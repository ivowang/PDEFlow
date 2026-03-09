from __future__ import annotations

import os
import shlex
import shutil
import time
import xml.etree.ElementTree as ET
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlencode

import httpx
from pypdf import PdfReader

from common import (
    canonicalize_artifact_id,
    canonicalize_repo_id,
    canonicalize_source_url,
    ensure_dir,
    read_json,
    repo_resolution_keys,
    short_hash,
    slugify,
    write_json,
)
from state import ArtifactDownloadMetadata, ArtifactRecord


_OWNER_REPO_RE = re.compile(r"github\.com[:/]+([^/\s]+)/([^/\s]+?)(?:\.git)?/?$", flags=re.IGNORECASE)


class _TransferError(RuntimeError):
    def __init__(self, failure_type: str, message: str):
        super().__init__(message)
        self.failure_type = failure_type
        self.message = message


class RetrievalToolsMixin:
    _KNOWN_REPO_MAP = {
        "pdebench": {
            "name": "PDEBench",
            "full_name": "pdebench/PDEBench",
            "html_url": "https://github.com/pdebench/PDEBench",
            "description": "PDE benchmark repository",
            "stars": 0,
            "default_branch": "main",
        },
        "deepxde": {
            "name": "deepxde",
            "full_name": "lululxvi/deepxde",
            "html_url": "https://github.com/lululxvi/deepxde",
            "description": "DeepXDE",
            "stars": 0,
            "default_branch": "master",
        },
        "physics-informed-deeponets": {
            "name": "Physics-informed-DeepONets",
            "full_name": "PredictiveIntelligenceLab/Physics-informed-DeepONets",
            "html_url": "https://github.com/PredictiveIntelligenceLab/Physics-informed-DeepONets",
            "description": "Physics-informed DeepONets",
            "stars": 0,
            "default_branch": "main",
        },
        "neuraloperator": {
            "name": "neuraloperator",
            "full_name": "neuraloperator/neuraloperator",
            "html_url": "https://github.com/neuraloperator/neuraloperator",
            "description": "NeuralOperator official repository",
            "stars": 0,
            "default_branch": "main",
        },
    }

    def _repo_cache_path(self) -> Path:
        return self.memory.repositories_dir / "repo_resolution_cache.json"

    def _load_repo_cache(self) -> dict[str, list[dict[str, Any]]]:
        path = self._repo_cache_path()
        if not path.exists():
            return {}
        payload = read_json(path)
        return payload if isinstance(payload, dict) else {}

    def _save_repo_cache(self, cache: dict[str, list[dict[str, Any]]]) -> None:
        write_json(self._repo_cache_path(), cache)

    def _cache_repo_result(self, query: str, items: list[dict[str, Any]]) -> None:
        cache = self._load_repo_cache()
        keys = {slugify(query)}
        for item in items:
            keys.update(repo_resolution_keys(item.get("full_name") or item.get("name") or "", item.get("html_url")))
        payload = [dict(item) for item in items]
        for key in keys:
            cache[key] = payload
        self._save_repo_cache(cache)
        for item in items:
            self.memory.record_repo_resolution({"query": query, "resolved": item})

    def _repo_from_cache(self, query: str) -> list[dict[str, Any]]:
        cache = self._load_repo_cache()
        results: list[dict[str, Any]] = []
        for key in repo_resolution_keys(query):
            results.extend(cache.get(key, []))
        deduped: dict[str, dict[str, Any]] = {}
        for item in results:
            deduped[item["full_name"]] = item
        return list(deduped.values())

    def _heuristic_repo_candidates(self, query: str) -> list[dict[str, Any]]:
        lowered = query.lower()
        candidates: list[dict[str, Any]] = []
        for key, item in self._KNOWN_REPO_MAP.items():
            if key in lowered or item["full_name"].lower() in lowered or item["name"].lower() in lowered:
                candidates.append({**item, "resolution_source": "known_canonical_source"})
        if "github.com/" in lowered:
            match = _OWNER_REPO_RE.search(query)
            if match:
                owner, repo = match.group(1), match.group(2).replace(".git", "")
                candidates.append(
                    {
                        "name": repo,
                        "full_name": f"{owner}/{repo}",
                        "html_url": f"https://github.com/{owner}/{repo}",
                        "description": "heuristic github url candidate",
                        "stars": 0,
                        "default_branch": "main",
                        "resolution_source": "heuristic_url_probe",
                    }
                )
        return candidates

    def _validate_repo_candidate(self, item: dict[str, Any]) -> bool:
        repo_url = item.get("html_url")
        if not repo_url:
            return False
        probe = self.run_command(
            f"git ls-remote --heads {shlex.quote(repo_url)}",
            cwd=self.repo_root,
            allow_failure=True,
            emit_progress=False,
        )
        return probe["returncode"] == 0

    def search_arxiv_papers(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        if not self.config.execution.network_enabled:
            raise RuntimeError("Network-backed retrieval is disabled in the current config.")
        params = urlencode(
            {
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": min(max_results, self.config.retrieval.max_search_results),
                "sortBy": "relevance",
                "sortOrder": "descending",
            }
        )
        url = f"https://export.arxiv.org/api/query?{params}"
        response = httpx.get(url, timeout=self.config.retrieval.http_timeout_seconds)
        response.raise_for_status()
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(response.text)
        papers: list[dict[str, Any]] = []
        for entry in root.findall("atom:entry", ns):
            entry_id = entry.findtext("atom:id", default="", namespaces=ns)
            title = " ".join((entry.findtext("atom:title", default="", namespaces=ns) or "").split())
            summary = " ".join((entry.findtext("atom:summary", default="", namespaces=ns) or "").split())
            authors = [
                " ".join((author.findtext("atom:name", default="", namespaces=ns) or "").split())
                for author in entry.findall("atom:author", ns)
            ]
            published = entry.findtext("atom:published", default="", namespaces=ns)
            pdf_url = ""
            for link in entry.findall("atom:link", ns):
                if link.attrib.get("title") == "pdf":
                    pdf_url = link.attrib.get("href", "")
                    break
            papers.append(
                {
                    "paper_id": entry_id.rsplit("/", 1)[-1] if entry_id else slugify(title),
                    "title": title,
                    "authors": authors,
                    "year": int(published[:4]) if len(published) >= 4 and published[:4].isdigit() else None,
                    "abstract": summary,
                    "source_url": entry_id,
                    "pdf_url": pdf_url,
                }
            )
        self._record_tool_event("search_arxiv_papers", {"query": query, "count": len(papers)})
        return papers

    def search_github_repositories(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        if not self.config.execution.network_enabled:
            raise RuntimeError("Network-backed retrieval is disabled in the current config.")
        cached = self._repo_from_cache(query)
        if cached:
            self._record_tool_event("search_github_repositories", {"query": query, "count": len(cached), "cached": True})
            return cached[:max_results]
        headers = {"Accept": "application/vnd.github+json"}
        token = os.getenv("GITHUB_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        response = httpx.get(
            "https://api.github.com/search/repositories",
            params={"q": query, "per_page": min(max_results, self.config.retrieval.max_search_results)},
            headers=headers,
            timeout=self.config.retrieval.http_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        items = []
        for item in payload.get("items", []):
            items.append(
                {
                    "name": item["name"],
                    "full_name": item["full_name"],
                    "html_url": item["html_url"],
                    "description": item.get("description") or "",
                    "stars": item.get("stargazers_count", 0),
                    "default_branch": item.get("default_branch", "main"),
                }
            )
        if not items:
            heuristic_items = []
            for candidate in self._heuristic_repo_candidates(query):
                if self._validate_repo_candidate(candidate):
                    heuristic_items.append(candidate)
            if heuristic_items:
                items = heuristic_items[:max_results]
        if items:
            self._cache_repo_result(query, items)
        self._record_tool_event("search_github_repositories", {"query": query, "count": len(items)})
        return items

    def fetch_url_text(self, url: str, max_chars: int = 20000) -> dict[str, Any]:
        if not self.config.execution.network_enabled:
            raise RuntimeError("Network-backed retrieval is disabled in the current config.")
        response = httpx.get(url, timeout=self.config.retrieval.http_timeout_seconds, follow_redirects=True)
        response.raise_for_status()
        text = response.text[:max_chars]
        self._record_tool_event("fetch_url_text", {"url": url, "chars": len(text)})
        return {"url": url, "text": text, "status_code": response.status_code}

    def _download_httpx(
        self,
        url: str,
        part_path: Path,
        attempt_index: int,
    ) -> dict[str, Any]:
        existing_bytes = part_path.stat().st_size if part_path.exists() else 0
        headers: dict[str, str] = {}
        mode = "ab"
        resumed = existing_bytes > 0
        if resumed:
            headers["Range"] = f"bytes={existing_bytes}-"
        else:
            mode = "wb"
        timeout = httpx.Timeout(
            connect=self.config.retrieval.http_timeout_seconds,
            read=self.config.retrieval.no_progress_timeout_seconds,
            write=self.config.retrieval.http_timeout_seconds,
            pool=self.config.retrieval.http_timeout_seconds,
        )
        started_at = time.monotonic()
        bytes_downloaded = existing_bytes
        window_started = started_at
        window_bytes = 0
        try:
            with httpx.stream("GET", url, timeout=timeout, follow_redirects=True, headers=headers) as response:
                response.raise_for_status()
                if resumed and response.status_code != 206:
                    part_path.unlink(missing_ok=True)
                    bytes_downloaded = 0
                    mode = "wb"
                    resumed = False
                with part_path.open(mode) as handle:
                    for chunk in response.iter_bytes(chunk_size=self.config.retrieval.download_chunk_size_bytes):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        handle.flush()
                        bytes_downloaded += len(chunk)
                        window_bytes += len(chunk)
                        now = time.monotonic()
                        if now - window_started >= self.config.retrieval.throughput_window_seconds:
                            throughput = window_bytes / max(now - window_started, 1e-6)
                            if throughput < self.config.retrieval.min_transfer_throughput_bytes_per_second:
                                raise _TransferError(
                                    "transfer_stalled",
                                    (
                                        f"Throughput {throughput:.1f} B/s below minimum "
                                        f"{self.config.retrieval.min_transfer_throughput_bytes_per_second} B/s"
                                    ),
                                )
                            window_started = now
                            window_bytes = 0
        except httpx.ReadTimeout as exc:
            raise _TransferError("transfer_timeout", str(exc)) from exc
        except httpx.ConnectError as exc:
            raise _TransferError("connection_refused", str(exc)) from exc
        except httpx.HTTPStatusError as exc:
            raise _TransferError("transfer_http_error", str(exc)) from exc
        elapsed = time.monotonic() - started_at
        return {
            "transfer_method": "httpx",
            "attempt_count": attempt_index,
            "bytes_downloaded": bytes_downloaded,
            "elapsed_time": elapsed,
            "average_throughput": bytes_downloaded / elapsed if elapsed > 0 else None,
            "resumed": resumed,
        }

    def download_file(
        self,
        url: str,
        target_path: str,
        artifact_id: str | None = None,
        artifact_type: str = "dataset",
        strategy_id: str = "direct_remote_download",
        source_type: str = "remote_url",
        canonical_target_id: str | None = None,
        expected_checksum: str | None = None,
        checksum_algorithm: str = "md5",
        min_size_bytes: int | None = None,
        required_keys: list[str] | None = None,
    ) -> dict[str, Any]:
        if not self.config.execution.network_enabled:
            raise RuntimeError("Network-backed retrieval is disabled in the current config.")
        path = self._resolve_path(target_path, default_root=self.shared_workspace_root)
        ensure_dir(path.parent)
        inferred_canonical_id, inferred_spec = canonicalize_artifact_id(
            artifact_id or path.name,
            local_path=str(path),
            title=path.name,
            metadata={
                "benchmark": "PDEBench" if "pdebench" in str(path).lower() else None,
                "expected_filename": path.name,
            },
            artifact_type=artifact_type,
        )
        attempt_signature = short_hash(
            strategy_id,
            canonicalize_source_url(url),
            str(path),
            expected_checksum or "",
            checksum_algorithm,
        )
        artifact = ArtifactRecord(
            artifact_id=artifact_id or slugify(path.name),
            canonical_id=canonical_target_id or inferred_canonical_id,
            raw_aliases=[artifact_id or slugify(path.name)],
            artifact_type=artifact_type,
            title=path.name,
            rationale="downloaded via managed downloader",
            source_url=url,
            local_path=str(path),
            status="downloaded",
            semantic_spec=inferred_spec,
            metadata={
                **({"official_checksum": expected_checksum} if expected_checksum else {}),
                **({"checksum_algorithm": checksum_algorithm} if expected_checksum else {}),
                **({"min_size_bytes": min_size_bytes} if min_size_bytes is not None else {}),
                **({"required_keys": required_keys} if required_keys else {}),
            },
        )

        if path.exists():
            validated_existing = self.validate_artifact_record(artifact, quarantine_on_failure=True)
            if validated_existing.validation and validated_existing.validation.ready_for_training:
                self.memory.record_artifact(validated_existing)
                payload = {
                    "url": url,
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "validation_status": validated_existing.status,
                    "reused_existing": True,
                    "strategy_id": "reuse_validated_local",
                    "attempt_signature": attempt_signature,
                }
                self._record_tool_event("download_file", payload)
                return payload

        part_path = path.with_name(f"{path.name}.part")
        if part_path.exists():
            self._record_tool_event(
                "download_file",
                {
                    "url": url,
                    "path": str(part_path),
                    "validation_status": "partial_file_leftover",
                    "strategy_id": strategy_id,
                    "attempt_signature": attempt_signature,
                },
            )

        attempt_count = 0
        last_failure_type = None
        last_failure_message = None
        transfer_stats: dict[str, Any] = {
            "transfer_method": "httpx",
            "attempt_count": 0,
            "bytes_downloaded": 0,
            "elapsed_time": 0.0,
            "average_throughput": None,
            "resumed": False,
        }
        max_attempts = self.config.retrieval.max_transfer_attempts
        while attempt_count < max_attempts:
            attempt_count += 1
            try:
                transfer_stats = self._download_httpx(url, part_path, attempt_index=attempt_count)
                os.replace(part_path, path)
                break
            except _TransferError as exc:
                last_failure_type = exc.failure_type
                last_failure_message = exc.message
                if attempt_count >= max_attempts:
                    artifact = artifact.model_copy(
                        update={
                            "status": "download_failed",
                            "download_metadata": ArtifactDownloadMetadata(
                                source_url=url,
                            local_path=str(path),
                            source_type=source_type,
                            canonical_target_id=canonical_target_id or inferred_canonical_id,
                            strategy_id=strategy_id,
                            attempt_signature=attempt_signature,
                            file_size=part_path.stat().st_size if part_path.exists() else 0,
                            validation_status="download_failed",
                            transfer_method="httpx",
                                attempt_count=attempt_count,
                                bytes_downloaded=part_path.stat().st_size if part_path.exists() else 0,
                                elapsed_time=transfer_stats.get("elapsed_time", 0.0) or 0.0,
                                average_throughput=transfer_stats.get("average_throughput"),
                                failure_type=last_failure_type,
                                failure_message=last_failure_message,
                                resumed=bool(transfer_stats.get("resumed")),
                            ),
                        }
                    )
                    self.memory.record_artifact(artifact)
                    payload = {
                        "url": url,
                        "path": str(path),
                        "size_bytes": part_path.stat().st_size if part_path.exists() else 0,
                        "validation_status": "download_failed",
                        "failure_type": last_failure_type,
                        "attempt_count": attempt_count,
                        "strategy_id": strategy_id,
                        "attempt_signature": attempt_signature,
                    }
                    self._record_tool_event("download_file", payload)
                    return payload
                time.sleep(min(10, attempt_count * 2))

        artifact = artifact.model_copy(
            update={
                "download_metadata": ArtifactDownloadMetadata(
                    source_url=url,
                    source_type=source_type,
                    local_path=str(path),
                    canonical_target_id=canonical_target_id or inferred_canonical_id,
                    strategy_id=strategy_id,
                    attempt_signature=attempt_signature,
                    file_size=path.stat().st_size if path.exists() else 0,
                    validation_status="downloaded",
                    transfer_method=transfer_stats.get("transfer_method"),
                    attempt_count=attempt_count,
                    bytes_downloaded=transfer_stats.get("bytes_downloaded", 0),
                    elapsed_time=float(transfer_stats.get("elapsed_time", 0.0) or 0.0),
                    average_throughput=transfer_stats.get("average_throughput"),
                    resumed=bool(transfer_stats.get("resumed")),
                )
            }
        )
        validated = self.validate_artifact_record(artifact, quarantine_on_failure=True)
        download_metadata = validated.download_metadata or ArtifactDownloadMetadata(
            source_url=url,
            local_path=str(path),
            file_size=path.stat().st_size if path.exists() else 0,
        )
        validated = validated.model_copy(
            update={
                "download_metadata": download_metadata.model_copy(
                    update={
                        "source_url": url,
                        "source_type": source_type,
                        "local_path": str(path),
                        "canonical_target_id": canonical_target_id or inferred_canonical_id,
                        "strategy_id": strategy_id,
                        "attempt_signature": attempt_signature,
                        "validation_status": validated.status,
                        "attempt_count": attempt_count,
                        "transfer_method": transfer_stats.get("transfer_method"),
                        "bytes_downloaded": transfer_stats.get("bytes_downloaded", 0),
                        "elapsed_time": float(transfer_stats.get("elapsed_time", 0.0) or 0.0),
                        "average_throughput": transfer_stats.get("average_throughput"),
                        "resumed": bool(transfer_stats.get("resumed")),
                    }
                )
            }
        )
        self.memory.record_artifact(validated)
        payload = {
            "url": url,
            "path": str(path),
            "size_bytes": path.stat().st_size if path.exists() else 0,
            "validation_status": validated.status,
            "attempt_count": attempt_count,
            "transfer_method": transfer_stats.get("transfer_method"),
            "strategy_id": strategy_id,
            "attempt_signature": attempt_signature,
        }
        self._record_tool_event("download_file", payload)
        return payload

    def extract_pdf_text(self, pdf_path: str, max_pages: int = 6) -> dict[str, Any]:
        path = self._resolve_path(pdf_path)
        reader = PdfReader(str(path))
        text_chunks: list[str] = []
        pages_to_read = min(max_pages, len(reader.pages))
        for index in range(pages_to_read):
            text_chunks.append(reader.pages[index].extract_text() or "")
        text = "\n".join(text_chunks)
        self._record_tool_event("extract_pdf_text", {"path": str(path), "pages": pages_to_read})
        return {"path": str(path), "pages_read": pages_to_read, "text": text}

    def clone_repository(self, repo_url: str, destination_name: str | None = None) -> dict[str, Any]:
        if not self.config.execution.network_enabled:
            raise RuntimeError("Network-backed retrieval is disabled in the current config.")
        raw_name = destination_name or repo_url.rsplit("/", 1)[-1].replace(".git", "")
        repo_name = slugify(Path(raw_name).name.replace(".git", "")) or "repository"
        local_path = self.shared_workspace_root / "repos" / repo_name
        ensure_dir(local_path.parent)
        if local_path.exists():
            result = {
                "status": "available",
                "path": str(local_path),
                "repo_url": repo_url,
                "canonical_id": canonicalize_repo_id(repo_name, repo_url),
            }
            self._cache_repo_result(repo_url, [{
                "name": Path(repo_name).name,
                "full_name": canonicalize_repo_id(repo_name, repo_url),
                "html_url": repo_url,
                "description": "cached from successful clone",
                "stars": 0,
                "default_branch": "main",
                "resolution_source": "clone_cache",
            }])
            self._record_tool_event("clone_repository", result)
            return result
        command = f"timeout 300 git clone --depth 1 {shlex.quote(repo_url)} {shlex.quote(str(local_path))}"
        run_result = self.run_command(
            command,
            cwd=self.repo_root,
            env_overrides={"GIT_TERMINAL_PROMPT": "0"},
            allow_failure=True,
        )
        if run_result["returncode"] != 0:
            if local_path.exists():
                shutil.rmtree(local_path, ignore_errors=True)
            result = {
                "status": "failed",
                "path": str(local_path),
                "repo_url": repo_url,
                "error": run_result["stderr_tail"] or f"Failed to clone {repo_url}",
                "log_path": run_result["log_path"],
            }
            self._record_tool_event("clone_repository", result)
            return result
        commit_result = self.run_command("git rev-parse HEAD", cwd=local_path, allow_failure=True)
        result = {
            "status": "cloned",
            "path": str(local_path),
            "repo_url": repo_url,
            "commit": commit_result["stdout_tail"] or None,
            "canonical_id": canonicalize_repo_id(repo_name, repo_url),
        }
        self._cache_repo_result(repo_url, [{
            "name": Path(repo_name).name,
            "full_name": canonicalize_repo_id(repo_name, repo_url),
            "html_url": repo_url,
            "description": "cached from successful clone",
            "stars": 0,
            "default_branch": "main",
            "resolution_source": "clone_cache",
        }])
        self._record_tool_event("clone_repository", result)
        return result
