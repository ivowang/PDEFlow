from __future__ import annotations

import io
import json
import os
import shlex
import shutil
import time
import tarfile
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
from state import ArtifactDownloadMetadata, ArtifactRecord, RepositoryRecord


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

    def _iter_sibling_artifact_registry_paths(self) -> list[Path]:
        family_root = self.workspace_family_root
        try:
            candidates = sorted(
                family_root.glob("*/artifacts/artifact_registry.jsonl"),
                key=lambda item: item.stat().st_mtime if item.exists() else 0.0,
                reverse=True,
            )
        except OSError:
            return []
        return candidates[:32]

    def _candidate_local_discovery_dirs(self, roots: list[Path], artifact_type: str) -> list[Path]:
        candidate_dirs: list[Path] = []
        common_relatives = {
            "dataset": (
                "",
                "external_assets",
                "external_assets/data",
                "external_assets/datasets",
                "data",
                "datasets",
            ),
            "checkpoint": (
                "",
                "external_assets",
                "external_assets/checkpoints",
                "checkpoints",
                "models",
            ),
        }.get(artifact_type, ("", "external_assets"))
        for root in roots:
            if not root.exists():
                continue
            if root.is_file():
                continue
            for relative in common_relatives:
                candidate = root / relative if relative else root
                if candidate.exists() and candidate.is_dir():
                    candidate_dirs.append(candidate)
            try:
                child_runs = sorted(
                    (item for item in root.iterdir() if item.is_dir()),
                    key=lambda item: item.stat().st_mtime,
                    reverse=True,
                )
            except OSError:
                child_runs = []
            for child in child_runs[:12]:
                for relative in common_relatives:
                    candidate = child / relative if relative else child
                    if candidate.exists() and candidate.is_dir():
                        candidate_dirs.append(candidate)
        deduped: list[Path] = []
        seen: set[str] = set()
        for candidate in candidate_dirs:
            try:
                key = str(candidate.resolve())
            except OSError:
                continue
            if key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)
        return deduped

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

    def _github_owner_repo(self, repo_url: str) -> tuple[str, str] | None:
        match = _OWNER_REPO_RE.search(repo_url)
        if not match:
            return None
        owner = match.group(1)
        repo = match.group(2).replace(".git", "")
        return owner, repo

    def _github_archive_branches(self, repo_url: str) -> list[str]:
        owner_repo = self._github_owner_repo(repo_url)
        if owner_repo is None:
            return []
        owner, repo = owner_repo
        cache = self._load_repo_cache()
        branches: list[str] = []
        for key in repo_resolution_keys(repo, repo_url):
            for item in cache.get(key, []):
                if item.get("html_url") == repo_url and item.get("default_branch"):
                    branch = str(item["default_branch"]).strip()
                    if branch and branch not in branches:
                        branches.append(branch)
        for branch in ("main", "master"):
            if branch not in branches:
                branches.append(branch)
        return branches

    def _extract_github_archive(self, repo_url: str, destination: Path) -> dict[str, Any] | None:
        owner_repo = self._github_owner_repo(repo_url)
        if owner_repo is None:
            return None
        owner, repo = owner_repo
        for branch in self._github_archive_branches(repo_url):
            archive_url = f"https://codeload.github.com/{owner}/{repo}/tar.gz/refs/heads/{branch}"
            try:
                response = httpx.get(
                    archive_url,
                    timeout=self.config.retrieval.http_timeout_seconds,
                    follow_redirects=True,
                )
                response.raise_for_status()
            except Exception:
                continue
            top_level = None
            with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as archive:
                members = archive.getmembers()
                for member in members:
                    name = member.name
                    if not name:
                        continue
                    parts = Path(name).parts
                    if not parts:
                        continue
                    if top_level is None:
                        top_level = parts[0]
                    relative_parts = parts[1:] if parts and parts[0] == top_level else parts
                    if not relative_parts:
                        continue
                    target = destination.joinpath(*relative_parts)
                    resolved_target = target.resolve()
                    if not str(resolved_target).startswith(str(destination.resolve())):
                        raise RuntimeError(f"Archive extraction escaped destination: {resolved_target}")
                    if member.isdir():
                        ensure_dir(resolved_target)
                        continue
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        continue
                    ensure_dir(resolved_target.parent)
                    with resolved_target.open("wb") as handle:
                        shutil.copyfileobj(extracted, handle)
            return {
                "status": "archive_downloaded",
                "path": str(destination),
                "repo_url": repo_url,
                "archive_url": archive_url,
                "branch": branch,
                "canonical_id": canonicalize_repo_id(repo, repo_url),
            }
        return None

    def search_arxiv_papers(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        if not self.config.execution.network_enabled:
            raise RuntimeError("Network-backed retrieval is disabled in the current config.")
        result_limit = min(max_results, self.config.retrieval.max_search_results, 5)
        params = urlencode(
            {
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": result_limit,
                "sortBy": "relevance",
                "sortOrder": "descending",
            }
        )
        url = f"https://export.arxiv.org/api/query?{params}"
        try:
            response = httpx.get(url, timeout=self.config.retrieval.http_timeout_seconds)
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            status_code = error.response.status_code if error.response is not None else None
            if status_code == 429:
                self._record_tool_event(
                    "search_arxiv_papers",
                    {
                        "query": query,
                        "count": 0,
                        "rate_limited": True,
                        "status_code": status_code,
                    },
                )
                return []
            raise
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
                    "authors": authors[:5],
                    "year": int(published[:4]) if len(published) >= 4 and published[:4].isdigit() else None,
                    "abstract": summary[:600],
                    "source_url": entry_id,
                    "pdf_url": pdf_url,
                }
            )
        papers = papers[:result_limit]
        self._record_tool_event("search_arxiv_papers", {"query": query, "count": len(papers)})
        return papers

    def search_github_repositories(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        if not self.config.execution.network_enabled:
            raise RuntimeError("Network-backed retrieval is disabled in the current config.")
        cached = self._repo_from_cache(query)
        if cached:
            self._record_tool_event("search_github_repositories", {"query": query, "count": len(cached), "cached": True})
            limit = min(max_results, self.config.retrieval.max_search_results, 5)
            return cached[:limit]
        headers = {"Accept": "application/vnd.github+json"}
        token = os.getenv("GITHUB_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        limit = min(max_results, self.config.retrieval.max_search_results, 5)
        response = httpx.get(
            "https://api.github.com/search/repositories",
            params={"q": query, "per_page": limit},
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
                items = heuristic_items[:limit]
        if items:
            items = items[:limit]
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
        *,
        strategy_id: str | None = None,
        attempt_signature: str | None = None,
        artifact_label: str | None = None,
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
        heartbeat_interval = max(30, int(self.config.retrieval.throughput_window_seconds // 2) or 30)
        last_heartbeat = started_at
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
                        if now - last_heartbeat >= heartbeat_interval:
                            elapsed = now - started_at
                            throughput = bytes_downloaded / elapsed if elapsed > 0 else None
                            self._record_tool_event(
                                "download_progress",
                                {
                                    "url": url,
                                    "path": str(part_path),
                                    "artifact_id": artifact_label,
                                    "strategy_id": strategy_id,
                                    "attempt_signature": attempt_signature,
                                    "attempt_count": attempt_index,
                                    "bytes_downloaded": bytes_downloaded,
                                    "elapsed_time": elapsed,
                                    "average_throughput": throughput,
                                    "transfer_method": "httpx",
                                    "resumed": resumed,
                                },
                            )
                            last_heartbeat = now
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

    def _download_with_curl(
        self,
        url: str,
        part_path: Path,
        attempt_index: int,
        *,
        strategy_id: str | None = None,
        attempt_signature: str | None = None,
        artifact_label: str | None = None,
    ) -> dict[str, Any]:
        existing_bytes = part_path.stat().st_size if part_path.exists() else 0
        started_at = time.monotonic()
        command = (
            "curl -L --fail --continue-at - "
            f"--output {shlex.quote(str(part_path))} {shlex.quote(url)}"
        )
        result = self.run_command(
            command,
            cwd=part_path.parent,
            allow_failure=True,
            job_kind="download",
            stall_timeout_seconds=self.config.retrieval.no_progress_timeout_seconds,
        )
        if result["returncode"] != 0:
            stderr = result["stderr_tail"] or result["stdout_tail"] or "curl download failed"
            failure_type = "transfer_failed"
            lowered = stderr.lower()
            if "timed out" in lowered:
                failure_type = "transfer_timeout"
            elif "connection refused" in lowered:
                failure_type = "connection_refused"
            elif "could not resolve" in lowered or "failed to connect" in lowered:
                failure_type = "transfer_unreachable"
            raise _TransferError(failure_type, stderr)
        bytes_downloaded = part_path.stat().st_size if part_path.exists() else existing_bytes
        elapsed = time.monotonic() - started_at
        return {
            "transfer_method": "curl",
            "attempt_count": attempt_index,
            "bytes_downloaded": bytes_downloaded,
            "elapsed_time": elapsed,
            "average_throughput": bytes_downloaded / elapsed if elapsed > 0 else None,
            "resumed": existing_bytes > 0,
        }

    def _download_with_wget(
        self,
        url: str,
        part_path: Path,
        attempt_index: int,
        *,
        strategy_id: str | None = None,
        attempt_signature: str | None = None,
        artifact_label: str | None = None,
    ) -> dict[str, Any]:
        existing_bytes = part_path.stat().st_size if part_path.exists() else 0
        started_at = time.monotonic()
        command = (
            "wget --continue --tries=1 "
            f"--timeout={int(self.config.retrieval.http_timeout_seconds)} "
            f"--output-document={shlex.quote(str(part_path))} {shlex.quote(url)}"
        )
        result = self.run_command(
            command,
            cwd=part_path.parent,
            allow_failure=True,
            job_kind="download",
            stall_timeout_seconds=self.config.retrieval.no_progress_timeout_seconds,
        )
        if result["returncode"] != 0:
            stderr = result["stderr_tail"] or result["stdout_tail"] or "wget download failed"
            failure_type = "transfer_failed"
            lowered = stderr.lower()
            if "timed out" in lowered or "timeout" in lowered:
                failure_type = "transfer_timeout"
            elif "connection refused" in lowered:
                failure_type = "connection_refused"
            elif "unable to resolve host address" in lowered or "failed: connection refused" in lowered:
                failure_type = "transfer_unreachable"
            raise _TransferError(failure_type, stderr)
        bytes_downloaded = part_path.stat().st_size if part_path.exists() else existing_bytes
        elapsed = time.monotonic() - started_at
        return {
            "transfer_method": "wget",
            "attempt_count": attempt_index,
            "bytes_downloaded": bytes_downloaded,
            "elapsed_time": elapsed,
            "average_throughput": bytes_downloaded / elapsed if elapsed > 0 else None,
            "resumed": existing_bytes > 0,
        }

    def _default_local_discovery_roots(self, artifact_type: str = "dataset") -> list[Path]:
        roots = [self.shared_workspace_root]
        if artifact_type in {"dataset", "checkpoint"}:
            if self.workspace_family_root not in {Path("/"), Path("/tmp"), Path("/var/tmp")}:
                roots.append(self.workspace_family_root)
            roots.append(self.memory.root)
        deduped: list[Path] = []
        for root in roots:
            resolved = root.resolve()
            if resolved not in deduped and resolved.exists():
                deduped.append(resolved)
        return deduped

    def _ready_route_artifact_for_optional_download_deferral(self) -> ArtifactRecord | None:
        artifacts = self.memory.load_artifacts()
        ready_datasets = [
            item
            for item in artifacts
            if item.artifact_type == "dataset" and item.status == "ready_for_training" and item.local_path
        ]
        if not ready_datasets:
            return None
        repositories = self.memory.load_repositories()
        environments = self.memory.load_environments()
        repo_ready = any(item.local_path and Path(item.local_path).exists() for item in repositories)
        env_ready = any(item.state.value == "ready" for item in environments)
        if not (repo_ready and env_ready):
            return None
        return max(
            ready_datasets,
            key=lambda item: (
                int(bool(item.semantic_spec and item.semantic_spec.equation == "Burgers")),
                int(bool(item.local_path)),
                int((item.validation.size_bytes if item.validation else 0) or 0),
            ),
        )

    def _artifact_materialization_path(self, artifact: ArtifactRecord) -> Path:
        if artifact.local_path:
            raw_path = Path(artifact.local_path)
            candidate = raw_path if raw_path.is_absolute() else self.shared_workspace_root / raw_path
            resolved = candidate.resolve()
            if self._is_within_managed_write_roots(resolved):
                return resolved
        metadata = artifact.metadata or {}
        semantic = artifact.semantic_spec
        filename = (
            (semantic.filename if semantic and semantic.filename else None)
            or str(metadata.get("expected_filename") or "").strip()
            or Path(artifact.source_url or artifact.title or artifact.artifact_id).name
            or f"{artifact.artifact_id}.bin"
        )
        if artifact.artifact_type == "dataset":
            root = self.shared_workspace_root / "datasets"
        elif artifact.artifact_type == "checkpoint":
            root = self.shared_workspace_root / "checkpoints"
        else:
            root = self.shared_workspace_root / "artifacts"
        official_path = str(metadata.get("official_path") or "").strip().strip("/")
        if official_path:
            return root / Path(official_path) / filename
        segments: list[str] = []
        if semantic and semantic.benchmark:
            segments.append(semantic.benchmark)
        if semantic and semantic.asset_family:
            segments.append(semantic.asset_family)
        elif semantic and semantic.equation:
            segments.append(semantic.equation)
        if semantic and semantic.split:
            segments.append(semantic.split)
        return root.joinpath(*segments, filename) if segments else root / filename

    def materialize_artifact_record(
        self,
        artifact: ArtifactRecord,
        *,
        strategy_id: str = "auto_materialize",
        source_type: str = "verified_remote_registry",
    ) -> dict[str, Any]:
        target_path = self._artifact_materialization_path(artifact)
        metadata = artifact.metadata or {}
        checksum = (
            metadata.get("official_md5")
            or metadata.get("expected_md5")
            or metadata.get("official_checksum")
        )
        checksum_algorithm = str(metadata.get("checksum_algorithm") or "md5")
        required_keys = list(metadata.get("required_keys", []))
        min_size_bytes = metadata.get("min_size_bytes")
        return self.download_file(
            url=artifact.source_url or "",
            target_path=str(target_path),
            artifact_id=artifact.artifact_id,
            artifact_type=artifact.artifact_type,
            strategy_id=strategy_id,
            source_type=source_type,
            canonical_target_id=artifact.canonical_id or artifact.artifact_id,
            expected_checksum=checksum,
            checksum_algorithm=checksum_algorithm,
            min_size_bytes=min_size_bytes,
            required_keys=required_keys,
        )

    def discover_local_artifacts(
        self,
        query: str,
        search_roots: list[str] | None = None,
        artifact_type: str = "dataset",
        canonical_target_id: str | None = None,
        expected_checksum: str | None = None,
        checksum_algorithm: str = "md5",
        min_size_bytes: int | None = None,
        required_keys: list[str] | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        roots = [
            self._resolve_path(root, default_root=self.workspace_family_root)
            for root in (search_roots or [])
        ] or self._default_local_discovery_roots(artifact_type)
        filename = Path(query).name
        root_prefixes = [str(root.resolve()) for root in roots]
        registry_records = self.memory.load_artifacts()
        known_bad_paths: set[str] = set()

        def _within_roots(path_str: str) -> bool:
            try:
                resolved = str(Path(path_str).resolve())
            except OSError:
                return False
            return any(resolved.startswith(prefix) for prefix in root_prefixes)

        cached_candidates: list[dict[str, Any]] = []
        for record in registry_records:
            if not record.local_path or Path(record.local_path).name != filename:
                continue
            if canonical_target_id and record.canonical_id not in {canonical_target_id, None}:
                continue
            if not _within_roots(record.local_path):
                continue
            if (
                record.download_metadata
                and record.download_metadata.source_type == "local_discovery"
                and not self._is_within_managed_write_roots(Path(record.local_path))
            ):
                continue
            if record.status in {"corrupted", "quarantined"}:
                known_bad_paths.add(str(Path(record.local_path).resolve()))
                continue
            validation = record.validation
            cached_candidates.append(
                {
                    "path": str(Path(record.local_path).resolve()),
                    "status": record.status,
                    "canonical_id": record.canonical_id,
                    "size_bytes": int(
                        (validation.size_bytes if validation else 0)
                        or (record.download_metadata.file_size if record.download_metadata else 0)
                        or 0
                    ),
                    "ready_for_training": bool(validation and validation.ready_for_training),
                    "cached": True,
                }
            )
        cached_candidates.sort(
            key=lambda item: (
                0 if item.get("ready_for_training") else 1,
                -(item.get("size_bytes") or 0),
                item.get("path", ""),
            )
        )
        if cached_candidates and cached_candidates[0].get("ready_for_training"):
            result = cached_candidates[: max(1, min(limit, 3))]
            self._record_tool_event(
                "discover_local_artifacts",
                {
                    "query": filename,
                    "roots": [str(root) for root in roots],
                    "count": len(result),
                    "cache_hit": True,
                },
            )
            return result

        sibling_registry_candidates: list[dict[str, Any]] = []
        sibling_ready_seen: set[str] = set()
        for registry_path in self._iter_sibling_artifact_registry_paths():
            try:
                with registry_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        raw = line.strip()
                        if not raw:
                            continue
                        try:
                            payload = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        local_path = str(payload.get("local_path") or "").strip()
                        if not local_path or Path(local_path).name != filename:
                            continue
                        if canonical_target_id and payload.get("canonical_id") not in {canonical_target_id, None}:
                            continue
                        if not _within_roots(local_path):
                            continue
                        try:
                            resolved = Path(local_path).resolve()
                        except OSError:
                            continue
                        path_str = str(resolved)
                        if path_str in known_bad_paths or path_str in sibling_ready_seen:
                            continue
                        if not resolved.exists() or not resolved.is_file():
                            continue
                        status = str(payload.get("status") or "")
                        if status in {"corrupted", "quarantined"}:
                            known_bad_paths.add(path_str)
                            continue
                        ready_for_training = status == "ready_for_training" or bool(
                            ((payload.get("validation") or {}).get("ready_for_training"))
                        )
                        sibling_registry_candidates.append(
                            {
                                "path": path_str,
                                "status": status or "downloaded",
                                "canonical_id": payload.get("canonical_id"),
                                "size_bytes": int(resolved.stat().st_size),
                                "ready_for_training": ready_for_training,
                                "cached": True,
                                "sibling_registry": str(registry_path),
                            }
                        )
                        sibling_ready_seen.add(path_str)
            except OSError:
                continue
        sibling_registry_candidates.sort(
            key=lambda item: (
                0 if item.get("ready_for_training") else 1,
                -(item.get("size_bytes") or 0),
                item.get("path", ""),
            )
        )
        if sibling_registry_candidates and sibling_registry_candidates[0].get("ready_for_training"):
            result = sibling_registry_candidates[: max(1, min(limit, 4))]
            self._record_tool_event(
                "discover_local_artifacts",
                {
                    "query": filename,
                    "roots": [str(root) for root in roots],
                    "count": len(result),
                    "sibling_registry_hit": True,
                },
            )
            return result

        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()
        search_dirs = self._candidate_local_discovery_dirs(roots, artifact_type)
        for root in search_dirs:
            try:
                iterator = root.rglob(filename)
                for path in iterator:
                    if not path.is_file():
                        continue
                    resolved = path.resolve()
                    path_str = str(resolved)
                    if path_str in seen:
                        continue
                    seen.add(path_str)
                    if path_str in known_bad_paths:
                        continue
                    if any(part in {"quarantine", ".git", "__pycache__"} for part in resolved.parts):
                        continue
                    artifact = ArtifactRecord(
                        artifact_id=slugify(filename),
                        canonical_id=canonical_target_id,
                        artifact_type=artifact_type,
                        title=filename,
                        rationale="discovered from existing local workspace",
                        source_url=str(resolved),
                        local_path=path_str,
                        status="downloaded",
                        metadata={
                            **({"official_checksum": expected_checksum} if expected_checksum else {}),
                            **({"checksum_algorithm": checksum_algorithm} if expected_checksum else {}),
                            **({"min_size_bytes": min_size_bytes} if min_size_bytes is not None else {}),
                            **({"required_keys": required_keys} if required_keys else {}),
                        },
                        download_metadata=ArtifactDownloadMetadata(
                            source_url=path_str,
                            source_type="local_discovery",
                            local_path=path_str,
                            canonical_target_id=canonical_target_id,
                            strategy_id="local_discovery",
                            attempt_signature=short_hash("local_discovery", filename, path_str),
                            file_size=resolved.stat().st_size,
                            validation_status="discovered",
                        ),
                    )
                    validated = self.validate_artifact_record(artifact, quarantine_on_failure=False)
                    candidates.append(
                        {
                            "path": path_str,
                            "status": validated.status,
                            "canonical_id": validated.canonical_id,
                            "size_bytes": resolved.stat().st_size,
                            "ready_for_training": bool(validated.validation and validated.validation.ready_for_training),
                        }
                    )
                    if candidates and candidates[-1]["ready_for_training"]:
                        break
                    if len(candidates) >= limit:
                        break
            except (FileNotFoundError, OSError):
                continue
            if len(candidates) >= limit or any(item.get("ready_for_training") for item in candidates):
                break
        if not candidates:
            if sibling_registry_candidates:
                candidates = sibling_registry_candidates[: max(1, min(limit, 4))]
            elif cached_candidates:
                candidates = cached_candidates[: max(1, min(limit, 3))]
        candidates.sort(
            key=lambda item: (
                0 if item.get("ready_for_training") else 1,
                -(item.get("size_bytes") or 0),
                item.get("path", ""),
            )
        )
        self._record_tool_event(
            "discover_local_artifacts",
            {
                "query": filename,
                "roots": [str(root) for root in roots],
                "search_dirs": [str(root) for root in search_dirs[:12]],
                "count": len(candidates),
            },
        )
        return candidates

    def _reuse_existing_local_copy(
        self,
        *,
        path: Path,
        artifact_id: str | None,
        artifact_type: str,
        canonical_target_id: str,
        expected_checksum: str | None,
        checksum_algorithm: str,
        min_size_bytes: int | None,
        required_keys: list[str] | None,
    ) -> ArtifactRecord | None:
        discovered = self.discover_local_artifacts(
            query=path.name,
            artifact_type=artifact_type,
            canonical_target_id=canonical_target_id,
            expected_checksum=expected_checksum,
            checksum_algorithm=checksum_algorithm,
            min_size_bytes=min_size_bytes,
            required_keys=required_keys,
        )
        for item in discovered:
            candidate_path = Path(item["path"]).resolve()
            if candidate_path == path.resolve():
                continue
            if item.get("status") != "ready_for_training":
                continue
            ensure_dir(path.parent)
            try:
                if path.exists():
                    path.unlink()
                os.link(candidate_path, path)
            except OSError:
                shutil.copy2(candidate_path, path)
            inferred_canonical_id, inferred_spec = canonicalize_artifact_id(
                artifact_id or path.name,
                local_path=str(path),
                title=path.name,
                metadata={"expected_filename": path.name},
                artifact_type=artifact_type,
            )
            artifact = ArtifactRecord(
                artifact_id=artifact_id or slugify(path.name),
                canonical_id=canonical_target_id or inferred_canonical_id,
                raw_aliases=[artifact_id or slugify(path.name), candidate_path.name],
                artifact_type=artifact_type,
                title=path.name,
                rationale="reused existing validated local copy",
                source_url=str(candidate_path),
                local_path=str(path),
                status="downloaded",
                semantic_spec=inferred_spec,
                metadata={
                    **({"official_checksum": expected_checksum} if expected_checksum else {}),
                    **({"checksum_algorithm": checksum_algorithm} if expected_checksum else {}),
                    **({"min_size_bytes": min_size_bytes} if min_size_bytes is not None else {}),
                    **({"required_keys": required_keys} if required_keys else {}),
                    "source_local_copy": str(candidate_path),
                },
                download_metadata=ArtifactDownloadMetadata(
                    source_url=str(candidate_path),
                    source_type="local_discovery",
                    local_path=str(path),
                    canonical_target_id=canonical_target_id or inferred_canonical_id,
                    strategy_id="local_discovery",
                    attempt_signature=short_hash("local_discovery", str(candidate_path), str(path)),
                    file_size=path.stat().st_size if path.exists() else 0,
                    validation_status="downloaded",
                    transfer_method="local_copy",
                    attempt_count=1,
                    bytes_downloaded=path.stat().st_size if path.exists() else 0,
                    elapsed_time=0.0,
                    average_throughput=None,
                    resumed=False,
                ),
            )
            validated = self.validate_artifact_record(artifact, quarantine_on_failure=False)
            if validated.validation and validated.validation.ready_for_training:
                self.memory.record_artifact(validated)
                return validated
        return None

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
        path = self._resolve_managed_write_path(target_path, default_root=self.shared_workspace_root)
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

        if artifact_type == "dataset" and self.current_phase == "acquisition":
            ready_route_artifact = self._ready_route_artifact_for_optional_download_deferral()
            if ready_route_artifact is not None:
                payload = {
                    "url": url,
                    "path": str(path),
                    "size_bytes": 0,
                    "validation_status": "deferred_optional_dataset_download",
                    "strategy_id": strategy_id,
                    "attempt_signature": attempt_signature,
                    "reused_existing": True,
                    "ready_route_artifact_id": ready_route_artifact.canonical_id or ready_route_artifact.artifact_id,
                    "ready_route_artifact_path": ready_route_artifact.local_path,
                    "reason": (
                        "A runnable acquisition route already exists with repo, env, and at least one "
                        "ready dataset artifact. Deferred optional dataset download."
                    ),
                }
                self._record_tool_event("download_file", payload)
                return payload

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

        if artifact_type in {"dataset", "checkpoint"}:
            reused_copy = self._reuse_existing_local_copy(
                path=path,
                artifact_id=artifact_id,
                artifact_type=artifact_type,
                canonical_target_id=canonical_target_id or inferred_canonical_id,
                expected_checksum=expected_checksum,
                checksum_algorithm=checksum_algorithm,
                min_size_bytes=min_size_bytes,
                required_keys=required_keys,
            )
            if reused_copy is not None:
                payload = {
                    "url": url,
                    "path": str(path),
                    "size_bytes": path.stat().st_size if path.exists() else 0,
                    "validation_status": reused_copy.status,
                    "reused_existing": True,
                    "strategy_id": "local_discovery",
                    "attempt_signature": short_hash("local_discovery", str(path)),
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
        transfer_methods = [self._download_httpx, self._download_with_curl, self._download_with_wget]
        for method_index, transfer_method in enumerate(transfer_methods):
            attempt_count = 0
            while attempt_count < max_attempts:
                attempt_count += 1
                try:
                    try:
                        transfer_stats = transfer_method(
                            url,
                            part_path,
                            attempt_index=attempt_count,
                            strategy_id=strategy_id,
                            attempt_signature=attempt_signature,
                            artifact_label=artifact.artifact_id,
                        )
                    except TypeError as exc:
                        if "unexpected keyword argument" not in str(exc):
                            raise
                        transfer_stats = transfer_method(url, part_path, attempt_index=attempt_count)
                    os.replace(part_path, path)
                    last_failure_type = None
                    last_failure_message = None
                    break
                except _TransferError as exc:
                    last_failure_type = exc.failure_type
                    last_failure_message = exc.message
                    if attempt_count >= max_attempts:
                        break
                    time.sleep(min(2.0, 0.5 * attempt_count))
            if path.exists():
                break
            if method_index < len(transfer_methods) - 1:
                self._record_tool_event(
                    "download_file",
                    {
                        "url": url,
                        "path": str(part_path),
                        "validation_status": "retrying_with_fallback_method",
                        "failure_type": last_failure_type,
                        "strategy_id": strategy_id,
                        "attempt_signature": attempt_signature,
                    },
                )

        if not path.exists():
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
                        transfer_method=transfer_stats.get("transfer_method"),
                        attempt_count=max_attempts,
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
                "attempt_count": max_attempts,
                "strategy_id": strategy_id,
                "attempt_signature": attempt_signature,
            }
            self._record_tool_event("download_file", payload)
            return payload

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
            repository = RepositoryRecord(
                repo_id=repo_name,
                canonical_id=canonicalize_repo_id(repo_name, repo_url),
                raw_aliases=[repo_name],
                name=Path(repo_name).name,
                remote_url=repo_url,
                local_path=str(local_path),
                bootstrap_status="cloned",
            )
            self.memory.record_repository(repository)
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
            archive_result = self._extract_github_archive(repo_url, local_path)
            if archive_result is not None:
                self.memory.record_repository(
                    RepositoryRecord(
                        repo_id=repo_name,
                        canonical_id=canonicalize_repo_id(repo_name, repo_url),
                        raw_aliases=[repo_name],
                        name=Path(repo_name).name,
                        remote_url=repo_url,
                        local_path=str(local_path),
                        bootstrap_status="archive_downloaded",
                        notes=[f"archive_url={archive_result['archive_url']}", f"branch={archive_result['branch']}"],
                    )
                )
                self._cache_repo_result(repo_url, [{
                    "name": Path(repo_name).name,
                    "full_name": canonicalize_repo_id(repo_name, repo_url),
                    "html_url": repo_url,
                    "description": "cached from archive fallback",
                    "stars": 0,
                    "default_branch": archive_result["branch"],
                    "resolution_source": "archive_fallback",
                }])
                self._record_tool_event("clone_repository", archive_result)
                return archive_result
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
        self.memory.record_repository(
            RepositoryRecord(
                repo_id=repo_name,
                canonical_id=canonicalize_repo_id(repo_name, repo_url),
                raw_aliases=[repo_name],
                name=Path(repo_name).name,
                remote_url=repo_url,
                local_path=str(local_path),
                bootstrap_status="cloned",
                notes=[f"commit={commit_result['stdout_tail'] or 'unknown'}"],
            )
        )
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
