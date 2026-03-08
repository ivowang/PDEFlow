from __future__ import annotations

import os
import shlex
import shutil
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx
from pypdf import PdfReader

from common import ensure_dir, slugify
from state import ArtifactDownloadMetadata, ArtifactRecord


class _TransferError(RuntimeError):
    def __init__(self, failure_type: str, message: str):
        super().__init__(message)
        self.failure_type = failure_type
        self.message = message


class RetrievalToolsMixin:
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
        expected_checksum: str | None = None,
        checksum_algorithm: str = "md5",
        min_size_bytes: int | None = None,
        required_keys: list[str] | None = None,
    ) -> dict[str, Any]:
        if not self.config.execution.network_enabled:
            raise RuntimeError("Network-backed retrieval is disabled in the current config.")
        path = self._resolve_path(target_path, default_root=self.shared_workspace_root)
        ensure_dir(path.parent)
        artifact = ArtifactRecord(
            artifact_id=artifact_id or slugify(path.name),
            artifact_type=artifact_type,
            title=path.name,
            rationale="downloaded via managed downloader",
            source_url=url,
            local_path=str(path),
            status="downloaded",
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
                    }
                    self._record_tool_event("download_file", payload)
                    return payload
                time.sleep(min(10, attempt_count * 2))

        artifact = artifact.model_copy(
            update={
                "download_metadata": ArtifactDownloadMetadata(
                    source_url=url,
                    local_path=str(path),
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
                        "local_path": str(path),
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
            result = {"status": "available", "path": str(local_path), "repo_url": repo_url}
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
        }
        self._record_tool_event("clone_repository", result)
        return result
