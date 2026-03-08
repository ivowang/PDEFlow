from __future__ import annotations

import os
import shlex
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx
from pypdf import PdfReader

from common import ensure_dir, slugify


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

    def download_file(self, url: str, target_path: str) -> dict[str, Any]:
        if not self.config.execution.network_enabled:
            raise RuntimeError("Network-backed retrieval is disabled in the current config.")
        path = self._resolve_path(target_path, default_root=self.shared_workspace_root)
        ensure_dir(path.parent)
        with httpx.stream("GET", url, timeout=self.config.retrieval.http_timeout_seconds, follow_redirects=True) as response:
            response.raise_for_status()
            with path.open("wb") as handle:
                for chunk in response.iter_bytes():
                    handle.write(chunk)
        self._record_tool_event("download_file", {"url": url, "path": str(path)})
        return {"url": url, "path": str(path), "size_bytes": path.stat().st_size}

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
