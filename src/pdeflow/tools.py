from __future__ import annotations

from collections import deque
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from typing import Any
from urllib.parse import urlencode
import xml.etree.ElementTree as ET

import httpx
from pypdf import PdfReader

from .config import SystemConfig
from .memory import ResearchMemory
from .schemas import EnvironmentSnapshot, SecretStatus
from .utils import ensure_dir, now_utc, short_hash, slugify, write_json

try:
    from agents import function_tool
except ImportError:  # pragma: no cover
    function_tool = None


class ResearchTools:
    """Executable tool surface exposed to specialist agents."""

    def __init__(self, config: SystemConfig, memory: ResearchMemory, repo_root: Path):
        self.config = config
        self.memory = memory
        self.repo_root = repo_root.resolve()
        shared_root = Path(config.execution.workspace_root)
        if shared_root.is_absolute():
            resolved_shared_root = shared_root.resolve()
        else:
            resolved_shared_root = (memory.root / shared_root).resolve()
        if not str(resolved_shared_root).startswith(str(memory.root.resolve())):
            raise ValueError(
                "execution.workspace_root must resolve inside execution.work_directory."
            )
        self.shared_workspace_root = ensure_dir(resolved_shared_root)
        self.run_workspace_root = ensure_dir(memory.root / "workspaces")

    def _record_tool_event(self, tool_name: str, payload: dict[str, Any]) -> None:
        self.memory.record_tool_event({"tool": tool_name, "timestamp": now_utc(), **payload})

    def _allowed_roots(self) -> list[Path]:
        return [
            self.repo_root,
            self.shared_workspace_root,
            self.memory.root.resolve(),
            self.run_workspace_root.resolve(),
        ]

    def _resolve_path(self, path_str: str, default_root: Path | None = None) -> Path:
        raw_path = Path(path_str)
        candidate = raw_path if raw_path.is_absolute() else (default_root or self.repo_root) / raw_path
        resolved = candidate.resolve()
        allowed = any(str(resolved).startswith(str(root)) for root in self._allowed_roots())
        if not allowed:
            raise ValueError(f"Path is outside managed roots: {resolved}")
        return resolved

    def inspect_secret_status(self) -> list[SecretStatus]:
        statuses = [
            SecretStatus(
                env_var=spec.env_var,
                purpose=spec.purpose,
                required=spec.required,
                is_set=bool(os.getenv(spec.env_var)),
                resolution_hint=f"Export {spec.env_var} in the shell before starting the autonomous workflow.",
            )
            for spec in self.config.secrets
        ]
        for status in statuses:
            self.memory.record_secret_status(status)
        return statuses

    def inspect_compute_environment(self) -> EnvironmentSnapshot:
        python_executable = sys.executable
        python_version = sys.version.split()[0]
        uv_result = self.run_command("uv --version", allow_failure=True)
        gpu_result = self.run_command(
            "nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader",
            allow_failure=True,
        )
        available_gpu_ids: list[int] = []
        gpu_descriptions: dict[str, str] = {}
        notes: list[str] = []
        if gpu_result["returncode"] == 0 and gpu_result["stdout_tail"]:
            for line in gpu_result["stdout_tail"].splitlines():
                parts = [part.strip() for part in line.split(",")]
                if len(parts) >= 3 and parts[0].isdigit():
                    gpu_id = int(parts[0])
                    available_gpu_ids.append(gpu_id)
                    gpu_descriptions[str(gpu_id)] = f"{parts[1]} ({parts[2]})"
            notes.append("GPU inventory detected via nvidia-smi.")
        else:
            notes.append("GPU probing failed; falling back to configured inventory hints.")
            gpu_descriptions = dict(self.config.resource_policy.gpu_inventory_hint)
            available_gpu_ids = [int(item) for item in gpu_descriptions if item.isdigit()]
            if gpu_result["stderr_tail"]:
                notes.append(gpu_result["stderr_tail"])
        selected = [gpu_id for gpu_id in self.config.resource_policy.preferred_gpu_ids if gpu_id in available_gpu_ids]
        if not selected:
            selected = available_gpu_ids[: self.config.resource_policy.max_parallel_experiments]
        snapshot = EnvironmentSnapshot(
            python_executable=python_executable,
            python_version=python_version,
            uv_available=uv_result["returncode"] == 0,
            uv_version=uv_result["stdout_tail"] or None,
            available_gpu_ids=available_gpu_ids,
            selected_gpu_ids=selected,
            gpu_descriptions=gpu_descriptions,
            notes=notes,
        )
        return snapshot

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
        repo_name = destination_name or slugify(repo_url.rsplit("/", 1)[-1].replace(".git", ""))
        local_path = self.shared_workspace_root / "repos" / repo_name
        ensure_dir(local_path.parent)
        if local_path.exists():
            result = {"status": "available", "path": str(local_path), "repo_url": repo_url}
            self._record_tool_event("clone_repository", result)
            return result
        command = f"git clone {shlex.quote(repo_url)} {shlex.quote(str(local_path))}"
        run_result = self.run_command(command, cwd=self.repo_root, allow_failure=False)
        if run_result["returncode"] != 0:
            raise RuntimeError(run_result["stderr_tail"] or f"Failed to clone {repo_url}")
        commit_result = self.run_command("git rev-parse HEAD", cwd=local_path, allow_failure=True)
        result = {
            "status": "cloned",
            "path": str(local_path),
            "repo_url": repo_url,
            "commit": commit_result["stdout_tail"] or None,
        }
        self._record_tool_event("clone_repository", result)
        return result

    def inspect_directory_tree(self, path: str, max_depth: int = 2, max_entries: int = 200) -> dict[str, Any]:
        root = self._resolve_path(path)
        entries: list[dict[str, Any]] = []
        root_depth = len(root.parts)
        for current_root, dirnames, filenames in os.walk(root):
            current_path = Path(current_root)
            depth = len(current_path.parts) - root_depth
            if depth > max_depth:
                dirnames[:] = []
                continue
            for dirname in sorted(dirnames):
                entries.append({"type": "dir", "path": str((current_path / dirname).relative_to(root))})
                if len(entries) >= max_entries:
                    break
            if len(entries) >= max_entries:
                break
            for filename in sorted(filenames):
                entries.append({"type": "file", "path": str((current_path / filename).relative_to(root))})
                if len(entries) >= max_entries:
                    break
            if len(entries) >= max_entries:
                break
        result = {"root": str(root), "entries": entries}
        self._record_tool_event("inspect_directory_tree", {"path": str(root), "entries": len(entries)})
        return result

    def read_text_file(self, path: str, max_chars: int = 20000) -> dict[str, Any]:
        resolved = self._resolve_path(path)
        text = resolved.read_text(encoding="utf-8")[:max_chars]
        self._record_tool_event("read_text_file", {"path": str(resolved), "chars": len(text)})
        return {"path": str(resolved), "text": text}

    def search_in_directory(
        self,
        path: str,
        pattern: str,
        glob: str | None = None,
        max_hits: int = 100,
    ) -> dict[str, Any]:
        root = self._resolve_path(path)
        rg_binary = shutil.which("rg")
        hits: list[dict[str, Any]] = []
        if rg_binary:
            command = [rg_binary, "-n", "--hidden", "--follow", pattern, str(root)]
            if glob:
                command[1:1] = ["-g", glob]
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            for line in completed.stdout.splitlines()[:max_hits]:
                file_path, line_no, line_text = line.split(":", 2)
                hits.append({"path": file_path, "line": int(line_no), "text": line_text.strip()})
        else:
            compiled = re.compile(pattern)
            for file_path in root.rglob(glob or "*"):
                if not file_path.is_file():
                    continue
                try:
                    text = file_path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    continue
                for line_no, line_text in enumerate(text.splitlines(), start=1):
                    if compiled.search(line_text):
                        hits.append({"path": str(file_path), "line": line_no, "text": line_text.strip()})
                        if len(hits) >= max_hits:
                            break
                if len(hits) >= max_hits:
                    break
        self._record_tool_event("search_in_directory", {"path": str(root), "pattern": pattern, "hits": len(hits)})
        return {"root": str(root), "hits": hits}

    def find_files(self, path: str, pattern: str) -> dict[str, Any]:
        root = self._resolve_path(path)
        matches = [str(item) for item in root.rglob(pattern) if item.is_file()]
        self._record_tool_event("find_files", {"path": str(root), "pattern": pattern, "count": len(matches)})
        return {"root": str(root), "matches": matches}

    def detect_project_manifests(self, path: str) -> dict[str, Any]:
        root = self._resolve_path(path)
        manifest_names = [
            "pyproject.toml",
            "uv.lock",
            "requirements.txt",
            "setup.py",
            "environment.yml",
            "environment.yaml",
            "Makefile",
        ]
        manifests = [name for name in manifest_names if (root / name).exists()]
        entrypoints = []
        for pattern in ("train*.py", "*train*.sh", "scripts/*.py", "scripts/*.sh"):
            entrypoints.extend(str(item) for item in root.rglob(pattern) if item.is_file())
        result = {"path": str(root), "manifests": manifests, "entrypoints": entrypoints[:50]}
        self._record_tool_event("detect_project_manifests", result)
        return result

    def bootstrap_python_environment(self, project_path: str) -> dict[str, Any]:
        if not self.config.execution.auto_bootstrap_environments:
            raise RuntimeError("Automatic environment bootstrapping is disabled in the current config.")
        if not self.config.execution.allow_package_installation:
            raise RuntimeError("Package installation is disabled in the current config.")
        project_root = self._resolve_path(project_path)
        manifest_info = self.detect_project_manifests(str(project_root))
        manifests = set(manifest_info["manifests"])
        commands: list[str] = []
        if "pyproject.toml" in manifests:
            commands.append("uv sync")
        elif "requirements.txt" in manifests:
            commands.extend(["uv venv", "uv pip install -r requirements.txt"])
        elif "setup.py" in manifests:
            commands.extend(["uv venv", "uv pip install -e ."])
        else:
            return {
                "project_path": str(project_root),
                "status": "unsupported",
                "manifests": sorted(manifests),
                "commands": [],
                "results": [],
            }
        results = []
        overall_status = "ready"
        for command in commands:
            result = self.run_command(command, cwd=project_root, allow_failure=True)
            results.append(result)
            if result["returncode"] != 0:
                overall_status = "failed"
                break
        payload = {
            "project_path": str(project_root),
            "status": overall_status,
            "manifests": sorted(manifests),
            "commands": commands,
            "results": results,
        }
        self._record_tool_event("bootstrap_python_environment", payload)
        return payload

    def copy_tree(self, source_path: str, destination_path: str) -> dict[str, Any]:
        source = self._resolve_path(source_path)
        destination = self._resolve_path(destination_path, default_root=self.run_workspace_root)
        if destination.exists():
            raise FileExistsError(f"Destination already exists: {destination}")
        ensure_dir(destination.parent)
        shutil.copytree(source, destination)
        payload = {"source": str(source), "destination": str(destination)}
        self._record_tool_event("copy_tree", payload)
        return payload

    def write_text_file(self, path: str, content: str) -> dict[str, Any]:
        resolved = self._resolve_path(path, default_root=self.run_workspace_root)
        ensure_dir(resolved.parent)
        resolved.write_text(content, encoding="utf-8")
        payload = {"path": str(resolved), "chars": len(content)}
        self._record_tool_event("write_text_file", payload)
        return payload

    def write_json_file(self, path: str, payload: Any) -> dict[str, Any]:
        resolved = self._resolve_path(path, default_root=self.run_workspace_root)
        write_json(resolved, payload)
        result = {"path": str(resolved)}
        self._record_tool_event("write_json_file", result)
        return result

    def write_patch_file(self, path: str, patch_text: str) -> dict[str, Any]:
        return self.write_text_file(path, patch_text)

    def apply_patch_file(self, repo_path: str, patch_path: str) -> dict[str, Any]:
        repository = self._resolve_path(repo_path)
        patch_file = self._resolve_path(patch_path)
        return self.run_command(
            f"git apply {shlex.quote(str(patch_file))}",
            cwd=repository,
            allow_failure=False,
        )

    def run_command(
        self,
        command: str,
        cwd: str | Path | None = None,
        env_overrides: dict[str, str] | None = None,
        gpu_ids: list[int] | None = None,
        log_path: str | None = None,
        allow_failure: bool = False,
    ) -> dict[str, Any]:
        if not self.config.execution.allow_shell_commands:
            raise RuntimeError("Shell command execution is disabled in the current config.")
        working_directory = self._resolve_path(str(cwd), default_root=self.repo_root) if cwd else self.repo_root
        log_file = (
            self._resolve_path(log_path, default_root=self.memory.logs_dir)
            if log_path
            else self.memory.logs_dir / f"cmd-{short_hash(command, str(working_directory), now_utc())}.log"
        )
        ensure_dir(log_file.parent)
        env = os.environ.copy()
        if env_overrides:
            env.update(env_overrides)
        if gpu_ids:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in gpu_ids)
        process = subprocess.Popen(
            shlex.split(command),
            cwd=str(working_directory),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        tail_lines: deque[str] = deque(maxlen=200)
        with log_file.open("w", encoding="utf-8") as handle:
            assert process.stdout is not None
            for line in process.stdout:
                handle.write(line)
                tail_lines.append(line.rstrip("\n"))
        return_code = process.wait()
        stdout_tail = "\n".join(tail_lines)
        result = {
            "command": command,
            "cwd": str(working_directory),
            "returncode": return_code,
            "stdout_tail": stdout_tail,
            "stderr_tail": "" if return_code == 0 else stdout_tail,
            "log_path": str(log_file),
        }
        self._record_tool_event("run_command", result)
        if return_code != 0 and not allow_failure:
            raise RuntimeError(stdout_tail or f"Command failed: {command}")
        return result

    def parse_json_file(self, path: str) -> dict[str, Any]:
        resolved = self._resolve_path(path)
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        result = {"path": str(resolved), "payload": payload}
        self._record_tool_event("parse_json_file", {"path": str(resolved)})
        return result

    def parse_metrics_file(self, path: str) -> dict[str, Any]:
        resolved = self._resolve_path(path)
        suffix = resolved.suffix.lower()
        if suffix == ".json":
            payload = json.loads(resolved.read_text(encoding="utf-8"))
            result = {"path": str(resolved), "metrics": payload}
            self._record_tool_event("parse_metrics_file", {"path": str(resolved), "format": "json"})
            return result
        text = resolved.read_text(encoding="utf-8")
        metrics: dict[str, Any] = {}
        for match in re.finditer(r"([A-Za-z0-9_./-]+)\s*[:=]\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", text):
            key = match.group(1)
            value = match.group(2)
            metrics[key] = float(value)
        result = {"path": str(resolved), "metrics": metrics}
        self._record_tool_event("parse_metrics_file", {"path": str(resolved), "format": "text", "count": len(metrics)})
        return result

    def write_report(self, filename: str, content: str) -> Path:
        path = self.memory.reports_dir / filename
        ensure_dir(path.parent)
        path.write_text(content, encoding="utf-8")
        self._record_tool_event("write_report", {"path": str(path), "chars": len(content)})
        return path

    def build_function_tools(self) -> list[Any]:
        if function_tool is None:
            return []

        @function_tool
        def inspect_secret_status() -> list[dict[str, Any]]:
            """Inspect configured secrets and whether they are available in the current environment."""
            return [item.model_dump(mode="python") for item in self.inspect_secret_status()]

        @function_tool
        def inspect_compute_environment() -> dict[str, Any]:
            """Inspect Python, uv, and GPU availability on the current machine."""
            return self.inspect_compute_environment().model_dump(mode="python")

        @function_tool
        def search_arxiv_papers(query: str, max_results: int = 5) -> list[dict[str, Any]]:
            """Search arXiv for papers relevant to the current research question."""
            return self.search_arxiv_papers(query, max_results=max_results)

        @function_tool
        def search_github_repositories(query: str, max_results: int = 5) -> list[dict[str, Any]]:
            """Search GitHub repositories relevant to the current research question."""
            return self.search_github_repositories(query, max_results=max_results)

        @function_tool
        def fetch_url_text(url: str, max_chars: int = 20000) -> dict[str, Any]:
            """Fetch text content from a remote URL such as a README, docs page, or dataset page."""
            return self.fetch_url_text(url, max_chars=max_chars)

        @function_tool
        def download_file(url: str, target_path: str) -> dict[str, Any]:
            """Download a remote file into the managed workspace."""
            return self.download_file(url, target_path=target_path)

        @function_tool
        def extract_pdf_text(pdf_path: str, max_pages: int = 6) -> dict[str, Any]:
            """Extract text from a locally downloaded PDF to support literature analysis."""
            return self.extract_pdf_text(pdf_path, max_pages=max_pages)

        @function_tool
        def clone_repository(repo_url: str, destination_name: str | None = None) -> dict[str, Any]:
            """Clone a Git repository into the managed workspace."""
            return self.clone_repository(repo_url, destination_name=destination_name)

        @function_tool
        def inspect_directory_tree(path: str, max_depth: int = 2, max_entries: int = 200) -> dict[str, Any]:
            """Inspect a local directory tree after cloning or downloading assets."""
            return self.inspect_directory_tree(path, max_depth=max_depth, max_entries=max_entries)

        @function_tool
        def read_text_file(path: str, max_chars: int = 20000) -> dict[str, Any]:
            """Read a local text file such as a README, config, script, or log."""
            return self.read_text_file(path, max_chars=max_chars)

        @function_tool
        def search_in_directory(path: str, pattern: str, glob: str | None = None, max_hits: int = 100) -> dict[str, Any]:
            """Search local files for patterns such as train entrypoints, checkpoints, or metrics."""
            return self.search_in_directory(path, pattern=pattern, glob=glob, max_hits=max_hits)

        @function_tool
        def find_files(path: str, pattern: str) -> dict[str, Any]:
            """Find local files matching a glob pattern."""
            return self.find_files(path, pattern=pattern)

        @function_tool
        def detect_project_manifests(path: str) -> dict[str, Any]:
            """Detect Python project manifests and likely training entrypoints in a repository."""
            return self.detect_project_manifests(path)

        @function_tool
        def bootstrap_python_environment(project_path: str) -> dict[str, Any]:
            """Bootstrap a Python environment for a cloned repository using uv where possible."""
            return self.bootstrap_python_environment(project_path)

        @function_tool
        def copy_tree(source_path: str, destination_path: str) -> dict[str, Any]:
            """Copy a repository or workspace tree to create a child program candidate."""
            return self.copy_tree(source_path, destination_path)

        @function_tool
        def write_text_file(path: str, content: str) -> dict[str, Any]:
            """Write or overwrite a text file inside the managed workspace."""
            return self.write_text_file(path, content)

        @function_tool
        def write_json_file(path: str, payload_json: str) -> dict[str, Any]:
            """Write a JSON file inside the managed workspace."""
            return self.write_json_file(path, json.loads(payload_json))

        @function_tool
        def write_patch_file(path: str, patch_text: str) -> dict[str, Any]:
            """Write a unified diff patch file inside the managed workspace."""
            return self.write_patch_file(path, patch_text)

        @function_tool
        def apply_patch_file(repo_path: str, patch_path: str) -> dict[str, Any]:
            """Apply a git patch file to a repository workspace."""
            return self.apply_patch_file(repo_path, patch_path)

        @function_tool
        def run_command(
            command: str,
            cwd: str | None = None,
            gpu_ids: list[int] | None = None,
            log_path: str | None = None,
            allow_failure: bool = False,
        ) -> dict[str, Any]:
            """Run a shell command in a managed directory and capture a persistent log."""
            return self.run_command(
                command,
                cwd=cwd,
                gpu_ids=gpu_ids,
                log_path=log_path,
                allow_failure=allow_failure,
            )

        @function_tool
        def parse_json_file(path: str) -> dict[str, Any]:
            """Parse a JSON file produced by an experiment or external repository."""
            return self.parse_json_file(path)

        @function_tool
        def parse_metrics_file(path: str) -> dict[str, Any]:
            """Parse a metrics JSON or a text log with key=value style metrics."""
            return self.parse_metrics_file(path)

        @function_tool
        def write_report(filename: str, content: str) -> dict[str, Any]:
            """Write a markdown report to the current run report directory."""
            return {"path": str(self.write_report(filename, content))}

        return [
            inspect_secret_status,
            inspect_compute_environment,
            search_arxiv_papers,
            search_github_repositories,
            fetch_url_text,
            download_file,
            extract_pdf_text,
            clone_repository,
            inspect_directory_tree,
            read_text_file,
            search_in_directory,
            find_files,
            detect_project_manifests,
            bootstrap_python_environment,
            copy_tree,
            write_text_file,
            write_json_file,
            write_patch_file,
            apply_patch_file,
            run_command,
            parse_json_file,
            parse_metrics_file,
            write_report,
        ]
