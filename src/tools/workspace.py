from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
from typing import Any

from common import ensure_dir, write_json


class WorkspaceToolsMixin:
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
