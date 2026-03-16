from __future__ import annotations

from pathlib import Path
import re
import shlex
import shutil
import sys


class PythonRuntimeDiscoveryMixin:
    def _find_project_root(self, path: Path) -> Path:
        candidate = path if path.is_dir() else path.parent
        manifest_names = (
            "pyproject.toml",
            "uv.lock",
            "requirements.txt",
            "setup.py",
            "environment.yml",
            "environment.yaml",
        )
        while True:
            if any((candidate / name).exists() for name in manifest_names):
                return candidate
            if candidate == candidate.parent:
                return path if path.is_dir() else path.parent
            candidate = candidate.parent

    def _environment_python(self, environment_path: Path) -> Path:
        return environment_path / "bin" / "python"

    def _environment_bin(self, environment_path: Path) -> Path:
        return environment_path / "bin"

    def _extract_requires_python(self, project_root: Path) -> str | None:
        pyproject = project_root / "pyproject.toml"
        if not pyproject.exists():
            return None
        text = pyproject.read_text(encoding="utf-8")
        match = re.search(r'requires-python\s*=\s*"([^"]+)"', text)
        return match.group(1).strip() if match else None

    def _extract_pyproject_dependencies(self, project_root: Path) -> list[str]:
        pyproject = project_root / "pyproject.toml"
        if not pyproject.exists():
            return []
        text = pyproject.read_text(encoding="utf-8")
        marker = "dependencies = ["
        start = text.find(marker)
        if start == -1:
            return []
        index = start + len(marker)
        depth = 1
        while index < len(text) and depth > 0:
            char = text[index]
            if char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
            index += 1
        block = text[start + len(marker) : index - 1]
        return [item.strip() for item in re.findall(r'"([^"]+)"', block)]

    def _preferred_python_spec(self, project_root: Path, python_spec: str | None = None) -> str | None:
        if python_spec:
            return python_spec
        requires_python = self._extract_requires_python(project_root)
        if not requires_python:
            return None
        if "<3.10" in requires_python:
            return "3.9"
        if "<3.11" in requires_python:
            return "3.10"
        if "<3.12" in requires_python:
            return "3.11"
        match = re.search(r">=\s*([0-9]+\.[0-9]+)", requires_python)
        return match.group(1) if match else None

    def _find_python_interpreter(self, project_root: Path, python_spec: str | None = None) -> str:
        preferred_spec = self._preferred_python_spec(project_root, python_spec)
        if preferred_spec:
            find_result = self.run_command(
                f"{self._uv_command_prefix()} uv python find {shlex.quote(preferred_spec)}",
                cwd=self.repo_root,
                allow_failure=True,
                emit_progress=False,
            )
            if find_result["returncode"] == 0 and find_result["stdout_tail"].strip():
                return find_result["stdout_tail"].strip().splitlines()[-1].strip()
            if self.config.execution.allow_package_installation and self.config.execution.network_enabled:
                install_result = self.run_command(
                    (
                        f"{self._uv_command_prefix()} uv python install "
                        f"--install-dir {shlex.quote(str(self.managed_python_root))} "
                        f"{shlex.quote(preferred_spec)}"
                    ),
                    cwd=self.repo_root,
                    allow_failure=True,
                )
                if install_result["returncode"] == 0:
                    find_result = self.run_command(
                        f"{self._uv_command_prefix()} uv python find {shlex.quote(preferred_spec)}",
                        cwd=self.repo_root,
                        allow_failure=True,
                        emit_progress=False,
                    )
                    if find_result["returncode"] == 0 and find_result["stdout_tail"].strip():
                        return find_result["stdout_tail"].strip().splitlines()[-1].strip()
        for candidate in ("/usr/bin/python3", shutil.which("python3"), sys.executable):
            if candidate and Path(candidate).exists():
                return str(Path(candidate).resolve())
        raise RuntimeError("No usable Python interpreter was found for managed environment creation.")
