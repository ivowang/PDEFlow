from __future__ import annotations

import os
from pathlib import Path
import shlex

from common import short_hash, slugify


class ManagedEnvironmentMixin:
    def ensure_python_environment(
        self,
        project_path: str,
        environment_name: str | None = None,
        python_spec: str | None = None,
        dependency_strategy: str = "auto",
        editable_install: bool = True,
    ) -> dict[str, object]:
        if not self.config.execution.auto_bootstrap_environments:
            raise RuntimeError("Automatic environment bootstrapping is disabled in the current config.")
        if not self.config.execution.allow_package_installation:
            raise RuntimeError("Package installation is disabled in the current config.")

        initial_root = self._resolve_path(project_path)
        project_root = self._find_project_root(initial_root)
        environment_slug = environment_name or f"{slugify(project_root.name)}-{short_hash(str(project_root))}"
        environment_path = self.managed_env_root / environment_slug
        python_interpreter = self._find_python_interpreter(project_root, python_spec)
        manifest_info = self.detect_project_manifests(str(project_root))
        manifests = set(manifest_info["manifests"])
        results: list[dict[str, object]] = []
        attempted_commands: list[str] = []

        create_command = (
            "uv venv --seed --allow-existing "
            f"--python {shlex.quote(python_interpreter)} {shlex.quote(str(environment_path))}"
        )
        attempted_commands.append(create_command)
        create_result = self.run_command(create_command, cwd=self.repo_root, allow_failure=True)
        results.append(create_result)
        if create_result["returncode"] != 0:
            payload = {
                "project_path": str(project_root),
                "requested_path": str(initial_root),
                "environment_name": environment_slug,
                "environment_path": str(environment_path),
                "python_interpreter": python_interpreter,
                "status": "failed",
                "manifests": sorted(manifests),
                "strategy": dependency_strategy,
                "attempted_commands": attempted_commands,
                "results": results,
            }
            self._record_tool_event("ensure_python_environment", payload)
            return payload

        environment_python = self._environment_python(environment_path)
        strategy_used = "empty"
        if "pyproject.toml" in manifests:
            sync_command = f"env VIRTUAL_ENV={shlex.quote(str(environment_path))} uv sync --project {shlex.quote(str(project_root))} --active --no-dev"
            attempted_commands.append(sync_command)
            sync_result = self.run_command(sync_command, cwd=project_root, allow_failure=True)
            results.append(sync_result)
            if sync_result["returncode"] == 0:
                strategy_used = "uv_sync"
            elif dependency_strategy in {"auto", "minimal"}:
                dependencies = self._extract_pyproject_dependencies(project_root)
                if dependencies:
                    install_command = (
                        f"uv pip install --python {shlex.quote(str(environment_python))} "
                        + " ".join(shlex.quote(item) for item in dependencies)
                    )
                    attempted_commands.append(install_command)
                    install_result = self.run_command(install_command, cwd=project_root, allow_failure=True)
                    results.append(install_result)
                    if install_result["returncode"] == 0 and editable_install:
                        editable_command = (
                            f"uv pip install --python {shlex.quote(str(environment_python))} "
                            f"--editable {shlex.quote(str(project_root))} --no-deps"
                        )
                        attempted_commands.append(editable_command)
                        editable_result = self.run_command(editable_command, cwd=project_root, allow_failure=True)
                        results.append(editable_result)
                        if editable_result["returncode"] == 0:
                            strategy_used = "uv_pip_minimal_editable"
                    elif install_result["returncode"] == 0:
                        strategy_used = "uv_pip_minimal"
        elif "requirements.txt" in manifests:
            install_command = (
                f"uv pip install --python {shlex.quote(str(environment_python))} "
                f"--requirements {shlex.quote(str(project_root / 'requirements.txt'))}"
            )
            attempted_commands.append(install_command)
            install_result = self.run_command(install_command, cwd=project_root, allow_failure=True)
            results.append(install_result)
            if install_result["returncode"] == 0:
                strategy_used = "requirements"
        elif "setup.py" in manifests:
            install_command = (
                f"uv pip install --python {shlex.quote(str(environment_python))} "
                f"--editable {shlex.quote(str(project_root))}"
            )
            attempted_commands.append(install_command)
            install_result = self.run_command(install_command, cwd=project_root, allow_failure=True)
            results.append(install_result)
            if install_result["returncode"] == 0:
                strategy_used = "editable_setup"

        verify_command = f"{shlex.quote(str(environment_python))} -c \"import sys; print(sys.executable); print(sys.version.split()[0])\""
        attempted_commands.append(verify_command)
        verify_result = self.run_command(verify_command, cwd=project_root, allow_failure=True, emit_progress=False)
        results.append(verify_result)
        pip_command = f"{shlex.quote(str(environment_python))} -m pip --version"
        attempted_commands.append(pip_command)
        pip_result = self.run_command(pip_command, cwd=project_root, allow_failure=True, emit_progress=False)
        results.append(pip_result)

        success = verify_result["returncode"] == 0 and strategy_used != "empty"
        payload = {
            "project_path": str(project_root),
            "requested_path": str(initial_root),
            "environment_name": environment_slug,
            "environment_path": str(environment_path),
            "python_interpreter": python_interpreter,
            "environment_python": str(environment_python),
            "status": "ready" if success else "failed",
            "manifests": sorted(manifests),
            "strategy": strategy_used,
            "attempted_commands": attempted_commands,
            "results": results,
        }
        self._record_tool_event("ensure_python_environment", payload)
        return payload

    def inspect_python_environment(self, environment_path: str, modules: list[str] | None = None) -> dict[str, object]:
        resolved_env = self._resolve_path(environment_path, default_root=self.managed_env_root)
        environment_python = self._environment_python(resolved_env)
        python_result = self.run_command(
            f"{shlex.quote(str(environment_python))} -c \"import sys; print(sys.executable); print(sys.version.split()[0])\"",
            cwd=resolved_env,
            allow_failure=True,
            emit_progress=False,
        )
        pip_result = self.run_command(
            f"{shlex.quote(str(environment_python))} -m pip --version",
            cwd=resolved_env,
            allow_failure=True,
            emit_progress=False,
        )
        module_results: dict[str, bool] = {}
        for module in modules or []:
            probe = self.run_command(
                f"{shlex.quote(str(environment_python))} -c \"import {module}\"",
                cwd=resolved_env,
                allow_failure=True,
                emit_progress=False,
            )
            module_results[module] = probe["returncode"] == 0
        payload = {
            "environment_path": str(resolved_env),
            "python_path": str(environment_python),
            "python_available": python_result["returncode"] == 0,
            "python_version": python_result["stdout_tail"].splitlines()[-1].strip() if python_result["returncode"] == 0 and python_result["stdout_tail"] else None,
            "pip_available": pip_result["returncode"] == 0,
            "pip_version": pip_result["stdout_tail"] or None,
            "modules": module_results,
        }
        self._record_tool_event("inspect_python_environment", payload)
        return payload

    def bootstrap_python_environment(self, project_path: str) -> dict[str, object]:
        payload = self.ensure_python_environment(project_path=project_path, dependency_strategy="auto")
        self._record_tool_event("bootstrap_python_environment", payload)
        return payload

    def run_in_environment(
        self,
        environment_path: str,
        command: str,
        cwd: str | Path | None = None,
        gpu_ids: list[int] | None = None,
        log_path: str | None = None,
        allow_failure: bool = True,
    ) -> dict[str, object]:
        resolved_env = self._resolve_path(environment_path, default_root=self.managed_env_root)
        env_bin = self._environment_bin(resolved_env)
        env_overrides = {
            "VIRTUAL_ENV": str(resolved_env),
            "PATH": f"{env_bin}:{os.environ.get('PATH', '')}",
        }
        result = self.run_command(
            command,
            cwd=cwd,
            env_overrides=env_overrides,
            gpu_ids=gpu_ids,
            log_path=log_path,
            allow_failure=allow_failure,
        )
        self._record_tool_event(
            "run_in_environment",
            {
                "environment_path": str(resolved_env),
                "command": command,
                "cwd": result["cwd"],
                "returncode": result["returncode"],
                "log_path": result["log_path"],
            },
        )
        return result
