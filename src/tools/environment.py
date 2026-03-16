from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import re
from collections import deque

from common import canonicalize_env_id
from state import EnvironmentRecord, EnvironmentResolutionState


class ManagedEnvironmentMixin:
    _RUNTIME_DEPENDENCY_PREFIXES = (
        "torch",
        "torchvision",
        "torchaudio",
        "triton",
        "nvidia-",
        "nvidia_",
        "deepxde",
        "tensorflow",
        "pyro",
        "jax",
    )

    def _gpu_available_for_env_setup(self) -> bool:
        snapshot = self.inspect_compute_environment()
        return bool(snapshot.selected_gpu_ids or snapshot.available_gpu_ids)

    def _normalize_dependency_name(self, dependency: str) -> str:
        cleaned = dependency.strip()
        cleaned = cleaned.split(";", 1)[0].strip()
        cleaned = cleaned.split("[", 1)[0].strip()
        cleaned = re.split(r"[<>=!~ @]", cleaned, maxsplit=1)[0].strip()
        return cleaned.lower().replace("_", "-")

    def _split_base_and_runtime_dependencies(self, dependencies: list[str]) -> tuple[list[str], list[str]]:
        base_dependencies: list[str] = []
        runtime_dependencies: list[str] = []
        for dependency in dependencies:
            name = self._normalize_dependency_name(dependency)
            if any(name.startswith(prefix) for prefix in self._RUNTIME_DEPENDENCY_PREFIXES):
                runtime_dependencies.append(dependency)
            else:
                base_dependencies.append(dependency)
        return base_dependencies, runtime_dependencies

    def _default_gpu_probe_ids(self) -> list[int] | None:
        snapshot = self.inspect_compute_environment()
        selected = list(snapshot.selected_gpu_ids or snapshot.available_gpu_ids)
        return selected or None

    def _cached_environment_signal(self, environment_path: Path) -> dict[str, object]:
        run_root = environment_path.parent.parent
        signal: dict[str, object] = {
            "gpu_runtime_ready": False,
            "env_ready": False,
            "torch_available": False,
            "h5py_available": False,
            "hydra_available": False,
            "torch_cuda_version": "",
            "generated_at": "",
        }
        capability_path = run_root / "state" / "capability_matrix.jsonl"
        if not capability_path.exists():
            return signal
        recent_records: deque[dict[str, object]] = deque(maxlen=12)
        for line in capability_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                recent_records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        target_resolved = str(environment_path.resolve())
        target_raw = str(environment_path)
        for record in reversed(recent_records):
            candidate_path = str(record.get("environment_path") or "").strip()
            if not candidate_path:
                continue
            try:
                candidate_resolved = str(Path(candidate_path).resolve())
            except OSError:
                candidate_resolved = candidate_path
            if candidate_resolved != target_resolved and candidate_path != target_raw:
                continue
            signal.update(
                {
                    "gpu_runtime_ready": bool(record.get("gpu_runtime_ready")),
                    "env_ready": bool(record.get("env_ready")),
                    "torch_available": bool(record.get("torch_available")),
                    "h5py_available": bool(record.get("h5py_available")),
                    "hydra_available": bool(record.get("hydra_available")),
                    "torch_cuda_version": str(record.get("torch_cuda_version") or ""),
                    "generated_at": str(record.get("generated_at") or ""),
                }
            )
            return signal
        return signal

    def _cuda_version_score(self, value: object) -> tuple[int, int]:
        raw = str(value or "").strip()
        match = re.search(r"(\d+)\.(\d+)", raw)
        if not match:
            return (0, 0)
        return (int(match.group(1)), int(match.group(2)))

    def _probe_torch_cuda(
        self,
        environment_python: Path,
        cwd: Path,
        *,
        gpu_ids: list[int] | None = None,
    ) -> dict[str, object]:
        probe_command = (
            f"{shlex.quote(str(environment_python))} -c "
            "\"import json, torch; "
            "print(json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda, "
            "'available': bool(torch.cuda.is_available())}))\""
        )
        probe_result = self.run_command(
            probe_command,
            cwd=cwd,
            allow_failure=True,
            stall_timeout_seconds=20,
            emit_progress=False,
            gpu_ids=gpu_ids or self._default_gpu_probe_ids(),
        )
        payload: dict[str, object] = {
            "returncode": probe_result["returncode"],
            "stdout_tail": probe_result["stdout_tail"],
            "stderr_tail": probe_result["stderr_tail"],
            "available": False,
            "torch": None,
            "cuda": None,
        }
        if probe_result["returncode"] == 0 and probe_result["stdout_tail"]:
            try:
                parsed = json.loads(probe_result["stdout_tail"].splitlines()[-1].strip())
                payload.update(parsed)
            except Exception:
                pass
        return payload

    def _coherent_torch_repair_commands(self, environment_python: Path) -> list[tuple[str, str]]:
        python = shlex.quote(str(environment_python))
        cleanup = (
            f"{python} -m pip uninstall -y torch torchvision torchaudio functorch triton"
        )
        uv_prefix = self._uv_command_prefix()
        candidate_specs = [
            (
                "cu128",
                f"{cleanup} && {uv_prefix} uv pip install --python {python} "
                "--index-url https://download.pytorch.org/whl/cu128 "
                "--upgrade --force-reinstall torch torchvision",
            ),
            (
                "cu124",
                f"{cleanup} && {uv_prefix} uv pip install --python {python} "
                "--index-url https://download.pytorch.org/whl/cu124 "
                "--upgrade --force-reinstall torch torchvision",
            ),
            (
                "cu121",
                f"{cleanup} && {uv_prefix} uv pip install --python {python} "
                "--index-url https://download.pytorch.org/whl/cu121 "
                "--upgrade --force-reinstall torch torchvision",
            ),
            (
                "cu118",
                f"{cleanup} && {uv_prefix} uv pip install --python {python} "
                "--index-url https://download.pytorch.org/whl/cu118 "
                "--upgrade --force-reinstall torch torchvision",
            ),
        ]
        return candidate_specs

    def _repair_gpu_torch_install(
        self,
        *,
        project_root: Path,
        environment_python: Path,
        dependencies: list[str],
        attempted_commands: list[str],
        results: list[dict[str, object]],
        fallback_recipe: list[str],
    ) -> tuple[bool, str | None]:
        normalized_runtime_names = {self._normalize_dependency_name(item) for item in dependencies}
        torch_requested = any(
            name.startswith("torch") or name == "torchvision"
            for name in normalized_runtime_names
        ) or "pdebench" in project_root.name.lower()
        if not torch_requested or not self._gpu_available_for_env_setup():
            return False, None

        initial_probe = self._probe_torch_cuda(
            environment_python,
            project_root,
            gpu_ids=self._default_gpu_probe_ids(),
        )
        results.append(
            {
                "command": "torch_cuda_probe",
                "cwd": str(project_root),
                "returncode": int(initial_probe["returncode"]),
                "stdout_tail": str(initial_probe.get("stdout_tail") or ""),
                "stderr_tail": str(initial_probe.get("stderr_tail") or ""),
                "log_path": None,
                "emit_progress": False,
                "job_kind": "environment_probe",
            }
        )
        if bool(initial_probe.get("available")):
            return False, None

        for runtime_profile, install_command in self._coherent_torch_repair_commands(environment_python):
            attempted_commands.append(install_command)
            install_result = self.run_command(install_command, cwd=project_root, allow_failure=True)
            results.append(install_result)
            if install_result["returncode"] != 0:
                continue
            probe = self._probe_torch_cuda(
                environment_python,
                project_root,
                gpu_ids=self._default_gpu_probe_ids(),
            )
            results.append(
                {
                    "command": f"torch_cuda_probe[{runtime_profile}]",
                    "cwd": str(project_root),
                    "returncode": int(probe["returncode"]),
                    "stdout_tail": str(probe.get("stdout_tail") or ""),
                    "stderr_tail": str(probe.get("stderr_tail") or ""),
                    "log_path": None,
                    "emit_progress": False,
                    "job_kind": "environment_probe",
                }
            )
            if bool(probe.get("available")):
                fallback_recipe.append(install_command)
                return True, runtime_profile
        return False, str(initial_probe.get("stderr_tail") or initial_probe.get("stdout_tail") or "torch_cuda_unavailable")

    def _probe_reusable_environment(
        self,
        environment_path: Path,
        *,
        require_gpu_runtime: bool = True,
    ) -> bool:
        inspection = self.inspect_python_environment(
            str(environment_path),
            modules=["torch", "h5py", "hydra"],
        )
        if not inspection.get("python_available") or not inspection.get("pip_available"):
            return False
        modules = inspection.get("modules", {})
        if not bool(modules.get("h5py")):
            return False
        if not bool(modules.get("torch")):
            return False
        torch_probe = self._probe_torch_cuda(
            self._environment_python(environment_path),
            environment_path.parent,
            gpu_ids=self._default_gpu_probe_ids(),
        )
        if not bool(torch_probe.get("returncode") == 0):
            return False
        if require_gpu_runtime:
            return bool(torch_probe.get("available"))
        return True

    def _candidate_environment_score(
        self,
        candidate: Path,
        *,
        require_gpu_runtime: bool,
    ) -> tuple[int, int, float]:
        cached = self._cached_environment_signal(candidate)
        return (
            int(bool(cached.get("gpu_runtime_ready"))) if require_gpu_runtime else int(bool(cached.get("env_ready"))),
            *self._cuda_version_score(cached.get("torch_cuda_version")),
            int(bool(cached.get("h5py_available"))) + int(bool(cached.get("hydra_available"))),
            candidate.lstat().st_mtime if candidate.exists() else 0.0,
        )

    def _find_reusable_environment(
        self,
        *,
        environment_slug: str,
        current_target: Path,
        require_gpu_runtime: bool = True,
    ) -> Path | None:
        family_root = self.workspace_family_root
        candidates = sorted(
            family_root.glob(f"*/envs/{environment_slug}"),
            key=lambda item: self._candidate_environment_score(
                item,
                require_gpu_runtime=require_gpu_runtime,
            ),
            reverse=True,
        )
        deduped_candidates: list[Path] = []
        seen_resolved: set[str] = set()
        for candidate in candidates:
            try:
                resolved_key = str(candidate.resolve())
            except OSError:
                resolved_key = str(candidate)
            if resolved_key in seen_resolved:
                continue
            seen_resolved.add(resolved_key)
            deduped_candidates.append(candidate)
        candidates = deduped_candidates
        max_reuse_probes = 12
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved == current_target.resolve():
                continue
            if not candidate.is_dir():
                continue
            if not (candidate / "bin" / "python").exists():
                continue
            try:
                if self._probe_reusable_environment(
                    candidate,
                    require_gpu_runtime=require_gpu_runtime,
                ):
                    return candidate
            except Exception:
                continue
            max_reuse_probes -= 1
            if max_reuse_probes <= 0:
                break
        return None

    def _allow_sibling_environment_reuse(self, *, require_gpu_runtime: bool) -> bool:
        return True

    def _link_reused_environment(self, source_env: Path, target_env: Path) -> None:
        ensure_target_parent = target_env.parent
        ensure_target_parent.mkdir(parents=True, exist_ok=True)
        if target_env.exists():
            return
        target_env.symlink_to(source_env, target_is_directory=True)

    def ensure_python_environment(
        self,
        project_path: str,
        environment_name: str | None = None,
        python_spec: str | None = None,
        dependency_strategy: str = "auto",
        editable_install: bool = True,
        require_gpu_runtime: bool = True,
    ) -> dict[str, object]:
        if not self.config.execution.auto_bootstrap_environments:
            raise RuntimeError("Automatic environment bootstrapping is disabled in the current config.")
        if not self.config.execution.allow_package_installation:
            raise RuntimeError("Package installation is disabled in the current config.")

        initial_root = self._resolve_path(project_path)
        project_root = self._find_project_root(initial_root)
        environment_slug = canonicalize_env_id(environment_name or project_root.name, project_hint=project_root.name)
        environment_path = self.managed_env_root / environment_slug
        python_interpreter = self._find_python_interpreter(project_root, python_spec)
        manifest_info = self.detect_project_manifests(str(project_root))
        manifests = set(manifest_info["manifests"])
        results: list[dict[str, object]] = []
        attempted_commands: list[str] = []
        dependencies: list[str] = []
        base_dependencies: list[str] = []
        runtime_dependencies: list[str] = []

        reusable_env = None
        if self._allow_sibling_environment_reuse(require_gpu_runtime=require_gpu_runtime):
            reusable_env = self._find_reusable_environment(
                environment_slug=environment_slug,
                current_target=environment_path,
                require_gpu_runtime=require_gpu_runtime,
            )
        if reusable_env is not None:
            self._link_reused_environment(reusable_env, environment_path)
            payload = {
                "project_path": str(project_root),
                "requested_path": str(initial_root),
                "environment_name": environment_slug,
                "environment_path": str(environment_path),
                "python_interpreter": python_interpreter,
                "environment_python": str(self._environment_python(environment_path)),
                "status": "ready",
                "state": EnvironmentResolutionState.READY.value,
                "manifests": sorted(manifests),
                "strategy": "reused_sibling_env",
                "failure_reason": None,
                "failure_category": None,
                "fallback_recipe": [f"reuse:{reusable_env}"],
                "attempted_commands": attempted_commands,
                "results": results,
            }
            self.memory.record_environment(
                EnvironmentRecord(
                    env_id=environment_slug,
                    canonical_id=environment_slug,
                    project_path=str(project_root),
                    environment_path=str(environment_path),
                    python_interpreter=python_interpreter,
                    state=EnvironmentResolutionState.READY,
                    strategy="reused_sibling_env",
                    attempted_commands=attempted_commands,
                    manifests=sorted(manifests),
                    fallback_recipe=[f"reuse:{reusable_env}"],
                )
            )
            self._record_tool_event("ensure_python_environment", payload)
            return payload

        create_command = (
            f"{self._uv_command_prefix()} uv venv --seed --allow-existing "
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
                "state": EnvironmentResolutionState.BROKEN.value,
                "failure_category": "create_failed",
                "failure_reason": create_result["stderr_tail"] or create_result["stdout_tail"],
                "manifests": sorted(manifests),
                "strategy": dependency_strategy,
                "attempted_commands": attempted_commands,
                "results": results,
            }
            self.memory.record_environment(
                EnvironmentRecord(
                    env_id=environment_slug,
                    canonical_id=environment_slug,
                    project_path=str(project_root),
                    environment_path=str(environment_path),
                    python_interpreter=python_interpreter,
                    state=EnvironmentResolutionState.BROKEN,
                    strategy=dependency_strategy,
                    attempted_commands=attempted_commands,
                    manifests=sorted(manifests),
                    failure_reason=payload["failure_reason"],
                    failure_category="create_failed",
                )
            )
            self._record_tool_event("ensure_python_environment", payload)
            return payload

        environment_python = self._environment_python(environment_path)
        strategy_used = "empty"
        env_state = EnvironmentResolutionState.CREATING
        failure_reason: str | None = None
        failure_category: str | None = None
        fallback_recipe: list[str] = []
        if "pyproject.toml" in manifests:
            sync_command = (
                f"{self._uv_command_prefix(VIRTUAL_ENV=str(environment_path))} "
                f"uv sync --project {shlex.quote(str(project_root))} --active --no-dev"
            )
            attempted_commands.append(sync_command)
            sync_result = self.run_command(sync_command, cwd=project_root, allow_failure=True)
            results.append(sync_result)
            if sync_result["returncode"] == 0:
                strategy_used = "uv_sync"
                env_state = EnvironmentResolutionState.READY
            elif dependency_strategy in {"auto", "minimal"}:
                env_state = EnvironmentResolutionState.SYNC_FAILED
                failure_category = "uv_sync_failed"
                failure_reason = sync_result["stderr_tail"] or sync_result["stdout_tail"]
                dependencies = self._extract_pyproject_dependencies(project_root)
                if dependencies:
                    base_dependencies, runtime_dependencies = self._split_base_and_runtime_dependencies(dependencies)
                if base_dependencies:
                    env_state = EnvironmentResolutionState.FALLBACK_INSTALLING
                    install_command = (
                        f"{self._uv_command_prefix()} uv pip install --python {shlex.quote(str(environment_python))} "
                        + " ".join(shlex.quote(item) for item in base_dependencies)
                    )
                    attempted_commands.append(install_command)
                    install_result = self.run_command(install_command, cwd=project_root, allow_failure=True)
                    results.append(install_result)
                    if install_result["returncode"] == 0 and editable_install:
                        fallback_recipe = [install_command]
                        env_state = EnvironmentResolutionState.EDITABLE_INSTALLING
                        editable_command = (
                            f"{self._uv_command_prefix()} uv pip install --python {shlex.quote(str(environment_python))} "
                            f"--editable {shlex.quote(str(project_root))} --no-deps"
                        )
                        attempted_commands.append(editable_command)
                        editable_result = self.run_command(editable_command, cwd=project_root, allow_failure=True)
                        results.append(editable_result)
                        if editable_result["returncode"] == 0:
                            strategy_used = "uv_pip_minimal_editable"
                            env_state = EnvironmentResolutionState.READY
                            fallback_recipe.append(editable_command)
                    elif install_result["returncode"] == 0:
                        strategy_used = "uv_pip_minimal"
                        env_state = EnvironmentResolutionState.READY
                        fallback_recipe = [install_command]
                elif dependencies:
                    runtime_dependencies = list(dependencies)
                    env_state = EnvironmentResolutionState.READY
                    strategy_used = "runtime_only_bootstrap"
        elif "requirements.txt" in manifests:
            install_command = (
                f"{self._uv_command_prefix()} uv pip install --python {shlex.quote(str(environment_python))} "
                f"--requirements {shlex.quote(str(project_root / 'requirements.txt'))}"
            )
            attempted_commands.append(install_command)
            install_result = self.run_command(install_command, cwd=project_root, allow_failure=True)
            results.append(install_result)
            if install_result["returncode"] == 0:
                strategy_used = "requirements"
                env_state = EnvironmentResolutionState.READY
                fallback_recipe = [install_command]
        elif "setup.py" in manifests:
            install_command = (
                f"{self._uv_command_prefix()} uv pip install --python {shlex.quote(str(environment_python))} "
                f"--editable {shlex.quote(str(project_root))}"
            )
            attempted_commands.append(install_command)
            install_result = self.run_command(install_command, cwd=project_root, allow_failure=True)
            results.append(install_result)
            if install_result["returncode"] == 0:
                strategy_used = "editable_setup"
                env_state = EnvironmentResolutionState.READY
                fallback_recipe = [install_command]

        verify_command = f"{shlex.quote(str(environment_python))} -c \"import sys; print(sys.executable); print(sys.version.split()[0])\""
        attempted_commands.append(verify_command)
        verify_result = self.run_command(verify_command, cwd=project_root, allow_failure=True, emit_progress=False)
        results.append(verify_result)
        pip_command = f"{shlex.quote(str(environment_python))} -m pip --version"
        attempted_commands.append(pip_command)
        pip_result = self.run_command(pip_command, cwd=project_root, allow_failure=True, emit_progress=False)
        results.append(pip_result)

        repaired_cuda, cuda_detail = self._repair_gpu_torch_install(
            project_root=project_root,
            environment_python=environment_python,
            dependencies=runtime_dependencies or dependencies,
            attempted_commands=attempted_commands,
            results=results,
            fallback_recipe=fallback_recipe,
        ) if require_gpu_runtime else (False, None)
        if repaired_cuda:
            strategy_used = f"{strategy_used}+gpu_torch_repair" if strategy_used != "empty" else "gpu_torch_repair"

        success = verify_result["returncode"] == 0 and strategy_used != "empty"
        payload = {
            "project_path": str(project_root),
            "requested_path": str(initial_root),
            "environment_name": environment_slug,
            "environment_path": str(environment_path),
            "python_interpreter": python_interpreter,
            "environment_python": str(environment_python),
            "status": "ready" if success else "failed",
            "state": (EnvironmentResolutionState.READY if success else EnvironmentResolutionState.BROKEN).value
            if env_state != EnvironmentResolutionState.SYNC_FAILED
            else env_state.value,
            "manifests": sorted(manifests),
            "strategy": strategy_used,
            "failure_reason": None if success else (failure_reason or str(cuda_detail) or pip_result["stderr_tail"] or verify_result["stderr_tail"]),
            "failure_category": None if success else (failure_category or "verification_failed"),
            "fallback_recipe": fallback_recipe,
            "attempted_commands": attempted_commands,
            "results": results,
        }
        self.memory.record_environment(
            EnvironmentRecord(
                env_id=environment_slug,
                canonical_id=environment_slug,
                project_path=str(project_root),
                environment_path=str(environment_path),
                python_interpreter=python_interpreter,
                state=EnvironmentResolutionState.READY if success else (env_state if env_state != EnvironmentResolutionState.READY else EnvironmentResolutionState.BROKEN),
                strategy=strategy_used,
                attempted_commands=attempted_commands,
                manifests=sorted(manifests),
                failure_reason=payload["failure_reason"],
                failure_category=payload["failure_category"],
                fallback_recipe=fallback_recipe,
            )
        )
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
                (
                    f"{shlex.quote(str(environment_python))} -c "
                    "\"import importlib.util, json; "
                    f"print(json.dumps({{'available': importlib.util.find_spec('{module}') is not None}}))\""
                ),
                cwd=resolved_env,
                allow_failure=True,
                emit_progress=False,
                stall_timeout_seconds=20,
            )
            if probe["returncode"] != 0:
                module_results[module] = False
                continue
            try:
                payload = json.loads((probe["stdout_tail"] or "").splitlines()[-1].strip())
                module_results[module] = bool(payload.get("available"))
            except Exception:
                module_results[module] = False
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
        job_kind: str = "environment_command",
        stall_timeout_seconds: int | None = None,
        emit_progress: bool = True,
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
            job_kind=job_kind,
            stall_timeout_seconds=stall_timeout_seconds,
            emit_progress=emit_progress,
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
