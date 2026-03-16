from __future__ import annotations

from pathlib import Path
import re
import shlex

from common import plan_requires_fno, plan_requires_pinn
from state import (
    ArtifactRecord,
    ArtifactStatus,
    CapabilityMatrix,
    ExperimentPlan,
    PreflightCheckResult,
    PreflightReport,
)
from common import ensure_dir, now_utc, short_hash


_CONFIG_ASSIGN_RE = re.compile(r"^\+{1,2}([A-Za-z0-9_.-]+)=(.+)$")


class PreflightValidationMixin:
    def _split_launch_tokens(self, command: str) -> list[str]:
        try:
            return shlex.split(command)
        except ValueError:
            return command.split()

    def _infer_entrypoint(self, plan: ExperimentPlan) -> Path | None:
        working_directory = Path(plan.working_directory)
        for token in self._split_launch_tokens(plan.launch_command):
            candidate = Path(token)
            if candidate.suffix == ".py":
                path = candidate if candidate.is_absolute() else (working_directory / candidate)
                if path.exists():
                    return path.resolve()
        return None

    def _is_inline_python(self, command: str) -> bool:
        stripped = command.strip()
        return "python - <<" in stripped or "python -c " in stripped or stripped.startswith("python - <<")

    def _looks_like_placeholder_command(self, command: str) -> bool:
        lowered = command.lower()
        return "launch command truncated" in lowered or "malformed output" in lowered

    def _infer_environment_path(self, plan: ExperimentPlan, capability_matrix: CapabilityMatrix | None) -> str | None:
        if plan.environment.get("VIRTUAL_ENV"):
            return plan.environment["VIRTUAL_ENV"]
        match = re.search(r"(/[^\s'\"]+/bin/python)", plan.launch_command)
        if match:
            return str(Path(match.group(1)).parent.parent)
        if capability_matrix and capability_matrix.environment_path:
            return capability_matrix.environment_path
        return self._discover_environment_path()

    def _parse_assignments(self, command: str) -> dict[str, str]:
        assignments: dict[str, str] = {}
        for token in self._split_launch_tokens(command):
            match = _CONFIG_ASSIGN_RE.match(token)
            if match:
                assignments[match.group(1)] = match.group(2).strip("'\"")
        return assignments

    def _candidate_config_paths(self, config_name: str, working_directory: Path, entrypoint: Path | None) -> list[Path]:
        candidate_roots = [working_directory]
        if entrypoint is not None:
            candidate_roots.extend(
                [
                    entrypoint.parent,
                    entrypoint.parent / "config",
                    entrypoint.parent.parent,
                    entrypoint.parent.parent / "config",
                ]
            )
        candidates: list[Path] = []
        for root in candidate_roots:
            candidates.extend(
                [
                    root / "config" / "args" / config_name,
                    root / "config" / config_name,
                    root / "args" / config_name,
                    root / config_name,
                ]
            )
        for root in candidate_roots:
            try:
                for match in root.rglob(config_name):
                    candidates.append(match)
                    if len(candidates) >= 32:
                        break
            except OSError:
                continue
            if len(candidates) >= 32:
                break
        deduped: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            path_str = str(candidate)
            if path_str not in seen:
                seen.add(path_str)
                deduped.append(candidate)
        return deduped

    def _check_config_assignments(self, plan: ExperimentPlan, entrypoint: Path | None) -> list[PreflightCheckResult]:
        checks: list[PreflightCheckResult] = []
        assignments = self._parse_assignments(plan.launch_command)
        for key, value in assignments.items():
            if value.startswith("/path/to"):
                checks.append(
                    PreflightCheckResult(
                        name=f"assignment:{key}",
                        passed=False,
                        category="config",
                        details=f"Placeholder path remains in launch command: {value}",
                    )
                )
        config_name = next((value for key, value in assignments.items() if key == "args"), None)
        if config_name:
            working_directory = Path(plan.working_directory)
            config_paths = self._candidate_config_paths(config_name, working_directory, entrypoint)
            if not any(path.exists() for path in config_paths):
                checks.append(
                    PreflightCheckResult(
                        name="config_exists",
                        passed=False,
                        category="config",
                        details=(
                            f"Referenced config {config_name} was not found near {working_directory} "
                            f"or the resolved entrypoint."
                        ),
                    )
                )
        return checks

    def _check_capability_requirements(
        self,
        plan: ExperimentPlan,
        capability_matrix: CapabilityMatrix | None,
    ) -> list[PreflightCheckResult]:
        if capability_matrix is None:
            return []
        checks: list[PreflightCheckResult] = []
        if plan_requires_pinn(plan):
            checks.append(
                PreflightCheckResult(
                    name="pinn_backend_ready",
                    passed=capability_matrix.pinn_ready,
                    category="environment",
                    details=(
                        "PINN route requires DeepXDE with a supported backend. "
                        f"deepxde_installed={capability_matrix.deepxde_installed} "
                        f"deepxde_backend={capability_matrix.deepxde_backend} "
                        f"tensorflow_available={capability_matrix.tensorflow_available}"
                    ),
                )
            )
        if plan_requires_fno(plan):
            checks.append(
                PreflightCheckResult(
                    name="fno_runtime_ready",
                    passed=capability_matrix.fno_ready,
                    category="environment",
                    details=(
                        "FNO route requires the PDEBench codepath, torch runtime, h5py, and a launch-ready environment. "
                        f"fno_ready={capability_matrix.fno_ready} "
                        f"torch_runtime_ready={capability_matrix.torch_runtime_ready} "
                        f"h5py_available={capability_matrix.h5py_available}"
                    ),
                )
            )
        return checks

    def preflight_experiment_plan(
        self,
        plan: ExperimentPlan,
        artifacts: list[ArtifactRecord],
        capability_matrix: CapabilityMatrix | None = None,
    ) -> PreflightReport:
        checks: list[PreflightCheckResult] = []
        related_artifact_ids = list(plan.required_artifact_ids)
        if plan.job_kind != "experiment":
            recommended_route = plan.job_kind
            if plan.plan_id == "__no_executable_plans__":
                if capability_matrix and capability_matrix.target_dataset_preparing:
                    recommended_route = "acquisition"
                elif capability_matrix and capability_matrix.fallback_assets_available:
                    recommended_route = "fallback_execution"
                elif capability_matrix and capability_matrix.target_dataset_blocked:
                    recommended_route = "acquisition"
                else:
                    recommended_route = "planning"
            return PreflightReport(
                report_id=f"preflight-{short_hash(plan.plan_id, now_utc())}",
                plan_id=plan.plan_id,
                program_id=plan.program_id,
                passed=False,
                failed_checks=[
                    PreflightCheckResult(
                        name="job_kind",
                        passed=False,
                        category="routing",
                        details=f"Non-experiment job kind {plan.job_kind} must stay outside experiment launch.",
                    )
                ],
                blocking_reason=plan.notes[0] if plan.notes else "plan is not an actual experiment",
                recommended_route=recommended_route,
                related_artifact_ids=related_artifact_ids,
            )

        entrypoint = self._infer_entrypoint(plan)
        if entrypoint is None and self._is_inline_python(plan.launch_command):
            checks.append(
                PreflightCheckResult(
                    name="entrypoint_inline_python",
                    passed=True,
                    category="import",
                    details="Inline Python fallback experiment does not require a file entrypoint.",
                )
            )
        elif entrypoint is None:
            details = "Could not resolve a Python entrypoint from the launch command."
            if self._looks_like_placeholder_command(plan.launch_command):
                details = (
                    "Launch command appears to be malformed placeholder output rather than an executable program."
                )
            checks.append(
                PreflightCheckResult(
                    name="entrypoint_exists",
                    passed=False,
                    category="import",
                    details=details,
                )
            )
        else:
            compile_result = self.run_command(
                f"python3 -m py_compile {shlex.quote(str(entrypoint))}",
                cwd=entrypoint.parent,
                allow_failure=True,
                emit_progress=False,
                job_kind="preflight",
            )
            checks.append(
                PreflightCheckResult(
                    name="entrypoint_import",
                    passed=compile_result["returncode"] == 0,
                    category="import",
                    details="entrypoint compiled successfully"
                    if compile_result["returncode"] == 0
                    else str(compile_result["stderr_tail"]),
                )
            )

        checks.extend(self._check_config_assignments(plan, entrypoint))
        checks.extend(self._check_capability_requirements(plan, capability_matrix))

        required_artifacts: dict[str, ArtifactRecord] = {}
        for item in artifacts:
            keys = [item.artifact_id]
            if item.canonical_id:
                keys.append(item.canonical_id)
            for key in keys:
                if key in plan.required_artifact_ids and key not in required_artifacts:
                    required_artifacts[key] = item
        for artifact_id in plan.required_artifact_ids:
            artifact = required_artifacts.get(artifact_id)
            if artifact is None:
                checks.append(
                    PreflightCheckResult(
                        name=f"artifact:{artifact_id}",
                        passed=False,
                        category="dataset",
                        details="Required artifact was not found in the current run state.",
                    )
                )
                continue
            ready = artifact.status == ArtifactStatus.READY_FOR_TRAINING.value
            detail = "artifact is training-ready"
            if not ready:
                reason = artifact.validation.failure_reasons if artifact.validation else [artifact.status]
                detail = f"artifact status={artifact.status}; reasons={reason}"
            checks.append(
                PreflightCheckResult(
                    name=f"artifact:{artifact_id}",
                    passed=ready,
                    category="dataset" if artifact.artifact_type == "dataset" else "artifact",
                    details=detail,
                )
            )

        log_path = self._resolve_managed_write_path(plan.log_path, default_root=self.memory.experiments_dir)
        ensure_dir(log_path.parent)
        writable_path = log_path.parent / ".preflight_write_test"
        try:
            writable_path.write_text("ok", encoding="utf-8")
            writable_path.unlink(missing_ok=True)
            checks.append(
                PreflightCheckResult(
                    name="output_writable",
                    passed=True,
                    category="filesystem",
                    details=f"Output directory is writable: {log_path.parent}",
                )
            )
        except OSError as exc:
            checks.append(
                PreflightCheckResult(
                    name="output_writable",
                    passed=False,
                    category="filesystem",
                    details=str(exc),
                )
            )

        env_path = self._infer_environment_path(plan, capability_matrix)
        if env_path:
            inspection_modules = ["torch", "h5py"]
            if plan_requires_pinn(plan):
                inspection_modules.extend(["deepxde", "tensorflow"])
            inspection = self.inspect_python_environment(env_path, modules=inspection_modules)
            checks.append(
                PreflightCheckResult(
                    name="environment_python",
                    passed=bool(inspection["python_available"]),
                    category="environment",
                    details=f"python_available={inspection['python_available']} pip_available={inspection['pip_available']}",
                )
            )
            if plan.gpu_ids:
                cuda_probe = self.run_in_environment(
                    env_path,
                    "python -c \"import torch; print(int(torch.cuda.is_available()))\"",
                    allow_failure=True,
                    emit_progress=False,
                    job_kind="preflight",
                    gpu_ids=plan.gpu_ids,
                )
                checks.append(
                    PreflightCheckResult(
                        name="cuda_init",
                        passed=cuda_probe["returncode"] == 0 and cuda_probe["stdout_tail"].strip().endswith("1"),
                        category="environment",
                        details=cuda_probe["stdout_tail"] or str(cuda_probe["stderr_tail"]),
                    )
                )
            if plan_requires_pinn(plan):
                backend_probe = self.run_in_environment(
                    env_path,
                    (
                        "python - <<'PY'\n"
                        "try:\n"
                        "    import deepxde as dde\n"
                        "    backend = getattr(getattr(dde, 'backend', None), 'backend_name', 'unknown')\n"
                        "    print(backend)\n"
                        "except Exception as exc:\n"
                        "    raise SystemExit(f'{type(exc).__name__}: {exc}')\n"
                        "PY"
                    ),
                    allow_failure=True,
                    emit_progress=False,
                    job_kind="preflight",
                    stall_timeout_seconds=60,
                    gpu_ids=plan.gpu_ids,
                )
                backend_name = backend_probe["stdout_tail"].splitlines()[-1].strip() if backend_probe["stdout_tail"] else None
                details = backend_probe["stderr_tail"] or backend_probe["stdout_tail"] or "DeepXDE backend probe returned no output."
                backend_ready = backend_probe["returncode"] == 0 and bool(backend_name)
                checks.append(
                    PreflightCheckResult(
                        name="pinn_backend_import",
                        passed=backend_ready,
                        category="environment",
                        details=details,
                    )
                )
        elif capability_matrix:
            checks.append(
                PreflightCheckResult(
                    name="environment_python",
                    passed=capability_matrix.python_available,
                    category="environment",
                    details="Used capability matrix fallback because no plan-local environment path was found.",
                )
            )

        passed = all(item.passed for item in checks)
        failed_checks = [item for item in checks if not item.passed]
        blocking_reason = failed_checks[0].details if failed_checks else None
        recommended_route = None
        if failed_checks:
            categories = {item.category for item in failed_checks}
            if any(item.name == "entrypoint_exists" and "placeholder" in item.details.lower() for item in failed_checks):
                recommended_route = "planning"
            elif "dataset" in categories or "artifact" in categories:
                recommended_route = "acquisition"
            elif "environment" in categories or "import" in categories:
                recommended_route = "environment_repair"
            else:
                recommended_route = "planning"
        report = PreflightReport(
            report_id=f"preflight-{short_hash(plan.plan_id, now_utc())}",
            plan_id=plan.plan_id,
            program_id=plan.program_id,
            passed=passed,
            failed_checks=failed_checks,
            blocking_reason=blocking_reason,
            recommended_route=recommended_route,
            related_artifact_ids=related_artifact_ids,
            log_path=str(log_path.parent / f"{plan.plan_id}.preflight.log"),
        )
        self._record_tool_event(
            "preflight_experiment_plan",
            {
                "plan_id": plan.plan_id,
                "passed": passed,
                "failed_checks": [item.name for item in failed_checks],
                "blocking_reason": blocking_reason,
                "recommended_route": recommended_route,
            },
        )
        return report
