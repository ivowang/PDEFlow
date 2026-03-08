from __future__ import annotations

from pathlib import Path
import re
import shlex

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

    def _check_config_assignments(self, plan: ExperimentPlan) -> list[PreflightCheckResult]:
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
            config_paths = [
                working_directory / "config" / "args" / config_name,
                working_directory / "config" / config_name,
                working_directory / config_name,
            ]
            if not any(path.exists() for path in config_paths):
                checks.append(
                    PreflightCheckResult(
                        name="config_exists",
                        passed=False,
                        category="config",
                        details=f"Referenced config {config_name} was not found near {working_directory}.",
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
                blocking_reason="plan is not an actual experiment",
                recommended_route=plan.job_kind,
                related_artifact_ids=related_artifact_ids,
            )

        entrypoint = self._infer_entrypoint(plan)
        if entrypoint is None:
            checks.append(
                PreflightCheckResult(
                    name="entrypoint_exists",
                    passed=False,
                    category="import",
                    details="Could not resolve a Python entrypoint from the launch command.",
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

        checks.extend(self._check_config_assignments(plan))

        required_artifacts = {item.artifact_id: item for item in artifacts if item.artifact_id in plan.required_artifact_ids}
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

        log_path = self._resolve_path(plan.log_path, default_root=self.memory.experiments_dir)
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
            inspection = self.inspect_python_environment(env_path, modules=["torch", "h5py"])
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
                )
                checks.append(
                    PreflightCheckResult(
                        name="cuda_init",
                        passed=cuda_probe["returncode"] == 0 and cuda_probe["stdout_tail"].strip().endswith("1"),
                        category="environment",
                        details=cuda_probe["stdout_tail"] or str(cuda_probe["stderr_tail"]),
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
            if "dataset" in categories or "artifact" in categories:
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
            },
        )
        return report
