from __future__ import annotations

from typing import Any

from state import (
    CapabilityMatrix,
    CodingPhaseOutput,
    ExperimentPhaseOutput,
    ExperimentPlan,
    ExperimentPlanningPhaseOutput,
    PreflightValidationPhaseOutput,
    ReflectionPhaseOutput,
    ResearchPhase,
    ResearchState,
)
from tools import ResearchTools
from common import dedupe_strings, now_utc, short_hash, upsert_by_attr
from .base import BaseResearchAgent


class CoderAgent(BaseResearchAgent):
    name = "CoderAgent"
    phase = ResearchPhase.CODING
    output_model = CodingPhaseOutput

    def build_instructions(self, state: ResearchState) -> str:
        return """
You are the coding specialist.
Implement the latest method design by creating a child workspace, editing files, and running basic validation commands.

You must use tools to:
- inspect the baseline repository and its entrypoints
- create a child workspace from the parent program or repository
- write or modify code files
- run at least one validation command such as import, compile, or a smoke test
- if repo dependencies are missing, provision a managed uv environment with tools instead of hand-writing `python -m venv` / `pip` flows

Rules:
- Do not claim a file was edited unless you used a tool to write it.
- Prefer minimal, targeted code changes that correspond directly to the method design.
- Return actual changed files and workspace paths.
"""

    def build_payload(self, state: ResearchState, tools: ResearchTools) -> dict[str, Any]:
        parent_programs = [item.model_dump(mode="python") for item in state.program_candidates[-8:]]
        return {
            "research_brief": state.research_brief.model_dump(mode="python"),
            "latest_method_designs": [item.model_dump(mode="python") for item in state.method_designs[-2:]],
            "available_programs": parent_programs,
            "selected_baseline_program_id": state.selected_baseline_program_id,
            "repositories": [item.model_dump(mode="python") for item in state.repositories[-8:]],
            "run_workspace_root": str(tools.run_workspace_root),
        }

    def apply_output(self, state: ResearchState, tools: ResearchTools, output: CodingPhaseOutput) -> str:
        state.program_candidates = upsert_by_attr(state.program_candidates, output.program_candidates, "program_id")
        state.next_actions = output.next_actions
        for item in output.program_candidates:
            tools.memory.register_program(item)
        return output.summary


class ExperimentPlannerAgent(BaseResearchAgent):
    name = "ExperimentPlannerAgent"
    phase = ResearchPhase.EXPERIMENT_PLANNING
    output_model = ExperimentPlanningPhaseOutput

    def build_instructions(self, state: ResearchState) -> str:
        return """
You are the experiment-planning specialist.
Build concrete execution plans for baseline and candidate programs.

You must:
- decide which programs to run next
- produce concrete setup commands, launch commands, working directories, logs, and expected outputs
- choose GPU usage and experiment duration indirectly through the command and stopping rules
- use only `ready_for_training` artifacts for actual experiment plans
- do not emit acquisition, repair, or data-download jobs here; those belong to acquisition
- do not emit plans that point to blocked, corrupted, quarantined, or checksum-mismatched artifacts
- prefer managed uv environments over ad hoc `python -m venv`, `source`, or bare `pip install`
- a blocked or setup-failed baseline does not count as a completed baseline
- if the selected baseline has no completed experiment record with real outputs, do not schedule downstream candidate launch plans except prerequisite acquisition/verification or matched baseline reruns
- if the manager route focus requests fallback_execution, prefer an evidence-generating fallback experiment over returning an empty plan set
- do not retry acquisition-dependent plans unchanged when the blocker registry says the same dataset acquisition route is exhausted
- inspect the repository entrypoint and config semantics before writing launch commands; if the code separates dataset filename from dataset root, pass both correctly and override placeholder defaults with verified local artifact paths

Rules:
- Do not invent commands that are impossible to run from the inspected repository layout.
- If no baseline experiment exists yet, include a baseline plan.
- Stopping rules should be scientific and metric-driven, not based on an arbitrary wall-clock cap.
"""

    def build_payload(self, state: ResearchState, tools: ResearchTools) -> dict[str, Any]:
        return {
            "research_brief": state.research_brief.model_dump(mode="python"),
            "environment_snapshot": state.environment_snapshot.model_dump(mode="python") if state.environment_snapshot else None,
            "external_artifacts": [item.model_dump(mode="python") for item in state.external_artifacts[-20:]],
            "repositories": [item.model_dump(mode="python") for item in state.repositories[-10:]],
            "program_candidates": [item.model_dump(mode="python") for item in state.program_candidates[-10:]],
            "method_designs": [item.model_dump(mode="python") for item in state.method_designs[-2:]],
            "existing_experiments": [item.model_dump(mode="python") for item in state.experiment_records[-10:]],
            "existing_plans": [item.model_dump(mode="python") for item in state.experiment_plans[-10:]],
            "managed_env_root": str(tools.managed_env_root),
            "preferred_log_root": str(tools.memory.experiments_dir),
            "selected_baseline_program_id": state.selected_baseline_program_id,
            "failure_summaries": state.failure_summaries[-12:],
            "capability_matrix": state.capability_matrix.model_dump(mode="python") if state.capability_matrix else None,
            "classified_failures": [item.model_dump(mode="python") for item in state.classified_failures[-12:]],
            "blocker_registry": [item.model_dump(mode="python") for item in state.blocker_registry[-12:]],
            "route_history": [item.model_dump(mode="python") for item in state.route_history[-6:]],
            "active_route_id": state.active_route_id,
            "active_route_focus": state.active_route_focus,
            "hitl_events": [item.model_dump(mode="python") for item in state.hitl_events[-6:]],
            "manual_asset_roots": state.manual_asset_roots,
            "skipped_target_entities": state.skipped_target_entities,
            "human_guidance_notes": state.human_guidance_notes[-12:],
        }

    def apply_output(self, state: ResearchState, tools: ResearchTools, output: ExperimentPlanningPhaseOutput) -> str:
        if not output.experiment_plans and "fallback_execution" in state.active_route_focus:
            fallback_plan = self._build_fallback_plan(state, tools)
            if fallback_plan is not None:
                output = output.model_copy(
                    update={
                        "experiment_plans": [fallback_plan],
                        "summary": (
                            output.summary
                            + " Added a deterministic evidence-generating fallback experiment because the active route requests fallback execution."
                        ),
                        "next_actions": [
                            *output.next_actions,
                            "Run the fallback smoke experiment to generate empirical evidence while exact target datasets remain blocked.",
                        ],
                    }
                )
        state.experiment_plans = upsert_by_attr(state.experiment_plans, output.experiment_plans, "plan_id")
        state.next_actions = output.next_actions
        for item in output.experiment_plans:
            tools.memory.record_experiment_plan(item)
        return output.summary

    def _build_fallback_plan(self, state: ResearchState, tools: ResearchTools) -> ExperimentPlan | None:
        capability = state.capability_matrix
        if capability is None or not capability.env_ready or not capability.codepath_ready:
            return None
        repository = next((item for item in state.repositories if item.local_path), None)
        if repository is None:
            return None
        env_path = repository.environment_path or capability.environment_path
        if not env_path:
            return None
        ready_artifacts = [
            item for item in state.external_artifacts
            if item.status == "ready_for_training" and item.artifact_type in {"checkpoint", "dataset"}
        ]
        artifact_paths = [item.local_path for item in ready_artifacts if item.local_path][:3]
        artifact_asserts = "\n".join(
            f"assert Path({path!r}).exists(), {path!r}" for path in artifact_paths
        )
        report_path = tools.memory.experiments_dir / "fallback_smoke_metrics.json"
        script = (
            "python - <<'PY'\n"
            "from pathlib import Path\n"
            "import json\n"
            "import torch\n"
            "import h5py\n"
            "import pdebench\n"
            f"{artifact_asserts}\n"
            "payload = {\n"
            "  'mode': 'fallback_smoke',\n"
            "  'torch_version': torch.__version__,\n"
            "  'cuda_available': bool(torch.cuda.is_available()),\n"
            f"  'checked_artifacts': {artifact_paths!r},\n"
            "}\n"
            f"Path({str(report_path)!r}).write_text(json.dumps(payload), encoding='utf-8')\n"
            "print(json.dumps(payload))\n"
            "PY"
        )
        return ExperimentPlan(
            plan_id=f"fallback-smoke-{short_hash(state.run_name, str(state.cycle_index), now_utc())}",
            title="Fallback smoke evidence run",
            program_id=state.selected_baseline_program_id or "fallback-smoke",
            repo_id=repository.canonical_id or repository.repo_id,
            job_kind="experiment",
            working_directory=repository.local_path,
            setup_commands=[],
            launch_command=script,
            environment={"VIRTUAL_ENV": env_path},
            gpu_ids=list(state.environment_snapshot.selected_gpu_ids if state.environment_snapshot else []),
            required_artifact_ids=[item.canonical_id or item.artifact_id for item in ready_artifacts[:3]],
            preflight_required=True,
            expected_outputs=[str(report_path)],
            success_criteria=["The fallback smoke run completes and emits a JSON payload proving repo/env execution works."],
            stopping_rules=["Stop immediately on import or filesystem failure."],
            log_path=str(tools.memory.experiments_dir / "fallback_smoke.log"),
            status="planned",
            notes=[
                "evidence_generating_fallback",
                "This plan exists to break zero-evidence stagnation while exact target datasets remain blocked.",
            ],
        )


class PreflightValidationAgent(BaseResearchAgent):
    name = "PreflightValidationAgent"
    phase = ResearchPhase.PREFLIGHT_VALIDATION
    output_model = PreflightValidationPhaseOutput

    def build_instructions(self, state: ResearchState) -> str:
        return "Deterministic preflight validation agent."

    def build_payload(self, state: ResearchState, tools: ResearchTools) -> dict[str, Any]:
        return {}

    def run(self, state: ResearchState, tools: ResearchTools, runtime: Any) -> str:
        pending_plans = [
            plan for plan in state.experiment_plans
            if plan.status == "planned" and plan.job_kind == "experiment"
        ]
        if not pending_plans:
            zero_reason = "No experiment plans are launch-eligible after planning."
            if state.capability_matrix and state.capability_matrix.target_dataset_blocked:
                zero_reason = (
                    "No experiment plans are launch-eligible because the exact target datasets remain blocked "
                    "and the planner produced no fallback executable plans."
                )
            report = tools.preflight_experiment_plan(
                ExperimentPlan(
                    plan_id="__no_executable_plans__",
                    title="No executable plans",
                    program_id=state.selected_baseline_program_id or "none",
                    job_kind="preflight",
                    working_directory=state.work_directory,
                    launch_command="true",
                    log_path=str(tools.memory.preflight_dir / "no_executable_plans.log"),
                    status="blocked",
                    notes=[zero_reason],
                ),
                state.external_artifacts,
                state.capability_matrix,
            )
            capability_matrix = tools.probe_capability_matrix(
                artifacts=state.external_artifacts,
                repository_paths=[repo.local_path for repo in state.repositories],
                environment_path=state.capability_matrix.environment_path if state.capability_matrix else None,
            )
            output = PreflightValidationPhaseOutput(
                summary="Preflight validated pending experiment plans. passed=0 blocked=1.",
                preflight_reports=[report],
                capability_matrix=capability_matrix,
                failure_summaries=[zero_reason],
                zero_plan_reason=zero_reason,
                recommended_route=report.recommended_route or "acquisition",
                next_actions=["Pivot to fallback execution or alternate acquisition strategy; do not retry the same blocked route unchanged."],
            )
            applied_summary = self.apply_output(state, tools, output)
            self.record_diary(state, tools, applied_summary)
            return applied_summary
        reports = [
            tools.preflight_experiment_plan(plan, state.external_artifacts, state.capability_matrix)
            for plan in pending_plans
        ]
        capability_matrix = tools.probe_capability_matrix(
            artifacts=state.external_artifacts,
            repository_paths=[repo.local_path for repo in state.repositories],
            environment_path=state.capability_matrix.environment_path if state.capability_matrix else None,
        )
        failure_summaries = [
            f"Preflight blocked {report.plan_id}: {report.blocking_reason}"
            for report in reports
            if not report.passed and report.blocking_reason
        ]
        summary = (
            "Preflight validated pending experiment plans. "
            f"passed={sum(1 for report in reports if report.passed)} "
            f"blocked={sum(1 for report in reports if not report.passed)}."
        )
        output = PreflightValidationPhaseOutput(
            summary=summary,
            preflight_reports=reports,
            capability_matrix=capability_matrix,
            failure_summaries=failure_summaries,
            recommended_route=next((report.recommended_route for report in reports if report.recommended_route), None),
            next_actions=[
                "Launch only preflight-passed plans."
                if any(report.passed for report in reports)
                else "Route back to acquisition or environment repair using failed preflight checks."
            ],
        )
        applied_summary = self.apply_output(state, tools, output)
        self.record_diary(state, tools, applied_summary)
        return applied_summary

    def apply_output(self, state: ResearchState, tools: ResearchTools, output: PreflightValidationPhaseOutput) -> str:
        state.preflight_reports = upsert_by_attr(state.preflight_reports, output.preflight_reports, "report_id")
        state.failure_summaries = dedupe_strings([*state.failure_summaries, *output.failure_summaries])
        state.next_actions = output.next_actions
        if output.capability_matrix is not None:
            state.capability_matrix = output.capability_matrix
            tools.memory.record_capability_matrix(output.capability_matrix)
        report_by_plan = {report.plan_id: report for report in output.preflight_reports}
        for plan in state.experiment_plans:
            report = report_by_plan.get(plan.plan_id)
            if report is None:
                continue
            plan.preflight_status = "passed" if report.passed else "failed"
            if not report.passed:
                plan.status = "blocked"
            tools.memory.record_preflight_report(report)
        return output.summary


class ExperimentAgent(BaseResearchAgent):
    name = "ExperimentAgent"
    phase = ResearchPhase.EXPERIMENT
    output_model = ExperimentPhaseOutput

    def build_instructions(self, state: ResearchState) -> str:
        return """
You are the experiment execution specialist.
Use tools to run the planned experiments, parse their outputs, and return only observed results.

You must:
- execute setup and launch commands only from preflight-passed experiment plans
- parse metrics from produced files or logs
- record failures when commands or metrics extraction fail
- do not perform acquisition or repair work here; failed prerequisites should stay blocked and route back to acquisition/preflight
- only report a plan as completed when its intended execution actually ran and produced observed outputs; setup failures are blockers, not completions
- validate code-level path semantics before launch: when a repository expects both a dataset filename and a dataset root, ensure the launch command contains the correct local root instead of falling back to placeholder defaults discovered in repository code or configs

Rules:
- Do not fabricate metrics or success claims.
- Prefer the newest planned experiments that have not been completed.
- Update best-known results only from actual parsed outputs.
"""

    def build_payload(self, state: ResearchState, tools: ResearchTools) -> dict[str, Any]:
        pending_plans = [
            item.model_dump(mode="python")
            for item in state.experiment_plans
            if item.status == "planned" and item.job_kind == "experiment" and item.preflight_status == "passed"
        ]
        return {
            "research_brief": state.research_brief.model_dump(mode="python"),
            "pending_experiment_plans": pending_plans[-4:],
            "existing_experiments": [item.model_dump(mode="python") for item in state.experiment_records[-10:]],
            "preflight_reports": [item.model_dump(mode="python") for item in state.preflight_reports[-10:]],
            "program_candidates": [item.model_dump(mode="python") for item in state.program_candidates[-10:]],
            "repositories": [item.model_dump(mode="python") for item in state.repositories[-10:]],
            "external_artifacts": [item.model_dump(mode="python") for item in state.external_artifacts[-20:]],
            "managed_env_root": str(tools.managed_env_root),
            "best_known_results": state.best_known_results,
        }

    def apply_output(self, state: ResearchState, tools: ResearchTools, output: ExperimentPhaseOutput) -> str:
        actual_experiments = [item for item in output.experiment_records if item.job_kind == "experiment"]
        auxiliary_records = [item for item in output.experiment_records if item.job_kind != "experiment"]
        state.experiment_records = upsert_by_attr(state.experiment_records, actual_experiments, "experiment_id")
        state.execution_records = upsert_by_attr(state.execution_records, auxiliary_records, "experiment_id")
        state.best_known_results.update(output.best_known_results)
        state.failure_summaries = dedupe_strings([*state.failure_summaries, *output.failure_summaries])
        state.next_actions = output.next_actions
        latest_record_by_plan = {record.plan_id: record for record in output.experiment_records}
        for plan in state.experiment_plans:
            record = latest_record_by_plan.get(plan.plan_id)
            if record is None:
                continue
            if record.status == "completed":
                plan.status = "completed"
            elif "block" in record.status or "setup" in record.status:
                plan.status = "blocked"
            else:
                plan.status = "failed"
        for item in actual_experiments:
            tools.memory.record_experiment(item)
            tools.memory.update_program_result(
                program_id=item.program_id,
                status="evaluated" if item.status == "completed" else item.status,
                metrics=item.metrics,
                failure_reason="; ".join(item.failure_modes) if item.failure_modes else None,
            )
        for item in auxiliary_records:
            tools.memory.record_execution(item)
        return output.summary


class ReflectionAgent(BaseResearchAgent):
    name = "ReflectionAgent"
    phase = ResearchPhase.REFLECTION
    output_model = ReflectionPhaseOutput

    def build_instructions(self, state: ResearchState) -> str:
        return """
You are the reflection specialist.
Assess whether the latest research cycle produced meaningful progress and decide whether the system should continue iterating.

Rules:
- Compare against actual baseline or parent-program evidence when available.
- Distinguish method-level gains from accidental engineering artifacts.
- If progress is insufficient or blocked, say why and propose the next move.
- Return machine-readable control signals inside each reflection record:
  - recommended_route_id
  - preferred_recovery_strategies
  - forbidden_attempt_signatures
  - blocked_entities
  - material_change_required
  - escalation_required
- If the same infrastructure blocker repeated without new evidence, mark escalation_required or force a pivot to a different strategy.
"""

    def build_payload(self, state: ResearchState, tools: ResearchTools) -> dict[str, Any]:
        return {
            "research_brief": state.research_brief.model_dump(mode="python"),
            "cycle_index": state.cycle_index,
            "selected_baseline_program_id": state.selected_baseline_program_id,
            "hypotheses": [item.model_dump(mode="python") for item in state.hypotheses[-4:]],
            "method_designs": [item.model_dump(mode="python") for item in state.method_designs[-4:]],
            "program_candidates": [item.model_dump(mode="python") for item in state.program_candidates[-8:]],
            "experiment_records": [item.model_dump(mode="python") for item in state.experiment_records[-10:]],
            "best_known_results": state.best_known_results,
            "failure_summaries": state.failure_summaries[-12:],
            "classified_failures": [item.model_dump(mode="python") for item in state.classified_failures[-12:]],
            "capability_matrix": state.capability_matrix.model_dump(mode="python") if state.capability_matrix else None,
            "blocker_registry": [item.model_dump(mode="python") for item in state.blocker_registry[-12:]],
            "route_history": [item.model_dump(mode="python") for item in state.route_history[-8:]],
            "cycle_deltas": [item.model_dump(mode="python") for item in state.cycle_deltas[-4:]],
            "hitl_events": [item.model_dump(mode="python") for item in state.hitl_events[-6:]],
            "manual_asset_roots": state.manual_asset_roots,
            "skipped_target_entities": state.skipped_target_entities,
            "human_guidance_notes": state.human_guidance_notes[-12:],
        }

    def apply_output(self, state: ResearchState, tools: ResearchTools, output: ReflectionPhaseOutput) -> str:
        state.reflections = upsert_by_attr(state.reflections, output.reflections, "reflection_id")
        state.next_actions = output.next_actions
        if output.terminate_research and output.reflections:
            state.termination_decision = output.reflections[-1].stop_reason or output.reflections[-1].verdict
        for reflection in output.reflections:
            if not reflection.linked_failure_ids and state.classified_failures:
                reflection.linked_failure_ids = [item.failure_id for item in state.classified_failures[-5:]]
            self.record_semantic_notes(state, tools, reflection.accepted_lessons)
        return output.summary
