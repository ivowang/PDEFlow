from __future__ import annotations

from typing import Any

from state import (
    CapabilityMatrix,
    CodingPhaseOutput,
    ExperimentPhaseOutput,
    ExperimentPlanningPhaseOutput,
    PreflightValidationPhaseOutput,
    ReflectionPhaseOutput,
    ResearchPhase,
    ResearchState,
)
from tools import ResearchTools
from common import dedupe_strings, upsert_by_attr
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
        }

    def apply_output(self, state: ResearchState, tools: ResearchTools, output: ExperimentPlanningPhaseOutput) -> str:
        state.experiment_plans = upsert_by_attr(state.experiment_plans, output.experiment_plans, "plan_id")
        state.next_actions = output.next_actions
        for item in output.experiment_plans:
            tools.memory.record_experiment_plan(item)
        return output.summary


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
