from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from .runtime import RuntimeAdapter
from .schemas import (
    AcquisitionPhaseOutput,
    CodingPhaseOutput,
    DiagnosisPhaseOutput,
    DiaryEntry,
    ExperimentPhaseOutput,
    ExperimentPlanningPhaseOutput,
    HypothesisPhaseOutput,
    LiteraturePhaseOutput,
    MethodDesignPhaseOutput,
    ProblemFramingPhaseOutput,
    ReflectionPhaseOutput,
    ReportingPhaseOutput,
    ResearchPhase,
    ResearchState,
)
from .tools import ResearchTools
from .utils import dedupe_strings, now_utc, short_hash, upsert_by_attr


class BaseResearchAgent(ABC):
    name = "BaseResearchAgent"
    phase = ResearchPhase.LITERATURE_REVIEW
    output_model: type[BaseModel]

    def record_diary(self, state: ResearchState, tools: ResearchTools, summary: str) -> None:
        entry = DiaryEntry(
            entry_id=f"{self.phase.value}-{short_hash(state.run_name, summary, now_utc())}",
            phase=self.phase.value,
            title=f"{self.phase.value.replace('_', ' ').title()} completed",
            body=summary,
            tags=[self.phase.value, self.name],
        )
        state.research_diary.append(entry)
        tools.memory.record_diary(entry)
        tools.memory.record_episode(label=entry.title, body=entry.body, phase=self.phase)

    def record_semantic_notes(self, state: ResearchState, tools: ResearchTools, notes: list[str]) -> None:
        for note in notes:
            if note not in state.semantic_memory_notes:
                state.semantic_memory_notes.append(note)
                tools.memory.record_semantic(note=note, source=self.name)

    def build_tools(self, tools: ResearchTools) -> list[Any]:
        return tools.build_function_tools()

    @abstractmethod
    def build_instructions(self, state: ResearchState) -> str:
        raise NotImplementedError

    @abstractmethod
    def build_payload(self, state: ResearchState, tools: ResearchTools) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def apply_output(self, state: ResearchState, tools: ResearchTools, output: BaseModel) -> str:
        raise NotImplementedError

    def run(self, state: ResearchState, tools: ResearchTools, runtime: RuntimeAdapter) -> str:
        output = runtime.run_structured(
            specialist_name=self.name,
            instructions=self.build_instructions(state),
            payload=self.build_payload(state, tools),
            session_id=f"{state.run_name}-{self.phase.value}-cycle-{state.cycle_index}",
            output_type=self.output_model,
            tools=self.build_tools(tools),
        )
        summary = self.apply_output(state, tools, output)
        self.record_diary(state, tools, summary)
        return summary


class LiteratureAgent(BaseResearchAgent):
    name = "LiteratureAgent"
    phase = ResearchPhase.LITERATURE_REVIEW
    output_model = LiteraturePhaseOutput

    def build_instructions(self, state: ResearchState) -> str:
        return """
You are the literature specialist in an autonomous scientific research system.
Use the available tools to search for relevant papers, fetch evidence, and read enough source material to produce structured literature notes.

Rules:
- Do not invent papers, URLs, claims, or limitations. Use tools first.
- Prefer recent and foundational sources that directly support the current research question.
- Download and inspect PDFs when the abstract is insufficient to support a research claim.
- Extract limitations and research opportunities, not generic summaries.
- Return only structured output matching the schema.
"""

    def build_payload(self, state: ResearchState, tools: ResearchTools) -> dict[str, Any]:
        return {
            "research_brief": state.research_brief.model_dump(mode="python"),
            "already_known_papers": [note.title for note in state.literature_notes],
        }

    def apply_output(self, state: ResearchState, tools: ResearchTools, output: LiteraturePhaseOutput) -> str:
        state.literature_notes = upsert_by_attr(state.literature_notes, output.literature_notes, "paper_id")
        state.method_taxonomy = output.method_taxonomy
        state.open_questions = dedupe_strings([*state.open_questions, *output.open_questions])
        state.next_actions = output.next_actions
        for note in output.literature_notes:
            tools.memory.record_literature(note)
        self.record_semantic_notes(state, tools, output.semantic_notes)
        return output.summary


class AcquisitionAgent(BaseResearchAgent):
    name = "AcquisitionAgent"
    phase = ResearchPhase.ACQUISITION
    output_model = AcquisitionPhaseOutput

    def build_instructions(self, state: ResearchState) -> str:
        return """
You are the acquisition and environment specialist.
Your job is to decide what external artifacts are required to execute real research on the supplied problem, then use tools to acquire and inspect them.

You must:
- inspect secrets and local compute first
- search for relevant repositories, benchmark code, data sources, checkpoints, and documentation
- clone repositories and bootstrap environments when needed
- identify at least one viable baseline program candidate grounded in acquired assets
- when recent experiment or setup failures expose a concrete missing prerequisite, prioritize fixing that blocker before collecting optional new assets
- verify that downloaded datasets, checkpoints, and repositories exist at the exact local paths needed by upcoming experiment plans

Rules:
- Do not assume repository URLs, dataset locations, or checkpoint links unless a tool confirmed them.
- Treat environment setup as part of the research loop.
- Return only artifacts and repositories that were actually verified with tools.
"""

    def build_payload(self, state: ResearchState, tools: ResearchTools) -> dict[str, Any]:
        return {
            "research_brief": state.research_brief.model_dump(mode="python"),
            "literature_titles": [note.title for note in state.literature_notes[:12]],
            "open_questions": state.open_questions[:12],
            "existing_repositories": [repo.name for repo in state.repositories],
            "failure_summaries": state.failure_summaries[-12:],
            "recent_experiment_records": [record.model_dump(mode="python") for record in state.experiment_records[-8:]],
            "existing_experiment_plans": [plan.model_dump(mode="python") for plan in state.experiment_plans[-12:]],
            "work_directory": state.work_directory,
        }

    def apply_output(self, state: ResearchState, tools: ResearchTools, output: AcquisitionPhaseOutput) -> str:
        state.environment_snapshot = output.environment_snapshot
        state.secret_status = upsert_by_attr(state.secret_status, output.secret_status, "env_var")
        state.external_artifacts = upsert_by_attr(state.external_artifacts, output.external_artifacts, "artifact_id")
        state.repositories = upsert_by_attr(state.repositories, output.repositories, "repo_id")
        state.program_candidates = upsert_by_attr(state.program_candidates, output.program_candidates, "program_id")
        state.selected_baseline_program_id = output.selected_baseline_program_id or state.selected_baseline_program_id
        state.acquisition_notes = dedupe_strings([*state.acquisition_notes, *output.acquisition_notes])
        state.next_actions = output.next_actions
        for secret in output.secret_status:
            tools.memory.record_secret_status(secret)
        for artifact in output.external_artifacts:
            tools.memory.record_artifact(artifact)
        for repository in output.repositories:
            tools.memory.record_repository(repository)
        for program in output.program_candidates:
            tools.memory.register_program(program)
        self.record_semantic_notes(state, tools, output.semantic_notes + output.acquisition_notes)
        return output.summary


class ProblemFramingAgent(BaseResearchAgent):
    name = "ProblemFramingAgent"
    phase = ResearchPhase.PROBLEM_FRAMING
    output_model = ProblemFramingPhaseOutput

    def build_instructions(self, state: ResearchState) -> str:
        return """
You are the problem-framing specialist.
Synthesize the research brief, acquired assets, and literature into a concrete research framing.

You must:
- define what constitutes success
- separate method innovation from engineering tuning
- identify candidate directions worth testing

Rules:
- Ground every direction in observed limitations or constraints.
- Explicitly explain why a tuning-only direction is not a method innovation.
- Return concise but concrete evaluation criteria.
"""

    def build_payload(self, state: ResearchState, tools: ResearchTools) -> dict[str, Any]:
        return {
            "research_brief": state.research_brief.model_dump(mode="python"),
            "literature_notes": [note.model_dump(mode="python") for note in state.literature_notes[:8]],
            "repositories": [repo.model_dump(mode="python") for repo in state.repositories[:8]],
            "baseline_programs": [
                program.model_dump(mode="python")
                for program in state.program_candidates
                if program.status.startswith("baseline")
            ],
        }

    def apply_output(self, state: ResearchState, tools: ResearchTools, output: ProblemFramingPhaseOutput) -> str:
        state.problem_framing_notes = output.problem_framing_notes
        state.evaluation_criteria = output.evaluation_criteria
        state.candidate_directions = upsert_by_attr(state.candidate_directions, output.candidate_directions, "direction_id")
        state.next_actions = output.next_actions
        return output.summary


class DiagnosisAgent(BaseResearchAgent):
    name = "DiagnosisAgent"
    phase = ResearchPhase.DIAGNOSIS
    output_model = DiagnosisPhaseOutput

    def build_instructions(self, state: ResearchState) -> str:
        return """
You are the diagnosis specialist.
Identify the main bottlenecks that block progress on the research question.

You may diagnose from:
- literature limitations
- acquired repository structure
- baseline program structure
- existing experiment logs if available

Rules:
- Do not fabricate metrics that were never observed.
- Focus on bottlenecks that meaningfully constrain research progress.
"""

    def build_payload(self, state: ResearchState, tools: ResearchTools) -> dict[str, Any]:
        return {
            "research_brief": state.research_brief.model_dump(mode="python"),
            "problem_framing_notes": state.problem_framing_notes,
            "evaluation_criteria": state.evaluation_criteria,
            "candidate_directions": [item.model_dump(mode="python") for item in state.candidate_directions],
            "program_candidates": [item.model_dump(mode="python") for item in state.program_candidates],
            "experiment_records": [item.model_dump(mode="python") for item in state.experiment_records[-6:]],
        }

    def apply_output(self, state: ResearchState, tools: ResearchTools, output: DiagnosisPhaseOutput) -> str:
        state.bottleneck_analysis = output.bottleneck_analysis
        state.next_actions = output.next_actions
        self.record_semantic_notes(state, tools, output.semantic_notes)
        return output.summary


class HypothesisAgent(BaseResearchAgent):
    name = "HypothesisAgent"
    phase = ResearchPhase.HYPOTHESIS
    output_model = HypothesisPhaseOutput

    def build_instructions(self, state: ResearchState) -> str:
        return """
You are the hypothesis specialist.
Propose one or more testable research hypotheses grounded in literature, baseline diagnosis, and acquired assets.

Rules:
- Do not propose pure hyperparameter sweeps as the main research hypothesis.
- Each hypothesis must specify expected gains, risks, code changes, and an evaluation plan.
- Prefer hypotheses that can be executed with available repos, data, and compute.
"""

    def build_payload(self, state: ResearchState, tools: ResearchTools) -> dict[str, Any]:
        return {
            "research_brief": state.research_brief.model_dump(mode="python"),
            "bottlenecks": state.bottleneck_analysis,
            "candidate_directions": [item.model_dump(mode="python") for item in state.candidate_directions],
            "existing_hypotheses": [item.model_dump(mode="python") for item in state.hypotheses],
            "selected_baseline_program_id": state.selected_baseline_program_id,
        }

    def apply_output(self, state: ResearchState, tools: ResearchTools, output: HypothesisPhaseOutput) -> str:
        state.hypotheses = upsert_by_attr(state.hypotheses, output.hypotheses, "hypothesis_id")
        state.next_actions = output.next_actions
        for item in output.hypotheses:
            tools.memory.record_idea(item)
        return output.summary


class MethodDesignAgent(BaseResearchAgent):
    name = "MethodDesignAgent"
    phase = ResearchPhase.METHOD_DESIGN
    output_model = MethodDesignPhaseOutput

    def build_instructions(self, state: ResearchState) -> str:
        return """
You are the method-design specialist.
Turn the selected hypothesis into a concrete method design that can be implemented in code.

Rules:
- Specify architecture, loss, data, training, inference, and physics integration changes where relevant.
- The design must point back to a parent program or baseline when possible.
- Keep the design concrete enough that the coding agent can implement it.
"""

    def build_payload(self, state: ResearchState, tools: ResearchTools) -> dict[str, Any]:
        return {
            "research_brief": state.research_brief.model_dump(mode="python"),
            "latest_hypotheses": [item.model_dump(mode="python") for item in state.hypotheses[-3:]],
            "available_programs": [item.model_dump(mode="python") for item in state.program_candidates[-8:]],
            "repositories": [item.model_dump(mode="python") for item in state.repositories[-8:]],
        }

    def apply_output(self, state: ResearchState, tools: ResearchTools, output: MethodDesignPhaseOutput) -> str:
        state.method_designs = upsert_by_attr(state.method_designs, output.method_designs, "design_id")
        state.next_actions = output.next_actions
        return output.summary


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
- use verified local artifacts when they exist, and add concrete download/bootstrap setup steps when required data or checkpoints are not yet local
- avoid plans that point to nonexistent dataset or checkpoint paths without a preceding acquisition step grounded in verified artifacts
- prefer managed uv environments over ad hoc `python -m venv`, `source`, or bare `pip install`
- a blocked or setup-failed baseline does not count as a completed baseline
- if the selected baseline has no completed experiment record with real outputs, do not schedule downstream candidate launch plans except prerequisite acquisition/verification or matched baseline reruns

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
        }

    def apply_output(self, state: ResearchState, tools: ResearchTools, output: ExperimentPlanningPhaseOutput) -> str:
        state.experiment_plans = upsert_by_attr(state.experiment_plans, output.experiment_plans, "plan_id")
        state.next_actions = output.next_actions
        for item in output.experiment_plans:
            tools.memory.record_experiment_plan(item)
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
- execute setup and launch commands from the pending experiment plans
- parse metrics from produced files or logs
- record failures when commands or metrics extraction fail
- repair invalid environment setup on the fly when possible by provisioning a managed uv environment and re-running inside it
- prefer `ensure_python_environment`, `inspect_python_environment`, and `run_in_environment` over ad hoc `python -m venv`, `source`, or `pip` shell sequences
- if setup fails because a required dataset, checkpoint, repository file, or environment dependency is missing, attempt to acquire or repair that prerequisite with tools before declaring the plan blocked
- only report a plan as completed when its intended execution actually ran and produced observed outputs; setup failures are blockers, not completions

Rules:
- Do not fabricate metrics or success claims.
- Prefer the newest planned experiments that have not been completed.
- Update best-known results only from actual parsed outputs.
"""

    def build_payload(self, state: ResearchState, tools: ResearchTools) -> dict[str, Any]:
        pending_plans = [item.model_dump(mode="python") for item in state.experiment_plans if item.status == "planned"]
        return {
            "research_brief": state.research_brief.model_dump(mode="python"),
            "pending_experiment_plans": pending_plans[-4:],
            "existing_experiments": [item.model_dump(mode="python") for item in state.experiment_records[-10:]],
            "program_candidates": [item.model_dump(mode="python") for item in state.program_candidates[-10:]],
            "repositories": [item.model_dump(mode="python") for item in state.repositories[-10:]],
            "external_artifacts": [item.model_dump(mode="python") for item in state.external_artifacts[-20:]],
            "managed_env_root": str(tools.managed_env_root),
            "best_known_results": state.best_known_results,
        }

    def apply_output(self, state: ResearchState, tools: ResearchTools, output: ExperimentPhaseOutput) -> str:
        state.experiment_records = upsert_by_attr(state.experiment_records, output.experiment_records, "experiment_id")
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
        for item in output.experiment_records:
            tools.memory.record_experiment(item)
            tools.memory.update_program_result(
                program_id=item.program_id,
                status="evaluated" if item.status == "completed" else item.status,
                metrics=item.metrics,
                failure_reason="; ".join(item.failure_modes) if item.failure_modes else None,
            )
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
        }

    def apply_output(self, state: ResearchState, tools: ResearchTools, output: ReflectionPhaseOutput) -> str:
        state.reflections = upsert_by_attr(state.reflections, output.reflections, "reflection_id")
        state.next_actions = output.next_actions
        if output.terminate_research and output.reflections:
            state.termination_decision = output.reflections[-1].stop_reason or output.reflections[-1].verdict
        for reflection in output.reflections:
            self.record_semantic_notes(state, tools, reflection.accepted_lessons)
        return output.summary


class ReporterAgent(BaseResearchAgent):
    name = "ReporterAgent"
    phase = ResearchPhase.REPORTING
    output_model = ReportingPhaseOutput

    def build_instructions(self, state: ResearchState) -> str:
        return """
You are the reporting specialist.
Use the structured state to produce durable markdown artifacts for the research run.

You must write reports with tools. At minimum create:
- literature_review.md
- acquisition_report.md
- idea_diary.md
- experiment_summary.md
- final_report.md

Rules:
- Ground every claim in recorded state.
- Do not reference outputs that were never produced.
- Return the paths of the files you actually wrote.
"""

    def build_payload(self, state: ResearchState, tools: ResearchTools) -> dict[str, Any]:
        return {
            "research_brief": state.research_brief.model_dump(mode="python"),
            "literature_notes": [item.model_dump(mode="python") for item in state.literature_notes],
            "acquisition_notes": state.acquisition_notes,
            "repositories": [item.model_dump(mode="python") for item in state.repositories],
            "program_candidates": [item.model_dump(mode="python") for item in state.program_candidates],
            "hypotheses": [item.model_dump(mode="python") for item in state.hypotheses],
            "method_designs": [item.model_dump(mode="python") for item in state.method_designs],
            "experiment_records": [item.model_dump(mode="python") for item in state.experiment_records],
            "reflections": [item.model_dump(mode="python") for item in state.reflections],
            "next_actions": state.next_actions,
            "report_directory": str(tools.memory.reports_dir),
        }

    def apply_output(self, state: ResearchState, tools: ResearchTools, output: ReportingPhaseOutput) -> str:
        state.generated_reports = upsert_by_attr(state.generated_reports, output.generated_reports, "report_id")
        for report in output.generated_reports:
            tools.memory.record_report(report)
        return output.summary
