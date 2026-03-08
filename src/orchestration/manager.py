from __future__ import annotations

from pathlib import Path
import traceback

from research_agents import (
    AcquisitionAgent,
    CoderAgent,
    DiagnosisAgent,
    ExperimentAgent,
    ExperimentPlannerAgent,
    HypothesisAgent,
    LiteratureAgent,
    MethodDesignAgent,
    PreflightValidationAgent,
    ProblemFramingAgent,
    ReflectionAgent,
    ReporterAgent,
)
from config import SystemConfig
from integrations.command_grounding import ground_experiment_plan
from memory import ResearchMemory
from orchestration.failures import classify_state_failures
from runtime import RuntimeAdapter
from state import ResearchPhase, ResearchState
from tools import ResearchTools
from .routing import ManagerRoutingMixin
from .specs import CycleRoute, PhaseSpec

class ResearchManager(ManagerRoutingMixin):
    """Manager-centered orchestration for autonomous research execution."""

    def __init__(self, config: SystemConfig, repo_root: Path):
        self.config = config
        self.repo_root = repo_root
        self.work_directory = config.resolve_work_directory(repo_root)
        self.memory = ResearchMemory(root=self.work_directory)
        self.tools = ResearchTools(config=config, memory=self.memory, repo_root=repo_root)
        self.runtime = RuntimeAdapter(
            runtime_config=config.runtime,
            session_db_path=str(self.memory.sessions_db),
        )
        self.agents = {
            "literature": LiteratureAgent(),
            "acquisition": AcquisitionAgent(),
            "problem": ProblemFramingAgent(),
            "diagnosis": DiagnosisAgent(),
            "hypothesis": HypothesisAgent(),
            "design": MethodDesignAgent(),
            "coder": CoderAgent(),
            "planner": ExperimentPlannerAgent(),
            "preflight": PreflightValidationAgent(),
            "experiment": ExperimentAgent(),
            "reflection": ReflectionAgent(),
            "reporter": ReporterAgent(),
        }
        self.front_phases = [
            PhaseSpec(ResearchPhase.LITERATURE_REVIEW, "literature", ("literature_notes", "method_taxonomy", "open_questions")),
            PhaseSpec(ResearchPhase.ACQUISITION, "acquisition", ("environment_snapshot", "external_artifacts", "repositories")),
            PhaseSpec(ResearchPhase.PROBLEM_FRAMING, "problem", ("problem_framing_notes", "candidate_directions")),
            PhaseSpec(ResearchPhase.DIAGNOSIS, "diagnosis", ("bottleneck_analysis", "next_actions")),
        ]
        self.iterative_phases = [
            PhaseSpec(ResearchPhase.HYPOTHESIS, "hypothesis", ("hypotheses",)),
            PhaseSpec(ResearchPhase.METHOD_DESIGN, "design", ("method_designs",)),
            PhaseSpec(ResearchPhase.CODING, "coder", ("program_candidates",)),
            PhaseSpec(ResearchPhase.EXPERIMENT_PLANNING, "planner", ("experiment_plans",)),
            PhaseSpec(ResearchPhase.PREFLIGHT_VALIDATION, "preflight", ("preflight_reports", "capability_matrix")),
            PhaseSpec(ResearchPhase.EXPERIMENT, "experiment", ("experiment_records", "best_known_results")),
            PhaseSpec(ResearchPhase.REFLECTION, "reflection", ("reflections", "next_actions")),
        ]
        self.recovery_phases = [
            PhaseSpec(ResearchPhase.ACQUISITION, "acquisition", ("environment_snapshot", "external_artifacts", "repositories")),
            PhaseSpec(ResearchPhase.EXPERIMENT_PLANNING, "planner", ("experiment_plans",)),
            PhaseSpec(ResearchPhase.PREFLIGHT_VALIDATION, "preflight", ("preflight_reports", "capability_matrix")),
            PhaseSpec(ResearchPhase.EXPERIMENT, "experiment", ("experiment_records", "best_known_results")),
            PhaseSpec(ResearchPhase.REFLECTION, "reflection", ("reflections", "next_actions")),
        ]
        self.reporting_phase = PhaseSpec(ResearchPhase.REPORTING, "reporter", ("generated_reports",))

    def _log(self, message: str) -> None:
        self.memory.record_process(message)

    def _phase_snapshot(self, state: ResearchState, phase: ResearchPhase) -> list[str]:
        if phase == ResearchPhase.LITERATURE_REVIEW:
            return [
                f"Literature notes: {len(state.literature_notes)} papers, taxonomy entries: {len(state.method_taxonomy)}, open questions: {len(state.open_questions)}."
            ]
        if phase == ResearchPhase.ACQUISITION:
            lines = [
                f"Acquisition status: {len(state.external_artifacts)} artifacts, {len(state.repositories)} repositories, selected baseline={state.selected_baseline_program_id or 'unset'}."
            ]
            if state.capability_matrix:
                lines.append(
                    f"Capability matrix: baseline_ready={state.capability_matrix.baseline_ready_to_launch} "
                    f"target_dataset_ready={state.capability_matrix.target_dataset_ready}."
                )
            return lines
        if phase == ResearchPhase.PROBLEM_FRAMING:
            return [
                f"Problem framing produced {len(state.candidate_directions)} candidate directions and {len(state.evaluation_criteria)} evaluation criteria."
            ]
        if phase == ResearchPhase.DIAGNOSIS:
            return [
                f"Diagnosis recorded {len(state.bottleneck_analysis)} bottlenecks and {len(state.next_actions)} next actions."
            ]
        if phase == ResearchPhase.HYPOTHESIS:
            latest = state.hypotheses[-1] if state.hypotheses else None
            if latest:
                return [f"Latest hypothesis: {latest.hypothesis_id} | {latest.title}."]
        if phase == ResearchPhase.METHOD_DESIGN:
            latest = state.method_designs[-1] if state.method_designs else None
            if latest:
                return [f"Latest method design: {latest.design_id} | {latest.title}."]
        if phase == ResearchPhase.CODING:
            latest = state.program_candidates[-1] if state.program_candidates else None
            if latest:
                return [
                    f"Latest program candidate: {latest.program_id} | status={latest.status} | changed_files={len(latest.changed_files)}."
                ]
        if phase == ResearchPhase.EXPERIMENT_PLANNING:
            latest = state.experiment_plans[-1] if state.experiment_plans else None
            if latest:
                return [
                    f"Latest experiment plan: {latest.plan_id} | program={latest.program_id} | gpus={latest.gpu_ids}."
                ]
        if phase == ResearchPhase.PREFLIGHT_VALIDATION:
            latest = state.preflight_reports[-1] if state.preflight_reports else None
            if latest:
                return [
                    f"Latest preflight: {latest.plan_id} | passed={latest.passed} | failed_checks={len(latest.failed_checks)}."
                ]
        if phase == ResearchPhase.EXPERIMENT:
            latest = state.experiment_records[-1] if state.experiment_records else None
            if latest:
                return [
                    f"Latest experiment: {latest.experiment_id} | status={latest.status} | program={latest.program_id} | failures={len(latest.failure_modes)}."
                ]
        if phase == ResearchPhase.REFLECTION:
            latest = state.reflections[-1] if state.reflections else None
            if latest:
                lines = [
                    f"Reflection verdict: {latest.verdict} | continue_research={latest.continue_research}."
                ]
                if latest.evidence:
                    lines.append(f"Breakthrough evidence: {latest.evidence[0]}")
                if latest.stop_reason:
                    lines.append(f"Stop reason: {latest.stop_reason}")
                return lines
        if phase == ResearchPhase.REPORTING:
            return [f"Reports generated: {len(state.generated_reports)}."]
        return []

    def _initial_state(self) -> ResearchState:
        state = ResearchState(
            project_name=self.config.project_name,
            run_name=self.config.run_name,
            work_directory=str(self.work_directory),
            research_brief=self.config.research_brief,
        )
        self.memory.save_state(state, label="initial_state")
        return state

    def _ground_experiment_plans(self, state: ResearchState) -> None:
        grounded_plans = []
        for plan in state.experiment_plans:
            grounded_plan, messages = ground_experiment_plan(plan, state.external_artifacts)
            grounded_plans.append(grounded_plan)
            for message in messages:
                self._log(f"Plan grounding: {message}")
        state.experiment_plans = grounded_plans

    def _validate_artifacts(self, state: ResearchState) -> None:
        updated = []
        for artifact in state.external_artifacts:
            if not artifact.local_path:
                updated.append(artifact)
                continue
            local_path = Path(artifact.local_path)
            if artifact.artifact_type not in {"dataset", "checkpoint"} and local_path.suffix.lower() not in {
                ".h5",
                ".hdf5",
                ".pt",
                ".pth",
                ".ckpt",
            }:
                updated.append(artifact)
                continue
            validated = self.tools.validate_artifact_record(artifact, quarantine_on_failure=True)
            updated.append(validated)
            self.memory.record_artifact(validated)
        state.external_artifacts = updated

    def _refresh_capability_matrix(self, state: ResearchState) -> None:
        capability_matrix = self.tools.probe_capability_matrix(
            artifacts=state.external_artifacts,
            repository_paths=[repo.local_path for repo in state.repositories],
            environment_path=state.capability_matrix.environment_path if state.capability_matrix else None,
        )
        state.capability_matrix = capability_matrix
        self.memory.record_capability_matrix(capability_matrix)

    def _refresh_classified_failures(self, state: ResearchState) -> None:
        classified = classify_state_failures(
            state,
            max_transfer_attempts=self.config.retrieval.max_transfer_attempts,
        )
        existing = {failure.failure_id: failure for failure in state.classified_failures}
        for failure in classified:
            existing[failure.failure_id] = failure
            self.memory.record_failure(failure)
        state.classified_failures = list(existing.values())

    def _post_phase_sync(self, state: ResearchState, spec: PhaseSpec) -> None:
        if spec.phase in {ResearchPhase.ACQUISITION, ResearchPhase.PREFLIGHT_VALIDATION}:
            self._validate_artifacts(state)
        if spec.phase in {
            ResearchPhase.ACQUISITION,
            ResearchPhase.EXPERIMENT_PLANNING,
            ResearchPhase.PREFLIGHT_VALIDATION,
            ResearchPhase.EXPERIMENT,
        }:
            self._refresh_capability_matrix(state)
            self._refresh_classified_failures(state)

    def _run_phase(self, state: ResearchState, spec: PhaseSpec) -> str:
        if spec.phase in {ResearchPhase.EXPERIMENT_PLANNING, ResearchPhase.EXPERIMENT}:
            self._ground_experiment_plans(state)
        state.current_phase = spec.phase
        state.phase_history.append(spec.phase.value)
        agent = self.agents[spec.agent_key]
        self._log(
            f"Starting phase {spec.phase.value} with {agent.name}. Cycle={state.cycle_index}."
        )
        try:
            summary = agent.run(state, self.tools, self.runtime)
            if spec.phase in {ResearchPhase.EXPERIMENT_PLANNING, ResearchPhase.PREFLIGHT_VALIDATION}:
                self._ground_experiment_plans(state)
            self._post_phase_sync(state, spec)
            self.memory.record_phase(spec.phase, summary, list(spec.outputs))
            self.memory.save_state(state, label=spec.phase.value)
            self._log(
                f"Completed phase {spec.phase.value} with {agent.name}. Summary: {summary}"
            )
            for line in self._phase_snapshot(state, spec.phase):
                self._log(line)
            return summary
        except Exception:
            stack = traceback.format_exc()
            self._log(
                f"Phase {spec.phase.value} with {agent.name} failed. Traceback follows.\n{stack}"
            )
            self.memory.save_state(state, label=f"{spec.phase.value}_failed")
            raise

    def _should_continue(self, state: ResearchState) -> bool:
        if state.blocked_reason and state.termination_decision:
            return False
        if not state.reflections:
            return True
        latest = state.reflections[-1]
        if not latest.continue_research:
            return False
        if self.config.manager_safety_max_cycles is not None and state.cycle_index >= self.config.manager_safety_max_cycles:
            state.termination_decision = (
                f"Stopped after reaching manager_safety_max_cycles={self.config.manager_safety_max_cycles}."
            )
            return False
        return True

    def run(self) -> ResearchState:
        self.runtime.ensure_ready()
        self._log(
            f"Research run started. project={self.config.project_name}, run={self.config.run_name}, work_directory={self.work_directory}."
        )
        state = self._initial_state()
        for spec in self.front_phases:
            self._run_phase(state, spec)
        while True:
            state.cycle_index += 1
            self._log(f"Entering research cycle {state.cycle_index}.")
            route = self._select_cycle_route(state)
            self._log(
                "Manager selected cycle route: "
                f"{' -> '.join(spec.phase.value for spec in route.phases)}. "
                f"Reason: {route.reason}"
            )
            for spec in route.phases:
                if spec.phase == ResearchPhase.EXPERIMENT:
                    ready_plans = [
                        plan for plan in state.experiment_plans
                        if plan.status == "planned"
                        and plan.job_kind == "experiment"
                        and plan.preflight_status == "passed"
                    ]
                    if not ready_plans:
                        self._log(
                            "Skipping experiment phase because no preflight-passed experiment plans are available."
                        )
                        continue
                self._run_phase(state, spec)
            if not self._should_continue(state):
                break
        self._run_phase(state, self.reporting_phase)
        self.memory.save_state(state, label="final_state")
        self._log(
            f"Research run finished. cycles={state.cycle_index}, repositories={len(state.repositories)}, programs={len(state.program_candidates)}, experiments={len(state.experiment_records)}, reports={len(state.generated_reports)}."
        )
        return state
