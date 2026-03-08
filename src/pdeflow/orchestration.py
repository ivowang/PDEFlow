from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import traceback

from .agents import (
    AcquisitionAgent,
    CoderAgent,
    DiagnosisAgent,
    ExperimentAgent,
    ExperimentPlannerAgent,
    HypothesisAgent,
    LiteratureAgent,
    MethodDesignAgent,
    ProblemFramingAgent,
    ReflectionAgent,
    ReporterAgent,
)
from .config import SystemConfig
from .memory import ResearchMemory
from .runtime import RuntimeAdapter
from .schemas import ResearchPhase, ResearchState
from .tools import ResearchTools


@dataclass(frozen=True)
class PhaseSpec:
    phase: ResearchPhase
    agent_key: str
    outputs: tuple[str, ...]


class ResearchManager:
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
            return [
                f"Acquisition status: {len(state.external_artifacts)} artifacts, {len(state.repositories)} repositories, selected baseline={state.selected_baseline_program_id or 'unset'}."
            ]
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

    def _run_phase(self, state: ResearchState, spec: PhaseSpec) -> str:
        state.current_phase = spec.phase
        state.phase_history.append(spec.phase.value)
        agent = self.agents[spec.agent_key]
        self._log(
            f"Starting phase {spec.phase.value} with {agent.name}. Cycle={state.cycle_index}."
        )
        try:
            summary = agent.run(state, self.tools, self.runtime)
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
            for spec in self.iterative_phases:
                self._run_phase(state, spec)
            if not self._should_continue(state):
                break
        self._run_phase(state, self.reporting_phase)
        self.memory.save_state(state, label="final_state")
        self._log(
            f"Research run finished. cycles={state.cycle_index}, repositories={len(state.repositories)}, programs={len(state.program_candidates)}, experiments={len(state.experiment_records)}, reports={len(state.generated_reports)}."
        )
        return state
