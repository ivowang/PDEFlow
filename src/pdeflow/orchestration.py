from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
        self.memory = ResearchMemory(root=repo_root / config.output_root / config.run_name)
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

    def _initial_state(self) -> ResearchState:
        state = ResearchState(
            project_name=self.config.project_name,
            run_name=self.config.run_name,
            research_brief=self.config.research_brief,
        )
        self.memory.save_state(state, label="initial_state")
        return state

    def _run_phase(self, state: ResearchState, spec: PhaseSpec) -> str:
        state.current_phase = spec.phase
        state.phase_history.append(spec.phase.value)
        agent = self.agents[spec.agent_key]
        summary = agent.run(state, self.tools, self.runtime)
        self.memory.record_phase(spec.phase, summary, list(spec.outputs))
        self.memory.save_state(state, label=spec.phase.value)
        return summary

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
        state = self._initial_state()
        for spec in self.front_phases:
            self._run_phase(state, spec)
        while True:
            state.cycle_index += 1
            for spec in self.iterative_phases:
                self._run_phase(state, spec)
            if not self._should_continue(state):
                break
        self._run_phase(state, self.reporting_phase)
        self.memory.save_state(state, label="final_state")
        return state
