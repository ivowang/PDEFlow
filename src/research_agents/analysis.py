from __future__ import annotations

from typing import Any

from common import upsert_by_attr
from state import (
    DiagnosisPhaseOutput,
    HypothesisPhaseOutput,
    MethodDesignPhaseOutput,
    ProblemFramingPhaseOutput,
    ResearchPhase,
    ResearchState,
)
from tools import ResearchTools

from .base import BaseResearchAgent


class ResearchStrategistAgent(BaseResearchAgent):
    name = "ResearchStrategistAgent"

    _OUTPUT_BY_PHASE = {
        ResearchPhase.PROBLEM_FRAMING: ProblemFramingPhaseOutput,
        ResearchPhase.DIAGNOSIS: DiagnosisPhaseOutput,
        ResearchPhase.HYPOTHESIS: HypothesisPhaseOutput,
        ResearchPhase.METHOD_DESIGN: MethodDesignPhaseOutput,
    }

    def __init__(self, phase: ResearchPhase):
        self.phase = phase
        self.output_model = self._OUTPUT_BY_PHASE[phase]

    def allowed_tool_names(self) -> set[str] | None:
        return set()

    def runtime_timeout_seconds(self) -> int | None:
        return 300

    def build_instructions(self, state: ResearchState) -> str:
        if self.phase == ResearchPhase.PROBLEM_FRAMING:
            return """
You are the research strategist.
For the problem-framing phase, synthesize the brief, literature, acquisition state, and recent memory into a concrete research framing.

You must:
- define what constitutes success
- separate method innovation from engineering tuning
- identify candidate directions worth testing
- use recent lessons and evaluation memories so the framing reflects what the system has already learned

Rules:
- Ground every direction in observed limitations or constraints.
- Explicitly explain why a tuning-only direction is not a method innovation.
- Return concise but concrete evaluation criteria.
"""
        if self.phase == ResearchPhase.DIAGNOSIS:
            return """
You are the research strategist.
For the diagnosis phase, identify the main bottlenecks that currently constrain progress.

You may diagnose from:
- literature limitations
- acquired repository structure
- baseline program structure
- experiment, preflight, and reflection memories

Rules:
- Do not fabricate metrics that were never observed.
- Focus on bottlenecks that meaningfully constrain research progress.
"""
        if self.phase == ResearchPhase.HYPOTHESIS:
            return """
You are the research strategist.
For the hypothesis phase, propose testable research hypotheses grounded in literature, diagnosis, evaluation history, and accumulated lessons.

Rules:
- Do not propose pure hyperparameter sweeps as the main research hypothesis.
- Each hypothesis must specify expected gains, risks, code changes, and an evaluation plan.
- Reuse lessons from prior accepted/rejected attempts so the proposal evolves instead of repeating itself.
"""
        return """
You are the research strategist.
For the method-design phase, turn the selected hypothesis into a concrete method design that can be implemented in code.

Rules:
- Specify architecture, loss, data, training, inference, and physics integration changes where relevant.
- The design must point back to a parent program or baseline when possible.
- Keep the design concrete enough that the engineering agent can implement it.
- Reflect prior lessons and evaluation evidence instead of designing in a vacuum.
"""

    def build_payload(self, state: ResearchState, tools: ResearchTools) -> dict[str, Any]:
        compact_brief = {
            "title": state.research_brief.title[:180],
            "question": state.research_brief.question[:300],
            "objectives": [item[:140] for item in state.research_brief.objectives[:3]],
            "constraints": [item[:140] for item in state.research_brief.constraints[:4]],
            "domain_tags": state.research_brief.domain_tags[:6],
        }
        shared_memory = {
            "recent_evaluations": [
                {
                    "title": item.title,
                    "summary": item.summary[:220],
                    "phase": item.phase,
                    "verdict": item.verdict,
                    "support_level": item.support_level,
                    "metrics": item.metrics,
                }
                for item in state.evaluation_memos[-6:]
            ],
            "recent_memory_notes": [
                {
                    "kind": item.kind,
                    "title": item.title,
                    "summary": item.summary[:220],
                    "phase": item.phase,
                }
                for item in state.memory_notes[-10:]
            ],
            "recent_reflections": [
                {
                    "title": item.title,
                    "summary": item.summary[:220],
                    "verdict": item.verdict,
                    "continue_research": item.continue_research,
                }
                for item in state.reflections[-4:]
            ],
            "semantic_memory_notes": state.semantic_memory_notes[-10:],
        }
        if self.phase == ResearchPhase.PROBLEM_FRAMING:
            return {
                "research_brief": compact_brief,
                "literature_notes": [
                    {
                        "title": note.title,
                        "method_family": note.method_family,
                        "physics_level": note.physics_level,
                        "strengths": note.strengths[:2],
                        "limitations": note.limitations[:2],
                        "research_opportunities": note.research_opportunities[:2],
                    }
                    for note in state.literature_notes[:6]
                ],
                "repositories": [
                    {
                        "name": repo.name,
                        "remote_url": repo.remote_url,
                        "bootstrap_status": repo.bootstrap_status,
                        "entrypoints": repo.entrypoints[:4],
                    }
                    for repo in state.repositories[:6]
                ],
                "baseline_programs": [
                    {
                        "program_id": program.program_id,
                        "title": program.title,
                        "status": program.status,
                        "repo_id": program.repo_id,
                        "summary": program.summary[:220],
                    }
                    for program in state.program_candidates
                    if program.status.startswith("baseline")
                ],
                **shared_memory,
            }
        if self.phase == ResearchPhase.DIAGNOSIS:
            return {
                "research_brief": compact_brief,
                "problem_framing_notes": state.problem_framing_notes[:8],
                "evaluation_criteria": state.evaluation_criteria[:6],
                "candidate_directions": [item.model_dump(mode="python") for item in state.candidate_directions[:6]],
                "program_candidates": [
                    {
                        "program_id": item.program_id,
                        "title": item.title,
                        "status": item.status,
                        "summary": item.summary[:220],
                    }
                    for item in state.program_candidates[:8]
                ],
                "experiment_records": [
                    {
                        "plan_id": item.plan_id,
                        "status": item.status,
                        "observations": item.observations[:3],
                        "failure_modes": item.failure_modes[:3],
                        "metrics": item.metrics,
                    }
                    for item in state.experiment_records[-4:]
                ],
                **shared_memory,
            }
        if self.phase == ResearchPhase.HYPOTHESIS:
            return {
                "research_brief": compact_brief,
                "bottlenecks": state.bottleneck_analysis[:8],
                "candidate_directions": [item.model_dump(mode="python") for item in state.candidate_directions[:6]],
                "existing_hypotheses": [item.model_dump(mode="python") for item in state.hypotheses[-4:]],
                "selected_baseline_program_id": state.selected_baseline_program_id,
                **shared_memory,
            }
        return {
            "research_brief": compact_brief,
            "latest_hypotheses": [item.model_dump(mode="python") for item in state.hypotheses[-3:]],
            "available_programs": [
                {
                    "program_id": item.program_id,
                    "title": item.title,
                    "status": item.status,
                    "summary": item.summary[:220],
                }
                for item in state.program_candidates[-6:]
            ],
            "repositories": [
                {
                    "name": item.name,
                    "remote_url": item.remote_url,
                    "bootstrap_status": item.bootstrap_status,
                    "entrypoints": item.entrypoints[:4],
                }
                for item in state.repositories[-6:]
            ],
            **shared_memory,
        }

    def apply_output(self, state: ResearchState, tools: ResearchTools, output: Any) -> str:
        if self.phase == ResearchPhase.PROBLEM_FRAMING:
            state.problem_framing_notes = output.problem_framing_notes
            state.evaluation_criteria = output.evaluation_criteria
            state.candidate_directions = upsert_by_attr(state.candidate_directions, output.candidate_directions, "direction_id")
            state.next_actions = output.next_actions
            return output.summary
        if self.phase == ResearchPhase.DIAGNOSIS:
            state.bottleneck_analysis = output.bottleneck_analysis
            state.next_actions = output.next_actions
            self.record_semantic_notes(state, tools, output.semantic_notes)
            return output.summary
        if self.phase == ResearchPhase.HYPOTHESIS:
            state.hypotheses = upsert_by_attr(state.hypotheses, output.hypotheses, "hypothesis_id")
            state.next_actions = output.next_actions
            for item in output.hypotheses:
                tools.memory.record_idea(item)
            return output.summary
        state.method_designs = upsert_by_attr(state.method_designs, output.method_designs, "design_id")
        state.next_actions = output.next_actions
        return output.summary


class ProblemFramingAgent(ResearchStrategistAgent):
    def __init__(self) -> None:
        super().__init__(ResearchPhase.PROBLEM_FRAMING)


class DiagnosisAgent(ResearchStrategistAgent):
    def __init__(self) -> None:
        super().__init__(ResearchPhase.DIAGNOSIS)


class HypothesisAgent(ResearchStrategistAgent):
    def __init__(self) -> None:
        super().__init__(ResearchPhase.HYPOTHESIS)


class MethodDesignAgent(ResearchStrategistAgent):
    def __init__(self) -> None:
        super().__init__(ResearchPhase.METHOD_DESIGN)
