from __future__ import annotations

from typing import Any

from state import (
    DiagnosisPhaseOutput,
    HypothesisPhaseOutput,
    MethodDesignPhaseOutput,
    ProblemFramingPhaseOutput,
    ResearchPhase,
    ResearchState,
)
from tools import ResearchTools
from common import upsert_by_attr
from .base import BaseResearchAgent


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
