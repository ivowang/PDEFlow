from __future__ import annotations

from typing import Any

from state import ReportingPhaseOutput, ResearchPhase, ResearchState
from tools import ResearchTools
from common import upsert_by_attr
from .base import BaseResearchAgent


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
