from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from runtime import RuntimeAdapter
from state import DiaryEntry, ResearchPhase, ResearchState
from tools import ResearchTools
from common import now_utc, short_hash


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

    def allowed_tool_names(self) -> set[str] | None:
        return None

    def build_tools(self, tools: ResearchTools) -> list[Any]:
        return tools.build_function_tools(self.allowed_tool_names())

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
