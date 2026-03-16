from __future__ import annotations

from abc import ABC, abstractmethod
import threading
import time
from typing import Any

from pydantic import BaseModel

from runtime import RuntimeAdapter
from state import DiaryEntry, MemoryKind, MemoryNote, ResearchPhase, ResearchState
from tools import ResearchTools
from common import now_utc, short_hash, upsert_by_attr


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
        stored_note = tools.memory.record_memory_note(
            MemoryNote(
                note_id=entry.entry_id,
                kind=MemoryKind.EPISODE,
                title=entry.title,
                summary=entry.body[:240],
                body=entry.body,
                cycle_index=state.cycle_index,
                phase=self.phase.value,
                tags=[self.phase.value, self.name],
            )
        )
        state.memory_notes = upsert_by_attr(state.memory_notes, [stored_note], "note_id")
        tools.memory.record_episode(label=entry.title, body=entry.body, phase=self.phase)

    def record_semantic_notes(self, state: ResearchState, tools: ResearchTools, notes: list[str]) -> None:
        for note in notes:
            if note not in state.semantic_memory_notes:
                state.semantic_memory_notes.append(note)
                tools.memory.record_semantic(note=note, source=self.name)

    def allowed_tool_names(self) -> set[str] | None:
        return None

    def max_turns(self) -> int | None:
        return None

    def runtime_timeout_seconds(self) -> int | None:
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
        started_at = time.monotonic()
        stop_heartbeat = threading.Event()
        heartbeat_interval_seconds = 60

        def _heartbeat() -> None:
            while not stop_heartbeat.wait(heartbeat_interval_seconds):
                elapsed = int(time.monotonic() - started_at)
                message = (
                    f"{self.name} is still working on {self.phase.value}. "
                    f"elapsed={elapsed}s cycle={state.cycle_index}."
                )
                tools.memory.record_agent_event(
                    agent_name=self.name,
                    phase=self.phase,
                    status="heartbeat",
                    cycle_index=state.cycle_index,
                    content=message,
                    payload={"elapsed_seconds": elapsed},
                    print_to_terminal=False,
                )
                tools.memory.record_process(message, print_to_terminal=True)

        heartbeat_thread = threading.Thread(
            target=_heartbeat,
            name=f"{self.name}-heartbeat",
            daemon=True,
        )
        heartbeat_thread.start()
        try:
            output = runtime.run_structured(
                specialist_name=self.name,
                instructions=self.build_instructions(state),
                payload=self.build_payload(state, tools),
                session_id=f"{state.run_name}-{self.phase.value}-cycle-{state.cycle_index}",
                output_type=self.output_model,
                tools=self.build_tools(tools),
                max_turns=self.max_turns(),
                runtime_timeout_seconds=self.runtime_timeout_seconds() or runtime.runtime_config.request_timeout_seconds,
            )
        finally:
            stop_heartbeat.set()
            heartbeat_thread.join(timeout=1)
        summary = self.apply_output(state, tools, output)
        self.record_diary(state, tools, summary)
        return summary
