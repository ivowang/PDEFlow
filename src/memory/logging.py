from __future__ import annotations

from pathlib import Path
from typing import Any

from common import append_jsonl, ensure_dir, now_utc, to_plain_data


class ResearchLogger:
    """Unified multi-granularity logger for research runs."""

    def __init__(self, root: Path, logs_dir: Path | None = None, process_path: Path | None = None):
        self.root = ensure_dir(root)
        self.logs_dir = ensure_dir(logs_dir or (self.root / "logs"))
        self.process_path = process_path or (self.root / "process.txt")
        self.debug_jsonl_path = self.logs_dir / "debug.jsonl"
        self.debug_log_path = self.logs_dir / "debug.log"
        self.core_jsonl_path = self.logs_dir / "core_progress.jsonl"
        self.core_log_path = self.logs_dir / "core_progress.log"
        self.agent_jsonl_path = self.logs_dir / "agent_activity.jsonl"
        self.agent_log_path = self.logs_dir / "agent_activity.log"
        self.phase_jsonl_path = self.logs_dir / "phase_events.jsonl"
        self.tool_jsonl_path = self.logs_dir / "tool_events.jsonl"

    def _line(self, message: str, timestamp: str | None = None) -> str:
        return f"[{timestamp or now_utc()}] {message}"

    def _append_text(self, path: Path, line: str) -> None:
        ensure_dir(path.parent)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def log_debug(
        self,
        message: str,
        *,
        category: str = "system",
        event_type: str = "message",
        payload: dict[str, Any] | None = None,
        print_to_terminal: bool = True,
        mirror_to_process: bool = True,
    ) -> None:
        timestamp = now_utc()
        event = {
            "timestamp": timestamp,
            "category": category,
            "event_type": event_type,
            "message": message,
            **to_plain_data(payload or {}),
        }
        append_jsonl(self.debug_jsonl_path, event)
        line = self._line(message, timestamp)
        self._append_text(self.debug_log_path, line)
        if mirror_to_process:
            self._append_text(self.process_path, line)
        if print_to_terminal:
            print(line, flush=True)

    def log_core_progress(
        self,
        message: str,
        *,
        kind: str = "milestone",
        phase: str | None = None,
        agent: str | None = None,
        cycle_index: int | None = None,
        payload: dict[str, Any] | None = None,
        print_to_terminal: bool = False,
    ) -> None:
        timestamp = now_utc()
        event = {
            "timestamp": timestamp,
            "kind": kind,
            "phase": phase,
            "agent": agent,
            "cycle_index": cycle_index,
            "message": message,
            **to_plain_data(payload or {}),
        }
        append_jsonl(self.core_jsonl_path, event)
        self._append_text(self.core_log_path, self._line(message, timestamp))
        self.log_debug(
            message,
            category="core_progress",
            event_type=kind,
            payload={"phase": phase, "agent": agent, "cycle_index": cycle_index, **(payload or {})},
            print_to_terminal=print_to_terminal,
            mirror_to_process=False,
        )

    def log_agent_event(
        self,
        *,
        agent_name: str,
        phase: str,
        status: str,
        cycle_index: int,
        content: str,
        payload: dict[str, Any] | None = None,
        print_to_terminal: bool = False,
    ) -> None:
        timestamp = now_utc()
        event = {
            "timestamp": timestamp,
            "agent_name": agent_name,
            "phase": phase,
            "status": status,
            "cycle_index": cycle_index,
            "content": content,
            **to_plain_data(payload or {}),
        }
        append_jsonl(self.agent_jsonl_path, event)
        line = self._line(
            f"AGENT {status.upper()} | agent={agent_name} phase={phase} cycle={cycle_index} | {content}",
            timestamp,
        )
        self._append_text(self.agent_log_path, line)
        self.log_debug(
            content,
            category="agent",
            event_type=status,
            payload={"agent_name": agent_name, "phase": phase, "cycle_index": cycle_index, **(payload or {})},
            print_to_terminal=print_to_terminal,
            mirror_to_process=False,
        )

    def log_phase_event(
        self,
        *,
        phase: str,
        summary: str,
        outputs: list[str],
        agent_name: str | None = None,
        cycle_index: int | None = None,
    ) -> None:
        append_jsonl(
            self.phase_jsonl_path,
            {
                "timestamp": now_utc(),
                "phase": phase,
                "agent_name": agent_name,
                "cycle_index": cycle_index,
                "summary": summary,
                "outputs": outputs,
            },
        )

    def log_tool_event(self, payload: dict[str, Any]) -> None:
        append_jsonl(self.tool_jsonl_path, payload)
        self.log_debug(
            payload.get("message", payload.get("tool", "tool_event")),
            category="tool",
            event_type=str(payload.get("tool", "tool_event")),
            payload=payload,
            print_to_terminal=False,
            mirror_to_process=False,
        )
