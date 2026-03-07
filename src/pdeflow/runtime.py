from __future__ import annotations

import json
import os
from typing import Any, TypeVar

from pydantic import BaseModel

try:
    from agents import Agent, RunConfig, Runner, SQLiteSession
except ImportError:  # pragma: no cover
    Agent = None
    RunConfig = None
    Runner = None
    SQLiteSession = None


OutputT = TypeVar("OutputT", bound=BaseModel)


class RuntimeAdapter:
    """Thin wrapper over OpenAI Agents SDK structured specialist execution."""

    def __init__(self, backend: str, model: str, session_db_path: str):
        self.backend = backend
        self.model = model
        self.session_db_path = session_db_path

    def ensure_ready(self) -> None:
        if self.backend != "openai_agents":
            raise RuntimeError(
                "This repository no longer ships a mock runtime. "
                "Set runtime.backend to 'openai_agents'."
            )
        if Agent is None or Runner is None:
            raise RuntimeError(
                "openai-agents is not installed in the active environment. Run `uv sync` first."
            )
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Export it before running the autonomous research workflow."
            )

    def _build_session(self, session_id: str) -> Any | None:
        if SQLiteSession is None:
            return None
        try:
            return SQLiteSession(session_id, self.session_db_path)
        except TypeError:
            try:
                return SQLiteSession(session_id=session_id, db_path=self.session_db_path)
            except TypeError:
                return None

    def run_structured(
        self,
        specialist_name: str,
        instructions: str,
        payload: dict[str, Any],
        session_id: str,
        output_type: type[OutputT],
        tools: list[Any] | None = None,
    ) -> OutputT:
        self.ensure_ready()
        agent = Agent(
            name=specialist_name,
            instructions=instructions,
            model=self.model,
            output_type=output_type,
            tools=tools or [],
        )
        run_config = RunConfig(workflow_name=f"pdeflow::{specialist_name.lower()}") if RunConfig else None
        session = self._build_session(session_id)
        result = Runner.run_sync(
            agent,
            json.dumps(payload, ensure_ascii=False, indent=2),
            session=session,
            run_config=run_config,
        )
        final_output = result.final_output
        if isinstance(final_output, output_type):
            return final_output
        if isinstance(final_output, BaseModel):
            return output_type.model_validate(final_output.model_dump(mode="python"))
        return output_type.model_validate(final_output)
