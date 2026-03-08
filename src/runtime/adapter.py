from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from .provider import Agent, Runner, RuntimeProviderMixin
from .structured_output import OutputT, StructuredOutputMixin


class RuntimeAdapter(RuntimeProviderMixin, StructuredOutputMixin):
    """Thin wrapper over OpenAI Agents SDK structured specialist execution."""

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
        use_prompt_json_output = self._should_use_prompt_json_output()
        agent = Agent(
            name=specialist_name,
            instructions=(
                self._build_prompt_json_instructions(instructions, output_type)
                if use_prompt_json_output
                else instructions
            ),
            model=self.model,
            output_type=None if use_prompt_json_output else output_type,
            tools=tools or [],
        )
        run_config = self._build_run_config(specialist_name)
        session = self._build_session(session_id)
        result = Runner.run_sync(
            agent,
            json.dumps(payload, ensure_ascii=False, indent=2),
            max_turns=self.runtime_config.max_turns,
            session=session,
            run_config=run_config,
        )
        final_output = result.final_output
        if use_prompt_json_output:
            return self._validate_prompt_json_output(final_output, output_type)
        if isinstance(final_output, output_type):
            return final_output
        if isinstance(final_output, BaseModel):
            return output_type.model_validate(final_output.model_dump(mode="python"))
        return output_type.model_validate(final_output)
