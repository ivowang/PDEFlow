from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from .provider import Agent, MaxTurnsExceeded, Runner, RuntimeProviderMixin
from .structured_output import OutputT, StructuredOutputMixin


class RuntimeAdapter(RuntimeProviderMixin, StructuredOutputMixin):
    """Thin wrapper over OpenAI Agents SDK structured specialist execution."""

    def _run_sync_with_session(
        self,
        agent: Any,
        payload: str,
        session: Any,
        run_config: Any,
        max_turns: int,
    ) -> Any:
        return Runner.run_sync(
            agent,
            payload,
            max_turns=max_turns,
            session=session,
            run_config=run_config,
        )

    def _retry_finalize_after_max_turns(
        self,
        *,
        specialist_name: str,
        instructions: str,
        payload: dict[str, Any],
        session_id: str,
        output_type: type[OutputT],
        run_config: Any,
        original_error: Exception,
    ) -> OutputT:
        finalizer = Agent(
            name=f"{specialist_name}Finalizer",
            instructions=self._build_prompt_json_instructions(
                instructions.strip()
                + "\n\n"
                + "The prior tool-using run exceeded the max turn budget.\n"
                + "Do not call any tools.\n"
                + "Use only the existing session context and tool results already gathered.\n"
                + "Now produce the final structured phase output.",
                output_type,
            ),
            model=self.model,
            output_type=None,
            tools=[],
        )
        retry_payload = json.dumps(
            {
                "original_payload": payload,
                "finalization_reason": "max_turns_exceeded",
                "max_turns_error": str(original_error),
            },
            ensure_ascii=False,
            indent=2,
        )
        retry_result = self._run_sync_with_session(
            finalizer,
            retry_payload,
            session=self._build_session(f"{session_id}-finalize"),
            run_config=run_config,
            max_turns=max(4, min(8, self.runtime_config.max_turns // 4 or 4)),
        )
        return self._validate_prompt_json_output(retry_result.final_output, output_type)

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
        serialized_payload = json.dumps(payload, ensure_ascii=False, indent=2)
        try:
            result = self._run_sync_with_session(
                agent,
                serialized_payload,
                session=session,
                run_config=run_config,
                max_turns=self.runtime_config.max_turns,
            )
        except Exception as error:
            if (
                MaxTurnsExceeded is not None
                and isinstance(error, MaxTurnsExceeded)
                and use_prompt_json_output
                and tools
            ):
                try:
                    return self._retry_finalize_after_max_turns(
                        specialist_name=specialist_name,
                        instructions=instructions,
                        payload=payload,
                        session_id=session_id,
                        output_type=output_type,
                        run_config=run_config,
                        original_error=error,
                    )
                except Exception:
                    raise error
            raise
        final_output = result.final_output
        if use_prompt_json_output:
            return self._validate_prompt_json_output(final_output, output_type)
        if isinstance(final_output, output_type):
            return final_output
        if isinstance(final_output, BaseModel):
            return output_type.model_validate(final_output.model_dump(mode="python"))
        return output_type.model_validate(final_output)
