from __future__ import annotations

import json
import re
import signal
from contextlib import contextmanager
from typing import Any

from pydantic import BaseModel

from .provider import Agent, MaxTurnsExceeded, Runner, RuntimeProviderMixin
from .structured_output import OutputT, StructuredOutputMixin


class RuntimeAdapter(RuntimeProviderMixin, StructuredOutputMixin):
    """Thin wrapper over OpenAI Agents SDK structured specialist execution."""

    def _build_run_config_with_budget(self, specialist_name: str, max_output_tokens: int) -> Any | None:
        try:
            return self._build_run_config(specialist_name, max_output_tokens=max_output_tokens)
        except TypeError:
            return self._build_run_config(specialist_name)

    @contextmanager
    def _runtime_timeout(self, timeout_seconds: int | None):
        if timeout_seconds is None or timeout_seconds <= 0:
            yield
            return
        if not hasattr(signal, "SIGALRM"):
            yield
            return

        def _handler(signum, frame):  # noqa: ANN001
            raise TimeoutError(f"Agent runtime timed out after {timeout_seconds}s.")

        previous = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, _handler)
        signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)

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

    def _run_direct_text_completion_once(
        self,
        *,
        instructions: str,
        payload_text: str,
        max_output_tokens: int,
    ) -> str:
        client = self._build_sync_openai_client()
        if self._resolved_use_responses_api():
            response = client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": payload_text},
                ],
                max_output_tokens=max_output_tokens,
            )
            output_text = getattr(response, "output_text", None)
            if isinstance(output_text, str) and output_text.strip():
                return output_text
            output = getattr(response, "output", None) or []
            parts: list[str] = []
            for item in output:
                for content_item in getattr(item, "content", None) or []:
                    text_value = getattr(content_item, "text", None)
                    if text_value:
                        parts.append(str(text_value))
            return "\n".join(parts).strip()

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": payload_text},
            ],
            max_tokens=max_output_tokens,
        )
        message = response.choices[0].message
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text_value = item.get("text")
                    if text_value:
                        parts.append(str(text_value))
                else:
                    text_value = getattr(item, "text", None)
                    if text_value:
                        parts.append(str(text_value))
            return "\n".join(parts).strip()
        return str(content)

    def _run_direct_text_completion(
        self,
        *,
        instructions: str,
        payload_text: str,
        timeout_seconds: int | None,
        max_output_tokens: int | None = None,
    ) -> str:
        target_max_tokens = max_output_tokens or self.runtime_config.max_output_tokens
        with self._runtime_timeout(timeout_seconds):
            while True:
                try:
                    return self._run_direct_text_completion_once(
                        instructions=instructions,
                        payload_text=payload_text,
                        max_output_tokens=target_max_tokens,
                    )
                except Exception as error:
                    reduced = self._reduced_max_output_tokens(error, target_max_tokens)
                    if reduced is None or reduced >= target_max_tokens:
                        raise
                    target_max_tokens = reduced

    def _reduced_max_output_tokens(self, error: Exception, current_max_tokens: int) -> int | None:
        message = str(error)
        match = re.search(r"can only afford\s+(\d+)", message)
        if not match:
            return None
        affordable = int(match.group(1))
        if affordable <= 0:
            return None
        if affordable >= current_max_tokens:
            return None
        return max(32, min(current_max_tokens - 1, affordable))

    def _extract_prompt_token_budget(self, error: Exception) -> tuple[int, int] | None:
        message = str(error)
        match = re.search(r"Prompt tokens limit exceeded:\s*(\d+)\s*>\s*(\d+)", message)
        if not match:
            return None
        requested = int(match.group(1))
        limit = int(match.group(2))
        if requested <= 0 or limit <= 0 or requested <= limit:
            return None
        return requested, limit

    def _compact_text(self, text: str, target_length: int) -> str:
        if len(text) <= target_length:
            return text
        if target_length <= 80:
            return text[:target_length]
        ellipsis = "\n...[truncated]...\n"
        head = max(40, int(target_length * 0.7))
        tail = max(20, target_length - head - len(ellipsis))
        return text[:head] + ellipsis + text[-tail:]

    def _compact_payload_value(self, value: Any, shrink_ratio: float) -> Any:
        ratio = max(0.2, min(shrink_ratio, 0.95))
        if isinstance(value, str):
            target = max(80, int(len(value) * ratio))
            return self._compact_text(value, target)
        if isinstance(value, list):
            if not value:
                return value
            keep = max(1, int(len(value) * max(ratio, 0.4)))
            return [self._compact_payload_value(item, ratio) for item in value[:keep]]
        if isinstance(value, dict):
            return {key: self._compact_payload_value(item, ratio) for key, item in value.items()}
        return value

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
        max_turns: int | None = None,
        runtime_timeout_seconds: int | None = None,
    ) -> OutputT:
        self.ensure_ready()
        use_prompt_json_output = self._should_use_prompt_json_output()
        if not tools:
            direct_instructions = self._build_prompt_json_instructions(instructions, output_type)
            current_max_tokens = min(self.runtime_config.max_output_tokens, 768)
            compacted_payload = payload
            for _attempt in range(4):
                serialized_payload = json.dumps(compacted_payload, ensure_ascii=False, separators=(",", ":"))
                try:
                    raw_text = self._run_direct_text_completion(
                        instructions=direct_instructions,
                        payload_text=serialized_payload,
                        timeout_seconds=runtime_timeout_seconds or self.runtime_config.request_timeout_seconds,
                        max_output_tokens=current_max_tokens,
                    )
                    return self._validate_prompt_json_output(raw_text, output_type)
                except Exception as error:
                    prompt_budget = self._extract_prompt_token_budget(error)
                    if prompt_budget is not None:
                        requested, limit = prompt_budget
                        shrink_ratio = max(0.25, min(0.85, (limit / requested) * 0.85))
                        compacted_payload = self._compact_payload_value(compacted_payload, shrink_ratio)
                        continue
                    reduced = self._reduced_max_output_tokens(error, current_max_tokens)
                    if reduced is not None and reduced < current_max_tokens:
                        current_max_tokens = reduced
                        continue
                    raise
            raise RuntimeError(f"{specialist_name} exceeded prompt budget after repeated payload compaction.")
        compacted_payload = payload
        current_max_tokens = min(self.runtime_config.max_output_tokens, 768)
        session = self._build_session(session_id)
        for _attempt in range(4):
            serialized_payload = json.dumps(compacted_payload, ensure_ascii=False, separators=(",", ":"))
            agent = Agent(
                name=specialist_name,
                instructions=(
                    self._build_prompt_json_instructions(instructions, output_type)
                    if use_prompt_json_output
                    else instructions
                ),
                model=self.model,
                model_settings=self._build_model_settings(current_max_tokens),
                output_type=None if use_prompt_json_output else output_type,
                tools=tools or [],
            )
            run_config = self._build_run_config_with_budget(specialist_name, current_max_tokens)
            try:
                with self._runtime_timeout(runtime_timeout_seconds):
                    result = self._run_sync_with_session(
                        agent,
                        serialized_payload,
                        session=session,
                        run_config=run_config,
                        max_turns=max_turns or self.runtime_config.max_turns,
                    )
                final_output = result.final_output
                if use_prompt_json_output:
                    return self._validate_prompt_json_output(final_output, output_type)
                if isinstance(final_output, output_type):
                    return final_output
                if isinstance(final_output, BaseModel):
                    return output_type.model_validate(final_output.model_dump(mode="python"))
                return output_type.model_validate(final_output)
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
                            payload=compacted_payload,
                            session_id=session_id,
                            output_type=output_type,
                            run_config=run_config,
                            original_error=error,
                        )
                    except Exception:
                        raise error
                prompt_budget = self._extract_prompt_token_budget(error)
                if prompt_budget is not None:
                    requested, limit = prompt_budget
                    shrink_ratio = max(0.25, min(0.85, (limit / requested) * 0.85))
                    compacted_payload = self._compact_payload_value(compacted_payload, shrink_ratio)
                    continue
                reduced = self._reduced_max_output_tokens(error, current_max_tokens)
                if reduced is not None and reduced < current_max_tokens:
                    current_max_tokens = reduced
                    continue
                raise
        raise RuntimeError(f"{specialist_name} exceeded prompt budget after repeated payload compaction.")
