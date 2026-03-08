from __future__ import annotations

import json
from typing import Any, TypeVar

from pydantic import BaseModel

from .provider import Agent, Runner


OutputT = TypeVar("OutputT", bound=BaseModel)


class StructuredOutputMixin:
    """Schema-guided JSON output validation and repair helpers."""

    def _should_use_prompt_json_output(self) -> bool:
        return self._resolved_provider() == "openrouter"

    def _build_prompt_json_instructions(
        self,
        instructions: str,
        output_type: type[OutputT],
    ) -> str:
        schema_json = json.dumps(output_type.model_json_schema(), ensure_ascii=False, indent=2)
        return (
            instructions.strip()
            + "\n\n"
            + "Output requirements:\n"
            + "- Return exactly one valid JSON object.\n"
            + "- Do not wrap the JSON in markdown fences.\n"
            + "- Do not add commentary before or after the JSON.\n"
            + "- Ensure the JSON validates against the following schema.\n\n"
            + schema_json
        )

    def _coerce_final_output_to_text(self, final_output: Any) -> str:
        if isinstance(final_output, str):
            return final_output
        if isinstance(final_output, BaseModel):
            return final_output.model_dump_json(indent=2)
        if isinstance(final_output, (dict, list)):
            return json.dumps(final_output, ensure_ascii=False, indent=2)
        return str(final_output)

    def _extract_balanced_json_object(self, text: str) -> str | None:
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]
        return None

    def _extract_json_object_text(self, text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```") and stripped.endswith("```"):
            lines = stripped.splitlines()
            if len(lines) >= 3:
                stripped = "\n".join(lines[1:-1]).strip()
        try:
            json.loads(stripped)
            return stripped
        except json.JSONDecodeError:
            pass

        candidate = self._extract_balanced_json_object(stripped)
        if candidate:
            json.loads(candidate)
            return candidate
        raise json.JSONDecodeError("No valid JSON object found in model output.", stripped, 0)

    def _repair_prompt_json_output(
        self,
        raw_text: str,
        output_type: type[OutputT],
        error_message: str,
    ) -> OutputT:
        schema_json = json.dumps(output_type.model_json_schema(), ensure_ascii=False, indent=2)
        repair_agent = Agent(
            name="JsonRepairAgent",
            instructions=(
                "You repair malformed JSON emitted by another model.\n"
                "Rules:\n"
                "- Return exactly one valid JSON object.\n"
                "- Preserve the original semantic content whenever possible.\n"
                "- Only repair syntax, quoting, commas, brackets, or obviously broken schema formatting.\n"
                "- Do not add commentary or markdown fences.\n"
                "- Ensure the result validates against the provided schema."
            ),
            model=self.model,
            output_type=None,
            tools=[],
        )
        repair_payload = json.dumps(
            {
                "schema": json.loads(schema_json),
                "validation_error": error_message,
                "malformed_output": raw_text,
            },
            ensure_ascii=False,
            indent=2,
        )
        repair_result = Runner.run_sync(
            repair_agent,
            repair_payload,
            max_turns=4,
            session=None,
            run_config=self._build_run_config("json_repair"),
        )
        repaired_text = self._coerce_final_output_to_text(repair_result.final_output)
        repaired_payload = json.loads(self._extract_json_object_text(repaired_text))
        return output_type.model_validate(repaired_payload)

    def _validate_prompt_json_output(
        self,
        final_output: Any,
        output_type: type[OutputT],
    ) -> OutputT:
        raw_text = self._coerce_final_output_to_text(final_output)
        try:
            payload = json.loads(self._extract_json_object_text(raw_text))
            return output_type.model_validate(payload)
        except Exception as error:
            return self._repair_prompt_json_output(raw_text, output_type, str(error))
