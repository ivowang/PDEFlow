from __future__ import annotations

import json
import os
from typing import Any, TypeVar

from pydantic import BaseModel

from .config import RuntimeConfig

try:
    from agents import Agent, OpenAIProvider, RunConfig, Runner, SQLiteSession
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover
    Agent = None
    OpenAIProvider = None
    RunConfig = None
    Runner = None
    SQLiteSession = None
    AsyncOpenAI = None


OutputT = TypeVar("OutputT", bound=BaseModel)


class RuntimeAdapter:
    """Thin wrapper over OpenAI Agents SDK structured specialist execution."""

    def __init__(self, runtime_config: RuntimeConfig, session_db_path: str):
        self.runtime_config = runtime_config
        self.backend = runtime_config.backend
        self.model = runtime_config.model
        self.session_db_path = session_db_path

    def _resolved_provider(self) -> str:
        return (self.runtime_config.provider or "openai").strip().lower()

    def _resolved_api_key_env_var(self) -> str:
        if self.runtime_config.api_key_env_var:
            return self.runtime_config.api_key_env_var
        if self._resolved_provider() == "openrouter":
            return "OPENROUTER_API_KEY"
        return "OPENAI_API_KEY"

    def _resolved_api_key(self) -> str | None:
        return os.getenv(self._resolved_api_key_env_var())

    def _resolved_base_url(self) -> str | None:
        if self.runtime_config.api_base_url:
            return self.runtime_config.api_base_url
        if self._resolved_provider() == "openrouter":
            return "https://openrouter.ai/api/v1"
        return None

    def _resolved_use_responses_api(self) -> bool:
        if self.runtime_config.use_responses_api is not None:
            return self.runtime_config.use_responses_api
        return self._resolved_provider() != "openrouter"

    def ensure_ready(self) -> None:
        if self.backend != "openai_agents":
            raise RuntimeError(
                "This repository no longer ships a mock runtime. "
                "Set runtime.backend to 'openai_agents'."
            )
        if Agent is None or Runner is None or OpenAIProvider is None or AsyncOpenAI is None:
            raise RuntimeError(
                "openai-agents and openai are not installed in the active environment. Run `uv sync` first."
            )
        if self._resolved_provider() not in {"openai", "openrouter"}:
            raise RuntimeError(
                f"Unsupported runtime.provider='{self.runtime_config.provider}'. "
                "Use 'openai' or 'openrouter'."
            )
        if not self._resolved_api_key():
            raise RuntimeError(
                f"{self._resolved_api_key_env_var()} is not set. "
                "Export it before running the autonomous research workflow."
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

    def _build_model_provider(self) -> Any:
        provider_name = self._resolved_provider()
        api_key = self._resolved_api_key()
        base_url = self._resolved_base_url()
        use_responses = self._resolved_use_responses_api()

        if provider_name == "openai":
            return OpenAIProvider(
                api_key=api_key,
                base_url=base_url,
                websocket_base_url=self.runtime_config.websocket_base_url,
                organization=self.runtime_config.organization,
                project=self.runtime_config.project,
                use_responses=use_responses,
            )

        default_headers: dict[str, str] = {}
        site_url = self.runtime_config.openrouter_site_url or os.getenv("OPENROUTER_SITE_URL")
        app_name = self.runtime_config.openrouter_app_name or os.getenv("OPENROUTER_APP_NAME")
        if site_url:
            default_headers["HTTP-Referer"] = site_url
        if app_name:
            default_headers["X-Title"] = app_name

        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers or None,
        )
        return OpenAIProvider(openai_client=client, use_responses=use_responses)

    def _build_run_config(self, specialist_name: str) -> Any | None:
        if RunConfig is None:
            return None
        return RunConfig(
            model=self.model,
            model_provider=self._build_model_provider(),
            workflow_name=f"pdeflow::{specialist_name.lower()}",
            tracing_disabled=self.runtime_config.tracing_disabled,
        )

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
        run_config = self._build_run_config(specialist_name)
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
