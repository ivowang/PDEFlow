from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class ResearchBriefConfig(BaseModel):
    """High-level research problem description supplied by the user."""

    title: str
    question: str
    background: str = ""
    objectives: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    deliverables: list[str] = Field(default_factory=list)
    domain_tags: list[str] = Field(default_factory=list)


class RuntimeConfig(BaseModel):
    backend: str = "openai_agents"
    model: str = "gpt-4.1"


class RetrievalConfig(BaseModel):
    paper_search_backend: str = "arxiv"
    repository_search_backend: str = "github"
    http_timeout_seconds: int = 60
    max_search_results: int = 10


class ExecutionConfig(BaseModel):
    network_enabled: bool = True
    auto_bootstrap_environments: bool = True
    allow_shell_commands: bool = True
    allow_package_installation: bool = True
    workspace_root: str = "external_assets"


class ResourcePolicyConfig(BaseModel):
    preferred_gpu_ids: list[int] = Field(default_factory=lambda: [6, 7])
    gpu_inventory_hint: dict[str, str] = Field(default_factory=dict)
    auto_decide_runtime: bool = True
    allow_unbounded_runtime: bool = True
    max_parallel_experiments: int = 2


class SecretConfig(BaseModel):
    env_var: str
    purpose: str
    required: bool = False


class SystemConfig(BaseModel):
    project_name: str
    run_name: str
    output_root: str = "runs"
    research_brief: ResearchBriefConfig
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    resource_policy: ResourcePolicyConfig = Field(default_factory=ResourcePolicyConfig)
    secrets: list[SecretConfig] = Field(default_factory=list)
    manager_safety_max_cycles: int | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> "SystemConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(payload)
