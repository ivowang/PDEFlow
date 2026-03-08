from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from common import now_utc


class PaperNote(BaseModel):
    paper_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    abstract: str = ""
    method_family: str = ""
    physics_level: str = ""
    key_claims: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    research_opportunities: list[str] = Field(default_factory=list)
    source_url: str = ""
    pdf_path: str | None = None
    note_path: str | None = None


class TaxonomyEntry(BaseModel):
    category: str
    methods: list[str] = Field(default_factory=list)
    shared_strengths: list[str] = Field(default_factory=list)
    shared_limitations: list[str] = Field(default_factory=list)
    research_opportunities: list[str] = Field(default_factory=list)


class SecretStatus(BaseModel):
    env_var: str
    purpose: str
    required: bool = False
    is_set: bool
    resolution_hint: str


class EnvironmentSnapshot(BaseModel):
    python_executable: str
    python_version: str
    uv_available: bool
    uv_version: str | None = None
    available_gpu_ids: list[int] = Field(default_factory=list)
    selected_gpu_ids: list[int] = Field(default_factory=list)
    gpu_descriptions: dict[str, str] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class ArtifactRecord(BaseModel):
    artifact_id: str
    artifact_type: str
    title: str
    rationale: str
    query: str = ""
    source_url: str | None = None
    local_path: str | None = None
    status: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class RepositoryRecord(BaseModel):
    repo_id: str
    name: str
    remote_url: str
    local_path: str
    bootstrap_status: str = "uninitialized"
    detected_manifests: list[str] = Field(default_factory=list)
    entrypoints: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class CandidateDirection(BaseModel):
    direction_id: str
    title: str
    innovation_kind: str
    rationale: str
    why_not_just_tuning: str
    expected_signal: str


class HypothesisRecord(BaseModel):
    hypothesis_id: str
    title: str
    statement: str
    rationale: str
    expected_gains: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    required_code_changes: list[str] = Field(default_factory=list)
    evaluation_plan: list[str] = Field(default_factory=list)
    innovation_kind: str
    status: str = "proposed"
    parent_program_id: str | None = None


class MethodDesign(BaseModel):
    design_id: str
    hypothesis_id: str
    title: str
    parent_program_id: str | None = None
    architecture_changes: list[str] = Field(default_factory=list)
    loss_changes: list[str] = Field(default_factory=list)
    data_changes: list[str] = Field(default_factory=list)
    training_strategy: list[str] = Field(default_factory=list)
    inference_strategy: list[str] = Field(default_factory=list)
    physics_integration: list[str] = Field(default_factory=list)
    implementation_steps: list[str] = Field(default_factory=list)
    evaluation_plan: list[str] = Field(default_factory=list)


class ProgramCandidate(BaseModel):
    program_id: str
    title: str
    summary: str
    repo_id: str | None = None
    workspace_path: str | None = None
    parent_program_id: str | None = None
    design_id: str | None = None
    hypothesis_id: str | None = None
    entry_command_hint: str | None = None
    status: str
    changed_files: list[str] = Field(default_factory=list)
    patch_paths: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=now_utc)


class ExperimentPlan(BaseModel):
    plan_id: str
    title: str
    program_id: str
    repo_id: str | None = None
    working_directory: str
    setup_commands: list[str] = Field(default_factory=list)
    launch_command: str
    environment: dict[str, str] = Field(default_factory=dict)
    gpu_ids: list[int] = Field(default_factory=list)
    expected_outputs: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    stopping_rules: list[str] = Field(default_factory=list)
    log_path: str
    status: str = "planned"
    notes: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=now_utc)


class ExperimentRecord(BaseModel):
    experiment_id: str
    plan_id: str
    program_id: str
    command: str
    working_directory: str
    status: str
    return_code: int | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    observations: list[str] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)
    log_path: str
    result_paths: list[str] = Field(default_factory=list)
    started_at: str = Field(default_factory=now_utc)
    finished_at: str | None = None


class ReflectionRecord(BaseModel):
    reflection_id: str
    cycle_index: int
    hypothesis_id: str | None = None
    verdict: str
    evidence: list[str] = Field(default_factory=list)
    accepted_lessons: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    continue_research: bool = False
    stop_reason: str | None = None
    created_at: str = Field(default_factory=now_utc)


class GeneratedReport(BaseModel):
    report_id: str
    title: str
    kind: str
    path: str
    created_at: str = Field(default_factory=now_utc)


class DiaryEntry(BaseModel):
    entry_id: str
    phase: str
    title: str
    body: str
    created_at: str = Field(default_factory=now_utc)
    tags: list[str] = Field(default_factory=list)
