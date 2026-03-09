from __future__ import annotations

from enum import Enum
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


class AssetSemanticSpec(BaseModel):
    benchmark: str | None = None
    asset_family: str | None = None
    equation: str | None = None
    split: str | None = None
    variant: str | None = None
    nu: str | None = None
    rho: str | None = None
    filename: str | None = None
    source_kind: str | None = None


class ArtifactStatus(str, Enum):
    DOWNLOADED = "downloaded"
    CHECKSUM_VERIFIED = "checksum_verified"
    FORMAT_VERIFIED = "format_verified"
    READY_FOR_TRAINING = "ready_for_training"
    CORRUPTED = "corrupted"
    QUARANTINED = "quarantined"
    VERIFIED_LOCAL = "verified_local"
    VERIFIED_REMOTE = "verified_remote"
    DOWNLOAD_FAILED = "download_failed"
    BLOCKED = "blocked"


class ArtifactChecksumRecord(BaseModel):
    algorithm: str = "md5"
    expected: str | None = None
    actual: str | None = None
    source: str | None = None
    matched: bool | None = None
    checked_at: str = Field(default_factory=now_utc)


class ArtifactValidationResult(BaseModel):
    validator: str
    status: ArtifactStatus
    exists: bool
    size_bytes: int = 0
    min_size_bytes: int | None = None
    size_ok: bool = False
    format_valid: bool | None = None
    ready_for_training: bool = False
    top_level_keys: list[str] = Field(default_factory=list)
    sample_read_target: str | None = None
    sample_shape: list[int] = Field(default_factory=list)
    checksum: ArtifactChecksumRecord | None = None
    failure_reasons: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)
    validated_at: str = Field(default_factory=now_utc)


class ArtifactDownloadMetadata(BaseModel):
    source_url: str | None = None
    source_type: str | None = None
    local_path: str | None = None
    canonical_target_id: str | None = None
    strategy_id: str | None = None
    attempt_signature: str | None = None
    download_timestamp: str = Field(default_factory=now_utc)
    file_size: int = 0
    checksum: ArtifactChecksumRecord | None = None
    validation_status: str | None = None
    transfer_method: str | None = None
    attempt_count: int = 0
    bytes_downloaded: int = 0
    elapsed_time: float = 0.0
    average_throughput: float | None = None
    failure_type: str | None = None
    failure_message: str | None = None
    resumed: bool = False


class ArtifactRecord(BaseModel):
    artifact_id: str
    canonical_id: str | None = None
    raw_aliases: list[str] = Field(default_factory=list)
    artifact_type: str
    title: str
    rationale: str
    query: str = ""
    source_url: str | None = None
    local_path: str | None = None
    status: str
    semantic_spec: AssetSemanticSpec | None = None
    validation: ArtifactValidationResult | None = None
    download_metadata: ArtifactDownloadMetadata | None = None
    quarantine_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class RepositoryRecord(BaseModel):
    repo_id: str
    canonical_id: str | None = None
    raw_aliases: list[str] = Field(default_factory=list)
    name: str
    remote_url: str
    local_path: str
    bootstrap_status: str = "uninitialized"
    environment_path: str | None = None
    environment_id: str | None = None
    resolution_source: str | None = None
    detected_manifests: list[str] = Field(default_factory=list)
    entrypoints: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class EnvironmentResolutionState(str, Enum):
    NOT_STARTED = "not_started"
    CREATING = "creating"
    SYNC_FAILED = "sync_failed"
    FALLBACK_INSTALLING = "fallback_installing"
    EDITABLE_INSTALLING = "editable_installing"
    READY = "ready"
    BROKEN = "broken"


class EnvironmentRecord(BaseModel):
    env_id: str
    canonical_id: str
    project_path: str
    environment_path: str
    repo_id: str | None = None
    python_interpreter: str | None = None
    state: EnvironmentResolutionState = EnvironmentResolutionState.NOT_STARTED
    strategy: str | None = None
    attempted_commands: list[str] = Field(default_factory=list)
    manifests: list[str] = Field(default_factory=list)
    failure_reason: str | None = None
    failure_category: str | None = None
    fallback_recipe: list[str] = Field(default_factory=list)
    updated_at: str = Field(default_factory=now_utc)


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
    job_kind: str = "experiment"
    working_directory: str
    setup_commands: list[str] = Field(default_factory=list)
    launch_command: str
    environment: dict[str, str] = Field(default_factory=dict)
    gpu_ids: list[int] = Field(default_factory=list)
    required_artifact_ids: list[str] = Field(default_factory=list)
    preflight_required: bool = True
    preflight_status: str | None = None
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
    job_kind: str = "experiment"
    command: str
    working_directory: str
    status: str
    return_code: int | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    observations: list[str] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)
    failure_ids: list[str] = Field(default_factory=list)
    log_path: str
    result_paths: list[str] = Field(default_factory=list)
    started_at: str = Field(default_factory=now_utc)
    finished_at: str | None = None


class PreflightCheckResult(BaseModel):
    name: str
    passed: bool
    details: str = ""
    category: str = ""


class PreflightReport(BaseModel):
    report_id: str
    plan_id: str
    program_id: str
    job_kind: str = "preflight"
    passed: bool
    failed_checks: list[PreflightCheckResult] = Field(default_factory=list)
    blocking_reason: str | None = None
    recommended_route: str | None = None
    related_artifact_ids: list[str] = Field(default_factory=list)
    log_path: str | None = None
    created_at: str = Field(default_factory=now_utc)


class FailureSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ClassifiedFailure(BaseModel):
    failure_id: str
    failure_type: str
    severity: FailureSeverity
    blocking: bool
    allow_experiment_launch: bool
    source_phase: str
    source_id: str | None = None
    summary: str
    remediation_steps: list[str] = Field(default_factory=list)
    fallback_strategy: str | None = None
    detected_from: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=now_utc)


class CapabilityMatrix(BaseModel):
    environment_path: str | None = None
    repo_ready: bool = False
    env_ready: bool = False
    codepath_ready: bool = False
    dataset_ready: bool = False
    baseline_launch_ready: bool = False
    experiment_plan_ready: bool = False
    scientific_iteration_ready: bool = False
    python_available: bool = False
    pip_available: bool = False
    torch_available: bool = False
    cuda_available: bool = False
    h5py_available: bool = False
    hydra_available: bool = False
    pdebench_trainable: bool = False
    deepxde_installed: bool = False
    deepxde_backend: str | None = None
    tensorflow_available: bool = False
    pinn_ready: bool = False
    fno_ready: bool = False
    unet_ready: bool = False
    target_dataset_ready: bool = False
    target_dataset_blocked: bool = False
    exact_target_shards_missing: list[str] = Field(default_factory=list)
    exact_target_shards_corrupted: list[str] = Field(default_factory=list)
    fallback_assets_available: bool = False
    baseline_ready_to_launch: bool = False
    generated_at: str = Field(default_factory=now_utc)


class BlockerRecord(BaseModel):
    blocker_id: str
    blocker_type: str
    target_entity: str
    first_seen_cycle: int
    last_seen_cycle: int
    repeat_count: int = 1
    last_attempt_signature: str | None = None
    evidence_summary: str = ""
    recovery_strategies_tried: list[str] = Field(default_factory=list)
    terminality: str = "temporary"
    route_exhausted: bool = False
    blocked_route_ids: list[str] = Field(default_factory=list)
    recommended_pivots: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=now_utc)
    updated_at: str = Field(default_factory=now_utc)


class RouteDecisionRecord(BaseModel):
    cycle_index: int
    route_id: str
    rationale: str
    blockers_considered: list[str] = Field(default_factory=list)
    allowed: bool = True
    material_change_summary: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=now_utc)


class CycleDeltaRecord(BaseModel):
    cycle_index: int
    snapshot_signature: str
    changed: bool = False
    summary: list[str] = Field(default_factory=list)
    newly_ready_artifacts: list[str] = Field(default_factory=list)
    newly_blocked_artifacts: list[str] = Field(default_factory=list)
    newly_ready_environments: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=now_utc)


class ReflectionRecord(BaseModel):
    reflection_id: str
    cycle_index: int
    hypothesis_id: str | None = None
    verdict: str
    evidence: list[str] = Field(default_factory=list)
    linked_failure_ids: list[str] = Field(default_factory=list)
    accepted_lessons: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    recommended_route_id: str | None = None
    preferred_recovery_strategies: list[str] = Field(default_factory=list)
    forbidden_attempt_signatures: list[str] = Field(default_factory=list)
    blocked_entities: list[str] = Field(default_factory=list)
    material_change_required: bool = False
    escalation_required: bool = False
    failure_category: str | None = None
    continue_research: bool = False
    stop_reason: str | None = None
    created_at: str = Field(default_factory=now_utc)


class HumanResponseType(str, Enum):
    CONFIRMED_DONE = "confirmed_done"
    INSTRUCTION = "instruction"
    REDUCE_SCOPE = "reduce_scope"
    ABORT = "abort"


class HITLStatus(str, Enum):
    REQUESTED = "requested"
    RESPONDED = "responded"
    REVALIDATED = "revalidated"
    RESUMED = "resumed"
    STILL_BLOCKED = "still_blocked"
    ABORTED = "aborted"


class HITLEvent(BaseModel):
    event_id: str
    cycle_index: int
    blocker_ids: list[str] = Field(default_factory=list)
    blocker_type: str
    target_entities: list[str] = Field(default_factory=list)
    escalation_reason: str
    requested_actions: list[str] = Field(default_factory=list)
    prompt_text: str
    response_type: HumanResponseType | None = None
    response_text: str | None = None
    validation_summary: list[str] = Field(default_factory=list)
    status: HITLStatus = HITLStatus.REQUESTED
    material_effect: str | None = None
    created_at: str = Field(default_factory=now_utc)
    responded_at: str | None = None


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
