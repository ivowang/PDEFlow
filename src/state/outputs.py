from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .entities import (
    ArtifactRecord,
    CapabilityMatrix,
    CandidateDirection,
    EnvironmentRecord,
    EnvironmentSnapshot,
    ExperimentPlan,
    ExperimentRecord,
    GeneratedReport,
    HypothesisRecord,
    MethodDesign,
    PaperNote,
    PreflightReport,
    ProgramCandidate,
    ReflectionRecord,
    RepositoryRecord,
    SecretStatus,
    TaxonomyEntry,
)


class LiteraturePhaseOutput(BaseModel):
    summary: str
    literature_notes: list[PaperNote]
    method_taxonomy: list[TaxonomyEntry]
    open_questions: list[str]
    semantic_notes: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)


class AcquisitionPhaseOutput(BaseModel):
    summary: str
    environment_snapshot: EnvironmentSnapshot
    environment_records: list[EnvironmentRecord] = Field(default_factory=list)
    capability_matrix: CapabilityMatrix | None = None
    secret_status: list[SecretStatus] = Field(default_factory=list)
    external_artifacts: list[ArtifactRecord] = Field(default_factory=list)
    repositories: list[RepositoryRecord] = Field(default_factory=list)
    program_candidates: list[ProgramCandidate] = Field(default_factory=list)
    selected_baseline_program_id: str | None = None
    acquisition_notes: list[str] = Field(default_factory=list)
    semantic_notes: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)


class ProblemFramingPhaseOutput(BaseModel):
    summary: str
    problem_framing_notes: list[str]
    evaluation_criteria: list[str]
    candidate_directions: list[CandidateDirection]
    next_actions: list[str] = Field(default_factory=list)


class DiagnosisPhaseOutput(BaseModel):
    summary: str
    bottleneck_analysis: list[str]
    next_actions: list[str] = Field(default_factory=list)
    semantic_notes: list[str] = Field(default_factory=list)


class HypothesisPhaseOutput(BaseModel):
    summary: str
    hypotheses: list[HypothesisRecord]
    next_actions: list[str] = Field(default_factory=list)


class MethodDesignPhaseOutput(BaseModel):
    summary: str
    method_designs: list[MethodDesign]
    next_actions: list[str] = Field(default_factory=list)


class CodingPhaseOutput(BaseModel):
    summary: str
    program_candidates: list[ProgramCandidate]
    next_actions: list[str] = Field(default_factory=list)


class ExperimentPlanningPhaseOutput(BaseModel):
    summary: str
    experiment_plans: list[ExperimentPlan]
    next_actions: list[str] = Field(default_factory=list)


class PreflightValidationPhaseOutput(BaseModel):
    summary: str
    preflight_reports: list[PreflightReport]
    capability_matrix: CapabilityMatrix | None = None
    failure_summaries: list[str] = Field(default_factory=list)
    zero_plan_reason: str | None = None
    recommended_route: str | None = None
    next_actions: list[str] = Field(default_factory=list)


class ExperimentPhaseOutput(BaseModel):
    summary: str
    experiment_records: list[ExperimentRecord]
    best_known_results: dict[str, dict[str, Any]] = Field(default_factory=dict)
    failure_summaries: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)


class ReflectionPhaseOutput(BaseModel):
    summary: str
    reflections: list[ReflectionRecord]
    next_actions: list[str] = Field(default_factory=list)
    terminate_research: bool = False


class ReportingPhaseOutput(BaseModel):
    summary: str
    generated_reports: list[GeneratedReport]
