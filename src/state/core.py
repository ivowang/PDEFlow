from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from config import ResearchBriefConfig
from .entities import (
    ArtifactRecord,
    CapabilityMatrix,
    CandidateDirection,
    ClassifiedFailure,
    DiaryEntry,
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


class ResearchPhase(str, Enum):
    LITERATURE_REVIEW = "literature_review"
    ACQUISITION = "acquisition"
    PROBLEM_FRAMING = "problem_framing"
    DIAGNOSIS = "diagnosis"
    HYPOTHESIS = "hypothesis"
    METHOD_DESIGN = "method_design"
    CODING = "coding"
    EXPERIMENT_PLANNING = "experiment_planning"
    PREFLIGHT_VALIDATION = "preflight_validation"
    EXPERIMENT = "experiment"
    REFLECTION = "reflection"
    REPORTING = "reporting"


class ResearchState(BaseModel):
    project_name: str
    run_name: str
    work_directory: str
    research_brief: ResearchBriefConfig
    current_phase: ResearchPhase = ResearchPhase.LITERATURE_REVIEW
    phase_history: list[str] = Field(default_factory=list)
    cycle_index: int = 0
    environment_snapshot: EnvironmentSnapshot | None = None
    capability_matrix: CapabilityMatrix | None = None
    secret_status: list[SecretStatus] = Field(default_factory=list)
    literature_notes: list[PaperNote] = Field(default_factory=list)
    method_taxonomy: list[TaxonomyEntry] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    external_artifacts: list[ArtifactRecord] = Field(default_factory=list)
    repositories: list[RepositoryRecord] = Field(default_factory=list)
    selected_baseline_program_id: str | None = None
    acquisition_notes: list[str] = Field(default_factory=list)
    evaluation_criteria: list[str] = Field(default_factory=list)
    problem_framing_notes: list[str] = Field(default_factory=list)
    bottleneck_analysis: list[str] = Field(default_factory=list)
    candidate_directions: list[CandidateDirection] = Field(default_factory=list)
    hypotheses: list[HypothesisRecord] = Field(default_factory=list)
    method_designs: list[MethodDesign] = Field(default_factory=list)
    program_candidates: list[ProgramCandidate] = Field(default_factory=list)
    experiment_plans: list[ExperimentPlan] = Field(default_factory=list)
    preflight_reports: list[PreflightReport] = Field(default_factory=list)
    execution_records: list[ExperimentRecord] = Field(default_factory=list)
    experiment_records: list[ExperimentRecord] = Field(default_factory=list)
    best_known_results: dict[str, dict[str, Any]] = Field(default_factory=dict)
    failure_summaries: list[str] = Field(default_factory=list)
    classified_failures: list[ClassifiedFailure] = Field(default_factory=list)
    reflections: list[ReflectionRecord] = Field(default_factory=list)
    semantic_memory_notes: list[str] = Field(default_factory=list)
    research_diary: list[DiaryEntry] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    generated_reports: list[GeneratedReport] = Field(default_factory=list)
    termination_decision: str | None = None
    blocked_reason: str | None = None
