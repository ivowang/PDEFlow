from __future__ import annotations

import csv
from pathlib import Path
import re
import shutil
import traceback

from research_agents import (
    AcquisitionAgent,
    EngineeringAgent,
    EvaluationAgent,
    LiteratureAgent,
    ResearchStrategistAgent,
    ReporterAgent,
)
from config import SystemConfig
from integrations.command_grounding import ground_experiment_plan
from memory import ResearchMemory
from orchestration.blockers import refresh_blocker_registry
from orchestration.failures import classify_state_failures
from orchestration.hitl import (
    blocked_artifacts_for_hitl,
    build_hitl_prompt,
    extract_absolute_paths,
    select_hitl_blockers,
    should_trigger_hitl,
)
from runtime import RuntimeAdapter
from state import (
    AcquisitionPhaseOutput,
    ArtifactDownloadMetadata,
    ArtifactRecord,
    ArtifactStatus,
    AssetSemanticSpec,
    CandidateDirection,
    CodingPhaseOutput,
    DiagnosisPhaseOutput,
    EnvironmentRecord,
    EnvironmentResolutionState,
    ExperimentPlan,
    ExperimentPlanningPhaseOutput,
    GeneratedReport,
    HITLEvent,
    HITLStatus,
    HypothesisRecord,
    HypothesisPhaseOutput,
    HumanResponseType,
    LiteraturePhaseOutput,
    MethodDesign,
    MethodDesignPhaseOutput,
    PaperNote,
    ProblemFramingPhaseOutput,
    ProgramCandidate,
    ReflectionRecord,
    ReflectionPhaseOutput,
    ResearchPhase,
    ResearchState,
    RepositoryRecord,
    ReportingPhaseOutput,
    RouteDecisionRecord,
    TaxonomyEntry,
)
from tools import ResearchTools
from common import canonicalize_env_id, dedupe_strings, now_utc, short_hash, upsert_by_attr
from .normalization import normalize_artifacts, normalize_environments, normalize_repositories
from .routing import ManagerRoutingMixin
from .specs import CycleRoute, PhaseSpec

class ResearchManager(ManagerRoutingMixin):
    """Manager-centered orchestration for autonomous research execution."""

    def __init__(self, config: SystemConfig, repo_root: Path):
        self.config = config
        self.repo_root = repo_root
        self.work_directory = config.resolve_work_directory(repo_root)
        self.memory = ResearchMemory(root=self.work_directory)
        self.tools = ResearchTools(config=config, memory=self.memory, repo_root=repo_root)
        self.runtime = RuntimeAdapter(
            runtime_config=config.runtime,
            session_db_path=str(self.memory.sessions_db),
        )
        self.agents = {
            "literature": LiteratureAgent(),
            "acquisition": AcquisitionAgent(),
            "problem": ResearchStrategistAgent(ResearchPhase.PROBLEM_FRAMING),
            "diagnosis": ResearchStrategistAgent(ResearchPhase.DIAGNOSIS),
            "hypothesis": ResearchStrategistAgent(ResearchPhase.HYPOTHESIS),
            "design": ResearchStrategistAgent(ResearchPhase.METHOD_DESIGN),
            "coder": EngineeringAgent(ResearchPhase.CODING),
            "planner": EngineeringAgent(ResearchPhase.EXPERIMENT_PLANNING),
            "preflight": EvaluationAgent(ResearchPhase.PREFLIGHT_VALIDATION),
            "experiment": EvaluationAgent(ResearchPhase.EXPERIMENT),
            "reflection": EvaluationAgent(ResearchPhase.REFLECTION),
            "reporter": ReporterAgent(),
        }
        self.front_phases = [
            PhaseSpec(ResearchPhase.LITERATURE_REVIEW, "literature", ("literature_notes", "method_taxonomy", "open_questions")),
            PhaseSpec(ResearchPhase.ACQUISITION, "acquisition", ("environment_snapshot", "external_artifacts", "repositories")),
            PhaseSpec(ResearchPhase.PROBLEM_FRAMING, "problem", ("problem_framing_notes", "candidate_directions")),
            PhaseSpec(ResearchPhase.DIAGNOSIS, "diagnosis", ("bottleneck_analysis", "next_actions")),
        ]
        self.iterative_phases = [
            PhaseSpec(ResearchPhase.HYPOTHESIS, "hypothesis", ("hypotheses",)),
            PhaseSpec(ResearchPhase.METHOD_DESIGN, "design", ("method_designs",)),
            PhaseSpec(ResearchPhase.CODING, "coder", ("program_candidates",)),
            PhaseSpec(ResearchPhase.EXPERIMENT_PLANNING, "planner", ("experiment_plans",)),
            PhaseSpec(ResearchPhase.PREFLIGHT_VALIDATION, "preflight", ("preflight_reports", "capability_matrix")),
            PhaseSpec(ResearchPhase.EXPERIMENT, "experiment", ("experiment_records", "best_known_results")),
            PhaseSpec(ResearchPhase.REFLECTION, "reflection", ("reflections", "next_actions")),
        ]
        self.recovery_phases = [
            PhaseSpec(ResearchPhase.ACQUISITION, "acquisition", ("environment_snapshot", "external_artifacts", "repositories")),
            PhaseSpec(ResearchPhase.EXPERIMENT_PLANNING, "planner", ("experiment_plans",)),
            PhaseSpec(ResearchPhase.PREFLIGHT_VALIDATION, "preflight", ("preflight_reports", "capability_matrix")),
            PhaseSpec(ResearchPhase.EXPERIMENT, "experiment", ("experiment_records", "best_known_results")),
            PhaseSpec(ResearchPhase.REFLECTION, "reflection", ("reflections", "next_actions")),
        ]
        self.reporting_phase = PhaseSpec(ResearchPhase.REPORTING, "reporter", ("generated_reports",))
        self._runtime_degraded = False

    def _log(self, message: str) -> None:
        self.memory.record_process(message)

    def _phase_snapshot(self, state: ResearchState, phase: ResearchPhase) -> list[str]:
        if phase == ResearchPhase.LITERATURE_REVIEW:
            return [
                f"Literature notes: {len(state.literature_notes)} papers, taxonomy entries: {len(state.method_taxonomy)}, open questions: {len(state.open_questions)}."
            ]
        if phase == ResearchPhase.ACQUISITION:
            lines = [
                f"Acquisition status: {len(state.external_artifacts)} artifacts, {len(state.repositories)} repositories, selected baseline={state.selected_baseline_program_id or 'unset'}."
            ]
            if state.capability_matrix:
                lines.append(
                    f"Capability matrix: baseline_ready={state.capability_matrix.baseline_ready_to_launch} "
                    f"target_dataset_ready={state.capability_matrix.target_dataset_ready} "
                    f"target_dataset_preparing={state.capability_matrix.target_dataset_preparing} "
                    f"gpu_runtime_ready={state.capability_matrix.gpu_runtime_ready}."
                )
            return lines
        if phase == ResearchPhase.PROBLEM_FRAMING:
            return [
                f"Problem framing produced {len(state.candidate_directions)} candidate directions and {len(state.evaluation_criteria)} evaluation criteria."
            ]
        if phase == ResearchPhase.DIAGNOSIS:
            return [
                f"Diagnosis recorded {len(state.bottleneck_analysis)} bottlenecks and {len(state.next_actions)} next actions."
            ]
        if phase == ResearchPhase.HYPOTHESIS:
            latest = state.hypotheses[-1] if state.hypotheses else None
            if latest:
                return [f"Latest hypothesis: {latest.hypothesis_id} | {latest.title}."]
        if phase == ResearchPhase.METHOD_DESIGN:
            latest = state.method_designs[-1] if state.method_designs else None
            if latest:
                return [f"Latest method design: {latest.design_id} | {latest.title}."]
        if phase == ResearchPhase.CODING:
            latest = state.program_candidates[-1] if state.program_candidates else None
            if latest:
                return [
                    f"Latest program candidate: {latest.program_id} | status={latest.status} | changed_files={len(latest.changed_files)}."
                ]
        if phase == ResearchPhase.EXPERIMENT_PLANNING:
            latest = state.experiment_plans[-1] if state.experiment_plans else None
            if latest:
                return [
                    f"Latest experiment plan: {latest.plan_id} | program={latest.program_id} | gpus={latest.gpu_ids}."
                ]
        if phase == ResearchPhase.PREFLIGHT_VALIDATION:
            latest = state.preflight_reports[-1] if state.preflight_reports else None
            if latest:
                return [
                    f"Latest preflight: {latest.plan_id} | passed={latest.passed} | failed_checks={len(latest.failed_checks)}."
                ]
        if phase == ResearchPhase.EXPERIMENT:
            latest = state.experiment_records[-1] if state.experiment_records else None
            if latest:
                return [
                    f"Latest experiment: {latest.experiment_id} | status={latest.status} | program={latest.program_id} | failures={len(latest.failure_modes)}."
                ]
        if phase == ResearchPhase.REFLECTION:
            latest = state.reflections[-1] if state.reflections else None
            if latest:
                lines = [
                    f"Reflection verdict: {latest.verdict} | continue_research={latest.continue_research}."
                ]
                if latest.evidence:
                    lines.append(f"Breakthrough evidence: {latest.evidence[0]}")
                if latest.stop_reason:
                    lines.append(f"Stop reason: {latest.stop_reason}")
                return lines
        if phase == ResearchPhase.HUMAN_INTERVENTION:
            latest = state.hitl_events[-1] if state.hitl_events else None
            if latest:
                return [
                    f"HITL status: {latest.status.value} | blocker_type={latest.blocker_type} | targets={len(latest.target_entities)}."
                ]
        if phase == ResearchPhase.REPORTING:
            return [f"Reports generated: {len(state.generated_reports)}."]
        return []

    def _artifact_expected_path(self, artifact) -> Path:
        return self.tools._artifact_materialization_path(artifact)

    def _acquisition_recovery_signature(self, state: ResearchState) -> tuple[object, ...]:
        artifact_bits = tuple(
            sorted(
                (
                    item.canonical_id or item.artifact_id,
                    item.status,
                    item.local_path or "",
                )
                for item in state.external_artifacts
            )
        )
        repository_bits = tuple(
            sorted(
                (
                    item.canonical_id or item.repo_id,
                    item.bootstrap_status,
                    item.local_path,
                )
                for item in state.repositories
            )
        )
        environment_bits = tuple(
            sorted(
                (
                    item.canonical_id or item.env_id,
                    item.state.value,
                    item.environment_path,
                )
                for item in state.environment_records
            )
        )
        capability_bits = (
            bool(state.capability_matrix and state.capability_matrix.repo_ready),
            bool(state.capability_matrix and state.capability_matrix.env_ready),
            bool(state.capability_matrix and state.capability_matrix.target_dataset_ready),
            bool(state.capability_matrix and state.capability_matrix.baseline_ready_to_launch),
        )
        return (
            artifact_bits,
            repository_bits,
            environment_bits,
            capability_bits,
            state.selected_baseline_program_id or "",
        )

    def _needs_acquisition_bootstrap(self, state: ResearchState) -> bool:
        capability = state.capability_matrix
        has_repo = any(item.local_path for item in state.repositories)
        has_env = any(item.environment_path for item in state.environment_records)
        has_baseline = bool(state.selected_baseline_program_id) or any(
            candidate.status.startswith("baseline") for candidate in state.program_candidates
        )
        capability_ready = bool(capability and (capability.repo_ready or capability.env_ready or capability.baseline_launch_ready))
        return not (has_repo or has_env or has_baseline or capability_ready)

    def _core_progress_messages(self, state: ResearchState, phase: ResearchPhase, summary: str) -> list[tuple[str, str]]:
        messages: list[tuple[str, str]] = []
        if phase == ResearchPhase.LITERATURE_REVIEW:
            messages.append(
                (
                    "literature_milestone",
                    f"Literature review established the working frame with {len(state.literature_notes)} papers and {len(state.open_questions)} open questions.",
                )
            )
        elif phase == ResearchPhase.ACQUISITION:
            if state.selected_baseline_program_id or state.repositories or state.external_artifacts:
                messages.append(
                    (
                        "acquisition_milestone",
                        "Acquisition updated executable research assets: "
                        f"repositories={len(state.repositories)} artifacts={len(state.external_artifacts)} "
                        f"selected_baseline={state.selected_baseline_program_id or 'unset'}.",
                    )
                )
        elif phase == ResearchPhase.HYPOTHESIS and state.hypotheses:
            latest = state.hypotheses[-1]
            messages.append(("hypothesis", f"Proposed hypothesis {latest.hypothesis_id}: {latest.title}."))
        elif phase == ResearchPhase.METHOD_DESIGN and state.method_designs:
            latest = state.method_designs[-1]
            messages.append(("method_design", f"Designed method {latest.design_id}: {latest.title}."))
        elif phase == ResearchPhase.CODING and state.program_candidates:
            latest = state.program_candidates[-1]
            messages.append(
                (
                    "implementation",
                    f"Prepared program candidate {latest.program_id} with status={latest.status} and {len(latest.changed_files)} changed files.",
                )
            )
        elif phase == ResearchPhase.EXPERIMENT_PLANNING and state.experiment_plans:
            latest = state.experiment_plans[-1]
            messages.append(("experiment_plan", f"Prepared experiment plan {latest.plan_id} for program {latest.program_id}."))
        elif phase == ResearchPhase.EXPERIMENT and state.experiment_records:
            latest = state.experiment_records[-1]
            metric_text = ", ".join(f"{key}={value}" for key, value in latest.metrics.items()) or "no parsed metrics"
            messages.append(
                (
                    "experiment_result",
                    f"Experiment {latest.experiment_id} finished with status={latest.status}; {metric_text}.",
                )
            )
        elif phase == ResearchPhase.REFLECTION and state.reflections:
            latest = state.reflections[-1]
            evidence = latest.evidence[0] if latest.evidence else "no breakthrough evidence recorded"
            messages.append(("reflection", f"Reflection verdict={latest.verdict}. {evidence}"))
        elif phase == ResearchPhase.REPORTING and state.generated_reports:
            messages.append(("reporting", f"Reporting generated {len(state.generated_reports)} durable report artifacts."))
        elif phase == ResearchPhase.HUMAN_INTERVENTION and state.hitl_events:
            latest = state.hitl_events[-1]
            messages.append(("hitl", f"Human intervention status={latest.status.value} for blocker_type={latest.blocker_type}."))
        elif summary.strip():
            messages.append(("phase_summary", summary.strip()))
        return messages

    def _initial_state(self) -> ResearchState:
        state = ResearchState(
            project_name=self.config.project_name,
            run_name=self.config.run_name,
            work_directory=str(self.work_directory),
            research_brief=self.config.research_brief,
        )
        self.memory.save_state(state, label="initial_state")
        return state

    def _normalize_state_assets(self, state: ResearchState) -> None:
        state.external_artifacts = normalize_artifacts(state.external_artifacts)
        state.repositories = normalize_repositories(state.repositories)
        state.environment_records = normalize_environments(state.environment_records)

    def _hydrate_state_from_memory(self, state: ResearchState) -> None:
        state.external_artifacts = upsert_by_attr(
            state.external_artifacts,
            self.memory.load_artifacts(),
            "artifact_id",
        )
        state.repositories = upsert_by_attr(
            state.repositories,
            self.memory.load_repositories(),
            "repo_id",
        )
        state.environment_records = upsert_by_attr(
            state.environment_records,
            self.memory.load_environments(),
            "env_id",
        )
        state.evaluation_memos = upsert_by_attr(
            state.evaluation_memos,
            self.memory.load_evaluation_memos(limit=200),
            "memo_id",
        )
        state.memory_notes = upsert_by_attr(
            state.memory_notes,
            self.memory.load_memory_notes(limit=400),
            "note_id",
        )
        self._normalize_state_assets(state)

    def _sync_environment_records(self, state: ResearchState) -> None:
        records = list(state.environment_records)
        if state.capability_matrix and state.capability_matrix.environment_path:
            env_path = state.capability_matrix.environment_path
            records.append(
                EnvironmentRecord(
                    env_id=canonicalize_env_id(env_path, project_hint=Path(env_path).name),
                    canonical_id=canonicalize_env_id(env_path, project_hint=Path(env_path).name),
                    project_path=str(Path(env_path).parent),
                    environment_path=env_path,
                    state=EnvironmentResolutionState.READY if state.capability_matrix.env_ready else EnvironmentResolutionState.BROKEN,
                    strategy="capability_probe",
                )
            )
        for repository in state.repositories:
            if repository.environment_path:
                records.append(
                    EnvironmentRecord(
                        env_id=repository.environment_id or canonicalize_env_id(repository.environment_path, project_hint=repository.name),
                        canonical_id=repository.environment_id or canonicalize_env_id(repository.environment_path, project_hint=repository.name),
                        project_path=repository.local_path,
                        environment_path=repository.environment_path,
                        repo_id=repository.canonical_id or repository.repo_id,
                        state=EnvironmentResolutionState.READY if repository.bootstrap_status == "ready" else EnvironmentResolutionState.NOT_STARTED,
                        strategy=repository.bootstrap_status,
                    )
                )
        state.environment_records = normalize_environments(records)
        for environment in state.environment_records:
            self.memory.record_environment(environment)

    def _ground_experiment_plans(self, state: ResearchState) -> None:
        grounded_plans = []
        for plan in state.experiment_plans:
            grounded_plan, messages = ground_experiment_plan(plan, state.external_artifacts)
            grounded_plans.append(grounded_plan)
            self.memory.record_experiment_plan(grounded_plan)
            for message in messages:
                self._log(f"Plan grounding: {message}")
        state.experiment_plans = grounded_plans

    def _validate_artifacts(self, state: ResearchState) -> None:
        updated = []
        for artifact in state.external_artifacts:
            if not artifact.local_path:
                updated.append(artifact)
                continue
            local_path = Path(artifact.local_path)
            if artifact.artifact_type not in {"dataset", "checkpoint"} and local_path.suffix.lower() not in {
                ".h5",
                ".hdf5",
                ".pt",
                ".pth",
                ".ckpt",
            }:
                updated.append(artifact)
                continue
            metadata = artifact.metadata or {}
            download_metadata = artifact.download_metadata
            allow_quarantine = True
            if metadata.get("human_provided") or metadata.get("manual_imported"):
                allow_quarantine = False
            if download_metadata and (
                download_metadata.strategy_id in {"local_discovery", "manual_local_import"}
                or download_metadata.source_type == "human_provided_local"
            ):
                allow_quarantine = False
            validated = self.tools.validate_artifact_record(artifact, quarantine_on_failure=allow_quarantine)
            updated.append(validated)
            self.memory.record_artifact(validated)
        state.external_artifacts = updated

    def _recover_artifacts_from_local_sources(self, state: ResearchState) -> None:
        search_roots = sorted(
            dict.fromkeys(
                [
                    str(self.tools.shared_workspace_root),
                    str(self.tools.workspace_family_root),
                    *state.manual_asset_roots,
                ]
            )
        )
        recovered: list[object] = []
        for artifact in state.external_artifacts:
            if artifact.artifact_type not in {"dataset", "checkpoint"}:
                continue
            if artifact.status == "ready_for_training":
                continue
            expected_path = self._artifact_expected_path(artifact)
            metadata = artifact.metadata or {}
            discovered = self.tools.discover_local_artifacts(
                query=expected_path.name,
                search_roots=search_roots,
                artifact_type=artifact.artifact_type,
                canonical_target_id=artifact.canonical_id or artifact.artifact_id,
                expected_checksum=metadata.get("official_md5") or metadata.get("official_checksum"),
                checksum_algorithm=str(metadata.get("checksum_algorithm") or "md5"),
                min_size_bytes=metadata.get("min_size_bytes"),
                required_keys=list(metadata.get("required_keys", [])),
                limit=8,
            )
            ready_candidate = next((item for item in discovered if item.get("ready_for_training")), None)
            if ready_candidate is None:
                continue
            candidate_path = Path(str(ready_candidate["path"]))
            if candidate_path.resolve() == expected_path.resolve() and expected_path.exists():
                continue
            expected_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                if expected_path.exists():
                    expected_path.unlink()
                try:
                    shutil.copy2(candidate_path, expected_path)
                except shutil.SameFileError:
                    pass
            except OSError:
                continue
            updated_download_metadata = (
                artifact.download_metadata.model_copy(
                    update={
                        "source_url": str(candidate_path),
                        "source_type": "local_discovery",
                        "local_path": str(expected_path),
                        "strategy_id": "manager_local_materialization",
                        "validation_status": "downloaded",
                        "attempt_signature": short_hash("manager_local_materialization", str(candidate_path), str(expected_path)),
                    }
                )
                if artifact.download_metadata
                else None
            )
            candidate = artifact.model_copy(
                update={
                    "local_path": str(expected_path),
                    "metadata": {**metadata, "manager_materialized": True},
                    "notes": [*artifact.notes, f"Manager materialized local candidate {candidate_path}"],
                    "download_metadata": updated_download_metadata,
                }
            )
            validated = self.tools.validate_artifact_record(candidate, quarantine_on_failure=False)
            recovered.append(validated)
            self.memory.record_artifact(validated)
            self._log(
                f"Recovered artifact from local source: {artifact.canonical_id or artifact.artifact_id} <- {candidate_path}"
            )
        if recovered:
            state.external_artifacts = upsert_by_attr(
                state.external_artifacts,
                recovered,
                "artifact_id",
            )
            self._normalize_state_assets(state)

    def _auto_materialize_pending_remote_artifacts(self, state: ResearchState) -> None:
        candidates = []
        for artifact in state.external_artifacts:
            if artifact.artifact_type not in {"dataset", "checkpoint"}:
                continue
            if artifact.status == "ready_for_training":
                continue
            if not artifact.source_url:
                continue
            if artifact.status not in {
                "verified_remote",
                "download_failed",
                "blocked",
                "downloaded",
                "checksum_verified",
                "format_verified",
            }:
                continue
            candidates.append(artifact)
        if not candidates:
            return
        candidates.sort(
            key=lambda item: (
                0 if self.tools._infer_exact_target(item) else 1,
                0 if item.artifact_type == "dataset" else 1,
                item.canonical_id or item.artifact_id,
            )
        )
        for artifact in candidates[:6]:
            target_path = self._artifact_expected_path(artifact)
            self._log(
                "Auto-materializing artifact "
                f"{artifact.canonical_id or artifact.artifact_id} -> {target_path}"
            )
            try:
                self.tools.materialize_artifact_record(
                    artifact,
                    strategy_id="manager_auto_materialize",
                    source_type="verified_remote_registry",
                )
            except Exception as error:  # noqa: BLE001
                self._log(
                    "Auto-materialization failed for "
                    f"{artifact.canonical_id or artifact.artifact_id}: {type(error).__name__}: {error}"
                )

    def _refresh_capability_matrix(self, state: ResearchState) -> None:
        self._sync_environment_records(state)
        preferred_environment_path = self._select_preferred_environment_path(state)
        capability_matrix = self.tools.probe_capability_matrix(
            artifacts=state.external_artifacts,
            repository_paths=[repo.local_path for repo in state.repositories],
            environment_path=preferred_environment_path,
        )
        state.capability_matrix = capability_matrix
        self.memory.record_capability_matrix(capability_matrix)

    def _can_short_circuit_acquisition(self, state: ResearchState) -> bool:
        capability = state.capability_matrix
        has_repo = any(item.local_path for item in state.repositories)
        has_env = any(item.environment_path for item in state.environment_records)
        has_baseline = bool(state.selected_baseline_program_id) or any(
            candidate.status.startswith("baseline") for candidate in state.program_candidates
        )
        if not (has_repo and has_env and has_baseline and capability):
            return False
        return bool(
            capability.target_dataset_ready
            or capability.scientific_iteration_ready
            or capability.fallback_assets_available
        )

    def _heuristic_acquisition_output(
        self,
        state: ResearchState,
        summary: str,
    ) -> AcquisitionPhaseOutput:
        environment_snapshot = state.environment_snapshot or self.tools.inspect_compute_environment()
        secret_status = state.secret_status or self.tools.inspect_secret_status()
        next_actions = [
            "Use the acquired baseline route to enter problem framing and diagnosis.",
            "Plan only experiments that match the currently executable capability state.",
        ]
        if state.capability_matrix and not state.capability_matrix.baseline_ready_to_launch:
            next_actions.append(
                "Prefer fallback execution or preflight-gated baseline evaluation if the preferred GPU runtime is not launch-ready."
            )
        return AcquisitionPhaseOutput(
            summary=summary,
            environment_snapshot=environment_snapshot,
            environment_records=state.environment_records,
            capability_matrix=state.capability_matrix,
            secret_status=secret_status,
            external_artifacts=state.external_artifacts,
            repositories=state.repositories,
            program_candidates=state.program_candidates,
            selected_baseline_program_id=state.selected_baseline_program_id,
            acquisition_notes=[
                "deterministic_manager_bootstrap",
                "acquisition_short_circuit_after_executable_route",
            ],
            semantic_notes=[
                "Acquisition now short-circuits once one executable baseline or fallback route exists."
            ],
            next_actions=next_actions,
        )

    def _select_preferred_environment_path(self, state: ResearchState) -> str | None:
        baseline_repo_id = None
        if state.selected_baseline_program_id:
            baseline_program = next(
                (item for item in state.program_candidates if item.program_id == state.selected_baseline_program_id),
                None,
            )
            if baseline_program is not None:
                baseline_repo_id = baseline_program.repo_id

        preferred_candidates: list[str] = []
        if baseline_repo_id:
            for repository in state.repositories:
                if (repository.canonical_id or repository.repo_id) == baseline_repo_id and repository.environment_path:
                    preferred_candidates.append(repository.environment_path)
        for repository in state.repositories:
            if repository.environment_path:
                preferred_candidates.append(repository.environment_path)
        for environment in state.environment_records:
            if environment.state.value == "ready":
                preferred_candidates.append(environment.environment_path)
        if state.capability_matrix and state.capability_matrix.environment_path:
            preferred_candidates.append(state.capability_matrix.environment_path)

        deduped: list[Path] = []
        seen: set[str] = set()
        for candidate in preferred_candidates:
            try:
                resolved = self.tools._resolve_path(candidate, default_root=self.tools.managed_env_root).resolve()
            except Exception:
                continue
            key = str(resolved)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(resolved)
        if not deduped:
            return state.capability_matrix.environment_path if state.capability_matrix else None

        repo_tokens = {
            Path(repository.local_path).name.lower()
            for repository in state.repositories
            if repository.local_path
        }

        def _score(path: Path) -> tuple[int, int, int, int]:
            env_name = path.name.lower()
            repo_match = int(any(token and token in env_name for token in repo_tokens))
            auxiliary = int(any(token in env_name for token in {"validator", "probe", "check", "inspect"}))
            ready_record = int(
                any(
                    environment.environment_path == str(path)
                    and environment.state.value == "ready"
                    for environment in state.environment_records
                )
            )
            current_run = int(str(path.resolve()).startswith(str(self.work_directory.resolve())))
            return (
                current_run,
                ready_record,
                repo_match,
                int(not auxiliary),
                int((path / "bin" / "python").exists()),
            )

        selected = max(deduped, key=_score)
        return str(selected)

    def _ensure_baseline_candidates(self, state: ResearchState) -> None:
        if any(candidate.status.startswith("baseline") for candidate in state.program_candidates):
            if not state.selected_baseline_program_id:
                baseline = next(
                    (candidate for candidate in state.program_candidates if candidate.status.startswith("baseline")),
                    None,
                )
                if baseline is not None:
                    state.selected_baseline_program_id = baseline.program_id
            return

        synthesized: list[ProgramCandidate] = []
        for repository in state.repositories:
            if not repository.local_path:
                continue
            manifests = repository.detected_manifests or []
            entry_hint = repository.entrypoints[0] if repository.entrypoints else None
            candidate = ProgramCandidate(
                program_id=f"baseline-{short_hash(repository.canonical_id or repository.repo_id, repository.local_path)}",
                title=f"Baseline from {repository.name}",
                summary="Synthesized baseline candidate from acquired repository state.",
                repo_id=repository.canonical_id or repository.repo_id,
                workspace_path=repository.local_path,
                entry_command_hint=entry_hint,
                status="baseline_discovered",
                changed_files=[],
                notes=[
                    "manager_synthesized_baseline",
                    *(f"manifest:{item}" for item in manifests[:4]),
                ],
            )
            synthesized.append(candidate)

        if not synthesized:
            return

        state.program_candidates = upsert_by_attr(state.program_candidates, synthesized, "program_id")
        for candidate in synthesized:
            self.memory.register_program(candidate)
        if not state.selected_baseline_program_id:
            state.selected_baseline_program_id = synthesized[0].program_id

    def _refresh_classified_failures(self, state: ResearchState) -> None:
        classified = classify_state_failures(
            state,
            max_transfer_attempts=self.config.retrieval.max_transfer_attempts,
        )
        existing = {failure.failure_id: failure for failure in state.classified_failures}
        for failure in classified:
            existing[failure.failure_id] = failure
            self.memory.record_failure(failure)
        state.classified_failures = list(existing.values())

    def _refresh_blockers(self, state: ResearchState) -> None:
        blockers, cycle_delta = refresh_blocker_registry(
            state,
            repeat_threshold=2,
            stagnation_threshold=4,
        )
        state.blocker_registry = blockers
        if state.cycle_deltas and state.cycle_deltas[-1].cycle_index == cycle_delta.cycle_index:
            state.cycle_deltas[-1] = cycle_delta
        else:
            state.cycle_deltas.append(cycle_delta)
        self.memory.record_cycle_delta(cycle_delta)
        for blocker in blockers:
            self.memory.record_blocker(blocker)

    def _post_phase_sync(self, state: ResearchState, spec: PhaseSpec) -> None:
        self._hydrate_state_from_memory(state)
        self._normalize_state_assets(state)
        if spec.phase == ResearchPhase.ACQUISITION:
            self._auto_materialize_pending_remote_artifacts(state)
            self._hydrate_state_from_memory(state)
            self._normalize_state_assets(state)
        if spec.phase in {ResearchPhase.ACQUISITION, ResearchPhase.PREFLIGHT_VALIDATION}:
            self._recover_artifacts_from_local_sources(state)
            self._validate_artifacts(state)
        if spec.phase == ResearchPhase.ACQUISITION:
            self._ensure_baseline_candidates(state)
        if spec.phase in {
            ResearchPhase.ACQUISITION,
            ResearchPhase.EXPERIMENT_PLANNING,
            ResearchPhase.PREFLIGHT_VALIDATION,
            ResearchPhase.EXPERIMENT,
        }:
            self._refresh_capability_matrix(state)
            self._refresh_classified_failures(state)
            self._refresh_blockers(state)

    def _log_cycle_context(self, state: ResearchState, route: CycleRoute) -> None:
        latest_delta = state.cycle_deltas[-1] if state.cycle_deltas else None
        active_blockers = [item for item in state.blocker_registry if item.repeat_count >= 1]
        blocker_summary = ", ".join(
            f"{item.blocker_type}:{item.target_entity}:repeat={item.repeat_count}:exhausted={item.route_exhausted}"
            for item in active_blockers[:6]
        ) or "none"
        delta_summary = ", ".join(latest_delta.summary[:6]) if latest_delta else "no_state_delta"
        self._log(f"Blocker summary: {blocker_summary}.")
        self._log(f"State delta: {delta_summary}.")
        self._log(
            f"Route rationale: id={route.route_id} focus={list(route.focus)} allowed=True reason={route.reason}"
        )

    def _record_route_decision(self, state: ResearchState, route: CycleRoute) -> None:
        latest_delta = state.cycle_deltas[-1] if state.cycle_deltas else None
        decision = RouteDecisionRecord(
            cycle_index=state.cycle_index,
            route_id=route.route_id,
            rationale=route.reason,
            blockers_considered=[item.blocker_id for item in state.blocker_registry if item.repeat_count >= 1],
            allowed=True,
            material_change_summary=list(latest_delta.summary if latest_delta else []),
        )
        state.route_history.append(decision)
        state.active_route_id = route.route_id
        state.active_route_reason = route.reason
        state.active_route_focus = list(route.focus)
        self.memory.record_route_decision(decision)

    def _log_hitl_event(self, state: ResearchState, event: HITLEvent) -> None:
        state.hitl_events.append(event)
        self.memory.record_hitl_event(event)
        self.memory.save_state(state, label="hitl_state")

    def _render_hitl_prompt(self, prompt_text: str) -> None:
        for line in prompt_text.rstrip().splitlines():
            self._log(line)

    def _refresh_hitl_context(
        self,
        state: ResearchState,
        fallback_blockers: list[object],
    ) -> tuple[list[object], list[object], str, list[str]]:
        blockers = select_hitl_blockers(
            state,
            repeat_threshold=self.config.execution.hitl_blocker_repeat_threshold,
            strategy_threshold=self.config.execution.hitl_strategy_threshold,
            allow_active=True,
        )
        if fallback_blockers:
            fallback_ids = {item.blocker_id for item in fallback_blockers}
            fallback_targets = {item.target_entity for item in fallback_blockers}
            relevant = [
                item
                for item in blockers
                if item.blocker_id in fallback_ids or item.target_entity in fallback_targets
            ]
            if not relevant:
                relevant = [
                    item
                    for item in state.blocker_registry
                    if item.blocker_id in fallback_ids or item.target_entity in fallback_targets
                ]
            blockers = relevant or blockers or list(fallback_blockers)
        artifacts = blocked_artifacts_for_hitl(state, blockers)
        prompt_text, requested_actions = build_hitl_prompt(state, blockers, artifacts)
        return blockers, artifacts, prompt_text, requested_actions

    def _human_instruction_requests_reduced_scope(self, instruction: str) -> bool:
        lowered = instruction.lower()
        return any(
            marker in lowered
            for marker in (
                "skip ",
                "skip this target",
                "reduce scope",
                "reduced scope",
                "only continue",
                "only run",
                "stop trying",
            )
        )

    def _apply_skip_scope(self, state: ResearchState, instruction: str | None = None) -> str:
        lowered = (instruction or "").lower()
        targets: list[str] = []
        if "reaction" in lowered or "reacdiff" in lowered:
            targets.append("ReactionDiffusion")
        if "burgers" in lowered:
            targets.append("Burgers")
        if not targets:
            if any((item.semantic_spec and item.semantic_spec.equation == "ReactionDiffusion") for item in state.external_artifacts):
                targets.append("ReactionDiffusion")
            elif any((item.semantic_spec and item.semantic_spec.equation == "Burgers") for item in state.external_artifacts):
                targets.append("Burgers")
        skipped: list[str] = []
        for artifact in state.external_artifacts:
            equation = artifact.semantic_spec.equation if artifact.semantic_spec else None
            if equation in targets:
                artifact.metadata["human_skip"] = True
                skipped.append(artifact.canonical_id or artifact.artifact_id)
        state.skipped_target_entities = sorted(set([*state.skipped_target_entities, *skipped]))
        note = (
            f"Human requested reduced scope. Skipped targets for equations={targets or ['unspecified']}."
        )
        state.human_guidance_notes.append(note)
        return note

    def _materialize_manual_files(
        self,
        state: ResearchState,
        search_roots: list[str],
        artifacts: list[object],
    ) -> tuple[list[str], list[object]]:
        validation_summary: list[str] = []
        updated_artifacts: list[object] = []
        for artifact in artifacts:
            expected_path = self._artifact_expected_path(artifact)
            if expected_path.exists():
                candidate = artifact.model_copy(
                    update={
                        "metadata": {**artifact.metadata, "human_provided": True},
                    }
                )
                validated = self.tools.validate_artifact_record(candidate, quarantine_on_failure=False)
                updated_artifacts.append(validated)
                if validated.status == "ready_for_training":
                    validation_summary.append(f"validated existing file {expected_path.name}")
                elif validated.status == "checksum_verified":
                    validation_summary.append(
                        f"checksum verified but format validation incomplete for {expected_path.name}"
                    )
                else:
                    validation_summary.append(
                        f"existing file still invalid {expected_path.name}: status={validated.status}"
                    )
                continue
            filename = expected_path.name
            located = None
            for root in search_roots:
                candidate_root = Path(root)
                if not candidate_root.exists():
                    continue
                matches = list(candidate_root.rglob(filename))
                if matches:
                    located = matches[0]
                    break
            if located is None:
                validation_summary.append(f"missing {filename}")
                continue
            expected_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(located, expected_path)
            updated_download_metadata = (
                artifact.download_metadata.model_copy(
                    update={
                        "local_path": str(expected_path),
                        "source_type": "human_provided_local",
                        "strategy_id": "manual_local_import",
                        "validation_status": "downloaded",
                    }
                )
                if artifact.download_metadata
                else None
            )
            candidate = artifact.model_copy(
                update={
                    "local_path": str(expected_path),
                    "metadata": {**artifact.metadata, "human_provided": True, "manual_imported": True},
                    "notes": [*artifact.notes, f"Imported from human-provided path {located}"],
                    "download_metadata": updated_download_metadata,
                }
            )
            validated = self.tools.validate_artifact_record(candidate, quarantine_on_failure=False)
            updated_artifacts.append(validated)
            if validated.status == "ready_for_training":
                validation_summary.append(f"validated imported file {filename}")
            elif validated.status == "checksum_verified":
                validation_summary.append(
                    f"imported file checksum verified but format validation incomplete {filename}"
                )
            else:
                validation_summary.append(f"imported but still invalid {filename}: status={validated.status}")
        return validation_summary, updated_artifacts

    def _revalidate_after_human_confirmation(self, state: ResearchState, blockers: list[object]) -> list[str]:
        self._hydrate_state_from_memory(state)
        self._normalize_state_assets(state)
        artifacts = blocked_artifacts_for_hitl(state, blockers)
        expected_roots = sorted(
            {
                str(self._artifact_expected_path(item).parent)
                for item in artifacts
            }
        )
        search_roots = sorted(dict.fromkeys([*expected_roots, *state.manual_asset_roots]))
        validation_summary, updated_artifacts = self._materialize_manual_files(state, search_roots, artifacts)
        if updated_artifacts:
            state.external_artifacts = upsert_by_attr(
                state.external_artifacts,
                updated_artifacts,
                "artifact_id",
            )
        self._normalize_state_assets(state)
        self._validate_artifacts(state)
        self._refresh_capability_matrix(state)
        self._refresh_classified_failures(state)
        self._refresh_blockers(state)
        self.memory.save_state(state, label="post_hitl_revalidation")
        return validation_summary

    def _handle_hitl(self, state: ResearchState, route: CycleRoute | None = None) -> bool:
        if not self.config.execution.hitl_enabled:
            return False
        if route is not None and route.route_id != "blocked-terminal":
            return False
        self._hydrate_state_from_memory(state)
        self._normalize_state_assets(state)
        if state.capability_matrix and state.capability_matrix.target_dataset_preparing:
            return False
        blockers = should_trigger_hitl(
            state,
            repeat_threshold=self.config.execution.hitl_blocker_repeat_threshold,
            strategy_threshold=self.config.execution.hitl_strategy_threshold,
        )
        if not blockers:
            return False
        blockers, artifacts, prompt_text, requested_actions = self._refresh_hitl_context(state, blockers)
        event = HITLEvent(
            event_id=f"hitl-{short_hash(state.run_name, str(state.cycle_index), now_utc())}",
            cycle_index=state.cycle_index,
            blocker_ids=[item.blocker_id for item in blockers],
            blocker_type=blockers[0].blocker_type,
            target_entities=[item.target_entity for item in blockers],
            escalation_reason="Repeated autonomous recovery attempts did not unblock progress.",
            requested_actions=requested_actions,
            prompt_text=prompt_text,
            status=HITLStatus.REQUESTED,
        )
        state.current_phase = ResearchPhase.HUMAN_INTERVENTION
        state.phase_history.append(ResearchPhase.HUMAN_INTERVENTION.value)
        self._log("Entering human_intervention phase due to exhausted autonomous recovery.")
        self._render_hitl_prompt(prompt_text)
        self._log_hitl_event(state, event)

        while True:
            try:
                response = input("HITL> ").strip()
            except EOFError:
                response = "4"
            if not response:
                continue
            selection = response.split(".", 1)[0].strip()
            if selection == "1":
                validation_summary = self._revalidate_after_human_confirmation(state, blockers)
                updated_event = event.model_copy(
                    update={
                        "response_type": HumanResponseType.CONFIRMED_DONE,
                        "response_text": response,
                        "validation_summary": validation_summary,
                        "status": HITLStatus.REVALIDATED,
                        "responded_at": now_utc(),
                    }
                )
                if state.capability_matrix and (
                    state.capability_matrix.target_dataset_ready or state.capability_matrix.baseline_launch_ready
                ):
                    state.blocked_reason = None
                    state.termination_decision = None
                    updated_event = updated_event.model_copy(
                        update={
                            "status": HITLStatus.RESUMED,
                            "material_effect": "validated_artifacts_ready",
                        }
                    )
                    state.hitl_events[-1] = updated_event
                    self.memory.record_hitl_event(updated_event)
                    self._log("HITL revalidation succeeded. Resuming autonomous execution.")
                    return True
                updated_event = updated_event.model_copy(
                    update={
                        "status": HITLStatus.STILL_BLOCKED,
                        "material_effect": "validation_failed",
                    }
                )
                blockers, artifacts, prompt_text, requested_actions = self._refresh_hitl_context(state, blockers)
                updated_event = updated_event.model_copy(
                    update={
                        "blocker_ids": [item.blocker_id for item in blockers],
                        "target_entities": [item.target_entity for item in blockers],
                        "requested_actions": requested_actions,
                        "prompt_text": prompt_text,
                    }
                )
                state.hitl_events[-1] = updated_event
                self.memory.record_hitl_event(updated_event)
                event = updated_event
                self._log(
                    "HITL confirmation did not unblock the run. Validation summary: "
                    + "; ".join(validation_summary or ["no matching validated files found"])
                )
                self._log("Updated HITL request follows.")
                self._render_hitl_prompt(prompt_text)
                continue
            if selection == "2":
                instruction = input("Instruction> ").strip()
                manual_paths = extract_absolute_paths(instruction)
                if manual_paths:
                    state.manual_asset_roots = sorted(set([*state.manual_asset_roots, *manual_paths]))
                state.human_guidance_notes.append(instruction)
                validation_summary = []
                skip_note = None
                if self._human_instruction_requests_reduced_scope(instruction):
                    skip_note = self._apply_skip_scope(state, instruction)
                    validation_summary.append(skip_note)
                if manual_paths:
                    validation_summary.extend(self._revalidate_after_human_confirmation(state, blockers))
                elif skip_note is not None:
                    self._refresh_capability_matrix(state)
                    self._refresh_classified_failures(state)
                    self._refresh_blockers(state)
                if state.capability_matrix and (
                    state.capability_matrix.target_dataset_ready
                    or state.capability_matrix.baseline_launch_ready
                    or skip_note is not None
                ):
                    state.blocked_reason = None
                    state.termination_decision = None
                    updated_event = event.model_copy(
                        update={
                            "response_type": HumanResponseType.INSTRUCTION,
                            "response_text": instruction,
                            "validation_summary": validation_summary,
                            "status": HITLStatus.RESUMED,
                            "material_effect": "human_strategy_override",
                            "responded_at": now_utc(),
                        }
                    )
                    state.hitl_events[-1] = updated_event
                    self.memory.record_hitl_event(updated_event)
                    self._log(
                        "Recorded human instruction and updated routing state. "
                        f"manual_asset_roots={state.manual_asset_roots} guidance_count={len(state.human_guidance_notes)}."
                    )
                    return True
                if manual_paths:
                    blockers, artifacts, prompt_text, requested_actions = self._refresh_hitl_context(state, blockers)
                    updated_event = event.model_copy(
                        update={
                            "response_type": HumanResponseType.INSTRUCTION,
                            "response_text": instruction,
                            "validation_summary": validation_summary,
                            "status": HITLStatus.STILL_BLOCKED,
                            "material_effect": "human_instruction_needs_followup",
                            "requested_actions": requested_actions,
                            "prompt_text": prompt_text,
                            "blocker_ids": [item.blocker_id for item in blockers],
                            "target_entities": [item.target_entity for item in blockers],
                            "responded_at": now_utc(),
                        }
                    )
                    state.hitl_events[-1] = updated_event
                    self.memory.record_hitl_event(updated_event)
                    event = updated_event
                    self._log(
                        "Human instruction did not fully unblock the run. Validation summary: "
                        + "; ".join(validation_summary or ["no matching validated files found"])
                    )
                    self._log("Updated HITL request follows.")
                    self._render_hitl_prompt(prompt_text)
                    continue
                updated_event = event.model_copy(
                    update={
                        "response_type": HumanResponseType.INSTRUCTION,
                        "response_text": instruction,
                        "validation_summary": validation_summary,
                        "status": HITLStatus.RESPONDED,
                        "material_effect": "human_strategy_override",
                        "responded_at": now_utc(),
                    }
                )
                state.hitl_events[-1] = updated_event
                self.memory.record_hitl_event(updated_event)
                self._log(
                    "Recorded human instruction. "
                    f"manual_asset_roots={state.manual_asset_roots} guidance_count={len(state.human_guidance_notes)}."
                )
                return True
            if selection == "3":
                instruction = input("Reduced-scope instruction> ").strip()
                note = self._apply_skip_scope(state, instruction)
                updated_event = event.model_copy(
                    update={
                        "response_type": HumanResponseType.REDUCE_SCOPE,
                        "response_text": instruction or "skip target and continue with reduced scope",
                        "validation_summary": [note],
                        "status": HITLStatus.RESUMED,
                        "material_effect": "reduced_scope",
                        "responded_at": now_utc(),
                    }
                )
                state.hitl_events[-1] = updated_event
                self.memory.record_hitl_event(updated_event)
                state.blocked_reason = None
                state.termination_decision = None
                self._log(note)
                return True
            if selection == "4":
                updated_event = event.model_copy(
                    update={
                        "response_type": HumanResponseType.ABORT,
                        "response_text": response,
                        "status": HITLStatus.ABORTED,
                        "material_effect": "human_abort",
                        "responded_at": now_utc(),
                    }
                )
                state.hitl_events[-1] = updated_event
                self.memory.record_hitl_event(updated_event)
                state.blocked_reason = "Aborted by human during HITL escalation."
                state.termination_decision = state.blocked_reason
                self._log(state.blocked_reason)
                return True
            self._log("Unrecognized HITL response. Reply with 1, 2, 3, or 4.")

    def _run_phase(self, state: ResearchState, spec: PhaseSpec) -> str:
        if spec.phase in {ResearchPhase.EXPERIMENT_PLANNING, ResearchPhase.EXPERIMENT}:
            self._ground_experiment_plans(state)
        state.current_phase = spec.phase
        state.phase_history.append(spec.phase.value)
        agent = self.agents[spec.agent_key]
        before_signature = self._acquisition_recovery_signature(state) if spec.phase == ResearchPhase.ACQUISITION else None
        self.tools.set_runtime_context(phase=spec.phase.value, cycle_index=state.cycle_index)
        self._log(
            f"Starting phase {spec.phase.value} with {agent.name}. Cycle={state.cycle_index}."
        )
        self.memory.record_agent_event(
            agent_name=agent.name,
            phase=spec.phase,
            status="started",
            cycle_index=state.cycle_index,
            content=f"Starting phase {spec.phase.value}.",
            payload={"outputs": list(spec.outputs)},
        )
        if spec.phase == ResearchPhase.ACQUISITION:
            bootstrap_summary = self._bootstrap_minimal_acquisition_recovery(state)
            if bootstrap_summary is not None:
                self._post_phase_sync(state, spec)
                output = self._heuristic_acquisition_output(state, bootstrap_summary)
                summary = agent.apply_output(state, self.tools, output)
                self.memory.record_phase(spec.phase, summary, list(spec.outputs))
                self.memory.record_agent_event(
                    agent_name=agent.name,
                    phase=spec.phase,
                    status="completed",
                    cycle_index=state.cycle_index,
                    content=summary,
                    payload={"outputs": list(spec.outputs), "phase_snapshot": self._phase_snapshot(state, spec.phase)},
                )
                for kind, message in self._core_progress_messages(state, spec.phase, summary):
                    self.memory.record_core_progress(
                        message,
                        kind=kind,
                        phase=spec.phase,
                        agent_name=agent.name,
                        cycle_index=state.cycle_index,
                        payload={"summary": summary},
                    )
                self.memory.save_state(state, label=spec.phase.value)
                self._log(
                    f"Completed phase {spec.phase.value} with {agent.name} via deterministic bootstrap. Summary: {summary}"
                )
                for line in self._phase_snapshot(state, spec.phase):
                    self._log(line)
                return summary
        if self._runtime_degraded:
            summary = self._recover_phase_without_runtime(state, spec, agent)
            if summary is not None:
                return summary
        try:
            summary = agent.run(state, self.tools, self.runtime)
            if spec.phase in {ResearchPhase.EXPERIMENT_PLANNING, ResearchPhase.PREFLIGHT_VALIDATION}:
                self._ground_experiment_plans(state)
            self._post_phase_sync(state, spec)
            if spec.phase == ResearchPhase.ACQUISITION and self._needs_acquisition_bootstrap(state):
                bootstrap_summary = self._bootstrap_minimal_acquisition_recovery(state)
                if bootstrap_summary is not None:
                    self._post_phase_sync(state, spec)
                    summary = f"{summary} {bootstrap_summary}".strip()
            self.memory.record_phase(spec.phase, summary, list(spec.outputs))
            self.memory.record_agent_event(
                agent_name=agent.name,
                phase=spec.phase,
                status="completed",
                cycle_index=state.cycle_index,
                content=summary,
                payload={"outputs": list(spec.outputs), "phase_snapshot": self._phase_snapshot(state, spec.phase)},
            )
            for kind, message in self._core_progress_messages(state, spec.phase, summary):
                self.memory.record_core_progress(
                    message,
                    kind=kind,
                    phase=spec.phase,
                    agent_name=agent.name,
                    cycle_index=state.cycle_index,
                    payload={"summary": summary},
                )
            self.memory.save_state(state, label=spec.phase.value)
            self._log(
                f"Completed phase {spec.phase.value} with {agent.name}. Summary: {summary}"
            )
            for line in self._phase_snapshot(state, spec.phase):
                self._log(line)
            return summary
        except Exception as exc:
            if self._should_disable_runtime(exc):
                self._runtime_degraded = True
                self._log(
                    f"Runtime backend marked degraded after {type(exc).__name__}: {exc}. "
                    "Subsequent recoverable phases will bypass live model calls."
                )
            if spec.phase == ResearchPhase.LITERATURE_REVIEW:
                summary = self._recover_literature_phase(state, spec, agent)
                if summary is not None:
                    return summary
            if spec.phase == ResearchPhase.ACQUISITION:
                self._hydrate_state_from_memory(state)
                self._post_phase_sync(state, spec)
                after_signature = self._acquisition_recovery_signature(state)
                bootstrap_summary: str | None = None
                if self._needs_acquisition_bootstrap(state) or before_signature == after_signature:
                    try:
                        bootstrap_summary = self._bootstrap_minimal_acquisition_recovery(state)
                    except Exception as bootstrap_error:  # noqa: BLE001
                        self._log(
                            "Acquisition bootstrap recovery failed with "
                            f"{type(bootstrap_error).__name__}: {bootstrap_error}"
                        )
                        bootstrap_summary = None
                    if bootstrap_summary is not None:
                        self._post_phase_sync(state, spec)
                        after_signature = self._acquisition_recovery_signature(state)
                if before_signature != after_signature:
                    summary = bootstrap_summary or (
                        "Recovered acquisition state from persisted tool side effects after model/runtime failure. "
                        f"artifacts={len(state.external_artifacts)} repositories={len(state.repositories)} "
                        f"environments={len(state.environment_records)}"
                    )
                    self.memory.record_phase(spec.phase, summary, list(spec.outputs))
                    self.memory.record_agent_event(
                        agent_name=agent.name,
                        phase=spec.phase,
                        status="recovered",
                        cycle_index=state.cycle_index,
                        content=summary,
                        payload={"outputs": list(spec.outputs), "phase_snapshot": self._phase_snapshot(state, spec.phase)},
                    )
                    for kind, message in self._core_progress_messages(state, spec.phase, summary):
                        self.memory.record_core_progress(
                            message,
                            kind=kind,
                            phase=spec.phase,
                            agent_name=agent.name,
                            cycle_index=state.cycle_index,
                            payload={"summary": summary, "recovered": True},
                        )
                    self.memory.save_state(state, label=spec.phase.value)
                    self._log(
                        f"Completed phase {spec.phase.value} with {agent.name} via recovery. Summary: {summary}"
                    )
                    for line in self._phase_snapshot(state, spec.phase):
                        self._log(line)
                    return summary
            summary = self._recover_phase_with_heuristics(state, spec, agent)
            if summary is not None:
                return summary
            stack = traceback.format_exc()
            self.memory.record_agent_event(
                agent_name=agent.name,
                phase=spec.phase,
                status="failed",
                cycle_index=state.cycle_index,
                content=stack,
                payload={"outputs": list(spec.outputs)},
            )
            self._log(
                f"Phase {spec.phase.value} with {agent.name} failed. Traceback follows.\n{stack}"
            )
            self.memory.save_state(state, label=f"{spec.phase.value}_failed")
            raise
        finally:
            self.tools.set_runtime_context(phase=None, cycle_index=None)

    def _recover_phase_without_runtime(self, state: ResearchState, spec: PhaseSpec, agent) -> str | None:
        if spec.phase == ResearchPhase.LITERATURE_REVIEW:
            return self._recover_literature_phase(state, spec, agent)  # type: ignore[arg-type]
        return self._recover_phase_with_heuristics(state, spec, agent)

    def _should_disable_runtime(self, exc: Exception) -> bool:
        message = f"{type(exc).__name__}: {exc}".lower()
        return any(
            token in message
            for token in [
                "insufficient credits",
                "error code: 402",
                "provider unavailable",
                "filtered.input",
                "max turns",
            ]
        )

    def _recover_literature_phase(
        self,
        state: ResearchState,
        spec: PhaseSpec,
        agent: LiteratureAgent,
    ) -> str | None:
        self._hydrate_state_from_memory(state)
        candidate_paths: list[Path] = []
        seen: set[str] = set()
        paper_artifacts = [
            artifact.local_path
            for artifact in state.external_artifacts
            if artifact.artifact_type == "paper" and artifact.local_path
        ]
        for raw_path in paper_artifacts:
            path = Path(raw_path)
            if not path.exists():
                continue
            resolved = str(path.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            candidate_paths.append(path)
        external_assets_root = self.work_directory / "external_assets"
        if external_assets_root.exists():
            for path in sorted(external_assets_root.rglob("*.pdf")):
                if not path.is_file():
                    continue
                resolved = str(path.resolve())
                if resolved in seen:
                    continue
                seen.add(resolved)
                candidate_paths.append(path)
        if not candidate_paths:
            return None
        paper_payloads: list[dict[str, str]] = []
        for path in candidate_paths[:6]:
            try:
                extracted = self.tools.extract_pdf_text(str(path), max_pages=4)
            except Exception:
                continue
            text = str(extracted.get("text") or "").strip()
            if not text:
                continue
            paper_payloads.append(
                {
                    "path": str(path),
                    "filename": path.name,
                    "excerpt": text[:1200],
                }
            )
            if len(paper_payloads) >= 4:
                break
        if not paper_payloads:
            return None
        compact_brief = {
            "title": state.research_brief.title[:180],
            "question": state.research_brief.question[:300],
            "objectives": [item[:140] for item in state.research_brief.objectives[:3]],
            "constraints": [item[:140] for item in state.research_brief.constraints[:4]],
            "domain_tags": state.research_brief.domain_tags[:6],
        }
        try:
            recovery_output = self.runtime.run_structured(
                specialist_name="LiteratureRecoveryAgent",
                instructions=(
                    agent.build_instructions(state).strip()
                    + "\n\n"
                    + "The prior tool-using literature phase gathered source material but failed to emit valid structured JSON.\n"
                    + "Use only the provided paper excerpts and research brief.\n"
                    + "Do not call tools.\n"
                    + "Keep the recovered output compact: at most 4 paper notes, concise abstracts, and 2-3 high-signal bullets per list field.\n"
                    + "Produce the final structured literature review output now."
                ),
                payload={
                    "research_brief": compact_brief,
                    "paper_excerpts": paper_payloads,
                    "already_known_papers": [note.title for note in state.literature_notes],
                },
                session_id=f"{state.run_name}-literature-recovery-cycle-{state.cycle_index}",
                output_type=LiteraturePhaseOutput,
                tools=[],
                runtime_timeout_seconds=max(180, self.config.runtime.request_timeout_seconds),
            )
        except Exception:
            recovery_output = self._heuristic_literature_output_from_sources(compact_brief, paper_payloads)
        summary = agent.apply_output(state, self.tools, recovery_output)
        self.memory.record_phase(spec.phase, summary, list(spec.outputs))
        self.memory.record_agent_event(
            agent_name=agent.name,
            phase=spec.phase,
            status="recovered",
            cycle_index=state.cycle_index,
            content=summary,
            payload={"outputs": list(spec.outputs), "phase_snapshot": self._phase_snapshot(state, spec.phase)},
        )
        for kind, message in self._core_progress_messages(state, spec.phase, summary):
            self.memory.record_core_progress(
                message,
                kind=kind,
                phase=spec.phase,
                agent_name=agent.name,
                cycle_index=state.cycle_index,
                payload={"summary": summary, "recovered": True},
            )
        self.memory.save_state(state, label=spec.phase.value)
        self._log(
            f"Completed phase {spec.phase.value} with {agent.name} via recovery. Summary: {summary}"
        )
        for line in self._phase_snapshot(state, spec.phase):
            self._log(line)
        return summary

    def _bootstrap_minimal_acquisition_recovery(self, state: ResearchState) -> str | None:
        query_text = " ".join(
            [
                state.research_brief.title,
                state.research_brief.question,
                *state.research_brief.domain_tags,
                *[note.title for note in state.literature_notes[:4]],
            ]
        ).lower()
        attached_repository = self._bootstrap_local_repository(state, query_text)
        if attached_repository is not None:
            self._log(
                "Acquisition bootstrap attached local repository "
                f"{attached_repository.name} at {attached_repository.local_path}."
            )
        bootstrapped_repos = 1 if attached_repository is not None else 0
        bootstrapped_envs = 1 if attached_repository and attached_repository.environment_path else 0
        if attached_repository is None:
            repo_candidates: list[dict[str, str]] = []
            queries = []
            if "pdebench" in query_text:
                queries.append("PDEBench benchmark official repository neural operators")
            if "deeponet" in query_text:
                queries.append("Physics-informed-DeepONets PredictiveIntelligenceLab")
            if "fourier neural operator" in query_text or "fno" in query_text or "neural operator" in query_text:
                queries.append("neuraloperator official repository")
            if not queries:
                queries.append(state.research_brief.title[:120])
            for query in queries[:3]:
                try:
                    repo_candidates.extend(self.tools.search_github_repositories(query, max_results=3))
                except Exception:
                    continue
            if not repo_candidates and "pdebench" in query_text:
                repo_candidates.append(
                    {
                        "name": "PDEBench",
                        "full_name": "pdebench/PDEBench",
                        "html_url": "https://github.com/pdebench/PDEBench",
                        "default_branch": "main",
                    }
                )
            for candidate in repo_candidates[:2]:
                repo_url = str(candidate.get("html_url") or "").strip()
                if not repo_url:
                    continue
                clone_result = self.tools.clone_repository(repo_url, destination_name=str(candidate.get("name") or "repository"))
                if clone_result.get("status") not in {"cloned", "available", "archive_downloaded"}:
                    continue
                local_path = str(clone_result.get("path") or "")
                if not local_path:
                    continue
                attached_repository = self._attach_repository_record(
                    state,
                    local_path=local_path,
                    repo_url=repo_url,
                    repo_name=str(candidate.get("name") or Path(local_path).name),
                    repo_id=str(candidate.get("full_name") or clone_result.get("canonical_id") or Path(local_path).name),
                    resolution_source=str(candidate.get("resolution_source") or "manager_bootstrap_remote"),
                )
                if attached_repository is not None:
                    self._log(
                        "Acquisition bootstrap cloned repository "
                        f"{attached_repository.name} at {attached_repository.local_path}."
                    )
                    bootstrapped_repos += 1
                    if attached_repository.environment_path:
                        bootstrapped_envs += 1
                    break

        if attached_repository is not None:
            self._seed_bootstrap_artifacts(state, attached_repository)

        if bootstrapped_repos == 0 and bootstrapped_envs == 0:
            return None
        return (
            "Recovered acquisition by manager bootstrap after model/runtime failure. "
            f"repositories={len(state.repositories)} environments={len(state.environment_records)} "
            f"bootstrapped_repos={bootstrapped_repos} bootstrapped_envs={bootstrapped_envs}"
        )

    def _discover_local_repository_paths(self, query_text: str) -> list[Path]:
        family_root = self.tools.workspace_family_root
        candidates: list[Path] = []
        preferred_patterns = [
            "*/external_assets/repos/*",
            "*/workspaces/*/pdebench",
            "*/workspaces/*",
        ]
        for pattern in preferred_patterns:
            for path in family_root.glob(pattern):
                if not path.is_dir():
                    continue
                if not (path / "pdebench" / "models" / "train_models_forward.py").exists():
                    continue
                candidates.append(path)
        seen: set[str] = set()
        ordered: list[Path] = []
        for path in candidates:
            resolved = str(path.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            ordered.append(path)
        if "pdebench" in query_text:
            ordered.sort(key=lambda item: ("pdebench" not in item.name.lower(), len(str(item))))
        return ordered[:8]

    def _attach_repository_record(
        self,
        state: ResearchState,
        *,
        local_path: str,
        repo_url: str,
        repo_name: str,
        repo_id: str,
        resolution_source: str,
    ) -> RepositoryRecord | None:
        manifest_info = self.tools.detect_project_manifests(local_path)
        repository = RepositoryRecord(
            repo_id=repo_id,
            canonical_id=repo_id,
            raw_aliases=[repo_name, Path(local_path).name],
            name=repo_name,
            remote_url=repo_url,
            local_path=local_path,
            bootstrap_status="cloned",
            resolution_source=resolution_source,
            detected_manifests=list(manifest_info.get("manifests") or [])[:8],
            entrypoints=list(manifest_info.get("entrypoints") or [])[:20],
            notes=["manager_bootstrap_recovery", resolution_source],
        )
        state.repositories = upsert_by_attr(state.repositories, [repository], "repo_id")
        self.memory.record_repository(repository)
        try:
            env_payload = self.tools.ensure_python_environment(
                local_path,
                environment_name=f"{Path(local_path).name}-env",
                dependency_strategy="minimal",
                require_gpu_runtime=True,
            )
        except Exception as error:  # noqa: BLE001
            env_payload = {
                "status": "failed",
                "failure_reason": f"{type(error).__name__}: {error}",
            }
        if env_payload.get("status") == "ready":
            repository = repository.model_copy(
                update={
                    "bootstrap_status": "ready",
                    "environment_path": str(env_payload.get("environment_path") or ""),
                    "environment_id": str(env_payload.get("environment_name") or ""),
                    "notes": [*repository.notes, "environment_ready"],
                }
            )
        elif env_payload.get("failure_reason"):
            repository = repository.model_copy(
                update={
                    "bootstrap_status": "env_failed",
                    "notes": [*repository.notes, str(env_payload.get("failure_reason"))[:240]],
                }
            )
        state.repositories = upsert_by_attr(state.repositories, [repository], "repo_id")
        self.memory.record_repository(repository)
        self._log(
            f"Attached repository record {repository.name} at {repository.local_path}; "
            f"bootstrap_status={repository.bootstrap_status}."
        )
        return repository

    def _bootstrap_local_repository(self, state: ResearchState, query_text: str) -> RepositoryRecord | None:
        for path in self._discover_local_repository_paths(query_text):
            repository = self._attach_repository_record(
                state,
                local_path=str(path),
                repo_url="local://pdebench",
                repo_name=path.name,
                repo_id=f"local-{path.name}",
                resolution_source="manager_local_repository_discovery",
            )
            if repository is not None:
                return repository
        return None

    def _pdebench_registry_rows(self, repository: RepositoryRecord) -> list[dict[str, str]]:
        repo_root = Path(repository.local_path)
        candidate_paths = [
            repo_root / "pdebench" / "data_download" / "pdebench_data_urls.csv",
            repo_root / "data_download" / "pdebench_data_urls.csv",
        ]
        for path in candidate_paths:
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                return [dict(row) for row in reader]
        return []

    def _select_pdebench_bootstrap_rows(self, state: ResearchState, rows: list[dict[str, str]]) -> list[dict[str, str]]:
        if not rows:
            return []

        def _pick(predicate, preferred_markers: tuple[str, ...]) -> dict[str, str] | None:
            filtered = [row for row in rows if predicate(row)]
            if not filtered:
                return None
            filtered.sort(
                key=lambda row: (
                    tuple(marker.lower() in f"{row.get('Filename', '')} {row.get('Path', '')}".lower() for marker in preferred_markers),
                    row.get("Filename", ""),
                ),
                reverse=True,
            )
            return filtered[0]

        selected: list[dict[str, str]] = []
        brief_text = " ".join(
            [
                state.research_brief.title,
                state.research_brief.question,
                state.research_brief.background,
                *state.research_brief.objectives,
                *state.research_brief.constraints,
            ]
        ).lower()
        if "burgers" in brief_text:
            row = _pick(
                lambda item: "burgers" in item.get("PDE", "").lower() and "train" in item.get("Path", "").lower(),
                ("Nu0.01",),
            )
            if row:
                selected.append(row)
        if "reaction" in brief_text or "diffusion" in brief_text:
            train_row = _pick(
                lambda item: "reacdiff" in item.get("PDE", "").lower() and "train" in item.get("Path", "").lower(),
                ("Nu1.0_Rho1.0",),
            )
            test_row = _pick(
                lambda item: "reacdiff" in item.get("PDE", "").lower() and "test" in item.get("Path", "").lower(),
                ("ReacDiff_react_Nu1.0_Rho1.0",),
            )
            if train_row:
                selected.append(train_row)
            if test_row:
                selected.append(test_row)
        deduped: list[dict[str, str]] = []
        seen: set[str] = set()
        for row in selected:
            filename = row.get("Filename", "")
            if filename and filename not in seen:
                deduped.append(row)
                seen.add(filename)
        return deduped[:4]

    def _build_registry_artifact(
        self,
        row: dict[str, str],
        *,
        local_path: str | None = None,
    ) -> ArtifactRecord:
        filename = row.get("Filename", "").strip()
        official_path = row.get("Path", "").strip()
        equation = "ReactionDiffusion" if "reacdiff" in row.get("PDE", "").lower() else ("Burgers" if "burgers" in row.get("PDE", "").lower() else row.get("PDE", ""))
        split = "test" if "test" in official_path.lower() else "train"
        semantic = AssetSemanticSpec(
            benchmark="PDEBench",
            asset_family="dataset",
            equation=equation or None,
            split=split,
            filename=filename or None,
        )
        metadata = {
            "official_md5": row.get("MD5", "").strip(),
            "official_path": official_path,
            "expected_filename": filename,
            "exact_target": True,
        }
        artifact = ArtifactRecord(
            artifact_id=filename or f"dataset-{short_hash(row.get('URL', ''), row.get('Path', ''))}",
            artifact_type="dataset",
            title=filename or "dataset",
            rationale="Manager bootstrap inferred an executable benchmark shard from the PDEBench registry.",
            source_url=row.get("URL", "").strip() or None,
            local_path=local_path,
            status=ArtifactStatus.DOWNLOADED.value if local_path else ArtifactStatus.VERIFIED_REMOTE.value,
            semantic_spec=semantic,
            metadata=metadata,
            download_metadata=ArtifactDownloadMetadata(
                source_url=row.get("URL", "").strip() or local_path,
                source_type="pdebench_registry",
                local_path=local_path,
                canonical_target_id=None,
                strategy_id="manager_registry_bootstrap",
                attempt_signature=short_hash("manager_registry_bootstrap", filename, official_path),
                validation_status="downloaded" if local_path else "verified_remote",
            ),
            notes=["manager_registry_bootstrap"],
        )
        if local_path:
            artifact = self.tools.validate_artifact_record(artifact, quarantine_on_failure=False)
        return artifact

    def _seed_bootstrap_artifacts(self, state: ResearchState, repository: RepositoryRecord) -> None:
        rows = self._select_pdebench_bootstrap_rows(state, self._pdebench_registry_rows(repository))
        if not rows:
            return
        search_roots = [
            str(self.tools.shared_workspace_root),
            str(self.tools.workspace_family_root),
            *state.manual_asset_roots,
        ]
        recovered: list[ArtifactRecord] = []
        for row in rows:
            filename = row.get("Filename", "").strip()
            expected_checksum = row.get("MD5", "").strip() or None
            discovered = self.tools.discover_local_artifacts(
                query=filename,
                search_roots=search_roots,
                artifact_type="dataset",
                expected_checksum=expected_checksum,
                checksum_algorithm="md5",
                limit=4,
            )
            local_path = None
            ready_candidate = next((item for item in discovered if item.get("ready_for_training")), None)
            if ready_candidate is not None:
                local_path = str(ready_candidate.get("path") or "")
            artifact = self._build_registry_artifact(row, local_path=local_path or None)
            recovered.append(artifact)
            self.memory.record_artifact(artifact)
        state.external_artifacts = upsert_by_attr(state.external_artifacts, recovered, "artifact_id")
        state.acquisition_notes = dedupe_strings(
            [
                *state.acquisition_notes,
                f"Manager bootstrap seeded {len(recovered)} PDEBench dataset targets from the official registry.",
            ]
        )

    def _complete_recovered_phase(
        self,
        state: ResearchState,
        spec: PhaseSpec,
        agent,
        summary: str,
    ) -> str:
        self.memory.record_phase(spec.phase, summary, list(spec.outputs))
        self.memory.record_agent_event(
            agent_name=agent.name,
            phase=spec.phase,
            status="recovered",
            cycle_index=state.cycle_index,
            content=summary,
            payload={"outputs": list(spec.outputs), "phase_snapshot": self._phase_snapshot(state, spec.phase)},
        )
        for kind, message in self._core_progress_messages(state, spec.phase, summary):
            self.memory.record_core_progress(
                message,
                kind=kind,
                phase=spec.phase,
                agent_name=agent.name,
                cycle_index=state.cycle_index,
                payload={"summary": summary, "recovered": True},
            )
        self.memory.save_state(state, label=spec.phase.value)
        self._log(
            f"Completed phase {spec.phase.value} with {agent.name} via recovery. Summary: {summary}"
        )
        for line in self._phase_snapshot(state, spec.phase):
            self._log(line)
        return summary

    def _recover_phase_with_heuristics(self, state: ResearchState, spec: PhaseSpec, agent) -> str | None:
        output = None
        if spec.phase == ResearchPhase.LITERATURE_REVIEW:
            output = self._heuristic_literature_output_from_state(state)
        elif spec.phase == ResearchPhase.PROBLEM_FRAMING:
            output = self._heuristic_problem_framing_output(state)
        elif spec.phase == ResearchPhase.DIAGNOSIS:
            output = self._heuristic_diagnosis_output(state)
        elif spec.phase == ResearchPhase.HYPOTHESIS:
            output = self._heuristic_hypothesis_output(state)
        elif spec.phase == ResearchPhase.METHOD_DESIGN:
            output = self._heuristic_method_design_output(state)
        elif spec.phase == ResearchPhase.CODING:
            output = self._heuristic_coding_output(state)
        elif spec.phase == ResearchPhase.EXPERIMENT_PLANNING:
            output = self._heuristic_experiment_planning_output(state)
        elif spec.phase == ResearchPhase.REFLECTION:
            output = self._heuristic_reflection_output(state)
        elif spec.phase == ResearchPhase.REPORTING:
            output = self._heuristic_reporting_output(state)
        if output is None:
            return None
        summary = agent.apply_output(state, self.tools, output)
        if spec.phase == ResearchPhase.EXPERIMENT_PLANNING:
            self._ground_experiment_plans(state)
        self._post_phase_sync(state, spec)
        return self._complete_recovered_phase(state, spec, agent, summary)

    def _heuristic_literature_output_from_state(self, state: ResearchState) -> LiteraturePhaseOutput:
        existing_notes = state.literature_notes[-6:]
        if existing_notes:
            summary = (
                "Recovered literature review from previously accumulated notes because the live model backend was unavailable."
            )
            return LiteraturePhaseOutput(
                summary=summary,
                literature_notes=existing_notes,
                method_taxonomy=state.method_taxonomy or [
                    TaxonomyEntry(
                        category="Neural operators",
                        methods=["FNO", "DeepONet"],
                        shared_strengths=["Strong operator-learning baselines for PDEBench-style tasks."],
                        shared_limitations=["Need task-specific diagnostics and physics participation to satisfy the challenge brief."],
                        research_opportunities=["Use executable baseline reproduction before proposing heavier method changes."],
                    )
                ],
                open_questions=state.open_questions or [
                    "How should physics participation be added without destabilizing short-window accuracy?",
                    "Which executable baseline is the fastest route to empirical evidence on PDEBench 1D?",
                ],
                semantic_notes=["literature_recovered_from_memory"],
                next_actions=[
                    "Continue with acquisition and executable baseline planning using the recovered literature scaffold.",
                ],
            )

        notes = [
            PaperNote(
                paper_id="heuristic-pdebench",
                title="PDEBench as the benchmark anchor",
                authors=[],
                year=2022,
                abstract="Use PDEBench as the benchmark substrate and baseline code source for executable 1D Burgers and Reaction-Diffusion experiments.",
                method_family="benchmark",
                physics_level="mixed",
                key_claims=[
                    "PDEBench provides standardized PDE datasets and baselines.",
                    "Executable benchmark reproduction is a prerequisite for meaningful method comparison.",
                ],
                strengths=["Directly aligned with the challenge task and baseline reproduction needs."],
                limitations=["Does not itself solve physics participation or diagnosis; those must be layered on top."],
                open_questions=["Which PDEBench 1D baseline can be reproduced most reliably first?"],
                research_opportunities=["Use PDEBench 1D baselines to establish an empirical anchor before method innovation."],
                source_url="local-heuristic://pdebench",
            ),
            PaperNote(
                paper_id="heuristic-fno",
                title="FNO as the first executable neural-operator baseline",
                authors=[],
                year=2021,
                abstract="Treat Fourier Neural Operator as the default baseline because it is already present in the acquired PDEBench repository and is compatible with short-window surrogate modeling.",
                method_family="neural operator",
                physics_level="data-driven baseline",
                key_claims=[
                    "FNO is the most practical first baseline path for PDEBench 1D execution.",
                    "Physics-aware extensions should be layered on a working FNO baseline rather than replacing it prematurely.",
                ],
                strengths=["Executable from the recovered PDEBench repository.", "Strong baseline for operator learning."],
                limitations=["Needs explicit physics participation to satisfy the task constraints."],
                open_questions=["What is the lightest physics-aware modification that preserves executability?"],
                research_opportunities=["Add residual regularization or inference-time correction after the baseline is reproduced."],
                source_url="local-heuristic://fno",
            ),
            PaperNote(
                paper_id="heuristic-physics-aware",
                title="Physics-aware operator learning as the first innovation family",
                authors=[],
                year=2021,
                abstract="Physics-informed operator learning suggests lightweight residual penalties or correction steps as the most plausible first innovations once a stable baseline exists.",
                method_family="physics-aware operator learning",
                physics_level="explicit physics participation",
                key_claims=[
                    "Physical information can enter through residual losses or correction operators.",
                    "Executable baseline reproduction should precede heavier physics-aware modifications.",
                ],
                strengths=["Matches the challenge constraint that physics must participate."],
                limitations=["Can add optimization burden if introduced before the baseline path is stable."],
                open_questions=["Should physics enter the loss, inference correction, or both?"],
                research_opportunities=["Start with a lightweight residual-weighted FNO variant after baseline reproduction."],
                source_url="local-heuristic://physics-aware-operator-learning",
            ),
        ]
        taxonomy = [
            TaxonomyEntry(
                category="Benchmark and baselines",
                methods=["PDEBench", "FNO", "DeepONet"],
                shared_strengths=["Provide an executable substrate for PDEBench 1D research."],
                shared_limitations=["Need physics participation and better diagnostics to meet the competition goal."],
                research_opportunities=["Establish one executable baseline and then add physics-aware modifications."],
            ),
            TaxonomyEntry(
                category="Physics-aware extensions",
                methods=["Residual regularization", "Physics correction", "PI-DeepONet-style constraints"],
                shared_strengths=["Inject physical information into training or inference."],
                shared_limitations=["Risk slowing or destabilizing training if introduced too early."],
                research_opportunities=["Use lightweight physics-aware mechanisms after the first baseline is reproduced."],
            ),
        ]
        open_questions = [
            "What is the fastest executable baseline path for PDEBench 1D Burgers and Reaction-Diffusion?",
            "How can physics participation be introduced without blocking executable training?",
            "Which diagnostics best expose short-window failure modes once the baseline is running?",
        ]
        return LiteraturePhaseOutput(
            summary=(
                "Recovered literature review heuristically because the live model backend was unavailable. "
                "The system will proceed with PDEBench as the benchmark anchor, FNO as the first executable baseline, "
                "and lightweight physics-aware operator learning as the first innovation family."
            ),
            literature_notes=notes,
            method_taxonomy=taxonomy,
            open_questions=open_questions,
            semantic_notes=["literature_recovered_heuristically", "provider_unavailable_recovery"],
            next_actions=[
                "Use the recovered literature scaffold to continue acquisition and executable baseline planning.",
            ],
        )

    def _heuristic_problem_framing_output(self, state: ResearchState) -> ProblemFramingPhaseOutput:
        criteria = dedupe_strings(
            [
                "Improve short-window predictive accuracy on PDEBench 1D tasks.",
                "Ensure physical information participates in training or inference.",
                "Maintain an executable baseline or fallback experiment path at every cycle.",
                "Favor method-level changes over hyperparameter-only tuning.",
            ]
        )
        directions = [
            {
                "title": "Physics-aware FNO baseline refinement",
                "innovation_kind": "method_innovation",
                "rationale": "Recovered literature and the challenge brief consistently point to FNO-style baselines plus physics-aware regularization as the fastest executable route.",
                "why_not_just_tuning": "It changes the loss structure and physical constraints rather than only scalar training hyperparameters.",
                "expected_signal": "Better physical consistency and lower short-window error once a baseline route is executable.",
            },
            {
                "title": "Evidence-first fallback execution",
                "innovation_kind": "workflow_strategy",
                "rationale": "When exact benchmark assets or CUDA runtime are unstable, the system should first produce executable empirical evidence instead of waiting indefinitely for the perfect route.",
                "why_not_just_tuning": "It changes the orchestration strategy and empirical evidence policy, not a training knob.",
                "expected_signal": "A completed smoke or fallback run that validates repo, environment, data, and reporting end-to-end.",
            },
        ]
        candidate_directions: list[CandidateDirection] = []
        for item in directions:
            candidate_directions.append(
                CandidateDirection(
                    direction_id=f"dir-{short_hash(state.run_name, item['title'])}",
                    title=item["title"],
                    innovation_kind=item["innovation_kind"],
                    rationale=item["rationale"],
                    why_not_just_tuning=item["why_not_just_tuning"],
                    expected_signal=item["expected_signal"],
                )
            )
        notes = [
            "Success requires an executable PDEBench-oriented baseline or fallback route, not only literature synthesis.",
            "Method innovation should attach to a runnable operator-learning baseline, preferably FNO-like, with explicit physical guidance.",
        ]
        if state.capability_matrix and not state.capability_matrix.baseline_ready_to_launch:
            notes.append("Infrastructure readiness is still part of the scientific problem framing because blocked data or runtime prevents hypothesis testing.")
        return ProblemFramingPhaseOutput(
            summary="Recovered problem framing from the brief, literature, and current acquisition state.",
            problem_framing_notes=notes,
            evaluation_criteria=criteria,
            candidate_directions=candidate_directions,
            next_actions=[
                "Secure one executable baseline or fallback evidence route.",
                "Prioritize physics-aware operator-learning changes once empirical execution is available.",
            ],
        )

    def _heuristic_diagnosis_output(self, state: ResearchState) -> DiagnosisPhaseOutput:
        capability = state.capability_matrix
        bottlenecks: list[str] = []
        semantic_notes: list[str] = []
        if capability is None or not capability.repo_ready:
            bottlenecks.append("A benchmark repository is not yet attached to the run in a launch-ready state.")
        if capability is None or not capability.env_ready:
            bottlenecks.append("The managed Python environment is not yet stable enough for repeatable baseline execution.")
        if capability and capability.target_dataset_blocked:
            bottlenecks.append("Exact target datasets remain blocked, so the preferred baseline route cannot launch.")
        if capability and not capability.gpu_runtime_ready and capability.gpu_runtime_required:
            bottlenecks.append("CUDA runtime is unavailable even though GPUs are visible; environment repair or CPU fallback is required.")
        if not state.experiment_records:
            bottlenecks.append("No empirical execution evidence exists yet, so hypothesis quality cannot be judged scientifically.")
        if not bottlenecks:
            bottlenecks.append("The main bottleneck is converting the recovered method idea into measurable empirical gains.")
        semantic_notes.extend(bottlenecks[:3])
        next_actions = [
            "Do not repeat unchanged blocked acquisition routes.",
            "Prefer an executable fallback experiment once repo/env/codepath are ready.",
        ]
        return DiagnosisPhaseOutput(
            summary="Recovered diagnosis from capability state, blockers, and available execution evidence.",
            bottleneck_analysis=bottlenecks[:6],
            next_actions=next_actions,
            semantic_notes=semantic_notes[:4],
        )

    def _heuristic_hypothesis_output(self, state: ResearchState) -> HypothesisPhaseOutput:
        parent_program_id = state.selected_baseline_program_id
        hypothesis = state.hypotheses[-1] if state.hypotheses else None
        if hypothesis is None:
            hypothesis = HypothesisRecord(
                hypothesis_id=f"hyp-{short_hash(state.run_name, state.cycle_index, 'physics-aware-fno')}",
                title="Physics-aware FNO baseline with lightweight residual regularization",
                statement="Adding a lightweight PDE residual or conservation-style auxiliary loss to an executable FNO baseline should improve short-window accuracy and physical consistency without sacrificing launchability.",
                rationale="Recovered literature repeatedly points to FNO-style baselines as the most practical operator-learning backbone and to physics-aware objectives as the clearest source of scientifically meaningful gains.",
                expected_gains=[
                    "Lower short-window error on executable PDEBench-style tasks.",
                    "Better physical consistency than a plain baseline.",
                ],
                risks=[
                    "Physics terms may destabilize early training if the baseline path is still fragile.",
                    "Infrastructure blockers may dominate until at least one empirical route is executable.",
                ],
                required_code_changes=[
                    "Add a lightweight residual-based auxiliary loss or residual-aware weighting to the baseline training path.",
                    "Preserve a runnable baseline path so the effect can be measured incrementally.",
                ],
                evaluation_plan=[
                    "Run a baseline or fallback executable experiment first.",
                    "Compare physical consistency and short-window metrics against the parent baseline.",
                ],
                innovation_kind="method_innovation",
                status="proposed",
                parent_program_id=parent_program_id,
            )
        return HypothesisPhaseOutput(
            summary="Recovered a literature-grounded hypothesis for the next executable iteration.",
            hypotheses=[hypothesis],
            next_actions=[
                "Convert the hypothesis into a concrete loss/training design.",
                "Keep an executable baseline or fallback route alive while testing the new idea.",
            ],
        )

    def _heuristic_method_design_output(self, state: ResearchState) -> MethodDesignPhaseOutput:
        hypothesis = state.hypotheses[-1] if state.hypotheses else None
        if hypothesis is None:
            return MethodDesignPhaseOutput(
                summary="No hypothesis was available, so method design recovery could only preserve the baseline route.",
                method_designs=[],
                next_actions=["Recover hypothesis generation before attempting code changes."],
            )
        design = state.method_designs[-1] if state.method_designs else None
        if design is None:
            design = MethodDesign(
                design_id=f"design-{short_hash(hypothesis.hypothesis_id, state.run_name)}",
                hypothesis_id=hypothesis.hypothesis_id,
                title="Residual-weighted physics-aware neural-operator baseline",
                parent_program_id=hypothesis.parent_program_id or state.selected_baseline_program_id,
                architecture_changes=[
                    "Keep the baseline neural-operator backbone intact to preserve executability.",
                ],
                loss_changes=[
                    "Add a lightweight PDE residual or conservation-style auxiliary loss term.",
                    "Allow adaptive weighting so the physics term can be ramped in after baseline convergence starts.",
                ],
                data_changes=[
                    "Use the currently validated PDEBench-compatible shard first; expand scope only after the pipeline is stable.",
                ],
                training_strategy=[
                    "Start from the executable baseline path.",
                    "Introduce the physics-aware term incrementally so preflight and smoke execution remain stable.",
                ],
                inference_strategy=[
                    "Retain the baseline rollout path for direct comparison.",
                ],
                physics_integration=[
                    "Compute a lightweight residual or consistency penalty from available PDE state tensors.",
                ],
                implementation_steps=[
                    "Preserve a runnable baseline program candidate.",
                    "Add the smallest possible physics-aware loss hook.",
                    "Validate imports and a smoke experiment before a heavier launch.",
                ],
                evaluation_plan=hypothesis.evaluation_plan,
            )
        return MethodDesignPhaseOutput(
            summary="Recovered a concrete method design from the current hypothesis and executable constraints.",
            method_designs=[design],
            next_actions=[
                "Prepare a child program candidate that preserves baseline executability.",
                "Prefer a smoke or fallback evidence run before heavier training.",
            ],
        )

    def _heuristic_coding_output(self, state: ResearchState) -> CodingPhaseOutput | None:
        baseline = next(
            (item for item in state.program_candidates if item.program_id == state.selected_baseline_program_id),
            None,
        ) or next((item for item in state.program_candidates if item.status.startswith("baseline")), None)
        repository = next((item for item in state.repositories if item.local_path), None)
        if baseline is None and repository is None:
            return None
        workspace = (baseline.workspace_path if baseline and baseline.workspace_path else None) or (repository.local_path if repository else None)
        repo_id = (baseline.repo_id if baseline else None) or (repository.canonical_id if repository else None)
        latest_design = state.method_designs[-1] if state.method_designs else None
        latest_hypothesis = state.hypotheses[-1] if state.hypotheses else None
        candidate = ProgramCandidate(
            program_id=f"prog-{short_hash(state.run_name, workspace or 'baseline-reuse', state.cycle_index)}",
            title="Recovered executable baseline-reuse candidate",
            summary="Manager recovery preserved an executable program candidate by reusing the current baseline workspace while keeping the latest design intent attached in metadata.",
            repo_id=repo_id,
            workspace_path=workspace,
            parent_program_id=baseline.program_id if baseline else None,
            design_id=latest_design.design_id if latest_design else None,
            hypothesis_id=latest_hypothesis.hypothesis_id if latest_hypothesis else None,
            entry_command_hint=(baseline.entry_command_hint if baseline else None),
            status="recovery_candidate",
            changed_files=[],
            notes=["manager_recovery_candidate", "baseline_reuse"],
        )
        return CodingPhaseOutput(
            summary="Recovered coding phase by preserving an executable program candidate tied to the latest design intent.",
            program_candidates=[candidate],
            next_actions=[
                "Plan a baseline or fallback experiment from the preserved executable candidate.",
            ],
        )

    def _build_recovery_fallback_plan(self, state: ResearchState) -> ExperimentPlan | None:
        repository = next((item for item in state.repositories if item.local_path), None)
        capability = state.capability_matrix
        env_path = None
        if repository and repository.environment_path:
            env_path = repository.environment_path
        elif capability and capability.environment_path:
            env_path = capability.environment_path
        if repository is None or env_path is None:
            return None
        ready_artifacts = [
            item for item in state.external_artifacts
            if item.artifact_type == "dataset" and item.status == ArtifactStatus.READY_FOR_TRAINING.value and item.local_path
        ]
        artifact_paths = [item.local_path for item in ready_artifacts[:3] if item.local_path]
        report_path = self.tools.memory.experiments_dir / f"fallback-smoke-{state.cycle_index}.json"
        checks = "\n".join(
            [
                "datasets = {}",
                *[
                    (
                        f"with h5py.File({path!r}, 'r') as handle:\n"
                        f"    datasets[{Path(path).name!r}] = sorted(handle.keys())[:5]"
                    )
                    for path in artifact_paths
                ],
            ]
        )
        script = (
            "python - <<'PY'\n"
            "from pathlib import Path\n"
            "import json\n"
            "import h5py\n"
            "import pdebench\n"
            "try:\n"
            "    import torch\n"
            "    torch_version = getattr(torch, '__version__', None)\n"
            "    cuda_available = bool(torch.cuda.is_available())\n"
            "except Exception as exc:\n"
            "    torch_version = None\n"
            "    cuda_available = False\n"
            "    torch_error = f'{type(exc).__name__}: {exc}'\n"
            "else:\n"
            "    torch_error = None\n"
            f"{checks}\n"
            "payload = {\n"
            "    'mode': 'fallback_smoke',\n"
            "    'torch_version': torch_version,\n"
            "    'cuda_available': cuda_available,\n"
            "    'torch_error': torch_error,\n"
            "    'datasets': datasets,\n"
            "}\n"
            f"Path({str(report_path)!r}).write_text(json.dumps(payload), encoding='utf-8')\n"
            "print(json.dumps(payload))\n"
            "PY"
        )
        return ExperimentPlan(
            plan_id=f"fallback-smoke-{short_hash(state.run_name, state.cycle_index, 'manager')}",
            title="Recovered fallback smoke experiment",
            program_id=state.selected_baseline_program_id or "fallback-smoke",
            repo_id=repository.canonical_id or repository.repo_id,
            job_kind="experiment",
            working_directory=repository.local_path,
            setup_commands=[],
            launch_command=script,
            environment={"VIRTUAL_ENV": env_path},
            gpu_ids=[],
            required_artifact_ids=[item.canonical_id or item.artifact_id for item in ready_artifacts[:3]],
            preflight_required=True,
            preflight_status=None,
            expected_outputs=[str(report_path)],
            success_criteria=["Import pdebench, open validated local datasets, and emit a machine-readable JSON payload."],
            stopping_rules=["Stop immediately on import or file-read failure."],
            log_path=str(self.tools.memory.experiments_dir / "fallback_smoke_recovery.log"),
            status="planned",
            notes=["evidence_generating_fallback", "manager_recovery_plan"],
        )

    def _heuristic_experiment_planning_output(self, state: ResearchState) -> ExperimentPlanningPhaseOutput | None:
        fallback_plan = self._build_recovery_fallback_plan(state)
        if fallback_plan is None:
            return ExperimentPlanningPhaseOutput(
                summary="Recovered experiment planning could not synthesize a runnable plan because repo or environment readiness is still missing.",
                experiment_plans=[],
                next_actions=[
                    "Return to acquisition or environment repair until one executable route is available.",
                ],
            )
        return ExperimentPlanningPhaseOutput(
            summary="Recovered experiment planning by synthesizing a deterministic evidence-generating fallback plan.",
            experiment_plans=[fallback_plan],
            next_actions=[
                "Run preflight on the fallback smoke plan.",
                "Use the resulting evidence to validate repo, environment, and dataset readiness.",
            ],
        )

    def _heuristic_reflection_output(self, state: ResearchState) -> ReflectionPhaseOutput:
        completed = [item for item in state.experiment_records if item.status == "completed"]
        failed = [item for item in state.experiment_records if item.status != "completed"]
        failure_ids = [item.failure_id for item in state.classified_failures[-6:]]
        if completed:
            latest = completed[-1]
            evidence = latest.observations[:3] or ["A deterministic fallback experiment completed and produced machine-readable evidence."]
            verdict = "supported"
            summary = "Recovered reflection concluded that the current cycle produced executable empirical evidence."
            continue_research = False
            stop_reason = "Recovered run produced executable evidence and durable reports; further scientific iteration can start from this new baseline state."
            recommended_route = None
            preferred: list[str] = []
            escalation_required = False
        else:
            blockers = [item.target_entity for item in state.blocker_registry if item.repeat_count >= 1][:6]
            evidence = state.failure_summaries[-4:] or ["No experiment produced completed metrics in the latest cycle."]
            verdict = "blocked"
            summary = "Recovered reflection concluded that infrastructure blockers still prevent meaningful empirical progress."
            continue_research = False if blockers else True
            stop_reason = (
                "No executable evidence was produced after deterministic recovery."
                if blockers
                else None
            )
            recommended_route = "fallback-execution" if state.capability_matrix and state.capability_matrix.scientific_iteration_ready else "recover-baseline-prerequisites"
            preferred = ["fallback_execution"] if recommended_route == "fallback-execution" else ["environment_repair", "local_discovery"]
            escalation_required = bool(blockers and not (state.capability_matrix and state.capability_matrix.scientific_iteration_ready))
        base_reflection = state.reflections[-1] if state.reflections else None
        reflection_payload = {
            "reflection_id": (
                base_reflection.reflection_id
                if base_reflection is not None
                else f"refl-{short_hash(state.run_name, state.cycle_index, now_utc())}"
            ),
            "cycle_index": state.cycle_index,
            "verdict": verdict,
            "evidence": evidence,
            "linked_failure_ids": failure_ids,
            "accepted_lessons": [
                "Prefer deterministic fallback execution when model-side structured output fails repeatedly.",
                "Do not wait for perfect assets before generating the first executable evidence.",
            ],
            "next_actions": state.next_actions[-4:] or ["Review generated reports and resume from the latest executable state."],
            "recommended_route_id": recommended_route,
            "preferred_recovery_strategies": preferred,
            "blocked_entities": [item.target_entity for item in state.blocker_registry if item.repeat_count >= 1][:6],
            "material_change_required": not bool(completed),
            "escalation_required": escalation_required,
            "failure_category": "infrastructure" if not completed else None,
            "continue_research": continue_research,
            "stop_reason": stop_reason,
        }
        reflection = (
            base_reflection.model_copy(update=reflection_payload)
            if base_reflection is not None
            else ReflectionRecord(**reflection_payload)
        )
        return ReflectionPhaseOutput(
            summary=summary,
            reflections=[reflection],
            next_actions=reflection.next_actions,
            terminate_research=not reflection.continue_research,
        )

    def _heuristic_reporting_output(self, state: ResearchState) -> ReportingPhaseOutput:
        report_payloads = {
            "literature_review.md": (
                "# Literature Review\n\n"
                + "\n".join(f"- {note.title}: {note.abstract}" for note in state.literature_notes[:8])
                + "\n"
            ),
            "acquisition_report.md": (
                "# Acquisition Report\n\n"
                f"- Repositories: {len(state.repositories)}\n"
                f"- Artifacts: {len(state.external_artifacts)}\n"
                f"- Selected baseline: {state.selected_baseline_program_id or 'unset'}\n"
                + "\n".join(f"- {note}" for note in state.acquisition_notes[-12:])
                + "\n"
            ),
            "idea_diary.md": (
                "# Idea Diary\n\n"
                + "\n".join(f"- {item.title}: {item.statement}" for item in state.hypotheses[-6:])
                + "\n"
            ),
            "experiment_summary.md": (
                "# Experiment Summary\n\n"
                + "\n".join(
                    f"- {record.experiment_id}: status={record.status}, metrics={record.metrics or {}}, failures={record.failure_modes}"
                    for record in state.experiment_records[-12:]
                )
                + "\n"
            ),
            "final_report.md": (
                "# Final Report\n\n"
                f"Run: {state.run_name}\n\n"
                f"- Literature notes: {len(state.literature_notes)}\n"
                f"- Repositories: {len(state.repositories)}\n"
                f"- Program candidates: {len(state.program_candidates)}\n"
                f"- Experiments: {len(state.experiment_records)}\n"
                f"- Best known results: {state.best_known_results}\n\n"
                "## Reflections\n\n"
                + "\n".join(f"- {item.verdict}: {item.stop_reason or '; '.join(item.evidence[:2])}" for item in state.reflections[-6:])
                + "\n"
            ),
        }
        generated: list[GeneratedReport] = []
        for filename, content in report_payloads.items():
            path = self.tools.write_report(filename, content)
            generated.append(
                GeneratedReport(
                    report_id=f"report-{short_hash(filename, state.run_name)}",
                    title=filename,
                    kind=filename.replace(".md", ""),
                    path=str(path),
                )
            )
        return ReportingPhaseOutput(
            summary=f"Recovered reporting by writing {len(generated)} durable markdown reports.",
            generated_reports=generated,
        )

    def _heuristic_literature_output_from_sources(
        self,
        compact_brief: dict[str, object],
        paper_payloads: list[dict[str, str]],
    ) -> LiteraturePhaseOutput:
        def _sentences(text: str, limit: int = 2) -> list[str]:
            chunks = [item.strip() for item in re.split(r"(?<=[.!?])\s+", text) if item.strip()]
            if chunks:
                return chunks[:limit]
            return [text[:180].strip()] if text.strip() else []

        def _family(text: str) -> tuple[str, str]:
            lowered = text.lower()
            if "fourier neural operator" in lowered or " fno" in lowered:
                return "FNO", "data-driven"
            if "deeponet" in lowered:
                return "DeepONet", "data-driven"
            if "physics-informed" in lowered or "pinn" in lowered:
                return "Physics-Informed", "physics-informed"
            if "benchmark" in lowered or "pdebench" in lowered:
                return "Benchmark", "evaluation"
            return "Operator Learning", "mixed"

        literature_notes: list[PaperNote] = []
        taxonomy: dict[str, TaxonomyEntry] = {}
        open_questions: list[str] = []
        semantic_notes: list[str] = []
        for item in paper_payloads[:4]:
            excerpt = item.get("excerpt", "")
            filename = item.get("filename", "paper.pdf")
            title = Path(filename).stem.replace("_", " ").replace("-", " ")
            method_family, physics_level = _family(f"{title}\n{excerpt}")
            lowered = excerpt.lower()
            strengths: list[str] = []
            limitations: list[str] = []
            opportunities: list[str] = []
            if "resolution" in lowered or "generalization" in lowered:
                strengths.append("Operator-level generalization or discretization transfer is highlighted.")
            if "physics-informed" in lowered or "residual" in lowered:
                strengths.append("Physical constraints or residual guidance are emphasized.")
            if "benchmark" in lowered or "dataset" in lowered:
                strengths.append("Provides benchmark or evaluation guidance relevant to PDEBench.")
            if "shock" in lowered or "discontinu" in lowered or "high-frequency" in lowered:
                limitations.append("Discontinuities or high-frequency dynamics remain difficult.")
            if "cost" in lowered or "expensive" in lowered or "slow" in lowered:
                limitations.append("Training or simulation cost remains a bottleneck.")
            if physics_level != "physics-informed":
                opportunities.append("Inject physics guidance without losing short-window predictive accuracy.")
            if method_family != "Benchmark":
                opportunities.append("Validate on PDEBench 1D Burgers and diffusion-reaction with physically informed objectives.")
            note = PaperNote(
                paper_id=Path(filename).stem,
                title=title[:180],
                abstract=" ".join(_sentences(excerpt, limit=2))[:320],
                method_family=method_family,
                physics_level=physics_level,
                key_claims=_sentences(excerpt, limit=2),
                strengths=strengths[:2] or ["Relevant evidence was recovered from a local source document."],
                limitations=limitations[:2] or ["Recovered deterministically after structured generation failed."],
                open_questions=opportunities[:2],
                research_opportunities=opportunities[:2] or ["Reconnect this evidence to an executable PDEBench baseline."],
                source_url=item.get("path", ""),
                pdf_path=item.get("path"),
            )
            literature_notes.append(note)
            entry = taxonomy.setdefault(
                method_family,
                TaxonomyEntry(
                    category=method_family,
                    methods=[],
                    shared_strengths=[],
                    shared_limitations=[],
                    research_opportunities=[],
                ),
            )
            entry.methods = list(dict.fromkeys([*entry.methods, title[:80]]))
            entry.shared_strengths = list(dict.fromkeys([*entry.shared_strengths, *note.strengths]))[:3]
            entry.shared_limitations = list(dict.fromkeys([*entry.shared_limitations, *note.limitations]))[:3]
            entry.research_opportunities = list(dict.fromkeys([*entry.research_opportunities, *note.research_opportunities]))[:3]
            open_questions.extend(note.open_questions[:1])
            semantic_notes.append(f"{method_family}: {'; '.join((note.strengths + note.limitations)[:2])}")

        question = str(compact_brief.get("question") or "").strip()
        if question:
            open_questions.append(f"Which executable baseline best addresses: {question[:140]}?")
        if not literature_notes:
            literature_notes.append(
                PaperNote(
                    paper_id="literature-recovery-empty",
                    title="Recovered literature summary",
                    abstract="Recovered from local sources after model output failure.",
                    method_family="Operator Learning",
                    physics_level="mixed",
                    key_claims=["Local literature artifacts were preserved even though the structured generation step failed."],
                    strengths=["The workflow can continue from local evidence."],
                    limitations=["This summary used deterministic recovery instead of model synthesis."],
                    research_opportunities=["Retry targeted synthesis later if a higher-fidelity narrative is needed."],
                )
            )
            taxonomy["Operator Learning"] = TaxonomyEntry(
                category="Operator Learning",
                methods=["Recovered literature summary"],
                shared_strengths=["Local evidence preserved."],
                shared_limitations=["Used deterministic fallback."],
                research_opportunities=["Retry focused synthesis after the core execution path is stable."],
            )
            open_questions.append("Which literature claims still need higher-fidelity synthesis?")
            semantic_notes.append("Literature recovery used deterministic fallback.")

        return LiteraturePhaseOutput(
            summary=(
                f"Recovered literature state from {len(literature_notes)} local sources. "
                f"Primary families: {', '.join(entry.category for entry in list(taxonomy.values())[:3])}."
            ),
            literature_notes=literature_notes[:4],
            method_taxonomy=list(taxonomy.values())[:4],
            open_questions=list(dict.fromkeys(open_questions))[:4],
            semantic_notes=list(dict.fromkeys(semantic_notes))[:4],
            next_actions=[
                "Use the recovered literature summary to ground baseline selection and problem framing.",
                "Prioritize executable baselines with physics-informed components and validated datasets.",
            ],
        )

    def _should_continue(self, state: ResearchState) -> bool:
        if state.blocked_reason and state.termination_decision:
            return False
        if not state.reflections:
            return True
        latest = state.reflections[-1]
        if not latest.continue_research:
            return False
        if self.config.manager_safety_max_cycles is not None and state.cycle_index >= self.config.manager_safety_max_cycles:
            state.termination_decision = (
                f"Stopped after reaching manager_safety_max_cycles={self.config.manager_safety_max_cycles}."
            )
            return False
        return True

    def run(self) -> ResearchState:
        self.runtime.ensure_ready()
        self._log(
            f"Research run started. project={self.config.project_name}, run={self.config.run_name}, work_directory={self.work_directory}."
        )
        state = self._initial_state()
        for spec in self.front_phases:
            self._run_phase(state, spec)
        while True:
            state.cycle_index += 1
            self._log(f"Entering research cycle {state.cycle_index}.")
            self._refresh_blockers(state)
            route = self._select_cycle_route(state)
            if self._handle_hitl(state, route=route):
                if state.termination_decision:
                    break
                self._refresh_blockers(state)
                route = self._select_cycle_route(state)
            self._record_route_decision(state, route)
            self._log(
                "Manager selected cycle route: "
                f"{' -> '.join(spec.phase.value for spec in route.phases)}. "
                f"route_id={route.route_id}. Reason: {route.reason}"
            )
            self._log_cycle_context(state, route)
            for spec in route.phases:
                if spec.phase == ResearchPhase.EXPERIMENT:
                    ready_plans = [
                        plan for plan in state.experiment_plans
                        if plan.status == "planned"
                        and plan.job_kind == "experiment"
                        and plan.preflight_status == "passed"
                    ]
                    if not ready_plans:
                        self._log(
                            "Skipping experiment phase because no preflight-passed experiment plans are available. "
                            f"active_route={state.active_route_id} blocked_reason={state.blocked_reason or 'none'}"
                        )
                        continue
                self._run_phase(state, spec)
            if not self._should_continue(state):
                break
        self._run_phase(state, self.reporting_phase)
        self.memory.save_state(state, label="final_state")
        self._log(
            f"Research run finished. cycles={state.cycle_index}, repositories={len(state.repositories)}, programs={len(state.program_candidates)}, experiments={len(state.experiment_records)}, reports={len(state.generated_reports)}."
        )
        return state
