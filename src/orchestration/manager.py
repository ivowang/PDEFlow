from __future__ import annotations

from pathlib import Path
import shutil
import traceback

from research_agents import (
    AcquisitionAgent,
    CoderAgent,
    DiagnosisAgent,
    ExperimentAgent,
    ExperimentPlannerAgent,
    HypothesisAgent,
    LiteratureAgent,
    MethodDesignAgent,
    PreflightValidationAgent,
    ProblemFramingAgent,
    ReflectionAgent,
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
    should_trigger_hitl,
)
from runtime import RuntimeAdapter
from state import (
    EnvironmentRecord,
    EnvironmentResolutionState,
    HITLEvent,
    HITLStatus,
    HumanResponseType,
    ResearchPhase,
    ResearchState,
    RouteDecisionRecord,
)
from tools import ResearchTools
from common import canonicalize_env_id, now_utc, short_hash, upsert_by_attr
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
            "problem": ProblemFramingAgent(),
            "diagnosis": DiagnosisAgent(),
            "hypothesis": HypothesisAgent(),
            "design": MethodDesignAgent(),
            "coder": CoderAgent(),
            "planner": ExperimentPlannerAgent(),
            "preflight": PreflightValidationAgent(),
            "experiment": ExperimentAgent(),
            "reflection": ReflectionAgent(),
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
                    f"target_dataset_ready={state.capability_matrix.target_dataset_ready}."
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
            validated = self.tools.validate_artifact_record(artifact, quarantine_on_failure=True)
            updated.append(validated)
            self.memory.record_artifact(validated)
        state.external_artifacts = updated

    def _refresh_capability_matrix(self, state: ResearchState) -> None:
        self._sync_environment_records(state)
        capability_matrix = self.tools.probe_capability_matrix(
            artifacts=state.external_artifacts,
            repository_paths=[repo.local_path for repo in state.repositories],
            environment_path=state.capability_matrix.environment_path if state.capability_matrix else None,
        )
        state.capability_matrix = capability_matrix
        self.memory.record_capability_matrix(capability_matrix)

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
        if spec.phase in {ResearchPhase.ACQUISITION, ResearchPhase.PREFLIGHT_VALIDATION}:
            self._validate_artifacts(state)
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

    def _materialize_manual_files(self, state: ResearchState, search_roots: list[str], artifacts: list[object]) -> list[str]:
        validation_summary: list[str] = []
        for artifact in artifacts:
            if not artifact.local_path:
                continue
            expected_path = Path(artifact.local_path)
            if expected_path.exists():
                validated = self.tools.validate_artifact_record(artifact, quarantine_on_failure=True)
                if validated.status == "ready_for_training":
                    validation_summary.append(f"validated existing file {expected_path.name}")
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
            artifact.local_path = str(expected_path)
            artifact.notes.append(f"Imported from human-provided path {located}")
            validated = self.tools.validate_artifact_record(artifact, quarantine_on_failure=True)
            if validated.status == "ready_for_training":
                validation_summary.append(f"validated imported file {filename}")
            else:
                validation_summary.append(f"imported but still invalid {filename}: status={validated.status}")
        return validation_summary

    def _revalidate_after_human_confirmation(self, state: ResearchState, blockers: list[object]) -> list[str]:
        artifacts = blocked_artifacts_for_hitl(state, blockers)
        expected_roots = sorted(
            {
                str(Path(item.local_path).parent)
                for item in artifacts
                if item.local_path
            }
        )
        search_roots = sorted(dict.fromkeys([*expected_roots, *state.manual_asset_roots]))
        validation_summary = self._materialize_manual_files(state, search_roots, artifacts)
        self._normalize_state_assets(state)
        self._validate_artifacts(state)
        self._refresh_capability_matrix(state)
        self._refresh_classified_failures(state)
        self._refresh_blockers(state)
        self.memory.save_state(state, label="post_hitl_revalidation")
        return validation_summary

    def _handle_hitl(self, state: ResearchState) -> bool:
        if not self.config.execution.hitl_enabled:
            return False
        blockers = should_trigger_hitl(
            state,
            repeat_threshold=self.config.execution.hitl_blocker_repeat_threshold,
            strategy_threshold=self.config.execution.hitl_strategy_threshold,
        )
        if not blockers:
            return False
        artifacts = blocked_artifacts_for_hitl(state, blockers)
        prompt_text, requested_actions = build_hitl_prompt(state, blockers, artifacts)
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
        for line in prompt_text.rstrip().splitlines():
            self._log(line)
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
                state.hitl_events[-1] = updated_event
                self.memory.record_hitl_event(updated_event)
                self._log(
                    "HITL confirmation did not unblock the run. Validation summary: "
                    + "; ".join(validation_summary or ["no matching validated files found"])
                )
                continue
            if selection == "2":
                instruction = input("Instruction> ").strip()
                manual_paths = extract_absolute_paths(instruction)
                if manual_paths:
                    state.manual_asset_roots = sorted(set([*state.manual_asset_roots, *manual_paths]))
                state.human_guidance_notes.append(instruction)
                validation_summary = []
                if manual_paths:
                    validation_summary = self._revalidate_after_human_confirmation(state, blockers)
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
        before_artifacts = len(state.external_artifacts)
        before_repositories = len(state.repositories)
        before_envs = len(state.environment_records)
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
        try:
            summary = agent.run(state, self.tools, self.runtime)
            if spec.phase in {ResearchPhase.EXPERIMENT_PLANNING, ResearchPhase.PREFLIGHT_VALIDATION}:
                self._ground_experiment_plans(state)
            self._post_phase_sync(state, spec)
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
        except Exception:
            if spec.phase == ResearchPhase.ACQUISITION:
                self._hydrate_state_from_memory(state)
                self._post_phase_sync(state, spec)
                if (
                    len(state.external_artifacts) > before_artifacts
                    or len(state.repositories) > before_repositories
                    or len(state.environment_records) > before_envs
                ):
                    summary = (
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
            if self._handle_hitl(state) and state.termination_decision:
                break
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
