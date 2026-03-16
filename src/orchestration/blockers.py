from __future__ import annotations

from collections import defaultdict

from common import short_hash
from state import (
    ArtifactRecord,
    ArtifactStatus,
    BlockerRecord,
    CapabilityMatrix,
    CycleDeltaRecord,
    EnvironmentRecord,
    ResearchState,
)


def _artifact_target_id(artifact: ArtifactRecord) -> str:
    return artifact.canonical_id or artifact.artifact_id


def _artifact_attempt_signature(artifact: ArtifactRecord) -> str:
    metadata = artifact.download_metadata
    if metadata and metadata.attempt_signature:
        return metadata.attempt_signature
    if metadata and metadata.strategy_id:
        return f"{metadata.strategy_id}:{metadata.transfer_method or 'unknown'}:{metadata.source_url or artifact.source_url or ''}"
    return artifact.status


def summarize_state_delta(state: ResearchState) -> CycleDeltaRecord:
    ready_artifacts = sorted(
        (item.canonical_id or item.artifact_id)
        for item in state.external_artifacts
        if item.status == ArtifactStatus.READY_FOR_TRAINING.value
    )
    pending_artifacts = sorted(
        (
            item.canonical_id or item.artifact_id,
            item.status,
            str(
                (item.validation.size_bytes if item.validation else 0)
                or (item.download_metadata.bytes_downloaded if item.download_metadata else 0)
            ),
        )
        for item in state.external_artifacts
        if item.status in {
            ArtifactStatus.DOWNLOADED.value,
            ArtifactStatus.CHECKSUM_VERIFIED.value,
            ArtifactStatus.FORMAT_VERIFIED.value,
            ArtifactStatus.VERIFIED_REMOTE.value,
        }
    )
    blocked_artifacts = sorted(
        (item.canonical_id or item.artifact_id)
        for item in state.external_artifacts
        if item.status in {
            ArtifactStatus.BLOCKED.value,
            ArtifactStatus.CORRUPTED.value,
            ArtifactStatus.QUARANTINED.value,
            ArtifactStatus.DOWNLOAD_FAILED.value,
        }
    )
    ready_envs = sorted(
        item.canonical_id for item in state.environment_records if item.state.value == "ready"
    )
    snapshot_signature = short_hash(
        "|".join(ready_artifacts),
        "|".join("::".join(item) for item in pending_artifacts),
        "|".join(blocked_artifacts),
        "|".join(ready_envs),
        str(state.capability_matrix.model_dump(mode="python") if state.capability_matrix else {}),
    )
    previous = state.cycle_deltas[-1] if state.cycle_deltas else None
    previous_ready = set(previous.newly_ready_artifacts) if previous else set()
    previous_blocked = set(previous.newly_blocked_artifacts) if previous else set()
    current_ready = set(ready_artifacts)
    current_blocked = set(blocked_artifacts)
    changed = previous is None or previous.snapshot_signature != snapshot_signature
    summary: list[str] = []
    new_ready = sorted(current_ready - previous_ready)
    new_blocked = sorted(current_blocked - previous_blocked)
    new_envs = sorted(set(ready_envs) - set(previous.newly_ready_environments if previous else []))
    if new_ready:
        summary.append(f"new_ready_artifacts={','.join(new_ready)}")
    if pending_artifacts:
        summary.append(
            "pending_artifacts="
            + ",".join(f"{artifact_id}:{status}" for artifact_id, status, _ in pending_artifacts[:6])
        )
    if new_blocked:
        summary.append(f"new_blocked_artifacts={','.join(new_blocked)}")
    if new_envs:
        summary.append(f"new_ready_envs={','.join(new_envs)}")
    if not summary:
        summary.append("no_material_state_change")
    return CycleDeltaRecord(
        cycle_index=state.cycle_index,
        snapshot_signature=snapshot_signature,
        changed=changed,
        summary=summary,
        newly_ready_artifacts=ready_artifacts,
        newly_blocked_artifacts=blocked_artifacts,
        newly_ready_environments=ready_envs,
    )


def _classify_artifact_blocker(artifact: ArtifactRecord, cycle_index: int) -> BlockerRecord | None:
    if artifact.artifact_type != "dataset":
        return None
    if artifact.status not in {
        ArtifactStatus.BLOCKED.value,
        ArtifactStatus.CORRUPTED.value,
        ArtifactStatus.QUARANTINED.value,
        ArtifactStatus.DOWNLOAD_FAILED.value,
    }:
        return None
    blocker_type = "dataset_acquisition_failure"
    evidence_parts = [f"status={artifact.status}"]
    validation = artifact.validation
    if validation and validation.failure_reasons:
        evidence_parts.extend(validation.failure_reasons[:3])
    if artifact.download_metadata and artifact.download_metadata.failure_type:
        evidence_parts.append(artifact.download_metadata.failure_type)
    strategy = artifact.download_metadata.strategy_id if artifact.download_metadata else None
    recommended = ["local_discovery", "mirror_resolution", "partial_salvage", "fallback_execution"]
    return BlockerRecord(
        blocker_id=f"blocker-{short_hash(blocker_type, _artifact_target_id(artifact))}",
        blocker_type=blocker_type,
        target_entity=_artifact_target_id(artifact),
        first_seen_cycle=cycle_index,
        last_seen_cycle=cycle_index,
        last_attempt_signature=_artifact_attempt_signature(artifact),
        evidence_summary="; ".join(item for item in evidence_parts if item),
        recovery_strategies_tried=[strategy] if strategy else [],
        terminality="temporary",
        recommended_pivots=recommended,
    )


def _classify_environment_blockers(state: ResearchState, cycle_index: int) -> list[BlockerRecord]:
    blockers: list[BlockerRecord] = []
    for env in state.environment_records:
        if env.state.value == "ready":
            continue
        blockers.append(
            BlockerRecord(
                blocker_id=f"blocker-{short_hash('env_resolution_failure', env.canonical_id)}",
                blocker_type="env_resolution_failure",
                target_entity=env.canonical_id,
                first_seen_cycle=cycle_index,
                last_seen_cycle=cycle_index,
                last_attempt_signature=env.strategy or env.state.value,
                evidence_summary=env.failure_reason or env.state.value,
                recovery_strategies_tried=[env.strategy] if env.strategy else [],
                terminality="temporary",
                recommended_pivots=["environment_repair", "fallback_execution"],
            )
        )
    return blockers


def _capability_blockers(capability: CapabilityMatrix | None, cycle_index: int, baseline_program_id: str | None) -> list[BlockerRecord]:
    blockers: list[BlockerRecord] = []
    if capability is None:
        return blockers
    if capability.environment_repair_needed:
        blockers.append(
            BlockerRecord(
                blocker_id=f"blocker-{short_hash('environment_runtime_failure', capability.environment_path or baseline_program_id or 'baseline')}",
                blocker_type="environment_runtime_failure",
                target_entity=capability.environment_path or baseline_program_id or "baseline-env",
                first_seen_cycle=cycle_index,
                last_seen_cycle=cycle_index,
                last_attempt_signature="gpu_runtime_unavailable",
                evidence_summary=(
                    f"torch_runtime_ready={capability.torch_runtime_ready} "
                    f"cuda_available={capability.cuda_available} "
                    f"gpu_runtime_required={capability.gpu_runtime_required} "
                    f"torch_version={capability.torch_version or 'unknown'} "
                    f"torch_cuda_version={capability.torch_cuda_version or 'unknown'}"
                ),
                recommended_pivots=["environment_repair", "fallback_execution"],
            )
        )
    if capability.target_dataset_blocked:
        blockers.append(
            BlockerRecord(
                blocker_id=f"blocker-{short_hash('dataset_acquisition_failure', baseline_program_id or 'baseline')}",
                blocker_type="dataset_acquisition_failure",
                target_entity=baseline_program_id or "baseline",
                first_seen_cycle=cycle_index,
                last_seen_cycle=cycle_index,
                last_attempt_signature="target_dataset_blocked",
                evidence_summary="Capability matrix indicates target dataset is blocked.",
                recommended_pivots=["local_discovery", "mirror_resolution", "fallback_execution", "terminate_blocked"],
            )
        )
    if (
        capability.env_ready
        and capability.codepath_ready
        and not capability.baseline_launch_ready
        and not capability.environment_repair_needed
        and not capability.target_dataset_blocked
        and not capability.target_dataset_preparing
    ):
        blockers.append(
            BlockerRecord(
                blocker_id=f"blocker-{short_hash('baseline_not_launch_ready', baseline_program_id or 'baseline')}",
                blocker_type="baseline_not_launch_ready",
                target_entity=baseline_program_id or "baseline",
                first_seen_cycle=cycle_index,
                last_seen_cycle=cycle_index,
                last_attempt_signature="capability_gate",
                evidence_summary=(
                    f"repo_ready={capability.repo_ready} env_ready={capability.env_ready} "
                    f"dataset_ready={capability.dataset_ready} baseline_launch_ready={capability.baseline_launch_ready}"
                ),
                recommended_pivots=["fallback_execution", "terminate_blocked"],
            )
        )
    return blockers


def refresh_blocker_registry(
    state: ResearchState,
    repeat_threshold: int,
    stagnation_threshold: int,
) -> tuple[list[BlockerRecord], CycleDeltaRecord]:
    cycle_delta = summarize_state_delta(state)
    existing = {item.blocker_id: item for item in state.blocker_registry}
    observations: list[BlockerRecord] = []
    for artifact in state.external_artifacts:
        blocker = _classify_artifact_blocker(artifact, state.cycle_index)
        if blocker:
            observations.append(blocker)
    observations.extend(_classify_environment_blockers(state, state.cycle_index))
    observations.extend(_capability_blockers(state.capability_matrix, state.cycle_index, state.selected_baseline_program_id))

    reflection = state.reflections[-1] if state.reflections else None
    if reflection and reflection.blocked_entities:
        for target in reflection.blocked_entities:
            blocker_type = reflection.failure_category or "dataset_acquisition_failure"
            blocker_id = f"blocker-{short_hash(blocker_type, target)}"
            observations.append(
                BlockerRecord(
                    blocker_id=blocker_id,
                    blocker_type=blocker_type,
                    target_entity=target,
                    first_seen_cycle=state.cycle_index,
                    last_seen_cycle=state.cycle_index,
                    last_attempt_signature="reflection",
                    evidence_summary=reflection.verdict,
                    recovery_strategies_tried=list(reflection.preferred_recovery_strategies),
                    terminality="persistent" if reflection.material_change_required else "temporary",
                    recommended_pivots=list(reflection.preferred_recovery_strategies),
                )
            )

    grouped: dict[str, list[BlockerRecord]] = defaultdict(list)
    for blocker in observations:
        grouped[blocker.blocker_id].append(blocker)

    refreshed: dict[str, BlockerRecord] = {}
    for blocker_id, group in grouped.items():
        observed = group[-1]
        previous = existing.get(blocker_id)
        repeat_count = 1
        recovery_strategies = set(observed.recovery_strategies_tried)
        if previous:
            if previous.last_seen_cycle == state.cycle_index:
                repeat_count = previous.repeat_count
            elif previous.last_attempt_signature == observed.last_attempt_signature:
                repeat_count = previous.repeat_count + 1
            else:
                repeat_count = previous.repeat_count
            recovery_strategies.update(previous.recovery_strategies_tried)
            first_seen = previous.first_seen_cycle
        else:
            first_seen = observed.first_seen_cycle
        exhausted = repeat_count >= repeat_threshold and not cycle_delta.changed
        terminality = observed.terminality
        if exhausted and repeat_count >= stagnation_threshold:
            terminality = "likely_unrecoverable_in_current_route"
        elif exhausted:
            terminality = "persistent"
        refreshed[blocker_id] = observed.model_copy(
            update={
                "first_seen_cycle": first_seen,
                "last_seen_cycle": state.cycle_index,
                "repeat_count": repeat_count,
                "recovery_strategies_tried": sorted(recovery_strategies),
                "route_exhausted": exhausted,
                "terminality": terminality,
                "updated_at": observed.updated_at,
            }
        )

    return list(refreshed.values()), cycle_delta
