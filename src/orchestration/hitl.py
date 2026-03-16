from __future__ import annotations

import re
from pathlib import Path

from common import canonicalize_artifact_id
from state import ArtifactRecord, BlockerRecord, ResearchState


_ABS_PATH_RE = re.compile(r"(/[^ \n\t\"']+)")
_TRAILING_PATH_PUNCTUATION = ";,)]}:"
_HITL_BLOCKER_TYPES = {
    "dataset_acquisition_failure",
    "env_resolution_failure",
    "repo_unavailable",
    "backend_missing",
}


def select_hitl_blockers(
    state: ResearchState,
    repeat_threshold: int,
    strategy_threshold: int,
    *,
    allow_active: bool = False,
) -> list[BlockerRecord]:
    if not state.blocker_registry:
        return []
    if state.capability_matrix and state.capability_matrix.target_dataset_preparing:
        return []
    candidates: list[BlockerRecord] = []
    for blocker in state.blocker_registry:
        if blocker.blocker_type not in _HITL_BLOCKER_TYPES:
            continue
        if not blocker.route_exhausted and blocker.terminality not in {
            "persistent",
            "likely_unrecoverable_in_current_route",
        }:
            continue
        if blocker.repeat_count < repeat_threshold and not blocker.route_exhausted:
            continue
        autonomous_options = {
            item
            for item in blocker.recommended_pivots
            if item not in {"terminate_blocked", "manual_intervention"}
        }
        if autonomous_options and not autonomous_options.issubset(set(blocker.recovery_strategies_tried)):
            continue
        if len(blocker.recovery_strategies_tried) < strategy_threshold and not blocker.route_exhausted:
            continue
        candidates.append(blocker)
    if not candidates:
        return []
    latest = state.hitl_events[-1] if state.hitl_events else None
    if not allow_active and latest and latest.status in {"requested", "still_blocked"}:
        return []
    return candidates


def should_trigger_hitl(
    state: ResearchState,
    repeat_threshold: int,
    strategy_threshold: int,
) -> list[BlockerRecord]:
    return select_hitl_blockers(
        state,
        repeat_threshold,
        strategy_threshold,
        allow_active=False,
    )


def blocked_artifacts_for_hitl(state: ResearchState, blockers: list[BlockerRecord]) -> list[ArtifactRecord]:
    targets = {item.target_entity for item in blockers}
    selected: list[ArtifactRecord] = []
    for artifact in state.external_artifacts:
        artifact_aliases = {
            alias
            for alias in [
                artifact.canonical_id,
                artifact.artifact_id,
                *artifact.raw_aliases,
            ]
            if alias
        }
        semantic_payload = (
            artifact.semantic_spec.model_dump(exclude_none=True)
            if artifact.semantic_spec is not None
            else {}
        )
        for target in list(targets):
            target_aliases = {target}
            target_canonical, _ = canonicalize_artifact_id(
                target,
                local_path=artifact.local_path,
                title=artifact.title,
                metadata={
                    **semantic_payload,
                    **artifact.metadata,
                },
                artifact_type=artifact.artifact_type,
            )
            if target_canonical:
                target_aliases.add(target_canonical)
            if artifact_aliases.intersection(target_aliases):
                selected.append(artifact)
                break
        else:
            continue
    return selected


def build_hitl_prompt(
    state: ResearchState,
    blockers: list[BlockerRecord],
    artifacts: list[ArtifactRecord],
) -> tuple[str, list[str]]:
    artifact_lines: list[str] = []
    destination_lines: list[str] = []
    requested_actions: list[str] = []
    if artifacts:
        for artifact in artifacts[:8]:
            filename = Path(artifact.local_path or artifact.title or artifact.artifact_id).name
            artifact_lines.append(f"- {filename}")
            if artifact.local_path:
                destination_lines.append(f"- {Path(artifact.local_path).parent}")
        requested_actions.append("Provide validated copies of the blocked dataset files.")
        requested_actions.append("Place them at the expected local paths or tell the agent about an alternate local mirror.")
    else:
        requested_actions.append("Tell the agent what alternate source, path, or reduced-scope route to use.")
    blocker_lines = "\n".join(
        f"- {item.blocker_type} on {item.target_entity}: {item.evidence_summary}"
        for item in blockers[:6]
    )
    strategy_lines = "\n".join(
        f"- {item.target_entity}: {', '.join(item.recovery_strategies_tried) or 'none recorded'}"
        for item in blockers[:6]
    )
    unique_destinations = "\n".join(sorted(dict.fromkeys(destination_lines))) or f"- {state.work_directory}"
    prompt = (
        "Human intervention required.\n"
        "The autonomous research system is blocked and cannot make further meaningful progress without help.\n\n"
        "Current blockers:\n"
        f"{blocker_lines}\n\n"
        "Autonomous recovery strategies already attempted:\n"
        f"{strategy_lines}\n\n"
        "Required files or assets:\n"
        f"{chr(10).join(artifact_lines) if artifact_lines else '- See blocker summary above.'}\n\n"
        "Place files under one of these expected directories, or provide an alternate local path:\n"
        f"{unique_destinations}\n\n"
        "Reply with one of:\n"
        "1. OK, I've already done that\n"
        "2. Tell the agent what to do differently\n"
        "3. Skip this target and continue with reduced scope\n"
        "4. Abort this run\n"
    )
    return prompt, requested_actions


def extract_absolute_paths(text: str) -> list[str]:
    cleaned: list[str] = []
    for match in _ABS_PATH_RE.finditer(text):
        candidate = match.group(1).rstrip(_TRAILING_PATH_PUNCTUATION)
        if candidate:
            cleaned.append(candidate)
    return sorted(dict.fromkeys(cleaned))
