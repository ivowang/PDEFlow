from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from common import (
    canonicalize_artifact_id,
    canonicalize_env_id,
    canonicalize_repo_id,
    choose_preferred_identifier,
)
from state import ArtifactRecord, AssetSemanticSpec, EnvironmentRecord, RepositoryRecord


_ARTIFACT_STATUS_PRIORITY = {
    "ready_for_training": 7,
    "format_verified": 6,
    "checksum_verified": 5,
    "downloaded": 4,
    "verified_local": 4,
    "verified_remote": 4,
    "blocked": 3,
    "download_failed": 2,
    "corrupted": 1,
    "quarantined": 0,
}


def _best_artifact(records: list[ArtifactRecord]) -> ArtifactRecord:
    return max(
        records,
        key=lambda item: (
            _ARTIFACT_STATUS_PRIORITY.get(item.status, -1),
            bool(item.validation and item.validation.ready_for_training),
            bool(item.local_path),
            len(item.notes),
        ),
    )


def _download_metadata_score(item: ArtifactRecord) -> tuple[int, int, int, int, int]:
    metadata = item.download_metadata
    if metadata is None:
        return (-1, -1, -1, -1, -1)
    checksum = metadata.checksum
    return (
        int(bool(checksum and checksum.expected)),
        int(bool(checksum and checksum.matched is True)),
        int(bool(metadata.source_url)),
        int(bool(metadata.local_path)),
        int(metadata.file_size or 0),
    )


def _pick_artifact_path(records: list[ArtifactRecord], chosen: ArtifactRecord) -> str | None:
    def path_score(path: str | None) -> tuple[int, int]:
        if not path:
            return (-1, -1)
        exists = int(Path(path).exists())
        return (exists, len(path))

    candidates = [item.local_path for item in records if item.local_path]
    if chosen.local_path:
        candidates.insert(0, chosen.local_path)
    if not candidates:
        return None
    return max(candidates, key=path_score)


def _pick_source_url(records: list[ArtifactRecord], chosen: ArtifactRecord) -> str | None:
    if chosen.source_url:
        return chosen.source_url
    for item in records:
        if item.source_url:
            return item.source_url
    return None


def _pick_download_metadata(records: list[ArtifactRecord], chosen: ArtifactRecord) -> object | None:
    candidates = [item for item in records if item.download_metadata is not None]
    if not candidates:
        return chosen.download_metadata
    selected = max(candidates, key=_download_metadata_score).download_metadata
    if selected is None:
        return chosen.download_metadata
    if chosen.download_metadata is None:
        return selected
    return selected.model_copy(
        update={
            "local_path": chosen.download_metadata.local_path or selected.local_path,
            "validation_status": chosen.download_metadata.validation_status or selected.validation_status,
            "attempt_signature": chosen.download_metadata.attempt_signature or selected.attempt_signature,
            "failure_type": chosen.download_metadata.failure_type or selected.failure_type,
            "failure_message": chosen.download_metadata.failure_message or selected.failure_message,
        }
    )


def normalize_artifacts(records: list[ArtifactRecord]) -> list[ArtifactRecord]:
    normalized_items: list[ArtifactRecord] = []
    for item in records:
        semantic_payload = (
            item.semantic_spec.model_dump(exclude_none=True)
            if item.semantic_spec is not None
            else {}
        )
        canonical_id, semantic_spec = canonicalize_artifact_id(
            item.canonical_id or item.artifact_id,
            local_path=item.local_path,
            title=item.title,
            metadata={
                **semantic_payload,
                **item.metadata,
                **({"source_url": item.source_url} if item.source_url else {}),
            },
            artifact_type=item.artifact_type,
        )
        normalized = item.model_copy(
            update={
                "canonical_id": canonical_id,
                "raw_aliases": sorted(set([item.artifact_id, canonical_id, *item.raw_aliases])),
                "semantic_spec": AssetSemanticSpec.model_validate(semantic_spec),
            }
        )
        normalized_items.append(normalized)
    grouped: dict[str, list[ArtifactRecord]] = defaultdict(list)
    for item in normalized_items:
        grouped[item.canonical_id or item.artifact_id].append(item)

    path_to_group: dict[str, str] = {}
    remapped: dict[str, str] = {}
    for canonical_id, items in list(grouped.items()):
        for item in items:
            if not item.local_path:
                continue
            path_key = str(Path(item.local_path).absolute())
            existing = path_to_group.get(path_key)
            if existing and existing != canonical_id:
                remapped[canonical_id] = existing
                break
            path_to_group[path_key] = canonical_id
    if remapped:
        collapsed: dict[str, list[ArtifactRecord]] = defaultdict(list)
        for canonical_id, items in grouped.items():
            collapsed[remapped.get(canonical_id, canonical_id)].extend(items)
        grouped = collapsed

    merged: list[ArtifactRecord] = []
    for canonical_id, items in grouped.items():
        chosen = _best_artifact(items)
        aliases = sorted({alias for item in items for alias in [item.artifact_id, canonical_id, *item.raw_aliases] if alias})
        notes = sorted({note for item in items for note in item.notes})
        metadata = dict(chosen.metadata)
        semantic_payload = (
            chosen.semantic_spec.model_dump(exclude_none=True)
            if chosen.semantic_spec is not None
            else {}
        )
        for item in items:
            metadata.update({key: value for key, value in item.metadata.items() if value not in (None, "", [], {})})
            if item.semantic_spec is not None:
                semantic_payload.update(
                    {
                        key: value
                        for key, value in item.semantic_spec.model_dump(exclude_none=True).items()
                        if value not in (None, "", [], {})
                    }
                )
        merged.append(
            chosen.model_copy(
                update={
                    "artifact_id": choose_preferred_identifier(aliases),
                    "canonical_id": canonical_id,
                    "raw_aliases": aliases,
                    "notes": notes,
                    "metadata": metadata,
                    "semantic_spec": AssetSemanticSpec.model_validate(semantic_payload),
                    "local_path": _pick_artifact_path(items, chosen),
                    "source_url": _pick_source_url(items, chosen),
                    "download_metadata": _pick_download_metadata(items, chosen),
                }
            )
        )
    return sorted(merged, key=lambda item: item.canonical_id or item.artifact_id)


def normalize_repositories(records: list[RepositoryRecord]) -> list[RepositoryRecord]:
    grouped: dict[str, list[RepositoryRecord]] = defaultdict(list)
    for item in records:
        canonical_id = canonicalize_repo_id(item.name or item.repo_id, item.remote_url)
        grouped[canonical_id].append(
            item.model_copy(
                update={
                    "canonical_id": canonical_id,
                    "raw_aliases": sorted(set([item.repo_id, canonical_id, *item.raw_aliases])),
                    "environment_id": item.environment_id or canonicalize_env_id(item.environment_path or item.name, project_hint=item.name),
                }
            )
        )
    normalized: list[RepositoryRecord] = []
    for canonical_id, items in grouped.items():
        chosen = max(items, key=lambda item: (item.bootstrap_status == "ready", bool(item.local_path), len(item.detected_manifests)))
        aliases = sorted({alias for item in items for alias in [item.repo_id, canonical_id, *item.raw_aliases] if alias})
        notes = sorted({note for item in items for note in item.notes})
        normalized.append(
            chosen.model_copy(
                update={
                    "repo_id": choose_preferred_identifier(aliases),
                    "canonical_id": canonical_id,
                    "raw_aliases": aliases,
                    "notes": notes,
                }
            )
        )
    return sorted(normalized, key=lambda item: item.canonical_id or item.repo_id)


def normalize_environments(records: list[EnvironmentRecord]) -> list[EnvironmentRecord]:
    grouped: dict[str, list[EnvironmentRecord]] = defaultdict(list)
    for item in records:
        canonical_id = canonicalize_env_id(item.canonical_id or item.env_id, project_hint=item.project_path)
        grouped[canonical_id].append(
            item.model_copy(
                update={
                    "env_id": canonical_id,
                    "canonical_id": canonical_id,
                }
            )
        )
    normalized: list[EnvironmentRecord] = []
    for canonical_id, items in grouped.items():
        chosen = max(items, key=lambda item: (item.state.value == "ready", len(item.attempted_commands)))
        manifests = sorted({manifest for item in items for manifest in item.manifests})
        recipe = []
        for item in items:
            for command in item.fallback_recipe:
                if command not in recipe:
                    recipe.append(command)
        normalized.append(
            chosen.model_copy(
                update={
                    "manifests": manifests,
                    "fallback_recipe": recipe,
                }
            )
        )
    return sorted(normalized, key=lambda item: item.canonical_id)
