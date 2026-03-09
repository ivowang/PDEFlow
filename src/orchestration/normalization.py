from __future__ import annotations

from collections import defaultdict

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


def normalize_artifacts(records: list[ArtifactRecord]) -> list[ArtifactRecord]:
    grouped: dict[str, list[ArtifactRecord]] = defaultdict(list)
    for item in records:
        canonical_id, semantic_spec = canonicalize_artifact_id(
            item.canonical_id or item.artifact_id,
            local_path=item.local_path,
            title=item.title,
            metadata=item.metadata,
            artifact_type=item.artifact_type,
        )
        normalized = item.model_copy(
            update={
                "canonical_id": canonical_id,
                "raw_aliases": sorted(set([item.artifact_id, canonical_id, *item.raw_aliases])),
                "semantic_spec": AssetSemanticSpec.model_validate(semantic_spec),
            }
        )
        grouped[canonical_id].append(normalized)
    merged: list[ArtifactRecord] = []
    for canonical_id, items in grouped.items():
        chosen = _best_artifact(items)
        aliases = sorted({alias for item in items for alias in [item.artifact_id, canonical_id, *item.raw_aliases] if alias})
        notes = sorted({note for item in items for note in item.notes})
        metadata = dict(chosen.metadata)
        for item in items:
            metadata.update({key: value for key, value in item.metadata.items() if value not in (None, "", [], {})})
        merged.append(
            chosen.model_copy(
                update={
                    "artifact_id": choose_preferred_identifier(aliases),
                    "canonical_id": canonical_id,
                    "raw_aliases": aliases,
                    "notes": notes,
                    "metadata": metadata,
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
