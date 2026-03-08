from __future__ import annotations

from typing import Iterable

from common import get_playbook, short_hash
from state import (
    ArtifactRecord,
    ClassifiedFailure,
    FailureSeverity,
    PreflightReport,
    ResearchState,
)


def _severity(value: str) -> FailureSeverity:
    return FailureSeverity(value)


def _make_failure(
    failure_type: str,
    source_phase: str,
    source_id: str | None,
    summary: str,
    detected_from: dict[str, object] | None = None,
) -> ClassifiedFailure:
    playbook = get_playbook(failure_type)
    return ClassifiedFailure(
        failure_id=f"{failure_type}-{short_hash(source_phase, source_id or 'none', summary)}",
        failure_type=failure_type,
        severity=_severity(playbook.severity),
        blocking=playbook.blocking,
        allow_experiment_launch=playbook.allow_experiment_launch,
        source_phase=source_phase,
        source_id=source_id,
        summary=summary,
        remediation_steps=list(playbook.remediation_steps),
        fallback_strategy=playbook.fallback_strategy,
        detected_from=detected_from or {},
    )


def classify_artifact_failures(
    artifacts: Iterable[ArtifactRecord],
    max_transfer_attempts: int,
) -> list[ClassifiedFailure]:
    failures: list[ClassifiedFailure] = []
    for artifact in artifacts:
        validation = artifact.validation
        if validation:
            reasons = " ".join(validation.failure_reasons).lower()
            if "checksum_mismatch" in reasons or (
                validation.checksum and validation.checksum.expected and validation.checksum.matched is False
            ):
                failures.append(
                    _make_failure(
                        "checksum_mismatch",
                        "acquisition",
                        artifact.artifact_id,
                        f"Artifact {artifact.artifact_id} failed checksum validation.",
                        {"artifact_id": artifact.artifact_id, "path": artifact.local_path},
                    )
                )
            elif "size_below_minimum" in reasons:
                failures.append(
                    _make_failure(
                        "dataset_truncated",
                        "acquisition",
                        artifact.artifact_id,
                        f"Artifact {artifact.artifact_id} is smaller than the minimum size threshold.",
                        {"artifact_id": artifact.artifact_id, "path": artifact.local_path},
                    )
                )
            elif any(marker in reasons for marker in ("hdf5_open_failed", "sample_read_failed", "missing_required_keys")):
                failures.append(
                    _make_failure(
                        "format_validation_failed",
                        "acquisition",
                        artifact.artifact_id,
                        f"Artifact {artifact.artifact_id} failed format validation.",
                        {"artifact_id": artifact.artifact_id, "path": artifact.local_path},
                    )
                )
        download_metadata = artifact.download_metadata
        if download_metadata and download_metadata.failure_type in {"transfer_stalled", "transfer_timeout"}:
            failures.append(
                _make_failure(
                    download_metadata.failure_type,
                    "acquisition",
                    artifact.artifact_id,
                    (
                        f"Transfer for artifact {artifact.artifact_id} failed with "
                        f"{download_metadata.failure_type}."
                    ),
                    {"artifact_id": artifact.artifact_id, "path": artifact.local_path},
                )
            )
            if download_metadata.attempt_count >= max_transfer_attempts:
                failures.append(
                    _make_failure(
                        "repeated_repair_failure",
                        "acquisition",
                        artifact.artifact_id,
                        f"Artifact {artifact.artifact_id} exhausted bounded repair attempts.",
                        {"artifact_id": artifact.artifact_id, "attempt_count": download_metadata.attempt_count},
                    )
                )
    return failures


def classify_preflight_failures(preflight_reports: Iterable[PreflightReport]) -> list[ClassifiedFailure]:
    failures: list[ClassifiedFailure] = []
    for report in preflight_reports:
        if report.passed:
            continue
        categories = {item.category for item in report.failed_checks}
        if "dataset" in categories or "artifact" in categories:
            failures.append(
                _make_failure(
                    "plan_depends_on_blocked_artifact",
                    "preflight_validation",
                    report.plan_id,
                    f"Plan {report.plan_id} depends on a blocked or not-ready artifact.",
                    {"plan_id": report.plan_id, "failed_checks": [item.name for item in report.failed_checks]},
                )
            )
        if "import" in categories:
            failures.append(
                _make_failure(
                    "import_error",
                    "preflight_validation",
                    report.plan_id,
                    f"Plan {report.plan_id} failed entrypoint import or compile preflight.",
                    {"plan_id": report.plan_id},
                )
            )
        if "environment" in categories and report.blocking_reason and "tensorflow" in report.blocking_reason.lower():
            failures.append(
                _make_failure(
                    "backend_missing",
                    "preflight_validation",
                    report.plan_id,
                    f"Plan {report.plan_id} requires a missing backend.",
                    {"plan_id": report.plan_id},
                )
            )
    return failures


def classify_text_failures(failure_summaries: Iterable[str]) -> list[ClassifiedFailure]:
    failures: list[ClassifiedFailure] = []
    for summary in failure_summaries:
        normalized = summary.lower()
        if "no module named" in normalized:
            failures.append(
                _make_failure(
                    "import_error",
                    "reflection",
                    None,
                    summary,
                )
            )
        if "tensorflow" in normalized and "missing" in normalized:
            failures.append(
                _make_failure(
                    "backend_missing",
                    "reflection",
                    None,
                    summary,
                )
            )
        if "uv sync failed" in normalized or "optional dependency" in normalized:
            failures.append(
                _make_failure(
                    "optional_dependency_sync_failed",
                    "acquisition",
                    None,
                    summary,
                )
            )
        if "github" in normalized and "results=0" in normalized:
            failures.append(
                _make_failure(
                    "github_search_empty",
                    "acquisition",
                    None,
                    summary,
                )
            )
    return failures


def classify_state_failures(state: ResearchState, max_transfer_attempts: int) -> list[ClassifiedFailure]:
    failures = [
        *classify_artifact_failures(state.external_artifacts, max_transfer_attempts=max_transfer_attempts),
        *classify_preflight_failures(state.preflight_reports[-8:]),
        *classify_text_failures(state.failure_summaries[-16:]),
    ]
    deduped: dict[str, ClassifiedFailure] = {}
    for failure in failures:
        deduped[failure.failure_id] = failure
    return list(deduped.values())
