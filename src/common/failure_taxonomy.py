from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FailurePlaybook:
    failure_type: str
    severity: str
    blocking: bool
    allow_experiment_launch: bool
    remediation_steps: tuple[str, ...]
    fallback_strategy: str | None = None


FAILURE_PLAYBOOKS: dict[str, FailurePlaybook] = {
    "dataset_truncated": FailurePlaybook(
        failure_type="dataset_truncated",
        severity="high",
        blocking=True,
        allow_experiment_launch=False,
        remediation_steps=(
            "Quarantine the local dataset shard.",
            "Re-download from the official source with resume and validation enabled.",
            "Require checksum and format validation before planning reuse.",
        ),
        fallback_strategy="Use only validated alternate shards or checkpoints if the exact shard remains unavailable.",
    ),
    "checksum_mismatch": FailurePlaybook(
        failure_type="checksum_mismatch",
        severity="critical",
        blocking=True,
        allow_experiment_launch=False,
        remediation_steps=(
            "Do not reuse the local file.",
            "Quarantine the mismatched file.",
            "Re-download from the official source and re-check checksum before launch.",
        ),
        fallback_strategy="Prefer a validated existing local copy; otherwise stop dependent plans until a trusted copy exists.",
    ),
    "format_validation_failed": FailurePlaybook(
        failure_type="format_validation_failed",
        severity="high",
        blocking=True,
        allow_experiment_launch=False,
        remediation_steps=(
            "Open the file with the format validator.",
            "Quarantine unreadable files.",
            "Acquire a fresh copy before retrying planning or launch.",
        ),
    ),
    "import_error": FailurePlaybook(
        failure_type="import_error",
        severity="high",
        blocking=True,
        allow_experiment_launch=False,
        remediation_steps=(
            "Probe the managed environment for the missing module.",
            "Repair the environment with uv sync or minimal dependency install.",
            "Re-run preflight before launch.",
        ),
    ),
    "backend_missing": FailurePlaybook(
        failure_type="backend_missing",
        severity="high",
        blocking=True,
        allow_experiment_launch=False,
        remediation_steps=(
            "Detect the backend expected by the selected framework.",
            "Install or switch to a supported backend.",
            "Avoid scheduling backend-dependent routes until preflight passes.",
        ),
        fallback_strategy="Prefer routes supported by the current environment capability matrix.",
    ),
    "optional_dependency_sync_failed": FailurePlaybook(
        failure_type="optional_dependency_sync_failed",
        severity="medium",
        blocking=False,
        allow_experiment_launch=True,
        remediation_steps=(
            "Record the failed optional dependency sync.",
            "Proceed with a verified minimal environment only if required training imports pass.",
        ),
    ),
    "github_search_empty": FailurePlaybook(
        failure_type="github_search_empty",
        severity="low",
        blocking=False,
        allow_experiment_launch=True,
        remediation_steps=(
            "Use known verified repositories already cloned locally.",
            "Broaden or rephrase the repository search query only if needed.",
        ),
    ),
    "transfer_stalled": FailurePlaybook(
        failure_type="transfer_stalled",
        severity="high",
        blocking=True,
        allow_experiment_launch=False,
        remediation_steps=(
            "Terminate the stalled transfer.",
            "Preserve a structured failure record with bytes and throughput.",
            "Retry with resume support and bounded attempts.",
        ),
        fallback_strategy="Use a validated existing local copy or stop until a healthy transfer route is available.",
    ),
    "transfer_timeout": FailurePlaybook(
        failure_type="transfer_timeout",
        severity="high",
        blocking=True,
        allow_experiment_launch=False,
        remediation_steps=(
            "Abort the timed-out transfer.",
            "Retry with resume support and bounded attempts.",
            "Do not expose the partial file as reusable.",
        ),
    ),
    "repeated_repair_failure": FailurePlaybook(
        failure_type="repeated_repair_failure",
        severity="critical",
        blocking=True,
        allow_experiment_launch=False,
        remediation_steps=(
            "Stop repeating the same repair path.",
            "Mark the artifact route as blocked for this run.",
            "Switch to a validated fallback route or surface an explicit blocked state.",
        ),
    ),
    "plan_depends_on_blocked_artifact": FailurePlaybook(
        failure_type="plan_depends_on_blocked_artifact",
        severity="critical",
        blocking=True,
        allow_experiment_launch=False,
        remediation_steps=(
            "Remove or block plans that depend on known-bad artifacts.",
            "Return to acquisition only if a bounded repair attempt remains.",
            "Otherwise route to fallback or blocked state.",
        ),
    ),
}


def get_playbook(failure_type: str) -> FailurePlaybook:
    return FAILURE_PLAYBOOKS[failure_type]
