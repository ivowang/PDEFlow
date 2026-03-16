from __future__ import annotations

from typing import Any

from .utils import now_utc, short_hash
from state import EvaluationMemo, ExperimentRecord, MemoryKind, MemoryNote, PreflightReport, ReflectionRecord, ResearchState


_LOWER_IS_BETTER_TOKENS = ("loss", "error", "rmse", "mae", "mse", "nll", "l1", "l2")
_HIGHER_IS_BETTER_TOKENS = ("acc", "accuracy", "f1", "r2", "score", "auc")


def _metric_direction(metric_name: str) -> str | None:
    lowered = metric_name.lower()
    if any(token in lowered for token in _LOWER_IS_BETTER_TOKENS):
        return "lower"
    if any(token in lowered for token in _HIGHER_IS_BETTER_TOKENS):
        return "higher"
    return None


def _compare_metrics(
    current: dict[str, Any],
    reference: dict[str, Any] | None,
) -> tuple[list[str], str]:
    if not current:
        return (["No quantitative metrics were parsed from this run."], "no_metrics")
    if not reference:
        return (
            ["This run produced quantitative evidence, but there was no comparable prior baseline metric snapshot."],
            "evidence_generated",
        )
    findings: list[str] = []
    improved = 0
    regressed = 0
    for key, value in current.items():
        if key not in reference:
            continue
        try:
            current_value = float(value)
            reference_value = float(reference[key])
        except (TypeError, ValueError):
            continue
        direction = _metric_direction(key)
        if direction == "lower":
            delta = reference_value - current_value
            if delta > 0:
                improved += 1
                findings.append(f"{key} improved from {reference_value} to {current_value}.")
            elif delta < 0:
                regressed += 1
                findings.append(f"{key} regressed from {reference_value} to {current_value}.")
        elif direction == "higher":
            delta = current_value - reference_value
            if delta > 0:
                improved += 1
                findings.append(f"{key} improved from {reference_value} to {current_value}.")
            elif delta < 0:
                regressed += 1
                findings.append(f"{key} regressed from {reference_value} to {current_value}.")
    if not findings:
        return (
            ["Metrics were produced, but no directly comparable scalar metric could be aligned against the prior reference."],
            "evidence_generated",
        )
    if improved > 0 and regressed == 0:
        return findings, "improved"
    if regressed > 0 and improved == 0:
        return findings, "regressed"
    return findings, "mixed"


def build_experiment_evaluation_memos(
    state: ResearchState,
    experiment_records: list[ExperimentRecord],
    prior_best_results: dict[str, dict[str, Any]],
) -> list[EvaluationMemo]:
    memos: list[EvaluationMemo] = []
    for record in experiment_records:
        if record.job_kind != "experiment":
            continue
        reference = prior_best_results.get(record.program_id) or (
            prior_best_results.get(state.selected_baseline_program_id or "", None)
            if record.program_id != state.selected_baseline_program_id
            else None
        )
        findings, metric_verdict = _compare_metrics(record.metrics, reference)
        if record.status == "completed" and record.metrics:
            verdict = metric_verdict
            support_level = "supported" if metric_verdict == "improved" else "partial"
            compared_to = (
                record.program_id
                if reference is prior_best_results.get(record.program_id)
                else state.selected_baseline_program_id
            )
        elif record.status == "completed":
            verdict = "completed_without_metrics"
            support_level = "inconclusive"
            compared_to = None
            findings = [*findings, "Execution completed, but no usable metric file was parsed."]
        else:
            verdict = "execution_blocked"
            support_level = "unsupported"
            compared_to = state.selected_baseline_program_id
            findings = [*findings, *(record.failure_modes or ["Execution did not reach a valid evaluation state."])]
        summary = (
            f"Evaluation for {record.experiment_id}: verdict={verdict}, status={record.status}, "
            f"metrics={len(record.metrics)}."
        )
        body_lines = [
            f"The experiment `{record.experiment_id}` for program `{record.program_id}` finished with status `{record.status}`.",
        ]
        if compared_to:
            body_lines.append(f"It was evaluated against the prior reference `{compared_to}`.")
        if findings:
            body_lines.append("Key findings:")
            body_lines.extend(f"- {item}" for item in findings)
        if record.failure_modes:
            body_lines.append("Observed failure modes:")
            body_lines.extend(f"- {item}" for item in record.failure_modes)
        recommended_actions = list(record.observations[:3])
        if not recommended_actions:
            if verdict == "improved":
                recommended_actions.append("Promote this direction into the next hypothesis/design iteration.")
            elif verdict == "regressed":
                recommended_actions.append("Do not repeat this variant unchanged; revise the design before rerunning.")
            elif verdict == "execution_blocked":
                recommended_actions.append("Route through blocker recovery or preflight repair before retrying execution.")
            else:
                recommended_actions.append("Collect a stronger comparison or richer metrics before claiming progress.")
        memos.append(
            EvaluationMemo(
                memo_id=f"eval-{short_hash(record.experiment_id, record.program_id, now_utc())}",
                cycle_index=state.cycle_index,
                phase="experiment",
                verdict=verdict,
                support_level=support_level,
                summary=summary,
                body="\n".join(body_lines),
                experiment_id=record.experiment_id,
                plan_id=record.plan_id,
                program_id=record.program_id,
                compared_to=compared_to,
                metrics=dict(record.metrics),
                findings=findings,
                failure_modes=list(record.failure_modes),
                recommended_actions=recommended_actions,
            )
        )
    return memos


def build_preflight_evaluation_memos(
    state: ResearchState,
    reports: list[PreflightReport],
) -> list[EvaluationMemo]:
    memos: list[EvaluationMemo] = []
    for report in reports:
        verdict = "preflight_passed" if report.passed else "preflight_blocked"
        support_level = "evidence_generated" if report.passed else "unsupported"
        findings = [
            check.details or check.name
            for check in report.failed_checks
        ] or ([report.blocking_reason] if report.blocking_reason else [])
        body_lines = [
            f"Preflight for plan `{report.plan_id}` {'passed' if report.passed else 'failed'}."
        ]
        if report.blocking_reason:
            body_lines.append(f"Blocking reason: {report.blocking_reason}")
        if findings:
            body_lines.append("Observed checks:")
            body_lines.extend(f"- {item}" for item in findings)
        recommended_actions = []
        if report.passed:
            recommended_actions.append("Launch this plan because the executable prerequisites passed preflight.")
        else:
            recommended_actions.append(
                f"Do not launch this plan until the blocking issue is resolved via {report.recommended_route or 'acquisition/recovery'}."
            )
        memos.append(
            EvaluationMemo(
                memo_id=f"preflight-{short_hash(report.report_id, now_utc())}",
                cycle_index=state.cycle_index,
                phase="preflight_validation",
                verdict=verdict,
                support_level=support_level,
                summary=f"Preflight memo for {report.plan_id}: {verdict}.",
                body="\n".join(body_lines),
                plan_id=report.plan_id,
                program_id=report.program_id,
                findings=findings,
                failure_modes=[report.blocking_reason] if report.blocking_reason else [],
                recommended_actions=recommended_actions,
            )
        )
    return memos


def build_reflection_memory_notes(
    state: ResearchState,
    reflection: ReflectionRecord,
) -> list[MemoryNote]:
    notes: list[MemoryNote] = []
    reflection_body = "\n".join(
        [
            f"Verdict: {reflection.verdict}",
            *([f"Stop reason: {reflection.stop_reason}"] if reflection.stop_reason else []),
            *(
                ["Evidence:"] + [f"- {item}" for item in reflection.evidence]
                if reflection.evidence
                else []
            ),
            *(
                ["Accepted lessons:"] + [f"- {item}" for item in reflection.accepted_lessons]
                if reflection.accepted_lessons
                else []
            ),
            *(
                ["Next actions:"] + [f"- {item}" for item in reflection.next_actions]
                if reflection.next_actions
                else []
            ),
        ]
    )
    notes.append(
        MemoryNote(
            note_id=f"reflection-{reflection.reflection_id}",
            kind=MemoryKind.REFLECTION,
            title=f"Reflection cycle {reflection.cycle_index}: {reflection.verdict}",
            summary=reflection.verdict,
            body=reflection_body,
            cycle_index=reflection.cycle_index,
            phase="reflection",
            tags=["reflection", reflection.verdict],
            related_ids={"reflection_id": reflection.reflection_id, **({"hypothesis_id": reflection.hypothesis_id} if reflection.hypothesis_id else {})},
        )
    )
    if reflection.accepted_lessons:
        notes.append(
            MemoryNote(
                note_id=f"lesson-{reflection.reflection_id}",
                kind=MemoryKind.LESSON,
                title=f"Accepted lessons from cycle {reflection.cycle_index}",
                summary=reflection.accepted_lessons[0],
                body="\n".join(f"- {item}" for item in reflection.accepted_lessons),
                cycle_index=reflection.cycle_index,
                phase="reflection",
                tags=["lesson", "accepted"],
                related_ids={"reflection_id": reflection.reflection_id},
            )
        )
    if reflection.preferred_recovery_strategies or reflection.forbidden_attempt_signatures:
        strategy_lines = []
        if reflection.preferred_recovery_strategies:
            strategy_lines.append("Preferred strategies:")
            strategy_lines.extend(f"- {item}" for item in reflection.preferred_recovery_strategies)
        if reflection.forbidden_attempt_signatures:
            strategy_lines.append("Forbidden retry signatures:")
            strategy_lines.extend(f"- {item}" for item in reflection.forbidden_attempt_signatures)
        notes.append(
            MemoryNote(
                note_id=f"strategy-{reflection.reflection_id}",
                kind=MemoryKind.STRATEGY,
                title=f"Strategy update from cycle {reflection.cycle_index}",
                summary=reflection.verdict,
                body="\n".join(strategy_lines),
                cycle_index=reflection.cycle_index,
                phase="reflection",
                tags=["strategy"],
                related_ids={"reflection_id": reflection.reflection_id},
            )
        )
    notes.append(
        MemoryNote(
            note_id=f"evolution-{reflection.reflection_id}",
            kind=MemoryKind.EVOLUTION,
            title=f"Evolution state after cycle {reflection.cycle_index}",
            summary=reflection.verdict,
            body="\n".join(
                [
                    f"The system completed cycle {reflection.cycle_index} with verdict `{reflection.verdict}`.",
                    f"continue_research={reflection.continue_research}",
                    *([f"failure_category={reflection.failure_category}"] if reflection.failure_category else []),
                    *([f"recommended_route_id={reflection.recommended_route_id}"] if reflection.recommended_route_id else []),
                    *(
                        ["Next actions:"] + [f"- {item}" for item in reflection.next_actions]
                        if reflection.next_actions
                        else []
                    ),
                ]
            ),
            cycle_index=reflection.cycle_index,
            phase="reflection",
            tags=["evolution", reflection.verdict],
            related_ids={"reflection_id": reflection.reflection_id},
        )
    )
    return notes
