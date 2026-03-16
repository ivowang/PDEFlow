from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any

from state import (
    AssetSemanticSpec,
    ArtifactRecord,
    BlockerRecord,
    CapabilityMatrix,
    ClassifiedFailure,
    CycleDeltaRecord,
    DiaryEntry,
    EnvironmentRecord,
    ExperimentPlan,
    ExperimentRecord,
    EvaluationMemo,
    GeneratedReport,
    HITLEvent,
    MemoryKind,
    MemoryNote,
    PaperNote,
    PreflightReport,
    ProgramCandidate,
    RepositoryRecord,
    ResearchPhase,
    ResearchState,
    RouteDecisionRecord,
    SecretStatus,
)
from common import (
    append_jsonl,
    canonicalize_artifact_id,
    choose_preferred_identifier,
    ensure_dir,
    now_utc,
    to_plain_data,
    write_json,
)
from common import read_jsonl
from .logging import ResearchLogger


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


def _normalize_loaded_artifacts(
    records: list[ArtifactRecord],
    *,
    preferred_root: Path | None = None,
) -> list[ArtifactRecord]:
    grouped: dict[str, list[ArtifactRecord]] = {}
    path_to_group: dict[str, str] = {}
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
        group_key = canonical_id
        if normalized.local_path:
            path_key = str(Path(normalized.local_path).absolute())
            group_key = path_to_group.get(path_key, group_key)
            path_to_group[path_key] = group_key
        grouped.setdefault(group_key, []).append(normalized)

    merged: list[ArtifactRecord] = []
    for canonical_id, items in grouped.items():
        chosen = max(
            items,
            key=lambda item: (
                _ARTIFACT_STATUS_PRIORITY.get(item.status, -1),
                int(bool(item.validation and item.validation.ready_for_training)),
                int(bool(item.local_path)),
                int(
                    (item.validation.size_bytes if item.validation else 0)
                    or (item.download_metadata.file_size if item.download_metadata else 0)
                    or 0
                ),
            ),
        )
        aliases = sorted({alias for item in items for alias in [item.artifact_id, canonical_id, *item.raw_aliases] if alias})
        metadata = dict(chosen.metadata)
        semantic_payload = (
            chosen.semantic_spec.model_dump(exclude_none=True)
            if chosen.semantic_spec is not None
            else {}
        )
        local_paths = [item.local_path for item in items if item.local_path]
        source_urls = [item.source_url for item in items if item.source_url]
        download_candidates = [item.download_metadata for item in items if item.download_metadata is not None]
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
        def _path_score(path: str | None) -> tuple[int, int, int]:
            if not path:
                return (-1, -1, -1)
            resolved = Path(path).resolve()
            exists = int(resolved.exists())
            preferred = int(preferred_root is not None and str(resolved).startswith(str(preferred_root.resolve())))
            return (exists, preferred, len(path))

        merged.append(
            chosen.model_copy(
                update={
                    "artifact_id": choose_preferred_identifier(aliases),
                    "canonical_id": canonical_id,
                    "raw_aliases": aliases,
                    "metadata": metadata,
                    "semantic_spec": AssetSemanticSpec.model_validate(semantic_payload),
                    "local_path": max(local_paths, key=_path_score, default=chosen.local_path),
                    "source_url": source_urls[0] if source_urls else chosen.source_url,
                    "download_metadata": max(
                        download_candidates,
                        key=lambda item: (
                            int(bool(item.checksum and item.checksum.matched is True)),
                            int(bool(item.local_path)),
                            int(item.file_size or 0),
                        ),
                        default=chosen.download_metadata,
                    ),
                }
            )
        )
    return sorted(merged, key=lambda item: item.canonical_id or item.artifact_id)


class ResearchMemory:
    """Filesystem-first memory hierarchy plus a small SQLite lineage store."""

    def __init__(self, root: Path):
        self.root = ensure_dir(root)
        self.process_path = self.root / "process.txt"
        self.state_dir = ensure_dir(root / "state")
        self.logs_dir = ensure_dir(root / "logs")
        self.command_logs_dir = ensure_dir(self.logs_dir / "commands")
        self.logger = ResearchLogger(root=self.root, logs_dir=self.logs_dir, process_path=self.process_path)
        self.memory_dir = ensure_dir(root / "memory")
        self.memory_index_dir = ensure_dir(self.memory_dir / "index")
        self.memory_episodes_dir = ensure_dir(self.memory_dir / "episodes")
        self.memory_evaluations_dir = ensure_dir(self.memory_dir / "evaluations")
        self.memory_reflections_dir = ensure_dir(self.memory_dir / "reflections")
        self.memory_lessons_dir = ensure_dir(self.memory_dir / "lessons")
        self.memory_strategy_dir = ensure_dir(self.memory_dir / "strategy")
        self.memory_evolution_dir = ensure_dir(self.memory_dir / "evolution")
        self.literature_dir = ensure_dir(root / "literature")
        self.reports_dir = ensure_dir(root / "reports")
        self.programs_dir = ensure_dir(root / "programs")
        self.experiments_dir = ensure_dir(root / "experiments")
        self.preflight_dir = ensure_dir(root / "preflight")
        self.diary_dir = ensure_dir(root / "diary")
        self.repositories_dir = ensure_dir(root / "repositories")
        self.artifacts_dir = ensure_dir(root / "artifacts")
        self.environments_dir = ensure_dir(root / "environments")
        self.sessions_db = self.state_dir / "agents_sessions.sqlite"
        self.program_db_path = self.programs_dir / "program_db.sqlite3"
        self._init_program_db()
        self.record_process(
            f"Initialized research workspace at {self.root}.",
            print_to_terminal=False,
        )

    def _init_program_db(self) -> None:
        with sqlite3.connect(self.program_db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS programs (
                    program_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    repo_id TEXT,
                    workspace_path TEXT,
                    parent_program_id TEXT,
                    design_id TEXT,
                    hypothesis_id TEXT,
                    entry_command_hint TEXT,
                    status TEXT NOT NULL,
                    changed_files_json TEXT NOT NULL,
                    patch_paths_json TEXT NOT NULL,
                    notes_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metrics_json TEXT,
                    failure_reason TEXT
                )
                """
            )
            conn.commit()

    def save_state(self, state: ResearchState, label: str = "current_state") -> Path:
        path = self.state_dir / f"{label}.json"
        write_json(path, state)
        return path

    def record_phase(self, phase: ResearchPhase, summary: str, artifacts: list[str]) -> None:
        self.logger.log_phase_event(
            phase=phase.value,
            summary=summary,
            outputs=artifacts,
        )
        append_jsonl(
            self.logs_dir / "research_log.jsonl",
            {"phase": phase.value, "summary": summary, "artifacts": artifacts},
        )

    def record_episode(self, label: str, body: str, phase: ResearchPhase) -> None:
        append_jsonl(
            self.memory_dir / "episodic_memory.jsonl",
            {"label": label, "body": body, "phase": phase.value},
        )

    def record_semantic(self, note: str, source: str) -> None:
        append_jsonl(self.memory_dir / "semantic_memory.jsonl", {"source": source, "note": note})

    def record_idea(self, payload: Any) -> None:
        append_jsonl(self.memory_dir / "idea_memory.jsonl", payload)

    def record_literature(self, note: PaperNote) -> None:
        append_jsonl(self.literature_dir / "paper_notes.jsonl", note)

    def record_secret_status(self, status: SecretStatus) -> None:
        append_jsonl(self.memory_dir / "secret_status.jsonl", status)

    def record_artifact(self, artifact: ArtifactRecord) -> None:
        append_jsonl(self.artifacts_dir / "artifact_registry.jsonl", artifact)

    def record_capability_matrix(self, capability_matrix: CapabilityMatrix) -> None:
        append_jsonl(self.state_dir / "capability_matrix.jsonl", capability_matrix)

    def record_environment(self, environment: EnvironmentRecord) -> None:
        append_jsonl(self.environments_dir / "environment_registry.jsonl", environment)

    def record_repository(self, repository: RepositoryRecord) -> None:
        append_jsonl(self.repositories_dir / "repository_registry.jsonl", repository)

    def record_repo_resolution(self, payload: dict[str, Any]) -> None:
        append_jsonl(self.repositories_dir / "repo_resolution_cache.jsonl", payload)

    def record_experiment_plan(self, plan: ExperimentPlan) -> None:
        append_jsonl(self.memory_dir / "experiment_plans.jsonl", plan)

    def record_preflight_report(self, report: PreflightReport) -> None:
        append_jsonl(self.preflight_dir / "preflight_reports.jsonl", report)

    def record_diary(self, entry: DiaryEntry) -> None:
        append_jsonl(self.diary_dir / "research_diary.jsonl", entry)

    def record_experiment(self, experiment: ExperimentRecord) -> None:
        append_jsonl(self.experiments_dir / "experiment_records.jsonl", experiment)

    def record_execution(self, execution: ExperimentRecord) -> None:
        append_jsonl(self.logs_dir / "execution_records.jsonl", execution)

    def record_report(self, report: GeneratedReport) -> None:
        append_jsonl(self.reports_dir / "generated_reports.jsonl", report)

    def _memory_kind_dir(self, kind: MemoryKind) -> Path:
        mapping = {
            MemoryKind.EPISODE: self.memory_episodes_dir,
            MemoryKind.EVALUATION: self.memory_evaluations_dir,
            MemoryKind.REFLECTION: self.memory_reflections_dir,
            MemoryKind.LESSON: self.memory_lessons_dir,
            MemoryKind.STRATEGY: self.memory_strategy_dir,
            MemoryKind.EVOLUTION: self.memory_evolution_dir,
        }
        return mapping[kind]

    def _render_memory_note(self, note: MemoryNote) -> str:
        tag_text = ", ".join(note.tags) if note.tags else "none"
        related_text = (
            "\n".join(f"- {key}: {value}" for key, value in sorted(note.related_ids.items()))
            if note.related_ids
            else "- none"
        )
        return (
            f"# {note.title}\n\n"
            f"- kind: {note.kind.value}\n"
            f"- cycle: {note.cycle_index}\n"
            f"- phase: {note.phase or 'n/a'}\n"
            f"- created_at: {note.created_at}\n"
            f"- tags: {tag_text}\n\n"
            "## Summary\n\n"
            f"{note.summary}\n\n"
            "## Details\n\n"
            f"{note.body}\n\n"
            "## Related IDs\n\n"
            f"{related_text}\n"
        )

    def record_memory_note(self, note: MemoryNote) -> MemoryNote:
        directory = self._memory_kind_dir(note.kind)
        path = directory / f"{note.note_id}.md"
        path.write_text(self._render_memory_note(note), encoding="utf-8")
        stored = note.model_copy(update={"path": str(path)})
        append_jsonl(self.memory_index_dir / "memory_index.jsonl", stored)
        append_jsonl(self.memory_dir / f"{note.kind.value}_notes.jsonl", stored)
        return stored

    def load_memory_notes(
        self,
        *,
        kinds: list[MemoryKind] | None = None,
        limit: int | None = None,
    ) -> list[MemoryNote]:
        notes = [
            MemoryNote.model_validate(item)
            for item in read_jsonl(self.memory_index_dir / "memory_index.jsonl")
        ]
        if kinds is not None:
            allowed = {kind.value for kind in kinds}
            notes = [note for note in notes if note.kind.value in allowed]
        notes.sort(key=lambda note: note.created_at)
        if limit is not None:
            notes = notes[-limit:]
        return notes

    def _render_evaluation_memo(self, memo: EvaluationMemo) -> str:
        metrics_text = (
            "\n".join(f"- {key}: {value}" for key, value in sorted(memo.metrics.items()))
            if memo.metrics
            else "- none"
        )
        findings_text = "\n".join(f"- {item}" for item in memo.findings) if memo.findings else "- none"
        failures_text = "\n".join(f"- {item}" for item in memo.failure_modes) if memo.failure_modes else "- none"
        actions_text = (
            "\n".join(f"- {item}" for item in memo.recommended_actions)
            if memo.recommended_actions
            else "- none"
        )
        return (
            f"# {memo.summary}\n\n"
            f"- cycle: {memo.cycle_index}\n"
            f"- phase: {memo.phase}\n"
            f"- verdict: {memo.verdict}\n"
            f"- support_level: {memo.support_level}\n"
            f"- experiment_id: {memo.experiment_id or 'n/a'}\n"
            f"- plan_id: {memo.plan_id or 'n/a'}\n"
            f"- program_id: {memo.program_id or 'n/a'}\n"
            f"- hypothesis_id: {memo.hypothesis_id or 'n/a'}\n"
            f"- compared_to: {memo.compared_to or 'n/a'}\n"
            f"- created_at: {memo.created_at}\n\n"
            "## Evaluation\n\n"
            f"{memo.body}\n\n"
            "## Metrics\n\n"
            f"{metrics_text}\n\n"
            "## Findings\n\n"
            f"{findings_text}\n\n"
            "## Failure Modes\n\n"
            f"{failures_text}\n\n"
            "## Recommended Actions\n\n"
            f"{actions_text}\n"
        )

    def record_evaluation_memo(self, memo: EvaluationMemo) -> tuple[EvaluationMemo, MemoryNote]:
        path = self.memory_evaluations_dir / f"{memo.memo_id}.md"
        path.write_text(self._render_evaluation_memo(memo), encoding="utf-8")
        stored = memo.model_copy(update={"path": str(path)})
        append_jsonl(self.memory_index_dir / "evaluation_index.jsonl", stored)
        append_jsonl(self.memory_dir / "evaluation_memos.jsonl", stored)
        note = MemoryNote(
            note_id=stored.memo_id,
            kind=MemoryKind.EVALUATION,
            title=stored.summary,
            summary=stored.summary,
            body=stored.body,
            cycle_index=stored.cycle_index,
            phase=stored.phase,
            tags=["evaluation", stored.verdict, stored.support_level],
            related_ids={
                key: value
                for key, value in {
                    "experiment_id": stored.experiment_id,
                    "plan_id": stored.plan_id,
                    "program_id": stored.program_id,
                    "hypothesis_id": stored.hypothesis_id,
                }.items()
                if value
            },
        )
        stored_note = self.record_memory_note(note)
        return stored, stored_note

    def load_evaluation_memos(self, limit: int | None = None) -> list[EvaluationMemo]:
        sources = [
            self.memory_index_dir / "evaluation_index.jsonl",
            self.memory_dir / "evaluation_memos.jsonl",
            self.memory_dir / "evaluations.jsonl",
        ]
        latest: dict[str, EvaluationMemo] = {}
        for source in sources:
            for item in read_jsonl(source):
                if not isinstance(item, dict):
                    continue
                if "memo_id" not in item or "verdict" not in item or "support_level" not in item:
                    continue
                memo = EvaluationMemo.model_validate(item)
                latest[memo.memo_id] = memo
        memos = list(latest.values())
        memos.sort(key=lambda memo: memo.created_at)
        if limit is not None:
            memos = memos[-limit:]
        return memos

    def record_failure(self, failure: ClassifiedFailure) -> None:
        append_jsonl(self.memory_dir / "classified_failures.jsonl", failure)

    def record_hitl_event(self, event: HITLEvent) -> None:
        append_jsonl(self.memory_dir / "hitl_events.jsonl", event)

    def record_blocker(self, blocker: BlockerRecord) -> None:
        append_jsonl(self.memory_dir / "blocker_registry.jsonl", blocker)

    def record_route_decision(self, route_decision: RouteDecisionRecord) -> None:
        append_jsonl(self.state_dir / "route_history.jsonl", route_decision)

    def record_cycle_delta(self, cycle_delta: CycleDeltaRecord) -> None:
        append_jsonl(self.state_dir / "cycle_deltas.jsonl", cycle_delta)

    def record_tool_event(self, payload: dict[str, Any]) -> None:
        self.logger.log_tool_event(payload)

    def record_agent_event(
        self,
        *,
        agent_name: str,
        phase: ResearchPhase,
        status: str,
        cycle_index: int,
        content: str,
        payload: dict[str, Any] | None = None,
        print_to_terminal: bool = False,
    ) -> None:
        self.logger.log_agent_event(
            agent_name=agent_name,
            phase=phase.value,
            status=status,
            cycle_index=cycle_index,
            content=content,
            payload=payload,
            print_to_terminal=print_to_terminal,
        )

    def record_core_progress(
        self,
        message: str,
        *,
        kind: str = "milestone",
        phase: ResearchPhase | None = None,
        agent_name: str | None = None,
        cycle_index: int | None = None,
        payload: dict[str, Any] | None = None,
        print_to_terminal: bool = False,
    ) -> None:
        self.logger.log_core_progress(
            message,
            kind=kind,
            phase=phase.value if phase is not None else None,
            agent=agent_name,
            cycle_index=cycle_index,
            payload=payload,
            print_to_terminal=print_to_terminal,
        )

    def _load_registry(self, path: Path, model_type: type[Any], key: str, fallback_key: str | None = None) -> list[Any]:
        latest: dict[str, Any] = {}
        for item in read_jsonl(path):
            model = model_type.model_validate(item)
            identifier = getattr(model, key)
            if identifier is None and fallback_key is not None:
                identifier = getattr(model, fallback_key)
            latest[str(identifier)] = model
        return list(latest.values())

    def load_artifacts(self) -> list[ArtifactRecord]:
        records = [
            ArtifactRecord.model_validate(item)
            for item in read_jsonl(self.artifacts_dir / "artifact_registry.jsonl")
        ]
        return _normalize_loaded_artifacts(records, preferred_root=self.root)

    def load_repositories(self) -> list[RepositoryRecord]:
        return self._load_registry(
            self.repositories_dir / "repository_registry.jsonl",
            RepositoryRecord,
            "canonical_id",
            fallback_key="repo_id",
        )

    def load_environments(self) -> list[EnvironmentRecord]:
        return self._load_registry(
            self.environments_dir / "environment_registry.jsonl",
            EnvironmentRecord,
            "canonical_id",
            fallback_key="env_id",
        )

    def record_process(self, message: str, print_to_terminal: bool = True) -> None:
        self.logger.log_debug(
            message,
            category="process",
            event_type="process_message",
            print_to_terminal=print_to_terminal,
            mirror_to_process=True,
        )

    def register_program(self, candidate: ProgramCandidate) -> None:
        payload = to_plain_data(candidate)
        with sqlite3.connect(self.program_db_path) as conn:
            conn.execute(
                """
                INSERT INTO programs (
                    program_id, title, summary, repo_id, workspace_path, parent_program_id,
                    design_id, hypothesis_id, entry_command_hint, status, changed_files_json,
                    patch_paths_json, notes_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(program_id) DO UPDATE SET
                    title=excluded.title,
                    summary=excluded.summary,
                    repo_id=excluded.repo_id,
                    workspace_path=excluded.workspace_path,
                    parent_program_id=excluded.parent_program_id,
                    design_id=excluded.design_id,
                    hypothesis_id=excluded.hypothesis_id,
                    entry_command_hint=excluded.entry_command_hint,
                    status=excluded.status,
                    changed_files_json=excluded.changed_files_json,
                    patch_paths_json=excluded.patch_paths_json,
                    notes_json=excluded.notes_json,
                    created_at=excluded.created_at
                """,
                (
                    payload["program_id"],
                    payload["title"],
                    payload["summary"],
                    payload["repo_id"],
                    payload["workspace_path"],
                    payload["parent_program_id"],
                    payload["design_id"],
                    payload["hypothesis_id"],
                    payload["entry_command_hint"],
                    payload["status"],
                    json.dumps(payload["changed_files"]),
                    json.dumps(payload["patch_paths"]),
                    json.dumps(payload["notes"]),
                    payload["created_at"],
                ),
            )
            conn.commit()

    def update_program_result(
        self,
        program_id: str,
        status: str,
        metrics: dict[str, Any] | None = None,
        failure_reason: str | None = None,
    ) -> None:
        with sqlite3.connect(self.program_db_path) as conn:
            conn.execute(
                """
                UPDATE programs
                SET status = ?, metrics_json = ?, failure_reason = ?
                WHERE program_id = ?
                """,
                (
                    status,
                    json.dumps(metrics) if metrics is not None else None,
                    failure_reason,
                    program_id,
                ),
            )
            conn.commit()
