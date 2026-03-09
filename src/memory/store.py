from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any

from state import (
    ArtifactRecord,
    BlockerRecord,
    CapabilityMatrix,
    ClassifiedFailure,
    CycleDeltaRecord,
    DiaryEntry,
    EnvironmentRecord,
    ExperimentPlan,
    ExperimentRecord,
    GeneratedReport,
    HITLEvent,
    PaperNote,
    PreflightReport,
    ProgramCandidate,
    RepositoryRecord,
    ResearchPhase,
    ResearchState,
    RouteDecisionRecord,
    SecretStatus,
)
from common import append_jsonl, ensure_dir, now_utc, to_plain_data, write_json
from common import read_jsonl
from .logging import ResearchLogger


class ResearchMemory:
    """Filesystem-first memory hierarchy plus a small SQLite lineage store."""

    def __init__(self, root: Path):
        self.root = ensure_dir(root)
        self.process_path = self.root / "process.txt"
        self.state_dir = ensure_dir(root / "state")
        self.logs_dir = ensure_dir(root / "logs")
        self.logger = ResearchLogger(root=self.root, logs_dir=self.logs_dir, process_path=self.process_path)
        self.memory_dir = ensure_dir(root / "memory")
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
        return self._load_registry(
            self.artifacts_dir / "artifact_registry.jsonl",
            ArtifactRecord,
            "canonical_id",
            fallback_key="artifact_id",
        )

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
