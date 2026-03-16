from __future__ import annotations

from pathlib import Path
import threading
import time
import re
from typing import Any

from state import AcquisitionPhaseOutput, LiteraturePhaseOutput, ResearchPhase, ResearchState
from tools import ResearchTools
from common import dedupe_strings, upsert_by_attr
from runtime import RuntimeAdapter
from .base import BaseResearchAgent


class LiteratureAgent(BaseResearchAgent):
    name = "LiteratureAgent"
    phase = ResearchPhase.LITERATURE_REVIEW
    output_model = LiteraturePhaseOutput

    def runtime_timeout_seconds(self) -> int | None:
        return 300

    def allowed_tool_names(self) -> set[str] | None:
        return {
            "search_arxiv_papers",
            "fetch_url_text",
            "discover_local_artifacts",
            "download_file",
            "extract_pdf_text",
            "read_text_file",
            "search_in_directory",
            "find_files",
        }

    def build_instructions(self, state: ResearchState) -> str:
        return """
You are the literature specialist in an autonomous scientific research system.
Use the available tools to search for relevant papers, fetch evidence, and read enough source material to produce structured literature notes.

Rules:
- Do not invent papers, URLs, claims, or limitations. Use tools first.
- Prefer recent and foundational sources that directly support the current research question.
- Download and inspect PDFs when the abstract is insufficient to support a research claim.
- Stop after you have enough evidence to cover the benchmark, main baselines, physics-informed variants, and a small number of concrete research opportunities.
- Do not repeatedly reacquire or reread the same paper once you already have its abstract or PDF text locally.
- Extract limitations and research opportunities, not generic summaries.
- Keep the final structured output compact: use at most 5 paper notes, keep abstracts and summaries to 1-2 concise sentences, and limit list fields to the highest-signal 2-4 items.
- Return only structured output matching the schema.
"""

    def build_payload(self, state: ResearchState, tools: ResearchTools) -> dict[str, Any]:
        return {
            "research_brief": self._compact_research_brief(state),
            "already_known_papers": [note.title for note in state.literature_notes],
        }

    def _compact_research_brief(self, state: ResearchState) -> dict[str, Any]:
        return {
            "title": state.research_brief.title[:180],
            "question": state.research_brief.question[:300],
            "objectives": [item[:140] for item in state.research_brief.objectives[:3]],
            "constraints": [item[:140] for item in state.research_brief.constraints[:4]],
            "domain_tags": state.research_brief.domain_tags[:6],
        }

    def _derive_queries(self, state: ResearchState) -> list[str]:
        corpus = " ".join(
            [
                state.research_brief.title,
                state.research_brief.question,
                state.research_brief.background,
                *state.research_brief.objectives,
                *state.research_brief.constraints,
                *state.research_brief.domain_tags,
            ]
        ).lower()
        queries: list[str] = []
        if "pdebench" in corpus:
            queries.append("PDEBench benchmark paper neural operator")
        if "fourier neural operator" in corpus or "fno" in corpus or "neural operator" in corpus:
            queries.append("Fourier Neural Operator for Parametric Partial Differential Equations")
        if "deeponet" in corpus:
            queries.append("DeepONet Learning nonlinear operators for identifying differential equations")
        if "physics-informed" in corpus or "physical information" in corpus:
            queries.append("physics-informed neural operator PDE")
            queries.append("physics-informed DeepONet operator learning PDE")
        if "pinn" in corpus:
            queries.append("physics-informed neural networks PDE")
        if "burgers" in corpus or "diffusion-reaction" in corpus or "reaction" in corpus:
            queries.append("PDEBench 1D Burgers diffusion reaction neural operator")
        if not queries:
            queries.append(state.research_brief.question)
        return dedupe_strings(queries)[:6]

    def _paper_score(self, paper: dict[str, Any], keywords: set[str]) -> tuple[int, int]:
        haystack = " ".join(
            str(paper.get(field) or "")
            for field in ("title", "abstract", "source_url")
        ).lower()
        overlap = sum(1 for keyword in keywords if keyword in haystack)
        title_bonus = sum(1 for keyword in keywords if keyword in str(paper.get("title") or "").lower())
        return (overlap, title_bonus)

    def _selected_papers(self, state: ResearchState, papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not papers:
            return []
        keywords = {
            token
            for token in re.findall(r"[a-z0-9][a-z0-9\-]+", " ".join(
                [
                    state.research_brief.title,
                    state.research_brief.question,
                    state.research_brief.background,
                    *state.research_brief.domain_tags,
                ]
            ).lower())
            if len(token) >= 4
        }
        ranked = sorted(
            papers,
            key=lambda item: (
                self._paper_score(item, keywords),
                int(bool(item.get("pdf_url"))),
                int(item.get("year") or 0),
            ),
            reverse=True,
        )
        selected: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in ranked:
            identifier = str(item.get("paper_id") or item.get("source_url") or item.get("title"))
            if identifier in seen:
                continue
            seen.add(identifier)
            selected.append(item)
            if len(selected) >= 3:
                break
        return selected

    def run(self, state: ResearchState, tools: ResearchTools, runtime: RuntimeAdapter) -> str:
        queries = self._derive_queries(state)
        candidate_papers: list[dict[str, Any]] = []
        for query in queries:
            try:
                candidate_papers.extend(tools.search_arxiv_papers(query, max_results=5))
            except Exception as error:  # noqa: BLE001
                tools.memory.record_process(
                    f"Literature search query failed and will be skipped: query='{query}' error={type(error).__name__}: {error}",
                    print_to_terminal=True,
                )
        selected = self._selected_papers(state, candidate_papers)
        paper_payloads: list[dict[str, Any]] = []
        for paper in selected:
            pdf_path: str | None = None
            pdf_url = str(paper.get("pdf_url") or "").strip()
            if pdf_url:
                filename = f"{paper.get('paper_id') or 'paper'}.pdf"
                local_matches = tools.discover_local_artifacts(
                    query=filename,
                    search_roots=[str(Path(state.work_directory).parent)],
                    artifact_type="paper",
                    canonical_target_id=str(paper.get("paper_id") or filename).lower(),
                    limit=5,
                )
                reusable_match = next(
                    (
                        item for item in local_matches
                        if item.get("path") and item.get("status") not in {"corrupted", "quarantined"}
                    ),
                    None,
                )
                if reusable_match is not None:
                    pdf_path = str(reusable_match["path"])
                else:
                    target = Path(state.work_directory) / "external_assets" / filename
                    download_result = tools.download_file(
                        pdf_url,
                        str(target),
                        artifact_id=str(paper.get("paper_id") or filename).lower(),
                        artifact_type="paper",
                        strategy_id="literature_pdf_download",
                        source_type="paper_pdf",
                    )
                    if download_result.get("validation_status") == "ready_for_training":
                        pdf_path = str(target)
            excerpt = ""
            if pdf_path:
                try:
                    extracted = tools.extract_pdf_text(pdf_path, max_pages=4)
                    excerpt = str(extracted.get("text") or "")[:800]
                except Exception:
                    excerpt = ""
            paper_payloads.append(
                {
                    "paper_id": str(paper.get("paper_id") or ""),
                    "title": str(paper.get("title") or ""),
                    "authors": list(paper.get("authors") or [])[:5],
                    "year": paper.get("year"),
                    "abstract": str(paper.get("abstract") or "")[:280],
                    "source_url": str(paper.get("source_url") or ""),
                    "pdf_url": pdf_url,
                    "pdf_path": pdf_path,
                    "excerpt": excerpt,
                }
            )
        started_at = time.monotonic()
        stop_heartbeat = threading.Event()

        def _heartbeat() -> None:
            while not stop_heartbeat.wait(60):
                elapsed = int(time.monotonic() - started_at)
                message = (
                    f"{self.name} is still working on {self.phase.value}. "
                    f"elapsed={elapsed}s cycle={state.cycle_index}."
                )
                tools.memory.record_agent_event(
                    agent_name=self.name,
                    phase=self.phase,
                    status="heartbeat",
                    cycle_index=state.cycle_index,
                    content=message,
                    payload={"elapsed_seconds": elapsed},
                    print_to_terminal=False,
                )
                tools.memory.record_process(message, print_to_terminal=True)

        heartbeat_thread = threading.Thread(
            target=_heartbeat,
            name=f"{self.name}-heartbeat",
            daemon=True,
        )
        heartbeat_thread.start()
        try:
            output = runtime.run_structured(
                specialist_name=self.name,
                instructions=self.build_instructions(state),
                payload={
                    "research_brief": self._compact_research_brief(state),
                    "candidate_papers": paper_payloads,
                },
                session_id=f"{state.run_name}-{self.phase.value}-cycle-{state.cycle_index}",
                output_type=self.output_model,
                tools=[],
                runtime_timeout_seconds=self.runtime_timeout_seconds() or runtime.runtime_config.request_timeout_seconds,
            )
        finally:
            stop_heartbeat.set()
            heartbeat_thread.join(timeout=1)
        summary = self.apply_output(state, tools, output)
        self.record_diary(state, tools, summary)
        return summary

    def apply_output(self, state: ResearchState, tools: ResearchTools, output: LiteraturePhaseOutput) -> str:
        state.literature_notes = upsert_by_attr(state.literature_notes, output.literature_notes, "paper_id")
        state.method_taxonomy = output.method_taxonomy
        state.open_questions = dedupe_strings([*state.open_questions, *output.open_questions])
        state.next_actions = output.next_actions
        for note in output.literature_notes:
            tools.memory.record_literature(note)
        self.record_semantic_notes(state, tools, output.semantic_notes)
        return output.summary


class AcquisitionAgent(BaseResearchAgent):
    name = "AcquisitionAgent"
    phase = ResearchPhase.ACQUISITION
    output_model = AcquisitionPhaseOutput

    def allowed_tool_names(self) -> set[str] | None:
        return {
            "inspect_secret_status",
            "inspect_compute_environment",
            "search_github_repositories",
            "fetch_url_text",
            "discover_local_artifacts",
            "download_file",
            "clone_repository",
            "inspect_directory_tree",
            "read_text_file",
            "search_in_directory",
            "find_files",
            "detect_project_manifests",
            "bootstrap_python_environment",
            "ensure_python_environment",
            "parse_json_file",
        }

    def max_turns(self) -> int | None:
        return 8

    def runtime_timeout_seconds(self) -> int | None:
        return 300

    def build_instructions(self, state: ResearchState) -> str:
        return """
You are the acquisition and environment specialist.
Your job is to decide what external artifacts are required to execute real research on the supplied problem, then use tools to acquire and inspect them.

You must:
- inspect secrets and local compute first
- search for relevant repositories, benchmark code, data sources, checkpoints, and documentation
- aggressively discover and reuse already-present local artifacts before attempting new remote transfers
- clone repositories and bootstrap environments when needed
- identify at least one viable baseline program candidate grounded in acquired assets
- when recent experiment or setup failures expose a concrete missing prerequisite, prioritize fixing that blocker before collecting optional new assets
- verify that downloaded datasets, checkpoints, and repositories exist at the exact local paths needed by upcoming experiment plans
- use the managed downloader and artifact validator instead of ad hoc `curl` or `wget` repair loops
- use `discover_local_artifacts` to search sibling run directories, manual asset roots, and mirrored paths for exact required filenames or compatible assets
- do not loop on the same local path with repeated low-level checksum/validation attempts; rely on persisted artifact status and switch strategy when a candidate is marked corrupted or blocked
- for dataset artifacts, persist official checksum metadata when available and do not treat path existence as validation
- if the route focus says `environment_repair`, treat environment runtime health as the primary blocker: rebuild or repair the managed environment until torch imports cleanly and the preferred execution backend is actually usable
- treat manager route focus as policy, not prose. If the route focus says `local_discovery`, `mirror_resolution`, or `partial_salvage`, change strategy accordingly and do not repeat a known-bad transfer path unchanged.
- stop once you have secured one executable baseline route or one evidence-generating fallback route; do not keep exploring optional assets in the same phase
- if a tool reports `deferred_optional_dataset_download`, that means the workspace already has enough assets to move forward with an executable route. Treat that as a stop signal and return immediately.
- treat `discover_local_artifacts` as a filename lookup tool, not a semantic search engine. Use exact filenames or checkpoint names extracted from repo/config/registry evidence, not broad natural-language queries.
- once you have: (1) a benchmark repo, (2) a usable managed environment, and (3) either exact target datasets or a validated fallback dataset/checkpoint set sufficient for one executable baseline path, you must stop and return structured output immediately.

Rules:
- Do not assume repository URLs, dataset locations, or checkpoint links unless a tool confirmed them.
- Treat environment setup as part of the research loop.
- Return only artifacts and repositories that were actually verified with tools.
"""

    def build_payload(self, state: ResearchState, tools: ResearchTools) -> dict[str, Any]:
        compact_brief = {
            "title": state.research_brief.title[:180],
            "question": state.research_brief.question[:300],
            "objectives": [item[:140] for item in state.research_brief.objectives[:3]],
            "constraints": [item[:140] for item in state.research_brief.constraints[:5]],
            "domain_tags": state.research_brief.domain_tags[:6],
        }
        return {
            "research_brief": compact_brief,
            "literature_titles": [note.title for note in state.literature_notes[:12]],
            "open_questions": state.open_questions[:12],
            "existing_repositories": [repo.name for repo in state.repositories],
            "failure_summaries": state.failure_summaries[-12:],
            "blocker_registry": [
                {
                    "blocker_type": item.blocker_type,
                    "target_entity": item.target_entity,
                    "repeat_count": item.repeat_count,
                    "evidence_summary": item.evidence_summary[:220],
                    "recovery_strategies_tried": item.recovery_strategies_tried[-4:],
                    "terminality": item.terminality,
                }
                for item in state.blocker_registry[-8:]
            ],
            "route_history": [
                {
                    "route_id": item.route_id,
                    "material_change_summary": item.material_change_summary[:4],
                    "rationale": item.rationale[:220],
                    "allowed": item.allowed,
                }
                for item in state.route_history[-5:]
            ],
            "recent_experiment_records": [
                {
                    "plan_id": record.plan_id,
                    "status": record.status,
                    "observations": record.observations[:3],
                    "failure_modes": record.failure_modes[:3],
                    "job_kind": record.job_kind,
                }
                for record in state.experiment_records[-6:]
            ],
            "existing_experiment_plans": [
                {
                    "plan_id": plan.plan_id,
                    "status": plan.status,
                    "title": plan.title[:160],
                    "notes": plan.notes[:3],
                    "required_artifact_ids": plan.required_artifact_ids[:5],
                }
                for plan in state.experiment_plans[-8:]
            ],
            "active_route_id": state.active_route_id,
            "active_route_focus": state.active_route_focus,
            "hitl_events": [
                {
                    "response_type": item.response_type,
                    "blocker_type": item.blocker_type,
                    "target_entities": item.target_entities[:4],
                    "status": item.status,
                    "validation_summary": item.validation_summary[:3],
                }
                for item in state.hitl_events[-4:]
            ],
            "manual_asset_roots": state.manual_asset_roots,
            "skipped_target_entities": state.skipped_target_entities,
            "human_guidance_notes": state.human_guidance_notes[-12:],
            "recent_memory_notes": [
                {
                    "kind": item.kind,
                    "title": item.title,
                    "summary": item.summary[:220],
                    "phase": item.phase,
                }
                for item in state.memory_notes[-10:]
            ],
            "recent_evaluation_memos": [
                {
                    "title": item.title,
                    "summary": item.summary[:220],
                    "phase": item.phase,
                    "verdict": item.verdict,
                    "support_level": item.support_level,
                    "metrics": item.metrics,
                }
                for item in state.evaluation_memos[-6:]
            ],
            "capability_matrix": state.capability_matrix.model_dump(mode="python") if state.capability_matrix else None,
            "existing_artifacts": [
                {
                    "artifact_id": item.canonical_id or item.artifact_id,
                    "canonical_id": item.canonical_id,
                    "title": item.title,
                    "status": item.status,
                    "local_path": item.local_path,
                    "source_url": item.source_url,
                    "exact_target": bool((item.metadata or {}).get("exact_target")),
                }
                for item in state.external_artifacts[-40:]
            ],
            "work_directory": state.work_directory,
            "local_discovery_roots": [
                str(tools.shared_workspace_root),
                str(tools.workspace_family_root),
                *state.manual_asset_roots,
            ],
            "acquisition_completion_target": (
                "Finish acquisition as soon as one executable baseline route or one executable fallback route is available. "
                "Do not keep collecting optional variants after that point."
            ),
        }

    def apply_output(self, state: ResearchState, tools: ResearchTools, output: AcquisitionPhaseOutput) -> str:
        state.environment_snapshot = output.environment_snapshot
        if output.environment_records:
            state.environment_records = upsert_by_attr(state.environment_records, output.environment_records, "canonical_id")
        if output.capability_matrix is not None:
            state.capability_matrix = output.capability_matrix
            tools.memory.record_capability_matrix(output.capability_matrix)
        state.secret_status = upsert_by_attr(state.secret_status, output.secret_status, "env_var")
        state.external_artifacts = upsert_by_attr(state.external_artifacts, output.external_artifacts, "artifact_id")
        state.repositories = upsert_by_attr(state.repositories, output.repositories, "repo_id")
        state.program_candidates = upsert_by_attr(state.program_candidates, output.program_candidates, "program_id")
        state.selected_baseline_program_id = output.selected_baseline_program_id or state.selected_baseline_program_id
        state.acquisition_notes = dedupe_strings([*state.acquisition_notes, *output.acquisition_notes])
        state.next_actions = output.next_actions
        for secret in output.secret_status:
            tools.memory.record_secret_status(secret)
        for environment in output.environment_records:
            tools.memory.record_environment(environment)
        for artifact in output.external_artifacts:
            tools.memory.record_artifact(artifact)
        for repository in output.repositories:
            tools.memory.record_repository(repository)
        for program in output.program_candidates:
            tools.memory.register_program(program)
        self.record_semantic_notes(state, tools, output.semantic_notes + output.acquisition_notes)
        if output.summary.strip():
            return output.summary
        return (
            "Acquisition updated the workspace state. "
            f"repositories={len(state.repositories)} artifacts={len(state.external_artifacts)} "
            f"environments={len(state.environment_records)} baseline={state.selected_baseline_program_id or 'unset'}."
        )
