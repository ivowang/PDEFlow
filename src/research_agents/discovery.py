from __future__ import annotations

from typing import Any

from state import AcquisitionPhaseOutput, LiteraturePhaseOutput, ResearchPhase, ResearchState
from tools import ResearchTools
from common import dedupe_strings, upsert_by_attr
from .base import BaseResearchAgent


class LiteratureAgent(BaseResearchAgent):
    name = "LiteratureAgent"
    phase = ResearchPhase.LITERATURE_REVIEW
    output_model = LiteraturePhaseOutput

    def allowed_tool_names(self) -> set[str] | None:
        return {
            "search_arxiv_papers",
            "fetch_url_text",
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
- Extract limitations and research opportunities, not generic summaries.
- Return only structured output matching the schema.
"""

    def build_payload(self, state: ResearchState, tools: ResearchTools) -> dict[str, Any]:
        return {
            "research_brief": state.research_brief.model_dump(mode="python"),
            "already_known_papers": [note.title for note in state.literature_notes],
        }

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
            "search_arxiv_papers",
            "search_github_repositories",
            "fetch_url_text",
            "download_file",
            "compute_file_checksum",
            "validate_artifact",
            "clone_repository",
            "inspect_directory_tree",
            "read_text_file",
            "search_in_directory",
            "find_files",
            "detect_project_manifests",
            "bootstrap_python_environment",
            "ensure_python_environment",
            "inspect_python_environment",
            "probe_capability_matrix",
            "parse_json_file",
        }

    def build_instructions(self, state: ResearchState) -> str:
        return """
You are the acquisition and environment specialist.
Your job is to decide what external artifacts are required to execute real research on the supplied problem, then use tools to acquire and inspect them.

You must:
- inspect secrets and local compute first
- search for relevant repositories, benchmark code, data sources, checkpoints, and documentation
- clone repositories and bootstrap environments when needed
- identify at least one viable baseline program candidate grounded in acquired assets
- when recent experiment or setup failures expose a concrete missing prerequisite, prioritize fixing that blocker before collecting optional new assets
- verify that downloaded datasets, checkpoints, and repositories exist at the exact local paths needed by upcoming experiment plans
- use the managed downloader and artifact validator instead of ad hoc `curl` or `wget` repair loops
- for dataset artifacts, persist official checksum metadata when available and do not treat path existence as validation
- treat manager route focus as policy, not prose. If the route focus says `local_discovery`, `mirror_resolution`, or `partial_salvage`, change strategy accordingly and do not repeat a known-bad transfer path unchanged.

Rules:
- Do not assume repository URLs, dataset locations, or checkpoint links unless a tool confirmed them.
- Treat environment setup as part of the research loop.
- Return only artifacts and repositories that were actually verified with tools.
"""

    def build_payload(self, state: ResearchState, tools: ResearchTools) -> dict[str, Any]:
        return {
            "research_brief": state.research_brief.model_dump(mode="python"),
            "literature_titles": [note.title for note in state.literature_notes[:12]],
            "open_questions": state.open_questions[:12],
            "existing_repositories": [repo.name for repo in state.repositories],
            "failure_summaries": state.failure_summaries[-12:],
            "blocker_registry": [item.model_dump(mode="python") for item in state.blocker_registry[-12:]],
            "route_history": [item.model_dump(mode="python") for item in state.route_history[-6:]],
            "recent_experiment_records": [record.model_dump(mode="python") for record in state.experiment_records[-8:]],
            "existing_experiment_plans": [plan.model_dump(mode="python") for plan in state.experiment_plans[-12:]],
            "active_route_id": state.active_route_id,
            "active_route_focus": state.active_route_focus,
            "hitl_events": [item.model_dump(mode="python") for item in state.hitl_events[-6:]],
            "manual_asset_roots": state.manual_asset_roots,
            "skipped_target_entities": state.skipped_target_entities,
            "human_guidance_notes": state.human_guidance_notes[-12:],
            "work_directory": state.work_directory,
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
