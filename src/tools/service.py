from __future__ import annotations

import json
from typing import Any

from common import load_openai_agents_sdk
from config import SystemConfig
from memory import ResearchMemory
from .base import ToolContext
from .environment import ManagedEnvironmentMixin
from .execution import CommandExecutionMixin
from .inspection import SystemInspectionMixin
from .python_runtime import PythonRuntimeDiscoveryMixin
from .reporting import ReportingToolsMixin
from .retrieval import RetrievalToolsMixin
from .workspace import WorkspaceToolsMixin

try:
    function_tool = getattr(load_openai_agents_sdk(), "function_tool", None)
except ImportError:  # pragma: no cover
    function_tool = None


class ResearchTools(
    ToolContext,
    SystemInspectionMixin,
    RetrievalToolsMixin,
    WorkspaceToolsMixin,
    PythonRuntimeDiscoveryMixin,
    ManagedEnvironmentMixin,
    CommandExecutionMixin,
    ReportingToolsMixin,
):
    """Executable tool surface exposed to specialist agents."""

    def __init__(self, config: SystemConfig, memory: ResearchMemory, repo_root):
        super().__init__(config=config, memory=memory, repo_root=repo_root)

    def build_function_tools(self) -> list[Any]:
        if function_tool is None:
            return []

        @function_tool
        def inspect_secret_status() -> list[dict[str, Any]]:
            """Inspect configured secrets and whether they are available in the current environment."""
            return [item.model_dump(mode="python") for item in self.inspect_secret_status()]

        @function_tool
        def inspect_compute_environment() -> dict[str, Any]:
            """Inspect Python, uv, and GPU availability on the current machine."""
            return self.inspect_compute_environment().model_dump(mode="python")

        @function_tool
        def search_arxiv_papers(query: str, max_results: int = 5) -> list[dict[str, Any]]:
            """Search arXiv for papers relevant to the current research question."""
            return self.search_arxiv_papers(query, max_results=max_results)

        @function_tool
        def search_github_repositories(query: str, max_results: int = 5) -> list[dict[str, Any]]:
            """Search GitHub repositories relevant to the current research question."""
            return self.search_github_repositories(query, max_results=max_results)

        @function_tool
        def fetch_url_text(url: str, max_chars: int = 20000) -> dict[str, Any]:
            """Fetch text content from a remote URL such as a README, docs page, or dataset page."""
            return self.fetch_url_text(url, max_chars=max_chars)

        @function_tool
        def download_file(url: str, target_path: str) -> dict[str, Any]:
            """Download a remote file into the managed workspace."""
            return self.download_file(url, target_path=target_path)

        @function_tool
        def extract_pdf_text(pdf_path: str, max_pages: int = 6) -> dict[str, Any]:
            """Extract text from a locally downloaded PDF to support literature analysis."""
            return self.extract_pdf_text(pdf_path, max_pages=max_pages)

        @function_tool
        def clone_repository(repo_url: str, destination_name: str | None = None) -> dict[str, Any]:
            """Clone a Git repository into the managed workspace."""
            return self.clone_repository(repo_url, destination_name=destination_name)

        @function_tool
        def inspect_directory_tree(path: str, max_depth: int = 2, max_entries: int = 200) -> dict[str, Any]:
            """Inspect a local directory tree after cloning or downloading assets."""
            return self.inspect_directory_tree(path, max_depth=max_depth, max_entries=max_entries)

        @function_tool
        def read_text_file(path: str, max_chars: int = 20000) -> dict[str, Any]:
            """Read a local text file such as a README, config, script, or log."""
            return self.read_text_file(path, max_chars=max_chars)

        @function_tool
        def search_in_directory(path: str, pattern: str, glob: str | None = None, max_hits: int = 100) -> dict[str, Any]:
            """Search local files for patterns such as train entrypoints, checkpoints, or metrics."""
            return self.search_in_directory(path, pattern=pattern, glob=glob, max_hits=max_hits)

        @function_tool
        def find_files(path: str, pattern: str) -> dict[str, Any]:
            """Find local files matching a glob pattern."""
            return self.find_files(path, pattern=pattern)

        @function_tool
        def detect_project_manifests(path: str) -> dict[str, Any]:
            """Detect Python project manifests and likely training entrypoints in a repository."""
            return self.detect_project_manifests(path)

        @function_tool
        def bootstrap_python_environment(project_path: str) -> dict[str, Any]:
            """Bootstrap a Python environment for a cloned repository using uv where possible."""
            return self.bootstrap_python_environment(project_path)

        @function_tool
        def ensure_python_environment(
            project_path: str,
            environment_name: str | None = None,
            python_spec: str | None = None,
            dependency_strategy: str = "auto",
            editable_install: bool = True,
        ) -> dict[str, Any]:
            """Create or repair a managed uv environment for a project inside the current research work directory."""
            return self.ensure_python_environment(
                project_path=project_path,
                environment_name=environment_name,
                python_spec=python_spec,
                dependency_strategy=dependency_strategy,
                editable_install=editable_install,
            )

        @function_tool
        def inspect_python_environment(environment_path: str, modules: list[str] | None = None) -> dict[str, Any]:
            """Inspect a managed Python environment and optionally probe selected imports."""
            return self.inspect_python_environment(environment_path, modules=modules)

        @function_tool
        def copy_tree(source_path: str, destination_path: str) -> dict[str, Any]:
            """Copy a repository or workspace tree to create a child program candidate."""
            return self.copy_tree(source_path, destination_path)

        @function_tool
        def write_text_file(path: str, content: str) -> dict[str, Any]:
            """Write or overwrite a text file inside the managed workspace."""
            return self.write_text_file(path, content)

        @function_tool
        def write_json_file(path: str, payload_json: str) -> dict[str, Any]:
            """Write a JSON file inside the managed workspace."""
            return self.write_json_file(path, json.loads(payload_json))

        @function_tool
        def write_patch_file(path: str, patch_text: str) -> dict[str, Any]:
            """Write a unified diff patch file inside the managed workspace."""
            return self.write_patch_file(path, patch_text)

        @function_tool
        def apply_patch_file(repo_path: str, patch_path: str) -> dict[str, Any]:
            """Apply a git patch file to a repository workspace."""
            return self.apply_patch_file(repo_path, patch_path)

        @function_tool
        def run_command(
            command: str,
            cwd: str | None = None,
            gpu_ids: list[int] | None = None,
            log_path: str | None = None,
            allow_failure: bool = True,
        ) -> dict[str, Any]:
            """Run a shell command in a managed directory and capture a persistent log."""
            return self.run_command(
                command,
                cwd=cwd,
                gpu_ids=gpu_ids,
                log_path=log_path,
                allow_failure=allow_failure,
            )

        @function_tool
        def run_in_environment(
            environment_path: str,
            command: str,
            cwd: str | None = None,
            gpu_ids: list[int] | None = None,
            log_path: str | None = None,
            allow_failure: bool = True,
        ) -> dict[str, Any]:
            """Run a command inside a managed Python environment created by the research system."""
            return self.run_in_environment(
                environment_path=environment_path,
                command=command,
                cwd=cwd,
                gpu_ids=gpu_ids,
                log_path=log_path,
                allow_failure=allow_failure,
            )

        @function_tool
        def parse_json_file(path: str) -> dict[str, Any]:
            """Parse a JSON file produced by an experiment or external repository."""
            return self.parse_json_file(path)

        @function_tool
        def parse_metrics_file(path: str) -> dict[str, Any]:
            """Parse a metrics JSON or a text log with key=value style metrics."""
            return self.parse_metrics_file(path)

        @function_tool
        def write_report(filename: str, content: str) -> dict[str, Any]:
            """Write a markdown report to the current run report directory."""
            return {"path": str(self.write_report(filename, content))}

        return [
            inspect_secret_status,
            inspect_compute_environment,
            search_arxiv_papers,
            search_github_repositories,
            fetch_url_text,
            download_file,
            extract_pdf_text,
            clone_repository,
            inspect_directory_tree,
            read_text_file,
            search_in_directory,
            find_files,
            detect_project_manifests,
            bootstrap_python_environment,
            ensure_python_environment,
            inspect_python_environment,
            copy_tree,
            write_text_file,
            write_json_file,
            write_patch_file,
            apply_patch_file,
            run_command,
            run_in_environment,
            parse_json_file,
            parse_metrics_file,
            write_report,
        ]
