from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from config import SystemConfig
from memory import ResearchMemory
from common import ensure_dir, now_utc


class ToolContext:
    """Shared tool runtime context, path policy, and progress/event logging."""

    def __init__(self, config: SystemConfig, memory: ResearchMemory, repo_root: Path):
        self.config = config
        self.memory = memory
        self.repo_root = repo_root.resolve()
        shared_root = Path(config.execution.workspace_root)
        if shared_root.is_absolute():
            resolved_shared_root = shared_root.resolve()
        else:
            resolved_shared_root = (memory.root / shared_root).resolve()
        if not str(resolved_shared_root).startswith(str(memory.root.resolve())):
            raise ValueError(
                "execution.workspace_root must resolve inside execution.work_directory."
            )
        self.shared_workspace_root = ensure_dir(resolved_shared_root)
        self.run_workspace_root = ensure_dir(memory.root / "workspaces")
        self.managed_env_root = ensure_dir(memory.root / "envs")
        self.managed_python_root = ensure_dir(memory.root / "pythons")
        self.quarantine_root = ensure_dir(memory.root / "quarantine")

    def _progress_message_for_tool_event(self, tool_name: str, payload: dict[str, Any]) -> str | None:
        if tool_name == "inspect_secret_status":
            return f"Inspected secret status. tracked={payload.get('count', 0)}."
        if tool_name == "inspect_compute_environment":
            return (
                "Inspected compute environment. "
                f"selected_gpus={payload.get('selected_gpu_ids', [])} uv_available={payload.get('uv_available', False)}."
            )
        if tool_name == "search_arxiv_papers":
            return f"Tool search_arxiv_papers finished. query='{payload.get('query', '')}' results={payload.get('count', 0)}."
        if tool_name == "search_github_repositories":
            return f"Tool search_github_repositories finished. query='{payload.get('query', '')}' results={payload.get('count', 0)}."
        if tool_name == "fetch_url_text":
            return f"Fetched remote text from {payload.get('url', '')}."
        if tool_name == "download_file":
            status = payload.get("validation_status")
            if status:
                return f"Downloaded file to {payload.get('path', '')}. validation_status={status}."
            return f"Downloaded file to {payload.get('path', '')}."
        if tool_name == "validate_artifact":
            return (
                f"Validated artifact {payload.get('artifact_id', '')}: "
                f"status={payload.get('status', 'unknown')} ready={payload.get('ready_for_training', False)}."
            )
        if tool_name == "probe_capability_matrix":
            return (
                "Capability probe finished. "
                f"baseline_ready={payload.get('baseline_ready_to_launch', False)} "
                f"target_dataset_ready={payload.get('target_dataset_ready', False)}."
            )
        if tool_name == "preflight_experiment_plan":
            return (
                f"Preflight finished for {payload.get('plan_id', '')}: "
                f"passed={payload.get('passed', False)}."
            )
        if tool_name == "extract_pdf_text":
            return f"Extracted PDF text from {payload.get('path', '')}."
        if tool_name == "clone_repository":
            if payload.get("status") == "failed":
                return (
                    "Repository clone failed for "
                    f"{payload.get('repo_url', '')}: {payload.get('error', 'unknown error')}."
                )
            return f"Repository available at {payload.get('path', '')} from {payload.get('repo_url', '')}."
        if tool_name == "detect_project_manifests":
            return f"Detected project manifests in {payload.get('path', '')}. count={len(payload.get('manifests', []))}."
        if tool_name == "bootstrap_python_environment":
            return f"Environment bootstrap finished for {payload.get('project_path', '')} with status={payload.get('status', 'unknown')}."
        if tool_name == "ensure_python_environment":
            return (
                "Managed environment ready for "
                f"{payload.get('project_path', '')}: status={payload.get('status', 'unknown')} "
                f"path={payload.get('environment_path', '')}."
            )
        if tool_name == "inspect_python_environment":
            return (
                "Inspected managed environment "
                f"{payload.get('environment_path', '')}: "
                f"python_ok={payload.get('python_available', False)} pip_ok={payload.get('pip_available', False)}."
            )
        if tool_name == "run_in_environment":
            command = str(payload.get("command", "")).strip()
            compact = command if len(command) <= 140 else command[:137] + "..."
            return (
                f"Environment command completed with returncode={payload.get('returncode', 'unknown')}: "
                f"{compact}"
            )
        if tool_name == "copy_tree":
            return f"Created child workspace at {payload.get('destination', '')}."
        if tool_name == "write_patch_file":
            return f"Wrote patch file {payload.get('path', '')}."
        if tool_name == "apply_patch_file":
            return f"Applied patch in {payload.get('cwd', '')} with returncode={payload.get('returncode', 'unknown')}."
        if tool_name == "parse_metrics_file":
            return f"Parsed metrics from {payload.get('path', '')}."
        if tool_name == "write_report":
            return f"Wrote report {payload.get('path', '')}."
        if tool_name == "run_command":
            if payload.get("emit_progress") is False:
                return None
            command = str(payload.get("command", "")).strip()
            compact = command if len(command) <= 140 else command[:137] + "..."
            return f"Command completed with returncode={payload.get('returncode', 'unknown')}: {compact}"
        return None

    def _record_tool_event(self, tool_name: str, payload: dict[str, Any]) -> None:
        self.memory.record_tool_event({"tool": tool_name, "timestamp": now_utc(), **payload})
        progress_message = self._progress_message_for_tool_event(tool_name, payload)
        if progress_message:
            self.memory.record_process(progress_message)

    def _allowed_roots(self) -> list[Path]:
        return [
            self.repo_root,
            self.shared_workspace_root,
            self.memory.root.resolve(),
            self.run_workspace_root.resolve(),
        ]

    def _command_requires_shell(self, command: str) -> bool:
        stripped = command.strip()
        if not stripped:
            return False
        shell_tokens = ("&&", "||", ";", "|", ">", "<", "$(", "`", "\n")
        if any(token in stripped for token in shell_tokens):
            return True
        first_token = stripped.split()[0]
        return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]*=.*", first_token))

    def _resolve_path(self, path_str: str, default_root: Path | None = None) -> Path:
        raw_path = Path(path_str)
        candidate = raw_path if raw_path.is_absolute() else (default_root or self.repo_root) / raw_path
        resolved = candidate.resolve()
        allowed = any(str(resolved).startswith(str(root)) for root in self._allowed_roots())
        if not allowed:
            raise ValueError(f"Path is outside managed roots: {resolved}")
        return resolved
