from __future__ import annotations

import re
import shlex
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
        self.current_phase: str | None = None
        self.current_cycle_index: int | None = None
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
        self.workspace_family_root = memory.root.parent.resolve()
        self.run_workspace_root = ensure_dir(memory.root / "workspaces")
        self.managed_env_root = ensure_dir(memory.root / "envs")
        self.managed_python_root = ensure_dir(memory.root / "pythons")
        self.quarantine_root = ensure_dir(memory.root / "quarantine")
        self.uv_cache_root = ensure_dir(memory.root / ".uv-cache")

    def set_runtime_context(self, *, phase: str | None = None, cycle_index: int | None = None) -> None:
        self.current_phase = phase
        self.current_cycle_index = cycle_index

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
            strategy = payload.get("strategy_id")
            attempt = payload.get("attempt_signature")
            if status:
                return (
                    f"Downloaded file to {payload.get('path', '')}. validation_status={status}. "
                    f"strategy_id={strategy or 'unknown'} attempt_signature={attempt or 'unknown'}."
                )
            return f"Downloaded file to {payload.get('path', '')}."
        if tool_name == "download_progress":
            path = payload.get("path", "")
            size_bytes = int(payload.get("bytes_downloaded", 0) or 0)
            elapsed = float(payload.get("elapsed_time", 0.0) or 0.0)
            throughput = payload.get("average_throughput")
            throughput_text = (
                f"{float(throughput):.1f} B/s" if isinstance(throughput, (int, float)) and throughput is not None else "unknown"
            )
            return (
                f"Download heartbeat for {path}: bytes_downloaded={size_bytes} "
                f"elapsed={elapsed:.1f}s avg_throughput={throughput_text}."
            )
        if tool_name == "discover_local_artifacts":
            return (
                f"Local artifact discovery finished. query='{payload.get('query', '')}' "
                f"results={payload.get('count', 0)} roots={payload.get('roots', [])}."
            )
        if tool_name == "validate_artifact":
            return (
                f"Validated artifact {payload.get('artifact_id', '')}: "
                f"status={payload.get('status', 'unknown')} ready={payload.get('ready_for_training', False)}."
            )
        if tool_name == "probe_capability_matrix":
            return (
                "Capability probe finished. "
                f"repo_ready={payload.get('repo_ready', False)} "
                f"env_ready={payload.get('env_ready', False)} "
                f"baseline_ready={payload.get('baseline_ready_to_launch', False)} "
                f"cuda_ready={payload.get('gpu_runtime_ready', False)} "
                f"target_dataset_ready={payload.get('target_dataset_ready', False)}."
            )
        if tool_name == "preflight_experiment_plan":
            return (
                f"Preflight finished for {payload.get('plan_id', '')}: "
                f"passed={payload.get('passed', False)} "
                f"recommended_route={payload.get('recommended_route', 'none')}."
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

    def _allowed_roots(self, include_workspace_family: bool = True) -> list[Path]:
        roots = [
            self.repo_root,
            self.shared_workspace_root,
            self.memory.root.resolve(),
            self.run_workspace_root.resolve(),
            self.managed_env_root.resolve(),
            self.managed_python_root.resolve(),
            self.quarantine_root.resolve(),
        ]
        if include_workspace_family:
            roots.append(self.workspace_family_root)
        deduped: list[Path] = []
        for root in roots:
            resolved = root.resolve()
            if resolved not in deduped:
                deduped.append(resolved)
        return deduped

    def _command_requires_shell(self, command: str) -> bool:
        stripped = command.strip()
        if not stripped:
            return False
        shell_tokens = ("&&", "||", ";", "|", ">", "<", "$(", "`", "\n")
        if any(token in stripped for token in shell_tokens):
            return True
        first_token = stripped.split()[0]
        return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]*=.*", first_token))

    def _uv_command_prefix(self, **extra_env: str) -> str:
        env_parts = [
            "env",
            f"UV_CACHE_DIR={shlex.quote(str(self.uv_cache_root))}",
            "UV_LINK_MODE=copy",
        ]
        for key, value in extra_env.items():
            env_parts.append(f"{key}={shlex.quote(str(value))}")
        return " ".join(env_parts)

    def _is_within_allowed_roots(
        self,
        path: Path,
        *,
        allow_workspace_family: bool = True,
    ) -> bool:
        resolved = path.resolve()
        return any(
            str(resolved).startswith(str(root))
            for root in self._allowed_roots(include_workspace_family=allow_workspace_family)
        )

    def _is_within_managed_write_roots(self, path: Path) -> bool:
        return self._is_within_allowed_roots(path, allow_workspace_family=False)

    def _resolve_path(
        self,
        path_str: str,
        default_root: Path | None = None,
        *,
        allow_workspace_family: bool = True,
    ) -> Path:
        raw_path = Path(path_str)
        candidate = raw_path if raw_path.is_absolute() else (default_root or self.repo_root) / raw_path
        resolved = candidate.resolve()
        if not self._is_within_allowed_roots(resolved, allow_workspace_family=allow_workspace_family):
            raise ValueError(f"Path is outside managed roots: {resolved}")
        return resolved

    def _resolve_managed_write_path(self, path_str: str, default_root: Path | None = None) -> Path:
        return self._resolve_path(
            path_str,
            default_root=default_root,
            allow_workspace_family=False,
        )
