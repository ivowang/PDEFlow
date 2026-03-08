from __future__ import annotations

from collections import deque
import os
from pathlib import Path
import shlex
import subprocess
import threading
import time

from common import ensure_dir, now_utc, short_hash


class CommandExecutionMixin:
    def run_command(
        self,
        command: str,
        cwd: str | Path | None = None,
        env_overrides: dict[str, str] | None = None,
        gpu_ids: list[int] | None = None,
        log_path: str | None = None,
        allow_failure: bool = False,
        emit_progress: bool = True,
    ) -> dict[str, object]:
        if not self.config.execution.allow_shell_commands:
            raise RuntimeError("Shell command execution is disabled in the current config.")
        working_directory = self._resolve_path(str(cwd), default_root=self.repo_root) if cwd else self.repo_root
        log_file = (
            self._resolve_path(log_path, default_root=self.memory.logs_dir)
            if log_path
            else self.memory.logs_dir / f"cmd-{short_hash(command, str(working_directory), now_utc())}.log"
        )
        ensure_dir(log_file.parent)
        env = os.environ.copy()
        if env_overrides:
            env.update(env_overrides)
        if gpu_ids:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in gpu_ids)
        compact = command if len(command) <= 140 else command[:137] + "..."
        if emit_progress:
            gpu_text = f" on GPUs {gpu_ids}" if gpu_ids else ""
            self.memory.record_process(
                f"Running command{gpu_text} in {working_directory}: {compact}"
            )
        popen_args = ["bash", "-lc", command] if self._command_requires_shell(command) else shlex.split(command)
        process = subprocess.Popen(
            popen_args,
            cwd=str(working_directory),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        tail_lines: deque[str] = deque(maxlen=200)
        heartbeat_stop = threading.Event()
        started_at = time.monotonic()

        def emit_heartbeat() -> None:
            interval_seconds = 60
            while not heartbeat_stop.wait(interval_seconds):
                if process.poll() is not None:
                    return
                elapsed = int(time.monotonic() - started_at)
                last_line = tail_lines[-1].strip() if tail_lines else ""
                log_size = log_file.stat().st_size if log_file.exists() else 0
                suffix = f" last_output={last_line[:160]}" if last_line else ""
                self.memory.record_process(
                    "Command heartbeat: "
                    f"elapsed={elapsed}s cwd={working_directory} log={log_file} bytes={log_size}. "
                    f"Command: {compact}.{suffix}"
                )

        heartbeat_thread = (
            threading.Thread(target=emit_heartbeat, name="pdeflow-command-heartbeat", daemon=True)
            if emit_progress
            else None
        )
        if heartbeat_thread is not None:
            heartbeat_thread.start()
        with log_file.open("w", encoding="utf-8") as handle:
            assert process.stdout is not None
            for line in process.stdout:
                handle.write(line)
                handle.flush()
                tail_lines.append(line.rstrip("\n"))
        return_code = process.wait()
        heartbeat_stop.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=1)
        stdout_tail = "\n".join(tail_lines)
        result = {
            "command": command,
            "cwd": str(working_directory),
            "returncode": return_code,
            "stdout_tail": stdout_tail,
            "stderr_tail": "" if return_code == 0 else stdout_tail,
            "log_path": str(log_file),
            "emit_progress": emit_progress,
        }
        self._record_tool_event("run_command", result)
        if return_code != 0 and not allow_failure:
            raise RuntimeError(stdout_tail or f"Command failed: {command}")
        return result
