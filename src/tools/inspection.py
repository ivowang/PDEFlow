from __future__ import annotations

import os
import sys

from state import EnvironmentSnapshot, SecretStatus


class SystemInspectionMixin:
    def inspect_secret_status(self) -> list[SecretStatus]:
        statuses = [
            SecretStatus(
                env_var=spec.env_var,
                purpose=spec.purpose,
                required=spec.required,
                is_set=bool(os.getenv(spec.env_var)),
                resolution_hint=f"Export {spec.env_var} in the shell before starting the autonomous workflow.",
            )
            for spec in self.config.secrets
        ]
        for status in statuses:
            self.memory.record_secret_status(status)
        self._record_tool_event("inspect_secret_status", {"count": len(statuses)})
        return statuses

    def inspect_compute_environment(self) -> EnvironmentSnapshot:
        python_executable = sys.executable
        python_version = sys.version.split()[0]
        uv_result = self.run_command("uv --version", allow_failure=True, emit_progress=False)
        gpu_result = self.run_command(
            "nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader",
            allow_failure=True,
            emit_progress=False,
        )
        available_gpu_ids: list[int] = []
        gpu_descriptions: dict[str, str] = {}
        notes: list[str] = []
        if gpu_result["returncode"] == 0 and gpu_result["stdout_tail"]:
            for line in gpu_result["stdout_tail"].splitlines():
                parts = [part.strip() for part in line.split(",")]
                if len(parts) >= 3 and parts[0].isdigit():
                    gpu_id = int(parts[0])
                    available_gpu_ids.append(gpu_id)
                    gpu_descriptions[str(gpu_id)] = f"{parts[1]} ({parts[2]})"
            notes.append("GPU inventory detected via nvidia-smi.")
        else:
            notes.append("GPU probing failed; falling back to configured inventory hints.")
            gpu_descriptions = dict(self.config.resource_policy.gpu_inventory_hint)
            available_gpu_ids = [int(item) for item in gpu_descriptions if item.isdigit()]
            if gpu_result["stderr_tail"]:
                notes.append(gpu_result["stderr_tail"])
        selected = [gpu_id for gpu_id in self.config.resource_policy.preferred_gpu_ids if gpu_id in available_gpu_ids]
        if not selected:
            selected = available_gpu_ids[: self.config.resource_policy.max_parallel_experiments]
        snapshot = EnvironmentSnapshot(
            python_executable=python_executable,
            python_version=python_version,
            uv_available=uv_result["returncode"] == 0,
            uv_version=uv_result["stdout_tail"] or None,
            available_gpu_ids=available_gpu_ids,
            selected_gpu_ids=selected,
            gpu_descriptions=gpu_descriptions,
            notes=notes,
        )
        self._record_tool_event(
            "inspect_compute_environment",
            {
                "python_version": snapshot.python_version,
                "uv_available": snapshot.uv_available,
                "selected_gpu_ids": snapshot.selected_gpu_ids,
            },
        )
        return snapshot
