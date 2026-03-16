from __future__ import annotations

import json
from pathlib import Path

from state import ArtifactStatus, ArtifactRecord, CapabilityMatrix


class CapabilityProbeMixin:
    _ARTIFACT_STATUS_PRIORITY = {
        ArtifactStatus.READY_FOR_TRAINING.value: 7,
        ArtifactStatus.FORMAT_VERIFIED.value: 6,
        ArtifactStatus.CHECKSUM_VERIFIED.value: 5,
        ArtifactStatus.DOWNLOADED.value: 4,
        ArtifactStatus.VERIFIED_LOCAL.value: 4,
        ArtifactStatus.VERIFIED_REMOTE.value: 4,
        ArtifactStatus.BLOCKED.value: 3,
        ArtifactStatus.DOWNLOAD_FAILED.value: 2,
        ArtifactStatus.CORRUPTED.value: 1,
        ArtifactStatus.QUARANTINED.value: 0,
    }

    def _artifact_probe_score(self, artifact: ArtifactRecord) -> tuple[int, int, int, int]:
        validation = artifact.validation
        size_bytes = int(
            (validation.size_bytes if validation else 0)
            or (artifact.download_metadata.file_size if artifact.download_metadata else 0)
            or 0
        )
        return (
            self._ARTIFACT_STATUS_PRIORITY.get(artifact.status, -1),
            int(bool(validation and validation.ready_for_training)),
            int(bool(artifact.local_path)),
            size_bytes,
        )

    def _dedupe_artifacts_for_probe(self, artifacts: list[ArtifactRecord]) -> list[ArtifactRecord]:
        selected: dict[str, ArtifactRecord] = {}
        for artifact in artifacts:
            key = artifact.canonical_id or artifact.artifact_id
            current = selected.get(key)
            if current is None or self._artifact_probe_score(artifact) > self._artifact_probe_score(current):
                selected[key] = artifact
        return list(selected.values())

    def _environment_candidate_score(
        self,
        path: Path,
        *,
        explicit_path: str | None = None,
        repository_paths: list[str] | None = None,
    ) -> tuple[int, int, int, int, int, int]:
        repository_paths = repository_paths or []
        resolved = path.resolve()
        explicit_bonus = 0
        if explicit_path:
            try:
                explicit_bonus = int(resolved == self._resolve_path(explicit_path, default_root=self.managed_env_root).resolve())
            except Exception:
                explicit_bonus = 0
        repo_tokens: set[str] = set()
        for repo_path in repository_paths:
            repo_name = Path(repo_path).name.strip().lower()
            if repo_name:
                repo_tokens.add(repo_name)
                repo_tokens.add(repo_name.replace("_", "-"))
        env_name = path.name.strip().lower()
        repo_match = int(any(token and token in env_name for token in repo_tokens))
        auxiliary_name = int(any(token in env_name for token in {"validator", "probe", "check", "inspect"}))
        python_exists = int((path / "bin" / "python").exists())
        return (
            python_exists,
            repo_match,
            int(not auxiliary_name),
            explicit_bonus,
            int(path.exists()),
            len(env_name),
        )

    def _discover_environment_path(
        self,
        explicit_path: str | None = None,
        repository_paths: list[str] | None = None,
    ) -> str | None:
        candidates: list[Path] = []
        if explicit_path:
            candidates.append(self._resolve_path(explicit_path, default_root=self.managed_env_root))
        try:
            candidates.extend(path for path in self.managed_env_root.iterdir() if path.is_dir())
        except FileNotFoundError:
            pass
        if not candidates:
            return None
        deduped: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            resolved = str(candidate.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            deduped.append(candidate)
        selected = max(
            deduped,
            key=lambda item: self._environment_candidate_score(
                item,
                explicit_path=explicit_path,
                repository_paths=repository_paths,
            ),
        )
        return str(selected)

    def _bool_module(self, modules: dict[str, bool], name: str) -> bool:
        return bool(modules.get(name) or modules.get(name.replace("-", "_")))

    def _infer_exact_target(self, artifact: ArtifactRecord) -> bool:
        metadata = artifact.metadata or {}
        if metadata.get("human_skip"):
            return False
        if metadata.get("exact_target") is not None:
            return bool(metadata.get("exact_target"))
        if artifact.artifact_type != "dataset":
            return False
        semantic = artifact.semantic_spec
        if semantic and semantic.benchmark == "PDEBench" and semantic.equation in {"Burgers", "ReactionDiffusion"}:
            return True
        if metadata.get("official_checksum") or metadata.get("official_md5"):
            return True
        if metadata.get("split") == "train" and (artifact.artifact_id.startswith("dataset-") or metadata.get("expected_filename")):
            return True
        return False

    def _artifact_ready(self, artifact: ArtifactRecord) -> bool:
        return artifact.status == ArtifactStatus.READY_FOR_TRAINING.value

    def _artifact_pending_preparation(self, artifact: ArtifactRecord) -> bool:
        return artifact.status in {
            ArtifactStatus.DOWNLOADED.value,
            ArtifactStatus.CHECKSUM_VERIFIED.value,
            ArtifactStatus.FORMAT_VERIFIED.value,
            ArtifactStatus.VERIFIED_REMOTE.value,
        }

    def _probe_torch_runtime(self, env_path: str, gpu_ids: list[int] | None = None) -> dict[str, object]:
        probe = self.run_in_environment(
            env_path,
            (
                "python - <<'PY'\n"
                "import json\n"
                "payload = {\n"
                "    'torch_import_ok': False,\n"
                "    'torch_version': None,\n"
                "    'torch_cuda_version': None,\n"
                "    'cuda_available': False,\n"
                "    'torchvision_available': False,\n"
                "    'torchvision_version': None,\n"
                "    'error': None,\n"
                "}\n"
                "try:\n"
                "    import torch\n"
                "    payload['torch_import_ok'] = True\n"
                "    payload['torch_version'] = getattr(torch, '__version__', None)\n"
                "    payload['torch_cuda_version'] = getattr(getattr(torch, 'version', None), 'cuda', None)\n"
                "    payload['cuda_available'] = bool(torch.cuda.is_available())\n"
                "except Exception as exc:\n"
                "    payload['error'] = f'{type(exc).__name__}: {exc}'\n"
                "try:\n"
                "    import torchvision\n"
                "    payload['torchvision_available'] = True\n"
                "    payload['torchvision_version'] = getattr(torchvision, '__version__', None)\n"
                "except Exception:\n"
                "    pass\n"
                "print(json.dumps(payload))\n"
                "PY"
            ),
            allow_failure=True,
            emit_progress=False,
            stall_timeout_seconds=90,
            gpu_ids=gpu_ids,
        )
        payload: dict[str, object] = {
            "torch_import_ok": False,
            "torch_version": None,
            "torch_cuda_version": None,
            "cuda_available": False,
            "torchvision_available": False,
            "torchvision_version": None,
            "error": probe["stderr_tail"] or probe["stdout_tail"] or None,
        }
        if probe["returncode"] == 0 and probe["stdout_tail"]:
            try:
                parsed = json.loads(probe["stdout_tail"].splitlines()[-1].strip())
                payload.update(parsed)
                if not payload.get("error") and probe["stdout_tail"].strip():
                    leading = "\n".join(probe["stdout_tail"].splitlines()[:-1]).strip()
                    if leading:
                        payload["error"] = leading
            except Exception:
                payload["error"] = probe["stdout_tail"]
        return payload

    def probe_capability_matrix(
        self,
        artifacts: list[ArtifactRecord] | None = None,
        repository_paths: list[str] | None = None,
        environment_path: str | None = None,
    ) -> CapabilityMatrix:
        artifacts = self._dedupe_artifacts_for_probe(artifacts or [])
        repository_paths = repository_paths or []
        env_path = self._discover_environment_path(
            environment_path,
            repository_paths=repository_paths,
        )
        modules: dict[str, bool] = {}
        python_available = False
        pip_available = False
        cuda_available = False
        torch_import_ok = False
        torch_runtime_ready = False
        torch_version = None
        torch_cuda_version = None
        torch_runtime_error = None
        torchvision_available = False
        torchvision_version = None
        deepxde_backend = None
        readiness_notes: list[str] = []
        compute_snapshot = self.inspect_compute_environment()
        gpu_runtime_required = bool(compute_snapshot.selected_gpu_ids or compute_snapshot.available_gpu_ids)
        if env_path:
            inspection = self.inspect_python_environment(
                env_path,
                modules=["torch", "h5py", "hydra", "deepxde", "tensorflow"],
            )
            modules = inspection["modules"]
            python_available = bool(inspection["python_available"])
            pip_available = bool(inspection["pip_available"])
            if self._bool_module(modules, "torch"):
                torch_probe = self._probe_torch_runtime(
                    env_path,
                    gpu_ids=compute_snapshot.selected_gpu_ids or compute_snapshot.available_gpu_ids,
                )
                torch_import_ok = bool(torch_probe.get("torch_import_ok"))
                torch_version = str(torch_probe.get("torch_version") or "") or None
                torch_cuda_version = str(torch_probe.get("torch_cuda_version") or "") or None
                cuda_available = bool(torch_probe.get("cuda_available"))
                torchvision_available = bool(torch_probe.get("torchvision_available"))
                torchvision_version = str(torch_probe.get("torchvision_version") or "") or None
                torch_runtime_error = str(torch_probe.get("error") or "") or None
                torch_runtime_ready = torch_import_ok
                if torch_runtime_error:
                    readiness_notes.append(f"torch_runtime={torch_runtime_error}")
            else:
                readiness_notes.append("torch_import_missing")
            if self._bool_module(modules, "deepxde"):
                backend_probe = self.run_in_environment(
                    env_path,
                    (
                        "python -c \"import deepxde; "
                        "print(getattr(getattr(deepxde, 'backend', None), 'backend_name', 'unknown'))\""
                    ),
                    allow_failure=True,
                    emit_progress=False,
                    stall_timeout_seconds=60,
                    gpu_ids=compute_snapshot.selected_gpu_ids or compute_snapshot.available_gpu_ids,
                )
                if backend_probe["returncode"] == 0:
                    deepxde_backend = backend_probe["stdout_tail"].splitlines()[-1].strip() or None
                else:
                    readiness_notes.append(
                        "deepxde_backend_probe_failed="
                        f"{(backend_probe['stderr_tail'] or backend_probe['stdout_tail'] or 'unknown').strip()[:160]}"
                    )

        repo_paths = [Path(path) for path in repository_paths]
        pdebench_trainable = any((path / "pdebench" / "models" / "train_models_forward.py").exists() for path in repo_paths)
        repo_ready = bool(repo_paths)
        env_ready = python_available and pip_available
        codepath_ready = pdebench_trainable
        exact_target_artifacts = [item for item in artifacts if self._infer_exact_target(item)]
        exact_corrupted = [
            (item.canonical_id or item.local_path or item.title)
            for item in exact_target_artifacts
            if item.status in {ArtifactStatus.CORRUPTED.value, ArtifactStatus.QUARANTINED.value}
        ]
        exact_pending = [
            (item.canonical_id or item.local_path or item.title)
            for item in exact_target_artifacts
            if self._artifact_pending_preparation(item)
        ]
        exact_missing = [
            (item.canonical_id or item.local_path or item.title)
            for item in exact_target_artifacts
            if item.status in {
                ArtifactStatus.DOWNLOAD_FAILED.value,
                ArtifactStatus.BLOCKED.value,
            }
        ]
        target_dataset_ready = bool(exact_target_artifacts) and all(self._artifact_ready(item) for item in exact_target_artifacts)
        target_dataset_preparing = bool(exact_pending) and not bool(exact_corrupted or exact_missing)
        fallback_assets_available = any(
            self._artifact_ready(item)
            and (
                item.artifact_type == "checkpoint"
                or (item.artifact_type == "dataset" and not self._infer_exact_target(item))
            )
            for item in artifacts
        )
        gpu_runtime_ready = (not gpu_runtime_required) or cuda_available
        environment_repair_needed = (
            env_ready
            and codepath_ready
            and (
                not torch_runtime_ready
                or (gpu_runtime_required and not gpu_runtime_ready)
            )
        )
        baseline_launch_ready = (
            target_dataset_ready
            and pdebench_trainable
            and torch_runtime_ready
            and self._bool_module(modules, "h5py")
            and gpu_runtime_ready
        )
        smoke_ready = repo_ready and env_ready and codepath_ready and torch_runtime_ready
        if target_dataset_ready and not baseline_launch_ready:
            if gpu_runtime_required and not gpu_runtime_ready:
                readiness_notes.append("gpu_runtime_unavailable")
            elif not torch_runtime_ready:
                readiness_notes.append("torch_runtime_unhealthy")
        matrix = CapabilityMatrix(
            environment_path=env_path,
            repo_ready=repo_ready,
            env_ready=env_ready,
            codepath_ready=codepath_ready,
            dataset_ready=target_dataset_ready,
            baseline_launch_ready=baseline_launch_ready,
            experiment_plan_ready=baseline_launch_ready,
            scientific_iteration_ready=baseline_launch_ready or fallback_assets_available or smoke_ready,
            python_available=python_available,
            pip_available=pip_available,
            torch_available=self._bool_module(modules, "torch"),
            torch_import_ok=torch_import_ok,
            torch_runtime_ready=torch_runtime_ready,
            torch_version=torch_version,
            torch_cuda_version=torch_cuda_version,
            torch_runtime_error=torch_runtime_error,
            torchvision_available=torchvision_available,
            torchvision_version=torchvision_version,
            cuda_available=cuda_available,
            gpu_runtime_required=gpu_runtime_required,
            gpu_runtime_ready=gpu_runtime_ready,
            h5py_available=self._bool_module(modules, "h5py"),
            hydra_available=self._bool_module(modules, "hydra"),
            pdebench_trainable=pdebench_trainable and python_available and torch_runtime_ready,
            deepxde_installed=self._bool_module(modules, "deepxde"),
            deepxde_backend=deepxde_backend,
            tensorflow_available=self._bool_module(modules, "tensorflow"),
            pinn_ready=self._bool_module(modules, "deepxde")
            and (deepxde_backend not in {None, "tensorflow"} or self._bool_module(modules, "tensorflow")),
            fno_ready=pdebench_trainable and torch_runtime_ready and self._bool_module(modules, "h5py"),
            unet_ready=pdebench_trainable and torch_runtime_ready and self._bool_module(modules, "h5py"),
            target_dataset_ready=target_dataset_ready,
            target_dataset_preparing=target_dataset_preparing,
            target_dataset_blocked=bool(exact_corrupted) or bool(exact_missing),
            exact_target_shards_pending=exact_pending,
            exact_target_shards_missing=exact_missing,
            exact_target_shards_corrupted=exact_corrupted,
            fallback_assets_available=fallback_assets_available,
            baseline_ready_to_launch=baseline_launch_ready,
            environment_repair_needed=environment_repair_needed,
            readiness_notes=readiness_notes,
        )
        self._record_tool_event("probe_capability_matrix", matrix.model_dump(mode="python"))
        return matrix
