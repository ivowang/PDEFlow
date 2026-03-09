from __future__ import annotations

from pathlib import Path

from state import ArtifactStatus, ArtifactRecord, CapabilityMatrix


class CapabilityProbeMixin:
    def _discover_environment_path(self, explicit_path: str | None = None) -> str | None:
        if explicit_path:
            resolved = self._resolve_path(explicit_path, default_root=self.managed_env_root)
            return str(resolved)
        environments = sorted(path for path in self.managed_env_root.iterdir() if path.is_dir())
        if not environments:
            return None
        return str(environments[0])

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

    def probe_capability_matrix(
        self,
        artifacts: list[ArtifactRecord] | None = None,
        repository_paths: list[str] | None = None,
        environment_path: str | None = None,
    ) -> CapabilityMatrix:
        artifacts = artifacts or []
        repository_paths = repository_paths or []
        env_path = self._discover_environment_path(environment_path)
        modules: dict[str, bool] = {}
        python_available = False
        pip_available = False
        cuda_available = False
        deepxde_backend = None
        if env_path:
            inspection = self.inspect_python_environment(
                env_path,
                modules=["torch", "h5py", "hydra", "deepxde", "tensorflow"],
            )
            modules = inspection["modules"]
            python_available = bool(inspection["python_available"])
            pip_available = bool(inspection["pip_available"])
            if self._bool_module(modules, "torch"):
                cuda_probe = self.run_in_environment(
                    env_path,
                    "python -c \"import torch; print(int(torch.cuda.is_available()))\"",
                    allow_failure=True,
                    emit_progress=False,
                )
                cuda_available = cuda_probe["returncode"] == 0 and cuda_probe["stdout_tail"].strip().endswith("1")
            if self._bool_module(modules, "deepxde"):
                backend_probe = self.run_in_environment(
                    env_path,
                    (
                        "python -c \"import deepxde; "
                        "print(getattr(getattr(deepxde, 'backend', None), 'backend_name', 'unknown'))\""
                    ),
                    allow_failure=True,
                    emit_progress=False,
                )
                if backend_probe["returncode"] == 0:
                    deepxde_backend = backend_probe["stdout_tail"].splitlines()[-1].strip() or None

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
        exact_missing = [
            (item.canonical_id or item.local_path or item.title)
            for item in exact_target_artifacts
            if item.status in {ArtifactStatus.DOWNLOAD_FAILED.value, ArtifactStatus.BLOCKED.value}
        ]
        target_dataset_ready = bool(exact_target_artifacts) and all(self._artifact_ready(item) for item in exact_target_artifacts)
        fallback_assets_available = any(
            self._artifact_ready(item)
            and (
                item.artifact_type == "checkpoint"
                or (item.artifact_type == "dataset" and not self._infer_exact_target(item))
            )
            for item in artifacts
        )
        baseline_launch_ready = target_dataset_ready and pdebench_trainable and self._bool_module(modules, "torch") and self._bool_module(modules, "h5py")
        smoke_ready = repo_ready and env_ready and codepath_ready and self._bool_module(modules, "torch")
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
            cuda_available=cuda_available,
            h5py_available=self._bool_module(modules, "h5py"),
            hydra_available=self._bool_module(modules, "hydra"),
            pdebench_trainable=pdebench_trainable and python_available and self._bool_module(modules, "torch"),
            deepxde_installed=self._bool_module(modules, "deepxde"),
            deepxde_backend=deepxde_backend,
            tensorflow_available=self._bool_module(modules, "tensorflow"),
            pinn_ready=self._bool_module(modules, "deepxde")
            and (deepxde_backend not in {None, "tensorflow"} or self._bool_module(modules, "tensorflow")),
            fno_ready=pdebench_trainable and self._bool_module(modules, "torch") and self._bool_module(modules, "h5py"),
            unet_ready=pdebench_trainable and self._bool_module(modules, "torch") and self._bool_module(modules, "h5py"),
            target_dataset_ready=target_dataset_ready,
            target_dataset_blocked=bool(exact_corrupted) or bool(exact_missing),
            exact_target_shards_missing=exact_missing,
            exact_target_shards_corrupted=exact_corrupted,
            fallback_assets_available=fallback_assets_available,
            baseline_ready_to_launch=baseline_launch_ready,
        )
        self._record_tool_event("probe_capability_matrix", matrix.model_dump(mode="python"))
        return matrix
