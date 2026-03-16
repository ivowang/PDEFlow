from __future__ import annotations

import hashlib
import io
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from config import ResearchBriefConfig, RuntimeConfig, SystemConfig
from memory import ResearchMemory
from state import ArtifactRecord
from orchestration.normalization import normalize_artifacts
from tools import ResearchTools
from tools.retrieval import _TransferError


def make_config(run_name: str = "artifact-test") -> SystemConfig:
    return SystemConfig(
        project_name="test-project",
        run_name=run_name,
        research_brief=ResearchBriefConfig(title="Test", question="Validate artifacts"),
        runtime=RuntimeConfig(provider="openrouter", model="openai/gpt-5.4"),
    )


class _FakeDataset:
    shape = (2, 3)
    ndim = 2

    def __getitem__(self, index):  # noqa: ANN001
        return 1


class _FakeH5File:
    def __init__(self, required_keys: list[str] | None = None):
        self._keys = required_keys or ["u"]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
        return False

    def keys(self):
        return self._keys

    def __contains__(self, item):  # noqa: ANN001
        return item in self._keys

    def __getitem__(self, item):  # noqa: ANN001
        return _FakeDataset()


class ArtifactValidationTests(unittest.TestCase):
    def test_memory_load_artifacts_normalizes_aliases_across_registry_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            memory = ResearchMemory(root=root)
            memory.record_artifact(
                ArtifactRecord(
                    artifact_id="1d_burgers_nu0.01",
                    canonical_id="doi:10.18419/darus-2986/1D/Burgers/Train/1D_Burgers_Sols_Nu0.01.hdf5",
                    artifact_type="dataset",
                    title="1D_Burgers_Sols_Nu0.01.hdf5",
                    rationale="first entry",
                    local_path="/tmp/1D_Burgers_Sols_Nu0.01.hdf5",
                    status="verified_local",
                    metadata={"benchmark": "PDEBench", "equation": "Burgers", "split": "train", "nu": "0.01"},
                )
            )
            memory.record_artifact(
                ArtifactRecord(
                    artifact_id="1d-burgers-sols-nu0-01-hdf5",
                    canonical_id="1d_burgers_nu0.01",
                    artifact_type="dataset",
                    title="1D_Burgers_Sols_Nu0.01.hdf5",
                    rationale="second entry",
                    local_path="/tmp/1D_Burgers_Sols_Nu0.01.hdf5",
                    status="ready_for_training",
                )
            )
            loaded = memory.load_artifacts()
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].canonical_id, "dataset-burgers-train-nu-0p01")
            self.assertIn("1d_burgers_nu0.01", loaded[0].raw_aliases)

    def test_download_file_rejects_path_escape_outside_run_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shared_runs_root = root / "runs"
            config = make_config(run_name="path-escape")
            config.execution.work_directory = str(shared_runs_root / "{run_name}")
            memory = ResearchMemory(root=shared_runs_root / config.run_name)
            tools = ResearchTools(config=config, memory=memory, repo_root=Path("/root/PDEFlow"))
            with self.assertRaises(ValueError):
                tools.download_file(
                    url="https://example.com/fno.pdf",
                    target_path="../../lit/fno.pdf",
                    artifact_id="fno-pdf",
                    artifact_type="paper",
                )

    def test_discover_local_artifacts_prefers_cached_ready_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            good_dir = root / "good" / "Train"
            bad_dir = root / "bad" / "Train"
            good_dir.mkdir(parents=True, exist_ok=True)
            bad_dir.mkdir(parents=True, exist_ok=True)
            good_path = good_dir / "1D_Burgers_Sols_Nu0.01.hdf5"
            bad_path = bad_dir / "1D_Burgers_Sols_Nu0.01.hdf5"
            good_path.write_bytes(b"good")
            bad_path.write_bytes(b"bad")
            config = make_config(run_name="cached-discovery")
            memory = ResearchMemory(root=root / "run")
            tools = ResearchTools(config=config, memory=memory, repo_root=Path("/root/PDEFlow"))
            memory.record_artifact(
                ArtifactRecord(
                    artifact_id="burgers-good",
                    canonical_id="dataset-burgers-train-nu-0p01",
                    artifact_type="dataset",
                    title=good_path.name,
                    rationale="cached ready",
                    local_path=str(good_path),
                    status="ready_for_training",
                    validation={
                        "validator": "hdf5",
                        "status": "ready_for_training",
                        "exists": True,
                        "size_bytes": good_path.stat().st_size,
                        "size_ok": True,
                        "ready_for_training": True,
                    },
                )
            )
            memory.record_artifact(
                ArtifactRecord(
                    artifact_id="burgers-bad",
                    canonical_id="dataset-burgers-train-nu-0p01",
                    artifact_type="dataset",
                    title=bad_path.name,
                    rationale="cached bad",
                    local_path=str(bad_path),
                    status="corrupted",
                )
            )
            results = tools.discover_local_artifacts(
                "1D_Burgers_Sols_Nu0.01.hdf5",
                search_roots=[str(root)],
                artifact_type="dataset",
                canonical_target_id="dataset-burgers-train-nu-0p01",
            )
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["path"], str(good_path.resolve()))
            self.assertTrue(results[0]["ready_for_training"])

    def test_discover_local_artifacts_persists_external_sibling_candidates_as_negative_knowledge(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shared_runs_root = root / "runs"
            prior_run_file = shared_runs_root / "prior-run" / "external_assets" / "data" / "1D" / "Burgers" / "Train" / "1D_Burgers_Sols_Nu1.0.hdf5"
            prior_run_file.parent.mkdir(parents=True, exist_ok=True)
            prior_run_file.write_bytes(b"payload")
            config = make_config(run_name="artifact-discovery")
            config.execution.work_directory = str(shared_runs_root / "{run_name}")
            memory = ResearchMemory(root=shared_runs_root / config.run_name)
            tools = ResearchTools(config=config, memory=memory, repo_root=Path("/root/PDEFlow"))

            with patch("tools.artifacts.h5py") as fake_h5py:
                fake_h5py.File.side_effect = lambda *args, **kwargs: _FakeH5File(["u"])
                results = tools.discover_local_artifacts(
                    "1D_Burgers_Sols_Nu1.0.hdf5",
                    search_roots=[str(shared_runs_root)],
                    artifact_type="dataset",
                    canonical_target_id="dataset-burgers-train-nu-1",
                    min_size_bytes=1,
                    required_keys=["u"],
                )

            self.assertTrue(results)
            self.assertEqual(Path(results[0]["path"]).resolve(), prior_run_file.resolve())
            persisted = memory.load_artifacts()
            self.assertEqual(len(persisted), 1)
            self.assertEqual(Path(persisted[0].local_path).resolve(), prior_run_file.resolve())
            self.assertEqual(persisted[0].status, "ready_for_training")

    def test_discover_local_artifacts_uses_sibling_registry_before_recursive_scan(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shared_runs_root = root / "runs"
            prior_run_root = shared_runs_root / "prior-run"
            prior_run_file = prior_run_root / "external_assets" / "data" / "1D" / "Burgers" / "Train" / "1D_Burgers_Sols_Nu1.0.hdf5"
            prior_run_file.parent.mkdir(parents=True, exist_ok=True)
            prior_run_file.write_bytes(b"payload")
            prior_memory = ResearchMemory(root=prior_run_root)
            prior_memory.record_artifact(
                ArtifactRecord(
                    artifact_id="burgers-prior",
                    canonical_id="dataset-burgers-train-nu-1",
                    artifact_type="dataset",
                    title=prior_run_file.name,
                    rationale="prior ready artifact",
                    local_path=str(prior_run_file),
                    status="ready_for_training",
                    validation={
                        "validator": "hdf5",
                        "status": "ready_for_training",
                        "exists": True,
                        "size_bytes": prior_run_file.stat().st_size,
                        "size_ok": True,
                        "ready_for_training": True,
                    },
                )
            )

            config = make_config(run_name="artifact-sibling-registry")
            config.execution.work_directory = str(shared_runs_root / "{run_name}")
            memory = ResearchMemory(root=shared_runs_root / config.run_name)
            tools = ResearchTools(config=config, memory=memory, repo_root=Path("/root/PDEFlow"))

            with patch.object(tools, "_candidate_local_discovery_dirs", return_value=[]):
                results = tools.discover_local_artifacts(
                    "1D_Burgers_Sols_Nu1.0.hdf5",
                    search_roots=[str(shared_runs_root)],
                    artifact_type="dataset",
                    canonical_target_id="dataset-burgers-train-nu-1",
                )

            self.assertTrue(results)
            self.assertEqual(Path(results[0]["path"]).resolve(), prior_run_file.resolve())
            self.assertTrue(results[0]["ready_for_training"])

    def test_download_file_reuses_validated_local_copy_from_sibling_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shared_runs_root = root / "runs"
            prior_run = shared_runs_root / "prior-run" / "external_assets" / "data" / "1D" / "Burgers" / "Train"
            prior_run.mkdir(parents=True, exist_ok=True)
            source_path = prior_run / "1D_Burgers_Sols_Nu0.01.hdf5"
            data = b"synthetic-hdf5-payload"
            source_path.write_bytes(data)
            checksum = hashlib.md5(data).hexdigest()

            config = make_config(run_name="artifact-test")
            config.execution.work_directory = str(shared_runs_root / "{run_name}")
            memory = ResearchMemory(root=shared_runs_root / config.run_name)
            tools = ResearchTools(config=config, memory=memory, repo_root=Path("/root/PDEFlow"))
            target_path = tools.shared_workspace_root / "data" / "1D" / "Burgers" / "Train" / "1D_Burgers_Sols_Nu0.01.hdf5"

            with patch("tools.artifacts.h5py") as fake_h5py:
                fake_h5py.File.side_effect = lambda *args, **kwargs: _FakeH5File(["u"])
                result = tools.download_file(
                    url="https://example.com/1D_Burgers_Sols_Nu0.01.hdf5",
                    target_path=str(target_path),
                    artifact_id="burgers-train",
                    expected_checksum=checksum,
                    min_size_bytes=1,
                    required_keys=["u"],
                )
            self.assertEqual(result["strategy_id"], "local_discovery")
            self.assertTrue(target_path.exists())

    def test_clone_repository_falls_back_to_github_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_config(run_name="repo-archive-fallback")
            memory = ResearchMemory(root=root)
            tools = ResearchTools(config=config, memory=memory, repo_root=Path("/root/PDEFlow"))
            archive_buffer = io.BytesIO()
            with tarfile.open(fileobj=archive_buffer, mode="w:gz") as archive:
                payload = b"print('ok')\n"
                info = tarfile.TarInfo(name="PDEBench-main/README.md")
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
            archive_bytes = archive_buffer.getvalue()

            def fake_run_command(command, **kwargs):  # noqa: ANN001
                if "git clone" in command:
                    return {
                        "returncode": 128,
                        "stdout_tail": "",
                        "stderr_tail": "fatal: gnutls_handshake() failed",
                        "log_path": str(root / "clone.log"),
                    }
                return {
                    "returncode": 0,
                    "stdout_tail": "",
                    "stderr_tail": "",
                    "log_path": str(root / "other.log"),
                }

            response = MagicMock()
            response.content = archive_bytes
            response.raise_for_status.return_value = None
            with patch.object(tools, "run_command", side_effect=fake_run_command), patch(
                "tools.retrieval.httpx.get",
                return_value=response,
            ):
                result = tools.clone_repository("https://github.com/pdebench/PDEBench")
            self.assertEqual(result["status"], "archive_downloaded")
            self.assertTrue((Path(result["path"]) / "README.md").exists())

    def test_download_file_falls_back_to_curl_after_httpx_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_config(run_name="download-fallback")
            memory = ResearchMemory(root=root)
            tools = ResearchTools(config=config, memory=memory, repo_root=Path("/root/PDEFlow"))
            target_path = root / "data" / "dataset.hdf5"
            payload = b"fallback-download"
            checksum = hashlib.md5(payload).hexdigest()

            def fake_curl(url, part_path, attempt_index):  # noqa: ANN001
                part_path.parent.mkdir(parents=True, exist_ok=True)
                part_path.write_bytes(payload)
                return {
                    "transfer_method": "curl",
                    "attempt_count": attempt_index,
                    "bytes_downloaded": len(payload),
                    "elapsed_time": 1.0,
                    "average_throughput": float(len(payload)),
                    "resumed": False,
                }

            with patch.object(
                tools,
                "_download_httpx",
                side_effect=_TransferError("transfer_timeout", "timeout"),
            ), patch.object(
                tools,
                "_download_with_curl",
                side_effect=fake_curl,
            ), patch("tools.artifacts.h5py") as fake_h5py:
                fake_h5py.File.side_effect = lambda *args, **kwargs: _FakeH5File(["u"])
                result = tools.download_file(
                    url="https://example.com/dataset.hdf5",
                    target_path=str(target_path),
                    artifact_id="dataset-fallback",
                    expected_checksum=checksum,
                    min_size_bytes=1,
                    required_keys=["u"],
                )
            self.assertEqual(result["transfer_method"], "curl")
            self.assertEqual(result["validation_status"], "ready_for_training")
            self.assertTrue(target_path.exists())

    def test_hdf5_validation_marks_ready_for_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_config()
            memory = ResearchMemory(root=root)
            tools = ResearchTools(config=config, memory=memory, repo_root=Path("/root/PDEFlow"))
            path = root / "dataset.hdf5"
            data = b"synthetic-hdf5-payload"
            path.write_bytes(data)
            checksum = hashlib.md5(data).hexdigest()
            artifact = ArtifactRecord(
                artifact_id="dataset-1",
                artifact_type="dataset",
                title="dataset",
                rationale="test",
                local_path=str(path),
                status="downloaded",
                metadata={"official_md5": checksum, "required_keys": ["u"], "min_size_bytes": 1},
            )
            with patch("tools.artifacts.h5py") as fake_h5py:
                fake_h5py.File.side_effect = lambda *args, **kwargs: _FakeH5File(["u"])
                validated = tools.validate_artifact_record(artifact, quarantine_on_failure=False)
            self.assertEqual(validated.status, "ready_for_training")
            self.assertTrue(validated.validation.ready_for_training)
            self.assertEqual(validated.validation.top_level_keys, ["u"])

    def test_checksum_mismatch_quarantines_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_config()
            memory = ResearchMemory(root=root)
            tools = ResearchTools(config=config, memory=memory, repo_root=Path("/root/PDEFlow"))
            path = root / "dataset.hdf5"
            path.write_bytes(b"payload")
            artifact = ArtifactRecord(
                artifact_id="dataset-2",
                artifact_type="dataset",
                title="dataset",
                rationale="test",
                local_path=str(path),
                status="downloaded",
                metadata={"official_md5": "deadbeef", "required_keys": ["u"], "min_size_bytes": 1},
            )
            with patch("tools.artifacts.h5py") as fake_h5py:
                fake_h5py.File.side_effect = lambda *args, **kwargs: _FakeH5File(["u"])
                validated = tools.validate_artifact_record(artifact, quarantine_on_failure=True)
            self.assertEqual(validated.status, "quarantined")
            self.assertIsNotNone(validated.quarantine_path)
            self.assertFalse(path.exists())

    def test_hdf5_validation_falls_back_to_subprocess_when_local_h5py_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_config()
            memory = ResearchMemory(root=root)
            tools = ResearchTools(config=config, memory=memory, repo_root=Path("/root/PDEFlow"))
            path = root / "dataset.hdf5"
            data = b"synthetic-hdf5-payload"
            path.write_bytes(data)
            artifact = ArtifactRecord(
                artifact_id="dataset-3",
                artifact_type="dataset",
                title="dataset",
                rationale="test",
                local_path=str(path),
                status="downloaded",
                metadata={"min_size_bytes": 1, "required_keys": ["u"]},
            )

            def fake_run_command(command, **kwargs):  # noqa: ANN001
                return {
                    "command": command,
                    "cwd": str(root),
                    "returncode": 0,
                    "stdout_tail": (
                        '{"format_valid": true, "top_level_keys": ["u"], '
                        '"sample_read_target": "u", "sample_shape": [2, 3], "failure_reasons": []}'
                    ),
                    "stderr_tail": "",
                    "log_path": str(root / "cmd.log"),
                    "emit_progress": False,
                    "job_kind": "command",
                }

            with patch("tools.artifacts.h5py", None):
                with patch("tools.artifacts.subprocess.run") as fake_subprocess:
                    fake_subprocess.return_value = MagicMock(
                        returncode=0,
                        stdout=(
                            '{"format_valid": true, "top_level_keys": ["u"], '
                            '"sample_read_target": "u", "sample_shape": [2, 3], "failure_reasons": []}'
                        ),
                        stderr="",
                    )
                    validated = tools.validate_artifact_record(artifact, quarantine_on_failure=False)
            self.assertEqual(validated.status, "ready_for_training")
            self.assertEqual(validated.validation.top_level_keys, ["u"])
            self.assertEqual(validated.validation.sample_read_target, "u")

    def test_validator_unavailable_does_not_quarantine_checksum_matched_manual_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_config()
            memory = ResearchMemory(root=root)
            tools = ResearchTools(config=config, memory=memory, repo_root=Path("/root/PDEFlow"))
            path = root / "dataset.hdf5"
            data = b"payload"
            path.write_bytes(data)
            checksum = hashlib.md5(data).hexdigest()
            artifact = ArtifactRecord(
                artifact_id="dataset-4",
                artifact_type="dataset",
                title="dataset",
                rationale="manual file",
                local_path=str(path),
                status="downloaded",
                metadata={"official_md5": checksum, "min_size_bytes": 1, "human_provided": True},
            )
            with patch("tools.artifacts.h5py", None), patch.object(
                tools,
                "_validate_hdf5_via_subprocess",
                return_value=None,
            ):
                validated = tools.validate_artifact_record(artifact, quarantine_on_failure=True)
            self.assertEqual(validated.status, "checksum_verified")
            self.assertIsNone(validated.quarantine_path)
            self.assertTrue(path.exists())

    def test_hdf5_validation_uses_bootstrapped_validator_python_when_local_h5py_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_config()
            memory = ResearchMemory(root=root)
            tools = ResearchTools(config=config, memory=memory, repo_root=Path("/root/PDEFlow"))
            path = root / "dataset.hdf5"
            path.write_bytes(b"payload")
            artifact = ArtifactRecord(
                artifact_id="dataset-validator-env",
                artifact_type="dataset",
                title="dataset",
                rationale="validator env",
                local_path=str(path),
                status="downloaded",
                metadata={"min_size_bytes": 1, "required_keys": ["u"]},
            )

            with patch("tools.artifacts.h5py", None), patch.object(
                tools,
                "_candidate_hdf5_validation_pythons",
                return_value=[],
            ), patch.object(
                tools,
                "_ensure_hdf5_validator_python",
                return_value="/tmp/fake-validator/bin/python",
            ), patch("tools.artifacts.subprocess.run") as fake_subprocess:
                fake_subprocess.return_value = MagicMock(
                    returncode=0,
                    stdout=(
                        '{"format_valid": true, "top_level_keys": ["u"], '
                        '"sample_read_target": "u", "sample_shape": [2, 3], "failure_reasons": []}'
                    ),
                    stderr="",
                )
                validated = tools.validate_artifact_record(artifact, quarantine_on_failure=False)
            self.assertEqual(validated.status, "ready_for_training")
            self.assertEqual(validated.validation.details.get("validator_python"), "/tmp/fake-validator/bin/python")

    def test_small_checksum_verified_hdf5_can_still_be_ready_when_format_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_config()
            memory = ResearchMemory(root=root)
            tools = ResearchTools(config=config, memory=memory, repo_root=Path("/root/PDEFlow"))
            path = root / "small_test_shard.hdf5"
            data = b"small-but-official"
            path.write_bytes(data)
            checksum = hashlib.md5(data).hexdigest()
            artifact = ArtifactRecord(
                artifact_id="dataset-5",
                artifact_type="dataset",
                title="small test shard",
                rationale="official small shard",
                local_path=str(path),
                status="downloaded",
                metadata={"official_md5": checksum, "required_keys": ["u"], "min_size_bytes": 1_000_000},
            )
            with patch("tools.artifacts.h5py") as fake_h5py:
                fake_h5py.File.side_effect = lambda *args, **kwargs: _FakeH5File(["u"])
                validated = tools.validate_artifact_record(artifact, quarantine_on_failure=False)
            self.assertEqual(validated.status, "ready_for_training")
            self.assertTrue(validated.validation.ready_for_training)

    def test_candidate_hdf5_validation_pythons_preserves_managed_env_symlink_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_config()
            memory = ResearchMemory(root=root)
            tools = ResearchTools(config=config, memory=memory, repo_root=Path("/root/PDEFlow"))
            target_python = root / "real-python"
            target_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            env_python = tools.managed_env_root / "probe-env" / "bin" / "python"
            env_python.parent.mkdir(parents=True, exist_ok=True)
            env_python.symlink_to(target_python)
            candidates = tools._candidate_hdf5_validation_pythons()
            self.assertIn(str(env_python.absolute()), candidates)
            self.assertNotIn(str(target_python.resolve()), candidates[:-1])

    def test_alias_merge_preserves_official_checksum_for_human_provided_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_config()
            memory = ResearchMemory(root=root)
            tools = ResearchTools(config=config, memory=memory, repo_root=Path("/root/PDEFlow"))
            path = root / "ReacDiff_react_Nu1.0_Rho1.0.hdf5"
            data = b"small-official-test-shard"
            path.write_bytes(data)
            checksum = hashlib.md5(data).hexdigest()
            artifacts = normalize_artifacts(
                [
                    ArtifactRecord(
                        artifact_id="doi:10.18419/darus-2986/1D/ReactionDiffusion/Test/ReacDiff_react_Nu1.0_Rho1.0.hdf5",
                        canonical_id="doi:10.18419/darus-2986/1D/ReactionDiffusion/Test/ReacDiff_react_Nu1.0_Rho1.0.hdf5",
                        artifact_type="dataset",
                        title="ReacDiff_react_Nu1.0_Rho1.0.hdf5",
                        rationale="official placeholder",
                        local_path=str(path),
                        status="download_failed",
                        metadata={
                            "official_checksum": checksum,
                            "min_size_bytes": 1_000_000,
                            "official_path": "1D/ReactionDiffusion/Test/",
                        },
                    ),
                    ArtifactRecord(
                        artifact_id="ReacDiff_react_Nu1.0_Rho1.0.hdf5",
                        canonical_id="pdebench-dataset-reactiondiffusion-test-nu-1-rho-1",
                        artifact_type="dataset",
                        title="human provided",
                        rationale="manual copy",
                        local_path=str(path),
                        status="blocked",
                        metadata={
                            "human_provided": True,
                            "min_size_bytes": 1_000_000,
                            "official_path": "1D/ReactionDiffusion/Test/",
                        },
                    ),
                ]
            )
            self.assertEqual(len(artifacts), 1)
            self.assertEqual(artifacts[0].metadata["official_checksum"], checksum)
            with patch("tools.artifacts.h5py") as fake_h5py:
                fake_h5py.File.side_effect = lambda *args, **kwargs: _FakeH5File(["u"])
                validated = tools.validate_artifact_record(artifacts[0], quarantine_on_failure=False)
            self.assertEqual(validated.status, "ready_for_training")
            self.assertTrue(validated.validation.ready_for_training)


if __name__ == "__main__":
    unittest.main()
