from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from config import ResearchBriefConfig, RuntimeConfig, SystemConfig
from memory import ResearchMemory
from state import ArtifactRecord
from tools import ResearchTools


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
                with patch.object(tools, "run_command", side_effect=fake_run_command):
                    validated = tools.validate_artifact_record(artifact, quarantine_on_failure=False)
            self.assertEqual(validated.status, "ready_for_training")
            self.assertEqual(validated.validation.top_level_keys, ["u"])
            self.assertEqual(validated.validation.sample_read_target, "u")


if __name__ == "__main__":
    unittest.main()
