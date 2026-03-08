from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from integrations import ground_experiment_plan
from state import ArtifactRecord, ExperimentPlan


class CommandGroundingTests(unittest.TestCase):
    def test_ground_experiment_plan_injects_verified_dataset_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "repo" / "operator_project"
            config_root = repo_root / "config"
            config_root.mkdir(parents=True, exist_ok=True)
            (repo_root / "train.py").write_text(
                "def main(cfg):\n"
                "    print(cfg.args.filename)\n"
                "    print(cfg.args.data_path)\n"
                "    print(cfg.args.root_path)\n",
                encoding="utf-8",
            )
            (config_root / "train.yaml").write_text(
                'data_path: "/path/to/data"\nroot_path: "../data"\n',
                encoding="utf-8",
            )

            dataset_dir = root / "external_assets" / "datasets" / "burgers"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            dataset_path = dataset_dir / "burgers_train.hdf5"
            dataset_path.write_bytes(b"ok")

            artifact = ArtifactRecord(
                artifact_id="dataset-burgers",
                artifact_type="dataset",
                title="Burgers train shard",
                rationale="verified test artifact",
                local_path=str(dataset_path),
                status="ready_for_training",
                metadata={"size_bytes": dataset_path.stat().st_size},
            )
            plan = ExperimentPlan(
                plan_id="plan-burgers",
                title="baseline",
                program_id="prog-burgers",
                working_directory=str(repo_root),
                setup_commands=["test -f /wrong/place/burgers_train.hdf5"],
                launch_command="python train.py ++args.filename='burgers_train.hdf5'",
                log_path=str(root / "logs" / "baseline.log"),
            )

            grounded_plan, messages = ground_experiment_plan(plan, [artifact])

            self.assertTrue(messages)
            self.assertIn("++args.data_path=", grounded_plan.launch_command)
            self.assertIn("++args.root_path=", grounded_plan.launch_command)
            self.assertIn(str(dataset_dir), grounded_plan.launch_command)
            self.assertIn(str(dataset_path), grounded_plan.setup_commands[0])
            self.assertIn("Grounded command paths from verified local artifacts", grounded_plan.notes[-1])

    def test_incomplete_artifact_is_not_used_for_grounding(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "repo" / "operator_project"
            config_root = repo_root / "config"
            config_root.mkdir(parents=True, exist_ok=True)
            (repo_root / "train.py").write_text(
                "def main(cfg):\n"
                "    print(cfg.args.filename)\n"
                "    print(cfg.args.data_path)\n",
                encoding="utf-8",
            )
            (config_root / "train.yaml").write_text(
                'data_path: "/path/to/data"\n',
                encoding="utf-8",
            )

            dataset_dir = root / "external_assets" / "datasets" / "reacdiff"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            dataset_path = dataset_dir / "reacdiff_train.hdf5"
            dataset_path.write_bytes(b"partial")

            artifact = ArtifactRecord(
                artifact_id="dataset-reacdiff",
                artifact_type="dataset",
                title="ReactionDiffusion shard",
                rationale="incomplete test artifact",
                local_path=str(dataset_path),
                status="corrupted",
                metadata={"expected_size_bytes": 1024},
            )
            plan = ExperimentPlan(
                plan_id="plan-reacdiff",
                title="baseline",
                program_id="prog-reacdiff",
                working_directory=str(repo_root),
                setup_commands=[],
                launch_command="python train.py ++args.filename='reacdiff_train.hdf5'",
                log_path=str(root / "logs" / "reacdiff.log"),
            )

            grounded_plan, messages = ground_experiment_plan(plan, [artifact])

            self.assertEqual("blocked", grounded_plan.status)
            self.assertIn("dataset-reacdiff", grounded_plan.required_artifact_ids)
            self.assertTrue(messages)


if __name__ == "__main__":
    unittest.main()
