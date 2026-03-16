from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from integrations import ground_experiment_plan
from state import ArtifactRecord, ExperimentPlan
from state import AssetSemanticSpec


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
            staging_root = root / "experiments" / "baseline_burgers" / "data_train"
            plan = ExperimentPlan(
                plan_id="plan-burgers",
                title="baseline",
                program_id="prog-burgers",
                working_directory=str(repo_root),
                setup_commands=[f"test -f {staging_root / dataset_path.name}"],
                launch_command="python train.py ++args.filename='burgers_train.hdf5'",
                log_path=str(root / "logs" / "baseline.log"),
            )

            grounded_plan, messages = ground_experiment_plan(plan, [artifact])

            self.assertTrue(messages)
            self.assertIn("++args.data_path=", grounded_plan.launch_command)
            self.assertIn("++args.root_path=", grounded_plan.launch_command)
            self.assertIn(str(staging_root), grounded_plan.launch_command)
            self.assertIn(str(dataset_path), grounded_plan.setup_commands[0])
            self.assertIn("Grounded command paths from verified local artifacts", grounded_plan.notes[-1])

    def test_ground_experiment_plan_prefers_plan_local_staging_root(self) -> None:
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

            sibling_dataset_dir = root / "other_run" / "external_assets" / "datasets" / "burgers"
            sibling_dataset_dir.mkdir(parents=True, exist_ok=True)
            dataset_path = sibling_dataset_dir / "burgers_train.hdf5"
            dataset_path.write_bytes(b"ok")

            staging_root = root / "experiments" / "baseline_burgers" / "data_train"
            artifact = ArtifactRecord(
                artifact_id="dataset-burgers",
                artifact_type="dataset",
                title="Burgers train shard",
                rationale="verified test artifact",
                local_path=str(dataset_path),
                status="ready_for_training",
            )
            plan = ExperimentPlan(
                plan_id="plan-burgers-local",
                title="baseline",
                program_id="prog-burgers",
                working_directory=str(repo_root),
                setup_commands=[
                    f"ln -sfn {dataset_path.parent} {staging_root}",
                    f"test -f {staging_root / dataset_path.name}",
                ],
                launch_command="python train.py ++args.filename='burgers_train.hdf5'",
                log_path=str(root / "logs" / "baseline.log"),
            )

            grounded_plan, _ = ground_experiment_plan(plan, [artifact])

            self.assertIn(str(staging_root), grounded_plan.launch_command)
            self.assertNotIn(str(dataset_path.parent), grounded_plan.launch_command)

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

    def test_ground_experiment_plan_prefers_current_run_artifact_over_sibling_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "current-run" / "external_assets" / "repos" / "project"
            config_root = repo_root / "config"
            config_root.mkdir(parents=True, exist_ok=True)
            (repo_root / "train.py").write_text(
                "def main(cfg):\n    print(cfg.args.data_path)\n",
                encoding="utf-8",
            )
            (config_root / "train.yaml").write_text('data_path: "/path/to/data"\n', encoding="utf-8")

            current_dataset_dir = root / "current-run" / "external_assets" / "datasets" / "burgers"
            sibling_dataset_dir = root / "older-run" / "external_assets" / "datasets" / "burgers"
            current_dataset_dir.mkdir(parents=True, exist_ok=True)
            sibling_dataset_dir.mkdir(parents=True, exist_ok=True)
            current_path = current_dataset_dir / "burgers_train.hdf5"
            sibling_path = sibling_dataset_dir / "burgers_train.hdf5"
            current_path.write_bytes(b"current")
            sibling_path.write_bytes(b"sibling")

            plan = ExperimentPlan(
                plan_id="plan-current",
                title="baseline",
                program_id="prog-burgers",
                working_directory=str(repo_root),
                setup_commands=[],
                launch_command="python train.py ++args.filename='burgers_train.hdf5'",
                log_path=str(root / "current-run" / "experiments" / "baseline" / "train.log"),
            )
            current_artifact = ArtifactRecord(
                artifact_id="dataset-current",
                canonical_id="dataset-burgers",
                artifact_type="dataset",
                title="burgers_train.hdf5",
                rationale="current run copy",
                local_path=str(current_path),
                status="ready_for_training",
            )
            sibling_artifact = ArtifactRecord(
                artifact_id="dataset-sibling",
                canonical_id="dataset-burgers",
                artifact_type="dataset",
                title="burgers_train.hdf5",
                rationale="sibling copy",
                local_path=str(sibling_path),
                status="ready_for_training",
            )

            grounded_plan, _ = ground_experiment_plan(plan, [sibling_artifact, current_artifact])
            self.assertIn(str(current_dataset_dir), grounded_plan.launch_command)
            self.assertNotIn(str(sibling_dataset_dir), grounded_plan.launch_command)

    def test_ground_experiment_plan_normalizes_managed_python_and_shell_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "repo"
            config_root = repo_root / "config"
            config_root.mkdir(parents=True, exist_ok=True)
            (repo_root / "train.py").write_text(
                "def main(cfg):\n    print(cfg.args.data_path)\n",
                encoding="utf-8",
            )
            (config_root / "train.yaml").write_text('data_path: "/path/to/data"\n', encoding="utf-8")
            dataset_dir = root / "run" / "external_assets" / "data"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            dataset_path = dataset_dir / "burgers_train.hdf5"
            dataset_path.write_bytes(b"ok")
            plan = ExperimentPlan(
                plan_id="plan-shell",
                title="baseline",
                program_id="prog-burgers",
                working_directory=str(repo_root),
                setup_commands=[
                    "uv run --python /tmp/env/bin/python python -c \"import torch\"",
                ],
                launch_command=(
                    "CUDA_VISIBLE_DEVICES=6 uv run --python /tmp/env/bin/python python train.py "
                    "++args.filename='burgers_train.hdf5' '2>&1' '|' tee /tmp/train.log"
                ),
                log_path=str(root / "run" / "experiments" / "baseline" / "train.log"),
            )
            artifact = ArtifactRecord(
                artifact_id="dataset-burgers",
                artifact_type="dataset",
                title="burgers_train.hdf5",
                rationale="ready",
                local_path=str(dataset_path),
                status="ready_for_training",
            )
            grounded_plan, _ = ground_experiment_plan(plan, [artifact])
            self.assertTrue(all("uv run --python" not in command for command in grounded_plan.setup_commands))
            self.assertTrue(any("/tmp/env/bin/python -c" in command for command in grounded_plan.setup_commands))
            self.assertIn(" | tee /tmp/train.log", grounded_plan.launch_command)
            self.assertNotIn("'|'", grounded_plan.launch_command)

    def test_ground_experiment_plan_repairs_truncated_baseline_from_repo_exemplar(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "repo"
            config_root = repo_root / "config"
            args_root = config_root / "args"
            args_root.mkdir(parents=True, exist_ok=True)
            (repo_root / "train_models_forward.py").write_text(
                "def main(cfg):\n"
                "    print(cfg.args.filename)\n"
                "    print(cfg.args.data_path)\n",
                encoding="utf-8",
            )
            (config_root / "config_rdb.yaml").write_text(
                'args:\n  data_path: "/path/to/swe2d/h5"\n  filename: "2D_rdb_NA_NA.h5"\n',
                encoding="utf-8",
            )
            (args_root / "config_ReacDiff.yaml").write_text(
                'model_name: "FNO"\nfilename: "ReacDiff_Nu0.5_Rho1.0.hdf5"\ninitial_step: 10\n',
                encoding="utf-8",
            )
            (repo_root / "run_forward_1D.sh").write_text(
                "CUDA_VISIBLE_DEVICES='0' python3 train_models_forward.py +args=config_ReacDiff.yaml "
                "++args.filename='ReacDiff_Nu0.5_Rho1.0.hdf5' ++args.model_name='FNO'\n",
                encoding="utf-8",
            )

            dataset_dir = root / "external_assets" / "data" / "1D" / "ReactionDiffusion" / "Train"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            dataset_path = dataset_dir / "ReacDiff_Nu1.0_Rho1.0.hdf5"
            dataset_path.write_bytes(b"ok")
            artifact = ArtifactRecord(
                artifact_id="dataset-reacdiff-target",
                artifact_type="dataset",
                title="Reaction-Diffusion target shard",
                rationale="ready",
                local_path=str(dataset_path),
                status="ready_for_training",
                semantic_spec=AssetSemanticSpec(
                    benchmark="PDEBench",
                    equation="ReactionDiffusion",
                    split="train",
                    nu="1",
                    rho="1",
                    filename=dataset_path.name,
                ),
            )
            plan = ExperimentPlan(
                plan_id="plan-rdb",
                title="Official PDEBench FNO baseline reproduction on 1D Reaction-Diffusion Nu=1.0 Rho=1.0",
                program_id="prog-rdb",
                working_directory=str(repo_root),
                setup_commands=[],
                launch_command=(
                    "python train_models_forward.py +args=config_rdb.yaml ++args.model_name=FNO "
                    "++args.if_training=True"
                ),
                log_path=str(root / "experiments" / "rdb" / "train.log"),
                notes=["Source output was truncated before full plan details were available."],
            )

            grounded_plan, messages = ground_experiment_plan(plan, [artifact])

            self.assertTrue(messages)
            self.assertIn("+args=config_ReacDiff.yaml", grounded_plan.launch_command)
            self.assertIn("++args.filename=ReacDiff_Nu1.0_Rho1.0.hdf5", grounded_plan.launch_command)
            self.assertIn(f"++args.data_path={dataset_dir}", grounded_plan.launch_command)

    def test_ground_experiment_plan_repairs_bare_override_from_selected_config_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "repo"
            config_root = repo_root / "config"
            args_root = config_root / "args"
            args_root.mkdir(parents=True, exist_ok=True)
            (repo_root / "train.py").write_text(
                "def main(cfg):\n"
                "    print(cfg.args.filename)\n"
                "    print(cfg.args.initial_step)\n",
                encoding="utf-8",
            )
            (config_root / "config.yaml").write_text('args:\n  data_path: "/path/to/data"\n', encoding="utf-8")
            (args_root / "config_Bgs.yaml").write_text(
                'filename: "1D_Burgers_Sols_Nu1.0.hdf5"\ninitial_step: 10\n',
                encoding="utf-8",
            )

            dataset_dir = root / "external_assets" / "data"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            dataset_path = dataset_dir / "1D_Burgers_Sols_Nu0.01.hdf5"
            dataset_path.write_bytes(b"ok")
            artifact = ArtifactRecord(
                artifact_id="dataset-burgers-target",
                artifact_type="dataset",
                title="Burgers target shard",
                rationale="ready",
                local_path=str(dataset_path),
                status="ready_for_training",
            )
            plan = ExperimentPlan(
                plan_id="plan-burgers-repair",
                title="Official PDEBench FNO baseline reproduction on 1D Burgers Nu=0.01",
                program_id="prog-burgers",
                working_directory=str(repo_root),
                setup_commands=[],
                launch_command=(
                    "python train.py +args=config_Bgs.yaml ++args.filename='1D_Burgers_Sols_Nu0.01.hdf5' "
                    "++args.initial_step"
                ),
                log_path=str(root / "experiments" / "burgers" / "train.log"),
            )

            grounded_plan, messages = ground_experiment_plan(plan, [artifact])

            self.assertTrue(messages)
            self.assertIn("++args.initial_step=10", grounded_plan.launch_command)
            self.assertNotIn("++args.initial_step ", grounded_plan.launch_command)


if __name__ == "__main__":
    unittest.main()
