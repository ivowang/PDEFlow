from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from config import ResearchBriefConfig, RuntimeConfig, SystemConfig
from memory import ResearchMemory
from orchestration import ResearchManager
from orchestration.normalization import normalize_artifacts
from research_agents import ExperimentPlannerAgent, PreflightValidationAgent
from state import (
    ArtifactRecord,
    BlockerRecord,
    CapabilityMatrix,
    CycleDeltaRecord,
    EnvironmentSnapshot,
    ExperimentPlanningPhaseOutput,
    RepositoryRecord,
    ResearchState,
)
from tools import ResearchTools


def make_config(run_name: str = "blocker-test") -> SystemConfig:
    return SystemConfig(
        project_name="test-project",
        run_name=run_name,
        research_brief=ResearchBriefConfig(title="Test brief", question="Can blocker routing avoid loops?"),
        runtime=RuntimeConfig(provider="openrouter", model="openai/gpt-5.4"),
    )


class CanonicalIdentityTests(unittest.TestCase):
    def test_normalize_artifacts_merges_aliases(self) -> None:
        artifacts = normalize_artifacts(
            [
                ArtifactRecord(
                    artifact_id="pdebench_reacdiff_nu1.0_rho1.0_train",
                    artifact_type="dataset",
                    title="ReactionDiffusion train",
                    rationale="alias a",
                    local_path="/tmp/ReacDiff_Nu1.0_Rho1.0.hdf5",
                    status="blocked",
                ),
                ArtifactRecord(
                    artifact_id="pdebench-reacdiff-nu1p0-rho1p0-train",
                    artifact_type="dataset",
                    title="ReactionDiffusion train",
                    rationale="alias b",
                    local_path="/tmp/ReacDiff_Nu1.0_Rho1.0.hdf5",
                    status="download_failed",
                ),
            ]
        )
        self.assertEqual(len(artifacts), 1)
        self.assertEqual(artifacts[0].canonical_id, "pdebench-dataset-reactiondiffusion-train-nu-1-rho-1")
        self.assertGreaterEqual(len(artifacts[0].raw_aliases), 2)


class BlockerRoutingTests(unittest.TestCase):
    def test_route_exhaustion_pivots_then_terminates(self) -> None:
        config = make_config(run_name="route-exhaust")
        manager = ResearchManager(config=config, repo_root=Path("/root/PDEFlow"))
        state = ResearchState(
            project_name=config.project_name,
            run_name=config.run_name,
            work_directory=str(manager.work_directory),
            research_brief=config.research_brief,
            capability_matrix=CapabilityMatrix(
                repo_ready=True,
                env_ready=True,
                codepath_ready=True,
                target_dataset_ready=False,
                target_dataset_blocked=True,
                baseline_ready_to_launch=False,
                scientific_iteration_ready=False,
            ),
            blocker_registry=[
                BlockerRecord(
                    blocker_id="blocker-a",
                    blocker_type="dataset_acquisition_failure",
                    target_entity="pdebench-dataset-burgers-train-nu-0p01",
                    first_seen_cycle=1,
                    last_seen_cycle=3,
                    repeat_count=3,
                    last_attempt_signature="direct",
                    evidence_summary="repeated failed transfers",
                    recovery_strategies_tried=["local_discovery"],
                    route_exhausted=True,
                )
            ],
        )
        route = manager._select_cycle_route(state)
        self.assertEqual(route.route_id, "recover-mirror-resolution")

        state.blocker_registry[0].recovery_strategies_tried = ["local_discovery", "mirror_resolution"]
        route = manager._select_cycle_route(state)
        self.assertEqual(route.route_id, "recover-partial-salvage")

        state.blocker_registry[0].recovery_strategies_tried = [
            "local_discovery",
            "mirror_resolution",
            "partial_salvage",
        ]
        state.cycle_deltas = [
            CycleDeltaRecord(
                cycle_index=3,
                snapshot_signature="same",
                changed=False,
                summary=["no_material_state_change"],
            )
        ]
        route = manager._select_cycle_route(state)
        self.assertEqual(route.route_id, "blocked-terminal")


class FallbackExecutionTests(unittest.TestCase):
    def test_planner_synthesizes_fallback_smoke_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "repo"
            repo_root.mkdir()
            config = make_config(run_name="fallback")
            memory = ResearchMemory(root=root)
            tools = ResearchTools(config=config, memory=memory, repo_root=Path("/root/PDEFlow"))
            state = ResearchState(
                project_name=config.project_name,
                run_name=config.run_name,
                work_directory=str(root),
                research_brief=config.research_brief,
                active_route_focus=["fallback_execution"],
                selected_baseline_program_id="prog-baseline",
                environment_snapshot=EnvironmentSnapshot(
                    python_executable="python",
                    python_version="3.10",
                    uv_available=True,
                    selected_gpu_ids=[6, 7],
                ),
                capability_matrix=CapabilityMatrix(
                    environment_path=str(root / "envs" / "pdebench-env"),
                    repo_ready=True,
                    env_ready=True,
                    codepath_ready=True,
                    scientific_iteration_ready=True,
                ),
                repositories=[
                    RepositoryRecord(
                        repo_id="pdebench",
                        canonical_id="github-pdebench-pdebench",
                        name="pdebench",
                        remote_url="https://github.com/pdebench/PDEBench",
                        local_path=str(repo_root),
                        bootstrap_status="ready",
                        environment_path=str(root / "envs" / "pdebench-env"),
                    )
                ],
            )
            output = ExperimentPlanningPhaseOutput(summary="no plans", experiment_plans=[], next_actions=[])
            summary = ExperimentPlannerAgent().apply_output(state, tools, output)
            self.assertIn("fallback experiment", summary.lower())
            self.assertEqual(len(state.experiment_plans), 1)
            self.assertEqual(state.experiment_plans[0].title, "Fallback smoke evidence run")


class PreflightZeroPlanTests(unittest.TestCase):
    def test_zero_plan_preflight_creates_structured_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_config(run_name="preflight-zero")
            memory = ResearchMemory(root=root)
            tools = ResearchTools(config=config, memory=memory, repo_root=Path("/root/PDEFlow"))
            state = ResearchState(
                project_name=config.project_name,
                run_name=config.run_name,
                work_directory=str(root),
                research_brief=config.research_brief,
                selected_baseline_program_id="prog-baseline",
                capability_matrix=CapabilityMatrix(
                    repo_ready=True,
                    env_ready=True,
                    codepath_ready=True,
                    target_dataset_ready=False,
                    target_dataset_blocked=True,
                    baseline_ready_to_launch=False,
                ),
            )
            summary = PreflightValidationAgent().run(state, tools, runtime=None)
            self.assertIn("blocked=1", summary)
            self.assertEqual(len(state.preflight_reports), 1)
            self.assertIn(state.preflight_reports[0].recommended_route, {"acquisition", "fallback_execution"})


class _FakeDataset:
    shape = (2, 3)
    ndim = 2

    def __getitem__(self, index):  # noqa: ANN001
        return 1


class _FakeH5File:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
        return False

    def keys(self):
        return ["u"]

    def __contains__(self, item):  # noqa: ANN001
        return item == "u"

    def __getitem__(self, item):  # noqa: ANN001
        return _FakeDataset()


class HITLTests(unittest.TestCase):
    def test_hitl_confirmation_revalidates_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_root = root / "external_assets" / "data" / "1D" / "Burgers" / "Train"
            data_root.mkdir(parents=True, exist_ok=True)
            dataset_path = data_root / "1D_Burgers_Sols_Nu0.01.hdf5"
            dataset_path.write_bytes(b"synthetic")
            config = make_config(run_name="hitl-resume")
            manager = ResearchManager(config=config, repo_root=Path("/root/PDEFlow"))
            manager.work_directory = root
            manager.memory = ResearchMemory(root=root)
            manager.tools = ResearchTools(config=config, memory=manager.memory, repo_root=Path("/root/PDEFlow"))
            state = ResearchState(
                project_name=config.project_name,
                run_name=config.run_name,
                work_directory=str(root),
                research_brief=config.research_brief,
                external_artifacts=[
                    ArtifactRecord(
                        artifact_id="pdebench_burgers_nu0.01_train",
                        canonical_id="pdebench-dataset-burgers-train-nu-0p01",
                        artifact_type="dataset",
                        title="burgers",
                        rationale="blocked dataset",
                        local_path=str(dataset_path),
                        status="blocked",
                        metadata={"min_size_bytes": 1, "required_keys": ["u"]},
                    )
                ],
                blocker_registry=[
                    BlockerRecord(
                        blocker_id="blocker-hitl",
                        blocker_type="dataset_acquisition_failure",
                        target_entity="pdebench-dataset-burgers-train-nu-0p01",
                        first_seen_cycle=1,
                        last_seen_cycle=3,
                        repeat_count=3,
                        last_attempt_signature="direct_remote_download",
                        evidence_summary="repeated download failure",
                        recovery_strategies_tried=["local_discovery", "mirror_resolution"],
                        route_exhausted=True,
                    )
                ],
            )
            with patch("builtins.input", side_effect=["1"]), patch("tools.artifacts.h5py") as fake_h5py:
                fake_h5py.File.side_effect = lambda *args, **kwargs: _FakeH5File()
                handled = manager._handle_hitl(state)
            self.assertTrue(handled)
            self.assertTrue(state.hitl_events)
            self.assertEqual(state.hitl_events[-1].status.value, "resumed")
            self.assertEqual(state.external_artifacts[0].status, "ready_for_training")


if __name__ == "__main__":
    unittest.main()
