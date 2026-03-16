from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from config import ResearchBriefConfig, RuntimeConfig, SystemConfig
from memory import ResearchMemory
from orchestration import ResearchManager
from orchestration.hitl import extract_absolute_paths
from orchestration.normalization import normalize_artifacts
from orchestration.specs import CycleRoute, PhaseSpec
from research_agents import ExperimentPlannerAgent, PreflightValidationAgent
from common import canonicalize_artifact_id
from state import (
    ArtifactRecord,
    ArtifactValidationResult,
    BlockerRecord,
    CapabilityMatrix,
    CycleDeltaRecord,
    EnvironmentSnapshot,
    ExperimentPlan,
    ExperimentPlanningPhaseOutput,
    ResearchPhase,
    RouteDecisionRecord,
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
        self.assertEqual(artifacts[0].canonical_id, "dataset-reactiondiffusion-train-nu-1-rho-1")
        self.assertGreaterEqual(len(artifacts[0].raw_aliases), 2)

    def test_hitl_path_extraction_strips_trailing_punctuation(self) -> None:
        extracted = extract_absolute_paths(
            "Use /data0/shared/pdebench_mirror and /data0/shared/burgers_train; for local discovery."
        )
        self.assertEqual(
            extracted,
            ["/data0/shared/burgers_train", "/data0/shared/pdebench_mirror"],
        )

    def test_canonicalize_artifact_id_ignores_run_name_test_token(self) -> None:
        canonical_id, spec = canonicalize_artifact_id(
            "fno-pdf",
            local_path="/data0/ziyi/PDEFlow_Runs/pde_codex_test_15/external_assets/fno.pdf",
            title="fno.pdf",
            artifact_type="paper",
        )
        self.assertEqual(canonical_id, "paper-fno")
        self.assertIsNone(spec.get("split"))


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

    def test_dataset_preparing_route_stays_in_acquisition(self) -> None:
        config = make_config(run_name="dataset-preparing")
        manager = ResearchManager(config=config, repo_root=Path("/root/PDEFlow"))
        state = ResearchState(
            project_name=config.project_name,
            run_name=config.run_name,
            work_directory=str(manager.work_directory),
            research_brief=config.research_brief,
            capability_matrix=CapabilityMatrix(
                repo_ready=True,
                env_ready=False,
                codepath_ready=True,
                target_dataset_ready=False,
                target_dataset_preparing=True,
                target_dataset_blocked=False,
                exact_target_shards_pending=["pdebench-dataset-burgers-train-nu-1"],
                baseline_ready_to_launch=False,
                scientific_iteration_ready=False,
            ),
        )
        route = manager._select_cycle_route(state)
        self.assertEqual(route.route_id, "continue-dataset-preparation")
        self.assertEqual(len(route.phases), 1)
        self.assertEqual(route.phases[0].phase.value, "acquisition")

    def test_environment_runtime_blocker_prefers_fallback_when_baseline_assets_are_ready(self) -> None:
        config = make_config(run_name="env-runtime")
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
                target_dataset_ready=True,
                torch_runtime_ready=True,
                gpu_runtime_required=True,
                gpu_runtime_ready=False,
                scientific_iteration_ready=True,
                baseline_launch_ready=False,
                environment_repair_needed=True,
            ),
            blocker_registry=[
                BlockerRecord(
                    blocker_id="env-blocker",
                    blocker_type="environment_runtime_failure",
                    target_entity="env:pdebench",
                    first_seen_cycle=1,
                    last_seen_cycle=1,
                    repeat_count=1,
                    last_attempt_signature="gpu_runtime_unavailable",
                    evidence_summary="cuda init failed",
                )
            ],
        )
        route = manager._select_cycle_route(state)
        self.assertEqual(route.route_id, "fallback-execution")
        state.route_history.append(
            RouteDecisionRecord(
                cycle_index=1,
                route_id="fallback-execution",
                rationale="fallback execution is immediately available",
            )
        )
        state.cycle_deltas = [
            CycleDeltaRecord(
                cycle_index=1,
                snapshot_signature="same",
                changed=False,
                summary=["no_material_state_change"],
            )
        ]
        route = manager._select_cycle_route(state)
        self.assertEqual(route.route_id, "fallback-execution")

    def test_acquisition_bootstrap_prefers_local_repository_before_remote_search(self) -> None:
        config = make_config(run_name="local-bootstrap")
        manager = ResearchManager(config=config, repo_root=Path("/root/PDEFlow"))
        state = ResearchState(
            project_name=config.project_name,
            run_name=config.run_name,
            work_directory=str(manager.work_directory),
            research_brief=ResearchBriefConfig(
                title="PDEBench Burgers research",
                question="Use PDEBench to study Burgers and ReactionDiffusion baselines.",
                domain_tags=["PDEBench", "FNO", "DeepONet"],
            ),
        )

        manager._bootstrap_local_repository = lambda state_arg, query_text: RepositoryRecord(  # type: ignore[method-assign]
            repo_id="local-pdebench",
            canonical_id="local-pdebench",
            name="pdebench",
            remote_url="local://pdebench",
            local_path="/tmp/pdebench",
            bootstrap_status="ready",
            environment_path="/tmp/envs/pdebench-env",
        )
        manager._seed_bootstrap_artifacts = lambda state_arg, repository: None  # type: ignore[method-assign]
        manager.tools.search_github_repositories = lambda *args, **kwargs: (_ for _ in ()).throw(  # type: ignore[method-assign]
            AssertionError("local bootstrap should not query GitHub first")
        )

        summary = manager._bootstrap_minimal_acquisition_recovery(state)
        self.assertIsNotNone(summary)
        self.assertIn("bootstrapped_repos=1", summary or "")


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
            self.assertEqual(state.experiment_plans[0].gpu_ids, [])

    def test_planner_replaces_gpu_only_plans_with_fallback_when_cuda_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "repo"
            repo_root.mkdir()
            config = make_config(run_name="fallback-override")
            memory = ResearchMemory(root=root)
            tools = ResearchTools(config=config, memory=memory, repo_root=Path("/root/PDEFlow"))
            state = ResearchState(
                project_name=config.project_name,
                run_name=config.run_name,
                work_directory=str(root),
                research_brief=config.research_brief,
                active_route_focus=["fallback_execution"],
                selected_baseline_program_id="prog-baseline",
                capability_matrix=CapabilityMatrix(
                    environment_path=str(root / "envs" / "pdebench-env"),
                    repo_ready=True,
                    env_ready=True,
                    codepath_ready=True,
                    baseline_launch_ready=False,
                    scientific_iteration_ready=True,
                    gpu_runtime_required=True,
                    gpu_runtime_ready=False,
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
            output = ExperimentPlanningPhaseOutput(
                summary="baseline only",
                experiment_plans=[
                    ExperimentPlan(
                        plan_id="baseline-gpu",
                        title="GPU baseline",
                        program_id="prog-baseline",
                        repo_id="github-pdebench-pdebench",
                        working_directory=str(repo_root),
                        launch_command="python train.py",
                        gpu_ids=[6],
                        log_path=str(root / "baseline.log"),
                    )
                ],
                next_actions=[],
            )
            ExperimentPlannerAgent().apply_output(state, tools, output)
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
                blocked_route = CycleRoute(
                    route_id="blocked-terminal",
                    phases=(PhaseSpec(ResearchPhase.REFLECTION, "reflection", ("reflections",)),),
                    reason="manual assistance required",
                    focus=("terminate_blocked",),
                )
                handled = manager._handle_hitl(state, route=blocked_route)
            self.assertTrue(handled)
            self.assertTrue(state.hitl_events)
            self.assertEqual(state.hitl_events[-1].status.value, "resumed")
            self.assertEqual(state.external_artifacts[0].status, "ready_for_training")

    def test_pending_dataset_preparation_does_not_trigger_hitl(self) -> None:
        config = make_config(run_name="hitl-pending")
        manager = ResearchManager(config=config, repo_root=Path("/root/PDEFlow"))
        state = ResearchState(
            project_name=config.project_name,
            run_name=config.run_name,
            work_directory=str(manager.work_directory),
            research_brief=config.research_brief,
            capability_matrix=CapabilityMatrix(
                repo_ready=True,
                env_ready=False,
                codepath_ready=True,
                target_dataset_ready=False,
                target_dataset_preparing=True,
                target_dataset_blocked=False,
                exact_target_shards_pending=["pdebench-dataset-burgers-train-nu-1"],
            ),
            external_artifacts=[
                ArtifactRecord(
                    artifact_id="pdebench_burgers_nu1_train",
                    canonical_id="pdebench-dataset-burgers-train-nu-1",
                    artifact_type="dataset",
                    title="burgers",
                    rationale="downloading",
                    local_path="/tmp/1D_Burgers_Sols_Nu1.0.hdf5",
                    status="checksum_verified",
                    validation=ArtifactValidationResult(
                        validator="hdf5",
                        status="checksum_verified",
                        exists=True,
                        ready_for_training=False,
                        size_bytes=1024,
                        size_ok=True,
                        format_valid=None,
                    ),
                )
            ],
            blocker_registry=[
                BlockerRecord(
                    blocker_id="blocker-a",
                    blocker_type="dataset_acquisition_failure",
                    target_entity="pdebench-dataset-burgers-train-nu-1",
                    first_seen_cycle=1,
                    last_seen_cycle=3,
                    repeat_count=3,
                    last_attempt_signature="direct",
                    evidence_summary="old state",
                    recovery_strategies_tried=["direct_remote_download", "local_discovery"],
                    route_exhausted=True,
                )
            ],
        )
        blocked_route = CycleRoute(
            route_id="blocked-terminal",
            phases=(PhaseSpec(ResearchPhase.REFLECTION, "reflection", ("reflections",)),),
            reason="manual assistance required",
            focus=("terminate_blocked",),
        )
        self.assertFalse(manager._handle_hitl(state, route=blocked_route))

    def test_hitl_does_not_trigger_when_autonomous_route_still_exists(self) -> None:
        config = make_config(run_name="hitl-autonomous-route")
        manager = ResearchManager(config=config, repo_root=Path("/root/PDEFlow"))
        state = ResearchState(
            project_name=config.project_name,
            run_name=config.run_name,
            work_directory=str(manager.work_directory),
            research_brief=config.research_brief,
            blocker_registry=[
                BlockerRecord(
                    blocker_id="blocker-dataset",
                    blocker_type="dataset_acquisition_failure",
                    target_entity="pdebench-dataset-burgers-train-nu-0p01",
                    first_seen_cycle=1,
                    last_seen_cycle=4,
                    repeat_count=4,
                    last_attempt_signature="direct_remote_download",
                    evidence_summary="transfer retries still ongoing",
                    recovery_strategies_tried=["local_discovery", "mirror_resolution"],
                    route_exhausted=True,
                    recommended_pivots=["local_discovery", "mirror_resolution", "partial_salvage", "fallback_execution"],
                )
            ],
        )
        non_terminal_route = CycleRoute(
            route_id="recover-partial-salvage",
            phases=(PhaseSpec(ResearchPhase.ACQUISITION, "acquisition", ("external_artifacts",)),),
            reason="Continue bounded salvage before involving a human.",
            focus=("partial_salvage",),
        )
        self.assertFalse(manager._handle_hitl(state, route=non_terminal_route))
        self.assertFalse(state.hitl_events)

    def test_hitl_reprints_updated_prompt_when_confirmation_stays_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            burgers_dir = root / "external_assets" / "data" / "1D" / "Burgers" / "Train"
            react_test_dir = root / "external_assets" / "data" / "1D" / "ReactionDiffusion" / "Test"
            react_train_dir = root / "external_assets" / "data" / "1D" / "ReactionDiffusion" / "Train"
            burgers_dir.mkdir(parents=True, exist_ok=True)
            react_test_dir.mkdir(parents=True, exist_ok=True)
            react_train_dir.mkdir(parents=True, exist_ok=True)
            (react_test_dir / "ReacDiff_react_Nu1.0_Rho1.0.hdf5").write_bytes(b"react-test")
            (react_train_dir / "ReacDiff_Nu1.0_Rho2.0.hdf5").write_bytes(b"react-train")

            config = make_config(run_name="hitl-reprompt")
            manager = ResearchManager(config=config, repo_root=Path("/root/PDEFlow"))
            manager.work_directory = root
            manager.memory = ResearchMemory(root=root)
            manager.tools = ResearchTools(config=config, memory=manager.memory, repo_root=Path("/root/PDEFlow"))

            def fake_refresh_capability_matrix(state: ResearchState) -> None:
                ready_artifacts = {item.canonical_id or item.artifact_id for item in state.external_artifacts if item.status == "ready_for_training"}
                state.capability_matrix = CapabilityMatrix(
                    repo_ready=True,
                    env_ready=True,
                    codepath_ready=True,
                    target_dataset_ready=False,
                    target_dataset_blocked=True,
                    baseline_ready_to_launch=False,
                    exact_target_shards_missing=["pdebench-dataset-burgers-train-nu-0p01"],
                    exact_target_shards_corrupted=[],
                    fallback_assets_available=bool(ready_artifacts),
                )

            manager._refresh_capability_matrix = fake_refresh_capability_matrix  # type: ignore[method-assign]

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
                        rationale="missing dataset",
                        local_path=str(burgers_dir / "1D_Burgers_Sols_Nu0.01.hdf5"),
                        status="blocked",
                        metadata={"min_size_bytes": 1, "required_keys": ["u"]},
                    ),
                    ArtifactRecord(
                        artifact_id="pdebench_reacdiff_test_rho1",
                        canonical_id="pdebench-dataset-reactiondiffusion-test-nu-1-rho-1",
                        artifact_type="dataset",
                        title="reacdiff test",
                        rationale="manual dataset",
                        local_path=str(react_test_dir / "ReacDiff_react_Nu1.0_Rho1.0.hdf5"),
                        status="blocked",
                        metadata={"min_size_bytes": 1, "required_keys": ["u"]},
                    ),
                    ArtifactRecord(
                        artifact_id="pdebench_reacdiff_train_rho2",
                        canonical_id="pdebench-dataset-reactiondiffusion-train-nu-1-rho-2",
                        artifact_type="dataset",
                        title="reacdiff train",
                        rationale="manual dataset",
                        local_path=str(react_train_dir / "ReacDiff_Nu1.0_Rho2.0.hdf5"),
                        status="blocked",
                        metadata={"min_size_bytes": 1, "required_keys": ["u"]},
                    ),
                ],
                blocker_registry=[
                    BlockerRecord(
                        blocker_id="blocker-burgers",
                        blocker_type="dataset_acquisition_failure",
                        target_entity="pdebench-dataset-burgers-train-nu-0p01",
                        first_seen_cycle=1,
                        last_seen_cycle=3,
                        repeat_count=3,
                        last_attempt_signature="direct_remote_download",
                        evidence_summary="repeated download failure",
                        recovery_strategies_tried=["direct_remote_download", "local_discovery"],
                        route_exhausted=True,
                    ),
                    BlockerRecord(
                        blocker_id="blocker-react-test",
                        blocker_type="dataset_acquisition_failure",
                        target_entity="pdebench-dataset-reactiondiffusion-test-nu-1-rho-1",
                        first_seen_cycle=1,
                        last_seen_cycle=3,
                        repeat_count=3,
                        last_attempt_signature="direct_remote_download",
                        evidence_summary="manual file expected",
                        recovery_strategies_tried=["direct_remote_download"],
                        route_exhausted=True,
                    ),
                    BlockerRecord(
                        blocker_id="blocker-react-train",
                        blocker_type="dataset_acquisition_failure",
                        target_entity="pdebench-dataset-reactiondiffusion-train-nu-1-rho-2",
                        first_seen_cycle=1,
                        last_seen_cycle=3,
                        repeat_count=3,
                        last_attempt_signature="partial_salvage",
                        evidence_summary="manual file expected",
                        recovery_strategies_tried=["partial_salvage"],
                        route_exhausted=True,
                    ),
                ],
            )

            with patch("builtins.input", side_effect=["1", "4"]), patch("tools.artifacts.h5py") as fake_h5py:
                fake_h5py.File.side_effect = lambda *args, **kwargs: _FakeH5File()
                blocked_route = CycleRoute(
                    route_id="blocked-terminal",
                    phases=(PhaseSpec(ResearchPhase.REFLECTION, "reflection", ("reflections",)),),
                    reason="manual assistance required",
                    focus=("terminate_blocked",),
                )
                handled = manager._handle_hitl(state, route=blocked_route)

            self.assertTrue(handled)
            self.assertEqual(state.hitl_events[-1].status.value, "aborted")
            statuses = {item.canonical_id or item.artifact_id: item.status for item in state.external_artifacts}
            self.assertEqual(statuses["dataset-reactiondiffusion-test-nu-1-rho-1"], "ready_for_training")
            self.assertEqual(statuses["dataset-reactiondiffusion-train-nu-1-rho-2"], "ready_for_training")
            self.assertEqual(statuses["dataset-burgers-train-nu-0p01"], "blocked")
            blocker_targets = {item.target_entity for item in state.blocker_registry if item.repeat_count >= 1}
            self.assertIn("dataset-burgers-train-nu-0p01", blocker_targets)
            self.assertNotIn("dataset-reactiondiffusion-test-nu-1-rho-1", blocker_targets)
            self.assertNotIn("dataset-reactiondiffusion-train-nu-1-rho-2", blocker_targets)
            process_log = manager.memory.process_path.read_text(encoding="utf-8")
            self.assertGreaterEqual(process_log.count("Human intervention required."), 2)
            self.assertGreaterEqual(process_log.count("1D_Burgers_Sols_Nu0.01.hdf5"), 2)

    def test_revalidate_after_human_confirmation_hydrates_checksum_rich_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            react_test_dir = root / "external_assets" / "datasets" / "pdebench" / "1D" / "ReactionDiffusion" / "Test"
            react_test_dir.mkdir(parents=True, exist_ok=True)
            dataset_path = react_test_dir / "ReacDiff_react_Nu1.0_Rho1.0.hdf5"
            data = b"small-official-test-shard"
            dataset_path.write_bytes(data)
            checksum = hashlib.md5(data).hexdigest()

            config = make_config(run_name="hitl-hydrate")
            manager = ResearchManager(config=config, repo_root=Path("/root/PDEFlow"))
            manager.work_directory = root
            manager.memory = ResearchMemory(root=root)
            manager.tools = ResearchTools(config=config, memory=manager.memory, repo_root=Path("/root/PDEFlow"))

            checksum_artifact = ArtifactRecord(
                artifact_id="doi:10.18419/darus-2986/1D/ReactionDiffusion/Test/ReacDiff_react_Nu1.0_Rho1.0.hdf5",
                canonical_id="doi:10.18419/darus-2986/1D/ReactionDiffusion/Test/ReacDiff_react_Nu1.0_Rho1.0.hdf5",
                artifact_type="dataset",
                title="ReacDiff_react_Nu1.0_Rho1.0.hdf5",
                rationale="official placeholder",
                local_path=str(dataset_path),
                status="download_failed",
                metadata={"official_checksum": checksum, "min_size_bytes": 1_000_000},
            )
            manager.memory.record_artifact(checksum_artifact)

            state = ResearchState(
                project_name=config.project_name,
                run_name=config.run_name,
                work_directory=str(root),
                research_brief=config.research_brief,
                external_artifacts=[
                    ArtifactRecord(
                        artifact_id="ReacDiff_react_Nu1.0_Rho1.0.hdf5",
                        canonical_id="pdebench-dataset-reactiondiffusion-test-nu-1-rho-1",
                        artifact_type="dataset",
                        title="human provided",
                        rationale="manual copy",
                        local_path=str(dataset_path),
                        status="blocked",
                        metadata={"human_provided": True, "min_size_bytes": 1_000_000},
                    )
                ],
                blocker_registry=[
                    BlockerRecord(
                        blocker_id="blocker-react-test",
                        blocker_type="dataset_acquisition_failure",
                        target_entity="pdebench-dataset-reactiondiffusion-test-nu-1-rho-1",
                        first_seen_cycle=1,
                        last_seen_cycle=3,
                        repeat_count=3,
                        last_attempt_signature="direct_remote_download",
                        evidence_summary="manual file expected",
                        recovery_strategies_tried=["direct_remote_download"],
                        route_exhausted=True,
                    )
                ],
            )

            manager._refresh_capability_matrix = lambda state_obj: setattr(  # type: ignore[method-assign]
                state_obj,
                "capability_matrix",
                CapabilityMatrix(
                    repo_ready=True,
                    env_ready=True,
                    codepath_ready=True,
                    target_dataset_ready=True,
                    target_dataset_blocked=False,
                    baseline_launch_ready=False,
                ),
            )

            with patch("tools.artifacts.h5py") as fake_h5py:
                fake_h5py.File.side_effect = lambda *args, **kwargs: _FakeH5File()
                summary = manager._revalidate_after_human_confirmation(state, list(state.blocker_registry))

            self.assertIn("validated existing file ReacDiff_react_Nu1.0_Rho1.0.hdf5", summary)
            self.assertEqual(len(state.external_artifacts), 1)
            self.assertEqual(state.external_artifacts[0].status, "ready_for_training")
            self.assertEqual(
                state.external_artifacts[0].metadata.get("official_checksum"),
                checksum,
            )


if __name__ == "__main__":
    unittest.main()
