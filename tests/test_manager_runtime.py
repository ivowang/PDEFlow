from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import BaseModel

from research_agents import ExperimentAgent, PreflightValidationAgent
from config import ResearchBriefConfig, RuntimeConfig, SystemConfig
from memory import ResearchMemory
from orchestration import ResearchManager
from orchestration.failures import classify_state_failures
from runtime import RuntimeAdapter
from state import (
    ArtifactDownloadMetadata,
    ArtifactRecord,
    ArtifactStatus,
    ArtifactValidationResult,
    CapabilityMatrix,
    ExperimentPhaseOutput,
    ExperimentPlan,
    ExperimentRecord,
    ResearchState,
)
from tools import ResearchTools


def make_config(run_name: str = "test-run") -> SystemConfig:
    return SystemConfig(
        project_name="test-project",
        run_name=run_name,
        research_brief=ResearchBriefConfig(
            title="Test brief",
            question="Can the system recover from blockers?",
        ),
        runtime=RuntimeConfig(provider="openrouter", model="openai/gpt-5.4"),
    )


class DummyOutput(BaseModel):
    summary: str
    count: int


class RuntimeAdapterTests(unittest.TestCase):
    def test_prompt_json_validation_uses_repair_fallback(self) -> None:
        adapter = RuntimeAdapter(RuntimeConfig(provider="openrouter"), session_db_path=":memory:")
        adapter._repair_prompt_json_output = lambda raw_text, output_type, error_message: output_type(  # type: ignore[method-assign]
            summary="repaired",
            count=1,
        )
        result = adapter._validate_prompt_json_output('{"summary":"broken" "count":1}', DummyOutput)
        self.assertEqual(result.summary, "repaired")
        self.assertEqual(result.count, 1)


class ExperimentPlanStatusTests(unittest.TestCase):
    def test_setup_failed_plan_is_blocked_not_completed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_config()
            memory = ResearchMemory(root=root)
            tools = ResearchTools(config=config, memory=memory, repo_root=Path("/root/PDEFlow"))
            state = ResearchState(
                project_name=config.project_name,
                run_name=config.run_name,
                work_directory=str(root),
                research_brief=config.research_brief,
                experiment_plans=[
                    ExperimentPlan(
                        plan_id="plan-1",
                        title="baseline",
                        program_id="prog-1",
                        working_directory=str(root),
                        launch_command="python train.py",
                        log_path=str(root / "logs" / "plan-1.log"),
                    )
                ],
            )
            output = ExperimentPhaseOutput(
                summary="setup failed",
                experiment_records=[
                    ExperimentRecord(
                        experiment_id="exp-1",
                        plan_id="plan-1",
                        program_id="prog-1",
                        command="python train.py",
                        working_directory=str(root),
                        status="setup_failed",
                        failure_modes=["required local dataset files were absent"],
                        log_path=str(root / "logs" / "exp-1.log"),
                    )
                ],
                failure_summaries=["required local dataset files were absent"],
            )
            summary = ExperimentAgent().apply_output(state, tools, output)
            self.assertIn("setup failed", summary)
            self.assertEqual(state.experiment_plans[0].status, "blocked")


class ManagerRoutingTests(unittest.TestCase):
    def test_manager_routes_to_recovery_when_dataset_blocker_detected(self) -> None:
        config = make_config(run_name="routing-test")
        manager = ResearchManager(config=config, repo_root=Path("/root/PDEFlow"))
        state = ResearchState(
            project_name=config.project_name,
            run_name=config.run_name,
            work_directory=str(manager.work_directory),
            research_brief=config.research_brief,
            failure_summaries=[
                "Both setup commands failed before launch because the required local HDF5 dataset files were still absent."
            ],
            experiment_records=[
                ExperimentRecord(
                    experiment_id="exp-blocked",
                    plan_id="plan-blocked",
                    program_id="prog-baseline",
                    command="python train.py",
                    working_directory=str(manager.work_directory),
                    status="setup_failed",
                    failure_modes=[
                        "expected local dataset paths under /data0/ziyi/PDEFlow_Runs/pde_4/data do not exist"
                    ],
                    log_path=str(manager.work_directory / "logs" / "exp-blocked.log"),
                )
            ],
            capability_matrix=CapabilityMatrix(
                fno_ready=True,
                target_dataset_ready=False,
                target_dataset_blocked=True,
                exact_target_shards_missing=["/tmp/missing.hdf5"],
                baseline_ready_to_launch=False,
            ),
        )
        route = manager._select_cycle_route(state)
        self.assertEqual(route.phases[0].phase.value, "acquisition")
        self.assertEqual(route.phases[1].phase.value, "experiment_planning")
        self.assertEqual(route.phases[2].phase.value, "preflight_validation")
        self.assertIn("baseline route is not launch-ready", route.reason)


class PreflightTests(unittest.TestCase):
    def test_preflight_blocks_plan_with_non_ready_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "repo").mkdir()
            (root / "repo" / "train.py").write_text("print('ok')\n", encoding="utf-8")
            config = make_config()
            memory = ResearchMemory(root=root)
            tools = ResearchTools(config=config, memory=memory, repo_root=Path("/root/PDEFlow"))
            state = ResearchState(
                project_name=config.project_name,
                run_name=config.run_name,
                work_directory=str(root),
                research_brief=config.research_brief,
                experiment_plans=[
                    ExperimentPlan(
                        plan_id="plan-1",
                        title="baseline",
                        program_id="prog-1",
                        working_directory=str(root / "repo"),
                        launch_command="python train.py",
                        required_artifact_ids=["dataset-1"],
                        log_path=str(root / "logs" / "plan.log"),
                    )
                ],
                external_artifacts=[
                    ArtifactRecord(
                        artifact_id="dataset-1",
                        artifact_type="dataset",
                        title="bad shard",
                        rationale="test",
                        local_path=str(root / "bad.hdf5"),
                        status="corrupted",
                        validation=ArtifactValidationResult(
                            validator="hdf5",
                            status=ArtifactStatus.CORRUPTED,
                            exists=True,
                            size_bytes=128,
                            min_size_bytes=1024,
                            size_ok=False,
                            format_valid=False,
                            ready_for_training=False,
                            failure_reasons=["size_below_minimum:128<1024"],
                        ),
                    )
                ],
                capability_matrix=CapabilityMatrix(environment_path=None),
            )
            summary = PreflightValidationAgent().run(state, tools, runtime=None)
            self.assertIn("blocked=1", summary)
            self.assertEqual(state.experiment_plans[0].status, "blocked")
            self.assertEqual(state.preflight_reports[0].recommended_route, "acquisition")


class FailureClassificationTests(unittest.TestCase):
    def test_state_failure_classification_marks_repeated_repair_failure(self) -> None:
        state = ResearchState(
            project_name="proj",
            run_name="run",
            work_directory="/tmp",
            research_brief=ResearchBriefConfig(title="x", question="y"),
            external_artifacts=[
                ArtifactRecord(
                    artifact_id="dataset-1",
                    artifact_type="dataset",
                    title="shard",
                    rationale="test",
                    local_path="/tmp/bad.hdf5",
                    status="corrupted",
                    download_metadata=ArtifactDownloadMetadata(
                        source_url="https://example.com/file.hdf5",
                        local_path="/tmp/bad.hdf5",
                        attempt_count=3,
                        failure_type="transfer_stalled",
                    ),
                    validation=ArtifactValidationResult(
                        validator="hdf5",
                        status=ArtifactStatus.CORRUPTED,
                        exists=True,
                        size_bytes=100,
                        min_size_bytes=1024,
                        size_ok=False,
                        format_valid=False,
                        ready_for_training=False,
                        failure_reasons=["size_below_minimum:100<1024"],
                    ),
                )
            ],
        )
        failures = classify_state_failures(state, max_transfer_attempts=3)
        failure_types = {item.failure_type for item in failures}
        self.assertIn("transfer_stalled", failure_types)
        self.assertIn("repeated_repair_failure", failure_types)


if __name__ == "__main__":
    unittest.main()
