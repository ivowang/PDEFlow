from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pydantic import BaseModel

from research_agents import ExperimentAgent
from config import ResearchBriefConfig, RuntimeConfig, SystemConfig
from memory import ResearchMemory
from orchestration import ResearchManager
from runtime import RuntimeAdapter
from state import ExperimentPhaseOutput, ExperimentPlan, ExperimentRecord, ResearchState
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
        )
        route = manager._select_cycle_route(state)
        self.assertEqual(route.phases[0].phase.value, "acquisition")
        self.assertEqual(route.phases[1].phase.value, "experiment_planning")
        self.assertIn("Hard external blocker", route.reason)


if __name__ == "__main__":
    unittest.main()
