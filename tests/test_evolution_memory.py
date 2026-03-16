from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from config import ResearchBriefConfig, RuntimeConfig, SystemConfig
from memory import ResearchMemory
from orchestration import ResearchManager
from research_agents import ExperimentAgent, PreflightValidationAgent, ReflectionAgent
from state import (
    ExperimentPhaseOutput,
    ExperimentRecord,
    PreflightCheckResult,
    PreflightReport,
    PreflightValidationPhaseOutput,
    ReflectionPhaseOutput,
    ReflectionRecord,
    ResearchState,
)
from tools import ResearchTools


def make_config(run_name: str = "evolution-test") -> SystemConfig:
    return SystemConfig(
        project_name="test-project",
        run_name=run_name,
        research_brief=ResearchBriefConfig(title="Evolution test", question="Can memory accumulate useful lessons?"),
        runtime=RuntimeConfig(provider="openrouter", model="openai/gpt-5.4"),
    )


class EvolutionMemoryTests(unittest.TestCase):
    def test_preflight_writes_evaluation_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_config("preflight-memory")
            memory = ResearchMemory(root=root)
            tools = ResearchTools(config=config, memory=memory, repo_root=Path("/root/PDEFlow"))
            state = ResearchState(
                project_name=config.project_name,
                run_name=config.run_name,
                work_directory=str(root),
                research_brief=config.research_brief,
            )
            output = PreflightValidationPhaseOutput(
                summary="preflight",
                preflight_reports=[
                    PreflightReport(
                        report_id="preflight-1",
                        plan_id="plan-1",
                        program_id="baseline-1",
                        passed=False,
                        failed_checks=[
                            PreflightCheckResult(
                                name="dataset_readability",
                                passed=False,
                                details="dataset file missing",
                                category="dataset",
                            )
                        ],
                        blocking_reason="dataset file missing",
                        recommended_route="acquisition",
                    )
                ],
                failure_summaries=["dataset file missing"],
            )
            PreflightValidationAgent().apply_output(state, tools, output)
            self.assertEqual(len(state.evaluation_memos), 1)
            self.assertTrue(Path(state.evaluation_memos[0].path).exists())
            self.assertTrue(any(note.kind.value == "evaluations" for note in state.memory_notes))

    def test_experiment_writes_evaluation_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_config("experiment-memory")
            memory = ResearchMemory(root=root)
            tools = ResearchTools(config=config, memory=memory, repo_root=Path("/root/PDEFlow"))
            state = ResearchState(
                project_name=config.project_name,
                run_name=config.run_name,
                work_directory=str(root),
                research_brief=config.research_brief,
                selected_baseline_program_id="baseline-1",
                best_known_results={"baseline-1": {"rmse": 0.25}},
            )
            output = ExperimentPhaseOutput(
                summary="experiment done",
                experiment_records=[
                    ExperimentRecord(
                        experiment_id="exp-1",
                        plan_id="plan-1",
                        program_id="candidate-1",
                        command="python train.py",
                        working_directory=str(root),
                        status="completed",
                        metrics={"rmse": 0.2},
                        log_path=str(root / "exp.log"),
                    )
                ],
            )
            ExperimentAgent().apply_output(state, tools, output)
            self.assertEqual(len(state.evaluation_memos), 1)
            self.assertEqual(state.evaluation_memos[0].verdict, "improved")
            self.assertTrue(Path(state.evaluation_memos[0].path).exists())

    def test_reflection_writes_lesson_strategy_and_evolution_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_config("reflection-memory")
            memory = ResearchMemory(root=root)
            tools = ResearchTools(config=config, memory=memory, repo_root=Path("/root/PDEFlow"))
            state = ResearchState(
                project_name=config.project_name,
                run_name=config.run_name,
                work_directory=str(root),
                research_brief=config.research_brief,
            )
            output = ReflectionPhaseOutput(
                summary="reflection",
                reflections=[
                    ReflectionRecord(
                        reflection_id="reflection-1",
                        cycle_index=2,
                        verdict="candidate improved baseline on short-window rmse",
                        evidence=["rmse improved from 0.25 to 0.20"],
                        accepted_lessons=["Physics-aware residual gating improved short-window error without destabilizing training."],
                        next_actions=["Promote residual gating into the next design iteration."],
                        preferred_recovery_strategies=["keep_method_direction"],
                        forbidden_attempt_signatures=["retry-plain-baseline"],
                        continue_research=True,
                    )
                ],
            )
            ReflectionAgent().apply_output(state, tools, output)
            kinds = {note.kind.value for note in state.memory_notes}
            self.assertIn("reflections", kinds)
            self.assertIn("lessons", kinds)
            self.assertIn("strategy", kinds)
            self.assertIn("evolution", kinds)

    def test_manager_uses_merged_agent_roles(self) -> None:
        config = make_config("manager-agent-map")
        manager = ResearchManager(config=config, repo_root=Path("/root/PDEFlow"))
        self.assertEqual(manager.agents["problem"].name, "ResearchStrategistAgent")
        self.assertEqual(manager.agents["diagnosis"].name, "ResearchStrategistAgent")
        self.assertEqual(manager.agents["hypothesis"].name, "ResearchStrategistAgent")
        self.assertEqual(manager.agents["design"].name, "ResearchStrategistAgent")
        self.assertEqual(manager.agents["coder"].name, "EngineeringAgent")
        self.assertEqual(manager.agents["planner"].name, "EngineeringAgent")
        self.assertEqual(manager.agents["preflight"].name, "EvaluationAgent")
        self.assertEqual(manager.agents["experiment"].name, "EvaluationAgent")
        self.assertEqual(manager.agents["reflection"].name, "EvaluationAgent")

