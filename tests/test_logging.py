from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from config import ResearchBriefConfig, RuntimeConfig, SystemConfig
from memory import ResearchMemory
from tools import ResearchTools
from state import ResearchPhase


def make_config(run_name: str = "logging-test") -> SystemConfig:
    return SystemConfig(
        project_name="test-project",
        run_name=run_name,
        research_brief=ResearchBriefConfig(title="Test brief", question="Can logs be unified?"),
        runtime=RuntimeConfig(provider="openrouter", model="openai/gpt-5.4"),
    )


class UnifiedLoggingTests(unittest.TestCase):
    def test_unified_logger_writes_process_core_agent_and_debug_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            memory = ResearchMemory(root=root)
            memory.record_process("debug message", print_to_terminal=False)
            memory.record_agent_event(
                agent_name="AcquisitionAgent",
                phase=ResearchPhase.ACQUISITION,
                status="completed",
                cycle_index=2,
                content="Acquired repository and dataset metadata.",
            )
            memory.record_core_progress(
                "Proposed a new physics-informed residual-gating idea.",
                kind="hypothesis",
                phase=ResearchPhase.HYPOTHESIS,
                agent_name="HypothesisAgent",
                cycle_index=2,
            )
            memory.record_tool_event({"tool": "search_github_repositories", "query": "PDEBench", "count": 1})

            process_text = (root / "process.txt").read_text(encoding="utf-8")
            self.assertIn("debug message", process_text)

            core_log = (root / "logs" / "core_progress.log").read_text(encoding="utf-8")
            self.assertIn("physics-informed residual-gating idea", core_log)

            agent_log = (root / "logs" / "agent_activity.log").read_text(encoding="utf-8")
            self.assertIn("AcquisitionAgent", agent_log)
            self.assertIn("Acquired repository and dataset metadata.", agent_log)

            debug_log = (root / "logs" / "debug.log").read_text(encoding="utf-8")
            self.assertIn("debug message", debug_log)

            tool_events = (root / "logs" / "tool_events.jsonl").read_text(encoding="utf-8")
            self.assertIn("search_github_repositories", tool_events)

    def test_command_logs_are_grouped_under_commands_subdirectory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            memory = ResearchMemory(root=root)
            tools = ResearchTools(config=make_config(), memory=memory, repo_root=root)

            result = tools.run_command("python3 -c \"print('ok')\"", cwd=root, emit_progress=False)

            log_path = Path(str(result["log_path"]))
            self.assertTrue(log_path.exists())
            self.assertEqual(log_path.parent, root / "logs" / "commands")
            self.assertIn("ok", log_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
