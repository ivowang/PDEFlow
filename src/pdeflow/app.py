from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from .config import SystemConfig
from .orchestration import ResearchManager


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the autonomous research workflow.")
    parser.add_argument(
        "--config",
        default="configs/research_problem.json",
        help="Path to the JSON config file.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional override for the run directory name.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    config = SystemConfig.from_file(repo_root / args.config)
    if args.run_name:
        config.run_name = args.run_name
    manager = ResearchManager(config=config, repo_root=repo_root)
    state = manager.run()
    print(f"Run completed: {state.run_name}")
    print(f"Output root: {repo_root / config.output_root / config.run_name}")
    print(f"Cycles completed: {state.cycle_index}")
    print(f"Repositories discovered: {len(state.repositories)}")
    print(f"Programs tracked: {len(state.program_candidates)}")
    print(f"Experiments recorded: {len(state.experiment_records)}")
    print(f"Reports written: {len(state.generated_reports)}")
    if state.termination_decision:
        print(f"Termination: {state.termination_decision}")
    return 0
