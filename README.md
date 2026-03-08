# PDEFlow

`PDEFlow` is a manager-centered autonomous research system for PDE neural operator research.

You provide a research problem in config. The system then autonomously runs literature review, asset acquisition, method design, coding, experiments, reflection, and reporting with real tools.

## Quick Start

1. Install dependencies with `uv`:

```bash
uv sync
```

2. Create `.env` and fill the model API key you want to use:

```bash
cp .env.example .env
```

Choose one:

- `OPENAI_API_KEY` for direct OpenAI access
- `OPENROUTER_API_KEY` for OpenRouter access

Optional:

- `GITHUB_TOKEN` for higher GitHub API rate limits
- `OPENROUTER_SITE_URL` and `OPENROUTER_APP_NAME` for OpenRouter metadata headers

3. Edit the research problem in [research_problem.json](configs/research_problem.json).

Important fields in `execution`:

- `work_directory`: the root working directory for a research run
- `workspace_root`: subdirectory inside `work_directory` used for downloaded assets and cloned repos

4. Run the system:

```bash
uv run python app.py --config configs/research_problem.json
```

Or:

```bash
uv run pdeflow --config configs/research_problem.json
```

To override the run name:

```bash
uv run python app.py --config configs/research_problem.json --run-name pde_round1
```

OpenRouter example:

```bash
uv run python app.py --config configs/pde.openrouter.json
```

## What The User Provides

The intended user input is only:

- a research question
- execution constraints
- optional API keys

The system is designed to decide for itself:

- which papers to search
- which repositories to inspect
- which datasets or checkpoints to acquire
- how to bootstrap environments
- which baseline to build on
- which hypotheses to test
- which experiments to run
- when to continue or stop

## How It Runs

The manager uses these phases:

1. `literature_review`
2. `acquisition`
3. `problem_framing`
4. `diagnosis`
5. `hypothesis`
6. `method_design`
7. `coding`
8. `experiment_planning`
9. `experiment`
10. `reflection`
11. `reporting`

The design is manager-centered. Specialist agents do not freely chat with each other; they operate through shared state and tools.

The phase vocabulary is fixed, but the cycle route is dynamic:

- normal route: `hypothesis -> method_design -> coding -> experiment_planning -> experiment -> reflection`
- recovery route: if the manager detects a hard blocker such as missing data, missing checkpoints, repo/bootstrap failure, or invalid environment state, it routes back through `acquisition -> experiment_planning -> experiment -> reflection`

This prevents the system from continuing with useless downstream steps after a prerequisite has failed.

## Outputs

Run artifacts are written under `execution.work_directory`:

- `state/`: structured state snapshots
- `logs/`: command logs and tool events
- `memory/`: episodic, semantic, and idea memory
- `literature/`: paper notes
- `programs/`: program lineage database
- `experiments/`: experiment records and logs
- `reports/`: generated markdown reports
- `workspaces/`: child program workspaces
- `envs/`: managed Python environments created by the system
- `pythons/`: managed Python interpreters downloaded by `uv` when needed

Downloaded datasets, cloned repositories, checkpoints, and other acquired assets are stored under
`<work_directory>/<workspace_root>/`.

The intended convention is that all research content for a run stays inside the configured
`work_directory`, including:

- downloaded assets
- generated code
- experiment logs and outputs
- state snapshots and memory
- research reports

Live progress is also written to `<work_directory>/process.txt` and printed to the terminal during execution.

## Key Files

- [config.py](src/pdeflow/config.py): config schema
- [schemas.py](src/pdeflow/schemas.py): state and memory schema
- [tools.py](src/pdeflow/tools.py): executable tool surface
- [agents.py](src/pdeflow/agents.py): specialist agents
- [orchestration.py](src/pdeflow/orchestration.py): manager loop
- [memory.py](src/pdeflow/memory.py): JSONL and SQLite memory
- [runtime.py](src/pdeflow/runtime.py): OpenAI Agents SDK runtime adapter

## Notes

- `runtime.provider=openai` requires `OPENAI_API_KEY`.
- `runtime.provider=openrouter` requires `OPENROUTER_API_KEY`.
- OpenRouter is wired through the OpenAI-compatible path in the Agents SDK and defaults to `chat_completions`.
- Under OpenRouter, structured phase outputs use schema-guided JSON plus a repair fallback so a malformed model JSON response does not immediately crash the phase.
- The current repository is live-only. There is no mock runtime.
- The system can create and repair managed `uv` environments under the run work directory instead of relying on a pre-existing project `venv`.
- The system uses real shell commands, downloads, repo cloning, and environment setup, so it should be run on a controlled research machine.
- `execution.workspace_root` is enforced to live inside `execution.work_directory`.
