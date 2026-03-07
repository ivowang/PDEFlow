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
uv run python app.py --config configs/research_problem.openrouter.json
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

## Workflow

The manager runs these phases:

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

## Outputs

Run artifacts are written to `runs/<run_name>/`:

- `state/`: structured state snapshots
- `logs/`: command logs and tool events
- `memory/`: episodic, semantic, and idea memory
- `literature/`: paper notes
- `programs/`: program lineage database
- `experiments/`: experiment records and logs
- `reports/`: generated markdown reports
- `workspaces/`: child program workspaces

Shared external assets are stored in `external_assets/`.

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
- The current repository is live-only. There is no mock runtime.
- The system uses real shell commands, downloads, repo cloning, and environment setup, so it should be run on a controlled research machine.

## Paper Draft

- [pdeflow_system.tex](docs/pdeflow_system.tex)
- [references.bib](docs/references.bib)
