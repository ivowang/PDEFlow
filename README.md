# PDEFlow

`PDEFlow` is a manager-centered autonomous research system for PDE neural operator research.

You provide a research problem in config. The system then autonomously runs literature review, asset acquisition, planning, preflight validation, experiments, reflection, human-in-the-loop escalation when needed, and reporting with real tools.

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
9. `preflight_validation`
10. `experiment`
11. `reflection`
12. `human_intervention`
13. `reporting`

The design is manager-centered. Specialist agents do not freely chat with each other; they operate through shared state and tools.

The phase vocabulary is fixed, but the cycle route is dynamic:

- normal route: `hypothesis -> method_design -> coding -> experiment_planning -> preflight_validation -> experiment -> reflection`
- recovery route: if the manager detects a hard blocker such as corrupted data, blocked artifacts, failed transfers, missing checkpoints, repo/bootstrap failure, or invalid environment state, it routes back through `acquisition -> experiment_planning -> preflight_validation -> reflection`
- HITL route: if the same blocker persists after repeated autonomous recovery attempts, the manager enters `human_intervention`, prints an actionable terminal request, waits for human input, and only resumes after the response materially changes state

This prevents the system from continuing with useless downstream steps after a prerequisite has failed.

Two reliability rules are now hard-gated:

- dataset and checkpoint artifacts must be `ready_for_training` before planning or launch
- real experiment launch only happens after `preflight_validation` passes for the exact plan

If a blocker is exhausted rather than recoverable, PDEFlow does not silently retry forever. It can escalate to HITL, asking the user to:

- confirm that files have been manually provided
- give a new local path or alternate instruction
- skip a blocked target and continue with reduced scope
- abort the run

Manual confirmation is never trusted blindly. The system re-scans and re-validates files before resuming.

## Outputs

Run artifacts are written under `execution.work_directory`:

- `state/`: structured state snapshots
- `logs/`: unified logging system output
- `memory/`: episodic, semantic, and idea memory
- `literature/`: paper notes
- `programs/`: program lineage database
- `artifacts/`: artifact registry with validation/checksum metadata
- `preflight/`: preflight reports for concrete experiment plans
- `memory/hitl_events.jsonl`: structured human-intervention requests, responses, and re-validation outcomes
- `experiments/`: experiment records and outputs
- `reports/`: generated markdown reports
- `workspaces/`: child program workspaces
- `envs/`: managed Python environments created by the system
- `pythons/`: managed Python interpreters downloaded by `uv` when needed
- `quarantine/`: corrupted artifacts moved out of the training path

Downloaded datasets, cloned repositories, checkpoints, and other acquired assets are stored under
`<work_directory>/<workspace_root>/`.

The intended convention is that all research content for a run stays inside the configured
`work_directory`, including:

- downloaded assets
- generated code
- experiment logs and outputs
- state snapshots and memory
- research reports

The unified logging system maintains three main granularities:

- `logs/core_progress.log` and `logs/core_progress.jsonl`: high-signal scientific milestones only, such as a new hypothesis, a concrete method design, an experiment result, or a meaningful reflection outcome
- `logs/agent_activity.log` and `logs/agent_activity.jsonl`: per-agent start/end/failure records with the agent’s phase-level content
- `logs/debug.log` and `logs/debug.jsonl`: the full execution trace, including routing rationale, tool activity, command progress, and failure details

Additional structured streams remain under `logs/` where useful, such as:

- `logs/tool_events.jsonl`
- `logs/phase_events.jsonl`
- `logs/commands/cmd-*.log` for full command stdout/stderr capture

`<work_directory>/process.txt` is now a compatibility mirror of the debug stream. It is still printed to the terminal during execution, but it is managed by the same unified logging system rather than by separate ad hoc writes.

## Package Layout

- [config/](src/config): runtime and execution config schema
- [state/](src/state): entities, global research state, and phase output models
- [tools/](src/tools): executable tool surface, split into inspection, retrieval, workspace, Python runtime discovery, managed environments, execution, and reporting modules
- [research_agents/](src/research_agents): base agent class plus discovery, analysis, execution, and reporting specialists
- [orchestration/](src/orchestration): manager loop and cycle routing
- [memory/](src/memory): JSONL and SQLite-backed memory
- [runtime/](src/runtime): provider resolution, structured-output handling, and OpenAI Agents SDK adapter
- [integrations/](src/integrations): generic repository/runtime integration helpers

## Notes

- `runtime.provider=openai` requires `OPENAI_API_KEY`.
- `runtime.provider=openrouter` requires `OPENROUTER_API_KEY`.
- OpenRouter is wired through the OpenAI-compatible path in the Agents SDK and defaults to `chat_completions`.
- Under OpenRouter, structured phase outputs use schema-guided JSON plus a repair fallback so a malformed model JSON response does not immediately crash the phase.
- Experiment plans are grounded against `ready_for_training` local artifacts by inspecting repository entrypoints and configs; this is implemented as a generic capability, not a benchmark-specific adapter.
- Artifact validation includes file existence, minimum-size checks, optional official checksum comparison, and HDF5 format validation. Corrupted files can be quarantined automatically.
- Large downloads use `.part` files, bounded retry/resume behavior, and post-download validation instead of treating path existence as success.
- The capability matrix and classified failure state are persisted in run state so repeated cycles do not rediscover the same blockers from scratch.
- Repeated unresolved blockers can trigger first-class human-in-the-loop escalation in the terminal; the manager records the request, waits for input, and consumes the response on the next step instead of continuing silent retry loops.
- Acquisition/repair work, preflight checks, and real experiments are tracked separately so experiment history is not polluted by data-repair attempts.
- Logging is unified under one logger that emits core scientific progress logs, agent activity logs, and full debug traces while still mirroring the live stream to `process.txt`.
- The current repository is live-only. There is no mock runtime.
- The system can create and repair managed `uv` environments under the run work directory instead of relying on a pre-existing project `venv`.
- The system uses real shell commands, downloads, repo cloning, and environment setup, so it should be run on a controlled research machine.
- `execution.workspace_root` is enforced to live inside `execution.work_directory`.
