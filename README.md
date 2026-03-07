# PDEFlow

`PDEFlow` is a manager-centered autonomous research system for PDE neural operator research.

The user provides a research problem in config. The system is then responsible for running the full research loop:

- retrieving and organizing literature
- discovering repositories, datasets, checkpoints, and technical documentation
- bootstrapping runnable environments with `uv`
- selecting and tracking baseline programs
- proposing literature-grounded hypotheses
- translating hypotheses into concrete method designs
- creating child program variants
- planning and executing experiments
- parsing results and reflecting on failures
- maintaining long-horizon research memory
- generating research reports

This repository is not a chatbot wrapper and not a benchmark-specific hyperparameter tuner. It is an executable research workflow system built around structured state, tool use, and iterative program evolution.

## Overview

The initial landing problem is PDE neural operator research for AI4Science-style settings, especially scenarios where the system must autonomously reason from a high-level task description to real assets, code, and experiments. The architecture is intentionally designed so the same workflow can later be reused for broader scientific ML tasks.

Key properties:

- manager-centered orchestration rather than free agent-to-agent chat
- explicit workflow phases rather than implicit conversational drift
- external state and local memory rather than hidden private agent memory
- executable tools rather than fake reasoning about repos, datasets, or experiments
- idea-level iteration rather than only local baseline tuning
- program lineage tracking to support self-evolving research loops

## What It Is And Is Not

`PDEFlow` is:

- an autonomous research systems scaffold
- a structured multi-agent runtime on top of the OpenAI Agents SDK
- a local memory and program-database system for research iteration
- a framework for literature-grounded hypothesis generation and experimental follow-through

`PDEFlow` is not:

- a general chat assistant
- a static benchmark runner with hardcoded PDEBench assumptions
- a pure optimizer sweep engine
- a toy multi-agent conversation simulator
- a no-op placeholder pipeline with mock assets

## System Goals

The repository is built for the following target behavior:

1. The user describes a scientific research problem in config.
2. The system decides which papers, repos, datasets, checkpoints, and docs it needs.
3. The system acquires and inspects those assets itself.
4. The system identifies bottlenecks and proposes method-level hypotheses.
5. The system implements child program candidates in isolated workspaces.
6. The system executes experiments, parses outcomes, and records lineage.
7. The system reflects on results and decides whether to continue another cycle.
8. The system writes durable reports from structured state.

## Design Principles

### 1. Manager-Centered Orchestration

The manager controls the workflow. Specialist agents do not freely talk to each other. They operate against shared state and a tool surface.

### 2. Explicit State Machines

Research is represented as named phases. Each phase has a clear output contract and updates shared state.

### 3. External State Over Hidden Memory

Long-lived research state is stored in JSONL, JSON, and SQLite so that the system can be inspected, resumed, audited, and extended.

### 4. Tool-Driven Research

Agents must use tools to inspect reality:

- search literature
- clone repos
- bootstrap environments
- inspect file trees
- run commands
- parse metrics
- write reports

### 5. Program Evolution As A First-Class Object

The system tracks parent programs, child programs, patches, experiments, and outcomes. This is the basis for future self-evolving code search.

### 6. Distinguish Method Innovation From Tuning

The workflow explicitly represents candidate directions, hypotheses, and method designs so the system can separate new methods from pure engineering sweeps.

## Workflow

The workflow is implemented as a phase/state machine with these phases:

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

The first four phases build context and execution readiness. The next six phases form an iterative research cycle. The final phase writes research artifacts to disk.

### Phase Semantics

- `literature_review`: search for relevant papers, extract limitations, build taxonomy, and identify open questions
- `acquisition`: inspect the machine, check secrets, discover and clone repos, download assets, and bootstrap environments
- `problem_framing`: define the concrete research target, evaluation criteria, and candidate directions
- `diagnosis`: identify the highest-value bottlenecks from literature, code structure, and existing experiment evidence
- `hypothesis`: propose testable, method-level research hypotheses
- `method_design`: convert a hypothesis into implementable architectural, loss, data, training, or inference changes
- `coding`: create a child workspace, modify code, and run validation or smoke checks
- `experiment_planning`: construct actual execution plans, commands, expected outputs, and stopping rules
- `experiment`: run commands, capture logs, parse metrics, and record failures
- `reflection`: assess whether the latest results justify continuation, branching, or termination
- `reporting`: write literature notes, acquisition reports, idea diaries, experiment summaries, and final reports

## Agent Roles

The current system includes the following specialist agents:

- `LiteratureAgent`
- `AcquisitionAgent`
- `ProblemFramingAgent`
- `DiagnosisAgent`
- `HypothesisAgent`
- `MethodDesignAgent`
- `CoderAgent`
- `ExperimentPlannerAgent`
- `ExperimentAgent`
- `ReflectionAgent`
- `ReporterAgent`

These agents are orchestrated by the manager in [orchestration.py](/root/PDEFlow/src/pdeflow/orchestration.py).

## Runtime Model

The runtime layer is implemented in [runtime.py](/root/PDEFlow/src/pdeflow/runtime.py) and uses the OpenAI Agents SDK as the primary agent runtime.

Important runtime properties:

- structured outputs via typed Pydantic models
- persistent agent sessions backed by SQLite
- tool-enabled specialist execution
- fail-fast behavior if `OPENAI_API_KEY` is not set

There is no mock runtime in the current repository state. The system is intended to run live.

## State And Memory Model

Structured schemas live in [schemas.py](/root/PDEFlow/src/pdeflow/schemas.py). They include:

- research brief
- current phase
- environment snapshot
- secret status
- literature notes and taxonomy
- artifact registry
- repository registry
- candidate directions
- hypotheses
- method designs
- program candidates
- experiment plans
- experiment records
- best-known results
- failure summaries
- reflections
- generated reports

The memory layer in [memory.py](/root/PDEFlow/src/pdeflow/memory.py) stores:

- episodic memory as JSONL
- semantic memory as JSONL
- literature notes as JSONL
- artifact and repository registries as JSONL
- idea memory as JSONL
- experiment plans and records as JSONL
- generated reports as JSONL
- program lineage as SQLite

This design makes the workflow inspectable and supports later research-on-research analysis.

## Tool Surface

The tool layer in [tools.py](/root/PDEFlow/src/pdeflow/tools.py) provides real executable capabilities.

Current tool categories:

- environment inspection
- secret inspection
- arXiv paper search
- GitHub repository search
- URL fetching
- file download
- PDF text extraction
- repository cloning
- directory tree inspection
- local file reading
- codebase search
- file discovery
- project manifest detection
- Python environment bootstrapping with `uv`
- tree copying for child workspaces
- file writing
- patch writing and patch application
- shell command execution
- JSON and metric parsing
- report writing

These tools are exposed to agents through OpenAI Agents SDK function tools. The intent is that agents decide when and how to use them, rather than having asset locations hardcoded in Python logic.

## Program Evolution Model

The system treats code evolution as a tracked research object:

- a baseline or acquired repository becomes a parent program
- a hypothesis motivates a method design
- the coding phase creates a child workspace and changed files
- the experiment phase evaluates the child program
- the lineage database records the relationship between parent and child

This is the basis for future self-evolving research loops similar in spirit to program-search systems, but grounded here in literature retrieval and scientific experimentation.

## Repository Structure

Top-level structure:

```text
.
├── app.py
├── configs/
├── pyproject.toml
├── src/pdeflow/
├── external_assets/
└── runs/
```

Important source files:

- [config.py](/root/PDEFlow/src/pdeflow/config.py): top-level config schema
- [schemas.py](/root/PDEFlow/src/pdeflow/schemas.py): state and phase output schemas
- [tools.py](/root/PDEFlow/src/pdeflow/tools.py): tool implementations
- [agents.py](/root/PDEFlow/src/pdeflow/agents.py): specialist agent definitions
- [orchestration.py](/root/PDEFlow/src/pdeflow/orchestration.py): manager loop
- [memory.py](/root/PDEFlow/src/pdeflow/memory.py): local memory stores
- [runtime.py](/root/PDEFlow/src/pdeflow/runtime.py): OpenAI Agents SDK adapter
- [app.py](/root/PDEFlow/src/pdeflow/app.py): CLI entrypoint

## Installation

This project is managed with `uv`.

### Requirements

- Python 3.10+
- `uv`
- network access if you want autonomous retrieval and downloads
- `OPENAI_API_KEY`

### Install Dependencies

```bash
uv sync
```

The repository includes [pyproject.toml](/root/PDEFlow/pyproject.toml) and `uv.lock`.

## Environment Variables

Create `.env` from [.env.example](/root/PDEFlow/.env.example):

```bash
cp .env.example .env
```

Supported variables:

- `OPENAI_API_KEY`: required for the agent runtime
- `GITHUB_TOKEN`: optional; improves GitHub API rate limits

The current arXiv retrieval path uses the public arXiv API and does not require a key.

## Configuration

The research problem is defined in [research_problem.json](/root/PDEFlow/configs/research_problem.json).

The config has five main parts:

- `research_brief`: the scientific question, background, objectives, constraints, and deliverables
- `runtime`: the agent backend and model
- `retrieval`: retrieval policy and search limits
- `execution`: network, shell, package installation, and workspace policy
- `resource_policy`: preferred GPU selection and runtime policy

### Minimal Mental Model

The user should think of config as:

1. the research question
2. the execution policy
3. the machine policy

The user should not need to manually enumerate the benchmark repo, dataset URL, or checkpoint URL unless they want to constrain the search space.

## Running The System

Default run:

```bash
uv run python app.py --config configs/research_problem.json
```

Package entrypoint:

```bash
uv run pdeflow --config configs/research_problem.json
```

Override run name:

```bash
uv run python app.py --config configs/research_problem.json --run-name pde_round1
```

If `OPENAI_API_KEY` is missing, the process stops immediately with a clear runtime error. This is intentional.

## What The User Supplies

The intended user input is:

- a scientific research problem
- execution constraints and safety policy
- optionally some environment variables or API keys

That is all.

The system is designed so the agent decides:

- what to search
- what to download
- what to inspect
- what code to modify
- what experiments to run
- when to continue or stop

## Outputs

Run-specific outputs are written under `runs/<run_name>/`.

Directory layout:

- `state/`: full structured states saved after phases
- `logs/`: command logs and tool events
- `memory/`: episodic, semantic, and idea memories
- `literature/`: paper notes
- `programs/`: lineage database and program metadata
- `experiments/`: experiment records and logs
- `reports/`: markdown reports
- `workspaces/`: child program workspaces created during coding

Shared downloaded or cloned assets are stored under:

- `external_assets/`

## Generated Reports

The reporting phase is expected to write durable markdown artifacts such as:

- literature review notes
- acquisition report
- idea diary
- experiment summary
- final research report

These are generated from structured state rather than free-form chat transcripts.

## Autonomous Acquisition

Acquisition is part of the research loop, not a manual prerequisite.

The acquisition phase can:

- inspect the machine and current Python environment
- search literature endpoints
- search repository registries
- fetch benchmark or documentation pages
- clone code repositories
- inspect project manifests and likely training entrypoints
- bootstrap environments using `uv`
- record acquired assets into structured memory

This is how the system is intended to discover PDEBench-related assets or any future research assets. They should not need to be manually hardcoded into the framework.

## Coding And Experiment Execution

The coding phase operates on real workspaces. It is intended to:

- inspect baseline code
- copy a parent workspace
- write or patch files
- run compile or smoke-test commands

The experiment-planning phase then defines:

- setup commands
- launch commands
- working directories
- GPU selection
- expected outputs
- success criteria
- stopping rules

The experiment phase executes commands, captures logs, and parses metrics from generated files or text logs.

## PDE Focus

This repository is currently configured for PDE neural operator research, but the architecture is broader than PDEBench-specific optimization.

The included research brief emphasizes:

- neural operator methods
- short-window PDE prediction
- physics-aware training or inference
- autonomous discovery of literature, repos, data, and checkpoints
- iterative scientific improvement rather than blind tuning

## Current Strengths

- explicit research workflow instead of prompt spaghetti
- real tool surface instead of placeholder reasoning
- manager-centered orchestration
- durable external memory
- program lineage tracking
- live runtime with OpenAI Agents SDK
- direct support for autonomous acquisition and environment setup

## Current Limitations

This repository is a real autonomous systems scaffold, but it still has important limits:

- research quality depends on the underlying model and tool decisions
- metric parsing is generic and not yet specialized for every scientific benchmark format
- domain-specific experiment interpretation is still stronger for some research areas than others
- unrestricted autonomous shell execution is powerful and should be used carefully
- the framework can discover and run external code, but correctness of third-party repos is not guaranteed

These are system-level limitations, not placeholders.

## Safety And Operational Notes

The framework is capable of:

- running shell commands
- cloning remote repositories
- downloading external files
- bootstrapping Python environments

This is necessary for autonomous research, but it also means runs should be performed in a controlled machine environment with clear permissions and resource expectations.

Recommended practice:

- run on a dedicated research machine or sandbox
- keep API keys scoped and minimal
- monitor first live runs
- review generated reports and child workspaces before long experiment campaigns

## Extending The System

High-value next extensions for PDE operator research include:

- richer result parsers for PDE rollout metrics
- dataset introspection for HDF5, NetCDF, and scientific metadata
- checkpoint compatibility validation
- physics residual libraries for common PDE families
- stronger benchmark family discovery across FNO, DeepONet, PINO, and related methods
- automated ablation planning
- paper-ready report generation and competition submission formatting

## Release Status

Current repository status:

- live runtime
- no mock mode
- `uv`-managed environment
- structured multi-agent workflow
- autonomous acquisition and execution interfaces

To perform a full live run, the remaining operational prerequisite is simply to provide `OPENAI_API_KEY` in the environment.
