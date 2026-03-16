from __future__ import annotations

from pathlib import Path
import re
import shlex
from typing import Any

from state import (
    CapabilityMatrix,
    CodingPhaseOutput,
    ExperimentPhaseOutput,
    ExperimentPlan,
    ExperimentPlanningPhaseOutput,
    ExperimentRecord,
    PreflightValidationPhaseOutput,
    ReflectionPhaseOutput,
    ResearchPhase,
    ResearchState,
)
from tools import ResearchTools
from common import (
    dedupe_strings,
    extract_plan_tee_outputs,
    now_utc,
    plan_is_baseline,
    plan_prefers_fallback,
    plan_requires_fno,
    plan_requires_pinn,
    short_hash,
    upsert_by_attr,
)
from common.evolution import (
    build_experiment_evaluation_memos,
    build_preflight_evaluation_memos,
    build_reflection_memory_notes,
)
from .base import BaseResearchAgent


def _compact_artifact_payload(item: Any) -> dict[str, Any]:
    semantic_spec = item.semantic_spec.model_dump(mode="python", exclude_none=True) if item.semantic_spec else None
    validation = item.validation
    validation_summary = None
    if validation is not None:
        validation_summary = {
            "status": validation.status,
            "ready_for_training": validation.ready_for_training,
            "failure_reasons": validation.failure_reasons[:4],
            "checksum_matched": validation.checksum.matched if validation.checksum else None,
        }
    return {
        "artifact_id": item.artifact_id,
        "canonical_id": item.canonical_id,
        "artifact_type": item.artifact_type,
        "title": item.title,
        "local_path": item.local_path,
        "status": item.status,
        "semantic_spec": semantic_spec,
        "validation": validation_summary,
        "notes": item.notes[-3:],
    }


def _compact_plan_payload(plan: ExperimentPlan) -> dict[str, Any]:
    return {
        "plan_id": plan.plan_id,
        "title": plan.title,
        "program_id": plan.program_id,
        "repo_id": plan.repo_id,
        "job_kind": plan.job_kind,
        "working_directory": plan.working_directory,
        "launch_command": plan.launch_command[:400],
        "gpu_ids": plan.gpu_ids,
        "required_artifact_ids": plan.required_artifact_ids,
        "preflight_status": plan.preflight_status,
        "status": plan.status,
        "expected_outputs": plan.expected_outputs[:4],
        "notes": plan.notes[-4:],
    }


def _compact_experiment_payload(record: ExperimentRecord) -> dict[str, Any]:
    return {
        "experiment_id": record.experiment_id,
        "plan_id": record.plan_id,
        "program_id": record.program_id,
        "job_kind": record.job_kind,
        "status": record.status,
        "return_code": record.return_code,
        "metrics": record.metrics,
        "failure_modes": record.failure_modes[:4],
        "failure_ids": record.failure_ids[:4],
        "log_path": record.log_path,
        "result_paths": record.result_paths[:4],
    }


class CoderAgent(BaseResearchAgent):
    name = "CoderAgent"
    phase = ResearchPhase.CODING
    output_model = CodingPhaseOutput

    def allowed_tool_names(self) -> set[str] | None:
        return {
            "inspect_directory_tree",
            "read_text_file",
            "search_in_directory",
            "find_files",
            "detect_project_manifests",
            "copy_tree",
            "write_text_file",
            "write_json_file",
            "write_patch_file",
            "apply_patch_file",
            "bootstrap_python_environment",
            "ensure_python_environment",
            "inspect_python_environment",
            "run_command",
            "run_in_environment",
        }

    def runtime_timeout_seconds(self) -> int | None:
        return 600

    def build_instructions(self, state: ResearchState) -> str:
        return """
You are the coding specialist.
Implement the latest method design by creating a child workspace, editing files, and running basic validation commands.

You must use tools to:
- inspect the baseline repository and its entrypoints
- create a child workspace from the parent program or repository
- write or modify code files
- run at least one validation command such as import, compile, or a smoke test
- if repo dependencies are missing, provision a managed uv environment with tools instead of hand-writing `python -m venv` / `pip` flows

Rules:
- Do not claim a file was edited unless you used a tool to write it.
- Prefer minimal, targeted code changes that correspond directly to the method design.
- Return actual changed files and workspace paths.
"""

    def build_payload(self, state: ResearchState, tools: ResearchTools) -> dict[str, Any]:
        parent_programs = [
            {
                "program_id": item.program_id,
                "title": item.title,
                "summary": item.summary,
                "repo_id": item.repo_id,
                "workspace_path": item.workspace_path,
                "parent_program_id": item.parent_program_id,
                "design_id": item.design_id,
                "hypothesis_id": item.hypothesis_id,
                "entry_command_hint": item.entry_command_hint,
                "status": item.status,
                "changed_files": item.changed_files[-6:],
                "notes": item.notes[-4:],
            }
            for item in state.program_candidates[-6:]
        ]
        return {
            "research_brief": state.research_brief.model_dump(mode="python"),
            "latest_method_designs": [item.model_dump(mode="python") for item in state.method_designs[-2:]],
            "available_programs": parent_programs,
            "selected_baseline_program_id": state.selected_baseline_program_id,
            "repositories": [
                {
                    "repo_id": item.repo_id,
                    "canonical_id": item.canonical_id,
                    "name": item.name,
                    "remote_url": item.remote_url,
                    "local_path": item.local_path,
                    "bootstrap_status": item.bootstrap_status,
                    "environment_path": item.environment_path,
                    "entrypoints": item.entrypoints[:6],
                    "notes": item.notes[-4:],
                }
                for item in state.repositories[-4:]
            ],
            "recent_evaluations": [
                {
                    "memo_id": item.memo_id,
                    "phase": item.phase,
                    "verdict": item.verdict,
                    "summary": item.summary,
                    "plan_id": item.plan_id,
                    "program_id": item.program_id,
                    "recommended_actions": item.recommended_actions[:4],
                }
                for item in state.evaluation_memos[-6:]
            ],
            "recent_memory_notes": [
                {
                    "note_id": item.note_id,
                    "kind": item.kind,
                    "title": item.title,
                    "summary": item.summary,
                    "phase": item.phase,
                    "cycle_index": item.cycle_index,
                }
                for item in state.memory_notes[-8:]
            ],
            "run_workspace_root": str(tools.run_workspace_root),
        }

    def apply_output(self, state: ResearchState, tools: ResearchTools, output: CodingPhaseOutput) -> str:
        state.program_candidates = upsert_by_attr(state.program_candidates, output.program_candidates, "program_id")
        state.next_actions = output.next_actions
        for item in output.program_candidates:
            tools.memory.register_program(item)
        return output.summary


class ExperimentPlannerAgent(BaseResearchAgent):
    name = "ExperimentPlannerAgent"
    phase = ResearchPhase.EXPERIMENT_PLANNING
    output_model = ExperimentPlanningPhaseOutput

    def allowed_tool_names(self) -> set[str] | None:
        return {
            "inspect_directory_tree",
            "read_text_file",
            "search_in_directory",
            "find_files",
            "detect_project_manifests",
            "inspect_python_environment",
            "parse_json_file",
            "parse_metrics_file",
        }

    def runtime_timeout_seconds(self) -> int | None:
        return 600

    def build_instructions(self, state: ResearchState) -> str:
        return """
You are the experiment-planning specialist.
Build concrete execution plans for baseline and candidate programs.

You must:
- decide which programs to run next
- produce concrete setup commands, launch commands, working directories, logs, and expected outputs
- choose GPU usage and experiment duration indirectly through the command and stopping rules
- use only `ready_for_training` artifacts for actual experiment plans
- do not emit acquisition, repair, or data-download jobs here; those belong to acquisition
- do not emit plans that point to blocked, corrupted, quarantined, or checksum-mismatched artifacts
- prefer managed uv environments over ad hoc `python -m venv`, `source`, or bare `pip install`
- a blocked or setup-failed baseline does not count as a completed baseline
- if the selected baseline has no completed experiment record with real outputs, do not schedule downstream candidate launch plans except prerequisite acquisition/verification or matched baseline reruns
- if the manager route focus requests fallback_execution, prefer an evidence-generating fallback experiment over returning an empty plan set
- do not retry acquisition-dependent plans unchanged when the blocker registry says the same dataset acquisition route is exhausted
- inspect the repository entrypoint and config semantics before writing launch commands; if the code separates dataset filename from dataset root, pass both correctly and override placeholder defaults with verified local artifact paths

Rules:
- Do not invent commands that are impossible to run from the inspected repository layout.
- If no baseline experiment exists yet, include a baseline plan.
- Stopping rules should be scientific and metric-driven, not based on an arbitrary wall-clock cap.
"""

    def build_payload(self, state: ResearchState, tools: ResearchTools) -> dict[str, Any]:
        return {
            "research_brief": state.research_brief.model_dump(mode="python"),
            "environment_snapshot": state.environment_snapshot.model_dump(mode="python") if state.environment_snapshot else None,
            "external_artifacts": [_compact_artifact_payload(item) for item in state.external_artifacts[-12:]],
            "repositories": [
                {
                    "repo_id": item.repo_id,
                    "canonical_id": item.canonical_id,
                    "name": item.name,
                    "remote_url": item.remote_url,
                    "local_path": item.local_path,
                    "bootstrap_status": item.bootstrap_status,
                    "environment_path": item.environment_path,
                    "entrypoints": item.entrypoints[:6],
                    "notes": item.notes[-4:],
                }
                for item in state.repositories[-6:]
            ],
            "program_candidates": [
                {
                    "program_id": item.program_id,
                    "title": item.title,
                    "summary": item.summary,
                    "repo_id": item.repo_id,
                    "workspace_path": item.workspace_path,
                    "parent_program_id": item.parent_program_id,
                    "design_id": item.design_id,
                    "hypothesis_id": item.hypothesis_id,
                    "entry_command_hint": item.entry_command_hint,
                    "status": item.status,
                    "changed_files": item.changed_files[-6:],
                    "notes": item.notes[-4:],
                }
                for item in state.program_candidates[-8:]
            ],
            "method_designs": [
                {
                    "design_id": item.design_id,
                    "hypothesis_id": item.hypothesis_id,
                    "title": item.title,
                    "parent_program_id": item.parent_program_id,
                    "training_strategy": item.training_strategy[:4],
                    "physics_integration": item.physics_integration[:4],
                    "implementation_steps": item.implementation_steps[:6],
                    "evaluation_plan": item.evaluation_plan[:6],
                }
                for item in state.method_designs[-2:]
            ],
            "existing_experiments": [_compact_experiment_payload(item) for item in state.experiment_records[-8:]],
            "existing_plans": [_compact_plan_payload(item) for item in state.experiment_plans[-8:]],
            "managed_env_root": str(tools.managed_env_root),
            "preferred_log_root": str(tools.memory.experiments_dir),
            "selected_baseline_program_id": state.selected_baseline_program_id,
            "failure_summaries": state.failure_summaries[-8:],
            "capability_matrix": state.capability_matrix.model_dump(mode="python") if state.capability_matrix else None,
            "classified_failures": [item.model_dump(mode="python") for item in state.classified_failures[-8:]],
            "blocker_registry": [item.model_dump(mode="python") for item in state.blocker_registry[-8:]],
            "route_history": [item.model_dump(mode="python") for item in state.route_history[-4:]],
            "active_route_id": state.active_route_id,
            "active_route_focus": state.active_route_focus,
            "recent_evaluations": [item.model_dump(mode="python") for item in state.evaluation_memos[-6:]],
            "recent_memory_notes": [item.model_dump(mode="python") for item in state.memory_notes[-8:]],
            "hitl_events": [item.model_dump(mode="python") for item in state.hitl_events[-4:]],
            "manual_asset_roots": state.manual_asset_roots,
            "skipped_target_entities": state.skipped_target_entities,
            "human_guidance_notes": state.human_guidance_notes[-6:],
        }

    def apply_output(self, state: ResearchState, tools: ResearchTools, output: ExperimentPlanningPhaseOutput) -> str:
        output = self._inject_gpu_smoke_plan(state, tools, output)
        if self._should_force_fallback_execution(state, output):
            fallback_plan = self._build_fallback_plan(state, tools)
            if fallback_plan is not None:
                retained_plans = [
                    plan for plan in output.experiment_plans
                    if "evidence_generating_fallback" in plan.notes
                ]
                output = output.model_copy(
                    update={
                        "experiment_plans": [*retained_plans, fallback_plan],
                        "summary": (
                            output.summary
                            + " Added a deterministic evidence-generating fallback experiment because the active route requests fallback execution and the current baseline path is not launch-ready."
                        ),
                        "next_actions": [
                            *output.next_actions,
                            "Run the fallback smoke experiment to generate empirical evidence while exact target datasets remain blocked.",
                        ],
                    }
                )
        output = self._filter_and_stage_plans(state, output)
        state.experiment_plans = upsert_by_attr(state.experiment_plans, output.experiment_plans, "plan_id")
        state.next_actions = output.next_actions
        for item in output.experiment_plans:
            tools.memory.record_experiment_plan(item)
        return output.summary

    def _inject_gpu_smoke_plan(
        self,
        state: ResearchState,
        tools: ResearchTools,
        output: ExperimentPlanningPhaseOutput,
    ) -> ExperimentPlanningPhaseOutput:
        completed_experiments_exist = any(
            record.job_kind == "experiment" and record.status == "completed"
            for record in state.experiment_records
        )
        if completed_experiments_exist:
            return output
        if any("gpu_smoke_evidence" in plan.notes for plan in output.experiment_plans):
            return output
        capability = state.capability_matrix
        if capability is None or not capability.fno_ready or not capability.gpu_runtime_ready:
            return output
        smoke_plan = self._build_gpu_smoke_plan(state, tools)
        if smoke_plan is None:
            return output
        return output.model_copy(
            update={
                "experiment_plans": [smoke_plan, *output.experiment_plans],
                "summary": output.summary + " Added a deterministic GPU smoke experiment to generate immediate executable evidence before long baseline training.",
                "next_actions": [
                    *output.next_actions,
                    "Run the GPU smoke experiment first to confirm data loading, FNO import, and CUDA execution on a verified PDEBench shard.",
                ],
            }
        )

    def _filter_and_stage_plans(
        self,
        state: ResearchState,
        output: ExperimentPlanningPhaseOutput,
    ) -> ExperimentPlanningPhaseOutput:
        capability = state.capability_matrix
        retained: list[ExperimentPlan] = []
        inactive: list[ExperimentPlan] = []
        notes: list[str] = []
        for plan in output.experiment_plans:
            if not self._plan_has_executable_launch(plan):
                reason = (
                    "Blocked because the launch command does not resolve to an executable Python entrypoint "
                    "and appears to be malformed or placeholder output."
                )
                notes.append(
                    f"Filtered {plan.plan_id} because its launch command is not executable and appears malformed."
                )
                inactive.append(
                    plan.model_copy(
                        update={
                            "status": "blocked",
                            "preflight_status": "failed",
                            "notes": [*plan.notes, reason],
                        }
                    )
                )
                continue
            if capability is not None:
                if plan_requires_pinn(plan) and not capability.pinn_ready:
                    reason = (
                        "Blocked because the current environment is not PINN-ready "
                        f"(deepxde_backend={capability.deepxde_backend}, tensorflow_available={capability.tensorflow_available})."
                    )
                    notes.append(
                        f"Filtered {plan.plan_id} because the current environment is not PINN-ready "
                        f"(deepxde_backend={capability.deepxde_backend}, tensorflow_available={capability.tensorflow_available})."
                    )
                    inactive.append(
                        plan.model_copy(
                            update={
                                "status": "blocked",
                                "preflight_status": "failed",
                                "notes": [*plan.notes, reason],
                            }
                        )
                    )
                    continue
                if plan_requires_fno(plan) and not capability.fno_ready:
                    reason = (
                        "Blocked because the current environment is not FNO-ready "
                        f"(torch_runtime_ready={capability.torch_runtime_ready}, h5py_available={capability.h5py_available})."
                    )
                    notes.append(
                        f"Filtered {plan.plan_id} because the current environment is not FNO-ready "
                        f"(torch_runtime_ready={capability.torch_runtime_ready}, h5py_available={capability.h5py_available})."
                    )
                    inactive.append(
                        plan.model_copy(
                            update={
                                "status": "blocked",
                                "preflight_status": "failed",
                                "notes": [*plan.notes, reason],
                            }
                        )
                    )
                    continue
            retained.append(plan)

        completed_experiments_exist = any(
            record.job_kind == "experiment" and record.status == "completed"
            for record in state.experiment_records
        )
        if not completed_experiments_exist and len(retained) > 1:
            retained = sorted(retained, key=lambda item: self._plan_priority(state, item))
            deferred = retained[1:]
            retained = retained[:1]
            notes.append(
                "Deferred additional experiment plans until the first executable baseline produces real evidence: "
                + ", ".join(item.plan_id for item in deferred)
                + "."
            )
            inactive.extend(
                item.model_copy(
                    update={
                        "status": "deferred",
                        "notes": [
                            *item.notes,
                            "Deferred until a first baseline execution produces empirical evidence.",
                        ],
                    }
                )
                for item in deferred
            )

        if not notes:
            return output
        return output.model_copy(
            update={
                "experiment_plans": [*retained, *inactive],
                "summary": output.summary + " " + " ".join(notes),
                "next_actions": [*output.next_actions, *notes],
            }
        )

    def _plan_has_executable_launch(self, plan: ExperimentPlan) -> bool:
        command = plan.launch_command.strip()
        lowered = command.lower()
        if "launch command truncated" in lowered or "malformed output" in lowered:
            return False
        if "python - <<" in command or "python -c " in command:
            return True
        try:
            tokens = shlex.split(command)
        except ValueError:
            return False
        working_directory = Path(plan.working_directory)
        for token in tokens:
            candidate = Path(token)
            if candidate.suffix == ".py":
                resolved = candidate if candidate.is_absolute() else (working_directory / candidate)
                if resolved.exists():
                    return True
        return False

    def _plan_priority(self, state: ResearchState, plan: ExperimentPlan) -> tuple[int, int, str]:
        priority = 50
        if "gpu_smoke_evidence" in plan.notes:
            priority -= 40
        if "fallback_execution" in state.active_route_focus and plan_prefers_fallback(plan):
            priority -= 30
        if state.selected_baseline_program_id and plan.program_id == state.selected_baseline_program_id:
            priority -= 20
        if plan_requires_fno(plan):
            priority -= 10
        if plan_is_baseline(plan):
            priority -= 5
        if plan_requires_pinn(plan):
            priority += 10
        return (priority, len(plan.required_artifact_ids), plan.plan_id)

    def _should_force_fallback_execution(
        self,
        state: ResearchState,
        output: ExperimentPlanningPhaseOutput,
    ) -> bool:
        if "fallback_execution" not in state.active_route_focus:
            return False
        if not output.experiment_plans:
            return True
        capability = state.capability_matrix
        if capability is None:
            return True
        if capability.baseline_ready_to_launch:
            return False
        if capability.gpu_runtime_required and not capability.gpu_runtime_ready:
            return not any("evidence_generating_fallback" in plan.notes for plan in output.experiment_plans)
        return False

    def _build_fallback_plan(self, state: ResearchState, tools: ResearchTools) -> ExperimentPlan | None:
        capability = state.capability_matrix
        if capability is None or not capability.env_ready or not capability.codepath_ready:
            return None
        repository = next((item for item in state.repositories if item.local_path), None)
        if repository is None:
            return None
        env_path = repository.environment_path or capability.environment_path
        if not env_path:
            return None
        ready_artifacts = [
            item for item in state.external_artifacts
            if item.status == "ready_for_training" and item.artifact_type in {"checkpoint", "dataset"}
        ]
        artifact_paths = [item.local_path for item in ready_artifacts if item.local_path][:3]
        artifact_asserts = "\n".join(
            f"assert Path({path!r}).exists(), {path!r}" for path in artifact_paths
        )
        report_path = tools.memory.experiments_dir / "fallback_smoke_metrics.json"
        script = (
            "python - <<'PY'\n"
            "from pathlib import Path\n"
            "import json\n"
            "import torch\n"
            "import h5py\n"
            "import pdebench\n"
            f"{artifact_asserts}\n"
            "payload = {\n"
            "  'mode': 'fallback_smoke',\n"
            "  'torch_version': torch.__version__,\n"
            "  'cuda_available': bool(torch.cuda.is_available()),\n"
            f"  'checked_artifacts': {artifact_paths!r},\n"
            "}\n"
            f"Path({str(report_path)!r}).write_text(json.dumps(payload), encoding='utf-8')\n"
            "print(json.dumps(payload))\n"
            "PY"
        )
        return ExperimentPlan(
            plan_id=f"fallback-smoke-{short_hash(state.run_name, str(state.cycle_index), now_utc())}",
            title="Fallback smoke evidence run",
            program_id=state.selected_baseline_program_id or "fallback-smoke",
            repo_id=repository.canonical_id or repository.repo_id,
            job_kind="experiment",
            working_directory=repository.local_path,
            setup_commands=[],
            launch_command=script,
            environment={"VIRTUAL_ENV": env_path},
            gpu_ids=[],
            required_artifact_ids=[item.canonical_id or item.artifact_id for item in ready_artifacts[:3]],
            preflight_required=True,
            expected_outputs=[str(report_path)],
            success_criteria=["The fallback smoke run completes and emits a JSON payload proving repo/env execution works."],
            stopping_rules=["Stop immediately on import or filesystem failure."],
            log_path=str(tools.memory.experiments_dir / "fallback_smoke.log"),
            status="planned",
            notes=[
                "evidence_generating_fallback",
                "This plan exists to break zero-evidence stagnation while exact target datasets remain blocked.",
            ],
        )

    def _build_gpu_smoke_plan(self, state: ResearchState, tools: ResearchTools) -> ExperimentPlan | None:
        repository = next((item for item in state.repositories if item.local_path), None)
        if repository is None:
            return None
        env_path = repository.environment_path or (state.capability_matrix.environment_path if state.capability_matrix else None)
        if not env_path:
            return None
        dataset = next(
            (
                item for item in state.external_artifacts
                if item.artifact_type == "dataset"
                and item.status == "ready_for_training"
                and item.local_path
                and "Burgers" in (item.title or item.local_path)
            ),
            None,
        )
        if dataset is None:
            dataset = next(
                (
                    item for item in state.external_artifacts
                    if item.artifact_type == "dataset" and item.status == "ready_for_training" and item.local_path
                ),
                None,
            )
        if dataset is None or dataset.local_path is None:
            return None
        dataset_path = Path(dataset.local_path)
        report_path = tools.memory.experiments_dir / "gpu_smoke_metrics.json"
        script = (
            "python - <<'PY'\n"
            "import os\n"
            "import json\n"
            "import time\n"
            "from pathlib import Path\n"
            "def log(message):\n"
            "    print(message, flush=True)\n"
            "log('gpu_smoke: start')\n"
            "import h5py\n"
            "import numpy as np\n"
            "import torch\n"
            "log(f'gpu_smoke: torch_imported version={getattr(torch, \"__version__\", \"unknown\")} cuda={getattr(getattr(torch, \"version\", None), \"cuda\", None)} visible={os.getenv(\"CUDA_VISIBLE_DEVICES\", \"all\")}')\n"
            "from pdebench.models.fno.fno import FNO1d\n"
            "log('gpu_smoke: imported_pdebench_modules')\n"
            f"dataset_path = Path({str(dataset_path)!r})\n"
            "log(f'gpu_smoke: dataset_path={dataset_path}')\n"
            "device = torch.device('cuda')\n"
            "start = time.time()\n"
            "cuda_available = bool(torch.cuda.is_available())\n"
            "log(f'gpu_smoke: cuda_available={cuda_available}')\n"
            "probe = torch.ones(1, device=device)\n"
            "torch.cuda.synchronize()\n"
            "log(f'gpu_smoke: device_probe_ok value={float(probe.item())}')\n"
            "with h5py.File(dataset_path, 'r') as handle:\n"
            "    tensor = handle['tensor']\n"
            "    sample = np.asarray(tensor[0, ::5, ::4], dtype=np.float32)\n"
            "    x_coord = np.asarray(handle.get('x-coordinate', np.linspace(0.0, 1.0, sample.shape[-1], dtype=np.float32))[::4], dtype=np.float32)\n"
            "log(f'gpu_smoke: raw_sample_loaded tensor={tuple(sample.shape)} x={tuple(x_coord.shape)}')\n"
            "sample = sample.T\n"
            "initial_step = 10\n"
            "xx = torch.from_numpy(sample[:, :initial_step]).contiguous()\n"
            "yy = torch.from_numpy(sample[:, initial_step:initial_step + 1]).contiguous().unsqueeze(-1)\n"
            "grid = torch.from_numpy(x_coord.reshape(-1, 1)).contiguous()\n"
            "log(f'gpu_smoke: sample_prepared xx={tuple(xx.shape)} yy={tuple(yy.shape)} grid={tuple(grid.shape)}')\n"
            "model = FNO1d(num_channels=1, width=20, modes=12, initial_step=initial_step).to(device)\n"
            "log('gpu_smoke: model_on_cuda')\n"
            "inp = xx.unsqueeze(0).reshape(1, xx.shape[0], -1).to(device)\n"
            "grid = grid.unsqueeze(0).to(device)\n"
            "log(f'gpu_smoke: tensors_on_cuda inp={tuple(inp.shape)} grid={tuple(grid.shape)}')\n"
            "torch.cuda.synchronize()\n"
            "with torch.no_grad():\n"
            "    out = model(inp, grid)\n"
            "torch.cuda.synchronize()\n"
            "log(f'gpu_smoke: forward_ok out={tuple(out.shape)}')\n"
            "payload = {\n"
            "    'mode': 'gpu_smoke_evidence',\n"
            "    'dataset': dataset_path.name,\n"
            "    'cuda_available': cuda_available,\n"
            "    'device': str(device),\n"
            "    'input_shape': list(inp.shape),\n"
            "    'output_shape': list(out.shape),\n"
            "    'elapsed_seconds': round(time.time() - start, 4),\n"
            "    'output_abs_mean': float(out.abs().mean().item()),\n"
            "}\n"
            f"Path({str(report_path)!r}).write_text(json.dumps(payload), encoding='utf-8')\n"
            "print(json.dumps(payload))\n"
            "PY"
        )
        return ExperimentPlan(
            plan_id=f"gpu-smoke-{short_hash(state.run_name, str(state.cycle_index), dataset.artifact_id)}",
            title="GPU smoke evidence run on verified PDEBench shard",
            program_id=state.selected_baseline_program_id or "gpu-smoke",
            repo_id=repository.canonical_id or repository.repo_id,
            job_kind="experiment",
            working_directory=repository.local_path,
            setup_commands=[],
            launch_command=script,
            environment={"VIRTUAL_ENV": env_path, "PYTHONUNBUFFERED": "1"},
            gpu_ids=[state.environment_snapshot.selected_gpu_ids[0]] if state.environment_snapshot.selected_gpu_ids else [],
            required_artifact_ids=[dataset.canonical_id or dataset.artifact_id],
            preflight_required=True,
            expected_outputs=[str(report_path)],
            success_criteria=[
                "Load a verified PDEBench dataset sample.",
                "Instantiate an FNO model and execute a forward pass on CUDA.",
                "Emit machine-readable GPU smoke metrics.",
            ],
            stopping_rules=["Stop immediately on import, dataset, or CUDA failure."],
            log_path=str(tools.memory.experiments_dir / "gpu_smoke.log"),
            status="planned",
            notes=["gpu_smoke_evidence", "bootstrap_empirical_evidence"],
        )


class PreflightValidationAgent(BaseResearchAgent):
    name = "PreflightValidationAgent"
    phase = ResearchPhase.PREFLIGHT_VALIDATION
    output_model = PreflightValidationPhaseOutput

    def build_instructions(self, state: ResearchState) -> str:
        return "Deterministic preflight validation agent."

    def build_payload(self, state: ResearchState, tools: ResearchTools) -> dict[str, Any]:
        return {}

    def _preferred_environment_path(
        self,
        state: ResearchState,
        pending_plans: list[ExperimentPlan] | None = None,
    ) -> str | None:
        for plan in pending_plans or []:
            env_path = str(plan.environment.get("VIRTUAL_ENV") or "").strip()
            if env_path:
                return env_path
        workdir_prefix = str(Path(state.work_directory).resolve())
        for environment in state.environment_records:
            if environment.state.value != "ready":
                continue
            try:
                resolved = str(Path(environment.environment_path).resolve())
            except OSError:
                continue
            if resolved.startswith(workdir_prefix):
                return environment.environment_path
        if state.capability_matrix and state.capability_matrix.environment_path:
            return state.capability_matrix.environment_path
        for environment in state.environment_records:
            if environment.state.value == "ready":
                return environment.environment_path
        return None

    def run(self, state: ResearchState, tools: ResearchTools, runtime: Any) -> str:
        pending_plans = [
            plan for plan in state.experiment_plans
            if plan.status == "planned" and plan.job_kind == "experiment"
        ]
        preferred_environment_path = self._preferred_environment_path(state, pending_plans)
        if not pending_plans:
            zero_reason = "No experiment plans are launch-eligible after planning."
            if state.capability_matrix and state.capability_matrix.target_dataset_preparing:
                zero_reason = (
                    "No experiment plans are launch-eligible yet because exact target datasets are still being "
                    "prepared or validated. Continue acquisition instead of escalating."
                )
            elif state.capability_matrix and state.capability_matrix.target_dataset_blocked:
                zero_reason = (
                    "No experiment plans are launch-eligible because the exact target datasets remain blocked "
                    "and the planner produced no fallback executable plans."
                )
            report = tools.preflight_experiment_plan(
                ExperimentPlan(
                    plan_id="__no_executable_plans__",
                    title="No executable plans",
                    program_id=state.selected_baseline_program_id or "none",
                    job_kind="preflight",
                    working_directory=state.work_directory,
                    launch_command="true",
                    log_path=str(tools.memory.preflight_dir / "no_executable_plans.log"),
                    status="blocked",
                    notes=[zero_reason],
                ),
                state.external_artifacts,
                state.capability_matrix,
            )
            capability_matrix = tools.probe_capability_matrix(
                artifacts=state.external_artifacts,
                repository_paths=[repo.local_path for repo in state.repositories],
                environment_path=preferred_environment_path,
            )
            output = PreflightValidationPhaseOutput(
                summary="Preflight validated pending experiment plans. passed=0 blocked=1.",
                preflight_reports=[report],
                capability_matrix=capability_matrix,
                failure_summaries=[zero_reason],
                zero_plan_reason=zero_reason,
                recommended_route=report.recommended_route or "acquisition",
                next_actions=["Pivot to fallback execution or alternate acquisition strategy; do not retry the same blocked route unchanged."],
            )
            applied_summary = self.apply_output(state, tools, output)
            self.record_diary(state, tools, applied_summary)
            return applied_summary
        reports = []
        for plan in pending_plans:
            plan_capability_matrix = tools.probe_capability_matrix(
                artifacts=state.external_artifacts,
                repository_paths=[repo.local_path for repo in state.repositories],
                environment_path=str(plan.environment.get("VIRTUAL_ENV") or "").strip() or preferred_environment_path,
            )
            reports.append(
                tools.preflight_experiment_plan(plan, state.external_artifacts, plan_capability_matrix)
            )
        capability_matrix = tools.probe_capability_matrix(
            artifacts=state.external_artifacts,
            repository_paths=[repo.local_path for repo in state.repositories],
            environment_path=preferred_environment_path,
        )
        failure_summaries = [
            f"Preflight blocked {report.plan_id}: {report.blocking_reason}"
            for report in reports
            if not report.passed and report.blocking_reason
        ]
        summary = (
            "Preflight validated pending experiment plans. "
            f"passed={sum(1 for report in reports if report.passed)} "
            f"blocked={sum(1 for report in reports if not report.passed)}."
        )
        output = PreflightValidationPhaseOutput(
            summary=summary,
            preflight_reports=reports,
            capability_matrix=capability_matrix,
            failure_summaries=failure_summaries,
            recommended_route=next((report.recommended_route for report in reports if report.recommended_route), None),
            next_actions=[
                "Launch only preflight-passed plans."
                if any(report.passed for report in reports)
                else "Route back to acquisition or environment repair using failed preflight checks."
            ],
        )
        applied_summary = self.apply_output(state, tools, output)
        self.record_diary(state, tools, applied_summary)
        return applied_summary

    def apply_output(self, state: ResearchState, tools: ResearchTools, output: PreflightValidationPhaseOutput) -> str:
        state.preflight_reports = upsert_by_attr(state.preflight_reports, output.preflight_reports, "report_id")
        state.failure_summaries = dedupe_strings([*state.failure_summaries, *output.failure_summaries])
        state.next_actions = output.next_actions
        if output.capability_matrix is not None:
            state.capability_matrix = output.capability_matrix
            tools.memory.record_capability_matrix(output.capability_matrix)
        report_by_plan = {report.plan_id: report for report in output.preflight_reports}
        for plan in state.experiment_plans:
            report = report_by_plan.get(plan.plan_id)
            if report is None:
                continue
            plan.preflight_status = "passed" if report.passed else "failed"
            if not report.passed:
                plan.status = "blocked"
            tools.memory.record_preflight_report(report)
        memos = build_preflight_evaluation_memos(state, output.preflight_reports)
        if memos:
            state.evaluation_memos = upsert_by_attr(state.evaluation_memos, memos, "memo_id")
            for memo in memos:
                stored_memo, stored_note = tools.memory.record_evaluation_memo(memo)
                state.evaluation_memos = upsert_by_attr(state.evaluation_memos, [stored_memo], "memo_id")
                state.memory_notes = upsert_by_attr(state.memory_notes, [stored_note], "note_id")
        return output.summary


class ExperimentAgent(BaseResearchAgent):
    name = "ExperimentAgent"
    phase = ResearchPhase.EXPERIMENT
    output_model = ExperimentPhaseOutput

    def allowed_tool_names(self) -> set[str] | None:
        return {
            "inspect_directory_tree",
            "read_text_file",
            "search_in_directory",
            "find_files",
            "inspect_python_environment",
            "probe_capability_matrix",
            "run_command",
            "run_in_environment",
            "parse_json_file",
            "parse_metrics_file",
        }

    def build_instructions(self, state: ResearchState) -> str:
        return """
You are the experiment execution specialist.
Use tools to run the planned experiments, parse their outputs, and return only observed results.

You must:
- execute setup and launch commands only from preflight-passed experiment plans
- parse metrics from produced files or logs
- record failures when commands or metrics extraction fail
- do not perform acquisition or repair work here; failed prerequisites should stay blocked and route back to acquisition/preflight
- only report a plan as completed when its intended execution actually ran and produced observed outputs; setup failures are blockers, not completions
- validate code-level path semantics before launch: when a repository expects both a dataset filename and a dataset root, ensure the launch command contains the correct local root instead of falling back to placeholder defaults discovered in repository code or configs

Rules:
- Do not fabricate metrics or success claims.
- Prefer the newest planned experiments that have not been completed.
- Update best-known results only from actual parsed outputs.
"""

    def build_payload(self, state: ResearchState, tools: ResearchTools) -> dict[str, Any]:
        pending_plans = [
            item.model_dump(mode="python")
            for item in state.experiment_plans
            if item.status == "planned" and item.job_kind == "experiment" and item.preflight_status == "passed"
        ]
        return {
            "research_brief": state.research_brief.model_dump(mode="python"),
            "pending_experiment_plans": pending_plans[-4:],
            "existing_experiments": [item.model_dump(mode="python") for item in state.experiment_records[-10:]],
            "preflight_reports": [item.model_dump(mode="python") for item in state.preflight_reports[-10:]],
            "program_candidates": [item.model_dump(mode="python") for item in state.program_candidates[-10:]],
            "repositories": [item.model_dump(mode="python") for item in state.repositories[-10:]],
            "external_artifacts": [item.model_dump(mode="python") for item in state.external_artifacts[-20:]],
            "managed_env_root": str(tools.managed_env_root),
            "best_known_results": state.best_known_results,
        }

    def run(self, state: ResearchState, tools: ResearchTools, runtime: Any) -> str:  # noqa: ARG002
        pending_plans = [
            plan for plan in state.experiment_plans
            if plan.status == "planned" and plan.job_kind == "experiment" and plan.preflight_status == "passed"
        ]
        if not pending_plans:
            output = ExperimentPhaseOutput(
                summary="No preflight-passed experiment plans were available for execution.",
                experiment_records=[],
                failure_summaries=[],
                next_actions=["Return to planning or preflight; do not enter an empty experiment loop."],
            )
            applied = self.apply_output(state, tools, output)
            self.record_diary(state, tools, applied)
            return applied

        ordered_plans = sorted(pending_plans, key=lambda item: self._execution_priority(state, item))
        records: list[ExperimentRecord] = []
        failure_summaries: list[str] = []
        best_known_results: dict[str, dict[str, Any]] = {}
        for plan in ordered_plans:
            record = self._execute_plan(state, tools, plan)
            records.append(record)
            if record.status == "completed" and record.metrics:
                best_known_results[record.program_id] = dict(record.metrics)
            if record.failure_modes:
                failure_summaries.extend(
                    f"{plan.plan_id}: {mode}" for mode in record.failure_modes
                )

        completed = sum(1 for item in records if item.status == "completed")
        failed = sum(1 for item in records if item.status != "completed")
        output = ExperimentPhaseOutput(
            summary=f"Executed experiment plans. completed={completed} failed={failed}.",
            experiment_records=records,
            best_known_results=best_known_results,
            failure_summaries=dedupe_strings(failure_summaries),
            next_actions=[
                "Reflect on completed experiment evidence and decide whether to reproduce, repair, or iterate on the method."
                if completed
                else "No experiment completed successfully; reflect on the observed launch failures before retrying."
            ],
        )
        applied = self.apply_output(state, tools, output)
        self.record_diary(state, tools, applied)
        return applied

    def _execution_priority(self, state: ResearchState, plan: ExperimentPlan) -> tuple[int, int, str]:
        priority = 50
        if "gpu_smoke_evidence" in plan.notes:
            priority -= 40
        if "fallback_execution" in state.active_route_focus and plan_prefers_fallback(plan):
            priority -= 30
        if state.selected_baseline_program_id and plan.program_id == state.selected_baseline_program_id:
            priority -= 20
        if plan_requires_fno(plan):
            priority -= 10
        if plan_is_baseline(plan):
            priority -= 5
        if plan_requires_pinn(plan):
            priority += 10
        return (priority, len(plan.required_artifact_ids), plan.plan_id)

    def _execute_plan(
        self,
        state: ResearchState,
        tools: ResearchTools,
        plan: ExperimentPlan,
    ) -> ExperimentRecord:
        started_at = now_utc()
        stall_timeout_seconds = tools.config.execution.experiment_no_progress_timeout_seconds
        for index, command in enumerate(plan.setup_commands):
            normalized_command = self._normalize_execution_command(command)
            result = tools.run_command(
                normalized_command,
                cwd=plan.working_directory,
                env_overrides=plan.environment,
                gpu_ids=plan.gpu_ids,
                log_path=str(Path(plan.log_path).with_name(f"{Path(plan.log_path).stem}.setup{index}.log")),
                allow_failure=True,
                emit_progress=True,
                job_kind="setup",
                stall_timeout_seconds=stall_timeout_seconds,
            )
            if result["returncode"] != 0:
                return ExperimentRecord(
                    experiment_id=f"exp-{short_hash(plan.plan_id, 'setup', started_at)}",
                    plan_id=plan.plan_id,
                    program_id=plan.program_id,
                    command=normalized_command,
                    working_directory=plan.working_directory,
                    status="setup_failed",
                    return_code=int(result["returncode"]),
                    observations=[],
                    failure_modes=[result["stderr_tail"] or result["stdout_tail"] or "setup command failed"],
                    log_path=str(result["log_path"]),
                    started_at=started_at,
                    finished_at=now_utc(),
                )

        launch_command = self._normalize_execution_command(plan.launch_command)
        launch_result = tools.run_command(
            launch_command,
            cwd=plan.working_directory,
            env_overrides=plan.environment,
            gpu_ids=plan.gpu_ids,
            log_path=plan.log_path,
            allow_failure=True,
            emit_progress=True,
            job_kind="experiment",
            stall_timeout_seconds=stall_timeout_seconds,
        )
        metrics, result_paths, observations = self._collect_plan_outputs(plan, tools)
        status = "completed" if int(launch_result["returncode"]) == 0 else "failed"
        failure_modes: list[str] = []
        if status != "completed":
            failure_modes.append(
                launch_result["stderr_tail"] or launch_result["stdout_tail"] or "experiment command failed"
            )
        elif not metrics:
            observations.append("Command completed without machine-readable metrics; inspect logs for qualitative evidence.")
        return ExperimentRecord(
            experiment_id=f"exp-{short_hash(plan.plan_id, 'launch', started_at)}",
            plan_id=plan.plan_id,
            program_id=plan.program_id,
            command=launch_command,
            working_directory=plan.working_directory,
            status=status,
            return_code=int(launch_result["returncode"]),
            metrics=metrics,
            observations=observations,
            failure_modes=failure_modes,
            log_path=str(launch_result["log_path"]),
            result_paths=result_paths,
            started_at=started_at,
            finished_at=now_utc(),
        )

    def _normalize_execution_command(self, command: str) -> str:
        return re.sub(
            r"(uv\s+run\s+--python\s+)(\S+)(\s+python\b)",
            lambda match: match.group(2),
            command,
        )

    def _collect_plan_outputs(
        self,
        plan: ExperimentPlan,
        tools: ResearchTools,
    ) -> tuple[dict[str, Any], list[str], list[str]]:
        candidate_paths: list[str] = []
        for candidate in [*plan.expected_outputs, plan.log_path, *extract_plan_tee_outputs(plan)]:
            if candidate and candidate not in candidate_paths:
                candidate_paths.append(candidate)
        result_paths: list[str] = []
        metrics: dict[str, Any] = {}
        observations: list[str] = []
        for candidate in candidate_paths:
            resolved = Path(candidate)
            if not resolved.is_absolute():
                resolved = Path(plan.working_directory) / resolved
            if not resolved.exists():
                continue
            result_paths.append(str(resolved))
            try:
                parsed = tools.parse_metrics_file(str(resolved))
            except Exception:
                continue
            if parsed.get("metrics"):
                metrics.update(parsed["metrics"])
        if result_paths:
            observations.append(
                "Observed output artifacts: " + ", ".join(Path(path).name for path in result_paths[:6])
            )
        return metrics, result_paths, observations

    def apply_output(self, state: ResearchState, tools: ResearchTools, output: ExperimentPhaseOutput) -> str:
        prior_best_results = {key: dict(value) for key, value in state.best_known_results.items()}
        actual_experiments = [item for item in output.experiment_records if item.job_kind == "experiment"]
        auxiliary_records = [item for item in output.experiment_records if item.job_kind != "experiment"]
        state.experiment_records = upsert_by_attr(state.experiment_records, actual_experiments, "experiment_id")
        state.execution_records = upsert_by_attr(state.execution_records, auxiliary_records, "experiment_id")
        state.best_known_results.update(output.best_known_results)
        state.failure_summaries = dedupe_strings([*state.failure_summaries, *output.failure_summaries])
        state.next_actions = output.next_actions
        latest_record_by_plan = {record.plan_id: record for record in output.experiment_records}
        for plan in state.experiment_plans:
            record = latest_record_by_plan.get(plan.plan_id)
            if record is None:
                continue
            if record.status == "completed":
                plan.status = "completed"
            elif "block" in record.status or "setup" in record.status:
                plan.status = "blocked"
            else:
                plan.status = "failed"
        for item in actual_experiments:
            tools.memory.record_experiment(item)
            tools.memory.update_program_result(
                program_id=item.program_id,
                status="evaluated" if item.status == "completed" else item.status,
                metrics=item.metrics,
                failure_reason="; ".join(item.failure_modes) if item.failure_modes else None,
            )
        for item in auxiliary_records:
            tools.memory.record_execution(item)
        memos = build_experiment_evaluation_memos(state, actual_experiments, prior_best_results)
        if memos:
            state.evaluation_memos = upsert_by_attr(state.evaluation_memos, memos, "memo_id")
            for memo in memos:
                stored_memo, stored_note = tools.memory.record_evaluation_memo(memo)
                state.evaluation_memos = upsert_by_attr(state.evaluation_memos, [stored_memo], "memo_id")
                state.memory_notes = upsert_by_attr(state.memory_notes, [stored_note], "note_id")
        return output.summary


class ReflectionAgent(BaseResearchAgent):
    name = "ReflectionAgent"
    phase = ResearchPhase.REFLECTION
    output_model = ReflectionPhaseOutput

    def allowed_tool_names(self) -> set[str] | None:
        return set()

    def runtime_timeout_seconds(self) -> int | None:
        return 300

    def build_instructions(self, state: ResearchState) -> str:
        return """
You are the reflection specialist.
Assess whether the latest research cycle produced meaningful progress and decide whether the system should continue iterating.

Rules:
- Compare against actual baseline or parent-program evidence when available.
- Distinguish method-level gains from accidental engineering artifacts.
- If progress is insufficient or blocked, say why and propose the next move.
- Return machine-readable control signals inside each reflection record:
  - recommended_route_id
  - preferred_recovery_strategies
  - forbidden_attempt_signatures
  - blocked_entities
  - material_change_required
  - escalation_required
- If the same infrastructure blocker repeated without new evidence, mark escalation_required or force a pivot to a different strategy.
"""

    def build_payload(self, state: ResearchState, tools: ResearchTools) -> dict[str, Any]:
        return {
            "research_brief": state.research_brief.model_dump(mode="python"),
            "cycle_index": state.cycle_index,
            "selected_baseline_program_id": state.selected_baseline_program_id,
            "hypotheses": [item.model_dump(mode="python") for item in state.hypotheses[-4:]],
            "method_designs": [item.model_dump(mode="python") for item in state.method_designs[-4:]],
            "program_candidates": [item.model_dump(mode="python") for item in state.program_candidates[-8:]],
            "experiment_records": [item.model_dump(mode="python") for item in state.experiment_records[-10:]],
            "best_known_results": state.best_known_results,
            "failure_summaries": state.failure_summaries[-12:],
            "classified_failures": [item.model_dump(mode="python") for item in state.classified_failures[-12:]],
            "capability_matrix": state.capability_matrix.model_dump(mode="python") if state.capability_matrix else None,
            "blocker_registry": [item.model_dump(mode="python") for item in state.blocker_registry[-12:]],
            "route_history": [item.model_dump(mode="python") for item in state.route_history[-8:]],
            "cycle_deltas": [item.model_dump(mode="python") for item in state.cycle_deltas[-4:]],
            "evaluation_memos": [item.model_dump(mode="python") for item in state.evaluation_memos[-10:]],
            "memory_notes": [item.model_dump(mode="python") for item in state.memory_notes[-16:]],
            "hitl_events": [item.model_dump(mode="python") for item in state.hitl_events[-6:]],
            "manual_asset_roots": state.manual_asset_roots,
            "skipped_target_entities": state.skipped_target_entities,
            "human_guidance_notes": state.human_guidance_notes[-12:],
        }

    def apply_output(self, state: ResearchState, tools: ResearchTools, output: ReflectionPhaseOutput) -> str:
        state.reflections = upsert_by_attr(state.reflections, output.reflections, "reflection_id")
        state.next_actions = output.next_actions
        if output.terminate_research and output.reflections:
            state.termination_decision = output.reflections[-1].stop_reason or output.reflections[-1].verdict
        for reflection in output.reflections:
            if not reflection.linked_failure_ids and state.classified_failures:
                reflection.linked_failure_ids = [item.failure_id for item in state.classified_failures[-5:]]
            self.record_semantic_notes(state, tools, reflection.accepted_lessons)
            for note in build_reflection_memory_notes(state, reflection):
                stored = tools.memory.record_memory_note(note)
                state.memory_notes = upsert_by_attr(state.memory_notes, [stored], "note_id")
        return output.summary


class EngineeringAgent(BaseResearchAgent):
    name = "EngineeringAgent"

    def __init__(self, phase: ResearchPhase):
        self.phase = phase
        self._delegate = CoderAgent() if phase == ResearchPhase.CODING else ExperimentPlannerAgent()
        self.output_model = self._delegate.output_model

    def allowed_tool_names(self) -> set[str] | None:
        return self._delegate.allowed_tool_names()

    def max_turns(self) -> int | None:
        return self._delegate.max_turns()

    def runtime_timeout_seconds(self) -> int | None:
        return self._delegate.runtime_timeout_seconds()

    def build_instructions(self, state: ResearchState) -> str:
        return self._delegate.build_instructions(state)

    def build_payload(self, state: ResearchState, tools: ResearchTools) -> dict[str, Any]:
        return self._delegate.build_payload(state, tools)

    def apply_output(self, state: ResearchState, tools: ResearchTools, output: Any) -> str:
        return self._delegate.apply_output(state, tools, output)


class EvaluationAgent(BaseResearchAgent):
    name = "EvaluationAgent"

    def __init__(self, phase: ResearchPhase):
        self.phase = phase
        if phase == ResearchPhase.PREFLIGHT_VALIDATION:
            self._delegate = PreflightValidationAgent()
        elif phase == ResearchPhase.EXPERIMENT:
            self._delegate = ExperimentAgent()
        else:
            self._delegate = ReflectionAgent()
        self.output_model = self._delegate.output_model

    def allowed_tool_names(self) -> set[str] | None:
        return self._delegate.allowed_tool_names()

    def max_turns(self) -> int | None:
        return self._delegate.max_turns()

    def runtime_timeout_seconds(self) -> int | None:
        return self._delegate.runtime_timeout_seconds()

    def build_instructions(self, state: ResearchState) -> str:
        return self._delegate.build_instructions(state)

    def build_payload(self, state: ResearchState, tools: ResearchTools) -> dict[str, Any]:
        return self._delegate.build_payload(state, tools)

    def apply_output(self, state: ResearchState, tools: ResearchTools, output: Any) -> str:
        return self._delegate.apply_output(state, tools, output)

    def run(self, state: ResearchState, tools: ResearchTools, runtime: Any) -> str:
        if self.phase in {ResearchPhase.PREFLIGHT_VALIDATION, ResearchPhase.EXPERIMENT}:
            return self._delegate.run(state, tools, runtime)
        return super().run(state, tools, runtime)
