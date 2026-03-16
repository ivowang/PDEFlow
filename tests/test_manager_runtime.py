from __future__ import annotations

import hashlib
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from pydantic import BaseModel

from research_agents import AcquisitionAgent, ExperimentAgent, ExperimentPlannerAgent, PreflightValidationAgent
from config import ResearchBriefConfig, RuntimeConfig, SystemConfig
from memory import ResearchMemory
from orchestration import ResearchManager
from orchestration.failures import classify_state_failures
from integrations.command_grounding import ground_experiment_plan
from runtime import RuntimeAdapter
from runtime.provider import MaxTurnsExceeded
from state import (
    ArtifactDownloadMetadata,
    ArtifactRecord,
    ArtifactStatus,
    ArtifactValidationResult,
    CapabilityMatrix,
    EnvironmentRecord,
    EnvironmentResolutionState,
    EnvironmentSnapshot,
    ExperimentPhaseOutput,
    ExperimentPlan,
    ExperimentPlanningPhaseOutput,
    ExperimentRecord,
    EvaluationMemo,
    LiteraturePhaseOutput,
    PaperNote,
    RepositoryRecord,
    ResearchPhase,
    ResearchState,
    TaxonomyEntry,
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


class DummyRunResult:
    def __init__(self, final_output):
        self.final_output = final_output


class _FakeDataset:
    shape = (2, 3)
    ndim = 2

    def __getitem__(self, index):  # noqa: ANN001
        return 1


class _FakeH5File:
    def __init__(self, required_keys: list[str] | None = None):
        self._keys = required_keys or ["u"]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
        return False

    def keys(self):
        return self._keys

    def __contains__(self, item):  # noqa: ANN001
        return item in self._keys

    def __getitem__(self, item):  # noqa: ANN001
        return _FakeDataset()


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

    def test_prompt_json_validation_accepts_string_wrapped_json_object(self) -> None:
        adapter = RuntimeAdapter(RuntimeConfig(provider="openrouter"), session_db_path=":memory:")
        result = adapter._validate_prompt_json_output(
            '"{\\"summary\\":\\"wrapped\\",\\"count\\":4}"',
            DummyOutput,
        )
        self.assertEqual(result.summary, "wrapped")
        self.assertEqual(result.count, 4)

    def test_run_structured_retries_finalization_after_max_turns(self) -> None:
        adapter = RuntimeAdapter(RuntimeConfig(provider="openrouter", model="openai/gpt-5.4"), session_db_path=":memory:")
        adapter.ensure_ready = lambda: None  # type: ignore[method-assign]
        adapter._build_run_config = lambda specialist_name: None  # type: ignore[method-assign]
        adapter._build_session = lambda session_id: object()  # type: ignore[method-assign]
        calls: list[tuple[Any, int]] = []

        class _Result:
            def __init__(self, final_output: str):
                self.final_output = final_output

        def fake_run_sync(agent, payload, session, run_config, max_turns):  # noqa: ANN001
            calls.append((agent, max_turns))
            if len(calls) == 1:
                raise MaxTurnsExceeded("Max turns (32) exceeded")
            return _Result('{"summary":"finalized","count":2}')

        adapter._run_sync_with_session = fake_run_sync  # type: ignore[method-assign]
        result = adapter.run_structured(
            specialist_name="AcquisitionAgent",
            instructions="Return JSON.",
            payload={"x": 1},
            session_id="session-1",
            output_type=DummyOutput,
            tools=["tool"],
        )
        self.assertEqual(result.summary, "finalized")
        self.assertEqual(result.count, 2)
        self.assertEqual(len(calls), 2)

    def test_run_structured_respects_runtime_timeout(self) -> None:
        adapter = RuntimeAdapter(RuntimeConfig(provider="openrouter", model="openai/gpt-5.4"), session_db_path=":memory:")
        adapter.ensure_ready = lambda: None  # type: ignore[method-assign]
        adapter._build_run_config = lambda specialist_name: None  # type: ignore[method-assign]
        adapter._build_session = lambda session_id: object()  # type: ignore[method-assign]

        def fake_run_sync(agent, payload, session, run_config, max_turns):  # noqa: ANN001
            time.sleep(2.0)
            raise AssertionError("runtime timeout did not interrupt the call")

        adapter._run_sync_with_session = fake_run_sync  # type: ignore[method-assign]
        with self.assertRaises(TimeoutError):
            adapter.run_structured(
                specialist_name="AcquisitionAgent",
                instructions="Return JSON.",
                payload={"x": 1},
                session_id="session-timeout",
                output_type=DummyOutput,
                tools=["tool"],
                runtime_timeout_seconds=1,
            )

    def test_run_structured_without_tools_uses_direct_completion_path(self) -> None:
        adapter = RuntimeAdapter(RuntimeConfig(provider="openrouter", model="openai/gpt-5.4"), session_db_path=":memory:")
        adapter.ensure_ready = lambda: None  # type: ignore[method-assign]
        adapter._run_sync_with_session = lambda *args, **kwargs: (_ for _ in ()).throw(  # type: ignore[method-assign]
            AssertionError("tool-free phases should not invoke Runner.run_sync")
        )
        adapter._run_direct_text_completion = lambda **kwargs: '{"summary":"ok","count":3}'  # type: ignore[method-assign]
        result = adapter.run_structured(
            specialist_name="ProblemFramingAgent",
            instructions="Return JSON.",
            payload={"x": 1},
            session_id="session-direct",
            output_type=DummyOutput,
            tools=[],
        )
        self.assertEqual(result.summary, "ok")
        self.assertEqual(result.count, 3)

    def test_runtime_extracts_affordable_token_budget_from_openrouter_error(self) -> None:
        adapter = RuntimeAdapter(RuntimeConfig(provider="openrouter", model="openai/gpt-5.4", max_output_tokens=4096), session_db_path=":memory:")
        reduced = adapter._reduced_max_output_tokens(
            Exception("This request requires fewer max_tokens. You requested up to 4096 tokens, but can only afford 2546."),
            4096,
        )
        self.assertEqual(reduced, 2546)

    def test_runtime_extracts_small_affordable_token_budget_from_openrouter_error(self) -> None:
        adapter = RuntimeAdapter(RuntimeConfig(provider="openrouter", model="openai/gpt-5.4", max_output_tokens=4096), session_db_path=":memory:")
        reduced = adapter._reduced_max_output_tokens(
            Exception("This request requires fewer max_tokens. You requested up to 256 tokens, but can only afford 151."),
            256,
        )
        self.assertEqual(reduced, 151)

    def test_run_structured_retries_with_compacted_payload_on_prompt_limit(self) -> None:
        adapter = RuntimeAdapter(RuntimeConfig(provider="openrouter", model="openai/gpt-5.4"), session_db_path=":memory:")
        adapter.ensure_ready = lambda: None  # type: ignore[method-assign]
        payload_lengths: list[int] = []

        def _completion(**kwargs):
            payload_text = kwargs["payload_text"]
            payload_lengths.append(len(payload_text))
            if len(payload_lengths) == 1:
                raise Exception("Prompt tokens limit exceeded: 2000 > 900.")
            return '{"summary":"ok","count":1}'

        adapter._run_direct_text_completion = _completion  # type: ignore[method-assign]
        result = adapter.run_structured(
            specialist_name="ProblemFramingAgent",
            instructions="Return JSON.",
            payload={"long_text": "x" * 4000, "items": ["y" * 400 for _ in range(6)]},
            session_id="session-compaction",
            output_type=DummyOutput,
            tools=[],
        )
        self.assertEqual(result.summary, "ok")
        self.assertEqual(len(payload_lengths), 2)
        self.assertLess(payload_lengths[1], payload_lengths[0])

    def test_tool_run_retries_with_compacted_payload_on_prompt_limit(self) -> None:
        adapter = RuntimeAdapter(RuntimeConfig(provider="openrouter", model="openai/gpt-5.4"), session_db_path=":memory:")
        adapter.ensure_ready = lambda: None  # type: ignore[method-assign]
        adapter._build_session = lambda session_id: None  # type: ignore[method-assign]
        adapter._build_run_config_with_budget = lambda specialist_name, max_output_tokens: None  # type: ignore[method-assign]
        payload_lengths: list[int] = []

        def _run_sync(agent, payload, session, run_config, max_turns):
            payload_lengths.append(len(payload))
            if len(payload_lengths) == 1:
                raise Exception("Prompt tokens limit exceeded: 2200 > 900.")
            return DummyRunResult({"summary": "ok", "count": 2})

        adapter._run_sync_with_session = _run_sync  # type: ignore[method-assign]
        result = adapter.run_structured(
            specialist_name="AcquisitionAgent",
            instructions="Return JSON.",
            payload={"long_text": "x" * 5000, "items": ["y" * 500 for _ in range(6)]},
            session_id="session-tool-compaction",
            output_type=DummyOutput,
            tools=[object()],
        )
        self.assertEqual(result.count, 2)
        self.assertEqual(len(payload_lengths), 2)
        self.assertLess(payload_lengths[1], payload_lengths[0])


class ToolWhitelistTests(unittest.TestCase):
    def test_acquisition_agent_does_not_expose_arbitrary_shell_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_config()
            memory = ResearchMemory(root=root)
            tools = ResearchTools(config=config, memory=memory, repo_root=Path("/root/PDEFlow"))
            tool_names = {
                getattr(tool, "name", getattr(tool, "__name__", None))
                for tool in AcquisitionAgent().build_tools(tools)
            }
            self.assertIn("clone_repository", tool_names)
            self.assertIn("discover_local_artifacts", tool_names)
            self.assertIn("download_file", tool_names)
            self.assertNotIn("compute_file_checksum", tool_names)
            self.assertNotIn("validate_artifact", tool_names)
            self.assertNotIn("run_command", tool_names)
            self.assertNotIn("run_in_environment", tool_names)
            self.assertNotIn("inspect_python_environment", tool_names)

    def test_run_command_uses_virtual_env_python_from_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fake_env = root / "fake-env"
            fake_bin = fake_env / "bin"
            fake_bin.mkdir(parents=True, exist_ok=True)
            fake_python = fake_bin / "python"
            fake_python.write_text("#!/bin/sh\necho fake-managed-python\n", encoding="utf-8")
            fake_python.chmod(0o755)
            config = make_config()
            memory = ResearchMemory(root=root / "run")
            tools = ResearchTools(config=config, memory=memory, repo_root=Path("/root/PDEFlow"))
            result = tools.run_command(
                "python -c \"print('system-python-should-not-run')\"",
                cwd=root,
                env_overrides={"VIRTUAL_ENV": str(fake_env)},
                emit_progress=False,
            )
            self.assertEqual(result["returncode"], 0)
            self.assertIn("fake-managed-python", str(result["stdout_tail"]))

    def test_run_command_sets_pytorch_cuda_check_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_config()
            memory = ResearchMemory(root=root / "run")
            tools = ResearchTools(config=config, memory=memory, repo_root=Path("/root/PDEFlow"))
            result = tools.run_command(
                "python3 -c \"import os; print(os.getenv('PYTORCH_NVML_BASED_CUDA_CHECK', 'missing'))\"",
                cwd=root,
                emit_progress=False,
            )
            self.assertEqual(result["returncode"], 0)
            self.assertEqual(str(result["stdout_tail"]).strip().splitlines()[-1], "1")

    def test_run_command_sets_cuda_device_order_for_gpu_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_config()
            memory = ResearchMemory(root=root / "run")
            tools = ResearchTools(config=config, memory=memory, repo_root=Path("/root/PDEFlow"))
            result = tools.run_command(
                "python3 -c \"import os; print(os.getenv('CUDA_DEVICE_ORDER', 'missing')); print(os.getenv('CUDA_VISIBLE_DEVICES', 'missing'))\"",
                cwd=root,
                gpu_ids=[6, 7],
                emit_progress=False,
            )
            self.assertEqual(result["returncode"], 0)
            lines = str(result["stdout_tail"]).strip().splitlines()
            self.assertEqual(lines[-2], "PCI_BUS_ID")
            self.assertEqual(lines[-1], "6,7")

    def test_capability_probe_prefers_repo_environment_over_auxiliary_validator(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_config(run_name="capability-prefer-repo-env")
            memory = ResearchMemory(root=root)
            tools = ResearchTools(config=config, memory=memory, repo_root=Path("/root/PDEFlow"))
            validator_python = tools.managed_env_root / "artifact-validator" / "bin" / "python"
            repo_python = tools.managed_env_root / "pdebench-env" / "bin" / "python"
            validator_python.parent.mkdir(parents=True, exist_ok=True)
            repo_python.parent.mkdir(parents=True, exist_ok=True)
            validator_python.write_text("")
            repo_python.write_text("")

            selected = tools._discover_environment_path(
                str(validator_python.parent.parent),
                repository_paths=[str(root / "external_assets" / "repos" / "pdebench")],
            )
            self.assertEqual(selected, str(repo_python.parent.parent.resolve()))

    def test_acquisition_agent_uses_phase_specific_turn_budget(self) -> None:
        self.assertEqual(AcquisitionAgent().max_turns(), 8)

    def test_inspect_python_environment_uses_find_spec_not_module_import(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_config()
            memory = ResearchMemory(root=root)
            tools = ResearchTools(config=config, memory=memory, repo_root=Path("/root/PDEFlow"))

            commands: list[str] = []

            def fake_run_command(command: str, **kwargs):  # noqa: ANN001
                commands.append(command)
                if "import sys" in command:
                    return {"returncode": 0, "stdout_tail": "/tmp/python\n3.10.12", "stderr_tail": "", "cwd": str(root), "log_path": None}
                if "-m pip --version" in command:
                    return {"returncode": 0, "stdout_tail": "pip 26.0.1", "stderr_tail": "", "cwd": str(root), "log_path": None}
                return {
                    "returncode": 0,
                    "stdout_tail": '{"available": true}',
                    "stderr_tail": "",
                    "cwd": str(root),
                    "log_path": None,
                }

            tools.run_command = fake_run_command  # type: ignore[method-assign]
            env_dir = root / "envs" / "probe-env" / "bin"
            env_dir.mkdir(parents=True, exist_ok=True)
            (env_dir / "python").write_text("")

            payload = tools.inspect_python_environment(str(root / "envs" / "probe-env"), modules=["torchvision"])
            self.assertTrue(payload["modules"]["torchvision"])
            joined = "\n".join(commands)
            self.assertIn("importlib.util.find_spec('torchvision')", joined)
            self.assertNotIn("import torchvision", joined)

    def test_acquisition_download_defers_optional_dataset_when_ready_route_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_config()
            memory = ResearchMemory(root=root)
            tools = ResearchTools(config=config, memory=memory, repo_root=Path("/root/PDEFlow"))
            repo_root = root / "external_assets" / "repos" / "pdebench"
            repo_root.mkdir(parents=True, exist_ok=True)
            ready_data = root / "external_assets" / "data" / "1D" / "Burgers" / "Train" / "1D_Burgers_Sols_Nu0.01.hdf5"
            ready_data.parent.mkdir(parents=True, exist_ok=True)
            ready_data.write_bytes(b"payload")
            memory.record_repository(
                RepositoryRecord(
                    repo_id="pdebench",
                    canonical_id="github-pdebench-pdebench",
                    name="pdebench",
                    remote_url="https://github.com/pdebench/PDEBench.git",
                    local_path=str(repo_root),
                    bootstrap_status="cloned",
                )
            )
            memory.record_environment(
                EnvironmentRecord(
                    env_id="pdebench-env",
                    canonical_id="pdebench-env",
                    project_path=str(repo_root),
                    environment_path=str(root / "envs" / "pdebench-env"),
                    state=EnvironmentResolutionState.READY,
                )
            )
            memory.record_artifact(
                ArtifactRecord(
                    artifact_id="burgers-ready",
                    canonical_id="burgers-ready",
                    artifact_type="dataset",
                    title="1D_Burgers_Sols_Nu0.01.hdf5",
                    rationale="ready baseline shard",
                    local_path=str(ready_data),
                    status=ArtifactStatus.READY_FOR_TRAINING.value,
                    validation=ArtifactValidationResult(
                        validator="test",
                        status=ArtifactStatus.READY_FOR_TRAINING,
                        exists=True,
                        size_bytes=ready_data.stat().st_size,
                        size_ok=True,
                        format_valid=True,
                        ready_for_training=True,
                    ),
                )
            )
            tools.set_runtime_context(phase="acquisition", cycle_index=0)
            payload = tools.download_file(
                url="https://example.com/ReacDiff_Nu0.5_Rho1.0.hdf5",
                target_path=str(root / "external_assets" / "datasets" / "1D" / "ReactionDiffusion" / "Train" / "ReacDiff_Nu0.5_Rho1.0.hdf5"),
                artifact_id="reacdiff-default",
                artifact_type="dataset",
            )
            self.assertEqual(payload["validation_status"], "deferred_optional_dataset_download")
            self.assertTrue(payload["reused_existing"])
            self.assertEqual(payload["ready_route_artifact_path"], str(ready_data))

    def test_search_arxiv_papers_truncates_results_for_context_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_config()
            memory = ResearchMemory(root=root)
            tools = ResearchTools(config=config, memory=memory, repo_root=Path("/root/PDEFlow"))
            entry_xml = """
            <entry>
              <id>http://arxiv.org/abs/1234.5678v1</id>
              <title> Sample Paper {i} </title>
              <summary>{summary}</summary>
              <published>2024-01-01T00:00:00Z</published>
              <author><name>Alice</name></author>
              <author><name>Bob</name></author>
              <link title="pdf" href="https://arxiv.org/pdf/1234.5678v1"/>
            </entry>
            """
            xml = (
                "<?xml version='1.0' encoding='UTF-8'?>"
                "<feed xmlns='http://www.w3.org/2005/Atom'>"
                + "".join(
                    entry_xml.format(i=index, summary=("x" * 1200))
                    for index in range(6)
                )
                + "</feed>"
            )

            class _Response:
                text = xml

                def raise_for_status(self) -> None:
                    return None

            with patch("tools.retrieval.httpx.get", return_value=_Response()):
                results = tools.search_arxiv_papers("pdebench", max_results=10)
            self.assertEqual(len(results), 5)
            self.assertTrue(all(len(item["abstract"]) <= 600 for item in results))

    def test_search_arxiv_papers_returns_empty_on_rate_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_config()
            memory = ResearchMemory(root=root)
            tools = ResearchTools(config=config, memory=memory, repo_root=Path("/root/PDEFlow"))
            request = httpx.Request("GET", "https://export.arxiv.org/api/query")
            response = httpx.Response(429, request=request)
            with patch(
                "tools.retrieval.httpx.get",
                side_effect=httpx.HTTPStatusError("rate limited", request=request, response=response),
            ):
                results = tools.search_arxiv_papers("pdebench", max_results=5)
            self.assertEqual(results, [])


class EnvironmentSelectionTests(unittest.TestCase):
    def test_preflight_prefers_plan_virtual_env_over_stale_capability_env(self) -> None:
        state = ResearchState(
            project_name="test-project",
            run_name="preflight-env-selection",
            work_directory="/tmp/current-run",
            research_brief=make_config().research_brief,
            capability_matrix=CapabilityMatrix(
                environment_path="/tmp/stale-run/envs/pdebench-env",
                repo_ready=True,
                env_ready=True,
                codepath_ready=True,
                dataset_ready=True,
                baseline_launch_ready=True,
                experiment_plan_ready=True,
                scientific_iteration_ready=True,
                python_available=True,
                pip_available=True,
                torch_available=True,
                torch_import_ok=True,
                torch_runtime_ready=True,
                cuda_available=True,
                gpu_runtime_required=True,
                gpu_runtime_ready=True,
                h5py_available=True,
                hydra_available=True,
                pdebench_trainable=True,
                deepxde_installed=False,
                pinn_ready=False,
                fno_ready=True,
                unet_ready=True,
                target_dataset_ready=True,
                target_dataset_preparing=False,
                target_dataset_blocked=False,
                fallback_assets_available=False,
                baseline_ready_to_launch=True,
                environment_repair_needed=False,
            ),
            environment_records=[
                EnvironmentRecord(
                    env_id="current-env",
                    canonical_id="current-env",
                    project_path="/tmp/project",
                    environment_path="/tmp/current-run/envs/pdebench-env",
                    state=EnvironmentResolutionState.READY,
                )
            ],
        )
        plan = ExperimentPlan(
            plan_id="gpu-smoke",
            title="GPU smoke",
            program_id="baseline-1",
            job_kind="experiment",
            working_directory="/tmp/project",
            launch_command="python -c \"print('ok')\"",
            environment={"VIRTUAL_ENV": "/tmp/current-run/envs/pdebench-env"},
            log_path="/tmp/gpu-smoke.log",
        )
        selected = PreflightValidationAgent()._preferred_environment_path(state, [plan])
        self.assertEqual(selected, "/tmp/current-run/envs/pdebench-env")

    def test_manager_prefers_current_run_ready_env_over_stale_ready_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            config = make_config(run_name="manager-env-selection")
            manager = ResearchManager(config=config, repo_root=repo_root)
            current_env = manager.work_directory / "envs" / "pdebench-env"
            stale_env = manager.work_directory.parent / "older-run" / "envs" / "pdebench-env"
            (current_env / "bin").mkdir(parents=True, exist_ok=True)
            (stale_env / "bin").mkdir(parents=True, exist_ok=True)
            (current_env / "bin" / "python").write_text("", encoding="utf-8")
            (stale_env / "bin" / "python").write_text("", encoding="utf-8")
            state = manager._initial_state()
            state.environment_records = [
                EnvironmentRecord(
                    env_id="stale-env",
                    canonical_id="stale-env",
                    project_path="/tmp/project",
                    environment_path=str(stale_env),
                    state=EnvironmentResolutionState.READY,
                ),
                EnvironmentRecord(
                    env_id="current-env",
                    canonical_id="current-env",
                    project_path="/tmp/project",
                    environment_path=str(current_env),
                    state=EnvironmentResolutionState.READY,
                ),
            ]
            selected = manager._select_preferred_environment_path(state)
            self.assertEqual(selected, str(current_env))

    def test_gpu_required_envs_do_not_reuse_sibling_envs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_config()
            memory = ResearchMemory(root=root / "run")
            tools = ResearchTools(config=config, memory=memory, repo_root=Path("/root/PDEFlow"))
            self.assertTrue(tools._allow_sibling_environment_reuse(require_gpu_runtime=True))
            self.assertTrue(tools._allow_sibling_environment_reuse(require_gpu_runtime=False))

    def test_runtime_dependency_split_treats_pyro_as_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_config()
            memory = ResearchMemory(root=root / "run")
            tools = ResearchTools(config=config, memory=memory, repo_root=Path("/root/PDEFlow"))
            base, runtime = tools._split_base_and_runtime_dependencies(
                ["scipy", "pyro-ppl", "torch~=2.7", "hydra-core"]
            )
            self.assertEqual(base, ["scipy", "hydra-core"])
            self.assertEqual(runtime, ["pyro-ppl", "torch~=2.7"])

    def test_probe_torch_cuda_uses_selected_gpu_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_config()
            memory = ResearchMemory(root=root / "run")
            tools = ResearchTools(config=config, memory=memory, repo_root=Path("/root/PDEFlow"))
            observed_gpu_ids: list[int] | None = None

            def fake_run_command(command: str, **kwargs):  # noqa: ANN001
                nonlocal observed_gpu_ids
                observed_gpu_ids = kwargs.get("gpu_ids")
                return {
                    "returncode": 0,
                    "stdout_tail": '{"torch":"2.7.1+cu118","cuda":"11.8","available":true}',
                    "stderr_tail": "",
                    "cwd": str(root),
                    "log_path": None,
                }

            tools.run_command = fake_run_command  # type: ignore[method-assign]
            tools.inspect_compute_environment = lambda: EnvironmentSnapshot(  # type: ignore[method-assign]
                python_executable="/usr/bin/python3",
                python_version="3.10.12",
                uv_available=True,
                available_gpu_ids=[0, 1, 6, 7],
                selected_gpu_ids=[6, 7],
            )
            payload = tools._probe_torch_cuda(root / "env" / "bin" / "python", root)
            self.assertEqual(observed_gpu_ids, [6, 7])
            self.assertTrue(payload["available"])

    def test_find_reusable_environment_prefers_cached_gpu_ready_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_config()
            memory = ResearchMemory(root=root / "current-run")
            tools = ResearchTools(config=config, memory=memory, repo_root=Path("/root/PDEFlow"))

            stale_run = root / "older-run"
            good_run = root / "gpu-ready-run"
            current_target = root / "current-run" / "envs" / "pdebench-env"
            for run_root in (stale_run, good_run):
                env_dir = run_root / "envs" / "pdebench-env" / "bin"
                env_dir.mkdir(parents=True, exist_ok=True)
                (env_dir / "python").write_text("", encoding="utf-8")
                state_dir = run_root / "state"
                state_dir.mkdir(parents=True, exist_ok=True)
            (stale_run / "state" / "capability_matrix.jsonl").write_text(
                json.dumps(
                    {
                        "environment_path": str(stale_run / "envs" / "pdebench-env"),
                        "gpu_runtime_ready": False,
                        "env_ready": True,
                        "h5py_available": True,
                        "hydra_available": True,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (good_run / "state" / "capability_matrix.jsonl").write_text(
                json.dumps(
                    {
                        "environment_path": str(good_run / "envs" / "pdebench-env"),
                        "gpu_runtime_ready": True,
                        "env_ready": True,
                        "h5py_available": True,
                        "hydra_available": True,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            tools._probe_reusable_environment = lambda *args, **kwargs: True  # type: ignore[method-assign]
            selected = tools._find_reusable_environment(
                environment_slug="pdebench-env",
                current_target=current_target,
                require_gpu_runtime=True,
            )
            self.assertEqual(selected, good_run / "envs" / "pdebench-env")

    def test_find_reusable_environment_dedupes_symlink_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_config()
            memory = ResearchMemory(root=root / "current-run")
            tools = ResearchTools(config=config, memory=memory, repo_root=Path("/root/PDEFlow"))

            source_run = root / "source-run"
            alias_run = root / "alias-run"
            source_env = source_run / "envs" / "pdebench-env"
            (source_env / "bin").mkdir(parents=True, exist_ok=True)
            (source_env / "bin" / "python").write_text("", encoding="utf-8")
            (alias_run / "envs").mkdir(parents=True, exist_ok=True)
            (alias_run / "envs" / "pdebench-env").symlink_to(source_env, target_is_directory=True)

            probe_calls: list[str] = []

            def fake_probe(path: Path, **kwargs):  # noqa: ANN001
                probe_calls.append(str(path.resolve()))
                return True

            tools._probe_reusable_environment = fake_probe  # type: ignore[method-assign]
            selected = tools._find_reusable_environment(
                environment_slug="pdebench-env",
                current_target=root / "current-run" / "envs" / "pdebench-env",
                require_gpu_runtime=True,
            )
            self.assertEqual(selected.resolve(), source_env.resolve())
            self.assertEqual(len(probe_calls), 1)

    def test_manager_bootstrap_requires_gpu_runtime_for_repo_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            manager = ResearchManager(config=make_config(run_name="bootstrap-gpu-required"), repo_root=repo_root)
            state = manager._initial_state()
            project = repo_root / "external_assets" / "repos" / "pdebench"
            project.mkdir(parents=True, exist_ok=True)
            (project / "pyproject.toml").write_text("[project]\nname='pdebench'\nversion='0.0.0'\n", encoding="utf-8")
            calls: list[bool] = []

            def fake_detect_project_manifests(path: str):  # noqa: ANN001
                return {"manifests": ["pyproject.toml"], "entrypoints": []}

            def fake_ensure_python_environment(project_path: str, **kwargs):  # noqa: ANN001
                calls.append(bool(kwargs.get("require_gpu_runtime")))
                return {
                    "status": "ready",
                    "environment_path": str(manager.work_directory / "envs" / "pdebench-env"),
                    "environment_name": "pdebench-env",
                }

            manager.tools.detect_project_manifests = fake_detect_project_manifests  # type: ignore[method-assign]
            manager.tools.ensure_python_environment = fake_ensure_python_environment  # type: ignore[method-assign]
            manager._attach_repository_record(
                state,
                local_path=str(project),
                repo_url="https://github.com/pdebench/PDEBench.git",
                repo_name="pdebench",
                repo_id="pdebench",
                resolution_source="test",
            )
            self.assertEqual(calls, [True])


class ExperimentPlanStatusTests(unittest.TestCase):
    def test_literature_phase_recovers_from_invalid_output_using_local_pdfs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = ResearchManager(make_config(run_name="lit-recovery"), repo_root=Path("/root/PDEFlow"))
            manager.work_directory = root
            manager.memory = ResearchMemory(root=root)
            manager.tools = ResearchTools(config=manager.config, memory=manager.memory, repo_root=Path("/root/PDEFlow"))
            manager.runtime = RuntimeAdapter(manager.config.runtime, session_db_path=str(manager.memory.sessions_db))
            literature_agent = manager.agents["literature"]
            dummy_pdf = root / "external_assets" / "paper.pdf"
            dummy_pdf.parent.mkdir(parents=True, exist_ok=True)
            dummy_pdf.write_bytes(b"%PDF-1.4")

            state = manager._initial_state()
            spec = manager.front_phases[0]

            literature_agent.run = lambda *args, **kwargs: (_ for _ in ()).throw(json.JSONDecodeError("bad", "", 0))  # type: ignore[method-assign]
            manager.tools.extract_pdf_text = lambda pdf_path, max_pages=4: {  # type: ignore[method-assign]
                "path": pdf_path,
                "pages_read": 1,
                "text": "Fourier Neural Operator and PDEBench benchmark evidence.",
            }
            manager.runtime.run_structured = lambda **kwargs: LiteraturePhaseOutput(  # type: ignore[method-assign]
                summary="Recovered literature summary",
                literature_notes=[
                    PaperNote(
                        paper_id="paper-1",
                        title="Recovered FNO paper",
                        source_url="https://example.com/fno",
                    )
                ],
                method_taxonomy=[TaxonomyEntry(category="operator", methods=["FNO"])],
                open_questions=["How to stabilize short-window rollout?"],
                semantic_notes=["Recovered from local PDF evidence."],
                next_actions=["Proceed to acquisition."],
            )

            summary = manager._run_phase(state, spec)
            self.assertEqual(summary, "Recovered literature summary")
            self.assertEqual(len(state.literature_notes), 1)
            self.assertEqual(state.literature_notes[0].title, "Recovered FNO paper")

    def test_literature_phase_falls_back_to_heuristic_sources_when_recovery_model_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = ResearchManager(make_config(run_name="lit-heuristic-fallback"), repo_root=Path("/root/PDEFlow"))
            manager.work_directory = root
            manager.memory = ResearchMemory(root=root)
            manager.tools = ResearchTools(config=manager.config, memory=manager.memory, repo_root=Path("/root/PDEFlow"))
            manager.runtime = RuntimeAdapter(manager.config.runtime, session_db_path=str(manager.memory.sessions_db))
            literature_agent = manager.agents["literature"]
            dummy_pdf = root / "external_assets" / "paper.pdf"
            dummy_pdf.parent.mkdir(parents=True, exist_ok=True)
            dummy_pdf.write_bytes(b"%PDF-1.4")

            state = manager._initial_state()
            spec = manager.front_phases[0]

            literature_agent.run = lambda *args, **kwargs: (_ for _ in ()).throw(json.JSONDecodeError("bad", "", 0))  # type: ignore[method-assign]
            manager.tools.extract_pdf_text = lambda pdf_path, max_pages=4: {  # type: ignore[method-assign]
                "path": pdf_path,
                "pages_read": 1,
                "text": "PDEBench benchmark paper. Fourier Neural Operator provides a practical neural-operator baseline. Physics-informed operator learning can add residual losses.",
            }
            manager.runtime.run_structured = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("provider unavailable"))  # type: ignore[method-assign]

            summary = manager._run_phase(state, spec)
            self.assertIn("Recovered literature", summary)
            self.assertGreaterEqual(len(state.literature_notes), 1)
            self.assertGreaterEqual(len(state.method_taxonomy), 1)
            self.assertGreaterEqual(len(state.open_questions), 1)

    def test_runtime_degraded_skips_live_model_call_for_recoverable_phase(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = ResearchManager(make_config(run_name="runtime-degraded"), repo_root=Path("/root/PDEFlow"))
            manager.work_directory = root
            manager.memory = ResearchMemory(root=root)
            manager.tools = ResearchTools(config=manager.config, memory=manager.memory, repo_root=Path("/root/PDEFlow"))
            manager.runtime = RuntimeAdapter(manager.config.runtime, session_db_path=str(manager.memory.sessions_db))
            state = manager._initial_state()
            spec = manager.iterative_phases[0]  # hypothesis
            agent = manager.agents[spec.agent_key]
            manager._runtime_degraded = True

            agent.run = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("live model should be bypassed"))  # type: ignore[method-assign]

            summary = manager._run_phase(state, spec)
            self.assertIn("Recovered", summary)
            self.assertGreaterEqual(len(state.hypotheses), 1)

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

    def test_preflight_blocks_pinn_plan_when_backend_is_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_config()
            memory = ResearchMemory(root=root)
            tools = ResearchTools(config=config, memory=memory, repo_root=Path("/root/PDEFlow"))
            entrypoint = root / "train.py"
            entrypoint.write_text("print('ok')\n", encoding="utf-8")
            ready_artifact = ArtifactRecord(
                artifact_id="dataset-ready",
                canonical_id="dataset-ready",
                artifact_type="dataset",
                title="dataset",
                rationale="ready",
                local_path=str(root / "dataset.hdf5"),
                status=ArtifactStatus.READY_FOR_TRAINING.value,
                validation=ArtifactValidationResult(
                    validator="test",
                    status=ArtifactStatus.READY_FOR_TRAINING,
                    exists=True,
                    ready_for_training=True,
                ),
            )
            report = tools.preflight_experiment_plan(
                ExperimentPlan(
                    plan_id="plan-pinn",
                    title="PINN baseline",
                    program_id="prog-pinn",
                    working_directory=str(root),
                    launch_command="python train.py",
                    required_artifact_ids=["dataset-ready"],
                    log_path=str(root / "logs" / "plan-pinn.log"),
                ),
                [ready_artifact],
                CapabilityMatrix(
                    python_available=True,
                    pip_available=True,
                    pinn_ready=False,
                    deepxde_installed=True,
                    deepxde_backend=None,
                    tensorflow_available=False,
                ),
            )
            self.assertFalse(report.passed)
            self.assertIn("pinn_backend_ready", [item.name for item in report.failed_checks])

    def test_experiment_planner_filters_incompatible_and_stages_initial_wave(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_config()
            memory = ResearchMemory(root=root)
            tools = ResearchTools(config=config, memory=memory, repo_root=Path("/root/PDEFlow"))
            (root / "train_pinn.py").write_text("print('pinn')\n", encoding="utf-8")
            (root / "train_fno.py").write_text("print('fno')\n", encoding="utf-8")
            state = ResearchState(
                project_name=config.project_name,
                run_name=config.run_name,
                work_directory=str(root),
                research_brief=config.research_brief,
                selected_baseline_program_id="prog-fno",
                capability_matrix=CapabilityMatrix(
                    fno_ready=True,
                    pinn_ready=False,
                    torch_runtime_ready=True,
                    h5py_available=True,
                ),
            )
            output = ExperimentPlanningPhaseOutput(
                summary="planned two routes",
                experiment_plans=[
                    ExperimentPlan(
                        plan_id="plan-pinn",
                        title="PINN comparator",
                        program_id="prog-pinn",
                        working_directory=str(root),
                        launch_command="python train_pinn.py",
                        log_path=str(root / "logs" / "plan-pinn.log"),
                    ),
                    ExperimentPlan(
                        plan_id="plan-fno",
                        title="FNO baseline",
                        program_id="prog-fno",
                        working_directory=str(root),
                        launch_command="python train_fno.py",
                        log_path=str(root / "logs" / "plan-fno.log"),
                    ),
                ],
            )
            summary = ExperimentPlannerAgent().apply_output(state, tools, output)
            self.assertIn("Filtered plan-pinn", summary)
            plans_by_id = {item.plan_id: item for item in state.experiment_plans}
            self.assertEqual(plans_by_id["plan-fno"].status, "planned")
            self.assertEqual(plans_by_id["plan-pinn"].status, "blocked")

    def test_experiment_planner_blocks_malformed_placeholder_launch_and_retains_executable_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_config()
            memory = ResearchMemory(root=root)
            tools = ResearchTools(config=config, memory=memory, repo_root=Path("/root/PDEFlow"))
            entrypoint = root / "smoke.py"
            entrypoint.write_text("print('ok')\n", encoding="utf-8")
            state = ResearchState(
                project_name=config.project_name,
                run_name=config.run_name,
                work_directory=str(root),
                research_brief=config.research_brief,
                selected_baseline_program_id="prog-baseline",
                capability_matrix=CapabilityMatrix(
                    fno_ready=True,
                    torch_runtime_ready=True,
                    h5py_available=True,
                ),
            )
            output = ExperimentPlanningPhaseOutput(
                summary="planner emitted baseline and smoke plans",
                experiment_plans=[
                    ExperimentPlan(
                        plan_id="plan-bad",
                        title="Malformed baseline",
                        program_id="prog-baseline",
                        working_directory=str(root),
                        launch_command="echo 'Launch command truncated in malformed output'",
                        log_path=str(root / "logs" / "bad.log"),
                    ),
                    ExperimentPlan(
                        plan_id="plan-smoke",
                        title="Smoke",
                        program_id="prog-smoke",
                        working_directory=str(root),
                        launch_command=f"python {entrypoint.name}",
                        log_path=str(root / "logs" / "smoke.log"),
                    ),
                ],
            )
            summary = ExperimentPlannerAgent().apply_output(state, tools, output)
            self.assertIn("Filtered plan-bad", summary)
            plans_by_id = {item.plan_id: item for item in state.experiment_plans}
            self.assertEqual(plans_by_id["plan-bad"].status, "blocked")
            self.assertEqual(plans_by_id["plan-smoke"].status, "planned")

    def test_experiment_planner_injects_gpu_smoke_before_first_long_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo = root / "repo"
            repo.mkdir(parents=True, exist_ok=True)
            config = make_config()
            memory = ResearchMemory(root=root)
            tools = ResearchTools(config=config, memory=memory, repo_root=Path("/root/PDEFlow"))
            state = ResearchState(
                project_name=config.project_name,
                run_name=config.run_name,
                work_directory=str(root),
                research_brief=config.research_brief,
                selected_baseline_program_id="prog-baseline",
                repositories=[
                    RepositoryRecord(
                        repo_id="pdebench",
                        canonical_id="pdebench",
                        name="pdebench",
                        remote_url="https://github.com/pdebench/PDEBench",
                        local_path=str(repo),
                        bootstrap_status="ready",
                        environment_path=str(root / "envs" / "pdebench-env"),
                    )
                ],
                external_artifacts=[
                    ArtifactRecord(
                        artifact_id="dataset-burgers",
                        canonical_id="dataset-burgers",
                        artifact_type="dataset",
                        title="1D_Burgers_Sols_Nu0.01.hdf5",
                        rationale="ready",
                        local_path=str(root / "data" / "1D_Burgers_Sols_Nu0.01.hdf5"),
                        status=ArtifactStatus.READY_FOR_TRAINING.value,
                    )
                ],
                capability_matrix=CapabilityMatrix(
                    fno_ready=True,
                    gpu_runtime_ready=True,
                    environment_path=str(root / "envs" / "pdebench-env"),
                ),
                environment_snapshot=EnvironmentSnapshot(
                    python_executable="/usr/bin/python3",
                    python_version="3.10.12",
                    uv_available=True,
                    selected_gpu_ids=[6],
                ),
            )
            output = ExperimentPlanningPhaseOutput(
                summary="baseline plan only",
                experiment_plans=[
                    ExperimentPlan(
                        plan_id="plan-baseline",
                        title="Baseline training",
                        program_id="prog-baseline",
                        working_directory=str(repo),
                        launch_command="python train.py",
                        log_path=str(root / "logs" / "baseline.log"),
                    )
                ],
            )
            summary = ExperimentPlannerAgent().apply_output(state, tools, output)
            self.assertIn("Added a deterministic GPU smoke experiment", summary)
            self.assertIn("gpu-smoke", state.experiment_plans[0].plan_id)
            self.assertIn("gpu_smoke_evidence", state.experiment_plans[0].notes)
            self.assertEqual(state.experiment_plans[0].status, "planned")

    def test_ground_experiment_plan_normalizes_managed_python_commands_without_artifact_match(self) -> None:
        plan = ExperimentPlan(
            plan_id="plan-uv",
            title="baseline",
            program_id="prog-1",
            working_directory="/tmp/repo",
            setup_commands=[
                "uv run --python /tmp/env/bin/python python -c \"print('ok')\"",
            ],
            launch_command=(
                "CUDA_VISIBLE_DEVICES=6 uv run --python /tmp/env/bin/python python "
                "train.py args.filename=missing.hdf5 > /tmp/train.log 2>&1"
            ),
            log_path="/tmp/train.log",
        )

        grounded, messages = ground_experiment_plan(plan, [])
        self.assertEqual(messages, [])
        self.assertEqual(
            grounded.setup_commands[0],
            "/tmp/env/bin/python -c \"print('ok')\"",
        )
        self.assertTrue(grounded.launch_command.startswith("CUDA_VISIBLE_DEVICES=6 /tmp/env/bin/python train.py"))

    def test_ground_experiment_plan_inserts_assignments_before_redirection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workdir = root / "repo"
            script = workdir / "train.py"
            config_dir = workdir / "config"
            config_dir.mkdir(parents=True, exist_ok=True)
            script.write_text("cfg.args.data_path\ncfg.args.root_path\n", encoding="utf-8")
            (config_dir / "config.yaml").write_text("data_path: /path/to/data\n", encoding="utf-8")
            dataset = root / "data" / "sample.hdf5"
            dataset.parent.mkdir(parents=True, exist_ok=True)
            dataset.write_bytes(b"data")
            artifact = ArtifactRecord(
                artifact_id="dataset-1",
                canonical_id="dataset-1",
                artifact_type="dataset",
                title=dataset.name,
                rationale="ready",
                local_path=str(dataset),
                status=ArtifactStatus.READY_FOR_TRAINING.value,
            )
            plan = ExperimentPlan(
                plan_id="plan-ground",
                title="baseline",
                program_id="prog-1",
                working_directory=str(workdir),
                launch_command=(
                    "uv run --python /tmp/env/bin/python python train.py "
                    "++args.filename=sample.hdf5 > /tmp/train.log 2>&1"
                ),
                log_path="/tmp/train.log",
            )
            grounded, _ = ground_experiment_plan(plan, [artifact])
            self.assertIn("++args.data_path=", grounded.launch_command)
            self.assertIn(" > /tmp/train.log 2>&1", grounded.launch_command)
            before_redirect, after_redirect = grounded.launch_command.split(" > ", 1)
            self.assertIn("++args.data_path=", before_redirect)
            self.assertNotIn("++args.data_path=", after_redirect)

    def test_experiment_agent_normalizes_managed_python_commands_before_execution(self) -> None:
        class _FakeTools:
            def __init__(self) -> None:
                self.commands: list[str] = []
                self.stall_timeouts: list[int | None] = []
                self.config = make_config()

            def run_command(self, command: str, **kwargs):  # noqa: ANN003
                self.commands.append(command)
                self.stall_timeouts.append(kwargs.get("stall_timeout_seconds"))
                return {
                    "returncode": 0,
                    "stderr_tail": "",
                    "stdout_tail": "",
                    "log_path": kwargs["log_path"],
                    "cwd": kwargs["cwd"],
                }

            def parse_metrics_file(self, path: str):  # noqa: ARG002
                return {"metrics": {}}

        tools = _FakeTools()
        agent = ExperimentAgent()
        plan = ExperimentPlan(
            plan_id="plan-uv",
            title="baseline",
            program_id="prog-1",
            working_directory="/tmp/repo",
            setup_commands=["uv run --python /tmp/env/bin/python python -c \"print('ok')\""],
            launch_command=(
                "CUDA_VISIBLE_DEVICES=6 uv run --python /tmp/env/bin/python python train.py > /tmp/train.log 2>&1"
            ),
            log_path="/tmp/train.log",
            preflight_status="passed",
        )
        record = agent._execute_plan(ResearchState(project_name="p", run_name="r", work_directory="/tmp", research_brief=make_config().research_brief), tools, plan)
        self.assertEqual(record.status, "completed")
        self.assertEqual(tools.commands[0], "/tmp/env/bin/python -c \"print('ok')\"")
        self.assertTrue(tools.commands[1].startswith("CUDA_VISIBLE_DEVICES=6 /tmp/env/bin/python train.py"))
        self.assertTrue(all(timeout == tools.config.execution.experiment_no_progress_timeout_seconds for timeout in tools.stall_timeouts))

    def test_experiment_agent_runs_deterministic_execution_and_parses_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_config()
            memory = ResearchMemory(root=root)
            tools = ResearchTools(config=config, memory=memory, repo_root=Path("/root/PDEFlow"))
            metrics_path = root / "metrics.json"
            state = ResearchState(
                project_name=config.project_name,
                run_name=config.run_name,
                work_directory=str(root),
                research_brief=config.research_brief,
                experiment_plans=[
                    ExperimentPlan(
                        plan_id="plan-fno",
                        title="FNO baseline",
                        program_id="prog-fno",
                        working_directory=str(root),
                        launch_command=(
                            "python3 - <<'PY'\n"
                            "from pathlib import Path\n"
                            "import json\n"
                            f"Path({str(metrics_path)!r}).write_text(json.dumps({{'rmse': 0.125, 'physics_residual': 0.04}}), encoding='utf-8')\n"
                            "print('rmse=0.125')\n"
                            "PY"
                        ),
                        preflight_status="passed",
                        expected_outputs=[str(metrics_path)],
                        log_path=str(root / "logs" / "plan-fno.log"),
                    )
                ],
            )
            summary = ExperimentAgent().run(state, tools, runtime=None)
            self.assertIn("completed=1", summary)
            self.assertEqual(state.experiment_records[0].status, "completed")
            self.assertAlmostEqual(state.experiment_records[0].metrics["rmse"], 0.125)
            self.assertIn("prog-fno", state.best_known_results)


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

    def test_manager_keeps_acquisition_active_while_exact_dataset_is_preparing(self) -> None:
        config = make_config(run_name="routing-preparing")
        manager = ResearchManager(config=config, repo_root=Path("/root/PDEFlow"))
        state = ResearchState(
            project_name=config.project_name,
            run_name=config.run_name,
            work_directory=str(manager.work_directory),
            research_brief=config.research_brief,
            capability_matrix=CapabilityMatrix(
                target_dataset_ready=False,
                target_dataset_preparing=True,
                target_dataset_blocked=False,
                baseline_ready_to_launch=False,
            ),
        )
        route = manager._select_cycle_route(state)
        self.assertEqual(route.route_id, "continue-dataset-preparation")
        self.assertEqual(len(route.phases), 1)
        self.assertEqual(route.phases[0].phase.value, "acquisition")

    def test_acquisition_phase_recovers_from_persisted_tool_side_effects(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_config(run_name="acq-recover")
            manager = ResearchManager(config=config, repo_root=Path("/root/PDEFlow"))
            manager.work_directory = root
            manager.memory = ResearchMemory(root=root)
            manager.tools = ResearchTools(config=config, memory=manager.memory, repo_root=Path("/root/PDEFlow"))
            state = ResearchState(
                project_name=config.project_name,
                run_name=config.run_name,
                work_directory=str(root),
                research_brief=config.research_brief,
                environment_snapshot=EnvironmentSnapshot(
                    python_executable="python",
                    python_version="3.10",
                    uv_available=True,
                ),
            )
            repo_root = root / "external_assets" / "repos" / "pdebench"
            repo_root.mkdir(parents=True, exist_ok=True)
            manager.memory.record_repository(
                RepositoryRecord(
                    repo_id="pdebench",
                    canonical_id="github-pdebench-pdebench",
                    name="pdebench",
                    remote_url="https://github.com/pdebench/PDEBench.git",
                    local_path=str(repo_root),
                    bootstrap_status="cloned",
                )
            )

            class _FailingAcquisitionAgent:
                name = "AcquisitionAgent"

                def run(self, state, tools, runtime):  # noqa: ANN001
                    raise MaxTurnsExceeded("Max turns (32) exceeded")

            manager.agents["acquisition"] = _FailingAcquisitionAgent()
            summary = manager._run_phase(
                state,
                manager.front_phases[1],
            )
            self.assertIn("Recovered acquisition state", summary)
            self.assertEqual(len(state.repositories), 1)
            self.assertEqual(state.repositories[0].canonical_id, "github-pdebench-pdebench")
            self.assertTrue(state.program_candidates)
            self.assertTrue(state.selected_baseline_program_id)

    def test_post_phase_sync_materializes_ready_local_artifact_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_config(run_name="materialize-local")
            manager = ResearchManager(config=config, repo_root=Path("/root/PDEFlow"))
            manager.work_directory = root
            manager.memory = ResearchMemory(root=root)
            manager.tools = ResearchTools(config=config, memory=manager.memory, repo_root=Path("/root/PDEFlow"))

            source_dir = root / "sibling" / "external_assets" / "data" / "1D" / "Burgers" / "Train"
            source_dir.mkdir(parents=True, exist_ok=True)
            source_path = source_dir / "1D_Burgers_Sols_Nu0.01.hdf5"
            source_path.write_bytes(b"payload")
            target_path = root / "external_assets" / "data" / "1D" / "Burgers" / "Train" / "1D_Burgers_Sols_Nu0.01.hdf5"

            state = ResearchState(
                project_name=config.project_name,
                run_name=config.run_name,
                work_directory=str(root),
                research_brief=config.research_brief,
                external_artifacts=[
                    ArtifactRecord(
                        artifact_id="pdebench-burgers-nu0.01",
                        canonical_id="pdebench-burgers-nu0.01",
                        artifact_type="dataset",
                        title="1D_Burgers_Sols_Nu0.01.hdf5",
                        rationale="expected artifact",
                        local_path=str(target_path),
                        status=ArtifactStatus.BLOCKED.value,
                        metadata={"min_size_bytes": 1, "required_keys": ["u"], "exact_target": True},
                    )
                ],
            )

            with patch.object(
                manager.tools,
                "discover_local_artifacts",
                return_value=[
                    {
                        "path": str(source_path),
                        "status": "ready_for_training",
                        "canonical_id": "pdebench-burgers-nu0.01",
                        "size_bytes": source_path.stat().st_size,
                        "ready_for_training": True,
                    }
                ],
            ), patch("tools.artifacts.h5py") as fake_h5py:
                fake_h5py.File.side_effect = lambda *args, **kwargs: _FakeH5File(["u"])
                manager._post_phase_sync(state, manager.front_phases[1])

            self.assertTrue(target_path.exists())
            self.assertEqual(state.external_artifacts[0].status, ArtifactStatus.READY_FOR_TRAINING.value)

    def test_post_phase_sync_auto_materializes_verified_remote_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shared_runs_root = root / "runs"
            prior_run = shared_runs_root / "prior-run" / "external_assets" / "datasets" / "1D" / "Burgers" / "Train"
            prior_run.mkdir(parents=True, exist_ok=True)
            source_path = prior_run / "1D_Burgers_Sols_Nu1.0.hdf5"
            payload = b"payload"
            source_path.write_bytes(payload)
            checksum = hashlib.md5(payload).hexdigest()

            config = make_config(run_name="auto-materialize")
            config.execution.work_directory = str(shared_runs_root / "{run_name}")
            manager = ResearchManager(config=config, repo_root=Path("/root/PDEFlow"))
            manager.work_directory = shared_runs_root / config.run_name
            manager.memory = ResearchMemory(root=manager.work_directory)
            manager.tools = ResearchTools(config=config, memory=manager.memory, repo_root=Path("/root/PDEFlow"))
            state = ResearchState(
                project_name=config.project_name,
                run_name=config.run_name,
                work_directory=str(manager.work_directory),
                research_brief=config.research_brief,
                external_artifacts=[
                    ArtifactRecord(
                        artifact_id="artifact-pdebench-burgers-nu1-train",
                        canonical_id="pdebench:1D/Burgers/Train/1D_Burgers_Sols_Nu1.0.hdf5",
                        artifact_type="dataset",
                        title="1D_Burgers_Sols_Nu1.0.hdf5",
                        rationale="verified remote shard",
                        source_url="https://example.com/1D_Burgers_Sols_Nu1.0.hdf5",
                        local_path=None,
                        status=ArtifactStatus.VERIFIED_REMOTE.value,
                        metadata={
                            "official_path": "1D/Burgers/Train",
                            "expected_md5": checksum,
                            "required_keys": ["u"],
                            "min_size_bytes": 1,
                            "exact_target": True,
                        },
                    )
                ],
            )
            with patch("tools.artifacts.h5py") as fake_h5py:
                fake_h5py.File.side_effect = lambda *args, **kwargs: _FakeH5File(["u"])
                manager._post_phase_sync(state, manager.front_phases[1])
            materialized = next(
                item for item in state.external_artifacts
                if item.title == "1D_Burgers_Sols_Nu1.0.hdf5"
            )
            self.assertEqual(materialized.status, ArtifactStatus.READY_FOR_TRAINING.value)
            self.assertTrue(Path(materialized.local_path).exists())

    def test_load_evaluation_memos_ignores_evaluation_notes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            memory = ResearchMemory(root=root)
            memo = EvaluationMemo(
                memo_id="memo-1",
                cycle_index=1,
                phase="preflight_validation",
                verdict="preflight_passed",
                support_level="evidence_generated",
                summary="memo",
                body="body",
                plan_id="plan-1",
                program_id="prog-1",
            )
            memory.record_evaluation_memo(memo)
            loaded = memory.load_evaluation_memos()
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].memo_id, "memo-1")

    def test_auto_materialize_ignores_sibling_run_local_path_when_choosing_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shared_runs_root = root / "runs"
            config = make_config(run_name="auto-materialize-sibling")
            config.execution.work_directory = str(shared_runs_root / "{run_name}")
            manager = ResearchManager(config=config, repo_root=Path("/root/PDEFlow"))
            manager.work_directory = shared_runs_root / config.run_name
            manager.memory = ResearchMemory(root=manager.work_directory)
            manager.tools = ResearchTools(config=config, memory=manager.memory, repo_root=Path("/root/PDEFlow"))

            sibling_path = shared_runs_root / "older-run" / "external_assets" / "datasets" / "1D" / "Burgers" / "Train" / "1D_Burgers_Sols_Nu1.0.hdf5"
            sibling_path.parent.mkdir(parents=True, exist_ok=True)
            sibling_path.write_bytes(b"payload")
            expected_target = manager.tools.shared_workspace_root / "datasets" / "1D" / "Burgers" / "Train" / "1D_Burgers_Sols_Nu1.0.hdf5"

            artifact = ArtifactRecord(
                artifact_id="artifact-pdebench-burgers-nu1-train",
                canonical_id="pdebench:1D/Burgers/Train/1D_Burgers_Sols_Nu1.0.hdf5",
                artifact_type="dataset",
                title="1D_Burgers_Sols_Nu1.0.hdf5",
                rationale="verified remote shard",
                source_url="https://example.com/1D_Burgers_Sols_Nu1.0.hdf5",
                local_path=str(sibling_path),
                status=ArtifactStatus.VERIFIED_REMOTE.value,
                metadata={"official_path": "1D/Burgers/Train"},
            )

            self.assertEqual(manager.tools._artifact_materialization_path(artifact), expected_target)


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

    def test_preflight_accepts_required_artifact_referenced_by_canonical_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "repo"
            repo_root.mkdir()
            (repo_root / "train.py").write_text("print('ok')\n", encoding="utf-8")
            env_root = root / "envs" / "pdebench-env"
            (env_root / "bin").mkdir(parents=True, exist_ok=True)
            config = make_config(run_name="preflight-canonical-artifact")
            memory = ResearchMemory(root=root)
            tools = ResearchTools(config=config, memory=memory, repo_root=Path("/root/PDEFlow"))
            tools.run_command = lambda *args, **kwargs: {  # type: ignore[method-assign]
                "returncode": 0,
                "stdout_tail": "",
                "stderr_tail": "",
                "cwd": str(kwargs.get("cwd") or repo_root),
                "log_path": None,
            }
            tools.inspect_python_environment = lambda *args, **kwargs: {  # type: ignore[method-assign]
                "python_available": True,
                "pip_available": True,
                "modules": {"torch": True, "h5py": True},
            }
            tools.run_in_environment = lambda *args, **kwargs: {  # type: ignore[method-assign]
                "returncode": 0,
                "stdout_tail": "1",
                "stderr_tail": "",
                "cwd": str(repo_root),
                "log_path": None,
            }
            artifact = ArtifactRecord(
                artifact_id="1D_Burgers_Sols_Nu0.01.hdf5",
                canonical_id="dataset-burgers-train-nu-0p01",
                artifact_type="dataset",
                title="1D_Burgers_Sols_Nu0.01.hdf5",
                rationale="test",
                local_path=str(root / "1D_Burgers_Sols_Nu0.01.hdf5"),
                status="ready_for_training",
                validation=ArtifactValidationResult(
                    validator="hdf5",
                    status=ArtifactStatus.READY_FOR_TRAINING,
                    exists=True,
                    size_bytes=1024,
                    min_size_bytes=1,
                    size_ok=True,
                    format_valid=True,
                    ready_for_training=True,
                    failure_reasons=[],
                ),
            )
            plan = ExperimentPlan(
                plan_id="plan-canonical",
                title="fallback smoke",
                program_id="prog-1",
                working_directory=str(repo_root),
                launch_command="python train.py",
                required_artifact_ids=["dataset-burgers-train-nu-0p01"],
                log_path=str(root / "logs" / "plan.log"),
            )
            report = tools.preflight_experiment_plan(
                plan,
                [artifact],
                CapabilityMatrix(environment_path=str(env_root)),
            )
            self.assertTrue(report.passed)

    def test_preflight_routes_placeholder_launch_failures_back_to_planning(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_config()
            memory = ResearchMemory(root=root)
            tools = ResearchTools(config=config, memory=memory, repo_root=Path("/root/PDEFlow"))
            report = tools.preflight_experiment_plan(
                ExperimentPlan(
                    plan_id="plan-placeholder",
                    title="Malformed baseline",
                    program_id="prog-1",
                    working_directory=str(root),
                    launch_command="echo 'Launch command truncated in malformed output'",
                    log_path=str(root / "logs" / "placeholder.log"),
                ),
                [],
                CapabilityMatrix(
                    python_available=True,
                    pip_available=True,
                    env_ready=True,
                ),
            )
            self.assertFalse(report.passed)
            self.assertEqual(report.recommended_route, "planning")

    def test_preflight_resolves_config_relative_to_entrypoint_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "repo"
            entrypoint = repo_root / "pdebench" / "models" / "train_models_forward.py"
            config_path = repo_root / "pdebench" / "models" / "config" / "config_Bgs.yaml"
            entrypoint.parent.mkdir(parents=True, exist_ok=True)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            entrypoint.write_text("print('ok')\n", encoding="utf-8")
            config_path.write_text("dummy: true\n", encoding="utf-8")
            env_root = root / "envs" / "pdebench-env"
            (env_root / "bin").mkdir(parents=True, exist_ok=True)
            config = make_config(run_name="preflight-config-resolution")
            memory = ResearchMemory(root=root)
            tools = ResearchTools(config=config, memory=memory, repo_root=Path("/root/PDEFlow"))
            tools.run_command = lambda *args, **kwargs: {  # type: ignore[method-assign]
                "returncode": 0,
                "stdout_tail": "",
                "stderr_tail": "",
                "cwd": str(kwargs.get("cwd") or repo_root),
                "log_path": None,
            }
            tools.inspect_python_environment = lambda *args, **kwargs: {  # type: ignore[method-assign]
                "python_available": True,
                "pip_available": True,
                "modules": {"torch": True, "h5py": True},
            }
            tools.run_in_environment = lambda *args, **kwargs: {  # type: ignore[method-assign]
                "returncode": 0,
                "stdout_tail": "1",
                "stderr_tail": "",
                "cwd": str(repo_root),
                "log_path": None,
            }
            plan = ExperimentPlan(
                plan_id="plan-config",
                title="baseline",
                program_id="prog-1",
                working_directory=str(repo_root),
                launch_command=(
                    f"python {entrypoint.relative_to(repo_root)} +args=config_Bgs.yaml "
                    "++args.model_name=FNO"
                ),
                required_artifact_ids=[],
                log_path=str(root / "logs" / "plan.log"),
                gpu_ids=[6],
            )
            report = tools.preflight_experiment_plan(
                plan,
                [],
                CapabilityMatrix(
                    environment_path=str(env_root),
                    fno_ready=True,
                    torch_runtime_ready=True,
                    h5py_available=True,
                ),
            )
            self.assertTrue(report.passed)


class CapabilityProbeTests(unittest.TestCase):
    def test_capability_probe_prefers_ready_artifact_over_stale_corrupted_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_config(run_name="capability-dedupe")
            memory = ResearchMemory(root=root)
            tools = ResearchTools(config=config, memory=memory, repo_root=Path("/root/PDEFlow"))
            matrix = tools.probe_capability_matrix(
                artifacts=[
                    ArtifactRecord(
                        artifact_id="burgers-old",
                        canonical_id="pdebench-burgers-nu0.01",
                        artifact_type="dataset",
                        title="old bad shard",
                        rationale="stale alias",
                        local_path=str(root / "old.hdf5"),
                        status=ArtifactStatus.CORRUPTED.value,
                        metadata={"exact_target": True},
                        validation=ArtifactValidationResult(
                            validator="hdf5",
                            status=ArtifactStatus.CORRUPTED,
                            exists=True,
                            ready_for_training=False,
                            size_bytes=10,
                            size_ok=False,
                            format_valid=False,
                            failure_reasons=["checksum_mismatch"],
                        ),
                    ),
                    ArtifactRecord(
                        artifact_id="burgers-new",
                        canonical_id="pdebench-burgers-nu0.01",
                        artifact_type="dataset",
                        title="new good shard",
                        rationale="ready alias",
                        local_path=str(root / "new.hdf5"),
                        status=ArtifactStatus.READY_FOR_TRAINING.value,
                        metadata={"exact_target": True},
                        validation=ArtifactValidationResult(
                            validator="hdf5",
                            status=ArtifactStatus.READY_FOR_TRAINING,
                            exists=True,
                            ready_for_training=True,
                            size_bytes=100,
                            size_ok=True,
                            format_valid=True,
                        ),
                    ),
                ],
                repository_paths=[],
                environment_path=None,
            )
            self.assertTrue(matrix.target_dataset_ready)
            self.assertFalse(matrix.target_dataset_blocked)

    def test_capability_probe_marks_verified_remote_exact_target_as_preparing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_config(run_name="capability-verified-remote")
            memory = ResearchMemory(root=root)
            tools = ResearchTools(config=config, memory=memory, repo_root=Path("/root/PDEFlow"))
            matrix = tools.probe_capability_matrix(
                artifacts=[
                    ArtifactRecord(
                        artifact_id="artifact-pdebench-burgers-nu1-train",
                        canonical_id="pdebench:1D/Burgers/Train/1D_Burgers_Sols_Nu1.0.hdf5",
                        artifact_type="dataset",
                        title="1D_Burgers_Sols_Nu1.0.hdf5",
                        rationale="verified remote shard",
                        source_url="https://example.com/1D_Burgers_Sols_Nu1.0.hdf5",
                        local_path=None,
                        status=ArtifactStatus.VERIFIED_REMOTE.value,
                        metadata={"exact_target": True},
                    )
                ],
                repository_paths=[],
                environment_path=None,
            )
            self.assertFalse(matrix.target_dataset_ready)
            self.assertTrue(matrix.target_dataset_preparing)
            self.assertIn(
                "pdebench:1D/Burgers/Train/1D_Burgers_Sols_Nu1.0.hdf5",
                matrix.exact_target_shards_pending,
            )

    def test_capability_probe_requires_gpu_runtime_for_baseline_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_config(run_name="capability-gpu")
            memory = ResearchMemory(root=root)
            tools = ResearchTools(config=config, memory=memory, repo_root=Path("/root/PDEFlow"))
            repo_root = root / "repo"
            (repo_root / "pdebench" / "models").mkdir(parents=True)
            (repo_root / "pdebench" / "models" / "train_models_forward.py").write_text("print('ok')\n", encoding="utf-8")
            env_root = root / "env"
            (env_root / "bin").mkdir(parents=True)

            with patch.object(
                tools,
                "inspect_compute_environment",
                return_value=EnvironmentSnapshot(
                    python_executable="python",
                    python_version="3.10",
                    uv_available=True,
                    available_gpu_ids=[6, 7],
                    selected_gpu_ids=[6, 7],
                ),
            ), patch.object(
                tools,
                "inspect_python_environment",
                return_value={
                    "environment_path": str(env_root),
                    "python_available": True,
                    "pip_available": True,
                    "modules": {"torch": True, "h5py": True, "hydra": True, "deepxde": False, "tensorflow": False},
                },
            ), patch.object(
                tools,
                "_probe_torch_runtime",
                return_value={
                    "torch_import_ok": True,
                    "torch_version": "2.5.1+cu121",
                    "torch_cuda_version": "12.1",
                    "cuda_available": False,
                    "torchvision_available": True,
                    "torchvision_version": "0.20.1+cu121",
                    "error": None,
                },
            ):
                matrix = tools.probe_capability_matrix(
                    artifacts=[
                        ArtifactRecord(
                            artifact_id="dataset-1",
                            canonical_id="dataset-1",
                            artifact_type="dataset",
                            title="ready shard",
                            rationale="test",
                            local_path=str(root / "dataset.hdf5"),
                            status=ArtifactStatus.READY_FOR_TRAINING.value,
                            metadata={"exact_target": True},
                            validation=ArtifactValidationResult(
                                validator="hdf5",
                                status=ArtifactStatus.READY_FOR_TRAINING,
                                exists=True,
                                ready_for_training=True,
                                size_bytes=100,
                                size_ok=True,
                                format_valid=True,
                            ),
                        )
                    ],
                    repository_paths=[str(repo_root)],
                    environment_path=str(env_root),
                )

            self.assertTrue(matrix.target_dataset_ready)
            self.assertTrue(matrix.torch_runtime_ready)
            self.assertTrue(matrix.gpu_runtime_required)
            self.assertFalse(matrix.gpu_runtime_ready)
            self.assertFalse(matrix.baseline_launch_ready)
            self.assertTrue(matrix.environment_repair_needed)


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


class EnvironmentBootstrapTests(unittest.TestCase):
    def test_environment_bootstrap_repairs_torch_for_cuda(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            project_root = root / "repo"
            project_root.mkdir()
            (project_root / "pyproject.toml").write_text(
                '[project]\nname = "pdebench"\ndependencies = ["torch", "numpy"]\n',
                encoding="utf-8",
            )
            config = make_config(run_name="env-repair")
            memory = ResearchMemory(root=root)
            tools = ResearchTools(config=config, memory=memory, repo_root=Path("/root/PDEFlow"))

            state = {"torch_probe_count": 0}

            def fake_run_command(command, cwd=None, **kwargs):  # noqa: ANN001
                command = str(command)
                if command.startswith("uv venv"):
                    return {"returncode": 0, "stdout_tail": "", "stderr_tail": "", "cwd": str(cwd), "log_path": None}
                if "uv sync" in command:
                    return {"returncode": 1, "stdout_tail": "", "stderr_tail": "sync failed", "cwd": str(cwd), "log_path": None}
                if "--index-url https://download.pytorch.org/whl/cu124" in command:
                    return {"returncode": 0, "stdout_tail": "", "stderr_tail": "", "cwd": str(cwd), "log_path": None}
                if "uv pip install" in command:
                    return {"returncode": 0, "stdout_tail": "", "stderr_tail": "", "cwd": str(cwd), "log_path": None}
                if command.endswith("-m pip --version"):
                    return {"returncode": 0, "stdout_tail": "pip 25.0", "stderr_tail": "", "cwd": str(cwd), "log_path": None}
                if "import sys; print(sys.executable)" in command:
                    return {"returncode": 0, "stdout_tail": f"{project_root}/.venv/bin/python\n3.10.0", "stderr_tail": "", "cwd": str(cwd), "log_path": None}
                if "import json, torch" in command:
                    state["torch_probe_count"] += 1
                    if state["torch_probe_count"] >= 2:
                        return {"returncode": 0, "stdout_tail": '{"torch":"2.6.0","cuda":"12.4","available":true}', "stderr_tail": "", "cwd": str(cwd), "log_path": None}
                    return {"returncode": 0, "stdout_tail": '{"torch":"2.6.0","cuda":null,"available":false}', "stderr_tail": "", "cwd": str(cwd), "log_path": None}
                return {"returncode": 0, "stdout_tail": "", "stderr_tail": "", "cwd": str(cwd), "log_path": None}

            tools.run_command = fake_run_command  # type: ignore[method-assign]
            tools.inspect_compute_environment = lambda: EnvironmentSnapshot(  # type: ignore[method-assign]
                python_executable="python",
                python_version="3.10",
                uv_available=True,
                available_gpu_ids=[6, 7],
                selected_gpu_ids=[6, 7],
            )

            payload = tools.ensure_python_environment(str(project_root))
            self.assertEqual(payload["status"], "ready")
            self.assertIn("gpu_torch_repair", str(payload["strategy"]))

    def test_environment_bootstrap_reuses_ready_sibling_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shared_runs_root = root / "runs"
            prior_env = shared_runs_root / "prior-run" / "envs" / "repo-env"
            (prior_env / "bin").mkdir(parents=True, exist_ok=True)
            (prior_env / "bin" / "python").write_text("", encoding="utf-8")
            project_root = shared_runs_root / "current-run" / "repo"
            project_root.mkdir(parents=True)
            (project_root / "pyproject.toml").write_text(
                '[project]\nname = "repo"\ndependencies = ["torch", "h5py"]\n',
                encoding="utf-8",
            )

            config = make_config(run_name="current-run")
            config.execution.work_directory = str(shared_runs_root / "{run_name}")
            memory = ResearchMemory(root=shared_runs_root / config.run_name)
            tools = ResearchTools(config=config, memory=memory, repo_root=Path("/root/PDEFlow"))

            with patch.object(tools, "_probe_reusable_environment", return_value=True):
                payload = tools.ensure_python_environment(str(project_root))

            expected_target = shared_runs_root / "current-run" / "envs" / "repo-env"
            self.assertEqual(payload["status"], "ready")
            self.assertEqual(payload["strategy"], "reused_sibling_env")
            self.assertTrue(expected_target.is_symlink())
            self.assertEqual(expected_target.resolve(), prior_env.resolve())

    def test_minimal_environment_bootstrap_skips_gpu_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            project_root = root / "repo"
            project_root.mkdir()
            (project_root / "pyproject.toml").write_text(
                '[project]\nname = "pdebench"\ndependencies = ["torch", "numpy", "h5py"]\n',
                encoding="utf-8",
            )
            config = make_config(run_name="env-minimal")
            memory = ResearchMemory(root=root)
            tools = ResearchTools(config=config, memory=memory, repo_root=Path("/root/PDEFlow"))

            seen_commands: list[str] = []

            def fake_run_command(command, cwd=None, **kwargs):  # noqa: ANN001
                command = str(command)
                seen_commands.append(command)
                if command.startswith("uv venv"):
                    return {"returncode": 0, "stdout_tail": "", "stderr_tail": "", "cwd": str(cwd), "log_path": None}
                if "uv sync" in command:
                    return {"returncode": 1, "stdout_tail": "", "stderr_tail": "sync failed", "cwd": str(cwd), "log_path": None}
                if "uv pip install" in command:
                    return {"returncode": 0, "stdout_tail": "", "stderr_tail": "", "cwd": str(cwd), "log_path": None}
                if command.endswith("-m pip --version"):
                    return {"returncode": 0, "stdout_tail": "pip 25.0", "stderr_tail": "", "cwd": str(cwd), "log_path": None}
                if "import sys; print(sys.executable)" in command:
                    return {"returncode": 0, "stdout_tail": f"{project_root}/.venv/bin/python\n3.10.0", "stderr_tail": "", "cwd": str(cwd), "log_path": None}
                if "import json, torch" in command:
                    self.fail("minimal bootstrap should not probe or repair GPU torch runtime")
                return {"returncode": 0, "stdout_tail": "", "stderr_tail": "", "cwd": str(cwd), "log_path": None}

            tools.run_command = fake_run_command  # type: ignore[method-assign]
            payload = tools.ensure_python_environment(
                str(project_root),
                dependency_strategy="minimal",
                require_gpu_runtime=False,
            )
            self.assertEqual(payload["status"], "ready")
            self.assertNotIn("gpu_torch_repair", str(payload["strategy"]))
            self.assertTrue(any("uv pip install" in command for command in seen_commands))


if __name__ == "__main__":
    unittest.main()
