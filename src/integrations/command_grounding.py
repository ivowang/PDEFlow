from __future__ import annotations

from pathlib import Path
import re
import shlex

from common import infer_dataset_semantic_spec
from state import ArtifactRecord, ExperimentPlan

PATH_KEY_RE = re.compile(r"(?:^|\.)([A-Za-z0-9_]*(?:path|root|folder|dir))$")
FILE_KEY_RE = re.compile(r"(?:^|\.)([A-Za-z0-9_]*(?:file|filename|dataset))$")
CONFIG_ASSIGN_RE = re.compile(r"^\+{1,2}([A-Za-z0-9_.-]+)=(.+)$")
BARE_OVERRIDE_RE = re.compile(r"^\+{1,2}([A-Za-z0-9_.-]+)$")
SCRIPT_EXTENSIONS = {".py"}
DATA_EXTENSIONS = {".h5", ".hdf5", ".npz", ".npy", ".pt", ".pth", ".ckpt", ".csv", ".json"}


def _split_shell(command: str) -> list[str] | None:
    try:
        return shlex.split(command)
    except ValueError:
        return None


def _command_assignments(tokens: list[str]) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for token in tokens:
        match = CONFIG_ASSIGN_RE.match(token)
        if match:
            assignments[match.group(1)] = match.group(2)
    return assignments


def _command_bare_overrides(tokens: list[str]) -> list[str]:
    bare: list[str] = []
    for token in tokens:
        match = BARE_OVERRIDE_RE.match(token)
        if match:
            bare.append(match.group(1))
    return bare


def _is_path_key(key: str) -> bool:
    return bool(PATH_KEY_RE.search(key))


def _is_file_key(key: str) -> bool:
    return bool(FILE_KEY_RE.search(key))


def _is_data_filename(value: str) -> bool:
    candidate = Path(value.strip("'\""))
    return candidate.suffix.lower() in DATA_EXTENSIONS and candidate.name == candidate.as_posix().split("/")[-1]


def _set_assignment(tokens: list[str], key: str, value: str) -> list[str]:
    replacement = f"++{key}={shlex.quote(value)}"
    updated = False
    normalized: list[str] = []
    for token in tokens:
        match = CONFIG_ASSIGN_RE.match(token)
        if match and match.group(1) == key:
            normalized.append(replacement)
            updated = True
        else:
            normalized.append(token)
    if not updated:
        insert_at = len(normalized)
        for index, token in enumerate(normalized):
            if token in {">", ">>", "<", "<<", "|", "||", "&&", ";", "2>&1"}:
                insert_at = index
                break
        normalized.insert(insert_at, replacement)
    return normalized


def _artifact_is_complete(artifact: ArtifactRecord) -> bool:
    return artifact.status == "ready_for_training"


def _infer_run_root(plan: ExperimentPlan) -> Path | None:
    try:
        log_path = Path(plan.log_path).resolve()
    except OSError:
        return None
    for parent in log_path.parents:
        if parent.name == "experiments":
            return parent.parent.resolve()
    return None


def _preferred_ready_artifact(
    plan: ExperimentPlan,
    matching_artifacts: list[ArtifactRecord],
) -> ArtifactRecord | None:
    ready = [artifact for artifact in matching_artifacts if _artifact_is_complete(artifact) and artifact.local_path]
    if not ready:
        return None
    run_root = _infer_run_root(plan)
    if run_root is None:
        return ready[0]
    ready.sort(
        key=lambda artifact: (
            0 if str(Path(artifact.local_path).resolve()).startswith(str(run_root)) else 1,
            len(str(artifact.local_path)),
        )
    )
    return ready[0]


def _artifact_spec(artifact: ArtifactRecord) -> dict[str, str]:
    if artifact.semantic_spec is not None:
        payload = artifact.semantic_spec.model_dump(exclude_none=True)
        if payload:
            return {key: str(value) for key, value in payload.items()}
    return infer_dataset_semantic_spec(
        artifact.artifact_id,
        artifact.title or "",
        artifact.local_path or "",
        str(artifact.metadata.get("official_path") or ""),
        str(artifact.metadata.get("expected_filename") or ""),
    )


def _plan_target_spec(plan: ExperimentPlan, assignments: dict[str, str]) -> dict[str, str]:
    spec = infer_dataset_semantic_spec(plan.title, plan.plan_id, *plan.required_artifact_ids, *plan.notes)
    if not spec.get("split"):
        training_value = next(
            (
                value.strip("'\"").lower()
                for key, value in assignments.items()
                if key.endswith("if_training")
            ),
            "",
        )
        if training_value in {"true", "1", "yes"}:
            spec["split"] = "train"
    return spec


def _spec_matches(plan_spec: dict[str, str], artifact: ArtifactRecord) -> bool:
    artifact_spec = _artifact_spec(artifact)
    if not plan_spec:
        return False
    equation = plan_spec.get("equation")
    if equation and artifact_spec.get("equation") != equation:
        return False
    for field in ("split", "nu", "rho"):
        expected = plan_spec.get(field)
        actual = artifact_spec.get(field)
        if expected and actual and expected != actual:
            return False
    return True


def _matching_dataset_artifacts(
    plan: ExperimentPlan,
    assignments: dict[str, str],
    artifacts: list[ArtifactRecord],
) -> list[ArtifactRecord]:
    required_ids = set(plan.required_artifact_ids)
    candidate_names = [
        Path(value.strip("'\"")).name
        for key, value in assignments.items()
        if _is_file_key(key) and _is_data_filename(value)
    ]
    direct_matches = [
        artifact
        for artifact in artifacts
        if artifact.local_path
        and (
            Path(artifact.local_path).name in candidate_names
            or artifact.artifact_id in required_ids
            or (artifact.canonical_id or "") in required_ids
        )
    ]
    if direct_matches:
        return direct_matches
    target_spec = _plan_target_spec(plan, assignments)
    return [
        artifact
        for artifact in artifacts
        if artifact.artifact_type == "dataset" and artifact.local_path and _spec_matches(target_spec, artifact)
    ]


def _resolve_script_path(tokens: list[str], working_directory: Path) -> Path | None:
    for token in tokens:
        candidate = Path(token)
        if candidate.suffix.lower() in SCRIPT_EXTENSIONS:
            path = candidate if candidate.is_absolute() else (working_directory / candidate)
            if path.exists():
                return path.resolve()
    return None


def _path_like_default(value: str) -> bool:
    normalized = value.strip().strip("'\"")
    if not normalized:
        return False
    if normalized.startswith("/path/to"):
        return True
    if normalized in {".", "..", "data", "./data", "../data"}:
        return True
    if normalized.startswith("../") or normalized.startswith("./"):
        return True
    if "/" in normalized or normalized.endswith(("\\", "/")):
        return True
    return False


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _discover_config_roots(working_directory: Path) -> list[Path]:
    roots: list[Path] = []
    for candidate in (working_directory / "config", working_directory.parent / "config", working_directory / "configs"):
        if candidate.exists() and candidate.is_dir():
            roots.append(candidate.resolve())
    return roots


def _discover_path_keys(script_path: Path, config_roots: list[Path]) -> dict[str, str]:
    path_defaults: dict[str, str] = {}
    script_text = _read_text(script_path)
    for match in re.finditer(r"cfg(?:\.[A-Za-z0-9_]+)*\.([A-Za-z0-9_]*(?:path|root|folder|dir))\b", script_text):
        path_defaults.setdefault(match.group(1), "")
    for config_root in config_roots:
        for yaml_path in config_root.rglob("*.yaml"):
            text = _read_text(yaml_path)
            for match in re.finditer(r"^\s*([A-Za-z0-9_]*(?:path|root|folder|dir))\s*:\s*([^\n#]+)", text, flags=re.MULTILINE):
                key = match.group(1).strip()
                value = match.group(2).strip()
                if _path_like_default(value):
                    if not path_defaults.get(key):
                        path_defaults[key] = value
    return path_defaults


def _discover_file_keys(script_path: Path, config_roots: list[Path]) -> dict[str, str]:
    file_defaults: dict[str, str] = {}
    script_text = _read_text(script_path)
    for match in re.finditer(r"cfg(?:\.[A-Za-z0-9_]+)*\.([A-Za-z0-9_]*(?:file|filename|dataset))\b", script_text):
        file_defaults.setdefault(match.group(1), "")
    for config_root in config_roots:
        for yaml_path in config_root.rglob("*.yaml"):
            text = _read_text(yaml_path)
            for match in re.finditer(r"^\s*([A-Za-z0-9_]*(?:file|filename|dataset))\s*:\s*([^\n#]+)", text, flags=re.MULTILINE):
                key = match.group(1).strip()
                value = match.group(2).strip().strip("'\"")
                if value:
                    file_defaults.setdefault(key, value)
    return file_defaults


def _read_yaml_scalars(path: Path) -> dict[str, str]:
    text = _read_text(path)
    defaults: dict[str, str] = {}
    for match in re.finditer(r"^\s*([A-Za-z0-9_]+)\s*:\s*([^\n#]+)", text, flags=re.MULTILINE):
        defaults[match.group(1).strip()] = match.group(2).strip().strip("'\"")
    return defaults


def _selected_config_defaults(assignments: dict[str, str], config_roots: list[Path]) -> dict[str, str]:
    defaults: dict[str, str] = {}
    for key, value in assignments.items():
        config_name = value.strip("'\"")
        if not config_name.endswith(".yaml"):
            continue
        namespace = key.split(".")[-1]
        for config_root in config_roots:
            candidates = [config_root / namespace / config_name, config_root / config_name]
            selected = next((candidate for candidate in candidates if candidate.exists()), None)
            if selected is None:
                continue
            for local_key, local_value in _read_yaml_scalars(selected).items():
                defaults.setdefault(f"{namespace}.{local_key}", local_value)
            break
    return defaults


def _sanitize_bare_overrides(
    tokens: list[str],
    config_defaults: dict[str, str],
) -> tuple[list[str], list[str]]:
    sanitized: list[str] = []
    messages: list[str] = []
    for token in tokens:
        match = BARE_OVERRIDE_RE.match(token)
        if not match:
            sanitized.append(token)
            continue
        key = match.group(1)
        default_key = next(
            (name for name in config_defaults if name == key or name.endswith(f".{key.split('.')[-1]}")),
            None,
        )
        if default_key is not None and config_defaults[default_key]:
            sanitized.append(f"++{key}={shlex.quote(config_defaults[default_key])}")
            messages.append(f"repaired bare override {key} using config default {config_defaults[default_key]}.")
        else:
            messages.append(f"dropped malformed bare override {key} because no repair value was available.")
    return sanitized, messages


def _replace_dataset_prechecks(setup_commands: list[str], dataset_path: Path) -> list[str]:
    normalized: list[str] = []
    quoted_path = shlex.quote(str(dataset_path))
    replaced = False
    for command in setup_commands:
        if "test -f" in command:
            updated = re.sub(r"test -f\s+(\S+)", f"test -f {quoted_path}", command, count=1)
            normalized.append(updated)
            replaced = True
        else:
            normalized.append(command)
    if not replaced:
        normalized.insert(0, f"test -f {quoted_path}")
    return normalized


def _normalize_shell_command(command: str) -> str:
    normalized = command
    replacements = {
        "'2>&1'": "2>&1",
        '"2>&1"': "2>&1",
        "'>'": ">",
        '">"': ">",
        "'>>'": ">>",
        '">>"': ">>",
        "'<'": "<",
        '"<"': "<",
        "'<<'": "<<",
        '"<<"': "<<",
        "'|'": "|",
        '"|"': "|",
        "'&&'": "&&",
        '"&&"': "&&",
        "'||'": "||",
        '"||"': "||",
        "';'": ";",
        '";"': ";",
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    normalized = re.sub(r"\s+\|\s+tee\b", " | tee", normalized)
    return normalized


def _normalize_managed_python_command(command: str) -> str:
    pattern = re.compile(r"(uv\s+run\s+--python\s+)(\S+)(\s+python\b)")
    return pattern.sub(lambda match: match.group(2), command)


def _normalize_plan_commands(plan: ExperimentPlan) -> ExperimentPlan:
    return plan.model_copy(
        update={
            "setup_commands": [
                _normalize_managed_python_command(_normalize_shell_command(command))
                for command in plan.setup_commands
            ],
            "launch_command": _normalize_managed_python_command(
                _normalize_shell_command(plan.launch_command)
            ),
        }
    )


def _infer_local_dataset_root(plan: ExperimentPlan, dataset_filename: str) -> Path | None:
    for command in plan.setup_commands:
        stripped = command.strip()
        if stripped.startswith("test -f "):
            try:
                checked_path = Path(shlex.split(stripped)[2])
            except (IndexError, ValueError):
                continue
            if checked_path.name == dataset_filename:
                return checked_path.parent
    for command in plan.setup_commands:
        stripped = command.strip()
        if stripped.startswith("ln -sfn "):
            try:
                tokens = shlex.split(stripped)
            except ValueError:
                continue
            if len(tokens) >= 4:
                destination = Path(tokens[3])
                if destination.name in {"data", "data_train", "data_root", "dataset", "datasets"}:
                    return destination
    return None


def _choose_exemplar_args(
    script_path: Path,
    ready_artifact: ArtifactRecord,
    assignments: dict[str, str],
) -> list[str] | None:
    model_name = next(
        (
            value.strip("'\"").lower()
            for key, value in assignments.items()
            if key.endswith("model_name")
        ),
        "",
    )
    equation = (_artifact_spec(ready_artifact).get("equation") or "").lower()
    artifact_filename = Path(ready_artifact.local_path or ready_artifact.title or "").name.lower()
    training_requested = next(
        (
            value.strip("'\"").lower() in {"true", "1", "yes"}
            for key, value in assignments.items()
            if key.endswith("if_training")
        ),
        True,
    )
    candidates: list[tuple[int, list[str]]] = []
    for shell_path in sorted(script_path.parent.glob("run*.sh")):
        for raw_line in _read_text(shell_path).splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or script_path.name not in line:
                continue
            lowered = line.lower()
            try:
                tokens = shlex.split(line)
            except ValueError:
                continue
            try:
                script_index = next(index for index, token in enumerate(tokens) if Path(token).name == script_path.name)
            except StopIteration:
                continue
            score = 0
            if artifact_filename and artifact_filename in lowered:
                score += 100
            if equation == "reactiondiffusion" and "reacdiff" in lowered:
                score += 40
            if equation == "burgers" and "burgers" in lowered:
                score += 40
            if model_name and f"++args.model_name='{model_name}'" in lowered:
                score += 20
            if training_requested and "if_training=false" not in lowered.replace(" ", ""):
                score += 10
            if "+args=" in line and "++args.filename" in line:
                score += 5
            if score > 0:
                candidates.append((score, tokens[script_index + 1 :]))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def ground_experiment_plan(
    plan: ExperimentPlan,
    artifacts: list[ArtifactRecord],
) -> tuple[ExperimentPlan, list[str]]:
    normalized_plan = _normalize_plan_commands(plan)
    tokens = _split_shell(normalized_plan.launch_command)
    if not tokens:
        return normalized_plan, [f"{plan.plan_id}: unable to parse launch command for grounding."]
    assignments = _command_assignments(tokens)
    matching_artifacts = _matching_dataset_artifacts(plan, assignments, artifacts)
    ready_artifact = _preferred_ready_artifact(normalized_plan, matching_artifacts)
    if ready_artifact is None:
        if matching_artifacts:
            blocked_ids = [item.artifact_id for item in matching_artifacts]
            notes = list(normalized_plan.notes)
            note = (
                "Plan depends on dataset artifacts that are not ready_for_training: "
                + ", ".join(blocked_ids)
            )
            if note not in notes:
                notes.append(note)
            blocked_plan = normalized_plan.model_copy(
                update={
                    "status": "blocked",
                    "required_artifact_ids": sorted(set([*normalized_plan.required_artifact_ids, *blocked_ids])),
                    "preflight_status": "blocked_artifact_dependency",
                    "notes": notes,
                }
            )
            return blocked_plan, [
                f"{normalized_plan.plan_id}: blocked because required dataset artifacts are not ready_for_training: {', '.join(blocked_ids)}."
            ]
        return normalized_plan, []
    dataset_path = Path(ready_artifact.local_path)

    working_directory = Path(normalized_plan.working_directory)
    script_path = _resolve_script_path(tokens, working_directory)
    if script_path is None:
        return (
            normalized_plan.model_copy(
                update={
                    "required_artifact_ids": sorted(set([*normalized_plan.required_artifact_ids, ready_artifact.artifact_id])),
                }
            ),
            [],
        )

    config_roots = _discover_config_roots(working_directory)
    config_defaults = _selected_config_defaults(assignments, config_roots)
    grounded_tokens = list(tokens)
    grounded_messages: list[str] = []
    exemplar_args = _choose_exemplar_args(script_path, ready_artifact, assignments)
    if exemplar_args and (
        _command_bare_overrides(tokens)
        or not matching_artifacts
        or any("truncated" in note.lower() or "malformed" in note.lower() for note in normalized_plan.notes)
    ):
        try:
            script_index = next(index for index, token in enumerate(tokens) if Path(token).suffix.lower() in SCRIPT_EXTENSIONS)
        except StopIteration:
            script_index = -1
        if script_index >= 0:
            grounded_tokens = tokens[: script_index + 1] + exemplar_args
            grounded_messages.append(
                f"{plan.plan_id}: repaired launch args from repository exemplar after detecting malformed or incomplete plan output."
            )
            assignments = _command_assignments(grounded_tokens)
            config_defaults = _selected_config_defaults(assignments, config_roots)
    grounded_tokens, repair_messages = _sanitize_bare_overrides(grounded_tokens, config_defaults)
    grounded_messages.extend(f"{plan.plan_id}: {message}" for message in repair_messages)
    assignments = _command_assignments(grounded_tokens)
    path_keys = _discover_path_keys(script_path, config_roots)
    file_keys = _discover_file_keys(script_path, config_roots)
    if not path_keys and not file_keys and not grounded_messages:
        return normalized_plan.model_copy(
            update={
                "required_artifact_ids": sorted(set([*normalized_plan.required_artifact_ids, ready_artifact.artifact_id])),
            }
        ), []
    local_dataset_root = _infer_local_dataset_root(normalized_plan, dataset_path.name)
    dataset_root = str((local_dataset_root or dataset_path.parent))
    for key, default_value in path_keys.items():
        matching_assignment_key = next((name for name in assignments if name.endswith(f".{key}") or name == key), None)
        if matching_assignment_key:
            current_value = assignments[matching_assignment_key].strip("'\"")
            if _path_like_default(current_value):
                grounded_tokens = _set_assignment(grounded_tokens, matching_assignment_key, dataset_root)
                grounded_messages.append(
                    f"{plan.plan_id}: replaced placeholder {matching_assignment_key} with {dataset_root}."
                )
            continue
        if _path_like_default(default_value):
            namespace = "args" if any(name.startswith("args.") for name in assignments) else ""
            assignment_key = f"{namespace}.{key}" if namespace else key
            grounded_tokens = _set_assignment(grounded_tokens, assignment_key, dataset_root)
            grounded_messages.append(
                f"{plan.plan_id}: injected {assignment_key}={dataset_root} from verified artifact {dataset_path.name}."
            )
    assignments = _command_assignments(grounded_tokens)
    for key in file_keys:
        matching_assignment_key = next((name for name in assignments if name.endswith(f".{key}") or name == key), None)
        if matching_assignment_key:
            current_value = Path(assignments[matching_assignment_key].strip("'\"")).name
            if current_value != dataset_path.name:
                grounded_tokens = _set_assignment(grounded_tokens, matching_assignment_key, dataset_path.name)
                grounded_messages.append(
                    f"{plan.plan_id}: redirected {matching_assignment_key} from {current_value} to verified artifact {dataset_path.name}."
                )
            continue
        namespace = "args" if any(name.startswith("args.") for name in assignments) else ""
        assignment_key = f"{namespace}.{key}" if namespace else key
        grounded_tokens = _set_assignment(grounded_tokens, assignment_key, dataset_path.name)
        grounded_messages.append(
            f"{plan.plan_id}: injected {assignment_key}={dataset_path.name} from verified artifact."
        )

    if not grounded_messages:
        return (
            normalized_plan.model_copy(
                update={
                    "required_artifact_ids": sorted(set([*normalized_plan.required_artifact_ids, ready_artifact.artifact_id])),
                }
            ),
            [],
        )

    grounded_setup = [
        _normalize_managed_python_command(_normalize_shell_command(command))
        for command in _replace_dataset_prechecks(normalized_plan.setup_commands, dataset_path)
    ]
    grounded_launch = _normalize_managed_python_command(_normalize_shell_command(shlex.join(grounded_tokens)))
    notes = list(normalized_plan.notes)
    note = (
        "Grounded command paths from verified local artifacts after inspecting repo entrypoint/config semantics: "
        f"dataset={dataset_path.name}, root={dataset_root}."
    )
    if note not in notes:
        notes.append(note)
    return (
        normalized_plan.model_copy(
            update={
                "launch_command": grounded_launch,
                "setup_commands": grounded_setup,
                "required_artifact_ids": sorted(set([*normalized_plan.required_artifact_ids, ready_artifact.artifact_id])),
                "notes": notes,
            }
        ),
        grounded_messages,
    )
