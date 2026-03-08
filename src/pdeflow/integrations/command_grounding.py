from __future__ import annotations

from pathlib import Path
import re
import shlex

from ..state import ArtifactRecord, ExperimentPlan

PATH_KEY_RE = re.compile(r"(?:^|\.)([A-Za-z0-9_]*(?:path|root|folder|dir))$")
FILE_KEY_RE = re.compile(r"(?:^|\.)([A-Za-z0-9_]*(?:file|filename|dataset))$")
CONFIG_ASSIGN_RE = re.compile(r"^\+{1,2}([A-Za-z0-9_.-]+)=(.+)$")
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
        normalized.append(replacement)
    return normalized


def _artifact_is_complete(artifact: ArtifactRecord) -> bool:
    if not artifact.local_path:
        return False
    path = Path(artifact.local_path)
    if not path.exists() or not path.is_file():
        return False
    status = artifact.status.lower()
    if any(marker in status for marker in ("partial", "corrupt", "failed", "blocked")):
        return False
    metadata = artifact.metadata or {}
    expected_size = metadata.get("expected_size_bytes")
    if expected_size is not None:
        try:
            if path.stat().st_size < int(expected_size):
                return False
        except (TypeError, ValueError):
            return False
    known_size = metadata.get("size_bytes")
    if known_size is not None:
        try:
            if path.stat().st_size < int(known_size):
                return False
        except (TypeError, ValueError):
            return False
    return True


def _resolve_dataset_artifact(assignments: dict[str, str], artifacts: list[ArtifactRecord]) -> Path | None:
    candidate_names = [
        Path(value.strip("'\"")).name
        for key, value in assignments.items()
        if _is_file_key(key) and _is_data_filename(value)
    ]
    for artifact in artifacts:
        if not _artifact_is_complete(artifact):
            continue
        if artifact.local_path and Path(artifact.local_path).name in candidate_names:
            return Path(artifact.local_path)
    return None


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


def ground_experiment_plan(
    plan: ExperimentPlan,
    artifacts: list[ArtifactRecord],
) -> tuple[ExperimentPlan, list[str]]:
    tokens = _split_shell(plan.launch_command)
    if not tokens:
        return plan, [f"{plan.plan_id}: unable to parse launch command for grounding."]
    assignments = _command_assignments(tokens)
    dataset_path = _resolve_dataset_artifact(assignments, artifacts)
    if dataset_path is None:
        return plan, []

    working_directory = Path(plan.working_directory)
    script_path = _resolve_script_path(tokens, working_directory)
    if script_path is None:
        return plan, []

    config_roots = _discover_config_roots(working_directory)
    path_keys = _discover_path_keys(script_path, config_roots)
    if not path_keys:
        return plan, []

    grounded_tokens = list(tokens)
    grounded_messages: list[str] = []
    dataset_root = str(dataset_path.parent)
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

    if not grounded_messages:
        return plan, []

    grounded_setup = _replace_dataset_prechecks(plan.setup_commands, dataset_path)
    grounded_launch = shlex.join(grounded_tokens)
    notes = list(plan.notes)
    note = (
        "Grounded command paths from verified local artifacts after inspecting repo entrypoint/config semantics: "
        f"dataset={dataset_path.name}, root={dataset_root}."
    )
    if note not in notes:
        notes.append(note)
    return (
        plan.model_copy(
            update={
                "launch_command": grounded_launch,
                "setup_commands": grounded_setup,
                "notes": notes,
            }
        ),
        grounded_messages,
    )
