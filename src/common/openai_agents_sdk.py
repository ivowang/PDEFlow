from __future__ import annotations

import importlib.machinery
import importlib.util
from pathlib import Path
import sys
from types import ModuleType


_SDK_MODULE: ModuleType | None = None


def _repo_src_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _is_local_src_path(path_entry: str) -> bool:
    if not path_entry:
        return True
    try:
        return Path(path_entry).resolve() == _repo_src_root()
    except OSError:
        return False


def load_openai_agents_sdk() -> ModuleType:
    global _SDK_MODULE
    if _SDK_MODULE is not None:
        return _SDK_MODULE

    search_paths = [entry for entry in sys.path if not _is_local_src_path(entry)]
    spec = importlib.machinery.PathFinder.find_spec("agents", search_paths)
    if spec is None or spec.loader is None:
        raise ImportError("Could not locate the installed OpenAI Agents SDK module `agents`.")

    local_agents_module = sys.modules.get("agents")
    sdk_module = importlib.util.module_from_spec(spec)
    sys.modules["agents"] = sdk_module
    try:
        spec.loader.exec_module(sdk_module)
    finally:
        if local_agents_module is not None:
            sys.modules["agents"] = local_agents_module
        else:
            sys.modules.pop("agents", None)
    _SDK_MODULE = sdk_module
    return sdk_module
