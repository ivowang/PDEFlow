from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, TypeVar

from pydantic import BaseModel


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def slugify(value: str) -> str:
    lowered = value.strip().lower()
    lowered = re.sub(r"[^a-z0-9]+", "-", lowered)
    return lowered.strip("-") or "item"


def short_hash(*parts: str) -> str:
    digest = hashlib.sha1("::".join(parts).encode("utf-8")).hexdigest()
    return digest[:10]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_plain_data(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return to_plain_data(value.model_dump(mode="python"))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): to_plain_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_plain_data(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(to_plain_data(payload), indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_plain_data(payload)) + "\n")


T = TypeVar("T")


def upsert_by_attr(existing: Iterable[T], incoming: Iterable[T], attr_name: str) -> list[T]:
    merged: OrderedDict[str, T] = OrderedDict()
    for item in existing:
        merged[str(getattr(item, attr_name))] = item
    for item in incoming:
        merged[str(getattr(item, attr_name))] = item
    return list(merged.values())


def dedupe_strings(items: Iterable[str]) -> list[str]:
    seen: OrderedDict[str, None] = OrderedDict()
    for item in items:
        normalized = item.strip()
        if normalized and normalized not in seen:
            seen[normalized] = None
    return list(seen.keys())
