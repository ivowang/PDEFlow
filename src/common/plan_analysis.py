from __future__ import annotations

import re
from typing import Any


_TEE_OUTPUT_RE = re.compile(r"\|\s*tee\s+([^\s]+)")


def plan_signal_text(plan: Any) -> str:
    notes = getattr(plan, "notes", []) or []
    parts = [
        getattr(plan, "title", None),
        getattr(plan, "program_id", None),
        getattr(plan, "repo_id", None),
        getattr(plan, "launch_command", None),
        " ".join(str(item) for item in notes),
    ]
    return " ".join(part for part in parts if part).lower()


def plan_requires_pinn(plan: Any) -> bool:
    text = plan_signal_text(plan)
    return any(token in text for token in ("pinn", "deepxde"))


def plan_requires_fno(plan: Any) -> bool:
    text = plan_signal_text(plan)
    return any(token in text for token in ("fno", "fourier"))


def plan_prefers_fallback(plan: Any) -> bool:
    text = plan_signal_text(plan)
    return "evidence_generating_fallback" in text or "fallback smoke" in text


def plan_is_baseline(plan: Any) -> bool:
    return "baseline" in plan_signal_text(plan)


def extract_plan_tee_outputs(plan: Any) -> list[str]:
    command = getattr(plan, "launch_command", "") or ""
    return [match.group(1) for match in _TEE_OUTPUT_RE.finditer(command)]
