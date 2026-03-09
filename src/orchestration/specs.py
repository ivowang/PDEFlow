from __future__ import annotations

from dataclasses import dataclass

from state import ResearchPhase


@dataclass(frozen=True)
class PhaseSpec:
    phase: ResearchPhase
    agent_key: str
    outputs: tuple[str, ...]


@dataclass(frozen=True)
class CycleRoute:
    route_id: str
    phases: tuple[PhaseSpec, ...]
    reason: str
    focus: tuple[str, ...] = ()
