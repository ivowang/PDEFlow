from __future__ import annotations

from .specs import CycleRoute, PhaseSpec
from state import ResearchPhase, ResearchState


class ManagerRoutingMixin:
    def _contains_any(self, haystack: str, needles: tuple[str, ...]) -> bool:
        return any(needle in haystack for needle in needles)

    def _recent_failure_text(self, state: ResearchState) -> str:
        parts: list[str] = []
        for record in state.experiment_records[-8:]:
            if record.status == "completed":
                continue
            parts.append(record.status)
            parts.extend(record.failure_modes)
            parts.extend(record.observations)
            parts.append(record.command)
            parts.append(record.working_directory)
        parts.extend(state.failure_summaries[-12:])
        parts.extend(state.next_actions[-12:])
        latest_reflection = state.reflections[-1] if state.reflections else None
        if latest_reflection:
            parts.append(latest_reflection.verdict)
            parts.extend(latest_reflection.evidence)
            parts.extend(latest_reflection.next_actions)
            if latest_reflection.stop_reason:
                parts.append(latest_reflection.stop_reason)
        return "\n".join(part for part in parts if part).lower()

    def _latest_program_requires_coding(self, state: ResearchState) -> bool:
        if not state.program_candidates:
            return False
        latest = state.program_candidates[-1]
        if latest.status.startswith("validated") or latest.status in {"evaluated", "completed"}:
            return False
        return bool(state.method_designs)

    def _select_cycle_route(self, state: ResearchState) -> CycleRoute:
        failure_text = self._recent_failure_text(state)
        missing_markers = (
            "missing",
            "absent",
            "not found",
            "does not exist",
            "failed before launch",
            "setup_failed",
            "blocked",
        )
        acquisition_markers = (
            "dataset",
            "hdf5",
            "checkpoint",
            "weights",
            "download",
            "repository",
            "clone",
            "artifact",
            "no module named",
            "ensurepip",
            "pip",
            "venv",
            "dependency",
            "environment",
        )
        if failure_text and self._contains_any(failure_text, missing_markers) and self._contains_any(
            failure_text, acquisition_markers
        ):
            return CycleRoute(
                phases=tuple(self.recovery_phases),
                reason=(
                    "Hard external blocker detected from recent failures. "
                    "Route back through acquisition and execution recovery before further hypothesis or coding work."
                ),
            )
        if self._latest_program_requires_coding(state):
            coding_route = (
                PhaseSpec(ResearchPhase.CODING, "coder", ("program_candidates",)),
                PhaseSpec(ResearchPhase.EXPERIMENT_PLANNING, "planner", ("experiment_plans",)),
                PhaseSpec(ResearchPhase.EXPERIMENT, "experiment", ("experiment_records", "best_known_results")),
                PhaseSpec(ResearchPhase.REFLECTION, "reflection", ("reflections", "next_actions")),
            )
            return CycleRoute(
                phases=coding_route,
                reason=(
                    "Latest program candidate is not yet validated; return to coding and execution before proposing new hypotheses."
                ),
            )
        return CycleRoute(
            phases=tuple(self.iterative_phases),
            reason="No hard blocker detected; run the normal research cycle.",
        )
