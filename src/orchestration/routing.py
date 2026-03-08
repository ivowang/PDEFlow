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
        failure_types = {failure.failure_type for failure in state.classified_failures if failure.blocking}
        capability = state.capability_matrix
        if capability and capability.target_dataset_blocked and not capability.fallback_assets_available and (
            "repeated_repair_failure" in failure_types or "checksum_mismatch" in failure_types
        ):
            state.blocked_reason = (
                "Exact target artifacts are blocked and no validated fallback route is currently executable."
            )
            state.termination_decision = state.blocked_reason
            return CycleRoute(
                phases=(PhaseSpec(ResearchPhase.REFLECTION, "reflection", ("reflections", "next_actions")),),
                reason=state.blocked_reason,
            )
        if "plan_depends_on_blocked_artifact" in failure_types:
            return CycleRoute(
                phases=(
                    PhaseSpec(ResearchPhase.ACQUISITION, "acquisition", ("environment_snapshot", "external_artifacts", "repositories")),
                    PhaseSpec(ResearchPhase.EXPERIMENT_PLANNING, "planner", ("experiment_plans",)),
                    PhaseSpec(ResearchPhase.PREFLIGHT_VALIDATION, "preflight", ("preflight_reports", "capability_matrix")),
                    PhaseSpec(ResearchPhase.REFLECTION, "reflection", ("reflections", "next_actions")),
                ),
                reason=(
                    "Recent plans depend on blocked artifacts. Reacquire or reroute before any launch."
                ),
            )
        if capability and not capability.baseline_ready_to_launch and (
            capability.target_dataset_blocked or "transfer_stalled" in failure_types or "transfer_timeout" in failure_types
        ):
            return CycleRoute(
                phases=tuple(self.recovery_phases),
                reason=(
                    "Capability matrix shows the baseline route is not launch-ready due to infrastructure blockers. "
                    "Route through acquisition, planning, and preflight recovery before further research work."
                ),
            )
        if self._latest_program_requires_coding(state):
            coding_route = (
                PhaseSpec(ResearchPhase.CODING, "coder", ("program_candidates",)),
                PhaseSpec(ResearchPhase.EXPERIMENT_PLANNING, "planner", ("experiment_plans",)),
                PhaseSpec(ResearchPhase.PREFLIGHT_VALIDATION, "preflight", ("preflight_reports", "capability_matrix")),
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
