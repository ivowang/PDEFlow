from __future__ import annotations

from .specs import CycleRoute, PhaseSpec
from state import ResearchPhase, ResearchState


class ManagerRoutingMixin:
    def _latest_program_requires_coding(self, state: ResearchState) -> bool:
        if not state.program_candidates:
            return False
        latest = state.program_candidates[-1]
        if latest.status.startswith("validated") or latest.status in {"evaluated", "completed"}:
            return False
        return bool(state.method_designs)

    def _route_history_contains(self, state: ResearchState, route_id: str) -> bool:
        return any(item.route_id == route_id for item in state.route_history[-8:])

    def _active_dataset_blockers(self, state: ResearchState) -> list[object]:
        return [
            blocker
            for blocker in state.blocker_registry
            if blocker.blocker_type == "dataset_acquisition_failure"
        ]

    def _active_environment_blockers(self, state: ResearchState) -> list[object]:
        return [
            blocker
            for blocker in state.blocker_registry
            if blocker.blocker_type in {"env_resolution_failure", "environment_runtime_failure"}
        ]

    def _build_recovery_route(self, route_id: str, reason: str, focus: tuple[str, ...]) -> CycleRoute:
        return CycleRoute(
            route_id=route_id,
            phases=tuple(self.recovery_phases),
            reason=reason,
            focus=focus,
        )

    def _build_fallback_execution_route(self, reason: str, focus: tuple[str, ...]) -> CycleRoute:
        return CycleRoute(
            route_id="fallback-execution",
            phases=(
                PhaseSpec(ResearchPhase.EXPERIMENT_PLANNING, "planner", ("experiment_plans",)),
                PhaseSpec(ResearchPhase.PREFLIGHT_VALIDATION, "preflight", ("preflight_reports", "capability_matrix")),
                PhaseSpec(ResearchPhase.EXPERIMENT, "experiment", ("experiment_records", "best_known_results")),
                PhaseSpec(ResearchPhase.REFLECTION, "reflection", ("reflections", "next_actions")),
            ),
            reason=reason,
            focus=focus,
        )

    def _build_acquisition_only_route(self, route_id: str, reason: str, focus: tuple[str, ...]) -> CycleRoute:
        return CycleRoute(
            route_id=route_id,
            phases=(
                PhaseSpec(ResearchPhase.ACQUISITION, "acquisition", ("environment_snapshot", "external_artifacts", "repositories")),
            ),
            reason=reason,
            focus=focus,
        )

    def _select_cycle_route(self, state: ResearchState) -> CycleRoute:
        capability = state.capability_matrix
        latest_reflection = state.reflections[-1] if state.reflections else None
        latest_delta = state.cycle_deltas[-1] if state.cycle_deltas else None
        no_material_change = bool(latest_delta and not latest_delta.changed)
        dataset_blockers = self._active_dataset_blockers(state)
        exhausted_dataset_blockers = [item for item in dataset_blockers if item.route_exhausted]
        environment_blockers = self._active_environment_blockers(state)
        exhausted_environment_blockers = [item for item in environment_blockers if item.route_exhausted]
        blocking_failure_types = {
            item.failure_type for item in state.classified_failures if item.blocking
        }
        if latest_reflection and latest_reflection.recommended_route_id == "fallback-execution":
            return self._build_fallback_execution_route(
                reason="Latest reflection explicitly recommends a fallback execution route.",
                focus=tuple(latest_reflection.preferred_recovery_strategies or ["fallback_execution"]),
            )

        if latest_reflection and latest_reflection.escalation_required:
            state.blocked_reason = latest_reflection.stop_reason or latest_reflection.verdict
            state.termination_decision = state.blocked_reason
            return CycleRoute(
                route_id="blocked-terminal",
                phases=(PhaseSpec(ResearchPhase.REFLECTION, "reflection", ("reflections", "next_actions")),),
                reason=state.blocked_reason,
                focus=("terminate_blocked",),
            )

        if capability and capability.target_dataset_preparing and not capability.target_dataset_blocked:
            return self._build_acquisition_only_route(
                route_id="continue-dataset-preparation",
                reason=(
                    "Exact target datasets are still being prepared or validated. "
                    "Continue acquisition and validation until they are training-ready instead of escalating or preflighting empty plans."
                ),
                focus=("complete_dataset_preparation", "environment_setup"),
            )

        if capability and capability.baseline_launch_ready and self._latest_program_requires_coding(state):
            return CycleRoute(
                route_id="coding-validation",
                phases=(
                    PhaseSpec(ResearchPhase.CODING, "coder", ("program_candidates",)),
                    PhaseSpec(ResearchPhase.EXPERIMENT_PLANNING, "planner", ("experiment_plans",)),
                    PhaseSpec(ResearchPhase.PREFLIGHT_VALIDATION, "preflight", ("preflight_reports", "capability_matrix")),
                    PhaseSpec(ResearchPhase.EXPERIMENT, "experiment", ("experiment_records", "best_known_results")),
                    PhaseSpec(ResearchPhase.REFLECTION, "reflection", ("reflections", "next_actions")),
                ),
                reason=(
                    "Latest program candidate is not yet validated and the baseline route is launch-ready."
                ),
                focus=("coding", "baseline_execution"),
            )

        if environment_blockers:
            if (
                capability
                and capability.target_dataset_ready
                and capability.env_ready
                and capability.codepath_ready
                and capability.scientific_iteration_ready
            ):
                return self._build_fallback_execution_route(
                    reason=(
                        "The exact target datasets and baseline codepath are ready, but the preferred GPU runtime is not launch-ready. "
                        "Prefer an executable fallback evidence path over re-entering acquisition."
                    ),
                    focus=("fallback_execution", "cpu_validation"),
                )
            repair_attempted = self._route_history_contains(state, "repair-environment-runtime")
            if not repair_attempted or not no_material_change:
                return self._build_recovery_route(
                    route_id="repair-environment-runtime",
                    reason=(
                        "Managed environment exists but runtime health is insufficient for the preferred baseline path. "
                        "Repair the environment before planning or launch."
                    ),
                    focus=("environment_repair",),
                )
            if capability and capability.scientific_iteration_ready:
                return self._build_fallback_execution_route(
                    reason=(
                        "GPU runtime repair attempts did not restore a launch-ready baseline route. "
                        "Switch to a CPU-executable evidence path instead of repeating the same repair loop."
                    ),
                    focus=("fallback_execution", "cpu_validation"),
                )
            if no_material_change and exhausted_environment_blockers:
                state.blocked_reason = (
                    "Environment runtime repair has been exhausted with no material state change and no executable fallback route."
                )
                state.termination_decision = state.blocked_reason
                return CycleRoute(
                    route_id="blocked-terminal",
                    phases=(PhaseSpec(ResearchPhase.REFLECTION, "reflection", ("reflections", "next_actions")),),
                    reason=state.blocked_reason,
                    focus=("terminate_blocked",),
                )

        if dataset_blockers:
            tried_strategies = {
                strategy
                for blocker in dataset_blockers
                for strategy in blocker.recovery_strategies_tried
            }
            preferred = tuple(latest_reflection.preferred_recovery_strategies) if latest_reflection else ()
            if capability and capability.scientific_iteration_ready and (
                "fallback_execution" in preferred
                or "adapted_loader" in preferred
                or "reduced_scope" in preferred
                or len(exhausted_dataset_blockers) >= 1
            ):
                return self._build_fallback_execution_route(
                    reason=(
                        "Exact target datasets remain blocked, but fallback-capable assets exist. "
                        "Prefer an evidence-generating fallback experiment over another unchanged repair loop."
                    ),
                    focus=("fallback_execution", "reduced_scope"),
                )
            if "local_discovery" not in tried_strategies:
                return self._build_recovery_route(
                    route_id="recover-local-discovery",
                    reason=(
                        "Dataset blockers are active. The next allowed recovery move is local file discovery and validation."
                    ),
                    focus=("local_discovery",),
                )
            if "mirror_resolution" not in tried_strategies:
                return self._build_recovery_route(
                    route_id="recover-mirror-resolution",
                    reason=(
                        "Direct dataset transfer path is not reliable. Switch to mirror or alternate source resolution."
                    ),
                    focus=("mirror_resolution",),
                )
            if "partial_salvage" not in tried_strategies:
                return self._build_recovery_route(
                    route_id="recover-partial-salvage",
                    reason=(
                        "Partial files exist for blocked datasets. Try bounded salvage and validation before another fresh transfer."
                    ),
                    focus=("partial_salvage",),
                )
            if capability and capability.scientific_iteration_ready:
                return self._build_fallback_execution_route(
                    reason=(
                        "Acquisition strategies are exhausted for the current exact target route. "
                        "Use a fallback evidence-generating execution path."
                    ),
                    focus=("fallback_execution",),
                )
            if no_material_change and exhausted_dataset_blockers:
                state.blocked_reason = (
                    "Repeated dataset acquisition strategies have been exhausted with no material state change and no executable fallback assets."
                )
                state.termination_decision = state.blocked_reason
                return CycleRoute(
                    route_id="blocked-terminal",
                    phases=(PhaseSpec(ResearchPhase.REFLECTION, "reflection", ("reflections", "next_actions")),),
                    reason=state.blocked_reason,
                    focus=("terminate_blocked",),
                )

        if "plan_depends_on_blocked_artifact" in blocking_failure_types:
            return CycleRoute(
                route_id="replan-blocked-artifacts",
                phases=(
                    PhaseSpec(ResearchPhase.EXPERIMENT_PLANNING, "planner", ("experiment_plans",)),
                    PhaseSpec(ResearchPhase.PREFLIGHT_VALIDATION, "preflight", ("preflight_reports", "capability_matrix")),
                    PhaseSpec(ResearchPhase.REFLECTION, "reflection", ("reflections", "next_actions")),
                ),
                reason="Existing plans depend on blocked artifacts; replan or fallback before launch.",
                focus=("replan", "blocked_artifact_avoidance"),
            )

        if capability and not capability.baseline_launch_ready:
            if capability.scientific_iteration_ready:
                return self._build_fallback_execution_route(
                    reason=(
                        "The preferred baseline is not launch-ready, but enough infrastructure is ready to attempt a fallback evidence run."
                    ),
                    focus=("fallback_execution",),
                )
            return self._build_recovery_route(
                route_id="recover-baseline-prerequisites",
                reason=(
                    "Capability matrix shows the baseline route is not launch-ready due to missing prerequisites."
                ),
                focus=("baseline_prerequisites",),
            )

        return CycleRoute(
            route_id="normal-research-cycle",
            phases=tuple(self.iterative_phases),
            reason="No hard blocker prevents the normal research cycle.",
            focus=("research",),
        )
