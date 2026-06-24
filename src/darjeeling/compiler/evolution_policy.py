from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

EvolutionBudgetProfile = Literal["standard", "fixed-inner", "smoke"]

LIVE_AGENT_MODES = frozenset({"codex-cli", "agent-session"})
AGENT_SESSION_MODE = "agent-session"


@dataclass(frozen=True)
class EvolutionEvidenceRequirements:
    quality_profile: EvolutionBudgetProfile = "fixed-inner"
    quality_evidence_class: str = "fixed_snapshot_research"
    min_rounds_requested: int = 16
    min_codex_cli_agent_rounds: int = 8
    min_teacher_labeled_traces: int | None = None
    agent_session_requires_one_completed_session: bool = True
    requires_private_selection_gate: bool = False
    requires_private_promotion_gate: bool = False
    requires_outer_replay: bool = True


@dataclass(frozen=True)
class OuterEvolutionPolicy:
    layer_name: str
    mode: str
    rounds_requested: int = 1
    budget_profile: EvolutionBudgetProfile = "standard"
    timeout_s: float | None = None
    max_agent_rounds: int | None = None
    inner_patience_rounds: int = 0
    stop_on_selection_gate: bool = False
    codex_command: str = "codex"
    codex_model: str | None = None
    local_search_consumes_llm: bool = False
    fixed_trace_snapshot_inner_loop: bool = True
    outer_replay_cadence_bound: bool = False
    prompt_strategy: str | None = None
    cost_policy: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def live_agent_mode(self) -> bool:
        return self.mode in LIVE_AGENT_MODES

    @property
    def agent_session_mode(self) -> bool:
        return self.mode == AGENT_SESSION_MODE


def resolve_max_agent_rounds(
    *,
    mode: str,
    budget_profile: EvolutionBudgetProfile,
    max_agent_rounds: int | None,
) -> int | None:
    if mode not in LIVE_AGENT_MODES:
        return max_agent_rounds
    if max_agent_rounds is not None:
        return max_agent_rounds
    if mode == AGENT_SESSION_MODE:
        return 1
    if budget_profile == "standard":
        return 3
    if budget_profile == "fixed-inner":
        return 16
    return 1


def profile_guidance_payload(
    policy: OuterEvolutionPolicy,
    *,
    schema_version: str,
    max_agent_rounds: int | None = None,
    recommended_quality_profile: EvolutionBudgetProfile = "fixed-inner",
    guidance_by_profile: dict[EvolutionBudgetProfile, str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_max_agent_rounds = (
        resolve_max_agent_rounds(
            mode=policy.mode,
            budget_profile=policy.budget_profile,
            max_agent_rounds=policy.max_agent_rounds,
        )
        if max_agent_rounds is None
        else max_agent_rounds
    )
    if policy.budget_profile == "fixed-inner":
        profile_role = "fixed_snapshot_research"
        guidance = (
            "Use this profile for the main fixed-snapshot research loop; "
            "it is deliberately decoupled from outer replay cadence."
        )
    elif policy.budget_profile == "smoke":
        profile_role = "connectivity_smoke"
        guidance = (
            "Use this profile only to check wiring; do not treat its result as "
            "evidence about evolution quality."
        )
    else:
        profile_role = "cost_capped_default"
        guidance = (
            "The standard profile is cost-capped. For codex-cli it may launch "
            "only a few live agent rounds, so failure here is not evidence that "
            "the evolution route has been exhausted."
        )
    if guidance_by_profile is not None:
        guidance = guidance_by_profile.get(policy.budget_profile, guidance)
    payload: dict[str, Any] = {
        "schema_version": schema_version,
        "profile": policy.budget_profile,
        "profile_role": profile_role,
        "recommended_quality_profile": recommended_quality_profile,
        "guidance": guidance,
        "fixed_trace_snapshot_inner_loop": policy.fixed_trace_snapshot_inner_loop,
        "outer_replay_cadence_bound": policy.outer_replay_cadence_bound,
        "agent_session_controls_internal_loop": policy.agent_session_mode,
        "local_search_consumes_llm": policy.local_search_consumes_llm,
        "codex_cli_rounds_consume_llm": policy.mode == "codex-cli",
        "live_agent_session_consumes_llm": policy.agent_session_mode,
        "effective_max_agent_rounds": resolved_max_agent_rounds,
        "agent_round_cap_is_cost_control": (
            policy.live_agent_mode and resolved_max_agent_rounds is not None
        ),
    }
    if extra:
        payload.update(extra)
    return payload


def budget_policy_payload(
    policy: OuterEvolutionPolicy,
    *,
    schema_version: str | None = None,
    profile_guidance_schema_version: str,
    max_agent_rounds: int | None = None,
    guidance_by_profile: dict[EvolutionBudgetProfile, str] | None = None,
    extra: dict[str, Any] | None = None,
    profile_guidance_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_max_agent_rounds = (
        resolve_max_agent_rounds(
            mode=policy.mode,
            budget_profile=policy.budget_profile,
            max_agent_rounds=policy.max_agent_rounds,
        )
        if max_agent_rounds is None
        else max_agent_rounds
    )
    payload: dict[str, Any] = {}
    if schema_version is not None:
        payload["schema_version"] = schema_version
    payload.update(
        {
            "inner_patience_rounds": policy.inner_patience_rounds,
            "stop_on_selection_gate": policy.stop_on_selection_gate,
            "max_agent_rounds": resolved_max_agent_rounds,
            "profile": policy.budget_profile,
            "profile_guidance": profile_guidance_payload(
                policy,
                schema_version=profile_guidance_schema_version,
                max_agent_rounds=resolved_max_agent_rounds,
                guidance_by_profile=guidance_by_profile,
                extra=profile_guidance_extra,
            ),
        }
    )
    if extra:
        payload.update(extra)
    return payload


def agent_budget_payload(
    policy: OuterEvolutionPolicy,
    *,
    schema_version: str,
    agent_rounds_started: int,
    agent_rounds_succeeded: int,
    max_agent_rounds: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_max_agent_rounds = (
        resolve_max_agent_rounds(
            mode=policy.mode,
            budget_profile=policy.budget_profile,
            max_agent_rounds=policy.max_agent_rounds,
        )
        if max_agent_rounds is None
        else max_agent_rounds
    )
    remaining = (
        None
        if resolved_max_agent_rounds is None
        else max(0, resolved_max_agent_rounds - agent_rounds_started)
    )
    payload: dict[str, Any] = {
        "schema_version": schema_version,
        "applies_to_mode": policy.live_agent_mode,
        "mode": policy.mode,
        "codex_command": policy.codex_command if policy.live_agent_mode else None,
        "codex_model": policy.codex_model if policy.live_agent_mode else None,
        "timeout_s": policy.timeout_s if policy.live_agent_mode else None,
        "max_agent_rounds": resolved_max_agent_rounds,
        "agent_rounds_started": agent_rounds_started,
        "agent_rounds_succeeded": agent_rounds_succeeded,
        "agent_rounds_remaining": remaining,
        "agent_session_scope": (
            "single_session_agent_controls_internal_loop"
            if policy.agent_session_mode
            else "one_codex_process_per_outer_round"
            if policy.mode == "codex-cli"
            else None
        ),
        "local_search_consumes_llm": policy.local_search_consumes_llm,
    }
    if policy.prompt_strategy is not None:
        payload["prompt_strategy"] = policy.prompt_strategy
    if policy.cost_policy is not None:
        payload["cost_policy"] = policy.cost_policy
    if extra:
        payload.update(extra)
    return payload


def evidence_policy_payload(
    policy: OuterEvolutionPolicy,
    *,
    schema_version: str,
    requirements: EvolutionEvidenceRequirements,
    rounds_completed: int | None = None,
    stop_reason: str | None = None,
    teacher_labeled_traces: int | None = None,
    max_agent_rounds: int | None = None,
    quality_claim_supported_text: str = "eligible_after_private_gates_and_outer_replay",
    unsupported_quality_claim_text: str = "not_supported_by_this_run",
    supported_interpretation: str,
    unsupported_interpretation: str,
) -> dict[str, Any]:
    resolved_max_agent_rounds = (
        resolve_max_agent_rounds(
            mode=policy.mode,
            budget_profile=policy.budget_profile,
            max_agent_rounds=policy.max_agent_rounds,
        )
        if max_agent_rounds is None
        else max_agent_rounds
    )
    blocking_reasons: list[str] = []

    if policy.budget_profile == "smoke":
        evidence_class = "connectivity_smoke"
        blocking_reasons.append("smoke profile only validates wiring")
    elif policy.budget_profile == "standard":
        evidence_class = "cost_capped_probe"
        blocking_reasons.append(
            "standard profile is cost-capped and may launch only a few live agent rounds"
        )
    elif policy.rounds_requested < requirements.min_rounds_requested:
        evidence_class = "short_fixed_snapshot_probe"
        blocking_reasons.append(
            "round budget "
            f"{policy.rounds_requested} is below quality minimum "
            f"{requirements.min_rounds_requested}"
        )
    elif (
        requirements.min_teacher_labeled_traces is not None
        and teacher_labeled_traces is not None
        and teacher_labeled_traces < requirements.min_teacher_labeled_traces
    ):
        evidence_class = "small_snapshot_probe"
        blocking_reasons.append(
            "teacher-labeled snapshot size "
            f"{teacher_labeled_traces} is below quality minimum "
            f"{requirements.min_teacher_labeled_traces}"
        )
    elif (
        policy.mode == "codex-cli"
        and resolved_max_agent_rounds is not None
        and resolved_max_agent_rounds < requirements.min_codex_cli_agent_rounds
    ):
        evidence_class = "agent_budget_capped_fixed_snapshot"
        blocking_reasons.append(
            "codex-cli agent round cap is below the quality evidence minimum"
        )
    elif policy.agent_session_mode and resolved_max_agent_rounds == 0:
        evidence_class = "agent_session_not_launched_probe"
        blocking_reasons.append("agent-session mode did not launch a live agent session")
    elif (
        policy.agent_session_mode
        and stop_reason is not None
        and requirements.agent_session_requires_one_completed_session
        and (
            stop_reason != "agent_session_completed"
            or rounds_completed is None
            or rounds_completed < 1
        )
    ):
        evidence_class = "incomplete_agent_session_probe"
        blocking_reasons.append(
            "agent-session did not complete one scoped candidate evaluation"
        )
    else:
        evidence_class = requirements.quality_evidence_class

    if (
        evidence_class == requirements.quality_evidence_class
        and not policy.agent_session_mode
        and stop_reason is not None
        and rounds_completed is not None
        and rounds_completed < min(policy.rounds_requested, requirements.min_rounds_requested)
    ):
        if stop_reason not in {"selection_gate_passed", "baseline_selection_gate_passed"}:
            evidence_class = "incomplete_fixed_snapshot_probe"
            blocking_reasons.append(
                f"completed {rounds_completed} rounds before reaching the requested evidence budget"
            )

    quality_claim_supported = evidence_class == requirements.quality_evidence_class
    return {
        "schema_version": schema_version,
        "evidence_class": evidence_class,
        "quality_claim_supported": quality_claim_supported,
        "quality_claim": (
            quality_claim_supported_text
            if quality_claim_supported
            else unsupported_quality_claim_text
        ),
        "result_interpretation": (
            supported_interpretation
            if quality_claim_supported
            else unsupported_interpretation
        ),
        "required_for_quality_claim": {
            "budget_profile": requirements.quality_profile,
            "min_rounds_requested": requirements.min_rounds_requested,
            "min_codex_cli_agent_rounds": requirements.min_codex_cli_agent_rounds,
            "agent_session_requires_one_completed_session": (
                requirements.agent_session_requires_one_completed_session
            ),
            "min_teacher_labeled_traces": requirements.min_teacher_labeled_traces,
            "requires_private_selection_gate": requirements.requires_private_selection_gate,
            "requires_private_promotion_gate": requirements.requires_private_promotion_gate,
            "requires_outer_replay": requirements.requires_outer_replay,
        },
        "blocking_reasons": blocking_reasons,
        "profile": policy.budget_profile,
        "mode": policy.mode,
        "rounds_requested": policy.rounds_requested,
        "rounds_completed": rounds_completed,
        "stop_reason": stop_reason,
        "teacher_labeled_traces": teacher_labeled_traces,
        "fixed_trace_snapshot_inner_loop": policy.fixed_trace_snapshot_inner_loop,
        "outer_replay_cadence_bound": policy.outer_replay_cadence_bound,
        "effective_max_agent_rounds": resolved_max_agent_rounds,
        "agent_round_cap_is_cost_control": (
            policy.live_agent_mode and resolved_max_agent_rounds is not None
        ),
    }
