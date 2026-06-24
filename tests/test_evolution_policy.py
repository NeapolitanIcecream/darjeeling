from darjeeling.compiler.evolution_policy import (
    EvolutionEvidenceRequirements,
    OuterEvolutionPolicy,
    agent_budget_payload,
    evidence_policy_payload,
    resolve_max_agent_rounds,
)


def test_outer_policy_resolves_live_agent_budget_profiles() -> None:
    assert (
        resolve_max_agent_rounds(
            mode="codex-cli",
            budget_profile="standard",
            max_agent_rounds=None,
        )
        == 3
    )
    assert (
        resolve_max_agent_rounds(
            mode="codex-cli",
            budget_profile="fixed-inner",
            max_agent_rounds=None,
        )
        == 16
    )
    assert (
        resolve_max_agent_rounds(
            mode="local-search",
            budget_profile="fixed-inner",
            max_agent_rounds=None,
        )
        is None
    )


def test_outer_policy_classifies_agent_session_without_launch() -> None:
    policy = OuterEvolutionPolicy(
        layer_name="L3",
        mode="agent-session",
        rounds_requested=1,
        budget_profile="fixed-inner",
        max_agent_rounds=0,
    )

    budget = agent_budget_payload(
        policy,
        schema_version="test-agent-budget-v1",
        agent_rounds_started=0,
        agent_rounds_succeeded=0,
    )
    evidence = evidence_policy_payload(
        policy,
        schema_version="test-evidence-v1",
        requirements=EvolutionEvidenceRequirements(
            min_rounds_requested=1,
            min_codex_cli_agent_rounds=1,
        ),
        rounds_completed=0,
        stop_reason="agent_session_budget_exhausted",
        supported_interpretation="supported",
        unsupported_interpretation="unsupported",
    )

    assert budget["agent_rounds_remaining"] == 0
    assert evidence["evidence_class"] == "agent_session_not_launched_probe"
    assert evidence["quality_claim_supported"] is False
