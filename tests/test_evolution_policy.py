from dataclasses import fields

from darjeeling.compiler.evolution_policy import (
    EvolutionRoundResult,
    EvolutionRunPolicy,
    EvolutionRunSummary,
    evolution_run_summary_payload,
)


def test_evolution_policy_exposes_only_round_controls() -> None:
    policy_fields = {field.name for field in fields(EvolutionRunPolicy)}
    round_fields = {field.name for field in fields(EvolutionRoundResult)}
    summary_fields = {field.name for field in fields(EvolutionRunSummary)}

    assert policy_fields == {
        "max_rounds",
        "round_timeout_s",
        "patience_rounds",
        "round_executor",
    }
    assert round_fields == {
        "round_index",
        "status",
        "candidate_ref",
        "metrics",
        "diagnostics",
        "improved",
        "adoptable",
        "stop_reason",
    }
    assert summary_fields == {
        "max_rounds",
        "rounds_completed",
        "stop_reason",
        "round_results",
    }


def test_evolution_summary_counts_completed_rounds() -> None:
    policy = EvolutionRunPolicy(max_rounds=3, round_executor="fake")
    payload = evolution_run_summary_payload(
        policy=policy,
        stop_reason="validation_failed",
        round_results=[
            EvolutionRoundResult(round_index=1, status="completed"),
            EvolutionRoundResult(
                round_index=2,
                status="validation_failed",
                stop_reason="candidate_validation_failed",
            ),
        ],
    )

    assert payload["max_rounds"] == 3
    assert payload["rounds_completed"] == 1
    assert payload["stop_reason"] == "validation_failed"
    assert payload["round_results"][0]["status"] == "completed"
    assert payload["round_results"][1]["stop_reason"] == "candidate_validation_failed"


def test_evolution_policy_has_no_deleted_budget_or_quality_fields() -> None:
    names = {
        *{field.name for field in fields(EvolutionRunPolicy)},
        *{field.name for field in fields(EvolutionRoundResult)},
        *{field.name for field in fields(EvolutionRunSummary)},
    }

    assert names.isdisjoint(
        {
            "max_agent_rounds",
            "agent_budget",
            "local_search_consumes_llm",
            "quality_claim_supported",
            "requires_outer_replay",
            "quality_profile",
            "quality_evidence_class",
            "budget_profile",
            "profile_guidance",
            "fixed_trace_snapshot_inner_loop",
        }
    )
