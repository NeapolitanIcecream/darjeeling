# Outer Evolution Policy Simplification Repair Report

Date: 2026-06-24

Worktree: `/Users/chenmohan/gits/darjeeling-outer-evolution-policy`
Branch: `codex/outer-evolution-policy`

## Objective

Repair the outer evolution policy refactor so Darjeeling uses one real round
model across L1, L2, and L3. The repair removes shared agent budget/profile and
quality evidence vocabulary that was either not executed or only existed for
legacy artifact compatibility.

## Design Outcome

- Core now exposes only generic run mechanics:
  `EvolutionRunPolicy(max_rounds, round_timeout_s, patience_rounds,
  round_executor)`, `EvolutionRoundResult`, and `EvolutionRunSummary`.
- Core no longer declares target quality, private gate requirements, replay
  requirements, profile guidance, cost policy, or fixed inner-loop cadence.
- L1 now executes real policy-controlled rounds. Each round has an isolated
  round workspace, transcript, report, diff, command result, validation result,
  and round result payload. Later rounds start from the latest successful
  candidate crate. Dry-run mode accepts one patch per round.
- L3 now uses `max_rounds` instead of `max_agent_sessions`. Each round runs one
  prompt evolution agent session, validates the prompt artifact, evaluates the
  candidate when replay is enabled, and records round results.
- L2 keeps its real multi-round train/evaluate behavior and target-local
  budget preset resolver, but no longer emits generic `budget_policy`,
  `evidence_policy`, `agent_budget`, `loop_cadence`, or agent-round caps.
- Target/outer replay reporting now carries selection, promotion, adoption,
  private holdout evidence, and `l2_target_round_policy` as target-owned
  payloads rather than core quality claims.

## Removed Compatibility Surface

Removed from runtime source and new summary payloads:

- `max_agent_rounds`, `max_agent_sessions`, `agent_budget`;
- shared `budget_policy`, `evidence_policy`, `profile_guidance`;
- core `quality_claim_supported`, `requires_private_*`, `requires_outer_replay`;
- `fixed_trace_snapshot_inner_loop`, `loop_cadence`, and inner-loop summary
  vocabulary;
- L1 `L1_AGENT_BUDGET_PROFILE`, `L1_AGENT_MAX_AGENT_ROUNDS`, and partial
  `L1_AGENT_ROUNDS` behavior;
- L3 `--budget-profile` and `--max-agent-sessions`;
- L2 `--max-agent-rounds` and legacy payload fields used only by old artifacts.

Historical artifacts remain interpretable through the old commit/report; new
runs do not carry those fields forward.

## Validation

All required validation passed in this worktree:

```text
uv run --extra dev pytest tests/test_evolution_policy.py tests/targets/nlu/test_l1_coding_agent.py tests/targets/nlu/test_l2_target_evolution.py tests/targets/nlu/test_l3_prompt_optimizer.py -q
63 passed in 13.67s

uv run --extra dev --extra massive pytest -q
315 passed in 31.42s

uv run --extra dev ruff check src tests
All checks passed!

git diff --check
passed
```

No paid benchmark or live teacher calls were run for this repair.
