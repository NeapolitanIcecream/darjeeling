# 2026-06-10 L2 target evidence policy

## Goal

Prevent short or cost-capped L2 target-evolution runs from being interpreted as
formal quality evidence. This addresses the earlier confusion where a small
number of live `codex-cli` rounds could look like the full L2 evolve design had
been tested.

## Design Change

`edge-mvp l2 target-evolve` now writes `evidence_policy` in:

- `summary.json`
- `workspace/l2_target/data/round_state.json`
- `workspace/l2_target/data/objective.json`

The policy separates profile intent from evidence strength:

- `standard` is a `cost_capped_probe`.
- `smoke` is a `connectivity_smoke`.
- Short explicit `fixed-inner` budgets are `short_fixed_snapshot_probe`.
- Fixed-inner runs with fewer than 500 teacher-labeled traces are
  `small_snapshot_probe`.
- Only sufficiently budgeted `fixed-inner` runs are `fixed_snapshot_research`.

Even `fixed_snapshot_research` is not adoption; it only becomes quality evidence
after private selection/promotion gates and outer e2e replay also pass.

The legacy compiler `L2_AGENT_MODE` path now records
`l2_agent_harness_role=legacy_patch_generation_not_target_evolve`, so its patch
artifact cannot be confused with the main target-dependent runtime path.

## Smoke

Command:

```bash
uv run edge-mvp l2 target-evolve \
  --traces runs/l2-list-fallback-tuned-3k-r1/traces.jsonl \
  --out-dir runs/l2-target-evidence-policy-smoke-r1 \
  --max-traces 80 \
  --mode dry-run \
  --budget-profile fixed-inner \
  --rounds 2
```

Result:

- `budget_policy.profile=fixed-inner`
- `budget_policy.profile_intent.profile_role=fixed_snapshot_research`
- `evidence_policy.evidence_class=short_fixed_snapshot_probe`
- `evidence_policy.quality_claim_supported=false`
- `evidence_policy.blocking_reasons=["round budget 2 is below quality minimum 16"]`
- `evidence_policy.teacher_labeled_traces=80`
- `round_state.json` and `objective.json` carry the same evidence class.
- Agent-visible workspace data contains train and visible validation folds only;
  `selection_holdout` and `promotion_holdout` are not present in workspace state
  files.

This smoke validates the evidence-policy wiring. It is not an L2 quality result.
