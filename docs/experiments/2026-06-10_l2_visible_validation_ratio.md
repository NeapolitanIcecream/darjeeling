# 2026-06-10 L2 visible validation ratio

## Goal

Follow up the real L2 `agent-session` budget experiment, where a candidate
passed private selection but failed private promotion. The working hypothesis is
that the agent-visible validation pressure should be adjustable without
exposing private holdout rows or private gate aggregates.

## Design Change

`edge-mvp l2 target-evolve` now accepts:

```bash
--visible-validation-ratio FLOAT
```

When omitted, existing defaults stay unchanged:

- one visible fold keeps the old 60/20/10/10 split;
- multiple visible folds default to the fixed-inner 50/30/10/10 shape.

When provided, the ratio controls the total agent-visible validation pool, while
private selection and private promotion remain outside the workspace. Summary,
`data/round_state.json`, and `data/objective.json` record the requested ratio.
`data_split_policy` also records the effective ratio after integer rounding and
intent-stratified group constraints.

The same implementation pass fixed a reporting bug exposed by the experiment:
private selection/promotion `safety_backlog` entries now use
`outer_summary_only_not_agent_workspace` visibility instead of the misleading
`visible_validation_only` label.

## Experiment

Command:

```bash
uv run edge-mvp l2 target-evolve \
  --traces runs/l2-list-fallback-tuned-3k-r1/traces.jsonl \
  --out-dir runs/l2-target-visible-ratio40-smoke-r2 \
  --max-traces 1000 \
  --mode local-search \
  --budget-profile fixed-inner \
  --target-scope lower_miss \
  --split-policy intent-stratified \
  --rounds 1 \
  --visible-validation-folds 5 \
  --visible-validation-ratio 0.4 \
  --visible-cross-audit-folds 3 \
  --local-search-trials 4 \
  --local-search-timeout-s 120 \
  --local-search-cross-audit-top-k 1
```

Result:

- Evidence class: `short_fixed_snapshot_probe`.
- Scoped lower-miss traces: `999` out of `1000`; `1` lower-layer accept
  excluded.
- Split: train `435`, visible validation `377`, private selection `97`,
  private promotion `90`.
- Requested visible validation ratio: `0.4`.
- Effective visible validation ratio: `0.37737737737737737`.
- Baseline visible validation: `14` accepted, `11` correct, `3` wrong; gate
  failed.
- Local-search candidate was not applied:
  `best visible/cross-audit config failed visible cross-audit safety gate`.
- Best visible cross-audit candidate: `4` accepted, `3` correct, `1` wrong;
  gate failed.
- Private selection: `6` accepted, `5` correct, `1` wrong; gate failed.
- Private promotion: `3` accepted, `2` correct, `1` wrong; gate failed.
- Adoption: `adopted=false`.

## Conclusion

The ratio switch works as a visible-only pressure control and does not change
private holdout visibility. The run is intentionally short, so it is not a
quality claim, but it supports using a larger visible pool before spending
another live L2 agent-session budget. The stronger visible pool caught visible
wrong accepts immediately, and cross-audit still vetoed the best local-search
config.

Next live-agent quality work should use the new ratio on a larger fixed snapshot
instead of iterating from private promotion failures.
