# 2026-06-10 L2 lower-miss target scope

## Goal

Test whether L2 target evolution should operate on the residual distribution
that actually reaches L2, rather than on the full teacher-labeled trace stream.

## Design Change

`edge-mvp l2 target-evolve` now accepts:

```bash
--target-scope teacher_train|lower_miss
```

- `teacher_train` is the default and preserves the previous behavior.
- `lower_miss` keeps only teacher-labeled traces where L0/L1 did not accept.

The run summary and agent-visible state files now record `target_scope` with:

- input teacher-labeled trace count
- scoped teacher-labeled trace count
- lower-layer accepted rows excluded
- selection basis

This scope changes only target-visible train/validation data. Private
selection/promotion holdouts remain outside the workspace.

The same experiment exposed a local-search safety issue: a config could pass
visible validation while failing visible cross-audit. Local-search now treats
enabled cross-audit rerank as a visible safety veto: if the best config fails
cross-audit, it is not written to `target/config.json`.

## Experiment

Command:

```bash
uv run edge-mvp l2 target-evolve \
  --traces runs/l2-list-fallback-tuned-3k-r1/traces.jsonl \
  --out-dir runs/l2-target-lower-miss-scope-veto-smoke-r1 \
  --max-traces 3000 \
  --mode local-search \
  --budget-profile fixed-inner \
  --target-scope lower_miss \
  --rounds 1 \
  --local-search-trials 4 \
  --visible-cross-audit-folds 2 \
  --local-search-cross-audit-top-k 1
```

Result:

- Input teacher-labeled traces: `3000`
- Scoped lower-miss traces: `2806`
- L0/L1 accepted rows excluded: `194`
- Evidence class: `short_fixed_snapshot_probe`
- Best local-search visible validation:
  - accepted `8`
  - correct `8`
  - wrong `0`
  - accepted accuracy `1.0`
  - passes gate `true`
- Best visible cross-audit:
  - accepted `26`
  - correct `22`
  - wrong `4`
  - accepted accuracy `0.8461538461538461`
  - passes gate `false`
- Local-search decision:
  - `cross_audit_safety_veto=true`
  - `applied=false`
  - reason: `best visible/cross-audit config failed visible cross-audit safety gate`

## Conclusion

The lower-miss scope works and should be used for residual-distribution L2
diagnostics. The more important finding is that visible validation alone is too
weak for local-search config adoption: cross-audit caught wrong accepts before
private selection was consulted. This supports using cross-audit as a cheap
visible safety veto whenever it is enabled.

## Local-search current-config fix

The first smoke also exposed that `current_visible_cross_audit` could be
evaluated after Optuna had left `target/config.json` at the last trial. That
made the current/best comparison noisy. The harness now restores the original
target config before evaluating current cross-audit, then writes candidate
configs only for reranked trials.

Follow-up smoke:

```bash
uv run edge-mvp l2 target-evolve \
  --traces runs/l2-list-fallback-tuned-3k-r1/traces.jsonl \
  --out-dir runs/l2-target-lower-miss-scope-veto-smoke-r2 \
  --max-traces 3000 \
  --mode local-search \
  --budget-profile fixed-inner \
  --target-scope lower_miss \
  --rounds 1 \
  --local-search-trials 4 \
  --visible-cross-audit-folds 2 \
  --local-search-cross-audit-top-k 1
```

Result after the fix:

- `search.current_visible_cross_audit` now matches baseline cross-audit:
  accepted `77`, correct `55`, wrong `22`, accepted accuracy
  `0.7142857142857143`, gate `false`.
- Best local-search visible validation still passes: accepted `8`, correct `8`,
  wrong `0`, gate `true`.
- Best visible cross-audit still fails: accepted `26`, correct `22`, wrong `4`,
  accepted accuracy `0.8461538461538461`, gate `false`.
- The candidate is still vetoed and not applied.

This does not change the quality conclusion because the run is still a
one-round `short_fixed_snapshot_probe`. It does make the local-search report
internally consistent.
