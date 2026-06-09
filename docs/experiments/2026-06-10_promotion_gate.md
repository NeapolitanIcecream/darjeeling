# 2026-06-10 promotion gate notes

## Goal

Reduce the artifact-set promotion risk where an overall objective improvement
can hide a regression in one weak layer.

## Design change

Promotion is still atomic at the artifact-set level, but the mainline decision
now has a per-layer regression hard gate:

- `PROMOTION_BLOCK_LAYER_REGRESSIONS=true` by default.
- If the overall objective would promote but any layer has significant accepted
  accuracy regression, wrong-accept regression, or p95 latency regression, the
  candidate is rejected with `promotion_reason="per-layer regression gate
  failed: ..."`.
- `regressed_layers` is still written for diagnosis.
- `promoted_with_layer_regression=true` is now reserved for explicit diagnostic
  paths where the gate is disabled or artifacts are force-promoted.

This does not implement shadow promotion, layer quarantine, or per-layer
rollback. It only prevents the known artifact-set masking failure from becoming
a silent promotion.

## Validation

Focused tests:

```bash
uv run pytest tests/test_replay_promotion.py tests/test_compiler_loop.py tests/test_settings.py -q
```

Result: 26 passed.

Smoke run:

```bash
mkdir -p runs/promotion-layer-gate-smoke-r1
cp runs/l2-list-fallback-tuned-3k-r1/teacher_cache.jsonl \
  runs/promotion-layer-gate-smoke-r1/teacher_cache.jsonl
uv run edge-mvp run \
  --run-dir runs/promotion-layer-gate-smoke-r1 \
  --teacher cache \
  --max-requests 40 \
  --compile-every 20
```

Result:

- Runtime layer counts: `L0=1`, `L1=1`, `L2=0`, `L3=0`, `L4=38`.
- Two compiler generations ran and both promoted.
- Both `promotion.json` records include
  `candidate_metrics.promotion_block_layer_regressions=true`.
- Both records have `regressed_layers=[]` and
  `promoted_with_layer_regression=false`.

The smoke run validates artifact wiring and record shape, not promotion quality
at scale. Larger replay remains required once L2/L1 candidates produce material
coverage shifts.
