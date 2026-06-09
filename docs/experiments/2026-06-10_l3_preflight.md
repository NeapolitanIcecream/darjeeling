# 2026-06-10 L3 preflight notes

## Goal

Make local L3 SLM readiness explicit and switchable without blocking default
experiments. Preflight should not load the model; it should only report whether
the current mode can proceed safely from existing settings and benchmark
artifacts.

## Design change

- `LOCAL_SLM_MODE=disabled`: pass, non-blocking, no benchmark required, no model
  load attempted.
- `LOCAL_SLM_MODE=shadow`: missing or failed benchmark is warn. The run can
  proceed because runtime degrades shadow failures to disabled.
- `LOCAL_SLM_MODE=guarded`: missing or failed benchmark is fail. Guarded L3 can
  become the final route, so experiments should not proceed without a successful
  hardware/model benchmark artifact.
- Successful benchmark artifacts expose bounded fields in preflight:
  `actual_device`, request count, and generation p50/p95 latency.

## Validation

Default disabled preflight:

```bash
uv run edge-mvp experiment preflight \
  --run-dir runs/l2-list-fallback-tuned-3k-r1 \
  --teacher cache \
  --out runs/l3-preflight-disabled-smoke-r1.json
```

Result:

- Overall status: `pass`.
- L3 readiness: `disabled_nonblocking`.
- `model_load_attempted=false`.
- `runtime_blocking=false`.
- `benchmark_required=false`.

Shadow without benchmark:

```bash
LOCAL_SLM_MODE=shadow uv run edge-mvp experiment preflight \
  --run-dir runs/l2-list-fallback-tuned-3k-r1 \
  --teacher cache \
  --out runs/l3-preflight-shadow-missing-smoke-r1.json
```

Result:

- Overall status: `pass`, because warn checks do not fail preflight.
- L3 status: `warn`.
- L3 readiness: `benchmark_missing`.
- `runtime_blocking=false`.
- `model_load_attempted=false`.

Focused tests:

```bash
uv run pytest tests/test_experiments.py tests/test_l3_local_slm.py -q
```

Result: 18 passed.

## Conclusion

L3 remains non-blocking by default, but guarded-mode experiments now require
evidence from `edge-mvp l3 bench --out ...`. This keeps hardware adaptation
explicit and avoids hidden model loads in experiment preflight.
