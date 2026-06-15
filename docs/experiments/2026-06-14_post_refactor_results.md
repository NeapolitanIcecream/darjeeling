# Post-Refactor Experiment Results

Date: 2026-06-14

Source plan: `docs/experiments/2026-06-14_post_refactor_experiment_plan.md`

Run root: `runs/post-refactor-20260614-2105`

Recorded runtime commit: `b819f99d1d17`

Note: the suite metadata records the git commit available to the CLI at run time. The experiment wiring in this results pass was applied in the working tree on top of that commit.

## Commands

```bash
uv run pytest tests/targets/nlu/test_experiments.py tests/targets/nlu/test_experiment_suite_cli.py tests/targets/nlu/test_report_l3_summary.py tests/targets/nlu/test_settings.py -q
uv run pytest tests/test_core_contracts.py tests/test_target_boundary.py tests/targets/nlu/test_target_core_boundary.py -q
uv run pytest -q

uv run edge-mvp-nlu experiment preflight \
  --run-dir runs/post-refactor-20260614-2105/preflight \
  --data-dir data/processed/massive_en_us \
  --teacher cache \
  --check-l1-build \
  --out runs/post-refactor-20260614-2105/preflight.json

uv run edge-mvp-nlu experiment preflight \
  --experiment l3-guarded \
  --run-dir runs/post-refactor-20260614-2105/preflight-l3-guarded \
  --data-dir data/processed/massive_en_us \
  --teacher cache \
  --out runs/post-refactor-20260614-2105/preflight-l3-guarded.json

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run edge-mvp-nlu l3 bench \
  --mode guarded \
  --data-dir data/processed/massive_en_us \
  --out runs/post-refactor-20260614-2105/preflight-l3-guarded/reports/l3_benchmark.json \
  --fail-on-error

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run edge-mvp-nlu experiment suite \
  --run-root runs/post-refactor-20260614-2105/smoke \
  --max-requests 10 \
  --compile-every 5 \
  --teacher cache \
  --data-dir data/processed/massive_en_us \
  --parallel 1

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run edge-mvp-nlu experiment suite \
  --run-root runs/post-refactor-20260614-2105/medium \
  --include-guarded-l3 \
  --max-requests 100 \
  --compile-every 50 \
  --teacher cache \
  --data-dir data/processed/massive_en_us \
  --parallel 1

uv run edge-mvp-nlu experiment preflight \
  --run-dir runs/post-refactor-20260614-2105/live-preflight \
  --data-dir data/processed/massive_en_us \
  --teacher live \
  --out runs/post-refactor-20260614-2105/live-preflight.json
```

## Validation

- Focused experiment/report/settings tests: `47 passed`.
- Boundary tests: `16 passed`.
- Full suite: `261 passed`.
- Base cache preflight: pass with processed MASSIVE data, cache teacher, L1 crate build, and L3 disabled.
- Initial guarded L3 preflight: failed because benchmark evidence was missing.
- Guarded L3 benchmark: success on `mps:0`, 3 requests, p50/p95 generation `1148.382/3073.672 ms`, accepted `0/3`.
- Guarded L3 preflight after benchmark: pass.
- Live teacher preflight: pass, but no live residual run was launched.

## Matrix

Medium cache-backed matrix:

- `max_requests`: 100
- `compile_every`: 50
- `teacher`: `cache`
- `data_dir`: `data/processed/massive_en_us`
- `parallel`: 1
- teacher cache source: `runs/oracle-cache-massive/teacher_cache.jsonl`
- comparison: `runs/post-refactor-20260614-2105/medium/comparison/comparison.csv`

| Run | Req | Frame exact | p95 ms | Weak cov | Weak acc | Wrong field | Full L4/100 | Residual L4/100 | L3 share | Cost/100 | Promotions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| main-evolution | 100 | 1.000 | 1.934 | 0.053 | 1.000 | 0.000 | 92.000 | 5.000 | 0.000 | 0.000 | 2 |
| workload-locality:uniform | 100 | 1.000 | 2.037 | 0.000 | 1.000 | 0.000 | 100.000 | 0.000 | 0.000 | 0.000 | 2 |
| workload-locality:zipf-mild | 100 | 1.000 | 1.884 | 0.010 | 1.000 | 0.000 | 99.000 | 0.000 | 0.000 | 0.000 | 1 |
| workload-locality:zipf-heavy | 100 | 1.000 | 2.468 | 0.053 | 1.000 | 0.000 | 92.000 | 5.000 | 0.000 | 0.000 | 2 |
| no-guard | 100 | 0.590 | 1.982 | 0.471 | 0.443 | 0.262 | 50.000 | 5.000 | 0.000 | 0.000 | 2 |
| no-audit | 100 | 1.000 | 1.894 | 0.053 | 1.000 | 0.000 | 92.000 | 5.000 | 0.000 | 0.000 | 2 |
| no-l2 | 100 | 1.000 | 0.059 | 0.029 | 1.000 | 0.000 | 97.000 | 0.000 | 0.000 | 0.000 | 2 |
| l2-global-student | 100 | 1.000 | 2.039 | 0.029 | 1.000 | 0.000 | 97.000 | 0.000 | 0.000 | 0.000 | 2 |
| l2-expert-bank | 100 | 1.000 | 1.820 | 0.053 | 1.000 | 0.000 | 92.000 | 5.000 | 0.000 | 0.000 | 2 |
| l3-disabled | 100 | 1.000 | 2.212 | 0.053 | 1.000 | 0.000 | 92.000 | 5.000 | 0.000 | 0.000 | 2 |
| l3-shadow | 100 | 1.000 | 1209.090 | 0.053 | 1.000 | 0.000 | 92.000 | 5.000 | 0.000 | 0.000 | 2 |
| l3-guarded | 100 | 0.970 | 1219.502 | 0.087 | 0.778 | 0.019 | 89.000 | 5.000 | 0.030 | 0.000 | 2 |

## Measured Versus Modeled

- The matrix is cache-backed. `serving_cost_per_100`, full L4 tokens, residual L4 tokens, and audit cost are zero because the oracle cache has no live token/cost usage.
- Full and residual L4 call counts are measured route outcomes over cached teacher responses.
- Non-L3 p95 latency is local replay/cache latency, not live L4 latency.
- L3 shadow and guarded latencies are measured local SLM latencies on `mps:0`.
- Promotion objectives include offline modeled residual/full L4 costs from the replay cost model. Those modeled values are not the same as measured live serving costs.
- Live teacher preflight passed, but no live residual measurement was run in this pass to avoid spending new L4 calls after the cache-backed matrix had succeeded.

## Conclusions

- The required post-refactor specs and suite wiring are present and tested in the NLU target, not core.
- `l2-expert-bank` and `main-evolution` reduced full L4 calls from `97/100` in `no-l2` and `l2-global-student` to `92/100`, with `5/100` residual L4 calls and no frame exact-match loss in the cache-backed matrix.
- Workload locality matters: `uniform` had no weak-field coverage and `100/100` full L4 calls; `zipf-heavy` reached weak coverage `0.053` and `5/100` residual calls.
- `no-guard` exposed the expected failure mode: full L4 calls dropped to `50/100`, but frame exact match fell to `0.590` and wrong accepted field rate rose to `0.262`.
- `l3-guarded` was runnable after benchmark preflight, but it hurt accuracy in this 100-request matrix: frame exact `0.970`, wrong accepted field rate `0.019`, L3 final share `0.030`, and p95 latency about `1.22s`.
- `l3-shadow` preserved frame exact match because it had no final routing authority, but it still added local generation latency in trace timing.
- No live cost, token, or residual latency savings should be claimed from this run.

## Blockers And Limits

- A full 3000-request all-variant matrix was not run. With L3 p95 generation around `1.2s` in the 100-request traces and `3.07s` in benchmark preflight, the L3 variants would dominate wall time.
- Live residual savings remain unmeasured. Credentials are present and preflight passes, but this pass stopped before spending new live L4 calls.
- Old pre-refactor layer-share-heavy experiments remain non-comparable to these results because the runtime now includes field-level patches, residual L4, override/conflict accounting, and field-aware objectives.
