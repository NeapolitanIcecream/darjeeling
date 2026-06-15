# Post-Refactor Fixed Workload Results

Date: 2026-06-15

Run root:

```text
runs/post-refactor-fixed-20260615-0954
```

Code commits used:

- Final repo HEAD after harness/report fixes: `5316a922d44c92ea13c78118b92daa3ef317e98e`
- Cache trace suite metadata commit: `5850281df0b400dd26804a2fab0d1161b82c8c37`
- Live trace suite metadata commit: `f028add9a1b553e32a9897792bfed3981976e530`

The cache suite was completed before the live recovery harness was added. The live suite then used the committed resume harness to preserve verified trace prefixes after transient live teacher empty-content failures. The final report-only artifacts were generated from `5316a922d44c92ea13c78118b92daa3ef317e98e`. Trace counts and comparison data below come from the fixed workload run root, not from old pre-refactor results.

`git status --short` captured after the run and before creating this results document:

```text
?? docs/experiments/2026-06-14_post_refactor_experiment_plan.md
?? docs/experiments/2026-06-14_post_refactor_results.md
```

## Commands

Preflight:

```bash
RUN_ROOT=runs/post-refactor-fixed-20260615-0954
DATA_DIR=data/processed/massive_en_us

uv run edge-mvp-nlu experiment preflight \
  --run-dir "$RUN_ROOT/preflight-cache" \
  --data-dir "$DATA_DIR" \
  --teacher cache \
  --check-l1-build \
  --out "$RUN_ROOT/preflight-cache.json"

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run edge-mvp-nlu l3 bench \
  --mode guarded \
  --data-dir "$DATA_DIR" \
  --out "$RUN_ROOT/preflight-l3-guarded/reports/l3_benchmark.json" \
  --fail-on-error

uv run edge-mvp-nlu experiment preflight \
  --experiment l3-guarded \
  --run-dir "$RUN_ROOT/preflight-l3-guarded" \
  --data-dir "$DATA_DIR" \
  --teacher cache \
  --out "$RUN_ROOT/preflight-l3-guarded.json"

uv run edge-mvp-nlu experiment preflight \
  --run-dir "$RUN_ROOT/preflight-live" \
  --data-dir "$DATA_DIR" \
  --teacher live \
  --out "$RUN_ROOT/preflight-live.json"
```

Cache-full suite:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run edge-mvp-nlu experiment suite \
  --run-root "$RUN_ROOT/cache-full" \
  --include-guarded-l3 \
  --max-requests 3000 \
  --compile-every 500 \
  --teacher cache \
  --data-dir "$DATA_DIR" \
  --parallel 1
```

Live suite initial command:

```bash
uv run edge-mvp-nlu experiment suite \
  --run-root "$RUN_ROOT/live-residual-500" \
  --experiment no-l2 \
  --experiment l2-global-student \
  --experiment l2-expert-bank \
  --experiment main-evolution \
  --max-requests 500 \
  --compile-every 100 \
  --teacher live \
  --data-dir "$DATA_DIR" \
  --parallel 1
```

The initial live command failed with transient live teacher/API failures, including `TeacherParseError: teacher response content is empty` and SSL/API connection errors. The command was retried once without resume and then stopped after the first variant failed again, to avoid overwriting more usable progress. Before the first live run, the live run directories did not contain `teacher_cache.jsonl`.

Live recovery command, repeated until all four variants reached 500 traces:

```bash
uv run edge-mvp-nlu experiment suite \
  --run-root "$RUN_ROOT/live-residual-500" \
  --experiment no-l2 \
  --experiment l2-global-student \
  --experiment l2-expert-bank \
  --experiment main-evolution \
  --max-requests 500 \
  --compile-every 100 \
  --teacher live \
  --data-dir "$DATA_DIR" \
  --parallel 1 \
  --resume-existing
```

Report artifact regeneration after adding `quality.json` and `promotions.jsonl`:

```bash
uv run python - <<'PY'
import csv
from pathlib import Path
from darjeeling.targets.nlu.reports import generate_run_report

run_root = Path("runs/post-refactor-fixed-20260615-0954")
run_dirs = []
for suite in ["cache-full", "live-residual-500"]:
    with (run_root / suite / "comparison" / "comparison.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            run_dirs.append(Path(row["run_dir"]))
for run_dir in run_dirs:
    generate_run_report(run_dir)
print(f"regenerated_reports={len(run_dirs)}")
PY
```

Post-run tests:

```bash
uv run pytest tests/targets/nlu/test_experiments.py tests/targets/nlu/test_experiment_suite_cli.py tests/targets/nlu/test_report_l3_summary.py tests/targets/nlu/test_settings.py -q
uv run pytest tests/test_core_contracts.py tests/test_target_boundary.py tests/targets/nlu/test_target_core_boundary.py -q
uv run pytest -q
git diff --check
```

## Preflight

| artifact | result |
|---|---:|
| `preflight-cache.json` | `status=pass` |
| `preflight-l3-guarded/reports/l3_benchmark.json` | `status=success`, `requests=3` |
| `preflight-l3-guarded.json` | `status=pass` |
| `preflight-live.json` | `status=pass` |

L3 guarded benchmark:

- Backend: `Qwen/Qwen2.5-0.5B-Instruct` on `mps:0`
- Requests: 3
- Failures: 0
- Parse failures: 0
- Accepted: 0
- Would accept: 0
- Repair count: 2
- Generation p50: 1295.892 ms
- Generation p95: 4414.115 ms
- Throughput: 0.4145 qps

## Cache-Full Matrix

Cache-full produced 12 trace sets, each with 3000 traces, for 36,000 cache-backed serving traces.

| experiment | stream | requests | FEM | p95 ms | full L4/100 | residual L4/100 | L0 | L1 | L2 | L3 | L4 | L3 share | promotions | attempts | regressions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no-l2 | zipf-heavy | 3000 | 1 | 0.031 | 49.100 | 0.000 | 1527 | 0 | 0 | 0 | 1473 | 0.000 | 6 | 6 | 0 |
| l2-global-student | zipf-heavy | 3000 | 0.998 | 2.962 | 48.667 | 0.000 | 1527 | 0 | 13 | 0 | 1460 | 0.000 | 6 | 6 | 0 |
| no-audit | zipf-heavy | 3000 | 0.998 | 2.805 | 45.300 | 3.367 | 1527 | 0 | 13 | 0 | 1460 | 0.000 | 6 | 6 | 0 |
| main-evolution | zipf-heavy | 3000 | 0.998 | 2.952 | 45.300 | 3.367 | 1527 | 0 | 13 | 0 | 1460 | 0.000 | 6 | 6 | 0 |
| l3-disabled | zipf-heavy | 3000 | 0.998 | 3.001 | 45.300 | 3.367 | 1527 | 0 | 13 | 0 | 1460 | 0.000 | 6 | 6 | 0 |
| l2-expert-bank | zipf-heavy | 3000 | 0.998 | 3.032 | 45.300 | 3.367 | 1527 | 0 | 13 | 0 | 1460 | 0.000 | 6 | 6 | 0 |
| workload-locality | zipf-heavy | 3000 | 0.998 | 3.081 | 45.300 | 3.367 | 1527 | 0 | 13 | 0 | 1460 | 0.000 | 6 | 6 | 0 |
| workload-locality | zipf-mild | 3000 | 0.999667 | 2.792 | 86.667 | 2.400 | 328 | 0 | 0 | 0 | 2672 | 0.000 | 1 | 2 | 0 |
| workload-locality | uniform | 3000 | 0.999667 | 4.060 | 92.800 | 1.033 | 185 | 0 | 0 | 0 | 2815 | 0.000 | 2 | 3 | 0 |
| l3-shadow | zipf-heavy | 3000 | 0.998 | 1085.915 | 54.267 | 2.300 | 1290 | 0 | 13 | 0 | 1697 | 0.000 | 5 | 5 | 0 |
| l3-guarded | zipf-heavy | 3000 | 0.971 | 984.976 | 43.000 | 2.767 | 1527 | 0 | 13 | 87 | 1373 | 0.029 | 6 | 6 | 0 |
| no-guard | zipf-heavy | 3000 | 0.743667 | 2.847 | 16.667 | 3.367 | 1527 | 0 | 872 | 0 | 601 | 0.000 | 6 | 6 | 0 |

Cache-backed cost columns are zero because this suite used the cache teacher. These are route-count and replay metrics, not live measured cost.

## Live Residual 500

Live residual produced 4 trace sets, each with 500 traces, for 2,000 live serving traces.

| experiment | requests | FEM | p95 ms | full L4/100 | tokens/100 | cost/100 USD | full calls | total tokens | total cost USD | full p50 ms | full p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| main-evolution | 500 | 0.714 | 4434.080 | 100.000 | 72100.0 | 0.038814 | 500 | 360500 | 0.194072 | 3050.450 | 4433.868 |
| no-l2 | 500 | 0.720 | 4562.357 | 100.000 | 72024.2 | 0.038693 | 500 | 360121 | 0.193466 | 3027.502 | 4562.208 |
| l2-expert-bank | 500 | 0.696 | 5523.434 | 100.000 | 72182.4 | 0.038946 | 500 | 360912 | 0.194731 | 3113.918 | 5523.309 |
| l2-global-student | 500 | 0.704 | 8277.282 | 100.000 | 72105.6 | 0.038823 | 500 | 360528 | 0.194117 | 3106.141 | 8277.066 |

Aggregate live measurement:

- Requests: 2,000
- Full live L4 calls: 2,000
- Residual live L4 calls: 0
- Full live tokens: 1,442,061
- Residual live tokens: 0
- Full live cost: 0.7763856 USD
- Residual live cost: 0.0 USD

Live teacher cache line counts after recovery:

| experiment | `teacher_cache.jsonl` lines |
|---|---:|
| no-l2 | 681 |
| l2-global-student | 526 |
| l2-expert-bank | 500 |
| main-evolution | 500 |

The extra cache lines in `no-l2` and `l2-global-student` are from interrupted live calls before trace-prefix resume was added. Live mode did not use those cache lines to answer requests; final trace counts are from `traces.jsonl`.

## Cache-Backed Route Counts

Main cache-backed `main-evolution` on zipf-heavy routed 1527 requests to L0, 13 to L2, and 1460 to L4. It reduced full L4 calls from `no-l2` 49.1/100 to 45.3/100 and added 3.367 residual L4 calls/100 in the cache-backed replay. The `no-guard` ablation accepted many more L2 results, 872 L2 chosen traces, but frame exact match dropped to 0.743667 and wrong accepted field rate rose to 0.157845. This supports keeping the guard.

The locality rows show the expected workload effect: uniform and zipf-mild had higher full L4 rates than zipf-heavy because fewer requests hit repeated cacheable/local patterns.

## L3 Interpretation

L3 was successfully preflighted and benchmarked, but it was not useful in this workload:

- `l3-shadow` added high latency, p95 1085.915 ms, without accepting L3 outputs.
- `l3-guarded` accepted 87/3000 requests, L3 share 0.029, but frame exact match fell from 0.998 in `main-evolution` to 0.971 and p95 latency remained high at 984.976 ms.
- The guarded bench accepted/would-accept 0/3 smoke requests.

Conclusion: post-refactor L3 wiring is operational, but the current guarded L3 configuration is not a good serving path for this fixed workload. It should remain guarded or shadow-only until calibration improves.

## Verification Output

Artifact/count/column verification:

```text
preflight-cache.json pass
preflight-l3-guarded.json pass
preflight-live.json pass
preflight-l3-guarded/reports/l3_benchmark.json success requests 3
cache-full rows 12 missing_cols []
cache-full no-l2 zipf-heavy 3000 artifacts_missing []
cache-full l2-global-student zipf-heavy 3000 artifacts_missing []
cache-full no-audit zipf-heavy 3000 artifacts_missing []
cache-full main-evolution zipf-heavy 3000 artifacts_missing []
cache-full l3-disabled zipf-heavy 3000 artifacts_missing []
cache-full l2-expert-bank zipf-heavy 3000 artifacts_missing []
cache-full workload-locality zipf-heavy 3000 artifacts_missing []
cache-full workload-locality zipf-mild 3000 artifacts_missing []
cache-full workload-locality uniform 3000 artifacts_missing []
cache-full l3-shadow zipf-heavy 3000 artifacts_missing []
cache-full l3-guarded zipf-heavy 3000 artifacts_missing []
cache-full no-guard zipf-heavy 3000 artifacts_missing []
cache-full commit 5850281df0b4 resume_existing None
live-residual-500 rows 4 missing_cols []
live-residual-500 main-evolution zipf-heavy 500 artifacts_missing []
live-residual-500 no-l2 zipf-heavy 500 artifacts_missing []
live-residual-500 l2-expert-bank zipf-heavy 500 artifacts_missing []
live-residual-500 l2-global-student zipf-heavy 500 artifacts_missing []
live-residual-500 commit f028add9a1b5 resume_existing True
```

The required comparison columns were present in both comparison CSV files:

```text
frame_exact_match, weak_field_coverage, weak_field_accuracy,
wrong_accepted_field_rate, l4_conflict_rate, full_l4_calls_per_100,
residual_l4_calls_per_100, full_l4_tokens_per_100,
residual_l4_tokens_per_100, serving_cost_per_100, audit_cost_per_100,
correct_weak_fields_avoiding_full_l4_per_100,
residual_l4_verified_fields_per_100, l3_share, promoted_generations,
promotion_attempts, promoted_with_layer_regression
```

Test results after the final report-artifact fix:

```text
uv run pytest tests/targets/nlu/test_experiments.py tests/targets/nlu/test_experiment_suite_cli.py tests/targets/nlu/test_report_l3_summary.py tests/targets/nlu/test_settings.py -q
52 passed in 1.60s

uv run pytest tests/test_core_contracts.py tests/test_target_boundary.py tests/targets/nlu/test_target_core_boundary.py -q
16 passed in 1.24s

uv run pytest -q
268 passed in 46.70s

git diff --check
passed
```

## Bugs Fixed During Execution

| commit | fix | tests |
|---|---|---|
| `ad1778e78d4a0f6ef15a7cdd163d0ff8e9779b05` | Seed fresh cache-backed experiment run directories from `runs/oracle-cache-massive/teacher_cache.jsonl`; keep live teacher runs unseeded. | `test_experiment_preflight_seeds_cache_teacher_for_fresh_run_dir`, `test_experiment_preflight_live_teacher_does_not_seed_cache`, `test_execute_replay_run_seeds_cache_teacher_for_fresh_run_dir`, `test_load_settings_uses_default_settings_yaml_from_cwd` |
| `5850281df0b400dd26804a2fab0d1161b82c8c37` | Write `results.json` for experiment suites before reporting failures, so failed suites remain auditable. | `test_experiment_suite_writes_results_json_before_reporting_failures`, `test_experiment_suite_builds_parallel_subprocess_plan` |
| `f028add9a1b553e32a9897792bfed3981976e530` | Add trace-prefix resume for interrupted live suites, with prefix request/utterance validation. | `test_run_replay_resume_existing_appends_missing_stream_suffix`, `test_run_replay_resume_existing_rejects_mismatched_prefix`, `test_experiment_suite_resume_existing_passes_resume_env` |
| `5316a922d44c92ea13c78118b92daa3ef317e98e` | Add required `reports/quality.json` and `reports/promotions.jsonl` artifacts to run reports. | `test_generate_run_report_writes_summary_metrics_artifacts_and_curves` |

## Conclusions

- Accuracy: cache-backed `main-evolution` reached 0.998 frame exact match on zipf-heavy; live `main-evolution` measured 0.714 because live teacher outputs differed from gold labels in this run.
- Full L4 reduction: cache-backed `main-evolution` reduced full L4 calls from `no-l2` 49.1/100 to 45.3/100 on zipf-heavy, with 3.367 residual L4 calls/100.
- Residual behavior: cache-backed residual L4 verified 3.067 fields/100 for `main-evolution`; live residual measurement did not exercise residual calls because all 2,000 live requests went to full L4.
- Latency: cache-backed non-L3 variants stayed near 3 ms p95. Live p95 ranged from 4434.080 ms to 8277.282 ms. L3 variants were much slower in cache replay due local SLM generation.
- Live cost: the fixed live suite measured 1,442,061 full L4 tokens and 0.7763856 USD total cost over 2,000 requests.
- Audit cost: audit cost remained 0.0 in the reported live suite and cache comparison.
- L3 usefulness: L3 is wired and benchmarked, but guarded L3 reduced accuracy and increased latency in this workload; it is not ready as a beneficial serving path under this configuration.
