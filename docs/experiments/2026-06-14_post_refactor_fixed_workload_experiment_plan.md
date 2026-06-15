# Post-Refactor Fixed Workload Experiment Plan

Date: 2026-06-14

Purpose: produce a reproducible post-refactor experiment package with a fixed amount
of work. This plan intentionally does not define a reduced acceptable workload. The
agent executing it must run the exact matrix below, fix failures that prevent the
matrix from running, and rerun the same commands until the stated artifacts exist.

## Non-Negotiable Constraints

- Keep all experiment wiring in the NLU target, NLU CLI, NLU reports, NLU tests, and
  experiment docs.
- Do not move NLU frame, intent, slot, utterance, MASSIVE, or failure-case logic into
  Darjeeling core.
- Do not add a scheduler, dashboard, plugin framework, dependency-injection system,
  schema DSL, or new experiment framework.
- Do not edit generated L1/L2/L3 workspaces directly. Change repo-level harnesses,
  prompts, contracts, tests, and adapters instead.
- Use fresh run roots. Do not merge new results into previous run directories.
- Do not compare pre-refactor result numbers as if they measured the current runtime.
- Do not claim live cost, token, or latency savings from cache-backed runs.
- Do not reduce request counts, remove variants, skip L3 variants, or switch teacher
  mode unless this plan is edited by the user.

## Fixed Workload

Run exactly these workloads.

### A. Repository Stabilization

1. Make the experiment wiring reproducible from a commit before final experiment
   runs begin.
2. Run:

   ```bash
   uv run pytest tests/targets/nlu/test_experiments.py tests/targets/nlu/test_experiment_suite_cli.py tests/targets/nlu/test_report_l3_summary.py tests/targets/nlu/test_settings.py -q
   uv run pytest tests/test_core_contracts.py tests/test_target_boundary.py tests/targets/nlu/test_target_core_boundary.py -q
   uv run pytest -q
   git diff --check
   ```

3. Commit the experiment wiring, tests, and this plan before running the final cache
   and live matrices. Record the resulting commit hash in the final writeup.

### B. Preflight Package

Use one timestamped root:

```bash
RUN_ROOT=runs/post-refactor-fixed-YYYYMMDD-HHMM
DATA_DIR=data/processed/massive_en_us
```

Run all four commands:

```bash
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

Required artifacts:

- `$RUN_ROOT/preflight-cache.json`
- `$RUN_ROOT/preflight-l3-guarded/reports/l3_benchmark.json`
- `$RUN_ROOT/preflight-l3-guarded.json`
- `$RUN_ROOT/preflight-live.json`

Each JSON preflight artifact must have `"status": "pass"`.

### C. Full Cache-Backed Matrix

Run exactly one 3000-request cache-backed suite:

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

This suite must produce exactly these comparable trace sets:

- `main-evolution`, `zipf-heavy`, 3000 requests.
- `workload-locality`, `uniform`, 3000 requests.
- `workload-locality`, `zipf-mild`, 3000 requests.
- `workload-locality`, `zipf-heavy`, 3000 requests.
- `no-guard`, `zipf-heavy`, 3000 requests.
- `no-audit`, `zipf-heavy`, 3000 requests.
- `no-l2`, `zipf-heavy`, 3000 requests.
- `l2-global-student`, `zipf-heavy`, 3000 requests.
- `l2-expert-bank`, `zipf-heavy`, 3000 requests.
- `l3-disabled`, `zipf-heavy`, 3000 requests.
- `l3-shadow`, `zipf-heavy`, 3000 requests.
- `l3-guarded`, `zipf-heavy`, 3000 requests.

Total required cache-backed serving traces: 36,000.

Required artifacts:

- `$RUN_ROOT/cache-full/suite.json`
- `$RUN_ROOT/cache-full/results.json`
- `$RUN_ROOT/cache-full/comparison/comparison.csv`
- `$RUN_ROOT/cache-full/comparison/comparison.html`
- `traces.jsonl`, `settings.json`, `reports/quality.json`, and `reports/promotions.jsonl`
  for every trace set listed above.

The comparison CSV must include these columns:

- `frame_exact_match`
- `weak_field_coverage`
- `weak_field_accuracy`
- `wrong_accepted_field_rate`
- `l4_conflict_rate`
- `full_l4_calls_per_100`
- `residual_l4_calls_per_100`
- `full_l4_tokens_per_100`
- `residual_l4_tokens_per_100`
- `serving_cost_per_100`
- `audit_cost_per_100`
- `correct_weak_fields_avoiding_full_l4_per_100`
- `residual_l4_verified_fields_per_100`
- `l3_share`
- `promoted_generations`
- `promotion_attempts`
- `promoted_with_layer_regression`

### D. Fixed Live Residual Measurement

Run exactly one live suite with four variants and 500 requests per variant:

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

Before the first live run, ensure the four live run directories do not already
contain `teacher_cache.jsonl`. The live suite must use fresh run directories under
`$RUN_ROOT/live-residual-500`.

This suite must produce exactly these trace sets:

- `no-l2`, `zipf-heavy`, 500 requests.
- `l2-global-student`, `zipf-heavy`, 500 requests.
- `l2-expert-bank`, `zipf-heavy`, 500 requests.
- `main-evolution`, `zipf-heavy`, 500 requests.

Total required live serving traces: 2,000.

Required artifacts:

- `$RUN_ROOT/live-residual-500/suite.json`
- `$RUN_ROOT/live-residual-500/results.json`
- `$RUN_ROOT/live-residual-500/comparison/comparison.csv`
- `$RUN_ROOT/live-residual-500/comparison/comparison.html`
- `traces.jsonl`, `settings.json`, `teacher_cache.jsonl`, `reports/quality.json`,
  and `reports/promotions.jsonl` for all four variants.

The final writeup must report live measured values from this suite separately from
cache-backed route counts.

### E. Post-Run Verification

Run all checks after both matrices finish:

```bash
uv run pytest tests/targets/nlu/test_experiments.py tests/targets/nlu/test_experiment_suite_cli.py tests/targets/nlu/test_report_l3_summary.py tests/targets/nlu/test_settings.py -q
uv run pytest tests/test_core_contracts.py tests/test_target_boundary.py tests/targets/nlu/test_target_core_boundary.py -q
uv run pytest -q
git diff --check
```

Add a small verification script or command block in the final writeup that proves:

- the cache matrix has 12 trace sets;
- each cache trace set has exactly 3000 records;
- the live matrix has 4 trace sets;
- each live trace set has exactly 500 records;
- all required comparison columns are present;
- all preflight JSON artifacts have `"status": "pass"`;
- the suite metadata commit hash matches the committed code used for the run.

### F. Final Writeup

Create:

```text
docs/experiments/2026-06-14_post_refactor_fixed_workload_results.md
```

The writeup must include:

- exact commit hash;
- `git status --short` after the run;
- exact run root;
- exact commands executed;
- preflight summary;
- cache matrix table with all 12 trace sets;
- live residual table with all 4 trace sets;
- measured live cost/token/latency section;
- cache-backed route-count section;
- L3 benchmark and L3 guarded/shadow interpretation;
- explicit statement that cache-backed zero cost is not live measured cost;
- conclusions for accuracy, full L4 call reduction, residual L4 behavior, p95
  latency, live cost, audit cost, and L3 usefulness;
- list of any bugs fixed while executing the plan, with test names covering them.

## Failure Handling

If a command fails, the agent must fix the underlying repo, environment, or data
problem and rerun the same command with the same workload. The agent must not replace
the failed workload with a smaller run, a cache-only substitute for the live suite, a
non-L3 substitute for the L3 variants, or a manually edited comparison table.

If an external service outage or quota failure blocks the live suite, the plan is not
complete. Preserve the failed run directory, record the exact error in the results
draft, and stop without marking the fixed workload complete.
