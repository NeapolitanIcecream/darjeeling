# Post-Refactor Experiment Plan

Date: 2026-06-14

Inputs:

- `/Users/chenmohan/Downloads/Darjeeling-research-0614-1.md`
- `docs/design/07_gpt55_pro_0614_1_refactor_plan.md`
- `docs/experiments/2026-06-14_gpt55_pro_review_0614_1_verification.md`

Goal: make the post-refactor experiment results trustworthy and easy to compare. The runtime refactor is done; this plan covers the remaining experiment work: missing ablations, readiness checks, staged reruns, comparison reports, and final interpretation.

## Constraints

- Keep experiment logic in the NLU target CLI, settings, reports, and tests. Do not move NLU concepts into core.
- Keep the experiment system simple: explicit experiment specs, explicit CLI commands, explicit run directories.
- Do not add a scheduler, dashboard service, plugin system, or new experiment framework.
- Preserve reproducibility: record settings, teacher mode, data path, run directory, commit hash, and exact commands.
- Treat live L4 calls as expensive. Use preflight and cached smoke runs before any full live rerun.
- Keep old and new experiment results separate. Do not compare old layer-share-heavy results as if they measured the same runtime.

## Problems To Solve

1. Old end-to-end experiment results no longer represent the current system because routing, residual L4, field metrics, L1/L2 behavior, and objective scoring changed.
2. The current experiment suite does not clearly cover every required ablation:
   - no audit;
   - L2 global student versus L2 expert bank;
   - L3 disabled, shadow/skipped, and guarded/learned-gate behavior.
3. Reports now have the right metric families, but the experiment workflow must make those metrics first-class in comparisons and final writeups.
4. Live residual L4 benefit is still an empirical question. Offline modeled savings must not be presented as measured live savings.
5. Run directories and teacher cache reuse need to be intentional so reruns are reproducible and not polluted by stale artifacts.

## Plan

### 1. Add Missing Experiment Specs

Add only the experiment specs needed for the review matrix.

- `no-audit`: set `lower_layer_audit_mode="disabled"` while leaving the serving cascade otherwise unchanged.
- `l2-global-student`: run the existing global L2 student path without expert-bank wrapping, likely by setting `l2_expert_bank_enabled=False`.
- `l2-expert-bank`: make the expert-bank variant explicit, even if it matches the default, so comparisons are readable.
- `l3-disabled`: set `local_slm_mode="disabled"`.
- `l3-shadow`: set `local_slm_mode="shadow"` for observation without final routing authority.
- `l3-guarded`: set `local_slm_mode="guarded"` and rely on the existing preflight benchmark requirement.

Keep each spec as a small `ExperimentSpec` entry and add CLI dispatch/tests only where current generic suite handling is insufficient.

### 2. Define The Required Matrix

Run these as separate directories under one timestamped root:

- `main-evolution` on `zipf-heavy`.
- `workload-locality` over `uniform`, `zipf-mild`, and `zipf-heavy`.
- `no-guard`.
- `no-audit`.
- `no-l2`.
- `l2-global-student`.
- `l2-expert-bank`.
- `l3-disabled`.
- `l3-shadow`.
- `l3-guarded` only when L3 preflight passes.

Use the same `max_requests`, `compile_every`, teacher mode, processed data directory, and commit for all comparable runs unless the plan explicitly says otherwise.

### 3. Stage Runs By Cost And Risk

Do not jump straight to a full live suite.

1. **Preflight**: verify processed data, teacher cache or live teacher availability, L1 crate, and L3 readiness.
2. **Tiny smoke**: run a cached or live-or-cache suite with small request counts to verify command wiring, report generation, and comparison columns.
3. **Medium cached run**: run enough requests to exercise compiler generations and promotion records without spending new teacher calls.
4. **Full rerun**: run the required matrix with the intended request count and teacher mode after smoke/medium results are clean.
5. **Live residual measurement**: if live teacher is enabled, explicitly report measured full/residual L4 tokens, cost, and latency separately from modeled offline replay values.

### 4. Make Reports Answer The Right Questions

Each run and suite comparison must surface:

- frame exact match;
- weak field coverage and accuracy;
- wrong accepted field rate;
- L4 conflict/override rate;
- full L4 calls/tokens/cost/latency;
- residual L4 calls/tokens/cost/latency;
- serving cost per 100 requests;
- audit cost per 100 requests;
- correct weak fields avoiding full L4 per 100 requests;
- residual L4 verified fields per 100 requests;
- promotion count and rejection reasons;
- L3 observation and benchmark status when applicable.

The final experiment note should explain whether a value is measured live data, cache-backed serving data, or modeled offline replay data.

### 5. Protect Reproducibility

- Store each experiment under a fresh timestamped root such as `runs/post-refactor-YYYYMMDD-HHMM`.
- Write a `suite.json` with experiment names, request count, compile cadence, teacher mode, data directory, and parallelism.
- Record the commit hash in a short run note or suite metadata if the CLI does not already do it.
- Keep teacher caches deliberately: copy a baseline cache into the suite only when the experiment is intended to be cache-backed.
- Clear old runtime artifacts before rerun, but do not delete teacher caches accidentally.
- Keep generated run outputs under ignored `runs/`; commit only code, tests, docs, and any compact hand-written analysis docs.

### 6. Validate The Experiment Tooling

Add focused tests for new specs and suite behavior:

- settings overrides apply as expected;
- CLI commands or suite entries dispatch the right spec;
- preflight blocks `l3-guarded` when benchmark evidence is missing;
- comparison output includes the post-refactor metrics for synthetic traces;
- `no-audit` produces no audit cost in report metrics;
- `l2-global-student` and `l2-expert-bank` differ in settings metadata.

Run at minimum:

```bash
uv run pytest tests/targets/nlu/test_experiments.py tests/targets/nlu/test_experiment_suite_cli.py tests/targets/nlu/test_report_l3_summary.py tests/targets/nlu/test_settings.py -q
uv run pytest tests/test_core_contracts.py tests/test_target_boundary.py tests/targets/nlu/test_target_core_boundary.py -q
uv run pytest -q
```

### 7. Produce The Final Experiment Writeup

After the full matrix runs, write a concise results document under `docs/experiments/` with:

- exact command block;
- commit hash and settings summary;
- run matrix table;
- comparison table using the new field/residual metrics;
- explicit measured-versus-modeled note;
- conclusions about whether full L4 calls, p95 latency, cost, and frame accuracy improved;
- known limitations and which old experiments are no longer comparable.

## Done Criteria

- Required experiment specs exist and are tested.
- Preflight and smoke runs succeed before any expensive full rerun.
- The full matrix either runs successfully or documents a concrete blocker with logs and next action.
- Comparison reports include the post-refactor field, residual L4, serving cost, and audit cost metrics.
- Final writeup distinguishes measured live values from modeled offline values.
- No new generic framework or core NLU leakage is introduced.
