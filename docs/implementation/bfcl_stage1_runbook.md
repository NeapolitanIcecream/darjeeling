# BFCL Stage 1 Experiment Runbook

Date: 2026-06-30

This runbook is the execution plan for a local Codex agent to move the BFCL
experiment through Stage 1 and produce the first concrete wall-clock and LLM API
cost measurements.

The substantive context lives here so a fresh agent can execute the work without
depending on chat history.

## Objective

Build and run the BFCL Stage 1 pilot experiment for Darjeeling.

Stage 1 is complete when the repository has a small BFCL experiment harness and
one successful pilot run with:

- deterministic BFCL data sampling;
- cached L4/reference outputs with actual API cost and latency recorded;
- an interactive L1/L2-only candidate search loop;
- Core-written validation feedback over multiple candidates;
- a final held-out test report after the attempt closes;
- experiment actual and production counterfactual cost/latency ledgers;
- a concise final report with run path, timings, API spend, candidate counts,
  coverage, wrong accepts, fallback share, and final accuracy after fallback.

The point of Stage 1 is operational evidence. High local coverage is welcome but
not required. A low-coverage run is still useful if the harness, cache, feedback
loop, and ledgers are correct.

## Starting Context

Read these files first:

- `AGENTS.md`
- `README.md`
- `docs/implementation/bfcl_experiment_pre_report.md`
- `docs/implementation/reboot/interactive_compile_loop_runbook.md`
- `docs/design/reboot/00_overall_design.md`
- `docs/design/reboot/modules/01_target_definition.md`
- `docs/design/reboot/modules/02_snapshot_reference.md`
- `docs/design/reboot/modules/03_agent_workspace.md`
- `docs/design/reboot/modules/05_candidate_evaluation.md`
- `docs/design/reboot/modules/11_compile_orchestration.md`
- `tests/test_10_compile_orchestration.py`
- `tests/test_11_end_to_end_thin_target.py`

Relevant current implementation surfaces:

- `compile_orchestration.run_interactive_compile_loop`
- `compile_orchestration.start_compile_launch(..., launch_async=True)`
- `agent_workspace.launch_target_adaptation_agent_async`
- `agent_workspace.candidate_submission_ready`
- `agent_workspace.provide_validation_feedback`
- `candidate_evaluation.evaluate_candidate_on_validation`
- `candidate_evaluation.evaluate_candidate_on_test`
- `release_runtime.create_release_without_artifacts`

## Worktree And Branch

Run the experiment in its own branch and worktree.

Suggested setup from `/Users/chenmohan/gits/darjeeling`:

```bash
git status --short
git worktree add ../darjeeling-bfcl-stage1 -b codex/bfcl-stage1-pilot main
cd ../darjeeling-bfcl-stage1
uv sync --extra dev
```

If the pre-report or this runbook exists only as an uncommitted file in the main
checkout, copy it into the new worktree before starting or commit the docs first.
Do not delete the worktree after completion.

At completion, create a git commit for tracked repository changes unless the
main session explicitly says not to. Leave ignored run artifacts in place for
inspection.

## Artifact Policy

Use tracked code for the experiment harness and ignored paths for downloaded
data, caches, generated targets, workspaces, logs, and reports.

Recommended tracked paths:

```text
experiments/bfcl_stage1/
  README.md
  run_stage1.py
  bfcl_data.py
  bfcl_target.py
  l4_cache.py
  report.py
  simple_agent.py
  tests/
```

Recommended ignored run root:

```text
runs/bfcl-stage1/<YYYYMMDD-HHMMSS>/
  data/
  cache/
  target/
  snapshots/
  workspaces/
  artifacts/
  reports/
  logs/
  manifest.json
  cost_ledger.json
  final_report.md
```

`runs/` is ignored. A fresh git worktree will not contain prior run artifacts.
If a later agent needs an existing run, point it at the absolute run path and
validate manifest hashes before reusing it.

## Decision To Support

Stage 1 should answer:

- Can Darjeeling run a BFCL pilot through cached L4 plus interactive L1/L2
  candidate search without leaking validation/test rows?
- What actual API cost and wall-clock time did the pilot consume?
- What production counterfactual cost and latency would the resulting release
  imply for the same pilot traffic?
- Is the harness stable enough to justify Stage 2 static-core work?

Evidence supporting Stage 2:

- Stage 1 completes without manual row-level fixes.
- All cache keys and sample manifests are reproducible.
- Final report separates validation-search metrics from final test metrics.
- Wrong accepts and fallback share are clearly measured.
- Actual API cost and wall-clock time are within the configured budget.

Evidence against Stage 2:

- The adapter cannot produce stable reference outputs or parse model outputs.
- Validation/test leakage is detected.
- The experiment cannot separate cached actual latency from production
  counterfactual latency.
- Candidate evaluation is too slow or too flaky for the 500-800 case pilot.

## Budget And Stop Rules

No budget has been user-approved in this file. Use these conservative defaults
unless the main session gives different numbers before paid calls:

- hard actual L4 API cache-build cap: `$25`;
- hard experiment-agent API cap: `$25` if the target adaptation command makes
  explicit provider API calls;
- hard wall-clock cap for the complete Stage 1 run: 6 hours;
- max interactive candidates in the pilot attempt: 5;
- max target-adaptation agent wall time: 60 minutes;
- max failed candidate evaluations before stopping: 3.

Do not count the human-facing/local Codex orchestration session as experiment
API cost. Count only API/model calls launched inside the Darjeeling experiment
path, such as L4 cache-building calls and any explicit target-adaptation agent
provider calls that write to `journal/agent_usage.json`.

If no provider API key is available, complete the no-network harness, dry-run
cache mode, and sample manifest work, then stop before paid Stage 1 and report
the missing credential as the blocker.

## Stage 1 Sample

Use a deterministic 600-case pilot from these public BFCL files:

| Category | Sample count |
| --- | ---: |
| `BFCL_v3_simple.json` | 160 |
| `BFCL_v3_multiple.json` | 100 |
| `BFCL_v3_parallel.json` | 100 |
| `BFCL_v3_parallel_multiple.json` | 100 |
| `BFCL_v3_irrelevance.json` | 140 |

Use stable per-category sampling:

- seed string: `darjeeling-bfcl-stage1-20260630`;
- sort candidate rows by stable hash of `{dataset_revision, category, row_id,
  seed}`;
- take the first `sample_count` rows per category.

Split the 600 rows stratified by category:

- train: 50% (300 rows);
- validation: 25% (150 rows);
- test: 25% (150 rows).

Represent split eligibility explicitly in generated `data.yaml` records:

- train rows: `["train"]`;
- validation rows: `["validation_candidate"]`;
- test rows: `["test_candidate"]`.

Cache may contain all 600 L4 outputs, but train/validation/test visibility must
follow Darjeeling boundaries. The agent may see train rows and aggregate
validation feedback only. It must not see raw validation/test rows, expected
outputs, row ids, or cached L4 raw outputs for validation/test rows.

## BFCL Target Shape

Keep BFCL-specific logic outside Darjeeling core. Put it under the experiment
harness.

Input schema should be target-owned and close to BFCL:

```json
{
  "type": "object",
  "required": ["row_id", "category", "question", "functions"],
  "properties": {
    "row_id": {"type": "string"},
    "category": {"type": "string"},
    "question": {"type": "array"},
    "functions": {"type": "array"}
  }
}
```

Output schema should normalize tool-call behavior:

```json
{
  "type": "object",
  "required": ["calls"],
  "properties": {
    "calls": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["name", "arguments"],
        "properties": {
          "name": {"type": "string"},
          "arguments": {"type": "object"}
        }
      }
    }
  }
}
```

Use `{"calls": []}` for no-call/irrelevance expected outputs.

Contract guidance:

- `validate_input` and `validate_output` should be schema-light and deterministic.
- `normalize_input` should include category, row id, normalized question text,
  and function schema digest, but row ids must not appear in agent-visible
  feedback.
- `split_group` should keep each row independent for Stage 1.
- `slice_tags` should include category and simple shape tags such as
  `call_count:zero`, `call_count:one`, `call_count:many` when available from the
  reference output.
- `is_correct` should compare normalized call names and arguments. For Stage 1,
  use exact normalized comparison and document limitations. Do not introduce a
  broad BFCL evaluator abstraction unless exact comparison is clearly wrong for
  a selected category.
- `redact_for_trace` should remove raw question text and keep category, input
  hash, function schema hash, and row-shape metadata.

For function-call categories, load gold outputs from
`possible_answer/<category-file>` when available. For `BFCL_v3_irrelevance.json`,
use `{"calls": []}` as gold.

If a selected row lacks parseable gold output, exclude it from Stage 1 and record
the exclusion in `manifest.json`.

## L4 Cache

Implement a Core-owned cached reference broker for Stage 1.

Cache key fields:

- dataset id and revision;
- row id;
- category;
- normalized question payload hash;
- function schema hash;
- prompt/template hash;
- model id;
- decoding/tool-call configuration hash;
- output schema version.

Cache record fields:

- cache key;
- row id and category;
- parsed normalized output;
- raw provider output or a redacted replay payload;
- prompt hash and function schema hash;
- model id and version;
- usage tokens;
- actual API cost;
- observed provider latency;
- finish reason;
- parse/schema status;
- error metadata.

Ledgers:

- `actual_api_cost`: real cost paid to build or refresh cache entries;
- `actual_api_latency_ms`: real latency observed during cache build;
- `cache_hit_cost`: zero for repeated evaluation runs;
- `production_counterfactual_l4_cost`: estimated cost if the same fallback call
  happened live in production;
- `production_counterfactual_l4_latency_ms`: cached observed latency or a clearly
  labeled estimate.

Run modes:

- `--cache-mode dry-run`: do not call the provider; validate keys and estimated
  token/cost counts only.
- `--cache-mode build-missing`: call provider only for missing cache entries,
  bounded by the cost cap.
- `--cache-mode reuse`: fail if required cache entries are missing.

For the first Stage 1 completion, `build-missing` is preferred so actual API
cost and observed latency are measured.

## Interactive Candidate Search

Stage 1 should use the interactive compile loop.

Use L1/L2 only:

- candidate routing must set `enabled_layers: ["L1", "L2"]`;
- do not submit L3 artifacts;
- reports should mark L3 as disabled, not zero coverage.

The target adaptation command can initially be a simple local script under
`experiments/bfcl_stage1/simple_agent.py`. It should:

1. Read the agent brief and train view.
2. Generate candidate `c1` under `submissions/c1/artifacts/l1` or `l2`.
3. Write `submissions/c1/READY` only after all files are complete.
4. Watch for `journal/feedback-c1.json`.
5. Generate at least one improved or deliberately different candidate `c2`.
6. Optionally continue to `c3` through `c5`.
7. Write `journal/agent_usage.json` if it makes explicit provider API calls.

The simple agent does not need to be clever. A deterministic train-derived
artifact is acceptable for the first Stage 1 run if it exercises:

- valid artifact packaging;
- accept/abstain behavior;
- multiple candidate submissions;
- validation feedback delivery;
- final test evaluation.

Do not directly edit generated L1/L2 workspace artifacts from the orchestration
session after the run starts. Change the tracked harness, prompt, simple agent,
or target contract instead.

## Required Runner

Add one command that can perform the full pilot from a checkout.

Suggested interface:

```bash
uv run python -m experiments.bfcl_stage1.run_stage1 \
  --run-root runs/bfcl-stage1 \
  --sample-size stage1 \
  --cache-mode build-missing \
  --max-actual-api-cost 25 \
  --max-agent-api-cost 25 \
  --max-candidates 5 \
  --max-agent-seconds 3600 \
  --l3 disabled
```

Also support:

```bash
uv run python -m experiments.bfcl_stage1.run_stage1 \
  --run-root runs/bfcl-stage1 \
  --sample-size stage1 \
  --cache-mode dry-run \
  --l3 disabled
```

The runner should write:

- generated target path;
- dataset revision;
- selected row ids by category and split;
- cache summary;
- compile loop result;
- validation report summary;
- final test report summary;
- cost ledger;
- final markdown report.

## Report Requirements

`final_report.md` must include:

- run id and absolute run path;
- git branch and commit;
- start/end timestamps and timezone;
- dataset id, dataset revision, and selected categories;
- sample counts by category and split;
- provider/model configuration used for L4 cache;
- cache mode, cache hit count, build-missing count, actual API cost, and observed
  L4 latency;
- production counterfactual all-L4 cost and latency;
- production counterfactual Darjeeling cost and latency after fallback;
- compile loop wall time;
- candidate count and feedback count;
- stop reason;
- agent usage cost, or a clear statement that no explicit target-adaptation API
  calls were made;
- accepted count, correct accepts, wrong accepts, precision, coverage, fallback
  share, final accuracy after fallback, and baseline L4 accuracy;
- metrics by category;
- validation/test separation statement;
- L3 disabled statement;
- known limitations and recommended Stage 2 next step.

`cost_ledger.json` must include machine-readable fields for:

- `experiment_actual.cache_build_api_cost`;
- `experiment_actual.target_agent_api_cost`;
- `experiment_actual.local_eval_wall_seconds`;
- `experiment_actual.total_wall_seconds`;
- `production_counterfactual.baseline_all_l4_cost`;
- `production_counterfactual.darjeeling_serving_cost`;
- `production_counterfactual.savings_per_1000_requests`;
- `production_counterfactual.estimated_payback_requests`;
- `notes`.

## Tests

Add focused tests for the harness. Prefer small fixtures and dry-run modes.

Minimum tests:

```bash
uv run --with pytest pytest tests -q
uv run --with ruff ruff check src tests experiments
uv run python -m experiments.bfcl_stage1.run_stage1 \
  --run-root runs/bfcl-stage1-test \
  --sample-size tiny \
  --cache-mode dry-run \
  --max-candidates 2 \
  --max-agent-seconds 120 \
  --l3 disabled
```

Harness-specific tests should verify:

- deterministic sampling;
- no validation/test raw rows in agent-visible feedback;
- candidate routing disables L3;
- `READY` marker is required before evaluation;
- dry-run cost ledger shape;
- generated target passes `uv run darjeeling target check <target>`;
- tiny run produces a final report.

Run the full repository tests before committing unless runtime is prohibitive. If
full tests cannot run, record exactly which tests were run and why full tests
were skipped.

## Done Criteria

Stage 1 is done when all are true:

- tracked BFCL Stage 1 harness exists;
- deterministic 600-case sample manifest is produced in the run;
- generated target passes `darjeeling target check`;
- L4 cache is built or reused for all selected Stage 1 rows;
- actual API cost and observed cache-build latency are recorded;
- one interactive compile attempt evaluates at least two candidates and writes
  at least two validation feedback files;
- candidate routing is L1/L2 only;
- final test evaluation runs only after the attempt closes;
- final report and JSON cost ledger are written under the run root;
- test and lint commands above pass, or failures are explained with logs;
- tracked changes are committed;
- final handoff reports branch, worktree path, commit hash, run path, API cost,
  wall time, and whether the worktree is clean.

## If Blocked

Work autonomously through implementation and debugging. Stop for the main
session only for:

- provider credentials are unavailable;
- the paid API budget would be exceeded;
- the selected model/provider must change;
- the experiment cannot meet the no-leakage boundary without changing core
  architecture;
- the run takes longer than the wall-clock cap.

If blocked, write `runs/bfcl-stage1/<run-id>/blocked_report.md` with:

- blocking condition;
- completed work;
- partial costs and wall time;
- logs needed for diagnosis;
- next concrete step.

## Completion Handoff

The executing agent's final message should be short and include:

- branch;
- worktree path;
- commit hash;
- run path;
- actual API cost;
- target-agent API cost, if any;
- total wall time;
- final test accuracy, local coverage, wrong accepts, fallback share;
- whether the worktree is clean;
- commands run.

Do not delete the worktree, branch, or ignored run artifacts.

## Short Prompt For A Fresh Local Codex Agent

Read `docs/implementation/bfcl_stage1_runbook.md` and execute it in a new
branch/worktree. Your objective is to implement the minimal BFCL Stage 1 harness,
run the 600-case pilot with cached L4 and L1/L2-only interactive candidate
search, and produce the final report plus cost ledger. Keep Darjeeling core
target-independent, keep BFCL logic in the experiment harness, do not expose raw
validation/test rows to the target adaptation agent, and commit tracked changes
when done.
