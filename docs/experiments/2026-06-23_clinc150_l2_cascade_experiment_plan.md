# CLINC150 L2 And Cascade Experiment Plan

Date: 2026-06-23

Purpose: run the next CLINC150 Phase 1 mechanism experiment now that the repaired
live L4 teacher gate passes. The goal is to test whether Darjeeling can
externalize a meaningful share of reliable L4 intent-classification behavior
into L2 while preserving final quality through L4 fallback and reducing L4
calls, tokens, cost, and latency.

This is not an NLU productization effort. CLINC150 is a compact closed-form
benchmark for validating the broader Darjeeling mechanism.

## Required Context Files

Read these before changing code:

- `AGENTS.md`
- `docs/experiments/2026-06-23_clinc150_phase1_experiment_plan.md`
- `docs/experiments/2026-06-23_clinc150_phase1_report.md`
- `docs/experiments/2026-06-23_clinc150_teacher_reliability_plan.md`
- `docs/experiments/2026-06-23_clinc150_teacher_reliability_report.md`
- `docs/experiments/2026-06-23_clinc150_teacher_reliability_cost_ledger.json`
- `src/darjeeling/targets/nlu/clinc150_phase1.py`
- `src/darjeeling/targets/nlu/main_cli.py`
- `src/darjeeling/targets/nlu/layers/l2_student.py`
- `tests/targets/nlu/test_clinc150_phase1.py`

## Current State

CLINC150 `data_full` is processed at:

```text
data/processed/clinc150_data_full
```

The repaired L4 teacher gate passed with:

```text
clinc150-intent-v2-label-cards
```

500-request validation gate result for that prompt:

- overall accuracy: 97.4%;
- in-scope accuracy: 98.4%;
- parse/schema failure: 0.0%;
- final empty-response failures: 0;
- observed cost: `$0.3982408`.

This clears only the teacher prerequisite. It does not prove L2
externalization, cost reduction, or cascade quality.

Existing target-local helper code already includes:

- train examples from gold records;
- train examples from parsed teacher rows;
- CLINC150 L2 training;
- CLINC150 L2 evaluation over thresholds;
- threshold metrics for accepted precision, accepted coverage, final cascade
  accuracy, all-L4 accuracy, L4 calls, tokens, cost, latency, and OOS behavior.

Prefer extending these simple helper paths and CLIs rather than adding a new
framework.

## Decision To Support

Make one of these decisions with evidence:

- **Proceed**: CLINC150 Phase 1 supports the mechanism claim well enough to keep
  using it. Evidence should show high accepted L2 precision, meaningful L2
  coverage, final cascade quality close to all-L4, and material L4 call/cost
  reduction on validation, locked test, and at least one replay stream.
- **Pause and repair**: the mechanism looks plausible, but a bounded harness,
  calibration, teacher-cache, or measurement defect prevents a trustworthy
  decision. Repair it and rerun without asking for routine choices.
- **Reject CLINC150 for Phase 1**: a reliable all-L4 teacher exists, but after
  reasonable simple L2 iterations the benchmark still does not permit meaningful
  high-precision L2 absorption.

Do not stop with a weak or purely verbal conclusion. If evidence is weak, define
the next hypothesis and test it.

## Success Targets

Primary quality targets:

- teacher-distilled L2 selected threshold has accepted precision >= 99%;
- lower-layer OOS false accept rate <= 2%;
- final L2+L4 cascade accuracy delta vs all-L4 is no worse than -0.5 percentage
  points on validation and locked test;
- parse/schema failures remain <= 0.5% for live teacher rows used as all-L4
  fallback.

Primary efficiency targets:

- L2 accepted coverage is meaningful. Use >= 50% as the initial target; if the
  best safe threshold is lower, run at least two simple iteration hypotheses
  before concluding the benchmark has weak L2 absorption.
- L4 calls per 100 requests drop materially. Use >= 30% reduction as the initial
  target.
- Report token, cost, and p50/p95 latency changes separately from quality.

Secondary diagnostics:

- gold-trained diagnostic L2 should show whether the current lightweight L2
  family can learn CLINC150 at all;
- teacher-distilled L2 should show the actual Darjeeling mechanism path;
- learning curves should show whether more teacher-visible traces help.

## L0 Isolation

The primary experiment is about L2. Disable or bypass L0 exact/cache behavior in
the main CLINC150 L2 and cascade measurements.

Reason:

- L0 can absorb exact repeats or easy cached requests before L2 sees them.
- That would make lower-layer savings look better while obscuring whether L2
  itself learned useful behavior.
- It would also make L2 accepted coverage and L4-call reduction harder to
  interpret across streams with different repeat rates.

Required reporting scope:

- Report `L2-only shadow` and `L2+L4 fallback` as the primary results.
- Treat `all-L4` as the quality/cost baseline.
- Do not include L0 hits in primary accepted coverage, L4 call reduction, cost
  reduction, or latency reduction claims.
- If a full Darjeeling cascade with L0 enabled is useful, report it only as a
  secondary appendix and keep it clearly separated from the primary L2 result.

Implementation guidance:

- Prefer the target-local CLINC150 L2 evaluation helpers, which already compare
  L2 predictions to teacher fallback rows without requiring the general runtime
  L0 layer.
- If reusing a runtime path that normally enables L0, add a simple experiment
  flag or target-local path to bypass L0 for this benchmark. Do not delete or
  globally disable L0.

## Non-Goals

- Do not continue MASSIVE work.
- Do not run open-ended CLINC150 prompt search.
- Do not add CLINC150 semantics to Darjeeling core.
- Do not build a generic job system, plugin system, dataset DSL, or broad
  evaluator framework.
- Do not directly edit isolated generated L1/L2/L3 workspaces.
- Do not tune on the locked test split. Use validation and train-derived
  development splits for decisions; run test only after a candidate threshold
  and simple design are locked.

## Budget

Most of this phase should be local and cheap. Paid L4 calls are allowed for
teacher trace generation and all-L4 baseline rows.

Use the repaired live path with resume and cost ledgers. Keep new observed spend
under `$30` unless the evidence strongly justifies using more of the previously
discussed `$100` ceiling. Record observed spend from artifacts, including
attempt costs and unknown-usage attempts.

## Experiment Root

Use:

```text
runs/clinc150-l2-cascade-20260623/
```

Suggested report:

```text
docs/experiments/2026-06-23_clinc150_l2_cascade_report.md
```

Suggested aggregate cost ledger:

```text
docs/experiments/2026-06-23_clinc150_l2_cascade_cost_ledger.json
```

## Autonomy And Iteration Method

Work in loops:

1. State the current hypothesis in the experiment notes.
2. Run the smallest test that can falsify it.
3. Record metrics and failures.
4. If it fails but the product goal still makes sense, make the next simple
   change and test again.
5. Stop only when the decision criteria are met or when further progress would
   require changing the product goal, budget, risk tolerance, or target/core
   boundary.

Do not ask the user to choose among routine tactics such as threshold grids,
L2 model family settings, train sample sizes, cached-vs-live rows, or report
format. Make those choices from local evidence.

Keep abstraction low:

- static CLIs or small target-local helpers are fine;
- direct JSON/JSONL/CSV artifacts are fine;
- small scripts under the NLU target are fine if they are tested or clearly
  reproducible;
- avoid new framework layers.

## Phase A: Harness Repair And CLIs

Add or refine minimal CLINC150 target-local commands if needed. The harness
should be able to:

- train diagnostic L2 from official train gold labels;
- train teacher-distilled L2 from parsed live teacher detail rows;
- evaluate a saved L2 bundle on validation/test/streams;
- evaluate thresholds with teacher rows as all-L4 fallback;
- write JSON summaries, optional details JSONL, and a cost/latency table.
- keep primary metrics on an L0-disabled path so accepted coverage and savings
  are attributable to L2.

Reusing existing functions in `clinc150_phase1.py` is preferred. If the existing
helpers are enough, keep code changes small.

Done when focused tests cover:

- teacher row to L2 training example conversion;
- threshold/cascade metrics with teacher fallback rows;
- selected threshold behavior;
- report or CLI artifact shape for the new experiment path.

## Phase B: Gold-Trained Diagnostic L2 Ceiling

Train L2 from official train gold labels. This is a diagnostic ceiling only.
It must not become a runtime artifact or a Darjeeling mechanism claim.

Run:

- validation evaluation;
- locked test evaluation after initial validation checks;
- learning curve for 250 / 1k / 3k / full train examples, if cheap;
- simple model/config iterations if the first result is weak.

Suggested simple iterations:

- threshold grid adjustments;
- existing `L2StudentConfig` options such as model family, max features,
  regularization, iterations, or MLP settings if available;
- OOS handling checks;
- stratified sample sizes.

Report:

- raw accuracy;
- accepted precision and coverage by threshold;
- selected threshold;
- OOS false accept rate;
- confusion families;
- latency.

Decision from Phase B:

- If gold-trained L2 cannot reach high accepted precision with meaningful
  coverage after simple iterations, CLINC150 may be a poor L2 absorption
  benchmark even though L4 can solve it.
- If gold-trained L2 works, continue to teacher-distilled L2.

Do not stop at the first weak diagnostic number. Iterate before deciding.

## Phase C: Teacher Trace Generation

Generate teacher-visible train rows using the locked prompt:

```text
clinc150-intent-v2-label-cards
```

Use the repaired `clinc150 teacher-eval --resume-existing` path.

Recommended order:

1. Train split 500-row smoke.
2. Train split 3k-row stream if smoke is reliable.
3. Full train split if cost and time remain reasonable.
4. Full validation teacher rows for all-L4 baseline if not already available.
5. Full test teacher rows only after the validation candidate is locked.

The train split has only 100 OOS examples, so make sure OOS behavior is measured
on validation/test and not overclaimed from train alone.

Use cached rows for downstream L2/cascade experiments. Do not relabel rows that
already exist unless there is a clear artifact mismatch or failed-run policy
reason.

## Phase D: Teacher-Distilled L2

Train L2 only from parsed teacher rows. Do not use validation/test gold labels
as training input.

Run at least:

- teacher-distilled L2 trained on 500 teacher rows;
- teacher-distilled L2 trained on 3k teacher rows;
- teacher-distilled L2 trained on full train teacher rows if generated;
- the same selected threshold logic across variants;
- evaluation on validation using teacher rows as all-L4 fallback;
- locked test evaluation once a validation candidate is selected.

Compare against:

- all-L4 teacher baseline;
- gold-trained diagnostic ceiling;
- L2-only shadow predictions;
- L2+L4 fallback cascade.

If teacher-distilled L2 is far below the gold ceiling:

- inspect teacher train row quality and class coverage;
- try a larger teacher-labeled sample;
- try simple L2 config changes;
- check whether OOS scarcity is driving false accepts;
- then rerun validation.

Do not switch metrics to make a bad model look good. The strict gold accuracy,
all-L4 comparison, accepted precision, coverage, and OOS false accept metrics
must remain visible.

## Phase E: Stream And Distribution Tests

Run stream tests after a validation threshold is selected.

At minimum:

- official validation distribution;
- official locked test distribution;
- uniform replay;
- zipf-heavy replay;
- a sequential or learning-curve style replay that shows how coverage changes
  as teacher-visible traces increase.

For each, report:

- requests;
- all-L4 accuracy;
- L2-only accuracy;
- final L2+L4 cascade accuracy with L0 disabled;
- delta vs all-L4;
- L2 accepted coverage;
- L2 accepted precision;
- lower-layer OOS false accept rate;
- L4 calls per 100 requests;
- L4 tokens and cost per request;
- p50/p95 latency;
- parse/schema failures;
- retry recovered rows and unknown usage attempts for live teacher artifacts.

Use paired cached teacher rows when comparing routing/cascade decisions. Use
live runs only for missing teacher rows or updated cost/latency claims.
Keep L0 disabled or bypassed for these primary stream comparisons.

## Phase F: Failure Analysis And Iteration

If targets fail, run bounded iterations before giving up.

Examples:

- If accepted precision is high but coverage is too low, test threshold and
  calibration changes, more teacher traces, and simple model-family changes.
- If coverage is high but precision is too low, raise threshold, inspect
  confusion families, and test simple OOS safety rules inside target-local L2
  evaluation or config.
- If cascade quality drops vs all-L4, identify whether the drop comes from
  L2 accepted wrong rows, teacher fallback errors, OOS false accepts, or
  artifact mismatch.
- If L4 cost reduction is weak, inspect accepted coverage and stream shape; do
  not hide quality loss behind lower cost.
- If live teacher reliability regresses, use resume and attempt diagnostics
  before changing prompts.

Record each iteration with:

- hypothesis;
- change;
- command or script;
- metrics;
- decision.

## Final Report

Write:

```text
docs/experiments/2026-06-23_clinc150_l2_cascade_report.md
```

The report must include:

- data source and previous teacher-gate context;
- exact teacher trace sources and costs;
- diagnostic gold-trained L2 ceiling;
- teacher-distilled L2 variants and learning curve;
- selected threshold and why it was selected;
- validation and locked test results;
- stream results;
- cost/latency reduction;
- failure analysis and iterations attempted;
- decision: proceed, pause/repair, or reject CLINC150;
- remaining risks.

Also write or update:

```text
docs/experiments/2026-06-23_clinc150_l2_cascade_cost_ledger.json
```

## Validation

Run at least:

```bash
uv run pytest tests/targets/nlu/test_clinc150_phase1.py tests/targets/nlu/test_teacher_eval.py tests/targets/nlu/test_l4_teacher.py -q
uv run pytest -q
uv run ruff check <touched python files>
git diff --check
```

For any paid live command, preserve the exact command, environment variables,
output directory, and observed cost in the report.

## Done Criteria

- The harness can train/evaluate diagnostic and teacher-distilled CLINC150 L2.
- Teacher rows are generated or reused with resume-safe artifacts and cost
  ledgers.
- Diagnostic L2 ceiling is measured.
- Teacher-distilled L2 is measured across at least two train sizes.
- A validation-selected threshold is evaluated on locked test.
- Cascade stream metrics show whether L4 calls/cost/latency fall without
  unacceptable quality loss.
- At least one autonomous iteration is performed if the first teacher-distilled
  result misses targets.
- Final report and aggregate cost ledger are written.
- All checks pass.
- Changes are organized into a git commit.
