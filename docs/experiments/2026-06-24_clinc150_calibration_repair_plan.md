# CLINC150 Calibration Repair Plan

Date: 2026-06-24

Purpose: repair the CLINC150 Phase 1 near miss from the L2 cascade experiment.
The previous run made the L2 absorption mechanism plausible but did not pass
locked test at the validation-selected threshold. This plan focuses on threshold
robustness, calibration, accepted-error analysis, and simple target-local safety
rules while keeping L0 disabled for primary results.

## Required Context Files

Read these before changing code:

- `AGENTS.md`
- `docs/experiments/2026-06-23_clinc150_l2_cascade_experiment_plan.md`
- `docs/experiments/2026-06-23_clinc150_l2_cascade_report.md`
- `docs/experiments/2026-06-23_clinc150_l2_cascade_cost_ledger.json`
- `docs/experiments/2026-06-23_clinc150_teacher_reliability_report.md`
- `src/darjeeling/targets/nlu/clinc150_phase1.py`
- `src/darjeeling/targets/nlu/main_cli.py`
- `tests/targets/nlu/test_clinc150_phase1.py`

## Current State

The previous experiment decision was `Pause and repair`.

Best validation candidate:

- model: full teacher-distilled CLINC150 L2;
- threshold: `0.98`;
- validation accepted precision: 99.10%;
- validation accepted coverage: 50.32%;
- validation cascade delta vs all-L4: -0.097 percentage points;
- validation L4 call reduction: 50.32%.

Locked test at the validation-locked threshold:

- threshold: `0.98`;
- accepted precision: 98.77%;
- accepted coverage: 42.73%;
- cascade delta vs all-L4: 0.000 percentage points;
- L4 call reduction: 42.73%.

Conservative locked-test threshold:

- threshold: `0.995`;
- accepted precision: 99.78%;
- accepted coverage: 24.56%;
- L4 call reduction: 24.56%.

Interpretation: the teacher and L2 harness are now good enough to continue. The
main defect is threshold/guard robustness, not L4 teacher reliability.

Existing artifacts to reuse:

- `runs/clinc150-l2-cascade-20260623/teacher-traces/train-full-stratified/teacher_live_vs_gold.details.jsonl`
- `runs/clinc150-l2-cascade-20260623/teacher-traces/validation-full/teacher_live_vs_gold.details.jsonl`
- `runs/clinc150-l2-cascade-20260623/teacher-traces/test-full/teacher_live_vs_gold.details.jsonl`
- `runs/clinc150-l2-cascade-20260623/distilled-l2/train-full/l2_student.joblib`
- `runs/clinc150-l2-cascade-20260623/distilled-l2/train-full/validation-cascade/clinc150_l2_predictions.jsonl`
- `runs/clinc150-l2-cascade-20260623/distilled-l2/train-full/test-cascade/clinc150_l2_predictions.jsonl`

Use these before making new paid L4 calls.

## L4 Replay Oracle Design

As part of this repair, make the existing CLINC150 `teacher_details` fallback
path explicit as a target-local experiment facility. This facility is for
benchmark accounting only. It is not production cache behavior and not a new
Darjeeling core concept.

Working name:

```text
L4 replay oracle
```

Purpose:

- Pay for live L4 benchmark rows once.
- Reuse those rows in later L1/L2 experiments without new live L4 calls.
- Still count fallback rows as L4 calls in experiment metrics.
- Preserve observed L4 statistics needed for experiment reports: model, output,
  parse/schema failure, tokens, cost, latency, retry diagnostics, and request
  coverage.

Boundary:

- Keep target semantics in the NLU/CLINC150 target. The target decides how to
  parse the L4 output, compare it with gold, and compute target-specific
  correctness.
- A future shared helper may carry target-independent fields such as
  `request_id`, `model`, `usage`, `cost_usd`, `latency_ms`, `parse_failure`,
  `attempts`, and opaque `output`, but it must not interpret CLINC150 labels or
  NLU frames.
- Do not fold this into `TeacherCache` in this plan. `TeacherCache` is runtime
  replay cache behavior: cache hits have zero marginal serving cost and local
  cache latency. The L4 replay oracle is counterfactual experiment accounting:
  fallback rows still count as L4 calls and use recorded L4 cost/latency.

Implementation expectation for this plan:

- Refactor the existing CLINC150 `teacher_details` fallback logic into a clearly
  named target-local helper, if doing so reduces duplication for calibration and
  future L1 experiments.
- Keep the helper small: load rows by `request_id`, validate request coverage,
  expose fallback frame/output, expose L4 statistics, and produce all-L4
  baseline metrics.
- Use the helper in the calibration repair where practical.
- Do not block calibration progress on a broad abstraction. If the existing
  helper path is already sufficient, document the facility and add only the
  minimal naming/tests needed to make the semantics clear.

## Decision To Support

Make one of these decisions with evidence:

- **Proceed**: a calibration-selected L2 guard passes locked test with accepted
  precision >= 99%, lower-layer OOS false accept rate <= 2%, cascade delta vs
  all-L4 no worse than -0.5 percentage points, and meaningful L2 coverage/L4
  call reduction. Use >= 40% locked-test coverage as the practical repair target
  for this phase, because the previous validation-selected candidate had 42.73%
  locked-test coverage while narrowly missing precision.
- **Pause and repair again**: the repair finds a plausible path but needs a
  bounded harness or calibration fix before the decision is trustworthy.
- **Reject CLINC150 for Phase 1**: after simple calibration, safety-margin, and
  veto iterations, the benchmark still cannot deliver robust high-precision L2
  absorption with meaningful coverage.

Do not declare success by switching to `0.995` alone unless coverage is also
meaningful. Do not declare failure until at least two simple calibration/guard
iterations have been tried.

## Non-Goals

- Do not change the benchmark, teacher prompt, or target task.
- Do not continue MASSIVE work.
- Do not use locked test to choose thresholds, margins, label vetoes, or model
  variants.
- Do not add CLINC150 semantics to Darjeeling core.
- Do not move L4 replay-oracle target semantics into Darjeeling core.
- Do not change `TeacherCache` semantics or make production cache hits carry
  counterfactual L4 cost by default.
- Do not build a broad calibration framework, job system, plugin system, or
  schema DSL.
- Do not enable L0 in the primary experiment.
- Do not hide strict metrics behind looser semantic-match diagnostics.

## L0 Isolation

Keep L0 disabled or bypassed for primary measurements. The report must keep
primary results in `L2-only shadow` and `L2+L4 fallback` terms. Any full cascade
with L0 enabled is optional appendix material only and must not support the
main decision.

## Budget

This repair should mostly use existing local artifacts. New paid L4 calls are
allowed only if an artifact is missing or clearly invalid. Keep new observed
spend under `$5` unless a stronger reason is recorded in the report. If paid
calls occur, use the repaired resume-safe live path and write a cost ledger.

## Experiment Root

Use:

```text
runs/clinc150-calibration-repair-20260624/
```

Write the final report to:

```text
docs/experiments/2026-06-24_clinc150_calibration_repair_report.md
```

If there is any new paid spend, write or update:

```text
docs/experiments/2026-06-24_clinc150_calibration_repair_cost_ledger.json
```

If there is no new paid spend, state that explicitly in the report and point to
the reused cost ledger from the previous phase.

## Autonomy And Iteration Method

Work autonomously through hypothesis, experiment, result, and iteration loops.
Do not stop at the first near miss or ask the user to choose routine tactics.

For each iteration, record:

- hypothesis;
- data split used;
- guard or calibration rule;
- validation/dev metrics;
- whether it is eligible for locked-test evaluation;
- locked-test result if evaluated;
- decision.

Escalate only if the next step would change the product goal, budget, target/core
boundary, or locked-test policy.

## Phase A: Calibration Harness

Add minimal target-local CLINC150 calibration utilities or CLIs. Prefer simple
functions in `clinc150_phase1.py` and CLI commands in `main_cli.py`.

The harness should support:

- loading existing `clinc150_l2_predictions.jsonl` files;
- generating prediction rows if needed from an existing L2 bundle;
- building train-derived calibration/dev splits from teacher train rows;
- selecting thresholds or guard rules on calibration/dev data;
- evaluating selected guards on validation and locked test;
- using the CLINC150 L4 replay oracle for cached L4 fallback accounting;
- writing JSON summaries and optional accepted-error JSONL files.

Do not create a generic calibration framework. Plain dictionaries, JSON/JSONL,
and static CLINC150 commands are enough.

Focused tests should cover:

- split construction is deterministic;
- locked test is not used by the selector;
- selected guard uses configured precision/OOS/cascade constraints;
- L4 replay-oracle fallback rows count as L4 calls and carry recorded
  cost/latency, while remaining distinct from L0 and `TeacherCache`;
- accepted-error summaries include enough fields to debug confusion families;
- L0 remains absent from primary metrics.

## Phase B: Train-Derived Calibration Splits

Build at least two train-derived evaluation views from teacher train artifacts:

1. **General calibration/dev split**
   - Use parsed teacher train rows and official train gold labels.
   - Keep it deterministic and stratified enough to represent many intents.
   - Do not include validation or test rows.

2. **OOS-heavy calibration slice**
   - Include official train OOS rows.
   - Include rows where teacher predicted `out_of_scope`.
   - Include high-risk rows where L2 predicts in-scope but the gold or teacher
     row indicates OOS.
   - Keep duplicate handling explicit.

Use these splits to select or reject guards before looking at locked test.

## Phase C: Baseline Error Audit

Audit accepted errors for the current full teacher-distilled model at threshold
`0.98`.

Run this on:

- train-derived calibration/dev;
- OOS-heavy calibration;
- official validation;
- locked test only for post-selection diagnosis, not for rule selection.

Summarize:

- accepted wrong count;
- accepted wrong by gold intent;
- accepted wrong by predicted intent;
- accepted wrong in-scope-to-in-scope confusions;
- accepted OOS false accepts;
- distributions of guard probability, top1 probability, margin, and entropy;
- examples for the top confusion families.

The goal is to learn whether a simple margin/entropy/OOS veto can remove wrong
accepts without collapsing coverage.

## Phase D: Calibration And Guard Iterations

Try simple, low-abstraction candidates. Use train-derived calibration and
official validation for selection. Locked test is for final confirmation only.

Candidate families:

1. **Threshold with safety margin**
   - Select threshold by point precision plus an explicit margin above 99%.
   - Also try a lower confidence bound such as Wilson lower bound for accepted
     precision if easy to implement.
   - Report both point precision and lower-bound precision.

2. **Margin and entropy guard**
   - Keep `guard_probability >= threshold`.
   - Add minimum `margin` and/or maximum `entropy` if calibration shows accepted
     wrong rows cluster there.
   - Prefer one or two scalar cutoffs, not a complex learned guard.

3. **OOS safety veto**
   - Add a simple veto for OOS-risk rows if calibration shows OOS false accepts
     dominate.
   - Examples may use prediction confidence, margin, entropy, or predicted OOS
     probability if available.
   - Do not hard-code locked-test examples or labels.

4. **Confusion-family veto**
   - If accepted in-scope mistakes cluster in a small number of predicted/gold
     families on calibration and validation, test a small target-local veto.
   - Keep it transparent and documented.
   - Do not encode test-only families.

5. **Threshold grid refinement**
   - Add thresholds between `0.98` and `0.995`, such as `0.982`, `0.985`,
     `0.987`, `0.99`, and `0.992`.
   - This should be tried before adding more complex rules.

For each candidate, report:

- accepted precision;
- accepted precision lower bound if implemented;
- accepted coverage;
- lower-layer OOS false accept rate;
- final cascade accuracy and delta vs all-L4;
- L4 call/cost/token/latency reduction;
- accepted wrong examples or confusion families.

## Phase E: Selection Policy

Select a candidate without using locked test.

A candidate is eligible for locked test only if:

- train-derived calibration/dev passes the quality targets;
- OOS-heavy calibration does not show unacceptable OOS false accepts;
- official validation passes accepted precision >= 99%, OOS false accept <= 2%,
  and cascade delta >= -0.5pp;
- validation coverage remains meaningful, preferably near or above 45%;
- the rule is simple enough for future maintainers to understand quickly.

If multiple candidates pass, choose the one with the highest validation coverage
among candidates with an adequate precision margin. Do not choose a fragile rule
only because it maximizes coverage.

## Phase F: Locked Test And Stream Confirmation

After selection, run locked test once for the selected candidate family. If the
first locked-test result fails narrowly and the reason is clear from pre-test
evidence, one bounded repair iteration is allowed, but record it as a second
test exposure.

Confirm on:

- locked test sequential;
- validation uniform;
- validation zipf-heavy;
- optional OOS-heavy diagnostic stream.

For each, keep L0 disabled and report:

- all-L4 accuracy;
- L2-only raw accuracy;
- final L2+L4 cascade accuracy;
- delta vs all-L4;
- accepted precision;
- accepted coverage;
- lower-layer OOS false accept rate;
- L4 calls per 100 requests;
- L4 cost/token/latency reduction.
- whether L4 statistics came from live calls in this phase or from the replay
  oracle.

## Final Report

Write:

```text
docs/experiments/2026-06-24_clinc150_calibration_repair_report.md
```

The report must include:

- context from the previous near miss;
- reused artifacts and whether any new paid cost occurred;
- the L4 replay-oracle source artifacts and accounting semantics;
- calibration/dev split definitions;
- OOS-heavy slice definition;
- baseline accepted-error audit;
- each autonomous iteration tried;
- selected guard and why it was selected;
- validation, locked test, and stream results;
- decision: Proceed, Pause and repair again, or Reject CLINC150;
- risks and next step.

## Validation

Run at least:

```bash
uv run pytest tests/targets/nlu/test_clinc150_phase1.py tests/targets/nlu/test_teacher_eval.py tests/targets/nlu/test_l4_teacher.py -q
uv run pytest -q
uv run ruff check <touched python files>
git diff --check
```

If new paid calls happen, validate cost ledger parsing and reconcile observed
spend against detail JSONL artifacts.

## Done Criteria

- Calibration/dev and OOS-heavy slices are reproducible and documented.
- The CLINC150 L4 replay-oracle semantics are explicit and reusable by future
  L1/L2 benchmark experiments.
- At least two simple repair hypotheses are evaluated unless the first one
  strongly passes all criteria.
- Locked test is used only after a candidate is selected.
- Primary results keep L0 disabled.
- A final decision is written with evidence.
- Final report and any cost ledger are written.
- All checks pass.
- Changes are organized into a git commit.
