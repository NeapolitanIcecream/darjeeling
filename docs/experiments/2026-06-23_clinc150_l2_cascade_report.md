# CLINC150 L2 And Cascade Experiment Report

Date: 2026-06-23

Decision: **Pause and repair**.

Full train teacher traces made the mechanism plausible: the full teacher-distilled
L2 at threshold `0.98` passed validation quality and efficiency targets with
99.10% accepted precision, 50.32% accepted coverage, 50.32% fewer L4 calls, and
only -0.097 percentage points cascade accuracy delta vs all-L4. The same
validation-locked threshold did not pass locked test accepted precision:
98.77%, below the 99% target. A conservative threshold, `0.995`, passed locked
test precision and quality but only accepted 24.56% of requests, below the
initial efficiency target.

The result supports the broader L2 absorption hypothesis but shows that the
current validation threshold selection is not robust enough to treat CLINC150
Phase 1 as passed.

## Data And Scope

Processed CLINC150 `data_full` source:

```text
data/processed/clinc150_data_full
```

Primary experiment root:

```text
runs/clinc150-l2-cascade-20260623/
```

Prompt and model path:

- Prompt: `clinc150-intent-v2-label-cards`
- Observed models: `gpt-5.5`, `gpt-5.5-2026-04-24`
- Live teacher environment for paid runs:
  `OPENAI_TIMEOUT_S=120 OPENAI_MAX_RETRIES=6 TEACHER_MAX_TOKENS=256`

Primary L2/cascade measurements bypassed L0 and used target-local CLINC150
helpers. Reported L4 savings are therefore attributable to L2 accepted rows and
not exact-cache hits.

## Harness Work

Implemented minimal target-local CLINC150 harness support:

- stratified CLINC150 sampling for train rows;
- `clinc150 l2-train` CLI for gold and teacher-distilled L2 bundles;
- `clinc150 l2-eval` CLI for validation/test/stream evaluation with L4 fallback
  teacher details;
- JSON train/eval summaries, optional prediction details JSONL, and
  cost/latency tables;
- selected-threshold quality guard requiring accepted precision >= 99%, OOS
  false accept rate <= 2%, and cascade delta no worse than -0.5pp when paired
  all-L4 rows are available.

Focused tests cover teacher rows to L2 examples, artifact shape, stratified
sampling, threshold/cascade metrics, and fallback cost/latency accounting.

## Teacher Traces And Cost

Aggregate cost ledger:

```text
docs/experiments/2026-06-23_clinc150_l2_cascade_cost_ledger.json
```

Observed paid spend for this experiment: `$20.9290952` attempt cost, under the
`$30` limit. Final-response cost was `$20.5362356`.

| Run | Split/stream | Requests | Parsed | Accuracy | Parse fail | Cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| train-500-stratified | train/stratified | 500 | 499 | 99.40% | 0.200% | $0.394370 |
| train-3000-stratified | train/stratified | 3000 | 2998 | 97.40% | 0.067% | $2.295830 |
| train-full-stratified | train/stratified | 15100 | 15072 | 97.72% | 0.185% | $11.559850 |
| validation-full | validation/sequential | 3100 | 3097 | 98.32% | 0.097% | $2.376348 |
| test-full | test/sequential | 5500 | 5497 | 96.02% | 0.055% | $4.302697 |

All live teacher artifacts met the <= 0.5% parse/schema failure target.

Important live commands:

```bash
OPENAI_TIMEOUT_S=120 OPENAI_MAX_RETRIES=6 TEACHER_MAX_TOKENS=256 \
uv run python -m darjeeling.targets.nlu.main_cli clinc150 teacher-eval \
  --split train \
  --out-dir runs/clinc150-l2-cascade-20260623/teacher-traces/train-full-stratified \
  --prompt-version clinc150-intent-v2-label-cards \
  --stream stratified \
  --max-workers 64 \
  --resume-existing \
  --no-fail-on-gate
```

```bash
OPENAI_TIMEOUT_S=120 OPENAI_MAX_RETRIES=6 TEACHER_MAX_TOKENS=256 \
uv run python -m darjeeling.targets.nlu.main_cli clinc150 teacher-eval \
  --split test \
  --out-dir runs/clinc150-l2-cascade-20260623/teacher-traces/test-full \
  --prompt-version clinc150-intent-v2-label-cards \
  --stream sequential \
  --max-workers 64 \
  --no-fail-on-gate
```

The full train run began at lower concurrency, was safely terminated with
SIGTERM after preserving JSONL rows, and resumed with `--resume-existing`. The
final complete artifact has 15100 rows.

## Diagnostic Gold L2 Ceiling

Gold-trained L2 is diagnostic only. It uses official train labels and is not a
Darjeeling mechanism artifact.

| Variant | Split | Raw acc | Reported threshold | Accepted precision | Coverage | OOS false accept |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| gold 250 | validation | 45.23% | 0.93 | 100.00% | 0.77% | 0.00% |
| gold 1000 | validation | 66.77% | 0.995 | 98.76% | 5.19% | 0.00% |
| gold 3000 | validation | 78.42% | 0.995 | 99.04% | 10.06% | 0.00% |
| gold full | validation | 87.03% | 0.97 | 99.26% | 56.81% | 1.00% |
| gold full | test | 74.09% | 0.98 | 99.07% | 44.84% | 1.50% |

Interpretation: the lightweight L2 family can absorb a meaningful share of
CLINC150 only with full training coverage. The gold diagnostic ceiling justified
continuing from 500/3k teacher traces to full teacher traces.

## Teacher-Distilled Learning Curve

Validation evaluation used full validation teacher rows as paired all-L4
fallback. Shown thresholds are either the selected threshold or the best
high-threshold diagnostic row when no threshold was selected.

| Variant | Raw L2 acc | All-L4 acc | Threshold | Accepted precision | Coverage | Cascade delta | L4 call reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| teacher 500 | 55.77% | 98.32% | 0.995 | 98.39% | 2.00% | 0.000pp | 2.00% |
| teacher 3000 | 77.65% | 98.32% | 0.995 | 98.80% | 5.39% | -0.032pp | 5.39% |
| teacher 3000, retrieval frames | 77.84% | 98.32% | 0.995 | 98.64% | 4.74% | -0.032pp | 4.74% |
| teacher 3000, MLP | 75.52% | 98.32% | 0.995 | 100.00% | 0.35% | 0.000pp | 0.35% |
| teacher full | 86.94% | 98.32% | 0.98 | 99.10% | 50.32% | -0.097pp | 50.32% |

Iteration notes:

- Hypothesis 1: 500 teacher rows were too sparse. Result: 3k rows improved raw
  L2 accuracy but still failed precision and coverage.
- Hypothesis 2: frame retrieval might improve conservative acceptance. Result:
  retrieval frames did not improve high-threshold coverage.
- Hypothesis 3: higher-capacity MLP might improve precision. Result: MLP was
  safe only at negligible coverage.
- Hypothesis 4: full teacher-visible train coverage is required. Result: full
  train passed validation quality/coverage and became the candidate.

## Validation Candidate

Validation selected threshold:

```text
0.98
```

Full teacher-distilled validation result at `0.98`:

| Metric | Value |
| --- | ---: |
| Requests | 3100 |
| All-L4 accuracy | 98.32% |
| L2-only raw accuracy | 86.94% |
| Accepted precision | 99.10% |
| Accepted coverage | 50.32% |
| OOS false accept rate | 1.00% |
| Final cascade accuracy | 98.23% |
| Delta vs all-L4 | -0.097pp |
| L4 calls per 100 requests | 49.68 |
| L4 call reduction | 50.32% |
| Token reduction | 50.02% |
| Cost reduction | 49.41% |
| p50 latency reduction | 99.48% |
| p95 latency reduction | 21.14% |
| Parse/schema failure rate | 0.097% |

This passes the validation quality targets and initial efficiency targets.

## Locked Test

The validation-locked threshold `0.98` did not pass locked test accepted
precision.

| Threshold | Accepted precision | Coverage | OOS false accept | Cascade delta | L4 call reduction | Cost reduction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.98 | 98.77% | 42.73% | 1.50% | 0.000pp | 42.73% | 40.76% |
| 0.995 | 99.78% | 24.56% | 0.30% | +0.109pp | 24.56% | 23.27% |

Interpretation:

- Quality preservation is strong: at `0.98`, final cascade accuracy matched the
  all-L4 baseline on test.
- OOS false accepts remained within the <= 2% target at `0.98`.
- The accepted precision miss is narrow but material because the target is
  strict and test was locked.
- The conservative threshold `0.995` is quality-safe but does not provide the
  target >= 30% L4 call reduction or >= 50% accepted coverage.

## Stream Results

Stream runs used cached validation teacher rows and repeated their cost/latency
when replay sampling repeated request ids. No additional live calls were made.

| Stream | Threshold | All-L4 acc | Accepted precision | Coverage | OOS false accept | Cascade delta | L4 call reduction | Cost reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| validation sequential | 0.98 | 98.32% | 99.10% | 50.32% | 1.00% | -0.097pp | 50.32% | 49.41% |
| test sequential | 0.98 | 96.02% | 98.77% | 42.73% | 1.50% | 0.000pp | 42.73% | 40.76% |
| validation uniform | 0.98 | 98.19% | 99.35% | 49.74% | 1.92% | +0.161pp | 49.74% | 48.74% |
| validation zipf-heavy | 0.98 | 97.48% | 98.78% | 39.77% | 0.88% | -0.129pp | 39.77% | 38.80% |
| validation zipf-heavy | 0.995 | 97.48% | 99.86% | 22.65% | 0.00% | 0.000pp | 22.65% | 22.62% |

Stream result: distribution shape matters. Uniform replay looks similar to
validation. Zipf-heavy replay and locked test both reduce accepted precision at
the validation-selected threshold.

## Failure Analysis

The failure is not a teacher reliability failure. Validation and test teacher
fallback rows both pass parse/schema gates, and the all-L4 baselines are strong.

The failure is not caused only by OOS false accepts. At locked test threshold
`0.98`, OOS false accept rate is 1.50%, within target, while accepted precision
is 98.77%. Errors include in-scope accepted intent mistakes as well as OOS
pressure.

The failure is primarily threshold robustness. Validation selected the first
threshold that cleared strict quality targets while preserving meaningful
coverage. That threshold did not generalize to locked test. A stronger safety
margin such as `0.995` generalizes quality but loses too much absorption.

## Remaining Risks And Next Repair

Recommended next repair remains target-local and low-abstraction:

- create a train-derived calibration/dev split from teacher train rows instead
  of selecting only on official validation;
- test an OOS-heavy calibration slice because train has only 100 official OOS
  rows and teacher full train produced 170 OOS predictions;
- select thresholds with a precision confidence margin, not only point
  precision;
- inspect accepted in-scope confusion families at threshold `0.98`;
- consider a simple target-local OOS/low-margin veto if it improves locked-test
  precision without collapsing coverage.

Do not promote `0.98` as passed. Do not promote `0.995` as a successful Phase 1
mechanism result because its L4 call reduction is only 24.56% on locked test.

## Validation Commands

Code validation commands are recorded in the final implementation handoff. Key
experiment commands:

```bash
uv run python -m darjeeling.targets.nlu.main_cli clinc150 l2-train \
  --source teacher \
  --teacher-details runs/clinc150-l2-cascade-20260623/teacher-traces/train-full-stratified/teacher_live_vs_gold.details.jsonl \
  --out-dir runs/clinc150-l2-cascade-20260623/distilled-l2/train-full
```

```bash
uv run python -m darjeeling.targets.nlu.main_cli clinc150 l2-eval \
  --bundle-path runs/clinc150-l2-cascade-20260623/distilled-l2/train-full/l2_student.joblib \
  --out-dir runs/clinc150-l2-cascade-20260623/distilled-l2/train-full/validation-cascade \
  --split validation \
  --teacher-details runs/clinc150-l2-cascade-20260623/teacher-traces/validation-full/teacher_live_vs_gold.details.jsonl \
  --write-details
```

```bash
uv run python -m darjeeling.targets.nlu.main_cli clinc150 l2-eval \
  --bundle-path runs/clinc150-l2-cascade-20260623/distilled-l2/train-full/l2_student.joblib \
  --out-dir runs/clinc150-l2-cascade-20260623/distilled-l2/train-full/test-cascade \
  --split test \
  --teacher-details runs/clinc150-l2-cascade-20260623/teacher-traces/test-full/teacher_live_vs_gold.details.jsonl \
  --thresholds 0.0,0.5,0.7,0.8,0.9,0.93,0.95,0.97,0.98,0.99,0.995 \
  --write-details
```
