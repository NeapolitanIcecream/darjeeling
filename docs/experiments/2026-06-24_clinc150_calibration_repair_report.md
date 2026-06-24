# CLINC150 Calibration Repair Report

Date: 2026-06-24

Decision: **Pause and repair again**.

The repair made the calibration and replay-accounting path explicit, but it did
not produce a locked-test-passing guard. The best second-exposure guard missed
the strict accepted-precision target by a tiny margin, 98.997% vs 99%, and fell
below the practical 40% locked-test coverage target at 38.07%. This is still a
plausible CLINC150 Phase 1 path, but the evidence is not strong enough to
proceed.

Primary measurements kept L0 disabled and used the target-local L2 shadow plus
L4 fallback path.

## Reused Artifacts And Cost

Experiment root:

```text
runs/clinc150-calibration-repair-20260624/
```

Main summaries:

- `runs/clinc150-calibration-repair-20260624/clinc150_calibration_repair_summary.json`
- `runs/clinc150-calibration-repair-20260624/safety-margin-995/clinc150_calibration_repair_summary.json`

Reused previous-phase artifacts:

- `runs/clinc150-l2-cascade-20260623/teacher-traces/train-full-stratified/teacher_live_vs_gold.details.jsonl`
- `runs/clinc150-l2-cascade-20260623/teacher-traces/validation-full/teacher_live_vs_gold.details.jsonl`
- `runs/clinc150-l2-cascade-20260623/teacher-traces/test-full/teacher_live_vs_gold.details.jsonl`
- `runs/clinc150-l2-cascade-20260623/distilled-l2/train-full/l2_student.joblib`
- `runs/clinc150-l2-cascade-20260623/distilled-l2/train-full/validation-cascade/clinc150_l2_predictions.jsonl`
- `runs/clinc150-l2-cascade-20260623/distilled-l2/train-full/test-cascade/clinc150_l2_predictions.jsonl`
- validation uniform and zipf-heavy prediction JSONL artifacts from the same run.

No new paid L4 calls were made in this repair, and no new cost ledger was
written. Cost accounting points back to
`docs/experiments/2026-06-23_clinc150_l2_cascade_cost_ledger.json`, which
recorded `$20.9290952` previous-phase observed attempt spend.

## L4 Replay Oracle

The existing CLINC150 `teacher_details` fallback path is now a named
target-local `Clinc150L4ReplayOracle`. It loads observed L4 rows by
`request_id`, validates request coverage, exposes the recorded teacher frame and
observed L4 statistics, and computes all-L4 baseline metrics.

This is benchmark accounting, not production cache behavior. A replay-oracle
fallback row still counts as an L4 call and carries recorded L4 tokens, cost,
latency, retry diagnostics, model, parse/schema failure state, and output. It
does not change `TeacherCache` semantics and does not add CLINC150 labels or NLU
frame interpretation to Darjeeling core.

## Calibration Views

Train-derived split source: parsed full train teacher rows with official train
gold labels. Split construction is deterministic, intent-stratified, and uses no
validation or test rows.

| View | Requests |
| --- | ---: |
| general calibration | 7,545 |
| general dev | 7,527 |
| parsed train rows | 15,072 |

OOS-heavy slice:

- requests: 183;
- official gold OOS rows: 100;
- teacher-predicted OOS rows: 170;
- rows where L2 predicted in-scope while gold or teacher indicated OOS: 67;
- duplicate handling: request-id deduplicated with reason counts preserved.

## Baseline Audit

Baseline guard: `guard_probability >= 0.98`.

| View | Accepted precision | Coverage | OOS false accept | Cascade delta |
| --- | ---: | ---: | ---: | ---: |
| train calibration | 99.62% | 88.28% | 4.00% | +0.027pp |
| train dev | 99.39% | 87.64% | 2.00% | 0.000pp |
| OOS-heavy | 25.00% | 2.19% | 3.00% | +0.546pp |
| validation | 99.10% | 50.32% | 1.00% | -0.097pp |

The baseline audit confirmed that validation looked good, but OOS-heavy exposed
too many lower-layer OOS false accepts. Validation accepted-error families
included `distance -> directions`, `improve_credit_score -> credit_score`,
`transactions -> spending_history`, plus one OOS false accept.

## Iterations

Candidate families evaluated on calibration dev, OOS-heavy, and validation:

| Family | Candidates |
| --- | ---: |
| threshold | 8 |
| threshold + margin | 40 |
| threshold + entropy | 32 |
| predicted-intent veto | 24 |

Locked test was not used for selection. It was evaluated only after a candidate
was selected.

### Iteration 1: Threshold, Margin, Entropy, Veto Grid

Selection policy: accepted precision >= 99%, OOS false accept <= 2%, cascade
delta >= -0.5pp on validation, with OOS-heavy OOS false accept <= 2%.

Selected guard:

```text
guard_probability >= 0.98 and margin >= 0.15
```

| Split | Accepted precision | Coverage | OOS false accept | Cascade delta | L4 call reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| calibration dev | 99.50% | 82.61% | 0.00% | 0.000pp | 82.61% |
| OOS-heavy | 33.33% | 1.64% | 2.00% | +0.546pp | 1.64% |
| validation | 99.10% | 50.13% | 1.00% | -0.097pp | 50.13% |
| locked test | 98.76% | 42.51% | 1.50% | 0.000pp | 42.51% |

Result: failed locked-test accepted precision. The miss was material despite
good coverage and quality preservation.

### Iteration 2: Safety Margin And Shared Confusion Veto

Bounded repair policy after the first locked-test miss: require >= 99.5%
accepted precision on calibration dev and validation before selection. The
selected veto intents were derived from shared calibration-dev and validation
accepted-error families, not from locked test.

Selected guard:

```text
guard_probability >= 0.985
veto predicted intents: credit_score, directions, spending_history
```

| Split | Accepted precision | Wilson lower 95 | Coverage | OOS false accept | Cascade delta | L4 call reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| calibration dev | 99.55% | 99.35% | 83.15% | 0.00% | 0.000pp | 83.15% |
| OOS-heavy | 33.33% | 6.15% | 1.64% | 2.00% | +0.546pp | 1.64% |
| validation | 99.51% | 98.98% | 45.74% | 1.00% | +0.032pp | 45.74% |
| locked test | 98.997% | 98.47% | 38.07% | 1.20% | 0.000pp | 38.07% |

Result: failed locked-test accepted precision by 21 wrong accepted rows out of
2,094 accepted rows, and missed the practical 40% coverage target. The remaining
locked accepted errors were dominated by OOS false accepts: 12 of 21 accepted
wrong rows. The validation-derived veto removed `credit_score`, `directions`,
and `spending_history` mistakes from validation, but the locked-test pressure
shifted to families such as `text`, `weather`, `expiration_date`,
`direct_deposit`, and `calories`.

## Stream Confirmation

For the second selected guard:

| Stream | Accepted precision | Coverage | OOS false accept | Cascade delta | L4 call reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| validation sequential | 99.51% | 45.74% | 1.00% | +0.032pp | 45.74% |
| validation uniform | 99.79% | 45.13% | 1.92% | +0.226pp | 45.13% |
| validation zipf-heavy | 99.31% | 37.26% | 0.88% | +0.032pp | 37.26% |
| OOS-heavy diagnostic | 33.33% | 1.64% | 2.00% | +0.546pp | 1.64% |

The stream results support the same diagnosis: the guard can preserve cascade
accuracy, but accepted precision is fragile under distribution shift and the
high-confidence OOS-risk tail is not calibrated well enough.

## Decision

Do not proceed. The strict locked-test criteria were not met:

- accepted precision target: missed at 98.997%;
- OOS false accept target: passed at 1.20%;
- cascade delta target: passed at 0.000pp vs all-L4;
- practical coverage target: missed at 38.07%.

Do not reject CLINC150 yet. The mechanism remains close: validation, uniform
validation, and cascade quality are strong, and the second locked result missed
precision by a very small count. The next repair needs better pre-test evidence
for OOS-risk calibration before another locked-test exposure.

## Next Step

Before any further locked-test run, add a target-local OOS-risk signal that can
be selected without gold or teacher labels at runtime. The likely minimal path
is to expose the L2 model's `out_of_scope` probability or rank among prediction
rows, then select a simple guard such as `threshold + min OOS margin` or
`threshold + max in-scope confidence when OOS probability is high`.

Also add train-derived cross-fold calibration for OOS-heavy slices. A future
candidate should pass multiple train-derived OOS-heavy folds and validation
streams with a stronger precision margin before consuming another locked-test
exposure.

## Validation

Validation results:

```bash
uv run --extra dev pytest tests/targets/nlu/test_clinc150_phase1.py tests/targets/nlu/test_teacher_eval.py tests/targets/nlu/test_l4_teacher.py -q
# 37 passed

uv run --extra dev --extra massive pytest -q
# 301 passed

uv run --extra dev pytest -q
# initial fresh-venv run failed before --extra massive because pandas was absent;
# after optional adapter dependencies were installed in the worktree venv, 301 passed

uv run --extra dev ruff check \
  src/darjeeling/targets/nlu/clinc150_phase1.py \
  src/darjeeling/targets/nlu/main_cli.py \
  tests/targets/nlu/test_clinc150_phase1.py
# passed

git diff --check
# passed
```
