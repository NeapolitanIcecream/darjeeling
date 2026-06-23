# CLINC150 Phase 1 Report

Date: 2026-06-23

## Decision

Update after reliability repair:

CLINC150 `data_full` is **unpaused for Phase 1 continuation**. The repaired
live teacher gate passed with `clinc150-intent-v2-label-cards` on the planned
500-request validation gate.

This report originally rejected CLINC150 for the current Phase 1
mechanism-validation run because the pre-repair live L4 teacher gate did not
pass either allowed prompt. That rejection is superseded by
`docs/experiments/2026-06-23_clinc150_teacher_reliability_report.md`.

The repaired gate clears the all-L4 teacher prerequisite for diagnostic L2,
teacher-distilled L2, and cascade stream experiments. It does not by itself
validate the full Darjeeling Phase 1 mechanism claim.

## Data Source

Processed directory: `data/processed/clinc150_data_full`

Pinned source:

- URL: `https://raw.githubusercontent.com/clinc/oos-eval/828f8093932c8fe6ca7936c3d2e52903b1c523de/data/data_full.json`
- Repository commit: `828f8093932c8fe6ca7936c3d2e52903b1c523de`
- SHA256: `36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0`
- License note: the pinned `clinc/oos-eval` repository LICENSE is Creative
  Commons Attribution 3.0 Unported. The current UCI CLINC150 metadata page lists
  Creative Commons Attribution 4.0 International; this experiment records the
  pinned GitHub source.

Split counts:

| split | total | OOS |
| --- | ---: | ---: |
| train | 15,100 | 100 |
| validation | 3,100 | 100 |
| test | 5,500 | 1,000 |

Mapping:

- In-scope rows: `{"intent": "<label>", "slots": {}, "is_abstain": false}`
- OOS rows: `{"intent": "out_of_scope", "slots": {}, "is_abstain": true}`

## Implementation

Added target-local CLINC150 support only:

- `src/darjeeling/targets/nlu/adapters/clinc150.py`
- `src/darjeeling/targets/nlu/clinc150_phase1.py`
- `edge-mvp-nlu clinc150 prepare`
- `edge-mvp-nlu clinc150 teacher-gate`
- `edge-mvp-nlu clinc150 teacher-eval`
- `edge-mvp-nlu clinc150 teacher-repeat`
- `edge-mvp-nlu clinc150 teacher-metrics`

CLINC-specific source parsing, OOS mapping, label-card prompts, OOS metrics, gate
thresholds, and L2 diagnostic helpers stay in the NLU target. Darjeeling core was
not changed.

## Original Pre-Repair Teacher Gate

Gate sample:

- 500 validation requests
- 3 validation requests per in-scope intent: 450 total
- 50 validation OOS requests
- Same request list for both prompt versions

Model/settings:

- `OPENAI_MODEL=gpt-5.5`
- `OPENAI_TIMEOUT_S=120`
- Default retry policy for the completed gate
- Strict JSON object with exactly one allowed `intent`
- Unknown labels, empty responses, extra fields, parse failures, and schema
  failures counted as hard failures

Artifacts:

- `runs/clinc150-phase1-20260623/teacher-gate-500/clinc150_teacher_gate_comparison.json`
- `runs/clinc150-phase1-20260623/teacher-gate-500/clinc150-intent-v1/`
- `runs/clinc150-phase1-20260623/teacher-gate-500/clinc150-intent-v2-label-cards/`

Results:

| prompt | requests | parsed | overall acc. | in-scope acc. | parse/schema failure | OOS precision | OOS recall | tokens/request | cost | p95 latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `clinc150-intent-v1` | 500 | 429 | 82.4% | 86.2% | 14.2% | 92.3% | 48.0% | 691.9 | $0.1592 | 18,951 ms |
| `clinc150-intent-v2-label-cards` | 500 | 441 | 87.2% | 90.0% | 11.8% | 96.9% | 62.0% | 5,417.0 | $0.3377 | 19,214 ms |

Gate targets:

| target | required | v1 | v2 |
| --- | ---: | ---: | ---: |
| in-scope accuracy | >= 97% | 86.2% | 90.0% |
| overall accuracy | >= 95% | 82.4% | 87.2% |
| parse/schema failure | <= 0.5% | 14.2% | 11.8% |

No prompt passed. No teacher was locked.

## Failure Analysis

In the original pre-repair run, the largest failure source was live teacher
reliability, not parsed label accuracy alone:

- v1 parse/schema failures: 71/500, including 70 empty-response failures and 1
  timeout.
- v2 parse/schema failures: 59/500, including 58 empty-response failures and 1
  connection error.
- Among parsed v2 rows, only 5/441 were wrong.

Representative parsed v2 misses:

| request | gold | teacher |
| --- | --- | --- |
| `you need all five answers` | `current_location` | `out_of_scope` |
| `what's the best restaurant in arizona for pizza` | `travel_suggestion` | `restaurant_suggestion` |
| `what's the definition of nuclear engineering` | `out_of_scope` | `definition` |
| `what is naval engineering` | `out_of_scope` | `definition` |
| `what are black holes` | `out_of_scope` | `definition` |

The label-card prompt improved parsed accuracy and OOS recall, but it increased
tokens/request by about 7.8x and still had a p95 latency near 19 seconds. A
low-concurrency, higher-retry v2 rerun was attempted to separate prompt quality
from concurrent API instability, but it did not complete in a bounded time and
was interrupted before writing artifacts. That supports treating live teacher
latency/reliability as a blocker rather than continuing the experiment.

Post-run root-cause note:

- The CLINC teacher implementation capped `max_completion_tokens` at 64.
- Successful rows were already close to that cap: v1 success p95 completion
  tokens was about 60, and v2 success p95 was about 62.
- For reasoning models, reasoning tokens share the completion budget with
  visible JSON output. A 64-token cap can therefore yield empty visible content
  even when the model could classify the request.

This means the empty-response failures should be treated first as a teacher-call
configuration defect, not as evidence that CLINC150 itself is unsuitable. The
cap has been removed in follow-up code; the teacher gate should be rerun before
making a final CLINC150 benchmark decision.

## Reliability Repair Rerun

Run root: `runs/clinc150-teacher-reliability-20260623`

Report:
`docs/experiments/2026-06-23_clinc150_teacher_reliability_report.md`

Cost ledger:
`docs/experiments/2026-06-23_clinc150_teacher_reliability_cost_ledger.json`

The repaired path added attempt-level diagnostics, incremental details JSONL
writes, CLINC150 live-eval resume, and observed-attempt cost accounting. Paid
runs used `TEACHER_MAX_TOKENS=256`, `OPENAI_TIMEOUT_S=120`, and bounded
project-level retries.

500-request teacher gate results:

| prompt | parsed | overall acc. | in-scope acc. | parse/schema failure | empty attempts | final empty failures | retry recovered rows | cost | passed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `clinc150-intent-v1` | 499/500 | 92.8% | 94.4% | 0.2% | 5 | 0 | 146 | $0.1996008 | no |
| `clinc150-intent-v2-label-cards` | 500/500 | 97.4% | 98.4% | 0.0% | 0 | 0 | 31 | $0.3982408 | yes |

The v2 label-card prompt passed the planned teacher-gate thresholds:

- overall accuracy >= 95%;
- in-scope accuracy >= 97%;
- parse/schema failure <= 0.5%.

Observed spend for the reliability repair, including two bounded smoke runs and
the 500-row gate, was `$0.6994568`, below the `$10` repair budget.

## L2 And Stream Phases

Not run.

The plan made a reliable all-L4 teacher/fallback a prerequisite for L2
distillation and stream cascade claims. Because both allowed teacher prompts
missed the gate by a large margin, running teacher-distilled L2, L2+L4 fallback,
full validation, locked test, uniform replay, sequential learning, or zipf-heavy
replay would not answer the intended mechanism question. It would instead
measure behavior under an unreliable teacher.

Gold-trained diagnostic L2 was also not run as a final result because the plan's
stop condition triggered before Phase C. Target-local helper code exists for a
future diagnostic ceiling, but no diagnostic number is reported here.

## Conclusion

The original pre-repair run did **not** validate the Phase 1 mechanism claim.
The repaired teacher gate now supports continuing CLINC150 Phase 1, but the
mechanism claim still requires diagnostic L2, teacher-distilled L2, cascade, and
stream evidence.

Supported conclusion:

> CLINC150 `data_full` has a low implementation-cost target-local path, and the
> repaired `gpt-5.5` v2 label-card CLINC teacher setup now passes the required
> reliability gate. Darjeeling may continue to diagnostic L2 and teacher-distilled
> L2, but should not claim L2 externalization or L4 cost reduction until the
> later phases pass.

Recommended next step:

Continue to diagnostic L2 and cascade phases with `clinc150-intent-v2-label-cards`
as the locked teacher prompt. Do not continue open-ended CLINC prompt search, do
not resume MASSIVE prompt search, and do not promote CLINC-specific labels, OOS
rules, thresholds, or metrics into core.

## Commands Run

```bash
uv run python -m darjeeling.targets.nlu.main_cli clinc150 prepare --out data/processed/clinc150_data_full

OPENAI_TIMEOUT_S=120 uv run python -m darjeeling.targets.nlu.main_cli clinc150 teacher-gate \
  --out-dir runs/clinc150-phase1-20260623/teacher-gate-500 \
  --data-dir data/processed/clinc150_data_full \
  --max-workers 8

OPENAI_TIMEOUT_S=120 OPENAI_MAX_RETRIES=6 uv run python -m darjeeling.targets.nlu.main_cli clinc150 teacher-gate \
  --out-dir runs/clinc150-phase1-20260623/teacher-gate-500-v2-retry \
  --data-dir data/processed/clinc150_data_full \
  --prompt-version clinc150-intent-v2-label-cards \
  --max-workers 2
```

The retry command was interrupted with exit code 130 before it wrote artifacts.

Focused verification before the live run:

```bash
uv run pytest tests/targets/nlu/test_clinc150_adapter.py tests/targets/nlu/test_clinc150_phase1.py tests/targets/nlu/test_l4_teacher.py -q
uv run pytest tests/targets/nlu/test_experiment_suite_cli.py tests/targets/nlu/test_teacher_eval.py tests/targets/nlu/test_l4_teacher.py tests/targets/nlu/test_target_core_boundary.py tests/test_target_boundary.py tests/test_core_contracts.py -q
uv run ruff check src/darjeeling/targets/nlu/adapters/clinc150.py src/darjeeling/targets/nlu/clinc150_phase1.py src/darjeeling/targets/nlu/teacher.py src/darjeeling/targets/nlu/layers/l4_cloud_llm.py src/darjeeling/targets/nlu/main_cli.py tests/targets/nlu/test_clinc150_adapter.py tests/targets/nlu/test_clinc150_phase1.py tests/targets/nlu/test_l4_teacher.py
```
