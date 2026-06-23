# CLINC150 Teacher Reliability Repair Report

Date: 2026-06-23

Run root: `runs/clinc150-teacher-reliability-20260623`

Cost ledger summary:
`docs/experiments/2026-06-23_clinc150_teacher_reliability_cost_ledger.json`

Detailed run ledgers:

- `runs/clinc150-teacher-reliability-20260623/smoke-v2-50/cost_ledger.json`
- `runs/clinc150-teacher-reliability-20260623/smoke-v2-50-lowconcurrency/cost_ledger.json`
- `runs/clinc150-teacher-reliability-20260623/teacher-gate-500/cost_ledger.json`

## Decision

Continue CLINC150 Phase 1.

The repaired live teacher path is now reliable enough to trust for the Phase 1
teacher gate. The locked prompt for the next CLINC150 phase should be
`clinc150-intent-v2-label-cards`.

This does not validate the full Darjeeling Phase 1 mechanism. It only clears the
teacher prerequisite that blocked diagnostic L2, teacher-distilled L2, and
cascade stream work.

## Code Repair

Implemented in the NLU target and teacher eval harness:

- attempt-level diagnostics for live OpenAI teacher calls;
- empty response retries remain project-level retries with SDK retries disabled;
- row-level final response cost and total attempt cost;
- incremental details JSONL writes with flush after each completed row;
- run identity manifests for paid live evals;
- CLINC150 `teacher-gate` and `teacher-eval` `--resume-existing`;
- cost ledgers generated from detail JSONL rows and attempt diagnostics.

No Darjeeling core target semantics were added.

## No-Cost Validation

```bash
uv run pytest tests/targets/nlu/test_l4_teacher.py tests/targets/nlu/test_teacher_eval.py tests/targets/nlu/test_clinc150_phase1.py -q
uv run pytest -q
uv run ruff check src/darjeeling/targets/nlu/layers/l4_cloud_llm.py src/darjeeling/targets/nlu/teacher_eval.py src/darjeeling/targets/nlu/clinc150_phase1.py src/darjeeling/targets/nlu/main_cli.py tests/targets/nlu/test_l4_teacher.py tests/targets/nlu/test_teacher_eval.py tests/targets/nlu/test_clinc150_phase1.py
git diff --check
```

Result: focused tests passed, full suite passed with 293 tests, touched-file
ruff passed, and `git diff --check` passed.

## Paid Runs

All paid runs used `OPENAI_MODEL=gpt-5.5`,
`TEACHER_MAX_TOKENS=256`, `OPENAI_TIMEOUT_S=120`, and project-level retries.

| run | prompt | requests | workers | retries | parsed | overall acc. | parse/schema failure | retry recovered rows | empty attempts | final empty failures | cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | `clinc150-intent-v2-label-cards` | 50 | 2 | 3 | 49 | 98.0% | 2.0% | 25 | 0 | 0 | $0.0572396 |
| smoke low-concurrency | `clinc150-intent-v2-label-cards` | 50 | 1 | 6 | 50 | 100.0% | 0.0% | 4 | 0 | 0 | $0.0443756 |
| gate | `clinc150-intent-v1` | 500 | 2 | 6 | 499 | 92.8% | 0.2% | 146 | 5 | 0 | $0.1996008 |
| gate | `clinc150-intent-v2-label-cards` | 500 | 2 | 6 | 500 | 97.4% | 0.0% | 31 | 0 | 0 | $0.3982408 |

Total observed spend for this repair: **$0.6994568**.

## Teacher Gate Result

| prompt | overall acc. | in-scope acc. | parse/schema failure | OOS precision | OOS recall | passed |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `clinc150-intent-v1` | 92.8% | 94.4% | 0.2% | 86.7% | 78.0% | no |
| `clinc150-intent-v2-label-cards` | 97.4% | 98.4% | 0.0% | 91.7% | 88.0% | yes |

The v2 label-card prompt passed all planned gate thresholds:

- in-scope accuracy >= 97%;
- overall accuracy >= 95%;
- parse/schema failure <= 0.5%.

## Failure Analysis

The previous empty-response failure mode was repaired:

- The 500-row v2 gate had 0 empty attempts and 0 final empty failures.
- The 500-row v1 gate had 5 empty attempts, all recovered by retry.
- The only v1 final parse/schema failure was a connection error, not an empty
  response.

Connection instability remains visible but bounded by retries:

- Initial 50-row v2 smoke at two workers had 39 connection-error attempts and 1
  final connection failure.
- Low-concurrency v2 smoke reduced final failures to 0.
- The full v2 gate at two workers and six retries had 31 recovered retry rows
  and 0 final failures.

Representative v2 semantic misses:

| request | gold | teacher |
| --- | --- | --- |
| `you need all five answers` | `current_location` | `out_of_scope` |
| `may you stop a paymet on my account` | `freeze_account` | `out_of_scope` |
| `do i have any meetings on my calendar today` | `meeting_schedule` | `calendar` |
| `what's the status of my vacation days` | `pto_request_status` | `pto_balance` |
| `what's the best restaurant in arizona for pizza` | `travel_suggestion` | `restaurant_suggestion` |
| `show me recent activity in my backyard` | `out_of_scope` | `smart_home` |

## Next Step

Proceed to the CLINC150 diagnostic L2 ceiling and then teacher-distilled L2
experiments using the locked v2 label-card teacher traces. Continue to record
attempt diagnostics and use resume for paid live paths.
