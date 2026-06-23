# Teacher Eval Iteration Report

Date: 2026-06-22

Run root: `runs/teacher-eval-iteration-20260622`

## Recommendation

Keep `teacher-v1` as the default teacher prompt.

Reject `teacher-v2-intent-first` as an adoption candidate. On the pilot sample it was worse on frame exact match, slot key exact match, parse reliability, latency, and cost.

Keep `teacher-v3-slot-conservative` as an experiment only. It slightly improved medium-sample frame exact match by 1 request out of 100 and improved slot pair F1, but did not improve intent accuracy, slightly lowered slot key exact match, and cost more. That is not strong enough to adopt as the default prompt.

## Code Changes

- `ensure_supported_teacher_prompt_version()` now rejects unknown teacher prompt versions using the static supported-version list.
- Added `teacher-v3-slot-conservative`, a single-call NLU target prompt that discourages inferred slots.
- Teacher eval now records per-request teacher call failures as failed rows with zero observed cost when no usage is returned, so a long paid benchmark preserves completed rows instead of losing the whole artifact.

No core framework, generic prompt registry, evaluator framework, or cascade/weak-layer tuning was added.

## Cost Ledger

Ledger: `runs/teacher-eval-iteration-20260622/cost_ledger.json`

Actual observed spend: **$0.1187132**

Estimated remaining budget: **$99.8812868**

Cost was computed from generated artifacts:

- `teacher_prompt_comparison.json.rows[*].cost_usd`
- `teacher_live_vs_gold.summary.json.full_l4.cost_usd`
- Detail JSONL cost sums were used as cross-checks.

One failed paid run is included: `cycle-3-v3-medium-validation` completed the v1 100-request artifact, then failed during the v3 half with an API timeout. Its observed cost is the completed v1 artifact cost, `$0.0393916`.

## No-Cost Readiness

Passed before paid runs:

```bash
uv run pytest tests/targets/nlu/test_l4_teacher.py::test_teacher_prompt_version_rejects_unknown_version -q
uv run pytest tests/targets/nlu/test_nlu_target.py::test_nlu_teacher_adapter_builds_prompt_and_parses_frame tests/targets/nlu/test_l4_teacher.py tests/targets/nlu/test_teacher_eval.py tests/targets/nlu/test_experiment_suite_cli.py -q
uv run pytest tests/test_core_contracts.py tests/test_target_boundary.py tests/targets/nlu/test_target_core_boundary.py -q
uv run python -m darjeeling.targets.nlu.main_cli teacher eval-live --help
uv run python -m darjeeling.targets.nlu.main_cli teacher compare-prompts --help
uv run ruff check src/darjeeling/targets/nlu/teacher.py tests/targets/nlu/test_l4_teacher.py tests/targets/nlu/test_nlu_target.py
```

## Paid Runs

All paid runs used `data/processed/massive_en_us`, split `validation`, stream `sequential`, and model setting `gpt-5.5`.

| Cycle | Artifact | Prompt | Requests | Frame exact | Intent acc. | Slot key exact | Slot pair F1 | Parse/schema failures | Cost | p95 ms | Decision |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 pilot | `cycle1-pilot/teacher_prompt_comparison.json` | `teacher-v1` | 20 | 0.60 | 0.95 | 0.75 | 0.700 | 0 | $0.0078308 | 6808 | baseline |
| 1 pilot | `cycle1-pilot/teacher_prompt_comparison.json` | `teacher-v2-intent-first` | 20 | 0.45 | 0.95 | 0.55 | 0.667 | 1 | $0.0108344 | 20536 | reject |
| 2 pilot | `cycle2-v3-pilot/teacher_prompt_comparison.json` | `teacher-v1` | 20 | 0.55 | 1.00 | 0.75 | 0.619 | 0 | $0.0078996 | 5981 | baseline |
| 2 pilot | `cycle2-v3-pilot/teacher_prompt_comparison.json` | `teacher-v3-slot-conservative` | 20 | 0.55 | 0.95 | 0.70 | 0.732 | 0 | $0.0087204 | 9433 | validate medium |
| 3 medium | `cycle3-v3-medium/teacher-v1/teacher_live_vs_gold.summary.json` | `teacher-v1` | 100 | 0.54 | 0.92 | 0.68 | 0.615 | 0 | $0.0393916 | 6683 | baseline |
| 3 medium | `cycle3-v3-medium/teacher-v3-slot-conservative/teacher_live_vs_gold.summary.json` | `teacher-v3-slot-conservative` | 100 | 0.55 | 0.92 | 0.67 | 0.667 | 0 | $0.0440364 | 6748 | experiment only |

## Failure Modes

Cycle 1 showed that v2's main regression was slot over-extraction, not intent selection:

- v1: 8 frame misses, 1 intent miss, 5 slot-key misses, 0 parse failures.
- v2: 11 frame misses, 1 intent miss, 9 slot-key misses, 1 empty-response parse failure.

Cycle 3 medium validation showed v3 reduced some slot over-extraction and slot-value errors but shifted some errors into missing slots:

- v1: 27 requests with extra slot keys, 14 with missing slot keys, 14 with slot-value errors, 8 intent misses.
- v3: 22 requests with extra slot keys, 17 with missing slot keys, 11 with slot-value errors, 8 intent misses.
- Pairwise frame exact: 50 requests both correct, 4 v1-only correct, 5 v3-only correct, 41 neither correct.

This supports the hypothesis that conservative slot wording changes slot behavior, but it does not produce a meaningful primary-metric gain.

## Final Decision

The 0622 intent-first teacher prompt change is not effective enough to adopt and should be rejected as a default.

The slot-conservative prompt is worth retaining only as an experimental prompt variant for future slot-focused work. The default should remain `teacher-v1` until a prompt variant improves frame exact match materially without unacceptable cost or latency regression.
