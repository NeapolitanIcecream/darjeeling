# Teacher Prompt Search Report

Date: 2026-06-22

Run root: `runs/teacher-prompt-search-20260622`

Cost ledger: `runs/teacher-prompt-search-20260622/cost_ledger.json`

## Decision

Keep `teacher-v1` as the default prompt after this search.

`teacher-v7-evidence-stable` is the best new candidate and should remain
experimental only. It produced repeatable frame-exact gains on two 200-request
validations, but the uniform-stream validation had worse intent accuracy, slot
metrics, parse reliability, cost, and p95 latency. That is not clean enough for
a default change.

Reject `teacher-v5-value-copy`, `teacher-v6-schema-checklist`, and
`teacher-v8-evidence-compact` as adoption candidates. Keep
`teacher-v4-slot-evidence` as a weaker experimental backup only; it improved the
sequential medium sample but was dominated by v7 there and had unstable pilot
behavior.

## Objective

The objective of this second iteration was to run a broader low-complexity
teacher prompt search beyond the prior v2/v3 pilots, with enough paid live
evidence to decide whether to adopt a new teacher prompt or keep `teacher-v1`
after sufficient search.

The prior evidence was insufficient because it rejected bad candidates
(`teacher-v2-intent-first`) and kept `teacher-v3-slot-conservative`
experimental, but did not test enough plausible prompt hypotheses to justify a
final default decision.

## Prompt Hypotheses

| Prompt | Hypothesis | Expected mechanism | Main risk | Result |
| --- | --- | --- | --- | --- |
| `teacher-v4-slot-evidence` | Require specific slot evidence beyond the chosen intent. | Reduce extra slot keys that restate intent-implied objects/actions. | Missing real slots; latency from longer prompt. | Improved first pilot and sequential medium, but unstable and below v7. |
| `teacher-v5-value-copy` | Emphasize exact utterance-span slot values. | Reduce capitalization, normalization, and value mismatch errors. | Keep extra slot keys while only changing values. | Rejected in pilot; worse frame exact and higher cost than v1. |
| `teacher-v6-schema-checklist` | Use a compact private checklist inside one JSON-only prompt. | Balance evidence discipline and slot recall. | More parse failures and lower intent stability. | Rejected as default candidate; slot F1 improved in pilot but parse/intent regressed. |
| `teacher-v7-evidence-stable` | Iterate v4 to keep intent stable when slots are uncertain and suppress filler time/date slots. | Preserve v4 frame gains while reducing v4 pilot regressions. | Still longer/slower; may miss slots. | Best candidate, but experimental only due uniform regressions. |
| `teacher-v8-evidence-compact` | Compress v7 wording to lower reliability/latency risk. | Same evidence rules with fewer prompt tokens. | Too terse; loses intent guidance. | Rejected; poor 40-request uniform result. |

## Cost

Budget limit: `$100.00`

Previous observed spend: `$0.1187132`

New observed spend: `$0.5669000`

Total observed spend: `$0.6856132`

Estimated remaining budget: `$99.3143868`

Cost was computed from generated artifacts:

- `teacher_prompt_comparison.json.rows[*].cost_usd`
- `teacher_live_vs_gold.summary.json.full_l4.cost_usd`
- detail JSONL `cost_usd` sums as cross-checks

## Readiness

No-cost readiness passed before paid prompt search:

```bash
uv run pytest tests/targets/nlu/test_l4_teacher.py tests/targets/nlu/test_teacher_eval.py tests/targets/nlu/test_experiment_suite_cli.py tests/targets/nlu/test_nlu_target.py -q
uv run pytest tests/test_core_contracts.py tests/test_target_boundary.py tests/targets/nlu/test_target_core_boundary.py -q
uv run python -m darjeeling.targets.nlu.main_cli teacher eval-live --help
uv run python -m darjeeling.targets.nlu.main_cli teacher compare-prompts --help
```

The unknown prompt-version hard-fail behavior already existed from the prior
iteration and remained covered by focused tests.

## Prior Error Taxonomy

Prior artifacts showed the dominant failure classes:

- `teacher-v1` medium 100-request sample: 27 extra-slot-key requests, 14
  missing-slot-key requests, 14 wrong-slot-value requests, 8 intent misses.
- `teacher-v3-slot-conservative` medium 100-request sample: 22 extra-slot-key
  requests, 17 missing-slot-key requests, 11 wrong-slot-value requests, 8
  intent misses.
- Pairwise v1/v3 on that sample: 50 both correct, 4 v1-only correct, 5
  v3-only correct, 41 neither correct.

This made slot evidence, value copying, and a middle-ground checklist the right
next hypotheses. Possible label ambiguity was treated as a diagnostic bucket for
neither-correct items, especially around close intent-family boundaries; it was
not used as adoption evidence.

## Pilot Results

All pilot runs used `data/processed/massive_en_us`, split `validation`, model
setting `gpt-5.5`, and `--min-frame-exact-match 0.0`.

### First Pilot: 40 Sequential Requests

Artifact: `runs/teacher-prompt-search-20260622/pilot-sequential-40/teacher_prompt_comparison.json`

| Prompt | Frame exact | Intent | Slot key exact | Slot F1 | Parse failures | Cost | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `teacher-v1` | 0.550 | 0.900 | 0.700 | 0.675 | 1 | $0.0152956 | 6,979 |
| `teacher-v4-slot-evidence` | 0.625 | 0.875 | 0.825 | 0.741 | 1 | $0.0179616 | 43,479 |
| `teacher-v5-value-copy` | 0.525 | 0.925 | 0.700 | 0.667 | 0 | $0.0181624 | 5,601 |
| `teacher-v6-schema-checklist` | 0.600 | 0.850 | 0.750 | 0.762 | 2 | $0.0174708 | 21,180 |

Paired outcomes against v1:

| Candidate | Both correct | v1 only | Candidate only | Neither |
| --- | ---: | ---: | ---: | ---: |
| `teacher-v4-slot-evidence` | 20 | 2 | 5 | 13 |

Decision: reject v5; keep v4 as main survivor; keep v6 as backup only.

### Iteration Pilot: 40 Sequential Requests

Artifact: `runs/teacher-prompt-search-20260622/iteration-v7-sequential-40/teacher_prompt_comparison.json`

| Prompt | Frame exact | Intent | Slot key exact | Slot F1 | Parse failures | Cost | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `teacher-v1` | 0.575 | 0.925 | 0.750 | 0.698 | 0 | $0.0156728 | 7,965 |
| `teacher-v4-slot-evidence` | 0.525 | 0.800 | 0.700 | 0.649 | 5 | $0.0157176 | 24,475 |
| `teacher-v7-evidence-stable` | 0.575 | 0.850 | 0.800 | 0.741 | 2 | $0.0175240 | 22,031 |

Paired outcomes against v1:

| Candidate | Both correct | v1 only | Candidate only | Neither |
| --- | ---: | ---: | ---: | ---: |
| `teacher-v4-slot-evidence` | 17 | 6 | 4 | 13 |
| `teacher-v7-evidence-stable` | 20 | 3 | 3 | 14 |

Decision: v7 did not win the pilot primary metric, but it kept the best slot F1
and was the evidence-based iteration. Advance v4 and v7 to medium validation
because pilot evidence was mixed rather than clearly failed.

## Medium Validation

### Sequential 200 Requests

Artifact: `runs/teacher-prompt-search-20260622/medium-sequential-200/teacher_prompt_comparison.json`

| Prompt | Frame exact | Intent | Slot key exact | Slot F1 | Parse failures | Cost | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `teacher-v1` | 0.515 | 0.890 | 0.665 | 0.668 | 4 | $0.0784636 | 18,206 |
| `teacher-v4-slot-evidence` | 0.605 | 0.905 | 0.730 | 0.723 | 3 | $0.0904764 | 11,175 |
| `teacher-v7-evidence-stable` | 0.620 | 0.910 | 0.770 | 0.721 | 2 | $0.0912464 | 7,960 |

Paired outcomes against v1:

| Candidate | Both correct | v1 only | Candidate only | Neither |
| --- | ---: | ---: | ---: | ---: |
| `teacher-v4-slot-evidence` | 97 | 6 | 24 | 73 |
| `teacher-v7-evidence-stable` | 96 | 7 | 28 | 69 |

Sequential taxonomy:

| Prompt | Intent miss | Extra slot keys | Missing slot keys | Wrong slot values | Parse/schema failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| `teacher-v1` | 18 | 57 | 25 | 22 | 4 |
| `teacher-v4-slot-evidence` | 16 | 39 | 28 | 18 | 3 |
| `teacher-v7-evidence-stable` | 16 | 29 | 30 | 21 | 2 |

Sequential validation strongly favored v7.

### Uniform 200 Requests

Artifact: `runs/teacher-prompt-search-20260622/medium-uniform-200-v7/teacher_prompt_comparison.json`

| Prompt | Frame exact | Intent | Slot key exact | Slot F1 | Parse failures | Cost | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `teacher-v1` | 0.470 | 0.865 | 0.630 | 0.659 | 1 | $0.0820740 | 8,752 |
| `teacher-v7-evidence-stable` | 0.515 | 0.825 | 0.625 | 0.645 | 10 | $0.0901800 | 28,268 |

Paired outcomes against v1:

| Candidate | Both correct | v1 only | Candidate only | Neither |
| --- | ---: | ---: | ---: | ---: |
| `teacher-v7-evidence-stable` | 82 | 12 | 21 | 85 |

Uniform taxonomy:

| Prompt | Intent miss | Extra slot keys | Missing slot keys | Wrong slot values | Parse/schema failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| `teacher-v1` | 26 | 64 | 28 | 25 | 1 |
| `teacher-v7-evidence-stable` | 25 | 43 | 41 | 18 | 10 |

Uniform validation repeated the frame-exact gain, but the regression profile is
not acceptable for a default prompt: intent accuracy, slot-key exact, slot F1,
parse reliability, cost, and p95 latency all worsened.

### Compact Follow-Up

Artifact: `runs/teacher-prompt-search-20260622/v8-uniform-40/teacher-v8-evidence-compact/teacher_live_vs_gold.summary.json`

`teacher-v8-evidence-compact` was tested on the same first 40 deterministic
uniform requests used by the v1/v7 uniform validation. It did not repair v7:
frame exact 0.475, intent 0.725, slot-key exact 0.600, slot F1 0.729, five parse
failures, cost `$0.0166548`, and p95 latency 43,874 ms. Paired outcomes against
the existing v1 rows were 15 both correct, 6 v1-only, 4 v8-only, and 15 neither.

Decision: reject v8.

## Code Changes

- Added static prompt versions in `src/darjeeling/targets/nlu/teacher.py`:
  - `teacher-v4-slot-evidence`
  - `teacher-v5-value-copy`
  - `teacher-v6-schema-checklist`
  - `teacher-v7-evidence-stable`
  - `teacher-v8-evidence-compact`
- Extended `tests/targets/nlu/test_l4_teacher.py` so supported prompt versions
  render distinct prompts and unknown versions still fail.

No Darjeeling core code was changed. No L1/L2/L3 tuning or cascade experiment
was run. Gold frames remained benchmark/report-only diagnostics.

## Commands Run

No-cost checks:

```bash
uv run pytest tests/targets/nlu/test_l4_teacher.py tests/targets/nlu/test_teacher_eval.py tests/targets/nlu/test_experiment_suite_cli.py tests/targets/nlu/test_nlu_target.py -q
uv run pytest tests/test_core_contracts.py tests/test_target_boundary.py tests/targets/nlu/test_target_core_boundary.py -q
uv run python -m darjeeling.targets.nlu.main_cli teacher eval-live --help
uv run python -m darjeeling.targets.nlu.main_cli teacher compare-prompts --help
uv run pytest tests/targets/nlu/test_l4_teacher.py tests/targets/nlu/test_teacher_eval.py -q
```

Paid runs:

```bash
OPENAI_TIMEOUT_S=120 uv run python -m darjeeling.targets.nlu.main_cli teacher compare-prompts --out-dir runs/teacher-prompt-search-20260622/pilot-sequential-40 --data-dir data/processed/massive_en_us --split validation --stream sequential --max-requests 40 --prompt-version teacher-v1 --prompt-version teacher-v4-slot-evidence --prompt-version teacher-v5-value-copy --prompt-version teacher-v6-schema-checklist --min-frame-exact-match 0.0
OPENAI_TIMEOUT_S=120 uv run python -m darjeeling.targets.nlu.main_cli teacher compare-prompts --out-dir runs/teacher-prompt-search-20260622/iteration-v7-sequential-40 --data-dir data/processed/massive_en_us --split validation --stream sequential --max-requests 40 --prompt-version teacher-v1 --prompt-version teacher-v4-slot-evidence --prompt-version teacher-v7-evidence-stable --min-frame-exact-match 0.0
OPENAI_TIMEOUT_S=120 uv run python -m darjeeling.targets.nlu.main_cli teacher compare-prompts --out-dir runs/teacher-prompt-search-20260622/medium-sequential-200 --data-dir data/processed/massive_en_us --split validation --stream sequential --max-requests 200 --prompt-version teacher-v1 --prompt-version teacher-v4-slot-evidence --prompt-version teacher-v7-evidence-stable --min-frame-exact-match 0.0
OPENAI_TIMEOUT_S=120 uv run python -m darjeeling.targets.nlu.main_cli teacher compare-prompts --out-dir runs/teacher-prompt-search-20260622/medium-uniform-200-v7 --data-dir data/processed/massive_en_us --split validation --stream uniform --max-requests 200 --prompt-version teacher-v1 --prompt-version teacher-v7-evidence-stable --min-frame-exact-match 0.0
OPENAI_TIMEOUT_S=120 uv run python -m darjeeling.targets.nlu.main_cli teacher eval-live --out-dir runs/teacher-prompt-search-20260622/v8-uniform-40/teacher-v8-evidence-compact --data-dir data/processed/massive_en_us --split validation --stream uniform --max-requests 40 --prompt-version teacher-v8-evidence-compact --min-frame-exact-match 0.0
```

## Final Recommendation

Do not change the default teacher prompt in this pass.

The search was broad enough to support that decision: three new prompt
candidates were piloted, one was iterated into v7, the best survivor reached two
200-request validations, and a compact repair attempt was tested after the
second validation exposed reliability/latency regressions.

`teacher-v7-evidence-stable` is worth keeping for future prompt work because it
reduced extra slot keys and improved frame exact on both medium samples. It is
not default-ready because the uniform validation showed too many parse failures,
weaker secondary metrics, higher cost, and much worse p95 latency.
