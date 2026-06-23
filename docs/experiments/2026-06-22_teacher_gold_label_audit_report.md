# Teacher Gold Label Audit Report

Date: 2026-06-22

Run root: `runs/teacher-gold-label-audit-20260622`

## Decision

The audit supports a **mixed causes** conclusion, with dataset convention mismatch
as the largest primary bucket and strict exact-match artifacts as a second large
bucket.

Teacher quality is not the dominant blocker by itself. In the audited sample,
only 16/83 rows were clear teacher mistakes against reasonable gold labels.
Rows where the teacher was semantically reasonable but missed MASSIVE-specific
label conventions, plus exact-match artifacts, account for 44/83 rows. Gold
ambiguity/noise is material but not dominant at 13/83 rows when combining
`gold_questionable` and `ambiguous_multiple_valid`.

It is plausible that a target-local L2 trained on gold or teacher-corrected
convention examples could beat live L4 on this benchmark, because many failures
are learnable target conventions: redundant generic slots, relation-vs-person
slot keys, date/time split boundaries, semicolon pair delimiters, casing, and
span normalization. That evidence should be obtained in target-local NLU
experiments only; no MASSIVE convention should be moved into Darjeeling core,
promotion, replay, or generic candidate selection.

## Artifacts Audited

Primary input details:

- `runs/teacher-prompt-search-20260622/medium-sequential-200/teacher-v1/teacher_live_vs_gold.details.jsonl`
- `runs/teacher-prompt-search-20260622/medium-sequential-200/teacher-v7-evidence-stable/teacher_live_vs_gold.details.jsonl`
- `runs/teacher-prompt-search-20260622/medium-uniform-200-v7/teacher-v1/teacher_live_vs_gold.details.jsonl`
- `runs/teacher-prompt-search-20260622/medium-uniform-200-v7/teacher-v7-evidence-stable/teacher_live_vs_gold.details.jsonl`

Generated audit artifacts:

- `runs/teacher-gold-label-audit-20260622/audit_sample.jsonl`
- `runs/teacher-gold-label-audit-20260622/audit_labels.jsonl`
- `runs/teacher-gold-label-audit-20260622/audit_summary.json`
- helper: `runs/teacher-gold-label-audit-20260622/build_audit_artifacts.py`

No paid live calls were made for this audit.

## Metric Sanity Check

The recomputed metrics match the prompt-search report.

| stream/prompt | rows | unique ids | frame exact | intent acc. | slot key exact | slot F1 | parse failures | cost | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sequential v1 | 200 | 200 | 0.515 | 0.890 | 0.665 | 0.668 | 4 | $0.078464 | 18,206 |
| sequential v7 | 200 | 200 | 0.620 | 0.910 | 0.770 | 0.721 | 2 | $0.091246 | 7,960 |
| uniform v1 | 200 | 186 | 0.470 | 0.865 | 0.630 | 0.659 | 1 | $0.082074 | 8,752 |
| uniform v7 | 200 | 186 | 0.515 | 0.825 | 0.625 | 0.645 | 10 | $0.090180 | 28,268 |

Paired outcomes:

| stream | counting | both correct | v1 only | v7 only | both wrong |
| --- | --- | ---: | ---: | ---: | ---: |
| sequential | raw and unique | 96 | 7 | 28 | 69 |
| uniform | raw | 82 | 12 | 21 | 85 |
| uniform | first unique request id | 75 | 11 | 19 | 81 |

## Sampling And Duplicate Handling

The sample was deterministic and globally de-duplicated by `request_id`. Main
buckets were filled first in stream order, then parse/API failure rows were
topped up. No bucket had a shortfall.

| bucket | sampled unique rows |
| --- | ---: |
| sequential rows where v1 and v7 are both wrong | 33 |
| uniform rows where v1 and v7 are both wrong | 20 |
| v7 correct and v1 wrong | 15 |
| v1 correct and v7 wrong | 15 |
| parse/API failure rows | 10 |
| span/case/punctuation candidates | 20 |

The final audit contains 83 unique request ids. Some rows satisfy multiple
buckets, so bucket counts sum above 83.

The uniform stream sampled with replacement: 200 raw rows contained 186 unique
request ids and 14 duplicate request ids. Four duplicate request ids had
non-identical live teacher outputs or outcomes across repeated calls:
`validation-509`, `validation-1807`, `validation-1884`, and `validation-712`.
Those repeated rows were not counted as independent label evidence.

## Taxonomy

Each audited row received exactly one primary class:

- `teacher_clear_error`
- `teacher_reasonable_dataset_convention`
- `gold_questionable`
- `ambiguous_multiple_valid`
- `exact_match_artifact`
- `infrastructure_failure`
- `unclear`

Secondary tags record recurring mechanisms such as `extra_slot`,
`missing_slot`, `slot_value_span`, `slot_key_boundary`, `relation_vs_person`,
`date_vs_time`, `music_likeness_vs_play_music`, and
`case_or_punctuation`.

## Audit Counts

| primary class | count | share |
| --- | ---: | ---: |
| `teacher_reasonable_dataset_convention` | 29 | 34.9% |
| `teacher_clear_error` | 16 | 19.3% |
| `exact_match_artifact` | 15 | 18.1% |
| `infrastructure_failure` | 10 | 12.0% |
| `gold_questionable` | 7 | 8.4% |
| `ambiguous_multiple_valid` | 6 | 7.2% |
| `unclear` | 0 | 0.0% |

Frequent secondary tags:

| tag | count |
| --- | ---: |
| `extra_slot` | 21 |
| `slot_value_span` | 19 |
| `intent_boundary` | 16 |
| `missing_slot` | 14 |
| `slot_key_boundary` | 12 |
| `case_or_punctuation` | 7 |
| `relation_vs_person` | 4 |

## Representative Examples

| class | request | evidence |
| --- | --- | --- |
| `teacher_reasonable_dataset_convention` | `validation-0`: "turn the lights off please" | v1 adds `device_type=lights`; gold keeps the frame slotless because the intent already encodes lights. |
| `teacher_reasonable_dataset_convention` | `validation-1568`: "schedule a meeting with my colleague" | teachers use `person=my colleague`; gold uses `relation=colleague`. |
| `exact_match_artifact` | `validation-1643`: "what is the exchange rate of u. s. d. to cad" | teachers preserve the currency pair but not gold's `u. s. d. ; cad` delimiter convention. |
| `exact_match_artifact` | `validation-71`: "please i want to hear we will rock you from queen" | v1 title-cases artist/song; v7 matches gold casing. |
| `teacher_clear_error` | `validation-118`: "find out if the olive garden will let me do takeaway" | v7 omits explicit `order_type=takeaway`; v1 and gold include it. |
| `teacher_clear_error` | `validation-1445`: "where is the train station" | both teachers choose recommendation/location intent instead of the gold transport query. |
| `gold_questionable` | `validation-1664`: "can you give me the exchange rate for dollar in rupees" | gold is empty `qa_factoid`, while both teachers choose `qa_currency`, which is more plausible. |
| `ambiguous_multiple_valid` | `validation-68`: "email me the lyrics to this song" | both email action and music-query readings are defensible without more annotation guidance. |
| `infrastructure_failure` | `validation-1760`: "what is the stock value of google" | v7 failed with a connection error while v1 matched gold. |

## Interpretation

The low teacher-vs-gold exact match is not just weak semantic parsing. A large
share comes from target/dataset conventions that a generic live L4 prompt has
not been taught: when to omit slots already implied by the intent, when to use
`relation` instead of `person`, how to split time/date fields, how to represent
paired values with semicolons, and how narrow slot spans should be.

Strict exact match also over-penalizes benign differences. Casing, possessives,
generic nouns, delimiters, and span boundaries explain 15/83 audited rows as the
primary issue and appear as secondary tags in more rows.

Clear teacher errors remain material. The 16 clear errors include missing
explicit slots, wrong slot keys, and intent boundaries where gold is reasonable.
Those errors are enough to justify continued target-local model/prompt work, but
they do not support the conclusion that live L4 semantic quality is the sole or
dominant blocker.

Infrastructure is also visible, especially for v7 on uniform validation. Ten
audited rows had connection errors or empty responses as the primary cause; this
supports keeping v7 experimental despite its frame-exact gains.

Gold ambiguity/noise is material but bounded in this sample. Seven rows had
questionable gold labels and six were genuinely ambiguous. This is large enough
to affect how exact-match failures are interpreted, but not large enough to
discard the benchmark.

## L2 Implications

A target-local L2 trained on gold or corrected convention examples could
plausibly beat live L4 on MASSIVE because many errors are stable, local
conventions rather than open-ended reasoning failures. The right test is a
held-out target-local benchmark that:

- trains only inside the NLU target/adapters;
- keeps gold out of Darjeeling core, compiler inputs, replay, and promotion;
- reports live L4, L2, and final-frame comparisons against held-out gold;
- includes report-only normalized diagnostics alongside strict exact match;
- tracks duplicate request ids and parse/API failures separately.

This audit does not prove L2 will win; it shows the error distribution is
compatible with a convention-trained L2 advantage.

## Recommendations

1. Keep `teacher-v1` as the default prompt for now. The audit does not overturn
   the prompt-search decision.
2. Add or run target-local diagnostic metrics for normalized slot value match,
   slot-key boundary errors, redundant generic slots, and relation/person
   mismatches. Keep these report-only.
3. Run a small target-local L2 convention experiment on held-out MASSIVE data,
   with gold used only inside the target benchmark/training boundary.
4. Investigate v7 reliability before any future prompt adoption. Its empty
   responses and latency regressions are real benchmark blockers.
5. For suspected gold noise, create a small reviewed list of examples rather
   than changing benchmark defaults. The audit examples are evidence, not a
   license to rewrite core rules.

## Commands Run

```bash
python3 runs/teacher-gold-label-audit-20260622/build_audit_artifacts.py
wc -l runs/teacher-gold-label-audit-20260622/audit_sample.jsonl runs/teacher-gold-label-audit-20260622/audit_labels.jsonl
python3 -m json.tool runs/teacher-gold-label-audit-20260622/audit_summary.json >/dev/null
uv run pytest tests/targets/nlu/test_teacher_eval.py tests/targets/nlu/test_l4_teacher.py -q
git diff --check
```

Results:

- `audit_sample.jsonl`: 83 rows.
- `audit_labels.jsonl`: 83 rows.
- `audit_summary.json`: valid JSON.
- Focused NLU tests: `16 passed in 1.36s`.
- `git diff --check`: passed.
