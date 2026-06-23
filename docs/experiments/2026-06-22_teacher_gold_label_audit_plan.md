# Teacher Gold Label Audit Plan

Date: 2026-06-22

Purpose: audit whether the low live teacher-vs-gold exact-match scores are mainly caused by teacher semantic mistakes, MASSIVE label convention mismatch, possible gold-label noise, or strict exact-match artifacts. This plan is the source of truth for the next agent. Do not rely on chat/session history.

## Required Context Files

Read these before doing analysis:

- `AGENTS.md`
- `docs/experiments/2026-06-22_gpt55_pro_review_verification.md`
- `docs/experiments/2026-06-22_teacher_eval_iteration_report.md`
- `docs/experiments/2026-06-22_teacher_prompt_search_report.md`
- `runs/teacher-prompt-search-20260622/cost_ledger.json`

Primary artifacts to inspect:

- `runs/teacher-prompt-search-20260622/medium-sequential-200/teacher-v1/teacher_live_vs_gold.details.jsonl`
- `runs/teacher-prompt-search-20260622/medium-sequential-200/teacher-v7-evidence-stable/teacher_live_vs_gold.details.jsonl`
- `runs/teacher-prompt-search-20260622/medium-uniform-200-v7/teacher-v1/teacher_live_vs_gold.details.jsonl`
- `runs/teacher-prompt-search-20260622/medium-uniform-200-v7/teacher-v7-evidence-stable/teacher_live_vs_gold.details.jsonl`

Also inspect:

- `src/darjeeling/targets/nlu/streams.py`
- `src/darjeeling/targets/nlu/teacher_eval.py`
- `src/darjeeling/targets/nlu/reports.py`

## Current State

The prompt search found that `teacher-v7-evidence-stable` improves frame exact match over `teacher-v1` on two 200-request validations, but it is not default-ready because the uniform validation regressed intent accuracy, slot F1, parse failures, cost, and p95 latency.

The main unresolved question is whether the low teacher-vs-gold exact match means:

1. the live L4 teacher is genuinely weak on NLU frame parsing;
2. the teacher is semantically reasonable but does not know MASSIVE's label conventions;
3. some MASSIVE gold labels are ambiguous or questionable;
4. strict exact match is over-penalizing harmless span, casing, normalization, or slot-boundary differences;
5. some errors are infrastructure parse/API failures rather than labeling quality.

This audit should answer that question with concrete evidence before more prompt search or L2-evolution conclusions are drawn.

## Desired Decision

The final report must support one of these conclusions:

- **Teacher quality is the dominant blocker**: most audited failures are clear teacher mistakes against reasonable gold labels.
- **Dataset convention mismatch is the dominant blocker**: many failures are semantically acceptable but do not match MASSIVE-specific intent/slot conventions.
- **Gold ambiguity/noise is material**: a non-trivial share of failures have questionable or ambiguous gold labels.
- **Mixed causes**: no single cause dominates; recommend the next experiment based on the largest actionable bucket.

The report should also state whether it is plausible that a target-local L2 trained on gold or teacher-corrected convention examples could beat live L4 on this benchmark, and what evidence would be needed to test that without leaking gold into Darjeeling core.

## Constraints

- Do not change Darjeeling core.
- Do not change default teacher prompt.
- Do not tune L1, L2, or L3 in this audit.
- Do not add dataset-specific rules to core, replay, promotion, or candidate selection.
- Gold frames may be used only for benchmark/report diagnostics and this audit.
- Keep raw per-request audit artifacts under ignored `runs/`.
- If adding helper scripts, keep them small and target-local or experiment-local. Prefer one-off scripts under `runs/teacher-gold-label-audit-20260622/` unless a reusable NLU diagnostic helper is clearly justified.
- Avoid paid live OpenAI calls by default. The existing artifacts are enough for this audit. If a paid call is truly needed, record why and keep cost under `$5`.

## Audit Taxonomy

For each audited disagreement, assign exactly one primary class and optional secondary tags.

Primary classes:

- `teacher_clear_error`: gold is reasonable and teacher is clearly wrong.
- `teacher_reasonable_dataset_convention`: teacher output is semantically reasonable, but gold follows a dataset-specific convention the teacher was not told.
- `gold_questionable`: gold appears wrong or less plausible than teacher output.
- `ambiguous_multiple_valid`: gold and teacher are both defensible without more annotation guidelines.
- `exact_match_artifact`: the disagreement is mostly casing, punctuation, possessive, span boundary, singular/plural, or another normalization artifact.
- `infrastructure_failure`: parse failure, connection error, empty response, or schema parsing issue.
- `unclear`: not enough context to judge.

Secondary tags should be short and consistent, for example:

- `intent_boundary`
- `slot_key_boundary`
- `slot_value_span`
- `case_or_punctuation`
- `extra_slot`
- `missing_slot`
- `schema_alias`
- `relation_vs_person`
- `date_vs_time`
- `general_quirky_boundary`
- `music_likeness_vs_play_music`
- `transport_or_recommendation_boundary`

## Sampling Design

Create a deterministic audit sample under:

```text
runs/teacher-gold-label-audit-20260622/
```

Minimum sample:

- 30 unique sequential-medium rows where `teacher-v1` and `teacher-v7` are both wrong.
- 20 unique uniform-medium rows where `teacher-v1` and `teacher-v7` are both wrong.
- 15 unique rows where `teacher-v7` is correct and `teacher-v1` is wrong.
- 15 unique rows where `teacher-v1` is correct and `teacher-v7` is wrong.
- 10 parse/API failure rows if available; otherwise include all available parse/API failures and note the count.
- 10 rows where the only apparent issue is slot value span/casing/punctuation, if available.

Use unique request ids for audit counts. The `uniform` stream uses sampling with replacement, so report both raw request count and unique request count. Do not treat duplicated uniform rows as independent label evidence.

If a bucket has fewer rows than requested, include all rows in that bucket and explain the shortfall.

## Method

1. Load the four primary details JSONL files.
2. Recompute summary metrics and paired outcomes as a sanity check.
3. Detect duplicate request ids in each stream and report them.
4. Build the deterministic audit sample.
5. For each sampled row, inspect the utterance, gold frame, `teacher-v1` frame, and `teacher-v7` frame.
6. Assign the primary class, secondary tags, confidence (`high`, `medium`, `low`), and a one-sentence rationale.
7. Aggregate counts by primary class and secondary tag.
8. Include representative examples for each major class. Keep examples concise and do not dump raw details files into docs.
9. Interpret what the audit implies for:
   - continued prompt search;
   - whether a gold/convention-trained L2 could beat live L4 on MASSIVE;
   - whether evaluation should add diagnostic normalized metrics in addition to exact match;
   - whether any suspected gold-label issue is large enough to affect conclusions.

## Expected Artifacts

Raw/working artifacts under ignored runs:

```text
runs/teacher-gold-label-audit-20260622/audit_sample.jsonl
runs/teacher-gold-label-audit-20260622/audit_labels.jsonl
runs/teacher-gold-label-audit-20260622/audit_summary.json
```

Final report:

```text
docs/experiments/2026-06-22_teacher_gold_label_audit_report.md
```

The report must include:

- objective and scope;
- artifacts audited;
- sampling method and duplicate handling;
- summary metrics sanity check;
- audit taxonomy;
- aggregate audit counts;
- representative examples;
- judgment on dataset convention mismatch vs teacher errors;
- implications for L2 trained on gold/convention examples;
- recommendations for next experiments;
- commands run.

## Done Criteria

- At least 80 unique disagreements are audited, unless there are fewer qualifying rows and the report explains why.
- Uniform duplicate handling is explicitly documented.
- The report distinguishes teacher mistakes, dataset convention mismatch, gold ambiguity/noise, exact-match artifacts, and infrastructure failures.
- The report does not claim dataset labels are bad without concrete examples and counts.
- The report does not recommend putting MASSIVE-specific conventions into Darjeeling core.
- Any code or helper script added is small, scoped, and either stored under `runs/` or clearly target/experiment-local.
- `uv run pytest tests/targets/nlu/test_teacher_eval.py tests/targets/nlu/test_l4_teacher.py -q` passes if code is changed.
- `git diff --check` passes.
