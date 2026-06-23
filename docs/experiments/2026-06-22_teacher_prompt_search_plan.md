# Teacher Prompt Search Experiment Plan

Date: 2026-06-22

Purpose: run a self-contained teacher prompt search that can support a real adoption decision. The prior iteration was useful, but it did not explore enough prompt hypotheses to justify a final conclusion about teacher prompt quality.

This document is the source of truth for the next agent. Do not rely on chat/session history.

## Required Context Files

Read these before changing code or running paid calls:

- `AGENTS.md`
- `docs/design/08_gpt55_pro_0622_teacher_eval_plan.md`
- `docs/experiments/2026-06-22_gpt55_pro_review_verification.md`
- `docs/experiments/2026-06-22_teacher_eval_iteration_plan.md`
- `docs/experiments/2026-06-22_teacher_eval_iteration_report.md`
- `runs/teacher-eval-iteration-20260622/cost_ledger.json`

Also inspect the current implementation:

- `src/darjeeling/targets/nlu/teacher.py`
- `src/darjeeling/targets/nlu/teacher_eval.py`
- `src/darjeeling/targets/nlu/layers/l4_cloud_llm.py`
- `src/darjeeling/targets/nlu/main_cli.py`
- `tests/targets/nlu/test_l4_teacher.py`
- `tests/targets/nlu/test_teacher_eval.py`

## Current State

The repository now has a benchmark-only teacher quality path:

- `edge-mvp-nlu teacher eval-live`
- `edge-mvp-nlu teacher compare-prompts`

Existing prompt versions:

- `teacher-v1`: current default, single full-frame prompt.
- `teacher-v2-intent-first`: two live calls, first intent then frame/slots. Prior pilot rejected it.
- `teacher-v3-slot-conservative`: single-call prompt that discourages guessed slots. Prior medium validation kept it experimental only.

Existing iteration result:

- Total observed live spend so far: `0.1187132` USD.
- Budget limit: `100.0` USD total, including previous spend.
- Prior conclusion:
  - keep `teacher-v1` as default for now;
  - reject `teacher-v2-intent-first`;
  - keep `teacher-v3-slot-conservative` experimental only.

This prior conclusion is not enough. It rejects bad candidates, but it does not prove that a serious low-complexity prompt search was done.

## Desired Conclusion

The next iteration must produce one of these two conclusions:

1. **Adopt a new prompt**: a candidate shows a clear, repeatable improvement over `teacher-v1` on frame exact match, or a deliberate target metric with no unacceptable frame/cost/latency regression.
2. **Keep `teacher-v1` after sufficient search**: multiple plausible low-complexity prompt hypotheses were tested, iterated, and validated, and none beat `teacher-v1` enough to adopt.

Do not merely reinterpret the prior result. The task is to do enough additional experimental work to support one of the conclusions above.

## Constraints

- Keep all prompt work inside the NLU target and NLU tests/docs.
- `gold_frame` may be used only by benchmark/report diagnostics.
- Do not expose `gold_frame` to `TeacherTrace`, compiler inputs, training, promotion replay, candidate selection, or weak-layer tuning.
- Do not tune L1, L2, or L3.
- Do not run full cascade experiments except a tiny report-rendering sanity check if needed.
- Keep abstraction tax low:
  - static prompt versions are fine;
  - small helper functions are fine;
  - do not add a generic prompt registry;
  - do not add a generic evaluator framework;
  - do not add dependency injection, plugin systems, schema DSLs, or new experiment infrastructure.
- Keep raw per-request artifacts under ignored `runs/`.
- Do not commit raw `details.csv` or `details.jsonl` files containing utterances, gold frames, or teacher frames.

## Budget And Cost Accounting

Paid live calls are allowed. Total budget remains **100 USD**, including the previous `0.1187132` USD.

Use a new run root, for example:

```text
runs/teacher-prompt-search-20260622
```

Maintain a cost ledger in the run root:

```text
runs/teacher-prompt-search-20260622/cost_ledger.json
```

The ledger must include:

- budget limit;
- previous observed spend;
- new observed spend;
- total observed spend;
- estimated remaining budget;
- one entry per paid or no-cost step;
- pre-run budget estimate for every paid command;
- artifact paths used for cost accounting;
- failure reason for failed paid runs.

Cost must come from observed artifacts:

- `teacher_prompt_comparison.json.rows[*].cost_usd`
- `teacher_live_vs_gold.summary.json.full_l4.cost_usd`
- cross-check with `teacher_live_vs_gold.details.jsonl` row `cost_usd` sums when detail files exist.

Before every paid run:

- estimate next cost using the latest observed cost/request for the prompt or a conservative multiplier;
- do not run if `previous_spend + new_spend_so_far + estimated_next_run > 100`.

Do not over-optimize for saving money. The budget exists to run enough experiments. Stop early only when the evidence is already strong, not merely because the first candidate failed.

## Experiment Design

### Step 0: Readiness

Before paid calls:

- verify unknown prompt versions hard-fail;
- run focused teacher/eval tests;
- run boundary tests;
- run CLI help smoke;
- initialize the new cost ledger.

Minimum commands:

```bash
uv run pytest tests/targets/nlu/test_l4_teacher.py tests/targets/nlu/test_teacher_eval.py tests/targets/nlu/test_experiment_suite_cli.py tests/targets/nlu/test_nlu_target.py -q
uv run pytest tests/test_core_contracts.py tests/test_target_boundary.py tests/targets/nlu/test_target_core_boundary.py -q
uv run python -m darjeeling.targets.nlu.main_cli teacher eval-live --help
uv run python -m darjeeling.targets.nlu.main_cli teacher compare-prompts --help
```

### Step 1: Error Taxonomy From Existing Artifacts

Analyze prior detail artifacts before designing new prompts:

- cycle 1 v1/v2 details;
- cycle 2 v1/v3 details;
- cycle 3 v1/v3 100-request details.

Create an error taxonomy summary in the new report:

- intent miss;
- extra slot keys;
- missing slot keys;
- wrong slot values;
- parse/schema failure;
- possible label ambiguity;
- pairwise outcomes: both correct, baseline-only correct, candidate-only correct, neither correct.

The taxonomy should identify the dominant errors that new prompt hypotheses target.

### Step 2: Prompt Hypotheses

Design at least **three new prompt candidates** beyond `teacher-v1`, `teacher-v2-intent-first`, and `teacher-v3-slot-conservative`.

Each candidate must have:

- a short prompt version name;
- one-sentence hypothesis;
- expected improvement mechanism;
- expected risk;
- planned pilot sample.

Acceptable candidate directions include:

- stricter slot evidence;
- slot value copying discipline;
- no hidden/default slots;
- intent stability with simpler single-call wording;
- explicit empty-slot preference for ambiguous slots;
- schema discipline for allowed intents/slots;
- short checklist inside one prompt, as long as the output remains strict JSON only.

Avoid complex protocols. Prefer single-call prompts unless a two-call design has a strong reason and a clear cost justification.

### Step 3: Pilot Screening

Run each new candidate against `teacher-v1` on the same deterministic 30-50 request sample.

For every pilot, report:

- frame exact match;
- intent accuracy;
- slot key exact match;
- slot pair precision/recall/F1;
- parse/schema failures;
- cost/request;
- p95 latency;
- paired outcomes against `teacher-v1`.

Rules:

- immediately reject candidates that are worse on frame exact and worse on cost/latency;
- keep candidates that improve frame exact or show a strong targeted slot improvement without frame regression;
- if a candidate shows a clear failure mode but a small prompt edit could address it, make one iteration and rerun a pilot.

There must be at least one prompt iteration based on pilot evidence unless all candidates fail obviously and independently.

### Step 4: Medium Validation

Run the best one or two candidates against `teacher-v1` on at least one 200-request sample.

If budget allows, run a second validation sample:

- different stream; or
- different contiguous validation range if CLI support exists; or
- a deterministic non-prefix sample if implemented simply.

Do not overfit to the first sequential validation prefix.

Medium validation must report paired outcomes:

- both correct;
- teacher-v1 only correct;
- candidate only correct;
- neither correct;
- frame exact delta;
- slot F1 delta;
- intent accuracy delta;
- cost/request delta;
- p95 latency delta.

If implementing bootstrap or McNemar-style checks is simple, do it. If not, use paired counts and a clear practical threshold.

### Step 5: Adoption Criteria

A prompt can be recommended as default only if it satisfies at least one:

- clear, repeatable frame exact improvement over `teacher-v1` on medium validation;
- no frame exact regression, meaningful slot F1 improvement, acceptable cost/latency, and a documented reason that slot quality is the intended optimization target.

A prompt should stay experimental if:

- it helps one secondary metric but not enough to justify default adoption;
- the gain is within noise, such as 1 request out of 100 or similarly weak evidence;
- it costs materially more without primary-metric gain.

Reject a prompt if:

- frame exact drops;
- parse/schema failures increase;
- cost/latency rises without a compensating quality gain;
- it only improves on one small pilot and fails medium validation.

## Required Report

Write a new final report under `docs/experiments/`, for example:

```text
docs/experiments/2026-06-22_teacher_prompt_search_report.md
```

The report must include:

- the exact objective of this second iteration;
- summary of prior insufficient evidence;
- cost ledger path;
- total spend including prior spend;
- commands run;
- prompt hypotheses;
- pilot table;
- medium validation table;
- paired comparison table;
- failure taxonomy;
- prompt/code changes made;
- tests/checks run;
- final adoption decision.

The report should make clear whether the conclusion is:

- adopt a new prompt;
- keep `teacher-v1` after sufficient search;
- keep a candidate only as experimental;
- or continue research because evidence is still inconclusive.

## Done Criteria

- This plan is followed without relying on chat history.
- At least three new prompt candidates are designed and piloted.
- At least one prompt is iterated based on pilot evidence, unless every new candidate clearly fails for independent reasons.
- At least one candidate reaches a 200-request validation against `teacher-v1`, unless all candidates are clearly worse in pilot.
- Cost is tracked from actual artifacts and total observed spend remains below 100 USD.
- Raw details remain under ignored `runs/`.
- A final report gives an evidence-backed adoption decision.
- Focused tests, boundary tests, ruff touched-file check, and `git diff --check` pass.
