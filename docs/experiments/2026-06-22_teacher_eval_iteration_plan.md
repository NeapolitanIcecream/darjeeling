# Teacher Eval Experiment Iteration Plan

Date: 2026-06-22

Inputs:

- `docs/design/08_gpt55_pro_0622_teacher_eval_plan.md`
- `docs/experiments/2026-06-22_gpt55_pro_review_verification.md`

Goal: use the new teacher-vs-gold and prompt-comparison paths to test whether the teacher changes are useful, then iterate on the smallest prompt/code changes needed to improve live L4 teacher quality without leaking gold into training or expanding into cascade tuning.

Budget: paid live LLM calls are allowed up to **100 USD** total for this iteration.

## Constraints

- Keep experiments benchmark-only and NLU-target-local.
- Do not use `gold_frame` in compiler inputs, training, promotion replay, candidate selection, or weak-layer tuning.
- Do not tune L1, L2, or L3 in this pass.
- Do not run full-cascade experiments except a tiny report-rendering sanity check if needed.
- Keep abstraction tax low. Prefer static prompt versions, small helpers, existing CLI/report code, and concise docs.
- Preserve raw per-request artifacts locally under `runs/`; do not commit raw utterance/gold/teacher detail files.

## Cost Rules

- Maintain a cost ledger in the experiment run root, for example `runs/teacher-eval-iteration-YYYYMMDD/cost_ledger.json`.
- Count actual cost from the generated summary artifacts, not from rough estimates:
  - use `teacher_live_vs_gold.summary.json.full_l4.cost_usd`;
  - use `teacher_prompt_comparison.json.rows[*].cost_usd`;
  - if needed, cross-check by summing per-row `usage.total_tokens` and the configured input/output prices.
- Before each paid command, estimate the next run from the latest observed cost/request for that prompt and sample size.
- Do not start a paid command if `actual_spend_so_far + estimated_next_run_cost > 100`.
- Prefer stopping around 90 USD unless there is a clear, bounded final validation run.
- Record failed runs too. If a failed call returns usage/cost in artifacts, include it; if no usage is returned, record zero observed cost with the failure reason.
- Final report must include actual observed spend, estimated remaining budget, and the artifacts used to compute spend.

## Required First Fix

Before running paid prompt comparisons, fix prompt-version validation:

- `ensure_supported_teacher_prompt_version()` should reject unknown versions using `SUPPORTED_TEACHER_PROMPT_VERSIONS`.
- Add a focused test proving an unknown prompt version is rejected.
- Re-run focused tests and boundary tests.

This prevents typoed prompt versions from silently behaving like the current full-frame prompt.

## Experiment Loop

Use short cycles. Each cycle should state a hypothesis, run the smallest useful experiment, inspect failures, and either revise the prompt/code or reject the hypothesis.

### Cycle 0: Readiness And No-Cost Checks

Hypothesis: the benchmark and comparison paths are usable and report costs correctly.

Actions:

- Run focused tests, boundary tests, and CLI help.
- Run any no-live smoke tests using fake teacher/test coverage only.
- Create the run root and initialize the cost ledger.

Stop and fix if tests fail, CLI help fails, or cost fields are missing from synthetic/fake artifacts.

### Cycle 1: Paid Pilot

Hypothesis: `teacher-v2-intent-first` changes teacher quality enough to justify further work.

Actions:

- Run `teacher-v1` and `teacher-v2-intent-first` on the same small sample, about 20 requests.
- Use one deterministic sample: same split, stream, and max requests for both prompts.
- Compare frame exact match, intent accuracy, slot metrics, parse/schema failures, tokens/request, cost/request, and p95 latency.
- Update the cost ledger from artifacts.

Decision:

- If v2 improves quality with acceptable cost, validate on a larger sample.
- If v2 is worse or only more expensive, inspect errors and revise or reject it.
- If both prompts are poor, identify whether errors are mostly intent, slot extraction, schema parsing, or label ambiguity.

### Cycle 2: Prompt/Parser Iteration

Hypothesis: a small target-local prompt change can reduce the dominant error class from Cycle 1.

Allowed changes:

- tighten `teacher-v2-intent-first` instructions;
- add one new static prompt version if useful, such as `teacher-v3-shortlist`;
- improve validation/reporting for invalid intents/slots;
- add concise tests for any new prompt path.

Disallowed changes:

- new generic prompt registry;
- new evaluator framework;
- weak-layer training changes;
- gold-backed teacher labels for training.

Run another small paid comparison after each change. Keep each iteration small unless the previous result clearly justifies scale-up.

### Cycle 3: Medium Validation

Hypothesis: the best prompt from earlier cycles improves quality on a less noisy sample without hiding cost regressions.

Actions:

- Run the current best prompt and baseline `teacher-v1` on the same 100-200 request sample.
- If budget allows, run one second validation sample with a different stream or split.
- Update the cost ledger and write a concise experiment report.

Decision:

- Adopt the new prompt only if it improves the main quality metric or a clearly important submetric without unacceptable cost/latency.
- If no prompt beats `teacher-v1`, document that result and keep `teacher-v1` as default.

## Report Requirements

Write a final experiment report under `docs/experiments/`.

Include:

- hypotheses tested;
- commands run;
- artifact paths under `runs/`;
- actual observed spend and how it was computed;
- quality/cost/latency comparison table;
- dominant failure modes;
- prompt/code changes made;
- final recommendation: adopt, keep as experimental, or reject;
- tests and checks run.

Keep raw per-request detail files out of committed docs. Summarize examples only if they are redacted and needed to explain a failure pattern.

## Done Criteria

- Prompt-version validation rejects unknown versions.
- At least one paid pilot compares `teacher-v1` and `teacher-v2-intent-first` on the same sample.
- Cost is tracked from actual artifact usage/cost and stays below 100 USD.
- At least one hypothesis is either supported, improved through a code/prompt iteration, or explicitly rejected.
- A medium validation run is completed for the best candidate, unless the pilot clearly shows no candidate is worth scaling.
- The final report states whether the current teacher changes are effective and what should happen next.
