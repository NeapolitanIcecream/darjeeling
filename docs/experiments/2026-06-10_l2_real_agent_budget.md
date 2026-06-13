# 2026-06-10 L2 real agent-session budgeted experiment

## Goal

Run a real L2 `agent-session` target-evolution experiment under an LLM budget of
`$100`, using fixed snapshots, private selection/promotion gates, and no private
feedback inside the agent workspace.

## Budget Control

All live runs used:

- `--mode agent-session`
- `--max-agent-rounds 1`
- `--timeout-s 1200`
- `--budget-profile fixed-inner`
- `--local-search-trials 12`
- `--local-search-timeout-s 180`

Only the Codex agent session consumes LLM budget. Local evaluation, Optuna, and
cross-audit are deterministic/local tool work.

Transcript usage and cost estimate, using the repository's configured L4 prices
(`$0.40/M` uncached input, `$0.10/M` cached input, `$1.60/M` output):

| Run | Input tokens | Cached input | Output tokens | Estimated cost |
| --- | ---: | ---: | ---: | ---: |
| `l2-real-agent-fixed500-r1` | 1,875,743 | 1,608,960 | 9,038 | `$0.2821` |
| `l2-real-agent-stratified1000-r2` | 1,415,939 | 1,270,144 | 6,065 | `$0.1950` |
| `l2-real-agent-stratified1000-r3` | 1,420,680 | 1,268,864 | 6,721 | `$0.1984` |

Total estimated LLM cost: `$0.6755`, well below the `$100` cap. The Codex
transcript records token usage but not a provider invoice; this estimate uses
the project settings pricing.

## Run r1: 500 chronological lower-miss snapshot

Command:

```bash
uv run edge-mvp-nlu l2 target-evolve \
  --traces runs/l2-list-fallback-tuned-3k-r1/traces.jsonl \
  --out-dir runs/l2-real-agent-fixed500-r1 \
  --max-traces 500 \
  --mode agent-session \
  --budget-profile fixed-inner \
  --target-scope lower_miss \
  --rounds 16 \
  --max-agent-rounds 1 \
  --visible-validation-folds 5 \
  --visible-cross-audit-folds 3 \
  --local-search-trials 12 \
  --local-search-timeout-s 180 \
  --local-search-cross-audit-top-k 1 \
  --timeout-s 1200
```

Result:

- Evidence class: `fixed_snapshot_research`
- Agent session: completed, `1` started, `1` succeeded
- Split: train `250`, visible validation folds `150`, private selection `50`,
  private promotion `50`
- Visible validation: accepted `6`, correct `6`, wrong `0`, gate passed
- Visible cross-audit: accepted `2`, correct `2`, wrong `0`, gate passed
- Private selection: accepted `0`, gate failed
- Private promotion: accepted `0`, gate failed
- Selection: `selected=false`
- Adoption: `adopted=false`

Diagnosis: the candidate was safe on visible data but too conservative or too
sparse for the private holdouts. The summary recommended a larger or
intent-stratified split.

## Run r2: 1000 intent-stratified lower-miss snapshot

Command changed `--max-traces 1000` and added `--split-policy
intent-stratified`.

Result:

- Evidence class: `fixed_snapshot_research`
- Agent session: completed, `1` started, `1` succeeded
- Split: train `514`, visible validation folds `307`, private selection `89`,
  private promotion `89`
- Visible validation: accepted `2`, correct `2`, wrong `0`, gate passed
- Visible cross-audit: accepted `12`, correct `12`, wrong `0`, gate passed
- Private selection: accepted `1`, correct `1`, wrong `0`, gate passed
- Private promotion: accepted `1`, correct `0`, wrong `1`, gate failed
- Selection: `selected=true`
- Adoption: `adopted=false`

The candidate improved over r1 by passing selection, but failed promotion. The
candidate also used single-row exact visible utterance postprocess rules. That
exposed a design gap in the agent-visible target-code policy: target-dependent
code is allowed, but single-visible-row memorization is not a reliable
generalization strategy.

## Design Fix

Commit `df58c9f` tightened the L2 target program/objective:

- `objective.json` now marks single-visible-row exact utterance exceptions and
  request-id memorization as invalid strategies.
- `program.md` tells the agent to prefer pattern-level lexical or slot-support
  rules backed by multiple visible examples or clear schema semantics.
- `target_code_policy` records the same generalization rule.

Focused verification after the fix:

```bash
uv run ruff check src/darjeeling/compiler/l2_target_evolution.py tests/test_l2_target_evolution.py
uv run pytest tests/test_l2_target_evolution.py -q
```

Both passed.

## Run r3: 1000 intent-stratified after the design fix

Command matched r2 after regenerating the workspace with the new
generalization policy.

Result:

- Evidence class: `fixed_snapshot_research`
- Agent session: completed, `1` started, `1` succeeded
- Visible validation: accepted `2`, correct `2`, wrong `0`, gate passed
- Visible cross-audit: accepted `11`, correct `11`, wrong `0`, gate passed
- Private selection: accepted `1`, correct `1`, wrong `0`, gate passed
- Private promotion: accepted `1`, correct `0`, wrong `1`, gate failed
- Selection: `selected=true`
- Adoption: `adopted=false`

The r3 candidate avoided the exact utterance exception style from r2, but still
failed private promotion with one wrong accept. Because private promotion rows
must not be fed back into the same or a follow-up agent session as examples, I
stopped the live-agent loop here rather than spending more LLM budget on
holdout-driven iteration.

## Conclusion

The real L2 `agent-session` path works under the budget cap: the agent edited
`target/`, used visible tools, produced scoped candidates, and the outer harness
selected/rejected candidates through private gates.

No target artifact was adopted. The best result selected on visible validation
and private selection, but failed private promotion. This is a useful negative
quality result: the current L2 target loop can find safe-looking visible
improvements, but promotion still catches generalization failures. The next
quality experiment should increase visible pressure without leaking private
rows, for example by using a larger fixed snapshot or stronger visible
cross-audit before considering any target promotion.

## Metric Trace

The agent-session internal eval trace is plotted in
`2026-06-10_l2_real_agent_eval_metrics.png`; the extracted rows are in
`2026-06-10_l2_real_agent_eval_metrics.csv`.

Point labels in the plot use:

- `VV`: visible validation
- `CX`: visible cross-audit
- `TR`: train audit
- `IN`: inner validation
