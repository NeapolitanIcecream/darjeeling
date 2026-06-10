# 2026-06-10 L2 visible validation ratio

## Goal

Follow up the real L2 `agent-session` budget experiment, where a candidate
passed private selection but failed private promotion. The working hypothesis is
that the agent-visible validation pressure should be adjustable without
exposing private holdout rows or private gate aggregates.

## Design Change

`edge-mvp l2 target-evolve` now accepts:

```bash
--visible-validation-ratio FLOAT
```

When omitted, existing defaults stay unchanged:

- one visible fold keeps the old 60/20/10/10 split;
- multiple visible folds default to the fixed-inner 50/30/10/10 shape.

When provided, the ratio controls the total agent-visible validation pool, while
private selection and private promotion remain outside the workspace. Summary,
`data/round_state.json`, and `data/objective.json` record the requested ratio.
`data_split_policy` also records the effective ratio after integer rounding and
intent-stratified group constraints.

The same implementation pass fixed a reporting bug exposed by the experiment:
private selection/promotion `safety_backlog` entries now use
`outer_summary_only_not_agent_workspace` visibility instead of the misleading
`visible_validation_only` label.

## Experiment

Command:

```bash
uv run edge-mvp l2 target-evolve \
  --traces runs/l2-list-fallback-tuned-3k-r1/traces.jsonl \
  --out-dir runs/l2-target-visible-ratio40-smoke-r2 \
  --max-traces 1000 \
  --mode local-search \
  --budget-profile fixed-inner \
  --target-scope lower_miss \
  --split-policy intent-stratified \
  --rounds 1 \
  --visible-validation-folds 5 \
  --visible-validation-ratio 0.4 \
  --visible-cross-audit-folds 3 \
  --local-search-trials 4 \
  --local-search-timeout-s 120 \
  --local-search-cross-audit-top-k 1
```

Result:

- Evidence class: `short_fixed_snapshot_probe`.
- Scoped lower-miss traces: `999` out of `1000`; `1` lower-layer accept
  excluded.
- Split: train `435`, visible validation `377`, private selection `97`,
  private promotion `90`.
- Requested visible validation ratio: `0.4`.
- Effective visible validation ratio: `0.37737737737737737`.
- Baseline visible validation: `14` accepted, `11` correct, `3` wrong; gate
  failed.
- Local-search candidate was not applied:
  `best visible/cross-audit config failed visible cross-audit safety gate`.
- Best visible cross-audit candidate: `4` accepted, `3` correct, `1` wrong;
  gate failed.
- Private selection: `6` accepted, `5` correct, `1` wrong; gate failed.
- Private promotion: `3` accepted, `2` correct, `1` wrong; gate failed.
- Adoption: `adopted=false`.

## Conclusion

The ratio switch works as a visible-only pressure control and does not change
private holdout visibility. The run is intentionally short, so it is not a
quality claim, but it supports using a larger visible pool before spending
another live L2 agent-session budget. The stronger visible pool caught visible
wrong accepts immediately, and cross-audit still vetoed the best local-search
config.

Next live-agent quality work should use the new ratio on a larger fixed snapshot
instead of iterating from private promotion failures.

## Live agent-session follow-up

Command:

```bash
uv run edge-mvp l2 target-evolve \
  --traces runs/l2-list-fallback-tuned-3k-r1/traces.jsonl \
  --out-dir runs/l2-real-agent-ratio40-stratified2000-r1/job \
  --max-traces 2000 \
  --mode agent-session \
  --budget-profile fixed-inner \
  --target-scope lower_miss \
  --split-policy intent-stratified \
  --rounds 16 \
  --max-agent-rounds 1 \
  --visible-validation-folds 5 \
  --visible-validation-ratio 0.4 \
  --visible-cross-audit-folds 3 \
  --local-search-trials 12 \
  --local-search-timeout-s 180 \
  --local-search-cross-audit-top-k 1 \
  --timeout-s 1200
```

Result:

- Evidence class: `fixed_snapshot_research`.
- Agent session: completed, `1` started, `1` succeeded.
- Split: train `787`, visible validation `774`, private selection `199`,
  private promotion `190`.
- Requested visible validation ratio: `0.4`.
- Effective visible validation ratio: `0.39692307692307693`.
- Baseline visible validation: `33` accepted, `25` correct, `8` wrong; gate
  failed.
- Final visible validation: `32` accepted, `32` correct, `0` wrong; gate passed.
- Final visible cross-audit: `27` accepted, `27` correct, `0` wrong; gate
  passed.
- Final train audit: `105` accepted, `104` correct, `1` wrong.
- Private selection: `8` accepted, `6` correct, `2` wrong; gate failed.
- Private promotion: `7` accepted, `5` correct, `2` wrong; gate failed.
- Adoption: `adopted=false`.

Interpretation:

- The larger visible pool helped the agent find a visibly safe target candidate,
  but visible validation plus cross-audit was still insufficient.
- The agent report explicitly noted the remaining train-audit wrong accept and
  chose to leave it because it considered the visible label contradictory. Under
  the project contract, visible teacher labels remain the local safety oracle;
  if a target rule disagrees, it should abstain instead of accepting.
- The follow-up implementation makes visible train-audit accepted-wrong count a
  safety gate before private selection. This does not make train coverage an
  optimization target; it only prevents a candidate with known visible
  train-audit wrong accepts from being selected.

## Post-fix smoke

After adding the train-audit safety gate, I reran the short deterministic
local-search probe:

```bash
uv run edge-mvp l2 target-evolve \
  --traces runs/l2-list-fallback-tuned-3k-r1/traces.jsonl \
  --out-dir runs/l2-target-visible-ratio40-trainaudit-smoke-r1/job \
  --max-traces 1000 \
  --mode local-search \
  --budget-profile fixed-inner \
  --target-scope lower_miss \
  --split-policy intent-stratified \
  --rounds 1 \
  --visible-validation-folds 5 \
  --visible-validation-ratio 0.4 \
  --visible-cross-audit-folds 3 \
  --local-search-trials 4 \
  --local-search-timeout-s 120 \
  --local-search-cross-audit-top-k 1
```

Result:

- Evidence class: `short_fixed_snapshot_probe`.
- Split and effective visible ratio matched the earlier short probe: train
  `435`, visible validation `377`, private selection `97`, private promotion
  `90`, effective visible ratio `0.37737737737737737`.
- Visible validation still failed: `14` accepted, `11` correct, `3` wrong.
- Train-audit safety gate failed: `47` accepted, `35` correct, `12` wrong.
- Summary recorded `passes_train_audit_safety_gate=false`.
- `round_state.json` recorded the then-current candidate selection gate:
  visible validation, visible train-audit safety, and private selection must all
  pass.
- Adoption remained `adopted=false`.

This smoke verifies the new gate is wired through summary and agent-visible
state. It is not a quality result.

## Train-audit live rerun

With the train-audit safety gate in place, I reran the live agent-session on the
same 2000-row lower-miss fixed snapshot:

```bash
uv run edge-mvp l2 target-evolve \
  --traces runs/l2-list-fallback-tuned-3k-r1/traces.jsonl \
  --out-dir runs/l2-real-agent-ratio40-trainaudit-r2/job \
  --max-traces 2000 \
  --mode agent-session \
  --budget-profile fixed-inner \
  --target-scope lower_miss \
  --split-policy intent-stratified \
  --rounds 16 \
  --max-agent-rounds 1 \
  --visible-validation-folds 5 \
  --visible-validation-ratio 0.4 \
  --visible-cross-audit-folds 3 \
  --local-search-trials 12 \
  --local-search-timeout-s 180 \
  --local-search-cross-audit-top-k 1 \
  --timeout-s 1200
```

Result:

- Evidence class: `fixed_snapshot_research`.
- Agent session: completed, `1` started, `1` succeeded.
- Split: train `787`, visible validation `774`, private selection `199`,
  private promotion `190`.
- Requested visible validation ratio: `0.4`.
- Effective visible validation ratio: `0.39692307692307693`.
- Baseline visible validation: `33` accepted, `25` correct, `8` wrong; gate
  failed.
- Final visible validation: `7` accepted, `7` correct, `0` wrong; gate passed.
- Final visible cross-audit: `5` accepted, `5` correct, `0` wrong; gate passed.
- Final train audit: `59` accepted, `59` correct, `0` wrong; train-audit safety
  gate passed.
- Private selection: `0` accepted, `0` wrong; gate failed.
- Private promotion: `3` accepted, `2` correct, `1` wrong; gate failed.
- Adoption: `adopted=false`.

Interpretation:

- The train-audit gate changed the agent behavior: it no longer stopped with a
  known visible train-audit wrong accept.
- The new failure mode is a conservative target that passes visible safety by
  reducing visible accepts from baseline `33` to `7`. Private selection then
  observes no accepts, while private promotion still observes one wrong accept.
- This is not a reason to expose private feedback to the agent. The visible side
  needs a small support floor so that near-zero coverage candidates do not reach
  candidate selection just because their remaining accepts are clean.

## Visible support gate

The follow-up implementation adds a visible support gate before private
selection. Candidate selection now requires:

- visible validation gate,
- visible support gate,
- visible train-audit accepted-wrong safety gate,
- private selection holdout gate.

The visible support rule is intentionally small: the candidate must keep at
least `2` correct accepts per visible validation fold before private selection.
For the five-fold fixed-inner runs here, the floor is `10` visible correct
accepts. This keeps the concept simple and prevents an overly conservative
target from being treated as selection-ready.

`private_holdout_evidence` now records:

- `visible_support_passing_rounds`,
- `inner_passing_visible_support_failed_rounds`,
- `inner_passing_train_audit_wrong_accept_rounds`.

The last count remains based on all inner-passing rounds, even when visible
support also fails, so known visible train-audit safety problems are not hidden
by an earlier support failure.

I saved the live agent target shape as a reproducible dry-run patch:

```text
docs/experiments/patches/l2_target_ratio40_trainaudit_conservative_r2.patch
```

Then I reran it under the updated harness:

```bash
uv run edge-mvp l2 target-evolve \
  --traces runs/l2-list-fallback-tuned-3k-r1/traces.jsonl \
  --out-dir runs/l2-real-agent-ratio40-visible-support-r6/job \
  --max-traces 2000 \
  --mode dry-run \
  --budget-profile fixed-inner \
  --target-scope lower_miss \
  --split-policy intent-stratified \
  --rounds 1 \
  --dry-run-patch docs/experiments/patches/l2_target_ratio40_trainaudit_conservative_r2.patch \
  --visible-validation-folds 5 \
  --visible-validation-ratio 0.4 \
  --visible-cross-audit-folds 3 \
  --local-search-trials 12 \
  --local-search-timeout-s 180 \
  --local-search-cross-audit-top-k 1
```

Result:

- Evidence class: `short_fixed_snapshot_probe`.
- Split and effective visible ratio matched the live train-audit rerun.
- Visible validation passed: `7` accepted, `7` correct, `0` wrong.
- Visible support failed: `7` correct accepts, required `10`.
- Train-audit safety also failed in this dry-run replay: `60` accepted,
  `59` correct, `1` wrong. The live run had `59/59/0`, so the replay is used as
  protocol verification rather than a claim of exact candidate reproduction.
- Visible cross-audit passed: `5` accepted, `5` correct, `0` wrong.
- Private selection: `0` accepted, `0` wrong; gate failed.
- Private promotion: `3` accepted, `2` correct, `1` wrong; gate failed.
- `selection_gate_diagnosis=visible_support_gate_failed`.
- `inner_passing_visible_support_failed_rounds=1`.
- `inner_passing_train_audit_wrong_accept_rounds=1`.
- Adoption remained `adopted=false`.

Conclusion:

- The support gate catches the low-coverage candidate before it can be described
  as merely a sparse private-selection result.
- The gate does not relax private isolation: private selection/promotion rows
  and aggregates remain outer-summary-only.
- The next live-agent run should keep visible safety first, but it must avoid
  solving safety by simply raising the threshold until visible coverage is too
  thin. Structural target work should preserve at least the visible support
  floor while clearing visible validation and train-audit wrong accepts.
