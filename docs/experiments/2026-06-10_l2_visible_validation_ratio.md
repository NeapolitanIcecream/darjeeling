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

## Visible support live rerun

After committing the visible support gate, I reran the live agent-session on the
same fixed snapshot:

```bash
uv run edge-mvp l2 target-evolve \
  --traces runs/l2-list-fallback-tuned-3k-r1/traces.jsonl \
  --out-dir runs/l2-real-agent-ratio40-visible-support-live-r1/job \
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
- Split and effective visible ratio matched the previous live runs.
- Baseline visible validation: `33` accepted, `25` correct, `8` wrong; gate
  failed.
- Final visible validation: `36` accepted, `36` correct, `0` wrong; gate passed.
- Visible support passed: `36` correct accepts, required `10`.
- Final train audit: `97` accepted, `97` correct, `0` wrong; train-audit safety
  passed.
- Final visible cross-audit: `36` accepted, `36` correct, `0` wrong; gate
  passed.
- Private selection: `8` accepted, `6` correct, `2` wrong; gate failed.
- Private promotion: `11` accepted, `8` correct, `3` wrong; gate failed.
- `selection_gate_diagnosis=selection_wrong_accepts_for_inner_passing_rounds`.
- Adoption remained `adopted=false`.

Interpretation:

- The support gate worked: the agent no longer solved safety by collapsing
  visible coverage. It retained more visible correct accepts than the baseline
  while clearing visible validation, train audit, and cross-audit wrong accepts.
- The remaining failure is private hidden wrong accepts, not sparse selection.
  The private wrongs are slot/schema boundary errors, but all visible
  accepted-wrong backlogs are empty.
- This suggests the visible diagnostics need a clearer "slot risk" queue for
  intent-correct slot mismatches after accepted-wrong backlog is empty. The
  data already exists in `family_diagnostics`; the problem is presentation and
  priority, not private leakage.

## Slot-risk diagnostics

The follow-up implementation adds `slot_risk_backlog` alongside the existing
`safety_backlog` in family diagnostics. It is derived only from visible
validation, visible train audit, and visible cross-audit metrics. It ranks
families with `intent_correct_slot_wrong` examples so the agent has a visible
queue to inspect before stopping or expanding coverage after accepted-wrong
backlogs are empty.

New agent-visible keys include:

- `latest_slot_risk_backlog`,
- `latest_train_audit_slot_risk_backlog`,
- `latest_visible_cross_audit_slot_risk_backlog`.

This does not change candidate selection/adoption gates and does not expose
private rows or private aggregate feedback.

Smoke command:

```bash
uv run edge-mvp l2 target-evolve \
  --traces runs/l2-list-fallback-tuned-3k-r1/traces.jsonl \
  --out-dir runs/l2-target-slot-risk-diagnostics-smoke-r1/job \
  --max-traces 1000 \
  --mode dry-run \
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
- `target_diagnostics.json` wrote all new slot-risk backlog keys.
- `latest_slot_risk_backlog`, `latest_train_audit_slot_risk_backlog`, and
  `latest_visible_cross_audit_slot_risk_backlog` all used
  `visibility=visible_validation_only`.
- The diagnostics file did not contain `selection_holdout` or
  `promotion_holdout`.
- The top visible slot-risk families were calendar-oriented in this short
  smoke; this is expected for an unmodified baseline and only verifies that the
  queue is populated and visible-only.

Next live-agent work should use these slot-risk queues after clearing
accepted-wrong backlogs. A target should not stop just because visible accepted
wrongs are gone if high-priority visible slot-risk families remain unaddressed.

## Slot-risk live result and high-guard view

Run:

```bash
uv run edge-mvp l2 target-evolve \
  --traces runs/l2-list-fallback-tuned-3k-r1/traces.jsonl \
  --out-dir runs/l2-real-agent-ratio40-slot-risk-live-r1/job \
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
- Baseline visible validation: `33` accepted, `25` correct, `8` wrong; gate
  failed.
- Final visible validation: `28` accepted, `28` correct, `0` wrong; gate passed.
- Visible support passed: `28` correct accepts, required `10`.
- Final train audit: `78` accepted, `78` correct, `0` wrong; train-audit safety
  passed.
- Final visible cross-audit: `31` accepted, `31` correct, `0` wrong; gate
  passed.
- Private selection: `7` accepted, `5` correct, `2` wrong; gate failed.
- Private promotion: `8` accepted, `7` correct, `1` wrong; gate failed.
- `selection_gate_diagnosis=selection_wrong_accepts_for_inner_passing_rounds`.
- Adoption remained `adopted=false`.

Interpretation:

- Slot-risk diagnostics improved the candidate shape but did not solve hidden
  wrong accepts. Compared with the previous support-gated live run, private
  promotion wrong accepts dropped from `3` to `1`, but private selection still
  had `2` wrong accepts.
- The remaining wrong accepts were low-frequency slot/schema boundary errors.
  The visible slot-risk queue did expose many slot mismatch families, but the
  top `items` were dominated by high-volume families such as calendar, weather,
  music, datetime, news, factoid, email, and transport.
- The design gap is therefore presentation, not another gate: a low-frequency
  family with very high guard probability can be more relevant to hidden wrong
  accepts than a high-volume family with lower guard pressure.

Follow-up implementation:

- `slot_risk_backlog` now keeps its existing count-ranked `items`.
- The same payload also includes `high_guard_items`, sorted by
  `max_slot_mismatch_guard_probability`, then slot-mismatch count.
- This field is present for visible validation, visible train audit, and visible
  cross-audit diagnostics. It remains diagnostic-only, visible-only in agent
  workspaces, and does not change candidate selection or adoption gates.

Smoke command:

```bash
uv run edge-mvp l2 target-evolve \
  --traces runs/l2-list-fallback-tuned-3k-r1/traces.jsonl \
  --out-dir runs/l2-target-slot-risk-high-guard-smoke-r1/job \
  --max-traces 1000 \
  --mode dry-run \
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

Smoke result:

- Evidence class: `short_fixed_snapshot_probe`.
- `latest_slot_risk_backlog`, `latest_train_audit_slot_risk_backlog`, and
  `latest_visible_cross_audit_slot_risk_backlog` all include
  `high_guard_items`.
- The diagnostics remained `visible_validation_only` and did not contain
  `selection_holdout` or `promotion_holdout`.
- The visible validation count-ranked top family was `calendar_set`, while the
  high-guard top family was `play_radio`; the cross-audit count-ranked top
  family was `calendar_set`, while its high-guard top family was
  `email_sendemail`. This verifies that the new view exposes low-frequency
  high-guard risks without replacing the volume queue.

## High-guard live result and slot-key deltas

Run:

```bash
uv run edge-mvp l2 target-evolve \
  --traces runs/l2-list-fallback-tuned-3k-r1/traces.jsonl \
  --out-dir runs/l2-real-agent-ratio40-high-guard-slot-risk-live-r1/job \
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
- Final visible validation: `28` accepted, `28` correct, `0` wrong; gate passed.
- Visible support passed: `28` correct accepts, required `10`.
- Final train audit: `82` accepted, `82` correct, `0` wrong; train-audit safety
  passed.
- Final visible cross-audit: `26` accepted, `26` correct, `0` wrong; gate
  passed.
- Private selection: `7` accepted, `5` correct, `2` wrong; gate failed.
- Private promotion: `8` accepted, `6` correct, `2` wrong; gate failed.
- Adoption remained `adopted=false`.

Interpretation:

- The high-guard view reached the agent: the initial visible validation
  high-guard top family was `play_radio`, and train-audit high-guard top family
  was `general_joke`.
- The final candidate still missed slot/schema boundary cases. The remaining
  private failures included slotless accepts where the teacher frame required
  keys such as `house_place` or `joke_type`.
- This suggests the queue now has the right families but still makes the agent
  infer the actual schema delta from examples. The next diagnostic should
  summarize slot key differences directly in each slot-risk item.

Follow-up implementation:

- Each slot-risk item now includes `missing_slot_keys`, `extra_slot_keys`, and
  `changed_slot_keys`, sorted by count.
- Aggregated visible cross-audit slot-risk backlogs merge the same key counts
  across folds.
- This remains visible-only diagnostic context and does not change candidate
  selection/adoption gates.

Smoke command:

```bash
uv run edge-mvp l2 target-evolve \
  --traces runs/l2-list-fallback-tuned-3k-r1/traces.jsonl \
  --out-dir runs/l2-target-slot-key-deltas-smoke-r1/job \
  --max-traces 1000 \
  --mode dry-run \
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

Smoke result:

- Evidence class: `short_fixed_snapshot_probe`.
- The visible validation high-guard slot-risk item was `play_radio` and listed
  missing `radio_name`.
- The visible train-audit high-guard slot-risk item was `iot_hue_lightoff` and
  listed missing `house_place`.
- The visible cross-audit high-guard slot-risk item was `email_sendemail` and
  listed missing `person`, `relation`, and `email_address`, plus extra
  `email_address` and `person`.
- `target_diagnostics.json`, `round_state.json`, and `objective.json` did not
  contain `selection_holdout` or `promotion_holdout`.

## Slot-key live result and intent-confusion backlog

Run:

```bash
uv run edge-mvp l2 target-evolve \
  --traces runs/l2-list-fallback-tuned-3k-r1/traces.jsonl \
  --out-dir runs/l2-real-agent-ratio40-slot-key-deltas-live-r1/job \
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
- Final visible validation: `24` accepted, `24` correct, `0` wrong; gate passed.
- Visible support passed: `24` correct accepts, required `10`.
- Final train audit: `56` accepted, `56` correct, `0` wrong; train-audit safety
  passed.
- Final visible cross-audit: `22` accepted, `22` correct, `0` wrong; gate
  passed.
- Private selection: `7` accepted, `5` correct, `2` wrong; gate failed.
- Private promotion: `5` accepted, `4` correct, `1` wrong; gate failed.
- Adoption remained `adopted=false`.

Interpretation:

- Slot-key deltas helped the agent write narrower visible safety rules, but
  private selection still failed on the same podcast/radio and radio room-place
  boundary errors.
- The visible raw workspace had examples for `general_joke.joke_type` and
  `play_podcasts.podcast_name`, but no visible `play_radio.house_place`
  example. That room-place error is therefore not directly recoverable from
  visible slot deltas in this fixed snapshot.
- The podcast/radio private failure is an intent-boundary error, not an
  intent-correct slot mismatch. It is not well represented by `slot_risk_backlog`.

Follow-up implementation:

- Family diagnostics now include `intent_confusion_backlog`, derived only from
  visible wrong-intent examples.
- Items are keyed by `teacher_intent` and `predicted_intent` pair and include
  `total`, `default_accepts`, `accepted_wrong`, `max_guard_probability`, and a
  few high-guard examples.
- `target_diagnostics.json` exposes latest/baseline train-audit and
  visible-cross-audit variants. The queue is diagnostic-only and does not change
  selection/adoption gates.

Smoke command:

```bash
uv run edge-mvp l2 target-evolve \
  --traces runs/l2-list-fallback-tuned-3k-r1/traces.jsonl \
  --out-dir runs/l2-target-intent-confusion-smoke-r1/job \
  --max-traces 1000 \
  --mode dry-run \
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

Smoke result:

- Evidence class: `short_fixed_snapshot_probe`.
- `latest_intent_confusion_backlog` wrote a visible `audio_volume_up ->
  audio_volume_mute` pair.
- `latest_visible_cross_audit_intent_confusion_backlog` wrote a visible
  `audio_volume_down -> audio_volume_up` pair.
- Both used `visibility=visible_validation_only`.
- `target_diagnostics.json`, `round_state.json`, and `objective.json` did not
  contain `selection_holdout` or `promotion_holdout`.

## Intent-confusion live result

Run:

```bash
uv run edge-mvp l2 target-evolve \
  --traces runs/l2-list-fallback-tuned-3k-r1/traces.jsonl \
  --out-dir runs/l2-real-agent-ratio40-intent-confusion-live-r1/job \
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
- Final visible validation: `29` accepted, `29` correct, `0` wrong; gate passed.
- Visible support passed: `29` correct accepts, required `10`.
- Final train audit: `89` accepted, `89` correct, `0` wrong; train-audit safety
  passed.
- Final visible cross-audit: `35` accepted, `35` correct, `0` wrong; gate
  passed.
- Private selection: `6` accepted, `4` correct, `2` wrong; gate failed.
- Private promotion: `9` accepted, `6` correct, `3` wrong; gate failed.
- Adoption remained `adopted=false`.

Interpretation:

- Intent-confusion diagnostics were visible to the agent and it produced a
  candidate with stronger visible metrics, but hidden wrong accepts did not
  improve. Private selection still failed on the same podcast/radio and radio
  room-place boundary patterns.
- The stable `play_radio.house_place` private miss has no visible
  `play_radio.house_place` support in this split. That makes it a visible-data
  coverage gap, not a presentation gap in the current slot-risk queue.
- Further diagnostics should avoid leaking private feedback or merely adding
  more family queues. The next plausible design direction is a visible schema
  cue summary across slots, not intents: for example, showing that room words
  commonly map to `house_place` even when the specific `play_radio` +
  `house_place` pair is absent. That would be a visible-only generalization aid,
  not a new gate.

## Visible slot-cue summary

The follow-up implementation adds `visible_slot_cue_summary` to
`target_diagnostics.json`. Unlike the safety, slot-risk, and intent-confusion
queues, this is not a risk queue. It is a compact visible schema index built
only from visible train and validation teacher rows.

Each item is keyed by `slot_key` and includes:

- `total`,
- `top_teacher_intents`,
- `top_values`,
- up to three visible examples with request id, utterance, teacher intent, and
  slot value.

The purpose is to let the agent see cross-intent slot semantics. For example,
even when the fixed split has no visible `play_radio.house_place` pair, it can
still show that room values such as `bedroom`, `kitchen`, `bathroom`, and
`living room` commonly map to `house_place` in visible teacher labels. This is
diagnostic-only and does not change selection/adoption gates.

Smoke command:

```bash
uv run edge-mvp l2 target-evolve \
  --traces runs/l2-list-fallback-tuned-3k-r1/traces.jsonl \
  --out-dir runs/l2-target-visible-slot-cue-budget40-smoke-r1/job \
  --max-traces 1000 \
  --mode dry-run \
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

Smoke result:

- Evidence class: `short_fixed_snapshot_probe`.
- `visible_slot_cue_summary` used `visibility=visible_validation_only`.
- `item_limit` was `40`, and all 40 visible slot-key budget slots were filled.
- Source splits were visible `train` plus all five visible validation folds.
- `house_place` appeared within the default 40 slot keys and listed room values
  such as `bedroom`, `house`, `kitchen`, `living room`, and `bathroom`.
- `podcast_descriptor` and `podcast_name` also appeared. `podcast_name`
  carried visible `play_podcasts` examples such as `go to the next episode of
  the united states of anxiety podcast`.
- Its examples came from visible rows such as `bedroom lights off now`, `turn
  off lights of kitchen`, and `can you turn my bathroom lights off`.
- `target_diagnostics.json`, `round_state.json`, and `objective.json` did not
  contain `selection_holdout` or `promotion_holdout`.

Live command:

```bash
uv run edge-mvp l2 target-evolve \
  --traces runs/l2-list-fallback-tuned-3k-r1/traces.jsonl \
  --out-dir runs/l2-real-agent-ratio40-visible-slot-cue-live-r1/job \
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

Live result before widening the slot-cue budget:

- Evidence class: `fixed_snapshot_research`.
- Final visible validation: `25` accepted, `25` correct, `0` wrong; gate
  passed.
- Visible support passed: `25` correct accepts, required `10`.
- Final train audit: `80` accepted, `80` correct, `0` wrong; safety passed.
- Final visible cross-audit: `23` accepted, `23` correct, `0` wrong; gate
  passed.
- Private selection: `6` accepted, `5` correct, `1` wrong; gate failed.
- Private promotion: `4` accepted, `4` correct, `0` wrong; gate passed.
- The private selection wrong accept was `play me a radio drama podcast`,
  teacher `play_podcasts` with `podcast_name=radio drama`, predicted
  `play_radio` with no slots.

Interpretation:

- The new visible slot-cue summary helped expose `house_place`, but the default
  16-item budget still hid lower-frequency `podcast_name`.
- Visible rows in the same split contain `podcast_name` examples such as
  `get me the latest episode of the friends podcast` and `play my video game
  news podcast starting where i left off`.
- The next implementation keeps the same diagnostic concept and widens the
  default item budget to 40 so low-frequency, high-signal schema cues are still
  agent-visible without introducing another queue or gate.

Budget-40 live command:

```bash
uv run edge-mvp l2 target-evolve \
  --traces runs/l2-list-fallback-tuned-3k-r1/traces.jsonl \
  --out-dir runs/l2-real-agent-ratio40-slot-cue-budget40-live-r1/job \
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

Budget-40 live result:

- Evidence class: `fixed_snapshot_research`.
- Final visible validation: `21` accepted, `21` correct, `0` wrong; gate
  passed.
- Visible support passed: `21` correct accepts, required `10`.
- Final train audit: `66` accepted, `66` correct, `0` wrong; safety passed.
- Final visible cross-audit: `23` accepted, `23` correct, `0` wrong; gate
  passed.
- Private selection: `7` accepted, `5` correct, `2` wrong; gate failed.
- Private promotion: `4` accepted, `3` correct, `1` wrong; gate failed.

Interpretation:

- The budget change made `podcast_name`, `podcast_descriptor`, `radio_name`,
  `house_place`, and `joke_type` visible in `target_diagnostics.json`, but the
  agent still did not systematically use those cues for slotless accepts.
- The private wrong accepts were all omitted-slot or media-boundary cases that
  visible cues could support: podcast cue omitted by a `play_radio` frame,
  room cue omitted by a `play_radio` frame, and `joke_type` omitted by a
  `general_joke` frame.
- The next implementation keeps the same diagnostic object but makes its usage
  explicit: each item exposes `slot_key_terms`, and the agent program tells the
  agent not to stop until it checks `slot_key_terms`, top values, and examples
  against slotless or missing-slot accepted frames.

Slot-key-terms smoke command:

```bash
uv run edge-mvp l2 target-evolve \
  --traces runs/l2-list-fallback-tuned-3k-r1/traces.jsonl \
  --out-dir runs/l2-target-visible-slot-cue-terms-smoke-r1/job \
  --max-traces 1000 \
  --mode dry-run \
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

Slot-key-terms smoke result:

- `visible_slot_cue_summary.usage_hint` tells the agent to use
  `slot_key_terms`, top values, and examples for slotless/missing-slot accepted
  frames.
- `podcast_name` exposed `slot_key_terms=["podcast", "name"]`.
- `house_place` exposed `slot_key_terms=["house", "place"]`.
- `joke_type` exposed `slot_key_terms=["joke", "type"]`.
- `program.md` included the stopping checklist for slotless/missing-slot
  accepted frames.
- `target_diagnostics.json`, `round_state.json`, and `objective.json` still did
  not contain `selection_holdout` or `promotion_holdout`.

Slot-key-terms live command:

```bash
uv run edge-mvp l2 target-evolve \
  --traces runs/l2-list-fallback-tuned-3k-r1/traces.jsonl \
  --out-dir runs/l2-real-agent-ratio40-slot-cue-terms-live-r1/job \
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

Slot-key-terms live result:

- Evidence class: `fixed_snapshot_research`.
- Final visible validation: `25` accepted, `25` correct, `0` wrong; gate
  passed.
- Visible support passed: `25` correct accepts, required `10`.
- Final train audit: `74` accepted, `74` correct, `0` wrong; safety passed.
- Final visible cross-audit: `25` accepted, `25` correct, `0` wrong; gate
  passed.
- Private selection: `7` accepted, `5` correct, `2` wrong; gate failed.
- Private promotion: `7` accepted, `6` correct, `1` wrong; gate failed.

Interpretation:

- The prompt-level usage hint was still not concrete enough. The agent added
  some room and joke vetoes, but did not add a podcast-specific `play_radio`
  veto and still missed the `play_radio.house_place` kitchen case.
- In the full 2000-trace split, `joke_type` did not appear in the 40-item
  visible slot-cue summary even though it appeared in the smaller smoke split.
- The next implementation widens the default slot-cue budget to 64 and adds
  explicit mandatory cue checks for non-podcast accepted intents with podcast
  cues, slotless accepts with visible room values, and `general_joke` accepts
  with `joke about ...` but no `joke_type`.

Budget-64 smoke command:

```bash
uv run edge-mvp l2 target-evolve \
  --traces runs/l2-list-fallback-tuned-3k-r1/traces.jsonl \
  --out-dir runs/l2-target-visible-slot-cue-budget64-smoke-r1/job \
  --max-traces 1000 \
  --mode dry-run \
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

Budget-64 smoke result:

- `visible_slot_cue_summary.item_limit` was `64`.
- The smoke split had `46` visible slot keys, all included in the summary.
- `podcast_name`, `house_place`, and `joke_type` were present with
  `slot_key_terms`.
- `program.md` contained the mandatory cue checks for podcast, room, and
  `joke about` omitted-slot cases.
- `target_diagnostics.json`, `round_state.json`, and `objective.json` still did
  not contain `selection_holdout` or `promotion_holdout`.

Budget-64 live command:

```bash
uv run edge-mvp l2 target-evolve \
  --traces runs/l2-list-fallback-tuned-3k-r1/traces.jsonl \
  --out-dir runs/l2-real-agent-ratio40-slot-cue-budget64-live-r1/job \
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

Budget-64 live result:

- Evidence class: `fixed_snapshot_research`.
- Final visible validation: `28` accepted, `28` correct, `0` wrong; gate
  passed.
- Visible support passed: `28` correct accepts, required `10`.
- Final train audit: `78` accepted, `78` correct, `0` wrong; safety passed.
- Final visible cross-audit: `26` accepted, `26` correct, `0` wrong; gate
  passed.
- Private selection: `7` accepted, `5` correct, `2` wrong; gate failed.
- Private promotion: `6` accepted, `6` correct, `0` wrong; gate passed.

Interpretation:

- Widening to 64 and adding concrete cue checks fixed the private promotion
  `joke_type` failure, but did not fix private selection.
- The two remaining private selection wrong accepts were stable:
  `play me a radio drama podcast` accepted as slotless `play_radio`, and
  `i want the ryan seacrest show on the radio in the kitchen` accepted as
  slotless `play_radio`.
- The agent still did not add a podcast-specific `play_radio` veto, and its
  room handling did not cover slotless `play_radio` with room values.
- The next design step should avoid more prose-only prompt pressure. A small
  visible-only executable cue-probe tool would let the agent run concrete
  checks such as "slotless `play_radio` must veto visible podcast and room
  cue probes" without exposing private holdout rows.

Slot-cue-probes smoke command:

```bash
uv run edge-mvp l2 target-evolve \
  --traces runs/l2-list-fallback-tuned-3k-r1/traces.jsonl \
  --out-dir runs/l2-target-visible-slot-cue-probes-smoke-r1/job \
  --max-traces 1000 \
  --mode dry-run \
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

uv run --project \
  runs/l2-target-visible-slot-cue-probes-smoke-r1/job/workspace/l2_target/system/darjeeling \
  python runs/l2-target-visible-slot-cue-probes-smoke-r1/job/workspace/l2_target/tools/evaluate.py \
  --workspace runs/l2-target-visible-slot-cue-probes-smoke-r1/job/workspace/l2_target \
  --split slot_cue_probes \
  --out runs/l2-target-visible-slot-cue-probes-smoke-r1/job/workspace/l2_target/runs/slot_cue_probes.json
```

Slot-cue-probes smoke result:

- `workspace_manifest.json`, `data/commands.md`, and `program.md` exposed
  `slot_cue_probes`.
- The default target failed all three visible-only probes:
  `non_podcast_podcast_cue`, `slotless_radio_room_cue`, and
  `general_joke_missing_joke_type`.
- The probe payload used `visibility=visible_validation_only` and
  `gate_role=diagnostic_only_not_selection_or_adoption_gate`.
- `target_diagnostics.json`, `round_state.json`, `objective.json`, and
  `runs/slot_cue_probes.json` still did not contain `selection_holdout` or
  `promotion_holdout`.

Slot-cue-probes live command:

```bash
uv run edge-mvp l2 target-evolve \
  --traces runs/l2-list-fallback-tuned-3k-r1/traces.jsonl \
  --out-dir runs/l2-real-agent-ratio40-slot-cue-probes-live-r1/job \
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

Slot-cue-probes live result:

- Evidence class: `fixed_snapshot_research`.
- Final visible validation: `29` accepted, `29` correct, `0` wrong; gate
  passed.
- Visible support passed: `29` correct accepts, required `10`.
- Final train audit: `105` accepted, `105` correct, `0` wrong; safety passed.
- Final visible cross-audit: `27` accepted, `27` correct, `0` wrong; gate
  passed.
- Slot-cue probes passed in all candidate probe runs.
- Private selection: `7` accepted, `7` correct, `0` wrong; gate passed.
- Private promotion: `7` accepted, `7` correct, `0` wrong; gate passed.
- `selection_decision.selected=true`, `adoption_decision.adopted=true`.

Interpretation:

- Making the slot cue checks executable changed agent behavior. It ran
  `slot_cue_probes`, observed the baseline failures, and implemented target
  postprocess/veto rules for podcast, room, and joke cues.
- The remaining private wrong-accept pattern from earlier runs disappeared:
  both selection and promotion safety backlogs were empty.
- The final candidate kept enough visible support and did not rely on an
  over-conservative config; the agent removed a searched config that collapsed
  visible support.
- This is the first run in the ratio-0.4 visible-pressure series where the
  fixed-snapshot target passed visible validation, visible support, train audit,
  visible cross-audit, private selection, and private promotion together.

Outer replay command:

```bash
cp -R runs/l2-list-fallback-tuned-3k-r1 \
  runs/l2-real-agent-ratio40-slot-cue-probes-outer-3k-r1

uv run edge-mvp l2 promote-target \
  --target-run runs/l2-real-agent-ratio40-slot-cue-probes-live-r1/job \
  --run-dir runs/l2-real-agent-ratio40-slot-cue-probes-outer-3k-r1

uv run edge-mvp l2 replay-target \
  --run-dir runs/l2-real-agent-ratio40-slot-cue-probes-outer-3k-r1 \
  --traces runs/l2-real-agent-ratio40-slot-cue-probes-outer-3k-r1/traces.jsonl \
  --out runs/l2-real-agent-ratio40-slot-cue-probes-outer-3k-r1/reports/l2_target_outer_replay.json
```

Outer replay result:

- Candidate generation: `gen_003_l2_target`; parent baseline:
  `gen_002_candidate`.
- Candidate was inner-adopted: `true`; it was not a non-adopted diagnostic
  stage.
- Baseline: `L0=2344`, `L1=4`, `L2=0`, `L4=652`, frame EM `1.0`, cost
  per 100 requests `0.217333`.
- Candidate: `L0=2344`, `L1=4`, `L2=32`, `L4=620`, frame EM `0.997333`,
  cost per 100 requests `0.206720`.
- L2 accepted accuracy was `24/32 = 0.75`; wrong accept rate was
  `0.002667`.
- Decision: not promoted, `accuracy regression exceeds epsilon`; regressed
  layer: `L2`.

Outer replay wrong accepts:

- Generic radio phrases were over-slotted as concrete `radio_name`: `play the
  radio station`, `play random radio station`, and `press play on the radio`.
- `on the radio it is time for good music` missed the teacher `media_type`
  slot.
- Bare upcoming-events queries were accepted as `recommendation_events` instead
  of `calendar_query`.
- `what's the funniest joke` missed `joke_type`.
- `change the volume level to nineteen please` missed `change_amount`.

Interpretation:

- The first `slot_cue_probes` design was too narrow. It caught the private
  selection/promotion failures that had appeared earlier, but it did not test
  generic radio-name overfills, radio media-type cues, bare upcoming-events
  intent boundaries, joke adjectives, or spoken teen-number volume amounts.
- Re-evaluating the adopted target with the expanded probe code produced
  `8` probes, `3` passes, and `5` failures:
  `play_radio_generic_station_name`,
  `play_radio_music_media_type_cue`,
  `recommendation_events_bare_upcoming_events`,
  `general_joke_adjective_missing_joke_type`, and
  `audio_volume_spoken_amount_cue`.
- The next implementation keeps the same diagnostic-only split and broadens
  `slot_cue_probes` rather than introducing a new gate or more terminology.
  The goal is to make these visible-derived cue risks executable for the next
  agent run before private selection/promotion or outer replay are consulted.

Expanded-probes live command:

```bash
uv run edge-mvp l2 target-evolve \
  --traces runs/l2-list-fallback-tuned-3k-r1/traces.jsonl \
  --out-dir runs/l2-real-agent-ratio40-slot-cue-probes-expanded-live-r1/job \
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

Expanded-probes live result:

- Evidence class: `fixed_snapshot_research`.
- Final visible validation: `24` accepted, `24` correct, `0` wrong; gate
  passed.
- Visible support passed: `24` correct accepts, required `10`.
- Final train audit: `76` accepted, `76` correct, `0` wrong; safety passed.
- Final visible cross-audit: `23` accepted, `23` correct, `0` wrong; gate
  passed.
- Slot-cue probes passed: `8/8`.
- Private selection: `4` accepted, `4` correct, `0` wrong; gate passed.
- Private promotion: `4` accepted, `4` correct, `0` wrong; gate passed.
- `selection_decision.selected=true`, `adoption_decision.adopted=true`.

Expanded-probes outer replay result:

- Run: `runs/l2-real-agent-ratio40-slot-cue-probes-expanded-outer-3k-r1`.
- Baseline: `L0=2344`, `L1=4`, `L2=0`, `L4=652`, frame EM `1.0`, cost
  per 100 requests `0.217333`.
- Candidate: `L0=2344`, `L1=4`, `L2=21`, `L4=631`, frame EM `0.999333`,
  cost per 100 requests `0.210368`.
- L2 accepted accuracy was `19/21 = 0.904762`; wrong accept rate was
  `0.000667`.
- Decision: not promoted, `accuracy regression exceeds epsilon`; regressed
  layer: `L2`.

Remaining outer replay wrong accepts:

- `delete all the events of today` was accepted as `calendar_remove {}` but the
  teacher frame required `date=today`.
- `what's the funniest joke` was accepted as `general_joke {}` but the teacher
  frame required `joke_type=funniest`.

Interpretation:

- The expanded probes materially improved outer behavior: L2 wrong accepts
  dropped from `8` to `2`, and candidate L4 calls dropped by `21` instead of
  `32`, but zero-regression replay still failed.
- The two remaining failures are still visible-schema cue omissions. The next
  implementation adds two more visible-only probes:
  `calendar_remove_today_date_cue` and
  `general_joke_superlative_missing_joke_type`.
- Re-evaluating the expanded target with those new probes produced `10`
  probes, `8` passes, and exactly those `2` failures.

Expanded2-probes live command:

```bash
uv run edge-mvp l2 target-evolve \
  --traces runs/l2-list-fallback-tuned-3k-r1/traces.jsonl \
  --out-dir runs/l2-real-agent-ratio40-slot-cue-probes-expanded2-live-r1/job \
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

Expanded2-probes live result:

- Evidence class: `fixed_snapshot_research`.
- Final visible validation: `29` accepted, `29` correct, `0` wrong; gate
  passed.
- Visible support passed: `29` correct accepts, required `10`.
- Final train audit: `82` accepted, `82` correct, `0` wrong; safety passed.
- Final visible cross-audit: `30` accepted, `30` correct, `0` wrong; gate
  passed.
- Slot-cue probes passed: `10/10`.
- Private selection: `6` accepted, `4` correct, `2` wrong; gate failed.
- Private promotion: `7` accepted, `6` correct, `1` wrong; gate failed.
- `selection_decision.selected=false`, `adoption_decision.adopted=false`.

Private wrong accepts:

- Selection: `show me events that are going on right now` was accepted as
  `recommendation_events {}`, but teacher was `calendar_query {}`.
- Selection: `start coffee machine` was accepted as `iot_coffee {}`, but
  teacher required `device_type=coffee machine`.
- Promotion: `what on my list to do today evening` was accepted as
  `lists_query {date=today}`, but teacher also required
  `timeofday=evening`.

Interpretation:

- The extra probes fixed the previously observed 3k replay misses, but the
  agent added `target/config.json` with `accept_threshold=0.9` to restore more
  visible coverage. That threshold-lowering changed the risk profile: visible
  audits and probes passed, but private selection/promotion exposed new wrong
  accepts.
- The design response should not be another pile of one-off probes yet. The
  lower-tax fix is to clarify the existing optimization policy: once visible
  support passes, `target/config.json` must not lower `accept_threshold` merely
  to raise raw accepts. The agent should prefer target-local veto/postprocess
  rules and remove threshold-lowering config that only recovers coverage after
  safety vetoes.

Policy-guided live result:

- Run: `runs/l2-real-agent-ratio40-slot-cue-probes-policy-live-r1`.
- Evidence class: `fixed_snapshot_research`.
- Final visible validation: `30` accepted, `30` correct, `0` wrong; gate
  passed.
- Visible support passed: `30` correct accepts, required `10`.
- Final train audit: `93` accepted, `93` correct, `0` wrong; safety passed.
- Final visible cross-audit: `26` accepted, `26` correct, `0` wrong; gate
  passed.
- Slot-cue probes passed: `10/10`.
- Private selection: `5` accepted, `5` correct, `0` wrong; gate passed.
- Private promotion: `4` accepted, `4` correct, `0` wrong; gate passed.
- `selection_decision.selected=true`, `adoption_decision.adopted=true`.
- The selected snapshot did not contain `target/config.json`; `accept_threshold`
  stayed at `0.93`.

Policy-guided 3k outer replay:

- Run: `runs/l2-real-agent-ratio40-slot-cue-probes-policy-outer-3k-r1`.
- Baseline: `L0=2344`, `L1=4`, `L2=0`, `L4=652`, frame EM `1.0`, cost
  per 100 requests `0.217333`.
- Candidate: `L0=2344`, `L1=4`, `L2=27`, `L4=625`, frame EM `1.0`, cost
  per 100 requests `0.208378`.
- L2 accepted accuracy was `27/27 = 1.0`; wrong accept rate was `0.0`.
- Decision: promoted, `objective improved within gates`.

Policy-guided 10k outer replay:

- Run: `runs/l2-real-agent-ratio40-slot-cue-probes-policy-outer-10k-r1`.
- Baseline: `L0=9878`, `L1=1`, `L2=0`, `L4=121`, frame EM `1.0`, cost
  per 100 requests `0.012100`.
- Candidate: `L0=9878`, `L1=1`, `L2=6`, `L4=115`, frame EM `0.9999`, cost
  per 100 requests `0.011503`.
- L2 accepted accuracy was `5/6 = 0.833333`; wrong accept rate was `0.0001`.
- Decision: not promoted, `accuracy regression exceeds epsilon`.

10k wrong accept:

- `put on radio mango` was accepted as `play_radio {}`, but teacher required
  `radio_name=radio mango`.

Interpretation:

- The config policy fixed the previous private-gate failure and produced the
  first 3k outer replay in this series with zero frame regression and a
  promoted decision.
- The 10k replay found a longer-tail slot cue: radio-name phrases of the form
  `put on radio <name>`. This is still in the same visible slot-cue family, so
  the next implementation adds `play_radio_missing_radio_name_cue` to the
  existing `slot_cue_probes` split rather than creating a new diagnostic.
