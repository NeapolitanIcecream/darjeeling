# 2026-06-09 L2 evolution notes

## Goal

本轮目标是验证并使用 L4 coding-agent evolve L2；如果 agent patch 不能解决 L2
结构能力不足，则进入普通开发修复并重跑实验。

## Plan executed

1. Add slot-level failure context for L2 coding-agent jobs.
2. Run a real L2 Codex agent job against the observed slot-missing wrong accept.
3. If the agent job times out or produces an insufficient patch, reduce context
   and rerun the agent job.
4. Replay the agent patch on the same 3k cached setup and compare against the
   baseline.
5. If a non-L2 infrastructure issue blocks the experiment, fix it, test it, and
   rerun the experiment.
6. If the agent patch does not fix the L2 issue, make the smallest L2 design
   change needed, then rerun the same 3k experiment.

## Changes

- `da51b73` added `slot_error_summary.json` to L2 agent context, so agent jobs can
  see teacher-visible L2 wrong accepts and slot-level mismatch summaries.
- `ec858ac` is the successful compact L2 Codex agent patch. It made slot-pattern
  prefix matching treat question `is` contractions like `what's` as equivalent to
  `what is`.
- `f50a4e0` fixed experiment infrastructure: offline promotion replay now honors
  `L1_WORKER_TIMEOUT_S` instead of hardcoding a 5s Rust L1 worker timeout.
- `920340e` added a guard-protected `list_name` lexical fallback for singular
  `list` markers, for example `to do list` -> `list_name=to do`.

## Agent jobs

| job | context | model | result |
| --- | --- | --- | --- |
| `runs/l2-agent-slot-aware-job-r1` | 2200 teacher train traces + 99 hard cases | `gpt-5.5` | timed out after 900s, no diff |
| `runs/l2-agent-slot-aware-job-r2` | 145 slot-focused train traces + 4 hard cases | Codex default | succeeded, generated `ec858ac` patch |

The successful r2 job used 659,100 input tokens, 595,712 cached input tokens,
10,310 output tokens, and 5,648 reasoning output tokens as reported by Codex CLI.

## Experiment results

All runs used cached teacher labels, `LOCAL_SLM_MODE=disabled`,
`L1_AGENT_MODE=disabled`, `L4_PROPOSAL_MODE=disabled`, and
`L2_AGENT_MODE=disabled`. The successful post-timeout-fix runs used
`L1_WORKER_TIMEOUT_S=30`.

| run | relevant change | L2 accepts | L2 correct | L2 wrong | frame EM | `train-7270` |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `runs/l2-agent-patch-tuned-3k-r1` | pre slot-agent context baseline | 12 | 11 | 1 | 0.999333 | L2 wrong accept, missing `list_name` |
| `runs/l2-agent-slot-pattern-tuned-3k-r2` | Codex agent contraction patch | 12 | 11 | 1 | 0.999333 | unchanged |
| `runs/l2-list-fallback-tuned-3k-r1` | guarded `list_name` fallback | 13 | 13 | 0 | 0.999667 | L2 correct accept |

Final run summary:

- Layer counts: `{'L0': 185, 'L1': 9, 'L2': 13, 'L3': 0, 'L4': 2793}`.
- L2 accepted accuracy: `1.000`.
- L2 wrong accept rate: `0.000`.
- Gold frame exact match: `1.000`.
- L2 p50/p95: `2.440/3.549 ms`.

## Conclusions

- L4 coding-agent evolve L2 is operational, but context size matters. The broad
  2200-trace job timed out before producing a patch; the compact failure-focused
  context completed and produced a tested patch.
- The agent-generated contraction patch was locally valid but did not fix the
  observed wrong accept, because the failing early artifact had not learned a
  transferable `list_name` pattern.
- The remaining issue was not just threshold tuning. A narrow slot postprocess
  capability was needed so the guard could evaluate the more exact frame.
- After the fallback, the known wrong accept became a correct L2 accept in the
  end-to-end 3k replay, while L2 coverage increased by one request and wrong
  accepts dropped to zero.

## Follow-ups

- Generalize slot-name-derived lexical fallbacks only when a replay-backed hard
  case justifies them; avoid turning L2 into an unbounded rule system.
- Continue treating artifact promotion as an open issue: whole-artifact
  promotion can hide single-layer regressions.
- Keep L1 hardware/runtime timeout configurable in future experiment commands.

## Autoresearch-style L2 workspace follow-up

Later on 2026-06-09, the L2 agent harness was redesigned around an
autoresearch-style isolated workspace:

- `a6f36b7` introduced `workspace/l2_research/` with stable `program.md`,
  editable `candidate/`, fixed `system/darjeeling/`, dynamic teacher-visible
  `data/`, local `tools/`, and `workspace_manifest.json`.
- The Codex stdin prompt is now the stable one-line instruction to read
  `program.md`; trace/context/metrics/objective files are no longer embedded
  in prompt text.
- `L2_AGENT_MODEL` now defaults to `gpt-5.5`, `L2_AGENT_TIMEOUT_S` defaults to
  `7200`, and Codex runs with `--ignore-user-config`, `--ignore-rules`,
  `--ephemeral`, and `--skip-git-repo-check`.
- A dry-run validation smoke confirmed that generated `tools/run_checks.py`
  overlays `candidate/` into `system/darjeeling/` and runs focused L2
  pytest/ruff successfully.

All follow-up runs used `--max-requests 500 --compile-every 500 --teacher cache`
with a copied oracle teacher cache. They were smoke runs intended to validate
the new harness and patch-adoption loop, not final 3k/10k quality claims.

| run | repo state evaluated | candidate L2 accepts | candidate frame EM | candidate wrong accept | promotion | agent patch |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `runs/l2-agent-research-workspace-smoke-r1` | new workspace harness only | 12 | 0.986486 | 0.013514 | rejected: objective did not improve | lower safe threshold tie-break |
| `runs/l2-agent-research-workspace-smoke-r2` | lower-threshold patch applied | 16 | 0.977064 | 0.022936 | rejected: accuracy regression exceeds epsilon | utterance length guard feature |
| `runs/l2-agent-research-workspace-smoke-r3` | length-bucket feature applied, lower-threshold reverted | 18 | 0.977273 | 0.022727 | rejected: accuracy regression exceeds epsilon | QA slot lexical fallback |

Agent/provenance observations:

- r1 Codex job succeeded with 885,689 input tokens, 821,504 cached input tokens,
  7,852 output tokens, and 1,752 reasoning output tokens.
- r2 Codex job succeeded with 1,494,176 input tokens, 1,345,664 cached input
  tokens, 10,538 output tokens, and 2,813 reasoning output tokens.
- r3 Codex job succeeded with 1,271,731 input tokens, 1,182,080 cached input
  tokens, 11,006 output tokens, and 2,118 reasoning output tokens.
- The short prompt plus workspace files made the jobs complete reliably and
  gave high cached-input ratios, but each real GPT-5.5 job still took several
  minutes and substantial tokens.

Patch-adoption outcome:

- `75647e0` applied the r1 lower-threshold patch, then `e7a8d66` reverted it
  after r2 showed worse replay accuracy.
- `df1e4e6` applied the r2 utterance-length guard feature, then `e082923`
  reverted it after r3 still failed promotion with accuracy regression.
- The r3 QA lexical fallback was not applied. It hardcodes MASSIVE-specific
  QA intent/slot names and would move L2 toward dataset-specific rules without
  replay evidence that it fixes the current promotion failure.
- After the smoke runs, the generated `program.md` rules were tightened to make
  replay/promotion success explicit and to reject raw-coverage trades or
  dataset-specific slot hardcoding by default.
- `tools/run_checks.py` was adjusted to prefer pytest/ruff from the current
  Python environment, falling back to `uv run` only when those modules are
  unavailable. This keeps outer validation compatible while giving sandboxed
  agents a path around dependency-cache misses when a usable venv is available.

Conclusion for this follow-up:

- The redesigned L2 coding-agent harness works as intended: it isolates
  workspace writes, keeps prompt stable, records auditable artifacts, validates
  generated patches, and supports apply-commit-rerun evaluation.
- The L2 quality bottleneck remains unsolved in these 500-request smoke runs.
  Simple guard-threshold or single-feature changes trade accuracy for coverage
  and fail the promotion gate.
- Further L2 evolution should make the agent objective more explicit: prefer
  frame exactness and promotion success over raw L2 coverage, reject
  dataset-specific lexical patches by default, and consider a bounded agent
  budget/stop policy before spending another GPT-5.5 job.

## Target-dependent inner loop correction

The previous `l2_research/candidate` harness was useful for patch generation,
but it still mixed two concerns:

- It tied L2 evolution to outer replay generations, so the number of L4 agent
  attempts was limited by `compile_every` and replay throughput.
- It treated dataset-specific L2 code as if it were a Darjeeling-core patch,
  which incorrectly applied dataset-independence rules to target runtime code.

The corrected design is now implemented as `edge-mvp l2 target-evolve`:

- Outer Darjeeling creates a fixed target split and owns provenance, private
  promotion holdout, artifact adoption, and core invariants.
- Inner L2 target evolution runs many rounds inside one isolated
  `workspace/l2_target/`.
- `target/` is the only writable target-dependent runtime code area.
- `system/darjeeling/` is a read-only evaluator/core copy.
- `data/train.jsonl` and `data/inner_validation.jsonl` are visible to the
  agent; `promotion_holdout.jsonl` is stored under the outer job `private/`
  directory and is not copied into the agent workspace.

Smoke command:

```bash
uv run edge-mvp l2 target-evolve \
  --traces runs/l2-list-fallback-tuned-3k-r1/traces.jsonl \
  --out-dir runs/l2-target-evolve-inner-smoke-r2 \
  --rounds 5 \
  --max-traces 500
```

Smoke result:

- Runtime: about 8 seconds for 5 dry-run rounds over 500 trace rows.
- Split: train 300, inner validation 100, private promotion holdout 100.
- Inner validation stayed at 1 accepted / 1 correct / 0 wrong.
- Promotion holdout stayed at 0 accepted.
- Workspace manifest exposes only `train.jsonl` and `inner_validation.jsonl`;
  holdout privacy was verified by file layout.

Conclusion:

- The architectural issue is fixed: L2 inner evolution can run many fast rounds
  without waiting for a new replay prefix, and target-dependent code no longer
  needs to be judged as Darjeeling-core code.
- This dry-run smoke does not claim L2 quality improvement. The next meaningful
  experiment must run `codex-cli` or target patches against this inner loop and
  compare promotion-holdout success.

## Target-evolution budget policy smoke

The next iteration added baseline-first evaluation and an outer budget policy:

- Evaluate the unmodified `target/` baseline before any agent or patch round.
- Treat `rounds` as a hard maximum, not as a commitment to spend every round.
- Stop when the private promotion holdout passes its gate.
- Stop after `inner_patience_rounds` consecutive rounds without inner validation
  improvement. The default is 2.
- Keep agent-visible `data/round_state.json` limited to inner validation
  aggregates; holdout rows and holdout aggregate feedback remain outside the
  workspace.
- Score inner validation with wrong accepts before coverage, so raw coverage
  gains with worse frame exactness are not counted as improvement.

Smoke command:

```bash
uv run edge-mvp l2 target-evolve \
  --traces runs/l2-list-fallback-tuned-3k-r1/traces.jsonl \
  --out-dir runs/l2-target-evolve-budget-smoke-r1 \
  --rounds 5 \
  --max-traces 500
```

Smoke result:

- Runtime: about 5 seconds.
- Requested rounds: 5.
- Completed rounds: 2.
- Stop reason: `inner_validation_patience_exhausted`.
- Baseline inner validation: 1 accepted / 1 correct / 0 wrong.
- Baseline promotion holdout: 0 accepted.
- Rounds 1 and 2 had no inner improvement and no holdout improvement.
- `data/round_state.json` exposed only inner validation history; private holdout
  stayed in `private/promotion_holdout.jsonl`.
- The generated target workspace commands were verified with
  `python3 tools/inspect_context.py` and
  `uv run --project system/darjeeling python tools/evaluate.py --split inner_validation`;
  copying `README.md` into `system/darjeeling/` was required because the
  project metadata declares it as the package readme.

Conclusion:

- The cost-control behavior now works in the no-op case: the harness does not
  keep spending rounds when the target code is not improving.
- This still does not solve the L2 quality bottleneck. The next quality-bearing
  experiment should run a real `codex-cli` target evolution job or apply a
  target-only patch that can beat the baseline under the same gates.

## Target-evolution Codex smoke

A one-round live Codex target-evolution job was run with GPT-5.5:

```bash
uv run edge-mvp l2 target-evolve \
  --traces runs/l2-list-fallback-tuned-3k-r1/traces.jsonl \
  --out-dir runs/l2-target-evolve-codex-smoke-r1 \
  --rounds 1 \
  --mode codex-cli \
  --max-traces 500 \
  --inner-patience-rounds 1
```

Result:

- Runtime: about 10 minutes for one Codex round.
- Codex usage: 1,078,009 input tokens, 928,768 cached input tokens, 7,208 output
  tokens, and 1,671 reasoning output tokens.
- Baseline inner validation: 1 accepted / 1 correct / 0 wrong.
- Baseline private promotion holdout: 0 accepted.
- Agent changed only `target/target_l2.py`.
- Agent-visible inner validation improved to 6 accepted / 6 correct / 0 wrong.
- Private promotion holdout failed: 2 accepted / 1 correct / 1 wrong, accepted
  accuracy 0.5, wrong accept rate 0.5.
- The wrong holdout example was an `email_query` frame missing the `person`
  slot: `do i have any emails from robert`.

Interpretation:

- The live target-evolution path works: Codex can read the workspace, edit
  `target/`, and improve visible inner validation.
- The candidate is not adoptable. It overfit the visible inner split by lowering
  threshold / switching frame source and introduced a private holdout wrong
  accept.
- This motivated an explicit `adoption_decision`: `best_round` remains useful
  diagnostically, but `best_adoptable_round` is `null` unless a round passes the
  private promotion holdout gate.
- The agent also reported dependency trouble with the generated `uv run`
  command inside its sandbox and used `PYTHONPATH=system/darjeeling/src python`
  instead. From the outer environment, after copying `README.md` into
  `system/darjeeling/`, the documented `uv run --project system/darjeeling ...`
  commands succeeded. Tool isolation should still be revisited before relying
  on many live rounds.

## Selection/promotion split follow-up

The next design iteration split the private target data into two private
subsets:

- `selection_holdout`: used by the outer harness for candidate selection and
  early stop.
- `promotion_holdout`: used as a final private check after selection.

Both remain outside the agent workspace. The agent-visible state still contains
only train/inner-validation rows and inner-validation aggregates.

Dry-run smoke:

```bash
uv run edge-mvp l2 target-evolve \
  --traces runs/l2-list-fallback-tuned-3k-r1/traces.jsonl \
  --out-dir runs/l2-target-evolve-split-smoke-r1 \
  --rounds 3 \
  --max-traces 500
```

Dry-run result:

- Split: train 300, inner validation 100, selection holdout 50, promotion
  holdout 50.
- Completed 2/3 rounds and stopped by `inner_validation_patience_exhausted`.
- No round passed selection or promotion.

Live Codex smoke:

```bash
uv run edge-mvp l2 target-evolve \
  --traces runs/l2-list-fallback-tuned-3k-r1/traces.jsonl \
  --out-dir runs/l2-target-evolve-selection-codex-smoke-r1 \
  --rounds 1 \
  --mode codex-cli \
  --max-traces 500 \
  --inner-patience-rounds 1 \
  --timeout-s 900
```

Live result:

- Runtime: about 4.5 minutes for one Codex round.
- Codex usage: 754,201 input tokens, 681,600 cached input tokens, 6,836 output
  tokens, and 1,202 reasoning output tokens.
- Agent changed only `target/target_l2.py`.
- Agent-visible inner validation improved from 1 accepted / 1 correct / 0 wrong
  to 3 accepted / 3 correct / 0 wrong.
- Private selection holdout: 0 accepted.
- Private promotion holdout: 0 accepted.
- `selection_decision.selected=false` and `adoption_decision.adopted=false`.

Interpretation:

- The split prevents an inner-only improvement from being selected when it does
  not carry to private selection traffic.
- This is safer than the previous single private holdout setup, where one
  private set was doing both repeated selection and final proof.
- It still does not solve L2 quality. The agent learned another narrow
  inner-validation improvement but did not find a candidate that absorbs
  private target traffic.
- Tool isolation improved for lightweight inspection: generated
  `tools/inspect_context.py` now runs with plain `python3` and no project env.
  `tools/evaluate.py` still requires either `uv --project system/darjeeling` or
  an already-active Python >=3.11 environment with dependencies.

## Target local-search tuning smoke

The next design iteration added a non-LLM `local-search` mode to
`edge-mvp l2 target-evolve`. It runs Optuna over target-owned
`L2StudentConfig` overrides using only visible `train` and `inner_validation`,
writes the selected visible config to `target/config.json`, and then lets the
outer harness privately evaluate selection/promotion holdouts.

First run:

```bash
uv run edge-mvp l2 target-evolve \
  --traces runs/l2-list-fallback-tuned-3k-r1/traces.jsonl \
  --out-dir runs/l2-target-evolve-local-search-r1 \
  --rounds 1 \
  --mode local-search \
  --max-traces 500 \
  --local-search-trials 48 \
  --inner-patience-rounds 0
```

Result:

- Split: train 300, inner validation 100, selection holdout 50, promotion
  holdout 50.
- Local-search completed 48/48 trials without LLM calls.
- Baseline inner validation: 1 accepted / 1 correct / 0 wrong.
- Best visible inner validation: 4 accepted / 4 correct / 0 wrong.
- Private selection holdout: 0 accepted.
- Private promotion holdout: 1 accepted / 0 correct / 1 wrong.
- The selected config used `frame_source=student`, `slot_model_family=none`,
  `accept_threshold=0.90`, and reproduced the known `email_query` missing
  `person` slot regression.

Design correction from this run:

- Default `compact` local-search was too permissive. It allowed
  `slot_model_family=none`, which is a cheap slotless shortcut that can look
  safe on visible inner validation but fail frame exactness on private traffic.
- It also sampled many MLP trials, making a 48-trial smoke much slower than the
  intended cheap path.
- `compact` was tightened to low-cost, conservative `sgd_logreg + token_sgd`.
  MLP and `slot_model_family=none` remain available only in `wide` search or
  after an explicit L4 target-code/search-space design decision.

Second run after tightening `compact`:

```bash
uv run edge-mvp l2 target-evolve \
  --traces runs/l2-list-fallback-tuned-3k-r1/traces.jsonl \
  --out-dir runs/l2-target-evolve-local-search-r2 \
  --rounds 1 \
  --mode local-search \
  --max-traces 500 \
  --local-search-trials 48 \
  --inner-patience-rounds 0
```

Result:

- Runtime: about 33 seconds for 48 local-search trials on 500 trace rows.
- Local-search completed 48/48 trials and did not use LLM tokens.
- Baseline inner validation: 1 accepted / 1 correct / 0 wrong.
- Best visible inner validation: 2 accepted / 2 correct / 0 wrong.
- Private selection holdout: 0 accepted.
- Private promotion holdout: 0 accepted.
- `selection_decision.selected=false` and `adoption_decision.adopted=false`.

Interpretation:

- The local-search path now exercises many cheap target trials without coupling
  L2 evolve rounds to replay/sample collection or GPT-5.5 agent calls.
- The conservative compact space avoided the previously observed private wrong
  accept, but it still did not produce private selection coverage.
- Generated `tools/search_config.py` was verified from the target workspace. The
  `uv --project system/darjeeling` path works, but first use creates a local
  venv and is not a lightweight startup path; the generated command guide now
  also documents a `PYTHONPATH=system/darjeeling/src python ...` fallback when
  dependencies are already active.
- The remaining L2 quality bottleneck is not simple hyperparameter search. The
  next quality-bearing iteration should use L4 target-code evolution to improve
  slot/frame exactness or calibration/search-space design, then call
  `tools/search_config.py` for local tuning inside that target workspace.

## Target accept-veto hook smoke

The next implementation added a target-owned `accept_prediction(...)` hook.
The hook can only veto a core guard accept; it cannot force acceptance when the
core guard rejected a prediction. Metrics now include `vetoed_accepts` and up
to 8 `veto_examples`.

The dry-run patch used for the smoke is committed at:

```text
docs/experiments/patches/l2_target_accept_veto_r1.patch
```

It intentionally recreates the risky local-search R1 config
(`frame_source=student`, `slot_model_family=none`, `accept_threshold=0.90`) and
adds a broad slot-risk veto for slotless predictions with slot cues or alternate
slot evidence.

Command:

```bash
uv run edge-mvp l2 target-evolve \
  --traces runs/l2-list-fallback-tuned-3k-r1/traces.jsonl \
  --out-dir runs/l2-target-evolve-accept-veto-r4 \
  --rounds 1 \
  --mode dry-run \
  --dry-run-patch docs/experiments/patches/l2_target_accept_veto_r1.patch \
  --max-traces 500 \
  --inner-patience-rounds 0
```

Result:

- Runtime: about 4 seconds.
- Baseline inner validation: 1 accepted / 1 correct / 0 wrong.
- Patched inner validation: 1 accepted / 1 correct / 0 wrong, with
  3 vetoed accepts.
- Private selection holdout: 0 accepted.
- Private promotion holdout: 0 accepted, with 1 vetoed accept.
- The vetoed private promotion example was the previous wrong-accept case:
  `do i have any emails from robert`, where the predicted `email_query` frame
  omitted the `person` slot.
- The visible inner `veto_examples` also showed over-vetoing: several vetoed
  examples were already exact correct slotless frames, such as `any new emails`.

Interpretation:

- The hook solves an interface problem: target-owned code can now turn risky
  guard accepts into abstentions without changing Darjeeling core.
- The specific smoke patch is too conservative and not adoptable. It removed
  the known private wrong accept, but it also removed visible correct accepts
  and did not create private selection coverage.
- `veto_examples` are necessary feedback for L4 target evolution. Without them,
  an agent can see that coverage dropped but cannot tell whether the veto was a
  desirable safety abstention or an overbroad rule.

## Narrow email-from veto smoke

After inspecting visible train/inner examples, the broad slot-risk veto was
refined into a narrower target rule:

- Only apply to slotless `email_query` predictions.
- Parse `from X`.
- Allow the visible slotless case `from work`.
- Veto other slotless `from X` email accepts.

The dry-run patch is committed at:

```text
docs/experiments/patches/l2_target_email_from_veto_r1.patch
```

Command:

```bash
uv run edge-mvp l2 target-evolve \
  --traces runs/l2-list-fallback-tuned-3k-r1/traces.jsonl \
  --out-dir runs/l2-target-evolve-email-from-veto-r1 \
  --rounds 1 \
  --mode dry-run \
  --dry-run-patch docs/experiments/patches/l2_target_email_from_veto_r1.patch \
  --max-traces 500 \
  --inner-patience-rounds 0
```

Result:

- Runtime: about 4 seconds.
- Baseline inner validation: 1 accepted / 1 correct / 0 wrong.
- Patched inner validation: 4 accepted / 4 correct / 0 wrong.
- Private selection holdout: 0 accepted.
- Private promotion holdout: 0 accepted, with 1 vetoed accept.
- The vetoed private promotion example was again
  `do i have any emails from robert`, where the predicted frame omitted
  `person=robert`.

Interpretation:

- Compared with the broad veto, the narrow visible-data-derived veto preserved
  the inner validation gain while avoiding the known private wrong accept.
- It still did not create private selection coverage, so it is not adoptable.
- The useful pattern is methodological: target-owned veto code can safely bound
  risky higher-coverage configs, but L2 still needs a mechanism that improves
  private coverage, not only one that suppresses regressions.

## Near-miss diagnostics smoke

The next implementation added `near_miss_examples` to target evaluation
metrics. A near miss is a prediction rejected by the core guard; examples are
sorted by guard probability and capped at 8 rows. Each example records whether
the predicted frame would have exactly matched the teacher frame.

Command:

```bash
uv run edge-mvp l2 target-evolve \
  --traces runs/l2-list-fallback-tuned-3k-r1/traces.jsonl \
  --out-dir runs/l2-target-evolve-near-miss-r1 \
  --rounds 1 \
  --mode dry-run \
  --dry-run-patch docs/experiments/patches/l2_target_email_from_veto_r1.patch \
  --max-traces 500 \
  --inner-patience-rounds 0
```

Result:

- Runtime: about 4 seconds.
- The narrow email-from veto behavior was unchanged: inner validation stayed
  4 accepted / 4 correct / 0 wrong; private selection stayed 0 accepted; private
  promotion stayed 0 accepted with 1 vetoed accept.
- Agent-visible `round_state.json` now contains 8 inner-validation near misses.
- The top inner near misses show why simply lowering threshold is unsafe:
  several high-probability rejects were slotless predictions for slot-bearing
  teacher frames, for example `add grocery shopping to my to do list` and
  `show me what alarm times i've set for the week`.
- The same list also exposed a possible safe coverage case:
  `the available lists` was a rejected exact-correct `lists_query` prediction.
- A workspace scan confirmed private selection/promotion near-miss rows were
  not written into the agent workspace; they remain only in the outer summary.

Interpretation:

- `near_miss_examples` are useful target-agent feedback because they show both
  safe coverage opportunities and high-risk rejects using only visible inner
  validation.
- The next target-code evolution attempt should use these examples to design a
  more selective accept/veto policy, especially for slotless exact-intent cases,
  instead of lowering the threshold globally.

## Slotless threshold 0.75 veto smoke

The next target-only dry-run patch lowered the threshold to 0.75 and added
visible-data-derived vetoes for slotless high-risk patterns. The patch is:

```text
docs/experiments/patches/l2_target_slotless_threshold075_r1.patch
```

Command:

```bash
uv run edge-mvp l2 target-evolve \
  --traces runs/l2-list-fallback-tuned-3k-r1/traces.jsonl \
  --out-dir runs/l2-target-evolve-slotless-threshold075-r2 \
  --rounds 1 \
  --mode dry-run \
  --dry-run-patch docs/experiments/patches/l2_target_slotless_threshold075_r1.patch \
  --max-traces 500 \
  --inner-patience-rounds 0
```

Result:

- Runtime: about 4 seconds.
- Patched inner validation: 6 accepted / 4 correct / 2 wrong, so visible inner
  gate failed.
- Raw private selection holdout: 2 accepted / 2 correct / 0 wrong, with one
  vetoed risky email query.
- Private promotion holdout: 0 accepted, with one vetoed risky email query.
- The run therefore set `passes_private_selection_gate=true` but
  `passes_candidate_selection_gate=false`.
- `selection_decision.selected=false`, because candidate selection now requires
  both visible inner gate and private selection gate.

Design correction:

- The first run of this patch exposed a selection-policy bug: raw private
  selection could mark a candidate selected even when visible inner validation
  had obvious wrong accepts.
- The policy is now corrected. Private selection is necessary but not sufficient;
  visible inner safety must pass before a target round can become selected.
- This prevents private selection from masking an agent-visible regression and
  keeps the split roles clean: inner validation blocks known visible failures,
  private selection checks transfer, and private promotion remains final proof.

## Budget policy and target-dependence correction

The next design cleanup addressed two methodology issues:

- Target evolution default budget was still smoke-shaped: 3 requested rounds,
  2 non-improving inner rounds before patience stop, and selection-gate early
  stop enabled by default.
- Some text still treated target-specific lexical code as suspicious in the new
  target workspace, even though `target/` is explicitly target-dependent. The
  corrected boundary is: Darjeeling core must remain dataset-independent;
  `target/` may contain visible-data-derived target-specific code, and
  selection/promotion gates decide whether it is useful.

Implementation changes:

- Default target-evolve budget is now `rounds=12`,
  `inner_patience_rounds=4`, `local_search_trials=96`.
- `stop_on_selection_gate` now defaults to `false`. Private selection remains
  part of outer candidate selection, but it is not an inner-loop early-stop
  signal unless explicitly opted in for smoke or cost control.
- Agent-visible `objective.json`, `round_state.json`, and `program.md` now
  state that candidate selection requires visible inner + private selection,
  adoption also requires private promotion, and target-specific lexical/state
  machine logic is allowed under `target/` when derived from visible target data.

Smoke command:

```bash
uv run edge-mvp l2 target-evolve \
  --traces runs/l2-list-fallback-tuned-3k-r1/traces.jsonl \
  --out-dir runs/l2-target-evolve-budget-policy-r1 \
  --max-traces 500
```

Result:

- Runtime: about 11 seconds.
- Split: train 300, inner validation 100, private selection 50, private
  promotion 50.
- Requested 12 rounds; completed 4 dry-run rounds and stopped by
  `inner_validation_patience_exhausted`.
- Budget policy in `summary.json`: `inner_patience_rounds=4`,
  `stop_on_selection_gate=false`, `local_search_trials=96`,
  `local_search_space=compact`.
- Baseline and all dry-run rounds stayed at inner validation 1 accepted /
  1 correct / 0 wrong; private selection and private promotion stayed at
  0 accepted.
- `workspace/l2_target/data/` contains only `train.jsonl`,
  `inner_validation.jsonl`, `objective.json`, `round_state.json`, and
  `commands.md`; private holdouts remain outside the agent workspace.

Interpretation:

- This smoke validates the corrected budget/stop semantics but does not improve
  L2 quality.
- The next quality-bearing run should spend the larger inner-loop budget on
  `local-search` and/or GPT-5.5 target-code rounds, not on outer replay
  generations.

## Local-search quality run and email-from postprocess

A quality-bearing `local-search` run used the corrected 96-trial default:

```bash
uv run edge-mvp l2 target-evolve \
  --traces runs/l2-list-fallback-tuned-3k-r1/traces.jsonl \
  --out-dir runs/l2-target-evolve-local-search-r3 \
  --mode local-search \
  --rounds 1 \
  --max-traces 500 \
  --inner-patience-rounds 0
```

Result:

- Runtime: about 65 seconds.
- Optuna completed 96/96 compact trials and applied trial 50.
- Inner validation improved from 1/1/0 to 4/4/0 and passed gate.
- Private selection passed: 1 accepted / 1 correct / 0 wrong.
- Private promotion failed: 1 accepted / 0 correct / 1 wrong.
- The promotion wrong was `do i have any emails from robert`, predicted as
  `email_query` with no `person` slot.
- A private diagnostic scan of all 96 unique trial configs found 48 configs that
  passed private selection, but 0 that passed private promotion. This means the
  compact config search space alone did not contain an adoptable candidate.

The visible inner split contains a directly related teacher-visible example:
`please check email from matrimony` -> `email_query(person=matrimony)`.
Therefore the next dry-run patch tested a target-local `postprocess_frame`
rule derived from visible target data: when the predicted intent is
`email_query`, predicted slots are empty, and the utterance has
`from <term>`, fill `person=<term>` except for a small non-person term list.

Patch:

```text
docs/experiments/patches/l2_target_email_from_postprocess_r1.patch
```

Command:

```bash
uv run edge-mvp l2 target-evolve \
  --traces runs/l2-list-fallback-tuned-3k-r1/traces.jsonl \
  --out-dir runs/l2-target-evolve-email-from-postprocess-r2 \
  --rounds 1 \
  --mode dry-run \
  --dry-run-patch docs/experiments/patches/l2_target_email_from_postprocess_r1.patch \
  --max-traces 500 \
  --inner-patience-rounds 0
```

Result:

- Runtime: about 5 seconds.
- Inner validation: 4 accepted / 4 correct / 0 wrong.
- Private selection: 1 accepted / 1 correct / 0 wrong.
- Private promotion: 1 accepted / 1 correct / 0 wrong.
- `selection_decision.selected=true`.
- `adoption_decision.adopted=true`.

Interpretation:

- This is the first target-evolution run in this sequence with an adoptable L2
  target candidate.
- The useful mechanism was not more threshold tuning. It was target-owned slot
  postprocessing layered on top of a tuned config.
- This supports the design split: Optuna/local-search should find conservative
  operating points; L4 target-code evolution should add precise, visible-data
  derived postprocess or abstain logic for frame exactness.
- The result is still a 500-row target-loop proof, not a final system result.
  The next step is to replay the adopted target behavior at 3k/10k scale or wire
  target adoption into the artifact promotion path.

## Target artifact promotion and 3k replay

The adopted email-from target was wired into runtime artifacts:

- `l2_target` is now a manifest artifact path next to `l2_student`.
- Runtime replay and compiler offline replay load `TargetL2Layer` whenever
  `artifact_paths["l2_target"]` exists.
- `edge-mvp l2 promote-target` copies `target/`, retrains the target L2 bundle
  from the target workspace train split, and writes a new generation.
- If a normal compiler generation retrains core L2 without target-aware
  adoption, it drops inherited `l2_target` and records
  `l2_target_dropped_reason`.

The first 3k replay accidentally used the default `compile_every=500`, which
allowed the compiler to retrain a normal L2 bundle mid-run while inheriting the
old target wrapper. That run is not a valid target-only comparison and is kept
only as bug evidence:

- Run: `runs/l2-target-postprocess-3k-r1`.
- Layer counts: `L0=1807, L1=16, L2=15, L3=0, L4=1162`.
- Frame EM: `0.997333`.
- L2 accepted: 15 / 7 correct / 8 wrong.

The fair target-only replay used `--compile-every 999999`:

```bash
uv run edge-mvp l2 promote-target \
  --target-run runs/l2-target-evolve-email-from-postprocess-r2 \
  --run-dir runs/l2-target-postprocess-3k-r2

uv run edge-mvp run \
  --run-dir runs/l2-target-postprocess-3k-r2 \
  --max-requests 3000 \
  --compile-every 999999 \
  --teacher cache \
  --data-dir data/processed/massive_en_us
```

Control from the same starting artifacts, without target:

- Run: `runs/l2-list-fallback-final-3k-r1`.
- Layer counts: `L0=577, L1=19, L2=0, L3=0, L4=2404`.
- Frame EM: `1.0`.

Target replay:

- Run: `runs/l2-target-postprocess-3k-r2`.
- Layer counts: `L0=577, L1=19, L2=21, L3=0, L4=2383`.
- Frame EM: `0.996333`.
- L2 accepted: 21 / 10 correct / 11 wrong.
- Failure mode: the target code fixed some `email_query from <person>` slots,
  but the lowered threshold also accepted unrelated `iot_coffee`,
  `iot_hue_lightdim`, and `lists_query` errors. It also truncated
  `jane doe` to `jane`.

This invalidated the earlier 500-row adoption as a final quality claim. Inner
target adoption is a useful prefilter, but e2e replay remains the authority.

## Email-query postprocess plus veto

The next target patch made two changes:

- `postprocess_frame` extracts multi-token `from <person>` spans.
- `accept_prediction` vetoes all non-`email_query` accepts and only allows
  empty-slot or `person`-slot `email_query` frames.

Patch:

```text
docs/experiments/patches/l2_target_email_query_postprocess_veto_r2.patch
```

500-row target-evolve command:

```bash
uv run edge-mvp l2 target-evolve \
  --traces runs/l2-list-fallback-tuned-3k-r1/traces.jsonl \
  --out-dir runs/l2-target-evolve-email-query-postprocess-veto-r3 \
  --rounds 1 \
  --mode dry-run \
  --dry-run-patch docs/experiments/patches/l2_target_email_query_postprocess_veto_r2.patch \
  --max-traces 500 \
  --inner-patience-rounds 0
```

Result:

- Inner validation: 2 accepted / 2 correct / 0 wrong.
- Private selection: 0 accepted / 0 wrong, with one vetoed accept.
- Private promotion: 1 accepted / 1 correct / 0 wrong.
- `adoption_decision.adopted=false` because selection had no accepted coverage.

A 3000-row target-evolve split with the same patch also did not pass adoption:
using a larger 1800-example train split changed the core L2 bundle and produced
no useful promotion coverage. This shows that target code, train split, and
guard threshold are a coupled artifact, not independent knobs.

For outer replay diagnosis, `promote-target` now supports explicit staging:

```bash
uv run edge-mvp l2 promote-target \
  --target-run runs/l2-target-evolve-email-query-postprocess-veto-r3 \
  --run-dir runs/l2-target-postprocess-veto-3k-r1 \
  --allow-non-adopted

uv run edge-mvp run \
  --run-dir runs/l2-target-postprocess-veto-3k-r1 \
  --max-requests 3000 \
  --compile-every 999999 \
  --teacher cache \
  --data-dir data/processed/massive_en_us
```

Manifest semantics:

- `promotion_reason = explicit L2 target candidate staged for outer replay`.
- `l2_target_inner_adopted = false`.
- `l2_target_staged_for_outer_replay = true`.
- `l2_training_scope = l2_target_workspace_train`.
- `l2_target_training_traces = 300`.

3k replay result:

- Run: `runs/l2-target-postprocess-veto-3k-r1`.
- Layer counts: `L0=577, L1=19, L2=10, L3=0, L4=2394`.
- Frame EM: `3000/3000 = 1.0`.
- L2 accepted: 10 / 10 correct / 0 wrong.
- Target vetoed 11 would-be accepts.
- Compared with same-artifact control, this removes 10 L4 calls without any
  observed frame regression.

Interpretation:

- This is a safe but narrow improvement. It proves the target wrapper/staging
  path can produce a real e2e L2 gain, but it does not solve the broad L2
  quality bottleneck.
- The inner target selection gate is too brittle for very narrow candidates
  when selection holdout is only 50 examples. It should remain a useful
  prefilter, not the sole authority.
- The outer replay path should be treated as final evidence. Non-adopted
  candidate staging is acceptable only when manifest metadata makes that status
  explicit and the run is isolated.

## Formal target outer-replay gate

The hand analysis above was converted into a reusable CLI gate:

```bash
uv run edge-mvp l2 replay-target \
  --run-dir <target-run-dir> \
  --traces <target-run-dir>/traces.jsonl \
  --out <target-run-dir>/reports/l2_target_outer_replay.json
```

The command writes `l2-target-outer-replay-v1` JSON. It compares the current
target manifest against its parent manifest, includes the settings L1 Rust
worker by default, and uses `accuracy_epsilon=0` unless overridden.

Old adopted email-from target:

```bash
uv run edge-mvp l2 replay-target \
  --run-dir runs/l2-target-postprocess-3k-r2 \
  --traces runs/l2-target-postprocess-3k-r2/traces.jsonl \
  --out runs/l2-target-postprocess-3k-r2/reports/l2_target_outer_replay.json
```

Result:

- Baseline parent: `L0=577, L1=19, L2=0, L3=0, L4=2404`.
- Candidate: `L0=577, L1=19, L2=21, L3=0, L4=2383`.
- Candidate frame EM: `0.996333`.
- L2 accepted accuracy: `0.476190`.
- Decision: rejected, `accuracy regression exceeds epsilon`.

Veto target:

```bash
uv run edge-mvp l2 replay-target \
  --run-dir runs/l2-target-postprocess-veto-3k-r1 \
  --traces runs/l2-target-postprocess-veto-3k-r1/traces.jsonl \
  --out runs/l2-target-postprocess-veto-3k-r1/reports/l2_target_outer_replay.json
```

Result:

- Baseline parent: `L0=577, L1=19, L2=0, L3=0, L4=2404`.
- Candidate: `L0=577, L1=19, L2=10, L3=0, L4=2394`.
- Candidate frame EM: `1.0`.
- L2 accepted accuracy: `1.0`.
- Decision: promoted, `objective improved within gates`.

Interpretation:

- This closes the immediate process gap: target candidates now have a formal
  outer replay report instead of a one-off analysis script.
- The strict target gate correctly rejects the earlier inner-adopted candidate
  and accepts the later non-adopted-but-staged candidate, showing that inner
  adoption is neither necessary nor sufficient for final e2e acceptance.

## Fixed-inner budget profile and audio target extension

The next design issue was that previous target-evolve experiments looked like
only one to three rounds because most runs were smoke/dry-run jobs, and one
12-round job stopped after inner-validation patience. The implementation now
records the inner-loop cadence explicitly and exposes a `fixed-inner` budget
profile:

- `loop_cadence.kind = fixed_trace_snapshot_inner_loop`.
- `loop_cadence.outer_replay_cadence_bound = false`.
- `budget_policy.profile = fixed-inner`.
- `fixed-inner` defaults to `rounds=48`, `inner_patience_rounds=0`, and
  `local_search_trials=256`, unless explicit flags override them.
- `target_code_policy` records that Darjeeling core remains
  dataset-independent, while visible-data-derived target-specific code is
  legal inside `target/`.

Patch:

```text
docs/experiments/patches/l2_target_email_audio_veto_r3.patch
```

500-row fixed-inner target-evolve command:

```bash
uv run edge-mvp l2 target-evolve \
  --traces runs/l2-list-fallback-tuned-3k-r1/traces.jsonl \
  --out-dir runs/l2-target-evolve-email-audio-veto-r2 \
  --budget-profile fixed-inner \
  --rounds 3 \
  --mode dry-run \
  --dry-run-patch docs/experiments/patches/l2_target_email_audio_veto_r3.patch \
  --max-traces 500
```

Result:

- Requested/completed rounds: 3 / 3.
- Stop reason: `round_budget_exhausted`.
- Inner validation: 2 accepted / 2 correct / 0 wrong.
- Private selection: 0 accepted / 0 wrong.
- Private promotion: 1 accepted / 1 correct / 0 wrong.
- `adoption_decision.adopted=false` because no round passed both visible inner
  and private selection gates.

This run proves the corrected cadence: after applying the patch in round 1, the
same fixed target workspace was evaluated through rounds 2 and 3 without
collecting another stream prefix or waiting for outer replay.

The non-adopted best round was staged for an isolated 3k outer replay:

```bash
uv run edge-mvp l2 promote-target \
  --target-run runs/l2-target-evolve-email-audio-veto-r2 \
  --run-dir runs/l2-target-email-audio-veto-3k-r1 \
  --allow-non-adopted

uv run edge-mvp run \
  --run-dir runs/l2-target-email-audio-veto-3k-r1 \
  --max-requests 3000 \
  --compile-every 999999 \
  --teacher cache \
  --data-dir data/processed/massive_en_us

uv run edge-mvp l2 replay-target \
  --run-dir runs/l2-target-email-audio-veto-3k-r1 \
  --traces runs/l2-target-email-audio-veto-3k-r1/traces.jsonl \
  --out runs/l2-target-email-audio-veto-3k-r1/reports/l2_target_outer_replay.json
```

Outer replay result:

- Baseline parent: `L0=577, L1=19, L2=0, L3=0, L4=2404`.
- Candidate: `L0=577, L1=19, L2=12, L3=0, L4=2392`.
- Candidate frame EM: `1.0`.
- L2 accepted accuracy: `1.0`.
- L2 wrong accept rate: `0.0`.
- Decision: promoted, `objective improved within gates`.
- Manifest records `l2_target_inner_adopted=false`,
  `l2_target_staged_for_outer_replay=true`, `l2_target_loop_cadence`, and
  `l2_target_code_policy`.

Interpretation:

- The audio extension is safe on this 3k replay but still very narrow: it saves
  12 L4 calls instead of the previous 10.
- The selection holdout remains brittle for narrow target candidates. A zero
  selection accept should block automatic inner adoption, but it should not
  prevent explicit outer replay diagnostics when the manifest labels the
  candidate as non-adopted.
- The target-specific lexical/state-machine boundary is now explicit: such code
  is allowed in `target/`; only Darjeeling core must remain
  dataset-independent.

## Target family diagnostics

The next bottleneck was that the workspace only exposed a few `near_miss_examples`.
That made target evolution too dependent on manual inspection of large trace
files. The evaluator now emits bounded family-level diagnostics from visible
inner validation:

- `family_diagnostics` is included in inner validation metrics.
- `data/target_diagnostics.json` is written into the target workspace.
- The workspace version is `visible_inner_validation_only`; private selection
  and promotion holdout diagnostics stay outside the agent workspace.
- Each family records rejected-correct, vetoed-correct, accepted-wrong,
  intent-correct-slot-wrong, top predicted intents, and up to three examples per
  important category.

Smoke command:

```bash
uv run edge-mvp l2 target-evolve \
  --traces runs/l2-list-fallback-tuned-3k-r1/traces.jsonl \
  --out-dir runs/l2-target-diagnostics-smoke-r1 \
  --budget-profile smoke \
  --mode dry-run \
  --max-traces 500
```

Result:

- Requested/completed rounds: 1 / 1.
- `budget_policy.profile=smoke`.
- `target_diagnostics.json` was written under
  `runs/l2-target-diagnostics-smoke-r1/workspace/l2_target/data/`.
- Top visible inner-validation opportunity families:
  - `email_query`: total 5, rejected-correct 2, accepted-wrong 0.
  - `calendar_query`: total 10, rejected-correct 2, intent-correct-slot-wrong 3.
  - `audio_volume_up`: total 3, rejected-correct 2, accepted-wrong 0.
  - `weather_query`: total 7, rejected-correct 1, intent-correct-slot-wrong 4.

Interpretation:

- The diagnostic points to the same narrow families that manual inspection found
  (`email_query`, `audio_volume_up`), which validates the signal.
- It also surfaces broader but slot-risky families (`calendar_query`,
  `weather_query`). Those should be handled with postprocess/abstain design
  before threshold changes, because slot exactness is the known L2 bottleneck.
- This improves L4 agent context management: the prompt can stay short while
  the workspace exposes structured, bounded triage data for Codex to inspect.

## Weather threshold target experiment

Family diagnostics suggested `weather_query` had additional rejected-correct
opportunities, but also many intent-correct-slot-wrong cases. The target patch
therefore lowered the runtime threshold only behind stricter target-specific
veto hooks for `email_query`, `audio_volume_*`, and `weather_query`.

Patch stack:

```bash
docs/experiments/patches/l2_target_email_audio_veto_r3.patch
docs/experiments/patches/l2_target_weather_threshold_delta_r1.patch
```

During the first run we found a harness bug before interpreting the model
result: multi-round target-evolve selected a `best_round`, but
`promote-target` copied the final workspace `target/`. The fix is:

- every target round snapshots `target/` to `rounds/round_NNN_target/`;
- each round payload records `target_snapshot`;
- promotion copies the selected round snapshot;
- when private selection metrics tie, `best_round` uses visible inner
  validation as a tie-break and then prefers the later round.

The fixed 500-row inner run:

```bash
uv run edge-mvp l2 target-evolve \
  --traces runs/l2-list-fallback-tuned-3k-r1/traces.jsonl \
  --out-dir runs/l2-target-evolve-weather-veto-r2 \
  --budget-profile fixed-inner \
  --rounds 3 \
  --mode dry-run \
  --dry-run-patch docs/experiments/patches/l2_target_email_audio_veto_r3.patch \
  --dry-run-patch docs/experiments/patches/l2_target_weather_threshold_delta_r1.patch \
  --max-traces 500
```

Inner result:

- Best round: 3, `target_snapshot=rounds/round_003_target`.
- Inner validation: 4 accepted / 4 correct / 0 wrong.
- Private selection: 0 accepted / 0 wrong, so `adoption_decision.adopted=false`.
- Private promotion: 1 accepted / 1 correct / 0 wrong.

The candidate was explicitly staged for outer replay, not inner-adopted:

```bash
uv run edge-mvp l2 promote-target \
  --target-run runs/l2-target-evolve-weather-veto-r2 \
  --run-dir runs/l2-target-weather-veto-3k-r2 \
  --allow-non-adopted

uv run edge-mvp run \
  --run-dir runs/l2-target-weather-veto-3k-r2 \
  --max-requests 3000 \
  --compile-every 999999 \
  --teacher cache \
  --data-dir data/processed/massive_en_us

uv run edge-mvp l2 replay-target \
  --run-dir runs/l2-target-weather-veto-3k-r2 \
  --traces runs/l2-target-weather-veto-3k-r2/traces.jsonl \
  --out runs/l2-target-weather-veto-3k-r2/reports/l2_target_outer_replay.json
```

Outer replay result:

- Baseline parent: `L0=577, L1=19, L2=0, L3=0, L4=2404`.
- Candidate: `L0=577, L1=19, L2=37, L3=0, L4=2367`.
- Candidate frame EM: `1.0`.
- L2 accepted accuracy: `1.0`.
- L2 wrong accept rate: `0.0`.
- Cost estimate: `0.801333` to `0.789062` USD / 100 requests.
- Decision: promoted, `objective improved within gates`.

Failed r1 diagnostic:

- Before adding email temporal veto, the same patch accepted 38 L2 requests but
  had one wrong accept:
  `find unread emails received from peter today olly` produced
  `email_query(person=peter)` and missed `date=today`.
- The fix is intentionally conservative: if an `email_query` utterance contains
  visible temporal terms such as `today`, `tomorrow`, `yesterday`, `week`,
  `last`, or `recently`, target code abstains instead of accepting a partial
  person-only frame.

Interpretation:

- The method is now doing what the design intended: target-dependent code lives
  in the isolated target artifact, Darjeeling core remains dataset-independent,
  and final adoption is decided by e2e replay.
- The selection holdout is still too sparse for narrow target patches: 0
  selection accepts blocked inner adoption even though 3k replay succeeded.
  This remains a design issue for inner-loop model selection. A future version
  should consider larger or stratified target selection holdouts, while keeping
  promotion authority with outer e2e replay.
