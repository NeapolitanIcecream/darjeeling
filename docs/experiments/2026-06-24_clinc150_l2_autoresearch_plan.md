# CLINC150 L2 AutoResearch Experiment Plan

Date: 2026-06-24

Purpose: put the CLINC150 L2 repair back onto Darjeeling's intended L4
AutoResearch path. The previous calibration repair made the failure mode clear:
fixed teacher-distilled L2 plus hand-designed threshold/margin/veto guards is
not enough to robustly handle the high-confidence OOS tail. This plan asks the
next agent to make L4-driven target-local L2 research solve that class of
problem, without returning routine target-specific choices to the main session.

This is not a new benchmark selection task and not a Darjeeling core redesign.
The expected shape is:

```text
CLINC150 teacher-visible data + replay-oracle feedback
  -> target-local L2 AutoResearch workspace
  -> L4 agent proposes and implements L2 feature/model/guard/postprocess changes
  -> visible validation and replay-oracle evaluation
  -> locked-test confirmation only after selection
```

## Required Context Files

Read these before changing code:

- `AGENTS.md`
- `docs/experiments/2026-06-23_benchmark_selection_phase_note.md`
- `docs/experiments/2026-06-23_clinc150_phase1_experiment_plan.md`
- `docs/experiments/2026-06-23_clinc150_phase1_report.md`
- `docs/experiments/2026-06-23_clinc150_l2_cascade_experiment_plan.md`
- `docs/experiments/2026-06-23_clinc150_l2_cascade_report.md`
- `docs/experiments/2026-06-24_clinc150_calibration_repair_plan.md`
- `docs/experiments/2026-06-24_clinc150_calibration_repair_report.md`
- `src/darjeeling/targets/nlu/compiler/l2_target_evolution.py`
- `src/darjeeling/targets/nlu/layers/l2_student.py`
- `src/darjeeling/targets/nlu/layers/l2_target.py`
- `src/darjeeling/targets/nlu/clinc150_phase1.py`
- `src/darjeeling/targets/nlu/main_cli.py`
- `tests/targets/nlu/test_l2_target_evolution.py`
- `tests/targets/nlu/test_clinc150_phase1.py`

Optional but useful background:

- `docs/experiments/2026-06-09_l2_evolution.md`
- `docs/experiments/2026-06-10_l2_evidence_policy.md`

## Execution Isolation

Run this plan in a dedicated git branch and worktree. Do not implement the work
directly in the main worktree.

Suggested branch:

```text
codex/clinc150-l2-autoresearch
```

Suggested worktree:

```text
../darjeeling-clinc150-l2-autoresearch
```

If a suitable branch or worktree already exists, inspect it and continue there
instead of creating a duplicate. Use the dedicated worktree as the workspace root
for implementation, experiments, validation, reports, and the final commit.

The existing `runs/` artifacts are ignored by git and will not automatically
appear in a fresh worktree. Treat `/Users/chenmohan/gits/darjeeling` and
`/Users/chenmohan/gits/darjeeling-clinc150-calibration-repair` as read-only
artifact sources, or copy the minimal required artifacts into the dedicated
worktree before running commands. Validate copied row counts and coverage in the
report.

Do not delete the worktree or branch when complete. Report the branch, worktree
path, commit hash, and cleanliness.

## Current State

CLINC150 all-L4 teacher reliability has passed. Teacher-distilled L2 is
plausible but not accepted for Phase 1.

Previous fixed L2 result:

- threshold `0.98` locked test accepted precision: 98.77%;
- threshold `0.98` locked test accepted coverage: 42.73%;
- cascade delta vs all-L4: 0.000 percentage points.

Calibration repair result:

- selected guard: `guard_probability >= 0.985` plus predicted-intent vetoes
  `credit_score`, `directions`, `spending_history`;
- locked test accepted precision: 98.997%;
- locked test accepted coverage: 38.07%;
- accepted wrong: 21;
- OOS false accepts: 12 of 21 accepted wrong rows;
- cascade delta vs all-L4: 0.000 percentage points.

Interpretation:

- Fixed threshold/margin/entropy/veto search improved precision but missed the
  strict target and reduced coverage.
- The main remaining issue is target-specific: high-confidence OOS and intent
  boundary risk.
- This kind of target-specific repair should be handled by L4 AutoResearch in a
  target-local L2 workspace, not manually specified by the main session.

## Reused Artifacts

Reuse existing paid L4 benchmark artifacts before making new paid calls:

- `runs/clinc150-l2-cascade-20260623/teacher-traces/train-full-stratified/teacher_live_vs_gold.details.jsonl`
- `runs/clinc150-l2-cascade-20260623/teacher-traces/validation-full/teacher_live_vs_gold.details.jsonl`
- `runs/clinc150-l2-cascade-20260623/teacher-traces/test-full/teacher_live_vs_gold.details.jsonl`
- `runs/clinc150-l2-cascade-20260623/distilled-l2/train-full/l2_student.joblib`
- `runs/clinc150-l2-cascade-20260623/distilled-l2/train-full/validation-cascade/clinc150_l2_predictions.jsonl`
- `runs/clinc150-l2-cascade-20260623/distilled-l2/train-full/test-cascade/clinc150_l2_predictions.jsonl`
- `runs/clinc150-calibration-repair-20260624/safety-margin-995/clinc150_calibration_repair_summary.json`

New paid L4 benchmark calls should be zero in the normal case. If an artifact is
missing or invalid, regenerate only the missing rows using the resume-safe
teacher path and keep new observed benchmark spend under `$5`. Record any new
benchmark spend in a cost ledger.

The previous locked-test accepted-error artifact may be used only to verify or
quote the prior report's diagnosis. Do not feed locked-test accepted-error rows,
utterances, labels, or confusion families into AutoResearch context, candidate
selection, local search, prompt text, or generated target code.

If coding-agent or AutoResearch tool spend is visible to the agent, report it.
If it is not visible in artifacts, state that benchmark serving spend and
coding-agent spend are accounted separately.

## Core Design Boundary

Keep CLINC150 and NLU semantics in the NLU target or target-local experiment
code. Do not move intents, OOS conventions, label names, request ids, or
accepted-error examples into Darjeeling core.

Do not modify `TeacherCache` semantics. Use the target-local
`Clinc150L4ReplayOracle` for benchmark accounting: no live L4 call is made during
fallback replay, but fallback rows still count as L4 calls using recorded L4
tokens, cost, latency, parse/schema failure, and retry diagnostics.

Do not hand-edit generated L2 target artifacts in the repo-level workspace.
Repository changes should provide harnesses, prompts, context builders, tests,
and contracts. The generated target artifact should be produced by the L4
AutoResearch workspace or by a simple local-search mode inside that workspace.

Low abstraction tax applies to repo-level harnesses and APIs. It does not mean
the target-local L2 artifact must be elegant. Target-local generated code may be
direct and task-specific if it passes the experiment gates.

## What AutoResearch May Change

The L4 AutoResearch loop may modify target-local L2 behavior, including:

- `L2StudentConfig` search choices such as model family, ngram ranges, feature
  limits, MLP settings, frame source, and threshold defaults;
- prediction metadata needed by target-local guards, such as OOS probability,
  OOS rank, OOS margin, class probabilities, or support/prototype signals;
- target-local `target_l2.py` postprocess and `accept_prediction` logic;
- simple target-local auxiliary detectors or validators trained only on visible
  train/dev data;
- per-intent or per-family thresholds, vetoes, OOS detectors, or support checks
  if selected using visible data only;
- reporting and audit code needed to evaluate the candidate.

The L4 AutoResearch loop must not:

- read locked test labels or accepted-error examples while designing candidates;
- change Darjeeling core;
- change the benchmark or teacher prompt;
- hide strict exact-match metrics behind semantic diagnostics;
- use live L4 calls as a substitute for local validation when replay artifacts
  already exist;
- ask the main session to choose between routine target-specific fixes that can
  be evaluated locally.

## Decision To Support

Make one of these decisions with evidence:

- **Proceed with L2 AutoResearch candidate**: a candidate selected without
  locked test passes locked test with accepted precision >= 99%, lower-layer OOS
  false accept rate <= 2%, final L2+L4 cascade accuracy no worse than 0.5
  percentage points below all-L4, and meaningful locked-test L2 coverage. Use
  >= 40% locked-test coverage as the practical target because the first L2 run
  reached 42.73% and the calibration repair reached 38.07%.
- **Pause and repair AutoResearch harness**: the target-specific repair looks
  plausible, but the AutoResearch workspace, context, replay-oracle accounting,
  or candidate adoption path is not trustworthy enough yet.
- **Reject current L2 shape for CLINC150 Phase 1**: after genuine AutoResearch
  attempts, the system cannot produce robust high-precision L2 absorption with
  meaningful coverage on this benchmark.

Do not return to the main session merely because a target-specific choice is
needed. The purpose of this plan is to let L4 AutoResearch make those choices
inside the target-local boundary.

## Primary Experiment Shape

Primary results must keep lower-layer interference out:

```text
L2 shadow
L2 + L4 replay-oracle fallback
```

Keep L0 disabled or bypassed. Keep L1 and L3 disabled. Optional appendices may
show interaction with other layers, but they must not support the main decision.

## Experiment Root

Use:

```text
runs/clinc150-l2-autoresearch-20260624/
```

Write the final report to:

```text
docs/experiments/2026-06-24_clinc150_l2_autoresearch_report.md
```

If any new paid benchmark L4 calls occur, write or update:

```text
docs/experiments/2026-06-24_clinc150_l2_autoresearch_cost_ledger.json
```

If no new paid benchmark L4 calls occur, state that explicitly and point to the
reused previous ledgers.

## Phase A: AutoResearch Bridge For CLINC150

First inspect whether the existing `l2 target-evolve` path can operate directly
on CLINC150 teacher rows and L2 artifacts. Prefer reusing it. If a small bridge
is needed, implement it target-locally.

The bridge should support:

- constructing teacher-visible CLINC150 traces or an equivalent target-local
  AutoResearch dataset from existing teacher detail rows;
- creating visible train/dev/OOS-heavy contexts for the L4 agent without locked
  test labels;
- evaluating candidate L2 bundles and target modules with
  `Clinc150L4ReplayOracle`;
- writing summaries compatible with the existing report style;
- exporting selected candidate artifacts for outer replay-style evaluation.

Keep the bridge simple. Do not build a new general AutoResearch framework unless
the existing one cannot be extended in a small, target-local way.

Focused tests should cover:

- generated AutoResearch context excludes locked test labels and examples;
- replay-oracle fallback counts L4 rows correctly for AutoResearch candidates;
- candidate evaluation keeps L0/L1/L3 absent from primary metrics;
- selected candidates can be evaluated on validation and locked test only after
  selection;
- generated target artifacts stay target-local and do not alter core contracts.

## Phase B: Baseline Reproduction

Before running new research, reproduce:

- fixed teacher-distilled L2 validation and locked-test metrics;
- calibration repair selected-guard validation and locked-test metrics;
- OOS-heavy accepted-error diagnosis.

This proves that the AutoResearch bridge is evaluating the same problem. If the
numbers differ materially from the previous reports, fix the harness before
starting research.

## Phase C: L4 AutoResearch Loop

Run L4-driven AutoResearch over visible CLINC150 data. Use `agent-session` if the
existing L2 target-evolution workspace supports it cleanly; otherwise use the
best existing combination of `codex-cli`, `local-search`, and target-local
helper commands.

The L4 agent should receive:

- objective and strict gates;
- allowed modification surface;
- training/dev/OOS-heavy visible context;
- baseline metrics and accepted-error summaries;
- commands for training, evaluating, auditing, and writing reports.

The L4 agent should not receive locked-test labels or locked accepted-error
examples.

The agent must autonomously propose, implement, test, and iterate hypotheses.
Examples of acceptable hypotheses include, but are not limited to:

- expose and use OOS probability/rank/margin;
- train a target-local OOS-risk detector;
- search class-specific confidence thresholds;
- add support/prototype distance checks;
- change intent model family or feature set;
- change frame source;
- add target-local `accept_prediction` vetoes selected from visible evidence;
- train per-intent experts or small auxiliary heads if simple and target-local.

These are examples, not instructions. The agent should choose based on evidence.

Run enough iterations to make the decision credible. As a default, require at
least one long `agent-session` or at least three bounded research rounds unless
an earlier candidate strongly passes all visible gates.

## Phase D: Selection Policy

Select candidates without using locked test.

A candidate is eligible for locked test only if it passes visible gates:

- accepted precision >= 99.5% on official validation;
- Wilson lower bound or comparable conservative precision check is stronger than
  the previous repair;
- lower-layer OOS false accept rate <= 2% on validation and train-derived
  OOS-heavy views;
- final L2+L4 cascade delta vs all-L4 >= -0.5 percentage points;
- validation coverage remains meaningful, preferably >= 45%;
- validation uniform and zipf-heavy streams do not reveal a new obvious failure
  mode;
- the implementation is target-local, reproducible, and does not use locked test
  evidence.

If multiple candidates pass, select the one with the best validation coverage
among candidates with an adequate precision/OOS safety margin.

## Phase E: Locked Test And Stream Confirmation

After selection, run locked test once. If it fails narrowly and the reason is
clear from pre-test evidence, one bounded repair iteration is allowed, but record
it as another locked-test exposure.

Confirm on:

- locked test sequential;
- validation sequential;
- validation uniform;
- validation zipf-heavy;
- OOS-heavy diagnostic slices;
- any additional visible cross-folds created during AutoResearch.

For each view, report:

- all-L4 replay-oracle accuracy;
- L2-only raw accuracy;
- L2 accepted precision;
- L2 accepted coverage;
- accepted wrong count and examples;
- lower-layer OOS false accept count and rate;
- final L2+L4 cascade accuracy and delta vs all-L4;
- L4 calls per 100 requests;
- L4 token/cost/latency reduction;
- candidate source and whether L4 statistics came from replay oracle.

## Final Report

Write:

```text
docs/experiments/2026-06-24_clinc150_l2_autoresearch_report.md
```

The report must include:

- why this experiment moved from manual guard repair to L4 AutoResearch;
- artifacts reused and copied from previous runs;
- whether new paid benchmark L4 spend occurred;
- AutoResearch workspace, branch, and commands;
- visible data available to the L4 agent and data intentionally withheld;
- each autonomous hypothesis, implementation, result, and next iteration;
- selected candidate and why it was selected;
- validation, locked test, and stream results;
- OOS/intent-boundary analysis after the final candidate;
- decision: Proceed, Pause and repair AutoResearch harness, or Reject current L2
  shape for CLINC150 Phase 1;
- risks and next step.

## Validation

Run at least:

```bash
uv run pytest tests/targets/nlu/test_l2_target_evolution.py tests/targets/nlu/test_clinc150_phase1.py tests/targets/nlu/test_l4_teacher.py -q
uv run pytest -q
uv run ruff check <touched python files>
git diff --check
```

If optional adapter dependencies are required for the full suite, use the same
extras as prior CLINC150 work and record the exact command.

If new paid benchmark calls happen, validate cost ledger parsing and reconcile
observed spend against detail JSONL artifacts.

## Done Criteria

- Dedicated branch/worktree is used.
- Existing L2 AutoResearch infrastructure is reused or minimally extended for
  CLINC150.
- The L4 agent, not the main session, makes routine target-specific L2 repair
  choices.
- AutoResearch context excludes locked test labels and locked accepted-error
  examples.
- L4 replay-oracle accounting is used for fallback metrics.
- At least one long agent session or three bounded research rounds are attempted
  unless an earlier candidate strongly passes all visible gates.
- Locked test is used only after candidate selection.
- Primary results keep L0/L1/L3 disabled.
- Final report and any cost ledger are written.
- All checks pass or any skipped check is justified.
- Changes are organized into a git commit on the dedicated branch/worktree.
