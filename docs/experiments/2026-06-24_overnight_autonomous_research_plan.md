# Overnight Autonomous Research Sprint Plan

Date: 2026-06-24

This is an overnight, high-autonomy research and development plan for using the
remaining weekly Codex quota before it expires around **2026-06-25 08:00
Asia/Shanghai**.

The goal is not to finish a short checklist. The goal is to keep finding and
testing the next highest-value hypothesis about Darjeeling until the overnight
window is nearly used, while preserving engineering discipline and the
target/core boundary.

## Primary Objective

Improve Darjeeling's ability to externalize L4 capability into cheaper lower
layers without unacceptable quality loss.

Use this scorecard, in priority order:

1. Maintain strict quality:
   - accepted precision target: >= 99%;
   - lower-layer OOS false accept rate target: <= 2%;
   - final cascade accuracy delta vs all-L4 target: >= -0.5 percentage points.
2. Improve useful lower-layer coverage:
   - higher L1/L2 accepted coverage;
   - lower L4 calls per 100 requests;
   - lower replay-oracle cost/latency at the same quality bar.
3. Improve evidence quality:
   - better visible/holdout split pressure;
   - better selection gates;
   - better diagnostics, operating curves, and failure classification.
4. Preserve system generality:
   - target-specific optimization is allowed as target adaptation cost;
   - do not move CLINC150/NLU semantics into Darjeeling core;
   - distinguish reusable system evidence from benchmark-specific lift.

## Time And Quota Utilization Policy

This is a quota-utilization task. Do not stop after the first valid result.

Continue autonomous hypothesis -> experiment -> implementation -> validation
cycles until one of these happens:

- it is close to **2026-06-25 08:00 Asia/Shanghai**;
- the practical Codex quota/time budget is exhausted;
- all viable research tracks in this plan have either produced a useful result,
  been falsified with evidence, or are blocked by external constraints;
- continuing would require a product-goal, risk-tolerance, or paid-spend
  decision from the user.

Minimum expected effort:

- at least 5 substantial research/development cycles;
- explore at least 3 distinct tracks unless one track keeps producing strong,
  measurable gains;
- include at least 1 web/literature research pass;
- include at least 1 implementation/experiment pass;
- include at least 1 reporting/visualization/selection-gate improvement pass;
- do not end merely because tests pass or because one clean commit exists.

A clean commit and passing tests are checkpoints, not the finish line, before
the overnight window is substantially used.

Use stable intermediate commits when a cycle reaches a coherent checkpoint. If a
later cycle fails, keep the earlier useful commit and document the failed
branch of exploration.

Check wall-clock time explicitly during the sprint. Once it is about 45 minutes
before the expiry window, stop starting new risky implementation work and switch
to consolidation: finish the research log, write the final report, run the
appropriate validation commands, commit useful tracked changes, and report any
ignored artifacts that should be preserved.

## Execution Isolation

Use a separate branch/worktree for this night sprint:

```text
branch: codex/overnight-autonomous-research-20260624
worktree: ../darjeeling-overnight-research-20260624
```

If an appropriate worktree already exists, inspect and continue there instead
of creating a duplicate.

This task may run in parallel with the precision/coverage frontier repair task,
which is expected to continue in:

```text
worktree: /Users/chenmohan/gits/darjeeling-precision-coverage-visuals
branch: codex/precision-coverage-visuals
```

During parallel execution, treat the precision/coverage repair worktree as
read-only context. Do not edit files owned by that active task unless the repair
has been merged into the branch you are working on.

Avoid editing these files during parallel execution unless they have already
been merged and the change is clearly coordinated:

```text
src/darjeeling/eval/plots.py
src/darjeeling/targets/nlu/precision_coverage.py
docs/experiments/precision_coverage/
docs/experiments/2026-06-24_precision_coverage_visualization_report.md
docs/experiments/2026-06-24_precision_coverage_frontier_repair_plan.md
```

## Required Context

Read these before choosing the first track:

- `AGENTS.md`
- `docs/design/00_decisions.md`
- `docs/design/README.md`
- `docs/design/modules/l1_rust_programbank.md`
- `docs/design/modules/l2_student.md`
- `docs/design/modules/l4_agent_evolve_harness.md`
- `docs/design/modules/eval_reports.md`
- `docs/experiments/README.md`
- `docs/experiments/2026-06-24_clinc150_l1_agent_session_effect_plan.md`
- `docs/experiments/2026-06-24_clinc150_l1_agent_session_effect_report.md`
- `docs/experiments/2026-06-24_clinc150_l2_autoresearch_report.md`
- `docs/experiments/2026-06-24_precision_coverage_frontier_repair_plan.md`

If the precision/coverage repair worktree exists and is clean enough to read,
also inspect its report and figures as read-only context:

- `/Users/chenmohan/gits/darjeeling-precision-coverage-visuals/docs/experiments/2026-06-24_precision_coverage_visualization_report.md`
- `/Users/chenmohan/gits/darjeeling-precision-coverage-visuals/docs/experiments/precision_coverage/visual_qa.md`

Do not depend on those files being present in your own worktree unless the
precision/coverage branch has been merged.

## Research And Experiment Loop

Every cycle must write a short entry to:

```text
docs/experiments/2026-06-25_overnight_autonomous_research_log.md
```

Each entry should include:

- cycle number and timestamp;
- track name;
- hypothesis;
- expected metric impact;
- action taken;
- evidence collected;
- decision: continue, expand, repair, revert, or abandon;
- next hypothesis.

Default cycle structure:

1. Identify the current bottleneck from artifacts, reports, or fresh experiment
   output.
2. State a falsifiable hypothesis.
3. Choose the smallest implementation, experiment, or research pass that can
   test it.
4. Run the experiment or implement the patch.
5. Validate with focused tests, artifact checks, and metric comparisons.
6. Record the result.
7. If it works, look for the next bottleneck; if it fails, diagnose and try the
   next plausible repair.

Do not ask the user to pick among routine tactics. Escalate only if the next
step changes product goals, scope boundaries, risk tolerance, or paid spend.

## Candidate Tracks

These are priority tracks, not a fixed checklist. Choose dynamically based on
evidence and expected return.

### Track A: L1 Selection And Risk Pressure

Problem: the real L1 agent-session run proved the route works mechanically, but
the selected candidate reached only 96.91% locked-test accepted precision at
14.11% coverage. Visible validation alone was too weak.

Possible hypotheses:

- A zero-or-near-zero train-dev accepted-error gate would have rejected risky
  L1 candidates before locked test.
- Multi-fold train-derived dev slices expose phrase collisions better than a
  single train-dev split.
- Rule-level negative support and OOS/conflict vetoes can identify safer
  target-adapter overlays without modifying L1 artifacts.
- The next L1 agent-session run should feed stronger visible diagnostics and
  disallow candidates with train-dev accepted errors unless coverage is tiny.

Possible work:

- implement stronger L1 visible selection gates;
- add train-derived folds for L1 evaluation;
- add target-side L1 rule-risk overlay analysis, if not already provided by the
  visualization repair branch;
- rerun a small L1 visible-only evaluation over existing candidates;
- run a bounded follow-up L1 agent-session only if the harness and data pressure
  are ready and it will answer a clear hypothesis.

### Track B: L2 AutoResearch Multi-Round Repair

Problem: the first CLINC150 L2 AutoResearch bridge effectively tested only one
round and did not reach locked test. It was useful infrastructure evidence, but
not a strong test of the intended method.

Possible hypotheses:

- The L2 AutoResearch harness needs stronger round continuity, scratch
  candidate handling, and visible multi-fold pressure before a real multi-round
  run is meaningful.
- Agent-invoked Optuna/search can improve the accepted precision/coverage curve
  if the agent is given better visible diagnostics and cannot mutate the active
  config directly.
- Selection should use operating-curve evidence rather than a single threshold
  point.

Possible work:

- repair L2 target-evolution workspace loop issues;
- make scratch candidate search explicit;
- add visible cross-fold/OOS-heavy selection summaries if missing;
- run a multi-round L2 AutoResearch attempt using replay artifacts;
- compare candidate operating curves and selection decisions.

### Track C: Risk-Coverage / Selective Prediction Research

Use web research to import ideas from related work. Search and read enough to
turn ideas into concrete experiments, not just a literature summary.

Recommended topics:

- selective classification;
- classification with rejection option;
- risk-coverage curves;
- conformal prediction for abstention;
- calibration under distribution shift;
- cascaded inference and early exits;
- distillation with abstention or confidence calibration.

Every useful idea should become one of:

- an experiment implemented in this sprint;
- a design note with a concrete go/no-go criterion;
- a small target-side prototype over existing CLINC150 artifacts;
- a future plan only if implementation would exceed the overnight window.

Use primary sources where practical. Record citations/links in the final report.

### Track D: Experiment Evidence And Reporting

Problem: we have repeatedly needed better evidence before deciding whether L1
or L2 is improving.

Possible hypotheses:

- Standard risk/coverage curve data can improve candidate selection before
  locked-test exposure.
- Existing reports should show selected point, visible curve, diagnostic locked
  point, and next bottleneck in one compact table.
- Selection gates should emit machine-readable reason codes that make future
  Agent iterations less likely to stop early.

Possible work:

- improve selection summaries;
- add compact metric tables;
- add failed-hypothesis logs;
- add reusable report snippets without adding a broad framework;
- integrate repaired precision/coverage figures after the parallel repair
  branch is merged, or leave clear integration notes if it is not merged.

### Track E: Core Lifecycle Only If Evidence Justifies It

Core is intentionally thin. Do not expand it because a target-specific
experiment needs a target-specific diagnostic.

Consider core changes only if at least two layers or targets duplicate the same
target-independent lifecycle plumbing, such as:

- append-only round/candidate ledgers;
- path/hash manifest helpers;
- opaque metric row validation;
- generic resume/checkpoint handling;
- non-target-specific report artifact indexing.

Any core change must be small, target-neutral, and backed by tests that do not
mention CLINC150/NLU semantics.

## Data And Cost Policy

Use existing replay artifacts before making paid benchmark calls.

Known useful artifacts:

```text
/Users/chenmohan/gits/darjeeling/runs/clinc150-l2-cascade-20260623/
/Users/chenmohan/gits/darjeeling-clinc150-l1-agent-session-effect/runs/clinc150-l1-agent-session-effect-20260624/main-agent-session-5round/
/Users/chenmohan/gits/darjeeling-clinc150-calibration-repair/runs/clinc150-calibration-repair-20260624/
/Users/chenmohan/gits/darjeeling-clinc150-l2-autoresearch/runs/clinc150-l2-autoresearch-20260624/
```

Fresh worktrees do not contain ignored `runs/` artifacts. Use read-only
absolute paths or copy only the minimal files needed. Validate row counts and
hashes where practical.

Codex quota is the resource to use aggressively. Paid benchmark L4/API calls are
different from Codex session quota:

- use replay artifacts first for debugging, harness repair, and cheap
  iteration;
- do not avoid paid L4 calls when they answer a concrete experimental question,
  unlock a higher-quality validation, or materially reduce uncertainty;
- cap new paid benchmark L4 spend at `$100` for this sprint unless the user has
  separately approved more;
- estimate the intended paid run cost before launching it, then ledger all paid
  calls with observed usage/cost from artifacts;
- do not mix Codex session usage with benchmark serving spend.

## Locked-Test Policy

Do not use locked-test labels/details to design rules, thresholds, or target
adaptations.

Locked test may be used only for:

- already selected candidate confirmation;
- diagnostic plotting of predeclared visible policies;
- post-hoc reporting clearly marked as diagnostic.

If a track would require repeated locked-test tuning, switch to visible
train-derived folds or OOS-heavy visible diagnostics instead.

## Checkpointing

Create stable intermediate commits after coherent milestones. Examples:

- research note + experiment design;
- target-side gate implementation + tests;
- L1/L2 experiment runner repair + focused validation;
- completed experiment report and artifacts.

Keep the final tracked worktree clean. Keep ignored `runs/` artifacts available
for inspection. Do not delete worktrees or branches.

## Final Report

Write:

```text
docs/experiments/2026-06-25_overnight_autonomous_research_report.md
```

The report must include:

- branch, worktree, commit list, and clean/dirty status;
- time window actually used;
- Codex usage if available;
- paid benchmark L4 spend ledger or an explicit `$0` statement;
- research links and what was borrowed from them;
- cycle table with hypotheses, actions, evidence, and decisions;
- metric deltas for any implemented change;
- failed hypotheses and why they failed;
- what was committed vs left as ignored artifacts;
- which changes are target-specific adaptation and which are reusable system
  work;
- next recommended plan.

If useful, also write:

```text
docs/experiments/2026-06-25_overnight_related_work_notes.md
```

## Validation

Validation depends on the tracks touched. Always run:

```bash
git diff --check
uv run ruff check <touched Python files>
```

Run focused tests for touched areas. Examples:

```bash
uv run pytest tests/targets/nlu/test_clinc150_phase1.py -q
uv run pytest tests/targets/nlu/test_l1_coding_agent.py tests/targets/nlu/test_l1_rust_worker.py -q
uv run pytest tests/targets/nlu/test_l2_target_evolution.py -q
uv run pytest tests/test_precision_coverage_plots.py -q
```

Run full pytest when changes are broad or when final time permits. If optional
extras are required, use the repo-documented extra combination and record the
exact command.

## Done Criteria

- Existing active precision/coverage repair worktree was not modified.
- Separate overnight branch/worktree was used.
- At least 5 substantial cycles were recorded, unless the time window was
  genuinely unavailable or a hard external blocker stopped progress.
- At least 3 tracks were explored, unless one track kept producing strong,
  measurable gains and the report justifies focusing there.
- At least 1 web/research pass was converted into an experiment, design note, or
  implemented change.
- At least 1 implementation/experiment pass was completed and validated.
- Passing tests or one clean commit was not treated as sufficient early
  completion.
- Final report and research log were written.
- Useful changes were committed.
- Worktree status and ignored artifacts were reported.
