# Daytime Autonomous Research Sprint Plan

Date: 2026-06-25

This plan is a second attempt at a long-running autonomous research sprint after
the 2026-06-24 overnight plan ended after about 1.5 hours. The purpose is to
test whether we can give an agent a time-bounded research objective that keeps
working until the planned stop time while still preserving engineering quality,
cost accounting, and the target/core boundary.

The execution boundary is time, not a checklist.

## Execution Window

Run until **2026-06-25 17:00 Asia/Shanghai**.

Use **2026-06-25 16:30 Asia/Shanghai** as the soft stop:

- do not start a new risky implementation or long-running experiment after the
  soft stop;
- finish the current bounded run if it is already near completion;
- otherwise switch to consolidation: write the report, run validation, commit,
  and checkpoint remaining work.

Use **2026-06-25 17:00 Asia/Shanghai** as the hard stop:

- stop active research work;
- write the best available checkpoint even if the current hypothesis is
  incomplete;
- preserve ignored artifacts and report where they are;
- leave the tracked worktree clean if possible.

Do not write the final report, mark the task complete, or stop because a useful
patch exists before the soft stop. A passing test run, a clean commit, a failed
hypothesis, a timeout, or a rejected candidate is a checkpoint, not a stop
condition.

## Primary Scorecard

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
   - lower serving latency/cost at the same quality bar.
3. Improve evidence quality:
   - clearer visible/holdout split pressure;
   - better candidate selection gates;
   - better operating curves, diagnostics, and failure classification.
4. Preserve system generality:
   - target-specific optimization is allowed as target adaptation cost;
   - do not move CLINC150/NLU semantics into Darjeeling core;
   - distinguish reusable system evidence from benchmark-specific lift.

Support work such as harness repair, timeout cleanup, or report cleanup is
useful only when it enables the scorecard above. It cannot by itself satisfy
this sprint.

## Cost And Usage Accounting

Use separate ledgers. Do not mix these categories.

### API Spend

`api_spend_usd` is the only dollar-denominated experiment cost in this plan.

It includes:

- live teacher calls;
- L4 benchmark serving calls;
- OpenAI API calls made by benchmark/evaluation code;
- any other serving call with token usage and a price table.

The API budget cap is **$100**. Estimate the intended API cost before launching
a paid run, then record observed usage/cost from artifacts. Spending $0 is
acceptable only when replay artifacts are enough to answer the current
experimental question; do not avoid paid validation if it materially reduces
uncertainty about a promising candidate.

### L4 Agent-Session Usage

Darjeeling may launch L4 AutoResearch/evolve agent sessions as part of L1/L2
experiments. These are not counted as `api_spend_usd` when they run through
Codex subscription/quota rather than a priced API ledger.

Record them separately:

- layer and track;
- command or harness entry point;
- model if visible;
- start/end time or elapsed time;
- timeout setting;
- rounds requested/completed;
- stop reason;
- output artifact path.

This ledger is part of the research evidence because it measures how much L4
research effort was used to externalize capability, but it is not the $100 API
budget.

### Outer Executor Usage

The agent executing this plan is the outer research executor. Its own Codex
subscription/quota usage is not experiment cost and must not be folded into the
API budget or the L4 agent-session ledger. It may be reported as wall-clock time
only.

### Local Compute Notes

Record large local CPU runs, timeouts, and large ignored artifacts when they
matter for reproducibility.

Write the final usage ledger to:

```text
docs/experiments/2026-06-25_daytime_autonomous_research_usage_ledger.json
```

## Execution Isolation

Use a separate branch/worktree:

```text
branch: codex/daytime-autonomous-research-20260625
worktree: ../darjeeling-daytime-research-20260625
```

If the worktree already exists, inspect and continue there instead of creating a
duplicate.

The previous overnight branch is useful input, not proof that this plan is
done:

```text
branch: codex/overnight-autonomous-research-20260624
worktree: /Users/chenmohan/gits/darjeeling-overnight-research-20260624
commit: 61e6542dcb95c7fa16ec265939cfcaf422cbb621
```

Read its report and log. You may merge, cherry-pick, or reimplement useful
changes after verification, but that must be treated as setup or a checkpoint,
not as completion of this sprint.

Known preflight issues to account for:

- the overnight branch produced useful target-local L1/L2 gate and timeout
  fixes, but it stopped too early and did not materially improve L1/L2 effect;
- full pytest may expose a target/core boundary false positive caused by
  `pandas.DataFrame` in the merged plotting facility; repair the test/code
  wording if this appears, then continue the research sprint;
- optional parquet tests may require the repo's documented extras such as
  `--extra massive`.

## Required Context

Read these before choosing the first research direction:

- `AGENTS.md`
- `docs/design/00_decisions.md`
- `docs/design/README.md`
- `docs/design/modules/l1_rust_programbank.md`
- `docs/design/modules/l2_student.md`
- `docs/design/modules/l4_agent_evolve_harness.md`
- `docs/design/modules/eval_reports.md`
- `docs/experiments/README.md`
- `docs/experiments/2026-06-24_clinc150_l1_agent_session_effect_report.md`
- `docs/experiments/2026-06-24_clinc150_l2_autoresearch_report.md`
- `docs/experiments/2026-06-24_precision_coverage_visualization_report.md`
- `docs/experiments/precision_coverage/visual_qa.md`
- `/Users/chenmohan/gits/darjeeling-overnight-research-20260624/docs/experiments/2026-06-25_overnight_autonomous_research_report.md`
- `/Users/chenmohan/gits/darjeeling-overnight-research-20260624/docs/experiments/2026-06-25_overnight_autonomous_research_log.md`

If the overnight worktree is unavailable, continue from the reports and
artifacts available in the main repository.

## Operating Loop

This is not a checklist. Work continuously until the soft stop, choosing the
highest expected-value item from the backlog at each point.

Maintain a research log at:

```text
docs/experiments/2026-06-25_daytime_autonomous_research_log.md
```

Each cycle entry must include:

- timestamp and elapsed time;
- track;
- hypothesis;
- expected scorecard impact;
- action taken;
- evidence collected;
- decision: continue, expand, repair, revert, switch direction, or checkpoint;
- next hypothesis.

Use 60-90 minute checkpoints to decide whether to continue the current line,
scale it up, reduce it after timeout, or switch directions. These checkpoints
are pacing tools only; they are not permission to finish before 16:30.

Default cycle:

1. Identify the current bottleneck from artifacts, plots, reports, or fresh
   experiment output.
2. State a falsifiable hypothesis tied to the scorecard.
3. Choose the smallest implementation, experiment, paid validation, or research
   pass that can test it.
4. Run it.
5. Validate with metrics, focused tests, artifact checks, and standard
   precision/coverage plots when relevant.
6. Record the result.
7. If it works, expand or move to the next bottleneck. If it fails, diagnose,
   reduce scope, repair, or switch directions. Do not stop.

## Flexible Time Shape

Use this as a rhythm, not as a topic schedule:

- First 30-45 minutes: establish baseline, cost/usage ledger, current branch
  state, available artifacts, known preflight repairs, and the first highest
  value hypothesis.
- Middle period until the soft stop: run autonomous hypothesis -> experiment ->
  validation -> report cycles from the backlog below. Keep going even after
  useful commits.
- Last 30 minutes: consolidate, finish the report, update usage ledgers, run the
  best validation set that fits, commit, and record next steps.

If a promising experiment needs another 20-30 minutes near the soft stop, finish
that bounded run only if the final report and checkpoint can still be completed
by the hard stop.

## Effect Obligations

This sprint must try to improve L1 or L2 effect, not only block unsafe
candidates.

Before the soft stop, complete at least:

- one real L1 or L2 effect-improvement attempt that tries to move accepted
  coverage or the precision/coverage operating curve, not just tighten gates;
- one actual L4 agent-session/evolve/AutoResearch run after any required harness
  repair, unless the existing harness is demonstrably unable to launch;
- one standard precision/coverage output update or diagnostic comparison for
  the best relevant candidates;
- one explicit decision about whether API paid validation is useful for the best
  candidate or uncertainty found so far.

If no candidate improves, the final report must show which frontier or gate
prevented progress and what next experiment would attack that limit.

## Research Backlog

Choose dynamically. Do not treat this list as ordered or exhaustive.

### L1 Effect Improvement

Known issue: the real 5-round L1 agent-session run reached high visible
validation precision but failed locked test. The next work should try to improve
generalization and coverage under visible-only pressure.

Possible hypotheses:

- Stronger visible feedback, train-derived folds, OOS-heavy slices, and
  conflict-veto diagnostics can guide L1 evolution toward safer coverage.
- Target-adapter overlays can filter risky accepts without entering the L1
  artifact or distracting the L1 evolve agent.
- A larger or more explicit L1 ProgramBank may improve coverage if selection
  pressure rejects train-dev accepted errors.

Possible actions:

- run a real L1 agent-session evolution with improved visible diagnostics;
- implement or reuse target-side overlays over recorded L1 accepts, then plot
  the operating curve;
- add train-derived folds or risk summaries that the L1 harness can use without
  leaking locked-test labels;
- compare best L1 candidates against the 2026-06-24 frontier.

### L2 Effect Improvement

Known issue: L2 fixed-inner and first AutoResearch candidates reached good
visible metrics but did not transfer cleanly to selection-like pressure.

Possible hypotheses:

- Visible cross-audit, train-audit, and scratch config search can make multi-
  round AutoResearch produce safer candidates.
- Smaller bounded searches with resume/checkpoint are more effective than a
  single heavy local search that times out.
- Operating-curve selection can identify a safer threshold/guard point than a
  single validation threshold.

Possible actions:

- incorporate or repair the overnight branch's scratch-search, cross-audit, and
  timeout fixes if they are not already present;
- add explicit timeout/trial budget controls for agent-facing local search;
- run a multi-round L2 AutoResearch attempt with bounded searches and replay
  artifacts;
- rerun after timeout with reduced trial count or narrower search instead of
  stopping;
- plot visible, cross-audit, private diagnostic, and operating-curve evidence
  for each serious candidate.

### Paid API / Live Validation

Use the $100 API budget when it answers a concrete uncertainty.

Possible uses:

- live teacher validation for a promising candidate whose replay evidence is
  stale or incomplete;
- paid L4 spot checks on difficult/high-risk examples to decide whether replay
  labels or teacher artifacts are blocking progress;
- a small live benchmark shard if it can distinguish two candidate policies.

Do not spend API budget merely to burn money. Do not refuse to spend API budget
when a bounded paid run would materially improve the decision.

### Related Work To Experiment

Network research is allowed, but every useful idea must become one of:

- a concrete experiment;
- a design note with go/no-go criteria;
- an implementation patch with validation;
- a future plan only when implementation would exceed today's window.

Suggested topics:

- risk-coverage curves;
- selective classification;
- classification with rejection;
- conformal prediction for abstention;
- calibration under distribution shift;
- cascaded inference and early exits;
- teacher-student distillation with rejection or confidence calibration.

### Evidence And Reporting

Use the merged precision/coverage facility as the standard reporting surface.

Possible actions:

- regenerate or extend `round_metrics.jsonl`, `operating_points.jsonl`, and
  figures for new candidates;
- add compact tables that show selected point, visible curve, diagnostic
  holdout point, and next bottleneck;
- make gate reason codes machine-readable where they help future agents
  continue instead of stopping early.

### Core Vs Target Boundary

Target-specific CLINC150/NLU optimization is allowed and may be large or
hard-coded when it is part of target adaptation or benchmark upper-bound
exploration.

Darjeeling core changes are allowed only when they are target-independent
lifecycle/reporting/plumbing improvements. Core must not learn CLINC150 labels,
OOS conventions, intents, utterance patterns, phrase rules, or target failure
cases.

Generated L1/L2 artifacts may be ugly, large, or hard-coded. Repo-level harness
and core code should remain low-abstraction-tax.

## Locked-Test And Private Holdout Policy

Do not use locked-test labels/details to design rules, thresholds, or target
adaptations.

Private or locked diagnostics may be used for:

- already selected candidate confirmation;
- predeclared diagnostic plots;
- post-hoc failure reporting.

When a diagnostic fails, move the next design pressure back to visible folds,
OOS-heavy visible slices, train audit, or cross-audit. Do not tune directly on
locked-test rows.

## Checkpointing

Use intermediate commits for coherent milestones:

- baseline/preflight repair;
- harness repair with tests;
- L1/L2 experiment run and report;
- operating-curve/reporting update;
- final report and ledger.

Do not treat an intermediate commit as completion. Keep ignored `runs/`
artifacts available. Report large artifact paths instead of deleting them.

## Final Report

Write:

```text
docs/experiments/2026-06-25_daytime_autonomous_research_report.md
```

The report must include:

- exact execution time window and whether the soft/hard stop policy was
  followed;
- branch, worktree, commit list, and clean/dirty status;
- `api_spend_usd` ledger summary and path;
- L4 agent-session usage ledger summary and path;
- outer executor wall-clock time, reported separately from experiment cost;
- cycle table with hypotheses, actions, evidence, and decisions;
- metric deltas and precision/coverage figures for any serious L1/L2
  candidate;
- paid validation decision and rationale;
- failed hypotheses and what they ruled out;
- what was committed vs left as ignored artifacts;
- which changes are target-specific adaptation and which are reusable system
  work;
- next recommended plan.

## Validation

Always run:

```bash
git diff --check
uv run --extra dev ruff check <touched Python files>
```

Run focused tests for touched areas. Common examples:

```bash
uv run --extra dev pytest tests/targets/nlu/test_clinc150_phase1.py -q
uv run --extra dev pytest tests/targets/nlu/test_l1_coding_agent.py tests/targets/nlu/test_l1_rust_worker.py -q
uv run --extra dev pytest tests/targets/nlu/test_l2_target_evolution.py -q
uv run --extra dev pytest tests/test_precision_coverage_plots.py -q
```

When broad behavior changes, run the fullest suite that fits the time budget.
If optional CLINC150/parquet dependencies are needed, use:

```bash
uv run --extra dev --extra massive pytest -q
```

Record exact failures. Do not hide known unrelated failures; fix them when they
are cheap and blocking, otherwise document them.

## Completion Criteria

Completion is the 17:00 hard stop plus a committed checkpoint, not a checklist.

At hard stop, the work should have:

- research log;
- final report;
- usage ledger;
- useful tracked changes committed;
- relevant validation results recorded;
- ignored artifacts preserved and listed;
- clear next steps.

If the final candidate does not improve L1/L2 metrics, the sprint is still
useful only if it records a concrete frontier, bottleneck, or failed mechanism
that narrows the next experiment.
