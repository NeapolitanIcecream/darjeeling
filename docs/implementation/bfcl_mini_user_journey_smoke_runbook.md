# BFCL Mini User-Journey Smoke Runbook

Date: 2026-06-30

This runbook is for a short user-perspective trial after the reboot
implementation and agent guidance/permissions work. The goal is to discover
what blocks a first-time user from using Darjeeling on a tiny BFCL-shaped task.

This is not the BFCL Stage 1 experiment. It should collect product, workflow,
documentation, CLI, configuration, logging, paid-reference, agent-permission,
and report issues before the larger pilot work starts.

## Objective

Run a very small BFCL-derived journey as if you were a new Darjeeling user:

1. Start from the public README and tracked design/user docs.
2. Create or run a tiny BFCL target.
3. Launch a compile with L1/L2 enabled, L3 disabled, and L4 fallback available.
4. Let one target adaptation agent session submit at least one candidate.
5. Let Core write validation feedback.
6. Close the attempt and run final test evaluation if the journey reaches that
   point.
7. Record every place where the user journey is unclear, blocked, fragile, or
   too developer-oriented.

The primary output is a friction report and issue backlog, not a benchmark
score.

## Starting Context

Read these files first:

- `AGENTS.md`
- `README.md`
- `docs/implementation/bfcl_experiment_pre_report.md`
- `docs/implementation/bfcl_stage1_runbook.md`
- `docs/implementation/reboot/interactive_compile_loop_runbook.md`
- `docs/implementation/reboot/agent_guidance_and_permissions_runbook.md`
- `docs/design/reboot/00_overall_design.md`
- `docs/design/reboot/modules/03_agent_workspace.md`
- `docs/design/reboot/modules/05_candidate_evaluation.md`
- `docs/design/reboot/modules/06_release_runtime.md`
- `docs/design/reboot/modules/11_compile_orchestration.md`

If any of the BFCL implementation docs are local-only in the main checkout, copy
the needed documents into the worktree or use the absolute source paths. Record
what was copied.

## Worktree And Branch

Run this in a separate branch and worktree.

Suggested setup:

```bash
cd /Users/chenmohan/gits/darjeeling
git status --short
git worktree add ../darjeeling-bfcl-mini-user-journey -b codex/bfcl-mini-user-journey main
cd ../darjeeling-bfcl-mini-user-journey
uv sync --extra dev
```

Do not delete the worktree at the end. Keep ignored run artifacts available for
inspection.

## Smoke Scenario

Use the smallest BFCL-shaped task that can exercise Darjeeling's intended user
workflow.

Recommended sample:

- `BFCL_v3_simple`: 10-20 rows.
- `BFCL_v3_irrelevance`: 10-20 rows.
- Total: 20-40 rows.

Split:

- train: 50%.
- validation: 25%.
- test: 25%.

Runtime shape:

- L1 enabled.
- L2 enabled.
- L3 disabled.
- L4 fallback available, preferably through the real reference/cache path on the
  tiny sample.

Cost:

- Paid L4/reference calls are allowed for this smoke run.
- Hard actual API cap: `$10`.
- Keep the sample small enough that live L4 can run within the cap.
- Record every paid call or completed cached response in a cost ledger with
  model, usage, latency, status, and cost.
- If credentials are missing or the provider path fails, record the issue and
  fall back to a local deterministic reference only to keep discovering later
  user-journey issues.

The smoke run may use a simplified BFCL adapter. It does not need to implement
the official evaluator or all BFCL categories.

## User-Perspective Method

Act like a technically competent first-time user, not like the original system
designer.

For each step, record:

- the command or file the user is expected to find;
- what you tried;
- what happened;
- what you expected;
- whether the next step was obvious;
- whether the error message or log was enough to recover.

Start by following existing README/CLI/docs. If you must inspect internals to
make progress, log that as a user-journey issue.

## Fix Policy

This run should collect issues first.

Allowed during the smoke run:

- very small unblock fixes when the journey cannot continue;
- missing command help text if needed to make the next step discoverable;
- missing logs or report fields needed to understand the smoke result;
- tiny tests for any unblock fix.

Not allowed:

- broad architecture refactors;
- BFCL Stage 1 implementation;
- target-specific optimization work for coverage;
- new framework layers;
- large CLI redesigns;
- changes that hide the original friction without recording it first.

When you make an unblock fix, record the issue first, then record the fix and
the command that proved the journey could continue.

## Required Artifacts

Create one ignored run root:

```text
runs/bfcl-mini-user-journey-<YYYYMMDD-HHMMSS>/
  manifest.json
  command_log.md
  friction_log.md
  issue_backlog.jsonl
  user_journey_report.md
  copied_docs/
  mini_data/
  target/
  workspaces/
  logs/
  reports/
```

`command_log.md` should contain commands, important outputs, and timestamps.

`friction_log.md` should be chronological and written while working.

`issue_backlog.jsonl` should use one JSON object per issue:

```json
{
  "id": "BFCL-MINI-001",
  "severity": "P0|P1|P2|P3",
  "area": "docs|cli|config|target|compile|agent|evaluation|logging|reporting|cost|other",
  "summary": "...",
  "reproduction": "...",
  "expected": "...",
  "actual": "...",
  "impact": "...",
  "fixed_in_this_run": false,
  "fix_commit_or_file": null
}
```

Severity:

- P0: cannot start or continue the journey.
- P1: user can continue only by reading internals or guessing.
- P2: confusing, brittle, or missing observability.
- P3: polish.

## Suggested Flow

### 1. Baseline user setup

Try the documented setup path:

- install dependencies;
- run tests or demo as documented;
- run any relevant CLI help;
- record whether a new user can discover how to start a compile.

If there is no compile CLI, record that. You may still continue through the
Python API or a small smoke script, but mark the CLI gap clearly.

### 2. Build the tiny BFCL-shaped input

Create the minimal target-local data files and schemas needed for:

- one simple function-call case;
- one irrelevance/no-call case;
- train/validation/test split boundaries.

Keep this harness tiny. Prefer plain Python files and JSON/JSONL.

### 3. Build or stub L4 fallback

Prefer the real L4/reference cache path for this tiny sample. Build the cache
once, then reuse it for validation/test replay.

Use the local deterministic reference only when live L4 cannot run because of
credentials, provider failures, or time. Treat that as a user-journey issue, not
as the ideal path.

Record whether the system makes the distinction clear between:

- experiment actual cost;
- production counterfactual L4 fallback cost.

For this smoke run, exact production cost numbers are less important than
whether the report shape is understandable, but actual API spend must not be
under-counted.

### 4. Launch compile with agent guidance

Use the post-PR #2 path:

- pass an `AgentSearchGuidance` that encourages a simple baseline, then local
  search;
- set `AgentWorkspacePermissions(network_access=True, dependency_install=True)`
  unless the platform cannot support target-adaptation agent execution;
- disable L3 in routing/settings for this scenario.

Record the generated `AGENT_BRIEF.md`.

Check whether the brief is understandable to a new user and to the target
adaptation agent:

- Does it say what to optimize?
- Does it say where to write candidates?
- Does it say how to read validation feedback?
- Does it say what tools/permissions are available?
- Does it avoid exposing validation/test rows?

### 5. Let one candidate loop run

Try to get at least one candidate submitted through:

```text
submissions/<candidate>/artifacts/l1|l2
submissions/<candidate>/READY
```

Then verify:

- Core notices the candidate;
- Core runs validation;
- Core writes `journal/feedback-<candidate>.json`;
- the feedback is aggregate and agent-safe;
- the agent can continue or stop intentionally.

If this cannot run end to end, record the blocker and stop after collecting the
available artifacts.

### 6. Close and report

If the compile attempt reaches a closeable state, run final test evaluation and
produce a smoke report.

The report should say:

- whether the journey reached final test;
- candidate count;
- feedback count;
- final stop reason;
- actual paid API spend and whether it stayed below `$10`;
- whether L1/L2/L3/L4 routing behaved as intended;
- whether any raw validation/test material was exposed;
- top user-facing blockers;
- recommended repair order.

## Done Criteria

The run is complete when one of these is true:

- the tiny BFCL journey reaches final test evaluation and writes a final smoke
  report using the real L4/reference path or a documented fallback after the
  real path fails; or
- a P0 blocker prevents progress and the run has enough logs/artifacts to fix
  it confidently.

In both cases:

- `friction_log.md` exists and is chronological;
- `issue_backlog.jsonl` has structured issues;
- a cost ledger exists, even if spend is `$0`;
- `user_journey_report.md` summarizes the journey and top repairs;
- all unblock fixes are small and recorded;
- focused tests are added for any code changes;
- full tests, ruff, and `git diff --check` are run when code changed.

## Final Report Format

Write a concise tracked report under:

```text
docs/experiments/<YYYY-MM-DD>_bfcl_mini_user_journey_report.md
```

It should include:

- run root path;
- commit hash;
- commands run;
- whether the smoke journey completed;
- paid API spend and cost ledger path;
- top P0/P1 issues;
- fixes made in this run;
- remaining issues;
- recommendation: repair now, proceed to Stage 1, or rerun smoke after repairs.

## Completion

At the end:

- commit tracked changes unless the main session says not to;
- leave ignored run artifacts in place;
- report branch, worktree, commit hash, run root, and worktree cleanliness.
