# Outer Evolution Policy Refactor Plan

Date: 2026-06-24

Purpose: refactor Darjeeling's L1/L2/L3 evolution control so rounds, budget,
timeouts, stop policy, agent-run accounting, and evidence recording are owned by
the outer evolution policy, while target/layer adapters keep all task-specific
training, evaluation, feedback, and adoption logic.

This plan intentionally avoids the misleading term "shared inner lifecycle".
There should not be a new product layer between the outer lifecycle and target
adapters. The target-independent shape is:

```text
outer evolution policy
  -> run/generation/round/budget/agent-control mechanics
  -> layer-specific adapter hooks
  -> target-owned training, evaluation, feedback, and artifact adoption
```

## Required Context Files

Read these before changing code:

- `AGENTS.md`
- `docs/experiments/2026-06-24_evolution_lifecycle_alignment_note.md`
- `docs/experiments/2026-06-24_clinc150_l1_agent_session_evolution_plan.md`
- `docs/experiments/2026-06-24_clinc150_l1_programbank_report.md`
- `docs/experiments/2026-06-24_clinc150_l2_autoresearch_plan.md`
- `docs/experiments/2026-06-24_clinc150_l2_autoresearch_report.md`
- `src/darjeeling/contracts.py`
- `src/darjeeling/cli.py`
- `src/darjeeling/targets/nlu/replay.py`
- `src/darjeeling/targets/nlu/target.py`
- `src/darjeeling/targets/nlu/settings.py`
- `src/darjeeling/targets/nlu/compiler/loop.py`
- `src/darjeeling/targets/nlu/compiler/l1_program_compiler.py`
- `src/darjeeling/targets/nlu/compiler/l2_target_evolution.py`
- `src/darjeeling/targets/nlu/compiler/l3_prompt_optimizer.py`
- `tests/targets/nlu/test_l1_rust_worker.py`
- `tests/targets/nlu/test_l2_target_evolution.py`
- `tests/targets/nlu/test_l3_prompt_optimizer.py`
- `tests/targets/nlu/test_clinc150_phase1.py`

## Execution Isolation

Run this work in a dedicated git branch and worktree. Do not implement it in the
main worktree.

Suggested branch:

```text
codex/outer-evolution-policy
```

Suggested worktree:

```text
../darjeeling-outer-evolution-policy
```

If a suitable branch or worktree already exists, inspect it and continue there
instead of creating a duplicate. Use that worktree for implementation,
verification, docs, and the final commit.

This is a refactor task, not a paid benchmark task. Do not run new paid L4
benchmark calls unless a local smoke or characterization test unexpectedly
requires a missing artifact and no cheaper fake/cached path exists. If any paid
benchmark call is made, keep it minimal and record observed cost.

When complete, create a git commit and report the branch, worktree path, commit
hash, and whether the tracked worktree is clean. Do not delete the worktree or
branch.

## Current State

Darjeeling already has an outer replay and compiler generation path:

- `src/darjeeling/targets/nlu/replay.py` routes requests and triggers compiler
  generations through `compile_every`.
- `src/darjeeling/targets/nlu/compiler/loop.py` builds generation artifacts,
  evaluates candidate artifact sets, decides promotion, and writes manifests.
- `src/darjeeling/contracts.py` contains target-facing protocols such as
  `CompileContext` and `TargetCompiler`.

The control points are currently fragmented:

- L1 has `L4CodingAgentAdapter` and `run_l1_coding_agent_job`. It handles
  workspace packaging, agent launch, scope checking, transcript, diff,
  provenance, and validation, but it is effectively one job rather than a
  first-class policy-controlled evolution loop.
- L2 has the richest controller in `l2_target_evolution.py`: rounds, budget
  profiles, local-search/agent modes, visible/private splits, scope checks,
  candidate snapshots, selection/promotion gates, patience, summaries, and
  evidence policy. Much of this is policy/control plumbing that should not be
  uniquely L2-shaped.
- L3 has `run_l3_prompt_evolution`, which handles an isolated prompt workspace,
  agent-session launch, private holdouts, scope check, replay, selection, and
  adoption, but mostly as one long session.
- `src/darjeeling/targets/nlu/main_cli.py` resolves L2-specific budget profiles
  and flags. L1 and L3 have narrower knobs, so the repo does not yet expose one
  consistent outer policy model across layers.

## Design Boundary

The outer lifecycle may own generic control and accounting:

- max generations, compile cadence, and candidate generation attempts;
- per-layer evolution enablement;
- max rounds or max agent sessions;
- timeout per agent launch or local search step;
- budget profile and explicit budget overrides;
- patience / no-improvement stop policy;
- stop reasons;
- command/transcript/provenance paths;
- workspace scope checking mechanics;
- evidence-strength labeling so dry-runs, fake commands, smoke runs, and real
  agent sessions are not confused;
- common summary fields for requested/resolved policy and consumed budget.

Target/layer adapters must own semantics:

- L1 Rust ProgramBank build, test, benchmark, and candidate evaluation;
- L2 model, guard, target postprocess, local-search objective, OOS diagnostics,
  and adoption gates;
- L3 prompt rendering, local SLM replay, prompt validation, and adoption gates;
- NLU/CLINC150 labels, frames, intents, slots, OOS meaning, strict exact match,
  accepted-error interpretation, and target-specific feedback;
- concrete layer artifact formats.

Core or shared compiler code may carry target-owned payloads only as opaque
JSON/data files. It must not inspect NLU utterances, intents, frames, slots,
request ids, or CLINC150 failure cases.

Keep the refactor low-abstraction. Prefer small dataclasses, ordinary
functions, explicit adapters, and preserved JSON schemas over plugin systems,
dependency injection, schema DSLs, or a broad framework.

## Desired End State

By the end of this refactor:

1. There is an explicit outer evolution policy object or equivalent simple
   structure that resolves common controls such as rounds, max agent launches,
   timeout, patience, and evidence mode.
2. L1, L2, and L3 consume that policy through layer adapter hooks instead of
   each inventing unrelated control fields and summary semantics.
3. L2's existing behavior and artifact/report schemas are preserved unless a
   change is explicitly documented as a compatibility improvement.
4. L1 has a clear path to real multi-round agent evolution controlled by the
   outer policy, even if the first migrated implementation keeps existing tests
   on fake/dry-run commands.
5. L3 uses the same policy vocabulary for max agent sessions, timeout, stop
   reason, and evidence labeling.
6. Existing experiment plans can refer to outer policy knobs instead of
   hard-coding routine round/budget decisions in prose.
7. No NLU or CLINC150 semantics move into Darjeeling core.

## Implementation Plan

### 1. Characterize Before Refactoring

Add or strengthen focused tests that pin current observable behavior before
moving code:

- L2 target evolution summary contains the same important fields for a dry-run
  or fake-command fixture: `rounds_requested`, `rounds_completed`,
  `stop_reason`, `budget_policy`, `evidence_policy`, `agent_budget`,
  `selection_decision`, and `adoption_decision`.
- L1 coding-agent job still writes prompt, contexts, transcript, diff,
  commands, provenance, report, and scope violation records.
- L3 prompt evolution still honors `max_agent_sessions=0`, fake/failing command
  behavior, scope violations, private holdout separation, and summary fields.
- NLU compiler generation still records L1/L2/L3 candidate metrics and promotion
  data in the existing places.

Run these characterization tests before the main migration and again after each
major step.

### 2. Introduce The Outer Policy Vocabulary

Create the smallest useful shared policy structure. The exact file placement is
part of the implementation decision:

- If the module contains only target-independent controls and opaque payloads,
  prefer core/shared placement such as `src/darjeeling/compiler/`.
- If an early step still contains NLU-specific assumptions, keep it under
  `src/darjeeling/targets/nlu/compiler/` and document what must be removed
  before it can move to core.

The policy should cover common controls, not target metrics. Likely fields:

- layer name or candidate kind;
- mode: disabled, dry-run/fake, local-search, codex-cli, agent-session when
  applicable;
- budget profile plus resolved explicit values;
- max rounds;
- max agent launches/sessions;
- per-launch timeout;
- patience / stop-on-no-improvement policy;
- evidence mode or evidence strength;
- optional opaque metadata for target adapters.

Also add simple helpers to render:

- requested policy;
- resolved policy;
- consumed budget/round counts;
- stop reason;
- evidence classification.

Do not force every layer to support every mode. Unsupported modes should fail
with clear errors at adapter boundaries.

### 3. Extract Generic Agent/Workspace Mechanics Only Where It Reduces Duplication

Extract shared helpers only for mechanics that are clearly duplicated and
target-independent:

- command execution result shape;
- JSONL command writer;
- protected workspace snapshot and scope diff, parameterized by writable and
  ignored roots;
- transcript/report/provenance path conventions;
- agent command construction where inputs are generic strings and paths.

Do not extract prompt text, target context writing, candidate evaluation, local
search objectives, or adoption gates unless the extracted code treats payloads
as opaque adapter-owned data.

### 4. Migrate L1 To The Policy Shape

Keep L1 target semantics in `l1_program_compiler.py`, but make it consume the
outer policy vocabulary:

- map existing `Settings.l1_agent_*` fields into the shared policy;
- preserve existing single-job behavior by default;
- add the minimum runner shape needed for future multi-round L1 evolution:
  round result records, stop reason, agent budget counters, and evidence
  strength;
- ensure fake/dry-run/codex-cli/agent-session paths still produce the existing
  artifacts;
- do not directly edit generated ProgramBank artifacts in the repo-level
  workspace.

This step should make future L1 experiments depend on policy parameters rather
than prose instructions like "try three attempts".

### 5. Migrate L3 To The Policy Shape

Update `run_l3_prompt_evolution` and its CLI/config wiring so it uses the same
policy vocabulary for:

- max agent sessions;
- timeout;
- agent budget counters;
- stop reason;
- evidence classification;
- summary/report fields.

Preserve private holdout separation, prompt artifact validation, replay gates,
and adoption behavior.

### 6. Migrate L2 Last And Preserve Compatibility

Treat L2 as the reference implementation whose behavior should not silently
drift.

Move or adapt policy plumbing from `l2_target_evolution.py` and
`main_cli.py` so L2 uses the shared policy vocabulary while keeping L2-specific
logic local:

- visible/private split policy remains L2/NLU-owned;
- local-search objective remains L2-owned;
- OOS and intent diagnostics remain target-owned;
- selection and adoption gates remain L2-owned;
- existing CLINC150 reports and summaries remain readable.

Preserve or explicitly compatibility-map existing CLI flags:

- `--rounds`
- `--budget-profile`
- `--max-agent-rounds`
- `--timeout-s`
- `--inner-patience-rounds`
- local-search flags

If a summary schema changes, write a migration note and tests that prove old
reports or downstream readers still work, or that the new schema is deliberately
versioned.

### 7. Wire Settings And CLI Without Expanding Scope

Expose common policy controls consistently, but avoid a broad new configuration
system.

Acceptable work:

- keep existing environment variables and settings fields working;
- add small shared resolver functions;
- add missing L1/L3 settings only when needed to express the same policy knobs;
- update CLI help text so users can see that rounds/budget/timeout are outer
  policy controls.

Avoid:

- replacing all settings with a new DSL;
- changing benchmark definitions;
- changing teacher prompts;
- changing `TeacherCache` semantics;
- changing CLINC150/L2 quality thresholds as part of this refactor.

### 8. Documentation And Report

Write a concise completion report:

```text
docs/experiments/2026-06-24_outer_evolution_policy_refactor_report.md
```

The report should include:

- what moved into outer policy/shared mechanics;
- what stayed in L1/L2/L3 adapters;
- old vs new control-flow diagram or bullet summary;
- compatibility notes for L2 AutoResearch and existing CLINC150 reports;
- evidence that L1/L2/L3 still work through focused tests;
- any remaining duplication intentionally left in place and why.

Update earlier docs only if they would otherwise send the next agent down the
wrong path. Prefer small edits over broad rewrite.

## Validation

Run focused tests first:

```bash
uv run --extra dev pytest \
  tests/targets/nlu/test_l1_rust_worker.py \
  tests/targets/nlu/test_l2_target_evolution.py \
  tests/targets/nlu/test_l3_prompt_optimizer.py \
  tests/targets/nlu/test_clinc150_phase1.py \
  -q
```

Run full tests with the optional extras that have been needed in recent work:

```bash
uv run --extra dev --extra massive pytest -q
```

Run lint and whitespace checks:

```bash
uv run --extra dev ruff check src tests
git diff --check
```

If the full suite fails because of an optional dependency mismatch, first try
the known full command with extras. If any failure remains, fix it or clearly
document why it is unrelated and what focused tests cover the refactor.

## Done Criteria

This plan is complete when:

- a dedicated branch/worktree contains the implementation;
- L1, L2, and L3 all consume a common outer policy vocabulary for rounds,
  budget/session limits, timeout, stop reason, and evidence classification;
- shared code contains only target-independent mechanics and opaque payloads;
- target/layer semantics remain in target/layer modules;
- L2 target evolution behavior is protected by characterization tests and
  existing summary/report consumers still work;
- L1 has a policy-controlled evolution runner shape suitable for future real
  agent-session evolution;
- L3 uses the same policy vocabulary without losing private holdout separation;
- the completion report is written;
- focused tests, full tests with extras, ruff, and `git diff --check` pass or
  any residual issue is explicitly justified;
- the final changes are committed.

## Escalation Rules

Do not stop for routine implementation choices. Make a conservative choice,
write it down in the report, and verify it.

Escalate only if the work would require:

- moving NLU/CLINC150 semantics into core;
- changing the product goal or benchmark selection;
- changing quality thresholds or promotion semantics rather than preserving
  behavior;
- running paid benchmark calls beyond a small missing-artifact repair;
- deleting or rewriting large experiment artifacts.

Negative discoveries are not blockers by themselves. If a proposed extraction
is messier than expected, keep that piece local, extract the smaller safe part,
record the reason, and continue.
