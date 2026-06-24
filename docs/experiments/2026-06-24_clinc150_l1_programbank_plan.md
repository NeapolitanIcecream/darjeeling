# CLINC150 L1 ProgramBank Experiment Plan

Date: 2026-06-24

Purpose: validate the existing Darjeeling L1 technical route on the CLINC150
benchmark. This is not a redesign of L1. The experiment should use the current
Rust ProgramBank plus L4 coding-agent evolution path, then measure whether a
CPU-native L1 can absorb a meaningful share of CLINC150 requests while preserving
strict correctness through L4 fallback.

## Required Context Files

Read these before changing code:

- `AGENTS.md`
- `docs/experiments/2026-06-23_benchmark_selection_phase_note.md`
- `docs/experiments/2026-06-23_clinc150_phase1_experiment_plan.md`
- `docs/experiments/2026-06-23_clinc150_phase1_report.md`
- `docs/experiments/2026-06-23_clinc150_teacher_reliability_report.md`
- `docs/experiments/2026-06-23_clinc150_l2_cascade_experiment_plan.md`
- `docs/experiments/2026-06-23_clinc150_l2_cascade_report.md`
- `docs/experiments/2026-06-24_clinc150_calibration_repair_plan.md`
- `src/darjeeling/targets/nlu/compiler/l1_program_compiler.py`
- `src/darjeeling/targets/nlu/layers/l1_rust_programbank.py`
- `src/darjeeling/targets/nlu/native/l1_empty_programbank/`
- `src/darjeeling/targets/nlu/clinc150_phase1.py`
- `src/darjeeling/targets/nlu/main_cli.py`
- `tests/targets/nlu/test_l1_rust_worker.py`
- `tests/targets/nlu/test_clinc150_phase1.py`

## Execution Isolation

Run this plan in a dedicated git branch and worktree. Do not implement the
experiment directly in the main worktree.

Suggested branch:

```text
codex/clinc150-l1-programbank
```

Suggested worktree:

```text
../darjeeling-clinc150-l1-programbank
```

If a suitable branch or worktree already exists, inspect it and continue there
instead of creating a duplicate. Use the dedicated worktree as the workspace root
for implementation, experiments, validation, reports, and the final commit.

The existing `runs/` artifacts are ignored by git and will not automatically
appear in a fresh worktree. Treat `/Users/chenmohan/gits/darjeeling` as the
read-only artifact source for previous runs, or copy the required artifacts into
the dedicated worktree before running commands. Validate copied artifact row
counts and paths in the report.

## Current L1 Design Assumption

Use the existing L1 design:

- L1 is a Rust ProgramBank worker.
- The worker receives one request at a time and returns either abstain, a full
  CLINC150/NLU frame, or a precise patch.
- L4 coding-agent evolution edits the isolated Rust ProgramBank workspace.
- Outer replay/evaluation decides whether a candidate is good enough; candidate
  code does not self-certify.

Large hard-coded native logic is allowed and expected. The ProgramBank may grow
large, repetitive, table-heavy, or partially redundant. It may use direct
`if`/`else`, `match`, generated lookup tables, finite-state scanners, string
normalizers, validators, and intent-specific modules. Do not optimize for small
source size or elegance inside the generated L1 artifact. Optimize for:

- accepted correctness;
- accepted coverage;
- fast CPU execution;
- benchmark isolation;
- enough local organization that future evolution can continue.

Low abstraction tax still applies to Darjeeling repo-level harnesses, prompts,
adapters, and contracts. Do not introduce a new L1 DSL, framework, interpreter,
plugin system, or generic rule engine for this experiment.

## Reused Benchmark Artifacts

Reuse existing CLINC150 L4/teacher artifacts before making new paid calls:

- `runs/clinc150-l2-cascade-20260623/teacher-traces/train-full-stratified/teacher_live_vs_gold.details.jsonl`
- `runs/clinc150-l2-cascade-20260623/teacher-traces/validation-full/teacher_live_vs_gold.details.jsonl`
- `runs/clinc150-l2-cascade-20260623/teacher-traces/test-full/teacher_live_vs_gold.details.jsonl`
- `docs/experiments/2026-06-23_clinc150_l2_cascade_cost_ledger.json`
- `docs/experiments/2026-06-23_clinc150_teacher_reliability_cost_ledger.json`

If these artifacts are unavailable from the dedicated worktree, copy them from
the main worktree or regenerate only the missing rows using the resume-safe
CLINC150 teacher path. New paid L4 benchmark spend should be zero in the normal
case and must stay under `$5` unless the report records a stronger reason.

## L4 Replay Oracle Requirement

Use or implement a target-local CLINC150 L4 replay oracle for this experiment.
This is experiment accounting, not production cache behavior.

The replay oracle should:

- load paid L4/teacher detail rows by request id;
- validate request coverage before evaluation;
- expose the recorded L4 prediction/output;
- expose recorded L4 statistics such as model, tokens, cost, latency, attempts,
  parse/schema failure, and retry diagnostics when present;
- let L1+L4 fallback metrics count fallback rows as L4 calls using recorded L4
  cost/latency, even though no live L4 call is made during this experiment.

Keep CLINC150/NLU interpretation in the target. Do not move CLINC150 labels,
intents, OOS rules, or NLU frame semantics into Darjeeling core. Do not change
`TeacherCache` semantics.

If the L2 calibration-repair branch already has a suitable helper and it is easy
to reuse or port, use it. Otherwise implement the smallest target-local helper
needed for L1 and leave broader consolidation for a later merge.

## Decision To Support

Make one of these decisions with evidence:

- **Proceed with existing L1 route**: validation-selected L1 ProgramBank passes
  locked test with accepted precision >= 99%, lower-layer OOS false accept rate
  <= 2%, final L1+L4 cascade accuracy no worse than 0.5 percentage points below
  all-L4, and meaningful L1 coverage. Use >= 10% locked-test coverage as the
  minimum phase target and >= 25% as the stretch target.
- **Pause and repair harness/evolution**: the route looks plausible but the
  harness, prompt, replay-oracle accounting, or L1 agent loop is not reliable
  enough to trust the result.
- **Revisit L1 technical route**: after several autonomous ProgramBank evolution
  attempts, the existing route cannot produce high-precision accepted coverage
  above a trivial level, or repeatedly violates build/runtime/isolation
  constraints. Do not use large hard-coded code as a reason to reject the route.

Do not declare success from a tiny hand-picked pocket unless the report explains
why that coverage is meaningful for this phase. Do not declare failure until at
least three distinct L1 ProgramBank evolution attempts or repair iterations have
been tried, unless a hard blocker makes further work invalid.

## Non-Goals

- Do not redesign L1.
- Do not replace Rust ProgramBank with an L2 model, embedding retrieval system,
  generic rule DSL, or new framework.
- Do not use locked test to design rules, select pockets, tune validators, or
  choose candidate code.
- Do not enable L0, L2, or L3 in primary results.
- Do not continue MASSIVE work.
- Do not change the CLINC150 benchmark, teacher prompt, or strict exact-match
  target.
- Do not hide strict metrics behind looser semantic diagnostics.
- Do not add CLINC150 semantics to Darjeeling core.

## Primary Experiment Shape

Primary results must be L1-focused:

```text
L1 shadow
L1 + L4 replay-oracle fallback
```

Keep L0 disabled or bypassed. Keep L2 and L3 disabled. Optional appendices may
show how a candidate would interact with other layers, but those results must not
support the main decision.

## Experiment Root

Use:

```text
runs/clinc150-l1-programbank-20260624/
```

Write the final report to:

```text
docs/experiments/2026-06-24_clinc150_l1_programbank_report.md
```

If any new paid L4 benchmark calls occur, write or update:

```text
docs/experiments/2026-06-24_clinc150_l1_programbank_cost_ledger.json
```

If no new paid L4 calls occur, state that explicitly and point to the reused
previous ledgers. Keep benchmark serving cost separate from any coding-agent
operation cost that is not visible in replay artifacts.

## Phase A: Harness And Prompt Alignment

Add the smallest CLINC150 L1 evaluation harness needed to run the experiment.
Prefer target-local functions in `clinc150_phase1.py` and static CLI commands in
`main_cli.py`.

The harness should support:

- building a selected Rust ProgramBank crate;
- running it over CLINC150 train-derived dev, validation, and locked test splits;
- recording each request's L1 accepted/abstained result, frame/patch, reason,
  program path, native latency, and correctness against gold;
- evaluating L1+L4 replay-oracle fallback without live L4 calls;
- writing JSON summaries and accepted-error JSONL files;
- keeping locked test unavailable to selection/evolution steps.

Update the L1 coding-agent prompt and constraints if needed so they explicitly
allow large hard-coded Rust ProgramBank artifacts. Remove wording that may imply
the state machines, tables, or programs must stay small. The prompt should still
require abstain on uncertainty and forbid changes to outer evaluator,
promotion, teacher cache, benchmark labels, or Python orchestration.

Focused tests should cover:

- L1 evaluation can distinguish accept, abstain, wrong accept, and OOS false
  accept;
- L1+L4 replay-oracle fallback counts fallback rows as L4 calls with recorded
  L4 cost/latency;
- L0/L2/L3 are absent from primary CLINC150 L1 metrics;
- locked test data is not used by candidate selection helpers;
- the L1 agent prompt permits large hard-coded native logic while preserving
  benchmark isolation constraints.

## Phase B: Visible Training And Dev Context

Build visible L1 evolution context from CLINC150 train data and reused train L4
details. Do not include validation or locked test labels in generated L1
contexts.

The context should help the L1 agent find high-precision pockets:

- intent families and support counts;
- common lexical patterns;
- OOS-heavy examples and near-OOS risks;
- current L1 misses and wrong accepts from prior candidates;
- accepted-error examples from validation only after a candidate has been run,
  never locked-test examples for rule design.

Keep this context simple JSON/JSONL/Markdown. Do not build a new general
feature store.

## Phase C: Baselines

Run and record:

- empty/default L1 ProgramBank over validation and locked test, expected to
  abstain on all rows;
- all-L4 replay-oracle baseline on validation and locked test;
- optional simple hand-written smoke fixture only to verify harness behavior, not
  as the final candidate.

The baseline should prove that L1 acceptance and fallback accounting are wired
correctly before running long evolution loops.

## Phase D: Autonomous ProgramBank Evolution

Run autonomous L1 ProgramBank evolution against visible train/dev context and
official validation. Use the existing L4 coding-agent path where practical; do
not hand-edit final generated L1 artifacts in the repo-level workspace.

Each iteration should:

- state a hypothesis about which pockets can be safely absorbed;
- run or update an L1 agent job that edits an isolated Rust ProgramBank
  workspace;
- build and test the Rust crate;
- evaluate on train-derived dev and validation;
- audit accepted errors, OOS false accepts, conflicts, and slow paths;
- keep or reject the candidate based on validation evidence;
- feed safe findings into the next iteration.

Candidate strategies may include:

- high-support intent-specific modules;
- exact and normalized phrase tables;
- prefix/suffix/pattern scanners;
- OOS validators;
- negative guards that force abstain on risky phrases;
- generated lookup tables derived from visible train examples;
- module-level conflict detection where multiple programs match.

Large tables and repetitive code are acceptable. Dead or redundant code inside a
candidate is acceptable if it does not harm correctness, speed, build reliability,
or future evolution.

## Phase E: Selection Policy

Select candidates without using locked test.

A candidate is eligible for locked test only if validation shows:

- accepted precision >= 99%;
- lower-layer OOS false accept rate <= 2%;
- final L1+L4 cascade delta vs all-L4 >= -0.5 percentage points;
- L1 coverage >= 10%, or a clear report argument for why slightly lower coverage
  is still meaningful in this phase;
- no build failures, worker timeouts, or unacceptable latency regressions;
- implementation remains CPU-native Rust with no runtime network calls or hidden
  data access.

If multiple candidates pass, choose the one with the best validation coverage
among candidates with adequate precision margin and clean OOS behavior.

## Phase F: Locked Test And Stream Confirmation

After selecting a candidate, run locked test once. If it fails narrowly and the
reason is clear from pre-test evidence, one bounded repair iteration is allowed,
but record it as a second test exposure.

Confirm on:

- locked test sequential;
- validation uniform;
- validation zipf-heavy;
- OOS-heavy diagnostic slice if available.

For each view, report:

- all-L4 replay-oracle accuracy;
- L1-only accepted precision;
- L1 coverage;
- L1 wrong accepts;
- lower-layer OOS false accept rate;
- final L1+L4 cascade accuracy and delta vs all-L4;
- L4 calls per 100 requests;
- L4 token/cost/latency reduction;
- L1 native p50/p95 latency and throughput;
- source size and binary size as diagnostics, not success gates.

## Final Report

Write:

```text
docs/experiments/2026-06-24_clinc150_l1_programbank_report.md
```

The report must include:

- current L1 design summary;
- artifacts reused and copied from prior runs;
- whether any new paid L4 benchmark spend occurred;
- L4 replay-oracle accounting semantics;
- harness changes;
- prompt/constraint changes made for L1 ProgramBank evolution;
- each autonomous evolution attempt and its result;
- selected candidate and why it was selected;
- validation, locked test, and stream results;
- accepted-error and OOS false-accept audit;
- decision: Proceed, Pause and repair, or Revisit L1 route;
- risks and next step.

## Validation

Run at least:

```bash
cargo test --manifest-path <selected_l1_crate>/Cargo.toml
uv run pytest tests/targets/nlu/test_l1_rust_worker.py tests/targets/nlu/test_clinc150_phase1.py tests/targets/nlu/test_l4_teacher.py -q
uv run pytest -q
uv run ruff check <touched python files>
git diff --check
```

If new paid L4 calls happen, validate cost ledger parsing and reconcile observed
spend against detail JSONL artifacts.

## Done Criteria

- Dedicated branch/worktree is used.
- CLINC150 L1 harness exists and is tested.
- L4 replay-oracle fallback accounting is explicit and tested for L1.
- L1 agent prompt/constraints no longer discourage large native ProgramBank
  artifacts.
- At least three L1 ProgramBank evolution or repair iterations are attempted,
  unless an earlier candidate strongly passes all criteria.
- Locked test is used only after candidate selection.
- Primary results keep L0/L2/L3 disabled.
- Final report and any cost ledger are written.
- All checks pass or any skipped check is justified.
- Changes are organized into a git commit on the dedicated branch/worktree.
