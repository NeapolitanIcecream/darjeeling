# CLINC150 L1 Agent-Session Evolution Plan

Date: 2026-06-24

Superseded by:
`docs/experiments/2026-06-24_clinc150_l1_agent_session_effect_plan.md`.

Keep this document as historical context for the first L1 agent-session plan.
Use the effect plan for the next execution because it adds the target-dependent
optimization boundary and effect-side failure ladder.

Purpose: rerun the CLINC150 L1 experiment using the real L4 coding-agent
evolution path. The prior CLINC150 L1 ProgramBank run proved useful harness
wiring and replay-oracle accounting, but its candidates were produced through
dry-run patch jobs. That is not the designed L1 evolve method and cannot be
used as strong evidence about whether L4 can autonomously externalize CLINC150
capability into CPU-native L1 Rust code.

This plan keeps the existing L1 technical route: a Rust ProgramBank candidate
edited by an L4 coding agent, then judged by an outer evaluator. The goal is not
to make the Rust artifact elegant. The goal is to let a long-running agent
session build a large, fast, high-precision L1 program from teacher-visible
data, then measure whether L1+L4 replay fallback preserves strict CLINC150
quality while reducing L4 calls.

## Required Context Files

Read these before changing code or running experiments:

- `AGENTS.md`
- `docs/design/00_decisions.md`
- `docs/design/modules/l1_rust_programbank.md`
- `docs/design/modules/l4_layer.md`
- `docs/design/modules/testing.md`
- `docs/experiments/2026-06-23_benchmark_selection_phase_note.md`
- `docs/experiments/2026-06-23_clinc150_phase1_report.md`
- `docs/experiments/2026-06-23_clinc150_teacher_reliability_report.md`
- `docs/experiments/2026-06-23_clinc150_l2_cascade_report.md`
- `docs/experiments/2026-06-24_clinc150_l1_programbank_plan.md`
- `docs/experiments/2026-06-24_clinc150_l1_programbank_report.md`
- `src/darjeeling/targets/nlu/compiler/l1_program_compiler.py`
- `src/darjeeling/targets/nlu/layers/l1_rust_programbank.py`
- `src/darjeeling/targets/nlu/native/l1_empty_programbank/`
- `src/darjeeling/targets/nlu/clinc150_phase1.py`
- `src/darjeeling/targets/nlu/main_cli.py`
- `tests/targets/nlu/test_l1_coding_agent.py`
- `tests/targets/nlu/test_l1_rust_worker.py`
- `tests/targets/nlu/test_clinc150_phase1.py`

## Execution Isolation

Run this work in a dedicated branch and worktree. Do not implement or run the
experiment directly in the main worktree.

Suggested branch:

```text
codex/clinc150-l1-agent-session-evolve
```

Suggested worktree:

```text
../darjeeling-clinc150-l1-agent-session-evolve
```

Use that worktree for implementation, experiments, reports, and the final
commit. If an appropriate worktree already exists, inspect it and continue
there instead of creating a duplicate. Keep ignored `runs/` artifacts available
for inspection; do not assume they will be committed.

## Core Design Boundary

Keep Darjeeling core target-, dataset-, and application-independent. CLINC150
labels, NLU frames, out-of-scope policy, teacher rows, replay-oracle accounting,
and L1 candidate diagnostics belong in the NLU target or experiment artifacts.

Low abstraction tax applies to repository-level code: keep harnesses, CLI
commands, reports, and tests explicit and boring. Do not add a new plugin
system, generic L1 DSL, dependency-injection layer, feature store framework, or
cross-target rule engine for this experiment.

Low abstraction tax does not mean the generated L1 artifact must stay small.
Large hard-coded Rust tables, repetitive branches, scanners, validators,
lookup maps, generated modules, and partially redundant code are acceptable
inside an isolated L1 ProgramBank candidate if they improve precision, coverage,
and latency without breaking the contract.

## Prior Result And Correct Interpretation

The previous L1 report concluded `Pause and repair harness/evolution`:

- validation-selected candidate: 100.00% accepted precision, 60.35% coverage;
- locked test: 92.73% accepted precision, 39.76% coverage;
- lower-layer OOS false accept rate: 7.10%;
- cascade delta vs all-L4: -1.927 percentage points.

The main lesson is not that L1 ProgramBank is invalid. The main issue is that
the candidate was produced through dry-run patch jobs. Dry-run patch mode is a
fixture path for testing workspace packaging, context generation, provenance,
diff capture, and state-machine wiring. It applies a prepared patch and runs
checks. It does not launch a real L4 agent session and does not validate the
intended L1 evolve loop.

For this plan, dry-run may be used only as a smoke test. It cannot support the
quality conclusion. A result only counts as L1 evolve evidence if it launches
`agent-session` and records the agent transcript, commands, candidate diff,
scope check, validation metrics, and outer evaluator decision.

## Agent Input Surface

The L4 coding agent should run inside the isolated L1 workspace. Its default
editable surface is:

```text
workspace/l1_programbank/
```

Its scratch surface is:

```text
workspace/runs/
```

Its protected, read-only context surface is:

```text
workspace/contexts/
workspace/program.md
workspace/workspace_manifest.json
```

The agent should see teacher-visible data, not all benchmark data. It may use:

- train split utterances with existing L4 teacher labels;
- train-derived calibration/dev slices;
- visible validation metrics and accepted-error summaries after candidates are
  evaluated;
- aggregated intent-family summaries;
- OOS-heavy diagnostics derived from train and visible validation only;
- command guides and local evaluation tools.

The agent must not see or query:

- locked test labels;
- locked test L4 detail rows;
- private selection or promotion holdout labels;
- hidden pass/fail details from private gates;
- any artifact that lets it tune directly against the final locked test.

If a searchable local database is useful, build a simple target-local
teacher-visible database under the experiment workspace, such as JSONL plus a
small SQLite index. It may contain train teacher rows, train-derived dev rows,
visible candidate diagnostics, and visible validation accepted-error summaries.
It must not contain locked-test labels or locked-test L4 rows. Keep the database
format simple and document its source files and row counts in the report.

## Reused Benchmark Artifacts

Reuse existing CLINC150 artifacts before making any new paid teacher calls:

- `data/processed/clinc150_data_full/train.jsonl`
- `data/processed/clinc150_data_full/validation.jsonl`
- `data/processed/clinc150_data_full/test.jsonl`
- `runs/clinc150-l2-cascade-20260623/teacher-traces/train-full-stratified/teacher_live_vs_gold.details.jsonl`
- `runs/clinc150-l2-cascade-20260623/teacher-traces/validation-full/teacher_live_vs_gold.details.jsonl`
- `runs/clinc150-l2-cascade-20260623/teacher-traces/test-full/teacher_live_vs_gold.details.jsonl`
- `docs/experiments/2026-06-23_clinc150_l2_cascade_cost_ledger.json`
- `docs/experiments/2026-06-23_clinc150_teacher_reliability_cost_ledger.json`

The normal expected new paid benchmark L4 spend is `$0` because the experiment
should use replay-oracle rows. If required artifacts are missing, first copy
them from the main worktree and validate row counts. Regenerate missing teacher
rows only if copying is impossible. Any new paid benchmark L4 spend must be
ledgered and stay under `$5`.

Cost for running Codex/agent-session itself is separate from benchmark serving
cost and may not be visible in replay detail artifacts. Do not mix the two in
the replay cost ledger. Report benchmark L4 spend from detail rows and note
agent-session execution separately.

## Required Mode

The main experiment must use:

```text
L1_AGENT_MODE=agent-session
```

If the current CLI/harness does not expose enough control to run CLINC150 L1
agent-session evolution cleanly, add the smallest target-local or NLU compiler
entry point needed. Prefer extending the existing `L4CodingAgentAdapter` and
CLINC150 harness rather than writing a parallel system.

Do not use dry-run patches to generate final candidates. Do not hand-edit the
selected L1 candidate in the repository-level workspace. Human-written support
code in Darjeeling harnesses is fine; generated candidate Rust code should be
produced inside isolated L1 agent jobs.

## Decision To Support

Make one of these decisions with evidence:

- **Proceed with existing L1 route**: an agent-session evolved candidate passes
  locked test with accepted precision >= 99%, lower-layer OOS false accept rate
  <= 2%, final L1+L4 cascade accuracy no worse than 0.5 percentage points below
  all-L4, and meaningful L1 coverage. Use >= 10% locked-test coverage as the
  minimum phase target and >= 25% as the stretch target.
- **Pause and repair agent-session harness**: agent-session is the right route,
  but workspace tools, context shaping, visible dev gates, selection policy, or
  autonomous loop control are not yet reliable enough to trust the result.
- **Revisit L1 technical route**: after several real agent-session attempts,
  the route repeatedly cannot produce high-precision accepted coverage above a
  trivial level, or repeatedly violates build/runtime/isolation constraints.

Do not downgrade an under-supported result into a final conclusion. If evidence
is weak, repair the experiment and iterate. Do not reject L1 because the Rust
artifact is large, repetitive, or inelegant.

## Non-Goals

- Do not redesign L1.
- Do not replace Rust ProgramBank with L2, embeddings, a generic rule DSL, or a
  new model.
- Do not optimize the generated Rust candidate for small source size.
- Do not enable L0, L2, or L3 in primary results.
- Do not continue MASSIVE work.
- Do not change the CLINC150 benchmark or strict exact-match target.
- Do not tune against locked test.
- Do not change `TeacherCache` semantics.
- Do not move CLINC150/NLU semantics into Darjeeling core.

## Experiment Root And Outputs

Use:

```text
runs/clinc150-l1-agent-session-evolve-20260624/
```

Write the final report to:

```text
docs/experiments/2026-06-24_clinc150_l1_agent_session_evolution_report.md
```

If new paid benchmark L4 calls occur, write:

```text
docs/experiments/2026-06-24_clinc150_l1_agent_session_evolution_cost_ledger.json
```

## Phase A: Preflight And Harness Repair

Inspect the current L1 harness and verify what is already available after the
previous merge:

- CLINC150 L1 eval CLI;
- CLINC150 L4 replay oracle;
- L1 agent-session workspace creation;
- L1 prompt and constraints;
- provenance, transcript, commands, diff, and scope-check outputs;
- tests for dry-run, codex-cli, and agent-session modes.

Run a small agent-session smoke using a fake or very small controlled job only
to prove command wiring if needed. Label it as smoke. It must not become the
quality candidate.

Repair only the minimum harness gaps needed for a real CLINC150 L1
agent-session experiment. Likely acceptable additions:

- a CLINC150-specific command or script that prepares visible L1 context and
  launches `L4CodingAgentAdapter` in `agent-session`;
- a visible train/dev evaluation command for candidate crates;
- simple teacher-visible JSONL/SQLite lookup files under the experiment root;
- report helpers for agent-session attempts.

## Phase B: Teacher-Visible Data And Tools

Build the agent-visible context from train teacher rows and train-derived
calibration/dev rows. Include row counts and source hashes where practical.

The context should support autonomous exploration:

- intent family summaries;
- high-support phrases and normalized variants;
- OOS risk families;
- candidate accepted-error summaries from visible evaluations;
- commands for cargo test/build and CLINC150 visible eval;
- a clear objective that rewards high accepted precision first, then coverage,
  then latency/cost reduction.

If adding a lookup database, keep it target-local and simple:

- source: train teacher rows, train-derived dev, visible validation diagnostics;
- no locked test labels or locked test L4 rows;
- indexes: request id, intent, token, normalized phrase, OOS flag, maybe error
  family;
- query scripts documented in `commands.md`.

## Phase C: Baselines

Run and record:

- empty/default L1 ProgramBank on validation and locked test;
- all-L4 replay-oracle baseline on validation and locked test;
- previous dry-run selected candidate as historical reference only, not as a
  current candidate unless the agent-session independently recreates or revises
  it.

The baseline should prove that L1 acceptance and fallback accounting are wired
correctly before spending time on long agent sessions.

## Phase D: Real Agent-Session Evolution

Run real agent-session L1 evolution. Each attempt should launch an isolated
workspace, let the agent edit `workspace/l1_programbank/`, and record transcript,
commands, diff, provenance, and validation results.

Each iteration should:

- state the hypothesis being tested;
- launch or continue an agent-session job;
- build and test the Rust crate;
- evaluate on train-derived dev and official validation;
- audit accepted errors and OOS false accepts;
- update visible diagnostics for the next attempt;
- keep or reject the candidate based on visible evidence;
- autonomously repair routine issues instead of waiting for human direction.

The agent should try several strategies if needed:

- high-support phrase and normalized phrase tables;
- intent-specific modules;
- exact and boundary-aware token scanners;
- OOS validators and negative guards;
- conflict-family vetoes;
- conservative abstention when multiple programs match;
- generated tables derived from visible teacher data;
- validation-driven pruning using only visible diagnostics.

Try at least three real agent-session attempts or repair iterations unless an
earlier candidate strongly passes all criteria. If agent-session cannot launch
or cannot make edits, repair the harness first and document the blocker as
`Pause and repair agent-session harness`, not as an L1 route failure.

## Phase E: Selection Without Locked Test

Select candidates without locked-test labels.

A candidate is eligible for locked test only if visible validation shows:

- accepted precision >= 99%;
- lower-layer OOS false accept rate <= 2%;
- final L1+L4 cascade delta vs all-L4 >= -0.5 percentage points;
- L1 coverage >= 10%, or a clear report argument for why slightly lower
  coverage is still meaningful;
- no build failures, worker timeouts, or unacceptable latency regressions;
- clean scope check;
- no runtime network calls or hidden data access.

Prefer the eligible candidate with the highest validation coverage among those
with adequate safety margin and clean OOS behavior.

## Phase F: Locked Test And Confirmation

After selecting one candidate, run locked test once. Do not use locked-test
accepted errors to design new rules. If the locked-test failure is narrow and
the repair is justified entirely by visible pre-test evidence, one bounded
repair and second locked-test exposure is allowed; record it explicitly.

Confirm on:

- locked test sequential;
- validation sequential;
- validation uniform;
- validation zipf-heavy;
- OOS-heavy visible diagnostic slice.

Report:

- all-L4 replay-oracle accuracy;
- L1-only accepted precision;
- L1 coverage;
- L1 wrong accepts;
- lower-layer OOS false accept rate;
- final L1+L4 cascade accuracy and delta vs all-L4;
- L4 calls per 100 requests;
- L4 token/cost/latency reduction;
- L1 native p50/p95 latency and throughput;
- source size and binary size as diagnostics only.

## Final Report

Write:

```text
docs/experiments/2026-06-24_clinc150_l1_agent_session_evolution_report.md
```

The report must include:

- explicit statement that the main evidence used `agent-session`, not dry-run;
- branch, worktree, commit, and run root;
- artifacts reused or copied from prior runs;
- row counts for copied data and teacher details;
- whether any new paid benchmark L4 spend occurred;
- benchmark L4 replay-oracle accounting semantics;
- agent input surface and any teacher-visible database contents;
- harness changes;
- prompt/constraint changes;
- each agent-session attempt, hypothesis, commands, result, and decision;
- selected candidate and why it was selected;
- validation, locked-test, and stream confirmation metrics;
- accepted-error and OOS false-accept audit;
- final decision: Proceed, Pause and repair agent-session harness, or Revisit
  L1 route;
- next step.

## Validation

Run at least:

```bash
cargo test --manifest-path <selected_l1_crate>/Cargo.toml
uv run pytest tests/targets/nlu/test_l1_coding_agent.py tests/targets/nlu/test_l1_rust_worker.py tests/targets/nlu/test_clinc150_phase1.py tests/targets/nlu/test_l4_teacher.py -q
uv run pytest -q
uv run ruff check <touched python files>
git diff --check
```

If new paid benchmark L4 calls happen, validate the cost ledger and reconcile
observed spend against detail JSONL sums.

## Done Criteria

- Dedicated branch/worktree is used.
- Main evidence comes from real `agent-session` L1 evolution.
- Dry-run is used only for smoke/testing, if at all.
- Agent-visible data is teacher-visible only; locked test labels/details are
  protected from the agent.
- L1 candidate code is generated inside isolated L1 agent workspaces.
- At least three real agent-session attempts or repair iterations are tried,
  unless an earlier candidate strongly passes all criteria.
- Locked test is used only after candidate selection.
- Primary results keep L0/L2/L3 disabled.
- Final report and any cost ledger are written.
- Tests, lint, and `git diff --check` pass or skipped checks are justified.
- Changes are organized into a git commit on the dedicated branch/worktree.
