# CLINC150 L1 Agent-Session Effect Plan

Date: 2026-06-24

This plan supersedes
`docs/experiments/2026-06-24_clinc150_l1_agent_session_evolution_plan.md`.

The goal is to run the CLINC150 L1 experiment with the real L4 coding-agent
evolution path, then improve L1 effectiveness through target-owned feedback and
generated L1 code. The plan is intentionally not a core refactor plan.

## Decision To Support

Produce one of these decisions with evidence:

- **Proceed with the current L1 route**: a real agent-session evolved Rust
  ProgramBank passes locked test with accepted precision >= 99%, lower-layer OOS
  false accept rate <= 2%, final L1+L4 cascade accuracy no worse than 0.5
  percentage points below all-L4, and meaningful L1 coverage. Use >= 10% locked
  test coverage as the minimum phase target and >= 25% as the stretch target.
- **Continue target-side L1 adaptation**: real agent-session evolution works and
  shows improving visible evidence, but the candidate does not yet pass locked
  test. The report must identify the dominant effect failure mode and the next
  concrete target-side repair.
- **Repair L1 harness**: agent-session, workspace isolation, candidate build,
  visible evaluation, replay-oracle fallback accounting, or reporting is not
  reliable enough to support a quality conclusion.
- **Revisit the L1 artifact route**: after a real multi-round agent-session run
  with adequate target-side feedback and no harness blocker, Rust ProgramBank
  still cannot produce non-trivial safe accepted coverage.

Do not declare L1 invalid from dry-run patch evidence. Do not declare success
from visible validation alone.

## System Boundary

Darjeeling core should not be the main editing surface for this work. Core
already provides the outer round vocabulary:

```text
EvolutionRunPolicy(max_rounds, round_timeout_s, patience_rounds, round_executor)
EvolutionRoundResult
EvolutionRunSummary
```

For this plan, treat that as sufficient unless a concrete harness bug blocks
the experiment. Do not add OOS, intent, phrase, conflict-family, rule-evidence,
or CLINC150 semantics to core.

Target-dependent optimization is allowed in:

- the NLU target package;
- CLINC150 experiment harnesses;
- target-local tools under the experiment/workspace;
- generated isolated L1 Rust ProgramBank candidates.

This target-dependent work is an adaptation cost and benchmark upper-bound
exploration. The final report must separate reusable Darjeeling mechanism
evidence from CLINC150-specific lift.

## Required Context

Read these first:

- `AGENTS.md`
- `docs/design/00_decisions.md`
- `docs/design/README.md`
- `docs/design/modules/l1_rust_programbank.md`
- `docs/design/modules/l4_agent_evolve_harness.md`
- `docs/experiments/README.md`
- `docs/experiments/2026-06-23_clinc150_phase1_report.md`
- `docs/experiments/2026-06-23_clinc150_teacher_reliability_report.md`
- `docs/experiments/2026-06-24_clinc150_l1_programbank_report.md`
- `docs/experiments/2026-06-24_clinc150_l1_agent_session_evolution_plan.md`
- `src/darjeeling/compiler/evolution_policy.py`
- `src/darjeeling/targets/nlu/compiler/l1_program_compiler.py`
- `src/darjeeling/targets/nlu/layers/l1_rust_programbank.py`
- `src/darjeeling/targets/nlu/clinc150_phase1.py`
- `src/darjeeling/targets/nlu/main_cli.py`
- `tests/targets/nlu/test_l1_coding_agent.py`
- `tests/targets/nlu/test_l1_rust_worker.py`
- `tests/targets/nlu/test_clinc150_phase1.py`

## Execution Isolation

Run in a dedicated branch and worktree:

```text
branch: codex/clinc150-l1-agent-session-effect
worktree: ../darjeeling-clinc150-l1-agent-session-effect
```

If an appropriate worktree already exists, inspect and continue there instead
of creating a duplicate. Keep ignored `runs/` artifacts for inspection. Do not
delete the worktree or branch when finished.

## Benchmark Artifacts And Cost

Reuse existing CLINC150 data and replay-oracle rows before making paid calls:

- `data/processed/clinc150_data_full/train.jsonl`
- `data/processed/clinc150_data_full/validation.jsonl`
- `data/processed/clinc150_data_full/test.jsonl`
- `runs/clinc150-l2-cascade-20260623/teacher-traces/train-full-stratified/teacher_live_vs_gold.details.jsonl`
- `runs/clinc150-l2-cascade-20260623/teacher-traces/validation-full/teacher_live_vs_gold.details.jsonl`
- `runs/clinc150-l2-cascade-20260623/teacher-traces/test-full/teacher_live_vs_gold.details.jsonl`
- `docs/experiments/2026-06-23_clinc150_l2_cascade_cost_ledger.json`
- `docs/experiments/2026-06-23_clinc150_teacher_reliability_cost_ledger.json`

Fresh git worktrees may not contain ignored `runs/` or processed data. Copy the
minimum required artifacts from the main worktree or use read-only absolute
paths, then validate row counts and hashes where practical.

Expected new paid benchmark L4 spend is `$0` because this experiment should use
L4 replay-oracle rows. If required replay artifacts are missing and cannot be
copied, regenerate only the missing rows, ledger the observed spend, and keep
new paid benchmark L4 spend under `$5`.

Do not mix Codex/agent-session cost with benchmark serving cost. Report them
separately if both are visible.

## Primary Runtime Configuration

The main result must use:

```text
L1_AGENT_MODE=agent-session
L0 disabled
L2 disabled
L3 disabled
L4 replay-oracle fallback enabled
```

Use the shared outer round policy rather than manually simulating rounds. Start
with at least five real L1 agent-session rounds if the harness supports it.
Stopping earlier is acceptable only if a candidate strongly passes all visible
criteria or a real blocker is documented. If evidence is still improving and
cost/time are reasonable, extend the run rather than stopping after a single
near miss.

Dry-run may be used only for smoke tests. Dry-run candidates cannot support the
quality conclusion.

## Agent Input Surface

The L4 coding agent should edit only the isolated candidate surface:

```text
workspace/l1_programbank/
```

Scratch output belongs under:

```text
workspace/runs/
```

Protected context belongs under:

```text
workspace/contexts/
workspace/program.md
workspace/workspace_manifest.json
```

The agent may see teacher-visible data and diagnostics:

- train split utterances with L4 teacher labels;
- train-derived dev/calibration splits;
- visible validation aggregate metrics;
- visible validation accepted-error summaries after candidate evaluation;
- visible OOS-heavy diagnostics;
- visible intent/conflict-family summaries;
- rule-level positive/negative support summaries derived from visible data;
- local commands for build, test, visible eval, and diagnostics.

The agent must not see:

- locked test labels;
- locked test L4 detail rows;
- private selection or promotion holdout labels;
- private pass/fail details;
- any direct artifact that lets it tune against final locked test.

## Phase A: Preflight

Verify that the current L1 harness can run real `agent-session` rounds:

- worktree and branch isolation;
- candidate workspace creation;
- protected context and scope checks;
- transcript, command log, diff, provenance, and round summary output;
- candidate Rust build and tests;
- visible CLINC150 evaluation;
- L4 replay-oracle fallback accounting;
- `max_rounds` actually produces multiple real rounds.

Repair only blocking harness issues. Keep repair code target-local unless it is
pure target-independent lifecycle plumbing.

## Phase B: Baselines

Run and record:

- empty/default L1 ProgramBank on validation and locked test;
- all-L4 replay-oracle baseline on validation and locked test;
- previous dry-run selected candidate as historical reference only.

The previous dry-run candidate may inform diagnostics, but it must not become
the selected candidate unless agent-session independently recreates or improves
it using allowed visible evidence.

## Phase C: Visible Feedback System

Build target-owned visible feedback artifacts that help the agent improve
effectiveness without seeing locked test data.

Required feedback:

- accepted-error audit grouped by rule, intent, and OOS/in-scope status where
  possible;
- OOS-heavy visible slice;
- intent-conflict summary from visible data;
- phrase/token support summary with positive and negative support counts;
- per-slice validation summary, not only aggregate validation accuracy;
- latency and source-size diagnostics.

Recommended simple implementation:

- JSONL files plus a small Python script or SQLite index under the experiment
  root;
- deterministic generation from allowed source files;
- row counts and source paths recorded in the report;
- no new generic framework.

## Phase D: Effect-Side Improvement Ladder

Each real agent-session round should state the hypothesis it is testing, then
use the resulting diagnostics to choose the next repair. Do not treat a global
threshold change as the main repair unless diagnostics show that a uniform
threshold is the actual failure.

Use this ladder:

| Failure pattern | Expected repair direction |
| --- | --- |
| Accepted errors come from low-support positive phrases | Add rule-level positive/negative support checks, remove weak cues, require boundary-aware or multi-token evidence. |
| OOS false accepts dominate | Add OOS-first guards, high-risk phrase vetoes, negative support tables, and conservative abstain rules. |
| In-scope intent families are confused | Add conflict-family vetoes, disambiguating required cues, and abstain when multiple intent programs match. |
| Precision is safe but coverage is too low | Expand only high-support, low-negative-support rule families; avoid globally relaxing guard thresholds. |
| Coverage is high but precision collapses across slices | Prune unstable rule families and require slice stability before selection. |
| Validation is strong but visible OOS-heavy or zipf-heavy slice is weak | Treat the candidate as not eligible for locked test; repair the weak slice using visible diagnostics. |
| Agent writes brittle substring-only code | Generate structured tables, token scanners, normalization, rule provenance, and support metadata in the Rust candidate. |
| Candidate is slow or brittle at runtime | Keep the same semantics but simplify the generated tables/scanners and add native latency checks. |

Acceptable target-specific L1 artifact techniques include:

- large hard-coded Rust tables;
- normalized phrase maps;
- token and boundary-aware scanners;
- tries or perfect-hash maps;
- per-intent modules;
- required-token and forbidden-token combinations;
- OOS risk phrase tables;
- conflict veto tables;
- conservative abstain paths;
- generated rule provenance for debugging.

The generated Rust does not need to be elegant or small. It must be fast,
deterministic, isolated, buildable, and evaluated by replay.

## Phase E: Multi-Round Agent-Session Evolution

Run real L1 agent-session evolution. For each round:

1. Build the candidate from the previous accepted workspace or baseline.
2. Give the agent the current visible feedback package.
3. Let the agent edit candidate code and run allowed local tools.
4. Build and test the Rust crate.
5. Evaluate on train-derived dev and visible validation slices.
6. Audit accepted errors, OOS false accepts, conflict failures, coverage, and
   latency.
7. Keep, reject, or continue from the candidate based on visible evidence.
8. Record the hypothesis, commands, result, diff, and next decision.

Run at least five real rounds unless an earlier candidate clearly passes all
visible criteria. If the first real round fails because the agent cannot launch,
edit, build, or evaluate, classify the result as harness repair and fix it
before drawing any L1 quality conclusion.

## Phase F: Candidate Selection Without Locked Test

A candidate is eligible for locked test only if visible evidence shows:

- accepted precision >= 99%;
- lower-layer OOS false accept rate <= 2%;
- L1+L4 cascade delta vs all-L4 >= -0.5 percentage points;
- L1 coverage >= 10%, with >= 25% as stretch;
- no major weak visible slice, especially OOS-heavy or intent-conflict slices;
- no build failure, timeout, runtime network call, or scope violation;
- candidate changes are inside the allowed workspace;
- the candidate was produced by real agent-session, not dry-run.

Prefer the eligible candidate with the best combination of safety margin,
slice stability, and coverage. Do not select a candidate solely because it has
the highest aggregate validation coverage.

## Phase G: Locked Test And Confirmation

Run locked test only after visible candidate selection.

Do not use locked-test accepted-error details to design new rules. If locked
test narrowly fails and the repair is justified entirely by visible pre-test
evidence, one bounded second locked-test exposure is allowed. Record the
exposure, rationale, and exact information used.

Confirm:

- locked test sequential;
- validation sequential;
- validation uniform;
- validation zipf-heavy;
- visible OOS-heavy diagnostic slice;
- visible intent-conflict diagnostic slice;
- native L1 p50/p95 latency and throughput;
- final L4 replay-oracle fallback call rate.

## Reporting

Write:

```text
docs/experiments/2026-06-24_clinc150_l1_agent_session_effect_report.md
```

If new paid benchmark L4 calls occur, write:

```text
docs/experiments/2026-06-24_clinc150_l1_agent_session_effect_cost_ledger.json
```

The report must include:

- branch, worktree, commit hash, and clean/dirty status;
- reused/copied artifacts, row counts, and any hashes;
- paid benchmark L4 spend, expected `$0` unless regeneration was required;
- statement that main evidence used `agent-session`;
- harness changes;
- target-specific adaptation changes;
- explicit separation between reusable system evidence and CLINC150-specific
  lift;
- each round's hypothesis, feedback, candidate diff summary, metrics, failure
  classification, and next action;
- accepted-error, OOS false-accept, and intent-conflict audit;
- selected candidate and why it was eligible;
- locked-test result and any second exposure rationale;
- final decision from this plan's decision set;
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

If optional extras are required for full pytest, run the repo's documented full
test command with those extras and record the exact command.

## Done Criteria

- Dedicated branch/worktree was used.
- The main result used real `agent-session` L1 evolution.
- At least five real rounds were attempted, unless an earlier candidate clearly
  passed all visible criteria or a harness blocker was repaired and documented.
- L0/L2/L3 were disabled in the primary result.
- L4 replay-oracle fallback accounting was used.
- Target-dependent improvements stayed in target/harness/workspace/generated
  artifact code, not Darjeeling core.
- Locked test was used only after candidate selection.
- The final report distinguishes reusable system evidence from target-specific
  adaptation.
- Tests, lint, and diff checks passed, or any skipped check is justified.
- Changes are organized into a git commit on the dedicated branch/worktree.
