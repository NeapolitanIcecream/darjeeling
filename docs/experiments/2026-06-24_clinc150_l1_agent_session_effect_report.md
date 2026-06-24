# CLINC150 L1 Agent-Session Effect Report

Date: 2026-06-24

Decision: **Continue target-side L1 adaptation**.

The real `agent-session` L1 route works mechanically: Codex CLI launched in the
isolated L1 workspace, edited only `workspace/l1_programbank/`, produced
transcripts/diffs/provenance, built Rust crates, and the outer CLINC150
evaluator measured visible slices plus locked test with L4 replay-oracle
fallback. The best visible-safe candidate did not pass locked test: locked
accepted precision was 96.91% at 14.11% coverage, OOS false accept rate was
1.60%, and L1+L4 cascade accuracy was 0.291 percentage points below all-L4.

This is not evidence to reject Rust ProgramBank. It is evidence that the current
visible selection pressure is still too weak for CLINC150 phrase-rule
generalization.

## Scope

Branch/worktree:

```text
branch: codex/clinc150-l1-agent-session-effect
worktree: /Users/chenmohan/gits/darjeeling-clinc150-l1-agent-session-effect
base commit at report write: 143038d282ff
final experiment commit: recorded in the final handoff for this session
```

Primary run root:

```text
runs/clinc150-l1-agent-session-effect-20260624/main-agent-session-5round/
```

Smoke runs:

```text
runs/clinc150-l1-agent-session-effect-20260624/smoke-dry-run/
runs/clinc150-l1-agent-session-effect-20260624/smoke-agent-session/
```

Primary result used `L1_AGENT_MODE=agent-session` behavior through the
target-local CLI command:

```bash
uv run python -m darjeeling.targets.nlu.main_cli clinc150 l1-agent-session-effect \
  --mode agent-session --rounds 5 --timeout-s 900 \
  --out-dir runs/clinc150-l1-agent-session-effect-20260624/main-agent-session-5round
```

Primary measurements kept L0, L2, and L3 disabled. L4 fallback was replay-oracle
accounting over existing teacher detail rows.

## Reused Artifacts And Cost

No new paid benchmark L4 calls were made. New benchmark L4 spend was `$0.00`.
Codex/agent-session cost is separate from benchmark serving cost and is not
included in replay-oracle cost accounting.

| Artifact | Rows | SHA256 |
| --- | ---: | --- |
| `data/processed/clinc150_data_full/train.jsonl` | 15,100 | `fb488bc05ae0983d210ab1069fa7dcbae325dbbf07339d0b12846f8dff6d2887` |
| `data/processed/clinc150_data_full/validation.jsonl` | 3,100 | `0a8a7a3db696ed19faf88b531e5a0c1ebe12b6b8953ac1ab66b22028d37de430` |
| `data/processed/clinc150_data_full/test.jsonl` | 5,500 | `4287033a2bea1192ed0452edbfe8e7d7bb3f8d72bc9be6256bb95f9857525148` |
| `runs/clinc150-l2-cascade-20260623/teacher-traces/train-full-stratified/teacher_live_vs_gold.details.jsonl` | 15,100 | `242ec13214096bb106b389a54fed699a8ec646a25d37afbb250a54d9915be839` |
| `runs/clinc150-l2-cascade-20260623/teacher-traces/validation-full/teacher_live_vs_gold.details.jsonl` | 3,100 | `17cbbdc40ccb80faaeb7b12ead65266018c71f2f9d5c46557d02b99efe35ecdf` |
| `runs/clinc150-l2-cascade-20260623/teacher-traces/test-full/teacher_live_vs_gold.details.jsonl` | 5,500 | `da19fc889f8337218047128af9a1e1ca023066679273ab10de0d57a2b8aef10d` |

## Harness Changes

Target/core boundary was preserved. No CLINC150, OOS, intent, slot, or phrase
semantics were added to Darjeeling core.

Implemented:

- `codex exec --skip-git-repo-check` for L1 `agent-session` workspaces, because
  round workspaces are intentionally isolated and not git repositories.
- Opaque extra context payloads for the NLU L1 coding-agent harness, so target
  packages can place target-owned visible feedback under `workspace/contexts/`.
- `edge-mvp-nlu clinc150 l1-agent-session-effect`, a target-owned runner that
  builds visible CLINC150 feedback, launches real L1 agent sessions, evaluates
  candidates on train-dev and visible validation slices, selects without locked
  test, then runs locked confirmation.
- Focused tests for extra context payloads, `--skip-git-repo-check`,
  teacher-visible L1 traces, phrase support/negative support, and sanitized
  visible accepted-error feedback.

## Visible Feedback

Agent-visible context included:

- train teacher rows only;
- train-derived dev split ids;
- official validation aggregate metrics and sanitized accepted-error summaries;
- visible OOS-heavy and intent-conflict diagnostic slice summaries;
- train-derived phrase support with positive and negative support counts;
- CLINC150 command guide for cargo test, visible validation eval, and train
  visible eval.

The context intentionally avoided `gold_*` field names and did not include
locked-test labels, locked-test teacher details, private pass/fail signals, or
locked-test accepted-error details.

## Baselines

| View | All-L4 replay acc. | L1 accepted | L1 coverage | L1 precision |
| --- | ---: | ---: | ---: | ---: |
| Empty L1 validation | 98.323% | 0 | 0.000% | n/a |
| Empty L1 locked test | 96.018% | 0 | 0.000% | n/a |

The empty/default ProgramBank abstained on all rows, confirming fallback
accounting degenerates to all-L4 replay-oracle accounting.

## Agent-Session Rounds

All five main rounds were real `agent-session` runs. Each round copied the
latest candidate into an isolated workspace, launched Codex CLI, ran cargo
validation, then outer CLINC150 evals. Agent reports and transcripts are under:

```text
runs/clinc150-l1-agent-session-effect-20260624/main-agent-session-5round/l1-agent-jobs/
```

| Round | Visible validation precision / coverage | Train-dev precision / coverage | Train-dev wrong accepts | Eligible | Native p95 |
| ---: | ---: | ---: | ---: | --- | ---: |
| 1 | 100.00% / 14.97% | 99.78% / 18.08% | 3 | yes | 83 us |
| 2 | 100.00% / 16.87% | 99.54% / 20.17% | 7 | yes | 109 us |
| 3 | 100.00% / 22.13% | 99.43% / 25.72% | 11 | yes | 177 us |
| 4 | 100.00% / 36.42% | 99.23% / 34.46% | 20 | yes | 269 us |
| 5 | 100.00% / 44.06% | 98.69% / 38.43% | 38 | no | 315 us |

Rounds 2-5 increased validation coverage but steadily increased full
train-dev accepted errors. Selection therefore used visible precision floor and
OOS ceiling before validation coverage, and selected round 1.

Selected candidate:

```text
runs/clinc150-l1-agent-session-effect-20260624/main-agent-session-5round/l1-agent-jobs/round-001/workspace/l1_programbank
```

Selected candidate diagnostics:

- source: 330 lines across `src/lib.rs` and `src/frame.rs`;
- source directory size: 20 KiB;
- debug worker binary: 1,028 KiB;
- `cargo test`: 3 tests passed;
- validation sequential: 100.00% precision, 14.97% coverage;
- validation uniform: 100.00% precision, 15.97% coverage;
- validation zipf-heavy: 100.00% precision, 7.00% coverage;
- visible OOS-heavy: 0 accepts, 0 OOS false accepts;
- visible intent-conflict: 100.00% precision, 11.00% coverage.

## Locked Test

Locked test was used only after visible candidate selection in each real
agent-session run. There were two total locked-test exposures in this session:

1. a one-round `smoke-agent-session` run unexpectedly reached visible selection
   and ran locked confirmation;
2. the primary 5-round run performed the planned final locked confirmation.

No locked-test accepted-error details were used to design subsequent rules. The
primary run was started fresh and used only visible train-dev/validation feedback
for its rounds.

Primary selected candidate locked result:

| Metric | Result |
| --- | ---: |
| Accepted precision | 96.91% |
| Coverage | 14.11% |
| Correct accepts | 752 |
| Wrong accepts | 24 |
| OOS false accepts | 16 |
| OOS false accept rate | 1.60% |
| Cascade accuracy | 95.727% |
| All-L4 replay accuracy | 96.018% |
| Cascade delta vs all-L4 | -0.291 pp |
| L4 calls / 100 requests | 85.89 |
| L4 call reduction | 14.11% |
| L4 cost reduction | 13.44% |
| Native p50 / p95 | 73 us / 87 us |

The candidate passes the OOS false-accept and cascade-delta gates but fails the
>=99% accepted-precision gate. The failure is not narrow enough to justify a
third locked exposure.

## Interpretation

Reusable system evidence:

- The real L1 coding-agent harness can run isolated multi-round
  `agent-session` evolution.
- Scope checks, transcripts, diffs, provenance, cargo validation, and L1 worker
  evaluation are reliable enough to support an experiment conclusion.
- L4 replay-oracle fallback accounting works with L0/L2/L3 disabled.
- Rust ProgramBank candidates are fast enough for the intended native L1 role.

CLINC150-specific lift and failure:

- The agent can externalize a visible-safe CLINC150 subset into Rust phrase
  rules and reach up to 44.06% visible validation coverage with 100% visible
  validation precision.
- Coverage growth came from target-specific phrase/rule expansion in generated
  Rust, not from a reusable core mechanism.
- Full train-dev accepted errors rose monotonically with coverage, and even the
  safest selected round did not generalize to locked test.
- The dominant failure mode is still hidden phrase collision / insufficient
  visible negative support, not Rust latency, worker contract, replay fallback,
  or workspace isolation.

## Decision

**Continue target-side L1 adaptation.**

The next repair should be target-side and evidence-driven:

- reject candidates with any full train-dev accepted errors unless there is
  strong cross-slice evidence;
- expose multiple train-derived visible folds rather than a single train-dev
  split;
- build rule-level negative support and conflict-family vetoes into selection,
  not only into agent-visible prose;
- require stability across validation sequential, uniform, zipf-heavy,
  OOS-heavy, and intent-conflict slices before locked test;
- continue keeping CLINC150/NLU semantics in target code, feedback artifacts,
  and generated L1 ProgramBank code, not in Darjeeling core.

## Validation

Commands run before the final full validation:

```bash
uv run ruff check src/darjeeling/targets/nlu/clinc150_phase1.py src/darjeeling/targets/nlu/compiler/l1_program_compiler.py src/darjeeling/targets/nlu/main_cli.py tests/targets/nlu/test_clinc150_phase1.py tests/targets/nlu/test_l1_coding_agent.py
uv run pytest tests/targets/nlu/test_l1_coding_agent.py tests/targets/nlu/test_clinc150_phase1.py -q
cargo test --manifest-path runs/clinc150-l1-agent-session-effect-20260624/main-agent-session-5round/l1-agent-jobs/round-001/workspace/l1_programbank/Cargo.toml
uv run pytest tests/targets/nlu/test_l1_coding_agent.py tests/targets/nlu/test_l1_rust_worker.py tests/targets/nlu/test_clinc150_phase1.py tests/targets/nlu/test_l4_teacher.py -q
uv run pytest -q
uv run ruff check src/darjeeling/targets/nlu/clinc150_phase1.py src/darjeeling/targets/nlu/compiler/l1_program_compiler.py src/darjeeling/targets/nlu/main_cli.py tests/targets/nlu/test_clinc150_phase1.py tests/targets/nlu/test_l1_coding_agent.py tests/targets/nlu/test_l2_target_evolution.py
git diff --check
```

Results:

- focused ruff: passed;
- focused pytest: 35 passed;
- CLINC150 focused pytest: 27 passed;
- required focused pytest set: 60 passed;
- full pytest: 319 passed;
- selected L1 crate cargo test: passed.
- final touched-file ruff: passed;
- `git diff --check`: passed.
