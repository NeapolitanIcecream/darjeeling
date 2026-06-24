# CLINC150 L1 ProgramBank Experiment Report

Date: 2026-06-24

Decision: **Pause and repair harness/evolution**.

The existing Rust ProgramBank route can absorb a large validation share with
fast CPU-native code, but the validation-selected candidate did not generalize
to locked test. The selected candidate reached 100.00% validation accepted
precision at 60.35% coverage, then fell to 92.73% accepted precision and a
7.10% lower-layer OOS false-accept rate on locked test. That misses the 99%
precision, <=2% OOS false-accept, and -0.5pp cascade-delta gates by a wide
margin.

This is not enough evidence to revisit the L1 technical route. It is evidence
that the current L1 evolution/selection loop is too validation-fragile for
CLINC150 and needs a train-derived calibration/dev gate before another locked
test exposure.

## Scope

Dedicated branch/worktree:

```text
branch: codex/clinc150-l1-programbank
worktree: /Users/chenmohan/gits/darjeeling-clinc150-l1-programbank
```

Primary experiment root:

```text
runs/clinc150-l1-programbank-20260624/
```

Primary measurements kept L0, L2, and L3 disabled. Results are reported as:

- L1-only shadow;
- L1 + L4 replay-oracle fallback;
- all-L4 replay-oracle baseline.

## Reused Artifacts And Cost

No new paid L4 benchmark calls were made. New observed paid L4 spend for this
experiment was `$0.00`.

Cost ledger:

```text
docs/experiments/2026-06-24_clinc150_l1_programbank_cost_ledger.json
```

Prior ledgers reused:

- `docs/experiments/2026-06-23_clinc150_l2_cascade_cost_ledger.json`
  (`$20.9290952` observed spend);
- `docs/experiments/2026-06-23_clinc150_teacher_reliability_cost_ledger.json`
  (`$0.6994568` observed spend).

Copied artifact row counts were validated after copying from the main worktree:

| Artifact | Copied rows |
| --- | ---: |
| `data/processed/clinc150_data_full/train.jsonl` | 15,100 |
| `data/processed/clinc150_data_full/validation.jsonl` | 3,100 |
| `data/processed/clinc150_data_full/test.jsonl` | 5,500 |
| `runs/clinc150-l1-programbank-20260624/reused-teacher-traces/train-full-stratified/teacher_live_vs_gold.details.jsonl` | 15,100 |
| `runs/clinc150-l1-programbank-20260624/reused-teacher-traces/validation-full/teacher_live_vs_gold.details.jsonl` | 3,100 |
| `runs/clinc150-l1-programbank-20260624/reused-teacher-traces/test-full/teacher_live_vs_gold.details.jsonl` | 5,500 |

## Harness Changes

Implemented target-local CLINC150 L1 support in
`src/darjeeling/targets/nlu/clinc150_phase1.py` and
`src/darjeeling/targets/nlu/main_cli.py`:

- `Clinc150L4ReplayOracle` loads paid teacher detail rows by request id,
  validates request coverage, exposes recorded teacher frame/output, and exposes
  model, token, cost, latency, attempt, parse, retry, and empty-response stats.
- `evaluate_clinc150_l1` runs an L1 worker over CLINC150 records and records
  accepted/abstained output, frame/patch materialization, program path, native
  latency, correctness, accepted errors, OOS false accepts, and L1+L4 fallback
  metrics.
- `clinc150 l1-eval` builds a selected Rust ProgramBank crate and writes JSON
  summary, prediction details JSONL, accepted-error JSONL, and cost/latency
  table.
- `select_clinc150_l1_candidate` refuses locked-test selection splits.

Replay-oracle semantics are explicit experiment accounting, not production cache
behavior. Fallback rows count as L4 calls and retain recorded L4 cost, token,
latency, model, retry, and parse/schema diagnostics. `TeacherCache` semantics
were not changed.

The L1 coding-agent prompt and constraints now explicitly allow large,
repetitive, hard-coded CPU-native Rust ProgramBank logic while preserving
benchmark isolation and prohibiting changes to the outer evaluator, promotion
logic, teacher cache, and Python orchestration.

Focused tests cover accept/abstain/wrong accept/OOS false accept, replay-oracle
fallback cost/latency accounting, L0/L2/L3 absence from primary metrics,
locked-test selection refusal, and the updated large-hard-coded ProgramBank
prompt constraints.

## Baselines

| View | Requests | All-L4 acc | L1 precision | L1 coverage | Final cascade acc | L4 calls / 100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Empty L1 validation | 3,100 | 98.323% | n/a | 0.000% | 98.323% | 100.000 |
| Empty L1 locked test | 5,500 | 96.018% | n/a | 0.000% | 96.018% | 100.000 |

The empty/default ProgramBank abstained on all rows. This verified that L1+L4
fallback accounting degenerates to all-L4 replay-oracle accounting.

## Evolution Attempts

All attempts used isolated Rust ProgramBank workspaces under
`runs/clinc150-l1-programbank-20260624/l1-agent-jobs/`. The jobs used the
existing L1 coding-agent workspace/provenance path in dry-run patch mode and
ran `cargo test` inside each generated crate. Visible context came from copied
train teacher rows only; validation was used for candidate evaluation/selection;
locked test was not used until the selected candidate was fixed.

Attempt 1:

- Hypothesis: high-support train-unique phrase rules may be precise enough
  without validation pruning.
- Candidate: `attempt-1-train-unique-support10-raw`, 1,247 phrase rules.
- Validation result: 96.65% accepted precision, 34.65% coverage, 36 wrong
  accepts, 4.00% OOS false-accept rate, -0.742pp cascade delta.
- Decision: reject. Accepted errors showed broad phrase ambiguity.

Attempt 2:

- Hypothesis: validation accepted-error audit can prune risky train phrases
  while preserving meaningful coverage.
- Candidate: `attempt-2-validation-audited-support5`, 1,723 phrase rules.
- Validation result: 100.00% accepted precision, 44.97% coverage, 0 wrong
  accepts, 0.00% OOS false-accept rate, +0.323pp cascade delta.
- Decision: eligible, but attempt 3 had higher validation coverage with the
  same quality margin.

Attempt 3:

- Hypothesis: lower train support can raise coverage if each rule has clean
  validation shadow precision.
- Candidate: `attempt-3-validation-audited-support2`, 3,947 phrase rules.
- Validation result: 100.00% accepted precision, 60.35% coverage, 0 wrong
  accepts, 0.00% OOS false-accept rate, +0.419pp cascade delta.
- Decision: selected for locked test because it had the best validation coverage
  among candidates satisfying the quality gates.

## Selected Candidate Results

| View | All-L4 acc | L1 precision | L1 coverage | Wrong accepts | OOS false accept | Cascade acc | Delta vs all-L4 | L4 calls / 100 | Cost reduction | Native p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation sequential | 98.323% | 100.00% | 60.35% | 0 | 0.00% | 98.742% | +0.419pp | 39.645 | 59.99% | 697 us |
| Validation uniform | 98.194% | 100.00% | 59.87% | 0 | 0.00% | 98.645% | +0.452pp | 40.129 | 59.58% | 709 us |
| Validation zipf-heavy | 97.484% | 100.00% | 48.52% | 0 | 0.00% | 97.742% | +0.258pp | 51.484 | 47.37% | 735 us |
| Locked test sequential | 96.018% | 92.73% | 39.76% | 159 | 7.10% | 94.091% | -1.927pp | 60.236 | 39.63% | 715 us |

The locked test result fails the decision criteria:

- accepted precision target: >=99%; observed 92.73%;
- lower-layer OOS false-accept target: <=2%; observed 7.10%;
- cascade delta target: >= -0.5pp; observed -1.927pp.

The failure was not narrow, so no locked-test-driven repair was performed.

## Accepted-Error Audit

Validation selected candidate:

- sequential validation accepted errors: 0;
- validation uniform accepted errors: 0;
- validation zipf-heavy accepted errors: 0.

Locked test selected candidate:

- accepted wrong count: 159;
- OOS false accepts: 71;
- in-scope wrong accepts: 88.

Representative locked-test accepted errors:

| Request | Gold | L1 |
| --- | --- | --- |
| `repeat what the weather will be like` | `transfer` | `repeat` |
| `i want to change to a new allstate insurance plan` | `insurance_change` | `change_accent` |
| `am i safe to go to africa` | `travel_alert` | `vaccines` |
| `how can i increase my credit score` | `improve_credit_score` | `credit_limit_change` |
| `add laundry detergent to the list` | `shopping_list_update` | `todo_list_update` |

The error pattern points to validation-fragile substring rules, not a Rust worker
or replay-oracle accounting defect. The phrase table found many validation-clean
anchors that did not remain clean on locked test.

## Decision

**Pause and repair harness/evolution.**

Proceed is not supported because the validation-selected L1 failed locked-test
precision, OOS false-accept, and cascade quality gates. Revisit L1 route is also
not supported: the Rust ProgramBank build/run path is fast, isolated, and can
absorb large validation coverage. The main problem is candidate selection and
guard robustness.

## Risks And Next Step

Before another locked-test exposure, repair selection without test labels:

- create deterministic train-derived calibration/dev splits for L1 ProgramBank
  rule eligibility;
- require candidate rules to pass train-derived holdout precision and official
  validation precision, not validation-only per-rule cleanliness;
- add an OOS-heavy validation/dev slice before selection;
- prefer phrase rules with stronger train support, explicit word-boundary
  matching, and conflict-family vetoes derived from train/dev plus validation
  accepted-error audits;
- keep L0/L2/L3 disabled for primary L1 decisions;
- continue using replay-oracle fallback accounting and the same accepted-error
  JSONL audit path.

## Validation Commands

Checks run:

```bash
cargo test --manifest-path runs/clinc150-l1-programbank-20260624/l1-agent-jobs/attempt-3-validation-audited-support2/workspace/l1_programbank/Cargo.toml
uv run pytest tests/targets/nlu/test_l1_rust_worker.py tests/targets/nlu/test_clinc150_phase1.py tests/targets/nlu/test_l4_teacher.py -q
uv run ruff check src/darjeeling/targets/nlu/clinc150_phase1.py src/darjeeling/targets/nlu/main_cli.py src/darjeeling/targets/nlu/compiler/l1_program_compiler.py tests/targets/nlu/test_clinc150_phase1.py tests/targets/nlu/test_l1_rust_worker.py
uv sync --extra dev --extra massive
uv run pytest -q
git diff --check
```

Results:

- selected L1 crate `cargo test`: passed;
- focused pytest: 41 passed;
- touched-file ruff check: passed;
- full pytest: 301 passed;
- `git diff --check`: passed.

Key experiment commands included:

```bash
uv run python -m darjeeling.targets.nlu.main_cli clinc150 l1-eval --crate-dir runs/clinc150-l1-programbank-20260624/l1-agent-jobs/attempt-3-validation-audited-support2/workspace/l1_programbank --split validation --teacher-details runs/clinc150-l1-programbank-20260624/reused-teacher-traces/validation-full/teacher_live_vs_gold.details.jsonl --write-details
uv run python -m darjeeling.targets.nlu.main_cli clinc150 l1-eval --crate-dir runs/clinc150-l1-programbank-20260624/l1-agent-jobs/attempt-3-validation-audited-support2/workspace/l1_programbank --split test --teacher-details runs/clinc150-l1-programbank-20260624/reused-teacher-traces/test-full/teacher_live_vs_gold.details.jsonl --write-details
```
