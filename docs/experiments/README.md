# Experiment Index

This page points to the current experiment conclusions. Individual dated reports
remain the audit trail; this index is the short map for deciding what to read
next.

## Current Benchmark

The active Phase 1 benchmark is CLINC150 `data_full`.

Why CLINC150 replaced MASSIVE for the current phase:

- CLINC150 has a closed intent-label task with explicit out-of-scope rows.
- A repaired L4 teacher path can classify it reliably enough for Phase 1.
- The benchmark is still inside the NLU target, but it better matches the
  mechanism test: L4 should do the task well, and Darjeeling should try to move
  stable work from L4 into L1/L2 without a large quality drop.

MASSIVE remains useful as historical evidence. The June 22 audit found that many
MASSIVE errors were dataset convention mismatch or exact-match artifacts, so it
is not the current primary benchmark for mechanism validation.

## CLINC150 Teacher Gate

Report: [2026-06-23_clinc150_phase1_report.md](2026-06-23_clinc150_phase1_report.md)

Reliability repair:
[2026-06-23_clinc150_teacher_reliability_report.md](2026-06-23_clinc150_teacher_reliability_report.md)

Current decision: continue CLINC150 Phase 1.

Key result:

| Prompt | Overall accuracy | In-scope accuracy | Parse/schema failure | Decision |
| --- | ---: | ---: | ---: | --- |
| `clinc150-intent-v2-label-cards` | 97.4% | 98.4% | 0.0% | passed |

Important lesson: the first failed teacher gate was mainly a call-configuration
defect. `max_completion_tokens=64` was too tight for reasoning-model JSON
responses, so the repaired path uses a larger completion budget, attempt-level
diagnostics, retries, incremental JSONL writes, and resume support.

## CLINC150 L2

First full L2/cascade report:
[2026-06-23_clinc150_l2_cascade_report.md](2026-06-23_clinc150_l2_cascade_report.md)

Calibration repair:
[2026-06-24_clinc150_calibration_repair_report.md](2026-06-24_clinc150_calibration_repair_report.md)

L4 AutoResearch bridge:
[2026-06-24_clinc150_l2_autoresearch_report.md](2026-06-24_clinc150_l2_autoresearch_report.md)

Current decision: pause and repair the AutoResearch harness. Do not adopt an L2
candidate yet, but do not reject the L2 route.

Key results:

| Run | Split | Accepted precision | Coverage | Outcome |
| --- | --- | ---: | ---: | --- |
| Teacher-distilled L2, threshold `0.98` | validation | 99.10% | 50.32% | passed validation |
| Teacher-distilled L2, threshold `0.98` | locked test | 98.77% | 42.73% | missed 99% precision |
| Conservative threshold `0.995` | locked test | 99.78% | 24.56% | quality-safe, low coverage |
| Calibration repair guard | locked test | 98.997% | 38.07% | missed strict precision and practical coverage |
| First AutoResearch candidate | locked test | n/a | n/a | not exposed to locked test |

Latest diagnosis:

- Fixed teacher-distilled L2 can absorb a meaningful share of CLINC150.
- Validation-only threshold selection is too fragile under test and stream
  shifts.
- The first L4 AutoResearch run produced useful target-local infrastructure but
  overfit visible accepted-error cleanup. It failed private selection and did
  not reach locked test.
- Next L2 work should add stronger visible OOS-heavy and cross-fold pressure,
  make config search write scratch candidates instead of mutating the active
  config, and rerun a longer AutoResearch attempt.

## CLINC150 L1

Report: [2026-06-24_clinc150_l1_programbank_report.md](2026-06-24_clinc150_l1_programbank_report.md)

Next plan:
[2026-06-24_clinc150_l1_agent_session_effect_plan.md](2026-06-24_clinc150_l1_agent_session_effect_plan.md)

Current decision: pause and repair L1 harness/evolution before another locked
test exposure.

Key result:

| Split | L1 accepted precision | L1 coverage | OOS false accept | Outcome |
| --- | ---: | ---: | ---: | --- |
| validation | 100.00% | 60.35% | 0.00% | selected by validation |
| locked test | 92.73% | 39.76% | 7.10% | failed |

Latest diagnosis:

- The Rust ProgramBank route is fast and can absorb a large validation share.
- The failed run used dry-run patch evolution, not the full L4 agent-session
  evolve method.
- The validation-selected phrase table did not generalize to locked test. The
  next L1 experiment should use real `agent-session` evolution with
  train-derived calibration/dev and OOS-heavy pressure before any locked-test
  exposure.

## Outer Evolution Policy

Plan:
[2026-06-24_outer_evolution_policy_simplification_plan.md](2026-06-24_outer_evolution_policy_simplification_plan.md)

Report:
[2026-06-24_outer_evolution_policy_simplification_report.md](2026-06-24_outer_evolution_policy_simplification_report.md)

Current decision: merged.

The core policy now exposes only generic run mechanics:

```text
EvolutionRunPolicy(max_rounds, round_timeout_s, patience_rounds, round_executor)
EvolutionRoundResult
EvolutionRunSummary
```

Core no longer carries target quality claims, private gate requirements, replay
requirements, profile guidance, cost policy, or fixed inner-loop cadence. L1 and
L3 now execute real `max_rounds`; L2 keeps target-owned train/evaluate and gate
logic while reporting round summaries without the removed pseudo-abstractions.

## Cost Notes

Observed paid benchmark spend from the CLINC150 runs is recorded in the dated
cost ledgers. The L1 ProgramBank and L2 AutoResearch reports reused existing L4
teacher artifacts and added no new paid benchmark L4 calls.

Coding-agent session token usage may be reported separately when available. It
is not the same as benchmark serving spend.

## Precision-Coverage Visualization

Plan:
[2026-06-24_precision_coverage_visualization_plan.md](2026-06-24_precision_coverage_visualization_plan.md)

Repair plan:
[2026-06-24_precision_coverage_frontier_repair_plan.md](2026-06-24_precision_coverage_frontier_repair_plan.md)

Current decision: planned.

Future L1/L2 reports should standardize two static Seaborn figure families:
round/generation evolution curves for accepted precision and coverage, and
coverage-vs-precision operating frontiers for candidate-local accept-policy
trade-offs. L1 operating frontiers should be target-adapter overlays over
recorded L1 outputs, not requirements pushed into the generated L1 artifact.
The first implementation branch produced useful data infrastructure but the
frontier plot design was too mixed; the repair plan narrows each standard
frontier figure to one split, one candidate, and one explicit knob.

## Overnight Autonomous Research

Plan:
[2026-06-24_overnight_autonomous_research_plan.md](2026-06-24_overnight_autonomous_research_plan.md)

Current decision: planned.

This is a high-autonomy overnight sprint intended to use the remaining weekly
Codex quota before 2026-06-25 08:00 Asia/Shanghai. It should not stop after a
single successful patch; it should continue hypothesis, experiment,
implementation, validation, and reporting cycles across L1, L2, evidence, and
related-work tracks while respecting the target/core boundary.
