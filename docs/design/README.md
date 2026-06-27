# Darjeeling Design Index

This directory contains Darjeeling design documents. `../mvp_demo_proposal.md` is the original proposal; this directory records later engineering decisions and design discussions. When documents conflict, user decisions recorded here take precedence over the proposal and over the early implementation.

## Reboot Design

`reboot/` is the 2026-06-26 from-scratch target architecture based on GPT-5.5-Pro's 0626-2 feedback. It is a design target, not a description of the current implementation. It defines module boundaries for Core, Target Definition, Agent-managed workspace, L1/L2/L3 runtime artifacts, Candidate evaluation, Release runtime, and runtime feedback.

- [Reboot Design Index](reboot/README.md)
- [Reboot Overall Design](reboot/00_overall_design.md)
- [Reboot Implementation Runbook](reboot/implementation_runbook.md)

Read `reboot/` first when discussing the future complete version. Read the older design and experiment documents when discussing why the current code is shaped as it is.

## User Decisions

These decisions came from the user and have higher priority than the proposal and the early implementation:

1. **L1 should use a Rust native CPU program from the first version, not Python as the real L1 backend.**
   Python may be used for harnesses, tests, and glue code, but not as the basis for L1 latency or coverage claims.
2. **L1 is no longer DSL-first.**
   A DSL may remain as an optional table or rule helper format, but the main architecture path is an agent-maintained Rust source tree.
3. **L4 is a coding agent when evolving L1.**
   It is not "one L4 model instructing another agent." In L1 compiler mode, the L4 layer runs through a Codex CLI harness; the model is embedded in the coding agent and can edit files, build, test, and benchmark over multiple turns.
4. **L2 evolution is split between coding-agent structural work and local tuning tools.**
   The L4 coding agent handles L2 code, feature, and validation-protocol changes that require generalized intelligence. Local tools such as Optuna search inside the space designed by the agent. The older direct bounded-config proposal path remains only as a lightweight proposal path, not the final L2 evolution path.
5. **L1/L2/L3 evolution should use the same L4 coding-agent shape.**
   All three layers use isolated workspaces, round-local L4 agent sessions, agent-driven edit/evaluate/search loops, outer private checks, and outer replay release decisions. Their editable surfaces and tools differ: L1 is Rust/native code, L2 is target student code/Optuna/evaluation, and L3 is prompt/context/benchmark work.
6. **Darjeeling core should be tightened into a target-independent core.**
   NLU frame parsing is a target, not a built-in Core world model. Core keeps only target-independent layering, routing, trace flow, training/evolution harnesses, replay, release mechanics, artifact plumbing, and quality checks. `Frame(intent, slots)`, NLU teacher prompts, intent/slot diagnostics, and the MASSIVE loader belong in the NLU target or its adapters.
7. **L1/L2/L3 share an outer round policy, while target quality judgment stays in the target layer.**
   Core exposes only `max_rounds`, per-round timeout, patience, round executor, round result, and run summary. Core no longer carries pseudo-abstractions such as agent budget/profile/evidence that are not actually executed or only serve historical artifacts. It also does not define target quality or replay cadence semantics. L1/L2/L3 may use different workspaces and evaluators, but report the same round/run shape.
8. **Target-dependent optimization is allowed adaptation cost, not a Core contribution.**
   Darjeeling does not promise zero-target-knowledge magic. Users may provide target diagnostics, feedback generators, selection helpers, search tools, and target-specific L1/L2/L3 artifact code to explore upper-bound performance. Experiment reports must distinguish lift from target adaptation from reusable evidence about the Core/system method.

## Current State, 2026-06-24

- The current Phase 1 benchmark is CLINC150 `data_full`. MASSIVE remains as a historical NLU adapter and comparison artifact, but is no longer the main benchmark for current mechanism validation.
- The CLINC150 L4 teacher gate has passed: `clinc150-intent-v2-label-cards` reached 97.4% overall, 98.4% in-scope, and 0.0% parse/schema failure on the 500-row validation gate.
- L2 has shown absorption potential but has not reached the adoption target. The teacher-distilled L2 can reach 99.10% accepted precision / 50.32% coverage on validation, while locked test at the same threshold reached 98.77% accepted precision. Later guard and AutoResearch repair attempts still did not produce an adoptable locked-test candidate.
- The L1 Rust ProgramBank route remains active, but the latest CLINC150 dry-run patch experiment showed that validation-only phrase rules do not generalize: validation reached 100.00% accepted precision / 60.35% coverage, while locked test fell to 92.73% accepted precision. The next step should use real `agent-session` evolution and train-derived calibration/dev pressure rather than more patch-mode conclusions.
- The outer evolution policy refactor has been merged. Core's shared surface is now a plain round policy/summary; target layers continue to own their own workspaces, diagnostics, private checks, and adoption logic.
- Future CLINC150 L1/L2 upper-bound experiments may add target-specific optimization in the NLU target and isolated candidate workspaces. Those optimizations measure the payoff from target adaptation effort and should not flow back into Core defaults.

## Documents

- [00 User Decisions And Priority](00_decisions.md)
- [01 Overall Architecture](01_architecture.md)
- [02 Target Boundary Handoff](02_target_boundary_handoff.md)
- [03 Target-Independent Architecture](03_target_independent_architecture.md)
- [04 Target-Independent Refactor Plan](04_target_independent_refactor_plan.md)
- [modules/schemas.md](modules/schemas.md)
- [modules/settings.md](modules/settings.md)
- [modules/cli.md](modules/cli.md)
- [modules/data.md](modules/data.md)
- [modules/runtime.md](modules/runtime.md)
- [modules/l0_cache.md](modules/l0_cache.md)
- [modules/l1_rust_programbank.md](modules/l1_rust_programbank.md)
- [modules/l2_student.md](modules/l2_student.md)
- [modules/l3_local_slm.md](modules/l3_local_slm.md)
- [modules/l4_layer.md](modules/l4_layer.md)
- [modules/l4_agent_evolve_harness.md](modules/l4_agent_evolve_harness.md)
- [modules/l4_context.md](modules/l4_context.md)
- [modules/compiler.md](modules/compiler.md)
- [modules/replay_promotion.md](modules/replay_promotion.md)
- [modules/artifacts.md](modules/artifacts.md)
- [modules/eval_reports.md](modules/eval_reports.md)
- [modules/testing.md](modules/testing.md)
- [../experiments/README.md](../experiments/README.md)

## Naming

The project distribution name and Python import package name should be `darjeeling`. CLI command names keep the proposal's `edge-mvp` naming.

```text
[project]
name = "darjeeling"

[project.scripts]
edge-mvp = "darjeeling.cli:app"
edge-mvp-nlu = "darjeeling.targets.nlu.main_cli:app"
```

The current implementation already uses `src/darjeeling`. `edge-mvp` is the Core CLI and selects targets through a static target registry; `edge-mvp-nlu` exposes NLU target dataset/workflow commands.
