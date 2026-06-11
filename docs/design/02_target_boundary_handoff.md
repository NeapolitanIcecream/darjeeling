# Target Boundary Handoff

This handoff is for the next Codex session. Start from design. Do not begin by
moving code.

## Current State

The project has just clarified a system-wide boundary rule, recorded in
`AGENTS.md`: Darjeeling core must remain dataset- and application-independent.

This is not an L2-only issue. The previous L2 work exposed the problem because
`l2_target_evolution.py` now contains concrete radio, joke, calendar, and slot
examples, but the same boundary applies to all layers and their tooling.

The useful analogy is LLVM:

- target-independent LLVM core should not know details of a concrete backend;
- target-dependent backend code may know those details, but it must live behind
  a clear boundary.

For Darjeeling, "target-dependent" means code for a concrete application,
dataset, schema, or task. That code is allowed, but it must not be mixed into
Darjeeling core.

## Boundary To Preserve

There are now three conceptual regions:

1. Darjeeling core
   - Framework runtime, compiler/evaluation harnesses, artifact plumbing,
     replay/promotion mechanics, reusable prompts/program text, default
     diagnostics, default probe machinery, and shared tests.
   - Core may consume schema names, slot keys, utterances, labels, and examples
     as input data.
   - Core must not hard-code concrete application intent names, slot names,
     dataset utterances, labels, request ids, or experiment failure cases.

2. Application / dataset / task target-dependent code
   - Code that connects Darjeeling to a concrete task.
   - This code may know application schema names and may contain
     dataset-independent business logic for that task.
   - This region is not owned by the L4 agent, but it must remain separated
     from core.

3. L4-agent-owned task workspaces and generated artifacts
   - Isolated L1/L2/L3 task workspaces and generated target artifacts.
   - Repository coding agents should not directly edit these files. They should
     change repo-level harnesses, prompts, tests, adapters, or contracts that
     govern what the L4 flow produces.

## Why This Matters Now

The recent branch made L2 safer by adding visible slot-cue pressure and replay
checks. The final result was useful:

- 3k outer replay promoted: L2 accepted `29/29` correctly, frame exact match
  stayed `1.0`, cost dropped from `0.217333` to `0.207715`.
- 10k outer replay was safe but not promoted: L2 accepted `6/6` correctly, frame
  exact match stayed `1.0`, but objective did not improve after artifact
  complexity.

However, the implementation also put target-dependent examples and schema facts
into core. Examples include concrete radio/joke/calendar phrases and checks in
`src/darjeeling/compiler/l2_target_evolution.py`, along with tests and prompt
text that assume a specific task schema.

Those examples are legitimate experiment evidence, but they are not legitimate
core defaults.

## Known Pressure Points

The next design pass should audit all layers, not just L2:

- L0 cache generation and replay assumptions.
- L1 program bank, L1 coding-agent context, and native-program tests.
- L2 student training/evaluation, target evolution, diagnostics, and probes.
- L3 prompt optimizer and prompt/eval fixtures.
- L4 context construction and agent-evolve harnesses.
- Shared schemas, settings, reports, and experiment comparison code.
- Shared tests that use concrete application utterances or schema names.

This list is a starting inventory, not a proposed module layout.

## What The Next Session Should Do

Start with a design document or design diff that answers:

- What exactly belongs to Darjeeling core?
- Where should target-dependent application/dataset code live?
- How should core call target-dependent code without knowing task details?
- Which existing files currently violate the boundary?
- Which existing experiment artifacts remain evidence only, rather than
  becoming core defaults?

Only after that design is clear should implementation begin.
