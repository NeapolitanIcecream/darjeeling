# Target Boundary Handoff

This handoff is for the next Codex session. Start from design. Do not begin by
moving code.

## Current State

The project has just clarified a system-wide boundary rule, recorded in
`AGENTS.md`: Darjeeling core must remain dataset- and application-independent.

This is not an L2-only issue. The previous L2 work exposed the problem because
`l2_target_evolution.py` previously contained concrete application examples,
but the same boundary applies to all layers and their tooling.

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
into core. Examples included concrete application phrases and checks in
`src/darjeeling/compiler/l2_target_evolution.py`, along with tests and prompt
text that assumed a specific task schema.

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

## Design Pass: Target Boundary Refactor

This pass keeps the boundary smaller than a plugin system:

- Core may summarize visible target data generically. Examples:
  `visible_slot_cue_summary`, safety backlogs, intent-confusion backlogs,
  train/validation split metadata, and replay metrics.
- Core may execute a generic target-probe contract supplied as workspace data.
  It may not synthesize application-specific probes itself.
- Target-specific code and target-specific probe specs belong outside core,
  either in an application/dataset adapter or in an isolated target workspace.
- Experiment docs and artifacts may keep dataset-specific evidence, but that
  evidence must not become reusable prompt text, default diagnostics, or shared
  test fixtures.

The immediate implementation target is the L2 target-evolution slot-cue probe.
The old core behavior generated probes for concrete application-specific
intents and slots from `visible_slot_cue_summary`. That mixed useful
experiment evidence into core.
The refactor makes `slot_cue_probes` an optional data-driven diagnostic:

```text
data/slot_cue_probes.json
  -> generic evaluator contract
  -> target/target_l2.py postprocess/accept hooks
  -> diagnostic-only result
```

If the probe file is absent, core reports an empty diagnostic instead of
inventing target-specific probes. This preserves the workflow while removing
the default application knowledge.

While auditing, one smaller core default was also identified: the L3 CLI
benchmark fallback schema used concrete application names when no processed
dataset was available. That fallback should use neutral synthetic names because
real task schema must come from runtime data or an adapter.

## Design Pass: Shared Test Boundary

`AGENTS.md` treats shared tests as core, so tests need the same target boundary
as runtime and compiler code. The policy is intentionally small:

- Core/shared tests use neutral fixture schema and utterances such as
  `intent_alpha`, `intent_beta`, `slot_alpha`, `alpha request`, and
  `beta request`.
- Adapter tests may mention their concrete dataset because they are testing the
  adapter boundary itself.
- Demo target tests may mention their concrete demo schema, but they must be
  confined to explicitly demo-owned files or test cases and must not define core
  defaults.
- Experiment docs and archived patches may keep dataset-specific evidence.
  Design docs for current architecture should use generic terms such as hidden
  gold labels unless they are documenting an adapter.

This avoids a fixture registry or plugin abstraction. The enforcement mechanism
is a source scan in `tests/test_target_boundary.py` plus ordinary focused tests
for adapter/demo behavior.

## Second Pass: Runtime Defaults

The first pass was too narrow by itself. The second pass addresses runtime
defaults and adapter placement:

- `TaskSchema.schema_version` now defaults to a neutral `task-schema-v1`.
- Generic `DataRecord` lives in `darjeeling.data.records`.
- The MASSIVE loader lives in `darjeeling.adapters.massive`; runtime/replay reads
  processed `DataRecord` rows and does not import the MASSIVE adapter.
- The core `edge-mvp` CLI no longer owns dataset preparation. The bundled
  MASSIVE adapter exposes `edge-mvp-massive prepare` for converting that dataset
  into the shared `DataRecord` format.
- Core L1 settings and CLI defaults now point to `native/l1_empty_programbank`,
  a contract-only Rust worker that always abstains.
- The application-specific `native/l1_programbank` remains in the repository as
  an explicit demo/target crate and test fixture, not as the core default.
- Report source excerpts no longer name specific L1 demo source files; they read
  from the promoted crate generically.

This keeps the design small: there is no registry, plugin layer, or adapter
framework. The boundary is expressed by defaults, console entry points, and
import direction:

```text
core runtime/compiler/eval -> generic records, schemas, manifests, workspaces
adapter/demo code -> concrete dataset loader or concrete target crate
```
