# Target Boundary Handoff

> Historical audit snapshot. This handoff records the pre-refactor boundary
> problems and should be read together with
> [03 Target-Independent 架构](03_target_independent_architecture.md) and
> [04 Target-Independent 重构计划](04_target_independent_refactor_plan.md),
> whose execution log records the current module ownership.

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
- The core `edge-mvp` CLI no longer owns dataset preparation. The NLU target
  exposes `edge-mvp-nlu massive prepare` for converting MASSIVE into the
  target-owned record format.
- Core native defaults now point to `native/l1_empty_programbank`, a
  target-neutral contract-only Rust worker that always abstains. NLU settings
  point to `src/darjeeling/targets/nlu/native/l1_empty_programbank`.
- The repository no longer tracks an application-specific native L1 target
  crate. L1 accept-path tests use a neutral fixture under
  `tests/fixtures/l1_neutral_programbank`; real target crates are produced by
  the L4 workspace flow or supplied through explicit settings/artifacts.
- Report source excerpts no longer name specific L1 demo source files; they read
  from the promoted crate generically.

This keeps the design small: there is no registry, plugin layer, or adapter
framework. The boundary is expressed by defaults, console entry points, and
import direction:

```text
core runtime/compiler/eval -> generic records, schemas, manifests, workspaces
adapter/demo code -> concrete dataset loader or concrete target crate
```

## Third Pass: Strict Core Generalization Audit

Date: 2026-06-12.

The boundary above removed concrete MASSIVE/application defaults, but it is not
strict enough for the LLVM-style split now desired. The current repository is
best described as a dataset-independent NLU frame runtime, not a
target-independent Darjeeling core. NLU frame parsing is still embedded in core
contracts and should become a target implementation.

The desired split is:

```text
Darjeeling core
  -> layers, routing, trace flow, agent/evolution harnesses, replay,
     promotion, manifests, cost/latency accounting, quality gates

NLU target
  -> utterance input shape, Frame(intent, slots), TaskSchema, teacher prompt,
     label parser/equality, L1 ABI, L2 student, L3 prompt, NLU diagnostics,
     NLU report sections, MASSIVE adapter output mapping
```

### Current Violations

These are not concrete application leaks like `alarm_set` or `weather_query`;
they are target-contract leaks where core assumes the NLU task shape.

- `src/darjeeling/schemas.py` defines `Frame(intent, slots, is_abstain)` as the
  shared output type, and `LayerResult`, `TraceRecord`, and `TeacherTrace` all
  depend on it. This makes every runtime/compiler path a text-to-frame path.
- `src/darjeeling/layers/base.py`, `src/darjeeling/runtime/router.py`, and
  `src/darjeeling/runtime/replay.py` expose `try_answer(utterance) -> Frame`.
  Core should instead route target-neutral request payloads and opaque outputs.
- `src/darjeeling/runtime/replay.py::task_schema_from_records` derives
  `TaskSchema(intent_names, slot_names)` from `gold_frame`. Schema discovery is
  target logic and should be supplied by the target adapter.
- `src/darjeeling/layers/l4_cloud_llm.py` and
  `src/darjeeling/compiler/l4_context.py` implement an NLU teacher prompt and
  parser. Core may own L4 call/retry/caching mechanics, but the prompt,
  response format, parser, and cache key schema components should be target
  owned.
- `src/darjeeling/layers/l1_rust_programbank.py` and
  `native/l1_empty_programbank` define a native ABI that sends `utterance` and
  receives `Frame`. Core may own the subprocess harness, build, timeout, and
  benchmark wrapper; the request/response ABI should be target-owned or generic
  JSON.
- `src/darjeeling/layers/l2_student.py` is an NLU target implementation: intent
  classification, slot tagging, BIO reconstruction, slot postprocess,
  frame-retrieval, and frame-level guard features. It should move behind an NLU
  target boundary rather than remain a core layer.
- `src/darjeeling/layers/l2_target.py` is a target hook, but its hook names and
  types are still `postprocess_frame`, `accept_prediction`, and `Frame`.
- `src/darjeeling/layers/l3_local_slm.py` mixes a reusable local-LLM backend
  with an NLU prompt artifact, frame parser, and intent/slot validator. Split
  backend execution from NLU prompt/parse/validate logic.
- `src/darjeeling/compiler/loop.py` directly imports and orchestrates NLU layer
  implementations. It should become an orchestration loop that calls target
  compiler hooks for cache compilation, L1 candidate generation, L2 training,
  L3 prompt evolution, and replay.
- `src/darjeeling/compiler/l2_target_evolution.py` correctly uses an isolated
  workspace, but the harness itself is L2/NLU-specific: intent-stratified split,
  slot-cue probes, slot-risk backlog, intent-confusion backlog, and frame
  postprocess/veto tools. The reusable part is the agent workspace lifecycle,
  visible/private split discipline, scope checks, local-search hook, and
  adoption gates.
- `src/darjeeling/eval/metrics.py`, `src/darjeeling/compiler/objective.py`, and
  `src/darjeeling/eval/reports.py` expose `frame_exact_match`, intent/slot
  summaries, and L2/L3 NLU diagnostics as core report concepts. Core should own
  generic correctness/safety/cost/latency objectives and let the target render
  target-specific metric names.
- `src/darjeeling/settings.py` mixes orchestration settings with NLU/L2/L3
  target settings. Core settings should cover only generic runtime, teacher
  transport, agent harness, replay, promotion, and cost assumptions; target
  settings should be supplied by target config schemas.
- `tests/test_target_boundary.py` currently prevents concrete application and
  dataset terms from entering core, but it still treats `Frame`, `intent`,
  `slot`, and `utterance` as acceptable shared core terms. For strict core
  generalization, this test suite should be split into core-boundary tests and
  NLU-target tests.

### Things Already In The Right Direction

- The MASSIVE loader is isolated in `src/darjeeling/adapters/massive.py` and has
  its own CLI entry point.
- Hidden gold isolation is a useful core invariant, but the field names should
  become target-neutral over time, for example `gold_label`, `teacher_label`,
  and `final_output`.
- Artifact manifests, run directories, trace append, replay/promotion gates,
  visible/private split discipline, hard-buffer visibility, workspace scope
  checks, and agent provenance are mostly target-independent concepts.
- The L2 target workspace policy already distinguishes editable target code
  from protected system harness files. That pattern should become the generic
  target-evolution pattern.

### Refactor Plan

1. Introduce a target contract before moving large modules.
   Define a `TargetSpec` or `TargetAdapter` interface that owns input
   normalization, label parsing, output validation, equality/correctness,
   teacher prompt rendering/parsing, schema discovery, target diagnostics, and
   report rendering.
2. Make core schemas target-neutral.
   Replace public core references to `Frame`, `utterance`, `teacher_frame`,
   `gold_frame`, and `final_frame` with target-neutral request/output/label
   payloads. Keep compatibility shims until existing runs and tests migrate.
3. Extract the current NLU implementation into an explicit target package.
   A likely first location is `darjeeling.targets.nlu`, containing `Frame`,
   `TaskSchema`, NLU teacher, NLU L1 ABI helpers, NLU L2 student, NLU L3 prompt,
   NLU metrics, and NLU diagnostics.
4. Turn `compiler/loop.py` into orchestration only.
   It should split visible/private traces, call target/layer compiler hooks,
   replay candidates through generic layer handles, and apply promotion gates.
5. Split reusable harnesses from NLU diagnostics.
   Keep agent workspace creation, protected root checks, transcript/provenance,
   private gate discipline, and adoption decisions in core; move intent/slot
   family diagnostics and probe semantics to the NLU target.
6. Split CLI and settings.
   Keep core commands for run/report/preflight over a selected target. Move
   NLU-specific L2 train/tune/target-evolve and L3 prompt commands behind target
   command groups or a target CLI.
7. Upgrade architecture tests.
   Core tests should fail on `Frame`, `intent`, `slot`, `utterance`, and
   NLU-specific prompt text outside the NLU target package, adapters, fixtures,
   and experiment evidence. NLU target tests should keep the current behavior
   coverage.

### Structural Hotspots

A narrowed Cremona scan of `src/darjeeling` agreed with this priority order.
The highest-pressure files were:

- `src/darjeeling/compiler/l2_target_evolution.py`
- `src/darjeeling/cli.py`
- `src/darjeeling/eval/reports.py`
- `src/darjeeling/layers/l2_student.py`
- `src/darjeeling/compiler/loop.py`

The scan had partial signal health because coverage data was not supplied. A
full-repository scan also needs artifact directories such as `runs/` excluded;
otherwise audit tools can fail on excessive file arguments.
