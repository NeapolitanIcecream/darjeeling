# GPT-5.5 Review Refactor Plan

Date: 2026-06-13

Inputs:

- `/Users/chenmohan/Downloads/Darjeeling-research-0613.md`
- `docs/experiments/2026-06-13_gpt55_review_verification.md`

Goal: fully resolve the verified design issues, not just document or partially mitigate them. The result should make partial field acceptance, accepted-audit evidence, L2 target evolution, L2 local specialization, focus-task context, L3 residual gating, and compiler-loop modularity real parts of the system. This still must respect Darjeeling's existing target/core split and avoid framework-heavy abstractions.

## Non-Negotiable Constraints

- Do not add target-specific concepts to Darjeeling core. `Frame`, intent, slots, NLU labels, MASSIVE fields, and dataset examples stay in the NLU target or experiment docs.
- "Fully fix" means fully fix the NLU target's behavior and artifact lifecycle while keeping core generic. Core may route opaque JSON and manage generic artifacts/promotions, but it must not interpret intent, slot, frame, utterance, teacher frame, field coverage, or NLU residuals.
- Do not introduce dependency injection containers, schema DSLs, plugin frameworks, or broad runtime abstractions.
- Do not directly edit task-specific isolated L1/L2/L3 workspaces or generated target artifacts as the repo-level fix. Change the repo harnesses, contracts, prompts, tests, and adapters that create and evaluate them.
- Reduce terminology. Internally use ordinary names such as "patch", "composer", "expert", and "focus task"; avoid making every idea a new public concept.

## Plan

### 1. Add Teacher Audit For Lower-Layer Accepts

Fixes: missing accepted-audit evidence.

- Add NLU replay settings for lower-layer audit, with an eval/dev default that can audit every lower-layer accept and a runtime option for sampling.
- In `run_replay`, when L0/L1/L2/L3 accepts and the teacher cache has no label, optionally shadow L4 and write the result to the cache/trace.
- Record simple trace metadata: lower layer accepted, teacher audited, teacher disagreed.
- Update hard-case mining and reports to use these audit fields for wrong accepts and disagreement.
- Keep this NLU-target specific; do not change core trace contracts unless a later target-neutral need appears.

Done when eval replay can produce teacher labels for all lower-layer accepts, wrong accepts are visible in hard buffers, and existing replay tests plus new audit tests pass.

### 2. Introduce NLU Frame Patches And A Composer

Fixes: whole-frame cascade and full-frame trace/metric blind spots.

- Add an NLU-only `FramePatch` model as the primary NLU routing unit. It should carry accepted fields, not a new framework:
  - optional accepted intent
  - accepted slot values by slot key
  - source layer
  - confidence/metadata
  - whether the patch claims the frame is complete
- Add legacy adapters so existing full-frame layers still work:
  - accepted full frame -> complete patch
  - rejected layer -> empty patch
- Add an NLU composer that applies patches in layer order and produces the final full `Frame`.
- Make component-wise composition the normal NLU runtime path. L0/L1/L2/L3 may each contribute reliable fields; L4 fills the remaining fields and guarantees a complete final frame.
- Extend NLU traces/reports with field-level coverage and wrong-accepted-field metrics while preserving existing full-frame exact-match metrics.
- Keep the core `CascadeRouter` intact for target-neutral runtime. The NLU target can use its own patch-aware route path behind the target boundary.

Done when current whole-frame behavior is preserved through adapters, component-wise NLU routing is exercised in replay, L4 fills residual fields, and tests cover mixed-layer field composition plus full-frame legacy layers.

### 3. Make L2 Target Evolution A Main Compiler Candidate

Fixes: L2 target evolution living beside, not inside, the main promotion loop.

- Reuse the existing `l2_target_evolution.py` capability; do not rewrite it.
- Add a main-loop candidate path that can:
  - train the L2 student bundle,
  - run or load an L2 target evolution candidate,
  - assemble `l2_student.joblib` plus `l2_target.py`,
  - evaluate both together in offline replay,
  - promote both or neither.
- Keep the standalone CLI flows, but move shared logic into explicit helper functions so CLI and compiler use the same adoption rules.
- Stop dropping `l2_target` merely because the compiler retrained L2. Drop it only when no compatible target-aware candidate exists.

Done when `run_compiler_generation` can produce a candidate artifact set containing both `l2_student` and `l2_target`, replay it with `TargetL2Layer`, and reject/promote it through the same gates as other artifacts.

### 4. Add A Small L2 Expert Bank

Fixes: L2 being only a global student.

- Keep the existing global L2 student as the baseline/fallback.
- Add an NLU-specific expert-bank artifact with a plain manifest and explicit static wiring, not a generic plugin system.
- Implement real local specialization, at minimum:
  - intent binary experts for selected high-value intent families
  - slot extractor/canonicalizer experts for selected high-value slot keys
- Each expert owns its own training slice, guard threshold, validation metrics, and target postprocess/veto rules.
- Runtime experts emit `FramePatch` objects. The composer decides whether enough fields are covered.
- The compiler should select experts from workload evidence and promote them based on risk/coverage gates, not on expert count.

Done when intent and slot experts can be trained, replayed, selected, promoted, reported, and used by runtime composition; failed experts are rejected by promotion gates rather than shipped silently.

### 5. Replace Chronological Proposal Context With Focus Tasks

Fixes: L4 proposal context being raw first-N traces instead of high-ROI local work.

- Add a simple `focus_tasks.json` builder in the NLU compiler layer.
- Derive focus tasks from existing evidence:
  - hard buffer
  - lower-layer misses
  - wrong accepts and audit disagreements
  - L1 family context
  - L2 target family diagnostics
- A focus task should include a goal, positive examples, near negatives, current failures, current layer behavior, and a precision floor.
- Use focus tasks as the primary context for L1/L2/L3 agent and proposal inputs.
- Keep raw trace examples only as supporting evidence. Remove reliance on sorted first-50 trace windows as the main proposal signal.

Done when generic L4 proposal context and target-evolution workspaces are driven by ranked focus tasks with source trace IDs, and tests prove the context excludes gold labels.

### 6. Gate L3 By Residual Value

Fixes: L3 being structurally slow/fragile and not checked for workload value.

- Keep L3 optional as a consequence of evidence, not as an unfinished state. If L3 fails the residual value gate, runtime should deliberately skip it.
- Keep existing replay/promotion accuracy gates.
- Add a viability gate on the L2 residual: L3 can be runtime-guarded only if it improves expected cost/latency or L4-call avoidance without accuracy regression.
- Replace the current long default decode budget with a benchmark-backed bounded default.
- Add NLU-target prompt narrowing for residual cases, such as intent/slot shortlists, if it improves replay value. Avoid adding a model-serving backend or constrained-decoding subsystem unless the residual gate cannot be met otherwise.

Done when L3 prompt promotion records residual coverage, accuracy, wrong-accept rate, p95 latency, and expected value versus skipping L3.

### 7. Split The Compiler Loop Into Plain Candidate Steps

Fixes: `run_compiler_generation` becoming too large.

- Split the current function into explicit helpers:
  - `build_l0_candidate`
  - `evolve_l1_candidate`
  - `train_l2_student_candidate`
  - `evolve_l2_target_candidate`
  - `build_l3_prompt_candidate`
  - `assemble_candidate_artifact_set`
  - `evaluate_and_promote`
- Use one small return object such as `CandidatePart(paths, metrics, artifacts)`.
- Preserve current external behavior and manifest shape unless a migration is necessary for the earlier phases.
- Refactor only after tests pin the current behavior, so this step does not hide behavior changes.

Done when compiler-loop tests still pass, the main function is an orchestrator rather than the owner of every detail, and no generic lifecycle framework has been introduced.

### 8. Update Tests And Docs As Executable Boundaries

Fixes: future drift and handoff risk.

- Add focused tests per phase before broad rewrites.
- Update design docs for runtime, replay/promotion, compiler, L2, L3, and L4 context.
- Keep docs practical: describe the data that flows through the system and the gates that protect promotion, not new taxonomy.
- Add regression tests for target/core separation where new code touches boundaries.

Done when a new developer can understand the refactor from the docs without learning a large new vocabulary.

## Suggested Order

1. Lower-layer teacher audit.
2. NLU `FramePatch` plus composer and field-level metrics.
3. Compiler candidate-step extraction enough to support all remaining candidate sources cleanly.
4. L2 target evolution as a main compiler candidate.
5. L2 expert bank emitting patches.
6. Focus-task context for L1/L2/L3 proposal and agent inputs.
7. L3 residual viability gate and decode-budget reduction.
8. Final compiler-loop cleanup, full docs pass, and regression verification.

## Implementation Status

Implemented on 2026-06-13:

- Lower-layer audit settings and trace metadata are in `Settings` and NLU replay. Live
  teacher audit runs for lower-layer accepts when teacher mode permits it; cache-only runs
  record an explicit skip reason. Hard-buffer mining recognizes audit disagreement.
- `FramePatch`, NLU composer, and patch-aware NLU routing are target-owned. Legacy
  whole-frame layers are adapted to complete patches; L4 residual fill produces the final
  full frame. Core router and contracts remain target-neutral.
- Runtime traces, offline replay, compiler candidate metrics, and reports include
  field-level coverage and wrong accepted field metrics while preserving full-frame exact
  match gates.
- L2 target evolution is integrated into the main compiler candidate path. Adopted
  target snapshots are staged with the trained L2 bundle; compatible inherited targets are
  preserved; incompatible targets are dropped with an explicit reason.
- L2 expert bank artifacts train intent binary experts and slot value experts from
  teacher-visible workload evidence. Runtime experts emit incomplete patches and fall back
  to the global L2 student/target wrapper.
- Proposal context and L1 agent workspaces use ranked focus tasks as the primary dynamic
  context. Raw traces are supporting evidence only, and gold-leakage tests cover the new
  payloads.
- Guarded L3 runtime prompts are kept only when the residual value gate passes. Failing
  evidence removes runtime `l3_prompt` and sets effective `l3_mode=disabled`; default local
  decode budget is reduced to 64 tokens.
- `run_compiler_generation` now calls plain candidate helpers:
  `build_l0_candidate`, `evolve_l1_candidate`, `train_l2_student_candidate`,
  `evolve_l2_target_candidate`, `build_l3_prompt_candidate`,
  `assemble_candidate_artifact_set`, and `evaluate_and_promote`.
- Added regression tests for patch routing, audit, field metrics, L2 target/expert
  lifecycle, focus tasks, L3 residual gating, and target/core boundary separation.

Verification:

```bash
uv run pytest tests/targets/nlu -q
# 213 passed
```

## Agent Prompt

Use this shorter prompt when handing the work to a Codex agent:

```text
Work in /Users/chenmohan/gits/darjeeling.

Read these files first:
- AGENTS.md
- /Users/chenmohan/Downloads/Darjeeling-research-0613.md
- docs/experiments/2026-06-13_gpt55_review_verification.md
- docs/design/05_gpt55_review_refactor_plan.md

Complete the refactor plan in docs/design/05_gpt55_review_refactor_plan.md end to end. This is full-fix mode: do not stop at documentation, stubs, "future work", or disabled placeholders unless a gate deliberately rejects a feature based on replay evidence. Refine design where needed, update docs, implement code, add tests, run verification, fix failures, and iterate until the plan's design goals are met.

Non-negotiables:
- Keep Darjeeling core target/dataset/app independent.
- Keep NLU terms and schema details inside the NLU target/adapters/docs.
- Full fix means the NLU target becomes component-wise, audited, expert-capable, focus-task-driven, and residual-gated; it does not mean core learns Frame/intent/slot semantics.
- Keep abstraction tax low: plain functions/classes, explicit manifests, static wiring. No plugin systems, DI containers, schema DSLs, or generic lifecycle frameworks.
- Do not edit isolated/generated L1/L2/L3 workspaces as the repo fix; edit the harnesses, prompts, tests, adapters, and contracts that govern them.
- Preserve existing behavior through compatibility adapters while migrating.

Required outcomes:
- lower-layer teacher audit and disagreement evidence;
- NLU partial field patches plus composer as the normal NLU runtime path, with legacy whole-frame layers still working;
- field-level replay/report metrics;
- L2 target evolution integrated into main candidate -> replay -> promote;
- L2 expert bank for intent/slot specialization, selected and promoted from workload evidence;
- ranked focus-task contexts replacing chronological first-N proposal traces;
- L3 runtime use deliberately enabled or skipped by residual accuracy and cost/latency value;
- `run_compiler_generation` split into plain helper steps;
- updated tests and docs protecting target/core boundaries.

Maintain a progress doc under docs/experiments/. Work in small tested slices. Prefer NLU-specific implementation. Only move target-neutral plumbing to core, and keep payload semantics opaque there.

Before finishing, run at least:
  uv run pytest tests/targets/nlu -q
Run `uv run pytest tests -q` too if core contracts, artifact store, CLI target wiring, or router behavior changed. Record any tests you cannot run.

Done means all outcomes are implemented and exercised, or deliberately rejected by evidence-backed gates where runtime use would be harmful; relevant tests pass, docs explain the new boundaries, and no broad framework was introduced.
```
