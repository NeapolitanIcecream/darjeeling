# GPT-5.5-Pro 2026-06-14 Refactor Plan

Date: 2026-06-14

Inputs:

- `/Users/chenmohan/Downloads/Darjeeling-research-0614.md`
- `docs/experiments/2026-06-14_gpt55_pro_review_verification.md`

Goal: close every verified issue from the 2026-06-14 review while preserving the target/core boundary and keeping the system easy to understand. The work should make partial NLU patches produce real correctness, cost, latency, and reporting value without adding a broad framework.

## Constraints

- Keep Darjeeling core target-, dataset-, and application-independent. NLU frames, patches, intents, slots, utterances, teachers, residual fills, and field metrics stay under `src/darjeeling/targets/nlu`.
- Keep abstraction tax low. Prefer explicit functions, small data objects, current `FramePatch` and `FrameComposer` paths, static settings, and existing replay/report/objective structures.
- Do not add plugin systems, dependency-injection containers, schema DSLs, or generic patch frameworks.
- Do not directly edit generated task-specific L1/L2/L3 workspaces. Change repo-level harnesses, prompts, adapters, contracts, tests, and artifact generation code.
- Reduce vocabulary for future maintainers. Use ordinary names such as patch, composer, expert, residual L4, audit cost, and focus task.

## Plan

### 1. Make L3 Residual Detection Patch-Aware

Fixes U5.

- Treat a weak layer as having handled part of a request when it accepted either a complete frame or a non-empty patch.
- Use existing `accepted_field_keys`-style logic rather than inventing a new residual abstraction.
- Add a regression test where L2 emits only a patch and L3 residual value evaluation no longer counts that request as a pure residual.

### 2. Let L4 Correct Weak Field Conflicts

Fixes U3.

- Keep weak patches as ordinary patches, but make the L4 path explicit: when L4 is called, it may overwrite weak fields that conflict with the L4 frame.
- Extend `FrameComposer` with a small L4-specific apply/fill method instead of adding provisional/committed patch classes.
- Record conflict and override metadata such as `field_conflicts`, `field_overrides`, and updated `composer_field_sources`.
- Keep lower-layer complete accepts guarded by existing promotion/audit rules when L4 is not called.

### 3. Add A Real Residual L4 Fill/Verify Path

Fixes U2.

- Add an NLU-target residual L4 adapter that receives the utterance, accepted fields, and missing fields.
- The residual path should return only missing or verified fields as a `FramePatch`, with a lower token budget than full L4.
- Keep full L4 fallback for cases where no fields are accepted, residual fill fails, or conflict policy needs a full frame.
- Track separate usage and latency for full versus residual calls: full calls, residual calls, tokens, cost, latency, and fields avoided.
- Avoid changing core runtime contracts unless strictly required; target metadata is enough for reporting.

### 4. Tie Field Metrics To Objective Value

Fixes U4.

- Keep frame exact match, wrong accept rate, serving cost, and p95 latency as the main promotion gates.
- Add field-aware objective terms only where they map to real value: correct weak fields that avoid full L4 work, reduced full L4 calls, and verified residual work.
- Penalize wrong accepted fields and L4 conflicts/overrides.
- Do not reward raw weak field coverage when it does not reduce full L4 work or improve verified correctness.

### 5. Make L2 Expert Bank Less Self-Certifying

Fixes U6 and U7.

- Split expert training evidence into a train slice and a visible validation slice. A simple deterministic chronological or stable request-id split is enough.
- Train intent experts on the train slice and choose thresholds on validation.
- Build slot value tables on train and score accepted accuracy on validation.
- If multiple intent experts fire, accept the highest probability only when the margin is sufficient; otherwise abstain on intent.
- Merge slots only when they do not conflict and, when an intent is accepted, are compatible with that intent's observed slot signature.

### 6. Make L1 Patch-Native

Fixes U8.

- Add an optional `patch` field to the Rust `L1Result` and Python `RustL1Response`.
- Keep backward compatibility: old full-frame L1 results still work and can be adapted into complete patches.
- Allow L1 acceptance when either `frame` or `patch` is present.
- Update the empty native crate, test fixtures, wrapper tests, and L1 agent guidance so L1 programs can emit intent-only or slot-only patches.

### 7. Split Serving, Residual, Audit, And Labeling Costs

Fixes U9.

- Keep objective cost focused on serving behavior.
- Report full serving L4, residual serving L4, audit L4, and teacher-labeling costs separately.
- Surface the same split for calls, tokens, latency, and cost per 100 requests where data exists.
- Preserve raw audit metadata already written by replay and aggregate it in reports rather than changing trace shape unnecessarily.

### 8. Add Field-Level Focus Tasks As Supporting Context

Supports the overall closure loop without widening runtime scope.

- Extend existing focus tasks with field-level opportunities: high fallback fields, high conflict fields, wrong accepted fields, and fields frequently completed by L4 after weak partial accepts.
- Use these as proposal/agent context only. Do not add a new scheduler or task framework.

## Verification

Add or update focused tests for each step, then run:

```bash
uv run pytest tests/targets/nlu/test_patch_runtime.py tests/targets/nlu/test_l2_expert_bank.py tests/targets/nlu/test_l3_residual_gate.py tests/targets/nlu/test_replay_promotion.py tests/targets/nlu/test_l1_rust_worker.py tests/targets/nlu/test_report_l3_summary.py -q
uv run pytest tests/test_core_contracts.py tests/test_target_boundary.py tests/targets/nlu/test_target_core_boundary.py -q
uv run pytest -q
```

Done means:

- L3 residual gate ignores patch-handled requests.
- L4 can correct weak wrong fields when it is invoked.
- Partial patches can reduce full L4 work through a real residual path.
- Objective and reports show the cost/latency value of partial patches.
- L2 experts have visible validation and deterministic conflict handling.
- L1 can emit patches directly.
- Serving, residual, audit, and labeling cost are separately visible.
- Core remains target-independent under boundary tests.

## Agent Prompt

Use this prompt to hand the work to another Codex agent:

```text
Work in /Users/chenmohan/gits/darjeeling. Read and follow docs/design/06_gpt55_pro_0614_refactor_plan.md as the source of truth. Use /Users/chenmohan/Downloads/Darjeeling-research-0614.md and docs/experiments/2026-06-14_gpt55_pro_review_verification.md only as supporting context.

Implement the full plan end to end. Refine the design where necessary, update concise docs, develop code, add focused tests, run verification, debug failures, and iterate until every design goal in the plan is met. Continue solving new issues you uncover without asking for routine decisions.

Hard constraints: keep Darjeeling core target-independent; keep NLU frame/patch/residual/teacher logic inside the NLU target; do not edit generated L1/L2/L3 workspaces directly; avoid plugin systems, DI containers, schema DSLs, generic patch frameworks, and new terminology unless unavoidable. Prefer small explicit functions and existing repo patterns.

Deliverables: patch-aware L3 residual detection; L4 conflict override; real residual L4 fill/verify with separate usage metrics; field-aware objective tied to avoided full L4 work; L2 expert validation split and conflict policy; patch-native L1 Rust/Python contract; serving/residual/audit/labeling cost reporting; field-level focus-task context; tests and docs proving the boundary and behavior.

Before finishing, run the focused tests named in the plan and the full test suite with uv run pytest -q. If any test cannot run, document exactly why and what remains risky.
```
