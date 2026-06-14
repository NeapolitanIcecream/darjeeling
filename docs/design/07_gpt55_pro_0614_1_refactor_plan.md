# GPT-5.5-Pro 0614-1 Refactor Plan

Date: 2026-06-14

Inputs:

- `/Users/chenmohan/Downloads/Darjeeling-research-0614-1.md`
- `docs/experiments/2026-06-14_gpt55_pro_review_0614_1_verification.md`

Goal: fix the remaining issues confirmed by the 0614-1 verification pass before rerunning expensive end-to-end experiments. The current patch runtime, L4 override path, L1 patch contract, L2 expert bank, field-aware objective, and patch-aware L3 gate are real progress; do not redesign them from scratch.

## Constraints

- Keep Darjeeling core target-, dataset-, and application-independent.
- Keep NLU concepts under `src/darjeeling/targets/nlu`: frames, patches, intents, slots, utterances, residual L4, teacher prompts, field metrics, and NLU diagnostics.
- Keep abstraction tax low. Prefer existing `FramePatch`, `FrameComposer`, trace metadata, settings, reports, and small helper functions.
- Do not add plugin systems, dependency-injection containers, schema DSLs, or a generic patch framework.
- Do not directly edit generated L1/L2/L3 workspaces. Change repo-level harnesses, prompts, adapters, contracts, tests, and docs.
- Minimize new terms. Prefer plain names: residual L4, full L4, verified fields, weak fields, audit cost, serving cost.

## Non-Goals

- Do not move NLU schema knowledge into Darjeeling core.
- Do not redefine `chosen_layer` globally. Treat it as the layer that completed the route; add contribution metrics instead.
- Do not replace the L2 expert bank with a new training framework.
- Do not require cluster holdout or complex data mining in this pass.
- Do not rerun the full expensive experiment suite until residual teacher-frame semantics are fixed and tests are green.

## Plan

### 1. Fix Residual L4 Teacher-Frame Semantics

Fixes verified issue R1.

- Define a simple target-level rule for residual completion:
  - residual L4 may complete serving only when the final frame can be reconstructed;
  - residual L4 may become `teacher_frame` only when weak accepted fields are either verified, corrected, or removed by the residual patch;
  - otherwise replay must trigger full L4 audit before treating the trace as teacher-labeled.
- Keep this in NLU target code. A small helper around `FrameComposer`, residual patch metadata, and replay audit is enough.
- For cache-backed residual L4, derive verification from the cached full teacher frame as today.
- For live residual L4, either persist the reconstructed final frame with metadata such as `teacher_source="residual_live"` when verification is complete, or immediately full-audit when verification is incomplete.
- Add a regression test where L1/L2 accept fields, live residual L4 completes, no full cache exists, and the resulting trace still has a valid `teacher_frame` or an explicit full-audit reason.
- Update the existing test that currently asserts live residual calls do not append cache.

### 2. Tighten Residual Verification Semantics

Fixes verified issue R4.

- Update the residual teacher prompt to make `complete=true` mean: every previously accepted field is verified, corrected, or removed, and every missing required field needed for a complete frame is supplied.
- Keep using `FramePatch.metadata["verified_fields"]` and `metadata["removed_fields"]`; do not add a new schema unless tests show metadata is insufficient.
- Add validation that residual patches cannot silently ignore accepted weak fields when they claim completion.
- Record counts for residual completions without full verification, conflicts, overrides, removed fields, and verified fields.
- Prefer falling back to full L4 over accepting an unverified residual completion.

### 3. Separate Measured And Modeled Residual Cost

Fixes verified issue R2.

- Keep current live serving metrics for actual full/residual L4 calls: calls, tokens, cost, and latency.
- In offline replay and promotion metrics, label residual cost/latency estimates as modeled when they come from constants such as `RESIDUAL_L4_LATENCY_MS` or `RESIDUAL_L4_MIN_COST_FRACTION`.
- Add report fields that make the distinction visible without adding a new reporting layer:
  - measured full L4 tokens/latency/cost where live metadata exists;
  - measured residual L4 tokens/latency/cost where live metadata exists;
  - modeled residual L4 latency/cost in offline replay.
- Update docs/report text so experiment readers do not confuse modeled offline savings with measured live savings.

### 4. Show Weak-Layer Contribution Without Changing `chosen_layer`

Fixes verified issue R3.

- Keep `chosen_layer="L4"` when residual L4 completes the final frame.
- Add or surface contribution metrics based on existing composer metadata:
  - weak field coverage;
  - weak field accuracy;
  - correct weak fields avoiding full L4 per 100 requests;
  - residual L4 verified fields per 100 requests;
  - L4 conflict/override rate;
  - field source counts by layer.
- Ensure summary tables and CSVs make these metrics easier to find than raw layer share when evaluating patch runtime value.

### 5. Add Field And Residual Columns To Experiment Comparison

Fixes verified issue R6.

- Extend `comparison.csv` and comparison HTML with the key metrics needed after this refactor:
  - `weak_field_coverage`;
  - `weak_field_accuracy`;
  - `wrong_accepted_field_rate`;
  - `l4_conflict_rate`;
  - `full_l4_calls_per_100`;
  - `residual_l4_calls_per_100`;
  - `full_l4_tokens_per_100`;
  - `residual_l4_tokens_per_100`;
  - `serving_cost_per_100`;
  - `audit_cost_per_100`;
  - `correct_weak_fields_avoiding_full_l4_per_100`;
  - `residual_l4_verified_fields_per_100`.
- Source these from existing run reports, traces, promotion records, or direct trace aggregation. Prefer direct trace aggregation when it is simpler and stable.
- Keep layer share columns, but do not use them as the primary signal for patch-runtime value.

### 6. Strengthen L2 Expert Validation Modestly

Fixes verified issue R5.

- Replace pure stable request-id stride validation with a deterministic intent-stratified split when enough labeled examples exist.
- Keep a simple fallback to the current stable split for tiny datasets.
- Ensure every selected intent expert is validated with visible positive and negative examples when available.
- Ensure slot experts are validated on held-out examples and report accepted accuracy and wrong accepts.
- Keep conflict handling local and explicit: intent margin abstain, slot-intent signature abstain, and same-slot value conflict abstain.
- Do not add clustering, a new data splitter framework, or a new model family in this pass.

### 7. Add Focused Tests And Docs

- Add regression tests around each fixed issue:
  - residual live completion produces teacher labeling or full-audit metadata;
  - unverified residual completion cannot silently become trusted teacher data;
  - offline metrics identify modeled residual assumptions;
  - comparison CSV includes field and residual metrics;
  - `chosen_layer` remains L4 while weak contribution metrics are present;
  - L2 validation split is deterministic and stratified when data supports it.
- Update the relevant design/report docs concisely. Prefer one short section explaining residual L4 teacher-frame semantics and one short section explaining measured versus modeled residual metrics.
- Keep `docs/experiments/2026-06-14_gpt55_pro_review_0614_1_verification.md` as evidence, not as the implementation plan.

### 8. Rerun Readiness And Experiment Plan

- Do not rerun the full suite before steps 1-7 pass focused tests and boundary tests.
- After fixes, rerun at least a small cached/smoke experiment to verify reports and comparison output.
- When live teacher/cache/data are available, rerun:
  - main evolution;
  - Zipf mild, Zipf heavy, and uniform locality;
  - no guard / no audit;
  - no L2;
  - L2 global student versus expert bank;
  - L3 enabled / skipped / learned gate.
- Report old versus new results as not directly comparable unless the report includes the new field and residual metrics.

## Verification Commands

Run focused tests first:

```bash
uv run pytest tests/targets/nlu/test_patch_runtime.py tests/targets/nlu/test_l4_teacher.py tests/targets/nlu/test_l2_expert_bank.py tests/targets/nlu/test_l3_residual_gate.py tests/targets/nlu/test_replay_promotion.py tests/targets/nlu/test_report_l3_summary.py tests/targets/nlu/test_l1_rust_worker.py -q
```

Run boundary tests:

```bash
uv run pytest tests/test_core_contracts.py tests/test_target_boundary.py tests/targets/nlu/test_target_core_boundary.py -q
```

Before final handoff, run:

```bash
uv run pytest -q
```

## Done Criteria

- Residual live L4 completion no longer creates unlabeled compiler-visible traces by accident.
- Residual completion has clear verification semantics and safe full-L4 fallback.
- Reports distinguish measured live residual value from modeled offline residual value.
- Weak-layer contribution is visible without changing global `chosen_layer` meaning.
- Experiment comparison includes field, residual L4, serving cost, and audit cost metrics.
- L2 expert validation is deterministic, modestly stronger, and still easy to understand.
- Target/core boundary tests remain green.
- The implementation uses existing NLU target objects and metadata rather than a new framework.
