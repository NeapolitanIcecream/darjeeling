# GPT-5.5-Pro Updated Review Verification Log

Date: 2026-06-14

Source review: `/Users/chenmohan/.codex/attachments/72430fcd-db56-4534-a32f-ab2bade88bde/pasted-text.txt`

Scope: verify the updated review's concrete design claims against the local repository. This is a working log for intermediate evidence and final verdicts. It intentionally does not modify generated L1/L2/L3 workspaces.

## Status Key

- `pending`: not checked yet.
- `confirmed`: repository evidence supports the claim.
- `partially confirmed`: the claim is directionally right but needs qualification.
- `not confirmed`: repository evidence does not support the claim.

## Claims Under Review

| ID | Claim | Status | Notes |
| --- | --- | --- | --- |
| U1 | Partial/component-wise runtime now lives in the NLU target and replay uses it. | confirmed | `FramePatch`, `FrameComposer`, `route_nlu_layers`, runtime replay, and offline replay are NLU-target code. |
| U2 | Partial patches may not reduce L4 cost/latency because missing fields still call full L4. | confirmed | L4 is still invoked as a normal runtime layer; "residual fill" is currently composer-side extraction from a full L4 frame. |
| U3 | `FrameComposer` is first-writer-wins, so L4 fill cannot correct weak wrong fields. | confirmed | Intent and slots are only written when absent; L4 fill explicitly skips existing weak fields. |
| U4 | Objective metrics do not yet reward field-level progress or penalize wrong accepted fields. | confirmed | Field metrics are calculated and reported, but `ObjectiveMetrics` and `objective_score` do not include them. |
| U5 | L3 residual gate is not patch-aware and may treat partial L2 accepts as residuals. | confirmed | `_is_l2_residual` only treats accepted weak results with `frame is not None` as non-residual. |
| U6 | L2 expert bank thresholding/metrics are mostly train-slice self-evaluation. | confirmed | Intent classifier is fit and thresholded on the same trace slice; slot metrics also evaluate the same trace slice used to build value tables. |
| U7 | L2 expert bank conflict handling is weak: multiple intents are first-writer-ish and slots merge directly. | confirmed | `accepted_intent = accepted_intent or ...`; slots use `dict.update()` with no margin/signature compatibility policy. |
| U8 | L1 Rust worker is not patch-native; patch adaptation is only metadata/wrapper-side. | confirmed | Rust and Python L1 response schemas expose `frame`, not `patch`; accepted requires `response.frame is not None`. |
| U9 | Audit cost should be separated from serving cost in settings/reports. | partially confirmed | Audit metadata records live audit latency/cost, but reports/objective do not separately aggregate serving vs audit costs. |

## Evidence Log

Initial repository state:

- `git status --short` was clean before this document was created.
- The older `docs/experiments/2026-06-13_gpt55_review_verification.md` covers a previous GPT-5.5 review and is left unchanged.

Next steps:

1. Inspect NLU schemas, patch composer, target runtime routing, and replay to verify U1-U3 and U9.
2. Inspect objective/replay metrics and L3 residual gate to verify U4-U5.
3. Inspect L2 expert bank training/conflict logic and L1 Rust worker schema/wrapper to verify U6-U8.
4. Run focused tests for patch runtime, L2 expert bank, L3 residual gate, replay/reporting, and L1 Rust worker.

### 2026-06-14 Patch Runtime, L4 Fill, And Composer Pass

Evidence for U1:

- `src/darjeeling/targets/nlu/schemas.py` defines target-owned `FramePatch`.
- `src/darjeeling/targets/nlu/patches.py:12` defines target-owned `FrameComposer`; `src/darjeeling/targets/nlu/patches.py:74` defines target-owned `route_nlu_layers`.
- `src/darjeeling/targets/nlu/replay.py:86` calls `route_nlu_layers` in serving replay and records `composer_field_sources` in trace metadata.
- `src/darjeeling/targets/nlu/compiler/replay.py:498` uses the same composer approach for offline L4 fallback.

Verdict for U1: confirmed. The component-wise runtime exists in the NLU target and core contracts stay generic.

Core/target boundary spot check:

- `src/darjeeling/contracts.py` exposes generic `JsonObject`, `LayerResult.output`, `TraceRecord.input/final_output`, and target protocols; it does not define NLU frames, intents, slots, utterances, or dataset concepts.
- `tests/targets/nlu/test_target_core_boundary.py` scans non-target source for NLU schema terms and expects no offenders.
- `tests/test_target_boundary.py` maintains stricter target-boundary checks and dataset-independent core defaults.

Boundary test command:

```bash
uv run pytest tests/test_core_contracts.py tests/test_target_boundary.py tests/targets/nlu/test_target_core_boundary.py -q
```

Boundary test result:

- `16 passed in 0.83s`

Evidence for U2:

- `src/darjeeling/targets/nlu/patches.py:82` calls each layer's ordinary `try_answer({"utterance": utterance})`; L4 receives the same full request shape.
- `src/darjeeling/targets/nlu/patches.py:85` only after `result.frame is not None` calls `composer.fill_missing_from_frame(...)` with metadata `adapter: l4_residual_fill`.
- `src/darjeeling/targets/nlu/compiler/replay.py:498` does the same in offline replay, using the full `fallback_frame` and composer extraction.
- No separate residual teacher prompt, residual max-token setting, residual L4 usage bucket, or missing-field request object was found.

Verdict for U2: confirmed. Current "residual fill" avoids overwriting already accepted fields, but it does not yet reduce the full L4 call itself.

Evidence for U3:

- `src/darjeeling/targets/nlu/patches.py:20` only writes `accepted_intent` when the composer has no intent yet.
- `src/darjeeling/targets/nlu/patches.py:24` only writes a slot when that slot key is not already present.
- `src/darjeeling/targets/nlu/patches.py:41` builds L4 accepted slots by filtering out existing composer slots.
- `tests/targets/nlu/test_patch_runtime.py` explicitly expects L4 to add `slot_beta` while preserving a lower-layer wrong `slot_alpha`.

Verdict for U3: confirmed. Stronger layers fill missing fields only; they do not override conflicting weak fields.

### 2026-06-14 Objective And L3 Gate Pass

Evidence for U4:

- `src/darjeeling/targets/nlu/compiler/replay.py:235` through `src/darjeeling/targets/nlu/compiler/replay.py:361` calculates field metrics, including `weak_field_coverage`, `weak_field_accuracy`, and `wrong_accepted_field_rate`.
- `src/darjeeling/targets/nlu/reports.py:621` through `src/darjeeling/targets/nlu/reports.py:705` exports the same field metrics in reports.
- `src/darjeeling/targets/nlu/compiler/objective.py:16` defines `ObjectiveMetrics` without field-level metrics.
- `src/darjeeling/targets/nlu/compiler/objective.py:24` scores only frame exact match, wrong accept rate, cost, p95 latency, and artifact complexity.

Verdict for U4: confirmed. Field-level progress is observable, but it is not part of the promotion objective.

Evidence for U5:

- `src/darjeeling/targets/nlu/compiler/l3_residual_gate.py:91` defines `_is_l2_residual`.
- `src/darjeeling/targets/nlu/compiler/l3_residual_gate.py:92` only excludes traces where a weak result is accepted and has `result.frame is not None`.
- Partial expert-bank accepts are accepted `LayerResult` objects with `patch` and no `frame`, so they remain eligible as L3 residuals under this predicate.

Verdict for U5: confirmed. This is the smallest direct bug in the updated review.

### 2026-06-14 L2 Expert Bank Pass

Evidence for U6:

- `src/darjeeling/targets/nlu/layers/l2_experts.py:214` collects all labeled traces into one `labeled` slice.
- `src/darjeeling/targets/nlu/layers/l2_experts.py:309` fits the vectorizer on the same traces; `src/darjeeling/targets/nlu/layers/l2_experts.py:316` fits the classifier on that matrix.
- `src/darjeeling/targets/nlu/layers/l2_experts.py:317` then computes probabilities on the same matrix, and `src/darjeeling/targets/nlu/layers/l2_experts.py:318` selects thresholds from those same probabilities.
- `src/darjeeling/targets/nlu/layers/l2_experts.py:340` builds slot value tables from the trace slice and `src/darjeeling/targets/nlu/layers/l2_experts.py:354` evaluates metrics on that same slice.
- `src/darjeeling/targets/nlu/compiler/loop.py:819` builds the expert-bank candidate directly from the compiler training traces.

Verdict for U6: confirmed, with the review's qualification. Later promotion holdout can still catch some failures, but each expert's own selection metrics are train-slice metrics.

Evidence for U7:

- `src/darjeeling/targets/nlu/layers/l2_experts.py:134` iterates all intent experts.
- `src/darjeeling/targets/nlu/layers/l2_experts.py:139` uses `accepted_intent = accepted_intent or patch.accepted_intent`, so the first fired intent wins regardless of later probabilities.
- `src/darjeeling/targets/nlu/layers/l2_experts.py:142` iterates slot experts and `src/darjeeling/targets/nlu/layers/l2_experts.py:147` directly merges accepted slots with `accepted_slots.update(...)`.
- No probability margin, multi-intent conflict abstain, or accepted-intent/slot-signature compatibility check was found.

Verdict for U7: confirmed.

### 2026-06-14 L1 And Audit Reporting Pass

Evidence for U8:

- `src/darjeeling/targets/nlu/native/l1_empty_programbank/src/frame.rs:23` defines `L1Result`.
- `src/darjeeling/targets/nlu/native/l1_empty_programbank/src/frame.rs:27` exposes `frame: Option<Frame>` and no patch field.
- `src/darjeeling/targets/nlu/layers/l1_rust_programbank.py:33` defines the Python `RustL1Response` with `frame: Frame | None` and no patch field.
- `src/darjeeling/targets/nlu/layers/l1_rust_programbank.py:160` only marks L1 accepted when `response.accepted and response.frame is not None`.

Verdict for U8: confirmed.

Evidence for U9:

- `src/darjeeling/targets/nlu/settings.py:40` defaults `lower_layer_audit_mode` to `always`.
- `src/darjeeling/targets/nlu/replay.py:390` through `src/darjeeling/targets/nlu/replay.py:395` records live audit frame, source, disagreement, latency, and cost in trace metadata.
- `src/darjeeling/targets/nlu/reports.py:520` through `src/darjeeling/targets/nlu/reports.py:570` aggregates layer cost from `LayerResult.cost_usd`, not the trace metadata audit cost.
- `src/darjeeling/targets/nlu/reports.py:971` summarizes evolution as L4 calls, cost, p95, and frame EM, with no separate serving/audit/labeling cost columns.

Verdict for U9: partially confirmed. The raw metadata exists, so the system can separate audit costs later; the current report/objective presentation does not do that separation.

### 2026-06-14 Focused Verification Tests

Command:

```bash
uv run pytest tests/targets/nlu/test_patch_runtime.py tests/targets/nlu/test_l2_expert_bank.py tests/targets/nlu/test_l3_residual_gate.py tests/targets/nlu/test_replay_promotion.py tests/targets/nlu/test_l1_rust_worker.py tests/targets/nlu/test_report_l3_summary.py -q
```

Result:

- `39 passed in 3.97s`

What this validates:

- Patch runtime and composer behavior remain internally consistent with the current tests.
- Lower-layer audit metadata is recorded on replay.
- Offline replay field metrics, promotion replay, L2 expert-bank artifact creation, L3 residual gate behavior, L1 Rust worker integration, and report metric exports all pass their current test coverage.

Important qualification:

- Passing tests do not invalidate U2, U3, U4, U5, U6, U7, U8, or U9. Several tests intentionally codify the existing behavior being reviewed, especially composer fill-only semantics and train-slice expert selection.

## Final Verification Summary

The updated GPT-5.5-Pro review is mostly accurate against the local repository.

Confirmed positive direction:

- Core/target separation remains intact.
- Partial/component-wise patch routing has moved into the NLU target.
- Replay records composer field sources and lower-layer audit metadata.
- L2 expert bank and L3 residual value gate exist as target-level mechanisms.

Confirmed risks:

1. `U5` is the smallest direct bug: L3 residual gate is not patch-aware.
2. `U3` is the highest correctness risk: weak accepted fields cannot be corrected by L4 under current composer semantics.
3. `U2` is the highest cost/latency ROI risk: current L4 "residual fill" still depends on a full L4 frame call.
4. `U4` means field-level progress is visible but not optimized by the promotion objective.
5. `U6` and `U7` mean the L2 expert bank needs stronger validation and conflict policy before wider coverage is safe.
6. `U8` means L1 cannot yet naturally emit intent-only or slot-only patches.
7. `U9` is a reporting/accounting gap: audit metadata exists but serving cost and audit cost are not separately summarized.

Recommended order if this turns into implementation work:

1. Make L3 residual detection patch-aware.
2. Add an explicit L4 override/conflict policy for weak provisional fields.
3. Split full L4 calls from true residual L4 fill/verify calls and record separate usage.
4. Add field-aware objective terms only once they correspond to avoided full L4 work or verified correctness.
5. Add L2 expert validation split and conflict/margin handling.
6. Extend L1 Rust/Python worker contract to emit `FramePatch`.
7. Add report columns for serving L4 calls/cost and audit L4 calls/cost.
