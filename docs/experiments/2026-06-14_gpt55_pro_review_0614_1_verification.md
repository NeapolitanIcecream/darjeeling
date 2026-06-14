# GPT-5.5-Pro Review 0614-1 Verification Log

Date: 2026-06-14

Source review: `/Users/chenmohan/Downloads/Darjeeling-research-0614-1.md`

Scope: verify the review's concrete claims against the current local repository. This is a working log for intermediate evidence and final verdicts. It intentionally does not edit generated L1/L2/L3 workspaces.

Related historical note: `docs/experiments/2026-06-14_gpt55_pro_review_verification.md` records an older repository state and an older source review. Several findings there are now obsolete.

## Status Key

- `confirmed`: repository evidence supports the claim.
- `partially confirmed`: the claim is directionally right but needs qualification.
- `not confirmed`: repository evidence does not support the claim.

## Claims Under Review

| ID | Claim | Status | Evidence summary |
| --- | --- | --- | --- |
| P1 | L4 residual patch is connected into runtime routing before full L4 fallback. | confirmed | `route_nlu_layers()` calls `_try_l4_residual()` before ordinary L4 `try_answer()` when the composer already has accepted fields. |
| P2 | L4 can override weak-layer wrong fields and records conflicts/verified fields. | confirmed | `FrameComposer.apply_l4_patch()` overwrites conflicting intent/slot values, records `field_conflicts`, `field_overrides`, and updates `verified_fields`. |
| P3 | L1 is patch-native. | confirmed | Rust `FramePatch` and `L1Result.patch` exist; Python `RustL1Response.patch` is accepted by `RustProgramBankLayer`. |
| P4 | L2 expert bank gained validation split and conflict policies. | confirmed | Training uses stable request-id stride validation; runtime applies intent margin abstain, slot-intent signature abstain, and same-slot conflict abstain. |
| P5 | Objective now rewards/penalizes field-level progress and L4 call shape. | confirmed | `ObjectiveMetrics` and `objective_score()` include weak-field progress, residual verified fields, full L4 calls, wrong accepted field rate, and L4 conflict rate. |
| P6 | L3 residual gate is patch-aware. | confirmed | `_is_l2_residual()` checks `accepted_field_keys(frame_patch_from_layer_result(result))`, not only `result.frame`. |
| R1 | Residual L4 live completion may not produce a cache/teacher frame. | confirmed | Live residual returns a patch and does not append cache; replay audits only L0-L3 lower accepts, while residual-completed requests are `chosen_layer == "L4"`. |
| R2 | Offline residual L4 cost/latency includes modeled assumptions. | confirmed | Offline replay uses fixed `RESIDUAL_L4_LATENCY_MS = 300.0` and `RESIDUAL_L4_MIN_COST_FRACTION = 0.25`. |
| R3 | `chosen_layer` can understate weak-layer contribution when residual L4 completes. | confirmed | Runtime returns `chosen_layer="L4"` for residual completion even when weak fields were accepted; field source metadata carries the weak-layer contribution. |
| R4 | Residual teacher verification semantics need tightening. | partially confirmed | Prompt asks for missing/corrected/verification fields and composer records verified fields, but live residual does not reconstruct/cache a teacher frame and current tests explicitly allow no cache append. |
| R5 | L2 validation is still lightweight. | confirmed | Validation is deterministic request-id stride, not intent-stratified, near-negative, or cluster holdout. |
| R6 | Experiment comparison tables should include new field/cost metrics. | partially confirmed | Single-run reports include field and cost summaries, and evolution summary includes full/residual L4 counts. Cross-experiment `comparison.csv` still omits weak-field and residual cost columns. |

## Evidence Log

Initial repository state:

- `git status --short --branch` showed `## main...origin/main` before this document was created.
- Local instructions in `AGENTS.md` confirm NLU frame parsing and NLU diagnostics belong in target code, not Darjeeling core.

### Runtime Patch And Residual L4 Pass

Evidence for P1:

- `src/darjeeling/targets/nlu/patches.py` defines target-owned `FrameComposer`, `NluRouteResult`, and `route_nlu_layers`.
- `route_nlu_layers()` tries `_try_l4_residual()` before normal L4 `try_answer()` when the composer has accepted fields.
- `_try_l4_residual()` sends `utterance`, `accepted_fields`, and `missing_fields` into `try_residual_patch`.
- `src/darjeeling/targets/nlu/target.py` adapts that generic input object back to the legacy NLU L4 residual method.

Evidence for P2:

- `FrameComposer.apply_l4_patch()` handles conflicting `intent` and slot values by recording conflict/override metadata and replacing the existing value with the L4 value.
- `FrameComposer.fill_or_override_from_l4_frame()` converts a full L4 frame into an override patch, including `removed_fields` and `verified_fields`.
- `tests/targets/nlu/test_patch_runtime.py` has a focused test expecting full L4 to override a weak wrong slot.

Evidence for R1:

- `CachedTeacherLayer.try_answer()` appends live full L4 responses to `TeacherCache`.
- `CachedTeacherLayer.try_residual_patch()` live branch returns a `LayerResult` with patch metadata but does not call `self.cache.append()`.
- `tests/targets/nlu/test_l4_teacher.py` explicitly names and asserts `test_live_residual_teacher_call_uses_residual_budget_without_cache_append`.
- `src/darjeeling/targets/nlu/replay.py` calls `_lower_layer_audit()` after routing. `_lower_layer_audit()` treats only `chosen_layer in {"L0", "L1", "L2", "L3"}` as lower-layer accepts.
- Runtime residual completion returns `chosen_layer="L4"`, so no lower-layer audit is triggered. If no cached full teacher frame already exists, `TraceRecord.teacher_frame` remains `None`.

Verdict: the review's highest-priority risk is confirmed. A live residual-completed request can produce a final frame and still be missing a compiler-visible `teacher_frame`.

### Cost, Metrics, And Reporting Pass

Evidence for P5 and R2:

- `src/darjeeling/targets/nlu/compiler/objective.py` includes field-aware metrics and weights: `correct_weak_fields_avoiding_full_l4_per_100`, `residual_l4_verified_fields_per_100`, `full_l4_calls_per_100_requests`, `wrong_accepted_field_rate`, and `l4_conflict_rate`.
- `src/darjeeling/targets/nlu/compiler/replay.py` calculates those metrics and passes them into `ObjectiveMetrics`.
- The same offline replay file defines `RESIDUAL_L4_LATENCY_MS = 300.0` and `RESIDUAL_L4_MIN_COST_FRACTION = 0.25`, so offline residual value remains partly modeled.

Evidence for R3 and R6:

- Runtime residual completion records `composer.field_sources`, `field_conflicts`, `field_overrides`, and `verified_fields` in trace metadata, but `chosen_layer` remains `"L4"`.
- `src/darjeeling/targets/nlu/reports.py` now has a single-run L4 cost summary splitting `serving_full_l4`, `serving_residual_l4`, `audit_l4`, and `teacher_labeling_l4`.
- Evolution summary includes `full_L4/100` and `residual_L4/100`.
- Cross-experiment comparison fieldnames still emphasize frame exact match, latency, layer share, promotions, bottlenecks, and L1 benchmark fields; they do not include weak-field coverage, wrong accepted field rate, full/residual L4 cost, or residual verified fields.

Verdict: the review is correct that layer share is insufficient. The code has improved single-run reporting, but experiment comparison still needs the new field/cost columns.

### L1, L2, And L3 Pass

Evidence for P3:

- `src/darjeeling/targets/nlu/native/l1_empty_programbank/src/frame.rs` defines Rust `FramePatch` and `L1Result.patch`.
- `src/darjeeling/targets/nlu/layers/l1_rust_programbank.py` defines `RustL1Response.patch`.
- `RustProgramBankLayer.try_answer()` marks L1 accepted when the worker returns either a full frame or a patch, and serializes `frame_patch` metadata.

Evidence for P4 and R5:

- `L2ExpertTrainingConfig` includes `validation_fraction` and `intent_conflict_margin`.
- `train_l2_expert_bank()` splits labeled traces into train and validation via `_split_train_validation()`.
- `_split_train_validation()` sorts by request id and selects every `stride`th item for validation.
- `_select_intent_candidate()` abstains on close intent probabilities under the configured margin.
- `L2ExpertBank.try_patch()` skips slots incompatible with the selected intent signature and abstains on same-slot value conflicts.
- Tests cover validation trace counts, intent conflict abstain, and slot-intent signature abstain.

Evidence for P6:

- `src/darjeeling/targets/nlu/compiler/l3_residual_gate.py` imports `accepted_field_keys` and `frame_patch_from_layer_result`.
- `_is_l2_residual()` now excludes traces with any accepted weak-layer field patch from L0/L1/L2.

Verdict: the review's positive implementation claims are confirmed. The remaining L2 split concern is also confirmed as a limitation rather than a correctness bug.

## Current Recommendation

Before rerunning expensive end-to-end experiments, fix R1 or make the intended semantics explicit:

1. If residual L4 complete is trusted as teacher labeling, reconstruct `composer.to_frame()` and persist it as a cache/trace teacher frame with source metadata such as `teacher_source="residual_live"`.
2. If residual L4 is only a serving optimization, trigger a full L4 audit for residual-completed requests before using traces for compiler training.
3. Add regression coverage for replay traces where L1/L2 accept fields and live residual L4 completes without an existing cache entry.

After R1 is resolved, rerun the main evolution and end-to-end comparison experiments. Old experiments are not directly comparable because current routing, objective metrics, L1 contract, L2 policy, and L4 residual accounting have materially changed.

## Test Log

Focused test command:

```bash
uv run pytest tests/targets/nlu/test_patch_runtime.py tests/targets/nlu/test_l4_teacher.py tests/targets/nlu/test_l2_expert_bank.py tests/targets/nlu/test_l3_residual_gate.py tests/targets/nlu/test_replay_promotion.py tests/targets/nlu/test_report_l3_summary.py tests/targets/nlu/test_l1_rust_worker.py -q
```

Result:

- `55 passed in 2.93s`

Boundary test command:

```bash
uv run pytest tests/test_core_contracts.py tests/test_target_boundary.py tests/targets/nlu/test_target_core_boundary.py -q
```

Result:

- `16 passed in 0.87s`
