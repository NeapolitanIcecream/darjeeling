# GPT-5.5-Pro Review Verification Log

Date: 2026-06-13

Source review: `/Users/chenmohan/Downloads/Darjeeling-research-0613.md`

Scope: verify the review's major design claims against the local repository. This is a working log; it records intermediate evidence, not just final conclusions.

## Status Key

- `pending`: not checked yet.
- `confirmed`: the repository evidence supports the claim.
- `partially confirmed`: the claim is directionally right but needs qualification.
- `not confirmed`: the repository evidence does not support the claim.

## Claims Under Review

| ID | Claim | Status | Notes |
| --- | --- | --- | --- |
| C1 | Runtime is a whole-frame cascade, not component-wise routing/composition. | confirmed | Core `CascadeRouter` returns the first accepted full output. |
| C2 | NLU traces and schema record full frames rather than field-level accepted patches. | confirmed | NLU `Frame`, `LayerResult.frame`, and `TraceRecord.final_frame` are whole-frame objects. |
| C3 | Lower-layer accepted samples may lack teacher audit labels in replay. | confirmed | Runtime replay only reads `TeacherCache`; it does not force a shadow L4 audit for lower-layer accepts. |
| C4 | L2 target evolution exists but is not a first-class path in the main compiler loop. | confirmed | Main loop does not call `run_l2_target_evolution`; CLI can stage/promote it separately. |
| C5 | L2 remains primarily a global student rather than a micro-expert bank. | confirmed | One bundle owns one intent pipeline, slot tagger, guard, and retrieval index. |
| C6 | L3 is structurally expensive/fragile as a runtime tier and lacks viability gating. | partially confirmed | Structure/defaults match the risk claim; repo does have explicit L3 replay/promotion gates, but not a residual cost/latency viability gate in the main loop. |
| C7 | L4 proposal context is still too chronological/global and not bounty-driven. | confirmed | Generic proposal context sorts teacher traces by `request_id` and truncates to 50; L1 has richer family context separately. |
| C8 | `run_compiler_generation` is taking on too many responsibilities. | confirmed | The function spans most of `loop.py` and handles hard buffers, L0, L2 config/tuning/guard/train, L3 candidates, L1 agent, replay, promotion, and reports. |

## Evidence Log

Initial repository state:

- `git status --short` was clean before this document was created.
- The source review has 526 lines.

Next steps:

1. Inspect runtime contracts, router, NLU schemas, and trace/replay code for C1-C3.
2. Inspect compiler loop and target evolution code for C4, C6, and C8.
3. Inspect L2 student/target architecture and L4 context construction for C5 and C7.
4. Run focused tests if available for replay, compiler loop, L2 target evolution, and L3.

### 2026-06-13 Runtime And Replay Pass

Evidence for C1:

- `src/darjeeling/contracts.py:20` defines a core `LayerResult` with one `accepted` flag and one full `output: JsonObject | None`.
- `src/darjeeling/runtime/router.py:16` routes through layers in order and returns immediately at `result.accepted and result.output is not None`.
- `src/darjeeling/targets/nlu/replay.py:84` calls the core router once per utterance and validates the returned object as a complete NLU `Frame`.
- `src/darjeeling/targets/nlu/compiler/replay.py:369` has a parallel offline path with the same whole-frame semantics: L0/L1/L2/L3 return a complete `Frame`, otherwise L4 fallback supplies the complete teacher frame.

Verdict for C1: confirmed. There is no component-wise accept, patch merge, or composer in the runtime path.

Evidence for C2:

- `src/darjeeling/targets/nlu/schemas.py:10` defines `Frame(intent, slots, is_abstain)`.
- `src/darjeeling/targets/nlu/schemas.py:19` defines legacy NLU `LayerResult.frame: Frame | None`.
- `src/darjeeling/targets/nlu/schemas.py:30` defines `TraceRecord` with `teacher_frame`, `final_frame`, and full-frame `layer_results`.
- `src/darjeeling/targets/nlu/target.py:331` converts legacy layer results into core `output` by dumping a full frame.

Verdict for C2: confirmed. Traces record whole-frame results, not field-level accepted/rejected fields.

Evidence for C3:

- `src/darjeeling/targets/nlu/replay.py:59` loads `TeacherCache`.
- `src/darjeeling/targets/nlu/replay.py:84` routes the request through lower layers and L4 as needed.
- `src/darjeeling/targets/nlu/replay.py:90` sets `teacher_frame = teacher_cache.get(utterance)` after routing.
- There is no branch that calls L4 just because L0/L1/L2/L3 accepted. Therefore accepted lower-layer traces only get `teacher_frame` when the utterance was already in cache.

Verdict for C3: confirmed with qualification. The problem is missing forced/probabilistic shadow audit, not that lower-layer accepted traces can never have teacher labels.

### 2026-06-13 Compiler, L2, L3, And Context Pass

Evidence for C4:

- `src/darjeeling/targets/nlu/compiler/l2_target_evolution.py:78` defines `run_l2_target_evolution` with `dry-run`, `local-search`, `codex-cli`, and `agent-session` modes.
- `src/darjeeling/targets/nlu/main_cli.py:947` invokes that function from the `l2 target-evolve` CLI path.
- `src/darjeeling/targets/nlu/main_cli.py:1073` and `src/darjeeling/targets/nlu/main_cli.py:1305` provide separate `promote-target` and `replay-target` CLI paths for staged L2 target artifacts.
- `src/darjeeling/targets/nlu/compiler/loop.py` has no import or call of `run_l2_target_evolution`.
- `src/darjeeling/targets/nlu/compiler/loop.py:458` explicitly drops an existing `l2_target` artifact when it retrains the L2 bundle without target-aware adoption.
- `src/darjeeling/targets/nlu/compiler/loop.py:595` builds the candidate offline artifact set with `l2_bundle` but no `l2_target_path`, so main-loop replay cannot evaluate a fresh L2 target candidate.
- `tests/targets/nlu/test_compiler_loop.py:405` asserts this drop behavior.

Verdict for C4: confirmed. The review should be read precisely: L2 target evolution is not absent, but it is an adjacent CLI lifecycle rather than a first-class source in `run_compiler_generation`.

Evidence for C5:

- `src/darjeeling/targets/nlu/layers/l2_student.py:35` defines one global `L2StudentConfig` with global model-family and feature settings.
- `src/darjeeling/targets/nlu/layers/l2_student.py:422` defines `L2StudentBundle` with one intent pipeline, optional slot tagger, one guard model, and shared support/retrieval indices.
- `src/darjeeling/targets/nlu/layers/l2_student.py:641` trains one runtime intent pipeline and one runtime slot tagger over all selected examples.
- `src/darjeeling/targets/nlu/layers/l2_student.py:550` uses a single `L2StudentLayer` with a single guard threshold for runtime acceptance.
- `src/darjeeling/targets/nlu/layers/l2_target.py:13` wraps that global bundle with target-owned postprocess/veto hooks. It does not route among independent per-intent or per-slot experts.
- `src/darjeeling/targets/nlu/compiler/l2_distiller.py:17` and `src/darjeeling/targets/nlu/compiler/l2_tuner.py:292` tune global config fields such as intent model family, slot model family, frame source, and n-grams.

Verdict for C5: confirmed with nuance. L2 already has useful per-intent/signature calibration features and target postprocess/veto code, but its runtime shape is still a single global student, not a micro-expert bank.

Evidence for C6:

- `src/darjeeling/targets/nlu/settings.py:49` and `src/darjeeling/targets/nlu/layers/l3_local_slm.py:44` default to `Qwen/Qwen2.5-0.5B-Instruct`.
- `src/darjeeling/targets/nlu/settings.py:61` and `src/darjeeling/targets/nlu/layers/l3_local_slm.py:48` default `max_new_tokens` to 256.
- `src/darjeeling/targets/nlu/layers/l3_local_slm.py:62` renders a prompt containing the output schema, full intent list, full slot list, few-shot examples, and current utterance.
- `src/darjeeling/targets/nlu/layers/l3_local_slm.py:128` uses `AutoModelForCausalLM.generate(...)` with free text generation, then `src/darjeeling/targets/nlu/layers/l3_local_slm.py:328` parses/repairs JSON.
- `src/darjeeling/targets/nlu/layers/l3_local_slm.py:271` gates acceptance on model-reported confidence and schema validation.
- `src/darjeeling/targets/nlu/compiler/loop.py:502` records L3 prompt candidates as not runtime-promoted until regenerated/shadow replay.
- `src/darjeeling/targets/nlu/main_cli.py:630` validates explicit L3 prompt replay before promotion using request count, nonzero would-accept count, accepted accuracy, and wrong-accept rate.
- `src/darjeeling/targets/nlu/main_cli.py:2087` preflights L3 benchmark presence/status when L3 mode is enabled.
- However, neither `run_compiler_generation` nor `l3_promote_prompt` checks whether L3 improves expected cost/latency on the L2 residual. The main offline replay path only reuses recorded L3 accepts from traces, not a regenerated L3 candidate.

Verdict for C6: partially confirmed. The structural risk claim is supported. The "lacks viability gating" claim needs narrowing: the repo has explicit accuracy/wrong-accept gates for L3 prompt promotion, but not the residual cost/latency viability gate proposed by the review.

Evidence for C7:

- `src/darjeeling/targets/nlu/compiler/l4_context.py:97` defines generic proposal context for L2 config, guard, and L3 prompt proposals.
- `src/darjeeling/targets/nlu/compiler/l4_context.py:121` filters teacher-labeled traces, sorts by `request_id`, and truncates to `max_dynamic_traces=50`.
- `src/darjeeling/targets/nlu/compiler/l4_context.py:125` sends `current_artifact_summary`, `metrics`, and raw `teacher_traces`, not a ranked bounty/hot-cluster task package.
- `src/darjeeling/targets/nlu/compiler/l1_program_compiler.py:323` separately builds `context_families.json`, grouped by teacher intent and slot signature, prioritized by hard-case support and family size.
- `src/darjeeling/targets/nlu/compiler/l2_target_evolution.py` also has family diagnostics/backlogs, but those are part of the separate target-evolution workspace, not the generic `L4ProposalAdapter` context.

Verdict for C7: confirmed. The review's contrast between generic proposal context and richer L1 family context is accurate.

Evidence for C8:

- `src/darjeeling/targets/nlu/compiler/loop.py` has 1007 lines.
- `run_compiler_generation` spans `src/darjeeling/targets/nlu/compiler/loop.py:106` through `src/darjeeling/targets/nlu/compiler/loop.py:715`.
- Inside that one function it handles teacher-visible conversion, artifact store loading, split selection, hard-buffer merge/write, L0 exact cache, L2 proposal, guard proposal, Optuna tuning, L2 training, guard calibration, L3 prompt candidate creation, L1 agent invocation, L1 benchmark, offline candidate assembly, replay, promotion decisions, manifest creation, CSV metrics, and promotion JSON.
- The helper list after line 718 is mostly low-level support; the high-level candidate-source steps are not split into separate `build_l0_candidate`, `evolve_l1_candidate`, `train_l2_student_candidate`, etc.

Verdict for C8: confirmed. The function is workable but has become the orchestration monolith described in the review.

### 2026-06-13 Focused Verification Tests

Command:

```bash
uv run pytest tests/targets/nlu/test_replay_runtime.py tests/targets/nlu/test_compiler_loop.py tests/targets/nlu/test_l2_target_evolution.py tests/targets/nlu/test_l3_local_slm.py tests/targets/nlu/test_l4_context.py -q
```

Result:

- `74 passed in 37.43s`

What this validates:

- Runtime replay behavior, including manifest loading for L1/L2/L2 target/L3 prompt artifacts.
- Compiler-loop behavior, including L2 target drop-on-retrain and L3 prompt candidate non-promotion.
- L2 target evolution workspace, split, private holdout, local-search, and agent-session scaffolding behavior.
- L3 local SLM parsing/gating/benchmark behavior with test backends.
- Generic L4 proposal context construction and gold-label exclusion.
