# GPT-5.5-Pro Review 0622 Verification Log

Date: 2026-06-22

Source review: `/Users/chenmohan/Downloads/Darjeeling-research-0622.md`

Scope: verify the review's concrete claims against the local repository and the committed `post_refactor_fixed_20260615` review packet. This is a working log for intermediate evidence and final verdicts. It intentionally does not edit generated L1/L2/L3 workspaces.

## Status Key

- `pending`: not checked yet.
- `confirmed`: repository or packet evidence supports the claim.
- `partially confirmed`: the claim is directionally right but needs qualification.
- `not confirmed`: repository or packet evidence does not support the claim.

## Claims Under Review

| ID | Claim | Status | Evidence summary |
| --- | --- | --- | --- |
| Q1 | `cache-full/main-evolution` is strong under cache-backed replay: frame exact match is about 0.998, L0 handles about half the requests, L2 contributes very little, and residual L4 is used only a little. | confirmed | `cache-full/comparison.csv` and `main-evolution/quality.json` match the review's numbers. |
| Q2 | `cache-full` is not evidence of live serving cost/latency because teacher/cache paths report zero serving token/cost and cache-like L4 latency. | confirmed | `cache-full/main-evolution` reports zero serving tokens/cost and sub-millisecond cached L4 p95 buckets. |
| Q3 | `live-residual-500` is fundamentally all-L4: all four live runs have L4 share 1.0, weak-field coverage 0, full L4 calls 100/100, residual L4 calls 0, and frame exact match only about 0.696-0.720. | confirmed | `live-residual-500/comparison.csv` exactly matches this pattern. |
| Q4 | In the live runs, the weak layers did not degrade quality because they did not take over; live L4 itself is only about 70%-72% frame exact match against gold. | confirmed | Layer counts are all L4, and L4 accepted accuracy/forced global accuracy equals the gold frame exact match range. |
| Q5 | The live run settings mean this was not a full L4/coding-agent evolution run: proposal/evolution modes and L3 were disabled. | confirmed | `settings.sanitized.json` has `l4_proposal_mode=disabled`, `l1_agent_mode=disabled`, `l2_target_evolution_mode=disabled`, `local_slm_mode=disabled`. |
| Q6 | The live run reports a split between teacher-replay objective and gold evaluation: current all-L4 objective is 1.0 while gold frame exact match is about 0.714; candidate objective is about 0.943 and promotion is rejected by wrong accept rate. | confirmed | `metrics.csv` contains `gold_eval frame_exact_match=0.714`, `current_objective frame_exact_match=1.0`, `candidate_objective frame_exact_match=0.942708`, `wrong_accept_rate=0.057292`, `promoted=False`. |
| Q7 | Bottlenecks for live `main-evolution` include weak L1 coverage, teacher inconsistency, and an overly strict promotion gate. | confirmed | `quality.json` and `metrics.csv` both list these bottleneck codes and severities. |
| Q8 | L1 did not participate in these runs; this is an enablement/evolution issue rather than an ordinary tuning result. | confirmed | All inspected run summaries show L1 accepted/chosen count 0; preflight warns `L1_AGENT_MODE=disabled`. |
| Q9 | L2's current empirical issue is guard/precision calibration: cache-full accepted fields include nontrivial wrong accepts. | confirmed | `cache-full/main-evolution` records L2 field accepted 125, correct 111, wrong 14, accuracy 0.888; bottleneck is `weak_l2_guard_calibration`. |
| Q10 | L3 should stay shadow/diagnostic for now because preflight and guarded/cache results show zero reliable accepts and poor latency/quality. | partially confirmed | The data strongly supports the risk: preflight accepts 0 and p95 generation is about 4414ms; `l3-guarded` drops frame exact to 0.971 with L3 field accuracy 0.388 and p95 around 985ms. The "should stay" part is a recommendation rather than a repository fact. |
| Q11 | The review's recommended next experiments should start with a direct live teacher-vs-gold quality gate before more full-cascade tuning. | partially confirmed | The recommendation follows from confirmed metrics and code semantics. No dedicated `teacher-live-vs-gold` experiment exists yet; the live all-L4 suite is close but still wrapped in the full experiment harness. |
| Q12 | The current teacher prompt is a single full-frame prompt over the full intent/slot schema, so an intent-first or shortlist prompt experiment is a plausible next test. | confirmed | `teacher.py` builds one prompt with all intent names and all slot names and parses one `Frame`. |
| Q13 | Promotion reporting does not currently split candidate-vs-teacher errors from candidate-vs-gold or teacher-vs-gold errors. | confirmed | Promotion replay operates on `TeacherTrace` without `gold_frame`; metrics expose run-level `gold_eval` separately but candidate/current objectives are teacher-replay metrics. |

## Evidence Log

Initial repository state:

- `git status --short --branch` showed `## main...origin/main` before this document was created.
- Local instructions in `AGENTS.md` confirm NLU frame parsing, prompts, parsers, diagnostics, and dataset adapters belong in target code or adapters, not Darjeeling core.

Next steps:

1. Read the source review and map each concrete numeric/design claim to packet files.
2. Inspect `cache-full` comparison, per-run metrics, quality, and settings.
3. Inspect `live-residual-500` comparison, per-run metrics, quality, and settings.
4. Inspect NLU teacher/eval code paths to verify the teacher-objective versus gold-evaluation interpretation.
5. Run focused tests that cover report generation, replay/promotion metrics, residual L4, L1/L2/L3 routing, and target/core boundaries.

### 2026-06-22 Packet Metric Pass

Evidence for Q1:

- `docs/experiments/review_packets/post_refactor_fixed_20260615/cache-full/comparison.csv` reports `main-evolution` with `requests=3000`, `frame_exact_match=0.998`, `weak_field_coverage=0.530416`, `weak_field_accuracy=0.99583`, `full_l4_calls_per_100=45.3`, and `residual_l4_calls_per_100=3.366667`.
- The same row reports layer shares `L0=0.509`, `L1=0.0`, `L2=0.004333`, `L3=0.0`, `L4=0.486667`.
- `cache-full/run-details/main-evolution/quality.json` confirms layer counts `L0=1527`, `L2=13`, `L4=1460`.

Verdict for Q1: confirmed.

Evidence for Q2:

- `cache-full/run-details/main-evolution/quality.json` reports `serving_full_l4.tokens=0`, `serving_full_l4.cost_usd=0.0`, `serving_residual_l4.tokens=0`, and `serving_residual_l4.cost_usd=0.0`.
- Cached serving p95 is cache-like: full L4 p95 is about `0.0074ms`; residual L4 p95 is about `0.0191ms`.
- Therefore the review is right that cache-full cannot be read as live serving latency or cost evidence.

Verdict for Q2: confirmed.

Evidence for Q3 and Q4:

- `live-residual-500/comparison.csv` reports these frame exact matches: `main-evolution=0.714`, `no-l2=0.720`, `l2-expert-bank=0.696`, `l2-global-student=0.704`.
- All four live rows have `weak_field_coverage=0.0`, `full_l4_calls_per_100=100.0`, `residual_l4_calls_per_100=0.0`, and `L4 share=1.0`.
- `live-residual-500/run-details/main-evolution/quality.json` confirms `layer_counts` of `L4=500` and `L0/L1/L2/L3=0`, with `serving_full_l4.calls=500`.
- `live-residual-500/run-details/main-evolution/metrics.csv` reports L4 `accepted_accuracy=0.714`, `wrong_accept_rate=0.286`, and `forced_global_accuracy=0.714`.

Verdict for Q3: confirmed.

Verdict for Q4: confirmed. In these live runs, the observed gold failure is all-L4 behavior, not weak-layer takeover.

Evidence for Q5:

- `live-residual-500/run-details/main-evolution/settings.sanitized.json` reports `teacher=live`, `openai_model=gpt-5.5`, `l4_proposal_mode=disabled`, `l1_agent_mode=disabled`, `l2_target_evolution_mode=disabled`, `l2_tuning_mode=disabled`, and `local_slm_mode=disabled`.
- `preflight-live.json` also warns: `L1_AGENT_MODE=disabled; set agent-session for real L1 evolution experiments`.

Verdict for Q5: confirmed.

Evidence for Q6:

- `live-residual-500/run-details/main-evolution/metrics.csv` reports `gold_eval frame_exact_match=0.714`.
- The same file reports `current_objective frame_exact_match=1.0`, `candidate_objective frame_exact_match=0.9427083333333334`, `candidate_objective wrong_accept_rate=0.057291666666666664`, and `promotion promoted=False`.
- The review's interpretation that teacher-replay and gold evaluation are separate truth sources is supported by these metrics. Code-path review is still needed to confirm the exact semantics.

Verdict for Q6: confirmed from packet evidence.

Evidence for Q7:

- `live-residual-500/run-details/main-evolution/quality.json` lists `weak_l1_rule_coverage` as warning, `teacher_inconsistency` as error, and `overly_strict_promotion_gate` as warning.
- `metrics.csv` repeats the same bottleneck rows with evidence strings.

Verdict for Q7: confirmed.

Evidence for Q8:

- `cache-full/main-evolution`, `live-residual-500/main-evolution`, `l3-shadow`, and `l3-guarded` all report `L1` chosen count 0 and L1 field accepted 0.
- Preflight explicitly says L1 agent mode is disabled.

Verdict for Q8: confirmed.

Evidence for Q9:

- `cache-full/run-details/main-evolution/quality.json` reports `field_summary.layers.L2.accepted=125`, `correct=111`, `wrong=14`, and `accuracy=0.888`.
- The run bottleneck is `weak_l2_guard_calibration`, with evidence `L2 wrong accepts 6/13 (0.462) against teacher/gold-visible traces`.

Verdict for Q9: confirmed.

Evidence for Q10:

- `preflight/l3_benchmark.sanitized.json` reports `accepted=0`, `would_accept=0`, `generation_p50_ms=1295.891833`, `generation_p95_ms=4414.114558`, and `throughput_qps=0.4145`.
- `cache-full/comparison.csv` shows `l3-shadow` with `total_latency_p95_ms=1085.914987`, no L3 chosen share, and worse comparison score than the main run.
- `cache-full/run-details/l3-guarded/quality.json` reports L3 chosen count 87, frame exact match 0.971, L3 field accuracy 0.388, and 123 wrong L3 fields.

Verdict for Q10: partially confirmed. The risk diagnosis is data-backed; the exact product decision to keep L3 shadow-only is a recommendation.

### 2026-06-22 Teacher/Eval Code Semantics Pass

Evidence for Q6 and Q13:

- `src/darjeeling/targets/nlu/schemas.py` defines `TraceRecord` with both `gold_frame` and `teacher_frame`, but `TeacherTrace` intentionally omits `gold_frame`.
- `src/darjeeling/targets/nlu/compiler/loop.py` converts runtime traces through `compiler_inputs_from_traces()` and `traces_to_teacher_view()` before compilation, then `assert_teacher_visible_only()` rejects any compiler-visible gold field.
- `tests/targets/nlu/test_gold_leakage.py` locks this behavior: compiler inputs are `TeacherTrace` objects and do not contain `gold_frame`.
- `src/darjeeling/targets/nlu/compiler/replay.py` evaluates offline artifact sets by iterating labeled `TeacherTrace` objects and setting `expected = trace.teacher_frame`; current/candidate objective frame exact match, wrong accept rate, field metrics, and promotion gates are therefore teacher-replay metrics.
- `src/darjeeling/targets/nlu/reports.py` writes run-level `gold_eval frame_exact_match` from `TraceRecord.gold_frame`, and `comparison.csv` uses `_gold_frame_exact_match(traces)`.
- No code path or report field named `candidate_vs_gold`, `teacher_vs_gold`, `wrong_vs_gold`, or similar exists. The closest current split is run-level gold eval versus promotion/current/candidate teacher objectives.

Verdict for Q6: confirmed at code level. The live `current_objective=1.0` is expected for all-L4 teacher replay and does not contradict run-level gold frame exact match of 0.714.

Verdict for Q13: confirmed. Adding candidate-vs-gold diagnostics would need a deliberate benchmark-only reporting path; it should not leak gold into compiler inputs.

Evidence for Q11:

- Confirmed Q3/Q4 show the live suite is all-L4 and only 0.696-0.720 frame exact match against gold.
- Confirmed Q5 shows this suite was not exercising the full coding-agent evolution path.
- `src/darjeeling/targets/nlu/experiments.py` has experiment specs for cascade ablations such as `main-evolution`, `no-l2`, `l2-global-student`, and L3 variants, but no dedicated `teacher-live-vs-gold` experiment.
- The existing live suite is useful evidence, but a smaller direct teacher-vs-gold gate would isolate teacher prompt/schema quality from compilation, promotion, and weak-layer settings.

Verdict for Q11: partially confirmed. It is a justified next step, but it is a recommendation rather than a currently implemented check.

Evidence for Q12:

- `src/darjeeling/targets/nlu/teacher.py` builds a single system prompt that says to return one JSON frame, lists all allowed intents, lists all allowed slot names, and parses the response as `Frame`.
- There is no `teacher-v2-intent-first`, `teacher-v3-domain/intent-shortlist`, or two-stage intent/slot extraction implementation in the current source tree.

Verdict for Q12: confirmed.

### 2026-06-22 Focused Verification Tests

Command:

```bash
uv run pytest tests/targets/nlu/test_gold_leakage.py tests/targets/nlu/test_replay_runtime.py tests/targets/nlu/test_replay_promotion.py tests/targets/nlu/test_l4_teacher.py tests/targets/nlu/test_report_l3_summary.py tests/targets/nlu/test_l2_expert_bank.py tests/targets/nlu/test_l3_local_slm.py tests/targets/nlu/test_l3_residual_gate.py tests/targets/nlu/test_l1_rust_worker.py tests/test_core_contracts.py tests/test_target_boundary.py tests/targets/nlu/test_target_core_boundary.py -q
```

Result:

- `84 passed in 7.06s`

Format check:

```bash
git diff --check
```

Result:

- passed

What this validates:

- Compiler-visible NLU traces still exclude `gold_frame`.
- Runtime replay, teacher cache/live/residual behavior, promotion replay, report metrics, L1 Rust worker, L2 expert bank, L3 local SLM/gate, and target/core boundary tests are internally consistent.
- Passing tests do not contradict the review's main finding; they mostly confirm that the current split between teacher-visible compilation and gold-visible reporting is intentional.

## 2026-06-22 Top-3 Implementation Pass

Implemented from `docs/design/08_gpt55_pro_0622_teacher_eval_plan.md`:

1. Added a benchmark-only live teacher-vs-gold path under the NLU CLI: `edge-mvp-nlu teacher eval-live`.
2. Added prompt comparison on one fixed sample: `edge-mvp-nlu teacher compare-prompts`, defaulting to `teacher-v1` and `teacher-v2-intent-first`.
3. Added report-only `gold_diagnostics` rows and `quality.json.gold_diagnostics`, with summary text that separates teacher-replay metrics from gold-evaluation diagnostics.

Boundary notes:

- `gold_frame` remains in `TraceRecord` and report-only aggregation.
- `TeacherTrace`, compiler inputs, training, promotion replay, and candidate selection still use teacher-visible traces without gold.
- The new teacher quality gate calls live L4 directly and does not route through L0-L3, build artifacts, compile, promote, or feed gold back into teacher replay.

New artifacts:

- `teacher_live_vs_gold.summary.json`
- `teacher_live_vs_gold.details.csv`
- `teacher_live_vs_gold.details.jsonl`
- `teacher_prompt_comparison.json`
- `teacher_prompt_comparison.csv`

Focused verification command:

```bash
uv run pytest tests/targets/nlu/test_l4_teacher.py tests/targets/nlu/test_teacher_eval.py tests/targets/nlu/test_report_l3_summary.py tests/targets/nlu/test_experiment_suite_cli.py tests/targets/nlu/test_gold_leakage.py -q
```

Result:

- `49 passed in 2.56s`

Final implementation verification:

```bash
uv run pytest tests/targets/nlu/test_nlu_target.py::test_nlu_teacher_adapter_builds_prompt_and_parses_frame tests/targets/nlu/test_l4_teacher.py tests/targets/nlu/test_teacher_eval.py -q
uv run pytest tests/test_core_contracts.py tests/test_target_boundary.py tests/targets/nlu/test_target_core_boundary.py -q
uv run pytest -q
uv run python -m darjeeling.targets.nlu.main_cli teacher eval-live --help
uv run python -m darjeeling.targets.nlu.main_cli teacher compare-prompts --help
uv run ruff check src/darjeeling/targets/nlu/teacher.py src/darjeeling/targets/nlu/compiler/l4_context.py src/darjeeling/targets/nlu/layers/l4_cloud_llm.py src/darjeeling/targets/nlu/teacher_eval.py src/darjeeling/targets/nlu/main_cli.py src/darjeeling/targets/nlu/reports.py tests/targets/nlu/test_l4_teacher.py tests/targets/nlu/test_teacher_eval.py tests/targets/nlu/test_experiment_suite_cli.py tests/targets/nlu/test_report_l3_summary.py
git diff --check
```

Results:

- Targeted regression and new teacher eval tests: `15 passed in 0.87s`
- Boundary tests: `16 passed in 0.82s`
- Full suite: `276 passed in 51.89s`
- New CLI help commands rendered successfully.
- Ruff touched-file check passed.
- `git diff --check` passed.

## Final Verification Summary

The 2026-06-22 GPT-5.5-Pro review is largely accurate against the committed review packet and current repository.

Confirmed:

1. `cache-full` shows the mechanism can work under stable cache-backed teacher replay, but it is not live serving cost/latency evidence.
2. `live-residual-500` is all-L4 in every run; weak layers did not take over and therefore did not cause the live accuracy drop.
3. Live L4 teacher/fallback quality against MASSIVE gold is only about 0.696-0.720 frame exact match in this packet.
4. Promotion/current/candidate objectives are teacher-replay metrics, while `comparison.csv` and `quality.json` frame exact match are gold metrics when gold is present.
5. The live suite did not exercise full coding-agent evolution: L1 agent, L2 target evolution/tuning, L4 proposal, and L3 were disabled.
6. L1 is effectively absent in these runs.
7. L2's observed cache-backed issue is precision/guard calibration, not lack of raw activity alone.
8. L3 is not ready as a main serving path under this configuration: it is slow, has zero preflight accepts, and guarded routing hurts quality.
9. The current teacher prompt is still a single full-frame, full-schema prompt.
10. Promotion reports do not yet distinguish candidate-vs-teacher errors from candidate-vs-gold and teacher-vs-gold errors.

Qualified:

- The recommendation to run `teacher-live-vs-gold` first is not a repository fact, but it is the right next experimental move based on the confirmed evidence.
- Adding gold-based candidate diagnostics must be benchmark-only/report-only. The existing compiler gold-exclusion tests and AGENTS.md boundary instructions mean gold should not be introduced into core compiler or target training inputs.

Recommended next implementation/experiment order:

1. Add a small benchmark-only teacher quality gate that runs live L4 directly against MASSIVE gold and reports intent accuracy, slot accuracy, frame exact match, parse failures, invalid schema rate, tokens, and latency.
2. Add report-only diagnostics that compare teacher, candidate/final, and gold when gold is available, without exposing gold to compiler inputs.
3. Prototype teacher prompt variants: current full-frame prompt, intent-first then slot extraction, and intent/domain shortlist.
4. Only after teacher quality is understood, run isolated L1 field-bounty and L2 risk-coverage experiments.
5. Keep L3 as shadow/diagnostic until it has reliable accepts and acceptable latency.
