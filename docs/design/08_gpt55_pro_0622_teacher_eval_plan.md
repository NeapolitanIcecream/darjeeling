# GPT-5.5-Pro 0622 Teacher/Eval Plan

Date: 2026-06-22

Inputs:

- `/Users/chenmohan/Downloads/Darjeeling-research-0622.md`
- `docs/experiments/2026-06-22_gpt55_pro_review_verification.md`

Goal: fix the first three priorities from the 0622 review before doing more full-cascade tuning:

1. establish live L4 teacher quality against benchmark gold;
2. separate teacher-replay metrics from gold-evaluation diagnostics;
3. improve and compare teacher prompt variants.

The problem to avoid is simple: if live L4 is only about 70%-72% frame exact match against MASSIVE gold, weak-layer training and promotion will learn from noisy labels. First make that visible and measurable, then improve the teacher prompt, then return to L1/L2 cascade work.

## Constraints

- Keep Darjeeling core target-, dataset-, and application-independent.
- Keep NLU frames, intents, slots, teacher prompts, MASSIVE gold diagnostics, and prompt experiments inside `src/darjeeling/targets/nlu` or NLU experiment/docs code.
- Do not expose `gold_frame` to compiler inputs, teacher-visible traces, training, promotion replay, or candidate selection.
- Gold-based diagnostics are benchmark/report-only.
- Keep abstraction tax low. Prefer existing `TraceRecord`, `TeacherTrace`, `Frame`, report writers, CLI commands, settings, and small helper functions.
- Do not add plugin systems, dependency-injection containers, schema DSLs, a generic evaluator framework, or a new experiment platform.
- Prefer plain names a new developer can understand: `teacher-live-vs-gold`, `teacher_frame`, `gold_frame`, `final_frame`, `prompt_version`.

## Non-Goals

- Do not tune L1, L2, or L3 in this pass.
- Do not rerun the expensive full suite as the primary validation.
- Do not use gold labels to train or promote weak layers.
- Do not replace the existing teacher cache or replay system.
- Do not solve every teacher quality issue at once; first make quality measurable and compare a small prompt set.

## Plan

### 1. Add A Teacher-Live-Vs-Gold Quality Gate

Add a small NLU benchmark path that runs live L4 directly against a fixed MASSIVE sample and compares the returned frame to `gold_frame`.

Keep it separate from cascade replay. It should not build artifacts, run compiler generations, promote candidates, or route through L0-L3.

Minimum report fields:

- request count;
- frame exact match;
- intent accuracy;
- slot key/value exact metrics;
- parse failure count;
- invalid intent or slot count;
- abstain count if present;
- full L4 tokens, cost, and latency;
- teacher prompt version and model name.

Implementation should reuse the existing teacher adapter/layer where practical. If a small CLI command is needed, keep it under the NLU CLI surface with a direct name such as `teacher eval-live`.

### 2. Add Report-Only Teacher/Gold Diagnostics

When traces contain both `teacher_frame` and `gold_frame`, reports should make the disagreement visible without changing compiler behavior.

Add simple benchmark-only diagnostics:

- `teacher_vs_gold_frame_exact`;
- `teacher_vs_gold_intent_accuracy`;
- `final_vs_teacher_frame_exact`;
- `final_vs_gold_frame_exact`;
- counts for cases where final agrees with teacher but not gold;
- counts for cases where final agrees with gold but not teacher.

These diagnostics can live in run reports, quality JSON, or a small teacher-eval report. Use direct trace aggregation; do not create a new metric framework.

Keep the existing invariant that `TeacherTrace` excludes `gold_frame`. Add or preserve tests that prove compiler inputs do not contain gold labels.

### 3. Compare A Small Set Of Teacher Prompt Variants

Keep prompt work target-local and modest.

Start with three prompt versions:

- `teacher-v1`: current single full-frame prompt;
- `teacher-v2-intent-first`: first predict intent, then extract slots using that intent context;
- `teacher-v3-shortlist`: provide a narrowed intent or slot shortlist when a safe shortlist is available.

The first implementation may be simple. For example, `teacher-v2-intent-first` can be a target-local two-call path if that is faster to make correct than inventing a combined protocol. Record token and latency cost separately so quality gains are not confused with serving cost.

Prompt variants should be selected by existing settings or an explicit CLI option. Avoid introducing a generic prompt registry; a small static mapping is enough.

### 4. Run Small Experiments Before Cascade Work

After implementation, run a small matrix before any full-cascade experiment:

- `teacher-v1` on 100-200 examples;
- `teacher-v2-intent-first` on the same examples;
- `teacher-v3-shortlist` on the same examples if shortlist support is ready;
- one cache-backed sanity check to make sure reports still render.

Use the same sample across prompt variants. The output should make it obvious whether live teacher quality improved, what it cost, and where errors remain.

## Implemented Path

The benchmark-only teacher gate is target-local:

```bash
edge-mvp-nlu teacher eval-live \
  --data-dir data/processed/default \
  --split validation \
  --stream sequential \
  --max-requests 100 \
  --prompt-version teacher-v1 \
  --out-dir runs/teacher-live-vs-gold \
  --min-frame-exact-match 0.0
```

It runs live L4 directly against MASSIVE gold and writes:

- `teacher_live_vs_gold.summary.json`
- `teacher_live_vs_gold.details.csv`
- `teacher_live_vs_gold.details.jsonl`

The prompt comparison path uses one fixed sample for every prompt:

```bash
edge-mvp-nlu teacher compare-prompts \
  --data-dir data/processed/default \
  --split validation \
  --stream sequential \
  --max-requests 100 \
  --prompt-version teacher-v1 \
  --prompt-version teacher-v2-intent-first \
  --out-dir runs/teacher-prompt-comparison
```

`teacher-v1` is the current single full-frame prompt. `teacher-v2-intent-first` uses a live two-call path: first intent, then final frame/slots with that intent fixed. The comparison report includes frame exact match, intent accuracy, slot metrics, parse/schema failures, tokens, cost, and latency for each prompt.

Run reports now distinguish sources explicitly:

- `current_objective`, `candidate_objective`, and `promotion` are teacher-replay metrics.
- `gold_eval` and `gold_diagnostics` are report-only gold-evaluation diagnostics.
- `quality.json` includes `gold_diagnostics`.

## Tests

Add focused tests for:

- teacher-live-vs-gold report aggregation on synthetic traces;
- gold diagnostics in reports when both teacher and gold exist;
- no gold leakage into `TeacherTrace` or compiler inputs;
- prompt-version selection;
- parser handling of malformed teacher output;
- CLI/report artifact creation for the new teacher quality gate.

Run boundary tests before handoff:

```bash
uv run pytest tests/test_core_contracts.py tests/test_target_boundary.py tests/targets/nlu/test_target_core_boundary.py -q
```

Run the relevant NLU report/teacher tests:

```bash
uv run pytest tests/targets/nlu/test_gold_leakage.py tests/targets/nlu/test_l4_teacher.py tests/targets/nlu/test_report_l3_summary.py -q
```

Before final handoff, run the broader suite if feasible:

```bash
uv run pytest -q
```

## Done Criteria

- There is a documented, repeatable live teacher-vs-gold benchmark path.
- Reports clearly distinguish teacher-replay objective metrics from gold-evaluation diagnostics.
- Gold labels remain blocked from compiler/training/promotion inputs.
- At least the current prompt and one improved prompt variant can be compared on the same sample.
- The comparison reports accuracy, parse/schema failures, tokens, cost, and latency.
- The implementation stays in the NLU target/report/CLI layer and does not add a generic framework.
