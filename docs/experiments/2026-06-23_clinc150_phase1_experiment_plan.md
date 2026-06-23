# CLINC150 Phase 1 Experiment Plan

Date: 2026-06-23

Purpose: implement and evaluate CLINC150 `data_full` as a Phase 1 mechanism
validation benchmark for Darjeeling. The goal is to test L4 capability
externalization into L2, not to build an NLU-specific final system.

## Required Context Files

Read these before changing code:

- `AGENTS.md`
- `docs/experiments/2026-06-23_benchmark_selection_phase_note.md`
- `docs/experiments/2026-06-22_gpt55_pro_review_verification.md`
- `docs/experiments/2026-06-22_teacher_prompt_search_report.md`
- `docs/experiments/2026-06-22_teacher_gold_label_audit_report.md`
- `/Users/chenmohan/Downloads/Darjeeling-research-0623.md`

## Objective

Build the minimum CLINC150 path needed to answer:

> On a closed-form task where L4 is reliable, can a teacher-visible L2 intent
> absorber take over a large share of requests with very high accepted precision,
> while the final L0/L2/L4 cascade stays close to all-L4 quality and materially
> reduces L4 calls, cost, tokens, and latency?

## Non-Goals

- Do not continue MASSIVE teacher prompt search.
- Do not tune L1 or L3 in this phase.
- Do not claim general Darjeeling validity from CLINC150 alone.
- Do not move CLINC150 label semantics, OOS rules, thresholds, or examples into
  Darjeeling core.
- Do not add a broad plugin framework, schema DSL, or generic evaluator
  abstraction for this one benchmark.
- Do not let validation/test gold enter compiler inputs, promotion replay,
  runtime artifacts, or candidate selection.

## Design Principles

- Keep implementation low-abstraction and target-local.
- Reuse the existing NLU `Frame(intent, slots={})` shape.
- Treat OOS as a safety guard and fallback pressure, not as the only score.
- Prefer deterministic manifests, request IDs, checksums, and explicit split
  files over implicit sampling.
- Report all-L4 quality, cascade quality, L4 calls, tokens, cost, and latency
  separately.

## Phase A: Dataset Adapter

Add a CLINC150 adapter under the NLU target, for example:

```text
src/darjeeling/targets/nlu/adapters/clinc150.py
```

Requirements:

- Ingest a pinned CLINC150 `data_full` source.
- Record source URL or local source path, version, checksum, and license note in
  the processed manifest.
- Preserve official train/validation/test splits.
- Generate the existing processed record format used by NLU replay/eval.
- Map in-scope examples to:

```json
{"intent": "<label>", "slots": {}, "is_abstain": false}
```

- Map OOS examples to:

```json
{"intent": "out_of_scope", "slots": {}, "is_abstain": true}
```

- Add a CLI entry such as `edge-mvp-nlu clinc150 prepare`, or another
  repo-consistent command shape if the existing CLI suggests a better local
  pattern.

Done when:

- processed train/validation/test files can be generated reproducibly;
- manifest includes split counts, checksum, and source metadata;
- focused adapter tests cover split counts, OOS mapping, frame shape, and schema
  construction.

## Phase B: Teacher Gate

Before building L2, prove CLINC150 has a reliable L4 teacher/fallback baseline.

Implement at most two prompt versions:

- `clinc150-intent-v1`: allowed label list only;
- `clinc150-intent-v2-label-cards`: short label definitions plus 1-2 train-only
  examples per label, if this can be generated simply.

Constraints:

- Label cards and examples may use train split only.
- Validation/test examples must not enter prompts.
- Output must be strict JSON with exactly one allowed intent and no extra fields.
- Unknown labels and parse/schema failures are hard failures.
- Temperature and model settings should be recorded.
- Do not put a tight cap on teacher `max_completion_tokens`. Reasoning tokens count
  against the completion budget for reasoning models, so a small cap can produce
  empty visible JSON even for easy classification calls. Use a generous configured
  cap, then report actual token usage and cost from artifacts.

Run a 500-request validation gate:

- 3 validation requests per in-scope intent if available;
- 50 OOS validation requests;
- same request list for both prompt versions.

If one prompt passes, lock it and run full validation.

Suggested teacher gate targets:

- in-scope all-L4 accuracy >= 97%;
- overall all-L4 accuracy >= 95%;
- parse/schema failure <= 0.5%;
- repeated-call consistency >= 99% on a small fixed repeat sample.

If no prompt passes after these two prompt designs, stop and report that
CLINC150 is rejected as Phase 1 benchmark. Do not start open-ended prompt
search.

## Phase C: Diagnostic L2 Ceiling

Run a target-local diagnostic L2 trained from official train gold labels.

This is not a Darjeeling runtime artifact. It exists only to answer whether the
current light L2 approach can absorb CLINC150 at all.

Report:

- validation/test accuracy;
- accepted precision and coverage under confidence thresholds;
- OOS false accept rate;
- confusion families;
- learning curve for 250 / 1k / 3k / full train examples if cheap.

If gold-trained L2 cannot reach high accepted precision with meaningful
coverage, the benchmark may not exercise Darjeeling's intended low-layer
absorption path well enough.

## Phase D: Teacher-Distilled L2

Build the main L2 experiment from teacher-visible traces only.

Compare:

- all-L4 live/cached teacher baseline;
- gold-trained diagnostic L2;
- teacher-distilled L2 in shadow mode;
- teacher-distilled L2 + L4 fallback cascade.

Guard requirements:

- target accepted precision >= 99%;
- final accuracy delta vs all-L4 <= 0.5 percentage points;
- lower-layer OOS false accept <= 2%;
- report coverage and precision by in-scope/OOS split.

Gold may be used only by target-local evaluation/reporting, not by compiler
inputs, promotion replay, runtime artifacts, or candidate selection.

## Phase E: Stream Experiments

Run at least:

- official validation/test distribution for primary quality;
- sequential learning stream for incremental teacher-visible learning;
- uniform replay for generalization;
- zipf-heavy replay for hot request/caching economics.

For each run, report:

- all-L4 accuracy;
- final cascade accuracy;
- accuracy delta vs all-L4;
- L4 calls per 100 requests;
- L4 tokens and cost per request;
- p50/p95 latency;
- L2 accepted coverage;
- L2 accepted precision;
- OOS precision/recall/F1 and lower-layer OOS false accepts;
- parse/schema failures.

Use cache-backed paired replay for routing/promotion comparisons and separate
live runs for measured token/cost/latency claims.

## Phase F: Final Report

Write:

```text
docs/experiments/2026-06-23_clinc150_phase1_report.md
```

The report must include:

- whether CLINC150 passed the teacher gate;
- exact data source, checksum, split counts, and license note;
- prompt versions and locked teacher choice;
- teacher live metrics and repeat consistency;
- gold-trained diagnostic ceiling;
- teacher-distilled L2 shadow and cascade metrics;
- stream comparison tables;
- cost and latency accounting;
- failure analysis and OOS behavior;
- whether CLINC150 validates Phase 1 mechanism goals;
- whether to proceed to Phase 2 WikiSQL/BFCL or reject/revise benchmark choice.

## Success Criteria

The Phase 1 mechanism claim is supported only if:

- all-L4 teacher/fallback quality is high enough to trust;
- teacher-distilled L2 reaches very high accepted precision;
- the cascade holds final quality close to all-L4;
- L4 calls/tokens/cost/latency drop materially;
- results hold on a locked holdout and at least one nontrivial replay stream;
- the implementation keeps CLINC150-specific logic out of core.

## Commit Requirement

When complete, organize the changes and create a git commit unless the user
explicitly says not to.
