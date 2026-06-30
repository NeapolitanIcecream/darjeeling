# BFCL Experiment Pre-Report

Date: 2026-06-29
Last updated: 2026-06-30

This report records the pre-experiment decisions for evaluating Darjeeling on
the Berkeley Function Calling Leaderboard (BFCL). It is intended to be the
starting context for implementation and experiment agents.

## Executive Summary

The first serious public benchmark experiment should use BFCL before tau3-bench.
BFCL is smaller, more structured, easier to reproduce, and maps directly onto
Darjeeling's `accept`/`abstain` runtime model: the local artifact either emits a
complete function-call output or falls back to L4.

The experiment should not initially claim official BFCL leaderboard leadership.
It should report Darjeeling-style serving results:

- accepted-sample precision and wrong accept rate;
- local L1/L2 coverage;
- end-to-end accuracy after fallback;
- L4 call reduction;
- latency reduction;
- cost reduction per 1,000 requests;
- compile/agent cost and payback estimates.

Compile attempts should use an interactive multi-candidate search loop. The
agent submits a candidate, Core evaluates it on validation, Core writes
agent-safe feedback, and the same agent continues search from that feedback. The
final public result must come from test evaluation after the attempt is closed,
not from validation feedback used during search.

The first experiment should disable L3. The clean initial runtime shape is:

```text
L1: deterministic or generated local code artifacts
L2: target-specific local artifacts
L3: disabled
L4: cached reference/fallback broker
```

L4 should be precomputed for offline experiment repeatability, but reports must
carry two ledgers and two latency views:

- experiment actual: what this experiment actually spent and how long it took,
  including cache construction and agent iteration;
- production counterfactual: what serving would have cost and how long it would
  have taken if the same release used live L4 fallback in production.

## Why BFCL First

BFCL is the best first benchmark because it has a clean task boundary:

- input: user prompt plus function/tool schema;
- output: function call, no-call, or related structured response;
- correctness: function name and argument matching, plus executable subsets;
- fallback semantics: easy to define, because each case can fall through to a
  reference model output;
- Darjeeling value: measurable reduction in L4 calls without forcing local
  artifacts to guess.

tau3-bench remains valuable, but should be a second-stage benchmark. It is more
product-realistic and better for an agent workflow story, but it introduces
multi-turn policy state, tool execution state, user simulation, and task-level
success attribution. Those make first-result debugging harder.

## Benchmark Scale

As checked on 2026-06-29, the public Hugging Face BFCL dataset exposes
`BFCL_v3_*` JSONL files totaling 5,251 top-level cases. The official leaderboard
page currently labels the suite BFCL V4, so implementation should record both:
the official evaluator/version used and the concrete dataset revision/hash used.

Useful scale estimates:

| Scope | Cases | Model calls or turns | Input-size intuition |
| --- | ---: | ---: | --- |
| Pilot subset | 500-800 | 500-800 | small enough for rapid adapter iteration |
| Static core | 1,480 | 1,480 | about 235 input tokens per case before wrappers |
| Static all | 2,000 | 2,000 | about 0.5M raw input tokens before wrappers |
| Live | 2,251 | 2,251 | about 673 input tokens per case before wrappers |
| Multi-turn | 1,000 | about 4,528 turns | tool docs average about 5.2k tokens per case |
| Full public set | 5,251 | about 8,779 calls/turns | roughly 7M-9M input tokens under a low estimate; higher if multi-turn docs are repeated per turn |

The multi-turn category is the only scale-sensitive part. Its rows reference
separate tool documentation files, and those docs can dominate cost and latency.

## Recommended Experiment Stages

Each stage after the cache dry run should use interactive candidate search. The
default loop is:

```text
agent writes submissions/<candidate>/artifacts/l1|l2
agent creates submissions/<candidate>/READY after files are complete
Core evaluates the ready candidate on validation
Core writes journal/feedback-<candidate>.json with aggregate, agent-safe feedback
agent reads feedback and submits the next candidate
Core closes the attempt on candidate, time, cost, user-stop, or completion
Core runs test evaluation only after the attempt is closed
```

This keeps official validation inside Core while still letting the agent improve
L1/L2 artifacts across multiple search rounds.

### Stage 0: Harness And Cache Dry Run

Goal: prove that BFCL data loading, L4 cache building, evaluator invocation, and
Darjeeling request/output shapes are stable.

Suggested scope:

- 100-200 cases sampled from simple, multiple, parallel, parallel-multiple, and
  irrelevance categories.

Done criteria:

- every row has a stable cache key;
- cached L4 output includes usage, cost, latency, finish reason, model/version,
  prompt hash, tool schema hash, and raw evaluator input hash;
- candidate directories have an atomic ready marker so Core never evaluates a
  half-written submission;
- candidate accept/abstain decisions are made before fallback output is read;
- validation/test cache outputs are not exposed to the agent.

### Stage 1: Pilot Subset

Goal: iterate on the BFCL target adapter and first L1/L2 artifacts without paying
the full engineering cost of the complete benchmark.

Suggested scope:

- 500-800 stratified cases;
- include no-call/irrelevance cases early so abstention safety is tested.

Done criteria:

- one attempt can produce multiple L1/L2 candidates from Core-written validation
  feedback;
- end-to-end Darjeeling report can compare baseline L4, the selected L1/L2
  candidate, and fallback-composed serving output;
- wrong accepts are visible as release blockers;
- candidate count, feedback count, stop reason, and agent usage cost are
  recorded;
- report separates experiment actual and production counterfactual ledgers.

### Stage 2: Static Core

Goal: produce the first meaningful public result.

Suggested scope:

- simple, multiple, parallel, parallel-multiple;
- executable simple/multiple/parallel/parallel-multiple;
- irrelevance.

Approximate size: 1,480 cases.

This stage is the cleanest proof of Darjeeling's core hypothesis: stable
tool-calling behavior can be moved into local artifacts while unfamiliar or risky
requests fall back.

### Stage 3: Static All

Goal: measure broader schema and output variety while staying mostly single-turn.

Suggested additions:

- chatable;
- REST;
- SQL;
- Java;
- JavaScript.

Approximate total size: 2,000 cases.

### Stage 4: Live And Multi-Turn

Goal: stress the system on more realistic tool docs and multi-turn behavior.

This stage should happen after the static harness is stable. It may require a
separate target contract for multi-turn state, or at least a clearly separated
adapter mode. Do not let multi-turn complexity contaminate the first static
result.

### Stage 5: Full Public Set

Goal: one full BFCL-derived Darjeeling report.

The final report should still be broken down by category. A single aggregate
number is not enough, because Darjeeling's value depends on coverage, abstention,
and fallback behavior.

## Cached L4 Design

Offline benchmark experiments should allow precomputing L4 outputs for the full
evaluation set. This is different from production serving and is acceptable if
the cache is owned by the experiment harness, not by the artifact-writing agent.

Cache key should include at least:

- BFCL dataset revision and row id;
- normalized request payload;
- tool/function schema hash;
- prompt/template hash;
- reference model id and version;
- decoding configuration;
- evaluator mode/version.

Each cached response should include at least:

- parsed reference output;
- raw model output or enough replay material for audit;
- token usage;
- actual API cost at cache-build time;
- observed L4 latency;
- finish reason;
- parse/schema validation status;
- error metadata if the call failed.

During candidate evaluation, a cache hit should have near-zero actual API cost,
but should still contribute to the production counterfactual serving ledger.

## Dual Ledger And Dual Latency

Reports should carry these serving and experiment views separately.

Experiment actual:

- cache-build L4 API cost;
- agent/compile cost;
- local evaluation compute cost;
- wall-clock time for cache build, validation feedback rounds, test evaluation,
  and agent iteration;
- retry cost from failed API calls or failed agent attempts.

Production counterfactual:

- baseline all-L4 serving cost and latency;
- Darjeeling serving cost and latency with live L4 fallback;
- local compute cost for L1/L2;
- L4 fallback cost for abstained requests;
- savings per 1,000 requests;
- estimated payback requests after compile/agent cost.

The production counterfactual is the number users need before adopting a
Darjeeling release. The experiment actual ledger is what we need to control
research spend.

## L3 Disabled Constraint

L3 should be disabled for the first BFCL experiment.

Reasoning:

- L3 can mean a weaker cloud model, an edge model, or a local model.
- Hardware and deployment choices would dominate latency and cost.
- Results would be difficult to interpret and hard to reproduce.
- BFCL can already test Darjeeling's core local-when-safe behavior with L1/L2.

Reports should say `L3 disabled` rather than reporting L3 as zero coverage.

## Required Metrics

Minimum report fields:

- BFCL category and split scope;
- baseline L4 accuracy;
- L1 accepted count;
- L2 accepted count;
- local coverage;
- fallback share;
- accepted-sample precision;
- wrong accept count and wrong accept rate;
- final accuracy after fallback;
- accuracy delta versus baseline L4;
- latency p50/p95 under experiment actual and production counterfactual views;
- cost per 1,000 requests under experiment actual and production counterfactual
  views;
- compile/agent cost;
- candidate count;
- validation feedback count;
- interactive loop stop reason;
- validation-search wall time and final test wall time;
- estimated payback requests;
- cache hit rate;
- reference model, prompt hash, dataset revision, and evaluator version.

Category breakdowns are required. The full-set aggregate is secondary.

## Integrity Rules

The experiment must preserve Darjeeling's core boundary:

- Core may carry BFCL request, schema, label, and output data as opaque target
  data.
- BFCL-specific parsing and correctness belong in a BFCL target/adapter, not in
  Darjeeling core.
- The agent may see train data and aggregate validation feedback, but not raw
  validation/test rows or cached validation/test L4 outputs.
- Runtime artifacts must decide accept/abstain without reading fallback output.
- Validation feedback files must be aggregate and agent-safe; they must not
  include raw rows, row ids, expected outputs, split indices, or reconstructable
  holdout membership.
- Test evaluation and release decisions happen only after the interactive
  attempt is closed.
- A failed or user-visible test run should be treated as consumed holdout
  evidence for future claims.
- Do not edit generated task-specific L1/L2/L3 workspaces directly in the repo;
  change harnesses, prompts, adapters, or contracts that govern those workspaces.

## Expected Cost And Time Intuition

The L4 cost of a single formal run should be modest after caching. The major
cost is likely to be agent iteration while developing L1/L2 artifacts across
multiple validation-feedback rounds.

Approximate single-run intuition, assuming current low-cost and stronger
tool-calling model price ranges are refreshed before execution:

| Scope | Cheap reference model | Strong reference model | Time intuition |
| --- | ---: | ---: | --- |
| Pilot subset | less than $1 | less than $2 | 5-15 minutes |
| Static core | less than $1 | $1-$3 | 15-30 minutes |
| Static all | less than $1 | $2-$4 | 20-40 minutes |
| Live | $1-$3 | $5-$10 | 30-60 minutes |
| Multi-turn | $3-$8 | $15-$40 | 1-3 hours |
| Full public set | $5-$12 | $25-$60 | 2-5 hours |

Use roughly 10x the single-run estimate for development budgeting, because
baseline, cache validation, multi-candidate search, ablations, and reruns will
happen.

These are planning estimates, not final budget commitments. Refresh model prices
from the provider before launching paid runs.

## Risks

Main risks:

- official BFCL evaluator/version drift;
- public dataset filename/version mismatch with the leaderboard label;
- accidental leakage of cached validation/test L4 outputs into the agent
  workspace;
- accidental leakage of raw validation rows or expected outputs through
  feedback files;
- evaluating half-written candidate directories;
- treating cache-hit latency as production serving latency;
- overclaiming official leaderboard comparability when the experiment is really
  a Darjeeling wrapper/cost study;
- multi-turn state modeling complexity;
- category imbalance hiding weak slices in an aggregate score.

Mitigations:

- pin dataset revision, evaluator version, model id, prompt hash, and schema
  hash;
- report category-level metrics;
- keep the first public result scoped to static core;
- keep L3 disabled;
- maintain dual ledgers and dual latency views;
- require atomic ready markers before Core evaluates candidate submissions;
- store a concise experiment log and cost ledger for each run.

## Sources

- BFCL leaderboard: https://gorilla.cs.berkeley.edu/leaderboard.html
- BFCL public dataset: https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard
- tau-bench repository: https://github.com/sierra-research/tau-bench
- tau2-bench repository: https://github.com/sierra-research/tau2-bench
- Darjeeling README: ../../README.md
- Darjeeling overall design: ../design/reboot/00_overall_design.md
