# Benchmark Selection Phase Note

Date: 2026-06-23

## Current Conclusion

Darjeeling should stop using MASSIVE as the main benchmark for the next phase.
MASSIVE remains useful as a secondary stress test for open slot values and
annotation-convention robustness, but it is not a clean benchmark for the
current scientific question.

The current scientific question is not whether Darjeeling can outperform L4.
It is:

> When L4 is already a reliable high-ceiling solver, can Darjeeling externalize
> part of that capability into cheaper L1/L2 layers, reducing L4 calls, tokens,
> cost, and latency while keeping final quality close to the all-L4 baseline?

This requires a benchmark where L4 is strong enough to be a useful teacher and
fallback. MASSIVE failed that requirement in the current setup: live L4 was not
close to saturated under strict frame exact match, and the gold-label audit
showed that many failures were dataset convention mismatch or exact-match
artifacts rather than clear semantic teacher errors.

## Benchmark Strategy

Adopt CLINC150 `data_full` as a **Phase 1 mechanism-validation benchmark**, not
as Darjeeling's long-term representative benchmark.

CLINC150 is useful now because it is closed-form intent classification:

- output is one of 150 in-scope labels or OOS;
- there are no open slot values, span boundaries, casing artifacts, or
  slot-key convention disputes;
- existing NLU `Frame(intent, slots={})` machinery can be reused with low
  implementation cost;
- L2 should have a realistic path to high-precision absorption.

The Phase 1 claim should be narrow:

> A reliable L4 teacher plus teacher-visible L2 learning can reduce L4 calls on
> a closed intent-classification task with little or no final quality loss.

It should not be presented as evidence that Darjeeling is already a general
externalization system for arbitrary intelligent tasks.

## Generality Risk

The main risk is path dependence: CLINC150 is still NLU and still classification.
It may validate the economics and promotion loop while saying too little about
structured generation, tool use, executable verification, or multi-step
reasoning.

To control that risk, Phase 1 must have explicit exit criteria and a Phase 2
commitment:

- If CLINC150 teacher reliability is too low after at most two reasonable
  prompt designs, reject CLINC150 rather than starting another prompt-search
  loop.
- If CLINC150 validates the L4-externalization loop, move next to a structured
  executable benchmark such as WikiSQL or a frozen BFCL non-live single-turn
  subset before making broad claims about Darjeeling generality.

## Success Meaning

Success on CLINC150 means:

- all-L4 accuracy is high enough to be a credible teacher/fallback baseline;
- L2 accepts a large share of requests at very high accepted precision;
- final cascade accuracy is within a small delta of all-L4;
- L4 calls, tokens, cost, and latency drop materially across realistic streams.

It does not mean:

- Darjeeling has solved NLU generally;
- CLINC150 conventions should enter Darjeeling core;
- L2 should be trained from validation/test gold in the real system;
- future benchmark work can remain inside NLU classification.

## Boundary

CLINC150-specific labels, OOS handling, prompts, metrics, thresholds, and error
taxonomies belong in the NLU target, CLINC150 adapter, or experiment docs.
Darjeeling core must remain target-, dataset-, and application-independent.

