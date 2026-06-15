# 2026-06-15 Post-Refactor Fixed Workload Summary

Source result:
`docs/experiments/2026-06-14_post_refactor_fixed_workload_results.md`

Run root: `runs/post-refactor-fixed-20260615-0954`

## What Was Actually Measured

This run completed the fixed workload:

- Cache-backed matrix: 12 trace sets, 3000 requests each, 36,000 serving traces.
- Live residual suite: 4 trace sets, 500 requests each, 2,000 live traces.
- L3 readiness check: guarded local SLM benchmark with 3 smoke requests.

The cache-backed matrix is the main source for routing and layer-takeover
behavior. The live suite is mainly a live L4 cost and latency measurement; it did
not exercise residual live L4 calls because every live request routed to full L4.

## Main Cache-Backed Result

For `main-evolution` on `zipf-heavy`, 3000 requests:

| layer | takeover count | takeover rate | accepted accuracy |
| --- | ---: | ---: | ---: |
| L0 | 1527 | 50.90% | 100.00% |
| L1 | 0 | 0.00% | n/a |
| L2 | 13 | 0.43% | 53.85% |
| L3 | 0 | 0.00% | n/a |
| L4 | 1460 | 48.67% | 100.00% |

Frame exact match was `0.998`. Full L4 calls fell from `49.1/100` in `no-l2`
to `45.3/100`, with `3.367/100` residual L4 calls.

This is not strong evidence of generalized lower-layer learning. Most apparent
savings came from L0 on a locality-heavy stream. L2 covered very little traffic
and was not accurate enough when it did accept. L3 was disabled in the main run.

## Locality Effect

L0 coverage changes sharply with workload locality:

| workload | L0 takeover | L4 takeover | frame exact |
| --- | ---: | ---: | ---: |
| `zipf-heavy` | 50.90% | 48.67% | 0.998 |
| `zipf-mild` | 10.93% | 89.07% | 0.999667 |
| `uniform` | 6.17% | 93.83% | 0.999667 |

This means the current system looks much better on repeated or cache-friendly
traffic than on traffic with fewer exact repeats. If L0 is enabled in headline
results, it can dominate the story and make the whole cascade look more useful
than L1/L2/L3 actually are.

My view is that we need to debug each layer's effect separately, and decide
cautiously in future experiments whether L0 should be enabled. If L0 is disabled,
then L1, L2, and L3 are no longer learning the residual after L0, which introduces
some bias; but if L0 is enabled, then L1, L2, and L3 receive too little input, and
the overall system output becomes blindly optimistic because L0 cached too many
inputs.

## L1, L2, And L3 Readout

L1 was not meaningfully measured in this fixed workload. `L1_AGENT_MODE` was
disabled, the default native program abstained, and L1 accepted zero requests.
This run only checked L1 plumbing/preflight behavior, not L1 evolution quality or
L1 serving value.

L2 is still a quality bottleneck:

- Guarded `main-evolution`: L2 accepted 13/3000 requests, with 7/13 correct.
- `no-guard`: L2 accepted 872/3000 requests, but only 103/872 were correct and
  frame exact match fell to `0.743667`.

The guard is necessary, but the guarded L2 path is still too narrow and not
reliable enough.

L3 is operational but not useful as a serving path in this workload:

- `l3-shadow`: no final L3 accepts, but high local generation latency.
- `l3-guarded`: L3 accepted 87/3000 requests, only 6/87 correct, and frame exact
  match fell to `0.971`.

L3 should remain disabled, guarded for diagnostics, or shadow-only until its
calibration and output quality improve.

## Live Suite Readout

The live suite made 2,000 full live L4 calls:

- Total tokens: 1,442,061.
- Total cost: `0.7763856 USD`.
- Residual live calls: 0.
- Live `main-evolution` frame exact match against MASSIVE gold: `0.714`.
- Live teacher self-consistency: 100%, because final frames came from the live
  teacher.

The live suite therefore gives a useful L4 cost and latency baseline, but it does
not prove residual live savings.

## Historical Comparison

Compared with the 2026-06-08 full-layer pilot and controlled suite, this run is
much larger: 36,000 cache-backed traces plus 2,000 live traces versus a 60-request
pilot and 40-request controlled rows. The old run was useful for proving
end-to-end plumbing, but it is not directly comparable because the post-refactor
runtime now records field-level patches, residual L4 calls, conflict/override
metrics, and stricter promotion behavior.

The direction is still informative:

- L4 share is much lower on `zipf-heavy`, but mostly because L0 handles many exact
  repeated requests.
- L2 remains narrow. Earlier narrow L2 target experiments showed safe improvements
  in specific cases, but this fixed workload does not show broad L2 reliability.
- L3 remains unattractive. Earlier runs showed poor L3 shadow quality; this run
  confirms that guarded L3 can actively hurt quality when allowed to accept.

## Practical Conclusion

This fixed workload is a good post-refactor baseline and a good reproducibility
checkpoint. It is not evidence that Darjeeling has already achieved the intended
generalized lower-layer savings.

The next useful experiments should report results with L0 separated from the
headline cascade:

- L0 enabled versus L0 disabled.
- Metrics conditional on L0 miss.
- Unique-utterance or cold-start streams.
- Separate L1, L2, and L3 diagnostics with enough residual traffic for each layer.
- Live residual measurements only after cache-backed residual routing is known to
  trigger on the selected workload.
