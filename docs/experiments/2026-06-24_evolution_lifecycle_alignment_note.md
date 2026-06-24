# Evolution Lifecycle Alignment Note

Date: 2026-06-24

This note records the current design discussion. It is not an execution plan.
Write the actual shared-lifecycle optimization plan after the CLINC150 L2
AutoResearch work has completed and its branch has been merged.

## Current Understanding

Darjeeling already has a shared outer compiler generation and promotion loop in
`src/darjeeling/targets/nlu/compiler/loop.py`. That loop takes teacher-visible
traces, builds candidate artifacts across layers, assembles an artifact set,
runs offline replay, decides promotion, and writes the manifest.

The design concern is not that there is no shared upper-level pattern at all.
The concern is that later inner evolution paths have diverged:

- L1 has `L4CodingAgentAdapter` / `run_l1_coding_agent_job`, with workspace
  packaging, agent launch, scope check, transcript, commands, diff, and
  provenance, but it is still a single-job harness. It does not yet provide a
  first-class rounds, feedback, selection, and stop-policy loop.
- L2 has the most complete inner evolution implementation in
  `l2_target_evolution.py`: rounds, budget profiles, agent-session/local-search
  modes, visible/private splits, scope checks, candidate snapshots, selection
  and promotion gates, patience, summaries, and evidence policy.
- L3 has `run_l3_prompt_evolution`, with an isolated prompt workspace,
  agent-session, protected/private data separation, scope check, candidate
  replay, selection, and adoption decisions, but it is mostly a single
  long-session path rather than the richer L2 round loop.

The risk is gradual design drift: L1/L2/L3 share the same lifecycle shape, but
their inner loops now encode similar mechanics separately.

## What Should Be Shared

The shared layer should cover lifecycle mechanics, not task semantics:

- job and workspace layout;
- editable, protected, and scratch roots;
- workspace manifest;
- agent-session launch;
- transcript, report, command, diff, and provenance capture;
- protected-surface scope check;
- visible/private data separation;
- candidate snapshot;
- round state and stop reason;
- feedback pack convention;
- selection/adoption decision shell;
- evidence-strength policy so smoke, dry-run, and short probes are not mistaken
  for strong quality evidence.

## What Should Stay Layer Or Target Owned

Do not abstract away layer-specific behavior:

- L1 Rust ProgramBank build, test, benchmark, and evaluation;
- L2 target model, feature, guard, and postprocess training/evaluation;
- L3 prompt rendering, local SLM replay, and prompt artifact validation;
- CLINC150/NLU metrics, OOS semantics, frame exact match, teacher rows, and
  accepted-error interpretation;
- concrete artifact formats for each layer.

## Deferred Next Step

Do not write or execute the shared-lifecycle optimization plan yet. First let
the CLINC150 L2 AutoResearch plan finish and merge. Its result should inform
which parts of `l2_target_evolution.py` are genuinely reusable and which parts
are still L2-specific.

After that merge, write a scoped plan that:

- inventories the existing outer compiler loop plus L1/L2/L3 inner lifecycle
  mechanics;
- identifies the smallest shared lifecycle contract that reduces duplication;
- decides whether to reuse, extract, or rename pieces of the L2 implementation;
- connects L1 to the shared lifecycle without adding a broad framework;
- keeps target and layer semantics owned by adapters;
- preserves the low-abstraction-tax rule for repository-level code.
