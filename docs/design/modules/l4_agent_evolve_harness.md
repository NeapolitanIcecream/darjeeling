# L4 agent evolve harness

## Role

L4 agent evolve is the shared architecture for layer-owned artifacts that need
coding-agent style iteration. L1, L2, and L3 should use the same outer shape:

```text
layer-evolve job
  resolve EvolutionRunPolicy(max_rounds, timeout, patience, executor)
  for each round:
    prepare isolated workspace
    write program.md, objective, visible data, and tools
    run the configured round executor, usually an L4 agent session
    agent autonomously edits/evaluates/searches until it stops
    outer harness checks workspace scope
    outer harness evaluates target-owned visible/private gates
    record EvolutionRoundResult
  outer replay decides artifact promotion
```

The harness does not prescribe an internal script such as
`edit -> evaluate -> optuna -> evaluate`. The agent is the planner,
implementer, and local experiment driver inside a round.

## Round Boundary

- A layer-evolve job has one `max_rounds` value. Each round either launches an
  agent session or runs an explicitly selected non-agent executor such as
  `dry-run` or `local-search`.
- Agent sessions are round-local. A multi-round job may launch more than one
  session, and later rounds may start from the latest successful candidate.
- The agent can decide how many times to inspect context, edit files, run
  evaluate tools, run Optuna/search tools, debug, and stop.
- The prompt must stay short and stable. Dynamic state belongs in workspace
  files so the session keeps reasoning continuity while prompts remain cacheable.
- Agent stop reasons are agent-owned: visible target reached, no safe progress,
  risk too high, or budget near exhaustion.
- Harness controls constrain maximum rounds, per-round timeout, patience and
  executor. Target-local tools may also constrain trials or evaluation cost, but
  those are not core policy fields.

## Visibility

- Agent-visible data may include train rows, visible validation folds, visible
  diagnostics, and visible cross-audit diagnostics.
- Private selection and promotion rows are never copied into the workspace.
- Private gate aggregates are not fed back into the same round workspace.
- Agent-visible state must not include booleans or summaries derived from
  private gate outcomes; those belong only in the outer summary and promotion
  metadata.
- The agent can judge visible readiness, but adoption authority stays outside
  the session.

## Layer Surfaces

| Layer | Editable surface | Workspace tools |
| --- | --- | --- |
| L1 | Rust/C/C++ program bank code | compile, unit tests, bench, replay |
| L2 | target L2 code, postprocess, abstain, feature/search-space files | evaluate, Optuna/search, visible cross-audit |
| L3 | prompt templates, few-shot/context packing, routing/guard prompts | prompt eval, local SLM bench, latency/cost eval |

The architecture is homologous; the artifact surfaces are not identical.

## Current Implementation Status

- Core exposes only `EvolutionRunPolicy`, `EvolutionRoundResult` and
  `EvolutionRunSummary`. It does not carry target quality claims, private-gate
  requirements, profile guidance, evidence policy, cost policy or replay
  cadence.
- L1 now runs real `max_rounds`. Each round has its own workspace,
  transcript, report, diff, command result, validation result and round result.
  Later rounds start from the latest successful candidate crate.
- L2 has the first concrete `agent-session` entry point. It launches one Codex
  session for a target-evolution round over `workspace/l2_target/`, then
  evaluates the resulting `target/` candidate with the existing target-owned
  visible/private gates. L2 keeps its train/evaluate and selection/adoption
  logic in the NLU target.
- L2 agent-visible `round_state.json` reports visible validation readiness, not
  private selection/promotion pass/fail.
- L2 `tools/search_config.py` remains available as an agent-invoked Optuna tool.
  The old `local-search` mode is retained for deterministic tests and protocol
  probes, not as the preferred L2 evolve methodology.
- L2 `dry-run` remains a fixture path for tests and controlled patch replay.
- L3 now uses `max_rounds` instead of `max_agent_sessions`. Each round launches
  a prompt evolution session over `workspace/l3_prompt/`, exposes editable `prompt/`,
  protects `contexts/` and tools, provides prompt validation, visible prompt
  eval, local SLM bench, and latency/cost eval as workspace tools, snapshots the
  candidate prompt, and can run visible/private replay gates through the
  existing local SLM replay evaluator.
  The direct L3 prompt proposal path remains legacy bounded proposal support.
