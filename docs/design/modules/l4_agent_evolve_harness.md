# L4 agent evolve harness

## Role

L4 agent evolve is the shared architecture for layer-owned artifacts that need
coding-agent style iteration. L1, L2, and L3 should use the same outer shape:

```text
layer-evolve job
  prepare isolated workspace
  write program.md, objective, visible data, and tools
  launch one long-running L4 agent session
  agent autonomously edits/evaluates/searches until it stops
  outer harness checks workspace scope
  outer harness evaluates private selection/promotion gates
  outer replay decides artifact promotion
```

The harness does not prescribe an internal script such as
`edit -> evaluate -> optuna -> evaluate`. The agent is the planner,
implementer, and local experiment driver inside one session.

## Agent Session Boundary

- One L4 agent session closes inside one layer-evolve job.
- The agent can decide how many times to inspect context, edit files, run
  evaluate tools, run Optuna/search tools, debug, and stop.
- The prompt must stay short and stable. Dynamic state belongs in workspace
  files so the session keeps reasoning continuity while prompts remain cacheable.
- Agent stop reasons are agent-owned: visible target reached, no safe progress,
  risk too high, or budget near exhaustion.
- Harness budgets constrain wall time, LLM spend, tool calls, Optuna trials, and
  evaluation cost, but do not define a fixed step order.

## Visibility

- Agent-visible data may include train rows, visible validation folds, visible
  diagnostics, and visible cross-audit diagnostics.
- Private selection and promotion rows are never copied into the workspace.
- Private gate aggregates are not fed back into the same agent session.
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

- L1 now has an explicit `agent-session` mode. It launches one Codex session
  over an isolated workspace root with editable `l1_programbank/`, protected
  `program.md`/`contexts/`, `runs/` scratch output, scope checking, and
  provenance for the session policy.
- L2 has the first concrete `agent-session` entry point. It launches one Codex
  session over `workspace/l2_target/`, then evaluates the resulting `target/`
  candidate with the existing visible/private gates.
- L2 agent-visible `round_state.json` reports visible validation readiness, not
  private selection/promotion pass/fail. Failed launches, no-launch budget
  checks, and workspace scope violations are evidence-policy probes rather than
  quality evidence.
- L2 `tools/search_config.py` remains available as an agent-invoked Optuna tool.
  The old `local-search` mode is retained for deterministic tests and protocol
  probes, not as the preferred L2 evolve methodology.
- L2 `dry-run` remains a fixture path for tests and controlled patch replay.
- L3 should be migrated onto the same session boundary before its next real
  agent-evolve experiment; its current direct prompt proposal path is legacy.
