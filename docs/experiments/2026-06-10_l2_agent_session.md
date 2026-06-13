# 2026-06-10 L2 agent-session path

## Goal

Validate the next L2 target-evolution design: one long-running L4 agent session
per fixed target workspace, with Optuna/search exposed as a workspace tool
rather than as a separate outer evolve phase.

## Design Change

`edge-mvp-nlu l2 target-evolve` now accepts:

```bash
--mode agent-session
```

This mode:

- evaluates the unmodified baseline first
- launches one live agent command in `workspace/l2_target/`
- lets the agent decide its own internal edit/evaluate/search loop
- runs workspace scope checks after the session exits
- evaluates the resulting `target/` candidate with visible validation and
  private selection/promotion gates

Legacy `dry-run`, `local-search`, and multi-launch `codex-cli` modes remain for
fixtures, protocol probes, and compatibility. The preferred real L2 evolve path
is now `agent-session`.

## Smoke

Command used a no-op fake Codex command to validate wiring without LLM cost:

```bash
mkdir -p runs/l2-agent-session-noop-smoke-r1
L2_AGENT_CODEX_COMMAND=/usr/bin/true uv run edge-mvp-nlu l2 target-evolve \
  --traces runs/l2-list-fallback-tuned-3k-r1/traces.jsonl \
  --out-dir runs/l2-agent-session-noop-smoke-r1 \
  --max-traces 80 \
  --mode agent-session \
  --budget-profile smoke \
  --rounds 3 \
  > runs/l2-agent-session-noop-smoke-r1/stdout.json
```

Result:

- Mode: `agent-session`
- Requested rounds: `3`
- Completed evaluated candidates: `1`
- Stop reason: `agent_session_completed`
- Agent launches: started `1`, succeeded `1`, remaining `0`
- Agent session scope: `single_session_agent_controls_internal_loop`
- Split: train `48`, visible validation `16`, private selection `8`, private
  promotion `8`
- Evidence class: `connectivity_smoke`
- Workspace private holdout files:
  - `data/selection_holdout.jsonl`: absent
  - `data/promotion_holdout.jsonl`: absent
- Outer private holdout files:
  - `private/selection_holdout.jsonl`: present
  - `private/promotion_holdout.jsonl`: present

The no-op candidate was not adopted:

- `selection_decision.selected=false`
- `adoption_decision.adopted=false`

## Conclusion

The single-session L2 path is wired: the CLI can launch one agent process,
preserve private holdout isolation, evaluate the resulting target candidate, and
record the new agent-session metadata. This smoke is not an L2 quality result.

The next meaningful experiment should use real GPT-5.5 `agent-session` on a
small fixed snapshot, with the agent allowed to call `tools/evaluate.py` and
`tools/search_config.py` inside the same session.

## Smoke r2: private-gate visibility fix

After tightening the agent-visible state contract, I reran the no-op
`agent-session` smoke:

```bash
rm -rf runs/l2-agent-session-noop-smoke-r2
mkdir -p runs/l2-agent-session-noop-smoke-r2
L2_AGENT_CODEX_COMMAND=/usr/bin/true uv run edge-mvp-nlu l2 target-evolve \
  --traces runs/l2-list-fallback-tuned-3k-r1/traces.jsonl \
  --out-dir runs/l2-agent-session-noop-smoke-r2 \
  --max-traces 80 \
  --mode agent-session \
  --budget-profile smoke \
  --rounds 3 \
  > runs/l2-agent-session-noop-smoke-r2/stdout.json
```

Result:

- Mode: `agent-session`
- Completed evaluated candidates: `1`
- Stop reason: `agent_session_completed`
- Agent launches: started `1`, succeeded `1`
- Evidence class: `connectivity_smoke`
- Split: train `48`, visible validation `16`, private selection `8`, private
  promotion `8`
- Inner/selection/promotion accepts: all `0`
- `selection_decision.selected=false`
- `adoption_decision.adopted=false`

Visibility checks:

- `workspace/l2_target/data/selection_holdout.jsonl`: absent
- `workspace/l2_target/data/promotion_holdout.jsonl`: absent
- `private/selection_holdout.jsonl`: present
- `private/promotion_holdout.jsonl`: present
- `data/round_state.json`: contains `passes_visible_validation_gate`
- `data/round_state.json`: does not contain `passes_candidate_selection_gate`
- `data/round_state.json` and `data/target_diagnostics.json`: do not contain
  `selection_holdout` or `promotion_holdout`

Conclusion: the wiring still works after the fix, and private gate derived
state is no longer written into the agent-visible workspace. This remains a
connectivity smoke, not a quality result.
