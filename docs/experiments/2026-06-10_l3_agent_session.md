# 2026-06-10 L3 agent-session path

## Goal

Validate the L3 homologous agent-session boundary: one L4 coding-agent session
over an isolated prompt workspace, with editable prompt/context-packing/guard
files, protected visible context and tools, private holdouts outside the
workspace, candidate prompt snapshotting, and replay gates owned by the outer
harness.

## Design Change

`edge-mvp l3 prompt-evolve` is now the L3 prompt evolution entry point. It:

- creates `workspace/l3_prompt/`
- exposes editable `prompt/l3_prompt.json`, `prompt/context_packing.json`, and
  `prompt/routing_guard.md`
- writes protected `contexts/train.jsonl`, `contexts/visible_validation.jsonl`,
  `contexts/task_schema.json`, `contexts/objective.json`, and
  `contexts/local_slm_config.json`
- provides protected workspace tools for prompt validation, visible prompt eval,
  local SLM bench, and latency/cost eval
- stores private selection/promotion holdouts under the outer job `private/`
  directory, not in the workspace
- launches one Codex command with `--cd workspace/l3_prompt`
- scope-checks protected files after the session exits
- snapshots the candidate prompt to `candidates/candidate_l3_prompt.json`
- can run visible validation, private selection, and private promotion replay
  gates through the existing local SLM replay evaluator

`--skip-replay` exists only for smoke/no-model wiring checks. A skipped replay
run cannot select or adopt a candidate.

## Smoke

Command used a fake Codex command and skipped local SLM replay:

```bash
rm -rf runs/l3-agent-session-smoke-r3
mkdir -p runs/l3-agent-session-smoke-r3
L3_AGENT_CODEX_COMMAND=$(pwd)/runs/l3-agent-session-smoke-r1/fake_codex.py \
  uv run edge-mvp l3 prompt-evolve \
    --traces runs/l2-list-fallback-tuned-3k-r1/traces.jsonl \
    --out-dir runs/l3-agent-session-smoke-r3/job \
    --max-traces 40 \
    --skip-replay \
    > runs/l3-agent-session-smoke-r3/stdout.json
```

Result:

- Mode: `agent-session`
- Stop reason: `agent_session_completed`
- Agent sessions: started `1`, succeeded `1`
- Split: train `24`, visible validation `8`, private selection `4`, private
  promotion `4`
- Candidate prompt snapshot: `candidates/candidate_l3_prompt.json`
- Candidate prompt hash: present
- Workspace tools: present, concrete commands, no placeholder replay command
- Tool smoke: `validate_prompt.py` passed; `latency_cost_eval.py` wrote a visible
  validation-only estimate with `private_data_visible=false`
- Workspace private holdout files: absent
- Outer private holdout files: present
- Selection: `selected=false`
- Adoption: `adopted=false`

Conclusion: the L3 workspace/session/scope/snapshot wiring works. Because
`--skip-replay` was used, this is not an L3 quality result.
