# 2026-06-10 L1 agent-session path

## Goal

Validate the L1 homologous agent-session boundary: one L4 coding-agent session
over an isolated workspace root, with editable Rust ProgramBank source,
protected context/program files, scratch `runs/`, scope checking, and outer
replay/promotion as the adoption authority.

## Design Change

`L1_AGENT_MODE` now accepts:

```bash
agent-session
```

This mode:

- creates `workspace/l1_programbank/` as the editable Rust crate surface
- writes protected `workspace/program.md`, `workspace/workspace_manifest.json`,
  and `workspace/contexts/*`
- launches one Codex command with `--cd workspace/`
- gives the agent a short stable prompt that points it to `program.md`
- checks protected workspace files after the session exits and before
  validation
- records `agent_session` and `workspace_scope_policy` in `provenance.json`

Legacy `codex-cli` remains for compatibility; it is not the preferred real L1
evolve path.

## Smoke

A fake Codex command was used to avoid live LLM cost. It wrote one file under
editable `l1_programbank/` and one scratch note under `runs/`. Validation was
disabled for this wiring smoke.

Result:

- Mode: `agent-session`
- Return code: `0`
- Succeeded: `true`
- Editable marker: `workspace/l1_programbank/AGENT_SESSION_MARKER` present
- Scratch note: `workspace/runs/agent_note.txt` present
- Diff summary: 1 changed file, 1 addition
- Agent session policy: `single long-running L4 agent session`
- Internal loop control: `agent_decides_edit_compile_test_bench_replay_stop`
- Writable roots: `l1_programbank/`
- Scratch roots: `runs/`
- Protected roots: `contexts/`, `program.md`, `workspace_manifest.json`

Conclusion: the L1 harness now has an explicit agent-session path with the same
outer shape as L2. This smoke validates wiring and scope metadata only; it is
not a ProgramBank quality result.
