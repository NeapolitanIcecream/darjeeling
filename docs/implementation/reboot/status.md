# Implementation Status

## Active Implementation

Active architecture guidance remains under `docs/design/reboot/`.

The active implementation is under `src/darjeeling/` and is written from the
architecture docs. It uses plain dataclasses, functions, in-memory registries,
and filesystem-backed immutable snapshot/artifact/workspace stores.

## Active Surface

- Shared model and utility: `src/darjeeling/model.py`, `src/darjeeling/errors.py`, `src/darjeeling/util.py`.
- Modules: `target_definition.py`, `snapshot_reference.py`, `artifact_worker.py`, `agent_workspace.py`, `candidate_evaluation.py`, `release_runtime.py`, `runtime_trace_metrics.py`, `audit_monitoring.py`, `telemetry_recompile.py`, `compile_orchestration.py`.
- CLI: `src/darjeeling/cli.py`.
- Tests: `tests/test_01_target_definition.py` through `tests/test_11_end_to_end_thin_target.py`.

## Verification

- `uv run pytest tests -q`: 198 passed.
- `uv run ruff check src tests`: passed.
- `git diff --check`: passed.

## Implemented Interactive Compile Loop

The interactive compile loop runbook has been implemented in the active code
surface. `compile_orchestration.run_interactive_compile_loop` keeps one async
agent session alive while Core polls `READY` submissions, deduplicates
candidates through Core-owned state, evaluates validation candidates, writes
agent-safe feedback under `journal/feedback-<candidate>.json`, enforces time,
cost, user-stop, and candidate budgets, and closes the attempt with a terminal
session record. `agent_workspace.launch_target_adaptation_agent_async` provides
the non-blocking session launch and durable Core session records used by that
driver.

See `docs/implementation/reboot/interactive_compile_loop_runbook.md` for the
implementation checklist and boundary rationale.

## Deferred Items

Production hardening remains future work: persistent databases, durable queues,
long-running worker pools, OS-portable resource-limit adapters beyond the
current local sandbox boundary, and real external reference broker integrations.
