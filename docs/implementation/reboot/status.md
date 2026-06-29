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

- `uv run --with pytest pytest tests -q`: 163 passed.
- `uv run --with ruff ruff check src tests`: passed.

## Deferred Items

The interactive compile loop is not fully implemented yet. The current code has
the workspace, candidate submission, validation evaluation, and agent-safe
feedback pieces, but the high-level compile launch does not keep one agent
session alive while Core evaluates submissions and writes feedback back into the
workspace. See
`docs/implementation/reboot/interactive_compile_loop_runbook.md`.

Production hardening remains future work: persistent databases, durable queues,
long-running worker pools, OS-portable resource-limit adapters beyond the
current local sandbox boundary, and real external reference broker integrations.
