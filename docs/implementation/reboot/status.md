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

- `uv run --with pytest pytest tests -q`: 162 passed.
- `uv run --with ruff ruff check src tests`: passed.

## Deferred Items

No design-owned module requirement is intentionally deferred in this implementation slice. Production hardening remains future work: persistent databases, durable queues, long-running worker pools, OS-portable resource-limit adapters beyond the local `sandbox-exec` boundary, and real external reference broker integrations.
