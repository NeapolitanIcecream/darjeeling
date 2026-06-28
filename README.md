# Darjeeling

Darjeeling helps you run cheaper local artifacts in front of a strong reference
model. A local artifact can answer only when it is confident; otherwise
Darjeeling falls back to the reference model.

You define the task boundary. Darjeeling keeps that target-specific logic out of
the framework, then handles validation, evaluation, routing, fallback, tracing,
and the data needed to improve artifacts later.

## What You Provide

A target directory defines one task:

- `target.yaml` for metadata, requirements, and paths
- `schemas/input.json` and `schemas/output.json` for request and response shapes
- `contract.py` for target-owned validation, correctness, grouping, and redaction
- optional `reference.py` for adapting a reference model response
- optional `data.yaml` and `tests/` for target data and contract checks

Darjeeling treats target inputs, labels, outputs, and business rules as opaque
data.

## Install

```bash
uv sync --extra dev
```

## Check A Target

Validate a target directory:

```bash
uv run darjeeling target check /path/to/target
```

Require a reference adapter during the check:

```bash
uv run darjeeling target check /path/to/target --require-reference
```

The target directory contract is documented in
`docs/design/reboot/modules/01_target_definition.md`.

## Repository Map

- `src/darjeeling/`: active framework implementation
- `tests/`: active test suite
- `docs/design/reboot/`: architecture design documents

## Current Status

This repository contains a filesystem/in-memory implementation of the
architecture. It is suitable for development and design validation.

Production hardening remains future work:

- persistent stores and durable queues
- OS-portable resource-limit adapters
- real external reference broker integrations

Run the checks:

```bash
uv run --with pytest pytest tests -q
uv run --with ruff ruff check src tests
```
