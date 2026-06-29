# Darjeeling

Darjeeling helps LLM apps get faster and cheaper by turning well-bounded model
capabilities into local code and small models.

Local artifacts answer only when they are inside a checked reliability boundary.
Hard or unfamiliar requests still fall back to the main LLM, so the runtime can
reduce latency and inference cost without forcing local artifacts to guess.

## Five-Minute Demo

Run the no-network toy demo from a checkout:

```bash
uv sync --extra dev
uv run darjeeling demo thin-target
```

The demo creates a temporary toy target, starts with the simulated reference LLM
path, promotes a local artifact for known-safe requests, falls back for an
unfamiliar request, and prints precision, coverage, latency, fallback share, and
estimated saving. It uses toy data and does not spend API credits.

## Why This Exists

Many LLM products contain structured behavior that can become reliable outside
the main model. Darjeeling provides the runtime and evaluation loop for moving
that behavior into local artifacts while keeping fallback, validation, tracing,
and recompile paths explicit.

Darjeeling is not a cache. A cache can reuse an exact previous answer.
Darjeeling can route a new request to a local artifact when that artifact has
earned a reliability boundary for the target. If the artifact refuses, the main
LLM path still handles the request.

## Current Status

Darjeeling is an alpha runtime for design validation and early demo feedback.
The current stores and queues are filesystem or in-memory implementations.
Production hardening remains future work.

Current non-goals:

- production readiness;
- benchmark leadership claims;
- replacing hosted LLMs;
- broad claims about arbitrary open-ended chat workloads.

The clearest initial fit is structured LLM workflows where local artifacts can
safely accept some requests and fall back on the rest.

## Install

From a checkout:

```bash
git clone https://github.com/NeapolitanIcecream/darjeeling.git
cd darjeeling
uv sync --extra dev
```

The Python import package is still:

```python
import darjeeling
```

The CLI command is still:

```bash
darjeeling
```

The PyPI distribution name is `darjeeling-ai` because `darjeeling` is already
used by an unrelated package. After the alpha package is published, install it
with prereleases enabled:

```bash
uvx --from darjeeling-ai --prerelease allow darjeeling demo thin-target
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

A target directory defines one task:

- `target.yaml` for metadata, requirements, and paths;
- `schemas/input.json` and `schemas/output.json` for request and response
  shapes;
- `contract.py` for target-owned validation, correctness, grouping, and
  redaction;
- optional `reference.py` for adapting a reference model response;
- optional `data.yaml` and `tests/` for target data and contract checks.

Darjeeling treats target inputs, labels, outputs, and business rules as opaque
data. Target-specific parsing and business logic belong in target packages or
adapters, not in the core runtime.

The target directory contract is documented in
`docs/design/reboot/modules/01_target_definition.md`.

## How It Works

At a high level, a Darjeeling target moves through this loop:

1. Define the task boundary, schemas, contract checks, and reference path.
2. Build a snapshot from target-owned examples and approved runtime evidence.
3. Evaluate a candidate local artifact against validation and test splits.
4. Release the artifact only when it satisfies the target's quality gates.
5. Route runtime requests through local artifacts first and fall back when no
   local artifact accepts.
6. Record traces and approved feedback so later compile runs can improve the
   target.

Internal documents use layer names for implementation work. Local artifact
layers are named L1, L2, and L3; the fallback reference path is named L4. Those
names are useful when editing the runtime, but users should understand the
product first as local-when-safe with fallback-when-needed.

## Repository Map

- `src/darjeeling/`: active framework implementation
- `tests/`: active test suite
- `docs/design/reboot/`: architecture design documents
- `docs/releases/`: release notes
- `docs/launch/`: launch checklists and GitHub-side setup

## Validate The Repository

Run the local checks:

```bash
uv run --with pytest pytest tests -q
uv run --with ruff ruff check src tests
uv run darjeeling demo thin-target
```

CI runs the same test, lint, and demo checks on Python 3.11 and 3.12.
