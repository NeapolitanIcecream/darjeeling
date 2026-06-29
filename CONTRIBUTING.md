# Contributing

Darjeeling is in alpha. The most useful contributions right now are demo
feedback, target use cases, small documentation fixes, and focused runtime
checks that keep the target/core boundary clear.

## Start Here

1. Read `README.md`.
2. Run the demo:

   ```bash
   uv sync --extra dev
   uv run darjeeling demo thin-target
   ```

3. Run the checks:

   ```bash
   uv run --with pytest pytest tests -q
   uv run --with ruff ruff check src tests
   ```

## Target And Core Boundary

Keep Darjeeling core target-, dataset-, and application-independent. Target
schema names, request fields, labels, dataset fields, and business logic belong
in target packages, adapters, examples, or demo targets.

Use ordinary Python objects and explicit registries before adding abstractions.
Do not add plugin systems, dependency-injection containers, or schema DSLs
unless a maintainer asks for them.

## Pull Requests

- Keep changes scoped to one behavior or documentation task.
- Add tests when the change affects a stable public contract.
- For documentation-only changes, run at least the relevant command shown in the
  doc when practical.
- Do not commit generated runs, caches, local datasets, or API keys.
- Explain any skipped validation in the pull request description.

## Issues

Use the GitHub issue forms for demo feedback and use cases. For security
reports, follow `SECURITY.md` instead of opening a public issue with sensitive
details.
