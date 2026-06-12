# darjeeling

Profile-guided edge intelligence runtime MVP for the NLU replay demo described in
[docs/mvp_demo_proposal.md](docs/mvp_demo_proposal.md).

## Design

- [MVP demo proposal](docs/mvp_demo_proposal.md)
- [Module-level design](docs/design/README.md)

## Setup

```bash
uv sync --extra dev
```

The MASSIVE adapter dependencies are optional. Install them only when you need
`edge-mvp-nlu massive prepare`:

```bash
uv sync --extra dev --extra massive
```

`datasets` is pinned below 4.0 in the `massive` extra because the MASSIVE
adapter still loads `AmazonScience/massive` through its dataset script.

Create a `.env` from `.env.example` and set `OPENAI_API_KEY` unless the selected
run already has a complete teacher cache.

Optional run settings can live in `settings.yaml`. Environment variables and
`.env` override the YAML file. To select a file explicitly, put `--settings`
before the subcommand:

```bash
uv run edge-mvp --settings settings.yaml experiment preflight --run-dir runs/latest
```

## Smoke run

```bash
uv sync --extra dev --extra massive
uv run edge-mvp-nlu massive prepare --locale en-US
uv run edge-mvp experiment preflight --run-dir runs/latest --teacher live-or-cache
uv run edge-mvp run --stream zipf-heavy --max-requests 3000 --compile-every 500 --teacher live-or-cache
uv run edge-mvp report --run-dir runs/latest
uv run pytest
```

## Real L1 evolution run

The default config keeps `L1_AGENT_MODE=disabled`, so smoke runs do not launch
Codex CLI. For a real L1 evolution experiment, enable the coding-agent harness:

```bash
L1_AGENT_MODE=codex-cli L4_PROPOSAL_MODE=live \
  uv run edge-mvp experiment main-evolution --run-dir runs/main --teacher live-or-cache
```

Run preflight first. If `l1.agent` is `warn`, L1 evolution is still disabled; if
it is `fail`, the configured `codex` command or dry-run patch is missing.

The project is intentionally fail-fast for the main demo: if `OPENAI_API_KEY` is absent and no
teacher cache exists for the selected run directory, `edge-mvp run` exits instead of fabricating
labels.
