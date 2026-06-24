# darjeeling

Darjeeling is a profile-guided edge intelligence runtime MVP. It explores how a
strong cloud model, L4, can externalize stable parts of a workload into cheaper
local layers while preserving quality:

```text
L0 cache -> L1 native ProgramBank -> L2 trainable student -> L3 local SLM -> L4
```

The original demo target is NLU. The current Phase 1 benchmark is CLINC150
`data_full`; MASSIVE remains available as a historical NLU adapter and comparison
point.

## Design

- [MVP demo proposal](docs/mvp_demo_proposal.md)
- [Module-level design](docs/design/README.md)
- [Experiment index](docs/experiments/README.md)

Current design state:

- Darjeeling core is target-independent. NLU schemas, prompts, labels, CLINC150,
  MASSIVE, intent/slot diagnostics, and target-specific L1/L2/L3 logic stay under
  `src/darjeeling/targets/nlu` or experiment artifacts.
- L1, L2, and L3 share a small outer evolution policy: `max_rounds`,
  per-round timeout, patience, round executor, and round result summaries. Core
  does not declare target quality claims or private gate semantics.
- Generated L1/L2/L3 artifacts may be large or target-specific. They still must
  pass target-owned evaluation and outer replay/promotion gates before adoption.

## Setup

```bash
uv sync --extra dev
```

Dataset adapter dependencies are optional. Install them when you need CLINC150
or MASSIVE dataset preparation:

```bash
uv sync --extra dev --extra massive
```

The `massive` extra currently carries shared dataset tooling such as `pandas`,
`pyarrow`, and Hugging Face `datasets`. `datasets` is pinned below 4.0 because
the MASSIVE adapter still loads `AmazonScience/massive` through its dataset
script.

Create a `.env` from `.env.example` and set `OPENAI_API_KEY` unless the selected
run already has a complete teacher cache.

Optional run settings can live in `settings.yaml`. Environment variables and
`.env` override the YAML file. To select a file explicitly, put `--settings`
before the subcommand:

```bash
uv run edge-mvp-nlu --settings settings.yaml experiment preflight --run-dir runs/latest
```

## CLINC150 local setup

Prepare the current Phase 1 dataset:

```bash
uv sync --extra dev --extra massive
uv run edge-mvp-nlu clinc150 prepare --out data/processed/clinc150_data_full
```

The CLINC150 source is pinned and checksum-verified in
`src/darjeeling/targets/nlu/adapters/clinc150.py`.

Recent CLINC150 status:

- The repaired L4 teacher gate passed with `clinc150-intent-v2-label-cards`.
- Teacher-distilled L2 is promising but not adopted: validation passed at
  threshold `0.98`, locked test accepted precision was `98.77%`, below the
  `99%` gate.
- The latest L1 ProgramBank run failed locked test after a strong validation
  result: validation accepted precision `100.00%` at `60.35%` coverage, locked
  test accepted precision `92.73%`.

See [docs/experiments/README.md](docs/experiments/README.md) for the current
experiment map.

## MASSIVE smoke run

```bash
uv sync --extra dev --extra massive
uv run edge-mvp-nlu massive prepare --locale en-US
uv run edge-mvp-nlu experiment preflight --run-dir runs/latest --teacher live-or-cache
uv run edge-mvp run --stream zipf-heavy --max-requests 3000 --compile-every 500 --teacher live-or-cache
uv run edge-mvp report --run-dir runs/latest
uv run pytest
```

## Real L1 evolution run

The default config keeps `L1_AGENT_MODE=disabled`, so smoke runs do not launch
Codex CLI. For a real L1 evolution experiment, enable the coding-agent harness.
Use `agent-session` for current experiments; `codex-cli` remains available for
older paths and tests:

```bash
L1_AGENT_MODE=agent-session L4_PROPOSAL_MODE=live \
  uv run edge-mvp-nlu experiment main-evolution --run-dir runs/main --teacher live-or-cache
```

Run preflight first. If `l1.agent` is `warn`, L1 evolution is still disabled; if
it is `fail`, the configured `codex` command or dry-run patch is missing.

The project is intentionally fail-fast for the main demo: if `OPENAI_API_KEY` is absent and no
teacher cache exists for the selected run directory, `edge-mvp run` exits instead of fabricating
labels.
