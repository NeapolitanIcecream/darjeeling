# BFCL Mini User-Journey Repair Report

Date: 2026-06-30
Branch: `codex/bfcl-mini-user-journey-repair`
Worktree: `/Users/chenmohan/gits/darjeeling-bfcl-mini-user-journey-repair`

## Decision

The BFCL mini first-user journey is repaired for the four P1 smoke blockers. A
first user can now discover and run the compile path from the CLI and README,
configure an OpenAI-compatible reference provider/cache without reading internal
tests, let the interactive compile loop persist a Core-owned handoff record, and
reach final test through the public `darjeeling compile run` command.

This is not BFCL Stage 1 and does not claim BFCL coverage improvement. It only
repairs the mini user journey required before a larger BFCL experiment.

## Run

Repaired smoke command:

```bash
uv run python experiments/bfcl_mini_user_journey/run_smoke.py \
  --run-root runs/bfcl-mini-user-journey-repair-20260630-205358 \
  --max-actual-api-cost 10 \
  --live-reference auto
```

Run root:
`/Users/chenmohan/gits/darjeeling-bfcl-mini-user-journey-repair/runs/bfcl-mini-user-journey-repair-20260630-205358`

Smoke summary:
`/Users/chenmohan/gits/darjeeling-bfcl-mini-user-journey-repair/runs/bfcl-mini-user-journey-repair-20260630-205358/reports/smoke_summary.json`

Cost ledger:
`/Users/chenmohan/gits/darjeeling-bfcl-mini-user-journey-repair/runs/bfcl-mini-user-journey-repair-20260630-205358/reference_usage_ledger.json`

Actual paid spend recorded: `$0.019230`

The first live attempt made nine provider calls and exposed a smoke-runner
sandbox issue: the agent command pointed at `agent.py` under the run root, which
the macOS sandbox correctly denied from the isolated attempt workspace. The
runner now writes the same tiny smoke agent to an absolute script path and passes
that script through the narrowed `--agent-command` contract. The successful rerun
used the same run root and cache, added nine cache-hit ledger entries, and did
not increase paid spend.

## P1 Status

| ID | Status | Evidence |
| --- | --- | --- |
| `BFCL-MINI-001` | fixed | `darjeeling --help` exposes `compile`; `darjeeling compile --help` exposes `run`; `darjeeling compile run --help` documents required options. |
| `BFCL-MINI-004` | fixed | `ReferenceProviderConfig` loads a small JSON/YAML `openai_compatible` config, creates an implementation of the existing `ReferenceBroker` protocol, writes a cache, and maintains a usage ledger. Target-specific prompt/request/parse remains in target `reference.py`. |
| `BFCL-MINI-005` | fixed | `run_interactive_compile_loop` persists selected candidate and validation-report records under Core-owned attempt state and writes `interactive_compile_result.json`; `darjeeling compile run` uses that selected in-process handoff for final test. |
| `BFCL-MINI-006` | fixed | Public CLI exposes `--l4-deadline-ms`; default is `30000`; repaired smoke used `30000` for cold-start and compile routing. |

## Smoke Result

- Reference mode: live OpenAI-compatible provider through config.
- Final test reached: yes.
- Decision status: `eligible_for_release`.
- Decision blockers: none.
- Enabled layers: L1 and L2; L3 intentionally disabled for mini smoke.
- Agent-visible leakage scan: 0 suspect files.
- Paid spend: `$0.019230`, under the `$10` cap.
- Ledger totals: 9 provider calls, 9 cache hits.

## Verification

Passed:

```bash
uv run darjeeling --help
uv run darjeeling compile --help
uv run darjeeling compile run --help
uv run darjeeling demo thin-target
uv run python experiments/bfcl_mini_user_journey/run_smoke.py --run-root runs/bfcl-mini-user-journey-repair-20260630-205358 --max-actual-api-cost 10 --live-reference auto
uv run --with pytest pytest tests/test_02_snapshot_reference.py::test_openai_compatible_reference_config_writes_cache_and_usage_ledger tests/test_10_compile_orchestration.py::test_interactive_compile_loop_writes_feedback_while_agent_is_running tests/test_12_demo_cli.py::test_compile_cli_help_is_discoverable -q
uv run --with pytest pytest tests -q
uv run --with ruff ruff check src tests experiments
```

Full pytest result: 204 passed in 157.79s.

## Remaining Issues

- This run is still a mini user-journey smoke, not BFCL Stage 1 evidence.
- Reference spend is provider-reported when headers are available; otherwise it
  is estimated from configured token prices and provider usage.
- Target-adaptation agent execution remains macOS-only through
  `sandbox-exec`. Unsupported platforms should keep failing clearly until an
  explicit external runner/container design exists.
- The compile CLI is intentionally direct and low-abstraction. It is suitable
  for first-user repeatability but is not a broader provider framework or runner
  registry.
