# BFCL Mini User-Journey Repair Runbook

Date: 2026-06-30

This runbook repairs the first-user blockers found by the BFCL mini
user-journey smoke run. The goal is to make the tiny BFCL-shaped journey
discoverable and repeatable without requiring a user to read tests, write a
temporary reference broker, or reconstruct internal dataclasses after the
interactive compile loop closes.

This is still not BFCL Stage 1. Do not optimize BFCL coverage or expand the
benchmark scope while doing this repair.

## Source Evidence

Read these first:

- `AGENTS.md`
- `README.md`
- `docs/experiments/2026-06-30_bfcl_mini_user_journey_report.md`
- `docs/implementation/bfcl_mini_user_journey_smoke_runbook.md`
- `docs/implementation/bfcl_experiment_pre_report.md`
- `docs/implementation/bfcl_stage1_runbook.md`
- `docs/design/reboot/00_overall_design.md`
- `docs/design/reboot/modules/03_agent_workspace.md`
- `docs/design/reboot/modules/05_candidate_evaluation.md`
- `docs/design/reboot/modules/06_release_runtime.md`
- `docs/design/reboot/modules/11_compile_orchestration.md`

If the smoke branch has not been merged yet, use these absolute paths for the
completed run evidence:

```text
/Users/chenmohan/gits/darjeeling-bfcl-mini-user-journey/docs/experiments/2026-06-30_bfcl_mini_user_journey_report.md
/Users/chenmohan/gits/darjeeling-bfcl-mini-user-journey/runs/bfcl-mini-user-journey-20260630-195903/
```

Validate that the issue backlog contains the four P1 issues:

- `BFCL-MINI-001`: no discoverable compile CLI.
- `BFCL-MINI-004`: no documented provider-backed L4/reference cache config.
- `BFCL-MINI-005`: interactive loop does not hand off selected candidate/report
  for final test.
- `BFCL-MINI-006`: default live L4/reference deadline is too low and not
  discoverable.

## Objective

Repair the first-user path so the mini BFCL journey can be rerun with:

- a discoverable target-independent compile entrypoint;
- a documented provider/reference cache configuration path;
- an interactive loop result or persisted handoff that naturally supports final
  test evaluation;
- realistic and configurable L4/reference deadlines for live provider calls;
- cost accounting that clearly labels actual paid spend and estimated spend;
- docs that state the current macOS `sandbox-exec` requirement for
  target-adaptation agent execution.

The expected outcome is not better Precision/Coverage. The expected outcome is
that a technically competent user can repeat the mini journey without reading
internal tests or writing one-off glue code.

## Worktree And Branch

Use a separate branch and worktree.

Suggested setup:

```bash
cd /Users/chenmohan/gits/darjeeling
git status --short
git worktree add ../darjeeling-bfcl-mini-user-journey-repair -b codex/bfcl-mini-user-journey-repair main
cd ../darjeeling-bfcl-mini-user-journey-repair
uv sync --extra dev
```

If the smoke report/runbook files are not tracked on `main`, copy the minimum
needed docs from the completed smoke worktree and record what was copied.

## Scope

In scope:

- Small CLI/API additions that make compile launch discoverable.
- Small config file shapes for reference provider/cache and deadline settings.
- Returning or persisting enough interactive compile state to run final test
  without reconstructing internals.
- Documentation updates for the repaired path.
- Updating the BFCL mini smoke runner only as a verification harness.
- Focused tests for the new public path.

Out of scope:

- BFCL Stage 1.
- Official BFCL evaluator integration.
- Large CLI redesign.
- New plugin systems, runner registries, or provider frameworks.
- Target-specific optimization for BFCL coverage.
- Reintroducing Python portable sandboxing for target-adaptation agents.
- Broad changes to Snapshot, Release Runtime, or Candidate Evaluation unrelated
  to the smoke blockers.

## Repair Items

### 1. Add a discoverable compile entrypoint

Problem: `uv run darjeeling compile --help` currently fails with `No such
command 'compile'`.

Add the smallest useful target-independent CLI entrypoint. Suggested shape:

```text
darjeeling compile run <target-path> \
  --run-root <path> \
  --workspace-root <path> \
  --reference-config <path> \
  --agent-command <json-or-repeatable-arg> \
  --max-candidates <n> \
  --max-agent-seconds <seconds> \
  --max-cost <usd> \
  --enabled-layers L1,L2 \
  --l4-deadline-ms <ms>
```

Keep the first implementation direct. It may call the existing Core functions
without introducing a CLI framework beyond Typer.

Required behavior:

- `darjeeling compile --help` exists.
- `darjeeling compile run --help` explains the minimal required inputs.
- The command writes a run manifest and a concise report under `run-root`.
- If a feature is not implemented yet, the CLI should fail with a useful
  message, not a traceback.

Do not make BFCL-specific assumptions in Core CLI code.

### 2. Provide a documented reference provider/cache config path

Problem: the smoke had to implement a temporary OpenAI-compatible
`ReferenceBroker`.

Add a simple provider/cache config path that a benchmark user can discover and
use. Keep it plain JSON or YAML.

Suggested config shape:

```json
{
  "provider": "openai_compatible",
  "base_url_env": "OPENAI_BASE_URL",
  "api_key_env": "OPENAI_API_KEY",
  "model": "gpt-5.5",
  "timeout_ms": 30000,
  "max_completion_tokens": 2048,
  "price": {
    "input_per_million": 0.0,
    "output_per_million": 0.0
  },
  "cache_path": "reference_cache.jsonl"
}
```

Implementation notes:

- The provider config can live in a small module under Core if it remains
  target-independent and only implements the `ReferenceBroker` protocol.
- Target-specific prompt/request construction must stay in the target reference
  module through `build_reference_request` and `parse_reference_response`.
- Cache keys should include target contract hash, normalized input, prompt or
  request hash, model, base URL identity, and decoding config.
- Cost ledger entries must distinguish provider-reported cost from
  token-estimated cost. If no cost header is present, record
  `cost_status: "estimated-from-token-usage"`.
- Do not under-count failed or timed-out calls if usage/cost is available.

### 3. Close the interactive loop handoff

Problem: `run_interactive_compile_loop` returns counts and ledger paths, but not
the selected/final Candidate and validation Report needed for final test.

Fix this with the simplest stable contract. Acceptable approaches:

- return a typed or plain result containing the last evaluated candidate id,
  selected candidate id, validation report id/path, and evaluated submission
  ledger path; or
- persist a Core-owned `interactive_compile_result.json` next to the evaluated
  submissions ledger with the same information.

Preferred: do both if the implementation stays small. The return value is useful
for in-process callers; the persisted file is useful for CLI and post-run
inspection.

Required behavior:

- A caller can run final test evaluation after the loop closes without
  reconstructing `CandidateSubmission` from internal paths.
- Failed and skipped candidates remain represented in the ledger.
- No raw validation rows, row ids, expected outputs, or reconstructable holdout
  details are written into agent-visible files.
- The final-test handoff is Core-owned, not agent-writable.

### 4. Make L4/reference deadlines realistic and discoverable

Problem: the default `RoutingSettings.total_deadline_ms = 1000` timed out live
reasoning/reference calls during the smoke.

Do not blindly raise every runtime default if that would hurt production latency
semantics. Instead make the live reference path configurable and visible:

- CLI/config should accept an L4/reference deadline override.
- Provider/reference config should include `timeout_ms`.
- The compile report should record the effective timeout used for reference
  cache construction and cold-start fallback checks.
- README or usage docs should warn that live reasoning/reference models usually
  need a larger timeout than local L1/L2/L3 artifacts.

If changing the default is necessary, document the tradeoff and keep local layer
timeouts separate from reference model deadlines where possible.

### 5. Add user-facing docs for the mini path

Update README or a small getting-started doc so a new user can find:

- how to run the toy demo;
- how to run a real compile smoke;
- how to provide a reference provider config;
- how to enable agent guidance and workspace permissions;
- that target-adaptation agent execution currently requires macOS
  `sandbox-exec`;
- where costs and logs are written.

Keep the docs user-facing and short. Link to deeper design docs only for
architecture context.

### 6. Update the BFCL mini smoke verification

After repairs, rerun the mini smoke using the repaired public path instead of
the temporary internal workaround.

Requirements:

- Use the CLI or documented public API path from this repair.
- Use the documented reference provider/cache config path.
- Use live L4/reference calls when credentials are available, capped at `$10`.
- Maintain a cost ledger even if spend is `$0`.
- Reach final test without reconstructing internals.
- Confirm L1/L2 enabled, L3 disabled.
- Confirm no raw validation/test data appears in agent-visible files.
- Record whether each previous P1 is fixed.

## Additional Issues To Track

The smoke did not expose more P0/P1 blockers, but these P2 items should be
tracked while implementing the repairs:

- Cost estimates rely on a local/model price table when the gateway returns no
  cost header. The docs should make this explicit and configurable.
- Target-adaptation agent execution currently requires macOS `sandbox-exec`.
  Unsupported platforms should fail clearly and the docs should say so.
- The mini smoke runner is useful as a reproducibility harness, but it should
  not become the long-term product interface once `darjeeling compile run`
  exists.

## Validation

Run at least:

```bash
uv run darjeeling --help
uv run darjeeling compile --help
uv run darjeeling compile run --help
uv run darjeeling demo thin-target
uv run --with pytest pytest tests -q
uv run --with ruff ruff check src tests experiments
git diff --check
```

Also rerun the BFCL mini smoke through the repaired public path and write a new
tracked report:

```text
docs/experiments/<YYYY-MM-DD>_bfcl_mini_user_journey_repair_report.md
```

The report must include:

- run root path;
- command used;
- paid API spend;
- cost ledger path;
- whether final test was reached without internal reconstruction;
- fixed/not-fixed status for `BFCL-MINI-001`, `004`, `005`, and `006`;
- remaining P1/P2 issues;
- recommendation on whether BFCL Stage 1 can start.

## Done Criteria

This repair is complete when:

- all four P1 issues are fixed or explicitly downgraded with evidence;
- the mini smoke reaches final test through a documented public path;
- paid cost is tracked correctly and remains within the configured cap;
- docs explain the path well enough that a first-time user does not need to
  read internal tests;
- tests and lint pass;
- tracked changes are committed;
- ignored run artifacts remain available for inspection.
