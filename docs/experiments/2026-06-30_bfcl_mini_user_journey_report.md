# BFCL Mini User-Journey Smoke Report

Date: 2026-06-30
Branch: `codex/bfcl-mini-user-journey`
Worktree: `/Users/chenmohan/gits/darjeeling-bfcl-mini-user-journey`
Run root: `/Users/chenmohan/gits/darjeeling-bfcl-mini-user-journey/runs/bfcl-mini-user-journey-20260630-195903`

This was a first-user smoke run on a tiny BFCL-shaped task. It was not BFCL
Stage 1 and did not attempt coverage optimization.

## Decision

Do not start BFCL Stage 1 yet. The core primitives can complete a mini journey,
but the first-user path still requires internal Python knowledge for compile
launch, provider/reference setup, and final-test handoff after the interactive
loop closes.

## Journey Result

- README setup succeeded with `uv sync --extra dev`.
- `darjeeling demo thin-target` succeeded, but it is toy-only and simulated.
- No compile CLI exists: `uv run darjeeling compile --help` reports `No such command 'compile'`.
- A tiny BFCL-shaped target was generated under the ignored run root.
- Live L4/reference calls ran through the configured OpenAI-compatible gateway using `gpt-5.5`.
- L1/L2 routing was enabled and L3 was disabled.
- One sandboxed target-adaptation agent session submitted `submissions/c1/READY`.
- Core wrote one aggregate validation feedback file.
- Final test evaluation was reached through an orchestrator-side replay workaround.
- No raw validation/test row ids or prompts were found in agent-visible files.

## Cost And Usage

Cost ledger: `/Users/chenmohan/gits/darjeeling-bfcl-mini-user-journey/runs/bfcl-mini-user-journey-20260630-195903/usage_ledger.json`

- Budget cap: `$10.00`.
- Actual paid API spend recorded: `$0.02345`.
- Target-agent API spend: `$0.00`.
- Reference/cache API spend: `$0.02345`.
- L4/reference entries: 10.
- Pricing status: estimated from token usage because the gateway did not return a cost header.
- Reference model/path: `gpt-5.5` via the configured OpenAI-compatible base URL.

## Smoke Metrics

- Train/validation/test counts: 4 / 2 / 2.
- Candidate count: 1.
- Feedback count: 1.
- Stop reason: `candidate_limit`.
- Final decision: `eligible_for_release` on the tiny smoke sample.
- Validation local coverage: 50%.
- Validation accepted precision: 100% on 1 accept.
- Test local coverage: 50%.
- Test accepted precision: 100% on 1 accept.
- Test fallback share: 50%.
- Wrong accepts: 0.

These metrics only prove that the mini journey can execute. They are not BFCL
benchmark evidence.

## Top P0/P1 Issues

1. `BFCL-MINI-001` P1 CLI: no discoverable compile CLI. A user must inspect tests
   or internal Python modules to launch the compile loop.
2. `BFCL-MINI-004` P1 cost/config: no built-in provider-backed L4/reference
   cache configuration is discoverable. The smoke had to implement a temporary
   OpenAI-compatible `ReferenceBroker`.
3. `BFCL-MINI-005` P1 compile: `run_interactive_compile_loop` closes the attempt
   but does not return the selected `Candidate` or validation `Report`, so final
   test cannot be reached through the public return value.
4. `BFCL-MINI-006` P1 config: the default `RoutingSettings.total_deadline_ms`
   is 1000ms, which timed out live reasoning/reference calls during debugging.
   The final run required an explicit 30000ms L4 deadline.

No P0 remained after fallback/workaround: the final mini journey reached final
test.

## Fixes Made In This Run

- Added a tiny tracked smoke runner:
  `experiments/bfcl_mini_user_journey/run_smoke.py`.
- Copied local BFCL planning/runbook docs into the experiment branch so this
  worktree carries its execution context.
- No Darjeeling core behavior was changed.

## Commands Run

- `uv sync --extra dev`
- `uv run darjeeling --help`
- `uv run darjeeling demo --help`
- `uv run darjeeling compile --help`
- `uv run darjeeling target --help`
- `uv run darjeeling demo thin-target`
- `uv run python experiments/bfcl_mini_user_journey/run_smoke.py --run-root /Users/chenmohan/gits/darjeeling-bfcl-mini-user-journey/runs/bfcl-mini-user-journey-20260630-195903 --max-actual-api-cost 10 --live-reference auto`

## Validation

- `uv run darjeeling target check /Users/chenmohan/gits/darjeeling-bfcl-mini-user-journey/runs/bfcl-mini-user-journey-20260630-195903/target --require-reference`
  passed.
- `uv run --with ruff ruff check src tests experiments` passed.
- `uv run python experiments/bfcl_mini_user_journey/run_smoke.py --run-root /Users/chenmohan/gits/darjeeling-bfcl-mini-user-journey/runs/bfcl-mini-user-journey-verify --max-actual-api-cost 10 --live-reference disabled`
  passed as a no-paid deterministic replay of the smoke runner.
- `uv run --with pytest pytest tests -q` passed: 202 tests.
- `git diff --check` passed.

## Artifacts

- Run manifest: `/Users/chenmohan/gits/darjeeling-bfcl-mini-user-journey/runs/bfcl-mini-user-journey-20260630-195903/manifest.json`
- Command log: `/Users/chenmohan/gits/darjeeling-bfcl-mini-user-journey/runs/bfcl-mini-user-journey-20260630-195903/command_log.md`
- Friction log: `/Users/chenmohan/gits/darjeeling-bfcl-mini-user-journey/runs/bfcl-mini-user-journey-20260630-195903/friction_log.md`
- Issue backlog: `/Users/chenmohan/gits/darjeeling-bfcl-mini-user-journey/runs/bfcl-mini-user-journey-20260630-195903/issue_backlog.jsonl`
- User-journey report: `/Users/chenmohan/gits/darjeeling-bfcl-mini-user-journey/runs/bfcl-mini-user-journey-20260630-195903/user_journey_report.md`
- Final test report JSON: `/Users/chenmohan/gits/darjeeling-bfcl-mini-user-journey/runs/bfcl-mini-user-journey-20260630-195903/reports/final_test_report.json`
- Agent brief: `/Users/chenmohan/gits/darjeeling-bfcl-mini-user-journey/runs/bfcl-mini-user-journey-20260630-195903/workspaces/bfcl-mini/attempts/compile-db5dc286133344b1/attempt-92f8295e6d324ae7/AGENT_BRIEF.md`

## Recommendation

Repair the first-user path before BFCL Stage 1:

1. Add or document a target-independent compile entrypoint.
2. Add a documented provider/reference cache configuration path with budget
   accounting.
3. Make the interactive loop return or persist the selected candidate and
   validation report needed for final test.
4. Expose realistic L4/reference deadline configuration in user-facing setup.

After those are repaired, rerun this mini smoke once before launching the
500-800 row Stage 1 pilot.
