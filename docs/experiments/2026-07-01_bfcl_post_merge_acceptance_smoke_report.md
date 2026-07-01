# BFCL Post-Merge Acceptance Smoke Report

Date: 2026-07-01
Branch: `codex/bfcl-post-merge-acceptance-smoke`
Worktree: redacted local worktree path
Base commit: `dd8cbfaa9e61f4c355bc0db7cbf0e6b26145320f`
Final commit: reported by the completing session after this report is committed.

This run verified the BFCL mini public user path on `origin/main` after PR #3
merged. It was an acceptance smoke only. It did not run BFCL Stage 1, optimize
BFCL precision or coverage, or expand provider, command, or sandbox frameworks.

The post-merge acceptance runbook was local-only in the original checkout when
this smoke was run; the independent worktree was created from `origin/main` and
kept clean except for this tracked report and ignored `runs/` artifacts.

## Decision

Acceptance passed.

The public path was discoverable, deterministic no-paid smoke completed, live
reference smoke completed under budget, same-cache replay avoided additional
paid spend, compile wrote validation handoff and final reports, and agent-visible
holdout leakage checks did not find raw validation/test rows, holdout row ids,
expected outputs, or reference cache records.

## Run Roots

- Local deterministic run:
  `runs/bfcl-post-merge-acceptance-local-20260701-143328`
- Live reference run and replay:
  `runs/bfcl-post-merge-acceptance-live-20260701-143358`
- Live cost ledger:
  `runs/bfcl-post-merge-acceptance-live-20260701-143358/reference_usage_ledger.json`
- Live compile handoff:
  `runs/bfcl-post-merge-acceptance-live-20260701-143358/workspaces/bfcl-mini/attempts/<compile-id>/_core/<attempt-id>/interactive_compile_result.json`

## Commands Run

Setup:

```bash
git fetch origin main
git worktree add ../darjeeling-bfcl-post-merge-acceptance-smoke \
  -b codex/bfcl-post-merge-acceptance-smoke origin/main
cd ../darjeeling-bfcl-post-merge-acceptance-smoke
uv sync --extra dev
```

CLI discovery:

```bash
uv run darjeeling --help
uv run darjeeling compile --help
uv run darjeeling compile run --help
uv run darjeeling demo thin-target
```

Deterministic no-paid smoke:

```bash
uv run python experiments/bfcl_mini_user_journey/run_smoke.py \
  --run-root runs/bfcl-post-merge-acceptance-local-20260701-143328 \
  --live-reference disabled \
  --max-actual-api-cost 1
```

Live reference smoke:

```bash
uv run python experiments/bfcl_mini_user_journey/run_smoke.py \
  --run-root runs/bfcl-post-merge-acceptance-live-20260701-143358 \
  --live-reference required \
  --max-actual-api-cost 10
```

Same-cache replay:

```bash
uv run python experiments/bfcl_mini_user_journey/run_smoke.py \
  --run-root runs/bfcl-post-merge-acceptance-live-20260701-143358 \
  --live-reference required \
  --max-actual-api-cost 10
```

Artifact inspection and leakage checks used `jq`, `find`, and `rg` against the
run roots listed above.

## CLI Discovery

Passed.

- `darjeeling --help` exposes `target`, `demo`, and `compile`.
- `darjeeling compile --help` exposes `run`.
- `darjeeling compile run --help` exposes `--run-root`,
  `--reference-config`, `--agent-command`, `--max-cost`,
  `--enabled-layers`, and `--l4-deadline-ms`.
- `--agent-command` help says it accepts a JSON array containing an absolute
  executable, or a common interpreter with an absolute script path, and tells
  users to use an absolute wrapper script for complex commands.
- `darjeeling demo thin-target` completed with simulated reference fallback and
  local artifact routing.

## Deterministic No-Paid Smoke

Passed.

- Exit status: 0.
- Reference mode: `local-openai-compatible`.
- Reports present:
  - `compile/reports/compile_summary.json`
  - `compile/reports/test_report.json`
  - `compile/reports/final_report.json`
- Selected candidate: present.
- Selected validation report: present.
- Final decision: `eligible_for_release`.
- L4/reference deadline: `30000ms`.
- Enabled layers: `L1,L2`; L3 was disabled for this smoke.
- Reference ledger totals: 9 provider calls, 0 cache hits, `$0.000000` paid
  spend.

## Live Reference Smoke

Passed.

- Exit status: 0.
- Reference mode: `live`.
- Model: configured live reference model.
- First live run selected candidate: present.
- First live run selected validation report: present.
- First live run final decision: `eligible_for_release`.
- First live run ledger totals: 9 provider calls, 0 cache hits,
  `$0.018060` estimated paid spend.
- Each provider entry included model, usage, latency, status, and cost status.
- Cost status was `estimated-from-token-usage`, using the configured token
  prices because no provider cost header was available.
- Spend remained below the `$10` hard cap.
- The smoke runner invoked the public `darjeeling compile run` entrypoint.

## Cache Replay

Passed.

Rerunning the same command with the same live run root exited successfully.
The cumulative live ledger after replay showed:

```json
{
  "actual_paid_api_cost_usd": 0.018059999999999996,
  "cache_hit_count": 9,
  "provider_call_count": 9
}
```

The replay added 9 cache hits and did not increase paid spend.

The replay compile summary selected a candidate, recorded a validation report,
wrote a final test report, and wrote a final report.

## Artifact Inspection

Passed.

Inspected artifacts included:

- `manifest.json`
- `reference_config.json`
- `reference_usage_ledger.json`
- `reference_cache.jsonl`
- `compile/manifest.json`
- `compile/reports/compile_summary.json`
- `compile/reports/test_report.json`
- `compile/reports/final_report.json`
- `compile/workspaces/`

Findings:

- `compile/reports/compile_summary.json` contains selected candidate id,
  selected validation report id, final test report id, final report id,
  reference timeout, effective L4 deadline, and the interactive handoff path.
- `interactive_compile_result.json` contains selected candidate path,
  selected validation report path, evaluated submissions ledger path,
  feedback count, evaluated submission count, and `stop_reason:
  candidate_limit`.
- The final report decision is `eligible_for_release` with no release blockers.
- L1/L2 are enabled and L3 is disabled in compile manifests and final report
  routing evidence.
- The live reference deadline is `30000ms`.
- The agent sandbox file denies read and write access to
  `reference_cache.jsonl` and `reference_usage_ledger.json`.

## Leakage Checks

Passed.

The smoke runner's `leakage_scan` reported no suspect files. Manual scans of
the live replay agent workspace found:

- no validation/test row ids such as `mini-val-*` or `mini-test-*`;
- no holdout prompts such as `What is the weather in Berlin?`,
  `Write a short poem.`, or `Summarize this sentence.`;
- no raw holdout expected outputs such as Paris or Berlin tool-call arguments;
- no reference cache entries, provider response records, cache keys, cache-hit
  records, provider-call records, or raw provider messages/choices/usage.

The only broad-scan hits were intentionally visible redacted training rows and
the sandbox deny rules protecting the reference cache and ledger paths. The
agent-visible feedback file contained aggregate validation metrics and
`raw_rows_included: false`; it did not contain row-level keys.

## Issues And Fixes

No acceptance-blocking issues were found.

No code fixes were made. No tests were added because the acceptance path was not
blocked and the only tracked change is this report.

Residual notes:

- This remains a mini acceptance smoke, not BFCL Stage 1 evidence.
- The runbook file used for this post-merge acceptance smoke is local-only in
  the original checkout, not tracked on `origin/main`.
- The live spend is estimated from token usage and configured prices because the
  provider did not return a cost header.

## Validation Commands

Report-only validation passed after writing this report:

```bash
uv run darjeeling compile run --help
uv run python experiments/bfcl_mini_user_journey/run_smoke.py --help
git diff --check
```

The CLI discovery, deterministic smoke, live smoke, replay, artifact
inspection, cost ledger inspection, and leakage scans above also served as the
acceptance validation for this report.
