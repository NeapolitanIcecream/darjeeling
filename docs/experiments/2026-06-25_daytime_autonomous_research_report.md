# 2026-06-25 Daytime Autonomous Research Report

Branch/worktree:

- Branch: `codex/daytime-autonomous-research-20260625`
- Worktree: `/Users/chenmohan/gits/darjeeling-daytime-research-20260625`

Time boundaries:

- Work began after setup at about 2026-06-25 09:54 Asia/Shanghai.
- Soft-stop checkpoint began at 2026-06-25 16:23 Asia/Shanghai, before the requested 16:30 soft stop.
- No new experiments were launched after the soft-stop checkpoint.
- Final report and validation completed at 2026-06-25 16:25 Asia/Shanghai, before the 17:00 hard stop.

## Executive Conclusion

No L1 or L2 candidate was adoptable today, and no locked-test evaluation was triggered. The best result was diagnostic rather than promotable:

- Best private-transfer L2 candidate: r10, `runs/daytime-20260625/l2-autoresearch-agent-session-r10-8k-intent-risk-prompt`.
- r10 visible gates: visible validation, train audit, and visible cross-audit all had zero accepted wrongs.
- r10 private gates: selection still had 3 wrong accepts and promotion still had 1 wrong accept.
- r10 official validation sequential: 1558 accepts, 1544 correct, 14 wrong, precision 0.991014, coverage 0.502581, accuracy delta vs all-L4 -0.000968.
- r11 had slightly better official validation precision, but worse private transfer: private selection 9 wrong and promotion 5 wrong.

The evidence supports a specific bottleneck: current L2 visible folds and diagnostics can drive visible/train/cross-audit accepted wrongs to zero, but private failures persist in dense semantic-neighbor intent families. More visible density can expose more risk, but past a point it starves train support and encourages visible overfitting rather than private generalization.

## Implemented Harness Changes

- Added CLINC150 L2 AutoResearch controls for capped train traces, visible validation folds/ratio, visible cross-audit folds, local search timeout, and cross-audit top-k.
- Fixed generated L2 target workspaces so `commands.md` and `workspace_manifest.json` reflect configured local-search budgets.
- Added a process-level wall-clock timeout for `tools/search_config.py`, with structured `wall_clock_timeout` JSON output.
- Moved generated workspace `uv` environments and caches under `runs/` via `UV_PROJECT_ENVIRONMENT` and `UV_CACHE_DIR`, preventing protected `system/darjeeling/.venv` scope violations.
- Strengthened L2 target-program guidance to treat high-guard wrong-intent near misses as safety risks, even when visible `accepted_wrong` is zero.
- Added L1 prior accepted-error context in target-local CLINC150 L1 workspaces.
- Fixed precision-coverage backfill behavior for missing optional rows and unselected L1 diagnostic candidates.

## Experiment Results

L1:

- The best L1 four-round run nearly passed but still had one train-dev wrong accept in round 4.
- The explicit prior-error feedback context did not improve L1 transfer; the patched two-round run worsened from 9 train-dev wrong accepts in round 1 to 12 in round 2.
- L1 conclusion: simple prior-error discoverability is not the limiting factor; repairs open new broad-rule error families.

L2:

- r7 8k strong-visible reached visible/train/cross clean but failed private selection and promotion.
- r8 denser visible split exposed more baseline risk, but visible-clean candidates still failed private gates, and the run exposed a workspace `.venv` scope issue.
- r9 middle visible density was the best frontier candidate before prompt tightening: visible/train/cross clean, private selection 5 wrong, promotion 1 wrong.
- r10 intent-confusion prompt pressure reduced private selection wrongs from 5 to 3, with promotion still 1 wrong.
- r11 denser cross-audit made visible/cross clean possible under stronger stress but worsened private selection/promotion to 9/5 wrong.

## Budget And Usage

- `api_spend_usd`: `$0.00` observed, against a `$100.00` cap.
- All CLINC150 teacher rows and validations used copied processed data and existing replay artifacts.
- L4 agent-session usage is recorded separately in `docs/experiments/2026-06-25_daytime_autonomous_research_usage_ledger.json`.
- Observed completed L4 agent-session usage in the ledger: 67,438,236 input tokens, 62,720,768 cached input tokens, 386,063 output tokens, 107,793 reasoning output tokens.
- Outer executor usage is reported as wall-clock time only and is not folded into API spend or L4 agent-session usage.

## Validation Artifacts

- Research log: `docs/experiments/2026-06-25_daytime_autonomous_research_log.md`
- Usage ledger: `docs/experiments/2026-06-25_daytime_autonomous_research_usage_ledger.json`
- Precision-coverage output: `docs/experiments/precision_coverage/`
- Backfill summary after final run selection:
  - `round_metrics.jsonl`: 25 rows
  - `operating_points.jsonl`: 70 rows
  - `pareto_frontier.jsonl`: 61 rows

## Next Steps

- Add a visible-only diagnostic that ranks risky intent-neighbor pairs using train and multi-fold visible evidence before an accepted-wrong appears.
- Consider a selection-stage uncertainty gate for high-confidence semantic-neighbor pairs rather than relying only on accepted-wrong examples.
- Keep local-search process wall-clock enforcement; it materially improved experiment throughput and interpretability.
- Do not treat visible/train/cross clean as sufficient evidence for promotion on CLINC150-like dense intent sets without a stronger private-transfer proxy.
