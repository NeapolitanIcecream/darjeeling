# Overnight Autonomous Research Log - 2026-06-25

Plan executed from:

- `docs/experiments/2026-06-24_overnight_autonomous_research_plan.md`

Branch/worktree:

- Branch: `codex/overnight-autonomous-research-20260624`
- Worktree: `/Users/chenmohan/gits/darjeeling-overnight-research-20260624`

Budget policy:

- Paid L4 benchmark cap: `$100.00`
- Paid L4 benchmark spend so far: `$0.00`
- Replay artifacts were preferred for CLINC150 L1/L2 analysis and follow-up runs.

## Cycle 1 - Artifact Baseline And Standard Report Path

Hypothesis:

Precision/coverage plots from the merged visualization facility should become the standard comparison surface before deeper L1/L2 experiments.

Action:

- Read the overnight plan, repo design notes, target/core split instructions, and the 2026-06-24 L1/L2 experiment reports.
- Re-ran `clinc150 precision-coverage-backfill` against the original L1/L2/calibration/AutoResearch artifacts.

Evidence:

- Backfill completed with 30 round metrics, 75 operating points, and 63 Pareto points.
- Re-running against original source artifacts left `docs/experiments/precision_coverage/` with no git diff, confirming the standard report path is reproducible.

Decision:

Use precision/coverage figures as the default report output path for L1/L2 comparisons, but do not treat plots as a replacement for count-based acceptance gates.

## Cycle 2 - Related Work To Gate Design

Hypothesis:

Selective prediction literature can justify a bounded-risk-first gate without requiring a new core abstraction.

Action:

- Reviewed selective classification, reject-option, and NLP abstention work.
- Mapped accepted precision/coverage and L4 fallback to selective risk/coverage/reject.

Evidence:

- Notes are recorded in `docs/experiments/2026-06-25_overnight_related_work_notes.md`.
- The mapping supports simple target-local gate changes: maximize accepted coverage only after accepted-risk evidence is clean.

Decision:

Adopt bounded-risk-first selection as the guiding principle for CLINC150 L1/L2 candidate gates.

## Cycle 3 - L1 Selection Gate Reanalysis And Patch

Hypothesis:

The previous L1 selected candidate would have been rejected before locked-test exposure if selection had required zero train-dev wrong accepts, not only high visible validation precision.

Action:

- Added a CLINC150 L1 selection gate with a configurable `max_train_dev_wrong_accepts` threshold.
- Exposed gate details in per-round and summary payloads.
- Updated candidate selection to support a wrong-accept cap.
- Added tests for zero-wrong-accept selection and train-dev reason reporting.
- Reanalyzed the 2026-06-24 L1 summary under the stricter gate.

Evidence:

- Reanalysis artifact: `docs/experiments/2026-06-25_overnight_selection_gate_reanalysis.json`.
- New gate rejected all 5 prior rounds.
- The previously selected round 1 had visible validation precision 100% at 14.97% coverage, but train-dev had 3 wrong accepts.
- The same round later produced locked-test accepted precision 96.91%, coverage 14.11%, and 24 wrong accepts.

Decision:

Keep the default `max_train_dev_wrong_accepts=0` for CLINC150 L1 agent-session candidate selection. This is conservative but directly addresses the observed locked-test failure.

## Cycle 4 - L2 AutoResearch Search Config Externalization Patch

Hypothesis:

AutoResearch should not mutate active `target/config.json` during exploratory search; search should write a scratch candidate and require explicit apply.

Action:

- Changed L2 local search to write best search output under `runs/local_search_candidates/` by default.
- Added an explicit `--apply-best` path for intentional active-config writeback.
- Kept harness `mode="local-search"` behavior compatible by calling the apply path there.
- Updated workspace commands to expose both `search_config` and `apply_search_config`.
- Added tests covering scratch default and explicit apply behavior.

Evidence:

- Generated workspace manifest in the new run exposes:
  - `search_config`: scratch candidate path, restores active config.
  - `apply_search_config`: explicit active config update.
- Focused test coverage passed for both behaviors.

Decision:

Keep scratch candidate generation as the default AutoResearch agent-facing command. Explicit apply remains available for harness/local-search workflows.

## Cycle 5 - L2 Previous AutoResearch Gate Reanalysis

Hypothesis:

Perfect visible inner validation is not sufficient evidence for L2 adoption if private selection holdout remains unchanged or unsafe.

Action:

- Reanalyzed the 2026-06-24 L2 AutoResearch summary.

Evidence:

- Reanalysis artifact: `docs/experiments/2026-06-25_overnight_l2_autoresearch_gate_reanalysis.json`.
- Candidate round 1 inner validation: accepted 2043, wrong accepts 0, accepted accuracy 100.00%, coverage 44.99%.
- Candidate private selection: accepted 705, wrong accepts 4, accepted accuracy 99.43%, coverage 46.38%; gate failed.
- Baseline private selection: accepted 709, wrong accepts 4, accepted accuracy 99.44%, coverage 46.64%.
- Visible cross-audit still had 10 wrong accepts.

Decision:

Do not adopt visible-inner-only candidates. Future L2 rounds should explicitly pressure visible cross-audit/OOS-like wrong accepts and private-selection transfer before claiming improvement.

## Cycle 6 - Follow-up L2 AutoResearch Agent Session

Hypothesis:

With scratch search commands available, another fixed-inner AutoResearch session can use replay artifacts to search for a safer L2 target candidate without paid L4 calls.

Action:

- Started a 3-round CLINC150 L2 AutoResearch agent-session run using replay teacher details:
  - Output: `runs/overnight-20260625/l2-autoresearch-agent-session-r3`
  - Rounds: 3
  - Budget profile: `fixed-inner`
  - Local search trials: 16
  - Timeout: 1800 seconds

Evidence:

- Data preparation completed.
- Baseline wrote `target-evolution/rounds/baseline.json`.
- Round 1 completed and wrote `target-evolution/rounds/round_001.json`.
- Round 1 visible validation: accepted 2065, wrong accepts 0, coverage 45.47%.
- Round 1 train audit: accepted 6399, wrong accepts 0, coverage 84.99%.
- Round 1 visible cross-audit: accepted 6864, wrong accepts 7, coverage 56.87%; ratio gate passed but count-risk remained.
- Round 1 private selection: accepted 709, wrong accepts 4, gate failed.
- Round 1 private promotion: accepted 667, wrong accepts 3, gate passed.
- Round 2 started and ran a 32-trial visible local search with cross-audit reranking, but the agent session hit timeout before a selectable candidate was produced.
- No paid L4 calls have been made.

Decision:

Do not adopt the follow-up candidate. The run reproduced the core L2 failure mode: visible validation and train audit can be cleaned while cross-audit and private selection still expose accepted wrongs.

## Cycle 7 - L2 Visible Cross-Audit Count Gate

Hypothesis:

Visible cross-audit accepted-error counts catch L2 candidates that visible inner validation and train-audit gates miss.

Action:

- Added a visible cross-audit safety gate to L2 target evolution.
- The gate is enabled only when visible cross-audit metrics exist.
- Default policy requires the cross-audit metric gate to pass and `wrong_accepts <= 0`.
- Candidate selection and adoption now require this gate when visible cross-audit is enabled.
- Added tests for cross-audit wrong-accept rejection.
- Reanalyzed the previous and current L2 AutoResearch evidence.

Evidence:

- Reanalysis artifact: `docs/experiments/2026-06-25_overnight_l2_cross_audit_gate_reanalysis.json`.
- Previous AutoResearch round 1 had inner validation wrong accepts 0 and train-audit wrong accepts 0, but visible cross-audit wrong accepts 10 and private selection wrong accepts 4.
- Current follow-up round 1 had inner validation wrong accepts 0 and train-audit wrong accepts 0, but visible cross-audit wrong accepts 7 and private selection wrong accepts 4.

Decision:

Make visible cross-audit a selection safety gate when enabled. This is still target-evolution plumbing, not core schema logic, and it prevents using private holdouts to rescue visible selection-like failures.

## Cycle 8 - Timeout And Orphan Process Harness Repair

Hypothesis:

Long agent-session or local-search timeouts should produce auditable command records and clean up child processes instead of failing summary writing or leaving orphan CPU work.

Action:

- Round 2 timeout exposed a harness bug: `subprocess.TimeoutExpired.stdout` / `stderr` can be `bytes`, which broke `commands.jsonl` JSON serialization after `summary.json` had already been written.
- The same timeout left `tools/search_config.py` orphaned under PID 1; it was manually terminated after confirming the outer harness had already failed and written summary.
- Reworked `_run_command` to use `subprocess.Popen`, terminate the whole process group on timeout, and normalize timeout output to text.
- Added a regression test asserting timeout command results are JSON serializable.

Evidence:

- Before the fix, the CLI ended with `TypeError: Object of type bytes is not JSON serializable`.
- The failed run still wrote `target-evolution/summary.json` with `rounds_completed: 1` and `stop_reason: agent_session_failed`.
- Focused regression tests pass after the fix.

Decision:

Keep the process-group timeout fix. Future failed agent sessions should leave cleaner artifacts and should not require manual orphan cleanup.
