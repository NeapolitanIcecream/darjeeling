# Overnight Autonomous Research Report - 2026-06-25

Branch/worktree:

- Branch: `codex/overnight-autonomous-research-20260624`
- Worktree: `/Users/chenmohan/gits/darjeeling-overnight-research-20260624`

Plan:

- `docs/experiments/2026-06-24_overnight_autonomous_research_plan.md`

Budget:

- Paid L4 benchmark cap: `$100.00`
- Paid L4 benchmark spend: `$0.00`
- Replay artifacts were used for all CLINC150 L1/L2 analysis and follow-up runs.

## Decision Summary

Adopt the repository changes in this branch for the next CLINC150 L1/L2 research pass:

1. CLINC150 L1 selection now defaults to zero train-dev wrong accepts before locked-test exposure.
2. L2 AutoResearch search config now writes scratch candidate configs by default; active `target/config.json` writeback requires `--apply-best`.
3. L2 target evolution now treats visible cross-audit as a selection safety gate when it is enabled.
4. L2 command timeout handling now normalizes timeout output and terminates process groups, preventing non-JSON command records and orphaned search processes.

Do not adopt either analyzed L2 target candidate. Both the previous 2026-06-24 candidate and the current follow-up candidate still had private selection wrong accepts.

## Evidence

### L1 Selection Gate

Artifact:

- `docs/experiments/2026-06-25_overnight_selection_gate_reanalysis.json`

The stricter L1 gate rejected all 5 previous L1 rounds. The prior selected round had visible validation precision 100.00% at 14.97% coverage, but train-dev had 3 wrong accepts. Its locked-test result later had accepted precision 96.91%, coverage 14.11%, and 24 wrong accepts.

Conclusion: visible validation precision alone was too weak. The zero train-dev wrong-accept gate would have prevented that locked-test exposure.

### L2 Previous AutoResearch Reanalysis

Artifacts:

- `docs/experiments/2026-06-25_overnight_l2_autoresearch_gate_reanalysis.json`
- `docs/experiments/2026-06-25_overnight_l2_cross_audit_gate_reanalysis.json`

The previous L2 AutoResearch round reached inner validation wrong accepts 0 and train-audit wrong accepts 0, but visible cross-audit still had 10 wrong accepts and private selection had 4 wrong accepts.

Conclusion: visible inner validation and train audit can be clean while selection-like visible cross-audit and private selection remain unsafe.

### L2 Follow-up AutoResearch Run

Ignored run directory:

- `runs/overnight-20260625/l2-autoresearch-agent-session-r3`

Round 1 completed:

- Inner validation: accepted 2065, wrong accepts 0, coverage 45.47%.
- Train audit: accepted 6399, wrong accepts 0, coverage 84.99%.
- Visible cross-audit: accepted 6864, wrong accepts 7, coverage 56.87%.
- Private selection: accepted 709, wrong accepts 4, gate failed.
- Private promotion: accepted 667, wrong accepts 3, gate passed.

Round 2 ran a 32-trial local search and hit the agent-session timeout before producing an adopted candidate. The run summary reports `rounds_completed: 1` and `stop_reason: agent_session_failed`.

Conclusion: the agent improved visible inner/train safety, but cross-audit and private selection still blocked adoption.

## Related Work

The design is aligned with selective prediction / reject-option framing:

- Geifman and El-Yaniv, "Selective Classification for Deep Neural Networks": https://arxiv.org/abs/1705.08500
- Geifman and El-Yaniv, "SelectiveNet": https://arxiv.org/abs/1901.09192
- Hendrickx et al., "Machine Learning with a Reject Option: A survey": https://arxiv.org/abs/2107.11277
- Xin et al., "The Art of Abstention": https://aclanthology.org/2021.acl-long.84/

Darjeeling mapping:

- Accepted precision is the selective-risk constraint.
- Accepted coverage is optimized only after risk gates pass.
- L4 fallback is the reject path.
- Count-based wrong-accept gates are needed because high precision ratios can still hide unacceptable accepted-error counts.

## Code Changes

Target-local L1:

- Added `max_train_dev_wrong_accepts` to CLINC150 L1 agent-session selection.
- Added per-round selection gate payloads and reason codes.
- Added candidate selection support for wrong-accept caps.

Target-local L2:

- Added scratch candidate output for local search.
- Added explicit `--apply-best` active-config writeback.
- Added visible cross-audit safety gate payloads and selection/adoption integration.
- Added process-group timeout cleanup and JSON-safe timeout command records.

Tests:

- Added regression tests for L1 zero-wrong selection.
- Added tests for L2 scratch config behavior.
- Added tests for L2 cross-audit safety gate.
- Added timeout-result serialization regression test.

## Failed Hypotheses

- Visible L1 validation at 100% accepted precision is not enough to predict locked-test safety.
- L2 inner validation and train-audit perfection are not enough to predict private selection safety.
- A 32-trial cross-audit local search is too expensive to run inside a 30-minute agent-session timeout without tighter budget controls.

## Validation

Passed:

- `uv run --extra dev pytest tests/targets/nlu/test_clinc150_phase1.py tests/targets/nlu/test_l2_target_evolution.py tests/test_precision_coverage_plots.py -q`
- `uv run --extra dev ruff check src/darjeeling/targets/nlu/clinc150_phase1.py src/darjeeling/targets/nlu/compiler/l2_target_evolution.py tests/targets/nlu/test_clinc150_phase1.py tests/targets/nlu/test_l2_target_evolution.py`

Final pre-commit verification is recorded in the commit session output.

## Next Steps

1. Add a bounded `--timeout-s` or trial budget to agent-facing `search_config` commands so one local search cannot consume an entire round.
2. Re-run fixed-inner L2 AutoResearch with the new cross-audit gate active in the harness process.
3. Add compact precision/coverage panels for L2 visible cross-audit points next to inner/private selection points.
4. Treat any candidate with nonzero visible cross-audit wrong accepts as non-selectable unless a future experiment proves the gate is too conservative.

