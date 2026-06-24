# Precision-Coverage Visualization Facility Plan

Date: 2026-06-24

Purpose: make precision/coverage visualization a standard Darjeeling experiment
output, then backfill the current CLINC150 L1/L2 history with Seaborn figures.

The user-facing question is:

1. As L1/L2 evolves across rounds or generations, do accepted precision and
   coverage improve together, trade off, or regress?
2. For a fixed candidate, what operating trade-off is available if the target
   adapter changes the accept policy without asking the evolve agent to change
   the artifact?

This is a reporting and target-adapter facility, not a new evolve algorithm and
not a reason to move CLINC150 semantics into core.

## Decision To Support

Deliver a standard visualization pattern that future L1/L2 experiments can
reuse:

- **Evolution curve**: round/generation on the x-axis, accepted precision and
  coverage on the y-axis. This shows whether evolve is improving the candidate.
- **Operating frontier**: coverage on the x-axis, accepted precision on the
  y-axis. Each point is an operating policy for a fixed candidate; the plot
  highlights the Pareto frontier and shows how much precision can be bought by
  giving up coverage.

Backfill at least:

- the real CLINC150 L1 5-round agent-session run;
- the CLINC150 L2 cascade / threshold artifacts;
- the CLINC150 calibration-repair guard artifacts when the required summaries
  are available.

The final report should say which figures are fully supported by historical
artifacts and which are partial because an older run did not record enough
per-request or candidate-sweep detail.

## Architecture Boundary

Keep the split simple:

- `src/darjeeling/eval/` may contain a target-independent plotting helper,
  normalized point schema, Pareto-frontier helper, and Seaborn style utilities.
  It must not interpret OOS, intent, phrase, slot, CLINC150, or L1/L2-specific
  fields.
- `src/darjeeling/targets/nlu/` may contain CLINC150 extractors that parse
  historical L1/L2 artifacts and produce normalized precision/coverage rows.
- L1/L2 generated artifacts must not be modified merely to support plotting.
  The evolve agents should not need to know about Seaborn, Pareto frontiers, or
  operating sweeps.
- For L1, operating policies should be target-adapter overlays over recorded
  L1 outputs. They can filter raw L1 accepts into abstains for analysis and
  selection, but they should not require L1 ProgramBank itself to expose a
  confidence threshold.
- For L2, operating policies can sweep recorded guard probability, threshold,
  margin, entropy, or existing guard-candidate summaries when available.

If a target-side overlay is later promoted into serving behavior, replay must
evaluate `candidate artifact + target adapter overlay` together. That is still
target-layer behavior, not core logic.

## Required Context

Read these first:

- `AGENTS.md`
- `docs/design/00_decisions.md`
- `docs/design/README.md`
- `docs/design/modules/eval_reports.md`
- `docs/experiments/README.md`
- `docs/experiments/2026-06-24_clinc150_l1_agent_session_effect_plan.md`
- `/Users/chenmohan/gits/darjeeling-clinc150-l1-agent-session-effect/docs/experiments/2026-06-24_clinc150_l1_agent_session_effect_report.md`
- `docs/experiments/2026-06-23_clinc150_l2_cascade_report.md`
- `/Users/chenmohan/gits/darjeeling-clinc150-calibration-repair/docs/experiments/2026-06-24_clinc150_calibration_repair_report.md`
- `/Users/chenmohan/gits/darjeeling-clinc150-l2-autoresearch/docs/experiments/2026-06-24_clinc150_l2_autoresearch_report.md`

Useful code:

- `src/darjeeling/eval/plots.py`
- `src/darjeeling/targets/nlu/clinc150_phase1.py`
- `src/darjeeling/targets/nlu/main_cli.py`
- `tests/targets/nlu/test_clinc150_phase1.py`

## Execution Isolation

Run this work in its own branch and worktree:

```text
branch: codex/precision-coverage-visuals
worktree: ../darjeeling-precision-coverage-visuals
```

If an appropriate worktree already exists, inspect and continue there instead
of creating a duplicate. Keep ignored historical artifacts available for
inspection. Do not delete the worktree or branch on completion.

## Historical Artifact Inputs

The execution agent should validate paths before using them. Fresh worktrees do
not contain ignored `runs/` artifacts, so use read-only absolute paths or copy
only the small summaries/prediction files needed for plotting.

L1 real agent-session run:

```text
/Users/chenmohan/gits/darjeeling-clinc150-l1-agent-session-effect/runs/clinc150-l1-agent-session-effect-20260624/main-agent-session-5round/clinc150_l1_agent_session_effect_summary.json
```

That run also contains per-round prediction JSONL files under:

```text
/Users/chenmohan/gits/darjeeling-clinc150-l1-agent-session-effect/runs/clinc150-l1-agent-session-effect-20260624/main-agent-session-5round/evaluations/
```

Previous dry-run L1 historical reference:

```text
/Users/chenmohan/gits/darjeeling-clinc150-l1-programbank/runs/clinc150-l1-programbank-20260624/l1-agent-jobs/*/attempt_summary.json
```

L2 cascade history:

```text
/Users/chenmohan/gits/darjeeling/runs/clinc150-l2-cascade-20260623/
```

Important L2 files include:

```text
distilled-l2/train-full/validation-cascade/clinc150_l2_eval_summary.json
distilled-l2/train-full/validation-cascade/clinc150_l2_predictions.jsonl
distilled-l2/train-full/test-cascade/clinc150_l2_eval_summary.json
distilled-l2/train-full/test-cascade/clinc150_l2_predictions.jsonl
distilled-l2/train-full/validation-uniform-cascade/clinc150_l2_eval_summary.json
distilled-l2/train-full/validation-zipf-heavy-cascade/clinc150_l2_eval_summary.json
distilled-l2/train-3000/validation-cascade-tight-thresholds/clinc150_l2_eval_summary.json
```

Calibration repair history:

```text
/Users/chenmohan/gits/darjeeling-clinc150-calibration-repair/runs/clinc150-calibration-repair-20260624/clinc150_calibration_repair_summary.json
/Users/chenmohan/gits/darjeeling-clinc150-calibration-repair/runs/clinc150-calibration-repair-20260624/safety-margin-995/clinc150_calibration_repair_summary.json
```

L2 AutoResearch history:

```text
/Users/chenmohan/gits/darjeeling-clinc150-l2-autoresearch/runs/clinc150-l2-autoresearch-20260624/clinc150_l2_autoresearch_summary.json
/Users/chenmohan/gits/darjeeling-clinc150-l2-autoresearch/runs/clinc150-l2-autoresearch-20260624/agent-session-r1/target-evolution/summary.json
```

## Normalized Data Contract

Write normalized rows before plotting. Keep the schema ordinary JSONL/CSV so
reports can be regenerated without rerunning experiments.

Required output files:

```text
docs/experiments/precision_coverage/round_metrics.jsonl
docs/experiments/precision_coverage/operating_points.jsonl
docs/experiments/precision_coverage/pareto_frontier.jsonl
```

Suggested fields for `round_metrics.jsonl`:

```json
{
  "experiment_id": "clinc150-l1-agent-session-effect",
  "layer": "L1",
  "candidate_id": "round-001",
  "round": 1,
  "split": "visible_validation",
  "view": "sequential",
  "accepted_precision": 1.0,
  "coverage": 0.1497,
  "accepted": 464,
  "wrong_accepts": 0,
  "source_artifact": "/absolute/path/to/source.json"
}
```

Suggested fields for `operating_points.jsonl`:

```json
{
  "experiment_id": "clinc150-l1-agent-session-effect",
  "layer": "L1",
  "candidate_id": "round-001",
  "round": 1,
  "split": "train_dev",
  "policy_family": "l1_overlay_rule_support",
  "policy_label": "positive>=10, negative=0",
  "policy_value": 10,
  "accepted_precision": 0.9978,
  "coverage": 0.1808,
  "accepted": 1367,
  "wrong_accepts": 3,
  "pareto": true,
  "source_artifact": "/absolute/path/to/predictions.jsonl"
}
```

Target-specific metadata may be included under an opaque `metadata` object.
Core plotting code must not depend on it.

## Operating Policy Extraction

### L1

For historical L1, do not change the Rust candidate and do not ask an L1 agent
to add confidence scores.

The CLINC150 extractor should build operating points by filtering recorded L1
accepts after the fact:

1. Parse prediction rows from a candidate evaluation.
2. Derive a stable rule key from fields such as `program_path`, `reason`, and
   predicted intent.
3. Compute rule support/risk from an allowed visible calibration split, such as
   train-dev or visible validation.
4. Apply policy overlays to the same candidate's raw accepts:
   - raw accepts;
   - `positive_support >= N`;
   - `negative_support == 0`;
   - `OOS_false_support == 0`;
   - conflict-family veto if the historical artifact exposes enough data;
   - combinations of the above.
5. Recompute precision, coverage, OOS false accept, and cascade delta for each
   overlay.

Important boundary: locked-test rows may be plotted as post-selection
diagnostics, but locked-test labels/details must not be used to choose overlay
policies or tune thresholds. Mark any locked-test frontier as diagnostic.

This creates a candidate-local operating frontier. It can only filter raw L1
accepts into abstains; it cannot create new coverage. Pushing the frontier
right/up still requires future evolve rounds to produce better raw candidates.

### L2

For historical L2, prefer per-request prediction files when they contain guard
probabilities or confidences. Sweep thresholds over recorded predictions:

```text
threshold in [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.985, 0.99, 0.995]
```

Also include operating points from existing summary candidate families such as
threshold, threshold+margin, threshold+entropy, safety-margin guard, and
selected historical points. If older artifacts only contain summary-level
candidates, include them as discrete points and record the limitation in the
report.

## Seaborn Visual Design

Use Seaborn, not Plotly, for the first standard static figures.

Add `seaborn>=0.13` to project dependencies if it is not already available.
Use Matplotlib only for lower-level formatting. If adding Seaborn pulls pandas
into the base environment, make that explicit in the commit and report.

Produce at least these PNG figures:

```text
docs/experiments/precision_coverage/figures/clinc150_l1_evolution.png
docs/experiments/precision_coverage/figures/clinc150_l1_operating_frontier.png
docs/experiments/precision_coverage/figures/clinc150_l2_evolution.png
docs/experiments/precision_coverage/figures/clinc150_l2_operating_frontier.png
docs/experiments/precision_coverage/figures/clinc150_l1_l2_frontier_comparison.png
```

Also write SVG or PDF versions if easy, but PNG is required for visual QA.

Default style requirements:

- colorblind-safe palette;
- white background with light grid;
- line + marker for evolution curves;
- direct labels or a compact legend outside the plot area;
- axes formatted as percentages;
- visible gate reference lines, especially accepted precision 99%;
- split/view shown by facet or line style, not by overloaded colors;
- locked-test diagnostic points visually distinct from visible-selection data;
- no smoothing that invents data;
- enough right/top margin for labels;
- figure widths that fit reports without tiny text.

## Visual QA

After rendering, inspect the PNGs visually and iterate. The agent should use
the available image viewing tool for local files.

QA checklist:

- text is readable at report size;
- legend does not cover data;
- percent axes are clear and not misleading;
- precision and coverage are not plotted on incompatible scales without
  labeling;
- selected points and Pareto frontier are visually obvious;
- locked-test diagnostic points cannot be mistaken for agent-visible selection
  data;
- figures remain useful in grayscale or colorblind palettes;
- no chart has excessive empty space, clipped labels, or overlapping markers.

Write notes to:

```text
docs/experiments/precision_coverage/visual_qa.md
```

Record at least the first visual issues found and the final visual style
decisions. Do not stop after the first render if obvious issues remain.

## Report

Write:

```text
docs/experiments/2026-06-24_precision_coverage_visualization_report.md
```

The report must include:

- what code/data contracts were added;
- what historical artifacts were parsed;
- which plots are complete vs partial;
- the final standard visual pattern;
- thumbnails or links to generated figures;
- visual QA summary;
- how future L1/L2 experiments should emit round metrics and operating points;
- clear statement that L1 operating policies are target-adapter overlays, not
  L1 artifact requirements;
- limitations and next steps.

Update:

- `docs/experiments/README.md`
- `docs/design/modules/eval_reports.md`

## Tests And Validation

Add focused tests for:

- Pareto frontier selection;
- normalized row parsing/writing;
- L1 overlay filtering on a tiny fixture;
- L2 threshold sweep on a tiny fixture;
- plot smoke generation with a tiny fixture and non-empty PNG output.

Run at least:

```bash
uv run pytest tests/targets/nlu/test_clinc150_phase1.py tests/test_precision_coverage_plots.py -q
uv run ruff check <touched files>
git diff --check
```

If the new tests live under a different module path, adjust the command and
record the actual command in the report. If the dependency update changes the
lockfile, run the relevant package sync/update command and include the lockfile
in the commit.

## Done Criteria

- Dedicated branch/worktree used.
- Seaborn-based static figures generated from historical CLINC150 L1/L2 data.
- At least one evolution curve and one operating frontier are produced for L1.
- At least one evolution or historical candidate curve and one operating
  frontier/discrete frontier are produced for L2.
- Visual QA was performed and iterated until the figures are readable and
  suitable as standard report outputs.
- Normalized JSONL data files and Pareto rows are written.
- Target-specific extraction stays in NLU/CLINC150 code; generic plotting code
  does not interpret target semantics.
- L1 evolve agent does not need to know about operating frontiers or plotting.
- Report and docs are updated.
- Tests, lint, and diff checks pass or skipped checks are justified.
- Changes are organized into a git commit on the dedicated branch/worktree.
