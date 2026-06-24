# Precision-Coverage Frontier Repair Plan

Date: 2026-06-24

This plan repairs the first precision/coverage visualization implementation.
The first implementation added useful normalized data, Seaborn rendering, and
historical CLINC150 backfill, but the frontier figures did not express the
intended idea clearly. They mixed multiple splits, policy families, diagnostic
scopes, and Pareto markers into one scatter plot. The result looked like a
debug plot, not a standard operating trade-off curve.

The goal of this plan is to deliver the end-to-end effect originally intended:
for a fixed L1 or L2 candidate, show how accepted precision changes as coverage
is deliberately moved by one clear accept-policy knob.

## Starting Point

Start from the existing visualization implementation, not from scratch:

```text
source branch: codex/precision-coverage-visuals
source commit: 122f6c15ff4032052e1acfbf80371f43ea7af60c
source worktree: /Users/chenmohan/gits/darjeeling-precision-coverage-visuals
```

Continue in the existing visualization branch/worktree:

```text
branch: codex/precision-coverage-visuals
worktree: /Users/chenmohan/gits/darjeeling-precision-coverage-visuals
```

Do not create another worktree for this repair unless the existing visualization
worktree is missing or unusable. If it has uncommitted changes, inspect them
first and continue carefully without discarding user or agent work.

## Design Decision

The standard frontier plot must be a single operating curve, not a general
scatter plot. A standard operating curve has:

- one layer;
- one experiment;
- one candidate or candidate family;
- one evaluation split;
- one accept-policy knob;
- ordered knob values from strict to loose or loose to strict.

It should answer:

```text
If we move this knob, how much coverage do we buy, and how much precision do we lose?
```

Do not use a plot that requires the reader to mentally separate train-dev,
validation, locked test, OOS-heavy, conflict slices, raw policy, support policy,
risk policy, and Pareto markers all at once.

Pareto-frontier computation may remain in the data layer, but black-ring Pareto
markers must not be the main explanation. They can be used in an appendix/debug
figure only if useful.

## Architecture Boundary

Keep the boundary from the first implementation:

- target-neutral row I/O, validation, plotting helpers, and simple curve
  rendering live in `src/darjeeling/eval/plots.py`;
- CLINC150 artifact parsing, L1 overlay scoring, and L2 threshold sweeps live
  in `src/darjeeling/targets/nlu/precision_coverage.py`;
- generated L1 ProgramBank code is not modified;
- L1 evolve agents do not see plotting/frontier mechanics;
- CLINC150/NLU semantics do not enter core plotting code.

This repair may add stricter generic helpers such as
`plot_single_operating_curve` or `validate_operating_curve_rows`, but those
helpers must remain target-neutral and operate only on normalized fields.

## Required Context

Read these first:

- `AGENTS.md`
- `docs/design/00_decisions.md`
- `docs/design/modules/eval_reports.md`
- `docs/experiments/2026-06-24_precision_coverage_visualization_plan.md`
- `docs/experiments/2026-06-24_precision_coverage_visualization_report.md`
  from the source worktree if not present on main
- `docs/experiments/precision_coverage/visual_qa.md` from the source worktree
- `src/darjeeling/eval/plots.py`
- `src/darjeeling/targets/nlu/precision_coverage.py`
- `src/darjeeling/targets/nlu/main_cli.py`
- `tests/test_precision_coverage_plots.py`

Historical artifact inputs remain the same as the first plan:

- L1 summary:
  `/Users/chenmohan/gits/darjeeling-clinc150-l1-agent-session-effect/runs/clinc150-l1-agent-session-effect-20260624/main-agent-session-5round/clinc150_l1_agent_session_effect_summary.json`
- L2 cascade root:
  `/Users/chenmohan/gits/darjeeling/runs/clinc150-l2-cascade-20260623`
- Calibration summaries:
  `/Users/chenmohan/gits/darjeeling-clinc150-calibration-repair/runs/clinc150-calibration-repair-20260624/clinc150_calibration_repair_summary.json`
  and
  `/Users/chenmohan/gits/darjeeling-clinc150-calibration-repair/runs/clinc150-calibration-repair-20260624/safety-margin-995/clinc150_calibration_repair_summary.json`
- AutoResearch summary:
  `/Users/chenmohan/gits/darjeeling-clinc150-l2-autoresearch/runs/clinc150-l2-autoresearch-20260624/agent-session-r1/clinc150_l2_autoresearch_summary.json`

## Data Contract Repair

Keep the existing files:

```text
docs/experiments/precision_coverage/round_metrics.jsonl
docs/experiments/precision_coverage/operating_points.jsonl
docs/experiments/precision_coverage/pareto_frontier.jsonl
```

Add or consistently populate fields needed for single-knob curves:

```json
{
  "curve_id": "clinc150-l1-round001-train-dev-risk-tolerance",
  "curve_role": "standard_operating_curve",
  "knob_name": "risk_tolerance",
  "knob_value": 2,
  "knob_label": "medium: clean support >= 5",
  "knob_order": 2,
  "knob_direction": "strict_to_loose",
  "primary_curve": true
}
```

Rules:

- `curve_id` groups rows that may be connected by a line.
- A standard plot must not connect rows from different `curve_id`s.
- `knob_order` controls line order; do not sort by precision or coverage.
- `selection_scope=locked_test_diagnostic` rows must not be mixed into the
  primary agent-visible curve unless the figure title and filename explicitly
  say diagnostic.
- `pareto=true` is allowed but not required for standard figures.

## L1 Operating Curve Semantics

For L1, build one explicit target-adapter overlay knob from recorded L1 accepts.
Do not expose this knob to the L1 artifact or the L1 evolve agent.

Use a simple monotonic `risk_tolerance` curve. The exact formula may be adjusted
if the historical data requires it, but it must be explainable in the report and
must have ordered levels. A suggested version:

```text
0 strict: keep rules with wrong_support=0, oos_false_support=0, positive_support>=20
1 safe:   keep rules with wrong_support=0, oos_false_support=0, positive_support>=10
2 medium: keep rules with wrong_support=0, oos_false_support=0, positive_support>=5
3 loose:  keep rules with wrong_support=0, oos_false_support=0, positive_support>=2
4 raw:    keep all recorded L1 accepts
```

Support/risk stats should be computed from an allowed visible calibration split,
preferably the selected candidate's train-dev prediction rows. Apply the same
predefined ordered knob to:

- train-dev: primary visible risk curve;
- visible validation: secondary visible confirmation curve;
- locked test: diagnostic-only curve.

Do not compute the knob from locked-test outcomes. Locked test can only be used
to show what the predeclared visible overlay would have done.

The L1 operating curve is allowed to move only left from raw L1. It filters raw
accepts into abstains; it cannot create new accepted coverage. The report should
state this explicitly.

## L2 Operating Curve Semantics

For L2, use one explicit threshold knob over recorded guard probability:

```text
guard_threshold in [0.995, 0.99, 0.985, 0.98, 0.95, 0.90, 0.80, 0.70, 0.60, 0.50]
```

The main standard L2 curve should use:

```text
candidate: teacher-full
split: validation
knob_name: guard_threshold
```

Render locked test as a separate diagnostic curve using the same threshold
values, not in the same figure as the agent-visible validation curve.

Summary-only calibration repair and AutoResearch points can remain as discrete
context in tables or appendix plots, but they should not be connected to the
main threshold curve unless they share the same candidate, split, and knob.

## Required Standard Figures

Replace or supplement the first implementation's frontier figures with these
standard figures:

```text
docs/experiments/precision_coverage/figures/clinc150_l1_evolution.png
docs/experiments/precision_coverage/figures/clinc150_l1_train_dev_operating_curve.png
docs/experiments/precision_coverage/figures/clinc150_l1_validation_operating_curve.png
docs/experiments/precision_coverage/figures/clinc150_l1_locked_test_diagnostic_curve.png
docs/experiments/precision_coverage/figures/clinc150_l2_validation_threshold_curve.png
docs/experiments/precision_coverage/figures/clinc150_l2_locked_test_diagnostic_curve.png
docs/experiments/precision_coverage/figures/clinc150_l1_l2_visible_curve_comparison.png
```

The comparison figure should be readable even if L1 and L2 use different knobs.
Use separate facets or clearly separated panels if overlaying them would be
misleading. Do not connect L1 and L2 points to each other.

The old mixed scatter figures may be kept only as appendix/debug figures with
filenames such as:

```text
docs/experiments/precision_coverage/figures/debug_clinc150_l1_mixed_points.png
```

They must not be presented as the standard frontier output.

## Visual Design Requirements

For operating curves:

- one curve per subplot unless facets are explicit;
- x-axis is accepted coverage;
- y-axis is accepted precision;
- line connects ordered knob values;
- mark and label at least the strict endpoint, selected/nominal endpoint, and
  raw/loose endpoint;
- show the 99% precision gate;
- use arrows or labels to make strict-to-loose direction clear;
- do not use Pareto rings as the primary explanation;
- do not mix locked-test diagnostics with visible-selection curves;
- do not smooth or interpolate beyond observed operating points.

For evolution curves:

- keep round/generation on x-axis;
- show accepted precision and coverage as separate lines or facets;
- include train-dev and visible-validation splits only if both remain readable;
- keep the 99% precision gate readable and away from data labels.

## Visual QA

Run visual QA with the local image viewer. Iterate at least once after the first
render.

Checklist:

- Can a reader explain the knob after looking at the title, legend, and labels?
- Is there exactly one connected operating curve per subplot?
- Are locked-test diagnostics visually and semantically separate?
- Are axes percentages and ranges sensible?
- Are endpoint labels readable?
- Is the 99% precision gate visible without covering data?
- Does the comparison figure avoid implying that L1 and L2 points are on the
  same knob?
- Are old mixed scatter figures clearly marked as debug/appendix if retained?

Update:

```text
docs/experiments/precision_coverage/visual_qa.md
```

## Report

Update or rewrite:

```text
docs/experiments/2026-06-24_precision_coverage_visualization_report.md
```

The report must explicitly say:

- the first frontier design was insufficient because it mixed too many
  dimensions;
- the repaired standard is single split, single candidate, single knob per
  operating curve;
- L1's curve is a target-adapter overlay over recorded accepts, not an L1
  artifact feature;
- L2's curve is a guard-threshold sweep;
- which figures are standard outputs and which, if any, are debug appendix
  figures;
- what the repaired historical CLINC150 curves show.

Update:

- `docs/experiments/README.md`
- `docs/design/modules/eval_reports.md`

## Tests And Validation

Add or update tests for:

- standard operating curve rows must share one `curve_id`;
- plot helper refuses or does not connect mixed `curve_id`s unless explicitly
  faceted;
- L1 `risk_tolerance` overlay produces monotonic ordered points on a tiny
  fixture;
- L2 threshold sweep produces ordered `guard_threshold` points;
- locked-test diagnostic rows are not included in the visible standard curve;
- generated standard PNGs are non-empty.

Run at least:

```bash
uv run --extra dev pytest tests/test_precision_coverage_plots.py -q
uv run --extra dev pytest tests/targets/nlu/test_clinc150_phase1.py tests/test_precision_coverage_plots.py -q
uv run --extra dev ruff check src/darjeeling/eval/plots.py src/darjeeling/targets/nlu/precision_coverage.py src/darjeeling/targets/nlu/main_cli.py tests/test_precision_coverage_plots.py
git diff --check
```

Regenerate the figures from historical artifacts using the documented CLI and
record the exact command in the report.

## Done Criteria

- Existing visualization branch/worktree was used, unless it was missing or
  unusable and a replacement was clearly documented.
- Work starts from `codex/precision-coverage-visuals` and preserves useful data
  extraction/reporting infrastructure.
- Standard frontier figures are replaced by single-knob operating curves.
- L1 visible train-dev, L1 visible validation, and L1 locked diagnostic curves
  are separate figures or clearly separated facets.
- L2 validation and L2 locked diagnostic curves are separate figures or clearly
  separated facets.
- The L1/L2 comparison does not connect unrelated knobs or mix diagnostic data
  into the visible curve.
- Visual QA was performed and documented after at least one iteration.
- The report explains why the first frontier plot design was not good enough
  and what standard replaces it.
- Target-specific extraction remains in NLU code; generic plotting remains
  target-neutral.
- Tests, lint, and `git diff --check` pass.
- Changes are organized into a git commit on the dedicated branch/worktree.
