# Precision/Coverage Visual QA

Date: 2026-06-24

Figures inspected:

- `figures/clinc150_l1_evolution.png`
- `figures/clinc150_l1_train_dev_operating_curve.png`
- `figures/clinc150_l1_validation_operating_curve.png`
- `figures/clinc150_l1_locked_test_diagnostic_curve.png`
- `figures/clinc150_l2_validation_threshold_curve.png`
- `figures/clinc150_l2_locked_test_diagnostic_curve.png`
- `figures/clinc150_l1_l2_visible_curve_comparison.png`

## First Repaired Render Issues

- L1 train-dev and visible-validation endpoint labels sat too close to the
  title because several points are exactly at 100% accepted precision.
- The first L2 operating-curve arrow ran diagonally across the whole plot and
  looked like a second data series instead of a direction cue.
- In the comparison figure, L1 labels overlapped the subplot title for the same
  100% precision reason.

## Iterations

- Moved high-precision point labels below the points and right-edge labels to
  the inside of the plot.
- Replaced the long data-space direction arrow with a small
  `order: strict -> loose` note inside each subplot.
- Added slight y-axis headroom for 100% markers while keeping y-axis ticks
  capped at 100%, so the plots avoid the earlier 101% tick problem.
- Regenerated all JSONL and PNG outputs after the plotting adjustment.

## Final Check

- Each standard operating-curve subplot contains exactly one connected
  `curve_id`.
- L1 train-dev, L1 visible validation, and L1 locked-test diagnostic curves are
  separate figures.
- L2 validation and L2 locked-test diagnostic curves are separate figures.
- The comparison figure uses separate facets for L1 `risk_tolerance` and L2
  `guard_threshold`; it does not connect L1 and L2 points.
- Axes are percentages, the 99% precision gate remains visible, endpoint labels
  are readable, and no smoothing or interpolation is used.
- No mixed scatter/Pareto frontier figure is retained as a standard output.
