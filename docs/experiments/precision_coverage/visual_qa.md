# Precision/Coverage Visual QA

Date: 2026-06-24

Figures inspected:

- `figures/clinc150_l1_evolution.png`
- `figures/clinc150_l1_operating_frontier.png`
- `figures/clinc150_l2_evolution.png`
- `figures/clinc150_l2_operating_frontier.png`
- `figures/clinc150_l1_l2_frontier_comparison.png`

## First Render Issues

- Legend labels exposed machine field names such as `policy_family`,
  `selection_scope`, and underscore-separated split names.
- Evolution plots placed the `99% precision gate` label directly over the
  top precision curve.
- L1 frontier expanded the y-axis to a 101% tick even though all rates are
  capped at 100%.
- Locked-test diagnostic points were distinct by marker, but the legend wording
  needed to make the diagnostic status easier to scan.

## Final Style Decisions

- Use Seaborn with the colorblind palette, a white background, and a light grid.
- Format all precision/coverage axes as percentages.
- Keep legends outside the plot area so they do not cover points or frontier
  lines.
- Render evolution curves as line plus marker, with split shown by line style
  rather than color.
- Render operating points as scatter plots and overlay Pareto frontier points
  with black rings and connecting lines per candidate/split.
- Show locked-test diagnostic points with a different marker from agent-visible
  selection data.
- Keep the 99% accepted-precision reference line visible in every figure.
- Do not smooth curves or interpolate beyond the observed policies.

## Final Check

The second visual pass fixed the label and axis issues. Text is readable at
report size, legends do not cover data, locked-test diagnostics are visually
distinct, and the Pareto frontier markers remain visible in grayscale.
