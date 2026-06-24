from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

NORMALIZED_POINT_REQUIRED_FIELDS = (
    "experiment_id",
    "layer",
    "candidate_id",
    "split",
    "accepted_precision",
    "coverage",
    "source_artifact",
)


def plotly_available() -> bool:
    try:
        import plotly  # noqa: F401
    except ImportError:
        return False
    return True


def write_normalized_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> Path:
    """Write target-neutral precision/coverage rows as reviewable JSONL."""

    normalized_rows = [normalize_precision_coverage_row(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in normalized_rows),
        encoding="utf-8",
    )
    return path


def read_normalized_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        normalize_precision_coverage_row(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return rows


def normalize_precision_coverage_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    if "coverage" not in normalized and "accepted_coverage" in normalized:
        normalized["coverage"] = normalized["accepted_coverage"]
    missing = [
        field
        for field in NORMALIZED_POINT_REQUIRED_FIELDS
        if field not in normalized
    ]
    if missing:
        missing_fields = ", ".join(missing)
        raise ValueError(f"precision/coverage row missing required fields: {missing_fields}")
    normalized["experiment_id"] = str(normalized["experiment_id"])
    normalized["layer"] = str(normalized["layer"])
    normalized["candidate_id"] = str(normalized["candidate_id"])
    normalized["split"] = str(normalized["split"])
    normalized["source_artifact"] = str(normalized["source_artifact"])
    if normalized.get("accepted_precision") is not None:
        normalized["accepted_precision"] = float(normalized["accepted_precision"])
    normalized["coverage"] = float(normalized["coverage"])
    return normalized


def annotate_pareto_frontier(
    rows: Sequence[dict[str, Any]],
    *,
    group_keys: Sequence[str] = ("experiment_id", "layer", "candidate_id", "split"),
    precision_key: str = "accepted_precision",
    coverage_key: str = "coverage",
) -> list[dict[str, Any]]:
    """Return rows with a target-neutral Pareto flag for precision and coverage.

    Both dimensions are maximized. Rows with missing precision are retained but
    cannot be frontier points.
    """

    annotated = [dict(row, pareto=False) for row in rows]
    groups: dict[tuple[Any, ...], list[int]] = {}
    for index, row in enumerate(annotated):
        groups.setdefault(tuple(row.get(key) for key in group_keys), []).append(index)

    for indexes in groups.values():
        for index in indexes:
            row = annotated[index]
            precision = row.get(precision_key)
            coverage = row.get(coverage_key)
            if precision is None or coverage is None:
                continue
            row_precision = float(precision)
            row_coverage = float(coverage)
            dominated = False
            for other_index in indexes:
                if other_index == index:
                    continue
                other = annotated[other_index]
                other_precision = other.get(precision_key)
                other_coverage = other.get(coverage_key)
                if other_precision is None or other_coverage is None:
                    continue
                other_precision_float = float(other_precision)
                other_coverage_float = float(other_coverage)
                if (
                    other_precision_float >= row_precision
                    and other_coverage_float >= row_coverage
                    and (
                        other_precision_float > row_precision
                        or other_coverage_float > row_coverage
                    )
                ):
                    dominated = True
                    break
            row["pareto"] = not dominated
    return annotated


def pareto_frontier_rows(
    rows: Sequence[dict[str, Any]],
    *,
    group_keys: Sequence[str] = ("experiment_id", "layer", "candidate_id", "split"),
    precision_key: str = "accepted_precision",
    coverage_key: str = "coverage",
) -> list[dict[str, Any]]:
    return [
        row
        for row in annotate_pareto_frontier(
            rows,
            group_keys=group_keys,
            precision_key=precision_key,
            coverage_key=coverage_key,
        )
        if row.get("pareto")
    ]


def seaborn_available() -> bool:
    try:
        import seaborn  # noqa: F401
    except ImportError:
        return False
    return True


def plot_evolution_curve(
    rows: Sequence[dict[str, Any]],
    output_path: Path,
    *,
    title: str,
    x_key: str = "round",
) -> Path:
    plot_rows: list[dict[str, Any]] = []
    for row in rows:
        x_value = row.get(x_key)
        if x_value is None:
            continue
        if row.get("accepted_precision") is not None:
            plot_rows.append(
                {
                    "step": x_value,
                    "metric": "Accepted precision",
                    "value": float(row["accepted_precision"]),
                    "split": str(row.get("split", "unknown")),
                    "candidate_id": str(row.get("candidate_id", "")),
                }
            )
        if row.get("coverage") is not None:
            plot_rows.append(
                {
                    "step": x_value,
                    "metric": "Coverage",
                    "value": float(row["coverage"]),
                    "split": str(row.get("split", "unknown")),
                    "candidate_id": str(row.get("candidate_id", "")),
                }
            )
    if not plot_rows:
        raise ValueError("no plottable precision/coverage evolution rows")

    sns, plt, mticker, pd = _plotting_modules()
    dataframe = pd.DataFrame(plot_rows)
    dataframe = dataframe.sort_values(["split", "metric", "step"])
    dataframe["split"] = dataframe["split"].map(_display_label)

    sns.set_theme(style="whitegrid", context="talk", palette="colorblind")
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    sns.lineplot(
        data=dataframe,
        x="step",
        y="value",
        hue="metric",
        style="split",
        markers=True,
        dashes=True,
        errorbar=None,
        ax=ax,
    )
    ax.axhline(0.99, color="0.35", linestyle=":", linewidth=1.3)
    ax.text(
        0.02,
        0.99,
        "99% precision gate",
        color="0.25",
        fontsize=10,
        ha="left",
        va="bottom",
        transform=ax.get_yaxis_transform(),
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
    )
    ax.set_title(title, pad=14)
    ax.set_xlabel("Round / generation")
    ax.set_ylabel("Rate")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylim(_rate_axis_limits(dataframe["value"].tolist()))
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_operating_frontier(
    rows: Sequence[dict[str, Any]],
    frontier_rows: Sequence[dict[str, Any]],
    output_path: Path,
    *,
    title: str,
    hue_key: str = "policy_family",
    style_key: str = "selection_scope",
) -> Path:
    point_rows = [
        row
        for row in rows
        if row.get("accepted_precision") is not None and row.get("coverage") is not None
    ]
    if not point_rows:
        raise ValueError("no plottable precision/coverage operating rows")

    sns, plt, mticker, pd = _plotting_modules()
    points = pd.DataFrame(point_rows)
    hue_label = _display_label(hue_key)
    style_label = _display_label(style_key)
    if hue_key not in points:
        points[hue_key] = "policy"
    points[hue_label] = points[hue_key].fillna("policy").map(_display_label)
    if style_key not in points:
        points[style_key] = "agent_visible"
    points[style_label] = points[style_key].fillna("agent_visible").map(_display_label)

    sns.set_theme(style="whitegrid", context="talk", palette="colorblind")
    fig, ax = plt.subplots(figsize=(10.2, 6.1))
    sns.scatterplot(
        data=points,
        x="coverage",
        y="accepted_precision",
        hue=hue_label,
        style=style_label,
        s=82,
        alpha=0.86,
        ax=ax,
    )

    frontier_points = [
        row
        for row in frontier_rows
        if row.get("accepted_precision") is not None and row.get("coverage") is not None
    ]
    if frontier_points:
        frontier = pd.DataFrame(frontier_points)
        frontier["split"] = frontier["split"].map(_display_label)
        frontier["candidate_id"] = frontier["candidate_id"].map(_display_label)
        frontier = frontier.sort_values(["split", "candidate_id", "coverage"])
        for (_split, _candidate), group in frontier.groupby(["split", "candidate_id"]):
            if len(group) == 1:
                ax.scatter(
                    group["coverage"],
                    group["accepted_precision"],
                    s=150,
                    facecolors="none",
                    edgecolors="black",
                    linewidths=1.4,
                    zorder=4,
                )
                continue
            ax.plot(
                group["coverage"],
                group["accepted_precision"],
                color="black",
                linewidth=1.5,
                alpha=0.72,
                zorder=3,
            )
            ax.scatter(
                group["coverage"],
                group["accepted_precision"],
                s=150,
                facecolors="none",
                edgecolors="black",
                linewidths=1.4,
                zorder=4,
            )

    ax.axhline(0.99, color="0.35", linestyle=":", linewidth=1.3)
    ax.text(
        0.995,
        0.99,
        "99% precision gate",
        color="0.25",
        fontsize=10,
        ha="right",
        va="bottom",
        transform=ax.get_yaxis_transform(),
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
    )
    ax.set_title(title, pad=14)
    ax.set_xlabel("Accepted coverage")
    ax.set_ylabel("Accepted precision")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlim(_rate_axis_limits(points["coverage"].tolist(), lower_floor=0.0))
    ax.set_ylim(_rate_axis_limits(points["accepted_precision"].tolist()))
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def _plotting_modules() -> tuple[Any, Any, Any, Any]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import pandas as pd
    import seaborn as sns

    return sns, plt, mticker, pd


def _rate_axis_limits(
    values: Sequence[float],
    *,
    lower_floor: float | None = None,
) -> tuple[float, float]:
    finite_values = [float(value) for value in values if value is not None]
    if not finite_values:
        return (0.0, 1.0)
    low = min(finite_values)
    high = max(finite_values)
    pad = max((high - low) * 0.12, 0.015)
    lower = max(0.0, low - pad)
    if lower_floor is not None:
        lower = min(lower, lower_floor)
    upper = min(1.005 if high <= 1.0 else 1.02, high + pad)
    if upper - lower < 0.05:
        lower = max(0.0, upper - 0.05)
    return (lower, upper)


def _display_label(value: Any) -> str:
    return str(value).replace("_", " ")
