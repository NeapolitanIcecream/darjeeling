from __future__ import annotations

import json
import math
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
OPERATING_CURVE_REQUIRED_FIELDS = (
    "curve_id",
    "knob_name",
    "knob_label",
    "knob_order",
    "knob_direction",
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

    sns.set_theme(style="whitegrid", context="notebook", palette="colorblind")
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


def validate_operating_curve_rows(
    rows: Sequence[dict[str, Any]],
    *,
    require_single_curve: bool = True,
) -> list[dict[str, Any]]:
    """Validate normalized rows that may be connected as operating curves.

    The helper is target-neutral: it only checks that rows carry a stable curve
    identity, a single ordered knob inside each curve, and one selection scope.
    Target packages decide what the knob means.
    """

    plottable_rows = [
        normalize_precision_coverage_row(row)
        for row in rows
        if row.get("accepted_precision") is not None and row.get("coverage") is not None
    ]
    if not plottable_rows:
        raise ValueError("no plottable operating curve rows")

    missing_by_index: list[str] = []
    for index, row in enumerate(plottable_rows):
        missing = [field for field in OPERATING_CURVE_REQUIRED_FIELDS if field not in row]
        if missing:
            missing_by_index.append(f"row {index}: {', '.join(missing)}")
    if missing_by_index:
        raise ValueError(
            "operating curve row missing required fields: " + "; ".join(missing_by_index)
        )

    curve_ids = {str(row["curve_id"]) for row in plottable_rows}
    if require_single_curve and len(curve_ids) != 1:
        raise ValueError("operating curve rows must share one curve_id")

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in plottable_rows:
        groups.setdefault(str(row["curve_id"]), []).append(row)

    for curve_id, group in groups.items():
        knob_names = {str(row["knob_name"]) for row in group}
        if len(knob_names) != 1:
            raise ValueError(f"operating curve {curve_id} has mixed knob_name values")
        directions = {str(row["knob_direction"]) for row in group}
        if len(directions) != 1:
            raise ValueError(f"operating curve {curve_id} has mixed knob_direction values")
        scopes = {str(row.get("selection_scope", "agent_visible")) for row in group}
        if len(scopes) != 1:
            raise ValueError(f"operating curve {curve_id} has mixed selection_scope values")
        orders = [int(row["knob_order"]) for row in group]
        if len(orders) != len(set(orders)):
            raise ValueError(f"operating curve {curve_id} has duplicate knob_order values")

    return sorted(
        plottable_rows,
        key=lambda row: (str(row["curve_id"]), int(row["knob_order"])),
    )


def plot_single_operating_curve(
    rows: Sequence[dict[str, Any]],
    output_path: Path,
    *,
    title: str,
    subtitle: str | None = None,
) -> Path:
    """Render one ordered operating curve for one curve_id."""

    curve_rows = validate_operating_curve_rows(rows, require_single_curve=True)
    sns, plt, mticker, pd = _plotting_modules()
    dataframe = pd.DataFrame(curve_rows).sort_values("knob_order")

    sns.set_theme(style="whitegrid", context="notebook", palette="colorblind")
    fig, ax = plt.subplots(figsize=(9.4, 6.0))
    _draw_operating_curve(ax, dataframe, plt=plt, annotate_points=True)
    _format_operating_curve_axis(
        ax,
        mticker=mticker,
        title=title,
        subtitle=subtitle,
        coverage_values=dataframe["coverage"].tolist(),
        precision_values=dataframe["accepted_precision"].tolist(),
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_operating_curve_facets(
    rows: Sequence[dict[str, Any]],
    output_path: Path,
    *,
    title: str,
    columns: int = 2,
) -> Path:
    """Render multiple operating curves as explicit facets.

    Each subplot is still one curve_id. The helper never connects points across
    curve boundaries.
    """

    curve_rows = validate_operating_curve_rows(rows, require_single_curve=False)
    sns, plt, mticker, pd = _plotting_modules()
    dataframe = pd.DataFrame(curve_rows)
    curve_ids = list(dict.fromkeys(dataframe["curve_id"].astype(str).tolist()))
    if not curve_ids:
        raise ValueError("no operating curves to facet")

    columns = max(1, min(columns, len(curve_ids)))
    rows_count = (len(curve_ids) + columns - 1) // columns
    sns.set_theme(style="whitegrid", context="talk", palette="colorblind")
    fig, axes = plt.subplots(
        rows_count,
        columns,
        figsize=(8.2 * columns, 5.7 * rows_count),
        squeeze=False,
        sharex=False,
        sharey=False,
    )
    for index, curve_id in enumerate(curve_ids):
        ax = axes[index // columns][index % columns]
        group = dataframe[dataframe["curve_id"].astype(str) == curve_id].sort_values(
            "knob_order"
        )
        _draw_operating_curve(ax, group, plt=plt, annotate_points=True)
        _format_operating_curve_axis(
            ax,
            mticker=mticker,
            title=str(group.iloc[0].get("curve_title") or _display_label(curve_id)),
            subtitle=None,
            coverage_values=group["coverage"].tolist(),
            precision_values=group["accepted_precision"].tolist(),
        )
    for index in range(len(curve_ids), rows_count * columns):
        axes[index // columns][index % columns].axis("off")
    fig.suptitle(title, y=1.02, fontsize=18)
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
        group_columns = ["curve_id"] if "curve_id" in frontier else ["split", "candidate_id"]
        frontier = frontier.sort_values([*group_columns, "coverage"])
        for _, group in frontier.groupby(group_columns, sort=False):
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


def _draw_operating_curve(
    ax: Any,
    dataframe: Any,
    *,
    plt: Any,
    annotate_points: bool,
) -> None:
    palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#0072B2"])
    color = palette[0]
    ax.plot(
        dataframe["coverage"],
        dataframe["accepted_precision"],
        color=color,
        linewidth=2.2,
        marker="o",
        markersize=6.5,
        zorder=3,
    )
    for _, row in dataframe.iterrows():
        point_role = str(row.get("point_role") or "")
        if point_role:
            marker_size = 92 if "endpoint" in point_role else 80
            ax.scatter(
                [row["coverage"]],
                [row["accepted_precision"]],
                s=marker_size,
                color=color,
                edgecolors="black",
                linewidths=0.8,
                zorder=4,
            )

    if len(dataframe) >= 2:
        direction = _display_label(
            str(dataframe.iloc[0].get("knob_direction") or "knob direction")
        ).replace(" to ", " -> ")
        ax.text(
            0.02,
            0.04,
            f"order: {direction}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            color="0.25",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 1.8},
            zorder=5,
        )

    if annotate_points:
        for _, row in _annotation_rows(dataframe).iterrows():
            y_value = float(row["accepted_precision"])
            x_value = float(row["coverage"])
            x_offset = -8 if x_value >= 0.82 else 8
            y_offset = -12 if y_value >= 0.985 else 8
            horizontal_alignment = "right" if x_offset < 0 else "left"
            vertical_alignment = "top" if y_offset < 0 else "bottom"
            ax.annotate(
                _operating_point_label(row),
                xy=(row["coverage"], row["accepted_precision"]),
                xytext=(x_offset, y_offset),
                textcoords="offset points",
                fontsize=9,
                ha=horizontal_alignment,
                va=vertical_alignment,
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": "0.82",
                    "alpha": 0.9,
                },
                arrowprops={"arrowstyle": "-", "color": "0.65", "linewidth": 0.8},
                zorder=5,
            )


def _format_operating_curve_axis(
    ax: Any,
    *,
    mticker: Any,
    title: str,
    subtitle: str | None,
    coverage_values: Sequence[float],
    precision_values: Sequence[float],
) -> None:
    ax.axhline(0.99, color="0.35", linestyle=":", linewidth=1.3)
    ax.text(
        0.995,
        0.99,
        "99% precision gate",
        color="0.25",
        fontsize=9,
        ha="right",
        va="bottom",
        transform=ax.get_yaxis_transform(),
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
    )
    ax.set_title(title if subtitle is None else f"{title}\n{subtitle}", pad=14)
    ax.set_xlabel("Accepted coverage")
    ax.set_ylabel("Accepted precision")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlim(_rate_axis_limits(coverage_values, lower_floor=0.0))
    y_limits = _rate_axis_limits([*precision_values, 0.99])
    ax.set_ylim(y_limits)
    if max([*precision_values, 0.99]) <= 1.0 and y_limits[1] > 1.0:
        step = 0.01 if 1.0 - y_limits[0] <= 0.065 else 0.02
        start = math.ceil(y_limits[0] / step) * step
        ticks = []
        current = start
        while current <= 1.0 + 1e-9:
            ticks.append(round(current, 4))
            current += step
        ax.set_yticks(ticks)


def _annotation_rows(dataframe: Any) -> Any:
    if "point_role" in dataframe:
        annotated = dataframe[dataframe["point_role"].fillna("").astype(str) != ""]
        if not annotated.empty:
            return annotated
    indexes = {dataframe.index[0], dataframe.index[-1]}
    if len(dataframe) > 2:
        indexes.add(dataframe.index[len(dataframe) // 2])
    return dataframe.loc[list(indexes)].sort_values("knob_order")


def _operating_point_label(row: Any) -> str:
    label = row.get("annotation_label")
    if label:
        return str(label)
    point_role = _display_label(row.get("point_role") or "")
    knob_label = str(row.get("knob_label") or row.get("knob_value") or "")
    if point_role:
        return f"{point_role}\n{knob_label}"
    return knob_label


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
    upper = min(1.006 if high <= 1.0 else 1.02, high + pad)
    if upper - lower < 0.05:
        lower = max(0.0, upper - 0.05)
    return (lower, upper)


def _display_label(value: Any) -> str:
    return str(value).replace("_", " ")
