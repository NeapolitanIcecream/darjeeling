from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from darjeeling.eval.plots import (
    annotate_pareto_frontier,
    plot_evolution_curve,
    plot_operating_curve_facets,
    plot_single_operating_curve,
    write_normalized_jsonl,
)

CLINC150_L1_AGENT_SESSION_EXPERIMENT_ID = "clinc150-l1-agent-session-effect"
CLINC150_L2_CASCADE_EXPERIMENT_ID = "clinc150-l2-cascade"
CLINC150_CALIBRATION_REPAIR_EXPERIMENT_ID = "clinc150-calibration-repair"
CLINC150_L2_AUTORESEARCH_EXPERIMENT_ID = "clinc150-l2-autoresearch"

CLINC150_STANDARD_L2_THRESHOLDS = (
    0.995,
    0.99,
    0.985,
    0.98,
    0.95,
    0.90,
    0.80,
    0.70,
    0.60,
    0.50,
)


@dataclass(frozen=True)
class PrecisionCoverageBackfillResult:
    output_dir: Path
    round_metrics_path: Path
    operating_points_path: Path
    pareto_frontier_path: Path
    figure_paths: tuple[Path, ...]
    round_metric_count: int
    operating_point_count: int
    pareto_point_count: int


def backfill_clinc150_precision_coverage(
    *,
    output_dir: Path,
    l1_summary_path: Path,
    l2_cascade_root: Path,
    calibration_summary_paths: tuple[Path, ...] = (),
    autoresearch_summary_path: Path | None = None,
) -> PrecisionCoverageBackfillResult:
    """Backfill normalized precision/coverage rows and static figures from artifacts."""

    round_rows: list[dict[str, Any]] = []
    operating_rows: list[dict[str, Any]] = []

    round_rows.extend(clinc150_l1_round_metric_rows(l1_summary_path))
    operating_rows.extend(clinc150_l1_operating_point_rows(l1_summary_path))

    round_rows.extend(clinc150_l2_round_metric_rows(l2_cascade_root))
    operating_rows.extend(clinc150_l2_operating_point_rows(l2_cascade_root))

    for summary_path in calibration_summary_paths:
        operating_rows.extend(clinc150_calibration_repair_discrete_points(summary_path))

    if autoresearch_summary_path is not None:
        round_rows.extend(clinc150_l2_autoresearch_round_rows(autoresearch_summary_path))
        operating_rows.extend(clinc150_l2_autoresearch_discrete_points(autoresearch_summary_path))

    annotated_operating_rows = annotate_pareto_frontier(
        operating_rows,
        group_keys=("experiment_id", "layer", "candidate_id", "split", "curve_id"),
    )
    pareto_rows = [row for row in annotated_operating_rows if row.get("pareto")]

    output_dir.mkdir(parents=True, exist_ok=True)
    round_metrics_path = write_normalized_jsonl(
        round_rows,
        output_dir / "round_metrics.jsonl",
    )
    operating_points_path = write_normalized_jsonl(
        annotated_operating_rows,
        output_dir / "operating_points.jsonl",
    )
    pareto_frontier_path = write_normalized_jsonl(
        pareto_rows,
        output_dir / "pareto_frontier.jsonl",
    )
    figure_paths = tuple(
        _write_standard_figures(
            output_dir=output_dir,
            round_rows=round_rows,
            operating_rows=annotated_operating_rows,
            pareto_rows=pareto_rows,
        )
    )
    return PrecisionCoverageBackfillResult(
        output_dir=output_dir,
        round_metrics_path=round_metrics_path,
        operating_points_path=operating_points_path,
        pareto_frontier_path=pareto_frontier_path,
        figure_paths=figure_paths,
        round_metric_count=len(round_rows),
        operating_point_count=len(annotated_operating_rows),
        pareto_point_count=len(pareto_rows),
    )


def clinc150_l1_round_metric_rows(summary_path: Path) -> list[dict[str, Any]]:
    summary = _load_json(summary_path)
    source_repo_dir = _source_repo_dir(summary)
    rows: list[dict[str, Any]] = []
    for round_payload in summary.get("rounds", []):
        round_number = int(round_payload.get("round") or len(rows) + 1)
        candidate_id = _round_candidate_id(round_number)
        for split, evaluation in sorted((round_payload.get("evaluations") or {}).items()):
            metrics = ((evaluation.get("summary") or {}).get("l1_only") or {})
            if not metrics:
                continue
            rows.append(
                _metric_row(
                    experiment_id=CLINC150_L1_AGENT_SESSION_EXPERIMENT_ID,
                    layer="L1",
                    candidate_id=candidate_id,
                    round_number=round_number,
                    split=split,
                    view=_view_for_split(split),
                    metrics=metrics,
                    source_artifact=_resolve_source_artifact(
                        evaluation.get("summary_path") or evaluation.get("details_jsonl_path"),
                        source_repo_dir=source_repo_dir,
                        fallback_path=summary_path,
                    ),
                    selection_scope="agent_visible",
                    metadata={
                        "candidate_eligible": bool(round_payload.get("candidate_eligible")),
                        "failure_classification": round_payload.get("failure_classification"),
                    },
                )
            )

    locked_test = summary.get("locked_test")
    selected_round = (summary.get("selected_round") or {}).get("round") or 1
    if isinstance(locked_test, dict):
        metrics = ((locked_test.get("summary") or {}).get("l1_only") or {})
        if metrics:
            rows.append(
                _metric_row(
                    experiment_id=CLINC150_L1_AGENT_SESSION_EXPERIMENT_ID,
                    layer="L1",
                    candidate_id=_round_candidate_id(int(selected_round)),
                    round_number=int(selected_round),
                    split="locked_test",
                    view="sequential",
                    metrics=metrics,
                    source_artifact=_resolve_source_artifact(
                        locked_test.get("summary_path") or locked_test.get("details_jsonl_path"),
                        source_repo_dir=source_repo_dir,
                        fallback_path=summary_path,
                    ),
                    selection_scope="locked_test_diagnostic",
                    metadata={"diagnostic_only": True},
                )
            )
    return rows


def clinc150_l1_operating_point_rows(summary_path: Path) -> list[dict[str, Any]]:
    summary = _load_json(summary_path)
    source_repo_dir = _source_repo_dir(summary)
    selected_round = _l1_operating_curve_round(summary)
    candidate_id = _round_candidate_id(selected_round)
    selected_payload = _round_payload(summary, selected_round)
    if selected_payload is None:
        return []
    selected_for_locked_test = bool((summary.get("selected_round") or {}).get("round"))

    train_dev_evaluation = (selected_payload.get("evaluations") or {}).get("train_dev")
    if not train_dev_evaluation:
        return []
    support_path = _resolve_source_artifact(
        train_dev_evaluation.get("details_jsonl_path"),
        source_repo_dir=source_repo_dir,
        fallback_path=summary_path,
    )
    support_rows = _read_jsonl(support_path)
    support = _l1_rule_support(support_rows)
    policies = _l1_risk_tolerance_policies()

    rows: list[dict[str, Any]] = []
    evaluations = {
        split: evaluation
        for split, evaluation in (selected_payload.get("evaluations") or {}).items()
        if split in {"train_dev", "visible_validation"}
    }
    locked_test = summary.get("locked_test")
    if isinstance(locked_test, dict):
        evaluations["locked_test"] = locked_test

    for split in ("train_dev", "visible_validation", "locked_test"):
        evaluation = evaluations.get(split)
        if evaluation is None:
            continue
        details_path = _resolve_source_artifact(
            evaluation.get("details_jsonl_path"),
            source_repo_dir=source_repo_dir,
            fallback_path=summary_path,
        )
        if not details_path.exists():
            continue
        prediction_rows = _read_jsonl(details_path)
        for policy in policies:
            metrics = _l1_overlay_metrics(
                prediction_rows,
                support=support,
                policy=policy,
            )
            rows.append(
                _operating_row(
                    experiment_id=CLINC150_L1_AGENT_SESSION_EXPERIMENT_ID,
                    layer="L1",
                    candidate_id=candidate_id,
                    round_number=selected_round,
                    split=split,
                    view=_view_for_split(split),
                    policy_family="l1_risk_tolerance",
                    policy_label=str(policy["knob_label"]),
                    policy_value=policy["knob_value"],
                    curve_id=_curve_id(
                        CLINC150_L1_AGENT_SESSION_EXPERIMENT_ID,
                        "L1",
                        candidate_id,
                        split,
                        "risk-tolerance",
                    ),
                    curve_role=(
                        "diagnostic_operating_curve"
                        if split == "locked_test"
                        else "standard_operating_curve"
                    ),
                    knob_name="risk_tolerance",
                    knob_value=policy["knob_value"],
                    knob_label=str(policy["knob_label"]),
                    knob_order=int(policy["knob_order"]),
                    knob_direction="strict_to_loose",
                    primary_curve=split != "locked_test",
                    point_role=str(policy.get("point_role") or ""),
                    annotation_label=str(policy.get("annotation_label") or ""),
                    curve_title=_curve_title(
                        layer="L1",
                        split=split,
                        knob_name="risk_tolerance",
                    ),
                    metrics=metrics,
                    source_artifact=details_path,
                    selection_scope=(
                        "locked_test_diagnostic"
                        if split == "locked_test"
                        else "agent_visible"
                    ),
                    metadata={
                        "overlay_semantics": (
                            "target_adapter_posthoc_filter_over_recorded_l1_accepts"
                        ),
                        "candidate_eligible": bool(
                            selected_payload.get("candidate_eligible")
                        ),
                        "selected_for_locked_test": selected_for_locked_test,
                        "selection_basis": (
                            "selected_round"
                            if selected_for_locked_test
                            else "best_visible_validation_coverage_diagnostic"
                        ),
                        "support_split": "train_dev",
                        "support_artifact": str(support_path),
                        "risk_tolerance_rule": policy["rule_description"],
                    },
                )
            )
    return rows


def _l1_operating_curve_round(summary: dict[str, Any]) -> int:
    selected_round = (summary.get("selected_round") or {}).get("round")
    if selected_round is not None:
        return int(selected_round)

    candidates: list[tuple[float, float, int, int]] = []
    for index, round_payload in enumerate(summary.get("rounds", []), start=1):
        round_number = int(round_payload.get("round") or index)
        evaluations = round_payload.get("evaluations") or {}
        if "train_dev" not in evaluations:
            continue
        visible_metrics = (
            ((evaluations.get("visible_validation") or {}).get("summary") or {}).get(
                "l1_only"
            )
            or {}
        )
        train_metrics = (
            ((evaluations.get("train_dev") or {}).get("summary") or {}).get("l1_only")
            or {}
        )
        metrics = visible_metrics or train_metrics
        coverage = _metric_float(metrics, "coverage", "accepted_coverage") or 0.0
        precision = _metric_float(metrics, "accepted_precision") or 0.0
        wrong_accepts = _metric_int(metrics, "wrong_accepts", "accepted_wrong") or 0
        candidates.append((coverage, precision, -wrong_accepts, round_number))
    if not candidates:
        return 1
    return max(candidates)[3]


def clinc150_l2_round_metric_rows(l2_cascade_root: Path) -> list[dict[str, Any]]:
    candidates = (
        ("teacher-500", 1, "distilled-l2/train-500/validation-cascade", "validation"),
        ("teacher-3000", 2, "distilled-l2/train-3000/validation-cascade", "validation"),
        (
            "teacher-3000-retrieval",
            3,
            "distilled-l2/train-3000-retrieval/validation-cascade",
            "validation",
        ),
        ("teacher-3000-mlp", 4, "distilled-l2/train-3000-mlp/validation-cascade", "validation"),
        ("teacher-full", 5, "distilled-l2/train-full/validation-cascade", "validation"),
        ("teacher-full", 5, "distilled-l2/train-full/test-cascade", "locked_test"),
        (
            "teacher-full",
            5,
            "distilled-l2/train-full/validation-uniform-cascade",
            "validation_uniform",
        ),
        (
            "teacher-full",
            5,
            "distilled-l2/train-full/validation-zipf-heavy-cascade",
            "validation_zipf_heavy",
        ),
    )
    rows: list[dict[str, Any]] = []
    for candidate_id, round_number, relative_dir, split in candidates:
        summary_path = l2_cascade_root / relative_dir / "clinc150_l2_eval_summary.json"
        if not summary_path.exists():
            continue
        summary = _load_json(summary_path)
        metrics = _selected_l2_summary_threshold(summary)
        if metrics is None:
            continue
        rows.append(
            _metric_row(
                experiment_id=CLINC150_L2_CASCADE_EXPERIMENT_ID,
                layer="L2",
                candidate_id=candidate_id,
                round_number=round_number,
                split=split,
                view=_view_for_split(split),
                metrics=metrics,
                source_artifact=summary_path,
                selection_scope=(
                    "locked_test_diagnostic"
                    if split == "locked_test"
                    else "agent_visible"
                ),
                metadata={
                    "policy_family": "l2_guard_threshold",
                    "policy_label": _l2_policy_label(metrics),
                    "artifact_support": "full_summary_threshold_row",
                },
            )
        )
    return rows


def clinc150_l2_operating_point_rows(l2_cascade_root: Path) -> list[dict[str, Any]]:
    prediction_artifacts = (
        ("teacher-full", 5, "validation", "distilled-l2/train-full/validation-cascade"),
        ("teacher-full", 5, "locked_test", "distilled-l2/train-full/test-cascade"),
        (
            "teacher-full",
            5,
            "validation_uniform",
            "distilled-l2/train-full/validation-uniform-cascade",
        ),
        (
            "teacher-full",
            5,
            "validation_zipf_heavy",
            "distilled-l2/train-full/validation-zipf-heavy-cascade",
        ),
    )
    rows: list[dict[str, Any]] = []
    for candidate_id, round_number, split, relative_dir in prediction_artifacts:
        prediction_path = l2_cascade_root / relative_dir / "clinc150_l2_predictions.jsonl"
        if not prediction_path.exists():
            continue
        summary_path = l2_cascade_root / relative_dir / "clinc150_l2_eval_summary.json"
        selected_threshold = _selected_l2_threshold_value(summary_path)
        prediction_rows = _read_jsonl(prediction_path)
        for knob_order, threshold in enumerate(CLINC150_STANDARD_L2_THRESHOLDS):
            metrics = _l2_threshold_metrics(prediction_rows, threshold=threshold)
            point_role, annotation_label = _l2_threshold_point_annotation(
                threshold,
                selected_threshold=selected_threshold,
            )
            rows.append(
                _operating_row(
                    experiment_id=CLINC150_L2_CASCADE_EXPERIMENT_ID,
                    layer="L2",
                    candidate_id=candidate_id,
                    round_number=round_number,
                    split=split,
                    view=_view_for_split(split),
                    policy_family="guard_threshold",
                    policy_label=f"guard >= {threshold:g}",
                    policy_value=threshold,
                    curve_id=_curve_id(
                        CLINC150_L2_CASCADE_EXPERIMENT_ID,
                        "L2",
                        candidate_id,
                        split,
                        "guard-threshold",
                    ),
                    curve_role=(
                        "diagnostic_operating_curve"
                        if split == "locked_test"
                        else (
                            "standard_operating_curve"
                            if split == "validation"
                            else "context_operating_curve"
                        )
                    ),
                    knob_name="guard_threshold",
                    knob_value=threshold,
                    knob_label=f"guard >= {threshold:g}",
                    knob_order=knob_order,
                    knob_direction="strict_to_loose",
                    primary_curve=split == "validation",
                    point_role=point_role,
                    annotation_label=annotation_label,
                    curve_title=_curve_title(
                        layer="L2",
                        split=split,
                        knob_name="guard_threshold",
                    ),
                    metrics=metrics,
                    source_artifact=prediction_path,
                    selection_scope=(
                        "locked_test_diagnostic"
                        if split == "locked_test"
                        else "agent_visible"
                    ),
                    metadata={
                        "artifact_support": "per_request_threshold_sweep",
                        "selected_summary_threshold": selected_threshold,
                    },
                )
            )
    return rows


def clinc150_calibration_repair_discrete_points(summary_path: Path) -> list[dict[str, Any]]:
    summary = _load_json(summary_path)
    rows: list[dict[str, Any]] = []
    selected = summary.get("selected") or {}
    for split in ("calibration_dev", "validation", "oos_heavy"):
        metrics = selected.get(split)
        if isinstance(metrics, dict):
            rows.append(
                _summary_operating_row(
                    experiment_id=CLINC150_CALIBRATION_REPAIR_EXPERIMENT_ID,
                    candidate_id="selected-guard",
                    round_number=1,
                    split=split,
                    metrics=metrics,
                    summary_path=summary_path,
                    selection_scope="agent_visible",
                    metadata={"artifact_support": "partial_summary_only"},
                )
            )
    locked_test = summary.get("locked_test")
    if isinstance(locked_test, dict):
        rows.append(
            _summary_operating_row(
                experiment_id=CLINC150_CALIBRATION_REPAIR_EXPERIMENT_ID,
                candidate_id="selected-guard",
                round_number=1,
                split="locked_test",
                metrics=locked_test,
                summary_path=summary_path,
                selection_scope="locked_test_diagnostic",
                metadata={"artifact_support": "partial_summary_only"},
            )
        )
    for split, metrics in (summary.get("stream_confirmation") or {}).items():
        if isinstance(metrics, dict):
            rows.append(
                _summary_operating_row(
                    experiment_id=CLINC150_CALIBRATION_REPAIR_EXPERIMENT_ID,
                    candidate_id="selected-guard",
                    round_number=1,
                    split=str(split),
                    metrics=metrics,
                    summary_path=summary_path,
                    selection_scope="agent_visible",
                    metadata={"artifact_support": "partial_summary_only"},
                )
            )
    return rows


def clinc150_l2_autoresearch_round_rows(summary_path: Path) -> list[dict[str, Any]]:
    summary = _load_json(summary_path)
    metrics = ((summary.get("validation_evaluation") or {}).get("metrics") or {})
    if not metrics:
        return []
    return [
        _metric_row(
            experiment_id=CLINC150_L2_AUTORESEARCH_EXPERIMENT_ID,
            layer="L2",
            candidate_id="round-001-diagnostic",
            round_number=1,
            split="validation",
            view="sequential",
            metrics=metrics,
            source_artifact=summary_path,
            selection_scope="agent_visible",
            metadata={
                "artifact_support": "partial_summary_only",
                "selected_for_locked_test": False,
            },
        )
    ]


def clinc150_l2_autoresearch_discrete_points(summary_path: Path) -> list[dict[str, Any]]:
    summary = _load_json(summary_path)
    rows: list[dict[str, Any]] = []
    validation_metrics = ((summary.get("validation_evaluation") or {}).get("metrics") or {})
    if validation_metrics:
        rows.append(
            _summary_operating_row(
                experiment_id=CLINC150_L2_AUTORESEARCH_EXPERIMENT_ID,
                candidate_id="round-001-diagnostic",
                round_number=1,
                split="validation",
                metrics=validation_metrics,
                summary_path=summary_path,
                selection_scope="agent_visible",
                metadata={
                    "artifact_support": "partial_summary_only",
                    "selected_for_locked_test": False,
                },
            )
        )
    for split, metrics in (summary.get("stream_confirmation") or {}).items():
        if isinstance(metrics, dict):
            rows.append(
                _summary_operating_row(
                    experiment_id=CLINC150_L2_AUTORESEARCH_EXPERIMENT_ID,
                    candidate_id="round-001-diagnostic",
                    round_number=1,
                    split=str(split),
                    metrics=metrics,
                    summary_path=summary_path,
                    selection_scope="agent_visible",
                    metadata={
                        "artifact_support": "partial_summary_only",
                        "selected_for_locked_test": False,
                    },
                )
            )
    return rows


def _write_standard_figures(
    *,
    output_dir: Path,
    round_rows: list[dict[str, Any]],
    operating_rows: list[dict[str, Any]],
    pareto_rows: list[dict[str, Any]],
) -> list[Path]:
    del pareto_rows
    figures_dir = output_dir / "figures"
    figure_paths: list[Path] = []

    l1_round_rows = [
        row
        for row in round_rows
        if row["layer"] == "L1" and row["split"] in {"train_dev", "visible_validation"}
    ]
    figure_paths.append(
        plot_evolution_curve(
            l1_round_rows,
            figures_dir / "clinc150_l1_evolution.png",
            title="CLINC150 L1 Agent-Session Round Evolution",
        )
    )

    l1_candidate_id = _l1_operating_candidate_id(operating_rows)
    l1_train_dev_curve = _standard_curve_rows(
        operating_rows,
        layer="L1",
        candidate_id=l1_candidate_id,
        split="train_dev",
        knob_name="risk_tolerance",
    )
    _append_single_operating_curve_figure(
        figure_paths=figure_paths,
        rows=l1_train_dev_curve,
        output_path=figures_dir / "clinc150_l1_train_dev_operating_curve.png",
        title="CLINC150 L1 Train-Dev Operating Curve",
        subtitle="target-adapter risk_tolerance overlay",
    )

    l1_validation_curve = _standard_curve_rows(
        operating_rows,
        layer="L1",
        candidate_id=l1_candidate_id,
        split="visible_validation",
        knob_name="risk_tolerance",
    )
    _append_single_operating_curve_figure(
        figure_paths=figure_paths,
        rows=l1_validation_curve,
        output_path=figures_dir / "clinc150_l1_validation_operating_curve.png",
        title="CLINC150 L1 Visible-Validation Operating Curve",
        subtitle="target-adapter risk_tolerance overlay",
    )

    l1_locked_curve = _standard_curve_rows(
        operating_rows,
        layer="L1",
        candidate_id=l1_candidate_id,
        split="locked_test",
        knob_name="risk_tolerance",
    )
    _append_single_operating_curve_figure(
        figure_paths=figure_paths,
        rows=l1_locked_curve,
        output_path=figures_dir / "clinc150_l1_locked_test_diagnostic_curve.png",
        title="CLINC150 L1 Locked-Test Diagnostic Curve",
        subtitle="same visible risk_tolerance overlay, diagnostic only",
    )

    l2_validation_curve = _standard_curve_rows(
        operating_rows,
        layer="L2",
        candidate_id="teacher-full",
        split="validation",
        knob_name="guard_threshold",
    )
    _append_single_operating_curve_figure(
        figure_paths=figure_paths,
        rows=l2_validation_curve,
        output_path=figures_dir / "clinc150_l2_validation_threshold_curve.png",
        title="CLINC150 L2 Validation Threshold Curve",
        subtitle="guard_threshold sweep",
    )

    l2_locked_curve = _standard_curve_rows(
        operating_rows,
        layer="L2",
        candidate_id="teacher-full",
        split="locked_test",
        knob_name="guard_threshold",
    )
    _append_single_operating_curve_figure(
        figure_paths=figure_paths,
        rows=l2_locked_curve,
        output_path=figures_dir / "clinc150_l2_locked_test_diagnostic_curve.png",
        title="CLINC150 L2 Locked-Test Diagnostic Curve",
        subtitle="same guard_threshold sweep, diagnostic only",
    )

    visible_comparison_rows = [*l1_validation_curve, *l2_validation_curve]
    figure_paths.append(
        plot_operating_curve_facets(
            visible_comparison_rows,
            figures_dir / "clinc150_l1_l2_visible_curve_comparison.png",
            title="CLINC150 Visible Operating Curve Comparison",
            columns=2,
        )
    )

    l2_round_rows = [
        row
        for row in round_rows
        if row["layer"] == "L2"
        and row["experiment_id"] == CLINC150_L2_CASCADE_EXPERIMENT_ID
        and row["split"] == "validation"
    ]
    if l2_round_rows:
        figure_paths.append(
            plot_evolution_curve(
                l2_round_rows,
                figures_dir / "debug_clinc150_l2_evolution.png",
                title="CLINC150 L2 Candidate Evolution",
            )
        )
    return figure_paths


def _append_single_operating_curve_figure(
    *,
    figure_paths: list[Path],
    rows: list[dict[str, Any]],
    output_path: Path,
    title: str,
    subtitle: str,
) -> None:
    if not _has_plottable_precision_coverage_rows(rows):
        output_path.unlink(missing_ok=True)
        return
    figure_paths.append(
        plot_single_operating_curve(
            rows,
            output_path,
            title=title,
            subtitle=subtitle,
        )
    )


def _has_plottable_precision_coverage_rows(rows: list[dict[str, Any]]) -> bool:
    return any(
        row.get("accepted_precision") is not None and row.get("coverage") is not None
        for row in rows
    )


def _standard_curve_rows(
    operating_rows: list[dict[str, Any]],
    *,
    layer: str,
    candidate_id: str,
    split: str,
    knob_name: str,
) -> list[dict[str, Any]]:
    return [
        row
        for row in operating_rows
        if row["layer"] == layer
        and row["candidate_id"] == candidate_id
        and row["split"] == split
        and row.get("knob_name") == knob_name
    ]


def _l1_operating_candidate_id(operating_rows: list[dict[str, Any]]) -> str:
    for row in operating_rows:
        if (
            row["experiment_id"] == CLINC150_L1_AGENT_SESSION_EXPERIMENT_ID
            and row["layer"] == "L1"
            and row["split"] == "visible_validation"
            and row.get("knob_name") == "risk_tolerance"
        ):
            return str(row["candidate_id"])
    return "round-001"


def _metric_row(
    *,
    experiment_id: str,
    layer: str,
    candidate_id: str,
    round_number: int,
    split: str,
    view: str,
    metrics: dict[str, Any],
    source_artifact: Path,
    selection_scope: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    accepted = _metric_int(metrics, "accepted")
    wrong_accepts = _metric_int(metrics, "wrong_accepts", "accepted_wrong")
    accepted_precision = _metric_float(metrics, "accepted_precision")
    if wrong_accepts is None and accepted is not None and accepted_precision is not None:
        wrong_accepts = accepted - round(accepted * accepted_precision)
    return {
        "experiment_id": experiment_id,
        "layer": layer,
        "candidate_id": candidate_id,
        "round": round_number,
        "split": split,
        "view": view,
        "accepted_precision": accepted_precision,
        "coverage": _metric_float(metrics, "coverage", "accepted_coverage"),
        "accepted": accepted,
        "wrong_accepts": wrong_accepts,
        "source_artifact": str(source_artifact),
        "selection_scope": selection_scope,
        "metadata": metadata,
    }


def _operating_row(
    *,
    experiment_id: str,
    layer: str,
    candidate_id: str,
    round_number: int,
    split: str,
    view: str,
    policy_family: str,
    policy_label: str,
    policy_value: Any,
    curve_id: str,
    curve_role: str,
    knob_name: str,
    knob_value: Any,
    knob_label: str,
    knob_order: int,
    knob_direction: str,
    primary_curve: bool,
    point_role: str,
    annotation_label: str,
    curve_title: str,
    metrics: dict[str, Any],
    source_artifact: Path,
    selection_scope: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    row = _metric_row(
        experiment_id=experiment_id,
        layer=layer,
        candidate_id=candidate_id,
        round_number=round_number,
        split=split,
        view=view,
        metrics=metrics,
        source_artifact=source_artifact,
        selection_scope=selection_scope,
        metadata=metadata,
    )
    row.update(
        {
            "policy_family": policy_family,
            "policy_label": policy_label,
            "policy_value": policy_value,
            "curve_id": curve_id,
            "curve_role": curve_role,
            "knob_name": knob_name,
            "knob_value": knob_value,
            "knob_label": knob_label,
            "knob_order": knob_order,
            "knob_direction": knob_direction,
            "primary_curve": primary_curve,
            "point_role": point_role,
            "annotation_label": annotation_label,
            "curve_title": curve_title,
        }
    )
    return row


def _summary_operating_row(
    *,
    experiment_id: str,
    candidate_id: str,
    round_number: int,
    split: str,
    metrics: dict[str, Any],
    summary_path: Path,
    selection_scope: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    guard_rule = metrics.get("guard_rule") or {}
    policy_family = str(guard_rule.get("family") or metrics.get("guard_name") or "summary_guard")
    policy_label = str(metrics.get("guard_name") or _l2_policy_label(metrics))
    policy_value = metrics.get("threshold")
    return _operating_row(
        experiment_id=experiment_id,
        layer="L2",
        candidate_id=candidate_id,
        round_number=round_number,
        split=split,
        view=_view_for_split(split),
        policy_family=policy_family,
        policy_label=policy_label,
        policy_value=policy_value,
        curve_id=_curve_id(
            experiment_id,
            "L2",
            candidate_id,
            split,
            f"{_slug(policy_family)}-discrete",
        ),
        curve_role="discrete_context",
        knob_name=policy_family,
        knob_value=policy_value,
        knob_label=policy_label,
        knob_order=0,
        knob_direction="none",
        primary_curve=False,
        point_role="context",
        annotation_label=policy_label,
        curve_title=_curve_title(layer="L2", split=split, knob_name=policy_family),
        metrics=metrics,
        source_artifact=summary_path,
        selection_scope=selection_scope,
        metadata=metadata,
    )


def _l1_risk_tolerance_policies() -> list[dict[str, Any]]:
    return [
        {
            "knob_value": 0,
            "knob_order": 0,
            "knob_label": "strict: clean support >= 20",
            "annotation_label": "strict\nsupport >= 20",
            "point_role": "strict_endpoint",
            "min_positive": 20,
            "require_no_negative": True,
            "require_no_oos_false": True,
            "rule_description": (
                "wrong_support=0, oos_false_support=0, positive_support>=20"
            ),
        },
        {
            "knob_value": 1,
            "knob_order": 1,
            "knob_label": "safe: clean support >= 10",
            "annotation_label": "",
            "point_role": "",
            "min_positive": 10,
            "require_no_negative": True,
            "require_no_oos_false": True,
            "rule_description": (
                "wrong_support=0, oos_false_support=0, positive_support>=10"
            ),
        },
        {
            "knob_value": 2,
            "knob_order": 2,
            "knob_label": "medium: clean support >= 5",
            "annotation_label": "medium\nsupport >= 5",
            "point_role": "nominal",
            "min_positive": 5,
            "require_no_negative": True,
            "require_no_oos_false": True,
            "rule_description": (
                "wrong_support=0, oos_false_support=0, positive_support>=5"
            ),
        },
        {
            "knob_value": 3,
            "knob_order": 3,
            "knob_label": "loose: clean support >= 2",
            "annotation_label": "",
            "point_role": "",
            "min_positive": 2,
            "require_no_negative": True,
            "require_no_oos_false": True,
            "rule_description": (
                "wrong_support=0, oos_false_support=0, positive_support>=2"
            ),
        },
        {
            "knob_value": 4,
            "knob_order": 4,
            "knob_label": "raw: all recorded accepts",
            "annotation_label": "raw\nall accepts",
            "point_role": "loose_endpoint",
            "min_positive": None,
            "require_no_negative": False,
            "require_no_oos_false": False,
            "rule_description": "keep all recorded L1 accepts",
        },
    ]


def _l1_rule_support(rows: list[dict[str, Any]]) -> dict[str, Counter[str]]:
    support: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        if not row.get("l1_accepted"):
            continue
        key = _l1_rule_key(row)
        support[key]["positive"] += int(bool(row.get("l1_correct")))
        support[key]["negative"] += int(not row.get("l1_correct"))
        support[key]["oos_false"] += int(
            bool(row.get("gold_oos")) and not bool(row.get("l1_oos"))
        )
    return support


def _l1_overlay_metrics(
    rows: list[dict[str, Any]],
    *,
    support: dict[str, Counter[str]],
    policy: dict[str, Any],
) -> dict[str, Any]:
    accepted_rows = [
        row
        for row in rows
        if row.get("l1_accepted") and _l1_overlay_accepts(row, support=support, policy=policy)
    ]
    correct_accepts = sum(1 for row in accepted_rows if row.get("l1_correct"))
    wrong_accepts = len(accepted_rows) - correct_accepts
    requests = len(rows)
    oos_total = sum(1 for row in rows if row.get("gold_oos"))
    oos_false_accepts = sum(
        1
        for row in accepted_rows
        if row.get("gold_oos") and not row.get("l1_oos")
    )
    return {
        "accepted": len(accepted_rows),
        "accepted_correct": correct_accepts,
        "wrong_accepts": wrong_accepts,
        "accepted_precision": _rate(correct_accepts, len(accepted_rows)),
        "accepted_coverage": _rate(len(accepted_rows), requests),
        "lower_layer_oos_false_accepts": oos_false_accepts,
        "lower_layer_oos_false_accept_rate": _rate(oos_false_accepts, oos_total) or 0.0,
    }


def _l1_overlay_accepts(
    row: dict[str, Any],
    *,
    support: dict[str, Counter[str]],
    policy: dict[str, Any],
) -> bool:
    counts = support.get(_l1_rule_key(row), Counter())
    min_positive = policy.get("min_positive")
    if min_positive is not None and counts["positive"] < int(min_positive):
        return False
    if policy.get("require_no_negative") and counts["negative"] != 0:
        return False
    if policy.get("require_no_oos_false") and counts["oos_false"] != 0:
        return False
    return True


def _l1_rule_key(row: dict[str, Any]) -> str:
    return "|".join(
        (
            str(row.get("program_path") or ""),
            str(row.get("reason") or ""),
            str(row.get("l1_intent") or ""),
        )
    )


def _l2_threshold_metrics(rows: list[dict[str, Any]], *, threshold: float) -> dict[str, Any]:
    accepted_rows = [
        row
        for row in rows
        if float(row.get("guard_probability") or 0.0) >= threshold
    ]
    correct_accepts = sum(
        1
        for row in accepted_rows
        if row.get("predicted_frame") == row.get("gold_frame")
    )
    wrong_accepts = len(accepted_rows) - correct_accepts
    oos_total = sum(1 for row in rows if row.get("gold_oos"))
    lower_layer_oos_false_accepts = sum(
        1
        for row in accepted_rows
        if row.get("gold_oos") and not row.get("predicted_oos")
    )
    return {
        "accepted": len(accepted_rows),
        "accepted_correct": correct_accepts,
        "accepted_wrong": wrong_accepts,
        "accepted_precision": _rate(correct_accepts, len(accepted_rows)),
        "accepted_coverage": _rate(len(accepted_rows), len(rows)),
        "lower_layer_oos_false_accepts": lower_layer_oos_false_accepts,
        "lower_layer_oos_false_accept_rate": _rate(lower_layer_oos_false_accepts, oos_total)
        or 0.0,
    }


def _selected_l2_summary_threshold(summary: dict[str, Any]) -> dict[str, Any] | None:
    selected = summary.get("selected_threshold")
    if isinstance(selected, dict):
        return selected
    threshold_rows = [
        row
        for row in summary.get("thresholds", [])
        if row.get("accepted_precision") is not None
    ]
    if not threshold_rows:
        return None
    eligible = [
        row
        for row in threshold_rows
        if float(row.get("accepted_precision") or 0.0) >= 0.99
    ]
    if eligible:
        return max(eligible, key=lambda row: float(row.get("accepted_coverage") or 0.0))
    return max(threshold_rows, key=lambda row: float(row.get("threshold") or 0.0))


def _selected_l2_threshold_value(summary_path: Path) -> float | None:
    if not summary_path.exists():
        return None
    metrics = _selected_l2_summary_threshold(_load_json(summary_path))
    if metrics is None or metrics.get("threshold") is None:
        return None
    return float(metrics["threshold"])


def _l2_threshold_point_annotation(
    threshold: float,
    *,
    selected_threshold: float | None,
) -> tuple[str, str]:
    selected = (
        selected_threshold is not None and abs(float(threshold) - float(selected_threshold)) < 1e-9
    )
    if threshold == CLINC150_STANDARD_L2_THRESHOLDS[0]:
        label = "strict selected" if selected else "strict"
        return "strict_endpoint", f"{label}\n{threshold:g}"
    if threshold == CLINC150_STANDARD_L2_THRESHOLDS[-1]:
        label = "loose selected" if selected else "loose"
        return "loose_endpoint", f"{label}\n{threshold:g}"
    if selected:
        return "nominal", f"selected\n{threshold:g}"
    return "", ""


def _l2_policy_label(metrics: dict[str, Any]) -> str:
    threshold = metrics.get("threshold")
    if threshold is None:
        return "summary guard"
    return f"guard_probability >= {float(threshold):g}"


def _curve_id(
    experiment_id: str,
    layer: str,
    candidate_id: str,
    split: str,
    knob_slug: str,
) -> str:
    return "-".join(
        _slug(part)
        for part in (experiment_id, layer, candidate_id, split, knob_slug)
        if part
    )


def _curve_title(*, layer: str, split: str, knob_name: str) -> str:
    return f"{layer} {_title_label(split)} {_title_label(knob_name)}"


def _title_label(value: str) -> str:
    return str(value).replace("_", " ").replace("-", " ").title()


def _slug(value: Any) -> str:
    text = str(value).strip().lower().replace("_", "-").replace(" ", "-")
    slug = "".join(char if char.isalnum() or char == "-" else "-" for char in text)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-") or "value"


def _round_payload(summary: dict[str, Any], round_number: int) -> dict[str, Any] | None:
    for payload in summary.get("rounds", []):
        if int(payload.get("round") or 0) == round_number:
            return payload
    return None


def _round_candidate_id(round_number: int) -> str:
    return f"round-{round_number:03d}"


def _source_repo_dir(summary: dict[str, Any]) -> Path | None:
    value = summary.get("source_repo_dir")
    if not value:
        return None
    return Path(str(value))


def _resolve_source_artifact(
    value: str | Path | None,
    *,
    source_repo_dir: Path | None,
    fallback_path: Path,
) -> Path:
    if value is None:
        return fallback_path.resolve()
    path = Path(value)
    if path.is_absolute():
        return path
    if source_repo_dir is not None:
        candidate = source_repo_dir / path
        if candidate.exists():
            return candidate.resolve()
        return candidate
    candidate = fallback_path.parent / path
    if candidate.exists():
        return candidate.resolve()
    return path


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _metric_float(metrics: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = metrics.get(key)
        if value is not None:
            return float(value)
    return None


def _metric_int(metrics: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = metrics.get(key)
        if value is not None:
            return int(value)
    return None


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _view_for_split(split: str) -> str:
    if "zipf" in split:
        return "zipf_heavy"
    if "uniform" in split:
        return "uniform"
    if "oos_heavy" in split:
        return "oos_heavy"
    if "intent_conflict" in split:
        return "intent_conflict"
    return "sequential"
