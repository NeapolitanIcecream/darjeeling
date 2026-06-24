import json
from pathlib import Path

import pytest

from darjeeling.eval.plots import (
    annotate_pareto_frontier,
    plot_evolution_curve,
    plot_operating_curve_facets,
    plot_single_operating_curve,
    read_normalized_jsonl,
    validate_operating_curve_rows,
    write_normalized_jsonl,
)
from darjeeling.targets.nlu.precision_coverage import (
    clinc150_l1_operating_point_rows,
    clinc150_l2_operating_point_rows,
)


def test_pareto_frontier_marks_non_dominated_points() -> None:
    rows = [
        _point("low", coverage=0.10, precision=0.90),
        _point("frontier-left", coverage=0.15, precision=0.91),
        _point("frontier-right", coverage=0.20, precision=0.86),
        _point("empty", coverage=0.0, precision=None),
    ]

    annotated = annotate_pareto_frontier(rows)

    by_policy = {row["policy_label"]: row["pareto"] for row in annotated}
    assert by_policy == {
        "low": False,
        "frontier-left": True,
        "frontier-right": True,
        "empty": False,
    }


def test_normalized_rows_round_trip_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "round_metrics.jsonl"
    rows = [_point("raw", coverage=0.25, precision=0.99)]

    write_normalized_jsonl(rows, path)

    assert read_normalized_jsonl(path) == rows


def test_standard_operating_curve_rows_require_one_curve_id() -> None:
    rows = [
        _point("strict", coverage=0.20, precision=1.0, knob_order=0),
        _point("raw", coverage=0.35, precision=0.98, knob_order=1),
    ]

    validated = validate_operating_curve_rows(rows)

    assert [row["policy_label"] for row in validated] == ["strict", "raw"]


def test_single_operating_curve_rejects_mixed_curve_ids(tmp_path: Path) -> None:
    rows = [
        _point("strict", coverage=0.20, precision=1.0, knob_order=0, curve_id="curve-a"),
        _point("raw", coverage=0.35, precision=0.98, knob_order=1, curve_id="curve-b"),
    ]

    with pytest.raises(ValueError, match="share one curve_id"):
        plot_single_operating_curve(rows, tmp_path / "mixed.png", title="Mixed")


def test_l1_overlay_filters_recorded_accepts_by_visible_rule_support(tmp_path: Path) -> None:
    train_predictions = tmp_path / "train_dev.jsonl"
    locked_predictions = tmp_path / "locked_test.jsonl"
    _write_jsonl(
        train_predictions,
        [
            _l1_row("r1", "alpha phrase", accepted=True, correct=True),
            _l1_row("r2", "alpha phrase", accepted=True, correct=True),
            _l1_row("r3", "alpha phrase", accepted=True, correct=True),
            _l1_row("r4", "weak phrase", accepted=True, correct=False),
            _l1_row("r5", "abstain", accepted=False, correct=False),
        ],
    )
    _write_jsonl(
        locked_predictions,
        [
            _l1_row("t1", "alpha phrase", accepted=True, correct=True),
            _l1_row("t2", "weak phrase", accepted=True, correct=False),
        ],
    )
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "rounds": [
                    {
                        "round": 1,
                        "candidate_eligible": True,
                        "evaluations": {
                            "train_dev": {
                                "details_jsonl_path": str(train_predictions),
                                "summary": {"l1_only": {}},
                            }
                        },
                    }
                ],
                "selected_round": {"round": 1},
                "locked_test": {
                    "details_jsonl_path": str(locked_predictions),
                    "summary": {"l1_only": {}},
                },
            }
        ),
        encoding="utf-8",
    )

    rows = clinc150_l1_operating_point_rows(summary_path)

    raw = _find_policy(rows, split="train_dev", policy_label="raw: all recorded accepts")
    loose = _find_policy(rows, split="train_dev", policy_label="loose: clean support >= 2")
    ordered = [row for row in rows if row["split"] == "train_dev"]
    assert raw["accepted"] == 4
    assert raw["wrong_accepts"] == 1
    assert raw["accepted_precision"] == pytest.approx(0.75)
    assert loose["accepted"] == 3
    assert loose["wrong_accepts"] == 0
    assert loose["accepted_precision"] == pytest.approx(1.0)
    assert loose["coverage"] == pytest.approx(3 / 5)
    assert [row["knob_order"] for row in ordered] == [0, 1, 2, 3, 4]
    assert [row["coverage"] for row in ordered] == sorted(row["coverage"] for row in ordered)
    assert loose["metadata"]["overlay_semantics"] == (
        "target_adapter_posthoc_filter_over_recorded_l1_accepts"
    )
    assert {row["curve_id"] for row in ordered} == {
        "clinc150-l1-agent-session-effect-l1-round-001-train-dev-risk-tolerance"
    }
    assert all(row["selection_scope"] == "agent_visible" for row in ordered)

    locked_rows = [row for row in rows if row["split"] == "locked_test"]
    assert locked_rows
    assert all(row["selection_scope"] == "locked_test_diagnostic" for row in locked_rows)
    assert {row["curve_id"] for row in locked_rows}.isdisjoint(
        {row["curve_id"] for row in ordered}
    )


def test_l2_threshold_sweep_uses_prediction_guard_probability(tmp_path: Path) -> None:
    prediction_dir = tmp_path / "distilled-l2" / "train-full" / "validation-cascade"
    prediction_path = prediction_dir / "clinc150_l2_predictions.jsonl"
    _write_jsonl(
        prediction_path,
        [
            _l2_row("r1", guard=0.99, predicted_intent="alpha", gold_intent="alpha"),
            _l2_row("r2", guard=0.97, predicted_intent="beta", gold_intent="alpha"),
            _l2_row("r3", guard=0.60, predicted_intent="gamma", gold_intent="gamma"),
        ],
    )

    rows = clinc150_l2_operating_point_rows(tmp_path)

    loose = _find_policy(rows, split="validation", policy_label="guard >= 0.5")
    strict = _find_policy(rows, split="validation", policy_label="guard >= 0.98")
    ordered = [row for row in rows if row["split"] == "validation"]
    assert loose["accepted"] == 3
    assert loose["accepted_precision"] == pytest.approx(2 / 3)
    assert strict["accepted"] == 1
    assert strict["accepted_precision"] == pytest.approx(1.0)
    assert strict["coverage"] == pytest.approx(1 / 3)
    assert [row["knob_value"] for row in ordered] == [
        0.995,
        0.99,
        0.985,
        0.98,
        0.95,
        0.9,
        0.8,
        0.7,
        0.6,
        0.5,
    ]
    assert {row["curve_id"] for row in ordered} == {
        "clinc150-l2-cascade-l2-teacher-full-validation-guard-threshold"
    }


def test_plot_smoke_writes_non_empty_pngs(tmp_path: Path) -> None:
    evolution_rows = [
        _point("r1", coverage=0.20, precision=0.99, round_number=1),
        _point("r2", coverage=0.35, precision=0.995, round_number=2),
    ]
    operating_rows = annotate_pareto_frontier(
        [
            _point("safe", coverage=0.25, precision=1.0, knob_order=0),
            _point("raw", coverage=0.35, precision=0.98, knob_order=1),
        ]
    )
    comparison_rows = [
        *operating_rows,
        _point(
            "threshold",
            coverage=0.50,
            precision=0.99,
            knob_order=0,
            curve_id="curve-b",
            knob_name="guard_threshold",
        ),
    ]
    evolution_path = tmp_path / "evolution.png"
    curve_path = tmp_path / "curve.png"
    comparison_path = tmp_path / "comparison.png"

    plot_evolution_curve(evolution_rows, evolution_path, title="Tiny Evolution")
    plot_single_operating_curve(
        operating_rows,
        curve_path,
        title="Tiny Curve",
    )
    plot_operating_curve_facets(
        comparison_rows,
        comparison_path,
        title="Tiny Facets",
    )

    assert evolution_path.stat().st_size > 0
    assert curve_path.stat().st_size > 0
    assert comparison_path.stat().st_size > 0


def _point(
    policy_label: str,
    *,
    coverage: float,
    precision: float | None,
    round_number: int = 1,
    knob_order: int = 0,
    curve_id: str = "curve-a",
    knob_name: str = "risk_tolerance",
) -> dict:
    return {
        "experiment_id": "exp",
        "layer": "L1",
        "candidate_id": "candidate",
        "round": round_number,
        "split": "validation",
        "view": "sequential",
        "accepted_precision": precision,
        "coverage": coverage,
        "accepted": 10,
        "wrong_accepts": 0,
        "policy_family": "test_policy",
        "policy_label": policy_label,
        "policy_value": knob_order,
        "curve_id": curve_id,
        "curve_role": "standard_operating_curve",
        "knob_name": knob_name,
        "knob_value": knob_order,
        "knob_label": policy_label,
        "knob_order": knob_order,
        "knob_direction": "strict_to_loose",
        "primary_curve": True,
        "point_role": "strict_endpoint" if knob_order == 0 else "loose_endpoint",
        "annotation_label": policy_label,
        "curve_title": "Tiny Curve",
        "source_artifact": "/tmp/source.jsonl",
        "selection_scope": "agent_visible",
    }


def _l1_row(
    request_id: str,
    reason: str,
    *,
    accepted: bool,
    correct: bool,
) -> dict:
    intent = "alpha" if reason == "alpha phrase" else "beta"
    return {
        "request_id": request_id,
        "utterance": request_id,
        "gold_frame": {"intent": intent, "slots": {}, "is_abstain": False},
        "gold_intent": intent,
        "gold_oos": False,
        "l1_accepted": accepted,
        "l1_frame": {"intent": intent, "slots": {}, "is_abstain": False} if accepted else None,
        "l1_intent": intent if accepted else None,
        "l1_oos": False,
        "l1_correct": correct,
        "l1_outcome": "correct_accept" if accepted and correct else "wrong_accept",
        "program_path": "phrase_rule" if accepted else "abstain",
        "native_latency_us": 1,
        "integration_latency_ms": 1.0,
        "reason": reason,
    }


def _l2_row(
    request_id: str,
    *,
    guard: float,
    predicted_intent: str,
    gold_intent: str,
) -> dict:
    return {
        "request_id": request_id,
        "utterance": request_id,
        "gold_frame": {"intent": gold_intent, "slots": {}, "is_abstain": False},
        "gold_intent": gold_intent,
        "gold_oos": False,
        "predicted_frame": {"intent": predicted_intent, "slots": {}, "is_abstain": False},
        "predicted_intent": predicted_intent,
        "predicted_oos": False,
        "guard_probability": guard,
        "top1_probability": guard,
        "margin": 0.1,
        "entropy": 1.0,
        "latency_ms": 1.0,
    }


def _find_policy(rows: list[dict], *, split: str, policy_label: str) -> dict:
    for row in rows:
        if row["split"] == split and row["policy_label"] == policy_label:
            return row
    raise AssertionError(f"missing policy row: {split} {policy_label}")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
