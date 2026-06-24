import json
from pathlib import Path

import pytest

from darjeeling.eval.plots import (
    annotate_pareto_frontier,
    plot_evolution_curve,
    plot_operating_frontier,
    read_normalized_jsonl,
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


def test_l1_overlay_filters_recorded_accepts_by_visible_rule_support(tmp_path: Path) -> None:
    train_predictions = tmp_path / "train_dev.jsonl"
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
            }
        ),
        encoding="utf-8",
    )

    rows = clinc150_l1_operating_point_rows(summary_path)

    raw = _find_policy(rows, split="train_dev", policy_label="raw accepts")
    strict = _find_policy(rows, split="train_dev", policy_label="positive>=2, negative=0")
    assert raw["accepted"] == 4
    assert raw["wrong_accepts"] == 1
    assert raw["accepted_precision"] == pytest.approx(0.75)
    assert strict["accepted"] == 3
    assert strict["wrong_accepts"] == 0
    assert strict["accepted_precision"] == pytest.approx(1.0)
    assert strict["coverage"] == pytest.approx(3 / 5)
    assert strict["metadata"]["overlay_semantics"] == (
        "target_adapter_posthoc_filter_over_recorded_l1_accepts"
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

    loose = _find_policy(rows, split="validation", policy_label="guard_probability >= 0.5")
    strict = _find_policy(rows, split="validation", policy_label="guard_probability >= 0.98")
    assert loose["accepted"] == 3
    assert loose["accepted_precision"] == pytest.approx(2 / 3)
    assert strict["accepted"] == 1
    assert strict["accepted_precision"] == pytest.approx(1.0)
    assert strict["coverage"] == pytest.approx(1 / 3)


def test_plot_smoke_writes_non_empty_pngs(tmp_path: Path) -> None:
    evolution_rows = [
        _point("r1", coverage=0.20, precision=0.99, round_number=1),
        _point("r2", coverage=0.35, precision=0.995, round_number=2),
    ]
    operating_rows = annotate_pareto_frontier(
        [
            _point("raw", coverage=0.35, precision=0.98),
            _point("safe", coverage=0.25, precision=1.0),
        ]
    )
    evolution_path = tmp_path / "evolution.png"
    frontier_path = tmp_path / "frontier.png"

    plot_evolution_curve(evolution_rows, evolution_path, title="Tiny Evolution")
    plot_operating_frontier(
        operating_rows,
        [row for row in operating_rows if row["pareto"]],
        frontier_path,
        title="Tiny Frontier",
    )

    assert evolution_path.stat().st_size > 0
    assert frontier_path.stat().st_size > 0


def _point(
    policy_label: str,
    *,
    coverage: float,
    precision: float | None,
    round_number: int = 1,
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
