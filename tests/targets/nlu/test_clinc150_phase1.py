import pytest

from darjeeling.targets.nlu.clinc150_phase1 import (
    build_clinc150_gate_records,
    build_clinc150_label_cards,
    clinc150_metrics_from_teacher_rows,
    compare_repeated_teacher_rows,
    evaluate_clinc150_l2,
    select_l2_threshold,
    train_clinc150_l2,
    training_examples_from_gold_records,
)
from darjeeling.targets.nlu.data import DataRecord
from darjeeling.targets.nlu.schemas import Frame


def test_clinc150_gate_sample_is_intent_stratified_with_oos_tail() -> None:
    records = [
        _record("r1", "alpha one", "alpha", split="validation"),
        _record("r2", "alpha two", "alpha", split="validation"),
        _record("r3", "beta one", "beta", split="validation"),
        _record("r4", "beta two", "beta", split="validation"),
        _record("r5", "oos one", "out_of_scope", split="validation", abstain=True),
        _record("r6", "oos two", "out_of_scope", split="validation", abstain=True),
    ]

    sample = build_clinc150_gate_records(records, per_intent=1, oos_requests=1)

    assert [record.request_id for record in sample] == ["r1", "r3", "r5"]


def test_clinc150_label_cards_use_train_examples_only() -> None:
    records = [
        _record("train-1", "alpha train", "alpha"),
        _record("train-2", "alpha second", "alpha"),
        _record("train-3", "alpha third", "alpha"),
        _record("train-4", "not supported", "out_of_scope", abstain=True),
    ]

    cards = build_clinc150_label_cards(records, examples_per_label=2)

    assert cards == [
        {
            "intent": "alpha",
            "description": "alpha",
            "examples": ["alpha train", "alpha second"],
        },
        {
            "intent": "out_of_scope",
            "description": "unsupported or out-of-scope request",
            "examples": ["not supported"],
        },
    ]


def test_clinc150_teacher_metrics_split_in_scope_oos_and_gate() -> None:
    rows = [
        _teacher_row("r1", "alpha", "alpha"),
        _teacher_row("r2", "beta", "alpha"),
        _teacher_row("r3", "out_of_scope", "out_of_scope", abstain=True),
        _teacher_row("r4", "out_of_scope", "alpha", abstain=True),
    ]

    metrics = clinc150_metrics_from_teacher_rows(
        rows,
        min_overall_accuracy=0.5,
        min_in_scope_accuracy=0.5,
        max_parse_failure_rate=0.0,
    )

    assert metrics["overall_accuracy"] == pytest.approx(0.5)
    assert metrics["in_scope_accuracy"] == pytest.approx(0.5)
    assert metrics["oos_precision"] == pytest.approx(1.0)
    assert metrics["oos_recall"] == pytest.approx(0.5)
    assert metrics["passed_teacher_gate"] is True


def test_clinc150_repeat_consistency_compares_parsed_teacher_frames() -> None:
    first = [
        _teacher_row("r1", "alpha", "alpha"),
        _teacher_row("r2", "beta", "beta"),
    ]
    second = [
        _teacher_row("r1", "alpha", "alpha"),
        _teacher_row("r2", "beta", "alpha"),
    ]

    result = compare_repeated_teacher_rows(first, second)

    assert result["comparable_requests"] == 2
    assert result["consistent_requests"] == 1
    assert result["consistency"] == pytest.approx(0.5)


def test_clinc150_l2_eval_selects_high_precision_threshold() -> None:
    train_records = [
        _record("t1", "alpha train one", "alpha"),
        _record("t2", "alpha train two", "alpha"),
        _record("t3", "beta train one", "beta"),
        _record("t4", "beta train two", "beta"),
        _record("t5", "unsupported thing", "out_of_scope", abstain=True),
        _record("t6", "not in supported intents", "out_of_scope", abstain=True),
    ]
    eval_records = [
        _record("e1", "alpha train one", "alpha", split="validation"),
        _record("e2", "beta train two", "beta", split="validation"),
        _record(
            "e3",
            "not in supported intents",
            "out_of_scope",
            split="validation",
            abstain=True,
        ),
    ]
    bundle = train_clinc150_l2(
        training_examples_from_gold_records(train_records),
        accept_threshold=0.0,
    )

    result = evaluate_clinc150_l2(bundle=bundle, records=eval_records)

    assert result["requests"] == 3
    assert result["accuracy"] is not None
    selected = select_l2_threshold(
        [
            {
                "threshold": 0.5,
                "accepted_precision": 0.5,
                "accepted_coverage": 1.0,
                "lower_layer_oos_false_accept_rate": 0.0,
            },
            {
                "threshold": 0.9,
                "accepted_precision": 1.0,
                "accepted_coverage": 0.5,
                "lower_layer_oos_false_accept_rate": 0.0,
            },
        ]
    )
    assert selected is not None
    assert selected["threshold"] == 0.9


def _record(
    request_id: str,
    utterance: str,
    intent: str,
    *,
    split: str = "train",
    abstain: bool = False,
) -> DataRecord:
    return DataRecord(
        request_id=request_id,
        utterance=utterance,
        split=split,
        gold_frame=Frame(intent=intent, slots={}, is_abstain=abstain),
    )


def _teacher_row(
    request_id: str,
    gold_intent: str,
    teacher_intent: str,
    *,
    abstain: bool = False,
) -> dict:
    gold_frame = Frame(intent=gold_intent, slots={}, is_abstain=abstain).model_dump(mode="json")
    teacher_frame = Frame(
        intent=teacher_intent,
        slots={},
        is_abstain=teacher_intent == "out_of_scope",
    ).model_dump(mode="json")
    return {
        "request_id": request_id,
        "utterance": request_id,
        "gold_frame": gold_frame,
        "teacher_frame": teacher_frame,
        "parse_failure": False,
        "frame_exact": gold_frame == teacher_frame,
        "intent_correct": gold_intent == teacher_intent,
    }
