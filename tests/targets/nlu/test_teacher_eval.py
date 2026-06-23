import csv
import json
from pathlib import Path

import pytest

from darjeeling.targets.nlu.data import DataRecord
from darjeeling.targets.nlu.layers.l4_cloud_llm import TeacherCallResult, TeacherParseError
from darjeeling.targets.nlu.schemas import Frame, TaskSchema
from darjeeling.targets.nlu.settings import load_settings
from darjeeling.targets.nlu.teacher_eval import (
    evaluate_live_teacher_vs_gold,
    run_teacher_prompt_comparison,
    write_teacher_live_eval_artifacts,
)


class SequenceTeacher:
    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.utterances: list[str] = []

    def answer(self, utterance: str, task_schema: TaskSchema):
        del task_schema
        self.utterances.append(utterance)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_teacher_live_vs_gold_aggregates_quality_failures_and_costs() -> None:
    records = [
        DataRecord(
            request_id="r1",
            utterance="alpha with value",
            gold_frame=Frame(intent="intent_alpha", slots={"slot_alpha": "value"}),
        ),
        DataRecord(
            request_id="r2",
            utterance="beta no slot",
            gold_frame=Frame(intent="intent_beta"),
        ),
        DataRecord(
            request_id="r3",
            utterance="broken response",
            gold_frame=Frame(intent="intent_beta"),
        ),
    ]
    teacher = SequenceTeacher(
        [
            _call(Frame(intent="intent_alpha", slots={"slot_alpha": "value"}), tokens=10),
            _call(Frame(intent="intent_alpha", slots={"unknown_slot": "x"}), tokens=14),
            TeacherParseError("teacher returned invalid JSON"),
        ]
    )
    settings = load_settings()

    result = evaluate_live_teacher_vs_gold(
        records=records,
        task_schema=TaskSchema(
            intent_names=["intent_alpha", "intent_beta"],
            slot_names=["slot_alpha"],
        ),
        settings=settings,
        split="validation",
        stream="sequential",
        prompt_version="teacher-v1",
        min_frame_exact_match=0.5,
        teacher=teacher,
    )

    summary = result.summary
    assert summary["requests"] == 3
    assert summary["parsed_requests"] == 2
    assert summary["parse_failure_count"] == 1
    assert summary["invalid_slot_count"] == 1
    assert summary["frame_exact_match"] == pytest.approx(1 / 3)
    assert summary["intent_accuracy"] == pytest.approx(1 / 3)
    assert summary["slot_pair_precision"] == pytest.approx(0.5)
    assert summary["slot_pair_recall"] == pytest.approx(1.0)
    assert summary["full_l4"]["calls"] == 3
    assert summary["full_l4"]["tokens"] == 24
    assert summary["passed"] is False
    assert teacher.utterances == ["alpha with value", "beta no slot", "broken response"]


def test_teacher_live_vs_gold_writes_summary_and_detail_artifacts(tmp_path: Path) -> None:
    result = evaluate_live_teacher_vs_gold(
        records=[
            DataRecord(
                request_id="r1",
                utterance="alpha",
                gold_frame=Frame(intent="intent_alpha"),
            )
        ],
        task_schema=TaskSchema(intent_names=["intent_alpha"], slot_names=[]),
        settings=load_settings(),
        split="validation",
        stream="sequential",
        prompt_version="teacher-v1",
        teacher=SequenceTeacher([_call(Frame(intent="intent_alpha"), tokens=8)]),
    )

    artifacts = write_teacher_live_eval_artifacts(result, out_dir=tmp_path)

    summary = json.loads(artifacts.summary_json_path.read_text(encoding="utf-8"))
    ledger = json.loads(artifacts.cost_ledger_path.read_text(encoding="utf-8"))
    detail_rows = list(csv.DictReader(artifacts.details_csv_path.open(encoding="utf-8")))
    details_jsonl = artifacts.details_jsonl_path.read_text(encoding="utf-8")
    assert summary["benchmark"] == "teacher-live-vs-gold"
    assert summary["frame_exact_match"] == 1.0
    assert ledger["observed_attempt_cost_usd"] == summary["full_l4"]["cost_usd"]
    assert ledger["request_costs"][0]["request_id"] == "r1"
    assert detail_rows[0]["request_id"] == "r1"
    assert '"request_id": "r1"' in details_jsonl


def test_teacher_live_vs_gold_records_teacher_call_failure() -> None:
    result = evaluate_live_teacher_vs_gold(
        records=[
            DataRecord(
                request_id="r1",
                utterance="alpha",
                gold_frame=Frame(intent="intent_alpha"),
            )
        ],
        task_schema=TaskSchema(intent_names=["intent_alpha"], slot_names=[]),
        settings=load_settings(),
        split="validation",
        stream="sequential",
        prompt_version="teacher-v1",
        teacher=SequenceTeacher([RuntimeError("request timed out")]),
    )

    assert result.summary["requests"] == 1
    assert result.summary["parse_failure_count"] == 1
    assert result.summary["full_l4"]["cost_usd"] == 0.0
    assert result.rows[0]["error"] == "request timed out"


def test_teacher_live_vs_gold_counts_retry_attempt_diagnostics_and_cost() -> None:
    attempts = [
        {
            "attempt": 1,
            "latency_ms": 1.0,
            "success": False,
            "error_type": "TeacherParseError",
            "error_message": "teacher response content is empty",
            "response_model": "fake-teacher",
            "usage": {"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1},
            "finish_reason": "length",
            "visible_content_length": 0,
        },
        {
            "attempt": 2,
            "latency_ms": 1.0,
            "success": True,
            "error_type": "",
            "error_message": "",
            "response_model": "fake-teacher",
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
            "finish_reason": "stop",
            "visible_content_length": 24,
        },
    ]

    result = evaluate_live_teacher_vs_gold(
        records=[
            DataRecord(
                request_id="r1",
                utterance="alpha",
                gold_frame=Frame(intent="intent_alpha"),
            )
        ],
        task_schema=TaskSchema(intent_names=["intent_alpha"], slot_names=[]),
        settings=load_settings(),
        split="validation",
        stream="sequential",
        prompt_version="teacher-v1",
        teacher=SequenceTeacher(
            [_call(Frame(intent="intent_alpha"), tokens=18, attempt_diagnostics=attempts)]
        ),
    )

    row = result.rows[0]
    summary = result.summary
    assert row["attempt_count"] == 2
    assert row["empty_response_attempts"] == 1
    assert row["retry_recovered"] is True
    assert row["tokens"] == 19
    assert row["final_response_tokens"] == 18
    assert row["cost_usd"] == pytest.approx((11 * 0.40 + 8 * 1.60) / 1_000_000)
    assert row["final_response_cost_usd"] == pytest.approx((9 * 0.40 + 9 * 1.60) / 1_000_000)
    assert summary["attempt_count"] == 2
    assert summary["retry_recovered_rows"] == 1
    assert summary["empty_response_attempts"] == 1
    assert summary["full_l4"]["api_attempts"] == 2
    assert summary["full_l4"]["tokens"] == 19
    assert summary["full_l4"]["final_response_tokens"] == 18


def test_teacher_live_vs_gold_counts_final_empty_failure_usage() -> None:
    error = TeacherParseError("teacher response content is empty")
    error.attempt_diagnostics = [
        {
            "attempt": 1,
            "latency_ms": 1.0,
            "success": False,
            "error_type": "TeacherParseError",
            "error_message": "teacher response content is empty",
            "response_model": "fake-teacher",
            "usage": {"prompt_tokens": 5, "completion_tokens": 0, "total_tokens": 5},
            "finish_reason": "length",
            "visible_content_length": 0,
        }
    ]

    result = evaluate_live_teacher_vs_gold(
        records=[
            DataRecord(
                request_id="r1",
                utterance="alpha",
                gold_frame=Frame(intent="intent_alpha"),
            )
        ],
        task_schema=TaskSchema(intent_names=["intent_alpha"], slot_names=[]),
        settings=load_settings(),
        split="validation",
        stream="sequential",
        prompt_version="teacher-v1",
        teacher=SequenceTeacher([error]),
    )

    row = result.rows[0]
    assert row["parse_failure"] is True
    assert row["final_empty_response_failure"] is True
    assert row["cost_usd"] == pytest.approx((5 * 0.40) / 1_000_000)
    assert row["unknown_usage_attempts"] == 0
    assert result.summary["final_empty_response_failures"] == 1
    assert result.summary["full_l4"]["cost_usd"] == pytest.approx((5 * 0.40) / 1_000_000)


def test_teacher_prompt_comparison_uses_same_sample_for_each_prompt(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    records = [
        DataRecord(
            request_id="r1",
            utterance="alpha",
            gold_frame=Frame(intent="intent_alpha"),
        ),
        DataRecord(
            request_id="r2",
            utterance="beta",
            gold_frame=Frame(intent="intent_beta"),
        ),
    ]
    (data_dir / "validation.jsonl").write_text(
        "".join(record.model_dump_json() + "\n" for record in records),
        encoding="utf-8",
    )

    def teacher_factory(settings):
        if settings.teacher_prompt_version == "teacher-v1":
            return SequenceTeacher(
                [
                    _call(Frame(intent="intent_alpha"), tokens=5),
                    _call(Frame(intent="intent_beta"), tokens=5),
                ]
            )
        return SequenceTeacher(
            [
                _call(Frame(intent="intent_beta"), tokens=7),
                _call(Frame(intent="intent_beta"), tokens=7),
            ]
        )

    comparison = run_teacher_prompt_comparison(
        data_dir=data_dir,
        split="validation",
        stream="sequential",
        max_requests=2,
        prompt_versions=["teacher-v1", "teacher-v2-intent-first"],
        settings=load_settings(),
        out_dir=tmp_path / "out",
        teacher_factory=teacher_factory,
    )

    payload = json.loads(comparison.comparison_json_path.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(comparison.comparison_csv_path.open(encoding="utf-8")))
    assert payload["same_sample"] is True
    assert payload["sample_request_ids"] == ["r1", "r2"]
    assert [row["prompt_version"] for row in rows] == [
        "teacher-v1",
        "teacher-v2-intent-first",
    ]
    assert payload["rows"][0]["frame_exact_match"] == 1.0
    assert payload["rows"][1]["frame_exact_match"] == 0.5


def _call(
    frame: Frame,
    *,
    tokens: int,
    attempt_diagnostics: list[dict] | None = None,
) -> TeacherCallResult:
    return TeacherCallResult(
        frame=frame,
        raw_response=frame.model_dump_json(),
        usage={
            "prompt_tokens": tokens // 2,
            "completion_tokens": tokens - tokens // 2,
            "total_tokens": tokens,
        },
        model="fake-teacher",
        context_hash=f"ctx-{tokens}",
        prompt_cache_key=f"cache-{tokens}",
        attempt_diagnostics=attempt_diagnostics or [],
    )
