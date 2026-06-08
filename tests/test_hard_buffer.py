from pathlib import Path

from darjeeling.compiler.mining import (
    build_hard_buffer,
    hard_case_reason_counts,
    hard_case_traces,
    hard_case_visibility_counts,
    load_hard_buffer_jsonl,
    merge_hard_buffers,
    write_hard_buffer_jsonl,
)
from darjeeling.schemas import Frame, LayerResult, TraceRecord, traces_to_teacher_view


def test_hard_buffer_prioritizes_wrong_accepts_without_gold_leakage(tmp_path: Path) -> None:
    wrong_accept = TraceRecord(
        request_id="wrong",
        utterance="set alarm for seven",
        gold_frame=Frame(intent="alarm_set", slots={"time": "gold-seven"}),
        teacher_frame=Frame(intent="alarm_set", slots={"time": "seven"}),
        chosen_layer="L2",
        final_frame=Frame(intent="music_play"),
        layer_results=[
            LayerResult(
                layer="L2",
                accepted=True,
                frame=Frame(intent="music_play"),
                latency_ms=12.0,
            )
        ],
    )
    fallback = TraceRecord(
        request_id="fallback",
        utterance="weather tomorrow",
        gold_frame=Frame(intent="weather_query", slots={"date": "gold-tomorrow"}),
        teacher_frame=Frame(intent="weather_query", slots={"date": "tomorrow"}),
        chosen_layer="L4",
        final_frame=Frame(intent="weather_query", slots={"date": "tomorrow"}),
        layer_results=[
            LayerResult(layer="L1", accepted=False, latency_ms=3.0),
            LayerResult(
                layer="L4",
                accepted=True,
                frame=Frame(intent="weather_query", slots={"date": "tomorrow"}),
                latency_ms=900.0,
            ),
        ],
    )
    easy = TraceRecord(
        request_id="easy",
        utterance="play jazz",
        gold_frame=Frame(intent="music_play"),
        teacher_frame=Frame(intent="music_play"),
        chosen_layer="L2",
        final_frame=Frame(intent="music_play"),
        layer_results=[
            LayerResult(
                layer="L2",
                accepted=True,
                frame=Frame(intent="music_play"),
                latency_ms=5.0,
            )
        ],
    )

    hard_cases = build_hard_buffer(traces_to_teacher_view([fallback, easy, wrong_accept]))

    assert [case.request_id for case in hard_cases] == ["wrong", "fallback"]
    assert hard_cases[0].reason == "weak_wrong_accept"
    assert hard_cases[1].reason == "fallback_after_weak_abstain"
    assert hard_case_reason_counts(hard_cases) == {
        "fallback_after_weak_abstain": 1,
        "weak_wrong_accept": 1,
    }
    assert [trace.request_id for trace in hard_case_traces(hard_cases)] == [
        "wrong",
        "fallback",
    ]

    hard_buffer_path = write_hard_buffer_jsonl(tmp_path / "hard_buffer.jsonl", hard_cases)
    payload = hard_buffer_path.read_text(encoding="utf-8")
    assert "gold_frame" not in payload
    assert "gold-seven" not in payload
    assert "gold-tomorrow" not in payload
    assert "weak_wrong_accept" in payload

    loaded = load_hard_buffer_jsonl(hard_buffer_path)
    assert loaded[0].visibility == "train_visible"
    assert [case.request_id for case in loaded] == ["wrong", "fallback"]


def test_hard_buffer_merge_dedupes_by_highest_severity() -> None:
    old_case = build_hard_buffer(
        traces_to_teacher_view(
            [
                TraceRecord(
                    request_id="same",
                    utterance="play music",
                    teacher_frame=Frame(intent="music_play"),
                    chosen_layer="L4",
                    final_frame=Frame(intent="music_play"),
                    layer_results=[
                        LayerResult(layer="L4", accepted=True, latency_ms=1.0),
                    ],
                )
            ]
        )
    )
    new_case = build_hard_buffer(
        traces_to_teacher_view(
            [
                TraceRecord(
                    request_id="same",
                    utterance="play music",
                    teacher_frame=Frame(intent="music_play"),
                    chosen_layer="L2",
                    final_frame=Frame(intent="alarm_set"),
                    layer_results=[
                        LayerResult(
                            layer="L2",
                            accepted=True,
                            frame=Frame(intent="alarm_set"),
                            latency_ms=5.0,
                        ),
                    ],
                )
            ]
        )
    )

    merged = merge_hard_buffers([old_case, new_case], max_cases=10)

    assert len(merged) == 1
    assert merged[0].request_id == "same"
    assert merged[0].reason == "weak_wrong_accept"


def test_hard_buffer_visibility_filters_replay_only_cases() -> None:
    train_case = build_hard_buffer(
        traces_to_teacher_view(
            [
                TraceRecord(
                    request_id="train",
                    utterance="play music",
                    teacher_frame=Frame(intent="music_play"),
                    chosen_layer="L4",
                    final_frame=Frame(intent="music_play"),
                    layer_results=[LayerResult(layer="L4", accepted=True, latency_ms=900.0)],
                )
            ]
        ),
        visibility="train_visible",
    )
    replay_only_case = build_hard_buffer(
        traces_to_teacher_view(
            [
                TraceRecord(
                    request_id="holdout",
                    utterance="set alarm",
                    teacher_frame=Frame(intent="alarm_set"),
                    chosen_layer="L4",
                    final_frame=Frame(intent="alarm_set"),
                    layer_results=[LayerResult(layer="L4", accepted=True, latency_ms=900.0)],
                )
            ]
        ),
        visibility="replay_only",
    )
    hard_cases = merge_hard_buffers([train_case, replay_only_case], max_cases=10)

    assert hard_case_visibility_counts(hard_cases) == {
        "replay_only": 1,
        "train_visible": 1,
    }
    assert [trace.request_id for trace in hard_case_traces(hard_cases)] == [
        "train",
        "holdout",
    ]
    assert [
        trace.request_id for trace in hard_case_traces(hard_cases, visibility={"train_visible"})
    ] == ["train"]
