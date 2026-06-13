from __future__ import annotations

from collections import Counter
from typing import Any

from darjeeling.targets.nlu.schemas import Frame, TeacherTrace

FOCUS_TASK_SCHEMA_VERSION = "nlu-focus-tasks-v1"
WEAK_LAYERS = {"L0", "L1", "L2", "L3"}


def build_focus_tasks(
    traces: list[TeacherTrace],
    *,
    max_tasks: int = 8,
    examples_per_task: int = 5,
    precision_floor: float = 0.98,
) -> list[dict[str, Any]]:
    labeled = [trace for trace in traces if trace.teacher_frame is not None]
    groups: dict[tuple[str, tuple[str, ...]], list[TeacherTrace]] = {}
    for trace in labeled:
        assert trace.teacher_frame is not None
        key = _task_key(trace.teacher_frame)
        groups.setdefault(key, []).append(trace)
    ranked = sorted(
        groups.items(),
        key=lambda item: _focus_score(item[1]),
        reverse=True,
    )
    return [
        _focus_task_payload(
            key,
            group_traces,
            all_traces=labeled,
            rank=rank,
            examples_per_task=examples_per_task,
            precision_floor=precision_floor,
        )
        for rank, (key, group_traces) in enumerate(ranked[:max_tasks], start=1)
    ]


def focus_task_document(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    source_trace_ids = []
    seen: set[str] = set()
    for task in tasks:
        for trace_id in task.get("source_trace_ids", []):
            if trace_id in seen:
                continue
            seen.add(trace_id)
            source_trace_ids.append(trace_id)
    return {
        "schema_version": FOCUS_TASK_SCHEMA_VERSION,
        "task_count": len(tasks),
        "source_trace_ids": source_trace_ids,
        "tasks": tasks,
    }


def _focus_task_payload(
    key: tuple[str, tuple[str, ...]],
    traces: list[TeacherTrace],
    *,
    all_traces: list[TeacherTrace],
    rank: int,
    examples_per_task: int,
    precision_floor: float,
) -> dict[str, Any]:
    intent, slot_signature = key
    failures = [_trace_example(trace) for trace in traces if _is_focus_failure(trace)]
    positives = [_trace_example(trace) for trace in traces[:examples_per_task]]
    near_negatives = [
        _trace_example(trace)
        for trace in all_traces
        if trace.teacher_frame is not None and _task_key(trace.teacher_frame) != key
    ][:examples_per_task]
    lower_misses = [trace for trace in traces if _lower_layers_all_abstained(trace)]
    wrong_accepts = [trace for trace in traces if _weak_wrong_accept(trace)]
    audit_disagreements = [
        trace for trace in traces if trace.metadata.get("teacher_disagreed") is True
    ]
    chosen_counts = Counter(trace.chosen_layer for trace in traces)
    goal_parts = [f"improve {intent}"]
    if slot_signature:
        goal_parts.append("with slots " + ", ".join(slot_signature))
    else:
        goal_parts.append("intent-only coverage")
    return {
        "schema_version": "nlu-focus-task-v1",
        "task_id": f"focus-{rank:03d}-{intent}-{'-'.join(slot_signature) or 'intent'}",
        "rank": rank,
        "score": _focus_score(traces),
        "goal": " ".join(goal_parts),
        "teacher_intent": intent,
        "slot_signature": list(slot_signature),
        "precision_floor": precision_floor,
        "positive_examples": positives,
        "near_negative_examples": near_negatives,
        "current_failures": failures[:examples_per_task],
        "current_layer_behavior": {
            "trace_count": len(traces),
            "chosen_layer_counts": dict(sorted(chosen_counts.items())),
            "lower_miss_count": len(lower_misses),
            "weak_wrong_accept_count": len(wrong_accepts),
            "teacher_audit_disagreement_count": len(audit_disagreements),
        },
        "source_trace_ids": [trace.request_id for trace in traces],
    }


def _trace_example(trace: TeacherTrace) -> dict[str, Any]:
    return {
        "request_id": trace.request_id,
        "utterance": trace.utterance,
        "teacher_frame": trace.teacher_frame.model_dump(mode="json")
        if trace.teacher_frame is not None
        else None,
        "chosen_layer": trace.chosen_layer,
        "final_frame": trace.final_frame.model_dump(mode="json"),
        "layer_behavior": [
            {
                "layer": result.layer,
                "accepted": result.accepted,
                "reason": result.reason,
                "patch_accepted_fields": result.metadata.get("patch_accepted_fields", []),
            }
            for result in trace.layer_results
        ],
    }


def _task_key(frame: Frame) -> tuple[str, tuple[str, ...]]:
    return frame.intent, tuple(sorted(frame.slots))


def _focus_score(traces: list[TeacherTrace]) -> float:
    hard = sum(_is_focus_failure(trace) for trace in traces)
    lower_miss = sum(_lower_layers_all_abstained(trace) for trace in traces)
    wrong_accept = sum(_weak_wrong_accept(trace) for trace in traces)
    audit_disagreement = sum(
        trace.metadata.get("teacher_disagreed") is True for trace in traces
    )
    l4_calls = sum(trace.chosen_layer == "L4" for trace in traces)
    return float(
        (audit_disagreement * 25)
        + (wrong_accept * 20)
        + (hard * 10)
        + (lower_miss * 5)
        + (l4_calls * 3)
        + len(traces)
    )


def _is_focus_failure(trace: TeacherTrace) -> bool:
    return (
        trace.teacher_frame is not None
        and (
            trace.final_frame != trace.teacher_frame
            or trace.chosen_layer == "L4"
            or trace.metadata.get("teacher_disagreed") is True
        )
    )


def _lower_layers_all_abstained(trace: TeacherTrace) -> bool:
    lower_results = [result for result in trace.layer_results if result.layer in WEAK_LAYERS]
    return bool(lower_results) and all(not result.accepted for result in lower_results)


def _weak_wrong_accept(trace: TeacherTrace) -> bool:
    return (
        trace.teacher_frame is not None
        and trace.chosen_layer in WEAK_LAYERS
        and trace.final_frame != trace.teacher_frame
    )
