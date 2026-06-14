from __future__ import annotations

from collections import Counter
from typing import Any

from darjeeling.targets.nlu.patches import (
    accepted_field_keys,
    frame_field_values,
    frame_patch_from_layer_result,
)
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
        "field_opportunities": [],
        "tasks": tasks,
    }


def focus_task_document_with_fields(
    tasks: list[dict[str, Any]],
    traces: list[TeacherTrace],
) -> dict[str, Any]:
    document = focus_task_document(tasks)
    document["field_opportunities"] = build_field_opportunities(traces)
    return document


def build_field_opportunities(
    traces: list[TeacherTrace],
    *,
    max_fields: int = 12,
    examples_per_field: int = 4,
) -> list[dict[str, Any]]:
    labeled = [trace for trace in traces if trace.teacher_frame is not None]
    stats: dict[str, dict[str, Any]] = {}
    for trace in labeled:
        assert trace.teacher_frame is not None
        expected_fields = frame_field_values(trace.teacher_frame)
        weak_fields = _weak_accepted_field_values(trace)
        l4_fields = _l4_accepted_fields(trace)
        l4_after_weak = bool(weak_fields) and bool(l4_fields)
        for field_key, expected_value in expected_fields.items():
            entry = stats.setdefault(field_key, _empty_field_opportunity(field_key))
            if trace.chosen_layer == "L4":
                entry["fallback_count"] += 1
                _append_field_example(entry, trace, examples_per_field)
            if field_key in weak_fields and weak_fields[field_key] != expected_value:
                entry["wrong_accepted_count"] += 1
                _append_field_example(entry, trace, examples_per_field)
            if l4_after_weak and field_key in l4_fields:
                entry["completed_by_l4_after_weak_count"] += 1
                _append_field_example(entry, trace, examples_per_field)
        for conflict in _l4_field_conflicts(trace):
            field_key = str(conflict.get("field", ""))
            if not field_key:
                continue
            entry = stats.setdefault(field_key, _empty_field_opportunity(field_key))
            entry["conflict_count"] += 1
            _append_field_example(entry, trace, examples_per_field)
    ranked = sorted(
        stats.values(),
        key=lambda entry: (
            entry["conflict_count"],
            entry["wrong_accepted_count"],
            entry["completed_by_l4_after_weak_count"],
            entry["fallback_count"],
        ),
        reverse=True,
    )
    return [entry for entry in ranked[:max_fields] if _field_opportunity_score(entry) > 0]


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


def _weak_accepted_field_values(trace: TeacherTrace) -> dict[str, str]:
    values: dict[str, str] = {}
    for result in trace.layer_results:
        if result.layer not in WEAK_LAYERS or not result.accepted:
            continue
        patch = frame_patch_from_layer_result(result)
        if patch is None:
            continue
        if patch.accepted_intent is not None:
            values["intent"] = patch.accepted_intent
        values.update(
            {
                f"slots.{slot_key}": slot_value
                for slot_key, slot_value in patch.accepted_slots.items()
            }
        )
    return values


def _l4_accepted_fields(trace: TeacherTrace) -> set[str]:
    fields: set[str] = set()
    for result in trace.layer_results:
        if result.layer != "L4" or not result.accepted:
            continue
        fields.update(accepted_field_keys(frame_patch_from_layer_result(result)))
    return fields


def _l4_field_conflicts(trace: TeacherTrace) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    for result in trace.layer_results:
        if result.layer != "L4":
            continue
        raw_conflicts = result.metadata.get("field_conflicts", [])
        if isinstance(raw_conflicts, list):
            conflicts.extend(
                conflict for conflict in raw_conflicts if isinstance(conflict, dict)
            )
    raw_trace_conflicts = trace.metadata.get("field_conflicts", [])
    if isinstance(raw_trace_conflicts, list):
        conflicts.extend(
            conflict for conflict in raw_trace_conflicts if isinstance(conflict, dict)
        )
    return conflicts


def _empty_field_opportunity(field_key: str) -> dict[str, Any]:
    return {
        "field": field_key,
        "fallback_count": 0,
        "conflict_count": 0,
        "wrong_accepted_count": 0,
        "completed_by_l4_after_weak_count": 0,
        "examples": [],
    }


def _append_field_example(
    entry: dict[str, Any],
    trace: TeacherTrace,
    examples_per_field: int,
) -> None:
    examples = entry["examples"]
    if any(example["request_id"] == trace.request_id for example in examples):
        return
    if len(examples) >= examples_per_field:
        return
    examples.append(_trace_example(trace))


def _field_opportunity_score(entry: dict[str, Any]) -> int:
    return int(
        entry["conflict_count"] * 4
        + entry["wrong_accepted_count"] * 3
        + entry["completed_by_l4_after_weak_count"] * 2
        + entry["fallback_count"]
    )
