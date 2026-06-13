from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from darjeeling.targets.nlu.compiler.l4_context import assert_no_forbidden_context
from darjeeling.targets.nlu.schemas import LayerName, TeacherTrace

HardCaseVisibility = Literal["train_visible", "replay_only"]

HardCaseReason = Literal[
    "weak_wrong_accept",
    "teacher_audit_disagreement",
    "teacher_final_mismatch",
    "fallback_after_weak_abstain",
    "l4_fallback",
    "slow_path",
]

HARD_BUFFER_SCHEMA_VERSION = "hard-buffer-v1"
WEAK_LAYERS = {"L0", "L1", "L2", "L3"}
REASON_WEIGHTS: dict[HardCaseReason, float] = {
    "weak_wrong_accept": 100.0,
    "teacher_audit_disagreement": 95.0,
    "teacher_final_mismatch": 90.0,
    "fallback_after_weak_abstain": 70.0,
    "l4_fallback": 60.0,
    "slow_path": 20.0,
}


class HardCase(BaseModel):
    schema_version: str = HARD_BUFFER_SCHEMA_VERSION
    visibility: HardCaseVisibility = "train_visible"
    request_id: str
    reason: HardCaseReason
    reasons: list[HardCaseReason] = Field(default_factory=list)
    severity: float
    chosen_layer: LayerName
    total_latency_ms: float
    trace: TeacherTrace


def hot_intents(traces: list[TeacherTrace], limit: int = 10) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter(
        trace.teacher_frame.intent for trace in traces if trace.teacher_frame is not None
    )
    return counts.most_common(limit)


def build_hard_buffer(
    traces: list[TeacherTrace],
    *,
    max_cases: int = 100,
    slow_latency_ms: float = 750.0,
    visibility: HardCaseVisibility = "train_visible",
) -> list[HardCase]:
    """Mine teacher-visible traces that deserve extra compiler pressure."""

    by_request_id: dict[str, HardCase] = {}
    for trace in traces:
        hard_case = _hard_case_from_trace(
            trace,
            slow_latency_ms=slow_latency_ms,
            visibility=visibility,
        )
        if hard_case is None:
            continue
        existing = by_request_id.get(hard_case.request_id)
        if existing is None or _sort_key(hard_case) < _sort_key(existing):
            by_request_id[hard_case.request_id] = hard_case

    hard_cases = sorted(by_request_id.values(), key=_sort_key)
    return hard_cases[:max_cases]


def merge_hard_buffers(
    hard_buffers: list[list[HardCase]],
    *,
    max_cases: int = 100,
) -> list[HardCase]:
    by_request_id: dict[str, HardCase] = {}
    for hard_buffer in hard_buffers:
        for hard_case in hard_buffer:
            existing = by_request_id.get(hard_case.request_id)
            if existing is None or _sort_key(hard_case) < _sort_key(existing):
                by_request_id[hard_case.request_id] = hard_case
    return sorted(by_request_id.values(), key=_sort_key)[:max_cases]


def hard_case_traces(
    hard_cases: list[HardCase],
    *,
    visibility: set[HardCaseVisibility] | None = None,
) -> list[TeacherTrace]:
    seen: set[str] = set()
    traces: list[TeacherTrace] = []
    for hard_case in hard_cases:
        if visibility is not None and hard_case.visibility not in visibility:
            continue
        if hard_case.request_id in seen:
            continue
        seen.add(hard_case.request_id)
        traces.append(hard_case.trace)
    return traces


def hard_case_reason_counts(hard_cases: list[HardCase]) -> dict[str, int]:
    counts: Counter[str] = Counter(hard_case.reason for hard_case in hard_cases)
    return dict(sorted(counts.items()))


def hard_case_visibility_counts(hard_cases: list[HardCase]) -> dict[str, int]:
    counts: Counter[str] = Counter(hard_case.visibility for hard_case in hard_cases)
    return dict(sorted(counts.items()))


def write_hard_buffer_jsonl(path: Path, hard_cases: list[HardCase]) -> Path:
    payloads = [hard_case.model_dump(mode="json", exclude_none=True) for hard_case in hard_cases]
    assert_no_forbidden_context(payloads)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(hard_case.model_dump_json(exclude_none=True) + "\n" for hard_case in hard_cases),
        encoding="utf-8",
    )
    return path


def load_hard_buffer_jsonl(path: Path) -> list[HardCase]:
    if not path.exists():
        return []
    hard_cases = [
        HardCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert_no_forbidden_context(
        [hard_case.model_dump(mode="json", exclude_none=True) for hard_case in hard_cases]
    )
    return hard_cases


def _hard_case_from_trace(
    trace: TeacherTrace,
    *,
    slow_latency_ms: float,
    visibility: HardCaseVisibility,
) -> HardCase | None:
    if trace.teacher_frame is None:
        return None

    total_latency_ms = sum(result.latency_ms for result in trace.layer_results)
    reasons: list[HardCaseReason] = []

    if trace.chosen_layer in WEAK_LAYERS and trace.final_frame != trace.teacher_frame:
        reasons.append("weak_wrong_accept")
    if trace.metadata.get("teacher_disagreed") is True:
        reasons.append("teacher_audit_disagreement")
    elif trace.final_frame != trace.teacher_frame:
        reasons.append("teacher_final_mismatch")

    if trace.chosen_layer == "L4":
        if _has_lower_layer_abstain(trace):
            reasons.append("fallback_after_weak_abstain")
        else:
            reasons.append("l4_fallback")

    if total_latency_ms >= slow_latency_ms:
        reasons.append("slow_path")

    if not reasons:
        return None

    reason = max(reasons, key=lambda item: REASON_WEIGHTS[item])
    severity = _severity(reasons, total_latency_ms)
    return HardCase(
        request_id=trace.request_id,
        reason=reason,
        reasons=reasons,
        severity=severity,
        chosen_layer=trace.chosen_layer,
        total_latency_ms=total_latency_ms,
        trace=trace,
        visibility=visibility,
    )


def _has_lower_layer_abstain(trace: TeacherTrace) -> bool:
    return any(
        result.layer in WEAK_LAYERS and not result.accepted for result in trace.layer_results
    )


def _severity(reasons: list[HardCaseReason], total_latency_ms: float) -> float:
    primary = max(REASON_WEIGHTS[reason] for reason in reasons)
    latency_bonus = min(total_latency_ms / 1000.0, 10.0)
    multi_reason_bonus = max(0, len(reasons) - 1) * 0.01
    return round(primary + latency_bonus + multi_reason_bonus, 4)


def _sort_key(hard_case: HardCase) -> tuple[float, str, int, str]:
    return (
        -hard_case.severity,
        str(hard_case.reason),
        _visibility_priority(hard_case.visibility),
        hard_case.request_id,
    )


def _visibility_priority(visibility: HardCaseVisibility) -> int:
    return 0 if visibility == "train_visible" else 1
