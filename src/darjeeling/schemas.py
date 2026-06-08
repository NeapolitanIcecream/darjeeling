from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

LayerName = Literal["L0", "L1", "L2", "L3", "L4"]


class Frame(BaseModel):
    intent: str
    slots: dict[str, str] = Field(default_factory=dict)
    is_abstain: bool = False


class LayerResult(BaseModel):
    layer: LayerName
    accepted: bool
    frame: Frame | None = None
    confidence: float | None = None
    reason: str = ""
    latency_ms: float
    cost_usd: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class TraceRecord(BaseModel):
    request_id: str
    utterance: str
    gold_frame: Frame | None = None
    teacher_frame: Frame | None = None
    chosen_layer: LayerName
    final_frame: Frame
    layer_results: list[LayerResult]
    l4_usage: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class TeacherTrace(BaseModel):
    """Compiler-visible trace view.

    This model intentionally has no gold_frame field. Compiler code should accept
    TeacherTrace instances instead of TraceRecord to avoid accidental gold leakage.
    """

    model_config = ConfigDict(extra="forbid")

    request_id: str
    utterance: str
    teacher_frame: Frame | None = None
    chosen_layer: LayerName
    final_frame: Frame
    layer_results: list[LayerResult]
    l4_usage: dict[str, Any] = Field(default_factory=dict)
    timestamp: str


def to_teacher_trace(trace: TraceRecord) -> TeacherTrace:
    return TeacherTrace.model_validate(trace.model_dump(exclude={"gold_frame"}))


def traces_to_teacher_view(traces: list[TraceRecord]) -> list[TeacherTrace]:
    return [to_teacher_trace(trace) for trace in traces]
