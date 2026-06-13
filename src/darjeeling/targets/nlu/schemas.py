from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Frame(BaseModel):
    intent: str
    slots: dict[str, str] = Field(default_factory=dict)
    is_abstain: bool = False


LayerName = Literal["L0", "L1", "L2", "L3", "L4"]


class FramePatch(BaseModel):
    accepted_intent: str | None = None
    accepted_slots: dict[str, str] = Field(default_factory=dict)
    source_layer: LayerName
    confidence: float | None = None
    complete: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class LayerResult(BaseModel):
    layer: LayerName
    accepted: bool
    frame: Frame | None = None
    patch: FramePatch | None = None
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
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class TeacherTrace(BaseModel):
    """Compiler-visible trace view without the gold frame."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    utterance: str
    teacher_frame: Frame | None = None
    chosen_layer: LayerName
    final_frame: Frame
    layer_results: list[LayerResult]
    l4_usage: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: str


def to_teacher_trace(trace: TraceRecord) -> TeacherTrace:
    return TeacherTrace.model_validate(trace.model_dump(exclude={"gold_frame"}))


def traces_to_teacher_view(traces: list[TraceRecord]) -> list[TeacherTrace]:
    return [to_teacher_trace(trace) for trace in traces]


@dataclass(frozen=True)
class TaskSchema:
    intent_names: list[str]
    slot_names: list[str]
    schema_version: str = "task-schema-v1"

    @classmethod
    def from_payload(cls, payload: dict) -> TaskSchema:
        return cls(
            intent_names=[str(intent) for intent in payload.get("intent_names", [])],
            slot_names=[str(slot) for slot in payload.get("slot_names", [])],
            schema_version=str(payload.get("schema_version", "task-schema-v1")),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "intent_names": list(self.intent_names),
            "slot_names": list(self.slot_names),
            "schema_version": self.schema_version,
        }
