from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol, TypeAlias

from pydantic import BaseModel, ConfigDict, Field
from pydantic import JsonValue as PydanticJsonValue

from darjeeling.artifacts.store import ArtifactManifest

JsonValue: TypeAlias = PydanticJsonValue  # noqa: UP040
JsonObject: TypeAlias = dict[str, JsonValue]  # noqa: UP040

LayerName: TypeAlias = Literal["L0", "L1", "L2", "L3", "L4"]  # noqa: UP040


class LayerResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    layer: LayerName
    accepted: bool
    output: JsonObject | None = None
    confidence: float | None = None
    reason: str = ""
    latency_ms: float
    cost_usd: float = 0.0
    metadata: JsonObject = Field(default_factory=dict)


class TraceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    input: JsonObject
    gold_label: JsonObject | None = None
    teacher_label: JsonObject | None = None
    chosen_layer: LayerName
    final_output: JsonObject
    layer_results: list[LayerResult]
    l4_usage: JsonObject = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class TeacherTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    input: JsonObject
    teacher_label: JsonObject | None = None
    chosen_layer: LayerName
    final_output: JsonObject
    layer_results: list[LayerResult]
    l4_usage: JsonObject = Field(default_factory=dict)
    timestamp: str


def to_teacher_trace(trace: TraceRecord) -> TeacherTrace:
    return TeacherTrace.model_validate(trace.model_dump(exclude={"gold_label"}))


def traces_to_teacher_view(traces: Sequence[TraceRecord]) -> list[TeacherTrace]:
    return [to_teacher_trace(trace) for trace in traces]


class TargetSpec(Protocol):
    name: str
    schema_version: str

    def load_task_schema(self, records: Sequence[JsonObject]) -> JsonObject: ...

    def normalize_request(self, input: JsonObject) -> str: ...

    def validate_output(self, output: JsonObject, task_schema: JsonObject) -> None: ...

    def labels_equal(
        self,
        output: JsonObject,
        expected: JsonObject,
        *,
        task_schema: JsonObject,
    ) -> bool: ...

    def summarize_for_context(
        self,
        traces: Sequence[TeacherTrace],
        *,
        budget: int,
    ) -> JsonObject: ...


class TeacherAdapter(Protocol):
    prompt_version: str

    def build_messages(
        self,
        *,
        input: JsonObject,
        task_schema: JsonObject,
    ) -> list[dict[str, str]]: ...

    def parse_response(
        self,
        raw_response: str,
        *,
        task_schema: JsonObject,
    ) -> JsonObject: ...

    def cache_key_parts(self, *, task_schema: JsonObject) -> JsonObject: ...


class RuntimeLayer(Protocol):
    layer_name: LayerName

    def try_answer(self, input: JsonObject) -> LayerResult: ...


class TeacherRuntime(Protocol):
    def try_answer(self, input: JsonObject) -> LayerResult: ...


class TargetRuntime(Protocol):
    def build_layers(
        self,
        *,
        manifest: ArtifactManifest | None,
        teacher: TeacherRuntime,
        settings: JsonObject,
    ) -> Mapping[LayerName, RuntimeLayer | None]: ...


@dataclass(frozen=True)
class ArtifactCandidate:
    artifact_paths: Mapping[str, str]
    metadata: JsonObject = field(default_factory=dict)


@dataclass(frozen=True)
class CompileContext:
    run_dir: Path
    task_schema: JsonObject
    teacher_traces: Sequence[TeacherTrace]
    current_manifest: ArtifactManifest | None = None
    settings: JsonObject = field(default_factory=dict)


class TargetCompiler(Protocol):
    def propose_artifacts(
        self,
        context: CompileContext,
    ) -> Sequence[ArtifactCandidate]: ...
