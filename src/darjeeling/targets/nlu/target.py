from __future__ import annotations

from collections.abc import Mapping, Sequence

from darjeeling.artifacts.store import ArtifactManifest
from darjeeling.contracts import (
    JsonObject,
    LayerName,
    RuntimeLayer,
    TeacherRuntime,
    TeacherTrace,
)
from darjeeling.targets.nlu.data import normalize_utterance
from darjeeling.targets.nlu.schemas import Frame, TaskSchema
from darjeeling.targets.nlu.teacher import NluTeacherAdapter


class NluTargetSpec:
    name = "nlu"
    schema_version = "nlu-target-v1"

    def load_task_schema(self, records: Sequence[JsonObject]) -> JsonObject:
        frames = [
            Frame.model_validate(record["gold_frame"])
            for record in records
            if record.get("gold_frame") is not None
        ]
        intent_names = sorted({frame.intent for frame in frames})
        slot_names = sorted({slot_name for frame in frames for slot_name in frame.slots})
        return TaskSchema(intent_names=intent_names, slot_names=slot_names).to_payload()

    def normalize_request(self, input: JsonObject) -> str:
        utterance = input.get("utterance")
        if not isinstance(utterance, str):
            raise ValueError("NLU request requires utterance")
        return normalize_utterance(utterance)

    def validate_output(self, output: JsonObject, task_schema: JsonObject) -> None:
        schema = TaskSchema.from_payload(task_schema)
        frame = Frame.model_validate(output)
        if frame.intent not in schema.intent_names:
            raise ValueError(f"intent not allowed: {frame.intent}")
        unsupported_slots = sorted(set(frame.slots) - set(schema.slot_names))
        if unsupported_slots:
            raise ValueError(f"slot not allowed: {unsupported_slots[0]}")

    def labels_equal(
        self,
        output: JsonObject,
        expected: JsonObject,
        *,
        task_schema: JsonObject,
    ) -> bool:
        del task_schema
        return Frame.model_validate(output) == Frame.model_validate(expected)

    def summarize_for_context(
        self,
        traces: Sequence[TeacherTrace],
        *,
        budget: int,
    ) -> JsonObject:
        examples: list[JsonObject] = []
        for trace in traces[: max(0, budget)]:
            if trace.teacher_label is None:
                continue
            examples.append(
                {
                    "request_id": trace.request_id,
                    "input": trace.input,
                    "teacher_label": trace.teacher_label,
                    "chosen_layer": trace.chosen_layer,
                }
            )
        return {"target": self.name, "examples": examples}


class NluTargetRuntime:
    def build_layers(
        self,
        *,
        manifest: ArtifactManifest,
        teacher: TeacherRuntime,
        settings: JsonObject,
    ) -> Mapping[LayerName, RuntimeLayer | None]:
        del manifest, teacher, settings
        return {}


class NluTarget(NluTargetSpec):
    def __init__(self) -> None:
        self.teacher_adapter = NluTeacherAdapter()
        self.runtime = NluTargetRuntime()
