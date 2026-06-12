from __future__ import annotations

import json
from dataclasses import dataclass

from darjeeling.contracts import JsonObject
from darjeeling.targets.nlu.schemas import Frame, TaskSchema


class NluTeacherParseError(RuntimeError):
    pass


@dataclass(frozen=True)
class NluTeacherAdapter:
    prompt_version: str = "teacher-v1"

    def build_messages(
        self,
        *,
        input: JsonObject,
        task_schema: JsonObject,
    ) -> list[dict[str, str]]:
        schema = TaskSchema.from_payload(task_schema)
        utterance = _utterance_from_input(input)
        stable_prefix = build_teacher_system_prompt(schema, prompt_version=self.prompt_version)
        dynamic_tail = json.dumps({"utterance": utterance}, ensure_ascii=False, sort_keys=True)
        return [
            {"role": "system", "content": stable_prefix},
            {"role": "user", "content": dynamic_tail},
        ]

    def parse_response(
        self,
        raw_response: str,
        *,
        task_schema: JsonObject,
    ) -> JsonObject:
        del task_schema
        return parse_teacher_frame(raw_response).model_dump(mode="json")

    def cache_key_parts(self, *, task_schema: JsonObject) -> JsonObject:
        schema = TaskSchema.from_payload(task_schema)
        return {
            "prompt_version": self.prompt_version,
            "schema_version": schema.schema_version,
        }


def build_teacher_system_prompt(task_schema: TaskSchema, *, prompt_version: str) -> str:
    return "\n".join(
        [
            "You are the L4 teacher for Darjeeling, a schema-constrained frame task.",
            "Return strict JSON only.",
            "Do not include explanations or markdown.",
            "The JSON object must have this shape:",
            '{"intent": "intent_name", "slots": {"slot_name": "slot value"}, "is_abstain": false}',
            "Use only these intents:",
            json.dumps(task_schema.intent_names, ensure_ascii=False, sort_keys=True),
            "Use only these slot names when slots are present:",
            json.dumps(task_schema.slot_names, ensure_ascii=False, sort_keys=True),
            "If no slot is present, return an empty slots object.",
            f"Prompt version: {prompt_version}.",
        ]
    )


def parse_teacher_frame(raw_response: str) -> Frame:
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise NluTeacherParseError(f"teacher returned invalid JSON: {exc}") from exc
    try:
        return Frame.model_validate(payload)
    except Exception as exc:
        raise NluTeacherParseError(
            f"teacher response does not match NLU Frame schema: {exc}"
        ) from exc


def _utterance_from_input(input: JsonObject) -> str:
    utterance = input.get("utterance")
    if not isinstance(utterance, str) or not utterance:
        raise ValueError("NLU teacher input requires a non-empty utterance")
    return utterance
