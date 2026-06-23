from __future__ import annotations

import json
from dataclasses import dataclass

from darjeeling.contracts import JsonObject
from darjeeling.targets.nlu.schemas import Frame, TaskSchema


class NluTeacherParseError(RuntimeError):
    pass


TEACHER_PROMPT_V1 = "teacher-v1"
TEACHER_PROMPT_V2_INTENT_FIRST = "teacher-v2-intent-first"
TEACHER_PROMPT_V3_SLOT_CONSERVATIVE = "teacher-v3-slot-conservative"
TEACHER_PROMPT_V4_SLOT_EVIDENCE = "teacher-v4-slot-evidence"
TEACHER_PROMPT_V5_VALUE_COPY = "teacher-v5-value-copy"
TEACHER_PROMPT_V6_SCHEMA_CHECKLIST = "teacher-v6-schema-checklist"
TEACHER_PROMPT_V7_EVIDENCE_STABLE = "teacher-v7-evidence-stable"
TEACHER_PROMPT_V8_EVIDENCE_COMPACT = "teacher-v8-evidence-compact"
CLINC150_PROMPT_V1 = "clinc150-intent-v1"
CLINC150_PROMPT_V2_LABEL_CARDS = "clinc150-intent-v2-label-cards"
SUPPORTED_TEACHER_PROMPT_VERSIONS = (
    TEACHER_PROMPT_V1,
    TEACHER_PROMPT_V2_INTENT_FIRST,
    TEACHER_PROMPT_V3_SLOT_CONSERVATIVE,
    TEACHER_PROMPT_V4_SLOT_EVIDENCE,
    TEACHER_PROMPT_V5_VALUE_COPY,
    TEACHER_PROMPT_V6_SCHEMA_CHECKLIST,
    TEACHER_PROMPT_V7_EVIDENCE_STABLE,
    TEACHER_PROMPT_V8_EVIDENCE_COMPACT,
    CLINC150_PROMPT_V1,
    CLINC150_PROMPT_V2_LABEL_CARDS,
)


@dataclass(frozen=True)
class NluTeacherAdapter:
    prompt_version: str = TEACHER_PROMPT_V1

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
    ensure_supported_teacher_prompt_version(prompt_version)
    if is_clinc150_teacher_prompt_version(prompt_version):
        return build_clinc150_intent_system_prompt(
            task_schema,
            prompt_version=prompt_version,
        )
    if prompt_version == TEACHER_PROMPT_V2_INTENT_FIRST:
        return _build_intent_first_full_frame_prompt(task_schema, prompt_version=prompt_version)
    if prompt_version == TEACHER_PROMPT_V3_SLOT_CONSERVATIVE:
        return _build_slot_conservative_full_frame_prompt(
            task_schema,
            prompt_version=prompt_version,
        )
    if prompt_version == TEACHER_PROMPT_V4_SLOT_EVIDENCE:
        return _build_slot_evidence_full_frame_prompt(
            task_schema,
            prompt_version=prompt_version,
        )
    if prompt_version == TEACHER_PROMPT_V5_VALUE_COPY:
        return _build_value_copy_full_frame_prompt(
            task_schema,
            prompt_version=prompt_version,
        )
    if prompt_version == TEACHER_PROMPT_V6_SCHEMA_CHECKLIST:
        return _build_schema_checklist_full_frame_prompt(
            task_schema,
            prompt_version=prompt_version,
        )
    if prompt_version == TEACHER_PROMPT_V7_EVIDENCE_STABLE:
        return _build_evidence_stable_full_frame_prompt(
            task_schema,
            prompt_version=prompt_version,
        )
    if prompt_version == TEACHER_PROMPT_V8_EVIDENCE_COMPACT:
        return _build_evidence_compact_full_frame_prompt(
            task_schema,
            prompt_version=prompt_version,
        )
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


def build_teacher_intent_system_prompt(task_schema: TaskSchema, *, prompt_version: str) -> str:
    ensure_supported_teacher_prompt_version(prompt_version)
    return "\n".join(
        [
            "You are the L4 teacher for Darjeeling, a schema-constrained NLU task.",
            "Step 1: choose only the intent for the utterance.",
            "Return strict JSON only.",
            "Do not include explanations or markdown.",
            "The JSON object must have this shape:",
            '{"intent": "intent_name"}',
            "Use only these intents:",
            json.dumps(task_schema.intent_names, ensure_ascii=False, sort_keys=True),
            f"Prompt version: {prompt_version}-intent.",
        ]
    )


def build_teacher_slots_system_prompt(task_schema: TaskSchema, *, prompt_version: str) -> str:
    ensure_supported_teacher_prompt_version(prompt_version)
    return "\n".join(
        [
            "You are the L4 teacher for Darjeeling, a schema-constrained NLU task.",
            "Step 2: extract slots after the intent has already been fixed.",
            "Return strict JSON only.",
            "Do not include explanations or markdown.",
            "The JSON object must have this shape:",
            '{"intent": "intent_name", "slots": {"slot_name": "slot value"}, "is_abstain": false}',
            "The returned intent must exactly match the provided intent.",
            "Use only these slot names when slots are present:",
            json.dumps(task_schema.slot_names, ensure_ascii=False, sort_keys=True),
            "If no slot is present, return an empty slots object.",
            f"Prompt version: {prompt_version}-slots.",
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


def parse_clinc150_teacher_frame(raw_response: str, *, task_schema: TaskSchema) -> Frame:
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise NluTeacherParseError(f"teacher returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise NluTeacherParseError("CLINC150 teacher response must be a JSON object")
    if set(payload) != {"intent"}:
        raise NluTeacherParseError(
            "CLINC150 teacher response must contain exactly one field: intent"
        )
    intent = payload["intent"]
    if not isinstance(intent, str) or not intent:
        raise NluTeacherParseError("CLINC150 teacher response requires non-empty intent")
    if intent not in task_schema.intent_names:
        raise NluTeacherParseError(f"CLINC150 teacher returned unsupported intent: {intent}")
    return Frame(intent=intent, slots={}, is_abstain=intent == "out_of_scope")


def parse_teacher_intent(raw_response: str, *, task_schema: TaskSchema) -> str:
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise NluTeacherParseError(f"teacher returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise NluTeacherParseError("teacher intent response must be a JSON object")
    intent = payload.get("intent")
    if not isinstance(intent, str) or not intent:
        raise NluTeacherParseError("teacher intent response requires non-empty intent")
    if intent not in task_schema.intent_names:
        raise NluTeacherParseError(f"teacher returned unsupported intent: {intent}")
    return intent


def ensure_supported_teacher_prompt_version(prompt_version: str) -> None:
    if not prompt_version:
        raise ValueError("teacher prompt version must be non-empty")
    if prompt_version not in SUPPORTED_TEACHER_PROMPT_VERSIONS:
        supported = ", ".join(SUPPORTED_TEACHER_PROMPT_VERSIONS)
        raise ValueError(
            f"unsupported teacher prompt version {prompt_version!r}; "
            f"supported versions: {supported}"
        )


def is_clinc150_teacher_prompt_version(prompt_version: str) -> bool:
    return prompt_version in {CLINC150_PROMPT_V1, CLINC150_PROMPT_V2_LABEL_CARDS}


def build_clinc150_intent_system_prompt(
    task_schema: TaskSchema,
    *,
    prompt_version: str,
    label_cards: list[dict[str, object]] | None = None,
) -> str:
    ensure_supported_teacher_prompt_version(prompt_version)
    if prompt_version == CLINC150_PROMPT_V1:
        return "\n".join(
            [
                "You are the L4 teacher for a CLINC150 intent classification task.",
                "Return strict JSON only.",
                "Do not include explanations, markdown, slots, or extra fields.",
                "The JSON object must have exactly this shape:",
                '{"intent": "intent_name"}',
                "Choose exactly one allowed intent.",
                "Use out_of_scope only when the utterance does not fit any in-scope intent.",
                "Allowed intents:",
                json.dumps(task_schema.intent_names, ensure_ascii=False, sort_keys=True),
                f"Prompt version: {prompt_version}.",
            ]
        )
    cards = label_cards or _default_label_cards(task_schema.intent_names)
    return "\n".join(
        [
            "You are the L4 teacher for a CLINC150 intent classification task.",
            "Return strict JSON only.",
            "Do not include explanations, markdown, slots, or extra fields.",
            "The JSON object must have exactly this shape:",
            '{"intent": "intent_name"}',
            "Choose exactly one allowed intent.",
            "Use out_of_scope only when the utterance does not fit any in-scope intent.",
            "Use these label cards. Examples, when present, are train-split examples only:",
            json.dumps(cards, ensure_ascii=False, sort_keys=True),
            f"Prompt version: {prompt_version}.",
        ]
    )


def _default_label_cards(intent_names: list[str]) -> list[dict[str, object]]:
    return [
        {
            "intent": intent,
            "description": (
                "unsupported or out-of-scope request"
                if intent == "out_of_scope"
                else intent.replace("_", " ")
            ),
            "examples": [],
        }
        for intent in intent_names
    ]


def _build_intent_first_full_frame_prompt(
    task_schema: TaskSchema,
    *,
    prompt_version: str,
) -> str:
    return "\n".join(
        [
            "You are the L4 teacher for Darjeeling, a schema-constrained frame task.",
            "Return strict JSON only.",
            "Do not include explanations or markdown.",
            "Before writing the final JSON, decide the intent first, then extract slots for "
            "that intent.",
            "The JSON object must have this shape:",
            '{"intent": "intent_name", "slots": {"slot_name": "slot value"}, "is_abstain": false}',
            "Use only these intents:",
            json.dumps(task_schema.intent_names, ensure_ascii=False, sort_keys=True),
            "Use only these slot names when slots are present:",
            json.dumps(task_schema.slot_names, ensure_ascii=False, sort_keys=True),
            "If no slot is present, return an empty slots object.",
            "Do not include the intermediate intent decision in the response.",
            f"Prompt version: {prompt_version}.",
        ]
    )


def _build_slot_conservative_full_frame_prompt(
    task_schema: TaskSchema,
    *,
    prompt_version: str,
) -> str:
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
            "Choose the best intent, then extract only slots that are explicitly supported by "
            "the utterance text.",
            "Do not infer hidden defaults, categories, dates, devices, media types, or other "
            "slots that are not stated.",
            "Prefer an empty slots object over a guessed slot.",
            "If no slot is present, return an empty slots object.",
            f"Prompt version: {prompt_version}.",
        ]
    )


def _build_slot_evidence_full_frame_prompt(
    task_schema: TaskSchema,
    *,
    prompt_version: str,
) -> str:
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
            "Choose the best intent from the utterance as a whole.",
            "Add a slot only when the utterance gives specific evidence for that slot.",
            "Do not add a slot whose value merely repeats the object or action already implied "
            "by the chosen intent.",
            "Do not invent hidden defaults, device categories, media categories, meal types, "
            "dates, times, places, or names.",
            "When a slot is supported, use the shortest explicit phrase from the utterance.",
            "If no slot is specifically supported, return an empty slots object.",
            f"Prompt version: {prompt_version}.",
        ]
    )


def _build_value_copy_full_frame_prompt(
    task_schema: TaskSchema,
    *,
    prompt_version: str,
) -> str:
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
            "Choose the best intent first, then fill slots.",
            "For each slot value, copy the exact words from the utterance.",
            "Preserve lowercase, punctuation, possessives, and word order from the utterance.",
            "Do not title-case, canonicalize, singularize, pluralize, or paraphrase slot values.",
            "Omit a slot if the value would require guessing or normalizing beyond the utterance.",
            "If no slot is present, return an empty slots object.",
            f"Prompt version: {prompt_version}.",
        ]
    )


def _build_schema_checklist_full_frame_prompt(
    task_schema: TaskSchema,
    *,
    prompt_version: str,
) -> str:
    return "\n".join(
        [
            "You are the L4 teacher for Darjeeling, a schema-constrained frame task.",
            "Return strict JSON only.",
            "Do not include explanations or markdown.",
            "Use this private checklist, but output only the final JSON object:",
            "1. Pick exactly one allowed intent.",
            "2. Review the allowed slot names and include only slots directly expressed.",
            "3. Copy slot values from the utterance using the smallest complete phrase.",
            "4. Drop slots that are only hidden defaults, broad categories, or restatements of "
            "the chosen intent.",
            "5. If intent is clear but a slot is uncertain, keep the intent and omit that slot.",
            "The JSON object must have this shape:",
            '{"intent": "intent_name", "slots": {"slot_name": "slot value"}, "is_abstain": false}',
            "Use only these intents:",
            json.dumps(task_schema.intent_names, ensure_ascii=False, sort_keys=True),
            "Use only these slot names when slots are present:",
            json.dumps(task_schema.slot_names, ensure_ascii=False, sort_keys=True),
            "If no slot survives the checklist, return an empty slots object.",
            f"Prompt version: {prompt_version}.",
        ]
    )


def _build_evidence_stable_full_frame_prompt(
    task_schema: TaskSchema,
    *,
    prompt_version: str,
) -> str:
    return "\n".join(
        [
            "You are the L4 teacher for Darjeeling, a schema-constrained frame task.",
            "Return strict JSON only, with no explanations or markdown.",
            "The JSON object must have this shape:",
            '{"intent": "intent_name", "slots": {"slot_name": "slot value"}, "is_abstain": false}',
            "Use only these intents:",
            json.dumps(task_schema.intent_names, ensure_ascii=False, sort_keys=True),
            "Use only these slot names when slots are present:",
            json.dumps(task_schema.slot_names, ensure_ascii=False, sort_keys=True),
            "First choose the best intent from the whole utterance; do not change intent "
            "because a slot is uncertain.",
            "Include a slot only when it adds concrete information beyond the chosen intent.",
            "Omit objects, actions, broad categories, and hidden defaults already implied by "
            "the intent.",
            "Copy slot values from the utterance using the shortest complete phrase and the "
            "same casing.",
            "Do not add filler time/date words such as now or currently unless time/date is "
            "the requested value.",
            "If a slot is uncertain, keep the intent and omit the slot.",
            "If no slot is specifically supported, return an empty slots object.",
            f"Prompt version: {prompt_version}.",
        ]
    )


def _build_evidence_compact_full_frame_prompt(
    task_schema: TaskSchema,
    *,
    prompt_version: str,
) -> str:
    return "\n".join(
        [
            "You are the L4 teacher for Darjeeling, a schema-constrained frame task.",
            "Return strict JSON only:",
            '{"intent": "intent_name", "slots": {"slot_name": "slot value"}, "is_abstain": false}',
            "Allowed intents:",
            json.dumps(task_schema.intent_names, ensure_ascii=False, sort_keys=True),
            "Allowed slots:",
            json.dumps(task_schema.slot_names, ensure_ascii=False, sort_keys=True),
            "Pick the intent from the whole utterance.",
            "Slots must be explicit, add information beyond the intent, and use the shortest "
            "phrase copied from the utterance.",
            "Omit hidden defaults, broad categories, objects/actions implied by the intent, "
            "and filler time words such as now/currently.",
            "If a slot is uncertain, omit it; do not change a clear intent because of slots.",
            "Use {} when no slot is specifically supported.",
            f"Prompt version: {prompt_version}.",
        ]
    )


def _utterance_from_input(input: JsonObject) -> str:
    utterance = input.get("utterance")
    if not isinstance(utterance, str) or not utterance:
        raise ValueError("NLU teacher input requires a non-empty utterance")
    return utterance
