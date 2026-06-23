from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, Field

from darjeeling.targets.nlu.compiler.focus_tasks import (
    build_focus_tasks,
    focus_task_document_with_fields,
)
from darjeeling.targets.nlu.schemas import TeacherTrace
from darjeeling.targets.nlu.teacher import build_teacher_system_prompt

FORBIDDEN_CONTEXT_TERMS = (
    "gold_frame",
    "gold_intent",
    "gold_slots",
    "final eval labels",
    "future stream labels",
)


class L4ContextError(RuntimeError):
    pass


class ContextBlock(BaseModel):
    name: str
    content: Any
    source_trace_ids: list[str] = Field(default_factory=list)
    priority: float = 0.0


class L4RenderedContext(BaseModel):
    kind: Literal["teacher", "proposal"]
    prompt_version: str
    context_layout_version: str
    messages: list[dict[str, str]]
    context_hash: str
    source_trace_ids: list[str] = Field(default_factory=list)
    prompt_cache_key: str
    prompt_cache_retention: str
    stable_prefix: str
    dynamic_tail: str


def build_teacher_context(
    *,
    utterance: str,
    task_schema: Any,
    settings: Any,
) -> L4RenderedContext:
    stable_prefix = build_teacher_stable_prefix(task_schema=task_schema, settings=settings)
    dynamic_tail = json.dumps(
        {"utterance": utterance},
        ensure_ascii=False,
        sort_keys=True,
    )
    messages = [
        {"role": "system", "content": stable_prefix},
        {"role": "user", "content": dynamic_tail},
    ]
    assert_no_forbidden_context(messages)
    prompt_cache_key = (
        f"darjeeling:{settings.teacher_prompt_version}:"
        f"{getattr(task_schema, 'schema_version', 'schema-unknown')}"
    )
    return L4RenderedContext(
        kind="teacher",
        prompt_version=settings.teacher_prompt_version,
        context_layout_version="teacher-layout-v1",
        messages=messages,
        context_hash=context_hash(messages),
        source_trace_ids=[],
        prompt_cache_key=prompt_cache_key,
        prompt_cache_retention=settings.prompt_cache_retention,
        stable_prefix=stable_prefix,
        dynamic_tail=dynamic_tail,
    )


def build_residual_teacher_context(
    *,
    utterance: str,
    accepted_fields: dict[str, str],
    missing_fields: list[str],
    task_schema: Any,
    settings: Any,
) -> L4RenderedContext:
    stable_prefix = "\n".join(
        [
            "You are the L4 residual teacher for Darjeeling, a schema-constrained frame task.",
            "Return strict JSON only.",
            "Do not include explanations or markdown.",
            "The JSON object must have this shape:",
            '{"accepted_intent": null, "accepted_slots": {}, "complete": true, '
            '"metadata": {"verified_fields": [], "removed_fields": []}}',
            "complete=true means every accepted field was verified, corrected, or removed, "
            "and the frame is complete.",
            "If any accepted field was not checked, return complete=false.",
            "Put verified accepted field keys in metadata.verified_fields.",
            "Put removed accepted slot field keys in metadata.removed_fields.",
            "Return corrected fields in accepted_intent or accepted_slots.",
            "Only include fields that are missing, corrected, removed, or necessary to finish "
            "verification.",
            "Use field key `intent` for intent and `slots.<slot_name>` for slots.",
            "Use only these intents:",
            json.dumps(task_schema.intent_names, ensure_ascii=False, sort_keys=True),
            "Use only these slot names when slots are present:",
            json.dumps(task_schema.slot_names, ensure_ascii=False, sort_keys=True),
            f"Prompt version: {settings.teacher_prompt_version}-residual.",
        ]
    )
    dynamic_tail = json.dumps(
        {
            "utterance": utterance,
            "accepted_fields": accepted_fields,
            "missing_fields": missing_fields,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    messages = [
        {"role": "system", "content": stable_prefix},
        {"role": "user", "content": dynamic_tail},
    ]
    assert_no_forbidden_context(messages)
    schema_version = getattr(task_schema, "schema_version", "schema-unknown")
    prompt_version = f"{settings.teacher_prompt_version}-residual"
    return L4RenderedContext(
        kind="teacher",
        prompt_version=prompt_version,
        context_layout_version="teacher-residual-layout-v1",
        messages=messages,
        context_hash=context_hash(messages),
        source_trace_ids=[],
        prompt_cache_key=f"darjeeling:{prompt_version}:{schema_version}",
        prompt_cache_retention=settings.prompt_cache_retention,
        stable_prefix=stable_prefix,
        dynamic_tail=dynamic_tail,
    )


def build_teacher_stable_prefix(*, task_schema: Any, settings: Any) -> str:
    return build_teacher_system_prompt(
        task_schema,
        prompt_version=settings.teacher_prompt_version,
    )


def build_proposal_context(
    *,
    role: str,
    task_schema: Any,
    settings: Any,
    traces: list[TeacherTrace],
    output_schema: dict[str, Any],
    current_artifact_summary: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    max_dynamic_traces: int = 50,
) -> L4RenderedContext:
    stable_prefix = "\n".join(
        [
            f"You are Darjeeling L4 proposal adapter for {role}.",
            "Return strict JSON only.",
            "Use only teacher-visible traces. Never use gold labels.",
            "Output schema:",
            json.dumps(output_schema, ensure_ascii=False, sort_keys=True),
            "Intent schema:",
            json.dumps(task_schema.intent_names, ensure_ascii=False, sort_keys=True),
            "Slot schema:",
            json.dumps(task_schema.slot_names, ensure_ascii=False, sort_keys=True),
        ]
    )
    sorted_traces = sorted(
        (trace for trace in traces if trace.teacher_frame is not None),
        key=lambda trace: trace.request_id,
    )[:max_dynamic_traces]
    labeled_traces = [trace for trace in traces if trace.teacher_frame is not None]
    focus_tasks = focus_task_document_with_fields(
        build_focus_tasks(
            labeled_traces,
            max_tasks=min(8, max_dynamic_traces),
        ),
        labeled_traces,
    )
    dynamic_payload = {
        "current_artifact_summary": current_artifact_summary or {},
        "focus_tasks": focus_tasks,
        "metrics": metrics or {},
        "supporting_teacher_traces": [
            trace.model_dump(mode="json", exclude_none=True) for trace in sorted_traces
        ],
    }
    dynamic_tail = json.dumps(dynamic_payload, ensure_ascii=False, sort_keys=True)
    messages = [
        {"role": "system", "content": stable_prefix},
        {"role": "user", "content": dynamic_tail},
    ]
    assert_no_forbidden_context(messages)
    source_trace_ids = focus_tasks["source_trace_ids"] or [
        trace.request_id for trace in sorted_traces
    ]
    prompt_version = f"{role}-proposal-v1"
    schema_version = getattr(task_schema, "schema_version", "schema-unknown")
    return L4RenderedContext(
        kind="proposal",
        prompt_version=prompt_version,
        context_layout_version="proposal-layout-v2-focus-tasks",
        messages=messages,
        context_hash=context_hash(messages),
        source_trace_ids=source_trace_ids,
        prompt_cache_key=f"darjeeling:{prompt_version}:{schema_version}",
        prompt_cache_retention=settings.prompt_cache_retention,
        stable_prefix=stable_prefix,
        dynamic_tail=dynamic_tail,
    )


def context_hash(payload: Any) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def assert_no_forbidden_context(payload: Any) -> None:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    for term in FORBIDDEN_CONTEXT_TERMS:
        if term in serialized:
            raise L4ContextError(f"L4 context contains forbidden field: {term}")
