from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from darjeeling.runtime.cost import replay_cost_model_from_settings
from darjeeling.runtime.timing import elapsed_ms
from darjeeling.targets.nlu.compiler.l4_context import (
    build_residual_teacher_context,
    build_teacher_context,
)
from darjeeling.targets.nlu.data import normalize_utterance
from darjeeling.targets.nlu.patches import FIELD_INTENT, slot_field_key
from darjeeling.targets.nlu.schemas import Frame, FramePatch, LayerResult, TaskSchema
from darjeeling.targets.nlu.settings import Settings
from darjeeling.targets.nlu.teacher import (
    NluTeacherParseError,
)
from darjeeling.targets.nlu.teacher import (
    build_teacher_system_prompt as build_nlu_teacher_system_prompt,
)
from darjeeling.targets.nlu.teacher import (
    parse_teacher_frame as parse_nlu_teacher_frame,
)

TEACHER_MODES = {"live", "cache", "live-or-cache"}

__all__ = [
    "CachedTeacherLayer",
    "CloudLLMTeacher",
    "MissingTeacherError",
    "TaskSchema",
    "TeacherCache",
    "TeacherCallResult",
    "TeacherPatchCallResult",
    "TeacherParseError",
    "build_teacher_system_prompt",
    "create_chat_completion_with_retry",
    "has_valid_teacher_cache",
    "parse_teacher_frame",
    "parse_teacher_patch",
    "require_live_or_cached_teacher",
    "teacher_cache_key",
    "_extract_chat_content",
    "_extract_usage",
]


class MissingTeacherError(RuntimeError):
    pass


class TeacherParseError(RuntimeError):
    pass


@dataclass(frozen=True)
class TeacherCallResult:
    frame: Frame
    raw_response: str
    usage: dict[str, Any]
    model: str
    context_hash: str
    prompt_cache_key: str


@dataclass(frozen=True)
class TeacherPatchCallResult:
    patch: FramePatch
    raw_response: str
    usage: dict[str, Any]
    model: str
    context_hash: str
    prompt_cache_key: str


def has_valid_teacher_cache(cache_path: Path) -> bool:
    return cache_path.exists() and cache_path.stat().st_size > 0


def require_live_or_cached_teacher(
    settings: Settings,
    teacher_mode: str,
    cache_path: Path,
) -> None:
    if teacher_mode not in TEACHER_MODES:
        raise MissingTeacherError(
            f"unknown teacher mode {teacher_mode!r}; expected live, cache, or live-or-cache"
        )

    cache_exists = has_valid_teacher_cache(cache_path)
    if teacher_mode == "cache" and not cache_exists:
        raise MissingTeacherError(f"teacher cache is required but missing: {cache_path}")

    live_allowed = teacher_mode in {"live", "live-or-cache"}
    if live_allowed and not settings.openai_api_key and not cache_exists:
        raise MissingTeacherError(
            "OPENAI_API_KEY is absent and no valid teacher cache exists; refusing to mock labels"
        )


class CloudLLMTeacher:
    def __init__(self, settings: Settings, *, client: Any | None = None) -> None:
        self.settings = settings
        self._client = client

    def client(self):
        if self._client is not None:
            return self._client
        if not self.settings.openai_api_key:
            raise MissingTeacherError("OPENAI_API_KEY is required for live L4 calls")
        from openai import OpenAI

        return OpenAI(
            api_key=self.settings.openai_api_key,
            base_url=self.settings.openai_base_url or None,
            timeout=self.settings.openai_timeout_s,
            max_retries=0,
        )

    def answer(self, utterance: str, task_schema: TaskSchema) -> TeacherCallResult:
        context = build_teacher_context(
            utterance=utterance,
            task_schema=task_schema,
            settings=self.settings,
        )
        response = create_chat_completion_with_retry(
            self.client(),
            self.settings,
            response_check=_extract_chat_content,
            model=self.settings.openai_model,
            messages=context.messages,
            response_format={"type": "json_object"},
            max_completion_tokens=self.settings.teacher_max_tokens,
            prompt_cache_key=context.prompt_cache_key,
            prompt_cache_retention=context.prompt_cache_retention,
            timeout=self.settings.openai_timeout_s,
        )
        raw_response = _extract_chat_content(response)
        return TeacherCallResult(
            frame=parse_teacher_frame(raw_response),
            raw_response=raw_response,
            usage=_extract_usage(response),
            model=getattr(response, "model", self.settings.openai_model),
            context_hash=context.context_hash,
            prompt_cache_key=context.prompt_cache_key,
        )

    def residual_patch(
        self,
        *,
        utterance: str,
        task_schema: TaskSchema,
        accepted_fields: dict[str, str],
        missing_fields: list[str],
    ) -> TeacherPatchCallResult:
        context = build_residual_teacher_context(
            utterance=utterance,
            accepted_fields=accepted_fields,
            missing_fields=missing_fields,
            task_schema=task_schema,
            settings=self.settings,
        )
        response = create_chat_completion_with_retry(
            self.client(),
            self.settings,
            response_check=_extract_chat_content,
            model=self.settings.openai_model,
            messages=context.messages,
            response_format={"type": "json_object"},
            max_completion_tokens=min(
                self.settings.teacher_max_tokens,
                self.settings.residual_l4_max_tokens,
            ),
            prompt_cache_key=context.prompt_cache_key,
            prompt_cache_retention=context.prompt_cache_retention,
            timeout=self.settings.openai_timeout_s,
        )
        raw_response = _extract_chat_content(response)
        return TeacherPatchCallResult(
            patch=parse_teacher_patch(raw_response, task_schema=task_schema),
            raw_response=raw_response,
            usage=_extract_usage(response),
            model=getattr(response, "model", self.settings.openai_model),
            context_hash=context.context_hash,
            prompt_cache_key=context.prompt_cache_key,
        )


class TeacherCache:
    def __init__(
        self,
        frames_by_normalized_utterance: dict[str, Frame],
        *,
        path: Path | None = None,
    ) -> None:
        self.frames_by_normalized_utterance = frames_by_normalized_utterance
        self.path = path

    @classmethod
    def load(cls, path: Path) -> TeacherCache:
        if not path.exists():
            return cls({}, path=path)

        frames: dict[str, Frame] = {}
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            utterance = payload.get("utterance")
            teacher_frame = payload.get("teacher_frame")
            if not utterance or teacher_frame is None:
                raise MissingTeacherError(
                    "invalid teacher cache line "
                    f"{line_number}: utterance and teacher_frame required"
                )
            frames[normalize_utterance(utterance)] = Frame.model_validate(teacher_frame)
        return cls(frames, path=path)

    def get(self, utterance: str) -> Frame | None:
        return self.frames_by_normalized_utterance.get(normalize_utterance(utterance))

    def exact_cache_frames(self) -> dict[str, Frame]:
        return dict(self.frames_by_normalized_utterance)

    def append(
        self,
        *,
        utterance: str,
        frame: Frame,
        task_schema: TaskSchema,
        settings: Settings,
        raw_response: str,
        usage: dict[str, Any],
        model: str,
        context_hash: str | None = None,
        prompt_cache_key: str | None = None,
    ) -> None:
        if self.path is None:
            raise MissingTeacherError("teacher cache path is not configured")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        normalized = normalize_utterance(utterance)
        payload = {
            "cache_key": teacher_cache_key(
                normalized_utterance=normalized,
                task_schema=task_schema,
                settings=settings,
                model=model,
            ),
            "utterance": utterance,
            "normalized_utterance": normalized,
            "schema_version": task_schema.schema_version,
            "prompt_version": settings.teacher_prompt_version,
            "model": model,
            "teacher_frame": frame.model_dump(mode="json"),
            "raw_response": raw_response,
            "usage": usage,
            "context_hash": context_hash,
            "prompt_cache_key": prompt_cache_key,
            "created_at": datetime.now(UTC).isoformat(),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        self.frames_by_normalized_utterance[normalized] = frame


class CachedTeacherLayer:
    def __init__(
        self,
        cache: TeacherCache,
        *,
        allow_live: bool,
        use_cache: bool,
        settings: Settings,
        task_schema: TaskSchema,
        teacher: CloudLLMTeacher | None = None,
    ) -> None:
        self.cache = cache
        self.allow_live = allow_live
        self.use_cache = use_cache
        self.settings = settings
        self.task_schema = task_schema
        self.teacher = teacher or CloudLLMTeacher(settings)

    def try_answer(self, utterance: str) -> LayerResult:
        with elapsed_ms() as ms:
            if self.use_cache and (cached_frame := self.cache.get(utterance)) is not None:
                return LayerResult(
                    layer="L4",
                    accepted=True,
                    frame=cached_frame,
                    confidence=1.0,
                    reason="teacher cache hit",
                    latency_ms=ms(),
                    metadata={"teacher_source": "cache", "l4_call_kind": "full"},
                )

            if self.allow_live:
                call_result = self.teacher.answer(utterance, self.task_schema)
                self.cache.append(
                    utterance=utterance,
                    frame=call_result.frame,
                    task_schema=self.task_schema,
                    settings=self.settings,
                    raw_response=call_result.raw_response,
                    usage=call_result.usage,
                    model=call_result.model,
                    context_hash=call_result.context_hash,
                    prompt_cache_key=call_result.prompt_cache_key,
                )
                return LayerResult(
                    layer="L4",
                    accepted=True,
                    frame=call_result.frame,
                    confidence=1.0,
                    reason="live teacher call",
                    latency_ms=ms(),
                    cost_usd=replay_cost_model_from_settings(self.settings).layer_cost_usd(
                        "L4",
                        call_result.usage,
                    ),
                    metadata={
                        "teacher_source": "live",
                        "l4_call_kind": "full",
                        "model": call_result.model,
                        "usage": call_result.usage,
                        "context_hash": call_result.context_hash,
                        "prompt_cache_key": call_result.prompt_cache_key,
                    },
                )

            raise MissingTeacherError(f"teacher cache miss for utterance: {utterance!r}")

    def residual_field_keys(self) -> list[str]:
        return [
            FIELD_INTENT,
            *(slot_field_key(slot_name) for slot_name in self.task_schema.slot_names),
        ]

    def try_residual_patch(
        self,
        utterance: str,
        *,
        accepted_fields: dict[str, str],
        missing_fields: list[str],
    ) -> LayerResult:
        fields_avoided = len(accepted_fields)
        with elapsed_ms() as ms:
            if self.use_cache and (cached_frame := self.cache.get(utterance)) is not None:
                patch = _residual_patch_from_frame(
                    cached_frame,
                    accepted_fields=accepted_fields,
                    missing_fields=missing_fields,
                    metadata={
                        "adapter": "l4_residual_cache",
                        "teacher_source": "cache",
                        "l4_call_kind": "residual",
                        "fields_avoided": fields_avoided,
                    },
                )
                return LayerResult(
                    layer="L4",
                    accepted=True,
                    patch=patch,
                    confidence=1.0,
                    reason="teacher cache residual fill",
                    latency_ms=ms(),
                    metadata={
                        "teacher_source": "cache",
                        "l4_call_kind": "residual",
                        "fields_avoided": fields_avoided,
                        "accepted_fields": accepted_fields,
                        "missing_fields": missing_fields,
                        "frame_patch": patch.model_dump(mode="json"),
                    },
                )

            if self.allow_live:
                call_result = self.teacher.residual_patch(
                    utterance=utterance,
                    task_schema=self.task_schema,
                    accepted_fields=accepted_fields,
                    missing_fields=missing_fields,
                )
                patch = call_result.patch.model_copy(
                    update={
                        "source_layer": "L4",
                        "metadata": {
                            **call_result.patch.metadata,
                            "adapter": "l4_residual_live",
                            "teacher_source": "live",
                            "l4_call_kind": "residual",
                            "fields_avoided": fields_avoided,
                        },
                    }
                )
                return LayerResult(
                    layer="L4",
                    accepted=True,
                    patch=patch,
                    confidence=1.0,
                    reason="live residual teacher call",
                    latency_ms=ms(),
                    cost_usd=replay_cost_model_from_settings(self.settings).layer_cost_usd(
                        "L4",
                        call_result.usage,
                    ),
                    metadata={
                        "teacher_source": "live",
                        "l4_call_kind": "residual",
                        "fields_avoided": fields_avoided,
                        "accepted_fields": accepted_fields,
                        "missing_fields": missing_fields,
                        "model": call_result.model,
                        "usage": call_result.usage,
                        "context_hash": call_result.context_hash,
                        "prompt_cache_key": call_result.prompt_cache_key,
                        "frame_patch": patch.model_dump(mode="json"),
                    },
                )

            raise MissingTeacherError(f"teacher cache miss for utterance: {utterance!r}")


def create_chat_completion_with_retry(
    client: Any,
    settings: Settings,
    *,
    response_check: Callable[[Any], Any] | None = None,
    **kwargs: Any,
) -> Any:
    attempts = max(1, settings.openai_max_retries + 1)
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            response = client.chat.completions.create(**kwargs)
            if response_check is not None:
                response_check(response)
            return response
        except Exception as exc:
            last_exc = exc
            if attempt >= attempts - 1:
                raise
            delay = _openai_retry_delay(settings, attempt)
            if delay > 0:
                time.sleep(delay)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("OpenAI retry loop exited without a response")


def _openai_retry_delay(settings: Settings, attempt: int) -> float:
    base = max(0.0, settings.openai_retry_base_delay_s)
    cap = max(0.0, settings.openai_retry_max_delay_s)
    return min(cap, base * (2**attempt))


def build_teacher_system_prompt(task_schema: TaskSchema, settings: Settings) -> str:
    return build_nlu_teacher_system_prompt(
        task_schema,
        prompt_version=settings.teacher_prompt_version,
    )


def parse_teacher_frame(raw_response: str) -> Frame:
    try:
        return parse_nlu_teacher_frame(raw_response)
    except NluTeacherParseError as exc:
        raise TeacherParseError(str(exc)) from exc


def parse_teacher_patch(raw_response: str, *, task_schema: TaskSchema) -> FramePatch:
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise TeacherParseError(f"teacher returned invalid JSON: {exc}") from exc
    if "intent" in payload or "slots" in payload:
        frame = Frame.model_validate(payload)
        patch = FramePatch(
            accepted_intent=frame.intent,
            accepted_slots=dict(frame.slots),
            source_layer="L4",
            complete=True,
        )
    else:
        patch = FramePatch.model_validate(
            {
                "source_layer": "L4",
                "complete": True,
                **payload,
            }
        )
    _validate_patch_schema(patch, task_schema)
    return patch.model_copy(update={"source_layer": "L4"})


def teacher_cache_key(
    *,
    normalized_utterance: str,
    task_schema: TaskSchema,
    settings: Settings,
    model: str,
) -> str:
    return "|".join(
        [
            normalized_utterance,
            task_schema.schema_version,
            settings.teacher_prompt_version,
            model,
        ]
    )


def _extract_chat_content(response: Any) -> str:
    try:
        content = response.choices[0].message.content
    except Exception as exc:
        raise TeacherParseError(f"teacher response has no message content: {exc}") from exc
    if not isinstance(content, str) or not content.strip():
        raise TeacherParseError("teacher response content is empty")
    return content


def _extract_usage(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump(mode="json")
    if isinstance(usage, dict):
        return dict(usage)
    return {
        key: getattr(usage, key)
        for key in ["prompt_tokens", "completion_tokens", "total_tokens"]
        if hasattr(usage, key)
    }


def _validate_patch_schema(patch: FramePatch, task_schema: TaskSchema) -> None:
    if patch.accepted_intent is not None and patch.accepted_intent not in task_schema.intent_names:
        raise TeacherParseError(
            f"residual teacher returned unsupported intent: {patch.accepted_intent}"
        )
    unsupported_slots = sorted(set(patch.accepted_slots) - set(task_schema.slot_names))
    if unsupported_slots:
        raise TeacherParseError(
            f"residual teacher returned unsupported slot: {unsupported_slots[0]}"
        )


def _residual_patch_from_frame(
    frame: Frame,
    *,
    accepted_fields: dict[str, str],
    missing_fields: list[str],
    metadata: dict[str, Any],
) -> FramePatch:
    accepted_intent: str | None = None
    accepted_slots: dict[str, str] = {}
    removed_fields: list[str] = []
    verified_fields: list[str] = []

    if FIELD_INTENT in missing_fields or accepted_fields.get(FIELD_INTENT) != frame.intent:
        accepted_intent = frame.intent
    else:
        verified_fields.append(FIELD_INTENT)

    for field_key, accepted_value in accepted_fields.items():
        if not field_key.startswith("slots."):
            continue
        slot_key = field_key.removeprefix("slots.")
        if slot_key not in frame.slots:
            removed_fields.append(field_key)
            continue
        if frame.slots[slot_key] != accepted_value:
            accepted_slots[slot_key] = frame.slots[slot_key]
        else:
            verified_fields.append(field_key)

    for field_key in missing_fields:
        if not field_key.startswith("slots."):
            continue
        slot_key = field_key.removeprefix("slots.")
        if slot_key in frame.slots:
            accepted_slots[slot_key] = frame.slots[slot_key]

    return FramePatch(
        accepted_intent=accepted_intent,
        accepted_slots=accepted_slots,
        source_layer="L4",
        complete=True,
        metadata={
            **metadata,
            "removed_fields": removed_fields,
            "verified_fields": sorted(set(verified_fields)),
        },
    )
