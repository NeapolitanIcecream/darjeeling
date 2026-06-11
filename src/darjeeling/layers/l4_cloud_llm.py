from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from darjeeling.compiler.l4_context import build_teacher_context, build_teacher_stable_prefix
from darjeeling.data.frames import normalize_utterance
from darjeeling.runtime.cost import replay_cost_model_from_settings
from darjeeling.runtime.timing import elapsed_ms
from darjeeling.schemas import Frame, LayerResult
from darjeeling.settings import Settings

TEACHER_MODES = {"live", "cache", "live-or-cache"}


class MissingTeacherError(RuntimeError):
    pass


class TeacherParseError(RuntimeError):
    pass


@dataclass(frozen=True)
class TaskSchema:
    intent_names: list[str]
    slot_names: list[str]
    schema_version: str = "task-schema-v1"


@dataclass(frozen=True)
class TeacherCallResult:
    frame: Frame
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
                    metadata={"teacher_source": "cache"},
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
                        "model": call_result.model,
                        "usage": call_result.usage,
                        "context_hash": call_result.context_hash,
                        "prompt_cache_key": call_result.prompt_cache_key,
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
    return build_teacher_stable_prefix(task_schema=task_schema, settings=settings)


def parse_teacher_frame(raw_response: str) -> Frame:
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise TeacherParseError(f"teacher returned invalid JSON: {exc}") from exc
    try:
        return Frame.model_validate(payload)
    except Exception as exc:
        raise TeacherParseError(f"teacher response does not match Frame schema: {exc}") from exc


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
