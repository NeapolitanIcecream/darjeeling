from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from darjeeling.compiler.l4_context import build_proposal_context
from darjeeling.layers.l4_cloud_llm import (
    MissingTeacherError,
    TaskSchema,
    TeacherParseError,
    _extract_chat_content,
    _extract_usage,
    create_chat_completion_with_retry,
)
from darjeeling.schemas import TeacherTrace
from darjeeling.settings import Settings


class ProposalParseError(RuntimeError):
    pass


@dataclass(frozen=True)
class L4ProposalCallResult:
    role: str
    proposal: dict[str, Any]
    raw_response: str
    usage: dict[str, Any]
    model: str
    context_hash: str
    prompt_cache_key: str
    source_trace_ids: list[str]


class L4ProposalAdapter:
    def __init__(self, settings: Settings, *, client: Any | None = None) -> None:
        self.settings = settings
        self._client = client

    def client(self):
        if self._client is not None:
            return self._client
        if not self.settings.openai_api_key:
            raise MissingTeacherError("OPENAI_API_KEY is required for live L4 proposal calls")
        from openai import OpenAI

        return OpenAI(
            api_key=self.settings.openai_api_key,
            base_url=self.settings.openai_base_url or None,
        )

    def propose(
        self,
        *,
        role: str,
        task_schema: TaskSchema,
        traces: list[TeacherTrace],
        output_schema: dict[str, Any],
        current_artifact_summary: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        max_dynamic_traces: int = 50,
    ) -> L4ProposalCallResult:
        context = build_proposal_context(
            role=role,
            task_schema=task_schema,
            settings=self.settings,
            traces=traces,
            output_schema=output_schema,
            current_artifact_summary=current_artifact_summary,
            metrics=metrics,
            max_dynamic_traces=max_dynamic_traces,
        )
        try:
            response = create_chat_completion_with_retry(
                self.client(),
                self.settings,
                response_check=_extract_chat_content,
                model=self.settings.openai_model,
                messages=context.messages,
                response_format={"type": "json_object"},
                max_completion_tokens=self.settings.proposal_max_tokens,
                prompt_cache_key=context.prompt_cache_key,
                prompt_cache_retention=context.prompt_cache_retention,
            )
            raw_response = _extract_chat_content(response)
            proposal = parse_proposal(raw_response, output_schema)
        except TeacherParseError as exc:
            raise ProposalParseError(str(exc)) from exc
        return L4ProposalCallResult(
            role=role,
            proposal=proposal,
            raw_response=raw_response,
            usage=_extract_usage(response),
            model=getattr(response, "model", self.settings.openai_model),
            context_hash=context.context_hash,
            prompt_cache_key=context.prompt_cache_key,
            source_trace_ids=context.source_trace_ids,
        )


def parse_proposal(raw_response: str, output_schema: dict[str, Any]) -> dict[str, Any]:
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise ProposalParseError(f"L4 proposal returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProposalParseError("L4 proposal must be a JSON object")
    _validate_basic_json_schema(payload, output_schema, path="$")
    return payload


def _validate_basic_json_schema(
    payload: Any,
    schema: dict[str, Any],
    *,
    path: str,
) -> None:
    expected_type = schema.get("type")
    if expected_type is not None and not _matches_json_type(payload, expected_type):
        raise ProposalParseError(f"{path} expected type {expected_type!r}")

    if expected_type == "object" or isinstance(payload, dict):
        if not isinstance(payload, dict):
            raise ProposalParseError(f"{path} expected object")
        for key in schema.get("required", []):
            if key not in payload:
                raise ProposalParseError(f"{path}.{key} is required")
        properties = schema.get("properties") or {}
        for key, property_schema in properties.items():
            if key in payload and isinstance(property_schema, dict):
                _validate_basic_json_schema(payload[key], property_schema, path=f"{path}.{key}")

    if expected_type == "array" or isinstance(payload, list):
        if not isinstance(payload, list):
            raise ProposalParseError(f"{path} expected array")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(payload):
                _validate_basic_json_schema(item, item_schema, path=f"{path}[{index}]")


def _matches_json_type(payload: Any, expected_type: str | list[str]) -> bool:
    if isinstance(expected_type, list):
        return any(_matches_json_type(payload, item) for item in expected_type)
    if expected_type == "object":
        return isinstance(payload, dict)
    if expected_type == "array":
        return isinstance(payload, list)
    if expected_type == "string":
        return isinstance(payload, str)
    if expected_type == "number":
        return isinstance(payload, int | float) and not isinstance(payload, bool)
    if expected_type == "integer":
        return isinstance(payload, int) and not isinstance(payload, bool)
    if expected_type == "boolean":
        return isinstance(payload, bool)
    if expected_type == "null":
        return payload is None
    return True
