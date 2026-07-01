from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from darjeeling.model import ReferenceContext, ReferenceResponse
from darjeeling.util import read_json, stable_hash, utcnow, write_json_atomic


@dataclass(frozen=True)
class ReferenceProviderConfig:
    provider: str
    base_url_env: str
    api_key_env: str
    model: str
    timeout_ms: int = 30_000
    max_completion_tokens: int | None = 2_048
    price: dict[str, float] = field(default_factory=dict)
    cache_path: Path | None = None
    usage_ledger_path: Path | None = None
    endpoint_path: str = "/chat/completions"


def load_reference_provider_config(path: Path) -> ReferenceProviderConfig:
    config_path = path.resolve()
    if not config_path.exists():
        raise ValueError(f"reference config not found: {config_path}")
    if config_path.suffix.lower() in {".yaml", ".yml"}:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    else:
        raw = read_json(config_path)
    if not isinstance(raw, dict):
        raise ValueError("reference config must contain a mapping")
    unknown = set(raw) - {
        "provider",
        "base_url_env",
        "api_key_env",
        "model",
        "timeout_ms",
        "max_completion_tokens",
        "price",
        "cache_path",
        "usage_ledger_path",
        "endpoint_path",
    }
    if unknown:
        raise ValueError(f"unknown reference config fields: {sorted(unknown)}")
    provider = raw.get("provider")
    if provider != "openai_compatible":
        raise ValueError("only provider='openai_compatible' is supported")
    model = raw.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("reference config model must be non-empty text")
    base_url_env = raw.get("base_url_env", "OPENAI_BASE_URL")
    api_key_env = raw.get("api_key_env", "OPENAI_API_KEY")
    if not isinstance(base_url_env, str) or not base_url_env:
        raise ValueError("reference config base_url_env must be non-empty text")
    if not isinstance(api_key_env, str) or not api_key_env:
        raise ValueError("reference config api_key_env must be non-empty text")
    timeout_ms = _positive_int(raw.get("timeout_ms", 30_000), "timeout_ms")
    max_completion_tokens = raw.get("max_completion_tokens", 2_048)
    if max_completion_tokens is not None:
        max_completion_tokens = _positive_int(
            max_completion_tokens, "max_completion_tokens"
        )
    price = raw.get("price", {})
    if price is None:
        price = {}
    if not isinstance(price, dict):
        raise ValueError("reference config price must be a mapping")
    normalized_price = {
        str(key): _non_negative_float(value, f"price.{key}")
        for key, value in price.items()
    }
    endpoint_path = raw.get("endpoint_path", "/chat/completions")
    if not isinstance(endpoint_path, str) or not endpoint_path.startswith("/"):
        raise ValueError("reference config endpoint_path must start with '/'")
    cache_path = _optional_relative_path(config_path, raw.get("cache_path"))
    usage_ledger_path = _optional_relative_path(
        config_path, raw.get("usage_ledger_path")
    )
    return ReferenceProviderConfig(
        provider=provider,
        base_url_env=base_url_env,
        api_key_env=api_key_env,
        model=model,
        timeout_ms=timeout_ms,
        max_completion_tokens=max_completion_tokens,
        price=normalized_price,
        cache_path=cache_path,
        usage_ledger_path=usage_ledger_path,
        endpoint_path=endpoint_path,
    )


def build_reference_broker_from_config(path: Path) -> OpenAICompatibleReferenceBroker:
    return OpenAICompatibleReferenceBroker(load_reference_provider_config(path))


class OpenAICompatibleReferenceBroker:
    def __init__(self, config: ReferenceProviderConfig):
        self.config = config
        self.reference_version = config.model
        self.base_url = _read_base_url_env(config.base_url_env).rstrip("/")
        self.api_key = _read_env(config.api_key_env)
        self._cache: dict[str, dict[str, Any]] = {}
        if config.cache_path is not None and config.cache_path.exists():
            for line in config.cache_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                if isinstance(record, dict) and isinstance(record.get("cache_key"), str):
                    self._cache[record["cache_key"]] = record

    def call(self, request: dict[str, Any], context: ReferenceContext) -> ReferenceResponse:
        provider_request = self._provider_request(request)
        cache_key = self._cache_key(provider_request, context)
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._write_usage_event(
                {
                    "status": "cache_hit",
                    "cache_key": cache_key,
                    "cost_usd": 0.0,
                    "cost_status": "cache-hit",
                    "cached_cost_usd": cached.get("cost_usd"),
                    "usage": cached.get("usage", {}),
                    "latency_ms": 0.0,
                    "purpose": context.purpose,
                    "request_id": context.request_id,
                }
            )
            return ReferenceResponse(
                payload=dict(cached["payload"]),
                reference_source="versioned_l4",
                reference_version=str(cached.get("reference_version") or self.reference_version),
                usage=dict(cached.get("usage", {})),
                cost=0.0,
                latency_ms=0.0,
                finish_status=str(cached.get("finish_status") or "cache_hit"),
            )
        started = time.perf_counter()
        try:
            response = self._call_provider(provider_request, context)
        except Exception as exc:
            latency_ms = (time.perf_counter() - started) * 1000.0
            self._write_usage_event(
                {
                    "status": "error",
                    "cache_key": cache_key,
                    "cost_usd": 0.0,
                    "cost_status": "provider-error-no-usage",
                    "usage": {},
                    "latency_ms": latency_ms,
                    "purpose": context.purpose,
                    "request_id": context.request_id,
                    "error_class": exc.__class__.__name__,
                }
            )
            raise
        latency_ms = (time.perf_counter() - started) * 1000.0
        usage = _usage_from_response(response)
        provider_cost = _provider_cost_from_response(response)
        try:
            cost_usd, cost_status = _cost_from_usage(
                usage, self.config.price, provider_cost
            )
        except ValueError as exc:
            self._write_usage_event(
                {
                    "status": "error",
                    "cache_key": cache_key,
                    "cost_usd": 0.0,
                    "cost_status": "unknown-token-price",
                    "usage": usage,
                    "latency_ms": latency_ms,
                    "purpose": context.purpose,
                    "request_id": context.request_id,
                    "error_class": exc.__class__.__name__,
                }
            )
            raise
        finish_status = _finish_status(response)
        payload = _response_payload(response)
        record = {
            "cache_key": cache_key,
            "request_hash": stable_hash(provider_request),
            "payload": payload,
            "reference_version": self.reference_version,
            "usage": usage,
            "cost_usd": cost_usd,
            "cost_status": cost_status,
            "latency_ms": latency_ms,
            "finish_status": finish_status,
            "model": self.config.model,
            "base_url_identity": stable_hash(self.base_url),
            "created_at": utcnow(),
        }
        self._append_cache_record(record)
        self._write_usage_event(
            {
                "status": "provider_call",
                "cache_key": cache_key,
                "cost_usd": cost_usd,
                "cost_status": cost_status,
                "usage": usage,
                "latency_ms": latency_ms,
                "purpose": context.purpose,
                "request_id": context.request_id,
                "finish_status": finish_status,
            }
        )
        return ReferenceResponse(
            payload=payload,
            reference_source="versioned_l4",
            reference_version=self.reference_version,
            usage=usage,
            cost=cost_usd,
            latency_ms=latency_ms,
            finish_status=finish_status,
        )

    def _cache_key(self, request: dict[str, Any], context: ReferenceContext) -> str:
        metadata = context.metadata or {}
        decoding = {
            "max_completion_tokens": self.config.max_completion_tokens,
            "endpoint_path": self.config.endpoint_path,
        }
        return stable_hash(
            {
                "provider": self.config.provider,
                "base_url_identity": stable_hash(self.base_url),
                "model": self.config.model,
                "contract_hash": metadata.get("contract_hash"),
                "normalized_input": metadata.get("normalized_input"),
                "purpose": context.purpose,
                "request_hash": stable_hash(request),
                "decoding": decoding,
            }
        )

    def _call_provider(
        self, request: dict[str, Any], context: ReferenceContext
    ) -> dict[str, Any]:
        body = self._provider_request(request)
        if (
            self.config.max_completion_tokens is not None
            and "max_completion_tokens" not in body
            and "max_tokens" not in body
        ):
            body["max_completion_tokens"] = self.config.max_completion_tokens
        timeout_ms = min(
            self.config.timeout_ms,
            _positive_int(
                context.metadata.get("timeout_ms", self.config.timeout_ms),
                "context timeout_ms",
            ),
        )
        cancel_event = context.metadata.get("cancel_event")
        if getattr(cancel_event, "is_set", lambda: False)():
            raise TimeoutError("reference call canceled before provider request")
        url = self.base_url + self.config.endpoint_path
        http_request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                http_request, timeout=max(timeout_ms, 1) / 1000
            ) as response:
                raw = response.read().decode("utf-8")
                parsed = json.loads(raw)
                if not isinstance(parsed, dict):
                    raise RuntimeError("provider response must be a JSON object")
                cost = _cost_from_headers(response.headers)
                if cost is not None:
                    parsed["_darjeeling_provider_cost_usd"] = cost
                return parsed
        except urllib.error.HTTPError as exc:
            text = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"reference provider HTTP {exc.code}: {text[:500]}") from exc

    def _provider_request(self, request: dict[str, Any]) -> dict[str, Any]:
        body = dict(request)
        body["model"] = self.config.model
        return body

    def _append_cache_record(self, record: dict[str, Any]) -> None:
        self._cache[record["cache_key"]] = record
        if self.config.cache_path is None:
            return
        self.config.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config.cache_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_jsonable(record), sort_keys=True) + "\n")

    def _write_usage_event(self, event: dict[str, Any]) -> None:
        if self.config.usage_ledger_path is None:
            return
        path = self.config.usage_ledger_path
        if path.exists():
            ledger = read_json(path)
        else:
            ledger = {
                "schema_version": "darjeeling.reference_usage_ledger.v1",
                "entries": [],
                "totals": {},
            }
        entries = ledger.setdefault("entries", [])
        entry = {
            "timestamp": utcnow(),
            "provider": self.config.provider,
            "model": self.config.model,
            "base_url_identity": stable_hash(self.base_url),
            **event,
        }
        entries.append(entry)
        totals = ledger.setdefault("totals", {})
        cost = float(event.get("cost_usd") or 0.0)
        totals["actual_paid_api_cost_usd"] = (
            float(totals.get("actual_paid_api_cost_usd", 0.0)) + cost
        )
        totals["provider_call_count"] = sum(
            1 for item in entries if item.get("status") == "provider_call"
        )
        totals["cache_hit_count"] = sum(
            1 for item in entries if item.get("status") == "cache_hit"
        )
        write_json_atomic(path, ledger)


def _read_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"reference provider environment variable is not set: {name}")
    return value


def _read_base_url_env(name: str) -> str:
    value = os.environ.get(name)
    if value:
        return value
    if name == "OPENAI_BASE_URL":
        return "https://api.openai.com/v1"
    raise ValueError(f"reference provider environment variable is not set: {name}")


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
        raise ValueError(f"{name} must be a non-negative number")
    return float(value)


def _optional_relative_path(config_path: Path, value: Any) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("reference config paths must be non-empty text")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def _usage_from_response(response: dict[str, Any]) -> dict[str, Any]:
    usage = response.get("usage", {})
    return dict(usage) if isinstance(usage, dict) else {}


def _provider_cost_from_response(response: dict[str, Any]) -> float | None:
    value = response.get("_darjeeling_provider_cost_usd")
    return float(value) if isinstance(value, int | float) else None


def _cost_from_usage(
    usage: dict[str, Any], price: dict[str, float], provider_cost: float | None
) -> tuple[float, str]:
    if provider_cost is not None:
        return provider_cost, "provider-reported"
    input_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    output_tokens = int(
        usage.get("completion_tokens") or usage.get("output_tokens") or 0
    )
    if input_tokens or output_tokens:
        missing = []
        if input_tokens and "input_per_million" not in price:
            missing.append("price.input_per_million")
        if output_tokens and "output_per_million" not in price:
            missing.append("price.output_per_million")
        if missing:
            raise ValueError(
                "reference response reported token usage but reference config "
                f"is missing token price fields: {', '.join(missing)}"
            )
        input_price = float(price.get("input_per_million", 0.0))
        output_price = float(price.get("output_per_million", 0.0))
        cost = (
            input_tokens * input_price + output_tokens * output_price
        ) / 1_000_000
        return cost, "estimated-from-token-usage"
    return 0.0, "not-available"


def _response_payload(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    parsed = {"content": content}
                return parsed if isinstance(parsed, dict) else {"content": parsed}
            if isinstance(content, list):
                text = "".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict)
                )
                return {"content": text}
    return response


def _finish_status(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        finish = choices[0].get("finish_reason")
        if isinstance(finish, str):
            return finish
    return "unknown"


def _cost_from_headers(headers: Any) -> float | None:
    for name in [
        "x-litellm-response-cost",
        "x-response-cost",
        "x-request-cost",
        "x-cost-usd",
    ]:
        value = headers.get(name)
        if value is None:
            continue
        try:
            return float(value)
        except ValueError:
            continue
    return None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_jsonable(child) for child in value]
    return value
