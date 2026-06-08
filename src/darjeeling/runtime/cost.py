from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


def estimate_l4_cost_usd(
    input_tokens: int,
    output_tokens: int,
    input_usd_per_million: float,
    output_usd_per_million: float,
) -> float:
    return (input_tokens / 1_000_000.0) * input_usd_per_million + (
        output_tokens / 1_000_000.0
    ) * output_usd_per_million


@dataclass(frozen=True)
class ReplayCostModel:
    l0_cost_usd_per_request: float = 0.0
    l1_cost_usd_per_request: float = 0.0
    l2_cost_usd_per_request: float = 0.00005
    l3_cost_usd_per_request: float = 0.0
    l4_default_cost_usd_per_request: float = 0.01
    l4_input_usd_per_million: float = 0.40
    l4_cached_input_usd_per_million: float = 0.10
    l4_output_usd_per_million: float = 1.60

    def layer_cost_usd(self, layer: str, usage: Mapping[str, Any] | None = None) -> float:
        if layer == "L0":
            return self.l0_cost_usd_per_request
        if layer == "L1":
            return self.l1_cost_usd_per_request
        if layer == "L2":
            return self.l2_cost_usd_per_request
        if layer == "L3":
            return self.l3_cost_usd_per_request
        if layer == "L4":
            return estimate_l4_cost_from_usage(
                usage or {},
                input_usd_per_million=self.l4_input_usd_per_million,
                cached_input_usd_per_million=self.l4_cached_input_usd_per_million,
                output_usd_per_million=self.l4_output_usd_per_million,
                default_cost_usd=self.l4_default_cost_usd_per_request,
            )
        return 0.0


def replay_cost_model_from_settings(settings: Any) -> ReplayCostModel:
    return ReplayCostModel(
        l0_cost_usd_per_request=float(settings.l0_cost_usd_per_request),
        l1_cost_usd_per_request=float(settings.l1_cost_usd_per_request),
        l2_cost_usd_per_request=float(settings.l2_cost_usd_per_request),
        l3_cost_usd_per_request=float(settings.l3_cost_usd_per_request),
        l4_default_cost_usd_per_request=float(settings.l4_default_cost_usd_per_request),
        l4_input_usd_per_million=float(settings.l4_input_usd_per_million),
        l4_cached_input_usd_per_million=float(settings.l4_cached_input_usd_per_million),
        l4_output_usd_per_million=float(settings.l4_output_usd_per_million),
    )


def estimate_l4_cost_from_usage(
    usage: Mapping[str, Any],
    *,
    input_usd_per_million: float,
    cached_input_usd_per_million: float,
    output_usd_per_million: float,
    default_cost_usd: float = 0.0,
) -> float:
    input_tokens = _token_count(usage, "prompt_tokens", "input_tokens")
    output_tokens = _token_count(usage, "completion_tokens", "output_tokens")
    cached_tokens = min(_cached_input_tokens(usage), input_tokens)
    if input_tokens <= 0 and output_tokens <= 0:
        return default_cost_usd
    uncached_input_tokens = max(0, input_tokens - cached_tokens)
    return (
        estimate_l4_cost_usd(
            input_tokens=uncached_input_tokens,
            output_tokens=output_tokens,
            input_usd_per_million=input_usd_per_million,
            output_usd_per_million=output_usd_per_million,
        )
        + (cached_tokens / 1_000_000.0) * cached_input_usd_per_million
    )


def _token_count(usage: Mapping[str, Any], *keys: str) -> int:
    for key in keys:
        value = usage.get(key)
        if isinstance(value, int | float):
            return max(0, int(value))
    return 0


def _cached_input_tokens(usage: Mapping[str, Any]) -> int:
    for detail_key in ("prompt_tokens_details", "input_tokens_details"):
        details = usage.get(detail_key)
        if isinstance(details, Mapping):
            value = details.get("cached_tokens")
            if isinstance(value, int | float):
                return max(0, int(value))
    return 0
