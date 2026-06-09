from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from darjeeling.layers.l2_student import L2StudentConfig


@dataclass(frozen=True)
class L2Config:
    config_id: str
    intent_family: str = "sgd_logreg"
    slot_family: str = "token_sgd"
    guard_family: str = "logreg"


L2_CONFIG_PROPOSAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["slot_model_family"],
    "properties": {
        "intent_model_family": {"type": "string", "enum": ["sgd_logreg", "mlp"]},
        "slot_model_family": {"type": "string", "enum": ["token_sgd", "none"]},
        "min_examples": {"type": "integer", "minimum": 2},
        "max_features": {"type": "integer", "minimum": 100},
        "max_iter": {"type": "integer", "minimum": 10},
        "mlp_hidden_layer_sizes": {
            "type": "array",
            "items": {"type": "integer", "minimum": 1},
            "minItems": 1,
            "maxItems": 3,
        },
        "mlp_alpha": {"type": "number", "minimum": 0.0},
        "mlp_early_stopping": {"type": "boolean"},
        "word_ngram_range": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 2,
            "maxItems": 2,
        },
        "char_ngram_range": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 2,
            "maxItems": 2,
        },
    },
}


def l2_config_from_settings(settings: Any) -> L2StudentConfig:
    return L2StudentConfig(
        intent_model_family=settings.l2_intent_model_family,
        slot_model_family=settings.l2_slot_model_family,
        max_features=settings.l2_max_features,
        max_iter=settings.l2_max_iter,
        mlp_hidden_layer_sizes=settings.l2_mlp_hidden_layer_sizes,
        mlp_alpha=settings.l2_mlp_alpha,
        mlp_early_stopping=settings.l2_mlp_early_stopping,
    )


def l2_config_from_proposal(
    proposal: dict[str, Any],
    *,
    default: L2StudentConfig | None = None,
) -> L2StudentConfig:
    base = default or L2StudentConfig()
    allowed_fields = {
        "intent_model_family",
        "slot_model_family",
        "min_examples",
        "max_features",
        "max_iter",
        "mlp_hidden_layer_sizes",
        "mlp_alpha",
        "mlp_early_stopping",
        "word_ngram_range",
        "char_ngram_range",
    }
    payload = {
        field: proposal[field]
        for field in allowed_fields
        if field in proposal and proposal[field] is not None
    }
    for range_field in ["word_ngram_range", "char_ngram_range"]:
        if range_field in payload:
            payload[range_field] = _ngram_range(payload[range_field], field_name=range_field)
    if "mlp_hidden_layer_sizes" in payload:
        payload["mlp_hidden_layer_sizes"] = _positive_int_tuple(
            payload["mlp_hidden_layer_sizes"],
            field_name="mlp_hidden_layer_sizes",
        )
    config = L2StudentConfig(**{**base.model_dump(), **payload})
    if config.intent_model_family not in {"sgd_logreg", "mlp"}:
        raise ValueError(f"unsupported L2 intent_model_family: {config.intent_model_family}")
    if config.slot_model_family not in {"token_sgd", "none"}:
        raise ValueError(f"unsupported L2 slot_model_family: {config.slot_model_family}")
    return config


def _ngram_range(value: Any, *, field_name: str) -> tuple[int, int]:
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ValueError(f"{field_name} must be a two-item integer array")
    lower, upper = value
    if not isinstance(lower, int) or not isinstance(upper, int):
        raise ValueError(f"{field_name} must contain integers")
    if lower < 1 or upper < lower:
        raise ValueError(f"{field_name} must satisfy 1 <= lower <= upper")
    return (lower, upper)


def _positive_int_tuple(value: Any, *, field_name: str) -> tuple[int, ...]:
    if not isinstance(value, list | tuple) or not value:
        raise ValueError(f"{field_name} must be a non-empty integer array")
    if len(value) > 3:
        raise ValueError(f"{field_name} must contain at most three layers")
    layers = tuple(value)
    if any(not isinstance(layer, int) or isinstance(layer, bool) for layer in layers):
        raise ValueError(f"{field_name} must contain integers")
    if any(layer <= 0 for layer in layers):
        raise ValueError(f"{field_name} must contain positive integers")
    return layers
