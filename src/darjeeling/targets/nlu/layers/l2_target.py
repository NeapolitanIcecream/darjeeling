from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

from darjeeling.layers.l2_student import L2StudentBundle, guard_accepts
from darjeeling.runtime.timing import elapsed_ms
from darjeeling.schemas import LayerResult
from darjeeling.targets.nlu.schemas import Frame


class TargetL2Layer:
    """L2 runtime wrapper for NLU target-owned postprocess/veto code."""

    def __init__(self, bundle: L2StudentBundle, target_module_path: Path) -> None:
        self.bundle = bundle
        self.target_module_path = target_module_path
        self.target_module = load_target_module(target_module_path)

    def try_answer(self, utterance: str) -> LayerResult:
        with elapsed_ms() as ms:
            prediction = self.bundle.predict(utterance)
            metadata = prediction.model_dump(mode="json")
            raw_frame = prediction.frame
            frame = target_postprocess_frame(
                self.target_module,
                utterance=utterance,
                frame=raw_frame,
                metadata=metadata,
            )
            runtime_enabled = getattr(self.bundle.config, "runtime_enabled", True)
            default_accept = runtime_enabled and guard_accepts(
                prediction.guard_probability,
                self.bundle.config.accept_threshold,
            )
            accepted = target_accept_prediction(
                self.target_module,
                utterance=utterance,
                frame=frame,
                metadata=metadata,
                default_accept=default_accept,
            )
            return LayerResult(
                layer="L2",
                accepted=accepted,
                frame=frame if accepted else None,
                confidence=prediction.guard_probability,
                reason=_target_l2_reason(runtime_enabled, default_accept, accepted),
                latency_ms=ms(),
                metadata={
                    **metadata,
                    "raw_predicted_frame": raw_frame.model_dump(mode="json"),
                    "predicted_frame": frame.model_dump(mode="json"),
                    "target_module": str(self.target_module_path),
                    "target_postprocessed": frame != raw_frame,
                    "target_vetoed": bool(default_accept and not accepted),
                    "accept_threshold": self.bundle.config.accept_threshold,
                    "runtime_enabled": runtime_enabled,
                    "frame_source_config": self.bundle.config.frame_source,
                    "intent_model": self.bundle.config.intent_model_family,
                    "slot_model": "token_sgd" if self.bundle.slot_tagger else "none",
                },
            )


def load_target_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("target_l2_runtime", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import target module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def target_config_overrides(module: ModuleType) -> dict[str, Any]:
    function = getattr(module, "config_overrides", None)
    if function is None:
        return {}
    value = function()
    if not isinstance(value, dict):
        raise TypeError("target_l2.config_overrides() must return a dict")
    return value


def target_postprocess_frame(
    module: ModuleType,
    *,
    utterance: str,
    frame: Frame,
    metadata: dict[str, Any],
) -> Frame:
    function = getattr(module, "postprocess_frame", None)
    if function is None:
        return frame
    value = function(utterance, frame.model_dump(mode="json"), metadata)
    return Frame.model_validate(value)


def target_accept_prediction(
    module: ModuleType,
    *,
    utterance: str,
    frame: Frame,
    metadata: dict[str, Any],
    default_accept: bool,
) -> bool:
    function = getattr(module, "accept_prediction", None)
    if function is None:
        return default_accept
    value = function(
        utterance,
        frame.model_dump(mode="json"),
        metadata,
        default_accept,
    )
    if value is None:
        return default_accept
    if not isinstance(value, bool):
        raise TypeError("target_l2.accept_prediction() must return bool or None")
    if not value:
        return False
    return default_accept


def _target_l2_reason(runtime_enabled: bool, default_accept: bool, accepted: bool) -> str:
    if not runtime_enabled:
        return "runtime disabled"
    if default_accept and not accepted:
        return "target vetoed"
    return "guard accepted" if accepted else "guard rejected"
