from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from darjeeling.contracts import LayerResult as CoreLayerResult
from darjeeling.targets.nlu.schemas import Frame, FramePatch, LayerName, LayerResult

FIELD_INTENT = "intent"


@dataclass
class FrameComposer:
    accepted_intent: str | None = None
    accepted_slots: dict[str, str] = field(default_factory=dict)
    field_sources: dict[str, LayerName] = field(default_factory=dict)
    complete: bool = False
    complete_source_layer: LayerName | None = None

    def apply_patch(self, patch: FramePatch) -> None:
        if patch.accepted_intent is not None and self.accepted_intent is None:
            self.accepted_intent = patch.accepted_intent
            self.field_sources[FIELD_INTENT] = patch.source_layer
        for slot_key, slot_value in patch.accepted_slots.items():
            field_key = slot_field_key(slot_key)
            if slot_key not in self.accepted_slots:
                self.accepted_slots[slot_key] = slot_value
                self.field_sources[field_key] = patch.source_layer
        if patch.complete and self.accepted_intent is not None:
            self.complete = True
            self.complete_source_layer = patch.source_layer

    def fill_missing_from_frame(
        self,
        frame: Frame,
        *,
        source_layer: LayerName,
        confidence: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FramePatch:
        accepted_slots = {
            slot_key: slot_value
            for slot_key, slot_value in frame.slots.items()
            if slot_key not in self.accepted_slots
        }
        patch = FramePatch(
            accepted_intent=frame.intent if self.accepted_intent is None else None,
            accepted_slots=accepted_slots,
            source_layer=source_layer,
            confidence=confidence,
            complete=True,
            metadata=metadata or {},
        )
        self.apply_patch(patch)
        self.complete = True
        self.complete_source_layer = source_layer
        return patch

    def to_frame(self) -> Frame:
        if self.accepted_intent is None:
            raise ValueError("cannot compose NLU frame without an accepted intent")
        return Frame(intent=self.accepted_intent, slots=dict(self.accepted_slots))


@dataclass(frozen=True)
class NluRouteResult:
    final_frame: Frame
    chosen_layer: LayerName
    layer_results: list[LayerResult]
    l4_usage: dict[str, Any]
    composer: FrameComposer


def route_nlu_layers(runtime_layers: dict[str, Any], *, utterance: str) -> NluRouteResult:
    composer = FrameComposer()
    layer_results: list[LayerResult] = []
    l4_usage: dict[str, Any] = {}
    for layer_name in ("L0", "L1", "L2", "L3", "L4"):
        layer = runtime_layers.get(layer_name)
        if layer is None:
            continue
        core_result = layer.try_answer({"utterance": utterance})
        result = legacy_layer_result_from_core(core_result)
        patch = frame_patch_from_layer_result(result)
        if layer_name == "L4" and result.frame is not None:
            patch = composer.fill_missing_from_frame(
                result.frame,
                source_layer="L4",
                confidence=result.confidence,
                metadata={
                    "adapter": "l4_residual_fill",
                    **(patch.metadata if patch is not None else {}),
                },
            )
        elif patch is not None:
            composer.apply_patch(patch)
        result = result.model_copy(update={"patch": patch})
        result.metadata.update(field_metadata_for_patch(patch))
        layer_results.append(result)
        if layer_name == "L4" and result.metadata:
            usage = result.metadata.get("usage")
            if isinstance(usage, dict):
                l4_usage = usage
        if patch is not None and patch.complete and composer.complete:
            return NluRouteResult(
                final_frame=composer.to_frame(),
                chosen_layer=patch.source_layer,
                layer_results=layer_results,
                l4_usage=l4_usage,
                composer=composer,
            )
    raise ValueError("NLU route did not produce a complete frame")


def frame_patch_from_layer_result(result: LayerResult) -> FramePatch | None:
    if result.patch is not None:
        return result.patch
    metadata_patch = result.metadata.get("frame_patch")
    if isinstance(metadata_patch, dict):
        return FramePatch.model_validate(metadata_patch)
    if not result.accepted or result.frame is None:
        return None
    return frame_patch_from_frame(
        result.frame,
        source_layer=result.layer,
        confidence=result.confidence,
        metadata={"adapter": "legacy_full_frame"},
    )


def frame_patch_from_frame(
    frame: Frame,
    *,
    source_layer: LayerName,
    confidence: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> FramePatch:
    return FramePatch(
        accepted_intent=frame.intent,
        accepted_slots=dict(frame.slots),
        source_layer=source_layer,
        confidence=confidence,
        complete=True,
        metadata=metadata or {},
    )


def legacy_layer_result_from_core(result: CoreLayerResult) -> LayerResult:
    patch = _patch_from_core_metadata(result)
    return LayerResult(
        layer=result.layer,
        accepted=result.accepted,
        frame=Frame.model_validate(result.output) if result.output is not None else None,
        patch=patch,
        confidence=result.confidence,
        reason=result.reason,
        latency_ms=result.latency_ms,
        cost_usd=result.cost_usd,
        metadata=result.metadata,
    )


def core_layer_result_from_legacy(result: LayerResult) -> CoreLayerResult:
    metadata = dict(result.metadata)
    if result.patch is not None:
        metadata["frame_patch"] = result.patch.model_dump(mode="json")
    return CoreLayerResult(
        layer=result.layer,
        accepted=result.accepted,
        output=result.frame.model_dump(mode="json") if result.frame is not None else None,
        confidence=result.confidence,
        reason=result.reason,
        latency_ms=result.latency_ms,
        cost_usd=result.cost_usd,
        metadata=metadata,
    )


def accepted_field_keys(patch: FramePatch | None) -> set[str]:
    if patch is None:
        return set()
    fields = set()
    if patch.accepted_intent is not None:
        fields.add(FIELD_INTENT)
    fields.update(slot_field_key(slot_key) for slot_key in patch.accepted_slots)
    return fields


def frame_field_values(frame: Frame) -> dict[str, str]:
    values = {FIELD_INTENT: frame.intent}
    values.update(
        {
            slot_field_key(slot_key): slot_value
            for slot_key, slot_value in frame.slots.items()
        }
    )
    return values


def slot_field_key(slot_key: str) -> str:
    return f"slots.{slot_key}"


def field_metadata_for_patch(patch: FramePatch | None) -> dict[str, Any]:
    if patch is None:
        return {
            "patch_accepted_fields": [],
            "patch_complete": False,
        }
    return {
        "patch_accepted_fields": sorted(accepted_field_keys(patch)),
        "patch_complete": patch.complete,
        "patch_source_layer": patch.source_layer,
    }


def _patch_from_core_metadata(result: CoreLayerResult) -> FramePatch | None:
    metadata_patch = result.metadata.get("frame_patch")
    if isinstance(metadata_patch, dict):
        return FramePatch.model_validate(metadata_patch)
    return None
