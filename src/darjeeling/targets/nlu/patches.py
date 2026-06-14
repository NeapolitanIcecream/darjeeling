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
    field_conflicts: list[dict[str, Any]] = field(default_factory=list)
    field_overrides: list[dict[str, Any]] = field(default_factory=list)
    verified_fields: list[str] = field(default_factory=list)
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

    def apply_l4_patch(self, patch: FramePatch) -> FramePatch:
        metadata = dict(patch.metadata)
        conflicts: list[dict[str, Any]] = []
        overrides: list[dict[str, Any]] = []
        verified_fields = set(str(field) for field in metadata.get("verified_fields", []))
        removed_fields = [
            str(field)
            for field in metadata.get("removed_fields", [])
            if isinstance(field, str)
        ]

        if patch.accepted_intent is not None:
            if self.accepted_intent is None:
                self.accepted_intent = patch.accepted_intent
                self.field_sources[FIELD_INTENT] = patch.source_layer
            elif self.accepted_intent != patch.accepted_intent:
                conflict = self._field_conflict(
                    FIELD_INTENT,
                    old_value=self.accepted_intent,
                    new_value=patch.accepted_intent,
                    source_layer=patch.source_layer,
                )
                conflicts.append(conflict)
                overrides.append(conflict)
                self.accepted_intent = patch.accepted_intent
                self.field_sources[FIELD_INTENT] = patch.source_layer
            else:
                verified_fields.add(FIELD_INTENT)

        for slot_key, slot_value in patch.accepted_slots.items():
            field_key = slot_field_key(slot_key)
            if slot_key not in self.accepted_slots:
                self.accepted_slots[slot_key] = slot_value
                self.field_sources[field_key] = patch.source_layer
            elif self.accepted_slots[slot_key] != slot_value:
                conflict = self._field_conflict(
                    field_key,
                    old_value=self.accepted_slots[slot_key],
                    new_value=slot_value,
                    source_layer=patch.source_layer,
                )
                conflicts.append(conflict)
                overrides.append(conflict)
                self.accepted_slots[slot_key] = slot_value
                self.field_sources[field_key] = patch.source_layer
            else:
                verified_fields.add(field_key)

        for field_key in removed_fields:
            if not field_key.startswith("slots."):
                continue
            slot_key = field_key.removeprefix("slots.")
            if slot_key not in self.accepted_slots:
                continue
            conflict = self._field_conflict(
                field_key,
                old_value=self.accepted_slots[slot_key],
                new_value=None,
                source_layer=patch.source_layer,
            )
            conflicts.append(conflict)
            overrides.append(conflict)
            self.accepted_slots.pop(slot_key, None)
            self.field_sources.pop(field_key, None)

        if patch.complete and self.accepted_intent is not None:
            self.complete = True
            self.complete_source_layer = patch.source_layer

        self.field_conflicts.extend(conflicts)
        self.field_overrides.extend(overrides)
        self.verified_fields = sorted(set(self.verified_fields) | verified_fields)
        applied_metadata = {
            **metadata,
            "field_conflicts": conflicts,
            "field_overrides": overrides,
            "verified_fields": sorted(verified_fields),
            "removed_fields": removed_fields,
        }
        return patch.model_copy(update={"metadata": applied_metadata})

    def fill_or_override_from_l4_frame(
        self,
        frame: Frame,
        *,
        source_layer: LayerName,
        confidence: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FramePatch:
        patch_metadata = dict(metadata or {})
        accepted_intent = (
            frame.intent
            if self.accepted_intent is None or self.accepted_intent != frame.intent
            else None
        )
        accepted_slots = {
            slot_key: slot_value
            for slot_key, slot_value in frame.slots.items()
            if self.accepted_slots.get(slot_key) != slot_value
        }
        removed_fields = [
            slot_field_key(slot_key)
            for slot_key in self.accepted_slots
            if slot_key not in frame.slots
        ]
        verified_fields: list[str] = []
        if self.accepted_intent == frame.intent:
            verified_fields.append(FIELD_INTENT)
        verified_fields.extend(
            slot_field_key(slot_key)
            for slot_key, slot_value in frame.slots.items()
            if self.accepted_slots.get(slot_key) == slot_value
        )
        patch = FramePatch(
            accepted_intent=accepted_intent,
            accepted_slots=accepted_slots,
            source_layer=source_layer,
            confidence=confidence,
            complete=True,
            metadata={
                **patch_metadata,
                "removed_fields": removed_fields,
                "verified_fields": sorted(verified_fields),
            },
        )
        return self.apply_l4_patch(patch)

    def to_frame(self) -> Frame:
        if self.accepted_intent is None:
            raise ValueError("cannot compose NLU frame without an accepted intent")
        return Frame(intent=self.accepted_intent, slots=dict(self.accepted_slots))

    def field_values(self) -> dict[str, str]:
        if self.accepted_intent is None:
            values: dict[str, str] = {}
        else:
            values = {FIELD_INTENT: self.accepted_intent}
        values.update(
            {
                slot_field_key(slot_key): slot_value
                for slot_key, slot_value in self.accepted_slots.items()
            }
        )
        return values

    def accepted_field_keys(self) -> set[str]:
        return set(self.field_values())

    def missing_field_keys(self, candidate_field_keys: list[str]) -> list[str]:
        accepted = self.accepted_field_keys()
        return [field_key for field_key in candidate_field_keys if field_key not in accepted]

    def _field_conflict(
        self,
        field_key: str,
        *,
        old_value: str | None,
        new_value: str | None,
        source_layer: LayerName,
    ) -> dict[str, Any]:
        return {
            "field": field_key,
            "old_value": old_value,
            "new_value": new_value,
            "old_source": self.field_sources.get(field_key),
            "new_source": source_layer,
        }


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
        if layer_name == "L4":
            residual_result = _try_l4_residual(layer, utterance=utterance, composer=composer)
            if residual_result is not None:
                result = _apply_route_result(composer, residual_result, l4_override=True)
                layer_results.append(result)
                _record_l4_usage(l4_usage, result)
                if result.patch is not None and result.patch.complete and composer.complete:
                    return NluRouteResult(
                        final_frame=composer.to_frame(),
                        chosen_layer="L4",
                        layer_results=layer_results,
                        l4_usage=l4_usage,
                        composer=composer,
                    )
        core_result = layer.try_answer({"utterance": utterance})
        result = legacy_layer_result_from_core(core_result)
        if layer_name == "L4" and result.frame is not None:
            patch = composer.fill_or_override_from_l4_frame(
                result.frame,
                source_layer="L4",
                confidence=result.confidence,
                metadata={
                    "adapter": "l4_full_frame",
                    **result.metadata,
                },
            )
            result = result.model_copy(update={"patch": patch})
            result.metadata.update(patch.metadata)
            result.metadata["frame_patch"] = patch.model_dump(mode="json")
            result.metadata.update(field_metadata_for_patch(patch))
        else:
            result = _apply_route_result(
                composer,
                result,
                l4_override=layer_name == "L4",
            )
            patch = result.patch
        layer_results.append(result)
        if layer_name == "L4":
            _record_l4_usage(l4_usage, result)
        if patch is not None and patch.complete and composer.complete:
            return NluRouteResult(
                final_frame=composer.to_frame(),
                chosen_layer=patch.source_layer,
                layer_results=layer_results,
                l4_usage=l4_usage,
                composer=composer,
            )
    raise ValueError("NLU route did not produce a complete frame")


def _apply_route_result(
    composer: FrameComposer,
    result: LayerResult,
    *,
    l4_override: bool,
) -> LayerResult:
    patch = frame_patch_from_layer_result(result)
    if patch is not None:
        patch = composer.apply_l4_patch(patch) if l4_override else _apply_weak_patch(
            composer,
            patch,
        )
    updated = result.model_copy(update={"patch": patch})
    if patch is not None:
        updated.metadata.update(patch.metadata)
        updated.metadata["frame_patch"] = patch.model_dump(mode="json")
    updated.metadata.update(field_metadata_for_patch(patch))
    return updated


def _apply_weak_patch(composer: FrameComposer, patch: FramePatch) -> FramePatch:
    composer.apply_patch(patch)
    return patch


def _try_l4_residual(
    layer: Any,
    *,
    utterance: str,
    composer: FrameComposer,
) -> LayerResult | None:
    if not composer.accepted_field_keys():
        return None
    residual = getattr(layer, "try_residual_patch", None)
    if residual is None:
        return None
    candidate_fields = _residual_field_keys(layer)
    try:
        core_result = residual(
            {
                "utterance": utterance,
                "accepted_fields": composer.field_values(),
                "missing_fields": composer.missing_field_keys(candidate_fields),
            }
        )
    except Exception:
        return None
    return legacy_layer_result_from_core(core_result)


def _residual_field_keys(layer: Any) -> list[str]:
    field_keys = getattr(layer, "residual_field_keys", None)
    if field_keys is None:
        return []
    return [str(field_key) for field_key in field_keys()]


def _record_l4_usage(l4_usage: dict[str, Any], result: LayerResult) -> None:
    call_kind = str(result.metadata.get("l4_call_kind", "full"))
    bucket = "serving_residual" if call_kind == "residual" else "serving_full"
    usage = result.metadata.get("usage")
    if isinstance(usage, dict) and call_kind != "residual":
        l4_usage.update(usage)
    l4_usage[f"{bucket}_calls"] = int(l4_usage.get(f"{bucket}_calls", 0)) + 1
    l4_usage[f"{bucket}_cost_usd"] = (
        float(l4_usage.get(f"{bucket}_cost_usd", 0.0)) + result.cost_usd
    )
    l4_usage[f"{bucket}_latency_ms"] = (
        float(l4_usage.get(f"{bucket}_latency_ms", 0.0)) + result.latency_ms
    )
    if isinstance(usage, dict):
        l4_usage[f"{bucket}_tokens"] = int(l4_usage.get(f"{bucket}_tokens", 0)) + _usage_tokens(
            usage
        )
    fields_avoided = result.metadata.get("fields_avoided")
    if isinstance(fields_avoided, int | float):
        l4_usage["serving_fields_avoided"] = (
            float(l4_usage.get("serving_fields_avoided", 0.0)) + float(fields_avoided)
        )


def _usage_tokens(usage: dict[str, Any]) -> int:
    total = usage.get("total_tokens")
    if isinstance(total, int | float) and not isinstance(total, bool):
        return int(total)
    prompt = usage.get("prompt_tokens", usage.get("input_tokens", 0))
    completion = usage.get("completion_tokens", usage.get("output_tokens", 0))
    tokens = 0
    for value in (prompt, completion):
        if isinstance(value, int | float) and not isinstance(value, bool):
            tokens += int(value)
    return tokens


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
