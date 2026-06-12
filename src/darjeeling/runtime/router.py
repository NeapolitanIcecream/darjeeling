from __future__ import annotations

from collections.abc import Sequence

from darjeeling.contracts import (
    JsonObject,
    LayerResult,
    RuntimeLayer,
)
from darjeeling.targets.nlu.schemas import Frame
from darjeeling.targets.nlu.schemas import LayerResult as FrameLayerResult


class CascadeRouter:
    def __init__(self, layers: Sequence[RuntimeLayer]) -> None:
        self.layers = list(layers)

    def route(self, input: JsonObject) -> tuple[JsonObject, list[LayerResult]]:
        results: list[LayerResult] = []
        for layer in self.layers:
            result = layer.try_answer(input)
            results.append(result)
            if result.accepted and result.output is not None:
                return result.output, results
        raise RuntimeError("cascade exhausted without an accepted output")


class FrameCascadeRouter:
    def __init__(self, layers: Sequence[object]) -> None:
        self.layers = list(layers)

    def route(self, utterance: str) -> tuple[Frame, list[FrameLayerResult]]:
        results: list[FrameLayerResult] = []
        for layer in self.layers:
            result = layer.try_answer(utterance)
            results.append(result)
            if result.accepted and result.frame is not None:
                return result.frame, results
        raise RuntimeError("cascade exhausted without an accepted frame")


JsonCascadeRouter = CascadeRouter
