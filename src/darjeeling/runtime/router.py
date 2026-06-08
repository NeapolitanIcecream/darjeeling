from __future__ import annotations

from collections.abc import Sequence

from darjeeling.layers.base import RuntimeLayer
from darjeeling.schemas import Frame, LayerResult


class CascadeRouter:
    def __init__(self, layers: Sequence[RuntimeLayer]) -> None:
        self.layers = list(layers)

    def route(self, utterance: str) -> tuple[Frame, list[LayerResult]]:
        results: list[LayerResult] = []
        for layer in self.layers:
            result = layer.try_answer(utterance)
            results.append(result)
            if result.accepted and result.frame is not None:
                return result.frame, results
        raise RuntimeError("cascade exhausted without an accepted frame")
