from __future__ import annotations

from collections.abc import Sequence

from darjeeling.contracts import (
    JsonObject,
    LayerResult,
    RuntimeLayer,
)


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


JsonCascadeRouter = CascadeRouter
