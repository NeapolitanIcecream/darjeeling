from __future__ import annotations

from typing import Protocol

from darjeeling.schemas import LayerResult


class RuntimeLayer(Protocol):
    def try_answer(self, utterance: str) -> LayerResult:
        raise NotImplementedError
