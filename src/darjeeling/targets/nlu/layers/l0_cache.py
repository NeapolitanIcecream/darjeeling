from __future__ import annotations

from darjeeling.runtime.timing import elapsed_ms
from darjeeling.targets.nlu.data import normalize_utterance
from darjeeling.targets.nlu.schemas import Frame, LayerResult


class ExactCacheLayer:
    def __init__(self, cache: dict[str, Frame] | None = None) -> None:
        self.cache = cache or {}

    def add(self, utterance: str, frame: Frame) -> None:
        self.cache[normalize_utterance(utterance)] = frame

    def try_answer(self, utterance: str) -> LayerResult:
        with elapsed_ms() as ms:
            frame = self.cache.get(normalize_utterance(utterance))
            return LayerResult(
                layer="L0",
                accepted=frame is not None,
                frame=frame,
                confidence=1.0 if frame is not None else None,
                reason="exact cache hit" if frame is not None else "cache miss",
                latency_ms=ms(),
            )
