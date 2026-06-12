from __future__ import annotations

from darjeeling.targets.nlu.schemas import Frame


def frame_exact_match(predicted: Frame, expected: Frame) -> bool:
    return predicted == expected


def intent_matches(predicted: Frame, expected: Frame) -> bool:
    return predicted.intent == expected.intent
