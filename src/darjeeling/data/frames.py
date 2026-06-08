from __future__ import annotations

import re
from collections import defaultdict

from darjeeling.schemas import Frame

ANNOTATION_RE = re.compile(r"\[(?P<slot>[^:\]]+)\s*:\s*(?P<value>[^\]]+)\]")
SPACE_RE = re.compile(r"\s+")


def normalize_utterance(utterance: str) -> str:
    return SPACE_RE.sub(" ", utterance.strip().lower())


def strip_annotations(annotated_utterance: str) -> str:
    return normalize_utterance(
        ANNOTATION_RE.sub(lambda match: match.group("value").strip(), annotated_utterance)
    )


def normalized_template(annotated_utterance: str) -> str:
    return normalize_utterance(
        ANNOTATION_RE.sub(lambda match: f"[{match.group('slot').strip()}]", annotated_utterance)
    )


def frame_from_annotated_utterance(intent: str, annotated_utterance: str) -> Frame:
    slots: defaultdict[str, list[str]] = defaultdict(list)
    for match in ANNOTATION_RE.finditer(annotated_utterance):
        slot_name = match.group("slot").strip()
        slot_value = normalize_utterance(match.group("value"))
        if slot_name and slot_value:
            slots[slot_name].append(slot_value)

    flattened = {slot_name: " ; ".join(values) for slot_name, values in slots.items() if values}
    return Frame(intent=intent, slots=flattened)
