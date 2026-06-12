from darjeeling.data.frames import (
    frame_from_annotated_utterance,
    normalized_template,
    strip_annotations,
)


def test_frame_parser_extracts_slots_from_bracket_annotation() -> None:
    annotated = "alpha request [slot_alpha : value alpha extended]"

    frame = frame_from_annotated_utterance("intent_alpha", annotated)

    assert frame.intent == "intent_alpha"
    assert frame.slots == {"slot_alpha": "value alpha extended"}
    assert strip_annotations(annotated) == "alpha request value alpha extended"
    assert normalized_template(annotated) == "alpha request [slot_alpha]"
