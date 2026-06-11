from darjeeling.data.frames import (
    frame_from_annotated_utterance,
    normalized_template,
    strip_annotations,
)


def test_frame_parser_extracts_slots_from_massive_style_annotation() -> None:
    annotated = "alpha request for [time : seven tomorrow morning]"

    frame = frame_from_annotated_utterance("intent_alpha", annotated)

    assert frame.intent == "intent_alpha"
    assert frame.slots == {"time": "seven tomorrow morning"}
    assert strip_annotations(annotated) == "alpha request for seven tomorrow morning"
    assert normalized_template(annotated) == "alpha request for [time]"
