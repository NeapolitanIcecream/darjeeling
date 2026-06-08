from darjeeling.data.frames import (
    frame_from_annotated_utterance,
    normalized_template,
    strip_annotations,
)


def test_frame_parser_extracts_slots_from_massive_style_annotation() -> None:
    annotated = "set an alarm for [time : seven tomorrow morning]"

    frame = frame_from_annotated_utterance("alarm_set", annotated)

    assert frame.intent == "alarm_set"
    assert frame.slots == {"time": "seven tomorrow morning"}
    assert strip_annotations(annotated) == "set an alarm for seven tomorrow morning"
    assert normalized_template(annotated) == "set an alarm for [time]"
