from pathlib import Path

from darjeeling.layers.l2_student import (
    L2StudentBundle,
    L2StudentConfig,
    L2StudentLayer,
    L2TrainingExample,
    bio_tags_for_teacher_slots,
    slots_from_bio_tags,
    train_l2_student,
    train_slot_tagger,
)
from darjeeling.schemas import Frame


def _examples() -> list[L2TrainingExample]:
    return [
        L2TrainingExample(
            utterance="set an alarm for seven",
            teacher_frame=Frame(intent="alarm_set", slots={"time": "seven"}),
        ),
        L2TrainingExample(
            utterance="wake me at eight",
            teacher_frame=Frame(intent="alarm_set", slots={"time": "eight"}),
        ),
        L2TrainingExample(
            utterance="alarm at nine tomorrow",
            teacher_frame=Frame(intent="alarm_set", slots={"time": "nine tomorrow"}),
        ),
        L2TrainingExample(
            utterance="play some jazz",
            teacher_frame=Frame(intent="music_play"),
        ),
        L2TrainingExample(
            utterance="play music",
            teacher_frame=Frame(intent="music_play"),
        ),
        L2TrainingExample(
            utterance="start my playlist",
            teacher_frame=Frame(intent="music_play"),
        ),
    ]


def test_l2_student_trains_from_teacher_frames_and_predicts_intent() -> None:
    bundle = train_l2_student(
        _examples(),
        L2StudentConfig(accept_threshold=0.0, min_examples=4),
    )

    prediction = bundle.predict("set alarm at six")

    assert prediction.frame.intent == "alarm_set"
    assert 0.0 <= prediction.guard_probability <= 1.0


def test_token_slot_tagger_learns_teacher_slot_spans() -> None:
    tagger = train_slot_tagger(
        _examples(),
        L2StudentConfig(accept_threshold=0.0, min_examples=4),
    )

    assert tagger is not None
    prediction = tagger.predict("alarm at nine tomorrow")

    assert prediction.model_name == "token_sgd"
    assert prediction.slots == {"time": "nine tomorrow"}
    assert not prediction.invalid_bio
    assert 0.0 <= prediction.avg_probability <= 1.0


def test_slot_alignment_and_bio_reconstruction() -> None:
    tokens = ["alarm", "at", "nine", "tomorrow"]

    tags = bio_tags_for_teacher_slots(tokens, {"time": "nine tomorrow"})
    slots, invalid_bio = slots_from_bio_tags(tokens, tags)
    duplicate_slots, duplicate_invalid_bio = slots_from_bio_tags(
        tokens,
        ["B-time", "B-time", "O", "O"],
    )

    assert tags == ["O", "O", "B-time", "I-time"]
    assert slots == {"time": "nine tomorrow"}
    assert not invalid_bio
    assert duplicate_slots == {"time": "alarm"}
    assert duplicate_invalid_bio


def test_l2_student_layer_uses_guard_threshold() -> None:
    bundle = train_l2_student(
        _examples(),
        L2StudentConfig(accept_threshold=0.0, min_examples=4),
    )
    layer = L2StudentLayer(bundle)

    result = layer.try_answer("play music")

    assert result.layer == "L2"
    assert result.accepted
    assert result.frame is not None
    assert result.metadata["slot_model"] == "token_sgd"
    assert "guard_probability" in result.metadata


def test_l2_student_bundle_round_trips_with_joblib(tmp_path: Path) -> None:
    bundle = train_l2_student(
        _examples(),
        L2StudentConfig(accept_threshold=0.0, min_examples=4),
    )
    path = tmp_path / "l2_student.joblib"

    bundle.save(path)
    loaded = L2StudentBundle.load(path)

    assert loaded.predict("play jazz").frame.intent == "music_play"
    assert loaded.predict("alarm at nine tomorrow").frame.slots == {"time": "nine tomorrow"}
