from pathlib import Path

import numpy as np

from darjeeling.layers.l2_student import (
    ConstantGuard,
    FrameRetrievalResult,
    IntentCalibrationIndex,
    L2StudentBundle,
    L2StudentConfig,
    L2StudentLayer,
    L2TrainingExample,
    SlotPrediction,
    apply_slot_patterns,
    bio_tags_for_teacher_slots,
    filter_slots_for_intent,
    slot_patterns_by_intent_from_examples,
    slots_by_intent_from_examples,
    slots_from_bio_tags,
    train_guard,
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
    feature_union = bundle.intent_pipeline.named_steps["features"]
    assert [name for name, _transformer in feature_union.transformer_list] == ["word", "char"]


def test_l2_retrieval_frame_source_returns_nearest_teacher_frame() -> None:
    bundle = train_l2_student(
        _examples(),
        L2StudentConfig(
            accept_threshold=0.0,
            min_examples=4,
            frame_source="retrieval",
            slot_model_family="none",
        ),
    )

    prediction = bundle.predict("please alarm at nine tomorrow")

    assert prediction.frame == Frame(
        intent="alarm_set",
        slots={"time": "nine tomorrow"},
    )
    assert prediction.frame_source == "retrieval"
    assert prediction.retrieval_frame == prediction.frame
    assert prediction.retrieval_similarity > 0.5


def test_l2_retrieval_frame_source_requires_intent_agreement() -> None:
    class FakeIntentPipeline:
        classes_ = ["music_play", "alarm_set"]

        def predict_proba(self, utterances):
            return np.asarray([[0.95, 0.05] for _utterance in utterances])

    class DisagreeingFramePrototypeIndex:
        def nearest(self, intent_pipeline, utterance):
            return FrameRetrievalResult(
                frame=Frame(intent="alarm_set", slots={"time": "seven"}),
                nearest_similarity=0.91,
                margin=0.32,
            )

    bundle = L2StudentBundle(
        intent_pipeline=FakeIntentPipeline(),
        slot_tagger=None,
        guard_model=ConstantGuard(1.0),
        config=L2StudentConfig(accept_threshold=0.0, frame_source="retrieval"),
        frame_prototype_index=DisagreeingFramePrototypeIndex(),
    )

    prediction = bundle.predict("play something")

    assert prediction.frame == Frame(intent="music_play", slots={})
    assert prediction.frame_source == "student"
    assert prediction.retrieval_frame == Frame(
        intent="alarm_set",
        slots={"time": "seven"},
    )
    assert prediction.retrieval_intent_matches_student == 0.0


def test_l2_retrieval_does_not_return_exact_self_neighbor() -> None:
    bundle = train_l2_student(
        _examples(),
        L2StudentConfig(
            accept_threshold=0.0,
            min_examples=4,
            frame_source="retrieval",
            slot_model_family="none",
        ),
    )

    prediction = bundle.predict("alarm at nine tomorrow")

    assert prediction.frame != Frame(
        intent="alarm_set",
        slots={"time": "nine tomorrow"},
    )
    assert prediction.retrieval_frame == prediction.frame


def test_l2_student_trains_mlp_intent_family_and_reports_it() -> None:
    bundle = train_l2_student(
        _examples(),
        L2StudentConfig(
            accept_threshold=0.0,
            min_examples=4,
            intent_model_family="mlp",
            max_iter=300,
            mlp_hidden_layer_sizes=(8,),
        ),
    )
    layer = L2StudentLayer(bundle)

    result = layer.try_answer("play music")

    assert result.accepted
    assert result.metadata["intent_model"] == "mlp"
    assert result.metadata["predicted_frame"]["intent"] == "music_play"


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


def test_l2_filters_predicted_slots_by_intent_schema() -> None:
    slots_by_intent = slots_by_intent_from_examples(_examples())

    assert slots_by_intent["alarm_set"] == ("time",)
    assert filter_slots_for_intent(
        "music_play",
        {"time": "seven", "playlist_name": "jazz"},
        slots_by_intent,
    ) == {}


def test_l2_slot_patterns_fill_missing_schema_slots() -> None:
    examples = [
        L2TrainingExample(
            utterance="how old is carrie underwood",
            teacher_frame=Frame(intent="qa_factoid", slots={"person": "carrie underwood"}),
        ),
        L2TrainingExample(
            utterance="what does regal mean",
            teacher_frame=Frame(intent="qa_definition", slots={"definition_word": "regal"}),
        ),
    ]
    slots_by_intent = slots_by_intent_from_examples(examples)
    patterns = slot_patterns_by_intent_from_examples(examples)

    assert apply_slot_patterns(
        "qa_factoid",
        "how old is dolly parton",
        {},
        slots_by_intent,
        patterns,
    ) == {"person": "dolly parton"}
    assert apply_slot_patterns(
        "qa_definition",
        "what does hesitant mean",
        {},
        slots_by_intent,
        patterns,
    ) == {"definition_word": "hesitant"}


def test_l2_slot_patterns_fallback_to_suffix_context() -> None:
    examples = [
        L2TrainingExample(
            utterance="show grocery list",
            teacher_frame=Frame(intent="lists_query", slots={"list_name": "grocery"}),
        )
    ]
    slots_by_intent = slots_by_intent_from_examples(examples)
    patterns = slot_patterns_by_intent_from_examples(examples)

    assert apply_slot_patterns(
        "lists_query",
        "what's on my to do list",
        {},
        slots_by_intent,
        patterns,
    ) == {"list_name": "to do"}


def test_l2_bundle_drops_slots_not_seen_for_predicted_intent() -> None:
    class FakeIntentPipeline:
        classes_ = ["music_play", "alarm_set"]

        def predict_proba(self, utterances):
            return [[0.95, 0.05] for _utterance in utterances]

    class FakeSlotTagger:
        def predict(self, utterance: str) -> SlotPrediction:
            return SlotPrediction(slots={"time": "seven"})

    bundle = L2StudentBundle(
        intent_pipeline=FakeIntentPipeline(),
        slot_tagger=FakeSlotTagger(),
        guard_model=ConstantGuard(1.0),
        config=L2StudentConfig(accept_threshold=0.0),
        slots_by_intent={"music_play": (), "alarm_set": ("time",)},
    )

    prediction = bundle.predict("play music at seven")

    assert prediction.frame == Frame(intent="music_play", slots={})


def test_l2_guard_training_labels_postprocessed_frames() -> None:
    class FakeIntentPipeline:
        classes_ = ["qa_factoid"]

        def predict_proba(self, utterances):
            return [[1.0] for _utterance in utterances]

    class EmptySlotTagger:
        def predict(self, utterance: str) -> SlotPrediction:
            return SlotPrediction(slots={})

    examples = [
        L2TrainingExample(
            utterance="how old is carrie underwood",
            teacher_frame=Frame(intent="qa_factoid", slots={"person": "carrie underwood"}),
        )
    ]
    guard = train_guard(
        FakeIntentPipeline(),
        EmptySlotTagger(),
        examples,
        L2StudentConfig(),
        slots_by_intent=slots_by_intent_from_examples(examples),
        slot_patterns_by_intent=slot_patterns_by_intent_from_examples(examples),
    )

    assert isinstance(guard, ConstantGuard)
    assert guard.probability == 1.0


def test_intent_calibration_index_scores_predicted_intent_reliability() -> None:
    class FakeIntentPipeline:
        classes_ = ["calendar_query", "play_music"]

        def predict_proba(self, utterances):
            rows = []
            for utterance in utterances:
                if "calendar" in utterance:
                    rows.append([0.9, 0.1])
                else:
                    rows.append([0.1, 0.9])
            return np.asarray(rows)

    examples = [
        L2TrainingExample(
            utterance="calendar please",
            teacher_frame=Frame(intent="calendar_query"),
        ),
        L2TrainingExample(
            utterance="calendar list",
            teacher_frame=Frame(intent="calendar_query"),
        ),
        L2TrainingExample(
            utterance="play jazz",
            teacher_frame=Frame(intent="play_music", slots={"music_descriptor": "jazz"}),
        ),
        L2TrainingExample(
            utterance="play queen",
            teacher_frame=Frame(intent="play_music", slots={"artist_name": "queen"}),
        ),
    ]

    calibration = IntentCalibrationIndex.from_examples(
        FakeIntentPipeline(),
        slot_tagger=None,
        examples=examples,
        slots_by_intent=slots_by_intent_from_examples(examples),
    )
    calendar_score = calibration.score("calendar_query", {})
    music_score = calibration.score("play_music", {})

    assert calendar_score.predicted_intent_frame_accuracy == 1.0
    assert music_score.predicted_intent_frame_accuracy == 0.0
    assert music_score.predicted_intent_intent_accuracy == 1.0
    assert music_score.predicted_signature_frame_accuracy == 0.0


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


def test_l2_guard_probability_is_capped_by_signature_accuracy() -> None:
    class FakeIntentPipeline:
        classes_ = ["lists_query", "music_play"]

        def predict_proba(self, utterances):
            return np.asarray([[0.99, 0.01] for _utterance in utterances])

    calibration = IntentCalibrationIndex(
        predicted_intent_frame_accuracy={"lists_query": 0.72},
        predicted_intent_intent_accuracy={"lists_query": 1.0},
        predicted_intent_support={"lists_query": 0.10},
        predicted_intent_slotless_rate={"lists_query": 1.0},
        predicted_signature_frame_accuracy={("lists_query", ()): 0.72},
        predicted_signature_support={("lists_query", ()): 0.10},
    )
    bundle = L2StudentBundle(
        intent_pipeline=FakeIntentPipeline(),
        slot_tagger=None,
        guard_model=ConstantGuard(0.99),
        config=L2StudentConfig(accept_threshold=0.93),
        intent_calibration_index=calibration,
    )

    prediction = bundle.predict("what's on my to do list")

    assert prediction.raw_guard_probability == 0.99
    assert prediction.guard_calibration_cap == 0.72
    assert prediction.guard_probability == 0.72

    result = L2StudentLayer(bundle).try_answer("what's on my to do list")

    assert not result.accepted
    assert result.frame is None
    assert result.metadata["raw_guard_probability"] == 0.99
    assert result.metadata["guard_calibration_cap"] == 0.72


def test_l2_guard_signature_cap_ignores_moderately_reliable_signatures() -> None:
    class FakeIntentPipeline:
        classes_ = ["lists_query", "music_play"]

        def predict_proba(self, utterances):
            return np.asarray([[0.99, 0.01] for _utterance in utterances])

    calibration = IntentCalibrationIndex(
        predicted_intent_frame_accuracy={"lists_query": 0.80},
        predicted_intent_intent_accuracy={"lists_query": 1.0},
        predicted_intent_support={"lists_query": 0.10},
        predicted_intent_slotless_rate={"lists_query": 1.0},
        predicted_signature_frame_accuracy={("lists_query", ()): 0.80},
        predicted_signature_support={("lists_query", ()): 0.10},
    )
    bundle = L2StudentBundle(
        intent_pipeline=FakeIntentPipeline(),
        slot_tagger=None,
        guard_model=ConstantGuard(0.99),
        config=L2StudentConfig(accept_threshold=0.93),
        intent_calibration_index=calibration,
    )

    prediction = bundle.predict("what's on my list")

    assert prediction.guard_calibration_cap == 1.0
    assert prediction.guard_probability == 0.99


def test_l2_student_layer_reports_intent_support_metadata() -> None:
    bundle = train_l2_student(
        _examples(),
        L2StudentConfig(accept_threshold=0.0, min_examples=4),
    )
    layer = L2StudentLayer(bundle)

    result = layer.try_answer("wake me at eight")

    assert result.metadata["nearest_similarity"] > 0.0
    assert result.metadata["predicted_intent_similarity"] > 0.0
    assert -1.0 <= result.metadata["intent_support_margin"] <= 1.0
    assert "predicted_intent_frame_accuracy" in result.metadata
    assert "predicted_signature_frame_accuracy" in result.metadata
    assert result.metadata["frame_source"] in {"student", "retrieval"}
    assert "retrieval_similarity" in result.metadata


def test_l2_bundle_keeps_legacy_five_feature_guard_compatible() -> None:
    class FakeIntentPipeline:
        classes_ = ["music_play", "alarm_set"]

        def predict_proba(self, utterances):
            return [[0.95, 0.05] for _utterance in utterances]

    class LegacyFiveFeatureGuard:
        n_features_in_ = 5

        def predict_proba(self, features):
            assert features.shape == (1, 5)
            return np.asarray([[0.0, 1.0]])

    bundle = L2StudentBundle(
        intent_pipeline=FakeIntentPipeline(),
        slot_tagger=None,
        guard_model=LegacyFiveFeatureGuard(),
        config=L2StudentConfig(accept_threshold=0.0),
    )

    prediction = bundle.predict("play music")

    assert prediction.guard_probability == 1.0


def test_l2_student_layer_runtime_disabled_still_reports_prediction() -> None:
    bundle = train_l2_student(
        _examples(),
        L2StudentConfig(accept_threshold=0.0, min_examples=4, runtime_enabled=False),
    )
    layer = L2StudentLayer(bundle)

    result = layer.try_answer("play music")

    assert result.layer == "L2"
    assert not result.accepted
    assert result.frame is None
    assert result.reason == "runtime disabled"
    assert result.metadata["runtime_enabled"] is False
    assert result.metadata["predicted_frame"]["intent"] == "music_play"


def test_l2_student_bundle_round_trips_with_joblib(tmp_path: Path) -> None:
    bundle = train_l2_student(
        _examples(),
        L2StudentConfig(accept_threshold=0.0, min_examples=4),
    )
    path = tmp_path / "l2_student.joblib"

    bundle.save(path)
    loaded = L2StudentBundle.load(path)

    assert loaded.predict("play jazz").frame.intent == "music_play"
    assert loaded.predict("please alarm at nine tomorrow").frame.slots == {
        "time": "nine tomorrow"
    }
