from pathlib import Path

from darjeeling.layers.l2_student import L2StudentConfig
from darjeeling.layers.l2_target import TargetL2Layer
from darjeeling.schemas import Frame


class _Prediction:
    frame = Frame(intent="email_query")
    guard_probability = 0.99
    top1_probability = 0.99
    margin = 0.9
    entropy = 0.1
    slot_avg_probability = 1.0
    slot_invalid_bio = False
    nearest_similarity = 1.0
    predicted_intent_similarity = 1.0
    intent_support_margin = 1.0
    predicted_slot_count = 0.0
    predicted_has_slots = 0.0
    predicted_intent_frame_accuracy = 1.0
    predicted_intent_intent_accuracy = 1.0
    predicted_intent_support = 4.0
    predicted_intent_slotless_rate = 1.0
    predicted_signature_frame_accuracy = 1.0
    predicted_signature_support = 4.0
    frame_source = "student"
    student_frame = Frame(intent="email_query")
    retrieval_frame = None
    retrieval_similarity = 0.0
    retrieval_margin = 0.0
    retrieval_intent_matches_student = 0.0

    def model_dump(self, *, mode: str) -> dict:
        assert mode == "json"
        return {
            "frame": self.frame.model_dump(mode="json"),
            "guard_probability": self.guard_probability,
            "top1_probability": self.top1_probability,
            "margin": self.margin,
            "entropy": self.entropy,
            "slot_avg_probability": self.slot_avg_probability,
            "slot_invalid_bio": self.slot_invalid_bio,
            "nearest_similarity": self.nearest_similarity,
            "predicted_intent_similarity": self.predicted_intent_similarity,
            "intent_support_margin": self.intent_support_margin,
            "predicted_slot_count": self.predicted_slot_count,
            "predicted_has_slots": self.predicted_has_slots,
            "predicted_intent_frame_accuracy": self.predicted_intent_frame_accuracy,
            "predicted_intent_intent_accuracy": self.predicted_intent_intent_accuracy,
            "predicted_intent_support": self.predicted_intent_support,
            "predicted_intent_slotless_rate": self.predicted_intent_slotless_rate,
            "predicted_signature_frame_accuracy": self.predicted_signature_frame_accuracy,
            "predicted_signature_support": self.predicted_signature_support,
            "frame_source": self.frame_source,
            "student_frame": self.student_frame.model_dump(mode="json"),
            "retrieval_frame": None,
            "retrieval_similarity": self.retrieval_similarity,
            "retrieval_margin": self.retrieval_margin,
            "retrieval_intent_matches_student": self.retrieval_intent_matches_student,
        }


class _Bundle:
    def __init__(self, *, accept_threshold: float = 0.5) -> None:
        self.config = L2StudentConfig(accept_threshold=accept_threshold)
        self.slot_tagger = object()

    def predict(self, utterance: str) -> _Prediction:
        del utterance
        return _Prediction()


def test_target_l2_layer_applies_postprocess_before_accept(tmp_path: Path) -> None:
    target_path = tmp_path / "target_l2.py"
    target_path.write_text(
        """
def postprocess_frame(utterance, frame, metadata):
    del metadata
    if " from " in f" {utterance} ":
        updated = dict(frame)
        updated["slots"] = {"person": utterance.rsplit(" from ", 1)[1]}
        return updated
    return frame
""",
        encoding="utf-8",
    )

    result = TargetL2Layer(_Bundle(), target_path).try_answer(
        "do i have emails from robert"
    )

    assert result.accepted is True
    assert result.frame == Frame(intent="email_query", slots={"person": "robert"})
    assert result.metadata["raw_predicted_frame"] == {
        "intent": "email_query",
        "slots": {},
        "is_abstain": False,
    }
    assert result.metadata["target_postprocessed"] is True


def test_target_l2_layer_veto_cannot_force_guard_reject(tmp_path: Path) -> None:
    target_path = tmp_path / "target_l2.py"
    target_path.write_text(
        """
def accept_prediction(utterance, frame, metadata, default_accept):
    del utterance, frame, metadata, default_accept
    return True
""",
        encoding="utf-8",
    )
    bundle = _Bundle(accept_threshold=1.0)

    result = TargetL2Layer(bundle, target_path).try_answer("do i have email")

    assert result.accepted is False
    assert result.reason == "guard rejected"
