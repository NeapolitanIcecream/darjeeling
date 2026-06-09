from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from pydantic import BaseModel
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import normalize

from darjeeling.runtime.timing import elapsed_ms
from darjeeling.schemas import Frame, LayerResult, TeacherTrace


@dataclass(frozen=True)
class GuardDecision:
    probability: float
    threshold: float

    @property
    def accepted(self) -> bool:
        return self.probability >= self.threshold


class L2StudentConfig(BaseModel):
    accept_threshold: float = 0.93
    runtime_enabled: bool = True
    random_state: int = 17
    min_examples: int = 4
    intent_model_family: str = "sgd_logreg"
    word_ngram_range: tuple[int, int] = (1, 2)
    char_ngram_range: tuple[int, int] = (3, 5)
    max_features: int = 50_000
    max_iter: int = 1000
    mlp_hidden_layer_sizes: tuple[int, ...] = (64,)
    mlp_alpha: float = 0.0001
    mlp_early_stopping: bool = False
    slot_model_family: str = "token_sgd"


class L2TrainingExample(BaseModel):
    utterance: str
    teacher_frame: Frame


class L2Prediction(BaseModel):
    frame: Frame
    guard_probability: float
    top1_probability: float
    margin: float
    entropy: float
    slot_avg_probability: float = 1.0
    slot_invalid_bio: bool = False
    nearest_similarity: float = 0.0
    predicted_intent_similarity: float = 0.0
    intent_support_margin: float = 0.0
    predicted_slot_count: float = 0.0
    predicted_has_slots: float = 0.0
    predicted_intent_frame_accuracy: float = 0.0
    predicted_intent_intent_accuracy: float = 0.0
    predicted_intent_support: float = 0.0
    predicted_intent_slotless_rate: float = 0.0
    predicted_signature_frame_accuracy: float = 0.0
    predicted_signature_support: float = 0.0


@dataclass(frozen=True)
class IntentSupportFeatures:
    nearest_similarity: float = 0.0
    predicted_intent_similarity: float = 0.0
    intent_support_margin: float = 0.0


@dataclass(frozen=True)
class IntentCalibrationFeatures:
    predicted_slot_count: float = 0.0
    predicted_has_slots: float = 0.0
    predicted_intent_frame_accuracy: float = 0.0
    predicted_intent_intent_accuracy: float = 0.0
    predicted_intent_support: float = 0.0
    predicted_intent_slotless_rate: float = 0.0
    predicted_signature_frame_accuracy: float = 0.0
    predicted_signature_support: float = 0.0


class IntentPrototypeIndex:
    def __init__(
        self,
        *,
        prototype_intents: tuple[str, ...],
        prototype_matrix: Any,
    ) -> None:
        self.prototype_intents = prototype_intents
        self.prototype_matrix = prototype_matrix

    @classmethod
    def from_examples(
        cls,
        intent_pipeline: Pipeline,
        examples: list[L2TrainingExample],
    ) -> IntentPrototypeIndex:
        feature_step = intent_pipeline.named_steps["features"]
        texts = [example.utterance for example in examples]
        matrix = normalize(feature_step.transform(texts), copy=True)
        return cls(
            prototype_intents=tuple(example.teacher_frame.intent for example in examples),
            prototype_matrix=matrix,
        )

    def score(
        self,
        intent_pipeline: Pipeline,
        utterance: str,
        predicted_intent: str,
    ) -> IntentSupportFeatures:
        if not self.prototype_intents:
            return IntentSupportFeatures()
        feature_step = intent_pipeline.named_steps["features"]
        query = normalize(feature_step.transform([utterance]), copy=True)
        similarities = (query @ self.prototype_matrix.T).toarray().ravel()
        if similarities.size == 0:
            return IntentSupportFeatures()
        nearest_similarity = float(similarities.max())
        same_intent_similarities = [
            float(similarity)
            for similarity, intent in zip(similarities, self.prototype_intents, strict=True)
            if intent == predicted_intent
        ]
        other_intent_similarities = [
            float(similarity)
            for similarity, intent in zip(similarities, self.prototype_intents, strict=True)
            if intent != predicted_intent
        ]
        predicted_intent_similarity = (
            max(same_intent_similarities) if same_intent_similarities else 0.0
        )
        other_intent_similarity = (
            max(other_intent_similarities) if other_intent_similarities else 0.0
        )
        return IntentSupportFeatures(
            nearest_similarity=nearest_similarity,
            predicted_intent_similarity=predicted_intent_similarity,
            intent_support_margin=predicted_intent_similarity - other_intent_similarity,
        )


class IntentCalibrationIndex:
    def __init__(
        self,
        *,
        predicted_intent_frame_accuracy: dict[str, float],
        predicted_intent_intent_accuracy: dict[str, float],
        predicted_intent_support: dict[str, float],
        predicted_intent_slotless_rate: dict[str, float],
        predicted_signature_frame_accuracy: dict[tuple[str, tuple[str, ...]], float],
        predicted_signature_support: dict[tuple[str, tuple[str, ...]], float],
    ) -> None:
        self.predicted_intent_frame_accuracy = predicted_intent_frame_accuracy
        self.predicted_intent_intent_accuracy = predicted_intent_intent_accuracy
        self.predicted_intent_support = predicted_intent_support
        self.predicted_intent_slotless_rate = predicted_intent_slotless_rate
        self.predicted_signature_frame_accuracy = predicted_signature_frame_accuracy
        self.predicted_signature_support = predicted_signature_support

    @classmethod
    def from_examples(
        cls,
        intent_pipeline: Pipeline,
        slot_tagger: TokenSlotTagger | None,
        examples: list[L2TrainingExample],
        *,
        slots_by_intent: dict[str, tuple[str, ...]] | None = None,
        slot_patterns_by_intent: dict[str, dict[str, list[dict[str, tuple[str, ...]]]]]
        | None = None,
    ) -> IntentCalibrationIndex:
        slots_by_intent = slots_by_intent or {}
        slot_patterns_by_intent = slot_patterns_by_intent or {}
        intent_stats: dict[str, dict[str, int]] = {}
        signature_stats: dict[tuple[str, tuple[str, ...]], dict[str, int]] = {}
        for example in examples:
            intent_result = predict_intent(intent_pipeline, example.utterance)
            slot_prediction = (
                slot_tagger.predict(example.utterance)
                if slot_tagger is not None
                else SlotPrediction(slots={})
            )
            predicted_intent = str(intent_result["intent"])
            slots = _postprocess_slots(
                predicted_intent,
                example.utterance,
                slot_prediction.slots,
                slots_by_intent,
                slot_patterns_by_intent,
            )
            predicted_frame = Frame(intent=predicted_intent, slots=slots)
            intent_bucket = intent_stats.setdefault(
                predicted_intent,
                {"total": 0, "frame_correct": 0, "intent_correct": 0, "slotless": 0},
            )
            intent_bucket["total"] += 1
            intent_bucket["frame_correct"] += int(predicted_frame == example.teacher_frame)
            intent_bucket["intent_correct"] += int(
                predicted_intent == example.teacher_frame.intent
            )
            intent_bucket["slotless"] += int(not slots)

            signature = _slot_signature(slots)
            signature_bucket = signature_stats.setdefault(
                (predicted_intent, signature),
                {"total": 0, "frame_correct": 0},
            )
            signature_bucket["total"] += 1
            signature_bucket["frame_correct"] += int(predicted_frame == example.teacher_frame)

        total_examples = max(1, len(examples))
        return cls(
            predicted_intent_frame_accuracy={
                intent: stats["frame_correct"] / stats["total"]
                for intent, stats in intent_stats.items()
            },
            predicted_intent_intent_accuracy={
                intent: stats["intent_correct"] / stats["total"]
                for intent, stats in intent_stats.items()
            },
            predicted_intent_support={
                intent: stats["total"] / total_examples
                for intent, stats in intent_stats.items()
            },
            predicted_intent_slotless_rate={
                intent: stats["slotless"] / stats["total"]
                for intent, stats in intent_stats.items()
            },
            predicted_signature_frame_accuracy={
                key: stats["frame_correct"] / stats["total"]
                for key, stats in signature_stats.items()
            },
            predicted_signature_support={
                key: stats["total"] / total_examples
                for key, stats in signature_stats.items()
            },
        )

    def score(self, predicted_intent: str, slots: dict[str, str]) -> IntentCalibrationFeatures:
        signature = _slot_signature(slots)
        signature_key = (predicted_intent, signature)
        return IntentCalibrationFeatures(
            predicted_slot_count=float(len(slots)),
            predicted_has_slots=float(bool(slots)),
            predicted_intent_frame_accuracy=self.predicted_intent_frame_accuracy.get(
                predicted_intent,
                0.0,
            ),
            predicted_intent_intent_accuracy=self.predicted_intent_intent_accuracy.get(
                predicted_intent,
                0.0,
            ),
            predicted_intent_support=self.predicted_intent_support.get(
                predicted_intent,
                0.0,
            ),
            predicted_intent_slotless_rate=self.predicted_intent_slotless_rate.get(
                predicted_intent,
                0.0,
            ),
            predicted_signature_frame_accuracy=self.predicted_signature_frame_accuracy.get(
                signature_key,
                0.0,
            ),
            predicted_signature_support=self.predicted_signature_support.get(
                signature_key,
                0.0,
            ),
        )


class ConstantGuard:
    def __init__(self, probability: float) -> None:
        self.probability = float(probability)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        probabilities = np.full((features.shape[0], 2), 0.0)
        probabilities[:, 1] = self.probability
        probabilities[:, 0] = 1.0 - self.probability
        return probabilities


class SlotPrediction(BaseModel):
    slots: dict[str, str]
    avg_probability: float = 1.0
    invalid_bio: bool = False
    model_name: str = "none"


class TokenSlotTagger:
    def __init__(
        self,
        *,
        vectorizer: DictVectorizer,
        classifier: SGDClassifier,
    ) -> None:
        self.vectorizer = vectorizer
        self.classifier = classifier

    def predict(self, utterance: str) -> SlotPrediction:
        tokens = tokenize(utterance)
        if not tokens:
            return SlotPrediction(slots={}, avg_probability=0.0, model_name="token_sgd")

        features = [_token_features(tokens, index) for index in range(len(tokens))]
        matrix = _csr32(self.vectorizer.transform(features))
        tags = [str(tag) for tag in self.classifier.predict(matrix)]
        probabilities = self.classifier.predict_proba(matrix)
        max_probabilities = [float(row.max()) for row in probabilities]
        slots, invalid_bio = slots_from_bio_tags(tokens, tags)
        return SlotPrediction(
            slots=slots,
            avg_probability=float(sum(max_probabilities) / len(max_probabilities)),
            invalid_bio=invalid_bio,
            model_name="token_sgd",
        )


class L2StudentBundle:
    def __init__(
        self,
        *,
        intent_pipeline: Pipeline,
        slot_tagger: TokenSlotTagger | None,
        guard_model: LogisticRegression | ConstantGuard,
        config: L2StudentConfig,
        slots_by_intent: dict[str, tuple[str, ...]] | None = None,
        slot_patterns_by_intent: dict[str, dict[str, list[dict[str, tuple[str, ...]]]]]
        | None = None,
        intent_support_index: IntentPrototypeIndex | None = None,
        intent_calibration_index: IntentCalibrationIndex | None = None,
    ) -> None:
        self.intent_pipeline = intent_pipeline
        self.slot_tagger = slot_tagger
        self.guard_model = guard_model
        self.config = config
        self.slots_by_intent = slots_by_intent or {}
        self.slot_patterns_by_intent = slot_patterns_by_intent or {}
        self.intent_support_index = intent_support_index
        self.intent_calibration_index = intent_calibration_index

    def predict(self, utterance: str) -> L2Prediction:
        intent_result = predict_intent(self.intent_pipeline, utterance)
        slot_prediction = (
            self.slot_tagger.predict(utterance)
            if self.slot_tagger is not None
            else SlotPrediction(slots={})
        )
        predicted_intent = str(intent_result["intent"])
        slots = _postprocess_slots(
            predicted_intent,
            utterance,
            slot_prediction.slots,
            getattr(self, "slots_by_intent", {}),
            getattr(self, "slot_patterns_by_intent", {}),
        )
        support = _score_intent_support(
            getattr(self, "intent_support_index", None),
            self.intent_pipeline,
            utterance,
            predicted_intent,
        )
        calibration = _score_intent_calibration(
            getattr(self, "intent_calibration_index", None),
            predicted_intent,
            slots,
        )
        features = guard_features(
            intent_result["top_probability"],
            intent_result["margin"],
            intent_result["entropy"],
            slot_prediction.avg_probability,
            slot_prediction.invalid_bio,
            support.nearest_similarity,
            support.predicted_intent_similarity,
            support.intent_support_margin,
            calibration.predicted_slot_count,
            calibration.predicted_has_slots,
            calibration.predicted_intent_frame_accuracy,
            calibration.predicted_intent_intent_accuracy,
            calibration.predicted_intent_support,
            calibration.predicted_intent_slotless_rate,
            calibration.predicted_signature_frame_accuracy,
            calibration.predicted_signature_support,
        )
        features = _match_guard_feature_width(features, self.guard_model)
        guard_probability = float(self.guard_model.predict_proba(features)[0][1])
        return L2Prediction(
            frame=Frame(intent=predicted_intent, slots=slots),
            guard_probability=guard_probability,
            top1_probability=intent_result["top_probability"],
            margin=intent_result["margin"],
            entropy=intent_result["entropy"],
            slot_avg_probability=slot_prediction.avg_probability,
            slot_invalid_bio=slot_prediction.invalid_bio,
            nearest_similarity=support.nearest_similarity,
            predicted_intent_similarity=support.predicted_intent_similarity,
            intent_support_margin=support.intent_support_margin,
            predicted_slot_count=calibration.predicted_slot_count,
            predicted_has_slots=calibration.predicted_has_slots,
            predicted_intent_frame_accuracy=calibration.predicted_intent_frame_accuracy,
            predicted_intent_intent_accuracy=calibration.predicted_intent_intent_accuracy,
            predicted_intent_support=calibration.predicted_intent_support,
            predicted_intent_slotless_rate=calibration.predicted_intent_slotless_rate,
            predicted_signature_frame_accuracy=calibration.predicted_signature_frame_accuracy,
            predicted_signature_support=calibration.predicted_signature_support,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> L2StudentBundle:
        return joblib.load(path)


class L2StudentLayer:
    def __init__(self, bundle: L2StudentBundle) -> None:
        self.bundle = bundle

    def try_answer(self, utterance: str) -> LayerResult:
        with elapsed_ms() as ms:
            prediction = self.bundle.predict(utterance)
            runtime_enabled = getattr(self.bundle.config, "runtime_enabled", True)
            accepted = runtime_enabled and guard_accepts(
                prediction.guard_probability,
                self.bundle.config.accept_threshold,
            )
            return LayerResult(
                layer="L2",
                accepted=accepted,
                frame=prediction.frame if accepted else None,
                confidence=prediction.guard_probability,
                reason=_l2_layer_reason(runtime_enabled, accepted),
                latency_ms=ms(),
                metadata={
                    "predicted_frame": prediction.frame.model_dump(mode="json"),
                    "guard_probability": prediction.guard_probability,
                    "top1_probability": prediction.top1_probability,
                    "margin": prediction.margin,
                    "entropy": prediction.entropy,
                    "slot_avg_probability": prediction.slot_avg_probability,
                    "slot_invalid_bio": prediction.slot_invalid_bio,
                    "nearest_similarity": prediction.nearest_similarity,
                    "predicted_intent_similarity": prediction.predicted_intent_similarity,
                    "intent_support_margin": prediction.intent_support_margin,
                    "predicted_slot_count": prediction.predicted_slot_count,
                    "predicted_has_slots": prediction.predicted_has_slots,
                    "predicted_intent_frame_accuracy": (
                        prediction.predicted_intent_frame_accuracy
                    ),
                    "predicted_intent_intent_accuracy": (
                        prediction.predicted_intent_intent_accuracy
                    ),
                    "predicted_intent_support": prediction.predicted_intent_support,
                    "predicted_intent_slotless_rate": (
                        prediction.predicted_intent_slotless_rate
                    ),
                    "predicted_signature_frame_accuracy": (
                        prediction.predicted_signature_frame_accuracy
                    ),
                    "predicted_signature_support": prediction.predicted_signature_support,
                    "accept_threshold": self.bundle.config.accept_threshold,
                    "runtime_enabled": runtime_enabled,
                    "intent_model": self.bundle.config.intent_model_family,
                    "slot_model": "token_sgd" if self.bundle.slot_tagger else "none",
                },
            )


def guard_accepts(probability: float, threshold: float) -> bool:
    return GuardDecision(probability=probability, threshold=threshold).accepted


def _l2_layer_reason(runtime_enabled: bool, accepted: bool) -> str:
    if not runtime_enabled:
        return "runtime disabled"
    return "guard accepted" if accepted else "guard rejected"


def training_examples_from_teacher_traces(
    traces: list[TeacherTrace],
) -> list[L2TrainingExample]:
    return [
        L2TrainingExample(utterance=trace.utterance, teacher_frame=trace.teacher_frame)
        for trace in traces
        if trace.teacher_frame is not None
    ]


def train_l2_student(
    examples: list[L2TrainingExample],
    config: L2StudentConfig | None = None,
) -> L2StudentBundle:
    config = config or L2StudentConfig()
    if len(examples) < config.min_examples:
        raise ValueError(f"L2 training requires at least {config.min_examples} examples")
    labels = [example.teacher_frame.intent for example in examples]
    if len(set(labels)) < 2:
        raise ValueError("L2 intent student requires at least two teacher intents")

    train_examples, guard_examples = _split_examples(examples, config.random_state)
    calibration_intent_pipeline = train_intent_pipeline(train_examples, config)
    calibration_slot_tagger = train_slot_tagger(train_examples, config)
    calibration_slots_by_intent = slots_by_intent_from_examples(train_examples)
    calibration_slot_patterns = slot_patterns_by_intent_from_examples(train_examples)
    calibration_intent_support = IntentPrototypeIndex.from_examples(
        calibration_intent_pipeline,
        train_examples,
    )
    calibration_intent_reliability = IntentCalibrationIndex.from_examples(
        calibration_intent_pipeline,
        calibration_slot_tagger,
        guard_examples,
        slots_by_intent=calibration_slots_by_intent,
        slot_patterns_by_intent=calibration_slot_patterns,
    )
    guard_model = train_guard(
        calibration_intent_pipeline,
        calibration_slot_tagger,
        guard_examples,
        config,
        slots_by_intent=calibration_slots_by_intent,
        slot_patterns_by_intent=calibration_slot_patterns,
        intent_support_index=calibration_intent_support,
        intent_calibration_index=calibration_intent_reliability,
    )
    runtime_intent_pipeline = train_intent_pipeline(examples, config)
    runtime_slot_tagger = train_slot_tagger(examples, config)
    return L2StudentBundle(
        intent_pipeline=runtime_intent_pipeline,
        slot_tagger=runtime_slot_tagger,
        guard_model=guard_model,
        config=config,
        slots_by_intent=slots_by_intent_from_examples(examples),
        slot_patterns_by_intent=slot_patterns_by_intent_from_examples(examples),
        intent_support_index=IntentPrototypeIndex.from_examples(
            runtime_intent_pipeline,
            examples,
        ),
        intent_calibration_index=calibration_intent_reliability,
    )


def train_intent_pipeline(
    examples: list[L2TrainingExample],
    config: L2StudentConfig,
) -> Pipeline:
    intent_pipeline = Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "word",
                            TfidfVectorizer(
                                analyzer="word",
                                ngram_range=config.word_ngram_range,
                                max_features=config.max_features,
                            ),
                        ),
                        (
                            "char",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=config.char_ngram_range,
                                max_features=config.max_features,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "intent",
                _intent_classifier(config),
            ),
        ]
    )
    intent_pipeline.fit(
        [example.utterance for example in examples],
        [example.teacher_frame.intent for example in examples],
    )
    return intent_pipeline


def _intent_classifier(config: L2StudentConfig) -> SGDClassifier | MLPClassifier:
    if config.intent_model_family == "sgd_logreg":
        return SGDClassifier(
            loss="log_loss",
            random_state=config.random_state,
            max_iter=config.max_iter,
            tol=1e-3,
        )
    if config.intent_model_family == "mlp":
        hidden_layer_sizes = _mlp_hidden_layer_sizes(config.mlp_hidden_layer_sizes)
        return MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=config.mlp_alpha,
            early_stopping=config.mlp_early_stopping,
            random_state=config.random_state,
            max_iter=config.max_iter,
            tol=1e-3,
        )
    raise ValueError(f"unsupported L2 intent_model_family: {config.intent_model_family}")


def _mlp_hidden_layer_sizes(value: tuple[int, ...]) -> tuple[int, ...]:
    if not value:
        raise ValueError("mlp_hidden_layer_sizes must not be empty")
    if any(layer_size <= 0 for layer_size in value):
        raise ValueError("mlp_hidden_layer_sizes must contain positive integers")
    return tuple(value)


def train_guard(
    intent_pipeline: Pipeline,
    slot_tagger: TokenSlotTagger | None,
    examples: list[L2TrainingExample],
    config: L2StudentConfig,
    *,
    slots_by_intent: dict[str, tuple[str, ...]] | None = None,
    slot_patterns_by_intent: dict[str, dict[str, list[dict[str, tuple[str, ...]]]]]
    | None = None,
    intent_support_index: IntentPrototypeIndex | None = None,
    intent_calibration_index: IntentCalibrationIndex | None = None,
) -> LogisticRegression | ConstantGuard:
    slots_by_intent = slots_by_intent or {}
    slot_patterns_by_intent = slot_patterns_by_intent or {}
    feature_rows = []
    correct_labels = []
    for example in examples:
        intent_result = predict_intent(intent_pipeline, example.utterance)
        slot_prediction = (
            slot_tagger.predict(example.utterance)
            if slot_tagger is not None
            else SlotPrediction(slots={})
        )
        predicted_intent = str(intent_result["intent"])
        slots = _postprocess_slots(
            predicted_intent,
            example.utterance,
            slot_prediction.slots,
            slots_by_intent,
            slot_patterns_by_intent,
        )
        predicted_frame = Frame(intent=predicted_intent, slots=slots)
        support = _score_intent_support(
            intent_support_index,
            intent_pipeline,
            example.utterance,
            predicted_intent,
        )
        calibration = _score_intent_calibration(
            intent_calibration_index,
            predicted_intent,
            slots,
        )
        feature_rows.append(
            [
                intent_result["top_probability"],
                intent_result["margin"],
                intent_result["entropy"],
                slot_prediction.avg_probability,
                float(slot_prediction.invalid_bio),
                support.nearest_similarity,
                support.predicted_intent_similarity,
                support.intent_support_margin,
                calibration.predicted_slot_count,
                calibration.predicted_has_slots,
                calibration.predicted_intent_frame_accuracy,
                calibration.predicted_intent_intent_accuracy,
                calibration.predicted_intent_support,
                calibration.predicted_intent_slotless_rate,
                calibration.predicted_signature_frame_accuracy,
                calibration.predicted_signature_support,
            ]
        )
        correct_labels.append(int(predicted_frame == example.teacher_frame))

    features = np.asarray(feature_rows, dtype=float)
    labels = np.asarray(correct_labels, dtype=int)
    if len(set(labels.tolist())) < 2:
        return ConstantGuard(float(labels.mean()) if len(labels) else 0.0)
    guard = LogisticRegression(random_state=config.random_state)
    guard.fit(features, labels)
    return guard


def guard_features(
    top_probability: float,
    margin: float,
    entropy: float,
    slot_avg_probability: float = 1.0,
    slot_invalid_bio: bool = False,
    nearest_similarity: float = 0.0,
    predicted_intent_similarity: float = 0.0,
    intent_support_margin: float = 0.0,
    predicted_slot_count: float = 0.0,
    predicted_has_slots: float = 0.0,
    predicted_intent_frame_accuracy: float = 0.0,
    predicted_intent_intent_accuracy: float = 0.0,
    predicted_intent_support: float = 0.0,
    predicted_intent_slotless_rate: float = 0.0,
    predicted_signature_frame_accuracy: float = 0.0,
    predicted_signature_support: float = 0.0,
) -> np.ndarray:
    return np.asarray(
        [
            [
                top_probability,
                margin,
                entropy,
                slot_avg_probability,
                float(slot_invalid_bio),
                nearest_similarity,
                predicted_intent_similarity,
                intent_support_margin,
                predicted_slot_count,
                predicted_has_slots,
                predicted_intent_frame_accuracy,
                predicted_intent_intent_accuracy,
                predicted_intent_support,
                predicted_intent_slotless_rate,
                predicted_signature_frame_accuracy,
                predicted_signature_support,
            ]
        ],
        dtype=float,
    )


def _score_intent_support(
    support_index: IntentPrototypeIndex | None,
    intent_pipeline: Pipeline,
    utterance: str,
    predicted_intent: str,
) -> IntentSupportFeatures:
    if support_index is None:
        return IntentSupportFeatures()
    return support_index.score(intent_pipeline, utterance, predicted_intent)


def _score_intent_calibration(
    calibration_index: IntentCalibrationIndex | None,
    predicted_intent: str,
    slots: dict[str, str],
) -> IntentCalibrationFeatures:
    if calibration_index is None:
        return IntentCalibrationFeatures(
            predicted_slot_count=float(len(slots)),
            predicted_has_slots=float(bool(slots)),
        )
    return calibration_index.score(predicted_intent, slots)


def _match_guard_feature_width(features: np.ndarray, guard_model: Any) -> np.ndarray:
    expected_width = getattr(guard_model, "n_features_in_", None)
    if expected_width is None:
        return features
    width = int(expected_width)
    if features.shape[1] == width:
        return features
    if features.shape[1] > width:
        return features[:, :width]
    padding = np.zeros((features.shape[0], width - features.shape[1]), dtype=features.dtype)
    return np.hstack([features, padding])


def slots_by_intent_from_examples(
    examples: list[L2TrainingExample],
) -> dict[str, tuple[str, ...]]:
    slots_by_intent: dict[str, set[str]] = {}
    for example in examples:
        slots_by_intent.setdefault(example.teacher_frame.intent, set()).update(
            example.teacher_frame.slots
        )
    return {
        intent: tuple(sorted(slot_names))
        for intent, slot_names in sorted(slots_by_intent.items())
    }


def filter_slots_for_intent(
    intent: str,
    slots: dict[str, str],
    slots_by_intent: dict[str, tuple[str, ...]],
) -> dict[str, str]:
    if not slots_by_intent:
        return slots
    allowed = set(slots_by_intent.get(intent, ()))
    return {slot_name: value for slot_name, value in slots.items() if slot_name in allowed}


def _postprocess_slots(
    intent: str,
    utterance: str,
    slots: dict[str, str],
    slots_by_intent: dict[str, tuple[str, ...]],
    slot_patterns_by_intent: dict[str, dict[str, list[dict[str, tuple[str, ...]]]]],
) -> dict[str, str]:
    filtered = filter_slots_for_intent(intent, slots, slots_by_intent)
    return apply_slot_patterns(
        intent,
        utterance,
        filtered,
        slots_by_intent,
        slot_patterns_by_intent,
    )


def _slot_signature(slots: dict[str, str]) -> tuple[str, ...]:
    return tuple(sorted(slots))


def slot_patterns_by_intent_from_examples(
    examples: list[L2TrainingExample],
) -> dict[str, dict[str, list[dict[str, tuple[str, ...]]]]]:
    patterns: dict[str, dict[str, list[dict[str, tuple[str, ...]]]]] = {}
    seen: set[tuple[str, str, tuple[str, ...], tuple[str, ...]]] = set()
    for example in examples:
        tokens = tokenize(example.utterance)
        for slot_name, slot_value in sorted(example.teacher_frame.slots.items()):
            slot_tokens = tokenize(slot_value)
            if not tokens or not slot_tokens:
                continue
            start = _find_unoccupied_span(tokens, slot_tokens, [False] * len(tokens))
            if start is None:
                continue
            end = start + len(slot_tokens)
            prefix = tuple(tokens[max(0, start - 3) : start])
            suffix = tuple(tokens[end : min(len(tokens), end + 3)])
            key = (example.teacher_frame.intent, slot_name, prefix, suffix)
            if key in seen:
                continue
            seen.add(key)
            patterns.setdefault(example.teacher_frame.intent, {}).setdefault(slot_name, []).append(
                {
                    "prefix": prefix,
                    "suffix": suffix,
                }
            )
    return {
        intent: {
            slot_name: sorted(
                slot_patterns,
                key=lambda pattern: (
                    -(len(pattern["prefix"]) + len(pattern["suffix"])),
                    pattern["prefix"],
                    pattern["suffix"],
                ),
            )
            for slot_name, slot_patterns in sorted(slots.items())
        }
        for intent, slots in sorted(patterns.items())
    }


def apply_slot_patterns(
    intent: str,
    utterance: str,
    slots: dict[str, str],
    slots_by_intent: dict[str, tuple[str, ...]],
    slot_patterns_by_intent: dict[str, dict[str, list[dict[str, tuple[str, ...]]]]],
) -> dict[str, str]:
    allowed_slots = slots_by_intent.get(intent, ())
    if not allowed_slots:
        return slots
    patterns_by_slot = slot_patterns_by_intent.get(intent, {})
    if not patterns_by_slot:
        return slots
    tokens = tokenize(utterance)
    if not tokens:
        return slots
    updated = dict(slots)
    for slot_name in allowed_slots:
        for pattern in patterns_by_slot.get(slot_name, []):
            extracted = _extract_slot_with_context(tokens, pattern)
            if extracted is None:
                continue
            existing = tokenize(updated.get(slot_name, ""))
            if slot_name not in updated or len(tokenize(extracted)) > len(existing):
                updated[slot_name] = extracted
            break
    return updated


def _extract_slot_with_context(
    tokens: list[str],
    pattern: dict[str, tuple[str, ...]],
) -> str | None:
    prefix = tuple(pattern.get("prefix", ()))
    suffix = tuple(pattern.get("suffix", ()))
    starts = _candidate_starts(tokens, prefix)
    for start in starts:
        end = _candidate_end(tokens, suffix, start)
        if end is None or end <= start:
            continue
        if end - start > 8:
            continue
        return " ".join(tokens[start:end])
    return None


def _candidate_starts(tokens: list[str], prefix: tuple[str, ...]) -> list[int]:
    if not prefix:
        return [0]
    starts = []
    width = len(prefix)
    for index in range(0, len(tokens) - width + 1):
        if tuple(tokens[index : index + width]) == prefix:
            starts.append(index + width)
    return starts


def _candidate_end(tokens: list[str], suffix: tuple[str, ...], start: int) -> int | None:
    if not suffix:
        return len(tokens)
    width = len(suffix)
    for index in range(start, len(tokens) - width + 1):
        if tuple(tokens[index : index + width]) == suffix:
            return index
    return None


def train_slot_tagger(
    examples: list[L2TrainingExample],
    config: L2StudentConfig,
) -> TokenSlotTagger | None:
    if config.slot_model_family == "none":
        return None

    feature_rows: list[dict[str, Any]] = []
    labels: list[str] = []
    for example in examples:
        tokens = tokenize(example.utterance)
        if not tokens:
            continue
        tags = bio_tags_for_teacher_slots(tokens, example.teacher_frame.slots)
        for index in range(len(tokens)):
            feature_rows.append(_token_features(tokens, index))
            labels.append(tags[index])

    if len(set(labels)) < 2:
        return None

    vectorizer = DictVectorizer(sparse=True)
    features = _csr32(vectorizer.fit_transform(feature_rows))
    classifier = SGDClassifier(
        loss="log_loss",
        random_state=config.random_state,
        max_iter=config.max_iter,
        tol=1e-3,
    )
    classifier.fit(features, labels)
    return TokenSlotTagger(vectorizer=vectorizer, classifier=classifier)


def predict_intent(intent_pipeline: Pipeline, utterance: str) -> dict[str, float | str]:
    probabilities = intent_pipeline.predict_proba([utterance])[0]
    classes = list(intent_pipeline.classes_)
    sorted_indices = np.argsort(probabilities)[::-1]
    top_index = int(sorted_indices[0])
    second_probability = _second_probability(probabilities, sorted_indices)
    top_probability = float(probabilities[top_index])
    return {
        "intent": str(classes[top_index]),
        "top_probability": top_probability,
        "margin": top_probability - second_probability,
        "entropy": _entropy(probabilities),
    }


TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def bio_tags_for_teacher_slots(tokens: list[str], slots: dict[str, str]) -> list[str]:
    tags = ["O"] * len(tokens)
    occupied = [False] * len(tokens)
    for slot_name, slot_value in sorted(slots.items()):
        slot_tokens = tokenize(slot_value)
        if not slot_tokens:
            continue
        start = _find_unoccupied_span(tokens, slot_tokens, occupied)
        if start is None:
            continue
        for offset, _token in enumerate(slot_tokens):
            index = start + offset
            occupied[index] = True
            tags[index] = f"B-{slot_name}" if offset == 0 else f"I-{slot_name}"
    return tags


def slots_from_bio_tags(tokens: list[str], tags: list[str]) -> tuple[dict[str, str], bool]:
    slots: dict[str, list[str]] = {}
    invalid_bio = False
    index = 0
    while index < len(tokens):
        tag = tags[index]
        if tag == "O":
            index += 1
            continue
        if tag.startswith("I-"):
            invalid_bio = True
            index += 1
            continue
        if not tag.startswith("B-"):
            invalid_bio = True
            index += 1
            continue
        slot_name = tag[2:]
        values = [tokens[index]]
        index += 1
        while index < len(tokens) and tags[index] == f"I-{slot_name}":
            values.append(tokens[index])
            index += 1
        if slot_name in slots:
            invalid_bio = True
        else:
            slots[slot_name] = values
    return {slot_name: " ".join(values) for slot_name, values in slots.items()}, invalid_bio


def _find_unoccupied_span(
    tokens: list[str],
    slot_tokens: list[str],
    occupied: list[bool],
) -> int | None:
    span_len = len(slot_tokens)
    for start in range(0, len(tokens) - span_len + 1):
        if any(occupied[start : start + span_len]):
            continue
        if tokens[start : start + span_len] == slot_tokens:
            return start
    return None


def _token_features(tokens: list[str], index: int) -> dict[str, Any]:
    token = tokens[index]
    previous_token = tokens[index - 1] if index > 0 else "<BOS>"
    next_token = tokens[index + 1] if index + 1 < len(tokens) else "<EOS>"
    return {
        "token": token,
        "lower": token.lower(),
        "prefix2": token[:2],
        "suffix2": token[-2:],
        "prev": previous_token,
        "next": next_token,
        "is_digit": token.isdigit(),
        "position": index,
    }


def _split_examples(
    examples: list[L2TrainingExample],
    random_state: int,
) -> tuple[list[L2TrainingExample], list[L2TrainingExample]]:
    labels = [example.teacher_frame.intent for example in examples]
    try:
        train_examples, guard_examples = train_test_split(
            examples,
            test_size=0.35,
            random_state=random_state,
            stratify=labels,
        )
    except ValueError:
        train_examples, guard_examples = train_test_split(
            examples,
            test_size=0.35,
            random_state=random_state,
        )
    return list(train_examples), list(guard_examples)


def _entropy(probabilities: np.ndarray) -> float:
    return float(-sum(p * math.log(max(float(p), 1e-12)) for p in probabilities))


def _second_probability(probabilities: np.ndarray, sorted_indices: np.ndarray) -> float:
    if len(sorted_indices) <= 1:
        return 0.0
    return float(probabilities[int(sorted_indices[1])])


def _csr32(matrix):
    converted = matrix.tocsr(copy=True)
    converted.indices = converted.indices.astype("int32", copy=False)
    converted.indptr = converted.indptr.astype("int32", copy=False)
    return converted
