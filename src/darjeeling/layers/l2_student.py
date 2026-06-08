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
from sklearn.pipeline import Pipeline

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
    random_state: int = 17
    min_examples: int = 4
    word_ngram_range: tuple[int, int] = (1, 2)
    char_ngram_range: tuple[int, int] = (3, 5)
    max_features: int = 50_000
    max_iter: int = 1000
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
    ) -> None:
        self.intent_pipeline = intent_pipeline
        self.slot_tagger = slot_tagger
        self.guard_model = guard_model
        self.config = config

    def predict(self, utterance: str) -> L2Prediction:
        intent_result = predict_intent(self.intent_pipeline, utterance)
        slot_prediction = (
            self.slot_tagger.predict(utterance)
            if self.slot_tagger is not None
            else SlotPrediction(slots={})
        )
        features = guard_features(
            intent_result["top_probability"],
            intent_result["margin"],
            intent_result["entropy"],
            slot_prediction.avg_probability,
            slot_prediction.invalid_bio,
        )
        guard_probability = float(self.guard_model.predict_proba(features)[0][1])
        return L2Prediction(
            frame=Frame(intent=intent_result["intent"], slots=slot_prediction.slots),
            guard_probability=guard_probability,
            top1_probability=intent_result["top_probability"],
            margin=intent_result["margin"],
            entropy=intent_result["entropy"],
            slot_avg_probability=slot_prediction.avg_probability,
            slot_invalid_bio=slot_prediction.invalid_bio,
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
            accepted = guard_accepts(
                prediction.guard_probability,
                self.bundle.config.accept_threshold,
            )
            return LayerResult(
                layer="L2",
                accepted=accepted,
                frame=prediction.frame if accepted else None,
                confidence=prediction.guard_probability,
                reason="guard accepted" if accepted else "guard rejected",
                latency_ms=ms(),
                metadata={
                    "predicted_frame": prediction.frame.model_dump(mode="json"),
                    "guard_probability": prediction.guard_probability,
                    "top1_probability": prediction.top1_probability,
                    "margin": prediction.margin,
                    "entropy": prediction.entropy,
                    "slot_avg_probability": prediction.slot_avg_probability,
                    "slot_invalid_bio": prediction.slot_invalid_bio,
                    "accept_threshold": self.bundle.config.accept_threshold,
                    "slot_model": "token_sgd" if self.bundle.slot_tagger else "none",
                },
            )


def guard_accepts(probability: float, threshold: float) -> bool:
    return GuardDecision(probability=probability, threshold=threshold).accepted


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
    intent_pipeline = Pipeline(
        [
            (
                "features",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=config.word_ngram_range,
                    max_features=config.max_features,
                ),
            ),
            (
                "intent",
                SGDClassifier(
                    loss="log_loss",
                    random_state=config.random_state,
                    max_iter=config.max_iter,
                    tol=1e-3,
                ),
            ),
        ]
    )
    intent_pipeline.fit(
        [example.utterance for example in train_examples],
        [example.teacher_frame.intent for example in train_examples],
    )
    slot_tagger = train_slot_tagger(train_examples, config)
    guard_model = train_guard(intent_pipeline, slot_tagger, guard_examples, config)
    return L2StudentBundle(
        intent_pipeline=intent_pipeline,
        slot_tagger=slot_tagger,
        guard_model=guard_model,
        config=config,
    )


def train_guard(
    intent_pipeline: Pipeline,
    slot_tagger: TokenSlotTagger | None,
    examples: list[L2TrainingExample],
    config: L2StudentConfig,
) -> LogisticRegression | ConstantGuard:
    feature_rows = []
    correct_labels = []
    for example in examples:
        intent_result = predict_intent(intent_pipeline, example.utterance)
        slot_prediction = (
            slot_tagger.predict(example.utterance)
            if slot_tagger is not None
            else SlotPrediction(slots={})
        )
        predicted_frame = Frame(intent=intent_result["intent"], slots=slot_prediction.slots)
        feature_rows.append(
            [
                intent_result["top_probability"],
                intent_result["margin"],
                intent_result["entropy"],
                slot_prediction.avg_probability,
                float(slot_prediction.invalid_bio),
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
) -> np.ndarray:
    return np.asarray(
        [
            [
                top_probability,
                margin,
                entropy,
                slot_avg_probability,
                float(slot_invalid_bio),
            ]
        ],
        dtype=float,
    )


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
