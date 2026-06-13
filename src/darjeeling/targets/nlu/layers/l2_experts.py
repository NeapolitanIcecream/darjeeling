from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from darjeeling.runtime.timing import elapsed_ms
from darjeeling.targets.nlu.data import normalize_utterance
from darjeeling.targets.nlu.schemas import FramePatch, LayerResult, TeacherTrace

EXPERT_BANK_SCHEMA_VERSION = "l2-expert-bank-v1"


@dataclass(frozen=True)
class L2ExpertTrainingConfig:
    min_examples: int = 4
    max_intents: int = 4
    max_slots: int = 4
    min_accuracy: float = 0.95
    random_state: int = 17


@dataclass
class IntentBinaryExpert:
    intent: str
    vectorizer: TfidfVectorizer
    classifier: LogisticRegression
    threshold: float
    validation_metrics: dict[str, Any]

    @property
    def name(self) -> str:
        return f"intent:{self.intent}"

    def try_patch(self, utterance: str) -> tuple[FramePatch | None, dict[str, Any]]:
        matrix = self.vectorizer.transform([utterance])
        probability = _positive_probability(self.classifier, matrix)
        metadata = {
            "expert": self.name,
            "probability": probability,
            "threshold": self.threshold,
            "validation_metrics": self.validation_metrics,
        }
        if probability < self.threshold:
            return None, metadata
        return (
            FramePatch(
                accepted_intent=self.intent,
                source_layer="L2",
                confidence=probability,
                complete=False,
                metadata=metadata,
            ),
            metadata,
        )

    def manifest_entry(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": "intent_binary",
            "intent": self.intent,
            "threshold": self.threshold,
            "validation_metrics": self.validation_metrics,
        }


@dataclass
class SlotValueExpert:
    slot_key: str
    values_by_normalized_value: dict[str, str]
    validation_metrics: dict[str, Any]
    threshold: float = 1.0

    @property
    def name(self) -> str:
        return f"slot:{self.slot_key}"

    def try_patch(self, utterance: str) -> tuple[FramePatch | None, dict[str, Any]]:
        normalized_utterance = normalize_utterance(utterance)
        matched_value = None
        for normalized_value, canonical_value in sorted(
            self.values_by_normalized_value.items(),
            key=lambda item: (-len(item[0]), item[0]),
        ):
            if normalized_value and normalized_value in normalized_utterance:
                matched_value = canonical_value
                break
        metadata = {
            "expert": self.name,
            "threshold": self.threshold,
            "validation_metrics": self.validation_metrics,
        }
        if matched_value is None:
            return None, metadata
        return (
            FramePatch(
                accepted_slots={self.slot_key: matched_value},
                source_layer="L2",
                confidence=self.threshold,
                complete=False,
                metadata={**metadata, "matched_value": matched_value},
            ),
            {**metadata, "matched_value": matched_value},
        )

    def manifest_entry(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": "slot_value",
            "slot_key": self.slot_key,
            "threshold": self.threshold,
            "value_count": len(self.values_by_normalized_value),
            "validation_metrics": self.validation_metrics,
        }


@dataclass
class L2ExpertBank:
    intent_experts: list[IntentBinaryExpert] = field(default_factory=list)
    slot_experts: list[SlotValueExpert] = field(default_factory=list)
    selection_metrics: dict[str, Any] = field(default_factory=dict)

    def try_patch(self, utterance: str) -> tuple[FramePatch | None, list[dict[str, Any]]]:
        fired: list[dict[str, Any]] = []
        accepted_intent: str | None = None
        accepted_slots: dict[str, str] = {}
        confidences: list[float] = []
        for expert in self.intent_experts:
            patch, metadata = expert.try_patch(utterance)
            if patch is None:
                continue
            fired.append(metadata)
            accepted_intent = accepted_intent or patch.accepted_intent
            if patch.confidence is not None:
                confidences.append(patch.confidence)
        for expert in self.slot_experts:
            patch, metadata = expert.try_patch(utterance)
            if patch is None:
                continue
            fired.append(metadata)
            accepted_slots.update(patch.accepted_slots)
            if patch.confidence is not None:
                confidences.append(patch.confidence)
        if accepted_intent is None and not accepted_slots:
            return None, fired
        return (
            FramePatch(
                accepted_intent=accepted_intent,
                accepted_slots=accepted_slots,
                source_layer="L2",
                confidence=max(confidences) if confidences else None,
                complete=False,
                metadata={"experts": fired},
            ),
            fired,
        )

    def manifest_payload(self) -> dict[str, Any]:
        return {
            "schema_version": EXPERT_BANK_SCHEMA_VERSION,
            "intent_experts": [expert.manifest_entry() for expert in self.intent_experts],
            "slot_experts": [expert.manifest_entry() for expert in self.slot_experts],
            "selection_metrics": self.selection_metrics,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> L2ExpertBank:
        return joblib.load(path)


class L2ExpertBankLayer:
    def __init__(self, bank: L2ExpertBank, fallback_layer: Any | None = None) -> None:
        self.bank = bank
        self.fallback_layer = fallback_layer

    def try_answer(self, utterance: str) -> LayerResult:
        with elapsed_ms() as ms:
            patch, fired = self.bank.try_patch(utterance)
            if patch is not None:
                return LayerResult(
                    layer="L2",
                    accepted=True,
                    patch=patch,
                    confidence=patch.confidence,
                    reason="expert patch accepted",
                    latency_ms=ms(),
                    metadata={
                        "expert_bank_schema_version": EXPERT_BANK_SCHEMA_VERSION,
                        "experts_fired": fired,
                        "frame_patch": patch.model_dump(mode="json"),
                    },
                )
        if self.fallback_layer is not None:
            return self.fallback_layer.try_answer(utterance)
        return LayerResult(
            layer="L2",
            accepted=False,
            reason="no expert accepted",
            latency_ms=0.0,
            metadata={"expert_bank_schema_version": EXPERT_BANK_SCHEMA_VERSION},
        )


def train_l2_expert_bank(
    traces: list[TeacherTrace],
    config: L2ExpertTrainingConfig,
) -> L2ExpertBank | None:
    labeled = [trace for trace in traces if trace.teacher_frame is not None]
    if len(labeled) < config.min_examples:
        return None
    intent_experts: list[IntentBinaryExpert] = []
    slot_experts: list[SlotValueExpert] = []
    selected_intents = _selected_intents(labeled, config)
    selected_slots = _selected_slots(labeled, config)
    for intent in selected_intents:
        expert = _train_intent_expert(labeled, intent=intent, config=config)
        if expert is not None:
            intent_experts.append(expert)
    for slot_key in selected_slots:
        expert = _train_slot_expert(labeled, slot_key=slot_key, config=config)
        if expert is not None:
            slot_experts.append(expert)
    if not intent_experts and not slot_experts:
        return None
    return L2ExpertBank(
        intent_experts=intent_experts,
        slot_experts=slot_experts,
        selection_metrics={
            "schema_version": "l2-expert-selection-v1",
            "training_traces": len(labeled),
            "selected_intents": selected_intents,
            "selected_slots": selected_slots,
            "adopted_intent_experts": [expert.intent for expert in intent_experts],
            "adopted_slot_experts": [expert.slot_key for expert in slot_experts],
            "min_accuracy": config.min_accuracy,
            "min_examples": config.min_examples,
        },
    )


def write_l2_expert_manifest(path: Path, bank: L2ExpertBank) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(bank.manifest_payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _selected_intents(
    traces: list[TeacherTrace],
    config: L2ExpertTrainingConfig,
) -> list[str]:
    counts: Counter[str] = Counter(
        trace.teacher_frame.intent for trace in traces if trace.teacher_frame
    )
    return [
        intent
        for intent, count in counts.most_common(config.max_intents)
        if count >= config.min_examples
    ]


def _selected_slots(
    traces: list[TeacherTrace],
    config: L2ExpertTrainingConfig,
) -> list[str]:
    counts: Counter[str] = Counter(
        slot_key
        for trace in traces
        if trace.teacher_frame is not None
        for slot_key in trace.teacher_frame.slots
    )
    return [
        slot_key
        for slot_key, count in counts.most_common(config.max_slots)
        if count >= config.min_examples
    ]


def _train_intent_expert(
    traces: list[TeacherTrace],
    *,
    intent: str,
    config: L2ExpertTrainingConfig,
) -> IntentBinaryExpert | None:
    positives = [
        trace
        for trace in traces
        if trace.teacher_frame and trace.teacher_frame.intent == intent
    ]
    negatives = [
        trace
        for trace in traces
        if trace.teacher_frame and trace.teacher_frame.intent != intent
    ]
    if len(positives) < config.min_examples or not negatives:
        return None
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    matrix = vectorizer.fit_transform([trace.utterance for trace in traces])
    labels = [
        1 if trace.teacher_frame and trace.teacher_frame.intent == intent else 0
        for trace in traces
    ]
    classifier = LogisticRegression(random_state=config.random_state, max_iter=1000)
    classifier.fit(matrix, labels)
    probabilities = classifier.predict_proba(matrix)[:, list(classifier.classes_).index(1)]
    threshold, metrics = _select_threshold(
        probabilities=probabilities,
        labels=labels,
        min_accuracy=config.min_accuracy,
    )
    if threshold is None:
        return None
    return IntentBinaryExpert(
        intent=intent,
        vectorizer=vectorizer,
        classifier=classifier,
        threshold=threshold,
        validation_metrics=metrics,
    )


def _train_slot_expert(
    traces: list[TeacherTrace],
    *,
    slot_key: str,
    config: L2ExpertTrainingConfig,
) -> SlotValueExpert | None:
    values: Counter[str] = Counter(
        trace.teacher_frame.slots[slot_key]
        for trace in traces
        if trace.teacher_frame is not None and slot_key in trace.teacher_frame.slots
    )
    if sum(values.values()) < config.min_examples:
        return None
    values_by_normalized_value = {
        normalize_utterance(value): value
        for value, _count in values.most_common()
        if normalize_utterance(value)
    }
    if not values_by_normalized_value:
        return None
    metrics = _slot_expert_metrics(traces, slot_key, values_by_normalized_value)
    if metrics["accepted"] == 0 or metrics["accepted_accuracy"] < config.min_accuracy:
        return None
    return SlotValueExpert(
        slot_key=slot_key,
        values_by_normalized_value=values_by_normalized_value,
        validation_metrics=metrics,
    )


def _select_threshold(
    *,
    probabilities: Any,
    labels: list[int],
    min_accuracy: float,
) -> tuple[float | None, dict[str, Any]]:
    candidates = []
    for threshold in [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]:
        accepted_indices = [
            index for index, probability in enumerate(probabilities) if probability >= threshold
        ]
        accepted = len(accepted_indices)
        correct = sum(labels[index] == 1 for index in accepted_indices)
        wrong = accepted - correct
        accuracy = correct / accepted if accepted else 1.0
        candidates.append(
            {
                "threshold": threshold,
                "accepted": accepted,
                "correct_accepts": correct,
                "wrong_accepts": wrong,
                "accepted_accuracy": accuracy,
                "coverage": accepted / len(labels) if labels else 0.0,
            }
        )
    passing = [
        candidate
        for candidate in candidates
        if candidate["accepted"] > 0 and candidate["accepted_accuracy"] >= min_accuracy
    ]
    if not passing:
        return None, {
            "accepted": 0,
            "accepted_accuracy": 0.0,
            "wrong_accepts": 0,
            "coverage": 0.0,
            "candidates": candidates,
        }
    selected = max(
        passing,
        key=lambda candidate: (
            candidate["coverage"],
            candidate["accepted_accuracy"],
            -candidate["threshold"],
        ),
    )
    return float(selected["threshold"]), {**selected, "candidates": candidates}


def _slot_expert_metrics(
    traces: list[TeacherTrace],
    slot_key: str,
    values_by_normalized_value: dict[str, str],
) -> dict[str, Any]:
    accepted = 0
    correct = 0
    wrong = 0
    for trace in traces:
        if trace.teacher_frame is None:
            continue
        predicted = _slot_value_from_utterance(trace.utterance, values_by_normalized_value)
        if predicted is None:
            continue
        accepted += 1
        expected = trace.teacher_frame.slots.get(slot_key)
        if expected == predicted:
            correct += 1
        else:
            wrong += 1
    labeled = sum(1 for trace in traces if trace.teacher_frame is not None)
    return {
        "accepted": accepted,
        "correct_accepts": correct,
        "wrong_accepts": wrong,
        "accepted_accuracy": correct / accepted if accepted else 1.0,
        "coverage": accepted / labeled if labeled else 0.0,
        "value_count": len(values_by_normalized_value),
    }


def _slot_value_from_utterance(
    utterance: str,
    values_by_normalized_value: dict[str, str],
) -> str | None:
    normalized_utterance = normalize_utterance(utterance)
    for normalized_value, canonical_value in sorted(
        values_by_normalized_value.items(),
        key=lambda item: (-len(item[0]), item[0]),
    ):
        if normalized_value and normalized_value in normalized_utterance:
            return canonical_value
    return None


def _positive_probability(classifier: LogisticRegression, matrix: Any) -> float:
    positive_index = list(classifier.classes_).index(1)
    return float(classifier.predict_proba(matrix)[0][positive_index])
