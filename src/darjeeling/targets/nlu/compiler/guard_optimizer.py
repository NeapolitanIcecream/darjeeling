from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from darjeeling.schemas import TeacherTrace

GUARD_PROPOSAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "threshold_grid_start",
        "threshold_grid_stop",
        "threshold_grid_steps",
        "max_wrong_accept_rate",
    ],
    "properties": {
        "threshold_grid_start": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "threshold_grid_stop": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "threshold_grid_steps": {"type": "integer", "minimum": 1, "maximum": 64},
        "max_wrong_accept_rate": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "rationale": {"type": "string"},
    },
}


@dataclass(frozen=True)
class GuardSearchSpec:
    schema_version: str = "guard-search-v1"
    threshold_grid_start: float = 0.70
    threshold_grid_stop: float = 0.98
    threshold_grid_steps: int = 8
    max_wrong_accept_rate: float = 0.05
    rationale: str = "deterministic default"

    @property
    def grid(self) -> list[float]:
        return threshold_grid(
            start=self.threshold_grid_start,
            stop=self.threshold_grid_stop,
            steps=self.threshold_grid_steps,
        )


@dataclass(frozen=True)
class L2ThresholdEvaluation:
    threshold: float
    coverage: float
    accepted_accuracy: float
    wrong_accept_rate: float
    correct_accepts: int
    wrong_accepts: int
    accepted: int
    total: int


@dataclass(frozen=True)
class L2ThresholdSelection:
    threshold: float
    evaluation: L2ThresholdEvaluation
    candidates: list[L2ThresholdEvaluation]


@dataclass(frozen=True)
class L2PredictionRecord:
    guard_probability: float
    correct: bool


def threshold_grid(start: float = 0.7, stop: float = 0.98, steps: int = 8) -> list[float]:
    if steps <= 1:
        return [stop]
    stride = (stop - start) / (steps - 1)
    return [round(start + stride * idx, 4) for idx in range(steps)]


def guard_search_spec_from_proposal(
    proposal: dict[str, Any],
    *,
    default_max_wrong_accept_rate: float = 0.05,
) -> GuardSearchSpec:
    start = _bounded_float(
        proposal.get("threshold_grid_start", 0.70),
        field_name="threshold_grid_start",
    )
    stop = _bounded_float(
        proposal.get("threshold_grid_stop", 0.98),
        field_name="threshold_grid_stop",
    )
    if stop < start:
        raise ValueError("threshold_grid_stop must be >= threshold_grid_start")

    steps = proposal.get("threshold_grid_steps", 8)
    if not isinstance(steps, int) or isinstance(steps, bool):
        raise ValueError("threshold_grid_steps must be an integer")
    if not 1 <= steps <= 64:
        raise ValueError("threshold_grid_steps must be in [1, 64]")

    max_wrong_accept_rate = _bounded_float(
        proposal.get("max_wrong_accept_rate", default_max_wrong_accept_rate),
        field_name="max_wrong_accept_rate",
    )
    rationale = proposal.get("rationale", "")
    if rationale is None:
        rationale = ""
    if not isinstance(rationale, str):
        raise ValueError("rationale must be a string")
    return GuardSearchSpec(
        threshold_grid_start=start,
        threshold_grid_stop=stop,
        threshold_grid_steps=steps,
        max_wrong_accept_rate=max_wrong_accept_rate,
        rationale=rationale,
    )


def select_l2_accept_threshold(
    bundle: Any,
    traces: list[TeacherTrace],
    *,
    grid: list[float] | None = None,
    max_wrong_accept_rate: float = 0.05,
    min_accepted_accuracy: float = 0.93,
) -> L2ThresholdSelection | None:
    labeled = [trace for trace in traces if trace.teacher_frame is not None]
    if not labeled:
        return None

    prediction_records = _l2_prediction_records(bundle, labeled)
    thresholds = grid or threshold_grid()
    thresholds = _augment_thresholds_from_records(prediction_records, thresholds)
    candidates = [
        evaluate_l2_threshold_records(prediction_records, threshold=threshold)
        for threshold in thresholds
    ]
    eligible = [
        candidate
        for candidate in candidates
        if candidate.wrong_accept_rate <= max_wrong_accept_rate
        and candidate.accepted_accuracy >= min_accepted_accuracy
    ]
    zero_wrong_eligible = [
        candidate
        for candidate in eligible
        if candidate.accepted > 0 and candidate.wrong_accepts == 0
    ]
    if zero_wrong_eligible:
        selected = max(
            zero_wrong_eligible,
            key=lambda item: (
                item.coverage,
                item.accepted_accuracy,
                item.threshold,
            ),
        )
    elif eligible:
        selected = max(
            eligible,
            key=lambda item: (
                item.coverage,
                item.accepted_accuracy,
                -item.wrong_accept_rate,
                item.threshold,
            ),
        )
    else:
        selected = max(
            candidates,
            key=lambda item: (
                -item.wrong_accept_rate,
                item.accepted_accuracy,
                item.coverage,
                item.threshold,
            ),
        )
    return L2ThresholdSelection(
        threshold=selected.threshold,
        evaluation=selected,
        candidates=candidates,
    )


def _augment_thresholds_from_records(
    records: list[L2PredictionRecord],
    thresholds: list[float],
) -> list[float]:
    augmented = {round(_clamp_threshold(threshold), 6) for threshold in thresholds}
    for record in records:
        probability = _clamp_threshold(record.guard_probability)
        augmented.add(round(probability, 6))
        augmented.add(round(_clamp_threshold(probability + 1e-6), 6))
    return sorted(augmented)


def evaluate_l2_unguarded(bundle: Any, traces: list[TeacherTrace]) -> L2ThresholdEvaluation:
    """Evaluate L2 as if the accept threshold did not block predictions."""

    return evaluate_l2_threshold(bundle, traces, threshold=0.0)


def evaluate_l2_threshold(
    bundle: Any,
    traces: list[TeacherTrace],
    *,
    threshold: float,
) -> L2ThresholdEvaluation:
    return evaluate_l2_threshold_records(
        _l2_prediction_records(bundle, traces),
        threshold=threshold,
    )


def _l2_prediction_records(
    bundle: Any,
    traces: list[TeacherTrace],
) -> list[L2PredictionRecord]:
    records: list[L2PredictionRecord] = []
    for trace in traces:
        if trace.teacher_frame is None:
            continue
        prediction = bundle.predict(trace.utterance)
        records.append(
            L2PredictionRecord(
                guard_probability=_clamp_threshold(float(prediction.guard_probability)),
                correct=prediction.frame == trace.teacher_frame,
            )
        )
    return records


def evaluate_l2_threshold_records(
    records: list[L2PredictionRecord],
    *,
    threshold: float,
) -> L2ThresholdEvaluation:
    total = 0
    accepted = 0
    correct_accepts = 0
    wrong_accepts = 0
    for record in records:
        total += 1
        if record.guard_probability < threshold:
            continue
        accepted += 1
        if record.correct:
            correct_accepts += 1
        else:
            wrong_accepts += 1

    coverage = accepted / total if total else 0.0
    accepted_accuracy = correct_accepts / accepted if accepted else 1.0
    wrong_accept_rate = wrong_accepts / total if total else 0.0
    return L2ThresholdEvaluation(
        threshold=threshold,
        coverage=coverage,
        accepted_accuracy=accepted_accuracy,
        wrong_accept_rate=wrong_accept_rate,
        correct_accepts=correct_accepts,
        wrong_accepts=wrong_accepts,
        accepted=accepted,
        total=total,
    )


def _bounded_float(value: Any, *, field_name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be a number")
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be in [0, 1]")
    return value


def _clamp_threshold(value: float) -> float:
    return min(1.0, max(0.0, value))
