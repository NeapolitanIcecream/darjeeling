from __future__ import annotations

import random
import time
from collections import defaultdict
from typing import Any, Literal

from pydantic import BaseModel, Field

from darjeeling.compiler.guard_optimizer import (
    L2ThresholdEvaluation,
    evaluate_l2_unguarded,
    select_l2_accept_threshold,
)
from darjeeling.layers.l2_student import (
    L2StudentBundle,
    L2StudentConfig,
    train_l2_student,
    training_examples_from_teacher_traces,
)
from darjeeling.schemas import TeacherTrace


class L2TuneSpec(BaseModel):
    n_trials: int = Field(default=16, ge=1)
    timeout_s: float | None = Field(default=None, gt=0.0)
    validation_fraction: float = Field(default=0.25, gt=0.0, lt=0.8)
    split_policy: Literal["chronological", "stratified_random"] = "chronological"
    random_state: int = 17
    search_space: Literal["compact", "wide"] = "compact"
    max_wrong_accept_rate: float = Field(default=0.05, ge=0.0, le=1.0)
    min_accepted_accuracy: float = Field(default=0.93, ge=0.0, le=1.0)
    latency_weight: float = Field(default=0.01, ge=0.0)


class L2TuneTrialReport(BaseModel):
    number: int
    state: str
    value: float | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] | None = None
    unguarded: dict[str, Any] | None = None
    guarded: dict[str, Any] | None = None
    p95_latency_ms: float | None = None
    error: str | None = None


class L2TuneResult(BaseModel):
    schema_version: str = "l2-tune-v1"
    train_size: int
    validation_size: int
    split_policy: str
    n_trials_requested: int
    n_trials_completed: int
    best_trial_number: int | None = None
    best_value: float | None = None
    best_config: dict[str, Any] | None = None
    best_metrics: dict[str, Any] | None = None
    trials: list[L2TuneTrialReport]


def tune_l2_student(
    traces: list[TeacherTrace],
    *,
    base_config: L2StudentConfig | None = None,
    spec: L2TuneSpec | None = None,
) -> L2TuneResult:
    """Tune L2 hyperparameters using only teacher-visible traces."""

    spec = spec or L2TuneSpec()
    base_config = base_config or L2StudentConfig()
    train_traces, validation_traces = split_l2_tune_traces(
        traces,
        validation_fraction=spec.validation_fraction,
        split_policy=spec.split_policy,
        random_state=spec.random_state,
    )
    train_examples = training_examples_from_teacher_traces(train_traces)
    if len(train_examples) < base_config.min_examples:
        raise ValueError(
            f"L2 tuning requires at least {base_config.min_examples} training examples "
            f"after validation split; got {len(train_examples)}"
        )
    if len({example.teacher_frame.intent for example in train_examples}) < 2:
        raise ValueError("L2 tuning requires at least two teacher intents in training split")

    import optuna

    trial_reports: dict[int, L2TuneTrialReport] = {}

    def objective(trial: optuna.Trial) -> float:
        config = _sample_l2_config(
            trial,
            base_config=base_config,
            spec=spec,
            train_size=len(train_examples),
        )
        try:
            bundle = train_l2_student(train_examples, config)
            unguarded = evaluate_l2_unguarded(bundle, validation_traces)
            selection = select_l2_accept_threshold(
                bundle,
                validation_traces,
                max_wrong_accept_rate=spec.max_wrong_accept_rate,
                min_accepted_accuracy=spec.min_accepted_accuracy,
            )
            guarded = selection.evaluation if selection is not None else None
            p95_latency_ms = _p95_prediction_latency_ms(bundle, validation_traces)
            value = _l2_tune_objective(
                unguarded=unguarded,
                guarded=guarded,
                p95_latency_ms=p95_latency_ms,
                latency_weight=spec.latency_weight,
            )
            trial_reports[trial.number] = L2TuneTrialReport(
                number=trial.number,
                state="COMPLETE",
                value=value,
                params=dict(trial.params),
                config=config.model_dump(mode="json"),
                unguarded=_threshold_evaluation_payload(unguarded),
                guarded=_threshold_evaluation_payload(guarded) if guarded is not None else None,
                p95_latency_ms=p95_latency_ms,
            )
            return value
        except Exception as exc:
            trial_reports[trial.number] = L2TuneTrialReport(
                number=trial.number,
                state="FAIL",
                value=-1_000_000.0,
                params=dict(trial.params),
                config=config.model_dump(mode="json"),
                error=str(exc),
            )
            return -1_000_000.0

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=spec.random_state),
    )
    study.optimize(objective, n_trials=spec.n_trials, timeout=spec.timeout_s)

    reports = [
        trial_reports.get(
            trial.number,
            L2TuneTrialReport(
                number=trial.number,
                state=trial.state.name,
                value=trial.value,
                params=dict(trial.params),
            ),
        )
        for trial in study.trials
    ]
    completed_reports = [
        report
        for report in reports
        if report.state == "COMPLETE" and report.value is not None and report.config is not None
    ]
    best_report = (
        max(completed_reports, key=lambda report: report.value) if completed_reports else None
    )
    return L2TuneResult(
        train_size=len(train_examples),
        validation_size=len(training_examples_from_teacher_traces(validation_traces)),
        split_policy=spec.split_policy,
        n_trials_requested=spec.n_trials,
        n_trials_completed=len(completed_reports),
        best_trial_number=best_report.number if best_report is not None else None,
        best_value=best_report.value if best_report is not None else None,
        best_config=best_report.config if best_report is not None else None,
        best_metrics={
            "unguarded": best_report.unguarded,
            "guarded": best_report.guarded,
            "p95_latency_ms": best_report.p95_latency_ms,
        }
        if best_report is not None
        else None,
        trials=reports,
    )


def split_l2_tune_traces(
    traces: list[TeacherTrace],
    *,
    validation_fraction: float,
    split_policy: Literal["chronological", "stratified_random"] = "chronological",
    random_state: int,
) -> tuple[list[TeacherTrace], list[TeacherTrace]]:
    labeled = [trace for trace in traces if trace.teacher_frame is not None]
    if len(labeled) < 4:
        raise ValueError("L2 tuning requires at least four labeled teacher traces")

    by_intent: dict[str, list[TeacherTrace]] = defaultdict(list)
    for trace in labeled:
        assert trace.teacher_frame is not None
        by_intent[trace.teacher_frame.intent].append(trace)
    if len(by_intent) < 2:
        raise ValueError("L2 tuning requires at least two teacher intents")

    if split_policy == "chronological":
        validation_count = max(1, round(len(labeled) * validation_fraction))
        validation_count = min(validation_count, len(labeled) - 1)
        train = labeled[:-validation_count]
        validation = labeled[-validation_count:]
        if len({trace.teacher_frame.intent for trace in train if trace.teacher_frame}) < 2:
            raise ValueError("L2 tuning chronological train split has fewer than two intents")
        if not validation:
            raise ValueError("L2 tuning validation split is empty")
        return train, validation

    if split_policy != "stratified_random":
        raise ValueError(f"unsupported L2 tuning split_policy: {split_policy}")

    rng = random.Random(random_state)
    train: list[TeacherTrace] = []
    validation: list[TeacherTrace] = []
    for _intent, intent_traces in sorted(by_intent.items()):
        shuffled = list(intent_traces)
        rng.shuffle(shuffled)
        if len(shuffled) == 1:
            train.extend(shuffled)
            continue
        validation_count = max(1, round(len(shuffled) * validation_fraction))
        validation_count = min(validation_count, len(shuffled) - 1)
        validation.extend(shuffled[:validation_count])
        train.extend(shuffled[validation_count:])

    train.sort(key=lambda trace: trace.request_id)
    validation.sort(key=lambda trace: trace.request_id)
    if not validation:
        raise ValueError("L2 tuning validation split is empty")
    return train, validation


def _sample_l2_config(
    trial: Any,
    *,
    base_config: L2StudentConfig,
    spec: L2TuneSpec,
    train_size: int,
) -> L2StudentConfig:
    intent_model_family = trial.suggest_categorical(
        "intent_model_family",
        ["sgd_logreg", "mlp"],
    )
    slot_model_family = trial.suggest_categorical(
        "slot_model_family",
        ["token_sgd", "none"],
    )
    frame_source = trial.suggest_categorical(
        "frame_source",
        ["retrieval", "student"],
    )
    max_features_choices = (
        [1_000, 3_000, 5_000, 10_000]
        if spec.search_space == "compact"
        else [1_000, 3_000, 5_000, 10_000, 25_000, 50_000]
    )
    max_features = trial.suggest_categorical("max_features", max_features_choices)
    word_upper = trial.suggest_int("word_ngram_upper", 1, 4 if spec.search_space == "wide" else 3)
    char_lower = trial.suggest_int("char_ngram_lower", 2, 4)
    char_upper_limit = 6 if spec.search_space == "wide" else 5
    char_upper = trial.suggest_int("char_ngram_upper", char_lower, char_upper_limit)
    max_iter = trial.suggest_categorical(
        "max_iter",
        [200, 300, 500] if spec.search_space == "compact" else [200, 300, 500, 1000],
    )
    hidden_size_text = trial.suggest_categorical(
        "mlp_hidden_layer_sizes",
        ["32", "64", "128", "64,32"] if spec.search_space == "wide" else ["32", "64"],
    )
    mlp_early_stopping = False
    if intent_model_family == "mlp" and spec.search_space == "wide" and train_size >= 200:
        mlp_early_stopping = trial.suggest_categorical("mlp_early_stopping", [False, True])
    return base_config.model_copy(
        update={
            "intent_model_family": intent_model_family,
            "slot_model_family": slot_model_family,
            "frame_source": frame_source,
            "max_features": int(max_features),
            "max_iter": int(max_iter),
            "word_ngram_range": (1, int(word_upper)),
            "char_ngram_range": (int(char_lower), int(char_upper)),
            "mlp_hidden_layer_sizes": _parse_hidden_layer_sizes(hidden_size_text),
            "mlp_alpha": trial.suggest_float("mlp_alpha", 1e-5, 1e-2, log=True),
            "mlp_early_stopping": mlp_early_stopping,
        }
    )


def _parse_hidden_layer_sizes(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(",") if part)


def _l2_tune_objective(
    *,
    unguarded: L2ThresholdEvaluation,
    guarded: L2ThresholdEvaluation | None,
    p95_latency_ms: float,
    latency_weight: float,
) -> float:
    guarded_quality = 0.0
    wrong_accept_penalty = 0.0
    if guarded is not None:
        guarded_quality = guarded.coverage * guarded.accepted_accuracy
        wrong_accept_penalty = guarded.wrong_accept_rate
    return (
        unguarded.accepted_accuracy
        + guarded_quality
        - 2.0 * wrong_accept_penalty
        - latency_weight * p95_latency_ms
    )


def _p95_prediction_latency_ms(bundle: L2StudentBundle, traces: list[TeacherTrace]) -> float:
    latencies: list[float] = []
    for trace in traces:
        if trace.teacher_frame is None:
            continue
        started = time.perf_counter()
        bundle.predict(trace.utterance)
        latencies.append((time.perf_counter() - started) * 1000.0)
    if not latencies:
        return 0.0
    latencies.sort()
    index = min(len(latencies) - 1, int(round(0.95 * (len(latencies) - 1))))
    return latencies[index]


def _threshold_evaluation_payload(evaluation: L2ThresholdEvaluation) -> dict[str, Any]:
    return {
        "threshold": evaluation.threshold,
        "coverage": evaluation.coverage,
        "accepted_accuracy": evaluation.accepted_accuracy,
        "wrong_accept_rate": evaluation.wrong_accept_rate,
        "correct_accepts": evaluation.correct_accepts,
        "wrong_accepts": evaluation.wrong_accepts,
        "accepted": evaluation.accepted,
        "total": evaluation.total,
    }
