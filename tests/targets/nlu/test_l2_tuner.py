from __future__ import annotations

from darjeeling.targets.nlu.compiler.l2_tuner import (
    L2TuneSpec,
    residual_l2_validation_traces,
    split_l2_tune_traces,
    tune_l2_student,
)
from darjeeling.targets.nlu.layers.l2_student import L2StudentConfig
from darjeeling.targets.nlu.schemas import Frame, LayerResult, TeacherTrace


def test_l2_tuner_selects_best_config_and_reports_trials() -> None:
    result = tune_l2_student(
        _teacher_traces(),
        base_config=L2StudentConfig(min_examples=4, max_iter=100),
        spec=L2TuneSpec(
            n_trials=3,
            validation_fraction=0.25,
            random_state=7,
            latency_weight=0.0,
        ),
    )

    assert result.schema_version == "l2-tune-v1"
    assert result.train_size >= 8
    assert result.validation_size >= 2
    assert result.split_policy == "chronological"
    assert result.n_trials_requested == 3
    assert result.n_trials_completed >= 1
    assert result.best_trial_number is not None
    assert result.best_value is not None
    assert result.best_config is not None
    assert result.best_config["intent_model_family"] in {"sgd_logreg", "mlp"}
    assert result.best_metrics is not None
    assert result.validation_residual_size == result.validation_size
    assert result.objective_validation_size == result.validation_size
    assert result.objective_validation_source == "residual"
    assert result.best_metrics["unguarded"]["total"] == result.objective_validation_size
    assert len(result.trials) == 3
    assert any(trial.config == result.best_config for trial in result.trials)


def test_l2_tuner_disables_mlp_early_stopping_for_small_samples() -> None:
    result = tune_l2_student(
        _teacher_traces(),
        base_config=L2StudentConfig(min_examples=4, max_iter=100),
        spec=L2TuneSpec(
            n_trials=9,
            validation_fraction=0.25,
            random_state=17,
            latency_weight=0.0,
        ),
    )

    assert [trial.state for trial in result.trials] == ["COMPLETE"] * 9
    mlp_configs = [
        trial.config
        for trial in result.trials
        if trial.config is not None and trial.config["intent_model_family"] == "mlp"
    ]
    assert mlp_configs
    assert all(config["mlp_early_stopping"] is False for config in mlp_configs)


def test_l2_tune_split_keeps_each_intent_in_train_and_validation() -> None:
    train, validation = split_l2_tune_traces(
        _teacher_traces(),
        validation_fraction=0.25,
        split_policy="stratified_random",
        random_state=17,
    )

    train_intents = {trace.teacher_frame.intent for trace in train if trace.teacher_frame}
    validation_intents = {
        trace.teacher_frame.intent for trace in validation if trace.teacher_frame
    }

    assert train_intents == {"intent_alpha", "intent_beta"}
    assert validation_intents == {"intent_alpha", "intent_beta"}
    assert {trace.request_id for trace in train}.isdisjoint(
        {trace.request_id for trace in validation}
    )


def test_l2_tune_split_defaults_to_chronological_holdout() -> None:
    train, validation = split_l2_tune_traces(
        _teacher_traces(),
        validation_fraction=0.25,
        random_state=17,
    )

    assert [trace.request_id for trace in train] == [
        "m1",
        "m2",
        "m3",
        "m4",
        "m5",
        "m6",
        "a1",
        "a2",
        "a3",
    ]
    assert [trace.request_id for trace in validation] == ["a4", "a5", "a6"]


def test_l2_residual_validation_filters_lower_layer_hits_and_l0_repeats() -> None:
    train = [
        _trace("t1", "beta request", "intent_beta"),
        _trace("t2", "alpha request", "intent_alpha"),
    ]
    validation = [
        _trace("v1", "beta request", "intent_beta"),
        _trace("v2", "beta variant request", "intent_beta", lower_layer="L0"),
        _trace("v3", "alpha wake", "intent_alpha", lower_layer="L1"),
        _trace("v4", "beta novel request", "intent_beta"),
    ]

    residual = residual_l2_validation_traces(train, validation)

    assert [trace.request_id for trace in residual] == ["v4"]


def _teacher_traces() -> list[TeacherTrace]:
    rows = [
        ("m1", "beta request", "intent_beta", {}),
        ("m2", "beta request", "intent_beta", {}),
        ("m3", "beta alternate request", "intent_beta", {}),
        ("m4", "beta collection request", "intent_beta", {}),
        ("m5", "beta variant request", "intent_beta", {}),
        ("m6", "beta alternate request", "intent_beta", {}),
        ("a1", "alpha request value alpha", "intent_alpha", {"slot_alpha": "value alpha"}),
        ("a2", "alpha variant value beta", "intent_alpha", {"slot_alpha": "value beta"}),
        ("a3", "alpha variant value gamma", "intent_alpha", {"slot_alpha": "value gamma"}),
        ("a4", "alpha variant one", "intent_alpha", {}),
        ("a5", "alpha value delta", "intent_alpha", {"slot_beta": "value delta"}),
        ("a6", "alpha variant two", "intent_alpha", {}),
    ]
    return [
        _trace(request_id, utterance, intent, slots=slots)
        for request_id, utterance, intent, slots in rows
    ]


def _trace(
    request_id: str,
    utterance: str,
    intent: str,
    *,
    slots: dict[str, str] | None = None,
    lower_layer: str | None = None,
) -> TeacherTrace:
    frame = Frame(intent=intent, slots=slots or {})
    layer_results = []
    if lower_layer is not None:
        layer_results.append(
            LayerResult(
                layer=lower_layer,
                accepted=True,
                frame=frame,
                latency_ms=0.1,
            )
        )
    return TeacherTrace(
        request_id=request_id,
        utterance=utterance,
        teacher_frame=frame,
        chosen_layer="L4",
        final_frame=frame,
        layer_results=layer_results,
        timestamp="2026-06-09T00:00:00Z",
    )
