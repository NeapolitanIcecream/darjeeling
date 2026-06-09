from __future__ import annotations

from darjeeling.compiler.l2_tuner import L2TuneSpec, split_l2_tune_traces, tune_l2_student
from darjeeling.layers.l2_student import L2StudentConfig
from darjeeling.schemas import Frame, TeacherTrace


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
    assert result.n_trials_requested == 3
    assert result.n_trials_completed >= 1
    assert result.best_trial_number is not None
    assert result.best_value is not None
    assert result.best_config is not None
    assert result.best_config["intent_model_family"] in {"sgd_logreg", "mlp"}
    assert result.best_metrics is not None
    assert result.best_metrics["unguarded"]["total"] == result.validation_size
    assert len(result.trials) == 3
    assert any(trial.config == result.best_config for trial in result.trials)


def test_l2_tune_split_keeps_each_intent_in_train_and_validation() -> None:
    train, validation = split_l2_tune_traces(
        _teacher_traces(),
        validation_fraction=0.25,
        random_state=17,
    )

    train_intents = {trace.teacher_frame.intent for trace in train if trace.teacher_frame}
    validation_intents = {
        trace.teacher_frame.intent for trace in validation if trace.teacher_frame
    }

    assert train_intents == {"alarm_set", "music_play"}
    assert validation_intents == {"alarm_set", "music_play"}
    assert {trace.request_id for trace in train}.isdisjoint(
        {trace.request_id for trace in validation}
    )


def _teacher_traces() -> list[TeacherTrace]:
    rows = [
        ("m1", "play jazz", "music_play", {}),
        ("m2", "play music", "music_play", {}),
        ("m3", "start my playlist", "music_play", {}),
        ("m4", "play songs", "music_play", {}),
        ("m5", "play rock", "music_play", {}),
        ("m6", "start smooth jazz", "music_play", {}),
        ("a1", "set alarm for seven", "alarm_set", {"time": "seven"}),
        ("a2", "wake me at eight", "alarm_set", {"time": "eight"}),
        ("a3", "alarm at nine", "alarm_set", {"time": "nine"}),
        ("a4", "set morning alarm", "alarm_set", {}),
        ("a5", "wake me tomorrow", "alarm_set", {"date": "tomorrow"}),
        ("a6", "set evening alarm", "alarm_set", {}),
    ]
    return [
        TeacherTrace(
            request_id=request_id,
            utterance=utterance,
            teacher_frame=Frame(intent=intent, slots=slots),
            chosen_layer="L4",
            final_frame=Frame(intent=intent, slots=slots),
            layer_results=[],
            timestamp="2026-06-09T00:00:00Z",
        )
        for request_id, utterance, intent, slots in rows
    ]
