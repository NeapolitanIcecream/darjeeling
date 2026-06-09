from types import SimpleNamespace

import pytest

from darjeeling.compiler.guard_optimizer import (
    evaluate_l2_unguarded,
    guard_search_spec_from_proposal,
    select_l2_accept_threshold,
)
from darjeeling.compiler.l2_distiller import l2_config_from_proposal
from darjeeling.layers.l2_student import GuardDecision, guard_accepts
from darjeeling.schemas import Frame, TeacherTrace


def test_l2_guard_accepts_at_threshold() -> None:
    assert guard_accepts(0.93, 0.93)
    assert not guard_accepts(0.92, 0.93)
    assert GuardDecision(probability=0.99, threshold=0.93).accepted


def test_l2_guard_optimizer_prefers_highest_safe_coverage_threshold() -> None:
    predictions = {
        "correct-high": SimpleNamespace(
            frame=Frame(intent="music_play"),
            guard_probability=0.95,
        ),
        "correct-mid": SimpleNamespace(
            frame=Frame(intent="alarm_set"),
            guard_probability=0.82,
        ),
        "wrong-low": SimpleNamespace(
            frame=Frame(intent="music_play"),
            guard_probability=0.72,
        ),
    }
    bundle = SimpleNamespace(predict=lambda utterance: predictions[utterance])
    traces = [
        _teacher_trace("r1", "correct-high", "music_play"),
        _teacher_trace("r2", "correct-mid", "alarm_set"),
        _teacher_trace("r3", "wrong-low", "alarm_set"),
    ]

    selection = select_l2_accept_threshold(
        bundle,
        traces,
        grid=[0.7, 0.8, 0.9],
        max_wrong_accept_rate=0.0,
    )

    assert selection is not None
    assert selection.threshold == 0.82
    assert selection.evaluation.coverage == 2 / 3
    assert selection.evaluation.wrong_accept_rate == 0.0


def test_l2_guard_optimizer_uses_observed_probabilities_between_grid_points() -> None:
    predictions = {
        "correct": SimpleNamespace(
            frame=Frame(intent="music_play"),
            guard_probability=0.81,
        ),
        "wrong": SimpleNamespace(
            frame=Frame(intent="music_play"),
            guard_probability=0.80,
        ),
    }
    bundle = SimpleNamespace(predict=lambda utterance: predictions[utterance])
    traces = [
        _teacher_trace("r1", "correct", "music_play"),
        _teacher_trace("r2", "wrong", "alarm_set"),
    ]

    selection = select_l2_accept_threshold(
        bundle,
        traces,
        grid=[0.7, 0.9],
        max_wrong_accept_rate=0.0,
    )

    assert selection is not None
    assert selection.threshold == 0.81
    assert selection.evaluation.accepted == 1
    assert selection.evaluation.wrong_accept_rate == 0.0


def test_l2_guard_optimizer_enforces_minimum_accepted_accuracy() -> None:
    predictions = {
        "correct-high": SimpleNamespace(
            frame=Frame(intent="music_play"),
            guard_probability=0.90,
        ),
        "wrong-mid": SimpleNamespace(
            frame=Frame(intent="music_play"),
            guard_probability=0.80,
        ),
        "correct-low": SimpleNamespace(
            frame=Frame(intent="alarm_set"),
            guard_probability=0.10,
        ),
    }
    bundle = SimpleNamespace(predict=lambda utterance: predictions[utterance])
    traces = [
        _teacher_trace("r1", "correct-high", "music_play"),
        _teacher_trace("r2", "wrong-mid", "alarm_set"),
        _teacher_trace("r3", "correct-low", "alarm_set"),
    ]

    selection = select_l2_accept_threshold(
        bundle,
        traces,
        grid=[0.0, 0.7],
        max_wrong_accept_rate=0.50,
        min_accepted_accuracy=0.90,
    )

    assert selection is not None
    assert selection.threshold >= 0.800001
    assert selection.evaluation.accepted == 1
    assert selection.evaluation.accepted_accuracy == 1.0


def test_l2_guard_optimizer_prefers_zero_observed_wrong_accepts() -> None:
    predictions = {
        "correct-high": SimpleNamespace(
            frame=Frame(intent="music_play"),
            guard_probability=0.90,
        ),
        "wrong-mid": SimpleNamespace(
            frame=Frame(intent="music_play"),
            guard_probability=0.80,
        ),
        "correct-low": SimpleNamespace(
            frame=Frame(intent="alarm_set"),
            guard_probability=0.70,
        ),
    }
    bundle = SimpleNamespace(predict=lambda utterance: predictions[utterance])
    traces = [
        _teacher_trace("r1", "correct-high", "music_play"),
        _teacher_trace("r2", "wrong-mid", "alarm_set"),
        _teacher_trace("r3", "correct-low", "alarm_set"),
    ]

    selection = select_l2_accept_threshold(
        bundle,
        traces,
        grid=[0.7],
        max_wrong_accept_rate=0.50,
        min_accepted_accuracy=0.50,
    )

    assert selection is not None
    assert selection.threshold >= 0.800001
    assert selection.evaluation.wrong_accepts == 0


def test_l2_unguarded_evaluation_reports_threshold_zero_accuracy() -> None:
    predictions = {
        "correct": SimpleNamespace(
            frame=Frame(intent="music_play"),
            guard_probability=0.10,
        ),
        "wrong": SimpleNamespace(
            frame=Frame(intent="music_play"),
            guard_probability=0.05,
        ),
    }
    bundle = SimpleNamespace(predict=lambda utterance: predictions[utterance])
    traces = [
        _teacher_trace("r1", "correct", "music_play"),
        _teacher_trace("r2", "wrong", "alarm_set"),
    ]

    evaluation = evaluate_l2_unguarded(bundle, traces)

    assert evaluation.threshold == 0.0
    assert evaluation.accepted == 2
    assert evaluation.accepted_accuracy == 0.5
    assert evaluation.wrong_accept_rate == 0.5


def test_l2_config_from_proposal_accepts_only_bounded_fields() -> None:
    config = l2_config_from_proposal(
        {
            "slot_model_family": "none",
            "intent_model_family": "mlp",
            "mlp_hidden_layer_sizes": [32, 16],
            "mlp_alpha": 0.001,
            "mlp_early_stopping": True,
            "word_ngram_range": [1, 3],
            "char_ngram_range": [2, 4],
            "max_features": 1000,
            "ignored_field": "not allowed",
        }
    )

    assert config.slot_model_family == "none"
    assert config.intent_model_family == "mlp"
    assert config.mlp_hidden_layer_sizes == (32, 16)
    assert config.mlp_alpha == 0.001
    assert config.mlp_early_stopping is True
    assert config.word_ngram_range == (1, 3)
    assert config.char_ngram_range == (2, 4)
    assert config.max_features == 1000

    with pytest.raises(ValueError):
        l2_config_from_proposal({"slot_model_family": "unsupported"})
    with pytest.raises(ValueError):
        l2_config_from_proposal(
            {"slot_model_family": "token_sgd", "mlp_hidden_layer_sizes": [0]}
        )


def test_guard_search_spec_from_proposal_bounds_grid_and_wrong_accept_rate() -> None:
    spec = guard_search_spec_from_proposal(
        {
            "threshold_grid_start": 0.6,
            "threshold_grid_stop": 0.9,
            "threshold_grid_steps": 4,
            "max_wrong_accept_rate": 0.03,
            "rationale": "tighten guard",
        }
    )

    assert spec.grid == [0.6, 0.7, 0.8, 0.9]
    assert spec.max_wrong_accept_rate == 0.03
    assert spec.rationale == "tighten guard"

    with pytest.raises(ValueError):
        guard_search_spec_from_proposal(
            {
                "threshold_grid_start": 0.9,
                "threshold_grid_stop": 0.7,
                "threshold_grid_steps": 4,
                "max_wrong_accept_rate": 0.03,
            }
        )


def _teacher_trace(request_id: str, utterance: str, intent: str) -> TeacherTrace:
    return TeacherTrace(
        request_id=request_id,
        utterance=utterance,
        teacher_frame=Frame(intent=intent),
        chosen_layer="L4",
        final_frame=Frame(intent=intent),
        layer_results=[],
        timestamp="2026-06-08T00:00:00Z",
    )
