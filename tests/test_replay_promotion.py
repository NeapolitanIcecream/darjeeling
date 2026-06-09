from pathlib import Path

import darjeeling.compiler.replay as replay_module
from darjeeling.artifacts.store import LayerDelta
from darjeeling.compiler.objective import ObjectiveMetrics
from darjeeling.compiler.replay import (
    OfflineArtifactSet,
    decide_artifact_set_promotion,
    decide_promotion,
    evaluate_offline_artifact_set,
)
from darjeeling.runtime.cost import ReplayCostModel
from darjeeling.schemas import Frame, LayerResult, TeacherTrace


def test_replay_promotes_only_when_objective_improves_within_gates() -> None:
    current = ObjectiveMetrics(
        frame_exact_match=0.95,
        wrong_accept_rate=0.02,
        cost_usd_per_100_requests=1.0,
        p95_latency_ms=900.0,
    )
    candidate = ObjectiveMetrics(
        frame_exact_match=0.95,
        wrong_accept_rate=0.01,
        cost_usd_per_100_requests=0.25,
        p95_latency_ms=500.0,
    )

    decision = decide_promotion(current, candidate)

    assert decision.promoted
    assert decision.candidate_score > decision.current_score


def test_replay_rejects_wrong_accept_regression() -> None:
    current = ObjectiveMetrics(
        frame_exact_match=0.95,
        wrong_accept_rate=0.02,
        cost_usd_per_100_requests=1.0,
        p95_latency_ms=900.0,
    )
    candidate = ObjectiveMetrics(
        frame_exact_match=0.97,
        wrong_accept_rate=0.20,
        cost_usd_per_100_requests=0.1,
        p95_latency_ms=300.0,
    )

    decision = decide_promotion(current, candidate, max_wrong_accept_rate=0.05)

    assert not decision.promoted
    assert decision.reason == "wrong_accept_rate exceeds limit"


def test_artifact_set_promotion_blocks_layer_regression_by_default() -> None:
    current = ObjectiveMetrics(
        frame_exact_match=0.95,
        wrong_accept_rate=0.02,
        cost_usd_per_100_requests=1.0,
        p95_latency_ms=900.0,
    )
    candidate = ObjectiveMetrics(
        frame_exact_match=0.96,
        wrong_accept_rate=0.01,
        cost_usd_per_100_requests=0.25,
        p95_latency_ms=500.0,
    )

    decision = decide_artifact_set_promotion(
        current,
        candidate,
        per_layer_deltas={
            "L1": LayerDelta(
                coverage_delta=0.10,
                accepted_accuracy_delta=-0.03,
                wrong_accept_delta=0.0,
            ),
            "L2": LayerDelta(coverage_delta=0.20, accepted_accuracy_delta=0.04),
        },
    )

    assert not decision.promoted
    assert decision.reason == "per-layer regression gate failed: L1"
    assert not decision.promoted_with_layer_regression
    assert decision.regressed_layers == ["L1"]


def test_artifact_set_promotion_can_record_layer_regression_without_blocking() -> None:
    current = ObjectiveMetrics(
        frame_exact_match=0.95,
        wrong_accept_rate=0.02,
        cost_usd_per_100_requests=1.0,
        p95_latency_ms=900.0,
    )
    candidate = ObjectiveMetrics(
        frame_exact_match=0.96,
        wrong_accept_rate=0.01,
        cost_usd_per_100_requests=0.25,
        p95_latency_ms=500.0,
    )

    decision = decide_artifact_set_promotion(
        current,
        candidate,
        per_layer_deltas={
            "L1": LayerDelta(
                coverage_delta=0.10,
                accepted_accuracy_delta=-0.03,
                wrong_accept_delta=0.0,
            ),
        },
        block_layer_regressions=False,
    )

    assert decision.promoted
    assert decision.reason == "objective improved within gates"
    assert decision.promoted_with_layer_regression
    assert decision.regressed_layers == ["L1"]


def test_large_exact_l0_cache_can_promote_when_it_reduces_l4_without_regression() -> None:
    traces = [
        TeacherTrace(
            request_id=f"r{i}",
            utterance=f"cached request {i}",
            teacher_frame=Frame(intent="cached", slots={"i": str(i)}),
            chosen_layer="L4",
            final_frame=Frame(intent="cached", slots={"i": str(i)}),
            layer_results=[],
            timestamp="2026-06-08T00:00:00Z",
        )
        for i in range(300)
    ]
    l0_cache = {
        f"cached request {i}": Frame(intent="cached", slots={"i": str(i)})
        for i in range(125)
    }
    l0_cache.update(
        {
            f"extra cached request {i}": Frame(intent="cached_extra", slots={"i": str(i)})
            for i in range(2_282)
        }
    )

    current = evaluate_offline_artifact_set(traces, OfflineArtifactSet(l0_cache={}))
    candidate = evaluate_offline_artifact_set(
        traces,
        OfflineArtifactSet(l0_cache=l0_cache),
    )
    decision = decide_artifact_set_promotion(
        current.objective,
        candidate.objective,
        per_layer_deltas={},
    )

    assert candidate.objective.artifact_complexity < 50.0
    assert (
        candidate.objective.cost_usd_per_100_requests
        < current.objective.cost_usd_per_100_requests
    )
    assert decision.promoted


def test_offline_replay_counts_recorded_l3_accepts_and_wrong_accepts() -> None:
    traces = [
        TeacherTrace(
            request_id="r1",
            utterance="play music",
            teacher_frame=Frame(intent="music_play"),
            chosen_layer="L3",
            final_frame=Frame(intent="music_play"),
            layer_results=[
                LayerResult(
                    layer="L3",
                    accepted=True,
                    frame=Frame(intent="music_play"),
                    latency_ms=120.0,
                )
            ],
            timestamp="2026-06-08T00:00:00Z",
        ),
        TeacherTrace(
            request_id="r2",
            utterance="set alarm",
            teacher_frame=Frame(intent="alarm_set"),
            chosen_layer="L3",
            final_frame=Frame(intent="music_play"),
            layer_results=[
                LayerResult(
                    layer="L3",
                    accepted=True,
                    frame=Frame(intent="music_play"),
                    latency_ms=120.0,
                )
            ],
            timestamp="2026-06-08T00:00:01Z",
        ),
    ]

    result = evaluate_offline_artifact_set(traces, OfflineArtifactSet(l0_cache={}))

    assert result.layer_counts["L3"] == 2
    assert result.objective.frame_exact_match == 0.5
    assert result.objective.wrong_accept_rate == 0.5
    assert result.layer_metrics["L3"]["accepted_accuracy"] == 0.5


def test_offline_replay_uses_artifact_l1_worker_timeout(monkeypatch) -> None:
    captured: dict[str, float] = {}

    class FakeWorker:
        def __init__(self, binary_path: Path, *, timeout_s: float) -> None:
            captured["timeout_s"] = timeout_s

        def start(self) -> None:
            return None

        def answer(self, utterance: str):
            return type("FakeL1Response", (), {"accepted": False, "frame": None})()

        def close(self) -> None:
            return None

    monkeypatch.setattr(replay_module, "build_l1_binary", lambda crate_dir: crate_dir / "worker")
    monkeypatch.setattr(replay_module, "RustL1Worker", FakeWorker)

    result = evaluate_offline_artifact_set(
        [
            TeacherTrace(
                request_id="r1",
                utterance="unknown request",
                teacher_frame=Frame(intent="music_play"),
                chosen_layer="L4",
                final_frame=Frame(intent="music_play"),
                layer_results=[],
                timestamp="2026-06-08T00:00:00Z",
            )
        ],
        OfflineArtifactSet(
            l0_cache={},
            l1_crate_dir=Path("native/l1_programbank"),
            l1_worker_timeout_s=12.5,
        ),
    )

    assert captured["timeout_s"] == 12.5
    assert result.layer_counts["L4"] == 1


def test_offline_replay_uses_trace_l4_usage_for_cost() -> None:
    result = evaluate_offline_artifact_set(
        [
            TeacherTrace(
                request_id="r1",
                utterance="unknown request",
                teacher_frame=Frame(intent="music_play"),
                chosen_layer="L4",
                final_frame=Frame(intent="music_play"),
                layer_results=[],
                l4_usage={"prompt_tokens": 1_000_000, "completion_tokens": 500_000},
                timestamp="2026-06-08T00:00:00Z",
            )
        ],
        OfflineArtifactSet(l0_cache={}),
        cost_model=ReplayCostModel(
            l4_input_usd_per_million=2.0,
            l4_output_usd_per_million=8.0,
            l4_default_cost_usd_per_request=0.01,
        ),
    )

    assert result.layer_counts["L4"] == 1
    assert result.objective.cost_usd_per_100_requests == 600.0
    assert result.layer_metrics["L4"]["cost_usd_per_100_requests"] == 600.0
