import pytest

from darjeeling.compiler.l3_prompt_optimizer import (
    calibrate_l3_confidence_threshold,
    l3_prompt_artifact_from_proposal,
    l3_prompt_artifact_hash,
    replay_l3_prompt_artifact,
)
from darjeeling.layers.l3_local_slm import L3PromptArtifact, LocalSLMConfig
from darjeeling.layers.l4_cloud_llm import TaskSchema
from darjeeling.schemas import Frame, LayerResult, TraceRecord, traces_to_teacher_view


class FakeL3Backend:
    def generate(self, prompt: str, config: LocalSLMConfig) -> str:
        assert "Return strict JSON" in prompt
        return '{"intent": "music_play", "slots": {}, "confidence": 0.93}'

    def status(self) -> dict:
        return {"model_name": "fake", "actual_device": "fake-device", "loaded": True}


def test_l3_prompt_artifact_from_proposal_expands_teacher_visible_examples() -> None:
    trace = TraceRecord(
        request_id="r1",
        utterance="set alarm for seven",
        gold_frame=Frame(intent="alarm_set", slots={"time": "gold-seven"}),
        teacher_frame=Frame(intent="alarm_set", slots={"time": "seven"}),
        chosen_layer="L4",
        final_frame=Frame(intent="alarm_set", slots={"time": "seven"}),
        layer_results=[
            LayerResult(
                layer="L4",
                accepted=True,
                frame=Frame(intent="alarm_set", slots={"time": "seven"}),
                latency_ms=1.0,
            )
        ],
    )

    artifact = l3_prompt_artifact_from_proposal(
        {
            "system_prompt": "Return strict JSON only.",
            "confidence_threshold": 0.81,
            "few_shot_trace_ids": ["r1"],
        },
        traces=traces_to_teacher_view([trace]),
        prompt_version="candidate-v1",
    )

    assert artifact.prompt_version == "candidate-v1"
    assert artifact.confidence_threshold == 0.81
    assert artifact.few_shot_examples == [
        {
            "trace_id": "r1",
            "utterance": "set alarm for seven",
            "frame": {
                "intent": "alarm_set",
                "slots": {"time": "seven"},
                "is_abstain": False,
            },
        }
    ]
    assert "gold-seven" not in artifact.model_dump_json()


def test_l3_prompt_artifact_rejects_unknown_few_shot_trace_id() -> None:
    with pytest.raises(ValueError, match="not teacher-visible"):
        l3_prompt_artifact_from_proposal(
            {
                "system_prompt": "Return JSON.",
                "few_shot_trace_ids": ["missing"],
            },
            traces=[],
            prompt_version="candidate-v1",
        )


def test_l3_guard_calibration_selects_safe_confidence_threshold() -> None:
    traces = [
        _l3_shadow_trace("r1", confidence=0.9, predicted=Frame(intent="music_play")),
        _l3_shadow_trace("r2", confidence=0.8, predicted=Frame(intent="alarm_set")),
        _l3_shadow_trace("r3", confidence=0.4, predicted=Frame(intent="music_play")),
    ]

    result = calibrate_l3_confidence_threshold(traces, max_wrong_accept_rate=0.0)

    assert result is not None
    assert result.threshold == 0.9
    assert result.accepted_count == 1
    assert result.wrong_accept_count == 0
    assert result.accepted_accuracy == 1.0


def test_l3_prompt_replay_scores_generated_shadow_outputs() -> None:
    trace = TraceRecord(
        request_id="r1",
        utterance="play jazz",
        teacher_frame=Frame(intent="music_play"),
        chosen_layer="L4",
        final_frame=Frame(intent="music_play"),
        layer_results=[],
    )

    prompt_artifact = L3PromptArtifact(
        prompt_version="candidate-v1",
        system_prompt="Return strict JSON.",
        confidence_threshold=0.8,
    )
    payload = replay_l3_prompt_artifact(
        prompt_artifact=prompt_artifact,
        traces=[trace],
        task_schema=TaskSchema(intent_names=["music_play"], slot_names=[]),
        config=LocalSLMConfig(mode="shadow", confidence_threshold=0.8),
        backend=FakeL3Backend(),
    )

    assert payload["schema_version"] == "l3-prompt-replay-v1"
    assert payload["status"] == "success"
    assert payload["prompt_version"] == "candidate-v1"
    assert payload["prompt_sha256"] == l3_prompt_artifact_hash(prompt_artifact)
    assert payload["requests"] == 1
    assert payload["would_accept_count"] == 1
    assert payload["correct_accept_count"] == 1
    assert payload["accepted_accuracy"] == 1.0
    assert payload["wrong_accept_rate"] == 0.0
    assert payload["request_results"][0]["predicted_frame"]["intent"] == "music_play"


def _l3_shadow_trace(
    request_id: str,
    *,
    confidence: float,
    predicted: Frame,
) -> TraceRecord:
    return TraceRecord(
        request_id=request_id,
        utterance=f"{request_id} utterance",
        teacher_frame=Frame(intent="music_play"),
        chosen_layer="L4",
        final_frame=Frame(intent="music_play"),
        layer_results=[
            LayerResult(
                layer="L3",
                accepted=False,
                reason="shadow local SLM would accept",
                latency_ms=1.0,
                confidence=confidence,
                metadata={
                    "confidence": confidence,
                    "shadow_frame": predicted.model_dump(mode="json"),
                    "validation_errors": [],
                },
            )
        ],
    )
