import pytest

from darjeeling.layers.l3_local_slm import (
    L3LocalSLMLayer,
    LocalSLMConfig,
    LocalSLMLoadError,
    benchmark_l3_layer,
    parse_l3_output,
    validate_l3_output,
)
from darjeeling.layers.l4_cloud_llm import TaskSchema


class FakeBackend:
    def __init__(self, output: str | None = None, error: Exception | None = None) -> None:
        self.output = output
        self.error = error
        self.calls = 0

    def generate(self, prompt: str, config: LocalSLMConfig) -> str:
        self.calls += 1
        if self.error is not None:
            raise self.error
        assert "Current utterance" in prompt
        return self.output or "{}"

    def status(self) -> dict:
        return {
            "model_name": "fake",
            "actual_device": "fake-device",
            "loaded": self.calls > 0,
        }


def _task_schema() -> TaskSchema:
    return TaskSchema(intent_names=["intent_alpha", "intent_beta"], slot_names=["time"])


def test_l3_disabled_does_not_call_backend() -> None:
    backend = FakeBackend(output='{"intent": "intent_beta", "confidence": 1.0}')
    layer = L3LocalSLMLayer(
        config=LocalSLMConfig(mode="disabled"),
        task_schema=_task_schema(),
        backend=backend,
    )

    result = layer.try_answer("beta request")

    assert not result.accepted
    assert result.reason == "local SLM disabled"
    assert result.metadata["load_attempted"] is False
    assert backend.calls == 0


def test_l3_shadow_load_failure_degrades_to_disabled() -> None:
    layer = L3LocalSLMLayer(
        config=LocalSLMConfig(mode="shadow"),
        task_schema=_task_schema(),
        backend=FakeBackend(error=LocalSLMLoadError("model unavailable")),
    )

    result = layer.try_answer("beta request")

    assert not result.accepted
    assert result.metadata["requested_mode"] == "shadow"
    assert result.metadata["actual_mode"] == "disabled"
    assert result.metadata["error"] == "model unavailable"


def test_l3_guarded_load_failure_is_fail_fast() -> None:
    layer = L3LocalSLMLayer(
        config=LocalSLMConfig(mode="guarded"),
        task_schema=_task_schema(),
        backend=FakeBackend(error=LocalSLMLoadError("model unavailable")),
    )

    with pytest.raises(LocalSLMLoadError):
        layer.try_answer("beta request")


def test_l3_guarded_accepts_valid_high_confidence_frame() -> None:
    layer = L3LocalSLMLayer(
        config=LocalSLMConfig(mode="guarded", confidence_threshold=0.7),
        task_schema=_task_schema(),
        backend=FakeBackend(
            output='{"intent": "intent_alpha", "slots": {"time": "seven"}, "confidence": 0.91}'
        ),
    )

    result = layer.try_answer("alpha request for seven")

    assert result.accepted
    assert result.frame is not None
    assert result.frame.intent == "intent_alpha"
    assert result.frame.slots == {"time": "seven"}
    assert result.metadata["repair_used"] is False


def test_l3_shadow_never_accepts_even_when_frame_would_pass_guard() -> None:
    layer = L3LocalSLMLayer(
        config=LocalSLMConfig(mode="shadow", confidence_threshold=0.7),
        task_schema=_task_schema(),
        backend=FakeBackend(output='{"intent": "intent_beta", "slots": {}, "confidence": 0.91}'),
    )

    result = layer.try_answer("beta request")

    assert not result.accepted
    assert result.frame is None
    assert result.metadata["shadow_frame"] == {
        "intent": "intent_beta",
        "slots": {},
        "is_abstain": False,
    }
    assert result.metadata["would_accept"] is True
    assert result.reason == "shadow local SLM would accept"


def test_l3_parser_repairs_json_and_validates_schema() -> None:
    parsed = parse_l3_output(
        '{"intent": "intent_alpha", "slots": {"time": "seven",}, "confidence": 0.91}'
    )
    invalid = parse_l3_output(
        '{"intent": "unknown", "slots": {"unsupported": "x"}, "confidence": 0.91}'
    )

    assert parsed.repair_used
    assert parsed.frame.intent == "intent_alpha"
    assert parsed.frame.slots == {"time": "seven"}
    assert validate_l3_output(parsed, _task_schema()) == []
    assert validate_l3_output(invalid, _task_schema()) == [
        "intent not allowed: unknown",
        "slots not allowed: ['unsupported']",
    ]


def test_l3_benchmark_layer_records_latency_backend_and_parse_stats() -> None:
    layer = L3LocalSLMLayer(
        config=LocalSLMConfig(mode="shadow", confidence_threshold=0.7),
        task_schema=_task_schema(),
        backend=FakeBackend(
            output='{"intent": "intent_alpha", "slots": {"time": "seven"}, "confidence": 0.91}'
        ),
    )

    metrics = benchmark_l3_layer(layer, ["alpha request for seven"])

    assert metrics["schema_version"] == "l3-benchmark-v1"
    assert metrics["status"] == "success"
    assert metrics["requests"] == 1
    assert metrics["would_accept"] == 1
    assert metrics["parse_failures"] == 0
    assert metrics["backend"]["actual_device"] == "fake-device"
    assert metrics["request_results"][0]["confidence"] == 0.91
