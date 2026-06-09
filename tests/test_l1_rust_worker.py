from pathlib import Path

import pytest

from darjeeling.layers.l1_rust_programbank import (
    RustL1Worker,
    RustProgramBankLayer,
    build_l1_binary,
)


@pytest.fixture(scope="module")
def l1_binary() -> Path:
    return build_l1_binary(Path("native/l1_programbank"))


def test_rust_l1_worker_answers_alarm_request(l1_binary: Path) -> None:
    with RustL1Worker(l1_binary) as worker:
        response = worker.answer("set an alarm for seven", request_id="r1")

    assert response.request_id == "r1"
    assert response.accepted
    assert response.frame is not None
    assert response.frame.intent == "alarm_set"
    assert response.frame.slots == {"time": "seven"}
    assert response.native_latency_us >= 0
    assert response.program_path


@pytest.mark.parametrize(
    "utterance",
    [
        "weather in leisure city",
        "what will the weather be on friday",
        "show me what alarm times i've set for the week",
        "please set an alarm at seven am tomorrow morning",
        "please set a reminder alarm for three p. m. on saturday",
    ],
)
def test_rust_l1_worker_abstains_when_rule_would_need_unsupported_slots(
    l1_binary: Path,
    utterance: str,
) -> None:
    with RustL1Worker(l1_binary) as worker:
        response = worker.answer(utterance, request_id="r-risk")

    assert not response.accepted
    assert response.frame is None


def test_rust_l1_layer_abstains_on_unknown_request(l1_binary: Path) -> None:
    with RustL1Worker(l1_binary) as worker:
        layer = RustProgramBankLayer(worker)
        result = layer.try_answer("play some jazz")

    assert result.layer == "L1"
    assert not result.accepted
    assert result.frame is None
    assert "native_latency_us" in result.metadata
