from pathlib import Path

import pytest

from darjeeling.targets.nlu.layers.l1_rust_programbank import (
    RustL1Worker,
    RustProgramBankLayer,
    build_l1_binary,
)


@pytest.fixture(scope="module")
def l1_binary() -> Path:
    return build_l1_binary(Path("tests/fixtures/l1_neutral_programbank"))


@pytest.fixture(scope="module")
def empty_l1_binary() -> Path:
    return build_l1_binary(Path("native/l1_empty_programbank"))


def test_default_empty_rust_l1_worker_abstains(empty_l1_binary: Path) -> None:
    with RustL1Worker(empty_l1_binary) as worker:
        response = worker.answer("alpha accept red", request_id="r-empty")

    assert response.request_id == "r-empty"
    assert not response.accepted
    assert response.frame is None
    assert response.program_path == "abstain"


def test_rust_l1_worker_answers_neutral_alpha_request(l1_binary: Path) -> None:
    with RustL1Worker(l1_binary) as worker:
        response = worker.answer("alpha accept red", request_id="r1")

    assert response.request_id == "r1"
    assert response.accepted
    assert response.frame is not None
    assert response.frame.intent == "intent_alpha"
    assert response.frame.slots == {"slot_alpha": "red"}
    assert response.native_latency_us >= 0
    assert response.program_path


@pytest.mark.parametrize(
    "utterance",
    [
        "alpha accept one two three four",
        "alpha accept red/blue",
        "beta accept red",
        "alpha request red",
        "gamma request",
    ],
)
def test_rust_l1_worker_abstains_outside_fixture_contract(
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
        result = layer.try_answer("beta request")

    assert result.layer == "L1"
    assert not result.accepted
    assert result.frame is None
    assert "native_latency_us" in result.metadata


def test_legacy_l1_rust_module_reexports_nlu_target_layer() -> None:
    from darjeeling.layers.l1_rust_programbank import (
        RustProgramBankLayer as LegacyRustProgramBankLayer,
    )

    assert LegacyRustProgramBankLayer is RustProgramBankLayer
