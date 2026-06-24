from pathlib import Path

import pytest

from darjeeling.targets.nlu.compiler.l1_program_compiler import (
    _build_l1_agent_prompt,
    _constraints_text,
)
from darjeeling.targets.nlu.layers.l1_rust_programbank import (
    RustL1Worker,
    RustProgramBankLayer,
    build_l1_binary,
)
from darjeeling.targets.nlu.settings import DEFAULT_NLU_L1_CRATE_DIR


@pytest.fixture(scope="module")
def l1_binary() -> Path:
    return build_l1_binary(Path("tests/fixtures/l1_neutral_programbank"))


@pytest.fixture(scope="module")
def empty_l1_binary() -> Path:
    return build_l1_binary(DEFAULT_NLU_L1_CRATE_DIR)


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


def test_rust_l1_worker_accepts_patch_only_response(l1_binary: Path) -> None:
    with RustL1Worker(l1_binary) as worker:
        response = worker.answer("alpha intent", request_id="r-patch")

    assert response.request_id == "r-patch"
    assert response.accepted
    assert response.frame is None
    assert response.patch is not None
    assert response.patch.accepted_intent == "intent_alpha"
    assert response.patch.accepted_slots == {}
    assert response.patch.source_layer == "L1"
    assert response.patch.complete is False


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


def test_rust_l1_layer_accepts_patch_only_response(l1_binary: Path) -> None:
    with RustL1Worker(l1_binary) as worker:
        layer = RustProgramBankLayer(worker)
        result = layer.try_answer("alpha intent")

    assert result.layer == "L1"
    assert result.accepted
    assert result.frame is None
    assert result.patch is not None
    assert result.patch.accepted_intent == "intent_alpha"
    assert result.metadata["frame_patch"]["accepted_intent"] == "intent_alpha"


def test_l1_agent_prompt_allows_large_hard_coded_rust_programbank() -> None:
    prompt = _build_l1_agent_prompt(
        context_dir=Path("contexts"),
        workspace_crate_dir=Path("l1_programbank"),
    )
    constraints = _constraints_text()

    assert "Large, repetitive, CPU-native Rust ProgramBank logic is allowed" in prompt
    assert "large hard-coded Rust tables" in constraints
    assert "Do not modify the outer evaluator" in constraints
    assert "teacher cache" in constraints
