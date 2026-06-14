from darjeeling.targets.nlu.compiler.l3_residual_gate import evaluate_l3_residual_value
from darjeeling.targets.nlu.schemas import (
    Frame,
    FramePatch,
    LayerResult,
    TraceRecord,
    traces_to_teacher_view,
)
from darjeeling.targets.nlu.settings import load_settings


def test_l3_residual_gate_passes_when_residual_accepts_are_accurate_and_valuable() -> None:
    frame = Frame(intent="intent_beta")
    trace = TraceRecord(
        request_id="r1",
        utterance="beta request",
        teacher_frame=frame,
        chosen_layer="L3",
        final_frame=frame,
        layer_results=[
            LayerResult(layer="L0", accepted=False, latency_ms=0.1),
            LayerResult(layer="L1", accepted=False, latency_ms=1.0),
            LayerResult(layer="L2", accepted=False, latency_ms=5.0),
            LayerResult(layer="L3", accepted=True, frame=frame, latency_ms=40.0),
        ],
    )

    gate = evaluate_l3_residual_value(
        traces_to_teacher_view([trace]),
        settings=load_settings(),
    )

    assert gate["passes_gate"] is True
    assert gate["coverage"] == 1.0
    assert gate["accepted_accuracy"] == 1.0
    assert gate["wrong_accept_rate"] == 0.0
    assert gate["expected_latency_value_ms_per_request"] > 0.0


def test_l3_residual_gate_skips_when_l3_has_no_residual_accepts() -> None:
    frame = Frame(intent="intent_beta")
    trace = TraceRecord(
        request_id="r1",
        utterance="beta request",
        teacher_frame=frame,
        chosen_layer="L4",
        final_frame=frame,
        layer_results=[
            LayerResult(layer="L0", accepted=False, latency_ms=0.1),
            LayerResult(layer="L1", accepted=False, latency_ms=1.0),
            LayerResult(layer="L2", accepted=False, latency_ms=5.0),
            LayerResult(layer="L3", accepted=False, latency_ms=40.0),
            LayerResult(layer="L4", accepted=True, frame=frame, latency_ms=900.0),
        ],
    )

    gate = evaluate_l3_residual_value(
        traces_to_teacher_view([trace]),
        settings=load_settings(),
    )

    assert gate["passes_gate"] is False
    assert gate["reason"] == "L3 accepted no residual requests"


def test_l3_residual_gate_does_not_count_l2_patch_handled_request_as_residual() -> None:
    frame = Frame(intent="intent_beta")
    trace = TraceRecord(
        request_id="r1",
        utterance="beta request",
        teacher_frame=frame,
        chosen_layer="L3",
        final_frame=frame,
        layer_results=[
            LayerResult(layer="L0", accepted=False, latency_ms=0.1),
            LayerResult(layer="L1", accepted=False, latency_ms=1.0),
            LayerResult(
                layer="L2",
                accepted=True,
                patch=FramePatch(accepted_intent="intent_beta", source_layer="L2"),
                latency_ms=5.0,
            ),
            LayerResult(layer="L3", accepted=True, frame=frame, latency_ms=40.0),
        ],
    )

    gate = evaluate_l3_residual_value(
        traces_to_teacher_view([trace]),
        settings=load_settings(),
    )

    assert gate["residual_requests"] == 0
    assert gate["passes_gate"] is False
    assert gate["reason"] == "no L2 residual evidence"
