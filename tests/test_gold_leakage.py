from darjeeling.compiler.loop import assert_teacher_visible_only, compiler_inputs_from_traces
from darjeeling.schemas import Frame, LayerResult, TeacherTrace, TraceRecord


def test_compiler_inputs_do_not_contain_gold_frame() -> None:
    trace = TraceRecord(
        request_id="r1",
        utterance="alpha request for seven",
        gold_frame=Frame(intent="intent_alpha", slots={"time": "seven"}),
        teacher_frame=Frame(intent="intent_alpha", slots={"time": "7:00"}),
        chosen_layer="L4",
        final_frame=Frame(intent="intent_alpha", slots={"time": "7:00"}),
        layer_results=[
            LayerResult(
                layer="L4",
                accepted=True,
                frame=Frame(intent="intent_alpha", slots={"time": "7:00"}),
                latency_ms=10.0,
            )
        ],
    )

    compiler_inputs = compiler_inputs_from_traces([trace])

    assert isinstance(compiler_inputs[0], TeacherTrace)
    assert "gold_frame" not in TeacherTrace.model_fields
    assert "gold_frame" not in compiler_inputs[0].model_dump()
    assert_teacher_visible_only(compiler_inputs)
