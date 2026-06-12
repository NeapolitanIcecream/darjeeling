import pytest
from pydantic import ValidationError

from darjeeling.contracts import LayerResult, TeacherTrace, TraceRecord, to_teacher_trace
from darjeeling.runtime.exact_cache import ExactJsonCacheLayer, exact_cache_from_teacher_traces
from darjeeling.runtime.router import JsonCascadeRouter


class _NeutralTarget:
    name = "neutral"
    schema_version = "neutral-v1"

    def normalize_request(self, input: dict) -> str:
        return str(input["text"]).strip().lower()

    def validate_output(self, output: dict, task_schema: dict) -> None:
        if output.get("label") not in task_schema["labels"]:
            raise ValueError("unsupported label")


class _RejectLayer:
    layer_name = "L0"

    def try_answer(self, input: dict) -> LayerResult:
        return LayerResult(
            layer=self.layer_name,
            accepted=False,
            output=None,
            reason=f"no match for {input['text']}",
            latency_ms=0.1,
        )


class _AcceptLayer:
    layer_name = "L1"

    def try_answer(self, input: dict) -> LayerResult:
        return LayerResult(
            layer=self.layer_name,
            accepted=True,
            output={"label": f"seen:{input['text']}"},
            confidence=0.9,
            latency_ms=1.0,
        )


def test_teacher_trace_omits_private_label() -> None:
    trace = TraceRecord(
        request_id="r1",
        input={"text": "alpha"},
        gold_label={"label": "private"},
        teacher_label={"label": "public"},
        chosen_layer="L1",
        final_output={"label": "public"},
        layer_results=[
            LayerResult(
                layer="L1",
                accepted=True,
                output={"label": "public"},
                latency_ms=1.0,
            )
        ],
    )

    teacher_trace = to_teacher_trace(trace)

    assert "gold_label" not in TeacherTrace.model_fields
    assert teacher_trace.model_dump() == {
        "request_id": "r1",
        "input": {"text": "alpha"},
        "teacher_label": {"label": "public"},
        "chosen_layer": "L1",
        "final_output": {"label": "public"},
        "layer_results": [
            {
                "layer": "L1",
                "accepted": True,
                "output": {"label": "public"},
                "confidence": None,
                "reason": "",
                "latency_ms": 1.0,
                "cost_usd": 0.0,
                "metadata": {},
            }
        ],
        "l4_usage": {},
        "timestamp": trace.timestamp,
    }


def test_teacher_trace_rejects_private_label_field() -> None:
    trace = TraceRecord(
        request_id="r1",
        input={"text": "alpha"},
        gold_label={"label": "private"},
        teacher_label={"label": "public"},
        chosen_layer="L1",
        final_output={"label": "public"},
        layer_results=[],
    ).model_dump()
    trace["gold_label"] = {"label": "private"}

    with pytest.raises(ValidationError, match="gold_label"):
        TeacherTrace.model_validate(trace)


def test_json_cascade_router_routes_opaque_payloads() -> None:
    router = JsonCascadeRouter([_RejectLayer(), _AcceptLayer()])

    output, results = router.route({"text": "alpha"})

    assert output == {"label": "seen:alpha"}
    assert [result.layer for result in results] == ["L0", "L1"]
    assert results[0].accepted is False
    assert results[1].output == {"label": "seen:alpha"}


def test_exact_json_cache_layer_uses_target_normalization() -> None:
    layer = ExactJsonCacheLayer(
        {"alpha": {"label": "A"}},
        target=_NeutralTarget(),
    )

    hit = layer.try_answer({"text": "  Alpha  "})
    miss = layer.try_answer({"text": "Beta"})

    assert hit.accepted is True
    assert hit.output == {"label": "A"}
    assert hit.metadata == {"normalized_request": "alpha"}
    assert miss.accepted is False
    assert miss.output is None
    assert miss.metadata == {"normalized_request": "beta"}


def test_exact_cache_from_teacher_traces_uses_teacher_labels() -> None:
    traces = [
        TeacherTrace(
            request_id="r1",
            input={"text": " Alpha "},
            teacher_label={"label": "A"},
            chosen_layer="L4",
            final_output={"label": "A"},
            layer_results=[],
            timestamp="2026-06-12T00:00:00+00:00",
        ),
        TeacherTrace(
            request_id="r2",
            input={"text": "Beta"},
            teacher_label=None,
            chosen_layer="L4",
            final_output={"label": "B"},
            layer_results=[],
            timestamp="2026-06-12T00:00:00+00:00",
        ),
    ]

    cache = exact_cache_from_teacher_traces(
        traces,
        target=_NeutralTarget(),
        task_schema={"labels": ["A"]},
    )

    assert cache == {"alpha": {"label": "A"}}
