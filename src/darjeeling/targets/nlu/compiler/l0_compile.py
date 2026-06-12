from __future__ import annotations

from darjeeling.contracts import TeacherTrace as CoreTeacherTrace
from darjeeling.runtime.exact_cache import exact_cache_from_teacher_traces as exact_json_cache
from darjeeling.schemas import TeacherTrace
from darjeeling.targets.nlu.schemas import Frame
from darjeeling.targets.nlu.target import NluTargetSpec


def exact_cache_from_teacher_traces(traces: list[TeacherTrace]) -> dict[str, Frame]:
    cache = exact_json_cache(
        [_core_teacher_trace_from_legacy_trace(trace) for trace in traces],
        target=NluTargetSpec(),
    )
    return {
        normalized_request: Frame.model_validate(output)
        for normalized_request, output in cache.items()
    }


def _core_teacher_trace_from_legacy_trace(trace: TeacherTrace) -> CoreTeacherTrace:
    return CoreTeacherTrace(
        request_id=trace.request_id,
        input={"utterance": trace.utterance},
        teacher_label=(
            trace.teacher_frame.model_dump(mode="json")
            if trace.teacher_frame is not None
            else None
        ),
        chosen_layer=trace.chosen_layer,
        final_output=trace.final_frame.model_dump(mode="json"),
        layer_results=[],
        l4_usage=trace.l4_usage,
        timestamp=trace.timestamp,
    )
