from __future__ import annotations

from collections.abc import Mapping, Sequence

from darjeeling.contracts import JsonObject, LayerResult, TargetSpec, TeacherTrace


class ExactJsonCacheLayer:
    layer_name = "L0"

    def __init__(
        self,
        outputs_by_normalized_request: Mapping[str, JsonObject],
        *,
        target: TargetSpec,
    ) -> None:
        self.outputs_by_normalized_request = dict(outputs_by_normalized_request)
        self.target = target

    def try_answer(self, input: JsonObject) -> LayerResult:
        normalized = self.target.normalize_request(input)
        output = self.outputs_by_normalized_request.get(normalized)
        if output is None:
            return LayerResult(
                layer=self.layer_name,
                accepted=False,
                reason="exact cache miss",
                latency_ms=0.0,
                metadata={"normalized_request": normalized},
            )
        return LayerResult(
            layer=self.layer_name,
            accepted=True,
            output=dict(output),
            confidence=1.0,
            reason="exact cache hit",
            latency_ms=0.0,
            metadata={"normalized_request": normalized},
        )


def exact_cache_from_teacher_traces(
    traces: Sequence[TeacherTrace],
    *,
    target: TargetSpec,
    task_schema: JsonObject | None = None,
) -> dict[str, JsonObject]:
    cache: dict[str, JsonObject] = {}
    for trace in traces:
        if trace.teacher_label is None:
            continue
        if task_schema is not None:
            target.validate_output(trace.teacher_label, task_schema)
        cache[target.normalize_request(trace.input)] = dict(trace.teacher_label)
    return cache
