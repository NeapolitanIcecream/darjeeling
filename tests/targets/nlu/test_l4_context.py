import pytest

from darjeeling.settings import load_settings
from darjeeling.targets.nlu.compiler.l4_context import (
    L4ContextError,
    assert_no_forbidden_context,
    build_proposal_context,
    build_teacher_context,
)
from darjeeling.targets.nlu.schemas import (
    Frame,
    LayerResult,
    TaskSchema,
    TraceRecord,
    traces_to_teacher_view,
)


def _task_schema() -> TaskSchema:
    return TaskSchema(intent_names=["intent_alpha", "intent_beta"], slot_names=["slot_alpha"])


def test_teacher_context_keeps_stable_prefix_when_utterance_changes() -> None:
    settings = load_settings()
    first = build_teacher_context(
        utterance="beta request",
        task_schema=_task_schema(),
        settings=settings,
    )
    second = build_teacher_context(
        utterance="alpha request value alpha",
        task_schema=_task_schema(),
        settings=settings,
    )

    assert first.stable_prefix == second.stable_prefix
    assert first.dynamic_tail != second.dynamic_tail
    assert first.context_hash != second.context_hash
    assert first.prompt_cache_key == second.prompt_cache_key
    assert first.prompt_cache_key.startswith("darjeeling:teacher-v1:")


def test_proposal_context_uses_teacher_visible_traces_only() -> None:
    trace = TraceRecord(
        request_id="r1",
        utterance="alpha request value alpha",
        gold_frame=Frame(intent="intent_alpha", slots={"slot_alpha": "gold-value-alpha"}),
        teacher_frame=Frame(intent="intent_alpha", slots={"slot_alpha": "value alpha"}),
        chosen_layer="L4",
        final_frame=Frame(intent="intent_alpha", slots={"slot_alpha": "value alpha"}),
        layer_results=[
            LayerResult(
                layer="L4",
                accepted=True,
                frame=Frame(intent="intent_alpha", slots={"slot_alpha": "value alpha"}),
                latency_ms=1.0,
            )
        ],
    )

    rendered = build_proposal_context(
        role="l2",
        task_schema=_task_schema(),
        settings=load_settings(),
        traces=traces_to_teacher_view([trace]),
        output_schema={"type": "object", "required": ["family"]},
    )

    assert rendered.source_trace_ids == ["r1"]
    assert "gold_frame" not in rendered.dynamic_tail
    assert "gold-value-alpha" not in rendered.dynamic_tail
    assert "value alpha" in rendered.dynamic_tail


def test_context_guard_rejects_gold_payloads() -> None:
    with pytest.raises(L4ContextError):
        assert_no_forbidden_context({"gold_frame": {"intent": "intent_alpha"}})

