import json
from types import SimpleNamespace

import pytest

from darjeeling.compiler.l4_proposal import L4ProposalAdapter, ProposalParseError, parse_proposal
from darjeeling.layers.l4_cloud_llm import TaskSchema
from darjeeling.schemas import Frame, LayerResult, TraceRecord, traces_to_teacher_view
from darjeeling.settings import load_settings


class FakeProposalCompletions:
    def __init__(self, content: dict) -> None:
        self.content = content
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            model=kwargs["model"],
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(self.content)))],
            usage=SimpleNamespace(prompt_tokens=20, completion_tokens=8, total_tokens=28),
        )


class EmptyThenValidProposalCompletions(FakeProposalCompletions):
    def __init__(self, content: dict) -> None:
        super().__init__(content)
        self.empty_left = 1

    def create(self, **kwargs):
        if self.empty_left:
            self.empty_left -= 1
            self.calls.append(kwargs)
            return SimpleNamespace(
                model=kwargs["model"],
                choices=[SimpleNamespace(message=SimpleNamespace(content=""))],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=0, total_tokens=1),
            )
        return super().create(**kwargs)


class FakeProposalClient:
    def __init__(self, content: dict) -> None:
        self.completions = FakeProposalCompletions(content)
        self.chat = SimpleNamespace(completions=self.completions)


class EmptyThenValidProposalClient(FakeProposalClient):
    def __init__(self, content: dict) -> None:
        self.completions = EmptyThenValidProposalCompletions(content)
        self.chat = SimpleNamespace(completions=self.completions)


def test_l4_proposal_adapter_calls_direct_api_with_teacher_visible_context() -> None:
    trace = TraceRecord(
        request_id="r1",
        utterance="set alarm for seven",
        gold_frame=Frame(intent="alarm_set", slots={"time": "gold-seven"}),
        teacher_frame=Frame(intent="alarm_set", slots={"time": "seven"}),
        chosen_layer="L4",
        final_frame=Frame(intent="alarm_set", slots={"time": "seven"}),
        layer_results=[
            LayerResult(
                layer="L4",
                accepted=True,
                frame=Frame(intent="alarm_set", slots={"time": "seven"}),
                latency_ms=1.0,
            )
        ],
    )
    output_schema = {
        "type": "object",
        "required": ["family", "accept_threshold"],
        "properties": {
            "family": {"type": "string"},
            "accept_threshold": {"type": "number"},
        },
    }
    settings = load_settings()
    fake_client = FakeProposalClient({"family": "token_sgd", "accept_threshold": 0.93})
    adapter = L4ProposalAdapter(settings, client=fake_client)

    result = adapter.propose(
        role="l2",
        task_schema=TaskSchema(intent_names=["alarm_set"], slot_names=["time"]),
        traces=traces_to_teacher_view([trace]),
        output_schema=output_schema,
        metrics={"frame_exact_match": 0.9},
    )

    assert result.proposal == {"family": "token_sgd", "accept_threshold": 0.93}
    assert result.usage["total_tokens"] == 28
    assert result.source_trace_ids == ["r1"]
    call = fake_client.completions.calls[0]
    assert call["response_format"] == {"type": "json_object"}
    assert call["max_completion_tokens"] == settings.proposal_max_tokens
    assert call["timeout"] == settings.openai_timeout_s
    assert call["prompt_cache_key"].startswith("darjeeling:l2-proposal-v1:")
    rendered_messages = json.dumps(call["messages"], sort_keys=True)
    assert "gold_frame" not in rendered_messages
    assert "gold-seven" not in rendered_messages


def test_l4_proposal_adapter_retries_empty_completion_content() -> None:
    output_schema = {
        "type": "object",
        "required": ["family", "accept_threshold"],
        "properties": {
            "family": {"type": "string"},
            "accept_threshold": {"type": "number"},
        },
    }
    settings = load_settings().model_copy(
        update={
            "openai_max_retries": 1,
            "openai_retry_base_delay_s": 0.0,
        }
    )
    fake_client = EmptyThenValidProposalClient({"family": "token_sgd", "accept_threshold": 0.93})
    adapter = L4ProposalAdapter(settings, client=fake_client)

    result = adapter.propose(
        role="l2",
        task_schema=TaskSchema(intent_names=["alarm_set"], slot_names=[]),
        traces=[],
        output_schema=output_schema,
    )

    assert result.proposal == {"family": "token_sgd", "accept_threshold": 0.93}
    assert len(fake_client.completions.calls) == 2


def test_parse_proposal_rejects_invalid_json_or_missing_required_field() -> None:
    schema = {"type": "object", "required": ["family"]}

    with pytest.raises(ProposalParseError):
        parse_proposal("not json", schema)

    with pytest.raises(ProposalParseError):
        parse_proposal("{}", schema)
