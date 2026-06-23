import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from darjeeling.targets.nlu.layers.l4_cloud_llm import (
    CachedTeacherLayer,
    CloudLLMTeacher,
    TaskSchema,
    TeacherCache,
    TeacherParseError,
    parse_clinc150_teacher_frame,
    parse_teacher_frame,
    parse_teacher_patch,
)
from darjeeling.targets.nlu.settings import load_settings
from darjeeling.targets.nlu.teacher import (
    build_teacher_system_prompt,
    ensure_supported_teacher_prompt_version,
)


class FakeCompletions:
    def __init__(self) -> None:
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            model=kwargs["model"],
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {"intent": "intent_beta", "slots": {}, "is_abstain": False}
                        )
                    )
                )
            ],
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7, total_tokens=18),
        )


class FlakyCompletions(FakeCompletions):
    def __init__(self) -> None:
        super().__init__()
        self.failures_left = 1

    def create(self, **kwargs):
        if self.failures_left:
            self.failures_left -= 1
            raise RuntimeError("temporary network failure")
        return super().create(**kwargs)


class EmptyThenValidCompletions(FakeCompletions):
    def __init__(self) -> None:
        super().__init__()
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


class FakeClient:
    def __init__(self) -> None:
        self.completions = FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


class FlakyClient(FakeClient):
    def __init__(self) -> None:
        self.completions = FlakyCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


class EmptyThenValidClient(FakeClient):
    def __init__(self) -> None:
        self.completions = EmptyThenValidCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


def test_parse_teacher_frame_requires_frame_json() -> None:
    frame = parse_teacher_frame('{"intent":"intent_alpha","slots":{"slot_alpha":"value alpha"}}')

    assert frame.intent == "intent_alpha"
    assert frame.slots == {"slot_alpha": "value alpha"}


def test_parse_teacher_frame_rejects_malformed_json() -> None:
    with pytest.raises(TeacherParseError, match="invalid JSON"):
        parse_teacher_frame("not-json")


def test_teacher_prompt_versions_render_distinct_prompts() -> None:
    schema = TaskSchema(intent_names=["intent_alpha"], slot_names=["slot_alpha"])

    current = build_teacher_system_prompt(schema, prompt_version="teacher-v1")
    intent_first = build_teacher_system_prompt(
        schema,
        prompt_version="teacher-v2-intent-first",
    )
    slot_conservative = build_teacher_system_prompt(
        schema,
        prompt_version="teacher-v3-slot-conservative",
    )
    slot_evidence = build_teacher_system_prompt(
        schema,
        prompt_version="teacher-v4-slot-evidence",
    )
    value_copy = build_teacher_system_prompt(
        schema,
        prompt_version="teacher-v5-value-copy",
    )
    schema_checklist = build_teacher_system_prompt(
        schema,
        prompt_version="teacher-v6-schema-checklist",
    )
    evidence_stable = build_teacher_system_prompt(
        schema,
        prompt_version="teacher-v7-evidence-stable",
    )
    evidence_compact = build_teacher_system_prompt(
        schema,
        prompt_version="teacher-v8-evidence-compact",
    )
    clinc150_v1 = build_teacher_system_prompt(
        TaskSchema(intent_names=["intent_alpha", "out_of_scope"], slot_names=[]),
        prompt_version="clinc150-intent-v1",
    )
    clinc150_v2 = build_teacher_system_prompt(
        TaskSchema(intent_names=["intent_alpha", "out_of_scope"], slot_names=[]),
        prompt_version="clinc150-intent-v2-label-cards",
    )

    assert "Prompt version: teacher-v1." in current
    assert "decide the intent first" in intent_first
    assert "Prefer an empty slots object over a guessed slot." in slot_conservative
    assert "Add a slot only when the utterance gives specific evidence" in slot_evidence
    assert "copy the exact words from the utterance" in value_copy
    assert "Use this private checklist" in schema_checklist
    assert "do not change intent because a slot is uncertain" in evidence_stable
    assert "filler time/date words" in evidence_stable
    assert "Return strict JSON only:" in evidence_compact
    assert "omit it; do not change a clear intent" in evidence_compact
    assert "CLINC150 intent classification" in clinc150_v1
    assert '{"intent": "intent_name"}' in clinc150_v1
    assert "label cards" in clinc150_v2
    prompts = {
        current,
        intent_first,
        slot_conservative,
        slot_evidence,
        value_copy,
        schema_checklist,
        evidence_stable,
        evidence_compact,
        clinc150_v1,
        clinc150_v2,
    }
    assert len(prompts) == 10


def test_teacher_prompt_version_rejects_unknown_version() -> None:
    with pytest.raises(ValueError, match="unsupported teacher prompt version 'teacher-typo'"):
        ensure_supported_teacher_prompt_version("teacher-typo")


def test_parse_clinc150_teacher_frame_requires_single_allowed_intent() -> None:
    schema = TaskSchema(intent_names=["balance", "out_of_scope"], slot_names=[])

    frame = parse_clinc150_teacher_frame('{"intent":"out_of_scope"}', task_schema=schema)

    assert frame.intent == "out_of_scope"
    assert frame.slots == {}
    assert frame.is_abstain is True
    with pytest.raises(TeacherParseError, match="exactly one field"):
        parse_clinc150_teacher_frame(
            '{"intent":"balance","slots":{}}',
            task_schema=schema,
        )
    with pytest.raises(TeacherParseError, match="unsupported intent"):
        parse_clinc150_teacher_frame('{"intent":"not_allowed"}', task_schema=schema)


def test_parse_teacher_patch_accepts_residual_patch_json() -> None:
    patch = parse_teacher_patch(
        '{"accepted_intent":"intent_alpha","accepted_slots":{"slot_alpha":"value alpha"}}',
        task_schema=TaskSchema(intent_names=["intent_alpha"], slot_names=["slot_alpha"]),
    )

    assert patch.accepted_intent == "intent_alpha"
    assert patch.accepted_slots == {"slot_alpha": "value alpha"}
    assert patch.source_layer == "L4"
    assert patch.complete is True


def test_live_teacher_client_sets_sdk_timeout_and_disables_sdk_retries(monkeypatch) -> None:
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    settings = load_settings().model_copy(
        update={
            "openai_api_key": "test-key",
            "openai_timeout_s": 12.0,
        }
    )

    CloudLLMTeacher(settings).client()

    assert captured["api_key"] == "test-key"
    assert captured["timeout"] == 12.0
    assert captured["max_retries"] == 0


def test_live_teacher_call_appends_cache(tmp_path: Path) -> None:
    settings = load_settings()
    fake_client = FakeClient()
    cache = TeacherCache.load(tmp_path / "teacher_cache.jsonl")
    schema = TaskSchema(intent_names=["intent_beta"], slot_names=[])
    layer = CachedTeacherLayer(
        cache,
        allow_live=True,
        use_cache=True,
        settings=settings,
        task_schema=schema,
        teacher=CloudLLMTeacher(settings, client=fake_client),
    )

    result = layer.try_answer("beta sample request")

    assert result.accepted
    assert result.frame is not None
    assert result.frame.intent == "intent_beta"
    assert result.metadata["teacher_source"] == "live"
    assert result.cost_usd == pytest.approx((11 * 0.40 + 7 * 1.60) / 1_000_000)
    assert fake_client.completions.calls[0]["response_format"] == {"type": "json_object"}
    assert fake_client.completions.calls[0]["prompt_cache_key"].startswith("darjeeling:")
    assert fake_client.completions.calls[0]["timeout"] == settings.openai_timeout_s

    cache_lines = (tmp_path / "teacher_cache.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(cache_lines) == 1
    payload = json.loads(cache_lines[0])
    assert payload["utterance"] == "beta sample request"
    assert payload["teacher_frame"]["intent"] == "intent_beta"
    assert payload["usage"]["total_tokens"] == 18
    assert payload["context_hash"]
    assert payload["prompt_cache_key"].startswith("darjeeling:teacher-v1:")


def test_live_teacher_intent_first_prompt_makes_two_calls(tmp_path: Path) -> None:
    settings = load_settings().model_copy(
        update={
            "teacher_prompt_version": "teacher-v2-intent-first",
            "teacher_max_tokens": 128,
        }
    )
    fake_client = FakeClient()
    cache = TeacherCache.load(tmp_path / "teacher_cache.jsonl")
    schema = TaskSchema(intent_names=["intent_beta"], slot_names=[])
    layer = CachedTeacherLayer(
        cache,
        allow_live=True,
        use_cache=True,
        settings=settings,
        task_schema=schema,
        teacher=CloudLLMTeacher(settings, client=fake_client),
    )

    result = layer.try_answer("beta sample request")

    assert result.accepted
    assert result.frame is not None
    assert result.frame.intent == "intent_beta"
    assert result.metadata["prompt_version"] == "teacher-v2-intent-first"
    assert result.metadata["usage"]["total_tokens"] == 36
    assert len(fake_client.completions.calls) == 2
    assert fake_client.completions.calls[0]["max_completion_tokens"] == 64
    assert fake_client.completions.calls[1]["max_completion_tokens"] == 128
    assert "Step 1" in fake_client.completions.calls[0]["messages"][0]["content"]
    assert "Step 2" in fake_client.completions.calls[1]["messages"][0]["content"]


def test_live_residual_teacher_call_uses_residual_budget_and_metadata(
    tmp_path: Path,
) -> None:
    settings = load_settings().model_copy(
        update={
            "teacher_max_tokens": 256,
            "residual_l4_max_tokens": 32,
        }
    )
    fake_client = FakeClient()
    cache_path = tmp_path / "teacher_cache.jsonl"
    cache = TeacherCache.load(cache_path)
    layer = CachedTeacherLayer(
        cache,
        allow_live=True,
        use_cache=False,
        settings=settings,
        task_schema=TaskSchema(intent_names=["intent_alpha", "intent_beta"], slot_names=[]),
        teacher=CloudLLMTeacher(settings, client=fake_client),
    )

    result = layer.try_residual_patch(
        "beta sample request",
        accepted_fields={"intent": "intent_alpha"},
        missing_fields=[],
    )

    assert result.accepted
    assert result.frame is None
    assert result.patch is not None
    assert result.patch.accepted_intent == "intent_beta"
    assert result.patch.complete is True
    assert result.metadata["l4_call_kind"] == "residual"
    assert result.metadata["fields_avoided"] == 1
    assert result.metadata["usage"]["total_tokens"] == 18
    assert fake_client.completions.calls[0]["max_completion_tokens"] == 32


def test_live_teacher_retries_transient_completion_failure(tmp_path: Path) -> None:
    settings = load_settings().model_copy(
        update={
            "openai_max_retries": 1,
            "openai_retry_base_delay_s": 0.0,
        }
    )
    fake_client = FlakyClient()
    cache = TeacherCache.load(tmp_path / "teacher_cache.jsonl")
    layer = CachedTeacherLayer(
        cache,
        allow_live=True,
        use_cache=True,
        settings=settings,
        task_schema=TaskSchema(intent_names=["intent_beta"], slot_names=[]),
        teacher=CloudLLMTeacher(settings, client=fake_client),
    )

    result = layer.try_answer("beta sample request")

    assert result.accepted
    assert result.frame is not None
    assert result.frame.intent == "intent_beta"
    assert fake_client.completions.failures_left == 0
    assert len(fake_client.completions.calls) == 1


def test_live_teacher_retries_empty_completion_content(tmp_path: Path) -> None:
    settings = load_settings().model_copy(
        update={
            "openai_max_retries": 1,
            "openai_retry_base_delay_s": 0.0,
        }
    )
    fake_client = EmptyThenValidClient()
    layer = CachedTeacherLayer(
        TeacherCache.load(tmp_path / "teacher_cache.jsonl"),
        allow_live=True,
        use_cache=True,
        settings=settings,
        task_schema=TaskSchema(intent_names=["intent_beta"], slot_names=[]),
        teacher=CloudLLMTeacher(settings, client=fake_client),
    )

    result = layer.try_answer("beta sample request")

    assert result.accepted
    assert result.frame is not None
    assert result.frame.intent == "intent_beta"
    assert len(fake_client.completions.calls) == 2


def test_cache_hit_does_not_call_live_teacher(tmp_path: Path) -> None:
    (tmp_path / "teacher_cache.jsonl").write_text(
        json.dumps(
            {
                "utterance": "beta sample request",
                "teacher_frame": {"intent": "intent_beta", "slots": {}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    settings = load_settings()
    fake_client = FakeClient()
    layer = CachedTeacherLayer(
        TeacherCache.load(tmp_path / "teacher_cache.jsonl"),
        allow_live=True,
        use_cache=True,
        settings=settings,
        task_schema=TaskSchema(intent_names=["intent_beta"], slot_names=[]),
        teacher=CloudLLMTeacher(settings, client=fake_client),
    )

    result = layer.try_answer("beta sample request")

    assert result.accepted
    assert result.metadata["teacher_source"] == "cache"
    assert fake_client.completions.calls == []
