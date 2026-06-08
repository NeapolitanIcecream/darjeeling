import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from darjeeling.layers.l4_cloud_llm import (
    CachedTeacherLayer,
    CloudLLMTeacher,
    TaskSchema,
    TeacherCache,
    parse_teacher_frame,
)
from darjeeling.settings import load_settings


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
                            {"intent": "music_play", "slots": {}, "is_abstain": False}
                        )
                    )
                )
            ],
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7, total_tokens=18),
        )


class FakeClient:
    def __init__(self) -> None:
        self.completions = FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


def test_parse_teacher_frame_requires_frame_json() -> None:
    frame = parse_teacher_frame('{"intent":"alarm_set","slots":{"time":"seven"}}')

    assert frame.intent == "alarm_set"
    assert frame.slots == {"time": "seven"}


def test_live_teacher_call_appends_cache(tmp_path: Path) -> None:
    settings = load_settings()
    fake_client = FakeClient()
    cache = TeacherCache.load(tmp_path / "teacher_cache.jsonl")
    schema = TaskSchema(intent_names=["music_play"], slot_names=[])
    layer = CachedTeacherLayer(
        cache,
        allow_live=True,
        use_cache=True,
        settings=settings,
        task_schema=schema,
        teacher=CloudLLMTeacher(settings, client=fake_client),
    )

    result = layer.try_answer("play some jazz")

    assert result.accepted
    assert result.frame is not None
    assert result.frame.intent == "music_play"
    assert result.metadata["teacher_source"] == "live"
    assert result.cost_usd == pytest.approx((11 * 0.40 + 7 * 1.60) / 1_000_000)
    assert fake_client.completions.calls[0]["response_format"] == {"type": "json_object"}
    assert fake_client.completions.calls[0]["prompt_cache_key"].startswith("darjeeling:")

    cache_lines = (tmp_path / "teacher_cache.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(cache_lines) == 1
    payload = json.loads(cache_lines[0])
    assert payload["utterance"] == "play some jazz"
    assert payload["teacher_frame"]["intent"] == "music_play"
    assert payload["usage"]["total_tokens"] == 18
    assert payload["context_hash"]
    assert payload["prompt_cache_key"].startswith("darjeeling:teacher-v1:")


def test_cache_hit_does_not_call_live_teacher(tmp_path: Path) -> None:
    (tmp_path / "teacher_cache.jsonl").write_text(
        json.dumps(
            {
                "utterance": "play some jazz",
                "teacher_frame": {"intent": "music_play", "slots": {}},
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
        task_schema=TaskSchema(intent_names=["music_play"], slot_names=[]),
        teacher=CloudLLMTeacher(settings, client=fake_client),
    )

    result = layer.try_answer("play some jazz")

    assert result.accepted
    assert result.metadata["teacher_source"] == "cache"
    assert fake_client.completions.calls == []
