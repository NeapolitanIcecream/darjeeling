import json
import sys
from types import SimpleNamespace

import pytest

from darjeeling.targets import registry
from darjeeling.targets.nlu.adapters.massive import prepare_massive_dataset
from darjeeling.targets.nlu.data import (
    frame_from_annotated_utterance,
    normalized_template,
    strip_annotations,
)
from darjeeling.targets.nlu.schemas import Frame, TaskSchema
from darjeeling.targets.nlu.target import NluTargetSpec
from darjeeling.targets.nlu.teacher import NluTeacherAdapter, NluTeacherParseError


class _IntentFeature:
    def int2str(self, value):
        return "intent_alpha" if value == 0 else str(value)


class _FakeDataset(list):
    features = {"intent": _IntentFeature()}


class _FakeDataFrame:
    def __init__(self, rows):
        self.rows = rows

    def to_parquet(self, path, *, index):
        del index
        path.write_text("fake parquet\n", encoding="utf-8")


def test_nlu_target_is_available_from_static_registry() -> None:
    target = registry.get_target("nlu")

    assert target.name == "nlu"
    assert registry.available_targets() == ("nlu",)


def test_nlu_target_spec_loads_schema_and_compares_labels() -> None:
    target = NluTargetSpec()
    records = [
        {
            "utterance": "alpha request",
            "gold_frame": {"intent": "intent_alpha", "slots": {"slot_alpha": "value alpha"}},
        }
    ]

    task_schema = target.load_task_schema(records)

    assert task_schema == {
        "intent_names": ["intent_alpha"],
        "slot_names": ["slot_alpha"],
        "schema_version": "task-schema-v1",
    }
    assert target.normalize_request({"utterance": "  Alpha Request  "}) == "alpha request"
    target.validate_output(
        {"intent": "intent_alpha", "slots": {"slot_alpha": "value alpha"}},
        task_schema,
    )
    assert target.labels_equal(
        {"intent": "intent_alpha", "slots": {"slot_alpha": "value alpha"}},
        {"intent": "intent_alpha", "slots": {"slot_alpha": "value alpha"}},
        task_schema=task_schema,
    )


def test_nlu_frame_parser_extracts_slots_from_bracket_annotation() -> None:
    annotated = "alpha request [slot_alpha : value alpha extended]"

    frame = frame_from_annotated_utterance("intent_alpha", annotated)

    assert frame == Frame(intent="intent_alpha", slots={"slot_alpha": "value alpha extended"})
    assert strip_annotations(annotated) == "alpha request value alpha extended"
    assert normalized_template(annotated) == "alpha request [slot_alpha]"


def test_legacy_frame_parser_module_reexports_target_functions() -> None:
    from darjeeling.data.frames import (
        frame_from_annotated_utterance as legacy_frame_from_annotated_utterance,
    )

    assert legacy_frame_from_annotated_utterance is frame_from_annotated_utterance


def test_legacy_core_frame_alias_points_to_nlu_target_frame() -> None:
    from darjeeling.schemas import Frame as LegacyFrame

    assert LegacyFrame is Frame


def test_nlu_teacher_adapter_builds_prompt_and_parses_frame() -> None:
    adapter = NluTeacherAdapter(prompt_version="teacher-test")
    task_schema = TaskSchema(
        intent_names=["intent_alpha"],
        slot_names=["slot_alpha"],
    ).to_payload()

    messages = adapter.build_messages(
        input={"utterance": "alpha request"},
        task_schema=task_schema,
    )
    parsed = adapter.parse_response(
        json.dumps({"intent": "intent_alpha", "slots": {"slot_alpha": "value alpha"}}),
        task_schema=task_schema,
    )

    assert messages[0]["role"] == "system"
    assert "Return strict JSON only." in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": '{"utterance": "alpha request"}'}
    assert parsed == {
        "intent": "intent_alpha",
        "slots": {"slot_alpha": "value alpha"},
        "is_abstain": False,
    }
    assert adapter.cache_key_parts(task_schema=task_schema) == {
        "prompt_version": "teacher-test",
        "schema_version": "task-schema-v1",
    }


def test_nlu_teacher_adapter_rejects_invalid_json() -> None:
    with pytest.raises(NluTeacherParseError):
        NluTeacherAdapter().parse_response("not json", task_schema={})


def test_nlu_massive_adapter_prepares_target_records(tmp_path, monkeypatch) -> None:
    calls = []

    def load_dataset(path, locale, *, split, trust_remote_code):
        calls.append(
            {
                "path": path,
                "locale": locale,
                "split": split,
                "trust_remote_code": trust_remote_code,
            }
        )
        return _FakeDataset(
            [
                {
                    "utt": f"alpha {split}",
                    "annot_utt": "alpha [slot_alpha : value alpha]",
                    "intent": 0,
                    "domain": "fixture",
                }
            ]
        )

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=load_dataset),
    )
    monkeypatch.setitem(
        sys.modules,
        "pandas",
        SimpleNamespace(DataFrame=_FakeDataFrame),
    )

    result = prepare_massive_dataset("en-US", tmp_path)

    assert result == {"records": 3}
    assert {call["split"] for call in calls} == {"train", "validation", "test"}
    assert all(call["path"] == "AmazonScience/massive" for call in calls)
    assert all(call["locale"] == "en-US" for call in calls)
    assert all(call["trust_remote_code"] is True for call in calls)
    assert (tmp_path / "train.jsonl").exists()
    assert (tmp_path / "records.parquet").exists()
