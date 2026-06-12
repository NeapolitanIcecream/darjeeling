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


def test_nlu_metrics_compare_frames_and_intents() -> None:
    from darjeeling.targets.nlu.metrics import frame_exact_match, intent_matches

    expected = Frame(intent="intent_alpha", slots={"slot_alpha": "value alpha"})
    same_intent = Frame(intent="intent_alpha", slots={"slot_alpha": "other"})
    different_intent = Frame(intent="intent_beta", slots={"slot_alpha": "value alpha"})

    assert frame_exact_match(expected, expected) is True
    assert frame_exact_match(same_intent, expected) is False
    assert intent_matches(same_intent, expected) is True
    assert intent_matches(different_intent, expected) is False


def test_nlu_l0_compiler_uses_target_normalization() -> None:
    from darjeeling.targets.nlu.compiler.l0_compile import exact_cache_from_teacher_traces
    from darjeeling.targets.nlu.schemas import TeacherTrace as LegacyTeacherTrace

    frame = Frame(intent="intent_alpha")
    cache = exact_cache_from_teacher_traces(
        [
            LegacyTeacherTrace(
                request_id="r1",
                utterance="  Alpha Request  ",
                teacher_frame=frame,
                chosen_layer="L4",
                final_frame=frame,
                layer_results=[],
                timestamp="2026-06-12T00:00:00+00:00",
            )
        ]
    )

    assert cache == {"alpha request": frame}


def test_nlu_l0_cache_layer_normalizes_requests() -> None:
    from darjeeling.targets.nlu.layers.l0_cache import ExactCacheLayer

    layer = ExactCacheLayer()
    layer.add("  Alpha Request  ", Frame(intent="intent_alpha"))

    result = layer.try_answer("alpha request")

    assert result.accepted is True
    assert result.frame == Frame(intent="intent_alpha")


def test_nlu_l1_dsl_rule_matches_and_extracts_slots() -> None:
    from darjeeling.targets.nlu.layers.l1_program_bank import ProgramRule, render_rule

    rule = ProgramRule.model_validate(
        {
            "rule_id": "intent_alpha_001",
            "description": "alpha requests with explicit slot value",
            "condition": {
                "and": [
                    {"contains_any": ["alpha request", "alpha wake"]},
                    {
                        "regex_extract": {
                            "pattern": "(?:for|at) (?P<slot_alpha>.+)$",
                            "slot_map": {"slot_alpha": "slot_alpha"},
                        }
                    },
                ]
            },
            "action": {
                "accept": {
                    "intent": "intent_alpha",
                    "slots_from_regex": True,
                }
            },
        }
    )

    frame = rule.try_frame("Alpha request for value alpha extended")

    assert frame == Frame(intent="intent_alpha", slots={"slot_alpha": "value alpha extended"})
    assert "intent_alpha_001" in render_rule(rule)


def test_nlu_l1_dsl_rejects_unknown_operator() -> None:
    from darjeeling.targets.nlu.layers.l1_program_bank import ProgramRule

    with pytest.raises(ValueError, match="unsupported L1 operator"):
        ProgramRule.model_validate(
            {
                "rule_id": "bad_001",
                "condition": {"decision_tree": {"depth": 3}},
                "action": {"accept": {"intent": "intent_alpha"}},
            }
        )


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


def test_nlu_teacher_parser_uses_nlu_target_frame_type() -> None:
    from darjeeling.targets.nlu.layers.l4_cloud_llm import parse_teacher_frame

    parsed = parse_teacher_frame(
        json.dumps({"intent": "intent_alpha", "slots": {"slot_alpha": "value alpha"}})
    )

    assert isinstance(parsed, Frame)
    assert parsed == Frame(intent="intent_alpha", slots={"slot_alpha": "value alpha"})


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
