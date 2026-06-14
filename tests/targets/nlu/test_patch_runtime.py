import json
from pathlib import Path
from typing import Any

from darjeeling.artifacts.store import ArtifactManifest, ArtifactStore
from darjeeling.contracts import LayerResult as CoreLayerResult
from darjeeling.targets.nlu.compiler.replay import (
    OfflineArtifactSet,
    evaluate_offline_artifact_set,
)
from darjeeling.targets.nlu.data import DataRecord
from darjeeling.targets.nlu.layers.l4_cloud_llm import (
    TeacherCallResult,
    TeacherPatchCallResult,
)
from darjeeling.targets.nlu.patches import route_nlu_layers
from darjeeling.targets.nlu.replay import run_replay
from darjeeling.targets.nlu.schemas import (
    Frame,
    FramePatch,
    LayerResult,
    TraceRecord,
    traces_to_teacher_view,
)
from darjeeling.targets.nlu.settings import load_settings


class _PatchLayer:
    def __init__(self, result: CoreLayerResult) -> None:
        self.result = result

    def try_answer(self, _input: dict[str, Any]) -> CoreLayerResult:
        return self.result


class _ResidualPatchLayer:
    def __init__(self) -> None:
        self.residual_inputs: list[dict[str, Any]] = []

    def residual_field_keys(self) -> list[str]:
        return ["intent", "slots.slot_alpha", "slots.slot_beta"]

    def try_residual_patch(self, input: dict[str, Any]) -> CoreLayerResult:
        self.residual_inputs.append(input)
        return CoreLayerResult(
            layer="L4",
            accepted=True,
            output=None,
            latency_ms=4.0,
            cost_usd=0.001,
            metadata={
                "l4_call_kind": "residual",
                "fields_avoided": 2,
                "usage": {"total_tokens": 3},
                "frame_patch": FramePatch(
                    accepted_slots={"slot_beta": "teacher residual"},
                    source_layer="L4",
                    complete=True,
                    metadata={
                        "l4_call_kind": "residual",
                        "fields_avoided": 2,
                        "verified_fields": ["intent", "slots.slot_alpha"],
                    },
                ).model_dump(mode="json"),
            },
        )

    def try_answer(self, _input: dict[str, Any]) -> CoreLayerResult:
        raise AssertionError("full L4 should not be called after residual fill")


class _PatchRuntime:
    def build_layers(self, *, manifest, teacher, settings):
        del manifest, settings
        return {
            "L1": _PatchLayer(
                CoreLayerResult(
                    layer="L1",
                    accepted=True,
                    output=None,
                    latency_ms=1.0,
                    metadata={
                        "frame_patch": FramePatch(
                            accepted_intent="intent_alpha",
                            source_layer="L1",
                        ).model_dump(mode="json")
                    },
                )
            ),
            "L2": _PatchLayer(
                CoreLayerResult(
                    layer="L2",
                    accepted=True,
                    output=None,
                    latency_ms=2.0,
                    metadata={
                        "frame_patch": FramePatch(
                            accepted_slots={"slot_alpha": "lower value"},
                            source_layer="L2",
                        ).model_dump(mode="json")
                    },
                )
            ),
            "L4": teacher,
        }


class _PatchRuntimeTarget:
    name = "nlu"
    schema_version = "nlu-target-v1"
    runtime = _PatchRuntime()


def test_route_nlu_layers_lets_full_l4_override_weak_field_conflicts() -> None:
    layers = {
        "L1": _PatchLayer(
            CoreLayerResult(
                layer="L1",
                accepted=True,
                output=None,
                latency_ms=1.0,
                metadata={
                    "frame_patch": FramePatch(
                        accepted_intent="intent_alpha",
                        source_layer="L1",
                    ).model_dump(mode="json")
                },
            )
        ),
        "L2": _PatchLayer(
            CoreLayerResult(
                layer="L2",
                accepted=True,
                output=None,
                latency_ms=2.0,
                metadata={
                    "frame_patch": FramePatch(
                        accepted_slots={"slot_alpha": "lower value"},
                        source_layer="L2",
                    ).model_dump(mode="json")
                },
            )
        ),
        "L4": _PatchLayer(
            CoreLayerResult(
                layer="L4",
                accepted=True,
                output={
                    "intent": "intent_alpha",
                    "slots": {
                        "slot_alpha": "teacher value",
                        "slot_beta": "teacher residual",
                    },
                    "is_abstain": False,
                },
                latency_ms=10.0,
                metadata={"usage": {"total_tokens": 10}},
            )
        ),
    }

    result = route_nlu_layers(layers, utterance="alpha request")

    assert result.chosen_layer == "L4"
    assert result.final_frame == Frame(
        intent="intent_alpha",
        slots={"slot_alpha": "teacher value", "slot_beta": "teacher residual"},
    )
    assert result.layer_results[0].metadata["patch_accepted_fields"] == ["intent"]
    assert result.layer_results[1].metadata["patch_accepted_fields"] == ["slots.slot_alpha"]
    assert result.layer_results[2].patch is not None
    assert result.layer_results[2].patch.accepted_slots == {
        "slot_alpha": "teacher value",
        "slot_beta": "teacher residual",
    }
    assert result.layer_results[2].metadata["field_conflicts"] == [
        {
            "field": "slots.slot_alpha",
            "old_value": "lower value",
            "new_value": "teacher value",
            "old_source": "L2",
            "new_source": "L4",
        }
    ]
    assert result.composer.field_sources == {
        "intent": "L1",
        "slots.slot_alpha": "L4",
        "slots.slot_beta": "L4",
    }
    assert result.l4_usage["serving_full_calls"] == 1


def test_route_nlu_layers_uses_residual_l4_for_partial_patch_completion() -> None:
    l4 = _ResidualPatchLayer()
    layers = {
        "L1": _PatchLayer(
            CoreLayerResult(
                layer="L1",
                accepted=True,
                output=None,
                latency_ms=1.0,
                metadata={
                    "frame_patch": FramePatch(
                        accepted_intent="intent_alpha",
                        source_layer="L1",
                    ).model_dump(mode="json")
                },
            )
        ),
        "L2": _PatchLayer(
            CoreLayerResult(
                layer="L2",
                accepted=True,
                output=None,
                latency_ms=2.0,
                metadata={
                    "frame_patch": FramePatch(
                        accepted_slots={"slot_alpha": "lower value"},
                        source_layer="L2",
                    ).model_dump(mode="json")
                },
            )
        ),
        "L4": l4,
    }

    result = route_nlu_layers(layers, utterance="alpha request")

    assert result.chosen_layer == "L4"
    assert result.final_frame == Frame(
        intent="intent_alpha",
        slots={"slot_alpha": "lower value", "slot_beta": "teacher residual"},
    )
    assert l4.residual_inputs == [
        {
            "utterance": "alpha request",
            "accepted_fields": {
                "intent": "intent_alpha",
                "slots.slot_alpha": "lower value",
            },
            "missing_fields": ["slots.slot_beta"],
        }
    ]
    assert result.l4_usage["serving_residual_calls"] == 1
    assert result.l4_usage["serving_residual_tokens"] == 3
    assert result.l4_usage["serving_fields_avoided"] == 2.0
    assert result.l4_usage["serving_residual_verified_fields"] == 2.0


def test_run_replay_labels_verified_live_residual_completion(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_dir = tmp_path / "data"
    run_dir = tmp_path / "run"
    data_dir.mkdir()
    (data_dir / "train.jsonl").write_text(
        DataRecord(
            request_id="r1",
            utterance="alpha request",
            gold_frame=Frame(
                intent="intent_alpha",
                slots={"slot_alpha": "lower value", "slot_beta": "teacher residual"},
            ),
        ).model_dump_json()
        + "\n",
        encoding="utf-8",
    )

    def fake_residual_patch(self, *, utterance, task_schema, accepted_fields, missing_fields):
        del self, task_schema
        assert utterance == "alpha request"
        assert accepted_fields == {
            "intent": "intent_alpha",
            "slots.slot_alpha": "lower value",
        }
        assert missing_fields == ["slots.slot_beta"]
        return TeacherPatchCallResult(
            patch=FramePatch(
                accepted_slots={"slot_beta": "teacher residual"},
                source_layer="L4",
                complete=True,
                metadata={"verified_fields": ["intent", "slots.slot_alpha"]},
            ),
            raw_response="{}",
            usage={"total_tokens": 5},
            model="fake-teacher",
            context_hash="ctx",
            prompt_cache_key="cache",
        )

    def fake_answer(self, utterance, task_schema):
        raise AssertionError("verified residual completion should not call full L4")

    monkeypatch.setattr(
        "darjeeling.targets.nlu.layers.l4_cloud_llm.CloudLLMTeacher.residual_patch",
        fake_residual_patch,
    )
    monkeypatch.setattr(
        "darjeeling.targets.nlu.layers.l4_cloud_llm.CloudLLMTeacher.answer",
        fake_answer,
    )
    settings = load_settings().model_copy(update={"openai_api_key": "test-key"})

    run_replay(
        stream="sequential",
        max_requests=1,
        teacher_mode="live",
        run_dir=run_dir,
        data_dir=data_dir,
        settings=settings,
        target=_PatchRuntimeTarget(),
    )

    trace = json.loads((run_dir / "traces.jsonl").read_text(encoding="utf-8"))
    assert trace["chosen_layer"] == "L4"
    assert trace["teacher_frame"] == trace["final_frame"]
    assert trace["metadata"]["teacher_frame_source"] == "residual_live"
    assert trace["metadata"]["residual_l4_verified_teacher_frame"] is True
    assert trace["metadata"]["residual_l4_verified_field_count"] == 2.0


def test_run_replay_full_audits_unverified_live_residual_completion(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_dir = tmp_path / "data"
    run_dir = tmp_path / "run"
    data_dir.mkdir()
    (data_dir / "train.jsonl").write_text(
        DataRecord(
            request_id="r1",
            utterance="alpha request",
            gold_frame=Frame(
                intent="intent_alpha",
                slots={"slot_alpha": "teacher value", "slot_beta": "teacher residual"},
            ),
        ).model_dump_json()
        + "\n",
        encoding="utf-8",
    )

    def fake_residual_patch(self, *, utterance, task_schema, accepted_fields, missing_fields):
        del self, utterance, task_schema, accepted_fields, missing_fields
        return TeacherPatchCallResult(
            patch=FramePatch(
                accepted_slots={"slot_beta": "teacher residual"},
                source_layer="L4",
                complete=True,
            ),
            raw_response="{}",
            usage={"total_tokens": 5},
            model="fake-teacher",
            context_hash="ctx-residual",
            prompt_cache_key="cache-residual",
        )

    def fake_answer(self, utterance, task_schema):
        del self, utterance, task_schema
        return TeacherCallResult(
            frame=Frame(
                intent="intent_alpha",
                slots={"slot_alpha": "teacher value", "slot_beta": "teacher residual"},
            ),
            raw_response='{"intent":"intent_alpha","slots":{"slot_alpha":"teacher value","slot_beta":"teacher residual"}}',
            usage={"total_tokens": 11},
            model="fake-teacher",
            context_hash="ctx-full",
            prompt_cache_key="cache-full",
        )

    monkeypatch.setattr(
        "darjeeling.targets.nlu.layers.l4_cloud_llm.CloudLLMTeacher.residual_patch",
        fake_residual_patch,
    )
    monkeypatch.setattr(
        "darjeeling.targets.nlu.layers.l4_cloud_llm.CloudLLMTeacher.answer",
        fake_answer,
    )
    settings = load_settings().model_copy(update={"openai_api_key": "test-key"})

    run_replay(
        stream="sequential",
        max_requests=1,
        teacher_mode="live",
        run_dir=run_dir,
        data_dir=data_dir,
        settings=settings,
        target=_PatchRuntimeTarget(),
    )

    trace = json.loads((run_dir / "traces.jsonl").read_text(encoding="utf-8"))
    assert trace["chosen_layer"] == "L4"
    assert trace["teacher_frame"] == {
        "intent": "intent_alpha",
        "slots": {"slot_alpha": "teacher value", "slot_beta": "teacher residual"},
        "is_abstain": False,
    }
    assert trace["metadata"]["residual_l4_unverified_completions"] == 1
    assert trace["metadata"]["residual_l4_full_audit_reason"] == (
        "unverified_accepted_fields"
    )
    assert [result["metadata"]["l4_call_kind"] for result in trace["layer_results"] if result["layer"] == "L4"] == [
        "residual",
        "full",
    ]


def test_run_replay_audits_lower_layer_accept_and_records_disagreement(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_dir = tmp_path / "data"
    run_dir = tmp_path / "run"
    l0_dir = run_dir / "artifacts" / "generations" / "gen_001"
    data_dir.mkdir()
    l0_dir.mkdir(parents=True)
    (data_dir / "train.jsonl").write_text(
        DataRecord(
            request_id="r1",
            utterance="alpha request",
            gold_frame=Frame(intent="intent_alpha"),
        ).model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    (l0_dir / "l0_cache.json").write_text(
        json.dumps(
            {
                "schema_version": "l0-exact-v1",
                "cache_type": "exact",
                "frames_by_normalized_utterance": {
                    "alpha request": {"intent": "intent_beta", "slots": {}}
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    ArtifactStore(run_dir / "artifacts").promote(
        ArtifactManifest(
            artifact_set_id="gen_001_l0",
            generation=1,
            target_name="nlu",
            target_schema_version="nlu-target-v1",
            artifact_paths={"l0_cache": "generations/gen_001/l0_cache.json"},
            promotion_reason="test fixture",
        )
    )

    def fake_answer(self, utterance, task_schema):
        del self, utterance, task_schema
        return TeacherCallResult(
            frame=Frame(intent="intent_alpha"),
            raw_response='{"intent":"intent_alpha","slots":{}}',
            usage={"total_tokens": 7},
            model="fake-teacher",
            context_hash="ctx",
            prompt_cache_key="cache",
        )

    monkeypatch.setattr(
        "darjeeling.targets.nlu.layers.l4_cloud_llm.CloudLLMTeacher.answer",
        fake_answer,
    )
    settings = load_settings().model_copy(update={"openai_api_key": "test-key"})

    run_replay(
        stream="sequential",
        max_requests=1,
        teacher_mode="live",
        run_dir=run_dir,
        data_dir=data_dir,
        settings=settings,
    )

    trace = json.loads((run_dir / "traces.jsonl").read_text(encoding="utf-8"))
    assert trace["chosen_layer"] == "L0"
    assert trace["teacher_frame"]["intent"] == "intent_alpha"
    assert trace["metadata"]["lower_layer_accepted"] is True
    assert trace["metadata"]["teacher_audited"] is True
    assert trace["metadata"]["teacher_audit_source"] == "live"
    assert trace["metadata"]["teacher_disagreed"] is True
    assert trace["metadata"]["teacher_audit_tokens"] == 7
    assert "intent_alpha" in (run_dir / "teacher_cache.jsonl").read_text(encoding="utf-8")


def test_offline_replay_reports_field_level_wrong_accepts() -> None:
    expected = Frame(intent="intent_alpha", slots={"slot_alpha": "teacher value"})
    traces = [
        TraceRecord(
            request_id="r1",
            utterance="alpha request",
            teacher_frame=expected,
            chosen_layer="L4",
            final_frame=expected,
            layer_results=[
                LayerResult(layer="L4", accepted=True, frame=expected, latency_ms=1.0)
            ],
        )
    ]

    result = evaluate_offline_artifact_set(
        traces=traces_to_teacher_view(traces),
        artifact_set=OfflineArtifactSet(
            l0_cache={"alpha request": Frame(intent="intent_alpha", slots={"slot_alpha": "wrong"})}
        ),
    )

    assert result.layer_counts["L0"] == 1
    assert result.field_metrics["weak_accepted_fields"] == 2.0
    assert result.field_metrics["weak_correct_fields"] == 1.0
    assert result.field_metrics["weak_wrong_fields"] == 1.0
    assert result.field_metrics["wrong_accepted_field_rate"] == 0.5
    assert result.layer_metrics["L0"]["field_accepted_accuracy"] == 0.5
