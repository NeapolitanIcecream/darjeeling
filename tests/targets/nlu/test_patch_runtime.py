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
from darjeeling.targets.nlu.layers.l4_cloud_llm import TeacherCallResult
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


def test_route_nlu_layers_composes_partial_patches_and_l4_residual() -> None:
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
        slots={"slot_alpha": "lower value", "slot_beta": "teacher residual"},
    )
    assert result.layer_results[0].metadata["patch_accepted_fields"] == ["intent"]
    assert result.layer_results[1].metadata["patch_accepted_fields"] == ["slots.slot_alpha"]
    assert result.layer_results[2].patch is not None
    assert result.layer_results[2].patch.accepted_slots == {"slot_beta": "teacher residual"}
    assert result.composer.field_sources == {
        "intent": "L1",
        "slots.slot_alpha": "L2",
        "slots.slot_beta": "L4",
    }


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
