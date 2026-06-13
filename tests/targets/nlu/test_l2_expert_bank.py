import json
from pathlib import Path

from darjeeling.artifacts.store import ArtifactStore
from darjeeling.targets.nlu.compiler.loop import run_compiler_generation
from darjeeling.targets.nlu.layers.l2_experts import (
    L2ExpertBankLayer,
    L2ExpertTrainingConfig,
    train_l2_expert_bank,
)
from darjeeling.targets.nlu.schemas import Frame, LayerResult, TraceRecord, traces_to_teacher_view
from darjeeling.targets.nlu.settings import load_settings


def test_l2_expert_bank_trains_intent_and_slot_experts_that_emit_patch() -> None:
    traces = traces_to_teacher_view(_expert_traces())

    bank = train_l2_expert_bank(
        traces,
        L2ExpertTrainingConfig(
            min_examples=3,
            max_intents=1,
            max_slots=2,
            min_accuracy=0.95,
        ),
    )

    assert bank is not None
    manifest = bank.manifest_payload()
    assert manifest["schema_version"] == "l2-expert-bank-v1"
    assert [expert["intent"] for expert in manifest["intent_experts"]] == [
        "intent_alarm"
    ]
    assert [expert["slot_key"] for expert in manifest["slot_experts"]] == ["slot_time"]

    result = L2ExpertBankLayer(bank).try_answer("please alarm at seven am")

    assert result.accepted is True
    assert result.frame is None
    assert result.patch is not None
    assert result.patch.accepted_intent == "intent_alarm"
    assert result.patch.accepted_slots == {"slot_time": "seven am"}
    assert result.patch.complete is False
    assert result.metadata["frame_patch"]["accepted_intent"] == "intent_alarm"


def test_compiler_generation_writes_l2_expert_bank_artifact(tmp_path: Path) -> None:
    settings = load_settings().model_copy(
        update={
            "l2_expert_min_examples": 3,
            "l2_min_runtime_examples": 4,
            "force_promote_artifacts": True,
        }
    )

    result = run_compiler_generation(
        run_dir=tmp_path,
        traces=_expert_traces(),
        settings=settings,
    )

    assert result.manifest is not None
    assert "l2_expert_bank" in result.manifest.artifact_paths
    assert "l2_expert_manifest" in result.manifest.artifact_paths
    manifest_path = tmp_path / "artifacts" / result.manifest.artifact_paths["l2_expert_manifest"]
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["intent_experts"][0]["intent"] == "intent_alarm"
    assert payload["slot_experts"][0]["slot_key"] == "slot_time"
    metrics = result.manifest.candidate_metrics
    assert metrics["l2_expert_bank_trained"] is True
    assert metrics["l2_expert_bank"]["selection_metrics"]["adopted_slot_experts"] == [
        "slot_time"
    ]
    assert ArtifactStore(tmp_path / "artifacts").load_current_manifest() is not None


def _expert_traces() -> list[TraceRecord]:
    rows = [
        ("r1", "set alarm at seven am", "intent_alarm", {"slot_time": "seven am"}),
        ("r2", "alarm for eight am", "intent_alarm", {"slot_time": "eight am"}),
        ("r3", "wake me at nine am", "intent_alarm", {"slot_time": "nine am"}),
        ("r4", "please alarm at seven am", "intent_alarm", {"slot_time": "seven am"}),
        ("r5", "what is the weather", "intent_weather", {}),
        ("r6", "show weather forecast", "intent_weather", {}),
        ("r7", "weather tomorrow", "intent_weather", {}),
        ("r8", "forecast today", "intent_weather", {}),
    ]
    traces = []
    for request_id, utterance, intent, slots in rows:
        frame = Frame(intent=intent, slots=slots)
        traces.append(
            TraceRecord(
                request_id=request_id,
                utterance=utterance,
                teacher_frame=frame,
                chosen_layer="L4",
                final_frame=frame,
                layer_results=[
                    LayerResult(layer="L4", accepted=True, frame=frame, latency_ms=1.0)
                ],
            )
        )
    return traces
