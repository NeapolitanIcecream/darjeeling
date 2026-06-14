import json
from pathlib import Path

from darjeeling.artifacts.store import ArtifactStore
from darjeeling.targets.nlu.compiler.loop import run_compiler_generation
from darjeeling.targets.nlu.layers.l2_experts import (
    L2ExpertBank,
    L2ExpertBankLayer,
    L2ExpertTrainingConfig,
    train_l2_expert_bank,
)
from darjeeling.targets.nlu.schemas import (
    Frame,
    FramePatch,
    LayerResult,
    TraceRecord,
    traces_to_teacher_view,
)
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
    assert manifest["selection_metrics"]["schema_version"] == "l2-expert-selection-v2"
    assert manifest["selection_metrics"]["training_traces"] == 6
    assert manifest["selection_metrics"]["validation_traces"] == 2
    assert manifest["selection_metrics"]["split_policy"] == "intent_stratified"
    assert manifest["selection_metrics"]["validation_intents"] == {
        "intent_alarm": 1,
        "intent_weather": 1,
    }
    assert [expert["intent"] for expert in manifest["intent_experts"]] == [
        "intent_alarm"
    ]
    assert [expert["slot_key"] for expert in manifest["slot_experts"]] == ["slot_time"]
    assert manifest["intent_experts"][0]["validation_metrics"]["positive_examples"] == 1
    assert manifest["intent_experts"][0]["validation_metrics"]["negative_examples"] == 1
    assert manifest["slot_experts"][0]["validation_metrics"]["wrong_accepts"] == 0

    result = L2ExpertBankLayer(bank).try_answer("please alarm at seven am")

    assert result.accepted is True
    assert result.frame is None
    assert result.patch is not None
    assert result.patch.accepted_intent == "intent_alarm"
    assert result.patch.accepted_slots == {"slot_time": "seven am"}
    assert result.patch.complete is False
    assert result.metadata["frame_patch"]["accepted_intent"] == "intent_alarm"


def test_l2_expert_bank_abstains_on_close_intent_conflict() -> None:
    bank = L2ExpertBank(
        intent_experts=[
            _StaticIntentExpert("intent_alpha", 0.91),
            _StaticIntentExpert("intent_beta", 0.88),
        ],
        intent_conflict_margin=0.05,
    )

    patch, fired = bank.try_patch("ambiguous request")

    assert patch is None
    assert fired[-1]["policy"] == "intent_conflict_abstain"


def test_l2_expert_bank_skips_slots_incompatible_with_selected_intent() -> None:
    bank = L2ExpertBank(
        intent_experts=[_StaticIntentExpert("intent_alpha", 0.95)],
        slot_experts=[_StaticSlotExpert("slot_beta", "value beta")],
        intent_conflict_margin=0.05,
        slot_signatures_by_intent={"intent_alpha": {"slot_alpha"}},
    )

    patch, fired = bank.try_patch("alpha with beta-looking value")

    assert patch is not None
    assert patch.accepted_intent == "intent_alpha"
    assert patch.accepted_slots == {}
    assert any(item.get("policy") == "slot_intent_signature_abstain" for item in fired)


def test_l2_expert_bank_uses_stable_split_for_tiny_datasets() -> None:
    traces = traces_to_teacher_view(_expert_traces()[:4])

    bank = train_l2_expert_bank(
        traces,
        L2ExpertTrainingConfig(
            min_examples=2,
            max_intents=1,
            max_slots=1,
            min_accuracy=0.0,
        ),
    )

    assert bank is not None
    assert bank.manifest_payload()["selection_metrics"]["split_policy"] == (
        "stable_request_id_stride"
    )


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


class _StaticIntentExpert:
    def __init__(self, intent: str, confidence: float) -> None:
        self.intent = intent
        self.confidence = confidence
        self.name = f"intent:{intent}"

    def try_patch(self, _utterance: str):
        metadata = {
            "expert": self.name,
            "probability": self.confidence,
            "threshold": 0.0,
        }
        return (
            FramePatch(
                accepted_intent=self.intent,
                source_layer="L2",
                confidence=self.confidence,
            ),
            metadata,
        )


class _StaticSlotExpert:
    def __init__(self, slot_key: str, value: str) -> None:
        self.slot_key = slot_key
        self.value = value
        self.name = f"slot:{slot_key}"

    def try_patch(self, _utterance: str):
        metadata = {"expert": self.name, "threshold": 1.0}
        return (
            FramePatch(
                accepted_slots={self.slot_key: self.value},
                source_layer="L2",
                confidence=1.0,
            ),
            metadata,
        )


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
