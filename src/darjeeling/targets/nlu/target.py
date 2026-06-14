from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from darjeeling.artifacts.store import ArtifactManifest
from darjeeling.contracts import (
    ArtifactCandidate,
    CompileContext,
    JsonObject,
    LayerName,
    RuntimeLayer,
    TeacherRuntime,
    TeacherTrace,
)
from darjeeling.contracts import (
    LayerResult as CoreLayerResult,
)
from darjeeling.targets.nlu.data import normalize_utterance
from darjeeling.targets.nlu.patches import (
    core_layer_result_from_legacy,
    legacy_layer_result_from_core,
)
from darjeeling.targets.nlu.schemas import Frame, LayerResult, TaskSchema, TraceRecord
from darjeeling.targets.nlu.teacher import NluTeacherAdapter


class NluTargetSpec:
    name = "nlu"
    schema_version = "nlu-target-v1"

    def load_task_schema(self, records: Sequence[JsonObject]) -> JsonObject:
        frames = [
            Frame.model_validate(record["gold_frame"])
            for record in records
            if record.get("gold_frame") is not None
        ]
        intent_names = sorted({frame.intent for frame in frames})
        slot_names = sorted({slot_name for frame in frames for slot_name in frame.slots})
        return TaskSchema(intent_names=intent_names, slot_names=slot_names).to_payload()

    def normalize_request(self, input: JsonObject) -> str:
        utterance = input.get("utterance")
        if not isinstance(utterance, str):
            raise ValueError("NLU request requires utterance")
        return normalize_utterance(utterance)

    def validate_output(self, output: JsonObject, task_schema: JsonObject) -> None:
        schema = TaskSchema.from_payload(task_schema)
        frame = Frame.model_validate(output)
        if frame.intent not in schema.intent_names:
            raise ValueError(f"intent not allowed: {frame.intent}")
        unsupported_slots = sorted(set(frame.slots) - set(schema.slot_names))
        if unsupported_slots:
            raise ValueError(f"slot not allowed: {unsupported_slots[0]}")

    def labels_equal(
        self,
        output: JsonObject,
        expected: JsonObject,
        *,
        task_schema: JsonObject,
    ) -> bool:
        del task_schema
        return Frame.model_validate(output) == Frame.model_validate(expected)

    def summarize_for_context(
        self,
        traces: Sequence[TeacherTrace],
        *,
        budget: int,
    ) -> JsonObject:
        examples: list[JsonObject] = []
        for trace in traces[: max(0, budget)]:
            if trace.teacher_label is None:
                continue
            examples.append(
                {
                    "request_id": trace.request_id,
                    "input": trace.input,
                    "teacher_label": trace.teacher_label,
                    "chosen_layer": trace.chosen_layer,
                }
            )
        return {"target": self.name, "examples": examples}


class NluTargetRuntime:
    def build_layers(
        self,
        *,
        manifest: ArtifactManifest | None,
        teacher: TeacherRuntime,
        settings: JsonObject,
    ) -> Mapping[LayerName, RuntimeLayer | None]:
        from darjeeling.targets.nlu.layers.l1_rust_programbank import (
            RustL1Worker,
            RustProgramBankLayer,
            build_l1_binary,
        )
        from darjeeling.targets.nlu.layers.l2_experts import L2ExpertBank, L2ExpertBankLayer
        from darjeeling.targets.nlu.layers.l2_student import L2StudentBundle, L2StudentLayer
        from darjeeling.targets.nlu.layers.l2_target import TargetL2Layer
        from darjeeling.targets.nlu.layers.l3_local_slm import (
            L3PromptArtifact,
            build_l3_layer_from_settings,
        )
        from darjeeling.targets.nlu.settings import Settings

        nlu_settings = Settings.model_validate(settings["nlu_settings"])
        run_dir = Path(str(settings["run_dir"]))
        artifact_root = Path(str(settings.get("artifact_root", run_dir / "artifacts")))
        task_schema = TaskSchema.from_payload(_json_object(settings["task_schema"]))

        l0_layer = NluLayerAdapter(
            _l0_layer_from_manifest(manifest, artifact_root),
            layer_name="L0",
        )

        l1_source_dir = _l1_source_dir_from_manifest(manifest, artifact_root)
        if l1_source_dir is None:
            l1_source_dir = nlu_settings.l1_rust_crate_dir
        if (
            nlu_settings.l1_rust_binary is not None
            and l1_source_dir == nlu_settings.l1_rust_crate_dir
        ):
            l1_binary = nlu_settings.l1_rust_binary
        else:
            l1_binary = build_l1_binary(l1_source_dir)
        l1_worker = RustL1Worker(l1_binary, timeout_s=nlu_settings.l1_worker_timeout_s)
        l1_layer = NluLayerAdapter(
            RustProgramBankLayer(l1_worker),
            layer_name="L1",
            close=l1_worker.close,
        )

        l2_layer: RuntimeLayer | None = None
        if nlu_settings.l2_enabled:
            l2_path = _artifact_path(manifest, artifact_root, "l2_student")
            if l2_path is not None:
                bundle = L2StudentBundle.load(l2_path)
                if nlu_settings.l2_guard_mode == "always_accept":
                    bundle.config.accept_threshold = 0.0
                target_path = _artifact_path(manifest, artifact_root, "l2_target")
                legacy_l2_layer = (
                    TargetL2Layer(bundle, target_path)
                    if target_path is not None
                    else L2StudentLayer(bundle)
                )
                expert_bank_path = _artifact_path(manifest, artifact_root, "l2_expert_bank")
                if expert_bank_path is not None:
                    legacy_l2_layer = L2ExpertBankLayer(
                        L2ExpertBank.load(expert_bank_path),
                        legacy_l2_layer,
                    )
                l2_layer = NluLayerAdapter(legacy_l2_layer, layer_name="L2")

        l3_prompt_path = _artifact_path(manifest, artifact_root, "l3_prompt")
        prompt_artifact = (
            L3PromptArtifact.model_validate_json(l3_prompt_path.read_text(encoding="utf-8"))
            if l3_prompt_path is not None
            else None
        )
        l3_layer = NluLayerAdapter(
            build_l3_layer_from_settings(
                settings=nlu_settings,
                task_schema=task_schema,
                prompt_artifact=prompt_artifact,
            ),
            layer_name="L3",
        )

        return {
            "L0": l0_layer,
            "L1": l1_layer,
            "L2": l2_layer,
            "L3": l3_layer,
            "L4": teacher,
        }


class NluTargetCompiler:
    def propose_artifacts(
        self,
        context: CompileContext,
    ) -> Sequence[ArtifactCandidate]:
        from darjeeling.targets.nlu.compiler.loop import run_compiler_generation
        from darjeeling.targets.nlu.settings import Settings

        settings_payload = context.settings.get("nlu_settings", context.settings)
        nlu_settings = Settings.model_validate(settings_payload)
        result = run_compiler_generation(
            run_dir=context.run_dir,
            traces=[_legacy_trace_from_teacher_trace(trace) for trace in context.teacher_traces],
            settings=nlu_settings,
        )
        if result.manifest is None:
            return ()
        return (
            ArtifactCandidate(
                artifact_paths=result.manifest.artifact_paths,
                metadata={
                    "generation": result.generation,
                    "promoted": result.promoted,
                    "reason": result.reason,
                },
            ),
        )


class NluLayerAdapter:
    def __init__(
        self,
        legacy_layer: Any,
        *,
        layer_name: LayerName | None = None,
        close: Any | None = None,
    ) -> None:
        self.legacy_layer = legacy_layer
        self.layer_name = layer_name or getattr(legacy_layer, "layer_name", "L4")
        self._close = close

    def try_answer(self, input: JsonObject) -> CoreLayerResult:
        utterance = _request_utterance(input)
        return _core_layer_result_from_legacy(self.legacy_layer.try_answer(utterance))

    def residual_field_keys(self) -> list[str]:
        field_keys = getattr(self.legacy_layer, "residual_field_keys", None)
        if field_keys is None:
            return []
        return list(field_keys())

    def try_residual_patch(self, input: JsonObject) -> CoreLayerResult:
        residual = getattr(self.legacy_layer, "try_residual_patch", None)
        if residual is None:
            raise AttributeError("legacy layer does not support residual patches")
        utterance = _request_utterance(input)
        accepted_fields = input.get("accepted_fields")
        missing_fields = input.get("missing_fields")
        if not isinstance(accepted_fields, dict):
            accepted_fields = {}
        if not isinstance(missing_fields, list):
            missing_fields = []
        result = residual(
            utterance,
            accepted_fields={str(key): str(value) for key, value in accepted_fields.items()},
            missing_fields=[str(field) for field in missing_fields],
        )
        return _core_layer_result_from_legacy(result)

    def close(self) -> None:
        if self._close is not None:
            self._close()


class NluTarget(NluTargetSpec):
    def __init__(self) -> None:
        self.teacher_adapter = NluTeacherAdapter()
        self.runtime = NluTargetRuntime()
        self.compiler = NluTargetCompiler()

    def load_settings(self, *, settings_path: Path | None = None):
        from darjeeling.targets.nlu.settings import load_settings

        return load_settings(settings_path)

    def run_replay(
        self,
        *,
        stream: str,
        max_requests: int,
        compile_every: int,
        teacher: str,
        run_dir: Path,
        data_dir: Path,
        settings: Any,
    ):
        from darjeeling.targets.nlu.replay import run_replay

        return run_replay(
            stream=stream,
            max_requests=max_requests,
            teacher_mode=teacher,
            run_dir=run_dir,
            data_dir=data_dir,
            settings=settings,
            compile_every=compile_every,
            target=self,
        )

    def generate_report(self, *, run_dir: Path):
        from darjeeling.targets.nlu.reports import generate_run_report

        return generate_run_report(run_dir)


def _request_utterance(input: JsonObject) -> str:
    utterance = input.get("utterance")
    if not isinstance(utterance, str):
        raise ValueError("NLU request requires utterance")
    return utterance


def _json_object(value: object) -> JsonObject:
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value


def _artifact_path(
    manifest: ArtifactManifest | None,
    artifact_root: Path,
    key: str,
) -> Path | None:
    if manifest is None:
        return None
    path_text = manifest.artifact_paths.get(key)
    if not path_text:
        return None
    path = Path(path_text)
    if not path.is_absolute():
        path = artifact_root / path
    if not path.exists():
        raise FileNotFoundError(f"{key} artifact is missing: {path}")
    return path


def _l0_layer_from_manifest(
    manifest: ArtifactManifest | None,
    artifact_root: Path,
):
    from darjeeling.targets.nlu.layers.l0_cache import ExactCacheLayer

    l0_path = _artifact_path(manifest, artifact_root, "l0_cache")
    if l0_path is None:
        return ExactCacheLayer({})
    payload = json.loads(l0_path.read_text(encoding="utf-8"))
    frames = {
        normalized_utterance: Frame.model_validate(frame_payload)
        for normalized_utterance, frame_payload in payload.get(
            "frames_by_normalized_utterance", {}
        ).items()
    }
    return ExactCacheLayer(frames)


def _l1_source_dir_from_manifest(
    manifest: ArtifactManifest | None,
    artifact_root: Path,
) -> Path | None:
    if manifest is None:
        return None
    path_text = manifest.artifact_paths.get("l1_crate_dir")
    if not path_text:
        return None
    path = Path(path_text)
    if not path.is_absolute():
        path = artifact_root / path
    if not path.exists():
        raise FileNotFoundError(f"L1 crate artifact is missing: {path}")
    return path


def _core_layer_result_from_legacy(result: LayerResult) -> CoreLayerResult:
    return core_layer_result_from_legacy(result)


def _legacy_layer_result_from_core(result: CoreLayerResult) -> LayerResult:
    return legacy_layer_result_from_core(result)


def _legacy_trace_from_teacher_trace(trace: TeacherTrace) -> TraceRecord:
    utterance = _request_utterance(trace.input)
    teacher_frame = Frame.model_validate(trace.teacher_label) if trace.teacher_label else None
    return TraceRecord(
        request_id=trace.request_id,
        utterance=utterance,
        gold_frame=None,
        teacher_frame=teacher_frame,
        chosen_layer=trace.chosen_layer,
        final_frame=Frame.model_validate(trace.final_output),
        layer_results=[_legacy_layer_result_from_core(result) for result in trace.layer_results],
        l4_usage=trace.l4_usage,
        timestamp=trace.timestamp,
    )
