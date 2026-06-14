from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from darjeeling.artifacts.store import ArtifactManifest, ArtifactStore
from darjeeling.contracts import CompileContext
from darjeeling.contracts import LayerResult as CoreLayerResult
from darjeeling.contracts import TeacherTrace as CoreTeacherTrace
from darjeeling.targets.nlu.data import DataRecord
from darjeeling.targets.nlu.layers.l0_cache import ExactCacheLayer
from darjeeling.targets.nlu.layers.l1_rust_programbank import (
    build_l1_binary,
)
from darjeeling.targets.nlu.layers.l2_experts import L2ExpertBank, L2ExpertBankLayer
from darjeeling.targets.nlu.layers.l2_student import L2StudentBundle, L2StudentLayer
from darjeeling.targets.nlu.layers.l2_target import TargetL2Layer
from darjeeling.targets.nlu.layers.l3_local_slm import (
    L3PromptArtifact,
)
from darjeeling.targets.nlu.layers.l4_cloud_llm import (
    CachedTeacherLayer,
    MissingTeacherError,
    TaskSchema,
    TeacherCache,
)
from darjeeling.targets.nlu.patches import legacy_layer_result_from_core, route_nlu_layers
from darjeeling.targets.nlu.schemas import Frame, TraceRecord
from darjeeling.targets.nlu.settings import Settings
from darjeeling.targets.nlu.streams import StreamItem, build_uniform_stream, build_zipf_stream
from darjeeling.targets.nlu.target import NluLayerAdapter, NluTarget
from darjeeling.targets.nlu.trace import TraceWriter, read_traces


@dataclass(frozen=True)
class ReplaySummary:
    requests: int
    layer_counts: dict[str, int]
    traces_path: Path


def run_replay(
    *,
    stream: str,
    max_requests: int,
    teacher_mode: str,
    run_dir: Path,
    data_dir: Path,
    settings: Settings,
    compile_every: int | None = None,
    target: NluTarget | None = None,
) -> ReplaySummary:
    target = target or NluTarget()
    records = load_processed_records(data_dir)
    stream_items = select_stream(records, stream=stream, max_requests=max_requests)
    task_schema = task_schema_from_records(records)

    teacher_cache_path = run_dir / "teacher_cache.jsonl"
    teacher_cache = TeacherCache.load(teacher_cache_path)
    l4 = CachedTeacherLayer(
        teacher_cache,
        allow_live=teacher_mode in {"live", "live-or-cache"},
        use_cache=teacher_mode in {"cache", "live-or-cache"},
        settings=settings,
        task_schema=task_schema,
    )
    traces_path = run_dir / "traces.jsonl"
    trace_writer = TraceWriter(traces_path)
    layer_counts: dict[str, int] = {"L0": 0, "L1": 0, "L2": 0, "L3": 0, "L4": 0}

    runtime_layers = _build_runtime_layers(
        target=target,
        run_dir=run_dir,
        settings=settings,
        task_schema=task_schema,
        teacher_layer=l4,
    )
    try:
        for index, item in enumerate(stream_items, start=1):
            record = item.record
            utterance = record.utterance
            route_result = route_nlu_layers(runtime_layers, utterance=utterance)
            final_frame = route_result.final_frame
            chosen_layer = route_result.chosen_layer
            layer_results = route_result.layer_results

            layer_counts[chosen_layer] += 1
            teacher_frame, audit_metadata = _lower_layer_audit(
                l4,
                request_id=record.request_id,
                utterance=utterance,
                chosen_layer=chosen_layer,
                final_frame=final_frame,
                settings=settings,
                teacher_mode=teacher_mode,
            )
            trace_writer.append(
                TraceRecord(
                    request_id=record.request_id,
                    utterance=utterance,
                    gold_frame=record.gold_frame,
                    teacher_frame=teacher_frame,
                    chosen_layer=chosen_layer,
                    final_frame=final_frame,
                    layer_results=layer_results,
                    l4_usage=route_result.l4_usage,
                    metadata={
                        **audit_metadata,
                        "composer_field_sources": route_result.composer.field_sources,
                        "field_conflicts": route_result.composer.field_conflicts,
                        "field_overrides": route_result.composer.field_overrides,
                        "verified_fields": route_result.composer.verified_fields,
                    },
                )
            )
            if compile_every is not None and index % compile_every == 0:
                _run_target_compiler(
                    target=target,
                    run_dir=run_dir,
                    task_schema=task_schema,
                    settings=settings,
                    traces=read_traces(traces_path),
                )
                _close_runtime_layers(runtime_layers)
                runtime_layers = _build_runtime_layers(
                    target=target,
                    run_dir=run_dir,
                    settings=settings,
                    task_schema=task_schema,
                    teacher_layer=l4,
                )
    finally:
        _close_runtime_layers(runtime_layers)

    return ReplaySummary(
        requests=len(stream_items),
        layer_counts=layer_counts,
        traces_path=traces_path,
    )


def load_processed_records(data_dir: Path, *, split: str = "train") -> list[DataRecord]:
    path = data_dir / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"processed data split not found: {path}; prepare a dataset first"
        )
    return [
        DataRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def task_schema_from_records(records: list[DataRecord]) -> TaskSchema:
    intent_names = sorted({record.gold_frame.intent for record in records})
    slot_names = sorted({slot_name for record in records for slot_name in record.gold_frame.slots})
    return TaskSchema(intent_names=intent_names, slot_names=slot_names)


def load_l2_layer_from_manifest(
    run_dir: Path,
    *,
    settings: Settings | None = None,
) -> L2StudentLayer | None:
    if settings is not None and not settings.l2_enabled:
        return None
    manifest = ArtifactStore(run_dir / "artifacts").load_current_manifest()
    if manifest is None:
        return None
    l2_path_text = manifest.artifact_paths.get("l2_student")
    if not l2_path_text:
        return None
    l2_path = Path(l2_path_text)
    if not l2_path.is_absolute():
        l2_path = run_dir / "artifacts" / l2_path
    if not l2_path.exists():
        raise FileNotFoundError(f"L2 student artifact is missing: {l2_path}")
    bundle = L2StudentBundle.load(l2_path)
    if settings is not None and settings.l2_guard_mode == "always_accept":
        bundle.config.accept_threshold = 0.0
    target_path_text = manifest.artifact_paths.get("l2_target")
    if target_path_text:
        target_path = Path(target_path_text)
        if not target_path.is_absolute():
            target_path = run_dir / "artifacts" / target_path
        if not target_path.exists():
            raise FileNotFoundError(f"L2 target artifact is missing: {target_path}")
        fallback_layer = TargetL2Layer(bundle, target_path)
    else:
        fallback_layer = L2StudentLayer(bundle)
    expert_bank_path_text = manifest.artifact_paths.get("l2_expert_bank")
    if expert_bank_path_text:
        expert_bank_path = Path(expert_bank_path_text)
        if not expert_bank_path.is_absolute():
            expert_bank_path = run_dir / "artifacts" / expert_bank_path
        if not expert_bank_path.exists():
            raise FileNotFoundError(f"L2 expert bank artifact is missing: {expert_bank_path}")
        return L2ExpertBankLayer(L2ExpertBank.load(expert_bank_path), fallback_layer)
    return fallback_layer


def load_l1_source_dir_from_manifest(run_dir: Path) -> Path | None:
    manifest = ArtifactStore(run_dir / "artifacts").load_current_manifest()
    if manifest is None:
        return None
    l1_path_text = manifest.artifact_paths.get("l1_crate_dir")
    if not l1_path_text:
        return None
    l1_path = Path(l1_path_text)
    if not l1_path.is_absolute():
        l1_path = run_dir / "artifacts" / l1_path
    if not l1_path.exists():
        raise FileNotFoundError(f"L1 crate artifact is missing: {l1_path}")
    return l1_path


def load_l3_prompt_artifact_from_manifest(run_dir: Path) -> L3PromptArtifact | None:
    manifest = ArtifactStore(run_dir / "artifacts").load_current_manifest()
    if manifest is None:
        return None
    l3_path_text = manifest.artifact_paths.get("l3_prompt")
    if not l3_path_text:
        return None
    l3_path = Path(l3_path_text)
    if not l3_path.is_absolute():
        l3_path = run_dir / "artifacts" / l3_path
    if not l3_path.exists():
        raise FileNotFoundError(f"L3 prompt artifact is missing: {l3_path}")
    return L3PromptArtifact.model_validate_json(l3_path.read_text(encoding="utf-8"))


def l1_binary_path_for_source(source_dir: Path, *, settings: Settings) -> Path:
    if settings.l1_rust_binary is not None and source_dir == settings.l1_rust_crate_dir:
        return settings.l1_rust_binary
    return build_l1_binary(source_dir)


def load_l0_layer_from_manifest(run_dir: Path) -> ExactCacheLayer:
    manifest = ArtifactStore(run_dir / "artifacts").load_current_manifest()
    if manifest is None:
        return ExactCacheLayer({})
    l0_path_text = manifest.artifact_paths.get("l0_cache")
    if not l0_path_text:
        return ExactCacheLayer({})
    l0_path = Path(l0_path_text)
    if not l0_path.is_absolute():
        l0_path = run_dir / "artifacts" / l0_path
    if not l0_path.exists():
        raise FileNotFoundError(f"L0 cache artifact is missing: {l0_path}")
    payload = json.loads(l0_path.read_text(encoding="utf-8"))
    frames = {
        normalized_utterance: Frame.model_validate(frame_payload)
        for normalized_utterance, frame_payload in payload.get(
            "frames_by_normalized_utterance", {}
        ).items()
    }
    return ExactCacheLayer(frames)


def select_stream(
    records: list[DataRecord],
    *,
    stream: str,
    max_requests: int,
) -> list:
    if not records:
        return []
    if stream == "sequential":
        return [
            StreamItem(index=index, record=record)
            for index, record in enumerate(records[:max_requests])
        ]
    if stream == "uniform":
        return build_uniform_stream(records, max_requests=max_requests)
    if stream == "zipf-mild":
        return build_zipf_stream(records, max_requests=max_requests, exponent=0.8)
    if stream == "zipf-heavy":
        return build_zipf_stream(records, max_requests=max_requests, exponent=1.2)
    raise ValueError(f"unsupported stream type: {stream}")


def write_run_settings(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_runtime_layers(
    *,
    target: NluTarget,
    run_dir: Path,
    settings: Settings,
    task_schema: TaskSchema,
    teacher_layer: CachedTeacherLayer,
) -> dict[str, Any]:
    manifest = ArtifactStore(run_dir / "artifacts").load_current_manifest()
    _validate_manifest_target_identity(manifest, target=target)
    return dict(
        target.runtime.build_layers(
            manifest=manifest,
            teacher=NluLayerAdapter(teacher_layer, layer_name="L4"),
            settings={
                "run_dir": str(run_dir),
                "artifact_root": str(run_dir / "artifacts"),
                "task_schema": task_schema.to_payload(),
                "nlu_settings": settings.model_dump(mode="json"),
            },
        )
    )


def _validate_manifest_target_identity(
    manifest: ArtifactManifest | None,
    *,
    target: NluTarget,
) -> None:
    if manifest is None:
        return
    if (
        manifest.target_name == target.name
        and manifest.target_schema_version == target.schema_version
    ):
        return
    actual_name = manifest.target_name or "<missing>"
    actual_version = manifest.target_schema_version or "<missing>"
    raise ValueError(
        "artifact manifest target mismatch: "
        f"expected {target.name}/{target.schema_version}, "
        f"got {actual_name}/{actual_version}"
    )


def _close_runtime_layers(runtime_layers: Mapping[str, Any]) -> None:
    for layer in runtime_layers.values():
        close = getattr(layer, "close", None)
        if close is not None:
            close()


def _legacy_layer_result(result: CoreLayerResult):
    return legacy_layer_result_from_core(result)


def _lower_layer_audit(
    teacher_layer: CachedTeacherLayer,
    *,
    request_id: str,
    utterance: str,
    chosen_layer: str,
    final_frame: Frame,
    settings: Settings,
    teacher_mode: str,
) -> tuple[Frame | None, dict[str, Any]]:
    lower_layer_accepted = chosen_layer in {"L0", "L1", "L2", "L3"}
    teacher_frame = teacher_layer.cache.get(utterance)
    metadata: dict[str, Any] = {
        "lower_layer_accepted": lower_layer_accepted,
        "lower_layer_accepted_layer": chosen_layer if lower_layer_accepted else None,
        "teacher_audit_mode": settings.lower_layer_audit_mode,
        "teacher_audited": False,
        "teacher_audit_source": None,
        "teacher_audit_skipped_reason": None,
        "teacher_disagreed": False,
    }
    if not lower_layer_accepted:
        return teacher_frame, metadata
    if teacher_frame is not None:
        metadata["teacher_audited"] = True
        metadata["teacher_audit_source"] = "cache"
        metadata["teacher_disagreed"] = teacher_frame != final_frame
        return teacher_frame, metadata
    if not _should_audit_lower_accept(
        request_id=request_id,
        utterance=utterance,
        settings=settings,
    ):
        metadata["teacher_audit_skipped_reason"] = "sampling"
        return None, metadata
    if teacher_mode not in {"live", "live-or-cache"}:
        metadata["teacher_audit_skipped_reason"] = "teacher_live_disabled"
        return None, metadata
    try:
        audit_result = teacher_layer.try_answer(utterance)
    except MissingTeacherError as exc:
        metadata["teacher_audit_skipped_reason"] = str(exc)
        return None, metadata
    if audit_result.frame is None:
        metadata["teacher_audit_skipped_reason"] = "teacher_returned_no_frame"
        return None, metadata
    teacher_frame = audit_result.frame
    metadata["teacher_audited"] = True
    metadata["teacher_audit_source"] = audit_result.metadata.get("teacher_source", "live")
    metadata["teacher_disagreed"] = teacher_frame != final_frame
    metadata["teacher_audit_latency_ms"] = audit_result.latency_ms
    metadata["teacher_audit_cost_usd"] = audit_result.cost_usd
    usage = audit_result.metadata.get("usage")
    if isinstance(usage, dict):
        metadata["teacher_audit_tokens"] = _usage_tokens(usage)
    return teacher_frame, metadata


def _should_audit_lower_accept(
    *,
    request_id: str,
    utterance: str,
    settings: Settings,
) -> bool:
    if settings.lower_layer_audit_mode == "disabled":
        return False
    if settings.lower_layer_audit_mode == "always":
        return True
    if settings.lower_layer_audit_sample_rate <= 0.0:
        return False
    if settings.lower_layer_audit_sample_rate >= 1.0:
        return True
    digest = sha256(f"{request_id}\0{utterance}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
    return value <= settings.lower_layer_audit_sample_rate


def _usage_tokens(usage: dict[str, Any]) -> int:
    total = usage.get("total_tokens")
    if isinstance(total, int | float) and not isinstance(total, bool):
        return int(total)
    prompt = usage.get("prompt_tokens", usage.get("input_tokens", 0))
    completion = usage.get("completion_tokens", usage.get("output_tokens", 0))
    tokens = 0
    for value in (prompt, completion):
        if isinstance(value, int | float) and not isinstance(value, bool):
            tokens += int(value)
    return tokens


def _core_teacher_trace(trace: TraceRecord) -> CoreTeacherTrace:
    return CoreTeacherTrace(
        request_id=trace.request_id,
        input={"utterance": trace.utterance},
        teacher_label=trace.teacher_frame.model_dump(mode="json")
        if trace.teacher_frame is not None
        else None,
        chosen_layer=trace.chosen_layer,
        final_output=trace.final_frame.model_dump(mode="json"),
        layer_results=[
            CoreLayerResult(
                layer=result.layer,
                accepted=result.accepted,
                output=result.frame.model_dump(mode="json") if result.frame is not None else None,
                confidence=result.confidence,
                reason=result.reason,
                latency_ms=result.latency_ms,
                cost_usd=result.cost_usd,
                metadata=result.metadata,
            )
            for result in trace.layer_results
        ],
        l4_usage=trace.l4_usage,
        timestamp=trace.timestamp,
    )


def _run_target_compiler(
    *,
    target: NluTarget,
    run_dir: Path,
    task_schema: TaskSchema,
    settings: Settings,
    traces: list[TraceRecord],
) -> None:
    target.compiler.propose_artifacts(
        CompileContext(
            run_dir=run_dir,
            task_schema=task_schema.to_payload(),
            teacher_traces=[_core_teacher_trace(trace) for trace in traces],
            current_manifest=ArtifactStore(run_dir / "artifacts").load_current_manifest(),
            settings={"nlu_settings": settings.model_dump(mode="json")},
        )
    )
