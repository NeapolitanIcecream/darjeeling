from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from darjeeling.artifacts.store import ArtifactStore
from darjeeling.settings import Settings
from darjeeling.targets.nlu.compiler.loop import run_compiler_generation
from darjeeling.targets.nlu.data import DataRecord
from darjeeling.targets.nlu.layers.l0_cache import ExactCacheLayer
from darjeeling.targets.nlu.layers.l1_rust_programbank import (
    RustL1Worker,
    RustProgramBankLayer,
    build_l1_binary,
)
from darjeeling.targets.nlu.layers.l2_student import L2StudentBundle, L2StudentLayer
from darjeeling.targets.nlu.layers.l2_target import TargetL2Layer
from darjeeling.targets.nlu.layers.l3_local_slm import (
    L3LocalSLMLayer,
    L3PromptArtifact,
    build_l3_layer_from_settings,
)
from darjeeling.targets.nlu.layers.l4_cloud_llm import (
    CachedTeacherLayer,
    MissingTeacherError,
    TaskSchema,
    TeacherCache,
)
from darjeeling.targets.nlu.schemas import Frame, TraceRecord
from darjeeling.targets.nlu.streams import StreamItem, build_uniform_stream, build_zipf_stream
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
) -> ReplaySummary:
    records = load_processed_records(data_dir)
    stream_items = select_stream(records, stream=stream, max_requests=max_requests)
    task_schema = task_schema_from_records(records)

    teacher_cache_path = run_dir / "teacher_cache.jsonl"
    teacher_cache = TeacherCache.load(teacher_cache_path)
    l0 = load_l0_layer_from_manifest(run_dir)
    l4 = CachedTeacherLayer(
        teacher_cache,
        allow_live=teacher_mode in {"live", "live-or-cache"},
        use_cache=teacher_mode in {"cache", "live-or-cache"},
        settings=settings,
        task_schema=task_schema,
    )

    l1_source_dir = load_l1_source_dir_from_manifest(run_dir) or settings.l1_rust_crate_dir
    binary_path = l1_binary_path_for_source(l1_source_dir, settings=settings)
    l2_layer = load_l2_layer_from_manifest(run_dir, settings=settings)
    l3_layer = build_l3_layer_from_settings(
        settings=settings,
        task_schema=task_schema,
        prompt_artifact=load_l3_prompt_artifact_from_manifest(run_dir),
    )
    traces_path = run_dir / "traces.jsonl"
    trace_writer = TraceWriter(traces_path)
    layer_counts: dict[str, int] = {"L0": 0, "L1": 0, "L2": 0, "L3": 0, "L4": 0}

    rust_worker = RustL1Worker(binary_path, timeout_s=settings.l1_worker_timeout_s)
    try:
        l1 = RustProgramBankLayer(rust_worker)
        for index, item in enumerate(stream_items, start=1):
            record = item.record
            utterance = record.utterance
            layer_results = []

            l0_result = l0.try_answer(utterance)
            layer_results.append(l0_result)
            if l0_result.accepted and l0_result.frame is not None:
                final_frame = l0_result.frame
                chosen_layer = "L0"
            else:
                l1_result = l1.try_answer(utterance)
                layer_results.append(l1_result)
                if l1_result.accepted and l1_result.frame is not None:
                    final_frame = l1_result.frame
                    chosen_layer = "L1"
                else:
                    if l2_layer is not None:
                        l2_result = l2_layer.try_answer(utterance)
                        layer_results.append(l2_result)
                        if l2_result.accepted and l2_result.frame is not None:
                            final_frame = l2_result.frame
                            chosen_layer = "L2"
                        else:
                            final_frame, chosen_layer = _answer_with_l3_or_l4(
                                utterance=utterance,
                                l3_layer=l3_layer,
                                l4_layer=l4,
                                layer_results=layer_results,
                            )
                    else:
                        final_frame, chosen_layer = _answer_with_l3_or_l4(
                            utterance=utterance,
                            l3_layer=l3_layer,
                            l4_layer=l4,
                            layer_results=layer_results,
                        )

            layer_counts[chosen_layer] += 1
            teacher_frame = teacher_cache.get(utterance)
            trace_writer.append(
                TraceRecord(
                    request_id=record.request_id,
                    utterance=utterance,
                    gold_frame=record.gold_frame,
                    teacher_frame=teacher_frame,
                    chosen_layer=chosen_layer,
                    final_frame=final_frame,
                    layer_results=layer_results,
                )
            )
            if compile_every is not None and index % compile_every == 0:
                run_compiler_generation(
                    run_dir=run_dir,
                    traces=read_traces(traces_path),
                    settings=settings,
                )
                l0 = load_l0_layer_from_manifest(run_dir)
                l2_layer = load_l2_layer_from_manifest(run_dir, settings=settings)
                next_l1_source_dir = (
                    load_l1_source_dir_from_manifest(run_dir) or settings.l1_rust_crate_dir
                )
                if next_l1_source_dir != l1_source_dir:
                    rust_worker.close()
                    l1_source_dir = next_l1_source_dir
                    binary_path = l1_binary_path_for_source(l1_source_dir, settings=settings)
                    rust_worker = RustL1Worker(
                        binary_path,
                        timeout_s=settings.l1_worker_timeout_s,
                    )
                    l1 = RustProgramBankLayer(rust_worker)
    finally:
        rust_worker.close()

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
        return TargetL2Layer(bundle, target_path)
    return L2StudentLayer(bundle)


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


def _answer_with_l3_or_l4(
    *,
    utterance: str,
    l3_layer: L3LocalSLMLayer,
    l4_layer: CachedTeacherLayer,
    layer_results: list[Any],
) -> tuple[Frame, str]:
    l3_result = l3_layer.try_answer(utterance)
    layer_results.append(l3_result)
    if l3_result.accepted and l3_result.frame is not None:
        return l3_result.frame, "L3"

    l4_result = l4_layer.try_answer(utterance)
    layer_results.append(l4_result)
    if not l4_result.accepted or l4_result.frame is None:
        raise MissingTeacherError(f"L4 did not produce a frame for {utterance!r}")
    return l4_result.frame, "L4"
