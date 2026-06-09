from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from darjeeling.artifacts.store import ArtifactManifest, ArtifactStore, LayerDelta
from darjeeling.compiler.guard_optimizer import (
    GUARD_PROPOSAL_SCHEMA,
    GuardSearchSpec,
    evaluate_l2_unguarded,
    guard_search_spec_from_proposal,
    select_l2_accept_threshold,
)
from darjeeling.compiler.l0_compile import exact_cache_from_teacher_traces
from darjeeling.compiler.l1_program_compiler import L1CodingAgentError, L4CodingAgentAdapter
from darjeeling.compiler.l2_distiller import (
    L2_CONFIG_PROPOSAL_SCHEMA,
    l2_config_from_proposal,
    l2_config_from_settings,
)
from darjeeling.compiler.l2_tuner import L2TuneSpec, tune_l2_student
from darjeeling.compiler.l3_prompt_optimizer import (
    L3_PROMPT_PROPOSAL_SCHEMA,
    l3_prompt_artifact_from_proposal,
)
from darjeeling.compiler.l4_proposal import L4ProposalAdapter, ProposalParseError
from darjeeling.compiler.mining import (
    HardCase,
    build_hard_buffer,
    hard_case_reason_counts,
    hard_case_traces,
    hard_case_visibility_counts,
    load_hard_buffer_jsonl,
    merge_hard_buffers,
    write_hard_buffer_jsonl,
)
from darjeeling.compiler.replay import (
    OfflineArtifactSet,
    decide_artifact_set_promotion,
    evaluate_offline_artifact_set,
    layer_deltas,
    load_offline_artifact_set,
    split_teacher_traces,
)
from darjeeling.layers.l1_rust_programbank import (
    DEFAULT_BENCHMARK_UTTERANCES,
    benchmark_worker,
    build_l1_binary,
)
from darjeeling.layers.l2_student import (
    L2StudentConfig,
    train_l2_student,
    training_examples_from_teacher_traces,
)
from darjeeling.layers.l4_cloud_llm import MissingTeacherError, TaskSchema
from darjeeling.runtime.cost import replay_cost_model_from_settings
from darjeeling.schemas import TeacherTrace, TraceRecord, traces_to_teacher_view
from darjeeling.settings import Settings


def compiler_inputs_from_traces(traces: list[TraceRecord]) -> list[TeacherTrace]:
    return traces_to_teacher_view(traces)


def assert_teacher_visible_only(traces: list[TeacherTrace]) -> None:
    for trace in traces:
        dumped = trace.model_dump()
        if "gold_frame" in dumped:
            raise AssertionError("compiler-visible trace contains gold_frame")


@dataclass(frozen=True)
class CompilerGenerationResult:
    generation: int
    promoted: bool
    reason: str
    manifest: ArtifactManifest | None = None


def run_compiler_generation(
    *,
    run_dir: Path,
    traces: list[TraceRecord],
    settings: Settings,
) -> CompilerGenerationResult:
    teacher_traces = compiler_inputs_from_traces(traces)
    assert_teacher_visible_only(teacher_traces)

    store = ArtifactStore(run_dir / "artifacts")
    current_manifest = store.load_current_manifest()
    current_artifacts = load_offline_artifact_set(
        store.root,
        current_manifest,
        default_l1_crate_dir=settings.l1_rust_crate_dir,
    )
    current_artifacts = _apply_l2_ablation_to_offline_artifacts(current_artifacts, settings)
    generation = 1 if current_manifest is None else current_manifest.generation + 1
    generation_dir = store.generation_dir(generation)
    artifact_paths: dict[str, str] = (
        dict(current_manifest.artifact_paths) if current_manifest is not None else {}
    )
    candidate_metrics: dict[str, object] = {
        "teacher_traces_seen": len(teacher_traces),
        "compiler_slice": "l0_exact_l2_train_replay_gate",
    }
    split = split_teacher_traces(teacher_traces)
    candidate_metrics.update(
        {
            "teacher_train_size": len(split.teacher_train),
            "teacher_promotion_holdout_size": len(split.teacher_promotion_holdout),
            "teacher_regression_sample_size": len(split.teacher_regression_sample),
        }
    )
    new_train_hard_buffer = build_hard_buffer(
        split.teacher_train,
        max_cases=settings.hard_buffer_max_cases,
        visibility="train_visible",
    )
    new_replay_only_hard_buffer = build_hard_buffer(
        split.evaluation_traces,
        max_cases=settings.hard_buffer_max_cases,
        visibility="replay_only",
    )
    previous_hard_buffer = _load_previous_hard_buffer(store.root, current_manifest)
    hard_buffer = merge_hard_buffers(
        [previous_hard_buffer, new_train_hard_buffer, new_replay_only_hard_buffer],
        max_cases=settings.hard_buffer_max_cases,
    )
    hard_buffer_path = write_hard_buffer_jsonl(
        generation_dir / "hard_buffer.jsonl",
        hard_buffer,
    )
    agent_hard_buffer_traces = hard_case_traces(
        hard_buffer,
        visibility={"train_visible"},
    )
    replay_pressure_traces = hard_case_traces(hard_buffer)
    artifact_paths["hard_buffer"] = _artifact_relative_path(store.root, hard_buffer_path)
    candidate_metrics.update(
        {
            "hard_buffer_size": len(hard_buffer),
            "hard_buffer_new_size": len(new_train_hard_buffer) + len(new_replay_only_hard_buffer),
            "hard_buffer_new_train_visible_size": len(new_train_hard_buffer),
            "hard_buffer_new_replay_only_size": len(new_replay_only_hard_buffer),
            "hard_buffer_previous_size": len(previous_hard_buffer),
            "hard_buffer_reason_counts": hard_case_reason_counts(hard_buffer),
            "hard_buffer_visibility_counts": hard_case_visibility_counts(hard_buffer),
            "hard_buffer_agent_context_size": len(agent_hard_buffer_traces),
            "hard_buffer_replay_pressure_size": len(replay_pressure_traces),
            "hard_buffer_source": "current_manifest + teacher_train + replay_only_evaluation",
        }
    )
    generated_artifacts = False
    l1_candidate_generated = False

    exact_cache = exact_cache_from_teacher_traces(split.teacher_train)
    candidate_l0_cache = {**current_artifacts.l0_cache, **exact_cache}
    if exact_cache:
        l0_payload = {
            "schema_version": "l0-exact-v1",
            "cache_type": "exact",
            "frames_by_normalized_utterance": {
                key: frame.model_dump(mode="json")
                for key, frame in sorted(candidate_l0_cache.items())
            },
        }
        l0_path = store.write_generation_json(generation, "l0_cache.json", l0_payload)
        artifact_paths["l0_cache"] = _artifact_relative_path(store.root, l0_path)
        candidate_metrics["l0_cache_lines"] = len(candidate_l0_cache)
        candidate_metrics["l0_new_cache_lines"] = len(exact_cache)
        generated_artifacts = True
    else:
        candidate_metrics["l0_cache_lines"] = len(candidate_l0_cache)
        candidate_metrics["l0_new_cache_lines"] = 0

    candidate_l2_bundle = current_artifacts.l2_bundle
    candidate_l1_crate_dir = current_artifacts.l1_crate_dir
    lower_miss_train_traces = _l2_lower_miss_traces(split.teacher_train)
    l2_training_traces = _l2_training_traces_for_scope(
        split.teacher_train,
        scope=settings.l2_training_scope,
    )
    l2_examples = training_examples_from_teacher_traces(l2_training_traces)
    candidate_metrics["l2_training_scope"] = settings.l2_training_scope
    candidate_metrics["l2_teacher_train_traces"] = len(split.teacher_train)
    candidate_metrics["l2_lower_miss_train_traces"] = len(lower_miss_train_traces)
    candidate_metrics["l2_training_traces"] = len(l2_training_traces)
    candidate_metrics["l2_examples"] = len(l2_examples)
    candidate_metrics["l2_enabled"] = settings.l2_enabled
    candidate_metrics["l2_guard_mode"] = settings.l2_guard_mode
    candidate_metrics["l2_tuning_mode"] = settings.l2_tuning_mode
    candidate_metrics["l2_tuning_min_examples"] = settings.l2_tuning_min_examples
    candidate_metrics["l2_tuning_split_policy"] = settings.l2_tuning_split_policy
    candidate_metrics["l4_proposal_mode"] = settings.l4_proposal_mode
    l2_config = l2_config_from_settings(settings)
    guard_search_spec = GuardSearchSpec(
        max_wrong_accept_rate=settings.l2_max_wrong_accept_rate,
    )
    if not settings.l2_enabled:
        candidate_l2_bundle = None
        artifact_paths.pop("l2_student", None)
        candidate_metrics["l2_trained"] = False
        candidate_metrics["l2_training_error"] = "L2 disabled by settings"
    elif settings.l4_proposal_mode == "live" and l2_examples:
        try:
            l2_proposal_result = L4ProposalAdapter(settings).propose(
                role="l2",
                task_schema=_task_schema_from_teacher_traces(l2_training_traces),
                traces=l2_training_traces,
                output_schema=L2_CONFIG_PROPOSAL_SCHEMA,
                current_artifact_summary=_artifact_summary(current_manifest),
                metrics=candidate_metrics,
            )
            l2_config = l2_config_from_proposal(
                l2_proposal_result.proposal,
                default=l2_config,
            )
        except (MissingTeacherError, ProposalParseError, ValueError) as exc:
            candidate_metrics["l4_l2_proposal_succeeded"] = False
            candidate_metrics["l4_l2_proposal_error"] = str(exc)
        else:
            candidate_metrics["l4_l2_proposal_succeeded"] = True
            candidate_metrics["l4_l2_proposal"] = l2_proposal_result.proposal
            candidate_metrics["l4_l2_proposal_context_hash"] = l2_proposal_result.context_hash
            candidate_metrics["l4_l2_proposal_prompt_cache_key"] = (
                l2_proposal_result.prompt_cache_key
            )
            candidate_metrics["l4_l2_proposal_source_trace_ids"] = (
                l2_proposal_result.source_trace_ids
            )
        try:
            guard_proposal_result = L4ProposalAdapter(settings).propose(
                role="guard",
                task_schema=_task_schema_from_teacher_traces(l2_training_traces),
                traces=l2_training_traces,
                output_schema=GUARD_PROPOSAL_SCHEMA,
                current_artifact_summary=_artifact_summary(current_manifest),
                metrics=candidate_metrics,
            )
            guard_search_spec = guard_search_spec_from_proposal(
                guard_proposal_result.proposal,
                default_max_wrong_accept_rate=settings.l2_max_wrong_accept_rate,
            )
        except (MissingTeacherError, ProposalParseError, ValueError) as exc:
            candidate_metrics["l4_guard_proposal_succeeded"] = False
            candidate_metrics["l4_guard_proposal_error"] = str(exc)
        else:
            guard_path = store.write_generation_json(
                generation,
                "guard/guard_candidate.json",
                asdict(guard_search_spec),
            )
            artifact_paths["guard_candidate"] = _artifact_relative_path(store.root, guard_path)
            candidate_metrics["l4_guard_proposal_succeeded"] = True
            candidate_metrics["l4_guard_proposal"] = guard_proposal_result.proposal
            candidate_metrics["l4_guard_proposal_context_hash"] = guard_proposal_result.context_hash
            candidate_metrics["l4_guard_proposal_prompt_cache_key"] = (
                guard_proposal_result.prompt_cache_key
            )
            candidate_metrics["l4_guard_proposal_source_trace_ids"] = (
                guard_proposal_result.source_trace_ids
            )
            candidate_metrics["guard_search_spec"] = asdict(guard_search_spec)
            generated_artifacts = True
    if settings.l2_enabled and settings.l2_tuning_mode == "optuna" and l2_examples:
        try:
            if len(l2_examples) < settings.l2_tuning_min_examples:
                candidate_metrics["l2_tuning_succeeded"] = False
                candidate_metrics["l2_tuning_skipped_reason"] = (
                    f"requires at least {settings.l2_tuning_min_examples} examples"
                )
            else:
                tune_spec = L2TuneSpec(
                    n_trials=settings.l2_tuning_trials,
                    timeout_s=settings.l2_tuning_timeout_s,
                    validation_fraction=settings.l2_tuning_validation_fraction,
                    split_policy=settings.l2_tuning_split_policy,
                    random_state=l2_config.random_state,
                    search_space=settings.l2_tuning_search_space,
                    max_wrong_accept_rate=guard_search_spec.max_wrong_accept_rate,
                    min_accepted_accuracy=settings.l2_min_guarded_accuracy,
                    latency_weight=settings.l2_tuning_latency_weight,
                )
                tune_result = tune_l2_student(
                    l2_training_traces,
                    base_config=l2_config,
                    spec=tune_spec,
                )
                tune_path = store.write_generation_json(
                    generation,
                    "l2/l2_tuning.json",
                    tune_result.model_dump(mode="json"),
                )
                artifact_paths["l2_tuning"] = _artifact_relative_path(store.root, tune_path)
                candidate_metrics["l2_tuning_succeeded"] = tune_result.best_config is not None
                candidate_metrics["l2_tuning"] = {
                    "schema_version": tune_result.schema_version,
                    "train_size": tune_result.train_size,
                    "validation_size": tune_result.validation_size,
                    "split_policy": tune_result.split_policy,
                    "n_trials_requested": tune_result.n_trials_requested,
                    "n_trials_completed": tune_result.n_trials_completed,
                    "best_trial_number": tune_result.best_trial_number,
                    "best_value": tune_result.best_value,
                    "best_metrics": tune_result.best_metrics,
                }
                generated_artifacts = True
                if tune_result.best_config is not None:
                    l2_config = L2StudentConfig.model_validate(tune_result.best_config)
        except (ImportError, ValueError) as exc:
            candidate_metrics["l2_tuning_succeeded"] = False
            candidate_metrics["l2_tuning_error"] = str(exc)
    if settings.l2_enabled:
        candidate_metrics["l2_config"] = l2_config.model_dump(mode="json")
        try:
            l2_bundle = train_l2_student(l2_examples, l2_config)
        except ValueError as exc:
            candidate_metrics["l2_trained"] = False
            candidate_metrics["l2_training_error"] = str(exc)
        else:
            unguarded_evaluation = evaluate_l2_unguarded(l2_bundle, l2_training_traces)
            candidate_metrics["l2_unguarded_train"] = _threshold_evaluation_payload(
                unguarded_evaluation
            )
            if settings.l2_training_scope != "teacher_train":
                candidate_metrics["l2_unguarded_teacher_train"] = (
                    _threshold_evaluation_payload(
                        evaluate_l2_unguarded(l2_bundle, split.teacher_train),
                    )
                )
            if settings.l2_guard_mode == "always_accept":
                l2_bundle.config.runtime_enabled = True
                l2_bundle.config.accept_threshold = 0.0
                candidate_metrics["l2_guard_threshold"] = 0.0
                candidate_metrics["l2_runtime_enabled"] = True
                candidate_metrics["l2_guard_search"] = {
                    "selected": {
                        "threshold": 0.0,
                        "coverage": 1.0,
                        "accepted_accuracy": None,
                        "wrong_accept_rate": None,
                        "correct_accepts": None,
                        "wrong_accepts": None,
                        "accepted": None,
                        "total": len(l2_examples),
                    },
                    "candidates": [],
                    "mode": "always_accept",
                }
            else:
                threshold_selection = select_l2_accept_threshold(
                    l2_bundle,
                    l2_training_traces,
                    grid=guard_search_spec.grid,
                    max_wrong_accept_rate=guard_search_spec.max_wrong_accept_rate,
                    min_accepted_accuracy=settings.l2_min_guarded_accuracy,
                )
                if threshold_selection is not None:
                    l2_bundle.config.accept_threshold = threshold_selection.threshold
                    candidate_metrics["l2_guard_threshold"] = threshold_selection.threshold
                    candidate_metrics["l2_guard_search"] = {
                        "selected": _threshold_evaluation_payload(threshold_selection.evaluation),
                        "candidates": [
                            _threshold_evaluation_payload(candidate)
                            for candidate in threshold_selection.candidates
                        ],
                    }
                runtime_enabled = len(l2_examples) >= settings.l2_min_runtime_examples
                l2_bundle.config.runtime_enabled = runtime_enabled
                candidate_metrics["l2_runtime_enabled"] = runtime_enabled
                candidate_metrics["l2_min_runtime_examples"] = settings.l2_min_runtime_examples
                if not runtime_enabled:
                    candidate_metrics["l2_runtime_disabled_reason"] = (
                        f"requires at least {settings.l2_min_runtime_examples} examples"
                    )
            l2_dir = generation_dir / "l2"
            l2_path = l2_dir / "l2_student.joblib"
            l2_bundle.save(l2_path)
            artifact_paths["l2_student"] = _artifact_relative_path(store.root, l2_path)
            candidate_metrics["l2_trained"] = True
            candidate_l2_bundle = l2_bundle
            generated_artifacts = True

    if settings.l4_proposal_mode == "live" and split.teacher_train:
        try:
            l3_proposal_result = L4ProposalAdapter(settings).propose(
                role="l3",
                task_schema=_task_schema_from_teacher_traces(split.teacher_train),
                traces=split.teacher_train,
                output_schema=L3_PROMPT_PROPOSAL_SCHEMA,
                current_artifact_summary=_artifact_summary(current_manifest),
                metrics=candidate_metrics,
            )
            l3_prompt_artifact = l3_prompt_artifact_from_proposal(
                l3_proposal_result.proposal,
                traces=split.teacher_train,
                prompt_version=f"{settings.local_slm_prompt_version}-candidate-gen-{generation:03d}",
            )
        except (MissingTeacherError, ProposalParseError, ValueError) as exc:
            candidate_metrics["l4_l3_prompt_proposal_succeeded"] = False
            candidate_metrics["l4_l3_prompt_proposal_error"] = str(exc)
        else:
            l3_path = store.write_generation_json(
                generation,
                "l3/l3_prompt.candidate.json",
                l3_prompt_artifact.model_dump(mode="json"),
            )
            artifact_paths["l3_prompt_candidate"] = _artifact_relative_path(store.root, l3_path)
            candidate_metrics["l4_l3_prompt_proposal_succeeded"] = True
            candidate_metrics["l4_l3_prompt_proposal"] = l3_proposal_result.proposal
            candidate_metrics["l4_l3_prompt_proposal_context_hash"] = (
                l3_proposal_result.context_hash
            )
            candidate_metrics["l4_l3_prompt_proposal_prompt_cache_key"] = (
                l3_proposal_result.prompt_cache_key
            )
            candidate_metrics["l4_l3_prompt_proposal_source_trace_ids"] = (
                l3_proposal_result.source_trace_ids
            )
            candidate_metrics["l3_prompt_candidate_runtime_promoted"] = False
            candidate_metrics["l3_prompt_candidate_promotion_blocker"] = (
                "requires regenerated or shadow L3 replay before runtime promotion"
            )
            generated_artifacts = True

    if settings.l1_agent_mode != "disabled":
        candidate_metrics["l1_agent_mode"] = settings.l1_agent_mode
        try:
            l1_agent_result = L4CodingAgentAdapter(settings).run_l1_job(
                job_dir=generation_dir / "l1_agent",
                source_crate_dir=candidate_l1_crate_dir or settings.l1_rust_crate_dir,
                teacher_train=split.teacher_train,
                hard_cases=agent_hard_buffer_traces,
                current_metrics=(
                    current_manifest.candidate_metrics if current_manifest is not None else {}
                ),
                objective={
                    "accuracy_epsilon": settings.promotion_accuracy_epsilon,
                    "wrong_accept_limit": settings.l2_max_wrong_accept_rate,
                },
                run_validation=True,
            )
        except L1CodingAgentError as exc:
            candidate_metrics["l1_agent_succeeded"] = False
            candidate_metrics["l1_agent_error"] = str(exc)
        else:
            candidate_metrics["l1_agent_succeeded"] = l1_agent_result.succeeded
            candidate_metrics["l1_agent_return_code"] = l1_agent_result.return_code
            artifact_paths["l1_agent_dir"] = _artifact_relative_path(
                store.root,
                l1_agent_result.job_dir,
            )
            artifact_paths["l1_agent_diff"] = _artifact_relative_path(
                store.root,
                l1_agent_result.diff_path,
            )
            artifact_paths["l1_agent_report"] = _artifact_relative_path(
                store.root,
                l1_agent_result.report_path,
            )
            artifact_paths["l1_agent_commands"] = _artifact_relative_path(
                store.root,
                l1_agent_result.commands_path,
            )
            artifact_paths["l1_agent_transcript"] = _artifact_relative_path(
                store.root,
                l1_agent_result.transcript_path,
            )
            artifact_paths["l1_agent_provenance"] = _artifact_relative_path(
                store.root,
                l1_agent_result.provenance_path,
            )
            if l1_agent_result.succeeded:
                candidate_l1_crate_dir = l1_agent_result.workspace_crate_dir
                artifact_paths["l1_crate_dir"] = _artifact_relative_path(
                    store.root,
                    l1_agent_result.workspace_crate_dir,
                )
                generated_artifacts = True
                l1_candidate_generated = True
    else:
        candidate_metrics["l1_agent_mode"] = "disabled"

    if l1_candidate_generated and candidate_l1_crate_dir is not None:
        l1_benchmark_payload = _l1_generation_benchmark_payload(
            candidate_l1_crate_dir,
            timeout_s=settings.l1_worker_timeout_s,
        )
        l1_benchmark_path = store.write_generation_json(
            generation,
            "l1/l1_benchmark.json",
            l1_benchmark_payload,
        )
        artifact_paths["l1_benchmark"] = _artifact_relative_path(store.root, l1_benchmark_path)
        candidate_metrics["l1_benchmark_status"] = l1_benchmark_payload.get("status")
        if l1_benchmark_payload.get("status") == "success":
            candidate_metrics["l1_benchmark_native_p95_us"] = l1_benchmark_payload.get(
                "native_p95_us"
            )
            candidate_metrics["l1_benchmark_throughput_qps"] = l1_benchmark_payload.get(
                "throughput_qps"
            )
        else:
            candidate_metrics["l1_benchmark_error"] = l1_benchmark_payload.get("error")

    if not generated_artifacts:
        return CompilerGenerationResult(
            generation=generation,
            promoted=False,
            reason="no teacher-visible artifacts generated",
        )

    candidate_artifacts = OfflineArtifactSet(
        l0_cache=candidate_l0_cache,
        l1_crate_dir=candidate_l1_crate_dir,
        l2_bundle=candidate_l2_bundle,
    )
    evaluation_traces = (
        _dedupe_teacher_traces([*split.evaluation_traces, *replay_pressure_traces])
        if split.evaluation_traces
        else []
    )
    if not evaluation_traces:
        decision_reason = "promotion replay coverage is empty"
        current_replay = None
        candidate_replay = None
        per_layer_deltas: dict[str, LayerDelta] = {}
        promoted = False
        promoted_with_layer_regression = False
        regressed_layers: list[str] = []
    else:
        cost_model = replay_cost_model_from_settings(settings)
        current_replay = evaluate_offline_artifact_set(
            evaluation_traces,
            current_artifacts,
            cost_model=cost_model,
        )
        candidate_replay = evaluate_offline_artifact_set(
            evaluation_traces,
            candidate_artifacts,
            cost_model=cost_model,
        )
        per_layer_deltas = layer_deltas(current_replay, candidate_replay)
        decision = decide_artifact_set_promotion(
            current_replay.objective,
            candidate_replay.objective,
            per_layer_deltas=per_layer_deltas,
            accuracy_epsilon=settings.promotion_accuracy_epsilon,
            max_wrong_accept_rate=settings.l2_max_wrong_accept_rate,
        )
        decision_reason = decision.reason
        promoted = decision.promoted
        promoted_with_layer_regression = decision.promoted_with_layer_regression
        regressed_layers = decision.regressed_layers or []
        if settings.force_promote_artifacts:
            candidate_metrics["force_promote_artifacts"] = True
            candidate_metrics["force_promote_original_reason"] = decision_reason
            promoted = True
            decision_reason = f"force promoted by settings after: {decision_reason}"

    candidate_metrics.update(
        {
            "promotion_eval_size": len(evaluation_traces),
            "promotion_eval_hard_buffer_size": len(replay_pressure_traces)
            if split.evaluation_traces
            else 0,
            "current_objective": _objective_payload(current_replay.objective)
            if current_replay is not None
            else None,
            "candidate_objective": _objective_payload(candidate_replay.objective)
            if candidate_replay is not None
            else None,
            "current_layer_counts": current_replay.layer_counts
            if current_replay is not None
            else None,
            "candidate_layer_counts": candidate_replay.layer_counts
            if candidate_replay is not None
            else None,
        }
    )

    manifest = ArtifactManifest(
        artifact_set_id=f"gen_{generation:03d}_candidate",
        generation=generation,
        parent_artifact_set_id=(
            current_manifest.artifact_set_id if current_manifest is not None else None
        ),
        schema_versions={
            "artifact_manifest": "artifact-manifest-v1",
            "l0_cache": "l0-exact-v1",
        },
        artifact_paths=artifact_paths,
        candidate_metrics=candidate_metrics,
        per_layer_deltas=per_layer_deltas,
        promoted_with_layer_regression=promoted_with_layer_regression,
        regressed_layers=regressed_layers,
        promotion_reason=decision_reason,
        l3_mode=settings.local_slm_mode,
    )
    metrics_csv_path = _write_candidate_metrics_csv(
        generation_dir / "candidate_metrics.csv",
        manifest,
    )
    promotion_json_path = _write_promotion_json(
        generation_dir / "promotion.json",
        manifest,
        promoted=promoted,
        current_score=(current_replay.objective if current_replay is not None else None),
        candidate_score=(candidate_replay.objective if candidate_replay is not None else None),
    )
    manifest.artifact_paths["candidate_metrics_csv"] = _artifact_relative_path(
        store.root,
        metrics_csv_path,
    )
    manifest.artifact_paths["promotion_record"] = _artifact_relative_path(
        store.root,
        promotion_json_path,
    )
    if promoted:
        store.promote(manifest)
    else:
        store.write_generation_manifest(manifest)
    return CompilerGenerationResult(
        generation=generation,
        promoted=promoted,
        reason=decision_reason,
        manifest=manifest,
    )


def _artifact_relative_path(root: Path, path: Path) -> str:
    return str(path.relative_to(root))


def _apply_l2_ablation_to_offline_artifacts(
    artifacts: OfflineArtifactSet,
    settings: Settings,
) -> OfflineArtifactSet:
    if not settings.l2_enabled:
        return OfflineArtifactSet(
            l0_cache=artifacts.l0_cache,
            l1_crate_dir=artifacts.l1_crate_dir,
            l2_bundle=None,
        )
    if settings.l2_guard_mode == "always_accept" and artifacts.l2_bundle is not None:
        artifacts.l2_bundle.config.accept_threshold = 0.0
    return artifacts


def _load_previous_hard_buffer(
    artifacts_root: Path,
    current_manifest: ArtifactManifest | None,
) -> list[HardCase]:
    if current_manifest is None:
        return []
    hard_buffer_path_text = current_manifest.artifact_paths.get("hard_buffer")
    if not hard_buffer_path_text:
        return []
    hard_buffer_path = Path(hard_buffer_path_text)
    if not hard_buffer_path.is_absolute():
        hard_buffer_path = artifacts_root / hard_buffer_path
    return load_hard_buffer_jsonl(hard_buffer_path)


def _dedupe_teacher_traces(traces: list[TeacherTrace]) -> list[TeacherTrace]:
    seen: set[str] = set()
    deduped: list[TeacherTrace] = []
    for trace in traces:
        if trace.request_id in seen:
            continue
        seen.add(trace.request_id)
        deduped.append(trace)
    return deduped


def _l2_training_traces_for_scope(
    traces: list[TeacherTrace],
    *,
    scope: str,
) -> list[TeacherTrace]:
    if scope == "teacher_train":
        return traces
    if scope == "lower_miss":
        return _l2_lower_miss_traces(traces)
    raise ValueError(f"unsupported L2 training scope: {scope}")


def _l2_lower_miss_traces(traces: list[TeacherTrace]) -> list[TeacherTrace]:
    return [
        trace
        for trace in traces
        if trace.teacher_frame is not None and not _lower_layer_accepted(trace)
    ]


def _lower_layer_accepted(trace: TeacherTrace) -> bool:
    return any(
        result.layer in {"L0", "L1"} and result.accepted and result.frame is not None
        for result in trace.layer_results
    )


def _task_schema_from_teacher_traces(traces: list[TeacherTrace]) -> TaskSchema:
    teacher_frames = [trace.teacher_frame for trace in traces if trace.teacher_frame is not None]
    return TaskSchema(
        intent_names=sorted({frame.intent for frame in teacher_frames}),
        slot_names=sorted({slot_name for frame in teacher_frames for slot_name in frame.slots}),
        schema_version="teacher-trace-window-v1",
    )


def _artifact_summary(manifest: ArtifactManifest | None) -> dict[str, object]:
    if manifest is None:
        return {}
    return {
        "artifact_set_id": manifest.artifact_set_id,
        "generation": manifest.generation,
        "artifact_paths": sorted(manifest.artifact_paths),
        "candidate_metrics": manifest.candidate_metrics,
        "promotion_reason": manifest.promotion_reason,
    }


def _objective_payload(metrics) -> dict[str, float]:
    return {
        "frame_exact_match": metrics.frame_exact_match,
        "wrong_accept_rate": metrics.wrong_accept_rate,
        "cost_usd_per_100_requests": metrics.cost_usd_per_100_requests,
        "p95_latency_ms": metrics.p95_latency_ms,
        "artifact_complexity": metrics.artifact_complexity,
    }


def _threshold_evaluation_payload(evaluation) -> dict[str, float | int]:
    return {
        "threshold": evaluation.threshold,
        "coverage": evaluation.coverage,
        "accepted_accuracy": evaluation.accepted_accuracy,
        "wrong_accept_rate": evaluation.wrong_accept_rate,
        "correct_accepts": evaluation.correct_accepts,
        "wrong_accepts": evaluation.wrong_accepts,
        "accepted": evaluation.accepted,
        "total": evaluation.total,
    }


def _l1_generation_benchmark_payload(crate_dir: Path, *, timeout_s: float) -> dict[str, Any]:
    try:
        binary_path = build_l1_binary(crate_dir)
        metrics = benchmark_worker(
            binary_path,
            DEFAULT_BENCHMARK_UTTERANCES,
            timeout_s=timeout_s,
        )
        return {
            "schema_version": "l1-benchmark-v1",
            "status": "success",
            "corpus": "default_smoke",
            "crate_dir": str(crate_dir),
            "binary_path": str(binary_path),
            "source_size_bytes": _l1_source_size_bytes(crate_dir),
            "binary_size_bytes": binary_path.stat().st_size,
            **metrics,
        }
    except Exception as exc:
        return {
            "schema_version": "l1-benchmark-v1",
            "status": "error",
            "corpus": "default_smoke",
            "crate_dir": str(crate_dir),
            "error": str(exc),
        }


def _l1_source_size_bytes(crate_dir: Path) -> int:
    if not crate_dir.exists() or not crate_dir.is_dir():
        return 0
    total = 0
    for path in crate_dir.rglob("*"):
        if not path.is_file() or "target" in path.relative_to(crate_dir).parts:
            continue
        if path.suffix in {".rs", ".toml", ".lock"}:
            total += path.stat().st_size
    return total


def _write_candidate_metrics_csv(path: Path, manifest: ArtifactManifest) -> Path:
    rows = _candidate_metric_rows(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["generation", "artifact_set_id", "scope", "metric", "value"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def _candidate_metric_rows(manifest: ArtifactManifest) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric, value in sorted(manifest.candidate_metrics.items()):
        if isinstance(value, dict):
            for nested_metric, nested_value in sorted(value.items()):
                if isinstance(nested_value, dict):
                    for leaf_metric, leaf_value in sorted(nested_value.items()):
                        rows.append(
                            _metric_row(
                                manifest,
                                f"{metric}.{nested_metric}",
                                leaf_metric,
                                leaf_value,
                            )
                        )
                else:
                    rows.append(_metric_row(manifest, metric, nested_metric, nested_value))
        else:
            rows.append(_metric_row(manifest, "candidate", metric, value))
    for layer, delta in sorted(manifest.per_layer_deltas.items()):
        for metric, value in sorted(delta.model_dump(mode="json").items()):
            rows.append(_metric_row(manifest, f"layer_delta.{layer}", metric, value))
    return rows


def _metric_row(
    manifest: ArtifactManifest,
    scope: str,
    metric: str,
    value: Any,
) -> dict[str, Any]:
    return {
        "generation": manifest.generation,
        "artifact_set_id": manifest.artifact_set_id,
        "scope": scope,
        "metric": metric,
        "value": json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value,
    }


def _write_promotion_json(
    path: Path,
    manifest: ArtifactManifest,
    *,
    promoted: bool,
    current_score,
    candidate_score,
) -> Path:
    payload = {
        "artifact_set_id": manifest.artifact_set_id,
        "generation": manifest.generation,
        "parent_artifact_set_id": manifest.parent_artifact_set_id,
        "promoted": promoted,
        "promotion_reason": manifest.promotion_reason,
        "promoted_with_layer_regression": manifest.promoted_with_layer_regression,
        "regressed_layers": manifest.regressed_layers,
        "current_objective": _objective_payload(current_score)
        if current_score is not None
        else None,
        "candidate_objective": (
            _objective_payload(candidate_score) if candidate_score is not None else None
        ),
        "per_layer_deltas": {
            layer: delta.model_dump(mode="json")
            for layer, delta in sorted(manifest.per_layer_deltas.items())
        },
        "candidate_metrics": manifest.candidate_metrics,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
