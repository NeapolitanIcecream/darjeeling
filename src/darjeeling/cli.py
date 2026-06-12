from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console

from darjeeling.artifacts.store import ArtifactManifest, ArtifactStore
from darjeeling.runtime.cost import replay_cost_model_from_settings
from darjeeling.runtime.trace import read_traces
from darjeeling.settings import DEFAULT_PROCESSED_DATA_DIR, load_settings
from darjeeling.targets import registry as target_registry
from darjeeling.targets.nlu.compiler.l2_distiller import l2_config_from_settings
from darjeeling.targets.nlu.compiler.l2_target_evolution import (
    DEFAULT_TARGET_EVOLVE_ROUNDS,
    DEFAULT_TARGET_INNER_PATIENCE_ROUNDS,
    DEFAULT_TARGET_LOCAL_SEARCH_CROSS_AUDIT_TOP_K,
    DEFAULT_TARGET_LOCAL_SEARCH_TRIALS,
    DEFAULT_TARGET_VISIBLE_CROSS_AUDIT_FOLDS,
    L2TargetBudgetProfile,
    L2TargetEvolutionConfig,
    L2TargetEvolutionMode,
    L2TargetScope,
    L2TargetSplitPolicy,
    run_l2_target_evolution,
)
from darjeeling.targets.nlu.compiler.l2_tuner import L2TuneSpec, tune_l2_student
from darjeeling.targets.nlu.compiler.l3_prompt_optimizer import (
    L3PromptEvolutionConfig,
    l3_prompt_artifact_hash,
    replay_l3_prompt_artifact,
    run_l3_prompt_evolution,
)
from darjeeling.targets.nlu.compiler.replay import (
    OfflineReplayResult,
    decide_artifact_set_promotion,
    evaluate_offline_artifact_set,
    layer_deltas,
    load_offline_artifact_set,
)
from darjeeling.targets.nlu.experiments import (
    ExperimentSpec,
    apply_experiment_settings,
    experiment_metadata,
    experiment_spec,
)
from darjeeling.targets.nlu.layers.l1_rust_programbank import (
    DEFAULT_BENCHMARK_UTTERANCES,
    benchmark_worker,
    binary_path_for,
    build_l1_binary,
)
from darjeeling.targets.nlu.layers.l2_student import (
    L2StudentConfig,
    train_l2_student,
    training_examples_from_teacher_traces,
)
from darjeeling.targets.nlu.layers.l2_target import load_target_module, target_config_overrides
from darjeeling.targets.nlu.layers.l3_local_slm import (
    DEFAULT_L3_BENCHMARK_UTTERANCES,
    L3LocalSLMLayer,
    L3PromptArtifact,
    LocalSLMConfig,
    LocalSLMError,
    benchmark_l3_layer,
)
from darjeeling.targets.nlu.layers.l4_cloud_llm import (
    MissingTeacherError,
    TaskSchema,
    require_live_or_cached_teacher,
)
from darjeeling.targets.nlu.replay import (
    load_processed_records,
    run_replay,
    task_schema_from_records,
    write_run_settings,
)
from darjeeling.targets.nlu.reports import (
    generate_experiment_comparison_report,
    generate_run_report,
)
from darjeeling.targets.nlu.schemas import TeacherTrace, traces_to_teacher_view

app = typer.Typer(no_args_is_help=True)
console = Console()
error_console = Console(stderr=True)
_settings_path: Path | None = None


@app.callback()
def main(
    settings: Annotated[
        Path | None,
        typer.Option("--settings", help="Optional settings.yaml file."),
    ] = None,
) -> None:
    """Darjeeling edge runtime CLI."""

    global _settings_path
    _settings_path = settings


def _load_cli_settings():
    return load_settings(_settings_path)


@app.command()
def run(
    stream: Annotated[
        str,
        typer.Option(help="Replay stream type: uniform, zipf-mild, or zipf-heavy."),
    ] = "zipf-heavy",
    max_requests: Annotated[int, typer.Option(min=1)] = 3000,
    compile_every: Annotated[int, typer.Option(min=1)] = 500,
    teacher: Annotated[
        str,
        typer.Option(help="Teacher mode: live, cache, or live-or-cache."),
    ] = "live-or-cache",
    run_dir: Annotated[Path, typer.Option(help="Run output directory.")] = Path("runs/latest"),
    data_dir: Annotated[
        Path,
        typer.Option(help="Processed data directory produced by prepare."),
    ] = DEFAULT_PROCESSED_DATA_DIR,
    target: Annotated[
        str,
        typer.Option(help="Target name from the static target registry."),
    ] = "nlu",
) -> None:
    """Run the replay demo.

    This incremental implementation runs L0 exact cache, Rust L1, optional L2,
    configurable L3 local SLM mode, and cache/live L4 fallback.
    """

    settings = _load_cli_settings()
    try:
        summary = _execute_replay_run(
            stream=stream,
            max_requests=max_requests,
            compile_every=compile_every,
            teacher=teacher,
            run_dir=run_dir,
            data_dir=data_dir,
            target=target,
            settings=settings,
        )
    except (FileNotFoundError, LocalSLMError, MissingTeacherError, ValueError) as exc:
        error_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc

    console.print(
        f"ran {summary.requests} requests; traces={summary.traces_path}; "
        f"layers={summary.layer_counts}"
    )


@app.command()
def report(
    run_dir: Annotated[
        Path,
        typer.Option(help="Run directory to summarize."),
    ] = Path("runs/latest"),
    target: Annotated[
        str,
        typer.Option(help="Target name from the static target registry."),
    ] = "nlu",
) -> None:
    """Generate run report files."""

    target_registry.get_target(target)
    result = generate_run_report(run_dir)
    console.print(f"wrote {result.summary_path}")


def _execute_replay_run(
    *,
    stream: str,
    max_requests: int,
    compile_every: int,
    teacher: str,
    run_dir: Path,
    data_dir: Path,
    target: str = "nlu",
    settings,
):
    target_spec = target_registry.get_target(target)
    run_dir.mkdir(parents=True, exist_ok=True)
    require_live_or_cached_teacher(settings, teacher, run_dir / "teacher_cache.jsonl")
    write_run_settings(
        run_dir / "settings.json",
        _run_settings_payload(
            stream=stream,
            max_requests=max_requests,
            compile_every=compile_every,
            teacher=teacher,
            data_dir=data_dir,
            target_name=target_spec.name,
            target_schema_version=target_spec.schema_version,
            settings=settings,
        ),
    )
    return run_replay(
        stream=stream,
        max_requests=max_requests,
        teacher_mode=teacher,
        run_dir=run_dir,
        data_dir=data_dir,
        settings=settings,
        compile_every=compile_every,
    )


def _run_settings_payload(
    *,
    stream: str,
    max_requests: int,
    compile_every: int,
    teacher: str,
    data_dir: Path,
    target_name: str,
    target_schema_version: str,
    settings,
) -> dict:
    settings_payload = settings.model_dump(mode="json", exclude={"openai_api_key"})
    return {
        **settings_payload,
        "openai_api_key_present": settings.openai_api_key is not None,
        "stream": stream,
        "max_requests": max_requests,
        "compile_every": compile_every,
        "teacher": teacher,
        "data_dir": str(data_dir),
        "target_name": target_name,
        "target_schema_version": target_schema_version,
    }


experiments_app = typer.Typer(no_args_is_help=True)
app.add_typer(experiments_app, name="experiment")

l1_app = typer.Typer(no_args_is_help=True)
app.add_typer(l1_app, name="l1")

l2_app = typer.Typer(no_args_is_help=True)
app.add_typer(l2_app, name="l2")

l3_app = typer.Typer(no_args_is_help=True)
app.add_typer(l3_app, name="l3")


@l1_app.command("build")
def l1_build(
    crate_dir: Annotated[
        Path,
        typer.Option(help="Rust L1 ProgramBank crate directory."),
    ] = Path("native/l1_empty_programbank"),
    release: Annotated[bool, typer.Option(help="Build the release profile.")] = False,
) -> None:
    """Build the Rust L1 ProgramBank worker."""

    binary_path = build_l1_binary(crate_dir, release=release)
    console.print(f"built L1 worker: {binary_path}")


@l1_app.command("bench")
def l1_bench(
    crate_dir: Annotated[
        Path,
        typer.Option(help="Rust L1 ProgramBank crate directory."),
    ] = Path("native/l1_empty_programbank"),
    release: Annotated[bool, typer.Option(help="Use the release profile binary.")] = False,
    out: Annotated[
        Path | None,
        typer.Option(help="Optional JSON path for benchmark metrics."),
    ] = None,
    timeout_s: Annotated[
        float,
        typer.Option(min=0.1, help="Per-request worker timeout in seconds."),
    ] = 2.0,
) -> None:
    """Run a small native L1 worker smoke benchmark."""

    binary_path = binary_path_for(crate_dir, release=release)
    if not binary_path.exists():
        binary_path = build_l1_binary(crate_dir, release=release)
    metrics = benchmark_worker(
        binary_path,
        DEFAULT_BENCHMARK_UTTERANCES,
        timeout_s=timeout_s,
    )
    payload = {
        "schema_version": "l1-benchmark-v1",
        "status": "success",
        "crate_dir": str(crate_dir),
        "binary_path": str(binary_path),
        "release": release,
        **metrics,
    }
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    console.print_json(data=payload)


@l3_app.command("bench")
def l3_bench(
    out: Annotated[
        Path | None,
        typer.Option(help="Optional JSON path for benchmark metrics."),
    ] = None,
    model_name: Annotated[
        str | None,
        typer.Option(help="Override LOCAL_SLM_MODEL."),
    ] = None,
    device_policy: Annotated[
        str | None,
        typer.Option(help="Override LOCAL_SLM_DEVICE_POLICY: auto, cpu, mps, or cuda."),
    ] = None,
    mode: Annotated[
        str,
        typer.Option(help="Benchmark mode: shadow or guarded."),
    ] = "shadow",
    max_new_tokens: Annotated[
        int | None,
        typer.Option(min=1, help="Override LOCAL_SLM_MAX_NEW_TOKENS."),
    ] = None,
    data_dir: Annotated[
        Path,
        typer.Option(help="Optional processed data dir for task schema discovery."),
    ] = DEFAULT_PROCESSED_DATA_DIR,
    fail_on_error: Annotated[
        bool,
        typer.Option(help="Exit non-zero if the local SLM benchmark records an error."),
    ] = False,
) -> None:
    """Run an explicit local L3 SLM hardware/model benchmark."""

    settings = _load_cli_settings()
    payload: dict
    try:
        if mode == "disabled":
            raise ValueError("l3 bench requires mode shadow or guarded")
        config = LocalSLMConfig(
            model_name=model_name or settings.local_slm_model,
            mode=mode,
            device_policy=device_policy or settings.local_slm_device_policy,
            max_new_tokens=max_new_tokens or settings.local_slm_max_new_tokens,
            confidence_threshold=settings.local_slm_confidence_threshold,
            prompt_version=settings.local_slm_prompt_version,
        )
        layer = L3LocalSLMLayer(
            config=config,
            task_schema=_l3_benchmark_task_schema(data_dir),
        )
        metrics = benchmark_l3_layer(layer, DEFAULT_L3_BENCHMARK_UTTERANCES)
        payload = {
            **metrics,
            "config": config.model_dump(mode="json"),
            "corpus": "default_l3_smoke",
        }
    except (LocalSLMError, ValueError) as exc:
        payload = {
            "schema_version": "l3-benchmark-v1",
            "status": "error",
            "error": str(exc),
            "config": {
                "model_name": model_name or settings.local_slm_model,
                "mode": mode,
                "device_policy": device_policy or settings.local_slm_device_policy,
                "max_new_tokens": max_new_tokens or settings.local_slm_max_new_tokens,
                "confidence_threshold": settings.local_slm_confidence_threshold,
                "prompt_version": settings.local_slm_prompt_version,
            },
            "corpus": "default_l3_smoke",
        }

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    console.print_json(data=payload)
    if payload.get("status") == "error" and fail_on_error:
        raise typer.Exit(code=2)


@l3_app.command("replay-prompt")
def l3_replay_prompt(
    prompt: Annotated[Path, typer.Option(help="L3PromptArtifact JSON path.")],
    traces: Annotated[Path, typer.Option(help="Trace JSONL path with teacher_frame labels.")],
    out: Annotated[Path, typer.Option(help="Output l3-prompt-replay-v1 JSON path.")],
    data_dir: Annotated[
        Path,
        typer.Option(help="Optional processed data dir for task schema discovery."),
    ] = DEFAULT_PROCESSED_DATA_DIR,
    max_requests: Annotated[
        int | None,
        typer.Option(min=1, help="Optional maximum labeled traces to replay."),
    ] = None,
) -> None:
    """Replay an L3 prompt artifact explicitly with the local SLM."""

    settings = _load_cli_settings()
    prompt_artifact = L3PromptArtifact.model_validate_json(prompt.read_text(encoding="utf-8"))
    trace_records = read_traces(traces)
    config = LocalSLMConfig(
        model_name=settings.local_slm_model,
        mode="shadow",
        device_policy=settings.local_slm_device_policy,
        max_new_tokens=settings.local_slm_max_new_tokens,
        confidence_threshold=settings.local_slm_confidence_threshold,
        prompt_version=settings.local_slm_prompt_version,
    )
    payload = replay_l3_prompt_artifact(
        prompt_artifact=prompt_artifact,
        traces=trace_records,
        task_schema=_l3_benchmark_task_schema(data_dir),
        config=config,
        max_requests=max_requests,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    console.print_json(data=payload)


@l3_app.command("prompt-evolve")
def l3_prompt_evolve(
    traces: Annotated[Path, typer.Option(help="Trace JSONL path with teacher_frame labels.")],
    out_dir: Annotated[Path, typer.Option(help="Output directory for L3 prompt evolution.")],
    data_dir: Annotated[
        Path,
        typer.Option(help="Optional processed data dir for task schema discovery."),
    ] = DEFAULT_PROCESSED_DATA_DIR,
    max_traces: Annotated[
        int | None,
        typer.Option(min=1, help="Optional prefix of traces to use for the prompt split."),
    ] = None,
    max_agent_sessions: Annotated[
        int,
        typer.Option(min=0, help="Maximum live agent sessions; 0 prepares workspace only."),
    ] = 1,
    skip_replay: Annotated[
        bool,
        typer.Option(help="Skip local SLM replay; use only for wiring smoke/no-model runs."),
    ] = False,
    min_accepted_accuracy: Annotated[
        float,
        typer.Option(min=0.0, max=1.0, help="Minimum replay accepted accuracy gate."),
    ] = 0.90,
    max_wrong_accept_rate: Annotated[
        float,
        typer.Option(min=0.0, max=1.0, help="Maximum replay wrong accept rate gate."),
    ] = 0.05,
) -> None:
    """Run one L3 prompt-evolution agent session over an isolated workspace."""

    settings = _load_cli_settings()
    trace_records = read_traces(traces)
    if max_traces is not None:
        trace_records = trace_records[:max_traces]
    summary = run_l3_prompt_evolution(
        config=L3PromptEvolutionConfig(
            job_dir=out_dir,
            codex_command=settings.l3_agent_codex_command,
            codex_model=settings.l3_agent_model,
            timeout_s=settings.l3_agent_timeout_s,
            sandbox=settings.l3_agent_sandbox,
            approval_policy=settings.l3_agent_approval_policy,
            max_agent_sessions=max_agent_sessions,
            skip_replay=skip_replay,
            min_accepted_accuracy=min_accepted_accuracy,
            max_wrong_accept_rate=max_wrong_accept_rate,
            prompt_version=settings.local_slm_prompt_version,
        ),
        traces=trace_records,
        task_schema=_l3_benchmark_task_schema(data_dir),
        local_slm_config=LocalSLMConfig(
            model_name=settings.local_slm_model,
            mode="shadow",
            device_policy=settings.local_slm_device_policy,
            max_new_tokens=settings.local_slm_max_new_tokens,
            confidence_threshold=settings.local_slm_confidence_threshold,
            prompt_version=settings.local_slm_prompt_version,
        ),
    )
    console.print_json(data=summary)


@l3_app.command("promote-prompt")
def l3_promote_prompt(
    run_dir: Annotated[Path, typer.Option(help="Run directory whose artifact manifest to update.")],
    prompt: Annotated[Path, typer.Option(help="L3PromptArtifact JSON path to promote.")],
    replay: Annotated[Path, typer.Option(help="l3-prompt-replay-v1 JSON path.")],
    min_accepted_accuracy: Annotated[
        float,
        typer.Option(min=0.0, max=1.0, help="Minimum accepted accuracy required."),
    ] = 0.90,
    max_wrong_accept_rate: Annotated[
        float,
        typer.Option(min=0.0, max=1.0, help="Maximum wrong accept rate per labeled request."),
    ] = 0.05,
) -> None:
    """Promote an L3 prompt only after explicit regenerated replay passes gates."""

    try:
        manifest = _promote_l3_prompt_artifact(
            run_dir=run_dir,
            prompt_path=prompt,
            replay_path=replay,
            min_accepted_accuracy=min_accepted_accuracy,
            max_wrong_accept_rate=max_wrong_accept_rate,
        )
    except ValueError as exc:
        error_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc
    console.print_json(data=manifest.model_dump(mode="json"))


def _promote_l3_prompt_artifact(
    *,
    run_dir: Path,
    prompt_path: Path,
    replay_path: Path,
    min_accepted_accuracy: float,
    max_wrong_accept_rate: float,
) -> ArtifactManifest:
    prompt_artifact = L3PromptArtifact.model_validate_json(prompt_path.read_text(encoding="utf-8"))
    replay_payload = json.loads(replay_path.read_text(encoding="utf-8"))
    _validate_l3_prompt_replay_for_promotion(
        replay_payload,
        prompt_artifact=prompt_artifact,
        min_accepted_accuracy=min_accepted_accuracy,
        max_wrong_accept_rate=max_wrong_accept_rate,
    )

    store = ArtifactStore(run_dir / "artifacts")
    current_manifest = store.load_current_manifest()
    generation = 1 if current_manifest is None else current_manifest.generation + 1
    generation_dir = store.generation_dir(generation)
    l3_dir = generation_dir / "l3"
    l3_dir.mkdir(parents=True, exist_ok=True)
    promoted_prompt_path = l3_dir / "l3_prompt.json"
    promoted_replay_path = l3_dir / "l3_prompt.replay.json"
    promoted_prompt_path.write_text(
        prompt_artifact.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    promoted_replay_path.write_text(
        json.dumps(replay_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    artifact_paths = dict(current_manifest.artifact_paths) if current_manifest is not None else {}
    artifact_paths["l3_prompt"] = _artifact_path_relative_to_store(store, promoted_prompt_path)
    artifact_paths["l3_prompt_replay"] = _artifact_path_relative_to_store(
        store,
        promoted_replay_path,
    )
    candidate_metrics = (
        dict(current_manifest.candidate_metrics) if current_manifest is not None else {}
    )
    candidate_metrics.update(
        {
            "l3_prompt_runtime_promoted": True,
            "l3_prompt_replay_requests": replay_payload.get("requests"),
            "l3_prompt_replay_coverage": replay_payload.get("coverage"),
            "l3_prompt_replay_accepted_accuracy": replay_payload.get("accepted_accuracy"),
            "l3_prompt_replay_wrong_accept_rate": replay_payload.get("wrong_accept_rate"),
        }
    )
    promotion_record_path = store.write_generation_json(
        generation,
        "promotion.json",
        {
            "artifact_set_id": f"gen_{generation:03d}_l3_prompt",
            "generation": generation,
            "promoted": True,
            "promotion_reason": "explicit L3 prompt replay passed gates",
            "l3_prompt_replay": replay_payload,
        },
    )
    artifact_paths["promotion_record"] = _artifact_path_relative_to_store(
        store,
        promotion_record_path,
    )
    manifest = ArtifactManifest(
        artifact_set_id=f"gen_{generation:03d}_l3_prompt",
        generation=generation,
        parent_artifact_set_id=current_manifest.artifact_set_id if current_manifest else None,
        artifact_paths=artifact_paths,
        candidate_metrics=candidate_metrics,
        promotion_reason="explicit L3 prompt replay passed gates",
        l3_mode="guarded",
    )
    store.promote(manifest)
    return ArtifactStore(run_dir / "artifacts").load_current_manifest() or manifest


def _validate_l3_prompt_replay_for_promotion(
    payload: dict,
    *,
    prompt_artifact: L3PromptArtifact,
    min_accepted_accuracy: float,
    max_wrong_accept_rate: float,
) -> None:
    if payload.get("schema_version") != "l3-prompt-replay-v1":
        raise ValueError("L3 prompt replay artifact has an unsupported schema")
    if payload.get("status") != "success":
        raise ValueError("L3 prompt replay did not succeed")
    if payload.get("prompt_version") != prompt_artifact.prompt_version:
        raise ValueError("L3 prompt replay prompt version does not match the prompt artifact")
    if payload.get("prompt_sha256") != l3_prompt_artifact_hash(prompt_artifact):
        raise ValueError("L3 prompt replay hash does not match the prompt artifact")
    if int(payload.get("requests") or 0) <= 0:
        raise ValueError("L3 prompt replay has no labeled requests")
    if int(payload.get("would_accept_count") or 0) <= 0:
        raise ValueError("L3 prompt replay accepted no requests")
    accepted_accuracy = payload.get("accepted_accuracy")
    if not isinstance(accepted_accuracy, int | float) or accepted_accuracy < min_accepted_accuracy:
        raise ValueError("L3 prompt replay accepted accuracy is below the promotion gate")
    wrong_accept_rate = payload.get("wrong_accept_rate")
    if not isinstance(wrong_accept_rate, int | float) or wrong_accept_rate > max_wrong_accept_rate:
        raise ValueError("L3 prompt replay wrong accept rate exceeds the promotion gate")


def _artifact_path_relative_to_store(store: ArtifactStore, path: Path) -> str:
    return path.relative_to(store.root).as_posix()


def _l3_benchmark_task_schema(data_dir: Path) -> TaskSchema:
    if (data_dir / "train.jsonl").exists():
        return task_schema_from_records(load_processed_records(data_dir))
    return TaskSchema(
        intent_names=[
            "intent_alpha",
            "intent_beta",
            "intent_gamma",
        ],
        slot_names=[
            "slot_alpha",
            "slot_beta",
            "slot_gamma",
        ],
    )


@l2_app.command("train")
def l2_train(
    traces: Annotated[Path, typer.Option(help="Trace JSONL file with teacher_frame labels.")],
    out: Annotated[Path, typer.Option(help="Output joblib path for the L2 bundle.")],
    accept_threshold: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.93,
    min_examples: Annotated[int, typer.Option(min=2)] = 4,
) -> None:
    """Train a real sklearn L2 intent student from teacher traces."""

    trace_records = read_traces(traces)
    examples = training_examples_from_teacher_traces(traces_to_teacher_view(trace_records))
    bundle = train_l2_student(
        examples,
        L2StudentConfig(
            accept_threshold=accept_threshold,
            min_examples=min_examples,
        ),
    )
    bundle.save(out)
    console.print(f"trained L2 student with {len(examples)} teacher examples: {out}")


@l2_app.command("tune")
def l2_tune(
    traces: Annotated[Path, typer.Option(help="Trace JSONL file with teacher_frame labels.")],
    out: Annotated[Path, typer.Option(help="Output JSON path for the tuning report.")],
    n_trials: Annotated[int, typer.Option(min=1, help="Optuna trial budget.")] = 16,
    timeout_s: Annotated[
        float | None,
        typer.Option(min=0.1, help="Optional Optuna study timeout in seconds."),
    ] = None,
    validation_fraction: Annotated[
        float,
        typer.Option(min=0.05, max=0.75, help="Teacher-visible tuning holdout fraction."),
    ] = 0.25,
    split_policy: Annotated[
        str,
        typer.Option(help="Tuning split policy: chronological or stratified_random."),
    ] = "chronological",
    search_space: Annotated[
        str,
        typer.Option(help="Tuning search space: compact or wide."),
    ] = "compact",
    latency_weight: Annotated[
        float,
        typer.Option(min=0.0, help="Penalty per millisecond of validation p95 latency."),
    ] = 0.01,
) -> None:
    """Run Optuna tuning for L2 hyperparameters over teacher-visible traces."""

    if search_space not in {"compact", "wide"}:
        raise typer.BadParameter("search_space must be compact or wide")
    if split_policy not in {"chronological", "stratified_random"}:
        raise typer.BadParameter("split_policy must be chronological or stratified_random")
    settings = _load_cli_settings()
    base_config = l2_config_from_settings(settings)
    trace_records = read_traces(traces)
    result = tune_l2_student(
        traces_to_teacher_view(trace_records),
        base_config=base_config,
        spec=L2TuneSpec(
            n_trials=n_trials,
            timeout_s=timeout_s,
            validation_fraction=validation_fraction,
            split_policy=split_policy,
            random_state=base_config.random_state,
            search_space=search_space,
            max_wrong_accept_rate=settings.l2_max_wrong_accept_rate,
            min_accepted_accuracy=settings.l2_min_guarded_accuracy,
            latency_weight=latency_weight,
        ),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    console.print(
        f"tuned L2 with {result.n_trials_completed}/{result.n_trials_requested} "
        f"completed trials; best={result.best_value}; report={out}"
    )


@l2_app.command("target-evolve")
def l2_target_evolve(
    traces: Annotated[Path, typer.Option(help="Trace JSONL file with teacher_frame labels.")],
    out_dir: Annotated[Path, typer.Option(help="Output directory for target evolution artifacts.")],
    rounds: Annotated[
        int | None,
        typer.Option(min=1, help="Maximum inner target-evolution rounds."),
    ] = None,
    mode: Annotated[
        str,
        typer.Option(
            help="Evolution mode: agent-session, dry-run, local-search, or codex-cli.",
        ),
    ] = "dry-run",
    budget_profile: Annotated[
        str,
        typer.Option(
            help=(
                "Budget profile: standard, fixed-inner, or smoke. standard is "
                "cost-capped; fixed-inner is the formal fixed-snapshot research "
                "loop unless overridden by explicit budget flags."
            ),
        ),
    ] = "standard",
    dry_run_patch: Annotated[
        list[Path] | None,
        typer.Option(help="Patch to apply before a dry-run round; repeatable."),
    ] = None,
    max_traces: Annotated[
        int | None,
        typer.Option(min=1, help="Optional prefix of traces to use for the target split."),
    ] = None,
    split_policy: Annotated[
        str,
        typer.Option(help="Target split policy: chronological or intent-stratified."),
    ] = "chronological",
    target_scope: Annotated[
        str,
        typer.Option(
            help=(
                "Target trace scope: teacher_train uses all teacher-labeled traces; "
                "lower_miss keeps only traces where L0/L1 did not accept."
            ),
        ),
    ] = "teacher_train",
    visible_validation_folds: Annotated[
        int | None,
        typer.Option(
            min=1,
            help=(
                "Number of agent-visible validation folds. Values above 1 create "
                "inner_validation_shadow_* files and gate on their aggregate. "
                "Defaults are profile-specific."
            ),
        ),
    ] = None,
    visible_validation_ratio: Annotated[
        float | None,
        typer.Option(
            min=0.01,
            help=(
                "Optional agent-visible validation pool ratio for target splits. "
                "Defaults remain profile-specific; explicit values increase or "
                "decrease visible pressure without exposing private holdouts."
            ),
        ),
    ] = None,
    visible_cross_audit_folds: Annotated[
        int | None,
        typer.Option(
            min=0,
            help=(
                "Visible diagnostic-only cross-audit folds. 0 disables; values "
                "above 1 retrain on visible folds to expose selection-like safety "
                "pressure without reading private holdouts. Defaults are profile-specific."
            ),
        ),
    ] = None,
    inner_patience_rounds: Annotated[
        int | None,
        typer.Option(
            min=0,
            help="Stop after this many non-improving inner-validation rounds; 0 disables.",
        ),
    ] = None,
    stop_on_selection_gate: Annotated[
        bool,
        typer.Option(
            help=(
                "Opt in to stopping when the private selection holdout gate passes; "
                "default keeps spending the requested inner budget."
            ),
        ),
    ] = False,
    timeout_s: Annotated[
        float | None,
        typer.Option(min=0.1, help="Optional per-agent-round timeout override in seconds."),
    ] = None,
    local_search_trials: Annotated[
        int | None,
        typer.Option(min=1, help="Optuna trials per local-search target round."),
    ] = None,
    local_search_space: Annotated[
        str,
        typer.Option(help="Local search space: compact or wide."),
    ] = "compact",
    local_search_timeout_s: Annotated[
        float | None,
        typer.Option(min=0.1, help="Optional Optuna timeout per local-search target round."),
    ] = None,
    local_search_cross_audit_top_k: Annotated[
        int | None,
        typer.Option(
            min=0,
            help=(
                "Re-rank this many top local-search trials with visible cross-audit. "
                "Defaults are profile-specific; 0 disables."
            ),
        ),
    ] = None,
    max_agent_rounds: Annotated[
        int | None,
        typer.Option(
            min=0,
            help=(
                "Maximum live agent launches. agent-session defaults to 1; "
                "codex-cli defaults are profile-specific; 0 prepares/evaluates "
                "the workspace without launching Codex."
            ),
        ),
    ] = None,
) -> None:
    """Run an inner target-dependent L2 evolution loop over fixed trace splits."""

    if mode not in {"dry-run", "local-search", "codex-cli", "agent-session"}:
        raise typer.BadParameter(
            "mode must be dry-run, local-search, codex-cli, or agent-session",
        )
    if budget_profile not in {"standard", "fixed-inner", "smoke"}:
        raise typer.BadParameter("budget_profile must be standard, fixed-inner, or smoke")
    if split_policy not in {"chronological", "intent-stratified"}:
        raise typer.BadParameter("split_policy must be chronological or intent-stratified")
    if target_scope not in {"teacher_train", "lower_miss"}:
        raise typer.BadParameter("target_scope must be teacher_train or lower_miss")
    if local_search_space not in {"compact", "wide"}:
        raise typer.BadParameter("local_search_space must be compact or wide")
    if visible_validation_ratio is not None and not (0.0 < visible_validation_ratio < 0.80):
        raise typer.BadParameter(
            "visible_validation_ratio must be greater than 0 and less than 0.80",
        )
    evolution_mode: L2TargetEvolutionMode = mode  # type: ignore[assignment]
    resolved_budget_profile: L2TargetBudgetProfile = budget_profile  # type: ignore[assignment]
    resolved_split_policy: L2TargetSplitPolicy = split_policy  # type: ignore[assignment]
    resolved_target_scope: L2TargetScope = target_scope  # type: ignore[assignment]
    resolved_rounds, resolved_inner_patience, resolved_local_search_trials = (
        _resolve_l2_target_budget(
            budget_profile=resolved_budget_profile,
            rounds=rounds,
            inner_patience_rounds=inner_patience_rounds,
            local_search_trials=local_search_trials,
        )
    )
    resolved_max_agent_rounds = _resolve_l2_target_agent_rounds(
        mode=evolution_mode,
        budget_profile=resolved_budget_profile,
        max_agent_rounds=max_agent_rounds,
    )
    resolved_visible_validation_folds = _resolve_l2_target_visible_validation_folds(
        budget_profile=resolved_budget_profile,
        visible_validation_folds=visible_validation_folds,
    )
    resolved_visible_cross_audit_folds = _resolve_l2_target_visible_cross_audit_folds(
        budget_profile=resolved_budget_profile,
        visible_cross_audit_folds=visible_cross_audit_folds,
    )
    resolved_local_search_cross_audit_top_k = (
        _resolve_l2_target_local_search_cross_audit_top_k(
            budget_profile=resolved_budget_profile,
            visible_cross_audit_folds=resolved_visible_cross_audit_folds,
            local_search_cross_audit_top_k=local_search_cross_audit_top_k,
        )
    )
    settings = _load_cli_settings()
    trace_records = read_traces(traces)
    if max_traces is not None:
        trace_records = trace_records[:max_traces]
    summary = run_l2_target_evolution(
        config=L2TargetEvolutionConfig(
            source_repo_dir=Path.cwd(),
            job_dir=out_dir,
            rounds=resolved_rounds,
            mode=evolution_mode,
            dry_run_patches=tuple(dry_run_patch or ()),
            codex_command=settings.l2_target_agent_codex_command,
            codex_model=settings.l2_target_agent_model,
            timeout_s=timeout_s if timeout_s is not None else settings.l2_target_agent_timeout_s,
            local_search_trials=resolved_local_search_trials,
            local_search_space=local_search_space,  # type: ignore[arg-type]
            local_search_timeout_s=local_search_timeout_s,
            local_search_cross_audit_top_k=resolved_local_search_cross_audit_top_k,
            budget_profile=resolved_budget_profile,
            split_policy=resolved_split_policy,
            target_scope=resolved_target_scope,
            visible_validation_folds=resolved_visible_validation_folds,
            visible_validation_ratio=visible_validation_ratio,
            visible_cross_audit_folds=resolved_visible_cross_audit_folds,
            max_agent_rounds=resolved_max_agent_rounds,
            sandbox=settings.l2_target_agent_sandbox,
            approval_policy=settings.l2_target_agent_approval_policy,
            ignore_user_config=settings.l2_target_agent_ignore_user_config,
            ignore_rules=settings.l2_target_agent_ignore_rules,
            ephemeral=settings.l2_target_agent_ephemeral,
            min_accepted_accuracy=settings.l2_min_guarded_accuracy,
            max_wrong_accept_rate=settings.l2_max_wrong_accept_rate,
            inner_patience_rounds=resolved_inner_patience,
            stop_on_selection_gate=stop_on_selection_gate,
        ),
        traces=traces_to_teacher_view(trace_records),
    )
    console.print_json(data=summary)


def _resolve_l2_target_budget(
    *,
    budget_profile: L2TargetBudgetProfile,
    rounds: int | None,
    inner_patience_rounds: int | None,
    local_search_trials: int | None,
) -> tuple[int, int, int]:
    if budget_profile == "standard":
        default_rounds = DEFAULT_TARGET_EVOLVE_ROUNDS
        default_inner_patience = DEFAULT_TARGET_INNER_PATIENCE_ROUNDS
        default_local_search_trials = DEFAULT_TARGET_LOCAL_SEARCH_TRIALS
    elif budget_profile == "fixed-inner":
        default_rounds = 48
        default_inner_patience = 0
        default_local_search_trials = 32
    else:
        default_rounds = 1
        default_inner_patience = 0
        default_local_search_trials = 2
    return (
        rounds if rounds is not None else default_rounds,
        (
            inner_patience_rounds
            if inner_patience_rounds is not None
            else default_inner_patience
        ),
        (
            local_search_trials
            if local_search_trials is not None
            else default_local_search_trials
        ),
    )


def _resolve_l2_target_agent_rounds(
    *,
    mode: L2TargetEvolutionMode,
    budget_profile: L2TargetBudgetProfile,
    max_agent_rounds: int | None,
) -> int | None:
    if mode not in {"codex-cli", "agent-session"}:
        return max_agent_rounds
    if max_agent_rounds is not None:
        return max_agent_rounds
    if mode == "agent-session":
        return 1
    if budget_profile == "standard":
        return 3
    if budget_profile == "fixed-inner":
        return 16
    return 1


def _resolve_l2_target_visible_validation_folds(
    *,
    budget_profile: L2TargetBudgetProfile,
    visible_validation_folds: int | None,
) -> int:
    if visible_validation_folds is not None:
        return visible_validation_folds
    if budget_profile == "fixed-inner":
        return 5
    return 1


def _resolve_l2_target_visible_cross_audit_folds(
    *,
    budget_profile: L2TargetBudgetProfile,
    visible_cross_audit_folds: int | None,
) -> int:
    if visible_cross_audit_folds is not None:
        return visible_cross_audit_folds
    if budget_profile == "fixed-inner":
        return DEFAULT_TARGET_VISIBLE_CROSS_AUDIT_FOLDS
    return 0


def _resolve_l2_target_local_search_cross_audit_top_k(
    *,
    budget_profile: L2TargetBudgetProfile,
    visible_cross_audit_folds: int,
    local_search_cross_audit_top_k: int | None,
) -> int:
    if local_search_cross_audit_top_k is not None:
        return local_search_cross_audit_top_k
    if budget_profile == "fixed-inner" and visible_cross_audit_folds >= 2:
        return DEFAULT_TARGET_LOCAL_SEARCH_CROSS_AUDIT_TOP_K
    return 0


@l2_app.command("promote-target")
def l2_promote_target(
    target_run: Annotated[
        Path,
        typer.Option(help="Output directory from `edge-mvp l2 target-evolve`."),
    ],
    run_dir: Annotated[
        Path,
        typer.Option(help="Replay run directory whose artifact manifest should receive L2 target."),
    ],
    allow_non_adopted: Annotated[
        bool,
        typer.Option(
            help=(
                "Stage the best non-adopted target candidate for outer replay. "
                "Default only promotes target runs that passed inner adoption gates."
            ),
        ),
    ] = False,
) -> None:
    """Promote an adopted L2 target-evolve result into a replay artifact manifest."""

    try:
        manifest = _promote_l2_target_run(
            target_run=target_run,
            run_dir=run_dir,
            allow_non_adopted=allow_non_adopted,
        )
    except (FileNotFoundError, TypeError, ValueError) as exc:
        error_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc
    console.print_json(data=manifest.model_dump(mode="json"))


def _promote_l2_target_run(
    *,
    target_run: Path,
    run_dir: Path,
    allow_non_adopted: bool = False,
) -> ArtifactManifest:
    summary_path = target_run / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"L2 target summary is missing: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    adoption_decision = summary.get("adoption_decision") or {}
    inner_adopted = bool(adoption_decision.get("adopted"))
    selected_round_payload: dict[str, Any] | None = None
    if inner_adopted:
        selected_round = adoption_decision.get("round")
        if not isinstance(selected_round, int):
            raise ValueError("L2 target adopted round is missing")
        promotion_reason = "explicit L2 target adoption passed gates"
    elif allow_non_adopted:
        best_round = summary.get("best_round")
        if not isinstance(best_round, dict):
            raise ValueError("L2 target run has no best round to stage")
        selected_round = best_round.get("round")
        if not isinstance(selected_round, int):
            raise ValueError("L2 target best round is missing")
        selected_round_payload = best_round
        promotion_reason = "explicit L2 target candidate staged for outer replay"
    else:
        raise ValueError("L2 target run is not adopted; refusing to promote")

    if selected_round_payload is None:
        selected_round_payload = next(
            (
                round_payload
                for round_payload in summary.get("rounds", [])
                if round_payload.get("round") == selected_round
            ),
            None,
        )
    workspace_root = target_run / "workspace" / "l2_target"
    if not workspace_root.exists() and summary.get("workspace"):
        workspace_root = Path(str(summary["workspace"]))
    target_dir = _l2_target_dir_for_selected_round(
        target_run=target_run,
        workspace_root=workspace_root,
        selected_round_payload=selected_round_payload,
    )
    target_module_path = target_dir / "target_l2.py"
    train_path = workspace_root / "data" / "train.jsonl"
    if not target_module_path.exists():
        raise FileNotFoundError(f"L2 target module is missing: {target_module_path}")
    if not train_path.exists():
        raise FileNotFoundError(f"L2 target train split is missing: {train_path}")

    train_traces = [
        TeacherTrace.model_validate_json(line)
        for line in train_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    target_module = load_target_module(target_module_path)
    l2_config = L2StudentConfig(**target_config_overrides(target_module))
    l2_bundle = train_l2_student(
        training_examples_from_teacher_traces(train_traces),
        l2_config,
    )

    store = ArtifactStore(run_dir / "artifacts")
    current_manifest = store.load_current_manifest()
    generation = 1 if current_manifest is None else current_manifest.generation + 1
    generation_dir = store.generation_dir(generation)
    l2_dir = generation_dir / "l2"
    l2_dir.mkdir(parents=True, exist_ok=True)
    promoted_bundle_path = l2_dir / "l2_student.joblib"
    promoted_target_dir = l2_dir / "target"
    if promoted_target_dir.exists():
        shutil.rmtree(promoted_target_dir)
    shutil.copytree(target_dir, promoted_target_dir)
    l2_bundle.save(promoted_bundle_path)

    artifact_paths = dict(current_manifest.artifact_paths) if current_manifest else {}
    artifact_paths["l2_student"] = _artifact_path_relative_to_store(
        store,
        promoted_bundle_path,
    )
    artifact_paths["l2_target"] = _artifact_path_relative_to_store(
        store,
        promoted_target_dir / "target_l2.py",
    )

    candidate_metrics = (
        dict(current_manifest.candidate_metrics) if current_manifest is not None else {}
    )
    candidate_metrics.update(
        {
            "l2_target_runtime_promoted": True,
            "l2_target_inner_adopted": inner_adopted,
            "l2_target_staged_for_outer_replay": not inner_adopted,
            "l2_target_requires_outer_replay": True,
            "l2_target_source_run": str(target_run),
            "l2_target_adopted_round": selected_round if inner_adopted else None,
            "l2_target_staged_round": selected_round,
            "l2_target_mode": summary.get("mode"),
            "l2_target_data_split": summary.get("data_split"),
            "l2_target_data_split_policy": summary.get("data_split_policy"),
            "l2_target_loop_cadence": summary.get("loop_cadence"),
            "l2_target_agent_budget": summary.get("agent_budget"),
            "l2_target_code_policy": summary.get("target_code_policy"),
            "l2_target_workspace_scope_policy": summary.get("workspace_scope_policy"),
            "l2_target_private_holdout_evidence": summary.get(
                "private_holdout_evidence"
            ),
            "l2_target_selection_decision": summary.get("selection_decision"),
            "l2_target_adoption_decision": adoption_decision,
            "l2_target_training_traces": len(train_traces),
            "l2_target_training_source": str(train_path),
            "l2_target_inner_validation": (
                selected_round_payload.get("inner_validation")
                if isinstance(selected_round_payload, dict)
                else None
            ),
            "l2_target_selection_holdout": (
                selected_round_payload.get("selection_holdout")
                if isinstance(selected_round_payload, dict)
                else None
            ),
            "l2_target_promotion_holdout": (
                selected_round_payload.get("promotion_holdout")
                if isinstance(selected_round_payload, dict)
                else None
            ),
            "l2_config": l2_config.model_dump(mode="json"),
            "l2_examples": len(train_traces),
            "l2_trained": True,
            "l2_training_scope": "l2_target_workspace_train",
            "l2_training_traces": len(train_traces),
        }
    )
    promotion_record_path = store.write_generation_json(
        generation,
        "promotion.json",
        {
            "artifact_set_id": f"gen_{generation:03d}_l2_target",
            "generation": generation,
            "promoted": True,
            "promotion_reason": promotion_reason,
            "l2_target_summary": {
                "source_run": str(target_run),
                "inner_adopted": inner_adopted,
                "adoption_decision": adoption_decision,
                "selection_decision": summary.get("selection_decision"),
                "selected_round": selected_round,
            },
        },
    )
    artifact_paths["promotion_record"] = _artifact_path_relative_to_store(
        store,
        promotion_record_path,
    )
    manifest = ArtifactManifest(
        artifact_set_id=f"gen_{generation:03d}_l2_target",
        generation=generation,
        parent_artifact_set_id=current_manifest.artifact_set_id if current_manifest else None,
        schema_versions={
            "artifact_manifest": "artifact-manifest-v1",
            "l2_target": "l2-target-runtime-v1",
        },
        artifact_paths=artifact_paths,
        candidate_metrics=candidate_metrics,
        promotion_reason=promotion_reason,
        l3_mode=current_manifest.l3_mode if current_manifest else "disabled",
    )
    store.promote(manifest)
    return ArtifactStore(run_dir / "artifacts").load_current_manifest() or manifest


def _l2_target_dir_for_selected_round(
    *,
    target_run: Path,
    workspace_root: Path,
    selected_round_payload: dict[str, Any] | None,
) -> Path:
    snapshot = (
        selected_round_payload.get("target_snapshot")
        if isinstance(selected_round_payload, dict)
        else None
    )
    if isinstance(snapshot, str) and snapshot:
        snapshot_path = Path(snapshot)
        if not snapshot_path.is_absolute():
            snapshot_path = target_run / snapshot_path
        if not snapshot_path.exists():
            raise FileNotFoundError(f"L2 target round snapshot is missing: {snapshot_path}")
        return snapshot_path
    return workspace_root / "target"


@l2_app.command("replay-target")
def l2_replay_target(
    run_dir: Annotated[
        Path,
        typer.Option(help="Replay run directory containing a staged L2 target manifest."),
    ],
    traces: Annotated[
        Path,
        typer.Option(help="Trace JSONL with teacher_frame labels for outer replay."),
    ],
    out: Annotated[
        Path,
        typer.Option(help="Output JSON report path."),
    ],
    candidate_generation: Annotated[
        int | None,
        typer.Option(min=1, help="Candidate manifest generation; defaults to current."),
    ] = None,
    baseline_generation: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Baseline manifest generation; defaults to candidate parent.",
        ),
    ] = None,
    accuracy_epsilon: Annotated[
        float,
        typer.Option(min=0.0, help="Allowed frame exactness regression."),
    ] = 0.0,
    max_wrong_accept_rate: Annotated[
        float | None,
        typer.Option(min=0.0, help="Override max target replay wrong-accept rate."),
    ] = None,
    include_default_l1: Annotated[
        bool,
        typer.Option(
            "--include-default-l1/--no-include-default-l1",
            help="Include settings L1 Rust worker when the manifest has no L1 artifact.",
        ),
    ] = True,
) -> None:
    """Compare a staged L2 target artifact against its parent manifest."""

    try:
        payload = _replay_l2_target_manifest(
            run_dir=run_dir,
            traces_path=traces,
            candidate_generation=candidate_generation,
            baseline_generation=baseline_generation,
            accuracy_epsilon=accuracy_epsilon,
            max_wrong_accept_rate=max_wrong_accept_rate,
            include_default_l1=include_default_l1,
        )
    except (FileNotFoundError, TypeError, ValueError) as exc:
        error_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    console.print_json(data=payload)


def _replay_l2_target_manifest(
    *,
    run_dir: Path,
    traces_path: Path,
    candidate_generation: int | None,
    baseline_generation: int | None,
    accuracy_epsilon: float,
    max_wrong_accept_rate: float | None,
    include_default_l1: bool,
) -> dict[str, Any]:
    settings = _load_cli_settings()
    store = ArtifactStore(run_dir / "artifacts")
    candidate_manifest = _load_manifest_for_generation(store, candidate_generation)
    if "l2_target" not in candidate_manifest.artifact_paths:
        raise ValueError("candidate manifest does not contain an l2_target artifact")
    if baseline_generation is None:
        baseline_generation = _parent_generation(candidate_manifest)
    baseline_manifest = _load_manifest_for_generation(store, baseline_generation)

    trace_records = read_traces(traces_path)
    teacher_traces = traces_to_teacher_view(trace_records)
    if not any(trace.teacher_frame is not None for trace in teacher_traces):
        raise ValueError("L2 target replay requires at least one teacher-labeled trace")

    cost_model = replay_cost_model_from_settings(settings)
    default_l1_crate_dir = settings.l1_rust_crate_dir if include_default_l1 else None
    baseline_artifacts = load_offline_artifact_set(
        store.root,
        baseline_manifest,
        default_l1_crate_dir=default_l1_crate_dir,
        l1_worker_timeout_s=settings.l1_worker_timeout_s,
    )
    candidate_artifacts = load_offline_artifact_set(
        store.root,
        candidate_manifest,
        default_l1_crate_dir=default_l1_crate_dir,
        l1_worker_timeout_s=settings.l1_worker_timeout_s,
    )
    baseline_result = evaluate_offline_artifact_set(
        teacher_traces,
        baseline_artifacts,
        cost_model=cost_model,
    )
    candidate_result = evaluate_offline_artifact_set(
        teacher_traces,
        candidate_artifacts,
        cost_model=cost_model,
    )
    deltas = layer_deltas(baseline_result, candidate_result)
    decision = decide_artifact_set_promotion(
        baseline_result.objective,
        candidate_result.objective,
        per_layer_deltas=deltas,
        accuracy_epsilon=accuracy_epsilon,
        max_wrong_accept_rate=(
            max_wrong_accept_rate
            if max_wrong_accept_rate is not None
            else settings.l2_max_wrong_accept_rate
        ),
        block_layer_regressions=settings.promotion_block_layer_regressions,
    )
    return {
        "schema_version": "l2-target-outer-replay-v1",
        "status": "success",
        "run_dir": str(run_dir),
        "traces": str(traces_path),
        "baseline_generation": baseline_manifest.generation,
        "baseline_artifact_set_id": baseline_manifest.artifact_set_id,
        "candidate_generation": candidate_manifest.generation,
        "candidate_artifact_set_id": candidate_manifest.artifact_set_id,
        "candidate_inner_adopted": _target_inner_adopted(candidate_manifest),
        "candidate_staged_for_outer_replay": _target_staged_for_outer_replay(
            candidate_manifest
        ),
        "gates": {
            "accuracy_epsilon": accuracy_epsilon,
            "max_wrong_accept_rate": (
                max_wrong_accept_rate
                if max_wrong_accept_rate is not None
                else settings.l2_max_wrong_accept_rate
            ),
            "include_default_l1": include_default_l1,
            "block_layer_regressions": settings.promotion_block_layer_regressions,
        },
        "baseline": _offline_replay_payload(baseline_result),
        "candidate": _offline_replay_payload(candidate_result),
        "per_layer_deltas": {
            layer: delta.model_dump(mode="json") for layer, delta in deltas.items()
        },
        "decision": {
            "promoted": decision.promoted,
            "reason": decision.reason,
            "current_score": decision.current_score,
            "candidate_score": decision.candidate_score,
            "promoted_with_layer_regression": decision.promoted_with_layer_regression,
            "regressed_layers": decision.regressed_layers or [],
        },
    }


def _load_manifest_for_generation(
    store: ArtifactStore,
    generation: int | None,
) -> ArtifactManifest:
    if generation is None:
        manifest = store.load_current_manifest()
        if manifest is None:
            raise FileNotFoundError(f"current artifact manifest is missing: {store.root}")
        return manifest
    path = store.generation_dir(generation) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"artifact manifest is missing: {path}")
    return ArtifactManifest.model_validate_json(path.read_text(encoding="utf-8"))


def _parent_generation(manifest: ArtifactManifest) -> int:
    parent = manifest.parent_artifact_set_id
    if parent:
        parts = parent.split("_", 2)
        if len(parts) >= 2 and parts[0] == "gen" and parts[1].isdigit():
            return int(parts[1])
    if manifest.generation > 1:
        return manifest.generation - 1
    raise ValueError("baseline generation is required when candidate has no parent")


def _offline_replay_payload(result: OfflineReplayResult) -> dict[str, Any]:
    return {
        "requests": result.requests,
        "objective": asdict(result.objective),
        "layer_counts": result.layer_counts,
        "layer_metrics": result.layer_metrics,
    }


def _target_inner_adopted(manifest: ArtifactManifest) -> bool | None:
    metrics = manifest.candidate_metrics
    value = metrics.get("l2_target_inner_adopted")
    if isinstance(value, bool):
        return value
    adoption_decision = metrics.get("l2_target_adoption_decision")
    if isinstance(adoption_decision, dict):
        adopted = adoption_decision.get("adopted")
        if isinstance(adopted, bool):
            return adopted
    return None


def _target_staged_for_outer_replay(manifest: ArtifactManifest) -> bool | None:
    metrics = manifest.candidate_metrics
    value = metrics.get("l2_target_staged_for_outer_replay")
    if isinstance(value, bool):
        return value
    inner_adopted = _target_inner_adopted(manifest)
    if inner_adopted is not None:
        return not inner_adopted
    return None


@experiments_app.command("main-evolution")
def experiment_main_evolution(
    run_dir: Annotated[Path, typer.Option(help="Experiment run directory.")] = Path("runs/main"),
    stream: Annotated[str | None, typer.Option(help="Override experiment stream.")] = None,
    max_requests: Annotated[int | None, typer.Option(min=1)] = None,
    compile_every: Annotated[int | None, typer.Option(min=1)] = None,
    teacher: Annotated[
        str,
        typer.Option(help="Teacher mode: live, cache, or live-or-cache."),
    ] = "live-or-cache",
    data_dir: Annotated[
        Path,
        typer.Option(help="Processed data directory produced by prepare."),
    ] = DEFAULT_PROCESSED_DATA_DIR,
) -> None:
    _run_single_experiment(
        "main-evolution",
        run_dir=run_dir,
        stream=stream,
        max_requests=max_requests,
        compile_every=compile_every,
        teacher=teacher,
        data_dir=data_dir,
    )


@experiments_app.command("direct-l4-optimization")
def experiment_direct_l4_optimization(
    run_dir: Annotated[
        Path,
        typer.Option(help="Experiment run directory."),
    ] = Path("runs/direct-l4-optimization"),
    stream: Annotated[str | None, typer.Option(help="Override experiment stream.")] = None,
    max_requests: Annotated[int | None, typer.Option(min=1)] = None,
    compile_every: Annotated[int | None, typer.Option(min=1)] = None,
    teacher: Annotated[
        str,
        typer.Option(help="Teacher mode: live, cache, or live-or-cache."),
    ] = "live-or-cache",
    data_dir: Annotated[
        Path,
        typer.Option(help="Processed data directory produced by prepare."),
    ] = DEFAULT_PROCESSED_DATA_DIR,
) -> None:
    _run_single_experiment(
        "direct-l4-optimization",
        run_dir=run_dir,
        stream=stream,
        max_requests=max_requests,
        compile_every=compile_every,
        teacher=teacher,
        data_dir=data_dir,
    )


@experiments_app.command("l2-family")
def experiment_l2_family(
    run_dir: Annotated[
        Path,
        typer.Option(help="Experiment run directory."),
    ] = Path("runs/l2-family"),
    stream: Annotated[str | None, typer.Option(help="Override experiment stream.")] = None,
    max_requests: Annotated[int | None, typer.Option(min=1)] = None,
    compile_every: Annotated[int | None, typer.Option(min=1)] = None,
    teacher: Annotated[
        str,
        typer.Option(help="Teacher mode: live, cache, or live-or-cache."),
    ] = "live-or-cache",
    data_dir: Annotated[
        Path,
        typer.Option(help="Processed data directory produced by prepare."),
    ] = DEFAULT_PROCESSED_DATA_DIR,
) -> None:
    _run_single_experiment(
        "l2-family",
        run_dir=run_dir,
        stream=stream,
        max_requests=max_requests,
        compile_every=compile_every,
        teacher=teacher,
        data_dir=data_dir,
    )


@experiments_app.command("l2-mlp")
def experiment_l2_mlp(
    run_dir: Annotated[
        Path,
        typer.Option(help="Experiment run directory."),
    ] = Path("runs/l2-mlp"),
    stream: Annotated[str | None, typer.Option(help="Override experiment stream.")] = None,
    max_requests: Annotated[int | None, typer.Option(min=1)] = None,
    compile_every: Annotated[int | None, typer.Option(min=1)] = None,
    teacher: Annotated[
        str,
        typer.Option(help="Teacher mode: live, cache, or live-or-cache."),
    ] = "live-or-cache",
    data_dir: Annotated[
        Path,
        typer.Option(help="Processed data directory produced by prepare."),
    ] = DEFAULT_PROCESSED_DATA_DIR,
) -> None:
    _run_single_experiment(
        "l2-mlp",
        run_dir=run_dir,
        stream=stream,
        max_requests=max_requests,
        compile_every=compile_every,
        teacher=teacher,
        data_dir=data_dir,
    )


@experiments_app.command("l2-tuned")
def experiment_l2_tuned(
    run_dir: Annotated[
        Path,
        typer.Option(help="Experiment run directory."),
    ] = Path("runs/l2-tuned"),
    stream: Annotated[str | None, typer.Option(help="Override experiment stream.")] = None,
    max_requests: Annotated[int | None, typer.Option(min=1)] = None,
    compile_every: Annotated[int | None, typer.Option(min=1)] = None,
    teacher: Annotated[
        str,
        typer.Option(help="Teacher mode: live, cache, or live-or-cache."),
    ] = "live-or-cache",
    data_dir: Annotated[
        Path,
        typer.Option(help="Processed data directory produced by prepare."),
    ] = DEFAULT_PROCESSED_DATA_DIR,
) -> None:
    _run_single_experiment(
        "l2-tuned",
        run_dir=run_dir,
        stream=stream,
        max_requests=max_requests,
        compile_every=compile_every,
        teacher=teacher,
        data_dir=data_dir,
    )


@experiments_app.command("l2-tuned-lower-miss")
def experiment_l2_tuned_lower_miss(
    run_dir: Annotated[
        Path,
        typer.Option(help="Experiment run directory."),
    ] = Path("runs/l2-tuned-lower-miss"),
    stream: Annotated[str | None, typer.Option(help="Override experiment stream.")] = None,
    max_requests: Annotated[int | None, typer.Option(min=1)] = None,
    compile_every: Annotated[int | None, typer.Option(min=1)] = None,
    teacher: Annotated[
        str,
        typer.Option(help="Teacher mode: live, cache, or live-or-cache."),
    ] = "live-or-cache",
    data_dir: Annotated[
        Path,
        typer.Option(help="Processed data directory produced by prepare."),
    ] = DEFAULT_PROCESSED_DATA_DIR,
) -> None:
    _run_single_experiment(
        "l2-tuned-lower-miss",
        run_dir=run_dir,
        stream=stream,
        max_requests=max_requests,
        compile_every=compile_every,
        teacher=teacher,
        data_dir=data_dir,
    )


@experiments_app.command("no-guard")
def experiment_no_guard(
    run_dir: Annotated[
        Path,
        typer.Option(help="Experiment run directory."),
    ] = Path("runs/no-guard"),
    stream: Annotated[str | None, typer.Option(help="Override experiment stream.")] = None,
    max_requests: Annotated[int | None, typer.Option(min=1)] = None,
    compile_every: Annotated[int | None, typer.Option(min=1)] = None,
    teacher: Annotated[
        str,
        typer.Option(help="Teacher mode: live, cache, or live-or-cache."),
    ] = "live-or-cache",
    data_dir: Annotated[
        Path,
        typer.Option(help="Processed data directory produced by prepare."),
    ] = DEFAULT_PROCESSED_DATA_DIR,
) -> None:
    _run_single_experiment(
        "no-guard",
        run_dir=run_dir,
        stream=stream,
        max_requests=max_requests,
        compile_every=compile_every,
        teacher=teacher,
        data_dir=data_dir,
    )


@experiments_app.command("no-l2")
def experiment_no_l2(
    run_dir: Annotated[
        Path,
        typer.Option(help="Experiment run directory."),
    ] = Path("runs/no-l2"),
    stream: Annotated[str | None, typer.Option(help="Override experiment stream.")] = None,
    max_requests: Annotated[int | None, typer.Option(min=1)] = None,
    compile_every: Annotated[int | None, typer.Option(min=1)] = None,
    teacher: Annotated[
        str,
        typer.Option(help="Teacher mode: live, cache, or live-or-cache."),
    ] = "live-or-cache",
    data_dir: Annotated[
        Path,
        typer.Option(help="Processed data directory produced by prepare."),
    ] = DEFAULT_PROCESSED_DATA_DIR,
) -> None:
    _run_single_experiment(
        "no-l2",
        run_dir=run_dir,
        stream=stream,
        max_requests=max_requests,
        compile_every=compile_every,
        teacher=teacher,
        data_dir=data_dir,
    )


@experiments_app.command("hard-buffer")
def experiment_hard_buffer(
    run_dir: Annotated[
        Path,
        typer.Option(help="Experiment run directory."),
    ] = Path("runs/hard-buffer"),
    stream: Annotated[str | None, typer.Option(help="Override experiment stream.")] = None,
    max_requests: Annotated[int | None, typer.Option(min=1)] = None,
    compile_every: Annotated[int | None, typer.Option(min=1)] = None,
    teacher: Annotated[
        str,
        typer.Option(help="Teacher mode: live, cache, or live-or-cache."),
    ] = "live-or-cache",
    data_dir: Annotated[
        Path,
        typer.Option(help="Processed data directory produced by prepare."),
    ] = DEFAULT_PROCESSED_DATA_DIR,
) -> None:
    _run_single_experiment(
        "hard-buffer",
        run_dir=run_dir,
        stream=stream,
        max_requests=max_requests,
        compile_every=compile_every,
        teacher=teacher,
        data_dir=data_dir,
    )


@experiments_app.command("workload-locality")
def experiment_workload_locality(
    run_dir: Annotated[
        Path,
        typer.Option(help="Experiment run directory."),
    ] = Path("runs/workload-locality"),
    max_requests: Annotated[int | None, typer.Option(min=1)] = None,
    compile_every: Annotated[int | None, typer.Option(min=1)] = None,
    teacher: Annotated[
        str,
        typer.Option(help="Teacher mode: live, cache, or live-or-cache."),
    ] = "live-or-cache",
    data_dir: Annotated[
        Path,
        typer.Option(help="Processed data directory produced by prepare."),
    ] = DEFAULT_PROCESSED_DATA_DIR,
) -> None:
    _run_workload_locality_experiment(
        run_dir=run_dir,
        max_requests=max_requests,
        compile_every=compile_every,
        teacher=teacher,
        data_dir=data_dir,
    )


@experiments_app.command("compare")
def experiment_compare(
    run_dirs: Annotated[
        list[Path] | None,
        typer.Option("--run", help="Run directory to include. Repeat for multiple runs."),
    ] = None,
    root: Annotated[
        Path | None,
        typer.Option(help="Discover run directories recursively under this root."),
    ] = None,
    out_dir: Annotated[
        Path,
        typer.Option(help="Output directory for comparison.csv and comparison.html."),
    ] = Path("runs/experiment-comparison"),
) -> None:
    """Generate a cross-experiment comparison report from existing run dirs."""

    selected_run_dirs = list(run_dirs or [])
    if root is not None:
        selected_run_dirs.extend(_discover_run_dirs(root))
    selected_run_dirs = _dedupe_paths(selected_run_dirs)
    if not selected_run_dirs:
        error_console.print("[red]no run directories supplied or discovered[/red]")
        raise typer.Exit(code=2)
    result = generate_experiment_comparison_report(selected_run_dirs, out_dir)
    console.print(
        f"compared {len(selected_run_dirs)} run directories; "
        f"csv={result.comparison_csv_path}; html={result.comparison_html_path}"
    )


@experiments_app.command("suite")
def experiment_suite(
    run_root: Annotated[
        Path,
        typer.Option(help="Root directory for the experiment suite."),
    ] = Path("runs/suite"),
    experiment: Annotated[
        list[str] | None,
        typer.Option("--experiment", help="Experiment name to include. Repeatable."),
    ] = None,
    max_requests: Annotated[int, typer.Option(min=1)] = 300,
    compile_every: Annotated[int, typer.Option(min=1)] = 100,
    teacher: Annotated[
        str,
        typer.Option(help="Teacher mode: live, cache, or live-or-cache."),
    ] = "live-or-cache",
    data_dir: Annotated[
        Path,
        typer.Option(help="Processed data directory produced by prepare."),
    ] = DEFAULT_PROCESSED_DATA_DIR,
    parallel: Annotated[int, typer.Option(min=1, help="Maximum concurrent experiments.")] = 2,
    compare: Annotated[bool, typer.Option(help="Generate comparison report after success.")] = True,
) -> None:
    """Run an experiment suite with subprocess-level parallelism."""

    selected = list(experiment or DEFAULT_EXPERIMENT_SUITE)
    run_root.mkdir(parents=True, exist_ok=True)
    suite_payload = {
        "schema_version": "experiment-suite-v1",
        "experiments": selected,
        "max_requests": max_requests,
        "compile_every": compile_every,
        "teacher": teacher,
        "data_dir": str(data_dir),
        "parallel": parallel,
    }
    (run_root / "suite.json").write_text(
        json.dumps(suite_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    commands = _experiment_suite_commands(
        selected,
        run_root=run_root,
        max_requests=max_requests,
        compile_every=compile_every,
        teacher=teacher,
        data_dir=data_dir,
    )
    results = _run_experiment_suite_commands(commands, parallel=parallel)
    failed = [result for result in results if result["return_code"] != 0]
    if failed:
        for result in failed:
            error_console.print(
                f"[red]{result['experiment']} failed; log={result['log_path']}[/red]"
            )
        raise typer.Exit(code=2)

    comparison_path = ""
    if compare:
        run_dirs = _discover_run_dirs(run_root)
        if run_dirs:
            comparison = generate_experiment_comparison_report(
                run_dirs,
                run_root / "comparison",
            )
            comparison_path = str(comparison.comparison_csv_path)
    console.print(
        f"suite ran {len(results)} experiments under {run_root}; comparison={comparison_path}"
    )


@experiments_app.command("preflight")
def experiment_preflight(
    run_dir: Annotated[
        Path,
        typer.Option(help="Run directory whose teacher cache/report artifacts should be checked."),
    ] = Path("runs/latest"),
    data_dir: Annotated[
        Path,
        typer.Option(help="Processed data directory produced by prepare."),
    ] = DEFAULT_PROCESSED_DATA_DIR,
    teacher: Annotated[
        str,
        typer.Option(help="Teacher mode to validate: live, cache, or live-or-cache."),
    ] = "live-or-cache",
    out: Annotated[
        Path | None,
        typer.Option(help="Optional JSON path for preflight results."),
    ] = None,
    check_l1_build: Annotated[
        bool,
        typer.Option(help="Build the configured L1 Rust crate during preflight."),
    ] = False,
) -> None:
    """Check local readiness before running experiments."""

    settings = _load_cli_settings()
    payload = _experiment_preflight_payload(
        run_dir=run_dir,
        data_dir=data_dir,
        teacher=teacher,
        settings=settings,
        check_l1_build=check_l1_build,
    )
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    console.print_json(data=payload)
    if payload["status"] == "fail":
        raise typer.Exit(code=2)


def _experiment_preflight_payload(
    *,
    run_dir: Path,
    data_dir: Path,
    teacher: str,
    settings,
    check_l1_build: bool = False,
) -> dict:
    checks = [
        _preflight_data_check(data_dir),
        _preflight_teacher_check(run_dir=run_dir, teacher=teacher, settings=settings),
        _preflight_l1_check(settings=settings, check_l1_build=check_l1_build),
        _preflight_l1_agent_check(settings=settings),
        _preflight_l3_check(run_dir=run_dir, settings=settings),
    ]
    status = "fail" if any(check["status"] == "fail" for check in checks) else "pass"
    return {
        "schema_version": "experiment-preflight-v1",
        "status": status,
        "run_dir": str(run_dir),
        "data_dir": str(data_dir),
        "teacher": teacher,
        "checks": checks,
    }


def _preflight_data_check(data_dir: Path) -> dict:
    train_path = data_dir / "train.jsonl"
    if not train_path.exists():
        return {
            "name": "data.train_split",
            "status": "fail",
            "message": f"missing processed train split: {train_path}",
        }
    if train_path.stat().st_size == 0:
        return {
            "name": "data.train_split",
            "status": "fail",
            "message": f"processed train split is empty: {train_path}",
        }
    return {
        "name": "data.train_split",
        "status": "pass",
        "message": f"found {train_path}",
    }


def _preflight_teacher_check(*, run_dir: Path, teacher: str, settings) -> dict:
    cache_path = run_dir / "teacher_cache.jsonl"
    try:
        require_live_or_cached_teacher(settings, teacher, cache_path)
    except MissingTeacherError as exc:
        return {
            "name": "teacher",
            "status": "fail",
            "message": str(exc),
        }
    return {
        "name": "teacher",
        "status": "pass",
        "message": f"teacher mode {teacher!r} is usable",
    }


def _preflight_l1_check(*, settings, check_l1_build: bool) -> dict:
    crate_dir = settings.l1_rust_crate_dir
    cargo_toml = crate_dir / "Cargo.toml"
    if not cargo_toml.exists():
        return {
            "name": "l1.rust_crate",
            "status": "fail",
            "message": f"missing L1 Cargo.toml: {cargo_toml}",
        }
    if check_l1_build:
        try:
            binary_path = build_l1_binary(crate_dir)
        except Exception as exc:
            return {
                "name": "l1.rust_crate",
                "status": "fail",
                "message": f"L1 build failed: {exc}",
            }
        return {
            "name": "l1.rust_crate",
            "status": "pass",
            "message": f"built L1 worker: {binary_path}",
        }
    return {
        "name": "l1.rust_crate",
        "status": "pass",
        "message": f"found L1 crate: {crate_dir}",
    }


def _preflight_l1_agent_check(*, settings) -> dict:
    if settings.l1_agent_mode == "disabled":
        return {
            "name": "l1.agent",
            "status": "warn",
            "message": (
                "L1_AGENT_MODE=disabled; set agent-session for real L1 evolution experiments"
            ),
        }
    if settings.l1_agent_mode == "dry-run":
        if settings.l1_agent_dry_run_patch is None:
            return {
                "name": "l1.agent",
                "status": "fail",
                "message": "L1_AGENT_MODE=dry-run requires L1_AGENT_DRY_RUN_PATCH",
            }
        if not settings.l1_agent_dry_run_patch.exists():
            return {
                "name": "l1.agent",
                "status": "fail",
                "message": f"dry-run patch missing: {settings.l1_agent_dry_run_patch}",
            }
        return {
            "name": "l1.agent",
            "status": "pass",
            "message": f"dry-run patch found: {settings.l1_agent_dry_run_patch}",
        }
    if shutil.which(settings.l1_agent_codex_command) is None:
        return {
            "name": "l1.agent",
            "status": "fail",
            "message": f"codex command not found: {settings.l1_agent_codex_command}",
        }
    return {
        "name": "l1.agent",
        "status": "pass",
        "message": f"codex command found: {settings.l1_agent_codex_command}",
    }


def _preflight_l3_check(*, run_dir: Path, settings) -> dict:
    benchmark_path = run_dir / "reports" / "l3_benchmark.json"
    base = {
        "name": "l3.local_slm",
        "mode": settings.local_slm_mode,
        "model_name": settings.local_slm_model,
        "device_policy": settings.local_slm_device_policy,
        "benchmark_path": str(benchmark_path),
        "model_load_attempted": False,
        "runtime_blocking": settings.local_slm_mode == "guarded",
    }
    if settings.local_slm_mode == "disabled":
        return {
            **base,
            "status": "pass",
            "message": "local SLM disabled",
            "readiness": "disabled_nonblocking",
            "benchmark_required": False,
            "benchmark_status": "not_required",
        }

    if not benchmark_path.exists():
        status = "fail" if settings.local_slm_mode == "guarded" else "warn"
        return {
            **base,
            "status": status,
            "message": f"run `edge-mvp l3 bench --out {benchmark_path}` before relying on L3",
            "readiness": "benchmark_missing",
            "benchmark_required": True,
            "benchmark_status": "missing",
        }

    try:
        benchmark_payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        status = "fail" if settings.local_slm_mode == "guarded" else "warn"
        return {
            **base,
            "status": status,
            "message": f"L3 benchmark artifact is unreadable: {benchmark_path}",
            "readiness": "benchmark_unreadable",
            "benchmark_required": True,
            "benchmark_status": "unreadable",
            "error": str(exc),
        }

    benchmark_status = str(benchmark_payload.get("status", "unknown"))
    if benchmark_status != "success":
        status = "fail" if settings.local_slm_mode == "guarded" else "warn"
        return {
            **base,
            "status": status,
            "message": f"L3 benchmark did not succeed: {benchmark_status}",
            "readiness": "benchmark_failed",
            "benchmark_required": True,
            "benchmark_status": benchmark_status,
            "benchmark_error": benchmark_payload.get("error"),
        }

    backend = benchmark_payload.get("backend")
    actual_device = backend.get("actual_device") if isinstance(backend, dict) else None
    return {
        **base,
        "status": "pass",
        "message": f"found successful L3 benchmark artifact: {benchmark_path}",
        "readiness": "benchmark_success",
        "benchmark_required": True,
        "benchmark_status": "success",
        "actual_device": actual_device,
        "requests": benchmark_payload.get("requests"),
        "generation_p50_ms": benchmark_payload.get("generation_p50_ms"),
        "generation_p95_ms": benchmark_payload.get("generation_p95_ms"),
    }


def _discover_run_dirs(root: Path) -> list[Path]:
    if (root / "traces.jsonl").exists():
        return [root]
    return sorted({path.parent for path in root.rglob("traces.jsonl")})


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(path)
    return deduped


DEFAULT_EXPERIMENT_SUITE = (
    "main-evolution",
    "direct-l4-optimization",
    "l2-family",
    "no-guard",
    "no-l2",
    "hard-buffer",
    "workload-locality",
)


def _experiment_suite_commands(
    experiments: list[str],
    *,
    run_root: Path,
    max_requests: int,
    compile_every: int,
    teacher: str,
    data_dir: Path,
) -> list[dict]:
    commands: list[dict] = []
    for experiment_name in experiments:
        experiment_spec(experiment_name)
        command = [sys.executable, "-m", "darjeeling.cli"]
        if _settings_path is not None:
            command.extend(["--settings", str(_settings_path)])
        command.extend(
            [
                "experiment",
                experiment_name,
                "--run-dir",
                str(run_root / experiment_name),
                "--max-requests",
                str(max_requests),
                "--compile-every",
                str(compile_every),
                "--teacher",
                teacher,
                "--data-dir",
                str(data_dir),
            ]
        )
        commands.append(
            {
                "experiment": experiment_name,
                "command": command,
                "run_dir": run_root / experiment_name,
                "log_path": run_root / experiment_name / "suite.log",
            }
        )
    return commands


def _run_experiment_suite_commands(commands: list[dict], *, parallel: int) -> list[dict]:
    with ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = [executor.submit(_run_experiment_suite_command, command) for command in commands]
        return [future.result() for future in as_completed(futures)]


def _run_experiment_suite_command(command_spec: dict) -> dict:
    run_dir = Path(command_spec["run_dir"])
    log_path = Path(command_spec["log_path"])
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [str(part) for part in command_spec["command"]]
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
        check=False,
    )
    log_path.write_text(completed.stdout or "", encoding="utf-8")
    return {
        "experiment": command_spec["experiment"],
        "command": command,
        "run_dir": str(run_dir),
        "log_path": str(log_path),
        "return_code": completed.returncode,
    }


def _run_single_experiment(
    experiment_name: str,
    *,
    run_dir: Path,
    stream: str | None,
    max_requests: int | None,
    compile_every: int | None,
    teacher: str,
    data_dir: Path,
) -> None:
    spec = experiment_spec(experiment_name)
    _execute_experiment_run(
        spec,
        run_dir=run_dir,
        stream=stream or spec.default_stream,
        max_requests=max_requests or spec.default_max_requests,
        compile_every=compile_every or spec.default_compile_every,
        teacher=teacher,
        data_dir=data_dir,
    )


def _run_workload_locality_experiment(
    *,
    run_dir: Path,
    max_requests: int | None,
    compile_every: int | None,
    teacher: str,
    data_dir: Path,
) -> None:
    spec = experiment_spec("workload-locality")
    run_dir.mkdir(parents=True, exist_ok=True)
    top_level = experiment_metadata(
        spec,
        stream=",".join(spec.substreams),
        max_requests=max_requests or spec.default_max_requests,
        compile_every=compile_every or spec.default_compile_every,
        teacher=teacher,
        data_dir=str(data_dir),
    )
    (run_dir / "experiment.json").write_text(
        json.dumps(top_level, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for substream in spec.substreams:
        _execute_experiment_run(
            spec,
            run_dir=run_dir / substream,
            stream=substream,
            max_requests=max_requests or spec.default_max_requests,
            compile_every=compile_every or spec.default_compile_every,
            teacher=teacher,
            data_dir=data_dir,
        )


def _execute_experiment_run(
    spec: ExperimentSpec,
    *,
    run_dir: Path,
    stream: str,
    max_requests: int,
    compile_every: int,
    teacher: str,
    data_dir: Path,
) -> None:
    settings = apply_experiment_settings(_load_cli_settings(), spec)
    run_dir.mkdir(parents=True, exist_ok=True)
    _reset_experiment_run_state(run_dir)
    (run_dir / "experiment.json").write_text(
        json.dumps(
            experiment_metadata(
                spec,
                stream=stream,
                max_requests=max_requests,
                compile_every=compile_every,
                teacher=teacher,
                data_dir=str(data_dir),
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        summary = _execute_replay_run(
            stream=stream,
            max_requests=max_requests,
            compile_every=compile_every,
            teacher=teacher,
            run_dir=run_dir,
            data_dir=data_dir,
            target="nlu",
            settings=settings,
        )
        report_result = generate_run_report(run_dir)
    except (FileNotFoundError, LocalSLMError, MissingTeacherError, ValueError) as exc:
        error_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc
    console.print(
        f"experiment {spec.name} ran {summary.requests} requests; "
        f"reports={report_result.report_dir}; layers={summary.layer_counts}"
    )


def _reset_experiment_run_state(run_dir: Path) -> None:
    """Keep teacher cache but remove prior runtime artifacts from this experiment dir."""

    for directory_name in ["artifacts", "reports"]:
        directory = run_dir / directory_name
        if directory.exists():
            shutil.rmtree(directory)
    for file_name in ["traces.jsonl", "settings.json", "experiment.json"]:
        path = run_dir / file_name
        if path.exists():
            path.unlink()


if __name__ == "__main__":
    app()
