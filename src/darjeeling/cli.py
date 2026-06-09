from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from darjeeling.artifacts.store import ArtifactManifest, ArtifactStore
from darjeeling.compiler.l2_distiller import l2_config_from_settings
from darjeeling.compiler.l2_tuner import L2TuneSpec, tune_l2_student
from darjeeling.compiler.l3_prompt_optimizer import (
    l3_prompt_artifact_hash,
    replay_l3_prompt_artifact,
)
from darjeeling.data.massive import prepare_massive_dataset
from darjeeling.eval.experiments import (
    ExperimentSpec,
    apply_experiment_settings,
    experiment_metadata,
    experiment_spec,
)
from darjeeling.eval.reports import (
    generate_experiment_comparison_report,
    generate_run_report,
)
from darjeeling.layers.l1_rust_programbank import (
    DEFAULT_BENCHMARK_UTTERANCES,
    benchmark_worker,
    binary_path_for,
    build_l1_binary,
)
from darjeeling.layers.l2_student import (
    L2StudentConfig,
    train_l2_student,
    training_examples_from_teacher_traces,
)
from darjeeling.layers.l3_local_slm import (
    DEFAULT_L3_BENCHMARK_UTTERANCES,
    L3LocalSLMLayer,
    L3PromptArtifact,
    LocalSLMConfig,
    LocalSLMError,
    benchmark_l3_layer,
)
from darjeeling.layers.l4_cloud_llm import (
    MissingTeacherError,
    TaskSchema,
    require_live_or_cached_teacher,
)
from darjeeling.runtime.replay import (
    load_processed_records,
    run_replay,
    task_schema_from_records,
    write_run_settings,
)
from darjeeling.runtime.trace import read_traces
from darjeeling.schemas import traces_to_teacher_view
from darjeeling.settings import load_settings

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
def prepare(
    locale: Annotated[str, typer.Option(help="MASSIVE locale/config to prepare.")] = "en-US",
    out: Annotated[
        Path,
        typer.Option(help="Output directory for processed parquet/jsonl files."),
    ] = Path("data/processed/massive_en_us"),
) -> None:
    """Download and process MASSIVE records for replay."""

    result = prepare_massive_dataset(locale=locale, out_dir=out)
    console.print(f"prepared {result['records']} records in {out}")


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
        typer.Option(help="Processed MASSIVE data directory produced by prepare."),
    ] = Path("data/processed/massive_en_us"),
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
) -> None:
    """Generate run report files."""

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
    settings,
):
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
    ] = Path("native/l1_programbank"),
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
    ] = Path("native/l1_programbank"),
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
        typer.Option(help="Optional processed MASSIVE data dir for task schema discovery."),
    ] = Path("data/processed/massive_en_us"),
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
        typer.Option(help="Optional processed MASSIVE data dir for task schema discovery."),
    ] = Path("data/processed/massive_en_us"),
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
            "alarm_set",
            "music_play",
            "weather_query",
        ],
        slot_names=[
            "date",
            "location",
            "time",
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
        typer.Option(help="Processed MASSIVE data directory produced by prepare."),
    ] = Path("data/processed/massive_en_us"),
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
        typer.Option(help="Processed MASSIVE data directory produced by prepare."),
    ] = Path("data/processed/massive_en_us"),
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
        typer.Option(help="Processed MASSIVE data directory produced by prepare."),
    ] = Path("data/processed/massive_en_us"),
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
        typer.Option(help="Processed MASSIVE data directory produced by prepare."),
    ] = Path("data/processed/massive_en_us"),
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
        typer.Option(help="Processed MASSIVE data directory produced by prepare."),
    ] = Path("data/processed/massive_en_us"),
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
        typer.Option(help="Processed MASSIVE data directory produced by prepare."),
    ] = Path("data/processed/massive_en_us"),
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
        typer.Option(help="Processed MASSIVE data directory produced by prepare."),
    ] = Path("data/processed/massive_en_us"),
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
        typer.Option(help="Processed MASSIVE data directory produced by prepare."),
    ] = Path("data/processed/massive_en_us"),
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
        typer.Option(help="Processed MASSIVE data directory produced by prepare."),
    ] = Path("data/processed/massive_en_us"),
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
        typer.Option(help="Processed MASSIVE data directory produced by prepare."),
    ] = Path("data/processed/massive_en_us"),
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
        typer.Option(help="Processed MASSIVE data directory produced by prepare."),
    ] = Path("data/processed/massive_en_us"),
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
        typer.Option(help="Processed MASSIVE data directory produced by prepare."),
    ] = Path("data/processed/massive_en_us"),
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
            "message": "L1_AGENT_MODE=disabled; set codex-cli for real L1 evolution experiments",
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
    if settings.local_slm_mode == "disabled":
        return {
            "name": "l3.local_slm",
            "status": "pass",
            "message": "local SLM disabled",
        }
    benchmark_path = run_dir / "reports" / "l3_benchmark.json"
    if benchmark_path.exists():
        return {
            "name": "l3.local_slm",
            "status": "pass",
            "message": f"found L3 benchmark artifact: {benchmark_path}",
        }
    return {
        "name": "l3.local_slm",
        "status": "warn",
        "message": f"run `edge-mvp l3 bench --out {benchmark_path}` before relying on L3",
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
