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
from darjeeling.targets import registry as target_registry
from darjeeling.targets.nlu.adapters.clinc150 import prepare_clinc150_dataset
from darjeeling.targets.nlu.adapters.massive import prepare_massive_dataset
from darjeeling.targets.nlu.clinc150_phase1 import (
    DEFAULT_CLINC150_PROMPTS,
    DEFAULT_CLINC150_THRESHOLDS,
    Clinc150L4ReplayOracle,
    build_clinc150_gate_records,
    build_clinc150_label_cards,
    clinc150_metrics_from_teacher_rows,
    compare_repeated_teacher_rows,
    evaluate_clinc150_l1,
    evaluate_clinc150_l2,
    load_teacher_rows,
    run_clinc150_teacher_live_eval,
    sample_clinc150_records,
    train_clinc150_l2,
    training_examples_from_gold_records,
    training_examples_from_teacher_rows,
    write_clinc150_l1_eval_artifacts,
    write_clinc150_l2_eval_artifacts,
    write_clinc150_l2_train_artifacts,
    write_clinc150_teacher_cost_ledger,
)
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
    RustL1Worker,
    benchmark_worker,
    binary_path_for,
    build_l1_binary,
)
from darjeeling.targets.nlu.layers.l2_student import (
    L2StudentBundle,
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
    has_valid_teacher_cache,
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
from darjeeling.targets.nlu.settings import (
    DEFAULT_NLU_L1_CRATE_DIR,
    DEFAULT_PROCESSED_DATA_DIR,
    load_settings,
)
from darjeeling.targets.nlu.target import NluTargetSpec
from darjeeling.targets.nlu.teacher_eval import (
    DEFAULT_TEACHER_PROMPT_COMPARISON,
    run_teacher_live_vs_gold,
    run_teacher_prompt_comparison,
)
from darjeeling.targets.nlu.trace import read_traces

app = typer.Typer(no_args_is_help=True)
console = Console()
error_console = Console(stderr=True)
_settings_path: Path | None = None
_EXPERIMENT_RESUME_ENV = "DARJEELING_EXPERIMENT_RESUME_EXISTING"


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


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise typer.BadParameter("at least one float value is required")
    try:
        return tuple(float(part) for part in parts)
    except ValueError as exc:
        raise typer.BadParameter(f"invalid float list: {value}") from exc


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise typer.BadParameter("at least one integer value is required")
    try:
        result = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise typer.BadParameter(f"invalid integer list: {value}") from exc
    if any(part <= 0 for part in result):
        raise typer.BadParameter("integer values must be positive")
    return result


def _current_git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    commit = completed.stdout.strip()
    return commit or None


def _experiment_resume_existing() -> bool:
    return os.environ.get(_EXPERIMENT_RESUME_ENV, "").lower() in {"1", "true", "yes"}


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
    resume_existing: bool = False,
):
    target_spec = target_registry.get_target(target)
    run_dir.mkdir(parents=True, exist_ok=True)
    _seed_teacher_cache_for_cache_mode(settings=settings, teacher=teacher, run_dir=run_dir)
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
        resume_existing=resume_existing,
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
        "commit_hash": _current_git_commit(),
    }


experiments_app = typer.Typer(no_args_is_help=True)
app.add_typer(experiments_app, name="experiment")

l1_app = typer.Typer(no_args_is_help=True)
app.add_typer(l1_app, name="l1")

l2_app = typer.Typer(no_args_is_help=True)
app.add_typer(l2_app, name="l2")

l3_app = typer.Typer(no_args_is_help=True)
app.add_typer(l3_app, name="l3")

teacher_app = typer.Typer(no_args_is_help=True)
app.add_typer(teacher_app, name="teacher")

massive_app = typer.Typer(no_args_is_help=True)
app.add_typer(massive_app, name="massive")

clinc150_app = typer.Typer(no_args_is_help=True)
app.add_typer(clinc150_app, name="clinc150")


@massive_app.command("prepare")
def prepare_massive(
    locale: Annotated[str, typer.Option(help="MASSIVE locale/config to prepare.")] = "en-US",
    out: Annotated[
        Path,
        typer.Option(help="Output directory for processed parquet/jsonl files."),
    ] = DEFAULT_PROCESSED_DATA_DIR,
) -> None:
    """Download and process MASSIVE records for NLU replay."""

    result = prepare_massive_dataset(locale=locale, out_dir=out)
    console.print(f"prepared {result['records']} records in {out}")


@clinc150_app.command("prepare")
def prepare_clinc150(
    out: Annotated[
        Path,
        typer.Option(help="Output directory for processed CLINC150 parquet/jsonl files."),
    ] = Path("data/processed/clinc150_data_full"),
    source: Annotated[
        str | None,
        typer.Option(help="Optional local path or URL for data_full.json."),
    ] = None,
    expected_sha256: Annotated[
        str | None,
        typer.Option(help="Expected SHA256 for data_full.json; pass empty to skip."),
    ] = None,
) -> None:
    """Download or read pinned CLINC150 data_full and process it for NLU replay."""

    checksum = expected_sha256 if expected_sha256 else None
    if expected_sha256 is None:
        from darjeeling.targets.nlu.adapters.clinc150 import CLINC150_DATA_FULL_SHA256

        checksum = CLINC150_DATA_FULL_SHA256
    result = prepare_clinc150_dataset(
        out_dir=out,
        source=source,
        expected_sha256=checksum,
    )
    console.print(f"prepared {result['records']} CLINC150 records in {out}")
    console.print_json(data=result)


@clinc150_app.command("teacher-gate")
def clinc150_teacher_gate(
    out_dir: Annotated[
        Path,
        typer.Option(help="Output directory for CLINC150 teacher gate artifacts."),
    ] = Path("runs/clinc150-phase1/teacher-gate-500"),
    data_dir: Annotated[
        Path,
        typer.Option(help="Processed CLINC150 data directory."),
    ] = Path("data/processed/clinc150_data_full"),
    prompt_version: Annotated[
        list[str] | None,
        typer.Option("--prompt-version", help="Prompt version to evaluate; repeatable."),
    ] = None,
    max_workers: Annotated[
        int,
        typer.Option(min=1, help="Parallel live teacher calls."),
    ] = 1,
    resume_existing: Annotated[
        bool,
        typer.Option(
            "--resume-existing/--no-resume-existing",
            help="Resume from existing CLINC150 teacher gate details JSONL rows.",
        ),
    ] = False,
) -> None:
    """Run the fixed 500-request CLINC150 validation teacher gate."""

    settings = _load_cli_settings()
    train_records = load_processed_records(data_dir, split="train")
    validation_records = load_processed_records(data_dir, split="validation")
    gate_records = build_clinc150_gate_records(validation_records)
    task_schema = task_schema_from_records(validation_records)
    label_cards = build_clinc150_label_cards(train_records)
    prompt_versions = prompt_version or list(DEFAULT_CLINC150_PROMPTS)
    results = []
    artifacts = []
    for version in prompt_versions:
        result = run_clinc150_teacher_live_eval(
            records=gate_records,
            task_schema=task_schema,
            settings=settings,
            split="validation",
            stream="clinc150-gate-500",
            prompt_version=version,
            out_dir=out_dir / version,
            label_cards=label_cards if version.endswith("label-cards") else None,
            max_workers=max_workers,
            resume_existing=resume_existing,
        )
        artifacts.append(result)
        results.append(
            {
                "prompt_version": version,
                "summary_path": str(result.artifacts.summary_json_path),
                "metrics_path": str(result.clinc_metrics_path),
                "details_jsonl_path": str(result.artifacts.details_jsonl_path),
                "cost_ledger_path": str(result.artifacts.cost_ledger_path),
                **result.clinc_metrics,
            }
        )
    comparison = {
        "schema_version": "clinc150-teacher-gate-comparison-v1",
        "request_ids": [record.request_id for record in gate_records],
        "prompt_versions": prompt_versions,
        "rows": results,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = out_dir / "clinc150_teacher_gate_comparison.json"
    comparison_path.write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    cost_ledger_path = write_clinc150_teacher_cost_ledger(
        out_dir=out_dir,
        artifacts=artifacts,
        run_kind="teacher-gate",
    )
    console.print(f"wrote {comparison_path}")
    console.print(f"wrote {cost_ledger_path}")
    console.print_json(data=comparison)
    if not any(row.get("passed_teacher_gate") for row in results):
        raise typer.Exit(code=3)


@clinc150_app.command("teacher-eval")
def clinc150_teacher_eval(
    out_dir: Annotated[
        Path,
        typer.Option(help="Output directory for CLINC150 teacher artifacts."),
    ],
    data_dir: Annotated[
        Path,
        typer.Option(help="Processed CLINC150 data directory."),
    ] = Path("data/processed/clinc150_data_full"),
    split: Annotated[str, typer.Option(help="Processed data split to evaluate.")] = "validation",
    stream: Annotated[
        str,
        typer.Option(
            help=(
                "Sample stream type: sequential, stratified, uniform, zipf-mild, "
                "or zipf-heavy."
            ),
        ),
    ] = "sequential",
    max_requests: Annotated[int | None, typer.Option(min=1)] = None,
    prompt_version: Annotated[
        str,
        typer.Option(help="CLINC150 teacher prompt version."),
    ] = "clinc150-intent-v1",
    label_cards: Annotated[
        bool,
        typer.Option(help="Include train-only label cards for label-card prompt."),
    ] = True,
    max_workers: Annotated[
        int,
        typer.Option(min=1, help="Parallel live teacher calls."),
    ] = 1,
    resume_existing: Annotated[
        bool,
        typer.Option(
            "--resume-existing/--no-resume-existing",
            help="Resume from existing CLINC150 teacher eval details JSONL rows.",
        ),
    ] = False,
    fail_on_gate: Annotated[
        bool,
        typer.Option(
            "--fail-on-gate/--no-fail-on-gate",
            help="Exit nonzero when CLINC150 gate-style teacher thresholds fail.",
        ),
    ] = True,
) -> None:
    """Run live CLINC150 teacher evaluation on a processed split or stream."""

    settings = _load_cli_settings()
    train_records = load_processed_records(data_dir, split="train")
    records = load_processed_records(data_dir, split=split)
    sample_records = sample_clinc150_records(
        records,
        stream=stream,
        max_requests=max_requests,
    )
    cards = (
        build_clinc150_label_cards(train_records)
        if label_cards and prompt_version.endswith("label-cards")
        else None
    )
    result = run_clinc150_teacher_live_eval(
        records=sample_records,
        task_schema=task_schema_from_records(records),
        settings=settings,
        split=split,
        stream=stream,
        prompt_version=prompt_version,
        out_dir=out_dir,
        label_cards=cards,
        max_workers=max_workers,
        resume_existing=resume_existing,
    )
    cost_ledger_path = write_clinc150_teacher_cost_ledger(
        out_dir=out_dir,
        artifacts=[result],
        run_kind="teacher-eval",
    )
    console.print(f"wrote {result.artifacts.summary_json_path}")
    console.print(f"wrote {cost_ledger_path}")
    console.print_json(data=result.artifacts.summary)
    if fail_on_gate and not result.clinc_metrics.get("passed_teacher_gate"):
        raise typer.Exit(code=3)


@clinc150_app.command("teacher-repeat")
def clinc150_teacher_repeat(
    first: Annotated[Path, typer.Argument(help="First teacher details JSONL.")],
    second: Annotated[Path, typer.Argument(help="Second teacher details JSONL.")],
    out: Annotated[
        Path | None,
        typer.Option(help="Optional JSON output path for consistency metrics."),
    ] = None,
) -> None:
    """Compare two CLINC150 teacher detail files for repeated-call consistency."""

    result = compare_repeated_teacher_rows(load_teacher_rows(first), load_teacher_rows(second))
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    console.print_json(data=result)


@clinc150_app.command("teacher-metrics")
def clinc150_teacher_metrics(
    details: Annotated[Path, typer.Argument(help="Teacher details JSONL.")],
    out: Annotated[
        Path | None,
        typer.Option(help="Optional JSON output path for CLINC150 teacher metrics."),
    ] = None,
) -> None:
    """Compute CLINC150-specific metrics from teacher detail rows."""

    rows = load_teacher_rows(details)
    result = clinc150_metrics_from_teacher_rows(rows)
    if out is None:
        out = details.parent / "clinc150_teacher_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    console.print(f"wrote {out}")
    console.print_json(data=result)


@clinc150_app.command("l2-train")
def clinc150_l2_train(
    out_dir: Annotated[
        Path,
        typer.Option(help="Output directory for CLINC150 L2 artifacts."),
    ],
    data_dir: Annotated[
        Path,
        typer.Option(help="Processed CLINC150 data directory."),
    ] = Path("data/processed/clinc150_data_full"),
    source: Annotated[
        str,
        typer.Option(help="Training source: gold or teacher."),
    ] = "gold",
    split: Annotated[
        str,
        typer.Option(help="Processed split for gold training examples."),
    ] = "train",
    teacher_details: Annotated[
        Path | None,
        typer.Option(help="Teacher details JSONL for teacher-distilled training."),
    ] = None,
    sample_stream: Annotated[
        str,
        typer.Option(help="Gold sample stream: sequential, stratified, uniform, or zipf-heavy."),
    ] = "stratified",
    max_examples: Annotated[int | None, typer.Option(min=1)] = None,
    accept_threshold: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.99,
    random_state: Annotated[int, typer.Option()] = 17,
    intent_model_family: Annotated[
        str,
        typer.Option(help="L2 intent model family: sgd_logreg or mlp."),
    ] = "sgd_logreg",
    max_features: Annotated[int, typer.Option(min=1)] = 50_000,
    max_iter: Annotated[int, typer.Option(min=1)] = 1000,
    frame_source: Annotated[
        str,
        typer.Option(help="L2 frame source: student or retrieval."),
    ] = "student",
    mlp_hidden_layer_sizes: Annotated[
        str,
        typer.Option(help="Comma-separated MLP hidden layer sizes."),
    ] = "64",
    mlp_alpha: Annotated[float, typer.Option(min=0.0)] = 0.0001,
    mlp_early_stopping: Annotated[bool, typer.Option()] = False,
) -> None:
    """Train a target-local CLINC150 diagnostic or teacher-distilled L2 bundle."""

    if source not in {"gold", "teacher"}:
        raise typer.BadParameter("source must be gold or teacher")
    if source == "teacher" and teacher_details is None:
        raise typer.BadParameter("--teacher-details is required when --source=teacher")

    source_path: Path | None = None
    if source == "gold":
        records = load_processed_records(data_dir, split=split)
        sample_records = sample_clinc150_records(
            records,
            stream=sample_stream,
            max_requests=max_examples,
        )
        examples = training_examples_from_gold_records(sample_records)
    else:
        source_path = teacher_details
        rows = load_teacher_rows(teacher_details)
        if max_examples is not None:
            rows = rows[:max_examples]
        examples = training_examples_from_teacher_rows(rows)

    config = L2StudentConfig(
        accept_threshold=accept_threshold,
        random_state=random_state,
        min_examples=4,
        slot_model_family="none",
        frame_source=frame_source,
        intent_model_family=intent_model_family,
        max_features=max_features,
        max_iter=max_iter,
        mlp_hidden_layer_sizes=_parse_int_tuple(mlp_hidden_layer_sizes),
        mlp_alpha=mlp_alpha,
        mlp_early_stopping=mlp_early_stopping,
    )
    bundle = train_clinc150_l2(examples, config=config)
    artifact = write_clinc150_l2_train_artifacts(
        bundle=bundle,
        examples=examples,
        out_dir=out_dir,
        training_source=source,
        source_path=source_path,
        split=split if source == "gold" else None,
        sample_stream=sample_stream if source == "gold" else None,
        max_examples=max_examples,
    )
    console.print(f"wrote {artifact.bundle_path}")
    console.print(f"wrote {artifact.summary_path}")
    console.print_json(data=artifact.summary)


@clinc150_app.command("l2-eval")
def clinc150_l2_eval(
    bundle_path: Annotated[
        Path,
        typer.Option(help="Path to a saved CLINC150 L2 bundle."),
    ],
    out_dir: Annotated[
        Path,
        typer.Option(help="Output directory for CLINC150 L2 eval artifacts."),
    ],
    data_dir: Annotated[
        Path,
        typer.Option(help="Processed CLINC150 data directory."),
    ] = Path("data/processed/clinc150_data_full"),
    split: Annotated[str, typer.Option(help="Processed split to evaluate.")] = "validation",
    stream: Annotated[
        str,
        typer.Option(
            help="Eval stream: sequential, stratified, uniform, zipf-mild, or zipf-heavy.",
        ),
    ] = "sequential",
    max_requests: Annotated[int | None, typer.Option(min=1)] = None,
    teacher_details: Annotated[
        Path | None,
        typer.Option(help="Paired teacher details JSONL for all-L4 fallback metrics."),
    ] = None,
    thresholds: Annotated[
        str,
        typer.Option(help="Comma-separated L2 accept thresholds."),
    ] = ",".join(str(value) for value in DEFAULT_CLINC150_THRESHOLDS),
    write_details: Annotated[
        bool,
        typer.Option("--write-details/--no-write-details"),
    ] = False,
) -> None:
    """Evaluate a saved CLINC150 L2 bundle with L0 disabled and optional L4 fallback rows."""

    records = load_processed_records(data_dir, split=split)
    sample_records = sample_clinc150_records(
        records,
        stream=stream,
        max_requests=max_requests,
    )
    teacher_rows = load_teacher_rows(teacher_details) if teacher_details is not None else None
    result = evaluate_clinc150_l2(
        bundle=L2StudentBundle.load(bundle_path),
        records=sample_records,
        teacher_rows=teacher_rows,
        thresholds=_parse_float_tuple(thresholds),
        include_prediction_rows=write_details,
    )
    artifact = write_clinc150_l2_eval_artifacts(result=result, out_dir=out_dir)
    console.print(f"wrote {artifact.summary_path}")
    console.print(f"wrote {artifact.cost_latency_path}")
    if artifact.details_jsonl_path is not None:
        console.print(f"wrote {artifact.details_jsonl_path}")
    console.print_json(data=artifact.summary)


@clinc150_app.command("l1-eval")
def clinc150_l1_eval(
    crate_dir: Annotated[
        Path,
        typer.Option(help="Rust L1 ProgramBank crate directory."),
    ],
    out_dir: Annotated[
        Path,
        typer.Option(help="Output directory for CLINC150 L1 eval artifacts."),
    ],
    data_dir: Annotated[
        Path,
        typer.Option(help="Processed CLINC150 data directory."),
    ] = Path("data/processed/clinc150_data_full"),
    split: Annotated[str, typer.Option(help="Processed split to evaluate.")] = "validation",
    stream: Annotated[
        str,
        typer.Option(
            help="Eval stream: sequential, stratified, uniform, zipf-mild, or zipf-heavy.",
        ),
    ] = "sequential",
    max_requests: Annotated[int | None, typer.Option(min=1)] = None,
    teacher_details: Annotated[
        Path | None,
        typer.Option(help="Paired CLINC150 teacher details JSONL for replay-oracle fallback."),
    ] = None,
    release: Annotated[bool, typer.Option(help="Build and use the release profile.")] = False,
    timeout_s: Annotated[
        float,
        typer.Option(min=0.1, help="Per-request L1 worker timeout in seconds."),
    ] = 2.0,
    write_details: Annotated[
        bool,
        typer.Option("--write-details/--no-write-details"),
    ] = False,
) -> None:
    """Evaluate a Rust L1 ProgramBank over CLINC150 with L0/L2/L3 disabled."""

    records = load_processed_records(data_dir, split=split)
    sample_records = sample_clinc150_records(
        records,
        stream=stream,
        max_requests=max_requests,
    )
    replay_oracle = (
        Clinc150L4ReplayOracle.from_path(teacher_details)
        if teacher_details is not None
        else None
    )
    binary_path = build_l1_binary(crate_dir, release=release)
    with RustL1Worker(binary_path, timeout_s=timeout_s) as worker:
        result = evaluate_clinc150_l1(
            worker=worker,
            records=sample_records,
            replay_oracle=replay_oracle,
            include_prediction_rows=write_details,
        )
    artifact = write_clinc150_l1_eval_artifacts(result=result, out_dir=out_dir)
    console.print(f"wrote {artifact.summary_path}")
    console.print(f"wrote {artifact.cost_latency_path}")
    console.print(f"wrote {artifact.accepted_errors_path}")
    if artifact.details_jsonl_path is not None:
        console.print(f"wrote {artifact.details_jsonl_path}")
    console.print_json(data=artifact.summary)


@teacher_app.command("eval-live")
def teacher_eval_live(
    out_dir: Annotated[
        Path,
        typer.Option(help="Output directory for teacher-live-vs-gold artifacts."),
    ] = Path("runs/teacher-live-vs-gold"),
    data_dir: Annotated[
        Path,
        typer.Option(help="Processed data directory produced by massive prepare."),
    ] = DEFAULT_PROCESSED_DATA_DIR,
    split: Annotated[str, typer.Option(help="Processed data split to evaluate.")] = "validation",
    stream: Annotated[
        str,
        typer.Option(help="Sample stream type: sequential, uniform, zipf-mild, or zipf-heavy."),
    ] = "sequential",
    max_requests: Annotated[int, typer.Option(min=1)] = 100,
    prompt_version: Annotated[
        str | None,
        typer.Option(help="Teacher prompt version; defaults to settings."),
    ] = None,
    min_frame_exact_match: Annotated[
        float,
        typer.Option(
            min=0.0,
            max=1.0,
            help="Fail the quality gate if frame exact match is below this value.",
        ),
    ] = 0.0,
) -> None:
    """Run live L4 directly against MASSIVE gold without cascade replay."""

    settings = _load_cli_settings()
    effective_prompt_version = prompt_version or settings.teacher_prompt_version
    try:
        result = run_teacher_live_vs_gold(
            data_dir=data_dir,
            split=split,
            stream=stream,
            max_requests=max_requests,
            prompt_version=effective_prompt_version,
            settings=settings,
            out_dir=out_dir,
            min_frame_exact_match=min_frame_exact_match,
        )
    except (FileNotFoundError, MissingTeacherError, ValueError) as exc:
        error_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc

    console.print(f"wrote {result.summary_json_path}")
    console.print_json(data=result.summary)
    if result.summary.get("passed") is False:
        raise typer.Exit(code=3)


@teacher_app.command("compare-prompts")
def teacher_compare_prompts(
    out_dir: Annotated[
        Path,
        typer.Option(help="Output directory for teacher prompt comparison artifacts."),
    ] = Path("runs/teacher-prompt-comparison"),
    data_dir: Annotated[
        Path,
        typer.Option(help="Processed data directory produced by massive prepare."),
    ] = DEFAULT_PROCESSED_DATA_DIR,
    split: Annotated[str, typer.Option(help="Processed data split to evaluate.")] = "validation",
    stream: Annotated[
        str,
        typer.Option(help="Sample stream type: sequential, uniform, zipf-mild, or zipf-heavy."),
    ] = "sequential",
    max_requests: Annotated[int, typer.Option(min=1)] = 100,
    prompt_version: Annotated[
        list[str] | None,
        typer.Option("--prompt-version", help="Prompt version to evaluate; repeatable."),
    ] = None,
    min_frame_exact_match: Annotated[
        float,
        typer.Option(
            min=0.0,
            max=1.0,
            help="Fail the comparison if any prompt is below this frame exact threshold.",
        ),
    ] = 0.0,
) -> None:
    """Compare live teacher prompt versions on the same MASSIVE sample."""

    settings = _load_cli_settings()
    prompt_versions = prompt_version or list(DEFAULT_TEACHER_PROMPT_COMPARISON)
    try:
        result = run_teacher_prompt_comparison(
            data_dir=data_dir,
            split=split,
            stream=stream,
            max_requests=max_requests,
            prompt_versions=prompt_versions,
            settings=settings,
            out_dir=out_dir,
            min_frame_exact_match=min_frame_exact_match,
        )
    except (FileNotFoundError, MissingTeacherError, ValueError) as exc:
        error_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc

    console.print(f"wrote {result.comparison_json_path}")
    console.print_json(data=result.summary)
    if any(row.get("passed") is False for row in result.summary.get("rows", [])):
        raise typer.Exit(code=3)


@l1_app.command("build")
def l1_build(
    crate_dir: Annotated[
        Path,
        typer.Option(help="Rust L1 ProgramBank crate directory."),
    ] = DEFAULT_NLU_L1_CRATE_DIR,
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
    ] = DEFAULT_NLU_L1_CRATE_DIR,
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
        target_name=NluTargetSpec.name,
        target_schema_version=NluTargetSpec.schema_version,
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
        typer.Option(help="Output directory from `edge-mvp-nlu l2 target-evolve`."),
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
        target_name=NluTargetSpec.name,
        target_schema_version=NluTargetSpec.schema_version,
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


@experiments_app.command("no-audit")
def experiment_no_audit(
    run_dir: Annotated[
        Path,
        typer.Option(help="Experiment run directory."),
    ] = Path("runs/no-audit"),
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
        "no-audit",
        run_dir=run_dir,
        stream=stream,
        max_requests=max_requests,
        compile_every=compile_every,
        teacher=teacher,
        data_dir=data_dir,
    )


@experiments_app.command("l2-global-student")
def experiment_l2_global_student(
    run_dir: Annotated[
        Path,
        typer.Option(help="Experiment run directory."),
    ] = Path("runs/l2-global-student"),
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
        "l2-global-student",
        run_dir=run_dir,
        stream=stream,
        max_requests=max_requests,
        compile_every=compile_every,
        teacher=teacher,
        data_dir=data_dir,
    )


@experiments_app.command("l2-expert-bank")
def experiment_l2_expert_bank(
    run_dir: Annotated[
        Path,
        typer.Option(help="Experiment run directory."),
    ] = Path("runs/l2-expert-bank"),
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
        "l2-expert-bank",
        run_dir=run_dir,
        stream=stream,
        max_requests=max_requests,
        compile_every=compile_every,
        teacher=teacher,
        data_dir=data_dir,
    )


@experiments_app.command("l3-disabled")
def experiment_l3_disabled(
    run_dir: Annotated[
        Path,
        typer.Option(help="Experiment run directory."),
    ] = Path("runs/l3-disabled"),
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
        "l3-disabled",
        run_dir=run_dir,
        stream=stream,
        max_requests=max_requests,
        compile_every=compile_every,
        teacher=teacher,
        data_dir=data_dir,
    )


@experiments_app.command("l3-shadow")
def experiment_l3_shadow(
    run_dir: Annotated[
        Path,
        typer.Option(help="Experiment run directory."),
    ] = Path("runs/l3-shadow"),
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
        "l3-shadow",
        run_dir=run_dir,
        stream=stream,
        max_requests=max_requests,
        compile_every=compile_every,
        teacher=teacher,
        data_dir=data_dir,
    )


@experiments_app.command("l3-guarded")
def experiment_l3_guarded(
    run_dir: Annotated[
        Path,
        typer.Option(help="Experiment run directory."),
    ] = Path("runs/l3-guarded"),
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
        "l3-guarded",
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
    include_guarded_l3: Annotated[
        bool,
        typer.Option(
            "--include-guarded-l3/--skip-guarded-l3",
            help="Append l3-guarded to the default suite after guarded L3 preflight passes.",
        ),
    ] = False,
    resume_existing: Annotated[
        bool,
        typer.Option(
            "--resume-existing/--no-resume-existing",
            help=(
                "Resume from existing trace prefixes in run directories after interrupted "
                "live runs."
            ),
        ),
    ] = False,
) -> None:
    """Run an experiment suite with subprocess-level parallelism."""

    selected = list(experiment or DEFAULT_EXPERIMENT_SUITE)
    if experiment is None and include_guarded_l3 and "l3-guarded" not in selected:
        selected.append("l3-guarded")
    run_root.mkdir(parents=True, exist_ok=True)
    suite_payload = {
        "schema_version": "experiment-suite-v1",
        "experiments": selected,
        "max_requests": max_requests,
        "compile_every": compile_every,
        "teacher": teacher,
        "data_dir": str(data_dir),
        "parallel": parallel,
        "include_guarded_l3": include_guarded_l3,
        "resume_existing": resume_existing,
        "commit_hash": _current_git_commit(),
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
        resume_existing=resume_existing,
    )
    results = _run_experiment_suite_commands(commands, parallel=parallel)
    (run_root / "results.json").write_text(
        json.dumps(
            {
                "schema_version": "experiment-suite-results-v1",
                "suite_path": str(run_root / "suite.json"),
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
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
    experiment: Annotated[
        str | None,
        typer.Option("--experiment", help="Apply an experiment spec before checking readiness."),
    ] = None,
) -> None:
    """Check local readiness before running experiments."""

    settings = _load_cli_settings()
    spec: ExperimentSpec | None = None
    if experiment is not None:
        spec = experiment_spec(experiment)
        settings = apply_experiment_settings(settings, spec)
    payload = _experiment_preflight_payload(
        run_dir=run_dir,
        data_dir=data_dir,
        teacher=teacher,
        settings=settings,
        check_l1_build=check_l1_build,
        experiment=spec,
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
    experiment: ExperimentSpec | None = None,
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
        "experiment": experiment.name if experiment is not None else "",
        "settings_overrides": experiment.settings_overrides if experiment is not None else {},
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
        seed_path = _seed_teacher_cache_for_cache_mode(
            settings=settings,
            teacher=teacher,
            run_dir=run_dir,
        )
        require_live_or_cached_teacher(settings, teacher, cache_path)
    except MissingTeacherError as exc:
        return {
            "name": "teacher",
            "status": "fail",
            "message": str(exc),
        }
    payload = {
        "name": "teacher",
        "status": "pass",
        "message": f"teacher mode {teacher!r} is usable",
    }
    if seed_path is not None:
        payload["seed_cache_path"] = str(seed_path)
    return payload


def _seed_teacher_cache_for_cache_mode(*, settings, teacher: str, run_dir: Path) -> Path | None:
    if teacher != "cache":
        return None

    cache_path = run_dir / "teacher_cache.jsonl"
    if has_valid_teacher_cache(cache_path):
        return None

    seed_path = settings.teacher_cache_seed_path
    if seed_path is None:
        return None
    seed_path = seed_path.expanduser()
    if seed_path == cache_path:
        return None
    if not has_valid_teacher_cache(seed_path):
        raise MissingTeacherError(
            "teacher cache is required but missing and configured seed cache "
            f"is unavailable: {seed_path}"
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(seed_path, cache_path)
    return seed_path


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
            "message": f"run `edge-mvp-nlu l3 bench --out {benchmark_path}` before relying on L3",
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
    "workload-locality",
    "no-guard",
    "no-audit",
    "no-l2",
    "l2-global-student",
    "l2-expert-bank",
    "l3-disabled",
    "l3-shadow",
)


def _experiment_suite_commands(
    experiments: list[str],
    *,
    run_root: Path,
    max_requests: int,
    compile_every: int,
    teacher: str,
    data_dir: Path,
    resume_existing: bool = False,
) -> list[dict]:
    commands: list[dict] = []
    for experiment_name in experiments:
        experiment_spec(experiment_name)
        command = [sys.executable, "-m", "darjeeling.targets.nlu.main_cli"]
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
        command_spec = {
            "experiment": experiment_name,
            "command": command,
            "run_dir": run_root / experiment_name,
            "log_path": run_root / experiment_name / "suite.log",
        }
        if resume_existing:
            command_spec["env"] = {_EXPERIMENT_RESUME_ENV: "1"}
        commands.append(command_spec)
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
    env = os.environ.copy()
    env.update(command_spec.get("env", {}))
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
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
    resume_existing = _experiment_resume_existing()
    if not resume_existing:
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
            resume_existing=resume_existing,
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
