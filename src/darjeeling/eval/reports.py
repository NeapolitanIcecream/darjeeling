from __future__ import annotations

import csv
import html
import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from darjeeling.artifacts.store import ArtifactManifest, ArtifactStore
from darjeeling.compiler.l3_prompt_optimizer import calibrate_l3_confidence_threshold
from darjeeling.compiler.mining import load_hard_buffer_jsonl, write_hard_buffer_jsonl
from darjeeling.data.frames import normalize_utterance
from darjeeling.layers.l1_rust_programbank import (
    DEFAULT_BENCHMARK_UTTERANCES,
    benchmark_worker,
    binary_path_for,
    build_l1_binary,
)
from darjeeling.runtime.trace import read_traces
from darjeeling.schemas import Frame, LayerName, TraceRecord

L1_BENCHMARK_FILENAME = "l1_benchmark.json"
L3_BENCHMARK_FILENAME = "l3_benchmark.json"
HARD_CASES_FILENAME = "hard_cases.jsonl"
EXPERIMENT_COMPARISON_FIELDNAMES = [
    "experiment",
    "stream",
    "run_dir",
    "requests",
    "frame_exact_match",
    "total_latency_p95_ms",
    "comparison_score",
    "comparison_rank",
    "l0_share",
    "l1_share",
    "l2_share",
    "l3_share",
    "l4_share",
    "promoted_generations",
    "promotion_attempts",
    "promoted_with_layer_regression",
    "bottleneck_codes",
    "bottleneck_count",
    "l1_benchmark_native_p95_us",
    "l1_benchmark_throughput_qps",
]


def ensure_report_dir(run_dir: Path) -> Path:
    report_dir = run_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir


@dataclass(frozen=True)
class RunReportResult:
    report_dir: Path
    summary_path: Path
    metrics_csv_path: Path
    artifacts_csv_path: Path
    curves_html_path: Path
    hard_cases_path: Path
    l1_benchmark_path: Path | None = None


@dataclass(frozen=True)
class ExperimentComparisonResult:
    out_dir: Path
    comparison_csv_path: Path
    comparison_html_path: Path


@dataclass(frozen=True)
class BottleneckFinding:
    code: str
    label: str
    evidence: str
    severity: str = "warning"


class L1BenchmarkUnavailable(RuntimeError):
    pass


def generate_run_report(run_dir: Path) -> RunReportResult:
    report_dir = ensure_report_dir(run_dir)
    settings_text = _read_text_or_default(run_dir / "settings.json", "{}")
    current_manifest = ArtifactStore(run_dir / "artifacts").load_current_manifest()
    promotion_records = _load_promotion_records(run_dir)
    generation_manifests = _load_generation_manifests(run_dir)
    traces = _read_trace_records(run_dir / "traces.jsonl")
    l1_benchmark = _ensure_l1_benchmark(
        report_dir,
        run_dir=run_dir,
        settings_text=settings_text,
        current_manifest=current_manifest,
        traces=traces,
    )
    l3_benchmark = _optional_json_object(report_dir / L3_BENCHMARK_FILENAME)

    bottlenecks = _failed_experiment_bottlenecks(
        traces=traces,
        promotion_records=promotion_records,
        settings_text=settings_text,
    )
    metrics_rows = _metrics_rows(
        run_dir,
        traces,
        promotion_records,
        current_manifest,
        generation_manifests,
        bottlenecks,
        l1_benchmark,
        l3_benchmark,
    )
    artifact_rows = _artifact_rows(current_manifest, generation_manifests)
    hard_cases_path = _write_report_hard_cases(
        report_dir / HARD_CASES_FILENAME,
        run_dir=run_dir,
        current_manifest=current_manifest,
        generation_manifests=generation_manifests,
    )

    metrics_csv_path = _write_csv(
        report_dir / "metrics.csv",
        rows=metrics_rows,
        fieldnames=["scope", "generation", "layer", "metric", "value"],
    )
    artifacts_csv_path = _write_csv(
        report_dir / "artifacts.csv",
        rows=artifact_rows,
        fieldnames=[
            "generation",
            "artifact_set_id",
            "promoted",
            "artifact_name",
            "artifact_path",
        ],
    )
    curves_html_path = _write_curves_html(
        report_dir / "curves.html",
        traces=traces,
        promotion_records=promotion_records,
        current_manifest=current_manifest,
        run_dir=run_dir,
        bottlenecks=bottlenecks,
        l1_benchmark=l1_benchmark,
        l3_benchmark=l3_benchmark,
    )
    summary_path = _write_summary_md(
        report_dir / "summary.md",
        run_dir=run_dir,
        settings_text=settings_text,
        current_manifest=current_manifest,
        generation_manifests=generation_manifests,
        traces=traces,
        promotion_records=promotion_records,
        bottlenecks=bottlenecks,
        l1_benchmark=l1_benchmark,
        l3_benchmark=l3_benchmark,
    )
    return RunReportResult(
        report_dir=report_dir,
        summary_path=summary_path,
        metrics_csv_path=metrics_csv_path,
        artifacts_csv_path=artifacts_csv_path,
        curves_html_path=curves_html_path,
        hard_cases_path=hard_cases_path,
        l1_benchmark_path=report_dir / L1_BENCHMARK_FILENAME if l1_benchmark is not None else None,
    )


def generate_experiment_comparison_report(
    run_dirs: Sequence[Path],
    out_dir: Path,
) -> ExperimentComparisonResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _rank_experiment_comparison_rows(
        [_experiment_comparison_row(run_dir) for run_dir in run_dirs]
    )
    comparison_csv_path = _write_csv(
        out_dir / "comparison.csv",
        rows=rows,
        fieldnames=EXPERIMENT_COMPARISON_FIELDNAMES,
    )
    comparison_html_path = _write_experiment_comparison_html(
        out_dir / "comparison.html",
        rows=rows,
    )
    return ExperimentComparisonResult(
        out_dir=out_dir,
        comparison_csv_path=comparison_csv_path,
        comparison_html_path=comparison_html_path,
    )


def _experiment_comparison_row(run_dir: Path) -> dict[str, Any]:
    traces = _read_trace_records(run_dir / "traces.jsonl")
    settings_text = _read_text_or_default(run_dir / "settings.json", "{}")
    promotion_records = _load_promotion_records(run_dir)
    bottlenecks = _failed_experiment_bottlenecks(
        traces=traces,
        promotion_records=promotion_records,
        settings_text=settings_text,
    )
    metadata = _load_json_object(run_dir / "experiment.json")
    layer_counts = Counter(trace.chosen_layer for trace in traces)
    requests = len(traces)
    benchmark = _load_json_object(run_dir / "reports" / L1_BENCHMARK_FILENAME)

    row: dict[str, Any] = {
        "experiment": metadata.get("experiment") or run_dir.name,
        "stream": metadata.get("stream", ""),
        "run_dir": str(run_dir),
        "requests": requests,
        "frame_exact_match": _gold_frame_exact_match(traces),
        "total_latency_p95_ms": round(_percentile(_total_latencies_ms(traces), 95), 6),
        "promoted_generations": sum(1 for record in promotion_records if record.get("promoted")),
        "promotion_attempts": len(promotion_records),
        "promoted_with_layer_regression": sum(
            1 for record in promotion_records if record.get("promoted_with_layer_regression")
        ),
        "bottleneck_codes": ",".join(finding.code for finding in bottlenecks),
        "bottleneck_count": len(bottlenecks),
        "l1_benchmark_native_p95_us": "",
        "l1_benchmark_throughput_qps": "",
    }
    for layer in ["L0", "L1", "L2", "L3", "L4"]:
        row[f"{layer.lower()}_share"] = (
            round(layer_counts.get(layer, 0) / requests, 6) if requests else 0.0
        )
    if benchmark.get("status") == "success":
        row["l1_benchmark_native_p95_us"] = benchmark.get("native_p95_us", "")
        row["l1_benchmark_throughput_qps"] = benchmark.get("throughput_qps", "")
    return row


def _rank_experiment_comparison_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored_rows = []
    for row in rows:
        updated = dict(row)
        updated["comparison_score"] = round(_experiment_comparison_score(updated), 6)
        scored_rows.append(updated)
    scored_rows.sort(
        key=lambda row: (
            -float(row["comparison_score"]),
            str(row.get("experiment", "")),
            str(row.get("stream", "")),
        )
    )
    for rank, row in enumerate(scored_rows, start=1):
        row["comparison_rank"] = rank
    return scored_rows


def _experiment_comparison_score(row: dict[str, Any]) -> float:
    frame_exact = _float_or_zero(row.get("frame_exact_match"))
    latency_p95 = _float_or_zero(row.get("total_latency_p95_ms"))
    l4_share = _float_or_zero(row.get("l4_share"))
    bottleneck_count = _float_or_zero(row.get("bottleneck_count"))
    return 100.0 * frame_exact - 0.01 * latency_p95 - 10.0 * l4_share - 2.0 * bottleneck_count


def _float_or_zero(value: Any) -> float:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str) and value:
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _write_experiment_comparison_html(path: Path, *, rows: list[dict[str, Any]]) -> Path:
    table_rows = [
        "<tr>"
        + "".join(f"<th>{html.escape(field)}</th>" for field in EXPERIMENT_COMPARISON_FIELDNAMES)
        + "</tr>"
    ]
    for row in rows:
        table_rows.append(
            "<tr>"
            + "".join(
                f"<td>{html.escape(str(row.get(field, '')))}</td>"
                for field in EXPERIMENT_COMPARISON_FIELDNAMES
            )
            + "</tr>"
        )

    path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                '<html lang="en">',
                "<head>",
                '<meta charset="utf-8">',
                "<title>Darjeeling Experiment Comparison</title>",
                "<style>",
                "body{font-family:system-ui,sans-serif;margin:32px;line-height:1.4}",
                "table{border-collapse:collapse;margin-top:16px;font-size:13px}",
                "td,th{border:1px solid #ddd;padding:6px 8px;text-align:left}",
                "th{background:#f7f7f7}",
                "</style>",
                "</head>",
                "<body>",
                "<h1>Darjeeling Experiment Comparison</h1>",
                f"<p>Compared {len(rows)} run(s).</p>",
                "<h2>Bottleneck Summary</h2>",
                _experiment_bottleneck_summary_html(rows),
                "<h2>Ranked Runs</h2>",
                f"<table>{''.join(table_rows)}</table>",
                "</body>",
                "</html>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _experiment_bottleneck_summary_html(rows: list[dict[str, Any]]) -> str:
    counts: Counter[str] = Counter()
    for row in rows:
        codes = str(row.get("bottleneck_codes", ""))
        for code in codes.split(","):
            if code:
                counts[code] += 1
    if not counts:
        return "<p>No bottlenecks detected across compared runs.</p>"
    table_rows = ["<tr><th>bottleneck</th><th>runs</th></tr>"]
    for code, count in counts.most_common():
        table_rows.append(
            f"<tr><td>{html.escape(code)}</td><td>{html.escape(str(count))}</td></tr>"
        )
    return f"<table>{''.join(table_rows)}</table>"


def _gold_frame_exact_match(traces: list[TraceRecord]) -> float | str:
    labeled = [trace for trace in traces if trace.gold_frame is not None]
    if not labeled:
        return ""
    exact = sum(trace.final_frame == trace.gold_frame for trace in labeled)
    return round(exact / len(labeled), 6)


def _total_latencies_ms(traces: list[TraceRecord]) -> list[float]:
    return [sum(result.latency_ms for result in trace.layer_results) for trace in traces]


def _write_summary_md(
    path: Path,
    *,
    run_dir: Path,
    settings_text: str,
    current_manifest: ArtifactManifest | None,
    generation_manifests: list[ArtifactManifest],
    traces: list[TraceRecord],
    promotion_records: list[dict[str, Any]],
    bottlenecks: list[BottleneckFinding],
    l1_benchmark: dict[str, Any] | None,
    l3_benchmark: dict[str, Any] | None,
) -> Path:
    manifest_text = (
        json.dumps(current_manifest.model_dump(mode="json"), indent=2, sort_keys=True)
        if current_manifest is not None
        else "{}"
    )
    path.write_text(
        "# Run Summary\n\n"
        "## Report Artifacts\n\n"
        "- `metrics.csv`\n"
        "- `artifacts.csv`\n"
        "- `curves.html`\n"
        "- `hard_cases.jsonl`\n\n"
        f"{_layer_summary_section(traces)}\n\n"
        f"{_l2_unguarded_section(traces)}\n\n"
        f"{_l2_tuning_section(current_manifest)}\n\n"
        f"{_evolution_summary_section(promotion_records)}\n\n"
        f"{
            _artifact_summary_section(
                promotion_records,
                current_manifest,
                generation_manifests,
            )
        }\n\n"
        f"{_l1_report_section(run_dir, current_manifest, traces, l1_benchmark)}\n\n"
        "## Settings\n\n"
        "```json\n"
        f"{settings_text.strip()}\n"
        "```\n\n"
        "## Current Artifact Manifest\n\n"
        "```json\n"
        f"{manifest_text.strip()}\n"
        "```\n\n"
        f"{_promotion_report_section(run_dir)}\n\n"
        f"{_l3_report_section(run_dir, settings_text)}\n\n"
        f"{_l3_benchmark_section(l3_benchmark)}\n\n"
        f"{_failed_experiment_analysis_section(bottlenecks)}",
        encoding="utf-8",
    )
    return path


def _layer_summary_section(traces: list[TraceRecord]) -> str:
    if not traces:
        return "## Layer Summary\n\nNo traces found."

    rows = _layer_summary_rows(traces)
    lines = [
        "## Layer Summary",
        "",
        f"- requests: {len(traces)}",
        "",
        "| layer | coverage | accepted_accuracy | wrong_accept_rate | "
        "forced_global_accuracy | p50_ms | p95_ms | cost/100 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {layer} | {coverage} | {accepted_accuracy} | {wrong_accept_rate} | "
            "{forced_global_accuracy} | {p50_ms} | {p95_ms} | {cost_usd_per_100_requests} |".format(
                **{key: _format_layer_summary_value(key, row[key]) for key in row}
            )
        )
    gold_labeled = [trace for trace in traces if trace.gold_frame is not None]
    if gold_labeled:
        exact = sum(trace.final_frame == trace.gold_frame for trace in gold_labeled)
        lines.append("")
        lines.append(f"- gold frame exact match: {exact / len(gold_labeled):.3f}")
    return "\n".join(lines)


def _metrics_rows(
    run_dir: Path,
    traces: list[TraceRecord],
    promotion_records: list[dict[str, Any]],
    current_manifest: ArtifactManifest | None,
    generation_manifests: list[ArtifactManifest],
    bottlenecks: list[BottleneckFinding],
    l1_benchmark: dict[str, Any] | None,
    l3_benchmark: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.append(_metric_row("run", "", "", "requests", len(traces)))
    if current_manifest is not None:
        rows.append(
            _metric_row(
                "current_manifest",
                current_manifest.generation,
                "",
                "artifact_set_id",
                current_manifest.artifact_set_id,
            )
        )

    layer_counts = Counter(trace.chosen_layer for trace in traces)
    for layer in ["L0", "L1", "L2", "L3", "L4"]:
        count = layer_counts.get(layer, 0)
        share = count / len(traces) if traces else 0.0
        rows.append(_metric_row("run", "", layer, "chosen_count", count))
        rows.append(_metric_row("run", "", layer, "chosen_share", round(share, 6)))
    rows.extend(_layer_summary_metric_rows(traces))
    rows.extend(_evolution_summary_metric_rows(promotion_records))

    gold_labeled = [trace for trace in traces if trace.gold_frame is not None]
    if gold_labeled:
        exact = sum(trace.final_frame == trace.gold_frame for trace in gold_labeled)
        rows.append(
            _metric_row(
                "gold_eval",
                "",
                "",
                "frame_exact_match",
                round(exact / len(gold_labeled), 6),
            )
        )

    rows.extend(_latency_metric_rows(traces))
    rows.extend(_l2_unguarded_metric_rows(traces))
    rows.extend(_l1_metric_rows(traces))
    rows.extend(_l1_benchmark_metric_rows(l1_benchmark))
    rows.extend(
        _l1_generation_benchmark_metric_rows(run_dir, current_manifest, generation_manifests)
    )
    rows.extend(_l3_metric_rows(traces))
    rows.extend(_l3_benchmark_metric_rows(l3_benchmark))
    rows.extend(_promotion_metric_rows(promotion_records))
    rows.extend(_bottleneck_metric_rows(bottlenecks))
    return rows


def _layer_summary_rows(traces: list[TraceRecord]) -> list[dict[str, Any]]:
    request_count = len(traces)
    labeled_traces = [trace for trace in traces if _evaluation_label(trace) is not None]
    labeled_count = len(labeled_traces)
    rows: list[dict[str, Any]] = []
    for layer in ["L0", "L1", "L2", "L3", "L4"]:
        chosen_labeled = [trace for trace in labeled_traces if trace.chosen_layer == layer]
        chosen_correct = sum(
            trace.final_frame == _evaluation_label(trace) for trace in chosen_labeled
        )
        forced_labeled = [(trace, _last_layer_result(trace, layer)) for trace in labeled_traces]
        forced_correct = sum(
            result is not None and result.frame == _evaluation_label(trace)
            for trace, result in forced_labeled
        )
        latencies = [
            result.latency_ms
            for trace in traces
            for result in trace.layer_results
            if result.layer == layer
        ]
        cost_usd = sum(
            result.cost_usd
            for trace in traces
            for result in trace.layer_results
            if result.layer == layer
        )
        wrong_accept_count = len(chosen_labeled) - chosen_correct
        rows.append(
            {
                "layer": layer,
                "coverage": (
                    sum(1 for trace in traces if trace.chosen_layer == layer) / request_count
                    if request_count
                    else 0.0
                ),
                "accepted_accuracy": (
                    chosen_correct / len(chosen_labeled) if chosen_labeled else None
                ),
                "wrong_accept_rate": (
                    wrong_accept_count / labeled_count if labeled_count else None
                ),
                "forced_global_accuracy": (
                    forced_correct / labeled_count if labeled_count else None
                ),
                "p50_ms": _percentile(latencies, 50) if latencies else None,
                "p95_ms": _percentile(latencies, 95) if latencies else None,
                "cost_usd_per_100_requests": (
                    cost_usd / request_count * 100.0 if request_count else 0.0
                ),
            }
        )
    return rows


def _layer_summary_metric_rows(traces: list[TraceRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in _layer_summary_rows(traces):
        layer = str(summary["layer"])
        for metric in [
            "coverage",
            "accepted_accuracy",
            "wrong_accept_rate",
            "forced_global_accuracy",
            "p50_ms",
            "p95_ms",
            "cost_usd_per_100_requests",
        ]:
            value = summary[metric]
            rows.append(
                _metric_row(
                    "layer_summary",
                    "",
                    layer,
                    metric,
                    "" if value is None else round(float(value), 6),
                )
            )
    return rows


def _l2_unguarded_section(traces: list[TraceRecord]) -> str:
    stats = _l2_unguarded_stats(traces)
    lines = ["## L2 Unguarded Diagnostics", ""]
    if stats["evaluated"] == 0:
        lines.append("No L2 observations found.")
        return "\n".join(lines)
    lines.extend(
        [
            f"- evaluated: {stats['evaluated']}",
            f"- labeled: {stats['labeled']}",
            f"- runtime accepted: {stats['runtime_accepted']}",
            (
                "- threshold=0 accuracy: "
                f"{_format_layer_summary_value('unguarded_accuracy', stats['unguarded_accuracy'])}"
            ),
            (
                "- threshold=0 wrong prediction rate: "
                f"{
                    _format_layer_summary_value(
                        'unguarded_wrong_rate',
                        stats['unguarded_wrong_rate'],
                    )
                }"
            ),
            (
                "- L2 latency p50/p95: "
                f"{_format_layer_summary_value('p50_ms', stats['p50_ms'])}/"
                f"{_format_layer_summary_value('p95_ms', stats['p95_ms'])} ms"
            ),
            (
                "- intent support similarity p50/p95: "
                f"{
                    _format_layer_summary_value(
                        'predicted_intent_similarity_p50',
                        stats['predicted_intent_similarity_p50'],
                    )
                }/"
                f"{
                    _format_layer_summary_value(
                        'predicted_intent_similarity_p95',
                        stats['predicted_intent_similarity_p95'],
                    )
                }"
            ),
        ]
    )
    return "\n".join(lines)


def _l2_tuning_section(current_manifest: ArtifactManifest | None) -> str:
    lines = ["## L2 Tuning", ""]
    if current_manifest is None:
        lines.append("No current artifact manifest found.")
        return "\n".join(lines)
    tuning = current_manifest.candidate_metrics.get("l2_tuning")
    training_scope = current_manifest.candidate_metrics.get("l2_training_scope")
    if training_scope is not None:
        lines.append(f"- training scope: {training_scope}")
    if "l2_training_traces" in current_manifest.candidate_metrics:
        lines.append(
            "- teacher/lower-miss/target traces: "
            f"{current_manifest.candidate_metrics.get('l2_teacher_train_traces')}/"
            f"{current_manifest.candidate_metrics.get('l2_lower_miss_train_traces')}/"
            f"{current_manifest.candidate_metrics.get('l2_training_traces')}"
        )
    if not isinstance(tuning, dict):
        skipped_reason = current_manifest.candidate_metrics.get("l2_tuning_skipped_reason")
        if skipped_reason:
            lines.append(f"L2 tuning skipped: {skipped_reason}.")
        else:
            lines.append("No L2 tuning report recorded.")
        return "\n".join(lines)
    config = current_manifest.candidate_metrics.get("l2_config")
    best_metrics = tuning.get("best_metrics")
    guarded = best_metrics.get("guarded") if isinstance(best_metrics, dict) else None
    unguarded = best_metrics.get("unguarded") if isinstance(best_metrics, dict) else None
    lines.extend(
        [
            f"- trials completed/requested: {tuning.get('n_trials_completed')}/"
            f"{tuning.get('n_trials_requested')}",
            f"- train/validation size: {tuning.get('train_size')}/{tuning.get('validation_size')}",
            "- residual/objective validation size: "
            f"{tuning.get('validation_residual_size', 'n/a')}/"
            f"{tuning.get('objective_validation_size', 'n/a')} "
            f"({tuning.get('objective_validation_source', 'unknown')})",
            f"- split policy: {tuning.get('split_policy')}",
            f"- best trial/value: {tuning.get('best_trial_number')}/{tuning.get('best_value')}",
        ]
    )
    if isinstance(config, dict):
        lines.extend(
            [
                f"- selected frame source: {config.get('frame_source')}",
                f"- selected intent model: {config.get('intent_model_family')}",
                f"- selected slot model: {config.get('slot_model_family')}",
                f"- selected max_features: {config.get('max_features')}",
                f"- selected word ngram: {config.get('word_ngram_range')}",
                f"- selected char ngram: {config.get('char_ngram_range')}",
            ]
        )
    if isinstance(unguarded, dict):
        unguarded_accuracy = _format_layer_summary_value(
            "accepted_accuracy",
            unguarded.get("accepted_accuracy"),
        )
        lines.append(
            "- tuning validation unguarded accuracy: "
            f"{unguarded_accuracy}"
        )
    if isinstance(guarded, dict):
        lines.append(
            "- tuning validation guarded coverage/accuracy/wrong rate: "
            f"{_format_layer_summary_value('coverage', guarded.get('coverage'))}/"
            f"{_format_layer_summary_value('accepted_accuracy', guarded.get('accepted_accuracy'))}/"
            f"{_format_layer_summary_value('wrong_accept_rate', guarded.get('wrong_accept_rate'))}"
        )
    artifact_path = current_manifest.artifact_paths.get("l2_tuning")
    if artifact_path:
        lines.append(f"- tuning artifact: `{artifact_path}`")
    return "\n".join(lines)


def _l2_unguarded_metric_rows(traces: list[TraceRecord]) -> list[dict[str, Any]]:
    stats = _l2_unguarded_stats(traces)
    return [
        _metric_row("l2_unguarded", "", "L2", metric, value)
        for metric, value in stats.items()
    ]


def _l2_unguarded_stats(traces: list[TraceRecord]) -> dict[str, Any]:
    evaluated = 0
    labeled = 0
    correct = 0
    wrong = 0
    runtime_accepted = 0
    latencies: list[float] = []
    guard_probabilities: list[float] = []
    nearest_similarities: list[float] = []
    predicted_intent_similarities: list[float] = []
    intent_support_margins: list[float] = []
    invalid_slot_outputs = 0
    for trace in traces:
        expected = _evaluation_label(trace)
        for result in trace.layer_results:
            if result.layer != "L2":
                continue
            predicted = _l2_predicted_frame(result)
            if predicted is None:
                continue
            evaluated += 1
            runtime_accepted += int(result.accepted)
            latencies.append(result.latency_ms)
            if isinstance(result.confidence, int | float):
                guard_probabilities.append(float(result.confidence))
            metadata = result.metadata or {}
            if metadata.get("slot_invalid_bio") is True:
                invalid_slot_outputs += 1
            _append_numeric_metadata(
                nearest_similarities,
                metadata,
                "nearest_similarity",
            )
            _append_numeric_metadata(
                predicted_intent_similarities,
                metadata,
                "predicted_intent_similarity",
            )
            _append_numeric_metadata(
                intent_support_margins,
                metadata,
                "intent_support_margin",
            )
            if expected is None:
                continue
            labeled += 1
            if predicted == expected:
                correct += 1
            else:
                wrong += 1
    return {
        "evaluated": evaluated,
        "labeled": labeled,
        "correct": correct,
        "wrong": wrong,
        "runtime_accepted": runtime_accepted,
        "unguarded_accuracy": correct / labeled if labeled else None,
        "unguarded_wrong_rate": wrong / labeled if labeled else None,
        "runtime_accept_rate": runtime_accepted / evaluated if evaluated else 0.0,
        "p50_ms": _percentile(latencies, 50) if latencies else None,
        "p95_ms": _percentile(latencies, 95) if latencies else None,
        "guard_probability_p50": _percentile(guard_probabilities, 50)
        if guard_probabilities
        else None,
        "guard_probability_p95": _percentile(guard_probabilities, 95)
        if guard_probabilities
        else None,
        "nearest_similarity_p50": _percentile(nearest_similarities, 50)
        if nearest_similarities
        else None,
        "nearest_similarity_p95": _percentile(nearest_similarities, 95)
        if nearest_similarities
        else None,
        "predicted_intent_similarity_p50": _percentile(
            predicted_intent_similarities,
            50,
        )
        if predicted_intent_similarities
        else None,
        "predicted_intent_similarity_p95": _percentile(
            predicted_intent_similarities,
            95,
        )
        if predicted_intent_similarities
        else None,
        "intent_support_margin_p50": _percentile(intent_support_margins, 50)
        if intent_support_margins
        else None,
        "intent_support_margin_p95": _percentile(intent_support_margins, 95)
        if intent_support_margins
        else None,
        "slot_invalid_bio": invalid_slot_outputs,
    }


def _append_numeric_metadata(
    values: list[float],
    metadata: dict[str, Any],
    key: str,
) -> None:
    value = metadata.get(key)
    if isinstance(value, int | float) and not isinstance(value, bool):
        values.append(float(value))


def _l2_predicted_frame(result: Any) -> Frame | None:
    payload = result.metadata.get("predicted_frame") if result.metadata else None
    if payload is None and result.accepted:
        return result.frame
    if payload is None:
        return None
    try:
        return Frame.model_validate(payload)
    except ValueError:
        return None


def _evolution_summary_section(promotion_records: list[dict[str, Any]]) -> str:
    if not promotion_records:
        return "## Evolution Summary\n\nNo compiler generations found."

    lines = [
        "## Evolution Summary",
        "",
        "| generation | L4_calls/100 | cost/100 | p95_ms | frame_em | "
        "L0_share | L1_share | L2_share | L3_share | L4_share |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in _evolution_summary_rows(promotion_records):
        lines.append(
            "| {generation} | {l4_calls_per_100} | {cost_usd_per_100_requests} | "
            "{p95_latency_ms} | {frame_exact_match} | {L0_share} | {L1_share} | "
            "{L2_share} | {L3_share} | {L4_share} |".format(
                **{key: _format_markdown_cell(key, row.get(key)) for key in row}
            )
        )
    return "\n".join(lines)


def _evolution_summary_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: _sort_generation(item.get("generation"))):
        objective = record.get("candidate_objective") or {}
        candidate_metrics = record.get("candidate_metrics") or {}
        layer_counts = candidate_metrics.get("candidate_layer_counts") or {}
        eval_size = _positive_number(candidate_metrics.get("promotion_eval_size"))
        if eval_size is None:
            eval_size = sum(
                _positive_number(layer_counts.get(layer)) or 0.0
                for layer in ["L0", "L1", "L2", "L3", "L4"]
            )
        row: dict[str, Any] = {
            "generation": record.get("generation", ""),
            "l4_calls_per_100": _layer_calls_per_100(layer_counts, "L4", eval_size),
            "cost_usd_per_100_requests": _numeric_value(objective.get("cost_usd_per_100_requests")),
            "p95_latency_ms": _numeric_value(objective.get("p95_latency_ms")),
            "frame_exact_match": _numeric_value(objective.get("frame_exact_match")),
        }
        for layer in ["L0", "L1", "L2", "L3", "L4"]:
            row[f"{layer}_share"] = _layer_share(layer_counts, layer, eval_size)
        rows.append(row)
    return rows


def _evolution_summary_metric_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in _evolution_summary_rows(records):
        generation = summary["generation"]
        for metric in [
            "l4_calls_per_100",
            "cost_usd_per_100_requests",
            "p95_latency_ms",
            "frame_exact_match",
            "L0_share",
            "L1_share",
            "L2_share",
            "L3_share",
            "L4_share",
        ]:
            value = summary.get(metric)
            rows.append(
                _metric_row(
                    "evolution_summary",
                    generation,
                    "",
                    metric.lower(),
                    "" if value is None else round(float(value), 6),
                )
            )
    return rows


def _artifact_summary_section(
    promotion_records: list[dict[str, Any]],
    current_manifest: ArtifactManifest | None,
    generation_manifests: list[ArtifactManifest],
) -> str:
    rows = _artifact_summary_rows(promotion_records, current_manifest, generation_manifests)
    if not rows:
        return "## Artifact Summary\n\nNo artifact manifests found."

    lines = [
        "## Artifact Summary",
        "",
        "| artifact_id | type | generation | coverage_delta | accuracy_delta | "
        "cost_delta | promoted | reason |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {artifact_id} | {type} | {generation} | {coverage_delta} | "
            "{accuracy_delta} | {cost_delta} | {promoted} | {reason} |".format(
                **{key: _format_markdown_cell(key, row.get(key)) for key in row}
            )
        )
    return "\n".join(lines)


def _artifact_summary_rows(
    promotion_records: list[dict[str, Any]],
    current_manifest: ArtifactManifest | None,
    generation_manifests: list[ArtifactManifest],
) -> list[dict[str, Any]]:
    records_by_generation = {
        str(record.get("generation", "")): record for record in promotion_records
    }
    rows: list[dict[str, Any]] = []
    manifests = _deduped_manifests(current_manifest, generation_manifests)
    for manifest in manifests:
        record = records_by_generation.get(str(manifest.generation), {})
        promoted = bool(record.get("promoted", manifest.promoted))
        reason = str(record.get("promotion_reason") or manifest.promotion_reason)
        rows.append(
            {
                "artifact_id": manifest.artifact_set_id,
                "type": "artifact_set",
                "generation": manifest.generation,
                "coverage_delta": None,
                "accuracy_delta": _objective_delta(record, "frame_exact_match"),
                "cost_delta": _objective_delta(record, "cost_usd_per_100_requests"),
                "promoted": promoted,
                "reason": reason,
            }
        )
        for layer, delta in sorted(manifest.per_layer_deltas.items()):
            rows.append(
                {
                    "artifact_id": f"{manifest.artifact_set_id}:{layer}",
                    "type": layer,
                    "generation": manifest.generation,
                    "coverage_delta": _delta_value(delta, "coverage_delta"),
                    "accuracy_delta": _delta_value(delta, "accepted_accuracy_delta"),
                    "cost_delta": _delta_value(delta, "cost_delta"),
                    "promoted": promoted,
                    "reason": reason,
                }
            )

    if rows or not promotion_records:
        return rows

    for record in sorted(
        promotion_records,
        key=lambda item: _sort_generation(item.get("generation")),
    ):
        generation = record.get("generation", "")
        artifact_id = record.get("artifact_set_id") or f"gen_{generation}_candidate"
        rows.append(
            {
                "artifact_id": artifact_id,
                "type": "artifact_set",
                "generation": generation,
                "coverage_delta": None,
                "accuracy_delta": _objective_delta(record, "frame_exact_match"),
                "cost_delta": _objective_delta(record, "cost_usd_per_100_requests"),
                "promoted": bool(record.get("promoted", False)),
                "reason": str(record.get("promotion_reason", "")),
            }
        )
    return rows


def _evaluation_label(trace: TraceRecord) -> Frame | None:
    return trace.gold_frame or trace.teacher_frame


def _last_layer_result(trace: TraceRecord, layer: str) -> Any | None:
    for result in reversed(trace.layer_results):
        if result.layer == layer:
            return result
    return None


def _format_layer_summary_value(metric: str, value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if metric == "cost_usd_per_100_requests":
            return f"{value:.6f}"
        if abs(value) < 0.0000005:
            value = 0.0
        return f"{value:.6f}" if abs(value) < 0.001 and value != 0.0 else f"{value:.3f}"
    return str(value)


def _format_markdown_cell(metric: str, value: Any) -> str:
    text = _format_summary_value(metric, value)
    return text.replace("\n", " ").replace("|", "\\|")


def _format_summary_value(metric: str, value: Any) -> str:
    if value is None or value == "":
        return "n/a"
    if metric == "generation":
        return str(value)
    if isinstance(value, bool):
        return str(value)
    number = _numeric_value(value)
    if number is None:
        return str(value)
    if metric in {
        "cost_usd_per_100_requests",
        "cost_delta",
    }:
        return f"{number:.6f}"
    return f"{number:.3f}"


def _deduped_manifests(
    current_manifest: ArtifactManifest | None,
    generation_manifests: list[ArtifactManifest],
) -> list[ArtifactManifest]:
    manifests_by_generation = {manifest.generation: manifest for manifest in generation_manifests}
    if current_manifest is not None:
        manifests_by_generation[current_manifest.generation] = current_manifest
    return [manifests_by_generation[key] for key in sorted(manifests_by_generation)]


def _sort_generation(value: Any) -> tuple[int, str]:
    number = _numeric_value(value)
    if number is None:
        return (10**9, str(value))
    return (int(number), str(value))


def _positive_number(value: Any) -> float | None:
    number = _numeric_value(value)
    if number is None or number <= 0:
        return None
    return number


def _numeric_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str) and value:
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _layer_calls_per_100(layer_counts: Any, layer: str, total: float | None) -> float | None:
    if total is None or total <= 0:
        return None
    return (_layer_count(layer_counts, layer) / total) * 100.0


def _layer_share(layer_counts: Any, layer: str, total: float | None) -> float | None:
    if total is None or total <= 0:
        return None
    return _layer_count(layer_counts, layer) / total


def _layer_count(layer_counts: Any, layer: str) -> float:
    if not isinstance(layer_counts, dict):
        return 0.0
    return _numeric_value(layer_counts.get(layer)) or 0.0


def _objective_delta(record: dict[str, Any], metric: str) -> float | None:
    current = record.get("current_objective") or {}
    candidate = record.get("candidate_objective") or {}
    if not isinstance(current, dict) or not isinstance(candidate, dict):
        return None
    current_value = _numeric_value(current.get(metric))
    candidate_value = _numeric_value(candidate.get(metric))
    if current_value is None or candidate_value is None:
        return None
    return candidate_value - current_value


def _delta_value(delta: Any, name: str) -> float | None:
    if isinstance(delta, dict):
        return _numeric_value(delta.get(name))
    return _numeric_value(getattr(delta, name, None))


def _latency_metric_rows(traces: list[TraceRecord]) -> list[dict[str, Any]]:
    latencies_by_layer: dict[LayerName, list[float]] = defaultdict(list)
    for trace in traces:
        for result in trace.layer_results:
            latencies_by_layer[result.layer].append(result.latency_ms)

    rows: list[dict[str, Any]] = []
    for layer, latencies in sorted(latencies_by_layer.items()):
        rows.append(
            _metric_row("latency", "", layer, "p50_ms", round(_percentile(latencies, 50), 6))
        )
        rows.append(
            _metric_row("latency", "", layer, "p95_ms", round(_percentile(latencies, 95), 6))
        )
    return rows


def _l1_metric_rows(traces: list[TraceRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    native_latencies = _l1_native_latencies_us(traces)
    if native_latencies:
        rows.append(
            _metric_row(
                "l1_native",
                "",
                "L1",
                "native_latency_p50_us",
                round(_percentile(native_latencies, 50), 6),
            )
        )
        rows.append(
            _metric_row(
                "l1_native",
                "",
                "L1",
                "native_latency_p95_us",
                round(_percentile(native_latencies, 95), 6),
            )
        )
    for program_path, count in sorted(_l1_program_path_counts(traces).items()):
        rows.append(_metric_row("l1_program_path", "", "L1", program_path, count))
    return rows


def _l1_benchmark_metric_rows(l1_benchmark: dict[str, Any] | None) -> list[dict[str, Any]]:
    if l1_benchmark is None:
        return []
    rows = [_metric_row("l1_benchmark", "", "L1", "status", l1_benchmark.get("status", "unknown"))]
    if l1_benchmark.get("status") != "success":
        if error := l1_benchmark.get("error"):
            rows.append(_metric_row("l1_benchmark", "", "L1", "error", error))
        return rows

    for metric in [
        "requests",
        "accepted",
        "accepted_share",
        "integration_avg_ms",
        "integration_p50_ms",
        "integration_p95_ms",
        "native_avg_us",
        "native_p50_us",
        "native_p95_us",
        "native_max_us",
        "throughput_qps",
        "source_size_bytes",
        "binary_size_bytes",
    ]:
        value = l1_benchmark.get(metric)
        if isinstance(value, int | float | str):
            rows.append(_metric_row("l1_benchmark", "", "L1", metric, value))

    path_counts = l1_benchmark.get("program_path_counts")
    if isinstance(path_counts, dict):
        for program_path, count in sorted(path_counts.items()):
            rows.append(
                _metric_row(
                    "l1_benchmark_path",
                    "",
                    "L1",
                    str(program_path),
                    count,
                )
            )
    return rows


def _l1_generation_benchmark_metric_rows(
    run_dir: Path,
    current_manifest: ArtifactManifest | None,
    generation_manifests: list[ArtifactManifest],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest, payload in _l1_generation_benchmark_records(
        run_dir,
        current_manifest,
        generation_manifests,
    ):
        rows.append(
            _metric_row(
                "l1_generation_benchmark",
                manifest.generation,
                "L1",
                "status",
                payload.get("status", "unknown"),
            )
        )
        if payload.get("status") != "success":
            if error := payload.get("error"):
                rows.append(
                    _metric_row(
                        "l1_generation_benchmark",
                        manifest.generation,
                        "L1",
                        "error",
                        error,
                    )
                )
            continue
        for metric in [
            "requests",
            "accepted",
            "native_p95_us",
            "integration_p95_ms",
            "throughput_qps",
            "source_size_bytes",
            "binary_size_bytes",
        ]:
            value = payload.get(metric)
            if isinstance(value, int | float | str):
                rows.append(
                    _metric_row(
                        "l1_generation_benchmark",
                        manifest.generation,
                        "L1",
                        metric,
                        value,
                    )
                )
    return rows


def _l3_benchmark_metric_rows(l3_benchmark: dict[str, Any] | None) -> list[dict[str, Any]]:
    if l3_benchmark is None:
        return []
    rows = [_metric_row("l3_benchmark", "", "L3", "status", l3_benchmark.get("status", "unknown"))]
    if l3_benchmark.get("status") != "success":
        if error := l3_benchmark.get("error"):
            rows.append(_metric_row("l3_benchmark", "", "L3", "error", error))
        return rows

    for metric in [
        "requests",
        "accepted",
        "would_accept",
        "failures",
        "parse_failures",
        "repair_count",
        "generation_avg_ms",
        "generation_p50_ms",
        "generation_p95_ms",
        "confidence_avg",
        "confidence_p50",
        "confidence_p95",
        "throughput_qps",
        "duration_ms",
    ]:
        value = l3_benchmark.get(metric)
        if isinstance(value, int | float | str):
            rows.append(_metric_row("l3_benchmark", "", "L3", metric, value))
    backend = l3_benchmark.get("backend")
    if isinstance(backend, dict):
        for metric in ["model_name", "device_policy", "actual_device", "load_time_ms", "loaded"]:
            value = backend.get(metric)
            if isinstance(value, int | float | str | bool):
                rows.append(_metric_row("l3_benchmark_backend", "", "L3", metric, value))
    return rows


def _l3_metric_rows(traces: list[TraceRecord]) -> list[dict[str, Any]]:
    stats = _l3_observation_stats(traces)
    total = int(stats["trace_results"])
    if total == 0:
        return []

    rows = [
        _metric_row("l3_observation", "", "L3", "trace_results", total),
        _metric_row("l3_observation", "", "L3", "chosen_as_final", stats["chosen_l3"]),
        _metric_row(
            "l3_observation",
            "",
            "L3",
            "failure_rate",
            round(float(stats["failures"]) / total, 6),
        ),
        _metric_row(
            "l3_observation",
            "",
            "L3",
            "parse_failure_rate",
            round(float(stats["parse_failures"]) / total, 6),
        ),
        _metric_row(
            "l3_observation",
            "",
            "L3",
            "repair_rate",
            round(float(stats["repair_count"]) / total, 6),
        ),
        _metric_row(
            "l3_observation",
            "",
            "L3",
            "would_accept_count",
            stats["would_accept_count"],
        ),
    ]
    if stats["would_accept_labeled"]:
        rows.append(
            _metric_row(
                "l3_observation",
                "",
                "L3",
                "would_accept_accuracy",
                round(float(stats["would_accept_correct"]) / stats["would_accept_labeled"], 6),
            )
        )
    if stats["accepted_labeled"]:
        rows.append(
            _metric_row(
                "l3_observation",
                "",
                "L3",
                "guarded_accepted_accuracy",
                round(float(stats["accepted_correct"]) / stats["accepted_labeled"], 6),
            )
        )
    latencies = stats["latencies_ms"]
    if isinstance(latencies, list) and latencies:
        rows.append(
            _metric_row(
                "l3_observation",
                "",
                "L3",
                "generation_p50_ms",
                round(_percentile(latencies, 50), 6),
            )
        )
        rows.append(
            _metric_row(
                "l3_observation",
                "",
                "L3",
                "generation_p95_ms",
                round(_percentile(latencies, 95), 6),
            )
        )
    load_times = stats["load_times_ms"]
    if isinstance(load_times, list) and load_times:
        rows.append(
            _metric_row(
                "l3_observation",
                "",
                "L3",
                "model_load_time_p95_ms",
                round(_percentile(load_times, 95), 6),
            )
        )
    calibration = calibrate_l3_confidence_threshold(traces)
    if calibration is not None:
        rows.extend(
            [
                _metric_row(
                    "l3_guard_calibration",
                    "",
                    "L3",
                    "recommended_threshold",
                    round(calibration.threshold, 6),
                ),
                _metric_row(
                    "l3_guard_calibration",
                    "",
                    "L3",
                    "accepted_count",
                    calibration.accepted_count,
                ),
                _metric_row(
                    "l3_guard_calibration",
                    "",
                    "L3",
                    "wrong_accept_rate",
                    round(calibration.wrong_accept_rate, 6),
                ),
                _metric_row(
                    "l3_guard_calibration",
                    "",
                    "L3",
                    "coverage",
                    round(calibration.coverage, 6),
                ),
            ]
        )
    return rows


def _promotion_metric_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        generation = record.get("generation", "")
        rows.append(_metric_row("promotion", generation, "", "promoted", record.get("promoted")))
        rows.append(
            _metric_row(
                "promotion",
                generation,
                "",
                "promoted_with_layer_regression",
                record.get("promoted_with_layer_regression", False),
            )
        )
        for scope in ["current_objective", "candidate_objective"]:
            objective = record.get(scope) or {}
            for metric, value in sorted(objective.items()):
                rows.append(_metric_row(scope, generation, "", metric, value))
        for layer, delta in sorted((record.get("per_layer_deltas") or {}).items()):
            for metric, value in sorted(delta.items()):
                rows.append(_metric_row("layer_delta", generation, layer, metric, value))
    return rows


def _bottleneck_metric_rows(bottlenecks: list[BottleneckFinding]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for finding in bottlenecks:
        rows.append(_metric_row("bottleneck", "", "", finding.code, finding.evidence))
        rows.append(
            _metric_row(
                "bottleneck",
                "",
                "",
                f"{finding.code}.severity",
                finding.severity,
            )
        )
    return rows


def _artifact_rows(
    current_manifest: ArtifactManifest | None,
    generation_manifests: list[ArtifactManifest],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest in generation_manifests:
        rows.extend(_manifest_artifact_rows(manifest))
    if current_manifest is not None and current_manifest.generation not in {
        manifest.generation for manifest in generation_manifests
    }:
        rows.extend(_manifest_artifact_rows(current_manifest))
    return rows


def _manifest_artifact_rows(manifest: ArtifactManifest) -> list[dict[str, Any]]:
    return [
        {
            "generation": manifest.generation,
            "artifact_set_id": manifest.artifact_set_id,
            "promoted": manifest.promoted,
            "artifact_name": name,
            "artifact_path": artifact_path,
        }
        for name, artifact_path in sorted(manifest.artifact_paths.items())
    ]


def _write_report_hard_cases(
    path: Path,
    *,
    run_dir: Path,
    current_manifest: ArtifactManifest | None,
    generation_manifests: list[ArtifactManifest],
) -> Path:
    manifest = _latest_manifest_with_artifact(
        "hard_buffer",
        current_manifest=current_manifest,
        generation_manifests=generation_manifests,
    )
    if manifest is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        return path
    hard_buffer_path = _manifest_artifact_path(run_dir, manifest, "hard_buffer")
    hard_cases = load_hard_buffer_jsonl(hard_buffer_path) if hard_buffer_path is not None else []
    return write_hard_buffer_jsonl(path, hard_cases)


def _latest_manifest_with_artifact(
    artifact_name: str,
    *,
    current_manifest: ArtifactManifest | None,
    generation_manifests: list[ArtifactManifest],
) -> ArtifactManifest | None:
    manifests_by_generation = {manifest.generation: manifest for manifest in generation_manifests}
    if current_manifest is not None:
        manifests_by_generation[current_manifest.generation] = current_manifest
    candidates = [
        manifest
        for manifest in manifests_by_generation.values()
        if artifact_name in manifest.artifact_paths
    ]
    return max(candidates, key=lambda manifest: manifest.generation) if candidates else None


def _write_curves_html(
    path: Path,
    *,
    traces: list[TraceRecord],
    promotion_records: list[dict[str, Any]],
    current_manifest: ArtifactManifest | None,
    run_dir: Path,
    bottlenecks: list[BottleneckFinding],
    l1_benchmark: dict[str, Any] | None,
    l3_benchmark: dict[str, Any] | None,
) -> Path:
    path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                '<html lang="en">',
                "<head>",
                '<meta charset="utf-8">',
                "<title>Darjeeling Run Curves</title>",
                "<style>",
                "body{font-family:system-ui,sans-serif;margin:32px;line-height:1.4}",
                "table{border-collapse:collapse;margin-top:16px}",
                "td,th{border:1px solid #ddd;padding:6px 8px;text-align:left}",
                "svg{max-width:100%;height:auto;border:1px solid #ddd;background:#fff}",
                ".legend{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0}",
                ".swatch{display:inline-block;width:10px;height:10px;margin-right:4px}",
                "</style>",
                "</head>",
                "<body>",
                "<h1>Darjeeling Run Curves</h1>",
                "<h2>Cumulative Layer Share</h2>",
                _layer_share_svg(traces),
                _legend_html(),
                "<h2>L1 Program Paths</h2>",
                _l1_program_path_table_html(traces),
                "<h2>L1 Native Latency</h2>",
                _l1_native_latency_html(traces),
                "<h2>L1 Native Benchmark</h2>",
                _l1_benchmark_html(l1_benchmark),
                "<h2>L1 Benchmark By Generation</h2>",
                _l1_generation_benchmark_html(run_dir, current_manifest),
                "<h2>L3 Hardware Benchmark</h2>",
                _l3_benchmark_html(l3_benchmark),
                "<h2>L1 Artifact Summary</h2>",
                _l1_artifact_html(run_dir, current_manifest),
                "<h2>Failed Experiment Analysis</h2>",
                _failed_experiment_analysis_html(bottlenecks),
                "<h2>Promotion Records</h2>",
                _promotion_table_html(promotion_records),
                "</body>",
                "</html>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _layer_share_svg(traces: list[TraceRecord]) -> str:
    if not traces:
        return "<p>No traces found.</p>"

    width = 900
    height = 320
    pad = 40
    colors = {
        "L0": "#1f77b4",
        "L1": "#2ca02c",
        "L2": "#ff7f0e",
        "L3": "#9467bd",
        "L4": "#d62728",
    }
    cumulative: Counter[str] = Counter()
    points: dict[str, list[tuple[float, float]]] = {layer: [] for layer in colors}
    total = len(traces)
    for index, trace in enumerate(traces, start=1):
        cumulative[trace.chosen_layer] += 1
        x = pad + (width - 2 * pad) * (index - 1) / max(total - 1, 1)
        for layer in colors:
            share = cumulative[layer] / index
            y = height - pad - (height - 2 * pad) * share
            points[layer].append((x, y))

    polylines = []
    for layer, layer_points in points.items():
        joined = " ".join(f"{x:.2f},{y:.2f}" for x, y in layer_points)
        polylines.append(
            f'<polyline fill="none" stroke="{colors[layer]}" stroke-width="2" points="{joined}" />'
        )
    return "\n".join(
        [
            f'<svg viewBox="0 0 {width} {height}" role="img">',
            f'<line x1="{pad}" y1="{height - pad}" x2="{width - pad}" '
            f'y2="{height - pad}" stroke="#999" />',
            f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" stroke="#999" />',
            *polylines,
            "</svg>",
        ]
    )


def _legend_html() -> str:
    colors = {
        "L0": "#1f77b4",
        "L1": "#2ca02c",
        "L2": "#ff7f0e",
        "L3": "#9467bd",
        "L4": "#d62728",
    }
    items = [
        f'<span><span class="swatch" style="background:{color}"></span>{layer}</span>'
        for layer, color in colors.items()
    ]
    return f'<div class="legend">{"".join(items)}</div>'


def _promotion_table_html(records: list[dict[str, Any]]) -> str:
    if not records:
        return "<p>No promotion records found.</p>"
    rows = [
        "<tr><th>generation</th><th>promoted</th><th>reason</th>"
        "<th>candidate frame exact</th><th>wrong accept</th></tr>"
    ]
    for record in records:
        objective = record.get("candidate_objective") or {}
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(record.get('generation', '')))}</td>"
            f"<td>{html.escape(str(record.get('promoted', '')))}</td>"
            f"<td>{html.escape(str(record.get('promotion_reason', '')))}</td>"
            f"<td>{html.escape(str(objective.get('frame_exact_match', '')))}</td>"
            f"<td>{html.escape(str(objective.get('wrong_accept_rate', '')))}</td>"
            "</tr>"
        )
    return f"<table>{''.join(rows)}</table>"


def _l1_report_section(
    run_dir: Path,
    current_manifest: ArtifactManifest | None,
    traces: list[TraceRecord],
    l1_benchmark: dict[str, Any] | None,
) -> str:
    lines = ["## L1 Rust ProgramBank", ""]
    program_counts = _l1_program_path_counts(traces)
    native_latencies = _l1_native_latencies_us(traces)
    l1_results = [
        result for trace in traces for result in trace.layer_results if result.layer == "L1"
    ]
    accepted = sum(1 for result in l1_results if result.accepted)

    lines.append(f"- trace results: {len(l1_results)}")
    lines.append(f"- accepted: {accepted}")
    if native_latencies:
        lines.append(f"- native p50 latency: {_percentile(native_latencies, 50):.1f} us")
        lines.append(f"- native p95 latency: {_percentile(native_latencies, 95):.1f} us")
    else:
        lines.append("- native latency: no accepted L1 native latency recorded")

    if program_counts:
        lines.append("")
        lines.append("Top program paths:")
        for program_path, count in program_counts.most_common(10):
            lines.append(f"- `{program_path}`: {count}")
    else:
        lines.append("")
        lines.append("No L1 program paths recorded.")

    artifact_lines = _l1_artifact_summary_lines(run_dir, current_manifest)
    if artifact_lines:
        lines.extend(["", *artifact_lines])
    lines.extend(["", *_l1_benchmark_summary_lines(l1_benchmark)])
    return "\n".join(lines)


def _ensure_l1_benchmark(
    report_dir: Path,
    *,
    run_dir: Path,
    settings_text: str,
    current_manifest: ArtifactManifest | None,
    traces: list[TraceRecord],
) -> dict[str, Any] | None:
    benchmark_path = report_dir / L1_BENCHMARK_FILENAME
    if benchmark_path.exists():
        payload = _load_json_object(benchmark_path)
        return payload if payload else None

    try:
        payload = _run_l1_report_benchmark(
            run_dir=run_dir,
            settings_text=settings_text,
            current_manifest=current_manifest,
            traces=traces,
        )
    except L1BenchmarkUnavailable:
        return None
    except Exception as exc:
        payload = {
            "schema_version": "l1-benchmark-v1",
            "status": "error",
            "error": str(exc),
        }

    benchmark_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _run_l1_report_benchmark(
    *,
    run_dir: Path,
    settings_text: str,
    current_manifest: ArtifactManifest | None,
    traces: list[TraceRecord],
) -> dict[str, Any]:
    crate_dir, configured_binary = _l1_benchmark_target(
        run_dir=run_dir,
        settings_text=settings_text,
        current_manifest=current_manifest,
    )
    if crate_dir is None and configured_binary is None:
        raise L1BenchmarkUnavailable("no L1 crate or binary configured for this run")

    binary_path = configured_binary
    if binary_path is None:
        if crate_dir is None:
            raise L1BenchmarkUnavailable("no L1 crate configured for this run")
        binary_path = binary_path_for(crate_dir, release=False)
        if not binary_path.exists():
            binary_path = build_l1_binary(crate_dir, release=False)
    elif not binary_path.exists():
        raise FileNotFoundError(f"L1 binary does not exist: {binary_path}")

    utterances, corpus = _l1_benchmark_utterances(traces)
    started_at = perf_counter()
    metrics = benchmark_worker(
        binary_path, utterances, timeout_s=_l1_benchmark_timeout(settings_text)
    )
    duration_ms = (perf_counter() - started_at) * 1000.0

    return {
        "schema_version": "l1-benchmark-v1",
        "status": "success",
        "corpus": corpus,
        "crate_dir": str(crate_dir) if crate_dir is not None else "",
        "binary_path": str(binary_path),
        "source_size_bytes": _l1_source_size_bytes(crate_dir) if crate_dir is not None else 0,
        "binary_size_bytes": binary_path.stat().st_size,
        "duration_ms": duration_ms,
        **metrics,
    }


def _l1_benchmark_target(
    *,
    run_dir: Path,
    settings_text: str,
    current_manifest: ArtifactManifest | None,
) -> tuple[Path | None, Path | None]:
    if current_manifest is not None:
        crate_dir = _manifest_artifact_path(run_dir, current_manifest, "l1_crate_dir")
        if crate_dir is not None:
            return crate_dir, None

    settings_payload = _settings_payload(settings_text)
    binary_path = _path_setting(settings_payload, "l1_rust_binary")
    crate_dir = _path_setting(settings_payload, "l1_rust_crate_dir")
    return crate_dir, binary_path


def _l1_benchmark_timeout(settings_text: str) -> float:
    timeout = _settings_payload(settings_text).get("l1_worker_timeout_s", 2.0)
    return float(timeout) if isinstance(timeout, int | float) else 2.0


def _l1_benchmark_utterances(traces: list[TraceRecord]) -> tuple[list[str], str]:
    utterances: list[str] = []
    seen: set[str] = set()
    for trace in traces:
        normalized = normalize_utterance(trace.utterance)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        utterances.append(trace.utterance)
        if len(utterances) >= 128:
            break
    if utterances:
        return utterances, "trace_unique_utterances"
    return list(DEFAULT_BENCHMARK_UTTERANCES), "default_smoke"


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


def _path_setting(settings_payload: dict[str, Any], name: str) -> Path | None:
    value = settings_payload.get(name)
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        return None
    return Path(value).expanduser()


def _l1_benchmark_summary_lines(l1_benchmark: dict[str, Any] | None) -> list[str]:
    lines = ["L1 independent benchmark:"]
    if l1_benchmark is None:
        lines.append("- not configured for this report")
        return lines
    if l1_benchmark.get("status") != "success":
        lines.append(f"- status: `{l1_benchmark.get('status', 'unknown')}`")
        if error := l1_benchmark.get("error"):
            lines.append(f"- error: `{error}`")
        return lines

    lines.extend(
        [
            f"- corpus: `{l1_benchmark.get('corpus', 'unknown')}`",
            f"- requests: {l1_benchmark.get('requests', 0)}",
            f"- accepted share: {float(l1_benchmark.get('accepted_share', 0.0)):.3f}",
            f"- native p50 latency: {float(l1_benchmark.get('native_p50_us', 0.0)):.1f} us",
            f"- native p95 latency: {float(l1_benchmark.get('native_p95_us', 0.0)):.1f} us",
            f"- integration p95 latency: "
            f"{float(l1_benchmark.get('integration_p95_ms', 0.0)):.3f} ms",
            f"- throughput: {float(l1_benchmark.get('throughput_qps', 0.0)):.1f} qps",
            f"- source size: {l1_benchmark.get('source_size_bytes', 0)} bytes",
            f"- binary size: {l1_benchmark.get('binary_size_bytes', 0)} bytes",
        ]
    )
    path_counts = l1_benchmark.get("program_path_counts")
    if isinstance(path_counts, dict) and path_counts:
        lines.append("- benchmark path coverage:")
        for program_path, count in sorted(path_counts.items(), key=lambda item: str(item[0]))[:10]:
            lines.append(f"  - `{program_path}`: {count}")
    return lines


def _l1_artifact_summary_lines(
    run_dir: Path,
    current_manifest: ArtifactManifest | None,
) -> list[str]:
    if current_manifest is None:
        return []
    lines: list[str] = []
    if l1_crate_dir := current_manifest.artifact_paths.get("l1_crate_dir"):
        lines.append(f"- promoted L1 crate: `{l1_crate_dir}`")
    if diff_text := _read_manifest_artifact_excerpt(
        run_dir,
        current_manifest,
        "l1_agent_diff",
        max_lines=60,
    ):
        lines.extend(["", "L1 coding-agent diff excerpt:", "", "```diff", diff_text, "```"])
    elif source_text := _read_l1_source_excerpt(run_dir, current_manifest):
        lines.extend(["", "L1 source excerpt:", "", "```rust", source_text, "```"])
    return lines


def _read_manifest_artifact_excerpt(
    run_dir: Path,
    manifest: ArtifactManifest,
    artifact_name: str,
    *,
    max_lines: int,
) -> str:
    artifact_path = _manifest_artifact_path(run_dir, manifest, artifact_name)
    if artifact_path is None or not artifact_path.exists():
        return ""
    return "\n".join(artifact_path.read_text(encoding="utf-8").splitlines()[:max_lines])


def _read_l1_source_excerpt(run_dir: Path, manifest: ArtifactManifest) -> str:
    crate_dir = _manifest_artifact_path(run_dir, manifest, "l1_crate_dir")
    if crate_dir is None or not crate_dir.exists() or not crate_dir.is_dir():
        return ""
    source_paths = sorted((crate_dir / "src").rglob("*.rs"))
    preferred_paths = [crate_dir / "src" / "lib.rs", *source_paths]
    for source_path in preferred_paths:
        if source_path.exists() and source_path.is_file():
            return "\n".join(source_path.read_text(encoding="utf-8").splitlines()[:80])
    return ""


def _manifest_artifact_path(
    run_dir: Path,
    manifest: ArtifactManifest,
    artifact_name: str,
) -> Path | None:
    artifact_path_text = manifest.artifact_paths.get(artifact_name)
    if not artifact_path_text:
        return None
    artifact_path = Path(artifact_path_text)
    if artifact_path.is_absolute():
        return artifact_path
    return run_dir / "artifacts" / artifact_path


def _l1_program_path_counts(traces: list[TraceRecord]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for trace in traces:
        for result in trace.layer_results:
            if result.layer != "L1" or not result.metadata:
                continue
            program_path = result.metadata.get("program_path")
            if isinstance(program_path, str) and program_path:
                counts[program_path] += 1
    return counts


def _l1_native_latencies_us(traces: list[TraceRecord]) -> list[float]:
    latencies: list[float] = []
    for trace in traces:
        for result in trace.layer_results:
            if result.layer != "L1" or not result.metadata:
                continue
            value = result.metadata.get("native_latency_us")
            if isinstance(value, int | float):
                latencies.append(float(value))
    return latencies


def _l1_program_path_table_html(traces: list[TraceRecord]) -> str:
    counts = _l1_program_path_counts(traces)
    if not counts:
        return "<p>No L1 program paths recorded.</p>"
    rows = ["<tr><th>program path</th><th>accepted/observed count</th></tr>"]
    for program_path, count in counts.most_common(20):
        rows.append(
            f"<tr><td>{html.escape(program_path)}</td><td>{html.escape(str(count))}</td></tr>"
        )
    return f"<table>{''.join(rows)}</table>"


def _l1_native_latency_html(traces: list[TraceRecord]) -> str:
    native_latencies = _l1_native_latencies_us(traces)
    if not native_latencies:
        return "<p>No L1 native latency samples recorded.</p>"
    p50 = _percentile(native_latencies, 50)
    p95 = _percentile(native_latencies, 95)
    return (
        "<table>"
        "<tr><th>metric</th><th>microseconds</th></tr>"
        f"<tr><td>p50</td><td>{p50:.1f}</td></tr>"
        f"<tr><td>p95</td><td>{p95:.1f}</td></tr>"
        "</table>"
    )


def _l1_benchmark_html(l1_benchmark: dict[str, Any] | None) -> str:
    if l1_benchmark is None:
        return "<p>No L1 benchmark configured for this report.</p>"
    if l1_benchmark.get("status") != "success":
        return (
            "<p>L1 benchmark did not complete: "
            f"{html.escape(str(l1_benchmark.get('error', 'unknown error')))}</p>"
        )
    rows = ["<tr><th>metric</th><th>value</th></tr>"]
    for metric in [
        "corpus",
        "requests",
        "accepted",
        "accepted_share",
        "native_p50_us",
        "native_p95_us",
        "integration_p50_ms",
        "integration_p95_ms",
        "throughput_qps",
        "source_size_bytes",
        "binary_size_bytes",
    ]:
        rows.append(
            "<tr>"
            f"<td>{html.escape(metric)}</td>"
            f"<td>{html.escape(str(l1_benchmark.get(metric, '')))}</td>"
            "</tr>"
        )
    path_counts = l1_benchmark.get("program_path_counts")
    if isinstance(path_counts, dict):
        for program_path, count in sorted(path_counts.items()):
            rows.append(
                "<tr>"
                f"<td>path:{html.escape(str(program_path))}</td>"
                f"<td>{html.escape(str(count))}</td>"
                "</tr>"
            )
    return f"<table>{''.join(rows)}</table>"


def _l1_generation_benchmark_html(
    run_dir: Path,
    current_manifest: ArtifactManifest | None,
) -> str:
    records = _l1_generation_benchmark_records(
        run_dir,
        current_manifest,
        _load_generation_manifests(run_dir),
    )
    if not records:
        return "<p>No generation-scoped L1 benchmark artifacts found.</p>"
    rows = [
        "<tr><th>generation</th><th>status</th><th>native p95 us</th>"
        "<th>integration p95 ms</th><th>throughput qps</th><th>error</th></tr>"
    ]
    for manifest, payload in records:
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(manifest.generation))}</td>"
            f"<td>{html.escape(str(payload.get('status', 'unknown')))}</td>"
            f"<td>{html.escape(str(payload.get('native_p95_us', '')))}</td>"
            f"<td>{html.escape(str(payload.get('integration_p95_ms', '')))}</td>"
            f"<td>{html.escape(str(payload.get('throughput_qps', '')))}</td>"
            f"<td>{html.escape(str(payload.get('error', '')))}</td>"
            "</tr>"
        )
    return f"<table>{''.join(rows)}</table>"


def _l1_generation_benchmark_records(
    run_dir: Path,
    current_manifest: ArtifactManifest | None,
    generation_manifests: list[ArtifactManifest],
) -> list[tuple[ArtifactManifest, dict[str, Any]]]:
    manifests_by_generation = {manifest.generation: manifest for manifest in generation_manifests}
    if current_manifest is not None:
        manifests_by_generation[current_manifest.generation] = current_manifest

    records: list[tuple[ArtifactManifest, dict[str, Any]]] = []
    for manifest in [manifests_by_generation[key] for key in sorted(manifests_by_generation)]:
        path = _manifest_artifact_path(run_dir, manifest, "l1_benchmark")
        if path is None:
            continue
        payload = _load_json_object(path)
        if payload:
            records.append((manifest, payload))
    return records


def _l3_benchmark_section(l3_benchmark: dict[str, Any] | None) -> str:
    lines = ["## L3 Hardware Benchmark", ""]
    if l3_benchmark is None:
        lines.append(
            "No `reports/l3_benchmark.json` found. Run `edge-mvp l3 bench --out ...` to record one."
        )
        return "\n".join(lines)
    if l3_benchmark.get("status") != "success":
        lines.append(f"- status: `{l3_benchmark.get('status', 'unknown')}`")
        if error := l3_benchmark.get("error"):
            lines.append(f"- error: `{error}`")
        return "\n".join(lines)

    backend = l3_benchmark.get("backend") if isinstance(l3_benchmark.get("backend"), dict) else {}
    lines.extend(
        [
            f"- requests: {l3_benchmark.get('requests', 0)}",
            f"- accepted: {l3_benchmark.get('accepted', 0)}",
            f"- would accept: {l3_benchmark.get('would_accept', 0)}",
            f"- failures: {l3_benchmark.get('failures', 0)}",
            f"- parse failures: {l3_benchmark.get('parse_failures', 0)}",
            f"- generation p50/p95: "
            f"{float(l3_benchmark.get('generation_p50_ms', 0.0)):.3f}/"
            f"{float(l3_benchmark.get('generation_p95_ms', 0.0)):.3f} ms",
            f"- throughput: {float(l3_benchmark.get('throughput_qps', 0.0)):.3f} qps",
            f"- backend actual device: `{backend.get('actual_device', 'unknown')}`",
            f"- backend load time: `{backend.get('load_time_ms', 'unknown')}`",
        ]
    )
    return "\n".join(lines)


def _l3_benchmark_html(l3_benchmark: dict[str, Any] | None) -> str:
    if l3_benchmark is None:
        return "<p>No L3 hardware benchmark artifact found.</p>"
    if l3_benchmark.get("status") != "success":
        return (
            "<p>L3 hardware benchmark did not complete: "
            f"{html.escape(str(l3_benchmark.get('error', 'unknown error')))}</p>"
        )

    rows = ["<tr><th>metric</th><th>value</th></tr>"]
    for metric in [
        "requests",
        "accepted",
        "would_accept",
        "failures",
        "parse_failures",
        "repair_count",
        "generation_p50_ms",
        "generation_p95_ms",
        "confidence_p50",
        "confidence_p95",
        "throughput_qps",
    ]:
        rows.append(
            "<tr>"
            f"<td>{html.escape(metric)}</td>"
            f"<td>{html.escape(str(l3_benchmark.get(metric, '')))}</td>"
            "</tr>"
        )
    backend = l3_benchmark.get("backend")
    if isinstance(backend, dict):
        for metric, value in sorted(backend.items()):
            rows.append(
                "<tr>"
                f"<td>backend:{html.escape(str(metric))}</td>"
                f"<td>{html.escape(str(value))}</td>"
                "</tr>"
            )
    return f"<table>{''.join(rows)}</table>"


def _l1_artifact_html(run_dir: Path, current_manifest: ArtifactManifest | None) -> str:
    if current_manifest is None:
        return "<p>No current artifact manifest.</p>"
    parts = _l1_artifact_summary_lines(run_dir, current_manifest)
    if not parts:
        return "<p>No L1 artifact summary available.</p>"
    escaped = html.escape("\n".join(parts))
    return f"<pre>{escaped}</pre>"


def _promotion_report_section(run_dir: Path) -> str:
    records = _load_promotion_records(run_dir)
    if not records:
        return "## Promotion\n\nNo promotion records found."

    lines = ["## Promotion", ""]
    for record in records[-5:]:
        candidate_objective = record.get("candidate_objective") or {}
        current_objective = record.get("current_objective") or {}
        lines.extend(
            [
                (
                    f"- gen {record.get('generation')}: "
                    f"promoted={record.get('promoted')} "
                    f"reason=`{record.get('promotion_reason', '')}`"
                ),
                (
                    "  "
                    f"frame_exact={candidate_objective.get('frame_exact_match', 'n/a')} "
                    f"wrong_accept={candidate_objective.get('wrong_accept_rate', 'n/a')} "
                    f"current_frame_exact={current_objective.get('frame_exact_match', 'n/a')}"
                ),
                (
                    "  "
                    f"layer_regression={record.get('promoted_with_layer_regression', False)} "
                    f"regressed_layers={record.get('regressed_layers', [])}"
                ),
            ]
        )
    return "\n".join(lines)


def _failed_experiment_bottlenecks(
    *,
    traces: list[TraceRecord],
    promotion_records: list[dict[str, Any]],
    settings_text: str,
) -> list[BottleneckFinding]:
    findings: list[BottleneckFinding] = []
    request_count = len(traces)
    layer_counts = Counter(trace.chosen_layer for trace in traces)
    l4_share = layer_counts["L4"] / request_count if request_count else 0.0
    l1_accepts = _accepted_layer_count(traces, "L1")
    l2_results = _layer_results(traces, "L2")
    l3_results = _layer_results(traces, "L3")

    repeat_stats = _workload_repeat_stats(traces)
    if request_count >= 10 and repeat_stats["repeat_rate"] < 0.05:
        findings.append(
            BottleneckFinding(
                code="insufficient_workload_locality",
                label="insufficient workload locality",
                evidence=(
                    f"exact repeat rate is {repeat_stats['repeat_rate']:.3f}; "
                    f"{repeat_stats['unique_utterances']} unique utterances across "
                    f"{request_count} requests"
                ),
            )
        )

    if request_count and l4_share >= 0.50 and (l1_accepts / request_count) < 0.05:
        findings.append(
            BottleneckFinding(
                code="weak_l1_rule_coverage",
                label="weak L1 rule coverage",
                evidence=(
                    f"L1 accepted {l1_accepts}/{request_count} requests while "
                    f"L4 handled {layer_counts['L4']}/{request_count}"
                ),
            )
        )

    l2_calibration = _l2_guard_calibration_evidence(traces, l2_results)
    if l2_calibration is not None:
        findings.append(
            BottleneckFinding(
                code="weak_l2_guard_calibration",
                label="weak L2 guard calibration",
                evidence=l2_calibration,
            )
        )

    l3_instability = _l3_instability_evidence(l3_results, settings_text)
    if l3_instability is not None:
        findings.append(
            BottleneckFinding(
                code="local_slm_json_instability",
                label="local SLM JSON instability",
                evidence=l3_instability,
            )
        )

    teacher_conflict = _teacher_inconsistency_evidence(traces)
    if teacher_conflict is not None:
        findings.append(
            BottleneckFinding(
                code="teacher_inconsistency",
                label="teacher inconsistency",
                evidence=teacher_conflict,
                severity="error",
            )
        )

    promotion_gate = _promotion_gate_evidence(promotion_records)
    if promotion_gate is not None:
        findings.append(
            BottleneckFinding(
                code="overly_strict_promotion_gate",
                label="overly strict promotion gate",
                evidence=promotion_gate,
            )
        )

    return findings


def _failed_experiment_analysis_section(bottlenecks: list[BottleneckFinding]) -> str:
    lines = ["## Failed Experiment Analysis", ""]
    if not bottlenecks:
        lines.append("No bottleneck detected from recorded traces and promotion records.")
        return "\n".join(lines)
    for finding in bottlenecks:
        lines.append(f"- {finding.label}: {finding.evidence}")
    return "\n".join(lines)


def _failed_experiment_analysis_html(bottlenecks: list[BottleneckFinding]) -> str:
    if not bottlenecks:
        return "<p>No bottleneck detected from recorded traces and promotion records.</p>"
    rows = ["<tr><th>bottleneck</th><th>severity</th><th>evidence</th></tr>"]
    for finding in bottlenecks:
        rows.append(
            "<tr>"
            f"<td>{html.escape(finding.label)}</td>"
            f"<td>{html.escape(finding.severity)}</td>"
            f"<td>{html.escape(finding.evidence)}</td>"
            "</tr>"
        )
    return f"<table>{''.join(rows)}</table>"


def _accepted_layer_count(traces: list[TraceRecord], layer: LayerName) -> int:
    return sum(
        1
        for trace in traces
        for result in trace.layer_results
        if result.layer == layer and result.accepted
    )


def _layer_results(traces: list[TraceRecord], layer: LayerName):
    return [result for trace in traces for result in trace.layer_results if result.layer == layer]


def _workload_repeat_stats(traces: list[TraceRecord]) -> dict[str, float | int]:
    normalized = [normalize_utterance(trace.utterance) for trace in traces]
    unique = len(set(normalized))
    total = len(normalized)
    repeat_rate = (total - unique) / total if total else 0.0
    return {
        "unique_utterances": unique,
        "requests": total,
        "repeat_rate": repeat_rate,
    }


def _l2_guard_calibration_evidence(
    traces: list[TraceRecord],
    l2_results,
) -> str | None:
    if not l2_results:
        return None
    unguarded = _l2_unguarded_stats(traces)
    labeled_accepts = []
    wrong_accepts = 0
    for trace in traces:
        expected = trace.teacher_frame or trace.gold_frame
        if expected is None:
            continue
        for result in trace.layer_results:
            if result.layer != "L2" or not result.accepted or result.frame is None:
                continue
            labeled_accepts.append(result)
            wrong_accepts += int(result.frame != expected)
    if labeled_accepts:
        wrong_rate = wrong_accepts / len(labeled_accepts)
        if wrong_rate > 0.05:
            return (
                f"L2 wrong accepts {wrong_accepts}/{len(labeled_accepts)} "
                f"({wrong_rate:.3f}) against teacher/gold-visible traces"
            )
    elif len(l2_results) >= 10:
        if unguarded["labeled"]:
            accuracy = unguarded["unguarded_accuracy"]
            p95 = unguarded["p95_ms"]
            if isinstance(accuracy, int | float) and accuracy >= 0.80:
                return (
                    "L2 produced "
                    f"{len(l2_results)} results but accepted none; threshold=0 accuracy would be "
                    f"{accuracy:.3f} over {unguarded['labeled']} labeled observations "
                    f"with p95 latency {p95:.3f} ms"
                )
            if isinstance(accuracy, int | float):
                return (
                    "L2 produced "
                    f"{len(l2_results)} results but accepted none; threshold=0 accuracy is only "
                    f"{accuracy:.3f} over {unguarded['labeled']} labeled observations"
                )
        return f"L2 produced {len(l2_results)} results but accepted none"
    return None


def _l3_instability_evidence(l3_results, settings_text: str) -> str | None:
    if not l3_results:
        return None
    parse_failures = sum(1 for result in l3_results if "parse failed" in result.reason)
    failures = sum(1 for result in l3_results if "failed" in result.reason)
    failure_rate = (parse_failures + failures) / len(l3_results)
    configured_mode = _settings_payload(settings_text).get("local_slm_mode", "unknown")
    if failure_rate > 0.20:
        return (
            f"L3 parse/generation failures {parse_failures + failures}/{len(l3_results)} "
            f"({failure_rate:.3f}) in configured mode {configured_mode}"
        )
    return None


def _teacher_inconsistency_evidence(traces: list[TraceRecord]) -> str | None:
    frames_by_utterance: dict[str, dict[str, str]] = defaultdict(dict)
    original_utterances: dict[str, str] = {}
    for trace in traces:
        if trace.teacher_frame is None:
            continue
        normalized = normalize_utterance(trace.utterance)
        original_utterances.setdefault(normalized, trace.utterance)
        frame_key = trace.teacher_frame.model_dump_json()
        frames_by_utterance[normalized][frame_key] = trace.teacher_frame.intent
    for normalized, frames in sorted(frames_by_utterance.items()):
        if len(frames) > 1:
            intents = sorted(set(frames.values()))
            return (
                f"utterance {original_utterances[normalized]!r} has conflicting "
                f"teacher frames/intents: {intents}"
            )
    return None


def _promotion_gate_evidence(records: list[dict[str, Any]]) -> str | None:
    if not records:
        return None
    promoted = [record for record in records if record.get("promoted")]
    if promoted:
        return None
    reasons = Counter(str(record.get("promotion_reason", "unknown")) for record in records)
    most_common = ", ".join(f"{reason} x{count}" for reason, count in reasons.most_common(3))
    return f"no candidate was promoted across {len(records)} generation(s): {most_common}"


def _settings_payload(settings_text: str) -> dict[str, Any]:
    try:
        payload = json.loads(settings_text)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_promotion_records(run_dir: Path) -> list[dict[str, Any]]:
    generations_dir = run_dir / "artifacts" / "generations"
    if not generations_dir.exists():
        return []
    records = []
    for path in sorted(generations_dir.glob("gen_*/promotion.json")):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return records


def _l3_report_section(run_dir: Path, settings_text: str) -> str:
    try:
        settings_payload = json.loads(settings_text)
    except json.JSONDecodeError:
        settings_payload = {}

    traces = _read_trace_records(run_dir / "traces.jsonl")
    stats = _l3_observation_stats(traces)
    total = int(stats["trace_results"])
    actual_modes = stats["actual_modes"]
    actual_devices = stats["actual_devices"]
    latencies = stats["latencies_ms"]
    load_times = stats["load_times_ms"]
    confidences = stats["confidences"]
    would_accept_accuracy = _ratio_or_na(
        stats["would_accept_correct"],
        stats["would_accept_labeled"],
    )
    guarded_accepted_accuracy = _ratio_or_na(
        stats["accepted_correct"],
        stats["accepted_labeled"],
    )
    calibration = calibrate_l3_confidence_threshold(traces)
    calibration_text = _l3_calibration_text(calibration)
    return (
        "## L3 Local SLM\n\n"
        f"- configured mode: `{settings_payload.get('local_slm_mode', 'unknown')}`\n"
        f"- model: `{settings_payload.get('local_slm_model', 'unknown')}`\n"
        f"- device policy: `{settings_payload.get('local_slm_device_policy', 'unknown')}`\n"
        f"- trace results: {total}\n"
        f"- chosen as final layer: {stats['chosen_l3']}\n"
        f"- actual modes observed: `{', '.join(actual_modes) if actual_modes else 'none'}`\n"
        f"- actual devices observed: `{', '.join(actual_devices) if actual_devices else 'none'}`\n"
        f"- failures: {stats['failures']}\n"
        f"- parse failures: {stats['parse_failures']}\n"
        f"- parse failure rate: {_rate(stats['parse_failures'], total)}\n"
        f"- repair rate: {_rate(stats['repair_count'], total)}\n"
        f"- generation latency p50/p95: {_p50_p95_text(latencies, 'ms')}\n"
        f"- model load time p50/p95: {_p50_p95_text(load_times, 'ms')}\n"
        f"- confidence p50/p95: {_p50_p95_text(confidences, '')}\n"
        f"- shadow/guard would-accept count: {stats['would_accept_count']}\n"
        f"- shadow/guard would-accept accuracy: {would_accept_accuracy}\n"
        f"- guarded accepted accuracy: {guarded_accepted_accuracy}\n"
        f"- guard calibration: {calibration_text}\n"
    )


def _l3_observation_stats(traces: list[TraceRecord]) -> dict[str, Any]:
    l3_results = []
    chosen_l3 = 0
    actual_modes: set[str] = set()
    actual_devices: set[str] = set()
    load_times_ms: list[float] = []
    latencies_ms: list[float] = []
    confidences: list[float] = []
    failures = 0
    parse_failures = 0
    repair_count = 0
    would_accept_count = 0
    would_accept_labeled = 0
    would_accept_correct = 0
    accepted_labeled = 0
    accepted_correct = 0

    for trace in traces:
        chosen_l3 += int(trace.chosen_layer == "L3")
        expected = trace.teacher_frame or trace.gold_frame
        for result in trace.layer_results:
            if result.layer != "L3":
                continue
            l3_results.append(result)
            latencies_ms.append(result.latency_ms)
            metadata = result.metadata or {}
            actual_modes.add(str(metadata.get("actual_mode", "unknown")))
            backend = metadata.get("backend", {})
            if isinstance(backend, dict):
                actual_devices.add(str(backend.get("actual_device", "unknown")))
                load_time = _float_value(backend.get("load_time_ms"))
                if load_time is not None:
                    load_times_ms.append(load_time)
            confidence = _float_value(metadata.get("confidence"))
            if confidence is not None:
                confidences.append(confidence)
            failures += int("failed" in result.reason)
            parse_failures += int("parse failed" in result.reason)
            repair_count += int(metadata.get("repair_used") is True)

            predicted = _l3_predicted_frame(result)
            if metadata.get("would_accept") is True:
                would_accept_count += 1
                if expected is not None and predicted is not None:
                    would_accept_labeled += 1
                    would_accept_correct += int(predicted == expected)
            if result.accepted and result.frame is not None and expected is not None:
                accepted_labeled += 1
                accepted_correct += int(result.frame == expected)

    return {
        "trace_results": len(l3_results),
        "chosen_l3": chosen_l3,
        "actual_modes": sorted(actual_modes),
        "actual_devices": sorted(actual_devices),
        "load_times_ms": load_times_ms,
        "latencies_ms": latencies_ms,
        "confidences": confidences,
        "failures": failures,
        "parse_failures": parse_failures,
        "repair_count": repair_count,
        "would_accept_count": would_accept_count,
        "would_accept_labeled": would_accept_labeled,
        "would_accept_correct": would_accept_correct,
        "accepted_labeled": accepted_labeled,
        "accepted_correct": accepted_correct,
    }


def _l3_predicted_frame(result) -> Frame | None:
    if result.frame is not None:
        return result.frame
    metadata = result.metadata or {}
    shadow_frame = metadata.get("shadow_frame")
    if shadow_frame is None:
        return None
    try:
        return Frame.model_validate(shadow_frame)
    except ValueError:
        return None


def _float_value(value: Any) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def _rate(numerator: Any, denominator: int) -> str:
    if not denominator:
        return "n/a"
    return f"{float(numerator) / denominator:.3f}"


def _ratio_or_na(numerator: Any, denominator: Any) -> str:
    if not denominator:
        return "n/a"
    return f"{float(numerator) / denominator:.3f}"


def _p50_p95_text(values: Any, unit: str) -> str:
    if not isinstance(values, list) or not values:
        return "n/a"
    suffix = f" {unit}" if unit else ""
    return f"{_percentile(values, 50):.3f}/{_percentile(values, 95):.3f}{suffix}"


def _l3_calibration_text(calibration: Any) -> str:
    if calibration is None:
        return "n/a"
    accepted_accuracy = (
        "n/a" if calibration.accepted_accuracy is None else f"{calibration.accepted_accuracy:.3f}"
    )
    return (
        f"threshold={calibration.threshold:.3f}, "
        f"coverage={calibration.coverage:.3f}, "
        f"accepted_accuracy={accepted_accuracy}, "
        f"wrong_accept_rate={calibration.wrong_accept_rate:.3f}"
    )


def _load_generation_manifests(run_dir: Path) -> list[ArtifactManifest]:
    generations_dir = run_dir / "artifacts" / "generations"
    if not generations_dir.exists():
        return []
    manifests: list[ArtifactManifest] = []
    for path in sorted(generations_dir.glob("gen_*/manifest.json")):
        try:
            manifests.append(ArtifactManifest.model_validate_json(path.read_text(encoding="utf-8")))
        except ValueError:
            continue
    return manifests


def _read_trace_records(path: Path) -> list[TraceRecord]:
    if not path.exists():
        return []
    return read_traces(path)


def _read_text_or_default(path: Path, default: str) -> str:
    if not path.exists():
        return default
    return path.read_text(encoding="utf-8")


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _optional_json_object(path: Path) -> dict[str, Any] | None:
    payload = _load_json_object(path)
    return payload or None


def _write_csv(path: Path, *, rows: list[dict[str, Any]], fieldnames: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fieldnames})
    return path


def _metric_row(
    scope: str,
    generation: int | str,
    layer: str,
    metric: str,
    value: Any,
) -> dict[str, Any]:
    return {
        "scope": scope,
        "generation": generation,
        "layer": layer,
        "metric": metric,
        "value": value,
    }


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * percentile / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = rank - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction
