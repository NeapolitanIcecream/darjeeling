import csv
import json
from pathlib import Path

from darjeeling.artifacts.store import ArtifactManifest, ArtifactStore, LayerDelta
from darjeeling.targets.nlu.compiler.mining import build_hard_buffer, write_hard_buffer_jsonl
from darjeeling.targets.nlu.reports import (
    _l3_report_section,
    _promotion_report_section,
    generate_experiment_comparison_report,
    generate_run_report,
)
from darjeeling.targets.nlu.schemas import (
    Frame,
    FramePatch,
    LayerResult,
    TeacherTrace,
    TraceRecord,
)


def _nlu_manifest(**kwargs) -> ArtifactManifest:
    return ArtifactManifest(
        target_name="nlu",
        target_schema_version="nlu-target-v1",
        **kwargs,
    )


def test_l3_report_section_summarizes_mode_device_and_failures(tmp_path: Path) -> None:
    trace = TraceRecord(
        request_id="r1",
        utterance="beta request",
        teacher_frame=Frame(intent="intent_beta"),
        chosen_layer="L4",
        final_frame=Frame(intent="intent_beta"),
        layer_results=[
            LayerResult(
                layer="L3",
                accepted=False,
                reason="shadow local SLM failed; degraded to disabled",
                latency_ms=3.0,
                metadata={
                    "actual_mode": "disabled",
                    "backend": {"actual_device": "not-loaded"},
                },
            )
        ],
    )
    (tmp_path / "traces.jsonl").write_text(trace.model_dump_json() + "\n", encoding="utf-8")
    settings_text = json.dumps(
        {
            "local_slm_mode": "shadow",
            "local_slm_model": "Qwen/Qwen2.5-0.5B-Instruct",
            "local_slm_device_policy": "auto",
        }
    )

    section = _l3_report_section(tmp_path, settings_text)

    assert "configured mode: `shadow`" in section
    assert "actual modes observed: `disabled`" in section
    assert "actual devices observed: `not-loaded`" in section
    assert "failures: 1" in section


def test_l3_report_section_summarizes_shadow_calibration_stats(tmp_path: Path) -> None:
    trace = TraceRecord(
        request_id="r1",
        utterance="beta request",
        teacher_frame=Frame(intent="intent_beta"),
        chosen_layer="L4",
        final_frame=Frame(intent="intent_beta"),
        layer_results=[
            LayerResult(
                layer="L3",
                accepted=False,
                reason="shadow local SLM would accept",
                latency_ms=12.0,
                confidence=0.91,
                metadata={
                    "actual_mode": "shadow",
                    "backend": {
                        "actual_device": "mps",
                        "load_time_ms": 1234.0,
                    },
                    "repair_used": True,
                    "confidence": 0.91,
                    "would_accept": True,
                    "shadow_frame": {
                        "intent": "intent_beta",
                        "slots": {},
                        "is_abstain": False,
                    },
                },
            )
        ],
    )
    (tmp_path / "traces.jsonl").write_text(trace.model_dump_json() + "\n", encoding="utf-8")
    (tmp_path / "settings.json").write_text(
        json.dumps(
            {
                "local_slm_mode": "shadow",
                "local_slm_model": "Qwen/Qwen2.5-0.5B-Instruct",
                "local_slm_device_policy": "mps",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = generate_run_report(tmp_path)

    summary = result.summary_path.read_text(encoding="utf-8")
    metrics = result.metrics_csv_path.read_text(encoding="utf-8")
    assert "actual devices observed: `mps`" in summary
    assert "repair rate: 1.000" in summary
    assert "model load time p50/p95: 1234.000/1234.000 ms" in summary
    assert "shadow/guard would-accept accuracy: 1.000" in summary
    assert "guard calibration: threshold=" in summary
    assert "would_accept_accuracy" in metrics
    assert "model_load_time_p95_ms" in metrics
    assert "recommended_threshold" in metrics


def test_generate_run_report_includes_l3_benchmark_artifact(tmp_path: Path) -> None:
    (tmp_path / "traces.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "settings.json").write_text("{}\n", encoding="utf-8")
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    (report_dir / "l3_benchmark.json").write_text(
        json.dumps(
            {
                "schema_version": "l3-benchmark-v1",
                "status": "success",
                "requests": 3,
                "accepted": 0,
                "would_accept": 2,
                "failures": 0,
                "parse_failures": 0,
                "repair_count": 1,
                "generation_p50_ms": 120.0,
                "generation_p95_ms": 180.0,
                "throughput_qps": 4.2,
                "backend": {
                    "model_name": "fake-l3",
                    "device_policy": "mps",
                    "actual_device": "mps",
                    "load_time_ms": 321.0,
                    "loaded": True,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = generate_run_report(tmp_path)

    summary = result.summary_path.read_text(encoding="utf-8")
    metrics = result.metrics_csv_path.read_text(encoding="utf-8")
    curves = result.curves_html_path.read_text(encoding="utf-8")
    assert "## L3 Hardware Benchmark" in summary
    assert "generation p50/p95: 120.000/180.000 ms" in summary
    assert "actual_device,mps" in metrics
    assert "L3 Hardware Benchmark" in curves


def test_generate_run_report_writes_hard_cases_jsonl_from_latest_generation(
    tmp_path: Path,
) -> None:
    teacher_trace = TeacherTrace(
        request_id="hard-1",
        utterance="beta smooth request",
        teacher_frame=Frame(intent="intent_beta"),
        chosen_layer="L4",
        final_frame=Frame(intent="intent_beta"),
        layer_results=[
            LayerResult(
                layer="L1",
                accepted=False,
                frame=None,
                latency_ms=1.0,
                reason="abstain",
            )
        ],
        timestamp="2026-06-08T00:00:00Z",
    )
    (tmp_path / "traces.jsonl").write_text(
        TraceRecord(
            request_id="hard-1",
            utterance="beta smooth request",
            gold_frame=Frame(intent="intent_beta"),
            teacher_frame=Frame(intent="intent_beta"),
            chosen_layer="L4",
            final_frame=Frame(intent="intent_beta"),
            layer_results=teacher_trace.layer_results,
        ).model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "settings.json").write_text("{}\n", encoding="utf-8")
    generation_dir = tmp_path / "artifacts" / "generations" / "gen_001"
    hard_buffer_path = write_hard_buffer_jsonl(
        generation_dir / "hard_buffer.jsonl",
        build_hard_buffer([teacher_trace]),
    )
    manifest = _nlu_manifest(
        artifact_set_id="gen_001_candidate",
        generation=1,
        artifact_paths={"hard_buffer": str(hard_buffer_path.relative_to(tmp_path / "artifacts"))},
    )
    (generation_dir / "manifest.json").write_text(
        manifest.model_dump_json() + "\n",
        encoding="utf-8",
    )

    result = generate_run_report(tmp_path)

    hard_cases = result.hard_cases_path.read_text(encoding="utf-8")
    summary = result.summary_path.read_text(encoding="utf-8")
    assert result.hard_cases_path.name == "hard_cases.jsonl"
    assert "beta smooth request" in hard_cases
    assert "gold_frame" not in hard_cases
    assert "`hard_cases.jsonl`" in summary


def test_promotion_report_section_summarizes_generation_records(tmp_path: Path) -> None:
    generation_dir = tmp_path / "artifacts" / "generations" / "gen_001"
    generation_dir.mkdir(parents=True)
    (generation_dir / "promotion.json").write_text(
        json.dumps(
            {
                "generation": 1,
                "promoted": True,
                "promotion_reason": "objective improved within gates",
                "candidate_objective": {
                    "frame_exact_match": 1.0,
                    "wrong_accept_rate": 0.0,
                },
                "current_objective": {
                    "frame_exact_match": 0.5,
                    "wrong_accept_rate": 0.0,
                },
                "promoted_with_layer_regression": False,
                "regressed_layers": [],
            }
        ),
        encoding="utf-8",
    )

    section = _promotion_report_section(tmp_path)

    assert "gen 1: promoted=True" in section
    assert "objective improved within gates" in section
    assert "frame_exact=1.0" in section


def test_generate_run_report_writes_summary_metrics_artifacts_and_curves(
    tmp_path: Path,
) -> None:
    trace = TraceRecord(
        request_id="r1",
        utterance="beta request",
        gold_frame=Frame(intent="intent_beta"),
        teacher_frame=Frame(intent="intent_beta"),
        chosen_layer="L4",
        final_frame=Frame(intent="intent_beta"),
        layer_results=[
            LayerResult(
                layer="L4",
                accepted=True,
                frame=Frame(intent="intent_beta"),
                latency_ms=900.0,
            )
        ],
    )
    (tmp_path / "traces.jsonl").write_text(trace.model_dump_json() + "\n", encoding="utf-8")
    (tmp_path / "settings.json").write_text(
        json.dumps(
            {
                "local_slm_mode": "disabled",
                "local_slm_model": "Qwen/Qwen2.5-0.5B-Instruct",
                "local_slm_device_policy": "auto",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    store = ArtifactStore(tmp_path / "artifacts")
    manifest = _nlu_manifest(
        artifact_set_id="gen_001_candidate",
        generation=1,
        artifact_paths={
            "l0_cache": "generations/gen_001/l0_cache.json",
            "promotion_record": "generations/gen_001/promotion.json",
        },
        candidate_metrics={"hard_buffer_size": 1},
        promotion_reason="objective improved within gates",
    )
    store.promote(manifest)
    generation_dir = tmp_path / "artifacts" / "generations" / "gen_001"
    (generation_dir / "promotion.json").write_text(
        json.dumps(
            {
                "generation": 1,
                "promoted": True,
                "promotion_reason": "objective improved within gates",
                "candidate_objective": {
                    "frame_exact_match": 1.0,
                    "wrong_accept_rate": 0.0,
                },
                "current_objective": {
                    "frame_exact_match": 0.0,
                    "wrong_accept_rate": 0.0,
                },
                "promoted_with_layer_regression": False,
                "regressed_layers": [],
                "per_layer_deltas": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = generate_run_report(tmp_path)

    assert result.summary_path.exists()
    assert result.metrics_csv_path.exists()
    assert result.artifacts_csv_path.exists()
    assert result.curves_html_path.exists()
    assert "Layer Summary" in result.summary_path.read_text(encoding="utf-8")
    assert "frame_exact_match" in result.metrics_csv_path.read_text(encoding="utf-8")
    assert "l0_cache" in result.artifacts_csv_path.read_text(encoding="utf-8")
    assert "Cumulative Layer Share" in result.curves_html_path.read_text(encoding="utf-8")


def test_generate_run_report_includes_required_layer_summary_metrics(tmp_path: Path) -> None:
    traces = [
        TraceRecord(
            request_id="r1",
            utterance="alpha request",
            gold_frame=Frame(intent="intent_alpha"),
            teacher_frame=Frame(intent="intent_alpha"),
            chosen_layer="L1",
            final_frame=Frame(intent="intent_alpha"),
            layer_results=[
                LayerResult(
                    layer="L1",
                    accepted=True,
                    frame=Frame(intent="intent_alpha"),
                    latency_ms=1.0,
                    cost_usd=0.0,
                )
            ],
        ),
        TraceRecord(
            request_id="r2",
            utterance="beta request",
            gold_frame=Frame(intent="intent_beta"),
            teacher_frame=Frame(intent="intent_beta"),
            chosen_layer="L1",
            final_frame=Frame(intent="intent_alpha"),
            layer_results=[
                LayerResult(
                    layer="L1",
                    accepted=True,
                    frame=Frame(intent="intent_alpha"),
                    latency_ms=3.0,
                    cost_usd=0.0,
                )
            ],
        ),
        TraceRecord(
            request_id="r3",
            utterance="gamma",
            gold_frame=Frame(intent="intent_gamma"),
            teacher_frame=Frame(intent="intent_gamma"),
            chosen_layer="L4",
            final_frame=Frame(intent="intent_gamma"),
            layer_results=[
                LayerResult(
                    layer="L1",
                    accepted=False,
                    frame=None,
                    latency_ms=2.0,
                    cost_usd=0.0,
                ),
                LayerResult(
                    layer="L4",
                    accepted=True,
                    frame=Frame(intent="intent_gamma"),
                    latency_ms=900.0,
                    cost_usd=0.02,
                ),
            ],
        ),
    ]
    (tmp_path / "traces.jsonl").write_text(
        "".join(trace.model_dump_json() + "\n" for trace in traces),
        encoding="utf-8",
    )
    (tmp_path / "settings.json").write_text("{}\n", encoding="utf-8")

    result = generate_run_report(tmp_path)

    summary = result.summary_path.read_text(encoding="utf-8")
    metrics = list(csv.DictReader(result.metrics_csv_path.open(encoding="utf-8")))
    metric_lookup = {(row["scope"], row["layer"], row["metric"]): row["value"] for row in metrics}
    assert "| layer | coverage | accepted_accuracy | wrong_accept_rate |" in summary
    assert "| L1 | 0.667 | 0.500 | 0.333 | 0.333 | 2.000 | 2.900 | 0.000000 |" in summary
    assert metric_lookup[("layer_summary", "L1", "accepted_accuracy")] == "0.5"
    assert metric_lookup[("layer_summary", "L1", "wrong_accept_rate")] == "0.333333"
    assert metric_lookup[("layer_summary", "L1", "forced_global_accuracy")] == "0.333333"
    assert metric_lookup[("layer_summary", "L4", "cost_usd_per_100_requests")] == "0.666667"


def test_generate_run_report_splits_serving_residual_audit_and_labeling_costs(
    tmp_path: Path,
) -> None:
    traces = [
        TraceRecord(
            request_id="r1",
            utterance="alpha request",
            gold_frame=Frame(intent="intent_alpha"),
            teacher_frame=Frame(intent="intent_alpha"),
            chosen_layer="L4",
            final_frame=Frame(intent="intent_alpha"),
            layer_results=[
                LayerResult(
                    layer="L4",
                    accepted=True,
                    frame=Frame(intent="intent_alpha"),
                    latency_ms=900.0,
                    cost_usd=0.02,
                    metadata={
                        "l4_call_kind": "full",
                        "usage": {"total_tokens": 10},
                    },
                )
            ],
            metadata={
                "teacher_audited": True,
                "teacher_audit_cost_usd": 0.01,
                "teacher_audit_latency_ms": 800.0,
                "teacher_audit_tokens": 7,
                "teacher_labeling_cost_usd": 0.03,
                "teacher_labeling_latency_ms": 700.0,
                "teacher_labeling_tokens": 9,
            },
        ),
        TraceRecord(
            request_id="r2",
            utterance="beta request",
            gold_frame=Frame(intent="intent_beta"),
            teacher_frame=Frame(intent="intent_beta"),
            chosen_layer="L4",
            final_frame=Frame(intent="intent_beta"),
            layer_results=[
                LayerResult(
                    layer="L4",
                    accepted=True,
                    patch=FramePatch(
                        accepted_intent="intent_beta",
                        source_layer="L4",
                        complete=True,
                    ),
                    latency_ms=250.0,
                    cost_usd=0.005,
                    metadata={
                        "l4_call_kind": "residual",
                        "fields_avoided": 1,
                        "usage": {"total_tokens": 4},
                    },
                )
            ],
        ),
    ]
    (tmp_path / "traces.jsonl").write_text(
        "".join(trace.model_dump_json() + "\n" for trace in traces),
        encoding="utf-8",
    )
    (tmp_path / "settings.json").write_text("{}\n", encoding="utf-8")

    result = generate_run_report(tmp_path)

    summary = result.summary_path.read_text(encoding="utf-8")
    metrics = list(csv.DictReader(result.metrics_csv_path.open(encoding="utf-8")))
    metric_lookup = {
        (row["scope"], row["layer"], row["metric"]): row["value"] for row in metrics
    }
    assert "## L4 Cost Summary" in summary
    assert metric_lookup[("cost_summary", "serving_full_l4", "calls_per_100")] == "50.0"
    assert (
        metric_lookup[
            ("cost_summary", "serving_residual_l4", "cost_usd_per_100_requests")
        ]
        == "0.25"
    )
    assert metric_lookup[("cost_summary", "audit_l4", "tokens_per_100")] == "350.0"
    assert (
        metric_lookup[
            ("cost_summary", "teacher_labeling_l4", "cost_usd_per_100_requests")
        ]
        == "1.5"
    )
    assert metric_lookup[("cost_summary", "", "serving_fields_avoided")] == "1.0"


def test_generate_run_report_includes_l2_unguarded_diagnostics(tmp_path: Path) -> None:
    traces = [
        TraceRecord(
            request_id="r1",
            utterance="beta request",
            gold_frame=Frame(intent="intent_beta"),
            teacher_frame=Frame(intent="intent_beta"),
            chosen_layer="L4",
            final_frame=Frame(intent="intent_beta"),
            layer_results=[
                LayerResult(
                    layer="L2",
                    accepted=False,
                    frame=None,
                    confidence=0.12,
                    latency_ms=2.0,
                    metadata={
                        "predicted_frame": {"intent": "intent_beta", "slots": {}},
                        "slot_invalid_bio": False,
                        "nearest_similarity": 0.9,
                        "predicted_intent_similarity": 0.8,
                        "intent_support_margin": 0.7,
                    },
                ),
                LayerResult(
                    layer="L4",
                    accepted=True,
                    frame=Frame(intent="intent_beta"),
                    latency_ms=900.0,
                ),
            ],
        ),
        TraceRecord(
            request_id="r2",
            utterance="alpha request",
            gold_frame=Frame(intent="intent_alpha"),
            teacher_frame=Frame(intent="intent_alpha"),
            chosen_layer="L4",
            final_frame=Frame(intent="intent_alpha"),
            layer_results=[
                LayerResult(
                    layer="L2",
                    accepted=False,
                    frame=None,
                    confidence=0.08,
                    latency_ms=4.0,
                    metadata={
                        "predicted_frame": {"intent": "intent_beta", "slots": {}},
                        "slot_invalid_bio": False,
                        "nearest_similarity": 0.4,
                        "predicted_intent_similarity": 0.2,
                        "intent_support_margin": -0.3,
                    },
                ),
                LayerResult(
                    layer="L4",
                    accepted=True,
                    frame=Frame(intent="intent_alpha"),
                    latency_ms=900.0,
                ),
            ],
        ),
    ]
    (tmp_path / "traces.jsonl").write_text(
        "".join(trace.model_dump_json() + "\n" for trace in traces),
        encoding="utf-8",
    )
    (tmp_path / "settings.json").write_text("{}\n", encoding="utf-8")

    result = generate_run_report(tmp_path)

    summary = result.summary_path.read_text(encoding="utf-8")
    metrics = list(csv.DictReader(result.metrics_csv_path.open(encoding="utf-8")))
    metric_lookup = {(row["scope"], row["metric"]): row["value"] for row in metrics}
    assert "## L2 Unguarded Diagnostics" in summary
    assert "- threshold=0 accuracy: 0.500" in summary
    assert metric_lookup[("l2_unguarded", "unguarded_accuracy")] == "0.5"
    assert metric_lookup[("l2_unguarded", "p95_ms")] == "3.9"
    assert metric_lookup[("l2_unguarded", "predicted_intent_similarity_p50")] == "0.5"


def test_generate_run_report_includes_l2_tuning_summary(tmp_path: Path) -> None:
    (tmp_path / "traces.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "settings.json").write_text("{}\n", encoding="utf-8")
    manifest = _nlu_manifest(
        artifact_set_id="gen_001_candidate",
        generation=1,
        artifact_paths={"l2_tuning": "generations/gen_001/l2/l2_tuning.json"},
        candidate_metrics={
            "l2_training_scope": "lower_miss",
            "l2_teacher_train_traces": 100,
            "l2_lower_miss_train_traces": 40,
            "l2_training_traces": 40,
            "l2_config": {
                "intent_model_family": "mlp",
                "slot_model_family": "none",
                "max_features": 10000,
                "word_ngram_range": [1, 3],
                "char_ngram_range": [3, 3],
            },
            "l2_tuning": {
                "schema_version": "l2-tune-v1",
                "train_size": 58,
                "validation_size": 20,
                "validation_residual_size": 12,
                "objective_validation_size": 12,
                "objective_validation_source": "residual",
                "n_trials_requested": 12,
                "n_trials_completed": 12,
                "best_trial_number": 5,
                "best_value": 1.49,
                "best_metrics": {
                    "unguarded": {"accepted_accuracy": 0.75},
                    "guarded": {
                        "coverage": 0.75,
                        "accepted_accuracy": 1.0,
                        "wrong_accept_rate": 0.0,
                    },
                },
            },
        },
    )
    ArtifactStore(tmp_path / "artifacts").promote(manifest)

    result = generate_run_report(tmp_path)

    summary = result.summary_path.read_text(encoding="utf-8")
    assert "## L2 Tuning" in summary
    assert "- training scope: lower_miss" in summary
    assert "- teacher/lower-miss/target traces: 100/40/40" in summary
    assert "- trials completed/requested: 12/12" in summary
    assert "- residual/objective validation size: 12/12 (residual)" in summary
    assert "- selected intent model: mlp" in summary
    assert "- tuning validation unguarded accuracy: 0.750" in summary
    assert "- tuning validation guarded coverage/accuracy/wrong rate: 0.750/1.000/0.000" in summary
    assert "- tuning artifact: `generations/gen_001/l2/l2_tuning.json`" in summary


def test_generate_run_report_includes_evolution_and_artifact_summary_tables(
    tmp_path: Path,
) -> None:
    (tmp_path / "traces.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "settings.json").write_text("{}\n", encoding="utf-8")
    generation_dir = tmp_path / "artifacts" / "generations" / "gen_001"
    generation_dir.mkdir(parents=True)
    manifest = _nlu_manifest(
        artifact_set_id="gen_001_candidate",
        generation=1,
        artifact_paths={"l1_agent_diff": "generations/gen_001/diff.patch"},
        per_layer_deltas={
            "L1": LayerDelta(
                coverage_delta=0.25,
                accepted_accuracy_delta=0.10,
                cost_delta=-1.5,
            )
        },
        promoted=True,
        promotion_reason="objective improved within gates",
    )
    (generation_dir / "manifest.json").write_text(
        manifest.model_dump_json() + "\n",
        encoding="utf-8",
    )
    (generation_dir / "promotion.json").write_text(
        json.dumps(
            {
                "artifact_set_id": "gen_001_candidate",
                "generation": 1,
                "promoted": True,
                "promotion_reason": "objective improved within gates",
                "current_objective": {
                    "frame_exact_match": 0.8,
                    "cost_usd_per_100_requests": 4.0,
                    "p95_latency_ms": 900.0,
                },
                "candidate_objective": {
                    "frame_exact_match": 0.9,
                    "cost_usd_per_100_requests": 2.5,
                    "p95_latency_ms": 500.0,
                },
                "candidate_metrics": {
                    "promotion_eval_size": 4,
                    "candidate_layer_counts": {
                        "L0": 1,
                        "L1": 1,
                        "L2": 1,
                        "L3": 0,
                        "L4": 1,
                    },
                    "candidate_cost_metrics": {
                        "serving_full_l4_calls_per_100": 25.0,
                        "serving_residual_l4_calls_per_100": 0.0,
                    },
                },
                "per_layer_deltas": {
                    "L1": {
                        "coverage_delta": 0.25,
                        "accepted_accuracy_delta": 0.10,
                        "cost_delta": -1.5,
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = generate_run_report(tmp_path)

    summary = result.summary_path.read_text(encoding="utf-8")
    metrics = result.metrics_csv_path.read_text(encoding="utf-8")
    assert "## Evolution Summary" in summary
    assert (
        "| generation | full_L4/100 | residual_L4/100 | L4_calls/100 | cost/100 | p95_ms | frame_em | "
        "L0_share | L1_share | L2_share | L3_share | L4_share |"
    ) in summary
    assert (
        "| 1 | 25.000 | 0.000 | 25.000 | 2.500000 | 500.000 | 0.900 | 0.250 | 0.250 | 0.250 | 0.000 | 0.250 |"
    ) in summary
    assert "## Artifact Summary" in summary
    assert (
        "| artifact_id | type | generation | coverage_delta | accuracy_delta | "
        "cost_delta | promoted | reason |"
    ) in summary
    assert (
        "| gen_001_candidate | artifact_set | 1 | n/a | 0.100 | "
        "-1.500000 | True | objective improved within gates |"
    ) in summary
    assert (
        "| gen_001_candidate:L1 | L1 | 1 | 0.250 | 0.100 | "
        "-1.500000 | True | objective improved within gates |"
    ) in summary
    assert "evolution_summary,1,,l4_calls_per_100,25.0" in metrics
    assert "evolution_summary,1,,serving_full_l4_calls_per_100,25.0" in metrics


def test_generate_run_report_includes_l1_program_paths_and_diff_snippet(
    tmp_path: Path,
) -> None:
    trace = TraceRecord(
        request_id="r1",
        utterance="alpha request value alpha",
        gold_frame=Frame(intent="intent_alpha", slots={"slot_alpha": "value alpha"}),
        teacher_frame=Frame(intent="intent_alpha", slots={"slot_alpha": "value alpha"}),
        chosen_layer="L1",
        final_frame=Frame(intent="intent_alpha", slots={"slot_alpha": "value alpha"}),
        layer_results=[
            LayerResult(
                layer="L1",
                accepted=True,
                frame=Frame(intent="intent_alpha", slots={"slot_alpha": "value alpha"}),
                latency_ms=1.5,
                metadata={
                    "program_path": "programs/alpha::try_intent_alpha",
                    "native_latency_us": 42,
                },
            )
        ],
    )
    (tmp_path / "traces.jsonl").write_text(trace.model_dump_json() + "\n", encoding="utf-8")
    (tmp_path / "settings.json").write_text("{}\n", encoding="utf-8")
    generation_dir = tmp_path / "artifacts" / "generations" / "gen_001"
    generation_dir.mkdir(parents=True)
    (generation_dir / "diff.patch").write_text(
        "\n".join(
            [
                "diff --git a/src/programs/alpha.rs b/src/programs/alpha.rs",
                "+fn evolved_alpha_path() {}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (generation_dir / "l1_benchmark.json").write_text(
        json.dumps(
            {
                "schema_version": "l1-benchmark-v1",
                "status": "success",
                "requests": 3,
                "accepted": 2,
                "native_p95_us": 11.0,
                "integration_p95_ms": 0.7,
                "throughput_qps": 999.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    ArtifactStore(tmp_path / "artifacts").promote(
        _nlu_manifest(
            artifact_set_id="gen_001_l1",
            generation=1,
            artifact_paths={
                "l1_agent_diff": "generations/gen_001/diff.patch",
                "l1_benchmark": "generations/gen_001/l1_benchmark.json",
            },
            promotion_reason="test fixture",
        )
    )

    result = generate_run_report(tmp_path)

    summary = result.summary_path.read_text(encoding="utf-8")
    metrics = result.metrics_csv_path.read_text(encoding="utf-8")
    curves = result.curves_html_path.read_text(encoding="utf-8")
    assert "## L1 Rust ProgramBank" in summary
    assert "programs/alpha::try_intent_alpha" in summary
    assert "native p95 latency: 42.0 us" in summary
    assert "evolved_alpha_path" in summary
    assert "native_latency_p95_us" in metrics
    assert "l1_generation_benchmark" in metrics
    assert "L1 Program Paths" in curves
    assert "L1 Benchmark By Generation" in curves


def test_generate_run_report_writes_l1_benchmark_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    crate_dir = tmp_path / "candidate_l1"
    binary_path = crate_dir / "target" / "debug" / "darjeeling-l1-programbank"
    binary_path.parent.mkdir(parents=True)
    binary_path.write_text("fake binary", encoding="utf-8")
    (crate_dir / "src").mkdir()
    (crate_dir / "Cargo.toml").write_text("[package]\nname='fake'\n", encoding="utf-8")
    (crate_dir / "src" / "lib.rs").write_text("pub fn try_answer() {}\n", encoding="utf-8")

    trace = TraceRecord(
        request_id="r1",
        utterance="alpha request value alpha",
        gold_frame=Frame(intent="intent_alpha", slots={"slot_alpha": "value alpha"}),
        teacher_frame=Frame(intent="intent_alpha", slots={"slot_alpha": "value alpha"}),
        chosen_layer="L4",
        final_frame=Frame(intent="intent_alpha", slots={"slot_alpha": "value alpha"}),
        layer_results=[],
    )
    (tmp_path / "traces.jsonl").write_text(trace.model_dump_json() + "\n", encoding="utf-8")
    (tmp_path / "settings.json").write_text(
        json.dumps(
            {
                "l1_rust_crate_dir": str(crate_dir),
                "l1_worker_timeout_s": 0.25,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    calls = {}

    def fake_benchmark_worker(binary_arg, utterances, *, timeout_s):
        calls["binary_path"] = binary_arg
        calls["utterances"] = list(utterances)
        calls["timeout_s"] = timeout_s
        return {
            "requests": 1,
            "accepted": 1,
            "accepted_share": 1.0,
            "integration_avg_ms": 0.42,
            "integration_p50_ms": 0.42,
            "integration_p95_ms": 0.42,
            "native_avg_us": 7.0,
            "native_p50_us": 7.0,
            "native_p95_us": 7.0,
            "native_max_us": 7,
            "throughput_qps": 1200.0,
            "program_path_counts": {"programs/alpha::try_intent_alpha": 1},
        }

    monkeypatch.setattr("darjeeling.targets.nlu.reports.benchmark_worker", fake_benchmark_worker)

    result = generate_run_report(tmp_path)

    assert result.l1_benchmark_path is not None
    assert result.l1_benchmark_path.exists()
    payload = json.loads(result.l1_benchmark_path.read_text(encoding="utf-8"))
    assert payload["status"] == "success"
    assert payload["corpus"] == "trace_unique_utterances"
    assert payload["source_size_bytes"] > 0
    assert payload["binary_size_bytes"] == len("fake binary")
    assert calls == {
        "binary_path": binary_path,
        "utterances": ["alpha request value alpha"],
        "timeout_s": 0.25,
    }
    assert "L1 independent benchmark" in result.summary_path.read_text(encoding="utf-8")
    assert "native_p95_us" in result.metrics_csv_path.read_text(encoding="utf-8")
    assert "L1 Native Benchmark" in result.curves_html_path.read_text(encoding="utf-8")


def test_generate_experiment_comparison_report_summarizes_runs(tmp_path: Path) -> None:
    run_a = tmp_path / "main"
    run_b = tmp_path / "no-l2"
    run_a.mkdir()
    run_b.mkdir()
    _write_comparison_trace(
        run_a,
        experiment="main-evolution",
        stream="zipf-heavy",
        chosen_layer="L1",
        final_frame=Frame(intent="intent_alpha"),
        gold_frame=Frame(intent="intent_alpha"),
    )
    residual_frame = Frame(
        intent="intent_alpha",
        slots={"slot_alpha": "weak value", "slot_beta": "residual value"},
    )
    (run_a / "traces.jsonl").write_text(
        TraceRecord(
            request_id="main-evolution-r1",
            utterance="main-evolution utterance",
            gold_frame=residual_frame,
            teacher_frame=residual_frame,
            chosen_layer="L4",
            final_frame=residual_frame,
            layer_results=[
                LayerResult(
                    layer="L1",
                    accepted=True,
                    patch=FramePatch(
                        accepted_intent="intent_alpha",
                        accepted_slots={"slot_alpha": "weak value"},
                        source_layer="L1",
                    ),
                    latency_ms=1.0,
                ),
                LayerResult(
                    layer="L4",
                    accepted=True,
                    patch=FramePatch(
                        accepted_slots={"slot_beta": "residual value"},
                        source_layer="L4",
                        complete=True,
                        metadata={"verified_fields": ["intent", "slots.slot_alpha"]},
                    ),
                    latency_ms=250.0,
                    cost_usd=0.005,
                    metadata={
                        "l4_call_kind": "residual",
                        "usage": {"total_tokens": 4},
                        "residual_verification_complete": True,
                        "residual_verified_field_count": 2,
                    },
                ),
            ],
        ).model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    _write_comparison_trace(
        run_b,
        experiment="no-l2",
        stream="zipf-heavy",
        chosen_layer="L4",
        final_frame=Frame(intent="intent_beta"),
        gold_frame=Frame(intent="intent_gamma"),
    )
    promotion_dir = run_a / "artifacts" / "generations" / "gen_001"
    promotion_dir.mkdir(parents=True)
    (promotion_dir / "promotion.json").write_text(
        json.dumps(
            {
                "generation": 1,
                "promoted": True,
                "promoted_with_layer_regression": True,
                "candidate_objective": {},
                "current_objective": {},
                "per_layer_deltas": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    report_dir = run_a / "reports"
    report_dir.mkdir()
    (report_dir / "l1_benchmark.json").write_text(
        json.dumps(
            {
                "schema_version": "l1-benchmark-v1",
                "status": "success",
                "native_p95_us": 9.0,
                "throughput_qps": 1000.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = generate_experiment_comparison_report(
        [run_a, run_b],
        tmp_path / "comparison",
    )

    csv_text = result.comparison_csv_path.read_text(encoding="utf-8")
    html_text = result.comparison_html_path.read_text(encoding="utf-8")
    rows = list(csv.DictReader(result.comparison_csv_path.open(encoding="utf-8")))
    assert "main-evolution" in csv_text
    assert "no-l2" in csv_text
    assert rows[0]["experiment"] == "main-evolution"
    assert rows[0]["comparison_rank"] == "1"
    assert "comparison_score" in rows[0]
    assert rows[0]["weak_field_coverage"] == "0.666667"
    assert rows[0]["residual_l4_calls_per_100"] == "100.0"
    assert rows[0]["residual_l4_tokens_per_100"] == "400.0"
    assert rows[0]["serving_cost_per_100"] == "0.5"
    assert rows[0]["correct_weak_fields_avoiding_full_l4_per_100"] == "200.0"
    assert rows[0]["residual_l4_verified_fields_per_100"] == "200.0"
    assert "l1_benchmark_native_p95_us" in csv_text
    assert "9.0" in csv_text
    assert "Experiment Comparison" in html_text
    assert "Bottleneck Summary" in html_text
    assert "promoted_with_layer_regression" in html_text


def test_no_audit_comparison_reports_zero_audit_cost(tmp_path: Path) -> None:
    run_dir = tmp_path / "no-audit"
    run_dir.mkdir()
    _write_comparison_trace(
        run_dir,
        experiment="no-audit",
        stream="zipf-heavy",
        chosen_layer="L1",
        final_frame=Frame(intent="intent_alpha"),
        gold_frame=Frame(intent="intent_alpha"),
    )

    result = generate_experiment_comparison_report(
        [run_dir],
        tmp_path / "comparison",
    )

    rows = list(csv.DictReader(result.comparison_csv_path.open(encoding="utf-8")))
    assert rows[0]["experiment"] == "no-audit"
    assert rows[0]["audit_cost_per_100"] == "0.0"


def _write_comparison_trace(
    run_dir: Path,
    *,
    experiment: str,
    stream: str,
    chosen_layer: str,
    final_frame: Frame,
    gold_frame: Frame,
) -> None:
    (run_dir / "experiment.json").write_text(
        json.dumps({"experiment": experiment, "stream": stream}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "settings.json").write_text("{}\n", encoding="utf-8")
    trace = TraceRecord(
        request_id=f"{experiment}-r1",
        utterance=f"{experiment} utterance",
        gold_frame=gold_frame,
        teacher_frame=gold_frame,
        chosen_layer=chosen_layer,
        final_frame=final_frame,
        layer_results=[
            LayerResult(
                layer=chosen_layer,
                accepted=True,
                frame=final_frame,
                latency_ms=2.0,
            )
        ],
    )
    (run_dir / "traces.jsonl").write_text(trace.model_dump_json() + "\n", encoding="utf-8")


def test_generate_run_report_identifies_failed_experiment_bottlenecks(
    tmp_path: Path,
) -> None:
    traces = []
    for index in range(12):
        traces.append(
            TraceRecord(
                request_id=f"r{index}",
                utterance=f"unique request {index}",
                gold_frame=Frame(intent="intent_beta"),
                teacher_frame=Frame(intent="intent_beta"),
                chosen_layer="L4",
                final_frame=Frame(intent="intent_beta"),
                layer_results=[
                    LayerResult(layer="L1", accepted=False, latency_ms=1.0),
                    LayerResult(
                        layer="L3",
                        accepted=False,
                        reason="local SLM parse failed",
                        latency_ms=20.0,
                    ),
                    LayerResult(
                        layer="L4",
                        accepted=True,
                        frame=Frame(intent="intent_beta"),
                        latency_ms=900.0,
                    ),
                ],
            )
        )
    (tmp_path / "traces.jsonl").write_text(
        "".join(trace.model_dump_json() + "\n" for trace in traces),
        encoding="utf-8",
    )
    (tmp_path / "settings.json").write_text(
        json.dumps({"local_slm_mode": "shadow"}) + "\n",
        encoding="utf-8",
    )
    generation_dir = tmp_path / "artifacts" / "generations" / "gen_001"
    generation_dir.mkdir(parents=True)
    (generation_dir / "promotion.json").write_text(
        json.dumps(
            {
                "generation": 1,
                "promoted": False,
                "promotion_reason": "objective did not improve",
                "candidate_objective": {"frame_exact_match": 1.0},
                "current_objective": {"frame_exact_match": 1.0},
                "promoted_with_layer_regression": False,
                "regressed_layers": [],
                "per_layer_deltas": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = generate_run_report(tmp_path)

    summary = result.summary_path.read_text(encoding="utf-8")
    metrics = result.metrics_csv_path.read_text(encoding="utf-8")
    curves = result.curves_html_path.read_text(encoding="utf-8")
    assert "## Failed Experiment Analysis" in summary
    assert "insufficient workload locality" in summary
    assert "weak L1 rule coverage" in summary
    assert "local SLM JSON instability" in summary
    assert "overly strict promotion gate" in summary
    assert "insufficient_workload_locality" in metrics
    assert "Failed Experiment Analysis" in curves


def test_generate_run_report_identifies_teacher_inconsistency(tmp_path: Path) -> None:
    traces = [
        TraceRecord(
            request_id="r1",
            utterance="beta sample request",
            teacher_frame=Frame(intent="intent_beta"),
            chosen_layer="L4",
            final_frame=Frame(intent="intent_beta"),
            layer_results=[],
        ),
        TraceRecord(
            request_id="r2",
            utterance="beta sample request",
            teacher_frame=Frame(intent="intent_alpha"),
            chosen_layer="L4",
            final_frame=Frame(intent="intent_alpha"),
            layer_results=[],
        ),
    ]
    (tmp_path / "traces.jsonl").write_text(
        "".join(trace.model_dump_json() + "\n" for trace in traces),
        encoding="utf-8",
    )

    result = generate_run_report(tmp_path)

    summary = result.summary_path.read_text(encoding="utf-8")
    assert "teacher inconsistency" in summary
    assert "beta sample request" in summary
