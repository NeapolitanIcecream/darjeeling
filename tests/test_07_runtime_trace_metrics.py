from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Literal

import pytest

from darjeeling.model import (
    ArtifactManifest,
    ArtifactPackage,
    AuditDecisions,
    AuditRecord,
    CacheHit,
    CacheMiss,
    CascadeResult,
    CostLedger,
    GeneralizationSummary,
    LayerAttemptResult,
    ReferenceQualificationReport,
    Release,
    Report,
    RoutingSettings,
    RuntimeMetricWindow,
    RuntimeRequest,
    ServingResult,
    TargetRequirements,
)
from darjeeling.runtime_trace_metrics import (
    aggregate_runtime_metrics,
    detect_drift,
    detect_runtime_failure,
    write_trace,
)
from darjeeling.target_definition import load_checked_target
from darjeeling.util import utcnow


def _dummy_artifact(
    layer: Literal["L1", "L2", "L3"], tmp_path: Path, contract_hash: str
) -> ArtifactPackage:
    return ArtifactPackage(
        f"artifact-{layer[-1]}",
        layer,
        tmp_path / f"artifact-{layer}",
        ArtifactManifest(
            "v1",
            layer,
            ["python3", "worker.py"],
            ["python3", "healthcheck.py"],
            "jsonl",
            1000,
            64,
            "disabled",
            contract_hash,
        ),
        f"digest-{layer}",
        "snapshot-digest",
        {},
    )


def test_trace_redacts_raw_input_output(target_dir: Path) -> None:
    definition, contract, _ = load_checked_target(target_dir)
    serving = ServingResult(
        "rel",
        "cascade",
        "ok",
        CacheMiss("k"),
        None,
        {"label": "a"},
        "L4",
        0.01,
        3.0,
        None,
        None,
        None,
    )
    trace = write_trace(
        "t1",
        definition.contract_hash,
        contract,
        RuntimeRequest(
            "req", definition.name, {"text": "a:secret"}, metadata={"tenant": "tenant-1"}
        ),
        serving,
        AuditDecisions(None, None, [], 0.0, []),
    )
    assert trace.input_redacted == {"text": "<redacted>"}
    assert "a:secret" not in repr(trace)
    assert trace.metadata_buckets == {"tenant_bucket": "tenant"}


def test_runtime_metrics_reject_mixed_scope_and_reference_failures_trigger_failure(
    target_dir: Path,
) -> None:
    definition, contract, _ = load_checked_target(target_dir)
    release = Release(
        "rel",
        definition.name,
        definition.contract_hash,
        None,
        None,
        None,
        None,
        utcnow(),
        {"L1": None, "L2": None, "L3": None},
        RoutingSettings(),
        None,
    )
    trace = write_trace(
        "t1",
        definition.contract_hash,
        contract,
        RuntimeRequest("req", definition.name, {"text": "a:x"}),
        ServingResult(
            "rel",
            "cascade",
            "ok",
            CacheMiss("k"),
            None,
            {"label": "a"},
            "L4",
            0.01,
            3.0,
            None,
            None,
            None,
        ),
        AuditDecisions(None, None, [], 1.0, []),
    )
    audit = AuditRecord(
        "a1",
        "t1",
        definition.name,
        "rel",
        definition.contract_hash,
        "random",
        "error",
        1.0,
        None,
        None,
        None,
        None,
        None,
        None,
        "provider_error",
        "h",
        0.01,
        utcnow(),
    )
    metrics = aggregate_runtime_metrics(
        [trace],
        [audit],
        (utcnow() - timedelta(minutes=1), utcnow() + timedelta(minutes=1)),
        release,
    )
    decision = detect_runtime_failure(
        metrics, release, TargetRequirements(random_audit_reference_failure_rate_max=0.0)
    )
    assert decision.status == "rollback_recommended"
    bad_trace = __import__("dataclasses").replace(trace, release_id="other")
    with pytest.raises(Exception, match="scope"):
        aggregate_runtime_metrics(
            [bad_trace],
            [],
            (utcnow() - timedelta(minutes=1), utcnow() + timedelta(minutes=1)),
            release,
        )


def test_runtime_failure_detects_latency_cost_l4_failure_and_local_attempt_spikes(
    target_dir: Path, tmp_path: Path
) -> None:
    definition, contract, _ = load_checked_target(target_dir)
    release = Release(
        "rel",
        definition.name,
        definition.contract_hash,
        None,
        None,
        None,
        None,
        utcnow(),
        {"L1": _dummy_artifact("L1", tmp_path, definition.contract_hash), "L2": None, "L3": None},
        RoutingSettings(),
        None,
    )
    attempt = LayerAttemptResult(
        "L1", "artifact-1", "timeout", None, None, "slow_worker", 50.0, "raw timeout"
    )
    cascade = CascadeResult(
        "rel",
        [attempt],
        "ok",
        "L4",
        {"label": "a"},
        0.5,
        250.0,
        fallback_reason="timeout",
    )
    trace = write_trace(
        "t1",
        definition.contract_hash,
        contract,
        RuntimeRequest("req", definition.name, {"text": "a:x"}),
        ServingResult(
            "rel",
            "cascade",
            "ok",
            CacheMiss("k"),
            cascade,
            {"label": "a"},
            "L4",
            0.5,
            250.0,
            None,
            None,
            None,
        ),
        AuditDecisions(None, None, [], 0.0, []),
    )
    metrics = aggregate_runtime_metrics(
        [trace],
        [],
        (utcnow() - timedelta(minutes=1), utcnow() + timedelta(minutes=1)),
        release,
    )
    assert metrics.error_rates["local_timeout"] == 1.0
    assert metrics.latency["p95_latency_ms"] == 250.0
    assert metrics.cost["serving_cost_per_1000"] == 500.0
    metrics = replace(
        metrics,
        error_rates={**metrics.error_rates, "l4_fallback_failure": 0.01},
    )
    decision = detect_runtime_failure(
        metrics,
        release,
        TargetRequirements(p95_latency_ms_max=100, serving_cost_per_1000_max=100),
    )
    assert decision.status == "rollback_recommended"
    assert "local_timeout" in decision.reasons
    assert "l4_fallback_failure" in decision.reasons
    assert "p95_latency_ms_max" in decision.reasons
    assert "serving_cost_per_1000_max" in decision.reasons


def test_cache_hits_are_separate_from_local_coverage_denominator(
    target_dir: Path, tmp_path: Path
) -> None:
    definition, contract, _ = load_checked_target(target_dir)
    release = Release(
        "rel",
        definition.name,
        definition.contract_hash,
        None,
        None,
        None,
        None,
        utcnow(),
        {"L1": _dummy_artifact("L1", tmp_path, definition.contract_hash), "L2": None, "L3": None},
        RoutingSettings(),
        None,
    )
    traces = []
    for index in range(9):
        traces.append(
            write_trace(
                f"cache-{index}",
                definition.contract_hash,
                contract,
                RuntimeRequest(f"cache-req-{index}", definition.name, {"text": "a:x"}),
                ServingResult(
                    "rel",
                    "cache",
                    "ok",
                    CacheHit({"label": "a"}, f"k-{index}"),
                    None,
                    {"label": "a"},
                    "cache",
                    0.0,
                    1.0,
                    None,
                    None,
                    None,
                ),
                AuditDecisions(None, None, [], 0.0, []),
            )
        )
    local_cascade = CascadeResult(
        "rel",
        [
            LayerAttemptResult(
                "L1", "artifact-1", "accept", {"label": "a"}, 0.99, "prefix", 1.0
            )
        ],
        "ok",
        "L1",
        {"label": "a"},
        0.0,
        1.0,
    )
    traces.append(
        write_trace(
            "local-1",
            definition.contract_hash,
            contract,
            RuntimeRequest("local-req", definition.name, {"text": "a:x"}),
            ServingResult(
                "rel",
                "cascade",
                "ok",
                CacheMiss("k-local"),
                local_cascade,
                {"label": "a"},
                "L1",
                0.0,
                1.0,
                None,
                None,
                None,
            ),
            AuditDecisions(None, None, [], 0.0, []),
        )
    )
    metrics = aggregate_runtime_metrics(
        traces,
        [],
        (utcnow() - timedelta(minutes=1), utcnow() + timedelta(minutes=1)),
        release,
    )
    assert metrics.request_count == 10
    assert metrics.cache_hit_rate == 0.9
    assert metrics.local_coverage == 1.0
    assert metrics.layer_coverage["L1"] == 1.0


def test_write_trace_rejects_cache_hit_with_cascade_attempts(target_dir: Path) -> None:
    definition, contract, _ = load_checked_target(target_dir)
    cascade = CascadeResult(
        "rel",
        [LayerAttemptResult("L1", "artifact-1", "accept", {"label": "a"}, 0.99, "prefix", 1.0)],
        "ok",
        "L1",
        {"label": "a"},
        0.0,
        1.0,
    )
    with pytest.raises(Exception, match="cache trace"):
        write_trace(
            "cache-bad",
            definition.contract_hash,
            contract,
            RuntimeRequest("req", definition.name, {"text": "a:x"}),
            ServingResult(
                "rel",
                "cache",
                "ok",
                CacheHit({"label": "a"}, "k"),
                cascade,
                {"label": "a"},
                "cache",
                0.0,
                1.0,
                None,
                None,
                None,
            ),
            AuditDecisions(None, None, [], 0.0, []),
        )


def test_write_trace_rejects_l4_fallback_failure_with_output(target_dir: Path) -> None:
    definition, contract, _ = load_checked_target(target_dir)
    with pytest.raises(Exception, match="fallback failure"):
        write_trace(
            "l4-failed",
            definition.contract_hash,
            contract,
            RuntimeRequest("req", definition.name, {"text": "a:x"}),
            ServingResult(
                "rel",
                "cascade",
                "error",
                CacheMiss("k"),
                None,
                {"label": "a"},
                None,
                0.0,
                1.0,
                "l4_fallback_failure",
                "h",
                None,
            ),
            AuditDecisions(None, None, [], 0.0, []),
        )


def test_runtime_metrics_reject_inactive_layer_attempts(
    target_dir: Path, tmp_path: Path
) -> None:
    definition, contract, _ = load_checked_target(target_dir)
    cascade = CascadeResult(
        "rel",
        [LayerAttemptResult("L1", "artifact-1", "accept", {"label": "a"}, 0.99, "prefix", 1.0)],
        "ok",
        "L1",
        {"label": "a"},
        0.0,
        1.0,
    )
    trace = write_trace(
        "local-1",
        definition.contract_hash,
        contract,
        RuntimeRequest("req", definition.name, {"text": "a:x"}),
        ServingResult(
            "rel",
            "cascade",
            "ok",
            CacheMiss("k"),
            cascade,
            {"label": "a"},
            "L1",
            0.0,
            1.0,
            None,
            None,
            None,
        ),
        AuditDecisions(None, None, [], 0.0, []),
    )
    no_artifact_release = Release(
        "rel",
        definition.name,
        definition.contract_hash,
        None,
        None,
        None,
        None,
        utcnow(),
        {"L1": None, "L2": None, "L3": None},
        RoutingSettings(),
        None,
    )
    with pytest.raises(Exception, match="not enabled"):
        aggregate_runtime_metrics(
            [trace],
            [],
            (utcnow() - timedelta(minutes=1), utcnow() + timedelta(minutes=1)),
            no_artifact_release,
        )
    disabled_release = replace(
        no_artifact_release,
        artifacts={
            "L1": _dummy_artifact("L1", tmp_path, definition.contract_hash),
            "L2": None,
            "L3": None,
        },
        routing=RoutingSettings(enabled_layers=["L2", "L3"]),
    )
    with pytest.raises(Exception, match="not enabled"):
        aggregate_runtime_metrics(
            [trace],
            [],
            (utcnow() - timedelta(minutes=1), utcnow() + timedelta(minutes=1)),
            disabled_release,
        )
    foreign_artifact_trace = replace(
        trace,
        attempts=[
            replace(trace.attempts[0], artifact_id="foreign-artifact"),
        ],
    )
    active_release = replace(
        no_artifact_release,
        artifacts={
            "L1": _dummy_artifact("L1", tmp_path, definition.contract_hash),
            "L2": None,
            "L3": None,
        },
    )
    with pytest.raises(Exception, match="artifact"):
        aggregate_runtime_metrics(
            [foreign_artifact_trace],
            [],
            (utcnow() - timedelta(minutes=1), utcnow() + timedelta(minutes=1)),
            active_release,
        )


def test_local_invalid_output_counts_as_schema_failure_drift(
    target_dir: Path, tmp_path: Path
) -> None:
    definition, contract, _ = load_checked_target(target_dir)
    release = Release(
        "rel",
        definition.name,
        definition.contract_hash,
        None,
        None,
        None,
        None,
        utcnow(),
        {"L1": _dummy_artifact("L1", tmp_path, definition.contract_hash), "L2": None, "L3": None},
        RoutingSettings(),
        None,
    )
    cascade = CascadeResult(
        "rel",
        [
            LayerAttemptResult(
                "L1", "artifact-1", "invalid_output", {"bad": "shape"}, None, "schema", 1.0
            )
        ],
        "ok",
        "L4",
        {"label": "a"},
        0.01,
        3.0,
        fallback_reason="invalid_output",
    )
    trace = write_trace(
        "invalid-1",
        definition.contract_hash,
        contract,
        RuntimeRequest("req", definition.name, {"text": "a:x"}),
        ServingResult(
            "rel",
            "cascade",
            "ok",
            CacheMiss("k"),
            cascade,
            {"label": "a"},
            "L4",
            0.01,
            3.0,
            None,
            None,
            None,
        ),
        AuditDecisions(None, None, [], 0.0, []),
    )
    metrics = aggregate_runtime_metrics(
        [trace],
        [],
        (utcnow() - timedelta(minutes=1), utcnow() + timedelta(minutes=1)),
        release,
    )
    assert metrics.error_rates["local_invalid_output"] == 1.0
    assert metrics.schema_failure_rate == 1.0
    signal = detect_drift(metrics, release, None, {"schema_failure_rate_max": 0.0})
    assert signal.status == "recompile_recommended"
    assert signal.signals["schema_failure_rate"] == 1.0


def _minimal_report(definition, report_id: str = "report") -> Report:
    return Report(
        report_id,
        "final",
        "candidate",
        definition.name,
        definition.contract_hash,
        "snapshot",
        "base",
        {},
        [],
        ReferenceQualificationReport(
            definition.name,
            definition.contract_hash,
            "reference-v1",
            0,
            0,
            None,
            0.0,
            None,
            0.0,
            0.0,
            {},
            {},
            "pass",
        ),
        GeneralizationSummary(None, None, 0.0, 0.0, None, None),
        {},
        CostLedger(),
        {},
        None,
        None,
    )


def test_drift_detects_target_independent_runtime_shifts_and_baseline_scope(
    target_dir: Path,
) -> None:
    definition, _contract, _ = load_checked_target(target_dir)
    release = Release(
        "rel",
        definition.name,
        definition.contract_hash,
        "candidate",
        "snapshot",
        "snapshot-digest",
        "report",
        utcnow(),
        {"L1": None, "L2": None, "L3": None},
        RoutingSettings(),
        None,
    )
    metrics = RuntimeMetricWindow(
        definition.name,
        "rel",
        definition.contract_hash,
        utcnow() - timedelta(minutes=1),
        utcnow(),
        10,
        0.1,
        {"L1": 10, "L2": 0, "L3": 0},
        {"L1": 1, "L2": 0, "L3": 0},
        {"L1": 0.1, "L2": 0.0, "L3": 0.0},
        [],
        0.0,
        0.9,
        None,
        None,
        None,
        None,
        None,
        0,
        0,
        0,
        0,
        0,
        0,
        None,
        {"0.1": 10},
        {"fallback_timeout": 10},
        {"tenant_bucket": {"new": 10}},
        0.3,
        {"avg_ms": 120.0, "p95_latency_ms": 250.0},
        {"total": 2.0, "serving_cost_per_1000": 200.0},
        {},
    )
    report = _minimal_report(definition)
    signal = detect_drift(
        metrics,
        release,
        report,
        {
            "l4_fallback_rate_max": 0.2,
            "local_coverage_min": 0.5,
            "layer_coverage_min": {"L1": 0.5},
            "schema_failure_rate_max": 0.1,
            "latency_avg_ms_max": 50.0,
            "latency_p95_ms_max": 100.0,
            "serving_cost_per_1000_max": 100.0,
            "confidence_histogram_baseline": {"0.9": 10},
            "confidence_histogram_tvd_max": 0.1,
            "reason_code_counts_baseline": {"prefix_match": 10},
            "reason_code_tvd_max": 0.1,
            "source_metadata_counts_baseline": {"tenant_bucket": {"old": 10}},
            "source_metadata_tvd_max": 0.1,
        },
    )
    assert signal.status == "recompile_recommended"
    assert "local_coverage" in signal.signals
    assert "layer_coverage.L1" in signal.signals
    assert "confidence_histogram_tvd" in signal.signals
    assert "reason_code_tvd" in signal.signals
    assert "source_metadata_tvd.tenant_bucket" in signal.signals
    assert "latency.p95_latency_ms" in signal.signals
    assert "cost.serving_cost_per_1000" in signal.signals

    with pytest.raises(Exception, match="baseline report scope"):
        detect_drift(metrics, release, replace(report, target_name="other"), {})
