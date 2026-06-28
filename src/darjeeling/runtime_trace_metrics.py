from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import datetime
from math import ceil
from typing import Any

from darjeeling.errors import RuntimeErrorSafe
from darjeeling.model import (
    AuditDecisions,
    AuditRecord,
    DriftSignal,
    OnlineQualitySummary,
    Release,
    Report,
    RuntimeFailureDecision,
    RuntimeMetricWindow,
    RuntimeRequest,
    ServingResult,
    TargetRequirements,
    TargetRuntimeContract,
    Trace,
    TraceLayerAttempt,
)
from darjeeling.util import scoped_hash, stable_hash, utcnow


def redact_layer_attempts_for_trace(
    contract: TargetRuntimeContract,
    attempts: list,
) -> list[TraceLayerAttempt]:
    trace_attempts: list[TraceLayerAttempt] = []
    for attempt in attempts:
        output_redacted = (
            contract.redact_for_trace(attempt.output) if attempt.output is not None else None
        )
        trace_attempts.append(
            TraceLayerAttempt(
                layer=attempt.layer,
                artifact_id=attempt.artifact_id,
                decision=attempt.decision,
                output_redacted=output_redacted,
                output_hash=stable_hash(attempt.output) if attempt.output is not None else None,
                confidence=attempt.confidence,
                reason_code=attempt.reason,
                latency_ms=attempt.latency_ms,
                error_type=_trace_attempt_error_type(attempt),
                error_message_hash=stable_hash(attempt.error) if attempt.error else None,
            )
        )
    return trace_attempts


def _trace_attempt_error_type(attempt: Any) -> str | None:
    if attempt.decision == "error" and attempt.reason in {"health_failure", "circuit_open"}:
        return attempt.reason
    if attempt.decision in {"error", "timeout", "invalid_output", "protocol_error"}:
        return attempt.decision
    return None


def write_trace(
    trace_id: str,
    contract_hash: str,
    contract: TargetRuntimeContract,
    runtime_request: RuntimeRequest,
    serving_result: ServingResult,
    audit_decisions: AuditDecisions,
) -> Trace:
    attempts = []
    if serving_result.path == "cache":
        if serving_result.cascade_result is not None or serving_result.chosen_layer != "cache":
            raise RuntimeErrorSafe("cache trace cannot include cascade attempts")
    elif serving_result.chosen_layer == "cache":
        raise RuntimeErrorSafe("cascade trace cannot use cache as chosen layer")
    elif serving_result.cascade_result is not None:
        attempts = redact_layer_attempts_for_trace(contract, serving_result.cascade_result.attempts)
    if serving_result.error_type == "l4_fallback_failure" and serving_result.output is not None:
        raise RuntimeErrorSafe("L4 fallback failure trace cannot include final output")
    final_output = (
        contract.redact_for_trace(serving_result.output)
        if serving_result.output is not None
        else None
    )
    trace = Trace(
        trace_id=trace_id,
        request_id_hash=scoped_hash("request", runtime_request.request_id),
        target_name=runtime_request.target_name,
        contract_hash=contract_hash,
        release_id=serving_result.release_id,
        input_redacted=contract.redact_for_trace(runtime_request.input),
        input_hash=stable_hash(runtime_request.input),
        attempts=attempts,
        cache_hit=serving_result.path == "cache",
        status=serving_result.status,
        chosen_layer=serving_result.chosen_layer,
        final_output_redacted=final_output,
        final_output_hash=stable_hash(serving_result.output)
        if serving_result.output is not None
        else None,
        error_type=serving_result.error_type,
        error_message_hash=serving_result.error_message_hash,
        serving_cost=serving_result.serving_cost,
        latency_ms=serving_result.latency_ms,
        random_audit_probability=audit_decisions.random_sampling_probability,
        risk_audit_flags=audit_decisions.risk_flags,
        selected_audit_types=audit_decisions.selected_audit_types,
        timestamp=utcnow(),
        metadata_buckets=contract.bucket_runtime_metadata(runtime_request.metadata),
    )
    raw_text = repr(trace)
    if "provider secret" in raw_text.lower():
        raise RuntimeErrorSafe("raw provider text leaked into trace")
    return trace


def _scope_check_trace(trace: Trace, release: Release) -> None:
    if (
        trace.target_name != release.target_name
        or trace.release_id != release.release_id
        or trace.contract_hash != release.contract_hash
    ):
        raise RuntimeErrorSafe("trace scope does not match release")


def _scope_check_audit(audit: AuditRecord, release: Release) -> None:
    if (
        audit.target_name != release.target_name
        or audit.release_id != release.release_id
        or audit.contract_hash != release.contract_hash
    ):
        raise RuntimeErrorSafe("audit scope does not match release")


def _nearest_rank_percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def aggregate_runtime_metrics(
    traces: Iterable[Trace],
    audits: Iterable[AuditRecord],
    window: tuple[datetime, datetime],
    release: Release,
) -> RuntimeMetricWindow:
    traces = [trace for trace in traces if window[0] <= trace.timestamp <= window[1]]
    audits = [audit for audit in audits if window[0] <= audit.created_at <= window[1]]
    for trace in traces:
        _scope_check_trace(trace, release)
    for audit in audits:
        _scope_check_audit(audit, release)
    request_count = len(traces)
    cache_hits = sum(1 for trace in traces if trace.cache_hit)
    non_cache_count = request_count - cache_hits
    active_artifacts = {
        layer: artifact
        for layer in ["L1", "L2", "L3"]
        if layer in release.routing.enabled_layers
        and (artifact := release.artifacts.get(layer)) is not None
    }
    active_layers = set(active_artifacts)
    for trace in traces:
        if trace.chosen_layer in {"L1", "L2", "L3"} and trace.chosen_layer not in active_layers:
            raise RuntimeErrorSafe("trace layer is not enabled for release")
    local_accepts = sum(1 for trace in traces if trace.chosen_layer in active_layers)
    l4_count = sum(1 for trace in traces if trace.chosen_layer == "L4")
    attempt_counts = {layer: 0 for layer in ["L1", "L2", "L3"]}
    accept_counts = {layer: 0 for layer in ["L1", "L2", "L3"]}
    confidence_histogram: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    latencies = []
    costs = []
    source_metadata_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for trace in traces:
        latencies.append(trace.latency_ms)
        costs.append(trace.serving_cost)
        if trace.error_type:
            errors[trace.error_type] += 1
        for key, value in trace.metadata_buckets.items():
            source_metadata_counts[key][str(value)] += 1
        for attempt in trace.attempts:
            if attempt.layer not in active_layers:
                raise RuntimeErrorSafe("trace layer is not enabled for release")
            if attempt.artifact_id != active_artifacts[attempt.layer].artifact_id:
                raise RuntimeErrorSafe("trace artifact does not match release")
            attempt_counts[attempt.layer] += 1
            if attempt.decision == "accept":
                accept_counts[attempt.layer] += 1
            if attempt.error_type:
                errors[f"local_{attempt.error_type}"] += 1
            if attempt.confidence is not None:
                bucket = f"{int(attempt.confidence * 10) / 10:.1f}"
                confidence_histogram[bucket] += 1
            if attempt.reason_code:
                reason_counts[attempt.reason_code] += 1
    random_audits = [audit for audit in audits if audit.audit_type == "random"]
    random_success = [audit for audit in random_audits if audit.status == "ok"]
    evaluable = [audit for audit in random_success if audit.is_correct is not None]
    correct = [audit for audit in evaluable if audit.is_correct is True]
    wrong = [audit for audit in evaluable if audit.is_correct is False]
    precision = len(correct) / len(evaluable) if evaluable else None
    reference_failures = [audit for audit in random_audits if audit.status == "error"]
    schema_failure_count = errors.get("no_valid_output", 0) + errors.get(
        "local_invalid_output", 0
    )
    return RuntimeMetricWindow(
        target_name=release.target_name,
        release_id=release.release_id,
        contract_hash=release.contract_hash,
        window_start=window[0],
        window_end=window[1],
        request_count=request_count,
        local_coverage=local_accepts / non_cache_count if non_cache_count else 0.0,
        layer_attempt_counts=attempt_counts,
        layer_accept_counts=accept_counts,
        layer_coverage={
            layer: accept_counts[layer] / non_cache_count if non_cache_count else 0.0
            for layer in attempt_counts
        },
        configured_disabled_layers=[
            layer for layer in ["L1", "L2", "L3"] if layer not in release.routing.enabled_layers
        ],  # type: ignore[list-item]
        cache_hit_rate=cache_hits / request_count if request_count else 0.0,
        l4_fallback_rate=l4_count / request_count if request_count else 0.0,
        random_audit_precision=precision,
        random_audit_precision_lower_bound=max(0.0, precision - 0.05)
        if precision is not None
        else None,
        random_audit_precision_confidence_level=0.95 if evaluable else None,
        wrong_accept_estimate=len(wrong) / len(evaluable) if evaluable else None,
        wrong_accept_rate_upper_bound=min(1.0, (len(wrong) / len(evaluable)) + 0.05)
        if evaluable
        else None,
        random_audit_attempt_count=len(random_audits),
        random_audit_success_count=len(random_success),
        random_audit_evaluable_count=len(evaluable),
        random_audit_correct_count=len(correct),
        random_audit_wrong_count=len(wrong),
        random_audit_reference_failure_count=len(reference_failures),
        random_audit_reference_failure_rate=len(reference_failures) / len(random_audits)
        if random_audits
        else None,
        confidence_histogram=dict(confidence_histogram),
        reason_code_counts=dict(reason_counts),
        source_metadata_counts={key: dict(value) for key, value in source_metadata_counts.items()},
        schema_failure_rate=schema_failure_count / non_cache_count
        if non_cache_count
        else 0.0,
        latency={
            "avg_ms": sum(latencies) / len(latencies) if latencies else 0.0,
            "p95_latency_ms": _nearest_rank_percentile(latencies, 0.95),
        },
        cost={
            "total": sum(costs),
            "serving_cost_per_1000": (sum(costs) / request_count * 1000.0)
            if request_count
            else 0.0,
        },
        error_rates={
            key: value / request_count if request_count else 0.0 for key, value in errors.items()
        },
    )


def _check_metric_scope(metrics: RuntimeMetricWindow, release: Release) -> None:
    if (
        metrics.target_name != release.target_name
        or metrics.release_id != release.release_id
        or metrics.contract_hash != release.contract_hash
    ):
        raise RuntimeErrorSafe("metric window does not match release")


def detect_runtime_failure(
    metrics: RuntimeMetricWindow,
    release: Release,
    requirements: TargetRequirements,
) -> RuntimeFailureDecision:
    _check_metric_scope(metrics, release)
    reasons: list[str] = []
    triggered: dict[str, float] = {}

    def add_reason(name: str, value: float) -> None:
        if name not in reasons:
            reasons.append(name)
        triggered[name] = value

    if (
        metrics.random_audit_precision_lower_bound is not None
        and metrics.random_audit_precision_lower_bound < requirements.precision_min
    ):
        add_reason(
            "random_audit_precision_lower_bound", metrics.random_audit_precision_lower_bound
        )
    if (
        requirements.wrong_accept_rate_max is not None
        and metrics.wrong_accept_rate_upper_bound is not None
        and metrics.wrong_accept_rate_upper_bound > requirements.wrong_accept_rate_max
    ):
        add_reason("wrong_accept_rate_upper_bound", metrics.wrong_accept_rate_upper_bound)
    if (
        requirements.random_audit_reference_failure_rate_max is not None
        and metrics.random_audit_reference_failure_rate is not None
        and metrics.random_audit_reference_failure_rate
        > requirements.random_audit_reference_failure_rate_max
    ):
        add_reason(
            "random_audit_reference_failure_rate",
            metrics.random_audit_reference_failure_rate,
        )
    p95_latency = metrics.latency.get("p95_latency_ms")
    if (
        requirements.p95_latency_ms_max is not None
        and p95_latency is not None
        and p95_latency > requirements.p95_latency_ms_max
    ):
        add_reason("p95_latency_ms_max", float(p95_latency))
    serving_cost_per_1000 = metrics.cost.get("serving_cost_per_1000")
    if (
        requirements.serving_cost_per_1000_max is not None
        and serving_cost_per_1000 is not None
        and serving_cost_per_1000 > requirements.serving_cost_per_1000_max
    ):
        add_reason("serving_cost_per_1000_max", float(serving_cost_per_1000))
    l4_fallback_failure_rate = metrics.error_rates.get("l4_fallback_failure", 0.0)
    if l4_fallback_failure_rate > 0.0:
        add_reason("l4_fallback_failure", l4_fallback_failure_rate)
    for key, rate in metrics.error_rates.items():
        if rate > 0.2:
            add_reason(key, rate)
    return RuntimeFailureDecision(
        target_name=release.target_name,
        release_id=release.release_id,
        contract_hash=release.contract_hash,
        status="rollback_recommended" if reasons else "ok",
        reasons=reasons,
        triggered_metrics=triggered,
        decided_at=utcnow(),
    )


def detect_drift(
    metrics: RuntimeMetricWindow,
    release: Release,
    baseline_report: Report | None,
    drift_options: dict,
) -> DriftSignal:
    _check_metric_scope(metrics, release)
    if release.report_id is not None:
        if baseline_report is None:
            raise RuntimeErrorSafe("compiled release drift detection requires baseline report")
        if (
            baseline_report.report_id != release.report_id
            or baseline_report.target_name != release.target_name
            or baseline_report.contract_hash != release.contract_hash
        ):
            raise RuntimeErrorSafe("baseline report scope does not match release")
    signals = {}
    fallback_threshold = drift_options.get("l4_fallback_rate_max")
    if fallback_threshold is not None and metrics.l4_fallback_rate > fallback_threshold:
        signals["l4_fallback_rate"] = metrics.l4_fallback_rate
    schema_threshold = drift_options.get("schema_failure_rate_max")
    if schema_threshold is not None and metrics.schema_failure_rate > schema_threshold:
        signals["schema_failure_rate"] = metrics.schema_failure_rate
    local_coverage_min = drift_options.get("local_coverage_min")
    if local_coverage_min is not None and metrics.local_coverage < local_coverage_min:
        signals["local_coverage"] = metrics.local_coverage
    layer_coverage_min = drift_options.get("layer_coverage_min")
    if isinstance(layer_coverage_min, dict):
        for layer, threshold in layer_coverage_min.items():
            coverage = metrics.layer_coverage.get(layer)
            if coverage is not None and coverage < threshold:
                signals[f"layer_coverage.{layer}"] = coverage
    elif layer_coverage_min is not None:
        for layer, coverage in metrics.layer_coverage.items():
            if coverage < layer_coverage_min:
                signals[f"layer_coverage.{layer}"] = coverage
    latency_avg_max = drift_options.get("latency_avg_ms_max")
    if latency_avg_max is not None and metrics.latency.get("avg_ms", 0.0) > latency_avg_max:
        signals["latency.avg_ms"] = metrics.latency.get("avg_ms", 0.0)
    latency_p95_max = drift_options.get("latency_p95_ms_max")
    if (
        latency_p95_max is not None
        and metrics.latency.get("p95_latency_ms", 0.0) > latency_p95_max
    ):
        signals["latency.p95_latency_ms"] = metrics.latency.get("p95_latency_ms", 0.0)
    cost_per_1000_max = drift_options.get("serving_cost_per_1000_max")
    if (
        cost_per_1000_max is not None
        and metrics.cost.get("serving_cost_per_1000", 0.0) > cost_per_1000_max
    ):
        signals["cost.serving_cost_per_1000"] = metrics.cost.get("serving_cost_per_1000", 0.0)
    confidence_shift = _distribution_distance(
        metrics.confidence_histogram, drift_options.get("confidence_histogram_baseline")
    )
    confidence_shift_max = drift_options.get("confidence_histogram_tvd_max")
    if confidence_shift_max is not None and confidence_shift > confidence_shift_max:
        signals["confidence_histogram_tvd"] = confidence_shift
    reason_shift = _distribution_distance(
        metrics.reason_code_counts, drift_options.get("reason_code_counts_baseline")
    )
    reason_shift_max = drift_options.get("reason_code_tvd_max")
    if reason_shift_max is not None and reason_shift > reason_shift_max:
        signals["reason_code_tvd"] = reason_shift
    source_shift_max = drift_options.get("source_metadata_tvd_max")
    if source_shift_max is not None:
        for key, current_counts in metrics.source_metadata_counts.items():
            baseline_counts = drift_options.get("source_metadata_counts_baseline", {}).get(key)
            distance = _distribution_distance(current_counts, baseline_counts)
            if distance > source_shift_max:
                signals[f"source_metadata_tvd.{key}"] = distance
    status = "recompile_recommended" if signals else "none"
    return DriftSignal(
        target_name=release.target_name,
        release_id=release.release_id,
        contract_hash=release.contract_hash,
        status=status,
        signals=signals,
        compared_to_report_id=baseline_report.report_id if baseline_report else None,
        detected_at=utcnow(),
    )


def _distribution_distance(current: dict[str, int], baseline: Any) -> float:
    if not isinstance(baseline, dict):
        return 0.0
    current_total = sum(current.values())
    baseline_total = sum(value for value in baseline.values() if isinstance(value, int | float))
    if current_total <= 0 and baseline_total <= 0:
        return 0.0
    keys = set(current) | set(baseline)
    distance = 0.0
    for key in keys:
        current_share = current.get(key, 0) / current_total if current_total else 0.0
        baseline_value = baseline.get(key, 0)
        if not isinstance(baseline_value, int | float):
            baseline_value = 0
        baseline_share = baseline_value / baseline_total if baseline_total else 0.0
        distance += abs(current_share - baseline_share)
    return distance / 2.0


def summarize_online_quality(
    metrics: RuntimeMetricWindow,
    release: Release,
    report: Report | None,
) -> OnlineQualitySummary:
    _check_metric_scope(metrics, release)
    if release.report_id is not None and (report is None or report.report_id != release.report_id):
        raise RuntimeErrorSafe("compiled release online summary requires matching report")
    failure = detect_runtime_failure(metrics, release, TargetRequirements()).status
    drift = detect_drift(metrics, release, report, {}).status
    return OnlineQualitySummary(
        release_id=release.release_id,
        metrics=metrics,
        drift_status=drift,
        failure_status=failure,
        generated_at=utcnow(),
    )
