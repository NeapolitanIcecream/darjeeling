# Runtime Trace And Metrics Module

## Purpose

The Runtime Trace And Metrics module owns durable runtime traces, aggregate runtime metrics, online quality summaries, drift detection, and runtime failure decisions. It records enough information to monitor a Release without storing raw runtime payloads.

This module follows the [Runtime Feedback Overview](07_runtime_feedback_overview.md) data lifetime model and the [System Invariants](../00_overall_design.md#system-invariants).

## Boundary

Inputs:

- `RuntimeRequest` and `ServingResult` from Release Runtime.
- Selected Release metadata from Release Runtime.
- `AuditDecisions` from Audit Monitoring.
- `AuditRecord` values from Audit Monitoring for aggregate random-audit metrics.
- `TargetRuntimeContract` and redaction functions from Target Definition.
- Runtime requirements from Target Definition.
- Baseline `Report` from Candidate Evaluation when the current Release has compile provenance.

Outputs:

- `Trace` records to durable trace storage.
- `RuntimeMetricWindow` to CLI, reports, monitoring, Audit Monitoring, and Telemetry Evidence And Recompile.
- `RuntimeFailureDecision` to Release Runtime rollback automation.
- `DriftSignal` to Telemetry Evidence And Recompile.
- `OnlineQualitySummary` to CLI and dashboards.

This module must not expose raw runtime payloads to reports, agents, or future Snapshots.

## Data Types

### `Trace`

Fields:

- `trace_id: str`
- `request_id_hash: str`
- `target_name: str`
- `contract_hash: str`
- `release_id: str`
- `input_redacted: dict | None`
- `input_hash: str`
- `attempts: list[TraceLayerAttempt]`
- `cache_hit: bool`
- `status: Literal["ok", "error"]`
- `chosen_layer: Literal["cache", "L1", "L2", "L3", "L4"] | None`
- `final_output_redacted: dict | None`
- `final_output_hash: str | None`
- `error_type: Literal["deadline_exceeded", "l4_fallback_failure", "no_valid_output", "runtime_error"] | None`
- `error_message_hash: str | None`
- `serving_cost: float`
- `latency_ms: float`
- `random_audit_probability: float`
- `risk_audit_flags: list[str]`
- `selected_audit_types: list[Literal["random", "risk"]]`
- `timestamp: datetime`
- `metadata_buckets: dict`

`contract_hash` is the target contract version identifier used by trace, audit, and runtime metric records. It is the machine-verifiable form of the target version in the source design.

Trace side fields follow the durable runtime privacy invariant. `request_id_hash` is a scoped stable hash, `metadata_buckets` comes from Target Definition `bucket_runtime_metadata`, and `risk_audit_flags` are bounded allowlisted codes from Audit Monitoring.

### `TraceLayerAttempt`

Fields:

- `layer: Literal["L1", "L2", "L3"]`
- `artifact_id: str`
- `decision: Literal["accept", "abstain", "error", "timeout", "invalid_output", "protocol_error"]`
- `output_redacted: dict | None`
- `output_hash: str | None`
- `confidence: float | None`
- `reason_code: str | None`
- `latency_ms: float`
- `error_type: str | None`
- `error_message_hash: str | None`

Trace attempts never store raw layer outputs. Core redacts each attempted output with the target contract before writing trace data.

`reason_code` must be the validated worker `reason` code: bounded, identifier-like, and allowlisted by the artifact manifest when an allowlist is present. `error_type` is a stable error class, and `error_message_hash` is the only durable representation of raw error text.

### `RuntimeMetricWindow`

Fields:

- `target_name: str`
- `release_id: str`
- `contract_hash: str`
- `window_start: datetime`
- `window_end: datetime`
- `request_count: int`
- `local_coverage: float`
- `layer_attempt_counts: dict[str, int]`
- `layer_accept_counts: dict[str, int]`
- `layer_coverage: dict[str, float]`
- `configured_disabled_layers: list[LayerName]`
- `cache_hit_rate: float`
- `l4_fallback_rate: float`
- `random_audit_precision: float | None`
- `random_audit_precision_lower_bound: float | None`
- `random_audit_precision_confidence_level: float | None`
- `wrong_accept_estimate: float | None`
- `wrong_accept_rate_upper_bound: float | None`
- `random_audit_attempt_count: int`
- `random_audit_success_count: int`
- `random_audit_evaluable_count: int`
- `random_audit_correct_count: int`
- `random_audit_wrong_count: int`
- `random_audit_reference_failure_count: int`
- `random_audit_reference_failure_rate: float | None`
- `confidence_histogram: dict[str, int]`
- `reason_code_counts: dict[str, int]`
- `source_metadata_counts: dict[str, dict[str, int]]`
- `schema_failure_rate: float`
- `latency: dict`
- `cost: dict`
- `error_rates: dict`

### `RuntimeFailureDecision`

Fields:

- `target_name: str`
- `release_id: str`
- `contract_hash: str`
- `status: Literal["ok", "circuit_breaker", "rollback_recommended"]`
- `reasons: list[str]`
- `triggered_metrics: dict`
- `decided_at: datetime`

### `DriftSignal`

Fields:

- `target_name: str`
- `release_id: str`
- `contract_hash: str`
- `status: Literal["none", "watch", "recompile_recommended"]`
- `signals: dict`
- `compared_to_report_id: str | None`
- `detected_at: datetime`

### `OnlineQualitySummary`

Fields:

- `release_id: str`
- `metrics: RuntimeMetricWindow`
- `drift_status: str`
- `failure_status: str`
- `generated_at: datetime`

## Functions

### `write_trace`

Input:

- `trace_id: str`
- `contract_hash: str`
- `contract: TargetRuntimeContract`
- `runtime_request: RuntimeRequest`
- `serving_result: ServingResult`
- `audit_decisions: AuditDecisions`

Output:

- `Trace`

Purpose:

- Redact input and output using target contract.
- Set `Trace.contract_hash` from the explicit input. Release Runtime is responsible for passing the selected Release's `contract_hash`; `write_trace` does not infer it from `serving_result`.
- Redact or hash every layer attempt output before storing it as `TraceLayerAttempt`.
- Hash the runtime request id before durable trace write.
- Call Target Definition `bucket_runtime_metadata` and store only the returned allowlisted or bucketed metadata values.
- Record release id, cache hit status, redacted layer attempts when any, chosen layer, final status, latency, cost, random audit probability, selected audit types, risk flags, and runtime failure class.
- For cache hits, write an empty `attempts` list and `chosen_layer: "cache"`.
- For L4 fallback failures, write `status: "error"`, leave final output fields empty, and preserve the failure class without raw provider error text.
- Enforce the durable runtime privacy invariant for all persisted fields.

Used by:

- Release Runtime `serve_request`.

### `redact_layer_attempts_for_trace`

Input:

- `contract: TargetRuntimeContract`
- `attempts: list[LayerAttemptResult]`

Output:

- `list[TraceLayerAttempt]`

Purpose:

- Convert raw runtime attempt results into trace-safe attempt records.
- Apply target redaction to accepted outputs and hash outputs when redaction policy requires it.
- Preserve enough decision, latency, error type/hash, confidence, and reason-code data for debugging and metrics.

Used by:

- `write_trace`.

### `aggregate_runtime_metrics`

Input:

- `traces: Iterable[Trace]`
- `audits: Iterable[AuditRecord]`
- `window: TimeWindow`
- `release: Release`

Output:

- `RuntimeMetricWindow`

Purpose:

- Compute local Coverage, cache hit rate, L4 fallback rate, random-audit Precision, lower bound, wrong-accept estimate, upper bound, latency, cost, and error rates.
- Set `RuntimeMetricWindow.target_name`, `release_id`, and `contract_hash` from the supplied Release.
- Enforce the target and contract scope invariant for all included Trace and AuditRecord rows.
- Compute per-layer attempt counts, accept counts, and Coverage for L1, L2, and L3 using the Release's enabled route.
- Record configured-disabled layers separately so zero attempts from user configuration are not confused with abstention, health failures, or missing artifacts.
- When a Release has no lower-layer artifacts, report zero L1/L2/L3 attempts and Coverage, and report direct L4 usage through fallback rate, latency, and cost.
- Keep risk-audit findings separate from unbiased online Precision estimates.
- Do not silently drop selected random-audit samples whose reference call failed. Report attempted audit count, successful audit count, reference failure count, and reference failure rate separately from Precision.
- Build target-independent drift summaries from trace-safe fields: confidence histogram, reason-code counts, source metadata counts, schema failure rate, latency, and cost.

Used by:

- CLI `status`.
- monitoring dashboard.
- recompile trigger checks.

### `detect_runtime_failure`

Input:

- `metrics: RuntimeMetricWindow`
- `release: Release`
- `requirements: TargetRequirements`

Output:

- `RuntimeFailureDecision`

Purpose:

- Detect hard failures: random audit Precision lower bound below requirement, wrong accept upper bound too high, random audit reference failure rate above requirement, crash/timeout spikes, schema failures, latency/cost budget violations, or L4 fallback failure.
- Require `metrics.target_name`, `metrics.release_id`, and `metrics.contract_hash` to match the supplied Release before making a decision.
- Use the lower bound and upper bound fields from `RuntimeMetricWindow`; do not compare rollback policy against the point estimate alone.
- Treat excessive random audit reference failures as a runtime failure because online Precision can no longer be estimated reliably.
- Recommend circuit breaker or rollback when needed.

Used by:

- Release Runtime rollback automation.
- monitoring.

### `detect_drift`

Input:

- `metrics: RuntimeMetricWindow`
- `release: Release`
- `baseline_report: Report | None`
- `drift_options: DriftOptions`

Output:

- `DriftSignal`

Purpose:

- Detect target-independent drift signals from `RuntimeMetricWindow`: fallback rise, local or per-layer Coverage drop, confidence histogram shift, reason-code count shift, source metadata count shift, schema failure rate shift, latency shift, and cost shift.
- Require `metrics.target_name`, `metrics.release_id`, and `metrics.contract_hash` to match the supplied Release before making a drift decision.
- When the supplied Release has compile provenance, require it to point at `baseline_report.report_id`, and require the Report target and contract to match the Release.
- Hard-fail if the supplied Release has compile provenance and `baseline_report` is missing.
- When the supplied Release has no compile provenance, do not require a fake Report; detect only target-independent runtime shifts against drift options and accumulated runtime windows.
- Treat reason codes and source metadata bucket keys as opaque buckets; do not interpret target semantic labels in Core.

Used by:

- Telemetry Evidence And Recompile.
- monitoring automation.

### `summarize_online_quality`

Input:

- `metrics: RuntimeMetricWindow`
- `release: Release`
- `report: Report | None`

Output:

- `OnlineQualitySummary`

Purpose:

- Present online Precision, Coverage, Generalization drift, fallback, latency, and cost in the same vocabulary as compile reports.
- For a Release with no compile Report, present direct-L4 runtime quality, local Coverage as zero, fallback, latency, and cost without inventing Report-derived claims.
- Keep cache hit rate separate.

Used by:

- CLI `status`.
- dashboard/report pages.

## Invariants

- Runtime traces include release id and redacted layer attempts.
- Cache-hit traces are valid traces with empty layer attempts.
- Trace redaction is target-owned and Core-enforced.
- Runtime failure decisions use random audit lower bounds and wrong-accept upper bounds, not point estimates alone.
- Core drift detection uses target-independent aggregate buckets only.
