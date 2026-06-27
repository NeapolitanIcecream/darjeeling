# Audit Monitoring Module

## Purpose

The Audit Monitoring module selects random and risk audits, captures short-lived raw payloads only when an audit needs them, calls the Core-owned reference broker for audit, and writes durable audit records and summaries.

Random audit estimates online Precision. Risk audit finds problems. They must stay separate in storage, reports, and runtime decisions.

This module follows the [System Invariants](../00_overall_design.md#system-invariants), especially same-request evidence and durable runtime privacy.

This module follows the [Runtime Feedback Overview](07_runtime_feedback_overview.md) data lifetime model.

## Boundary

Inputs:

- `Trace` records and `ServingResult` from Runtime Trace And Metrics and Release Runtime.
- Audit settings from Release Runtime and Target Definition.
- Risk audit rules from target requirements or deployment config.
- `TargetRuntimeContract` and redaction functions from Target Definition.
- Reference broker from Snapshot And Reference.

Outputs:

- `AuditDecisions` to Release Runtime and Runtime Trace And Metrics.
- `SecureAuditPayload` to short-lived secure storage.
- `AuditRecord` values to durable audit storage and Runtime Trace And Metrics.
- `AuditReferenceResult` to Telemetry Evidence And Recompile for immediate evidence approval.
- `AuditSummary` to reports and monitoring.

This module must not write raw audit inputs or outputs to durable trace or audit storage.

## Data Types

### `AuditDecision`

Fields:

- `audit_type: Literal["random", "risk"]`
- `selected: bool`
- `sampling_probability: float | None`
- `risk_flags: list[str]`
- `reason: str | None`

### `AuditDecisions`

Fields:

- `random: AuditDecision | None`
- `synchronous_risk: AuditDecision | None`
- `selected_audit_types: list[Literal["random", "risk"]]`
- `random_sampling_probability: float`
- `risk_flags: list[str]`

Random and risk audit decisions are recorded separately because random audit supports unbiased Precision estimates while risk audit is biased diagnostic evidence. A request may have both.

`risk_flags` must be bounded allowlisted codes from `RiskAuditRules` and follow the durable runtime privacy invariant; detailed risk diagnostics belong in short-lived audit context or redacted reports, not durable Trace.

### `SecureAuditPayload`

Fields:

- `trace_id: str`
- `release_id: str`
- `input_raw: dict`
- `lower_layer_output_raw: dict | None`
- `expires_at: datetime`
- `storage_policy: Literal["memory_only", "encrypted_short_lived"]`

`SecureAuditPayload` exists only long enough to run reference labeling and correctness comparison. It is not a Trace and is not part of the long-term audit record.

### `AuditRecord`

Fields:

- `audit_id: str`
- `trace_id: str`
- `target_name: str`
- `release_id: str`
- `contract_hash: str`
- `audit_type: Literal["random", "risk"]`
- `status: Literal["ok", "error"]`
- `sampling_probability: float | None`
- `reference_output_redacted: dict | None`
- `reference_output_hash: str | None`
- `reference_source: str | None`
- `lower_layer_output_redacted: dict | None`
- `lower_layer_output_hash: str | None`
- `is_correct: bool | None`
- `error_type: Literal["timeout", "rate_limited", "empty_response", "parse_failure", "validation_failure", "provider_error"] | None`
- `error_message_hash: str | None`
- `cost: float`
- `created_at: datetime`

`AuditRecord` is written for every selected audit, including reference-call failures. A failed audit has `status: "error"`, no reference output, no correctness claim, and enough error class/cost data for audit failure reporting. `target_name`, `release_id`, and `contract_hash` are copied from the matching Trace. User feedback is not an audit type; it enters through `UserFeedbackRecord` and `approve_telemetry_evidence`.

### `AuditReferenceResult`

Fields:

- `status: Literal["ok", "error"]`
- `audit_record: AuditRecord`
- `reference_output_raw: dict | None`
- `reference_source: str | None`
- `cost: float`
- `latency_ms: float`
- `error_type: Literal["timeout", "rate_limited", "empty_response", "parse_failure", "validation_failure", "provider_error"] | None`
- `error_message_hash: str | None`
- `expires_at: datetime`

This is a short-lived in-memory result from an audit reference call. On success, it exists so approved telemetry evidence can be created while the raw reference output is still available. On failure, it preserves the failed attempt's cost, latency, and error class without pretending a correctness comparison happened. It is not written to durable trace storage.

### `AuditSummary`

Fields:

- `target_name: str`
- `release_id: str`
- `contract_hash: str`
- `window_start: datetime`
- `window_end: datetime`
- `random_attempt_count: int`
- `random_success_count: int`
- `random_reference_failure_count: int`
- `random_reference_failure_rate: float | None`
- `random_precision: float | None`
- `random_precision_lower_bound: float | None`
- `random_precision_confidence_level: float | None`
- `random_wrong_accept_rate_upper_bound: float | None`
- `risk_attempt_count: int`
- `risk_success_count: int`
- `risk_reference_failure_count: int`
- `risk_findings_by_flag: dict[str, int]`
- `cost: dict`
- `generated_at: datetime`

`AuditSummary` is the durable report shape for audit monitoring. It keeps random audit estimates separate from risk audit diagnostics and reports reference-call failures separately from Precision.

## Functions

### `decide_random_audit`

Input:

- `release: Release`
- `request: RuntimeRequest`
- `serving_result: ServingResult`
- `audit_settings: AuditSettings`

Output:

- `AuditDecision`

Purpose:

- Select lower-layer accepted requests for unbiased random audit using a fixed sampling probability.
- Do not treat cache hits as lower-layer accepts; cache audits, if enabled, are reported separately from local Coverage.
- Record the probability so online Precision estimates can be corrected.

Used by:

- Release Runtime.

### `decide_synchronous_risk_audit`

Input:

- `release: Release`
- `request: RuntimeRequest`
- `serving_result: ServingResult`
- `risk_rules: RiskAuditRules`

Output:

- `AuditDecision`

Purpose:

- Select suspicious requests for immediate risk audit while raw input and lower-layer output are still available.
- Mark these audits as biased risk audits.
- Emit only bounded allowlisted risk flag codes.
- Enable `capture_secure_audit_payload` so correctness comparison can use the original lower-layer output.

Used by:

- Release Runtime `serve_request`.

### `combine_audit_decisions`

Input:

- `random_decision: AuditDecision | None`
- `risk_decision: AuditDecision | None`

Output:

- `AuditDecisions`

Purpose:

- Preserve random and risk audit decisions separately while giving runtime one object to pass to trace writing, secure payload capture, and audit queueing.
- Keep random sampling probability separate from risk flags.

Used by:

- Release Runtime `serve_request`.

### `capture_secure_audit_payload`

Input:

- `trace_id: str`
- `runtime_request: RuntimeRequest`
- `serving_result: ServingResult`
- `audit_decisions: AuditDecisions`
- `secure_store: SecureAuditStore`

Output:

- `SecureAuditPayload | None`

Purpose:

- Capture raw input and raw lower-layer output only when at least one random or risk audit has been selected.
- Store the payload in memory or encrypted short-lived storage with an expiry.
- Keep this payload separate from Trace so redacted trace retention does not need raw outputs.

Used by:

- Release Runtime `serve_request`.
- `run_audit_reference_call`.

### `run_audit_reference_call`

Input:

- `trace: Trace`
- `secure_payload: SecureAuditPayload`
- `audit_type: Literal["random", "risk"]`
- `contract: TargetRuntimeContract`
- `broker: ReferenceBroker`

Output:

- `AuditReferenceResult`

Purpose:

- Enforce the same-request evidence invariant before calling the reference broker.
- Obtain reference output through Core broker only after the Trace, secure payload, and audit type have been matched to the same selected request.
- Compare raw lower-layer output from `SecureAuditPayload` to raw reference output with `is_correct` when applicable.
- If the reference call times out, fails to parse, returns empty output, or fails validation, write an error `AuditRecord` and return an error `AuditReferenceResult`.
- Redact or hash both lower-layer output and reference output before writing `AuditRecord`.
- Record target name, release id, and contract hash from Trace, plus cost, latency, reference source when available, sampling probability, redacted/hash outputs, status, and error class.
- Return short-lived `AuditReferenceResult` so approved telemetry evidence can be created before raw reference output is discarded on success.
- Delete or expire the secure payload and raw reference result after audit processing and evidence approval.

Used by:

- Audit workers.

### `decide_asynchronous_risk_diagnostic`

Input:

- `trace: Trace`
- `risk_rules: RiskAuditRules`

Output:

- `RiskDiagnosticDecision`

Purpose:

- Select suspicious historical traces for offline diagnostics after only redacted/hash trace data remains.
- May trigger reference calls, replay through a frozen Release, or future evidence collection.
- Cannot claim comparison against the original lower-layer raw output unless approved telemetry evidence or replay output is available.

Used by:

- Monitoring and diagnostics.

### `build_audit_summary`

Input:

- `audits: Iterable[AuditRecord]`
- `window: TimeWindow`
- `target_name: str`
- `release_id: str`
- `contract_hash: str`

Output:

- `AuditSummary`

Purpose:

- Aggregate audit records for report and monitoring surfaces.
- Enforce the target and contract scope invariant for all included audit records.
- Count selected random audits even when their reference call failed.
- Compute random audit Precision only from successful random audits with a correctness comparison.
- Compute a random audit Precision lower bound and wrong-accept upper bound from the evaluable random audit count.
- Report random reference failures separately so monitoring can detect when online Precision cannot be estimated reliably.
- Keep risk audit counts and findings separate from random audit estimates.

Used by:

- reports.
- monitoring dashboard.
- Runtime Trace And Metrics.

## Invariants

- Random audit and risk audit are recorded separately.
- Random audit records preserve sampling probability.
- Risk audit findings must not be reported as unbiased online Precision.
- Audit correctness comparison may use short-lived raw secure payloads, but durable traces and audit records store only redacted/hash outputs.
- Reference-call failures are recorded and counted; selected random-audit samples are not silently dropped.
