# Release Runtime Module

## Purpose

The Release Runtime module creates immutable Releases and serves requests through one fixed Release. A Release may have zero, one, or multiple lower-layer artifacts. When no L1/L2/L3 artifact is present, cache misses naturally fall through to Core-owned L4. This module owns routing, cache lookup, L1/L2/L3 invocation, Core-owned L4 fallback, output validation, trace writing, canary, rollback, and runtime circuit breakers.

Runtime must execute the same artifact bytes that Core evaluated. It must not regenerate models, search thresholds, or dynamically splice unevaluated per-layer artifacts at deploy time.

This module follows the [System Invariants](../00_overall_design.md#system-invariants), especially release atomicity and durable runtime privacy.

## Boundary

Inputs:

- `CandidateDecision` and `Report` from Candidate Evaluation when a Release includes evaluated artifacts.
- Frozen `ArtifactPackage` records from Artifact Worker.
- `TargetDefinition` and `TargetRuntimeContract` from Target Definition.
- `TargetCheckReport` and reference/L4 broker settings for cold start Releases with no local artifacts.
- User approval or pre-approved deployment config from CLI/user.
- Runtime requests from serving entrypoints.
- Reference broker from reference infrastructure for L4 fallback.

Outputs:

- `Release` records to Release Registry.
- Runtime responses to callers.
- `Trace` records to Runtime Trace And Metrics.
- Request-lifetime serving context to Audit Monitoring and Telemetry Evidence And Recompile.
- Channel pointer changes for shadow, canary, stable, and rollback.
- Runtime health and circuit breaker events.

This module must not let artifacts call L4 or each other directly.

## Data Types

### `Release`

Fields:

- `release_id: str`
- `target_name: str`
- `contract_hash: str`
- `candidate_id: str | None`
- `snapshot_id: str | None`
- `snapshot_digest: str | None`
- `report_id: str | None`
- `created_at: datetime`
- `artifacts: dict[LayerName, ArtifactPackage | None]`
- `routing: RoutingSettings`
- `approval: ApprovalRecord | None`
- `status: Literal["created", "shadow", "canary", "stable", "retired", "rolled_back"]`

If `candidate_id`, `snapshot_id`, `snapshot_digest`, `report_id`, or `approval` is present, all compile provenance fields are required and must satisfy the release atomicity invariant.

A Release with every lower-layer artifact set to `None` is valid. It is the cold start serving state used before the first compile succeeds. It has no Candidate, Report, Snapshot, or ApprovalRecord, and cache misses fall through to Core-owned L4.

### `RoutingSettings`

Fields:

- `cache_enabled: bool`
- `enabled_layers: list[LayerName]`
- `L1_timeout_ms: int | None`
- `L2_timeout_ms: int | None`
- `L3_timeout_ms: int | None`
- `total_deadline_ms: int`
- `circuit_breaker: CircuitBreakerSettings`
- `audit: AuditSettings`

`enabled_layers` is the user/deployment switch for lower layers. It defaults to `["L1", "L2", "L3"]`. A layer can have an artifact in the Release and still be disabled by routing. Disabled layers are skipped as if they were absent, but reports and traces must distinguish configured-disabled from missing or unhealthy when presenting diagnostics.

### `ApprovalRecord`

Fields:

- `approval_id: str`
- `candidate_id: str`
- `report_id: str`
- `target_name: str`
- `contract_hash: str`
- `snapshot_id: str`
- `approved_at: datetime`
- `approved_by: Literal["user", "preapproved_policy"]`

`ApprovalRecord` is the release authorization. It must reference the exact Candidate, Report, target, contract, and Snapshot being released.

### `RuntimeRequest`

Fields:

- `request_id: str`
- `target_name: str`
- `input: dict`
- `tenant_key: str | None`
- `deadline_ms: int | None`
- `metadata: dict`

### `RuntimeResponse`

Fields:

- `request_id: str`
- `release_id: str`
- `status: Literal["ok", "error"]`
- `output: dict | None`
- `chosen_layer: Literal["cache", "L1", "L2", "L3", "L4"] | None`
- `error_type: Literal["deadline_exceeded", "l4_fallback_failure", "no_valid_output", "runtime_error"] | None`
- `public_error_message: str | None`
- `latency_ms: float`
- `trace_id: str`

### `CascadeResult`

Fields:

- `release_id: str`
- `attempts: list[LayerAttemptResult]`
- `status: Literal["ok", "error"]`
- `chosen_layer: Literal["L1", "L2", "L3", "L4"] | None`
- `output: dict | None`
- `serving_cost: float`
- `latency_ms: float`
- `fallback_reason: str | None`
- `error_type: Literal["l4_fallback_failure", "no_valid_output", "runtime_error"] | None`
- `error_message_hash: str | None`
- `l4_fallback_result: L4FallbackResult | None`

### `L4FallbackResult`

Fields:

- `input_raw: dict`
- `status: Literal["ok", "error"]`
- `output_raw: dict | None`
- `output_validated: dict | None`
- `reference_source: Literal["versioned_l4", "verified_l4", "human", "gold"] | None`
- `cost: float`
- `latency_ms: float`
- `finish_status: str`
- `error_type: Literal["timeout", "rate_limited", "empty_response", "parse_failure", "validation_failure", "provider_error"] | None`
- `error_message_raw: str | None`
- `error_message_hash: str | None`

`L4FallbackResult` is produced by Core-owned L4 fallback. On success it may carry raw input/output only inside the serving context so `approve_telemetry_evidence` can create approved evidence when privacy policy allows it. On failure it may keep raw provider or parse error text only for the request lifetime. Caller responses use stable public messages, and durable trace stores only error class and hash.

### `ServingResult`

Fields:

- `release_id: str`
- `path: Literal["cache", "cascade"]`
- `status: Literal["ok", "error"]`
- `cache_result: CacheHit | CacheMiss`
- `cascade_result: CascadeResult | None`
- `output: dict | None`
- `chosen_layer: Literal["cache", "L1", "L2", "L3", "L4"] | None`
- `serving_cost: float`
- `latency_ms: float`
- `error_type: Literal["deadline_exceeded", "l4_fallback_failure", "no_valid_output", "runtime_error"] | None`
- `error_message_hash: str | None`
- `l4_fallback_result: L4FallbackResult | None`

For cache hits, `cascade_result` is `None` and no L1/L2/L3 attempt is recorded. Cache hit rate is reported separately and never counted as local Coverage. If L4 fallback fails after all local layers abstain or fail, `status` is `"error"`, `output` is `None`, and the failure is still traceable.

`l4_fallback_result` is request-lifetime serving context only. It exists so Telemetry Evidence And Recompile can approve successful L4 fallback input/output before raw payloads are discarded. It must not be written to Trace, cache, Release registry, or durable telemetry unless converted into `ApprovedTelemetryEvidence`.

## Functions

### `create_release_without_artifacts`

Input:

- `definition: TargetDefinition`
- `contract: TargetRuntimeContract`
- `target_check: TargetCheckReport`
- `reference_broker: ReferenceBroker`
- `routing: RoutingSettings`
- `registry: ReleaseRegistry`

Output:

- `Release`

Purpose:

- Create an immutable Release with no lower-layer artifacts after target checks and the Core-owned L4/reference path are usable.
- Store target name, contract hash, routing settings, and release id.
- Store no Candidate, Report, Snapshot, ApprovalRecord, or lower-layer artifacts.
- Route cache misses directly to Core-owned L4 fallback.

Used by:

- Target registration.
- Cold start deployment.

### `create_release`

Input:

- `candidate: Candidate`
- `snapshot: Snapshot`
- `base_release: Release`
- `report: Report`
- `approval: ApprovalRecord`
- `artifact_store: ArtifactStore`

Output:

- `Release`

Purpose:

- Create an immutable Release from a Candidate that passed Core evaluation.
- Persist artifact digests, routing settings, contract hash, snapshot id/digest, report id, and approval record.
- Hard-fail unless `report.report_stage == "final"`.
- Hard-fail unless `report.decision` is present and `report.decision.status == "eligible_for_release"`.
- Reject if Candidate bytes differ from evaluated bytes.
- Enforce the release atomicity invariant across Candidate, Report, evaluated Snapshot, base Release, and ApprovalRecord.
- Enforce the snapshot scope invariant using the evaluated Snapshot's id and digest.

Used by:

- CLI `release approve`
- automated pre-approved release flow

### `load_release`

Input:

- `release_id: str`
- `registry: ReleaseRegistry`

Output:

- `Release`

Purpose:

- Load a Release and verify registry metadata and artifact digests.

Used by:

- Runtime serving.
- Rollback.
- Evaluation baseline loading.

### `set_channel`

Input:

- `target_name: str`
- `channel: Literal["shadow", "canary", "stable"]`
- `release_id: str`
- `channel_options: ChannelOptions`

Output:

- `ChannelUpdateResult`

Purpose:

- Atomically update a deployment channel pointer.
- Record previous channel value for rollback.
- Ensure target contract compatibility.

Used by:

- CLI deploy.
- rollout automation.

### `select_release_for_request`

Input:

- `target_name: str`
- `request: RuntimeRequest`
- `registry: ReleaseRegistry`

Output:

- `Release`

Purpose:

- Choose one fixed Release for the whole request.
- Use stable hash routing for canary when enabled.
- Never switch layers across Releases mid-request.

Used by:

- `serve_request`

### `check_result_cache`

Input:

- `release: Release`
- `contract: TargetRuntimeContract`
- `input_value: ValidatedInput`
- `cache_policy: CachePolicy`

Output:

- `CacheHit | CacheMiss`

Purpose:

- Check versioned result cache before L1.
- Use contract hash, normalized input, and relevant runtime/reference version in the key.
- Return cache hit rate separately from L1/L2/L3 Coverage.

Used by:

- `serve_request`

### `write_result_cache`

Input:

- `release: Release`
- `contract: TargetRuntimeContract`
- `input_value: ValidatedInput`
- `output: ValidatedOutput`
- `cache_policy: CachePolicy`

Output:

- `CacheWriteResult`

Purpose:

- Store successful outputs when cache is enabled and target policy permits it.
- Include TTL and release provenance.

Used by:

- `serve_request`

### `prepare_workers`

Input:

- `release: Release`
- `worker_pool: WorkerPool`

Output:

- `PreparedReleaseWorkers`

Purpose:

- Start or reuse workers for the Release.
- Start only lower-layer artifacts that are present and enabled by `release.routing.enabled_layers`.
- Run healthchecks and warmup.
- Verify worker digests match the Release.

Used by:

- Deploy.
- Runtime serving.

### `run_cascade`

Input:

- `release: Release`
- `workers: PreparedReleaseWorkers`
- `contract: TargetRuntimeContract`
- `input_value: ValidatedInput`
- `runtime_context: RuntimeContext`

Output:

- `CascadeResult`

Purpose:

- Call only lower-layer artifacts that are present, enabled by `release.routing.enabled_layers`, and healthy under the circuit breaker, in L1, then L2, then L3 order.
- Return the first valid accept.
- On abstain, timeout, crash, protocol error, or invalid output, continue to the next layer.
- If no local artifact is present, or if no local layer accepts, call Core-owned L4 fallback.
- If L4 fallback runs, attach the request-lifetime `L4FallbackResult` to `CascadeResult`.
- If L4 fallback fails, return an error `CascadeResult` with no output and enough error detail for trace and runtime metrics.

Used by:

- `serve_request`
- shadow execution.

### `build_cache_serving_result`

Input:

- `release: Release`
- `cache_hit: CacheHit`
- `contract: TargetRuntimeContract`
- `runtime_context: RuntimeContext`

Output:

- `ServingResult`

Purpose:

- Validate cached output.
- Build the serving result for a cache hit without running L1/L2/L3.
- Preserve cache provenance and release information for trace.
- Set `l4_fallback_result` to `None`.

Used by:

- `serve_request`

### `call_l4_fallback`

Input:

- `contract: TargetRuntimeContract`
- `input_value: ValidatedInput`
- `broker: ReferenceBroker`
- `runtime_context: RuntimeContext`

Output:

- `L4FallbackResult`

Purpose:

- Call L4/reference through Core broker.
- Apply runtime retry, timeout, rate limit, usage, cost, and output validation.
- Return raw input, raw output, validated output, reference source, latency, and cost for the current serving context on success.
- Return a failure result with cost, latency, finish status, error type, short-lived raw error text when needed, and error hash on timeout, provider error, empty response, parse failure, or validation failure.

Used by:

- `run_cascade`

### `serve_request`

Input:

- `request: RuntimeRequest`
- `registry: ReleaseRegistry`
- `contract_loader: TargetContractLoader`
- `worker_pool: WorkerPool`
- `trace_id_generator: TraceIdGenerator`
- `trace_writer: TraceWriter`
- `audit_decider: AuditDecider`
- `risk_rules: RiskAuditRules`
- `secure_audit_store: SecureAuditStore`
- `audit_queue: AuditQueue`
- `telemetry_privacy_policy: TelemetryPrivacyPolicy`
- `approved_evidence_store: ApprovedTelemetryEvidenceStore`

Output:

- `RuntimeResponse`

Purpose:

- Validate input.
- Select one Release.
- Check result cache.
- On cache hit, validate cached output and skip cascade.
- On cache miss, run cascade.
- Validate final output when one exists.
- Follow the Runtime Feedback Overview Data Lifetime Model for raw serving payloads, trace writing, audit capture, and approved evidence creation.
- Allocate `trace_id` before audit decisions and secure audit payload capture.
- If the chosen layer is L4 fallback, ask the telemetry privacy policy whether raw input/output may be written as approved telemetry evidence.
- Pass `trace_id`, selected `release_id`, source event time, target name, contract hash, and `ServingResult.l4_fallback_result` to `approve_telemetry_evidence` before the serving context expires; do not persist raw fallback payload otherwise.
- Decide random audit and synchronous risk audit while the raw serving result is still available.
- Combine random and risk decisions into `AuditDecisions`.
- Capture a short-lived secure audit payload when an audit is selected.
- Write approved L4 fallback evidence before raw fallback output leaves the serving context, when policy allows it.
- If L4 fallback fails, return an error `RuntimeResponse`, write a failure trace, update error metrics, and make the event visible to rollback checks.
- Return only stable error codes and public-safe messages to callers. Raw provider, parser, or validation error text may exist only in short-lived serving context and must not be written to Trace.
- Verify the selected Release's contract hash against the loaded target contract, then write Trace with that contract hash, `AuditDecisions`, and redacted/hash layer attempts.
- Enqueue selected audits after Trace is written.
- Return response.

Used by:

- HTTP/server entrypoint.
- Batch serving entrypoint.

### `run_shadow_request`

Input:

- `request: RuntimeRequest`
- `stable_release: Release`
- `shadow_release: Release`
- `contract: TargetRuntimeContract`
- `worker_pool: WorkerPool`
- `runtime_context: RuntimeContext`
- `circuit_breaker_state: CircuitBreakerStateStore | None`

Output:

- `ShadowComparisonRecord`

Purpose:

- Run a shadow Release without affecting user-visible output.
- Compare accept/abstain, chosen layer, output, latency, errors, and fallback.

Used by:

- Shadow channel.
- Runtime Trace And Metrics.
- Audit Monitoring.

### `update_circuit_breaker`

Input:

- `release: Release`
- `layer: LayerName`
- `event: RuntimeHealthEvent`

Output:

- `CircuitBreakerState`

Purpose:

- Track worker crash, timeout, invalid output, and health failure.
- Temporarily disable a failing layer so requests continue to fallback safely.
- Keep health-based temporary disablement separate from `RoutingSettings.enabled_layers`, which is the user/deployment configuration.

Used by:

- `run_cascade`
- monitoring.

### `rollback_release`

Input:

- `target_name: str`
- `registry: ReleaseRegistry`
- `rollback_options: RollbackOptions`

Output:

- `RollbackResult`

Purpose:

- Move the stable channel pointer back to the previous eligible Release.
- Do not rebuild artifacts or modify workspaces.
- Keep failed Release available for analysis.

Used by:

- CLI rollback.
- automatic safety rollback.

### `retire_release`

Input:

- `release_id: str`
- `registry: ReleaseRegistry`

Output:

- `ReleaseRetirementRecord`

Purpose:

- Mark a Release as no longer active.
- Preserve artifacts, report, trace links, and rollback history.

Used by:

- Registry maintenance.

## Invariants

- Every request uses one fixed Release.
- Runtime bytes match evaluated bytes for every lower-layer artifact present in the Release.
- A Release may have no lower-layer artifacts.
- Fallback is Core behavior.
- Cache hits skip cascade and produce traces with no layer attempts.
- Cache hit rate is separate from local Coverage.
- Canary uses stable routing keys.
- Rollback is pointer movement, not rebuild.
- Runtime traces include `release_id`, `contract_hash`, redacted/hash layer attempts, cost, latency, and audit probability.
