# Telemetry Evidence And Recompile Module

## Purpose

The Telemetry Evidence And Recompile module owns the durable bridge from runtime observations to future compile data. It approves eligible runtime observations or user feedback, builds telemetry data sources for Snapshot And Reference, and requests future compile runs.

This module follows the [Runtime Feedback Overview](07_runtime_feedback_overview.md) data lifetime model and the [System Invariants](../00_overall_design.md#system-invariants).

## Boundary

Inputs:

- Successful L4 fallback results from Release Runtime.
- Successful audit reference results from Audit Monitoring.
- Target name and `ContractHash` from the caller that is approving evidence or building a telemetry data source.
- Durable `Trace` and `AuditRecord` values for provenance, metrics, and selection metadata.
- User feedback from external systems.
- `TargetDefinition`, `TelemetryPrivacyPolicy`, and `TargetRuntimeContract` from Target Definition.
- Stable `Release` metadata from Release Runtime for recompile requests, including cold start Releases with no lower-layer artifacts.
- `DriftSignal` and `RuntimeMetricWindow` from Runtime Trace And Metrics.

Outputs:

- `ApprovedTelemetryEvidence` to the controlled evidence store.
- `TelemetryDataSource` to Compile Orchestration, then Snapshot And Reference for future Snapshot builds.
- `RecompileRequest` to Compile Orchestration.
- `AgentVisibleTelemetrySummary` to Agent Workspace for future compile.

This module must not turn redacted trace or audit records directly into Snapshot rows.

## Data Types

### `ApprovedTelemetryEvidence`

Fields:

- `evidence_id: str`
- `trace_id: str | None`
- `release_id: str | None`
- `target_name: str`
- `contract_hash: str`
- `input_payload: dict`
- `reference_output_payload: dict`
- `reference_source: Literal["gold", "human", "versioned_l4", "verified_l4", "user_feedback"]`
- `source: Literal["l4_fallback", "random_audit", "risk_audit", "user_feedback"]`
- `approved_for: list[Literal["train", "validation_candidate", "test_candidate"]]`
- `privacy_review: PrivacyReviewRecord`
- `source_event_at: datetime`
- `created_at: datetime`

This is the controlled long-term evidence store used to build future Snapshots. It may contain raw, canonicalized, or otherwise trainable/evaluable payloads, but only after the target privacy policy approves that payload for future compile use.

Invariants:

- `created_at` is the time the evidence became approved and durable; it must not be earlier than the source event time, such as a fallback request time, audit completion time, or `UserFeedbackRecord.submitted_at`.
- `privacy_review.decision` must be `"approved"`.
- `approved_for` must be non-empty.
- `approved_for` is the effective permission for future Snapshot use and must be equal to, or a strict subset of, `privacy_review.approved_for`.
- `payload_form == "redacted_not_trainable"` must not produce approved telemetry evidence for train, validation, or test Snapshot use. Keep such records only as redacted trace/audit diagnostics.
- Runtime-derived sources (`l4_fallback`, `random_audit`, `risk_audit`) must set `release_id`. User feedback may leave it `None` when the feedback is not tied to a served Release.

### `UserFeedbackRecord`

Fields:

- `feedback_id: str`
- `target_name: str`
- `contract_hash: str`
- `input_payload: dict`
- `corrected_output_payload: dict`
- `submitted_at: datetime`
- `source: Literal["human_review", "business_system", "customer_correction"]`
- `reviewer_id_hash: str | None`
- `requested_approved_for: list[Literal["train", "validation_candidate", "test_candidate"]]`
- `metadata: dict`

User feedback is not automatically training or evaluation data. It must pass `approve_telemetry_evidence` before it can become `ApprovedTelemetryEvidence`.

### `PrivacyReviewRecord`

Fields:

- `review_id: str`
- `policy_version: str`
- `decision: Literal["approved", "rejected"]`
- `approved_for: list[Literal["train", "validation_candidate", "test_candidate"]]`
- `payload_form: Literal["raw", "canonicalized", "redacted_not_trainable"]`
- `redactions_applied: list[str]`
- `reviewed_at: datetime`
- `reviewer: Literal["policy", "human"]`
- `notes: list[str]`

### `TelemetryPrivacyPolicy`

Owned by Target Definition as part of `TargetDefinition.runtime_config`. This module consumes the policy and documents the fields it requires.

Fields:

- `policy_version: str`
- `allowed_sources: list[Literal["l4_fallback", "random_audit", "risk_audit", "user_feedback"]]`
- `default_approved_for_by_source: dict[str, list[Literal["train", "validation_candidate", "test_candidate"]]]`
- `raw_payload_allowed: bool`
- `canonicalization_required: bool`
- `human_review_required_sources: list[str]`

The policy decides whether a raw or canonicalized payload can be retained as approved telemetry evidence, and for which future split roles.

### `TelemetryDataSource`

Fields:

- `source_id: str`
- `target_name: str`
- `contract_hash: str`
- `cutoff_time: datetime`
- `records_uri: str`
- `default_split_eligibility: list[Literal["train", "validation_candidate", "test_candidate"]]`
- `per_record_split_eligibility_uri: str | None`
- `included_sources: list[str]`
- `provenance_digest: str`

`records_uri` contains only approved evidence records with `created_at <= cutoff_time`. It must include enough per-row provenance for Snapshot And Reference to create `SourceRecord` values without reading Trace or Audit raw payloads. When a compile ingests this source, `cutoff_time` must be no later than the `CompileLaunchDecision.snapshot_cutoff_time`.

`default_split_eligibility` may be used only when every included record has exactly the same effective `approved_for` value. If included records have different permissions, `per_record_split_eligibility_uri` is required and must map every record to its own eligibility list. Builders must fail rather than widen a record's permission with a batch default.

### `RecompileRequest`

Fields:

- `target_name: str`
- `contract_hash: str`
- `reason: RecompileReason`
- `telemetry_source: TelemetryDataSource | None`
- `base_release_id: str`
- `budget_hint: CompileBudget | None`
- `created_at: datetime`
- `requested_by: Literal["user", "scheduler", "monitoring"]`

### `AgentVisibleTelemetrySummary`

Fields:

- `target_name: str`
- `release_id: str`
- `contract_hash: str`
- `metrics_summary: dict`
- `drift_summary: dict`
- `telemetry_source_id: str | None`
- `redaction_policy_version: str`
- `generated_at: datetime`

This summary is aggregate, redacted, and governed by the durable runtime privacy and agent boundary invariants.

## Functions

### `approve_telemetry_evidence`

Input:

- `trace: Trace | None`
- `secure_payload: SecureAuditPayload | None`
- `audit_reference_result: AuditReferenceResult | None`
- `l4_fallback_result: L4FallbackResult | None`
- `audit_record: AuditRecord | None`
- `user_feedback: UserFeedbackRecord | None`
- `trace_id: str | None`
- `release_id: str | None`
- `source_event_at: datetime | None`
- `target_name: str`
- `contract_hash: str`
- `contract: TargetRuntimeContract`
- `privacy_policy: TelemetryPrivacyPolicy`

Output:

- `ApprovedTelemetryEvidence | None`

Purpose:

- Create future-compile evidence only from payloads approved by target privacy policy.
- Accept exactly one source path per call: successful L4 fallback, successful audit reference result, or user feedback.
- Enforce the target and contract scope invariant for every supplied carrier.
- Enforce the same-request and same-release evidence invariant for audit evidence carriers before writing evidence.
- Preserve usable input and reference output payloads for future Snapshot `SourceRecord` creation.
- For L4 fallback, require `trace_id`, `release_id`, `source_event_at`, and successful `L4FallbackResult`; the release must match the serving Release that produced the fallback result.
- For audit evidence, require the matching `Trace`, `SecureAuditPayload`, `AuditReferenceResult`, and `AuditRecord` while raw audit payloads are still available; hard-fail unless their `trace_id`, `release_id`, `target_name`, and `contract_hash` match.
- For user feedback, create evidence only after privacy review; raw `UserFeedbackRecord` never enters `TelemetryDataSource` directly.
- Return `None` unless the privacy review decision is `"approved"`, the effective `approved_for` is non-empty, and the effective `approved_for` does not exceed the review's approved split roles.
- Return `None` for `payload_form == "redacted_not_trainable"` because that payload cannot become trainable or evaluable Snapshot evidence.
- Set `created_at` to the approval/store time and keep it no earlier than the source event time.
- Enforce the runtime-to-future-compile bridge invariant: redacted Trace or AuditRecord data alone can never become Snapshot evidence.

Used by:

- Release Runtime `serve_request`.
- Audit workers.
- L4 fallback telemetry processing.
- User feedback ingestion.

### `build_telemetry_data_source`

Input:

- `traces: Iterable[Trace]`
- `audits: Iterable[AuditRecord]`
- `approved_evidence: Iterable[ApprovedTelemetryEvidence]`
- `target_name: str`
- `contract_hash: str`
- `cutoff_time: datetime`
- `contract: TargetRuntimeContract`

Output:

- `TelemetryDataSource`

Purpose:

- Convert approved telemetry evidence into a source that future snapshots can ingest.
- Set `TelemetryDataSource.target_name` and `TelemetryDataSource.contract_hash` from the explicit inputs.
- Enforce the target and contract scope invariant for included evidence, Trace rows, and AuditRecord rows.
- When included evidence has `release_id`, join Trace and AuditRecord metadata only from the same release.
- Join Trace and AuditRecord metadata only through `trace_id` values referenced by included evidence.
- Filter out every `ApprovedTelemetryEvidence` where `created_at > cutoff_time`; late-arriving evidence must not enter an older Snapshot.
- Carry each evidence record's `approved_for` into the data source as split eligibility.
- Write per-record split eligibility whenever the included evidence does not share one exact eligibility list.
- Preserve provenance and cutoff time.
- Apply privacy checks and deduplication hints, but leave final split assignment to Snapshot And Reference.

Used by:

- Compile Orchestration for future Snapshot builds.
- Snapshot And Reference ingests the resulting source only when Compile Orchestration starts a compile.

### `request_recompile`

Input:

- `definition: TargetDefinition`
- `base_release: Release`
- `reason: RecompileReason`
- `telemetry_source: TelemetryDataSource | None`
- `budget_hint: CompileBudget | None`

Output:

- `RecompileRequest`

Purpose:

- Create a request for a future compile using the current stable Release and fresh data. The current stable Release may have no lower-layer artifacts yet.
- Set `RecompileRequest.target_name` and `RecompileRequest.contract_hash` from the current `TargetDefinition`.
- Set `RecompileRequest.base_release_id` from the supplied stable Release.
- Enforce the target and contract scope invariant across the current `TargetDefinition`, base Release, and optional telemetry source.
- Do not choose the Snapshot cutoff time. Compile Orchestration fixes `snapshot_cutoff_time` when it accepts the request and rejects telemetry sources whose cutoff is later than that freeze point.
- Does not start an agent. Compile Orchestration decides whether and when to launch the compile.

Used by:

- CLI/user.
- monitoring automation.
- Compile Orchestration consumes the resulting request.

### `export_agent_visible_telemetry_summary`

Input:

- `metrics: RuntimeMetricWindow`
- `drift_signal: DriftSignal`
- `redaction_policy: RedactionPolicy`

Output:

- `AgentVisibleTelemetrySummary`

Purpose:

- Provide future agents with aggregate telemetry and drift summaries.
- Require `metrics.target_name`, `metrics.release_id`, and `metrics.contract_hash` to match `drift_signal.target_name`, `drift_signal.release_id`, and `drift_signal.contract_hash`.
- Set `AgentVisibleTelemetrySummary.contract_hash` from the matched metrics/drift scope.
- Exclude raw production traces unless those rows are intentionally converted into the next train snapshot.

Used by:

- Agent Workspace module in future compile.

## Invariants

- This module owns the runtime-to-future-compile bridge defined in the System Invariants.
- Trace and AuditRecord are used only for provenance, metrics, and selection metadata.
- Snapshot And Reference owns final data validation, deduplication, grouping, and split assignment.
- Recompile requests use the current stable Release as the base unless the caller explicitly selects another release. The first lower-layer compile after cold start uses the current Release with no local artifacts as its base.
