# Runtime Feedback Overview

## Purpose

This document defines the shared lifetime rules and end-to-end data flow for runtime feedback. It is an overview, not the owner of every field.

This document expands the runtime-related [System Invariants](../00_overall_design.md#system-invariants).

The concrete module designs are split into:

- [08 Runtime Trace And Metrics](08_runtime_trace_metrics.md)
- [09 Audit Monitoring](09_audit_monitoring.md)
- [10 Telemetry Evidence And Recompile](10_telemetry_evidence_recompile.md)

Together, these modules record runtime behavior, estimate online quality, detect drift, prepare rollback or recompile signals, and decide which runtime observations may enter a future Snapshot.

## Boundary

Inputs:

- Runtime request results, serving context, and health events from Release Runtime.
- `TargetRuntimeContract`, redaction functions, privacy policy, and runtime requirements from Target Definition.
- Reference broker from Snapshot And Reference.
- Release and Report metadata from Release Runtime and Candidate Evaluation.
- User corrections or business feedback from external systems.

Outputs:

- `Trace`, `RuntimeMetricWindow`, drift signals, and runtime failure decisions from Runtime Trace And Metrics.
- `AuditRecord` and `AuditSummary` from Audit Monitoring.
- `ApprovedTelemetryEvidence`, `TelemetryDataSource`, and `RecompileRequest` from Telemetry Evidence And Recompile.
- `TelemetryDataSource` and `RecompileRequest` to Compile Orchestration when runtime evidence should start a future compile.

This feedback chain must not directly mutate the active target definition, candidate reports, past snapshots, or active release pointers. It emits evidence and decisions for the owning modules to apply.

## Data Lifetime Model

The canonical rule is: durable trace/audit records are not training or evaluation data. Future compile data must come from approved telemetry evidence, then from a telemetry data source, then from Snapshot And Reference.

| Data carrier | May contain raw input/output | Retention | May become future Snapshot rows | Main purpose |
| --- | --- | --- | --- | --- |
| Serving context (`RuntimeRequest`, `LayerAttemptResult`, `CascadeResult`, `ServingResult`, `L4FallbackResult`) | Yes | Request lifetime only | No | Serve one request and allow immediate trace, audit, and evidence decisions. |
| `Trace` | No; redacted/hash only | Durable | No | Runtime metrics, debugging, drift detection, rollback signals. |
| `SecureAuditPayload` | Yes | Short-lived only | No | Let an audit compare the original lower-layer output with a fresh reference output. |
| `AuditReferenceResult` | Yes, reference output only on success | Short-lived in memory only | No | Record reference-call success/failure and let evidence approval run before raw reference output is discarded. |
| `AuditRecord` | No; redacted/hash only | Durable | No | Audit reports, online quality estimates, monitoring. |
| `UserFeedbackRecord` | Yes | Short-lived ingest carrier, or retained only in the owning external system | No | Carry user corrections into privacy review; it must not become a Snapshot row directly. |
| `ApprovedTelemetryEvidence` | Yes, if privacy policy allows; otherwise canonicalized | Durable | Yes, subject to `approved_for` and cutoff | Store future-compile evidence after privacy and split-role approval. |
| `TelemetryDataSource` | Points to approved evidence rows | Durable manifest | Yes | Pass approved evidence through Compile Orchestration into Snapshot And Reference with cutoff and split eligibility preserved. |
| `SourceRecord` / `SnapshotRecord` | Yes, after Snapshot validation | Snapshot lifetime | Already in Snapshot | Feed train, validation, and test views according to split rules. |

Invariants:

- Trace and AuditRecord are never used to reconstruct future `SourceRecord` payloads.
- Raw runtime payloads either expire after request/audit processing or are explicitly converted into `ApprovedTelemetryEvidence`.
- `ApprovedTelemetryEvidence` is the only durable bridge from runtime observations or user feedback into future compile data.
- `TelemetryDataSource` must filter by cutoff time and preserve each record's split eligibility.
- Snapshot And Reference is the only module that turns approved telemetry rows into train/validation/test records.
- Any new runtime-derived data path must be classified in this table before adding module-specific fields or exceptions.

## Canonical Runtime To Recompile Flow

1. Release Runtime serves the request and keeps raw input/output only in the serving context.
2. Runtime allocates `trace_id` before any audit payload or evidence capture.
3. If L4 fallback succeeds, Runtime may call `approve_telemetry_evidence` while raw input/output is still available. Failed L4 fallback produces metrics and trace data, not future Snapshot evidence.
4. Runtime decides random audit and synchronous risk audit while raw lower-layer output is still available.
5. If an audit is selected, Runtime captures `SecureAuditPayload`.
6. Runtime writes `Trace` using only redacted/hash payloads and enqueues selected audits.
7. Audit workers call the reference broker with `SecureAuditPayload`, write success or failure `AuditRecord`, and may call `approve_telemetry_evidence` while a successful `AuditReferenceResult` is still available.
8. User feedback also enters through `approve_telemetry_evidence`; raw feedback never goes directly into a telemetry data source.
9. `build_telemetry_data_source` selects approved evidence with `created_at <= cutoff_time`, preserves per-record split eligibility, and emits a `TelemetryDataSource`.
10. Compile Orchestration receives the `RecompileRequest` and `TelemetryDataSource`, then calls Snapshot And Reference.
11. Snapshot And Reference converts `TelemetryDataSource` rows into `SourceRecord` values and performs validation, deduplication, grouping, and split assignment.

## Ownership Map

Runtime Trace And Metrics owns:

- Durable `Trace`.
- `RuntimeMetricWindow`.
- Runtime failure and drift detection.
- Online quality summaries shown to users.

Audit Monitoring owns:

- Random and risk audit decisions.
- Short-lived secure audit payloads.
- Reference calls for audit.
- Durable audit records and audit summaries.

Telemetry Evidence And Recompile owns:

- Privacy-reviewed approved telemetry evidence.
- User feedback ingestion into approved evidence.
- Telemetry data sources for future Snapshots.
- Recompile requests for Compile Orchestration.
- Agent-visible telemetry summaries for Agent Workspace.

## Invariants Across The Split

- Random audit and risk audit remain separate across all three documents.
- Cache hits are reported separately and never counted as L1/L2/L3 Coverage.
- Runtime artifacts never call L4 or the next layer.
- Core drift detection treats reason codes and source metadata bucket keys as opaque buckets.
- Target-specific semantic drift analysis belongs to agent-generated compile-time scaffolding, not Core default logic.
