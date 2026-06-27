# Reboot Implementation Runbook

This runbook is for an implementation agent that will turn the reboot design into code. Treat the design documents in this directory as the sole source of truth, especially [00 Overall Design](00_overall_design.md) and the module documents under [modules/](modules). If this runbook, the current implementation, old experiment code, or historical docs conflict with the reboot design documents, the reboot design documents win.

The goal is not to preserve the current implementation shape. The goal is to implement the reboot architecture with the lowest reasonable abstraction tax while keeping the system target-independent in Core.

## Operating Rules

- Work in a dedicated git branch and worktree.
- Implement one module slice at a time, but keep the repository testable after each slice.
- Prefer plain Python data classes, typed dictionaries, ordinary functions, and explicit registries before adding framework machinery.
- Do not move target-specific logic into Core. Core may carry opaque target payloads, but it must not interpret fields such as intents, slots, labels, utterances, or dataset ids.
- Do not let L1/L2/L3 artifacts call L4, each other, validation/test data, or release registries directly.
- Do not expose validation/test rows, row ids, split indices, or reconstructable holdout membership to Agent Workspace.
- Do not count cache hits as L1/L2/L3 Coverage.
- Do not create a separate cold-start compile path. The first lower-layer compile is a `RecompileRequest`.
- Do not invent compatibility shims for historical experiment artifacts unless they are required by current tests or explicitly approved.
- Update design docs only when implementation exposes a real design gap. Keep changes narrow, update the affected module boundary first, then implement against the updated design.

## Module Acceptance Protocol

After each module slice, the implementation agent must decide whether the module is correctly and completely implemented by checking it against the design documents, not against the current code shape.

For the module being implemented, create or update a short status note with a traceability matrix:

- Each data type in the module document maps to implementation files and focused tests.
- Each function in the module document maps to implementation files and focused tests.
- Each boundary input and output maps to the producing module and consuming module, or is marked deferred.
- Each invariant maps to at least one positive test or negative test, unless the invariant is purely documentary.
- Each deferred item has a reason, owner, and the later module slice that will finish it.

A module is correct when:

- Its behavior matches the Purpose text of each implemented function.
- It accepts and returns the documented inputs and outputs, or a deliberately equivalent low-abstraction representation.
- It enforces the documented hard-fail conditions.
- It rejects the important negative cases described by the module's invariants.
- It does not add target-specific interpretation to Core.

A module is complete when:

- Every data type, function, boundary data flow, and invariant in that module document is implemented, tested, or explicitly deferred.
- All cross-module data flows used by the module are wired to the named producer and consumer modules.
- Agent-visible, durable runtime, holdout, release, snapshot, and telemetry boundaries match the design.
- Focused tests for the module pass.
- The repository remains runnable with the agreed full or broad test command for the current stage.

If the agent cannot make a module complete because the design is ambiguous or inconsistent, it must update the design documents first and then continue. Do not silently complete a different design.

## Suggested Implementation Order

### 0. Baseline And Status

Read:

- [README](README.md)
- [00 Overall Design](00_overall_design.md)
- [01 Target Definition](modules/01_target_definition.md)

Do:

- Inspect the current code and tests.
- Write a short implementation status note in the worktree that maps existing code to reboot modules.
- Identify code that can be reused without carrying old target-specific assumptions into Core.

Done when:

- The agent can name the files it will touch first.
- The current test command is known.
- The branch/worktree state is clean or the existing changes are intentionally accounted for.

### 1. Target Definition And Contract Hash

Read:

- [01 Target Definition](modules/01_target_definition.md)

Implement:

- Target definition loading and validation.
- `TargetRuntimeContract` and contract hash generation.
- Target-owned redaction, metadata bucketing, reason-code policy, correctness checks, and output validation hooks.
- Target checks that prove the L4/reference path is available before cold-start serving or compile.

Tests:

- Contract hash changes when runtime contract semantics change.
- Target-specific fields remain opaque to Core.
- Invalid target definitions fail early.

Done when:

- Other modules can depend on a stable `TargetDefinition`, `TargetRuntimeContract`, and `contract_hash`.

### 2. Snapshot And Reference

Read:

- [02 Snapshot And Reference](modules/02_snapshot_reference.md)
- [10 Telemetry Evidence And Recompile](modules/10_telemetry_evidence_recompile.md)

Implement:

- Source record collection from target data plus optional `TelemetryDataSource`.
- Snapshot cutoff handling and source watermarks.
- Deduplication without widening split eligibility.
- Train/validation/test split assignment.
- `ConsumedRowsManifest` handling.
- Reference qualification and reference usage ledger.

Tests:

- Telemetry source with mismatched target or contract is rejected.
- `telemetry_source.cutoff_time > snapshot_cutoff_time` is rejected.
- Consumed validation/test rows cannot re-enter hidden holdout splits.
- Versioned L4 agreement is not reported as gold/human correctness.

Done when:

- Compile can request immutable Snapshot handles and qualified reference summaries.

### 3. Artifact Worker Protocol

Read:

- [04 Artifact Worker](modules/04_artifact_worker.md)
- [06 Release Runtime](modules/06_release_runtime.md)

Implement:

- Worker request/response protocol for L1/L2/L3 artifacts.
- `accept` / `abstain` / `error` result handling.
- Ephemeral request ids for private evaluation.
- Worker sandbox limits.
- Trace-safe reason code enforcement.

Tests:

- Artifact workers cannot access L4/reference or next-layer services.
- Private evaluation never passes stable row ids to workers.
- Invalid worker outputs fail validation.

Done when:

- Candidate Evaluation and Release Runtime can call artifacts through one stable protocol.

### 4. Compile Orchestration And Agent Workspace

Read:

- [11 Compile Orchestration](modules/11_compile_orchestration.md)
- [03 Agent Workspace](modules/03_agent_workspace.md)
- [02 Snapshot And Reference](modules/02_snapshot_reference.md)

Implement:

- `RecompileRequest` planning and acceptance.
- Snapshot cutoff and options persistence in `CompileLaunchDecision`.
- Long-lived target workspace baseline and isolated compile attempts.
- Agent mount creation with train-only data, aggregate reports, and approved telemetry summaries.
- Agent workspace closure and candidate submission capture.

Tests:

- First compile after cold start uses the same `RecompileRequest` path as later compiles.
- Agent mounts never contain validation/test rows or row ids.
- Accepted workspace baseline advances only after an accepted release path.

Done when:

- Core can launch one target adaptation agent per compile attempt and receive `CandidateSubmission`.

### 5. Candidate Evaluation

Read:

- [05 Candidate Evaluation](modules/05_candidate_evaluation.md)
- [04 Artifact Worker](modules/04_artifact_worker.md)
- [02 Snapshot And Reference](modules/02_snapshot_reference.md)

Implement:

- Candidate freezing and manifest checks.
- Validation evaluation, aggregate feedback, and private evaluation request plans.
- Test evaluation after the agent attempt is closed.
- Metrics by reference source.
- `compare_candidates`.
- `finalize_report`.
- Agent-visible report summaries using `HoldoutConsumptionSummary`.

Tests:

- Validation/test Reports have `decision: None`.
- `create_release` cannot use validation/test Reports directly.
- Final Report is produced only after `CandidateDecision`.
- Agent-visible reports never include `ConsumedRowsManifest` details.
- Failed or visible test results produce consumed holdout manifests.

Done when:

- Candidate Evaluation can produce final release evidence without leaking holdout membership.

### 6. Release Runtime

Read:

- [06 Release Runtime](modules/06_release_runtime.md)
- [08 Runtime Trace And Metrics](modules/08_runtime_trace_metrics.md)
- [09 Audit Monitoring](modules/09_audit_monitoring.md)
- [10 Telemetry Evidence And Recompile](modules/10_telemetry_evidence_recompile.md)

Implement:

- Release creation without artifacts for cold start.
- Release creation from final Report plus eligible `CandidateDecision`.
- Layer routing with per-layer enabled switches.
- Cache hit path.
- Cascade path.
- Core-owned L4 fallback.
- Stable public error responses.
- Trace/audit/evidence scheduling hooks.

Tests:

- A Release with no lower-layer artifacts routes cache misses to L4.
- Disabling a layer skips that layer even when its artifact exists.
- `create_release` rejects non-final Reports and non-eligible decisions.
- L4 fallback failures are represented in response, trace, and metrics.
- Successful L4 fallback evidence carries trace, release, target, and contract provenance.

Done when:

- The system can serve from a fixed Release before and after lower-layer compile success.

### 7. Runtime Trace And Metrics

Read:

- [07 Runtime Feedback Overview](modules/07_runtime_feedback_overview.md)
- [08 Runtime Trace And Metrics](modules/08_runtime_trace_metrics.md)

Implement:

- Durable redacted/hash Trace records.
- Runtime metric windows scoped by target, release, and contract.
- Per-layer Coverage and attempt statistics.
- Precision lower-bound and wrong-accept upper-bound calculations from random audit records.
- Drift and runtime failure decisions.

Tests:

- Trace contains no raw input/output/provider error text.
- Runtime metrics reject mixed target/release/contract input.
- Random audit reference failure rates can trigger runtime failure decisions.

Done when:

- Monitoring can make rollback/recompile recommendations from durable runtime data without raw payloads.

### 8. Audit Monitoring

Read:

- [09 Audit Monitoring](modules/09_audit_monitoring.md)
- [07 Runtime Feedback Overview](modules/07_runtime_feedback_overview.md)

Implement:

- Random audit decisions.
- Synchronous risk audit flags with allowlisted codes.
- Short-lived secure audit payloads.
- Audit reference calls with success and failure records.
- Audit summaries.

Tests:

- Secure payload and Trace must match on trace id, release id, target, and contract.
- Audit reference failures are recorded and counted.
- Audit records do not persist raw reference output.

Done when:

- Online quality estimates and failure decisions can use audit evidence safely.

### 9. Telemetry Evidence And Recompile

Read:

- [10 Telemetry Evidence And Recompile](modules/10_telemetry_evidence_recompile.md)
- [02 Snapshot And Reference](modules/02_snapshot_reference.md)
- [11 Compile Orchestration](modules/11_compile_orchestration.md)

Implement:

- `ApprovedTelemetryEvidence` creation from L4 fallback, audit, and user feedback.
- Privacy review and split eligibility enforcement.
- `TelemetryDataSource` construction.
- `request_recompile`.
- Agent-visible telemetry summaries.

Tests:

- Redacted Trace or AuditRecord alone cannot become Snapshot rows.
- Runtime-derived approved evidence has release provenance.
- Evidence created after the telemetry cutoff does not enter the source.
- User feedback cannot bypass approval.

Done when:

- Runtime observations can feed future compile only through approved evidence and normal Snapshot building.

### 10. End-To-End Thin Target

Use a small target that is deliberately simple and not NLU-specific.

Implement:

- A minimal target definition.
- A no-artifact cold-start Release.
- One generated or hand-written lower-layer artifact.
- One compile run through Snapshot, Agent Workspace, Candidate Evaluation, final Report, Release, serving, Trace, audit, and recompile request.

Tests:

- End-to-end cold start to first compiled Release.
- Release with disabled layers.
- Rollback or runtime failure decision path.
- Future compile from approved telemetry.

Done when:

- The reboot architecture is executable without relying on CLINC150, MASSIVE, NLU frames, or historical experiment artifacts.

## Completion Criteria

The reboot implementation is complete when:

- Every module document has a corresponding implementation surface or an explicit documented reason for deferral.
- The main end-to-end path works from cold start to first lower-layer Release.
- Core remains target-independent under tests and code review.
- L1/L2/L3 artifacts are invoked only through the worker protocol.
- Agent Workspace can evolve compile-time scaffolding and runtime artifacts without seeing validation/test data.
- Runtime-to-future-compile data flows only through `ApprovedTelemetryEvidence` and `TelemetryDataSource`.
- Full tests pass, and focused reboot tests cover the module invariants above.

## Final Handoff Format

When an implementation agent finishes, report:

- Branch and worktree path.
- Commit hash.
- Module slices completed.
- Module slices intentionally deferred.
- Test commands and results.
- Any design doc changes made during implementation.
- Known risks or follow-up work.
