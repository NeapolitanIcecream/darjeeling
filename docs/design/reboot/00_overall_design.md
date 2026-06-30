# Overall Design

## Purpose

Darjeeling compiles a user-reviewed target definition into cheaper runtime artifacts while preserving Precision, increasing Coverage, and measuring Generalization. The system does not ask users to hand-maintain target-specific optimization code. Instead, Core starts a target adaptation agent that writes compile-time scaffolding and L1/L2/L3 runtime source in an isolated workspace. Core then evaluates and deploys only the frozen runtime artifacts that pass its own checks.

This design intentionally restarts from a small set of concepts. It should not preserve old layer-specific compiler paths, target package semantics, or L0 cache language.

## Core Concepts

| Concept | Meaning | Owner |
| --- | --- | --- |
| Target | User-reviewed task definition: schema, correctness, reference, data rules, requirements. | User reviews; Core loads and hashes |
| Snapshot | Frozen data version with train/validation/test boundaries and reference provenance. | Core |
| Artifact | A deployable L1, L2, or L3 worker package using the same `accept`/`abstain` protocol. | Agent writes; Core validates |
| Candidate | Frozen proposal for a complete runtime release, possibly reusing unchanged artifacts. | Agent submits; Core freezes |
| Report | Core-owned evaluation result for a Candidate. | Core |
| Release | Immutable deployable runtime state with routing settings, optional lower-layer artifacts, and optional compile provenance. If no lower-layer artifact is present, cache misses naturally fall through to Core-owned L4. | Core |
| Trace | Runtime record of requests, layer attempts, outputs, cost, audit, and drift signals. | Core |

Supporting terms such as workspace, compile run, router, registry, and broker are implementation modules, not user-facing concepts.

## Non-Goals

- No L0 capability layer. Cache is reported separately.
- No nested agents and no layer-specific agent roles.
- No per-layer compiler framework.
- No target-specific rules in core.
- No core-level partial-output composition. Runtime artifacts either accept a full output or abstain.
- No weighted score that trades wrong accepts for cost savings.
- No deployed runtime package containing training data, validation/test rows, agent transcripts, search scripts, or compile-time tools.

## System Invariants

These invariants apply across module boundaries. Module documents may reference them instead of repeating every field-level check.

- **Target and contract scope:** any record crossing runtime, evaluation, audit, telemetry, or recompile boundaries must belong to one `target_name` and `contract_hash`. Release-scoped records must also match `release_id`.
- **Snapshot scope:** snapshot, evaluation, report, and release boundaries must match `snapshot_id`; when a digest is available, they must also match `snapshot_digest`.
- **Same-request evidence:** trace, secure audit payload, audit result, audit record, and approved evidence carriers must agree on request provenance before raw input/output is compared or persisted.
- **Release atomicity:** any Release that carries Candidate, Report, Snapshot, or ApprovalRecord provenance must carry one matching set of all of them, after Core test evaluation has made the Candidate release-eligible. A Release with no lower-layer artifacts and no compile provenance can be created from a checked TargetDefinition and a usable Core-owned L4/reference path.
- **Durable runtime privacy:** durable Trace, AuditRecord, runtime metrics, and agent-visible telemetry contain only redacted, hashed, bucketed, or allowlisted values. Raw runtime payloads live only in request/audit lifetime carriers unless converted into approved evidence.
- **Runtime-to-future-compile bridge:** future Snapshot rows derived from runtime observations or user feedback must flow through `ApprovedTelemetryEvidence`, then `TelemetryDataSource`, then Compile Orchestration, then Snapshot And Reference. Trace and AuditRecord are never reconstructed into training or evaluation rows.
- **Hidden holdout consumption:** a failed test gate, or any user/agent-visible test result, consumes that test window for future compile claims. Consumed validation/test rows cannot remain hidden validation/test evidence in a future Snapshot.
- **Agent boundary:** the target adaptation agent may see train data, approved reports, and redacted aggregate telemetry. It must not see raw validation/test rows, registry credentials, active evaluator code, telemetry stores, or raw production traces.

## Module Map

```text
User-reviewed Target Definition
        |
        v
Target Definition Module
        |\
        | \--> Release Runtime Module creates Release with no local artifacts
        |
        v
Compile Orchestration Module <---- RecompileRequest
        |
        v
Snapshot And Reference Module
        |
        v
Agent Workspace Module <---- current Release / AgentVisibleReports
        |
        v
Candidate submission
        |
        v
Artifact Worker Module
        |
        v
Candidate Evaluation Module
        |
        v
Release Runtime Module
        |
        v
Runtime Feedback Overview
        |
        +--> Runtime Trace And Metrics Module
        +--> Audit Monitoring Module
        +--> Telemetry Evidence And Recompile Module
              |
              v
        Compile Orchestration Module for future compile
```

## Module Boundaries

### Target Definition Module

Inputs:

- `target_path` from CLI or config.
- Target files: `target.yaml`, JSON schemas, `contract.py`, optional `reference.py`, `data.yaml`, contract tests.

Outputs:

- `TargetDefinition` to Snapshot And Reference, Agent Workspace, Candidate Evaluation, Release Runtime.
- `ContractHash` to Snapshot, Candidate, Report, Release, Trace, Audit, and Telemetry Evidence.
- `TargetCheckReport` to CLI and user.
- `TargetRuntimeContract` to Snapshot And Reference, Candidate Evaluation, Release Runtime, and runtime feedback modules.

Responsibilities:

- Load target files.
- Validate the target definition.
- Expose target-owned functions as ordinary callables.
- Hash the active target definition.
- Keep target logic outside core interpretation.

### Compile Orchestration Module

Inputs:

- `RecompileRequest` from Telemetry Evidence And Recompile, monitoring automation, scheduler policy, or CLI/user.
- `TargetDefinition`, `TargetRuntimeContract`, and target check result from Target Definition.
- Current stable `Release` from Release Runtime.
- Optional `TelemetryDataSource` carried by a `RecompileRequest`.
- Existing `ConsumedRowsManifest` values from Snapshot And Reference and Candidate Evaluation.
- Compile budget, reference broker settings, and scheduler policy from CLI/config.

Outputs:

- Snapshot build invocation to Snapshot And Reference; receives `SnapshotBuildResult`.
- Compile run and agent launch invocations to Agent Workspace; receives `CompileRun`, `AgentAttempt`, and `AgentSessionHandle`.
- `CompileLaunchDecision`, including `snapshot_cutoff_time` and `snapshot_options` for accepted launches, to CLI/user, monitoring, and compile run storage.

Responsibilities:

- Decide whether a recompile may start under budget, concurrency, reference-quality, and target/contract scope rules.
- Call Snapshot And Reference to build the next frozen Snapshot.
- Call Agent Workspace to create the compile run and launch exactly one target adaptation agent.
- Defer or reject compile requests without evaluating Candidate quality.
- Avoid becoming a Candidate evaluator, Release approver, telemetry evidence builder, or target-specific optimizer.

### Snapshot And Reference Module

Inputs:

- `TargetDefinition` from Target Definition.
- User data sources declared by the target.
- Optional `TelemetryDataSource` from Compile Orchestration, originally produced by Telemetry Evidence And Recompile.
- Reference/L4 budget and broker config from Compile Orchestration/CLI config.

Outputs:

- `Snapshot` to Agent Workspace and Candidate Evaluation.
- Train view to Agent Workspace.
- Validation/test views to Candidate Evaluation only.
- `ReferenceUsageLedger` to Candidate Evaluation reports and cost reports.
- `ReferenceQualificationReport` to CLI/user, Agent Workspace preflight, and Candidate Evaluation reports.
- `ConsumedRowsManifest` to future Snapshot builds when validation or test rows are exposed.

Responsibilities:

- Read source data.
- Normalize, deduplicate, and group records using target contract functions.
- Call reference/L4 through a Core-owned broker when needed.
- Prove reference/L4 quality before compile starts, and distinguish agreement with L4 from independent gold correctness.
- Freeze train/validation/test boundaries.
- Ensure the agent only receives train data.

### Agent Workspace Module

Inputs:

- `TargetDefinition` from Target Definition.
- Train view from Snapshot.
- Long-term target workspace baseline from Agent Workspace storage.
- Current `Release` and historical `AgentVisibleReport` summaries from Release Runtime/Candidate Evaluation.
- Agent-visible telemetry summaries from Runtime Feedback.
- Runtime protocol docs from Artifact Worker.
- Compile budget, launch decision, user search guidance, and workspace
  permission policy from Compile Orchestration.

Outputs:

- Agent-managed workspace files.
- `CandidateSubmission` to Artifact Worker and Candidate Evaluation.
- Agent usage ledger to Report.
- Updated target workspace baseline after an accepted Release or explicit carry-forward decision.
- Optional target-change proposals for user review.

Responsibilities:

- Maintain a long-term agent-managed target workspace and create isolated compile-attempt clones from it.
- Launch exactly one target adaptation agent per attempt.
- Let the agent write compile-time scaffolding and L1/L2/L3 runtime source.
- Prevent validation/test and registry credentials from entering the workspace.
  User-enabled network research or workspace-local dependency installation
  authorization does not change the hidden holdout boundary.
- Run target-adaptation agents through the macOS `sandbox-exec` launch path for
  now; platforms without `sandbox-exec` need a future external runner/container
  design instead of a Python-level sandbox fallback.
- Close the agent session before test evaluation.

### Artifact Worker Module

Inputs:

- `CandidateSubmission` from Agent Workspace.
- Worker protocol settings from Core.
- Runtime artifact directories from submissions.

Outputs:

- Frozen `Artifact` package metadata to Candidate Evaluation and Release Runtime.
- Worker protocol check results to Candidate Evaluation.
- Layer attempt results to Candidate Evaluation and Release Runtime.

Responsibilities:

- Validate artifact manifests and packages.
- Start, call, and stop workers.
- Enforce the `accept`/`abstain` protocol.
- Treat invalid responses, crashes, and timeouts as safe fallback events.
- Ensure artifacts cannot call L4 or the next layer.

### Candidate Evaluation Module

Inputs:

- `CandidateSubmission` and frozen `Artifact` metadata from Agent Workspace/Artifact Worker.
- `TargetDefinition` from Target Definition.
- `Snapshot` validation/test views from Snapshot And Reference.
- `ReferenceQualificationReport` and reference provenance from Snapshot And Reference.
- Current baseline `Release` from Release Runtime.
- Agent, reference, audit, and local training/search usage ledgers from Agent Workspace, Snapshot And Reference, and evaluation tooling.

Outputs:

- `Report` to CLI/user and Release Runtime.
- `CandidateDecision` to Release Runtime.
- `AgentFeedback` and `AgentVisibleReport` summaries to Agent Workspace.
- `CostLedger` inside Report.
- `ConsumedRowsManifest` when validation/test evidence is consumed.

Responsibilities:

- Freeze and hash Candidates.
- Run protocol, safety, validation, ablation, residual, cascade, latency, and cost checks.
- Compute Precision, Coverage, and Generalization evidence.
- Split metrics by reference source so L4 agreement is not reported as gold correctness.
- Compute serving and compile-time cost without mixing compile cost into serving savings.
- Compare Candidate against current Release.
- Keep raw validation/test rows hidden from the agent.
- Run test only after the agent session is closed, and mark the test window consumed when the test gate fails or any test result becomes user/agent visible.

### Release Runtime Module

Inputs:

- Accepted `CandidateDecision` and `Report` from Candidate Evaluation for compiled Releases.
- `TargetDefinition`, target check result, and L4/reference broker settings for Releases with no local artifacts.
- Target approval or pre-approved release policy from CLI/user.
- Runtime requests from serving entrypoints.
- Optional cache entries from its own cache store.

Outputs:

- `Release` records to registry.
- Runtime responses to callers.
- `ServingResult` for trace writing, audit capture, and evidence decisions.
- `Trace` records with redacted layer attempts to Runtime Trace And Metrics.
- Channel changes for shadow/canary/stable/rollback.

Responsibilities:

- Create immutable Releases, including cold-start Releases with no local artifacts.
- Apply per-layer routing switches from `RoutingSettings.enabled_layers`; disabled layers are skipped even when their artifacts exist.
- Route cache hits directly to response/trace without running cascade.
- Route cache misses through enabled and present L1/L2/L3 artifacts, then Core-owned L4 fallback.
- Keep the request on one fixed Release.
- Validate outputs using target contract.
- Apply timeouts, circuit breakers, canary routing, and rollback.

### Runtime Feedback Modules

Inputs:

- `Trace` records from Release Runtime to Runtime Trace And Metrics.
- Random audit and risk audit settings from TargetDefinition/Release.
- L4/reference broker from Snapshot And Reference.
- Drift thresholds from target requirements.
- User corrections or business feedback from external systems.

Outputs:

- Runtime metrics, online quality summaries, drift signals, and runtime failure decisions.
- Audit records and audit summaries to user reports and monitoring.
- Telemetry-derived data sources to Compile Orchestration, then Snapshot And Reference.
- Recompile requests to Compile Orchestration.
- Agent-visible telemetry summaries to Agent Workspace.

Responsibilities:

- Runtime Trace And Metrics records durable traces, computes runtime metrics, and detects failure or drift.
- Audit Monitoring records unbiased random audits separately from risk audits.
- Telemetry Evidence And Recompile builds approved telemetry evidence from L4 fallbacks, audits, and user corrections before future snapshots ingest it.
- Runtime feedback emits recompile requests; Compile Orchestration decides whether and when to launch the next compile.

## Primary Data Flows

### Target Registration

```text
User files
  -> Target Definition Module
  -> TargetCheckReport
  -> ContractHash
  -> Release Runtime
  -> Release with no local artifacts
```

The target can start serving after target checks and the Core-owned L4/reference path are usable. At this point the Release simply has no L1/L2/L3 artifacts, so cache misses route to L4. The target is not ready for lower-layer externalization until a later compile builds a Snapshot and its schema, contract tests, reference parser checks, split grouping checks, redaction checks, and reference/L4 qualification pass or are explicitly marked insufficient.

### Cold Start Runtime

```text
TargetDefinition + target checks + L4/reference broker
  -> Release Runtime
  -> Release with no local artifacts
  -> runtime requests route directly to Core-owned L4
  -> Trace / audit / telemetry evidence
  -> RecompileRequest
```

The system can serve before any compile succeeds. In that state the current Release has no L1/L2/L3 artifacts and no Candidate Report. It still has a fixed Release id, contract hash, runtime traces, audit, cost accounting, and approved telemetry evidence. The first lower-layer compile is just a recompile request produced from this runtime state or requested manually by the user.

### Compile

```text
RecompileRequest + source data + current Release + compile budget
  -> Compile Orchestration
  -> Snapshot And Reference
  -> Snapshot + ReferenceQualificationReport
  -> Agent Workspace with train only
  -> CandidateSubmission
```

Compile cannot launch the target adaptation agent when the ReferenceQualificationReport fails. If it is insufficient, the runtime remains on the current Release, which may have no local artifacts, while the system collects more gold/human/reference evidence or waits for explicit user approval. The agent may create scaffolding and runtime source, but it cannot see validation/test data or alter target correctness.

A user-requested compile is still represented as a `RecompileRequest`; it may simply carry no `TelemetryDataSource`. This keeps cold start, manual compile, scheduled compile, and runtime-triggered compile on one path.

### Candidate Evaluation

```text
CandidateSubmission
  -> Artifact Worker protocol checks
  -> Candidate Evaluation on validation
  -> aggregate feedback to Agent
  -> repeat until budget/candidates exhausted
  -> close Agent
  -> Candidate Evaluation on test
  -> test Report
  -> compare candidates
  -> CandidateDecision
  -> final Report
```

Validation feedback is aggregate. Test results never return to the same agent run.
If the test gate fails, or if any test result is shown to a user or any agent, Core records the test window as consumed and future compile claims must use a new test window.

### Release And Runtime

```text
Accepted Candidate + approval
  -> Release
  -> shadow/canary/stable channel
  -> request
  -> optional cache hit or enabled/present L1 -> L2 -> L3 -> Core-owned L4 fallback
  -> RuntimeResponse
  -> runtime feedback handling
```

Artifacts never call L4 and never call each other. Core controls fallback, including L4 fallback failures. Trace, audit, approved telemetry evidence, and future Snapshot ingestion follow the Data Lifetime Model and Canonical Runtime To Recompile Flow in the Runtime Feedback Overview.

### Future Compile

```text
runtime observations + audits + user corrections
  -> TelemetryDataSource
  -> RecompileRequest
  -> Compile Orchestration
  -> Snapshot And Reference build_snapshot(telemetry_source=...)
  -> new Snapshot
  -> Agent Workspace for new Compile
```

Random audits support unbiased Precision estimates. Risk audits help find problems but are reported separately. Future Snapshot records come from approved telemetry evidence, not from redacted/hash Trace or AuditRecord fields.

## Standard Runtime Request Contract

Worker request:

```json
{
  "request_id": "r-123",
  "input": {},
  "deadline_ms": 20
}
```

Worker accept:

```json
{
  "decision": "accept",
  "output": {},
  "confidence": 0.997,
  "reason": "exact_pattern_42"
}
```

Worker abstain:

```json
{
  "decision": "abstain",
  "confidence": 0.41,
  "reason": "outside_supported_region"
}
```

`confidence` and `reason` are artifact-owned. Core records them but does not reinterpret every target with a universal threshold.

## Required Metrics

Reports must show:

- Precision among local accepts.
- Random audit Precision lower bound and wrong-accept upper bound for runtime rollback decisions.
- Metrics split by reference source, with versioned L4 rows labeled as agreement rather than gold correctness.
- Coverage handled by enabled L1/L2/L3 layers, excluding cache hits.
- Per-layer attempt counts, accept counts, and Coverage for L1, L2, and L3.
- Configured-disabled layers reported separately from missing artifacts, health failures, and abstentions.
- Wrong accept count and wrong accept rate.
- Validation to test Precision and Coverage change.
- Cohort floor across validation cohorts.
- Coverage retention from validation to test.
- Worst user-reviewed slice when enough samples exist.
- Candidate rank stability across validation shards.
- Future audit status when online evidence exists.
- Random audit attempt, success, and reference failure counts.
- Minimum accepted sample and minimum slice sample checks.
- Latency and cost by layer and by full cascade.
- L4 fallback share and cost.
- Cache hit rate separately.
- Serving L4 cost, local serving compute cost, random audit cost, risk audit cost, compile agent cost, reference labeling cost, and local training/search cost.
- `saving_per_1000_requests`, `compile_cost`, and `estimated_payback_requests`.
- Generalization evidence as a panel, not as a single magic score.
