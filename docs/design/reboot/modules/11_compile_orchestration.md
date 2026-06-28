# Compile Orchestration Module

## Purpose

The Compile Orchestration module is the thin Core caller that turns a `RecompileRequest` into a new compile attempt.

It owns scheduling, budget checks, concurrency checks, and the call order between Snapshot And Reference and Agent Workspace. It does not evaluate Candidates, interpret target fields, create Releases, generate telemetry evidence, or decide that a Candidate is good.

Manual user compile and the first compile after cold start are represented as `RecompileRequest` values. They may carry no `TelemetryDataSource`, but they still use the same base Release, budget, Snapshot, and Agent Workspace path as later recompiles. The base Release may simply have no lower-layer artifacts yet.

This module follows the [System Invariants](../00_overall_design.md#system-invariants), especially target and contract scope, snapshot scope, hidden holdout consumption, and the agent boundary.

## Boundary

Inputs:

- `RecompileRequest` from Telemetry Evidence And Recompile, CLI/user, monitoring automation, or scheduler policy.
- `TargetDefinition`, `TargetRuntimeContract`, and `TargetCheckReport` from Target Definition.
- Current stable `Release` from Release Runtime.
- Optional `TelemetryDataSource` carried by a `RecompileRequest`.
- Existing `ConsumedRowsManifest` values from Snapshot And Reference and Candidate Evaluation.
- `ReferenceBroker` settings, `CompileBudget`, `CompileOptions`, and `SchedulerPolicy` from CLI/config.
- Current active compile jobs from the compile run store.

Outputs:

- Snapshot build invocation to Snapshot And Reference; receives `SnapshotBuildResult`.
- Compile run and agent launch invocations to Agent Workspace; receives `CompileRun`, `AgentAttempt`, and `AgentSessionHandle`.
- `CompileLaunchDecision` to CLI/user, monitoring, and compile run storage.
- Deferred or rejected compile decision when budget, concurrency, target checks, reference qualification, or scope matching fails.

This module must not output raw validation/test rows, Candidate quality decisions, Release approvals, or runtime telemetry records.

## Data Flow

Recompile:

```text
Telemetry Evidence And Recompile
  -> RecompileRequest
  -> Compile Orchestration
  -> Snapshot And Reference build_snapshot(telemetry_source=...)
  -> Agent Workspace create_compile_run(...)
  -> Agent Workspace launch_target_adaptation_agent(...)
```

The module passes the resulting `Snapshot` to Agent Workspace only through `create_compile_run` and workspace mounts. It does not expose validation or test views to the agent.

## Data Types

### `SchedulerPolicy`

Fields:

- `max_concurrent_compiles: int`
- `allow_monitoring_recompile: bool`
- `allow_scheduled_recompile: bool`
- `require_user_approval_for_insufficient_reference: bool`
- `default_compile_budget: CompileBudget`
- `default_snapshot_options: SnapshotOptions`

### `CompileLaunchDecision`

Fields:

- `status: Literal["accepted", "deferred", "rejected"]`
- `target_name: str`
- `contract_hash: str`
- `base_release_id: str`
- `reason: str`
- `budget: CompileBudget | None`
- `telemetry_source_id: str | None`
- `snapshot_cutoff_time: datetime | None`
- `snapshot_options: SnapshotOptions | None`
- `created_at: datetime`

### `CompileRunStore`

Fields:

- `runs: dict[str, CompileRun]`

This is the minimal Core-owned compile run store used by orchestration to record accepted launches and by schedulers to provide current active jobs back into `plan_compile_launch`. It is deliberately a plain store boundary rather than a scheduler framework.

## Functions

### `plan_compile_launch`

Input:

- `definition: TargetDefinition`
- `contract: TargetRuntimeContract`
- `target_check: TargetCheckReport`
- `request: RecompileRequest`
- `base_release: Release`
- `consumed_manifests: list[ConsumedRowsManifest]`
- `active_jobs: list[CompileRun]`
- `policy: SchedulerPolicy`

Output:

- `CompileLaunchDecision`

Purpose:

- Verify that the request matches the active target name and contract hash.
- Verify that the base Release matches the same target and contract.
- Enforce scheduler policy for user, monitoring, and scheduled triggers.
- Enforce compile budget and max concurrent compile limits.
- Choose and persist `snapshot_cutoff_time` and `snapshot_options` for accepted launches.
- If the request carries a `TelemetryDataSource`, hard-fail accepted launches unless `telemetry_source.cutoff_time <= snapshot_cutoff_time`.
- For deferred or rejected launches, leave snapshot fields `None`.
- Decide whether the compile should be accepted, deferred, or rejected before any Snapshot or Agent work starts.

Used by:

- CLI compile.
- monitoring automation.
- scheduled recompile checks.

### `start_compile_launch`

Input:

- `decision: CompileLaunchDecision`
- `definition: TargetDefinition`
- `contract: TargetRuntimeContract`
- `target_check: TargetCheckReport`
- `data_config: DataConfig`
- `request: RecompileRequest`
- `base_release: Release`
- `consumed_manifests: list[ConsumedRowsManifest]`
- `broker: ReferenceBroker`
- `workspace_store: WorkspaceStore`
- `compile_run_store: CompileRunStore`
- `compile_options: CompileOptions`
- `agent_options: AgentAttemptOptions`
- `report_views: list[AgentVisibleReport]`
- `telemetry_summaries: list[AgentVisibleTelemetrySummary]`

Output:

- `CompileRun`
- `AgentAttempt`
- `AgentSessionHandle`

Purpose:

- Hard-fail unless `decision.status == "accepted"`.
- Hard-fail unless `decision.snapshot_cutoff_time` and `decision.snapshot_options` are present.
- Hard-fail if the request carries a `TelemetryDataSource` whose `cutoff_time` is later than `decision.snapshot_cutoff_time`.
- Call Snapshot And Reference `build_snapshot(...)`, passing the request's telemetry source, `decision.snapshot_cutoff_time`, consumed manifests, reference broker, and `decision.snapshot_options`.
- Refuse to continue when snapshot build or reference qualification is not acceptable under policy.
- If reference qualification is insufficient, continue only when the compile
  launch options explicitly approve insufficient reference evidence for this
  run.
- Load the target workspace, create a `CompileRun`, create an isolated `AgentAttempt`, mount allowed inputs, write the agent brief, and launch the single target adaptation agent.
- Pass selected historical `AgentVisibleReport` summaries and
  `AgentVisibleTelemetrySummary` values through to Agent Workspace mounting.
- Record the compile launch in the compile run store.

Used by:

- CLI compile.
- monitoring automation when scheduler policy permits automatic launch.
- scheduled recompile when scheduler policy permits automatic launch.

## Invariants

- Compile Orchestration only starts work for one target name and one contract hash at a time.
- It never passes raw validation/test rows to Agent Workspace.
- It never evaluates Candidate quality; Candidate Evaluation owns that.
- It never creates or approves a Release; Release Runtime owns that.
- It never turns runtime traces into Snapshot rows; Telemetry Evidence And Recompile and Snapshot And Reference own that path.
- It may defer or reject a compile for budget, concurrency, policy, stale base Release, failed target checks, or failed reference qualification.
