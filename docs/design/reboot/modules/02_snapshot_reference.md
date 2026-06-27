# Snapshot And Reference Module

## Purpose

The Snapshot And Reference module freezes data for a compile run and owns all Core-brokered L4/reference calls. It creates train/validation/test boundaries, records provenance, and ensures the target adaptation agent only sees train data.

This module turns mutable data sources into immutable evidence. It is the main defense against accidental validation/test leakage.

This module follows the [System Invariants](../00_overall_design.md#system-invariants), especially snapshot scope, hidden holdout consumption, and the runtime-to-future-compile bridge.

## Boundary

Inputs:

- `TargetDefinition` and `TargetRuntimeContract` from Target Definition.
- Target `data.yaml` source declarations.
- Optional `TelemetryDataSource` from Compile Orchestration, originally produced by Telemetry Evidence And Recompile.
- Optional existing reference cache.
- Reference budget and broker settings from Compile Orchestration/CLI config.

Outputs:

- `Snapshot` manifest to Agent Workspace and Candidate Evaluation.
- `TrainView` to Agent Workspace.
- `ValidationView` and `TestView` to Candidate Evaluation only.
- `ReferenceUsageLedger` to Candidate Evaluation reports and cost reports.
- `ReferenceFailureReport` to target check or compile preflight.
- `ReferenceQualificationReport` to target check, compile preflight, and Candidate Evaluation reports.
- `ConsumedRowsManifest` to future snapshot builds.

This module must not output validation/test raw rows to the agent.

## Data Types

### `SourceRecord`

Fields:

- `record_id: str`
- `input: dict`
- `reference_output: dict | None`
- `reference_source: Literal["gold", "human", "versioned_l4", "verified_l4", "user_feedback"] | None`
- `split_eligibility: list[Literal["train", "validation_candidate", "test_candidate"]]`
- `source_name: str`
- `source_timestamp: datetime | None`
- `metadata: dict`

### `SnapshotRecord`

Fields:

- `snapshot_record_id: str`
- `normalized_input_key: str`
- `split_group_key: str`
- `input: dict`
- `reference_output: dict`
- `reference_source: str`
- `split_eligibility: list[Literal["train", "validation_candidate", "test_candidate"]]`
- `reference_version: str | None`
- `slice_tags: list[str]`
- `source_provenance: dict`

### `Snapshot`

Fields:

- `snapshot_id: str`
- `snapshot_digest: str`
- `target_name: str`
- `contract_hash: str`
- `created_at: datetime`
- `cutoff_time: datetime`
- `source_watermarks: dict[str, str]`
- `telemetry_source_id: str | None`
- `records_digest: str`
- `split_manifest_digest: str`
- `reference_ledger_digest: str`
- `train_count: int`
- `validation_count: int`
- `test_count: int`
- `storage_uri: str`

`cutoff_time` is the maximum event time allowed into the snapshot. `source_watermarks` records the effective source-specific boundaries, such as file digests, source timestamps, or upstream cursor values. `telemetry_source_id` records the approved telemetry source used, if any.

### `SnapshotView`

Fields:

- `snapshot_id: str`
- `split: Literal["train", "validation", "test"]`
- `records_uri: str`
- `redaction_level: Literal["raw", "redacted", "aggregate"]`
- `record_count: int`

The agent receives only a train view, usually redacted when the target requires it.

### `ReferenceQualificationReport`

Fields:

- `target_name: str`
- `contract_hash: str`
- `reference_version: str`
- `gold_sample_count: int`
- `l4_agreement_count: int`
- `gold_correct_count: int | None`
- `agreement_rate: float`
- `gold_precision: float | None`
- `parse_failure_rate: float`
- `schema_failure_rate: float`
- `latency: dict`
- `cost: dict`
- `status: Literal["pass", "fail", "insufficient"]`
- `notes: list[str]`

This report states whether L4/reference is good enough to use for compile. If there is no independent gold or human sample, it reports agreement with the versioned L4/reference, not absolute correctness.

### `ConsumedRowsManifest`

Fields:

- `snapshot_id: str`
- `split: Literal["validation", "test"]`
- `record_ids: list[str]`
- `normalized_input_keys: list[str]`
- `split_group_keys: list[str]`
- `reason: str`
- `consumed_at: datetime`
- `visible_to: Literal["user", "agent", "external_report"]`
- `replacement_required: bool`

The stable keys let future snapshot builds identify consumed holdout examples even when new snapshot record ids are generated.

## Functions

### `load_data_config`

Input:

- `definition: TargetDefinition`

Output:

- `DataConfig`

Purpose:

- Read target-owned data source declarations.
- Validate allowed source types and required credentials references.
- Do not read actual records yet.

Used by:

- `collect_source_records`
- CLI `target check`

### `collect_source_records`

Input:

- `definition: TargetDefinition`
- `data_config: DataConfig`
- `telemetry_source: TelemetryDataSource | None`
- `cutoff_time: datetime`

Output:

- `list[SourceRecord]`

Purpose:

- Load records from declared user data and approved telemetry-derived data.
- Do not load records whose source event time is after `cutoff_time`.
- Treat `TelemetryDataSource` as the only runtime-derived input. Do not read Trace, AuditRecord, or SecureAuditPayload directly.
- Reject `TelemetryDataSource` when its `target_name` or `contract_hash` does not match the current `TargetDefinition`.
- Reject `TelemetryDataSource` when `telemetry_source.cutoff_time > cutoff_time`.
- For telemetry-derived data, ingest only rows already filtered by `TelemetryDataSource.cutoff_time`; do not load approved evidence with `created_at` after that cutoff.
- Preserve source provenance.
- Preserve split eligibility from telemetry evidence so approved training-only rows cannot enter validation/test and evaluation-only rows cannot enter agent train.
- Require per-record split eligibility when the telemetry source contains records with different permissions; do not replace mixed permissions with a broader batch default.
- For non-telemetry user data, assign split eligibility from `data.yaml` or the target default data policy.
- Avoid mixing future online evidence into an older snapshot.

Used by:

- `build_snapshot`

### `validate_source_records`

Input:

- `contract: TargetRuntimeContract`
- `records: Iterable[SourceRecord]`

Output:

- `ValidatedSourceBatch`

Purpose:

- Validate inputs.
- Validate existing reference outputs when present.
- Report malformed records before reference labeling or splitting.

Used by:

- `build_snapshot`
- target checks

### `reference_missing_outputs`

Input:

- `contract: TargetRuntimeContract`
- `records: Iterable[SourceRecord]`
- `broker: ReferenceBroker`
- `budget: ReferenceBudget`

Output:

- `ReferenceLabelResult`

Purpose:

- Call L4/reference only for records missing accepted reference outputs.
- Use Core-owned retries, timeout, rate limit, cache, usage, and cost ledger.
- Parse responses through target reference adapter and output validation.
- Preserve parse/schema failures instead of silently dropping them.

Used by:

- `build_snapshot`
- runtime audit labeling

### `qualify_reference_baseline`

Input:

- `definition: TargetDefinition`
- `contract: TargetRuntimeContract`
- `records: Iterable[SourceRecord]`
- `broker: ReferenceBroker`
- `qualification_options: ReferenceQualificationOptions`

Output:

- `ReferenceQualificationReport`

Purpose:

- Verify that the configured L4/reference path is reliable enough before lower-layer externalization starts.
- Measure parse failures, schema failures, latency, cost, and agreement against existing gold/human/reference outputs.
- Separately report independent gold correctness when gold/human labels exist.
- Fail or report insufficient evidence when the reference itself is not trustworthy enough.

Used by:

- `check_target_definition`
- `build_snapshot`
- Agent Workspace compile preflight.

### `call_reference`

Input:

- `contract: TargetRuntimeContract`
- `input_value: ValidatedInput`
- `broker: ReferenceBroker`
- `request_context: ReferenceContext`

Output:

- `ReferenceCallResult`

Purpose:

- Build request using target reference adapter.
- Execute through Core broker.
- Parse and validate response.
- Return output, usage, cost, latency, finish status, and errors.

Used by:

- `reference_missing_outputs`
- Release Runtime L4 fallback
- Telemetry Evidence And Recompile module.

### `deduplicate_records`

Input:

- `contract: TargetRuntimeContract`
- `records: Iterable[SourceRecord]`

Output:

- `DeduplicatedRecords`

Purpose:

- Compute `normalized_input_key` for each record.
- Remove exact duplicates or mark duplicate groups according to target data policy.
- When duplicate records have different `split_eligibility`, merge by intersection only; never keep a broader eligibility list from one source record.
- When duplicate records have conflicting reference outputs, reference sources, or no shared legal split after intersection, fail the snapshot build or exclude the duplicate group with an explicit reason according to snapshot options.
- Preserve duplicate source provenance, reference provenance, and the effective narrowed split eligibility for reporting and later split planning.
- Preserve duplicate counts for reporting.

Used by:

- `build_snapshot`

### `assign_split_groups`

Input:

- `contract: TargetRuntimeContract`
- `records: Iterable[SourceRecord]`

Output:

- `GroupedRecords`

Purpose:

- Compute `split_group_key` for each record.
- Keep related records together during train/validation/test split.
- Record group sizes for leakage diagnostics.

Used by:

- `build_split_plan`

### `build_split_plan`

Input:

- `grouped_records: GroupedRecords`
- `split_options: SplitOptions`
- `consumed_manifests: list[ConsumedRowsManifest]`

Output:

- `SplitManifest`

Purpose:

- Assign groups to train, validation, and test.
- Prefer chronological/source-aware/group-aware splits when metadata supports it.
- Respect every record's `split_eligibility`.
- Apply `ConsumedRowsManifest` and enforce the hidden holdout consumption invariant before assigning hidden validation/test rows.
- Never place a row into train unless it is eligible for `train`.
- Never place a row into validation unless it is eligible for `validation_candidate`.
- Never place a row into test unless it is eligible for `test_candidate`.
- For each split group, compute the intersection of legal splits across its records; fail the snapshot build or exclude the group with an explicit reason when no common legal split exists.
- Guarantee deterministic split assignment from a seed and input manifest.

Used by:

- `build_snapshot`

### `materialize_snapshot_records`

Input:

- `contract: TargetRuntimeContract`
- `records: Iterable[SourceRecord]`
- `split_manifest: SplitManifest`

Output:

- `list[SnapshotRecord]`

Purpose:

- Create immutable snapshot records with validated input, validated reference output, split group, normalized key, slice tags, and provenance.
- Preserve `split_eligibility` in each snapshot record for auditability.

Used by:

- `write_snapshot`

### `write_snapshot`

Input:

- `definition: TargetDefinition`
- `records: list[SnapshotRecord]`
- `split_manifest: SplitManifest`
- `reference_ledger: ReferenceUsageLedger`
- `cutoff_time: datetime`
- `source_watermarks: dict[str, str]`
- `telemetry_source_id: str | None`
- `storage: SnapshotStore`

Output:

- `Snapshot`

Purpose:

- Persist snapshot records and manifests in content-addressed or immutable storage.
- Record contract hash, cutoff time, source watermarks, optional telemetry source id, and digests.
- Return a snapshot handle used by later modules.

Used by:

- `build_snapshot`

### `build_snapshot`

Input:

- `definition: TargetDefinition`
- `contract: TargetRuntimeContract`
- `data_config: DataConfig`
- `telemetry_source: TelemetryDataSource | None`
- `consumed_manifests: list[ConsumedRowsManifest]`
- `broker: ReferenceBroker`
- `cutoff_time: datetime`
- `snapshot_options: SnapshotOptions`

Output:

- `SnapshotBuildResult`

Purpose:

- Orchestrate data loading, validation, reference labeling, deduplication, grouping, split assignment, and writing.
- Pass `cutoff_time` into source collection so user data and telemetry data are frozen to the same build window.
- Pass the optional `TelemetryDataSource` into `collect_source_records` so L4 fallbacks, random audits, and user corrections approved for future compile can enter the new Snapshot.
- Enforce the target and contract scope invariant for telemetry sources before any records are loaded.
- Hard-fail when a telemetry source was approved after the Snapshot freeze point: `telemetry_source.cutoff_time > cutoff_time`.
- Pass prior `ConsumedRowsManifest` values into split planning.
- Record the effective `cutoff_time`, per-source watermarks, and telemetry source id in the Snapshot manifest.
- Rely on the Runtime Feedback Overview Data Lifetime Model for runtime-derived evidence eligibility; this module only validates and splits the resulting source records.
- Fail if reference quality, parse failures, or split leakage makes the snapshot unfit for compile.

Used by:

- Compile Orchestration module for recompile, including the first compile after cold start.

### `load_snapshot_view`

Input:

- `snapshot: Snapshot`
- `split: Literal["train", "validation", "test"]`
- `redaction_level: Literal["raw", "redacted", "aggregate"]`

Output:

- `SnapshotView`

Purpose:

- Produce a read handle to one split.
- Enforce access rules: Agent Workspace can request train only; Candidate Evaluation can request validation/test.

Used by:

- Agent Workspace
- Candidate Evaluation

### `export_train_view_for_agent`

Input:

- `snapshot: Snapshot`
- `contract: TargetRuntimeContract`
- `agent_view_options: AgentViewOptions`
- `output_dir: Path`

Output:

- `TrainViewManifest`

Purpose:

- Export allowed train rows to the agent sandbox.
- Apply target redaction where required.
- Include reference provenance and slice tags only when allowed by target policy.

Used by:

- Agent Workspace module.

### `mark_consumed_holdout_rows`

Input:

- `snapshot: Snapshot`
- `split: Literal["validation", "test"]`
- `records: list[SnapshotRecordId]`
- `reason: str`
- `visible_to: Literal["user", "agent", "external_report"]`

Output:

- `ConsumedRowsManifest`

Purpose:

- Mark validation or test rows that were declassified, exposed for debug, or used in a human-visible row-level report.
- Record consumed snapshot record ids, normalized input keys, and split group keys.
- Ensure consumed rows cannot remain in hidden validation/test for future claims.
- Require a new validation or test window when consumed rows affect the held-out evidence.
- This function should be rare and explicit.

Used by:

- Candidate Evaluation when row-level debug is intentionally declassified.
- Candidate Evaluation after row-level test results are exposed.
- Future snapshot creation.

## Invariants

- Snapshot split membership is frozen before the agent starts.
- Agent receives train data only.
- Validation/test raw rows never enter the agent workspace.
- Reference calls are brokered by Core, not artifacts or generated code.
- Reference usage and cost are recorded per call.
- L4/reference qualification is recorded before compile and distinguishes agreement from gold correctness.
- Exposed validation/test rows are marked consumed and replaced by future holdout windows.
- Existing gold/human/reference provenance is preserved; L4 agreement is not mislabeled as absolute gold correctness.

## Alignment Against 0626-2

- Implements immutable Snapshot and train/validation/test boundary.
- Keeps L4/reference credentials, retries, cache, usage, and cost in Core.
- Lets target code define request/response parsing without owning credentials or evaluation authority.
- Supports future compile from approved telemetry evidence through `TelemetryDataSource` without leaking runtime evidence into old snapshots.
