# Target Definition Module

## Purpose

The Target Definition module loads and checks the user-reviewed task boundary. It exposes target-owned schema, correctness, grouping, redaction, and reference adapter functions to the rest of Core without letting Core understand target semantics.

The target definition is the small human-owned part of the system. Agent-generated compile-time scaffolding and runtime artifacts may depend on it, but cannot modify the active version.

## Boundary

Inputs:

- `target_path`: filesystem path selected by CLI/config.
- `target.yaml`: target metadata and requirements.
- `schemas/input.json`, `schemas/output.json`: JSON schemas.
- `contract.py`: user-reviewed correctness and data-boundary functions.
- Optional `reference.py`: target-owned reference request/response adapter.
- `data.yaml`: data source declarations.
- `tests/`: examples and contract tests.

Outputs:

- `TargetDefinition`: consumed by Snapshot And Reference, Agent Workspace, Candidate Evaluation, Release Runtime, and runtime feedback modules.
- `ContractHash`: consumed by Snapshot, Candidate, Report, Release, Trace, Audit, Telemetry Evidence, and cache keys.
- `TargetCheckReport`: consumed by CLI/user and stored in compile records.
- `TargetRuntimeContract`: a restricted callable view used during evaluation and runtime.
- `TelemetryPrivacyPolicy`: target-owned runtime policy consumed by runtime feedback modules through `TargetDefinition.runtime_config`.

This module must not output:

- A release decision.
- A layer-specific plan.
- A target-specific optimization rule for Core.
- Validation/test split membership selected by the agent.

## Data Types

### `TargetDefinition`

Fields:

- `name: str`
- `version: str`
- `target_path: Path`
- `input_schema: JsonSchema`
- `output_schema: JsonSchema`
- `contract_module_digest: str`
- `reference_module_digest: str | None`
- `requirements: TargetRequirements`
- `data_config: DataConfig`
- `runtime_config: RuntimeConfig`
- `contract_hash: str`

### `RuntimeConfig`

Fields:

- `telemetry_privacy_policy: TelemetryPrivacyPolicy`

`TelemetryPrivacyPolicy` belongs to the target runtime configuration because only the target owner can decide which runtime observations or user corrections may become future compile data. Runtime feedback modules consume the policy, but do not own it.

### `TargetRequirements`

Fields:

- `precision_min: float`
- `wrong_accept_rate_max: float | None`
- `validation_test_precision_drop_max: float | None`
- `validation_test_coverage_retention_min: float | None`
- `cohort_precision_floor_min: float | None`
- `candidate_rank_stability_min_shards: int | None`
- `future_audit_required_for_auto_release: bool`
- `min_accepted_samples: int`
- `min_slice_samples: int`
- `critical_slices: list[str]`
- `critical_slice_precision_min: float | None`
- `critical_slice_coverage_min: float | None`
- `coverage_objective: Literal["maximize", "hold", "none"]`
- `p95_latency_ms_max: int | None`
- `memory_mb_max: int | None`
- `throughput_per_second_min: float | None`
- `serving_cost_per_1000_max: float | None`
- `random_audit_rate: float`
- `random_audit_reference_failure_rate_max: float | None`
- `manual_approval_required: bool`

### `TargetRuntimeContract`

Callable view:

- `validate_input`
- `validate_output`
- `is_correct`
- `normalize_input`
- `split_group`
- `slice_tags`
- `redact_for_trace`
- `bucket_runtime_metadata`
- optional `build_reference_request`
- optional `parse_reference_response`

The runtime contract is passed as a callable object, not copied into generated artifacts.

## Functions

### `load_target_definition`

Input:

- `target_path: Path`

Output:

- `TargetDefinitionDraft`

Purpose:

- Read `target.yaml`, schema paths, `data.yaml`, and module paths.
- Resolve paths relative to `target_path`.
- Load metadata without executing target contract behavior beyond import-time safety checks.

Used by:

- `check_target_definition`
- CLI `target check`
- CLI `compile`

### `load_contract_module`

Input:

- `contract_path: Path`

Output:

- `ContractModule`

Purpose:

- Load `contract.py` in a restricted import context.
- Verify that required callables exist.
- Do not run evaluation, data loading, or reference calls.

Used by:

- `build_runtime_contract`
- `check_target_definition`

### `load_reference_module`

Input:

- `reference_path: Path | None`

Output:

- `ReferenceModule | None`

Purpose:

- Load optional target reference adapter.
- Verify that adapter functions are present when a custom L4/reference endpoint is declared.
- Keep credentials out of target code; credentials are provided later by the Core broker.

Used by:

- Snapshot And Reference module.
- Target checks.

### `build_runtime_contract`

Input:

- `definition_draft: TargetDefinitionDraft`
- `contract_module: ContractModule`
- `reference_module: ReferenceModule | None`

Output:

- `TargetRuntimeContract`

Purpose:

- Wrap user-reviewed target callables behind a stable Core-facing interface.
- Apply JSON schema validation wrappers around input and output validation.
- Keep target-specific semantics opaque to Core.

Used by:

- Snapshot building.
- Candidate evaluation.
- Runtime output validation.
- Trace redaction.

### `validate_input`

Input:

- `contract: TargetRuntimeContract`
- `value: dict`

Output:

- `ValidatedInput`

Purpose:

- Check input schema and target-level input invariants.
- Return a normalized in-memory value for downstream modules.
- Raise a structured validation error on invalid input.

Used by:

- Snapshot ingestion.
- Runtime serving.
- Evaluation replay.

### `validate_output`

Input:

- `contract: TargetRuntimeContract`
- `value: dict`

Output:

- `ValidatedOutput`

Purpose:

- Check output schema and target-level output invariants.
- Convert malformed artifact or L4 outputs into structured validation errors.

Used by:

- Reference labeling.
- Candidate evaluation.
- Runtime artifact accept handling.
- L4 fallback handling.

### `is_correct`

Input:

- `contract: TargetRuntimeContract`
- `output: ValidatedOutput`
- `reference: ValidatedOutput`

Output:

- `bool`

Purpose:

- Determine sample-level correctness.
- This is the only target-owned correctness function Core calls for Precision and wrong accept metrics.
- It must not inspect split membership, candidate identity, or release state.

Used by:

- Candidate Evaluation module.
- Audit aggregation.

### `normalize_input`

Input:

- `contract: TargetRuntimeContract`
- `input_value: ValidatedInput`

Output:

- `NormalizedInputKey`

Purpose:

- Produce a stable target-owned key for deduplication and cache keys.
- It may canonicalize target semantics, but Core treats the result as opaque text.

Used by:

- Snapshot deduplication.
- Runtime cache.
- Trace compaction.

### `split_group`

Input:

- `contract: TargetRuntimeContract`
- `record: SourceRecord`

Output:

- `SplitGroupKey`

Purpose:

- Place near-duplicate or related examples into the same split group.
- Prevent train/validation/test leakage.
- The agent cannot override this function.

Used by:

- Snapshot split builder.

### `slice_tags`

Input:

- `contract: TargetRuntimeContract`
- `record: SourceRecord`

Output:

- `list[str]`

Purpose:

- Attach user-reviewed slice labels used for worst-slice reporting.
- Core may aggregate by these strings but must not infer target semantics from them.

Used by:

- Candidate Evaluation reports.
- Runtime feedback reports.

### `redact_for_trace`

Input:

- `contract: TargetRuntimeContract`
- `value: dict`

Output:

- `dict`

Purpose:

- Remove or hash fields that should not be written to traces, reports, or agent-visible feedback.
- Applies to inputs, outputs, and reference records when needed.

Used by:

- Release Runtime trace writing.
- Agent feedback generation.
- Audit reporting.

### `bucket_runtime_metadata`

Input:

- `contract: TargetRuntimeContract`
- `metadata: dict`

Output:

- `dict`

Purpose:

- Convert raw runtime request metadata into bounded, allowlisted buckets that can be written to durable Trace and aggregated by Core.
- Drop or bucket any value that could identify a user, request, customer, or target-specific raw payload.
- Core may aggregate the returned keys and values, but must not interpret their target semantics.

Used by:

- Runtime Trace And Metrics `write_trace`.
- Runtime drift summaries.

### `build_reference_request`

Input:

- `contract: TargetRuntimeContract`
- `input_value: ValidatedInput`
- `reference_context: ReferenceContext`

Output:

- `ReferenceRequest`

Purpose:

- Convert a target input into a Core-brokered L4/reference request.
- Target code decides prompt/request shape; Core owns credentials, retries, timeout, cache, usage, and cost.

Used by:

- Snapshot And Reference module.
- Release Runtime L4 fallback.
- Audit sampling.

### `parse_reference_response`

Input:

- `contract: TargetRuntimeContract`
- `response: ReferenceResponse`

Output:

- `ValidatedOutput`

Purpose:

- Convert L4/reference output to the target output schema.
- Preserve parse/schema failure diagnostics without hiding L4 unreliability.

Used by:

- Snapshot And Reference module.
- Release Runtime L4 fallback.
- Audit sampling.

### `compute_contract_hash`

Input:

- `definition: TargetDefinition`

Output:

- `ContractHash`

Purpose:

- Hash target metadata, schemas, contract source digest, reference adapter source digest, requirements that affect evaluation, and reference version.
- Exclude mutable runtime reports and generated workspace files.

Used by:

- Snapshot manifest.
- Candidate manifest validation.
- Release registry.
- Trace records.
- Audit records.
- Telemetry evidence records and data sources.
- Cache keys.

### `check_target_definition`

Input:

- `target_path: Path`
- `check_options: TargetCheckOptions`

Output:

- `TargetCheckReport`

Purpose:

- Run schema checks, contract tests, positive/negative examples, reference parser checks, split group consistency, normalization collision checks, and redaction checks.
- Check only the reference adapter contract here; reference/L4 quality qualification is owned by the Snapshot And Reference module through `qualify_reference_baseline`.
- Return actionable failures before compile is allowed.

Used by:

- CLI `target check`.
- Compile preflight.

### `export_agent_readonly_target_view`

Input:

- `definition: TargetDefinition`
- `output_dir: Path`

Output:

- `TargetViewManifest`

Purpose:

- Copy or mount the approved target definition into the agent sandbox as read-only files.
- Include schemas, contract tests, examples allowed for train use, and runtime protocol docs.
- Exclude validation/test rows, credentials, registry state, and production secrets.

Used by:

- Agent Workspace module.

## Invariants

- The active target definition is user-reviewed and content-hashed.
- The target contract can define correctness but cannot decide release.
- Agent-generated proposals can suggest target changes, but cannot mutate the active target definition.
- Target functions may return opaque labels and keys; Core must not branch on target-specific meanings.
- A changed reference adapter or L4 behavior that affects outputs changes the contract hash.

## Alignment Against 0626-2

- Keeps the target definition small and user-reviewed.
- Prevents target-specific optimization from entering Core.
- Makes correctness explicit through `is_correct`.
- Keeps release and evaluation authority in Core.
- Supports user-reviewed split grouping and redaction.
- Avoids layer-specific compiler concepts.
