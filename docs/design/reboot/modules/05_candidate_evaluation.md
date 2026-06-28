# Candidate Evaluation Module

## Purpose

The Candidate Evaluation module freezes agent submissions, recomputes all official metrics, compares Candidates against the current Release, and produces Reports. It is the only module that can decide whether a Candidate is eligible for Release.

Agent-provided metrics are untrusted. They may be useful local evidence, but the Report is computed by Core using hidden validation/test data and frozen artifact bytes.

This module follows the [System Invariants](../00_overall_design.md#system-invariants), especially hidden holdout consumption and release atomicity.

## Boundary

Inputs:

- `CandidateSubmission` from Agent Workspace.
- `ArtifactPackage` validation/call support from Artifact Worker.
- `TargetDefinition` and `TargetRuntimeContract` from Target Definition.
- `Snapshot` validation/test views from Snapshot And Reference.
- `ReferenceQualificationReport` and reference provenance from Snapshot And Reference.
- Agent, reference, audit, and local training/search usage ledgers.
- Current baseline `Release` from Release Runtime.
- `ClosedAgentAttempt` before test evaluation.

Outputs:

- `Candidate` records to Report and Release Runtime.
- `Report` to CLI/user, Release Runtime, and Runtime Trace And Metrics.
- `CandidateDecision` to Release Runtime.
- `AgentFeedback` to Agent Workspace for validation-stage iteration only.
- `AgentVisibleReport` summaries to Agent Workspace for future compile attempts.

This module must not expose raw validation/test rows to the agent.

## Data Types

### `Candidate`

Fields:

- `candidate_id: str`
- `submission_id: str`
- `compile_id: str`
- `attempt_id: str`
- `target_name: str`
- `contract_hash: str`
- `snapshot_id: str`
- `base_release_id: str`
- `workspace_commit: str`
- `artifacts: dict[LayerName, ArtifactRef | InheritedArtifactRef | None]`
- `routing: RoutingSettings`
- `digest: str`
- `status: Literal["submitted", "frozen", "validation_failed", "validation_passed", "test_failed", "eligible_for_release", "rejected"]`

`routing.enabled_layers` controls the official cascade path for this Candidate. A Candidate may include an artifact for a layer that is disabled in routing; Core may evaluate that artifact in diagnostics, but official Precision, Coverage, latency, cost, and release eligibility follow the enabled route only.

`submission_id`, `compile_id`, and `attempt_id` preserve target-independent
lineage from Agent Workspace. Release-backed workspace baseline advancement
uses these fields to prove that an accepted Release came from the closed
attempt being promoted.

### `EvaluationRun`

Fields:

- `evaluation_id: str`
- `candidate_id: str`
- `split: Literal["validation", "test"]`
- `mode: Literal["standalone", "residual", "ablation", "full_cascade", "fault_fallback", "latency_cost"]`
- `request_order_digest: str`
- `ephemeral_request_id_salt_digest: str`
- `started_at: datetime`
- `completed_at: datetime | None`

### `MetricSummary`

Fields:

- `accepted_count: int`
- `correct_accept_count: int`
- `wrong_accept_count: int`
- `precision: float | None`
- `coverage: float`
- `wrong_accept_rate: float`
- `precision_lower_bound: float | None`
- `wrong_accept_upper_bound: float | None`

When `accepted_count == 0`, Precision is `None`, not `1.0`.

### `ReferenceSourceMetricSummary`

Fields:

- `reference_source: Literal["gold", "human", "versioned_l4", "verified_l4", "user_feedback"]`
- `sample_count: int`
- `metric: MetricSummary`
- `claim: Literal["gold_correctness", "human_correctness", "l4_agreement", "verified_l4_correctness", "user_feedback_correctness"]`
- `notes: list[str]`

Metrics grouped under `versioned_l4` are agreement with that L4/reference version unless the reference has separate gold/human verification.

### `GeneralizationSummary`

Fields:

- `validation_precision: float | None`
- `test_precision: float | None`
- `validation_coverage: float`
- `test_coverage: float`
- `precision_drop: float | None`
- `coverage_retention: float | None`
- `cohort_floor: dict | None`
- `worst_slice: dict | None`
- `candidate_rank_stability: dict | None`
- `future_audit: dict | None`
- `min_accepted_sample_check: Literal["pass", "fail", "insufficient"]`
- `min_slice_sample_check: Literal["pass", "fail", "insufficient"]`
- `evidence_status: Literal["pass", "fail", "insufficient"]`

### `CostLedger`

Fields:

- `serving_l4_cost: float`
- `serving_local_compute_cost: float`
- `random_audit_cost: float`
- `risk_audit_cost: float`
- `compile_agent_cost: float`
- `reference_labeling_cost: float`
- `local_training_search_cost: float`
- `compile_cost: float`
- `saving_per_1000_requests: float | None`
- `estimated_payback_requests: int | None`
- `notes: list[str]`

### `Report`

Fields:

- `report_id: str`
- `report_stage: Literal["validation", "test", "final"]`
- `candidate_id: str`
- `target_name: str`
- `contract_hash: str`
- `snapshot_id: str`
- `baseline_release_id: str`
- `metrics: dict`
- `metrics_by_reference_source: list[ReferenceSourceMetricSummary]`
- `reference_qualification: ReferenceQualificationReport`
- `generalization: GeneralizationSummary`
- `latency: dict`
- `cost: CostLedger`
- `safety: dict`
- `holdout_consumption: ConsumedRowsManifest | None`
- `decision: CandidateDecision | None`

Validation and test Reports are evaluation evidence. They have `decision: None`. `compare_candidates` consumes these Reports and produces a `CandidateDecision`. `finalize_report` creates the final Report only after the decision exists. This avoids requiring a release decision before Candidate comparison has happened.

### `CandidateDecision`

Fields:

- `decision_id: str`
- `candidate_id: str`
- `target_name: str`
- `contract_hash: str`
- `snapshot_id: str`
- `baseline_release_id: str`
- `status: Literal["eligible_for_release", "rejected", "insufficient_evidence"]`
- `requirement_results: list[RequirementCheckResult]`
- `comparison_summary: dict`
- `selected_operating_point: dict | None`
- `release_blockers: list[str]`
- `created_at: datetime`

`CandidateDecision` is Core-internal release evidence. It may contain complete requirement checks and comparison detail needed for release review, but it must not be mounted into an Agent Workspace directly.

### `AgentVisibleDecisionSummary`

Fields:

- `candidate_id: str`
- `status: Literal["eligible_for_release", "rejected", "insufficient_evidence"]`
- `headline_reason: str`
- `requirement_summary: dict`
- `comparison_summary: dict`
- `test_metrics_included: bool`
- `holdout_consumption: HoldoutConsumptionSummary | None`

This summary is the only decision view allowed in `AgentVisibleReport`. It must not include raw rows, row ids, per-row failures, split indices, or reconstructable slice/cohort members.

### `HoldoutConsumptionSummary`

Fields:

- `snapshot_id: str`
- `split: Literal["validation", "test"]`
- `reason_code: str`
- `consumed_at: datetime`
- `visible_to: Literal["user", "agent", "external_report"]`
- `replacement_required: bool`
- `record_count: int`
- `split_group_count: int`
- `manifest_digest: str`

`HoldoutConsumptionSummary` is the only holdout-consumption shape allowed in Agent-visible data. It is derived from `ConsumedRowsManifest`, but it must not contain `record_ids`, `normalized_input_keys`, `split_group_keys`, or any value that lets the agent reconstruct holdout membership.

### `L4BaselineSummary`

Fields:

- `release_id: str`
- `target_name: str`
- `contract_hash: str`
- `quality_summary: dict`
- `quality_by_reference_source: list[ReferenceSourceMetricSummary]`
- `reference_qualification: ReferenceQualificationReport`
- `latency_summary: dict`
- `cost_summary: dict`
- `notes: list[str]`

`L4BaselineSummary` describes the current Release when it has no lower-layer artifacts. It is not a fake `Report`; it is only the direct-L4 baseline used for comparison and cost/latency payback estimates. It must preserve reference source metrics and reference qualification so cold-start comparison cannot present versioned L4 agreement as gold or human correctness.

### `ReleaseBaseline`

Fields:

- `release: Release`
- `report: Report | None`
- `l4_baseline: L4BaselineSummary | None`

When `release` has lower-layer artifacts and compile provenance, `report` is required. When `release` has no lower-layer artifacts and no compile provenance, `l4_baseline` is required and `report` may be `None`. Implementations must not synthesize a fake `Report` for cold start.

### `AgentVisibleReport`

Fields:

- `report_id: str`
- `candidate_id: str`
- `target_name: str`
- `contract_hash: str`
- `snapshot_id: str`
- `baseline_release_id: str`
- `decision_summary: AgentVisibleDecisionSummary`
- `validation_metrics: dict`
- `test_metrics: dict | None`
- `test_metrics_included: bool`
- `holdout_consumption: HoldoutConsumptionSummary | None`
- `generalization_summary: dict`
- `cost_summary: dict`
- `created_at: datetime`

`AgentVisibleReport` is an aggregate view, not a copy of `Report`. It must not include raw train, validation, or test rows, row ids, request ids, per-row failures, split indices, or slice/cohort details that let an agent reconstruct hidden holdout membership.

## Functions

### `freeze_candidate`

Input:

- `submission: CandidateSubmission`
- `base_release: Release`
- `definition: TargetDefinition`
- `artifact_store: ArtifactStore`
- `source_snapshot_digest: str`

Output:

- `Candidate`

Purpose:

- Convert a mutable submission into an immutable Candidate.
- Freeze changed artifact packages.
- Pass the current evaluation snapshot digest into frozen artifact package metadata.
- Inherit unchanged artifacts from the base Release.
- If the base Release has no lower-layer artifacts, there are no unchanged artifacts to inherit.
- Compute candidate digest and workspace provenance.

Used by:

- `evaluate_candidate_on_validation`

### `validate_candidate_manifest`

Input:

- `candidate: Candidate`
- `definition: TargetDefinition`
- `base_release: Release`

Output:

- `CandidateManifestCheck`

Purpose:

- Check target name, contract hash, artifact layers, inherited artifacts, routing settings, and path safety.
- Reject Candidates that try to change evaluator, split, target contract, or registry state.

Used by:

- Candidate preflight.

### `run_protocol_preflight`

Input:

- `candidate: Candidate`
- `contract: TargetRuntimeContract`
- `artifact_worker: ArtifactWorkerClient`

Output:

- `ProtocolPreflightReport`

Purpose:

- Start each changed artifact.
- Run healthcheck and minimal protocol calls.
- Confirm timeout/error paths are safe.

Used by:

- Candidate preflight before expensive evaluation.

### `evaluate_standalone_layer`

Input:

- `candidate: Candidate`
- `layer: Literal["L1", "L2", "L3"]`
- `records: SnapshotView`
- `contract: TargetRuntimeContract`

Output:

- `LayerEvaluationResult`

Purpose:

- Run one layer against all records in a split.
- Randomize private evaluation order.
- Send only ephemeral per-run request ids to artifacts.
- Measure what the layer can identify by itself.
- This is diagnostic and not the official cascade Coverage.

Used by:

- Report diagnostics.

### `evaluate_residual_layer`

Input:

- `candidate: Candidate`
- `layer: Literal["L1", "L2", "L3"]`
- `residual_records: SnapshotView`
- `upstream_results: list[LayerEvaluationResult]`
- `contract: TargetRuntimeContract`

Output:

- `LayerEvaluationResult`

Purpose:

- Run a layer only on records that would actually reach it after upstream abstains/failures.
- Randomize private evaluation order and use ephemeral per-run request ids.
- Produce the official per-layer contribution metrics.

Used by:

- Full cascade evaluation.
- Per-layer Report tables.

### `evaluate_full_cascade`

Input:

- `candidate: Candidate`
- `records: SnapshotView`
- `contract: TargetRuntimeContract`
- `reference_policy: EvaluationReferencePolicy`

Output:

- `CascadeEvaluationResult`

Purpose:

- Run the same route Core would use at runtime: enabled and present L1/L2/L3 artifacts in order, then L4 fallback.
- Randomize private evaluation order and use ephemeral per-run request ids for artifact calls.
- Compute local accepts, local Precision, local Coverage, L4 fallback share, latency, cost, and failure behavior.
- Report configured-disabled layers separately from absent, unhealthy, or abstaining layers.
- For validation/test evaluation, reference outputs come from the Snapshot, not from artifact self-reporting.

Used by:

- Candidate selection.
- Release eligibility.

### `evaluate_changed_layer_ablation`

Input:

- `candidate: Candidate`
- `baseline_release: Release`
- `changed_layers: list[LayerName]`
- `records: SnapshotView`
- `contract: TargetRuntimeContract`

Output:

- `AblationResult`

Purpose:

- Compare changed artifacts alone and together with baseline unchanged artifacts.
- Detect interactions where improving one layer harms the residual distribution of another layer.
- Use `build_private_evaluation_request_plan` before any artifact call on validation or test rows.

Used by:

- Report diagnostics.

### `evaluate_fault_fallback`

Input:

- `candidate: Candidate`
- `fault_scenarios: list[FaultScenario]`
- `records: SnapshotView`
- `contract: TargetRuntimeContract`

Output:

- `FallbackSafetyResult`

Purpose:

- Inject worker crash, timeout, malformed response, invalid output, and L4 timeout scenarios.
- Confirm Core fallback, deadline, and trace behavior.
- Use `build_private_evaluation_request_plan` for every scenario that calls an artifact on validation or test rows.

Used by:

- Hard safety requirement.

### `measure_latency_and_cost`

Input:

- `candidate: Candidate`
- `records: SnapshotView`
- `contract: TargetRuntimeContract`
- `measurement_options: LatencyCostOptions`

Output:

- `LatencyCostResult`

Purpose:

- Measure p50/p95/p99 latency, throughput, memory, local compute cost, L4 fallback cost, audit cost, reference labeling cost, compile agent cost, local training/search cost, and cascade cost.
- Include the cost of upstream abstain layers in cascade latency.
- Use `build_private_evaluation_request_plan` when latency/cost measurement calls artifacts on validation or test rows.

Used by:

- Candidate comparison.
- Release reports.

### `build_cost_ledger`

Input:

- `latency_cost: LatencyCostResult`
- `agent_usage: AgentUsageLedger`
- `reference_usage: ReferenceUsageLedger`
- `audit_usage: AuditUsageLedger | None`
- `local_training_search_usage: LocalTrainingSearchUsageLedger | None`
- `baseline_cost: BaselineCostSummary`

Output:

- `CostLedger`

Purpose:

- Build the cost view used in Reports.
- Separate serving L4 cost, local serving compute, random audit, risk audit, compile agent, reference labeling, and local training/search costs.
- Compute `saving_per_1000_requests`, `compile_cost`, and `estimated_payback_requests` without hiding compile cost inside serving savings.

Used by:

- `evaluate_candidate_on_validation`
- `evaluate_candidate_on_test`

### `compute_metric_summary`

Input:

- `evaluation_result: CascadeEvaluationResult | LayerEvaluationResult`
- `confidence_options: ConfidenceOptions`

Output:

- `MetricSummary`

Purpose:

- Compute accepted count, correct accepts, wrong accepts, Precision, Coverage, wrong accept rate, and confidence bounds.
- Return Precision as `None` when there are no accepts.

Used by:

- Reports.
- Requirement checks.

### `compute_reference_source_metrics`

Input:

- `evaluation_result: CascadeEvaluationResult | LayerEvaluationResult`
- `reference_provenance: ReferenceProvenanceView`
- `confidence_options: ConfidenceOptions`

Output:

- `list[ReferenceSourceMetricSummary]`

Purpose:

- Compute the same accepted Precision, Coverage, and wrong-accept summaries grouped by `reference_source`.
- Label versioned L4 rows as L4 agreement unless independent gold/human verification exists.
- Prevent Reports from presenting L4 agreement as gold correctness.

Used by:

- `evaluate_candidate_on_validation`
- `evaluate_candidate_on_test`

### `compute_generalization_summary`

Input:

- `validation_result: CascadeEvaluationResult`
- `test_result: CascadeEvaluationResult | None`
- `slice_results: list[SliceMetricSummary]`
- `requirements: TargetRequirements`

Output:

- `GeneralizationSummary`

Purpose:

- Summarize whether Precision and Coverage transfer from validation to test and across user-reviewed slices.
- Include cohort floor, coverage retention, worst-slice stability, candidate rank stability, future audit status, and minimum accepted/slice sample checks.
- Report insufficient evidence instead of passing low-sample claims.

Used by:

- Candidate decision.
- User report.

### `check_candidate_requirements`

Input:

- `metrics: MetricSummary`
- `generalization: GeneralizationSummary`
- `latency_cost: LatencyCostResult`
- `requirements: TargetRequirements`

Output:

- `RequirementCheckResult`

Purpose:

- Enforce hard requirements: Precision lower bound, wrong accept upper bound, Generalization status, critical slices, latency, memory, throughput, and serving cost.
- Treat insufficient accepted samples or insufficient slice samples as insufficient evidence, not pass.
- Does not trade one hard requirement against another.

Used by:

- `compare_candidates`

### `compare_candidates`

Input:

- `candidate_reports: list[Report]`
- `baseline: ReleaseBaseline`
- `objective: CandidateObjective`

Output:

- `CandidateDecision`

Purpose:

- Consume validation/test Reports with `decision: None`; do not require finalized Reports as input.
- Reject candidates failing hard requirements.
- Among passing candidates, optimize Coverage, cost, or latency according to target objective.
- Require meaningful improvement over current Release.
- When the current Release has no lower-layer artifacts, compare against `baseline.l4_baseline` rather than requiring a baseline `Report`.
- When the current Release has compile provenance, compare against `baseline.report`.
- Produce a Pareto view when tradeoffs remain.

Used by:

- CLI report.
- Release eligibility.

### `finalize_report`

Input:

- `report: Report`
- `decision: CandidateDecision`

Output:

- `Report`

Purpose:

- Attach the CandidateDecision after `compare_candidates`.
- Set `report_stage` to `"final"`.
- Hard-fail unless report and decision match on candidate, target, contract, snapshot, and baseline release.
- Preserve holdout consumption status from the input Report.

Used by:

- Release Runtime module.
- CLI report.

### `build_agent_feedback`

Input:

- `report: Report`
- `feedback_policy: FeedbackPolicy`

Output:

- `AgentFeedback`

Purpose:

- Convert validation-stage Report into aggregate feedback that the agent can use.
- Exclude raw validation rows, expected outputs, request ids, split indices, and reconstructable small slices.

Used by:

- Agent Workspace module.

### `summarize_holdout_consumption`

Input:

- `manifest: ConsumedRowsManifest`

Output:

- `HoldoutConsumptionSummary`

Purpose:

- Convert Core-internal holdout consumption records into the only Agent-visible holdout consumption shape.
- Preserve split, consumption time, reason code, visibility class, replacement requirement, counts, and manifest digest.
- Drop `record_ids`, `normalized_input_keys`, `split_group_keys`, and any other value that could reconstruct holdout membership.

Used by:

- `build_agent_visible_decision_summary`
- `build_agent_visible_report`

### `build_agent_visible_decision_summary`

Input:

- `decision: CandidateDecision`
- `include_test_metrics: bool`
- `holdout_consumption: ConsumedRowsManifest | None`

Output:

- `AgentVisibleDecisionSummary`

Purpose:

- Convert the Core-internal CandidateDecision into the decision summary that may be mounted into an Agent Workspace.
- Preserve status, headline reason, aggregate requirement status, and aggregate comparison result.
- Include `HoldoutConsumptionSummary` when test metrics are visible.
- Convert any supplied `ConsumedRowsManifest` through `summarize_holdout_consumption`; never embed the manifest itself.
- Strip raw rows, row ids, per-row failures, split indices, fine-grained Pareto rows, and reconstructable slice/cohort details.

Used by:

- `build_agent_visible_report`

### `build_agent_visible_report`

Input:

- `report: Report`
- `include_test_metrics: bool`

Output:

- `AgentVisibleReport`

Purpose:

- Convert a completed Report into the historical summary that may be mounted into a later Agent Workspace.
- Hard-fail unless `report.report_stage == "final"` and `report.decision` is present.
- Preserve target, contract, snapshot, baseline release, `AgentVisibleDecisionSummary`, aggregate validation metrics, aggregate generalization summary, and aggregate cost summary.
- Include aggregate test metrics only when the corresponding test window has already been marked consumed.
- Hard-fail if `include_test_metrics` is true and `report.holdout_consumption` is missing.
- Convert `report.holdout_consumption` through `summarize_holdout_consumption`; never expose `ConsumedRowsManifest` to Agent Workspace.
- Strip raw rows, row ids, request ids, per-row failures, split indices, and reconstructable slice/cohort members.

Used by:

- Agent Workspace module.

### `evaluate_candidate_on_validation`

Input:

- `submission: CandidateSubmission`
- `definition: TargetDefinition`
- `snapshot: Snapshot`
- `base_release: Release`
- `reference_qualification: ReferenceQualificationReport`
- `agent_usage: AgentUsageLedger`
- `reference_usage: ReferenceUsageLedger`
- `audit_usage: AuditUsageLedger | None`
- `local_training_search_usage: LocalTrainingSearchUsageLedger | None`
- `baseline_cost: BaselineCostSummary`
- `evaluation_options: EvaluationOptions`

Output:

- `ValidationEvaluationResult`

Purpose:

- Freeze Candidate.
- Run preflight, validation cascade, residual, ablation, fallback, latency, and cost checks.
- Use `candidate.routing.enabled_layers` for official validation metrics.
- Compute metrics by reference source and include the ReferenceQualificationReport in the Report.
- Build a complete CostLedger from the supplied usage ledgers.
- Produce a validation Report with `report_stage: "validation"` and `decision: None`, plus agent-safe feedback.

Used by:

- Agent compile loop.

### `build_private_evaluation_request_plan`

Input:

- `records: SnapshotView`
- `evaluation_id: str`
- `random_seed: int`

Output:

- `PrivateEvaluationRequestPlan`

Purpose:

- Shuffle private evaluation rows before artifact calls.
- Map each snapshot row to an ephemeral request id valid only for this evaluation run.
- Keep the mapping inside Core evaluator storage; never mount it into artifact workers or agent workspaces.
- Required for every Candidate Evaluation mode that calls artifacts on validation or test rows, including standalone, residual, full cascade, ablation, fault fallback, and latency/cost measurement.

Used by:

- `evaluate_standalone_layer`
- `evaluate_residual_layer`
- `evaluate_full_cascade`
- `evaluate_changed_layer_ablation`
- `evaluate_fault_fallback`
- `measure_latency_and_cost`

### `evaluate_candidate_on_test`

Input:

- `candidate: Candidate`
- `closed_attempt: ClosedAgentAttempt`
- `definition: TargetDefinition`
- `snapshot: Snapshot`
- `base_release: Release`
- `reference_qualification: ReferenceQualificationReport`
- `agent_usage: AgentUsageLedger`
- `reference_usage: ReferenceUsageLedger`
- `audit_usage: AuditUsageLedger | None`
- `local_training_search_usage: LocalTrainingSearchUsageLedger | None`
- `baseline_cost: BaselineCostSummary`
- `evaluation_options: EvaluationOptions`

Output:

- `TestEvaluationResult`

Purpose:

- Confirm the agent is closed and Candidate bytes are frozen.
- Run final test evaluation.
- Use `candidate.routing.enabled_layers` for official test metrics.
- Compute metrics by reference source and include the ReferenceQualificationReport in the Report.
- Build a complete CostLedger from the supplied usage ledgers.
- Produce a test Report with `report_stage: "test"` and `decision: None`.
- Release eligibility requires `compare_candidates` followed by `finalize_report`.
- Do not return test rows or row-level failures to the same agent run.
- Enforce the hidden holdout consumption invariant for failed or visible test results.

Used by:

- Release Runtime module.

## Invariants

- Core recomputes official metrics.
- Agent metrics are never trusted for Release.
- Validation feedback is aggregate and safe.
- Test evaluation happens only after the agent attempt is closed.
- Precision, Coverage, and Generalization are reported separately.
- Reports distinguish gold/human correctness from versioned L4 agreement.
- Cache hits are excluded from local Coverage.
- Hard quality requirements cannot be overridden by cost or latency improvements.
