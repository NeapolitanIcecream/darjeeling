from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Protocol

LayerName = Literal["L1", "L2", "L3"]
SplitName = Literal["train", "validation", "test"]
SplitEligibility = Literal["train", "validation_candidate", "test_candidate"]
ReferenceSource = Literal["gold", "human", "versioned_l4", "verified_l4", "user_feedback"]


@dataclass(frozen=True)
class TargetRequirements:
    precision_min: float = 0.0
    wrong_accept_rate_max: float | None = None
    validation_test_precision_drop_max: float | None = None
    validation_test_coverage_retention_min: float | None = None
    cohort_precision_floor_min: float | None = None
    candidate_rank_stability_min_shards: int | None = None
    future_audit_required_for_auto_release: bool = False
    min_accepted_samples: int = 1
    min_slice_samples: int = 1
    critical_slices: list[str] = field(default_factory=list)
    critical_slice_precision_min: float | None = None
    critical_slice_coverage_min: float | None = None
    coverage_objective: Literal["maximize", "hold", "none"] = "maximize"
    p95_latency_ms_max: int | None = None
    memory_mb_max: int | None = None
    throughput_per_second_min: float | None = None
    serving_cost_per_1000_max: float | None = None
    random_audit_rate: float = 0.0
    random_audit_reference_failure_rate_max: float | None = None
    manual_approval_required: bool = False


@dataclass(frozen=True)
class TelemetryPrivacyPolicy:
    policy_version: str = "v1"
    allowed_sources: list[Literal["l4_fallback", "random_audit", "risk_audit", "user_feedback"]] = (
        field(default_factory=list)
    )
    default_approved_for_by_source: dict[str, list[SplitEligibility]] = field(default_factory=dict)
    raw_payload_allowed: bool = False
    canonicalization_required: bool = True
    human_review_required_sources: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RuntimeConfig:
    telemetry_privacy_policy: TelemetryPrivacyPolicy = field(default_factory=TelemetryPrivacyPolicy)


@dataclass(frozen=True)
class DataConfig:
    sources: list[dict[str, Any]] = field(default_factory=list)
    default_split_eligibility: list[SplitEligibility] = field(default_factory=lambda: ["train"])
    split_options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TargetDefinitionDraft:
    name: str
    version: str
    target_path: Path
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    input_schema_path: Path
    output_schema_path: Path
    contract_path: Path
    reference_path: Path | None
    requirements: TargetRequirements
    data_config: DataConfig
    runtime_config: RuntimeConfig
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TargetDefinition:
    name: str
    version: str
    target_path: Path
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    contract_module_digest: str
    reference_module_digest: str | None
    requirements: TargetRequirements
    data_config: DataConfig
    runtime_config: RuntimeConfig
    contract_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)
    reference_version: str | None = None


@dataclass(frozen=True)
class TargetCheckOptions:
    require_reference: bool = False


@dataclass(frozen=True)
class TargetCheckReport:
    target_name: str
    contract_hash: str | None
    status: Literal["pass", "fail"]
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TargetViewManifest:
    target_name: str
    contract_hash: str
    view_path: Path
    included_files: list[str]


@dataclass
class ContractModule:
    path: Path
    module: Any
    digest: str


@dataclass
class ReferenceModule:
    path: Path
    module: Any
    digest: str


@dataclass(frozen=True)
class ReferenceContext:
    purpose: str
    request_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReferenceResponse:
    payload: dict[str, Any]
    reference_source: ReferenceSource = "versioned_l4"
    reference_version: str | None = None
    finish_status: str = "stop"
    usage: dict[str, Any] = field(default_factory=dict)
    cost: float = 0.0
    latency_ms: float = 0.0


class ReferenceBroker(Protocol):
    reference_version: str

    def call(self, request: dict[str, Any], context: ReferenceContext) -> ReferenceResponse: ...


@dataclass(frozen=True)
class ReferenceBudget:
    max_calls: int | None = None
    max_cost: float | None = None


@dataclass(frozen=True)
class ReferenceCallResult:
    status: Literal["ok", "error"]
    output: dict[str, Any] | None
    reference_source: ReferenceSource | None
    reference_version: str | None
    usage: dict[str, Any]
    cost: float
    latency_ms: float
    finish_status: str
    error_type: str | None = None
    error_message_hash: str | None = None


@dataclass(frozen=True)
class ReferenceUsageLedger:
    call_count: int = 0
    cost: float = 0.0
    errors: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ReferenceFailureReport:
    failures: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class ReferenceQualificationOptions:
    min_gold_samples: int = 0
    min_agreement_rate: float = 0.0
    max_parse_failure_rate: float = 0.0
    max_schema_failure_rate: float = 0.0


@dataclass(frozen=True)
class ReferenceQualificationReport:
    target_name: str
    contract_hash: str
    reference_version: str
    gold_sample_count: int
    l4_agreement_count: int
    gold_correct_count: int | None
    agreement_rate: float
    gold_precision: float | None
    parse_failure_rate: float
    schema_failure_rate: float
    latency: dict[str, Any]
    cost: dict[str, Any]
    status: Literal["pass", "fail", "insufficient"]
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SourceRecord:
    record_id: str
    input: dict[str, Any]
    reference_output: dict[str, Any] | None
    reference_source: ReferenceSource | None
    split_eligibility: list[SplitEligibility]
    source_name: str
    source_timestamp: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SnapshotRecord:
    snapshot_record_id: str
    normalized_input_key: str
    split_group_key: str
    input: dict[str, Any]
    reference_output: dict[str, Any]
    reference_source: ReferenceSource
    split_eligibility: list[SplitEligibility]
    reference_version: str | None
    slice_tags: list[str]
    source_provenance: dict[str, Any]


@dataclass(frozen=True)
class Snapshot:
    snapshot_id: str
    snapshot_digest: str
    target_name: str
    contract_hash: str
    created_at: datetime
    cutoff_time: datetime
    source_watermarks: dict[str, str]
    telemetry_source_id: str | None
    records_digest: str
    split_manifest_digest: str
    reference_ledger_digest: str
    train_count: int
    validation_count: int
    test_count: int
    storage_uri: str


@dataclass(frozen=True)
class SnapshotView:
    snapshot_id: str
    split: SplitName
    records_uri: str
    redaction_level: Literal["raw", "redacted", "aggregate"]
    record_count: int


@dataclass(frozen=True)
class ConsumedRowsManifest:
    snapshot_id: str
    split: Literal["validation", "test"]
    record_ids: list[str]
    normalized_input_keys: list[str]
    split_group_keys: list[str]
    reason: str
    consumed_at: datetime
    visible_to: Literal["user", "agent", "external_report"]
    replacement_required: bool


@dataclass(frozen=True)
class SnapshotOptions:
    seed: int = 0
    validation_fraction: float = 0.2
    test_fraction: float = 0.2
    on_duplicate_conflict: Literal["fail", "exclude"] = "fail"
    allow_insufficient_reference: bool = False
    qualification_options: ReferenceQualificationOptions = field(
        default_factory=ReferenceQualificationOptions
    )
    reference_budget: ReferenceBudget = field(default_factory=ReferenceBudget)
    storage_root: Path = Path(".darjeeling/snapshots")


@dataclass(frozen=True)
class SnapshotBuildResult:
    snapshot: Snapshot
    train_view: SnapshotView
    reference_usage: ReferenceUsageLedger
    reference_qualification: ReferenceQualificationReport
    failure_report: ReferenceFailureReport


@dataclass(frozen=True)
class TrainViewManifest:
    snapshot_id: str
    snapshot_digest: str
    view_path: Path
    record_count: int
    redaction_level: str
    export_digest: str
    view_kind: Literal["agent_train_export"] = "agent_train_export"


@dataclass(frozen=True)
class AgentViewOptions:
    redaction_level: Literal["raw", "redacted"] = "redacted"


@dataclass(frozen=True)
class ArtifactManifest:
    api_version: str
    layer: LayerName
    start_command: list[str]
    healthcheck_command: list[str] | None
    protocol: Literal["jsonl"]
    timeout_ms: int
    memory_mb: int | None
    network: Literal["disabled"]
    contract_hash: str
    artifact_id: str | None = None
    allowed_reason_codes: list[str] | None = None


@dataclass(frozen=True)
class ArtifactPackage:
    artifact_id: str
    layer: LayerName
    package_path: Path
    manifest: ArtifactManifest
    digest: str
    source_snapshot_digest: str
    build_provenance: dict[str, Any]


@dataclass(frozen=True)
class PackagePolicy:
    forbidden_names: list[str] = field(
        default_factory=lambda: [
            "validation",
            "test",
            "holdout",
            "registry",
            "credentials",
            "broker",
        ]
    )


@dataclass(frozen=True)
class ArtifactCheckReport:
    artifact_dir: Path
    expected_layer: LayerName
    status: Literal["pass", "fail"]
    manifest: ArtifactManifest | None
    failures: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class WorkerLimits:
    timeout_ms: int | None = None
    memory_mb: int | None = None


@dataclass(frozen=True)
class WorkerHandle:
    package: ArtifactPackage
    limits: WorkerLimits
    healthy: bool = True


@dataclass(frozen=True)
class HealthcheckResult:
    status: Literal["pass", "fail"]
    message: str = ""


@dataclass(frozen=True)
class WorkerRequest:
    request_id: str
    input: dict[str, Any]
    deadline_ms: int


@dataclass(frozen=True)
class WorkerResponse:
    decision: Literal["accept", "abstain"]
    output: dict[str, Any] | None = None
    confidence: float | None = None
    reason: str | None = None


@dataclass(frozen=True)
class RawWorkerCallResult:
    status: Literal["ok", "timeout", "error"]
    response_bytes: bytes | None
    latency_ms: float
    error: str | None = None


@dataclass(frozen=True)
class ProtocolError:
    message: str


@dataclass(frozen=True)
class LayerAttemptResult:
    layer: LayerName
    artifact_id: str
    decision: Literal["accept", "abstain", "error", "timeout", "invalid_output", "protocol_error"]
    output: dict[str, Any] | None
    confidence: float | None
    reason: str | None
    latency_ms: float
    error: str | None = None


@dataclass(frozen=True)
class WorkerStopResult:
    stopped: bool
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ProtocolDocs:
    protocol_version: str
    path: Path | None
    text: str


@dataclass(frozen=True)
class RoutingSettings:
    cache_enabled: bool = False
    enabled_layers: list[LayerName] = field(default_factory=lambda: ["L1", "L2", "L3"])
    L1_timeout_ms: int | None = None
    L2_timeout_ms: int | None = None
    L3_timeout_ms: int | None = None
    total_deadline_ms: int = 1000
    circuit_breaker: dict[str, Any] = field(default_factory=dict)
    audit: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompileBudget:
    max_agent_seconds: int = 0
    max_candidates: int = 1
    max_cost: float | None = None


@dataclass(frozen=True)
class CompileOptions:
    objective: dict[str, Any] = field(default_factory=dict)
    allow_insufficient_reference_qualification: bool = False


@dataclass(frozen=True)
class CompileRun:
    compile_id: str
    target_name: str
    contract_hash: str
    snapshot_id: str
    snapshot_digest: str
    base_release_id: str
    workspace_baseline_commit: str
    started_at: datetime
    budget: CompileBudget
    status: Literal["running", "closing", "closed", "failed"]


@dataclass
class CompileRunStore:
    runs: dict[str, CompileRun] = field(default_factory=dict)


@dataclass(frozen=True)
class TargetWorkspace:
    target_name: str
    workspace_path: Path
    baseline_commit: str
    contract_hash: str
    last_accepted_release_id: str | None
    status: Literal["active", "archived"] = "active"


@dataclass(frozen=True)
class WorkspaceBaselineUpdate:
    target_name: str
    previous_commit: str
    new_commit: str
    source_attempt_id: str
    source_release_id: str | None
    reason: Literal["accepted_release", "explicit_carry_forward"]


@dataclass(frozen=True)
class AgentAttempt:
    attempt_id: str
    compile_id: str
    target_name: str
    contract_hash: str
    snapshot_id: str
    snapshot_digest: str
    workspace_path: Path
    source_workspace_commit: str
    initial_commit: str | None
    final_commit: str | None
    agent_model: str
    status: Literal["running", "closed", "failed", "timed_out"]


@dataclass(frozen=True)
class ClosedAgentAttempt:
    attempt_id: str
    compile_id: str
    target_name: str
    contract_hash: str
    snapshot_id: str
    snapshot_digest: str
    source_workspace_commit: str
    workspace_path: Path
    final_commit: str
    status: Literal["closed", "failed", "timed_out"]
    usage: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateSubmission:
    submission_id: str
    compile_id: str
    attempt_id: str
    submission_path: Path
    workspace_commit: str
    submitted_at: datetime
    declared_layers: list[LayerName]


@dataclass(frozen=True)
class AgentFeedback:
    candidate_id: str
    summary: dict[str, Any]
    requirement_results: list[dict[str, Any]]
    metrics: dict[str, Any]
    safe_slice_summaries: list[dict[str, Any]]
    latency_cost_summary: dict[str, Any]
    raw_rows_included: Literal[False] = False


@dataclass(frozen=True)
class WorkspaceStore:
    root: Path


@dataclass(frozen=True)
class AgentAttemptOptions:
    agent_model: str = "manual"
    agent_command: list[str] = field(default_factory=list)
    agent_timeout_seconds: int | None = None


@dataclass(frozen=True)
class WorkspaceMountManifest:
    attempt_id: str
    mount_path: Path
    entries: list[str]


@dataclass(frozen=True)
class AgentSessionHandle:
    attempt_id: str
    status: Literal["running", "not_started", "completed", "failed", "timed_out", "stopped"]
    command: list[str] = field(default_factory=list)
    pid: int | None = None
    started_at: datetime | None = None
    log_path: Path | None = None
    session_record_path: Path | None = None
    sandbox_mode: str | None = None
    timeout_seconds: int | None = None


@dataclass(frozen=True)
class FeedbackDeliveryRecord:
    attempt_id: str
    path: Path
    delivered_at: datetime


@dataclass(frozen=True)
class AgentUsageEvent:
    kind: str
    cost: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentUsageLedger:
    events: list[AgentUsageEvent] = field(default_factory=list)

    @property
    def cost(self) -> float:
        return sum(event.cost for event in self.events)


@dataclass(frozen=True)
class JournalEntry:
    title: str
    body: str
    created_at: datetime


@dataclass(frozen=True)
class TargetChangeProposal:
    path: Path
    summary: str


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    submission_id: str
    compile_id: str
    attempt_id: str
    target_name: str
    contract_hash: str
    snapshot_id: str
    base_release_id: str
    workspace_commit: str
    artifacts: dict[LayerName, ArtifactPackage | None]
    routing: RoutingSettings
    digest: str
    status: Literal[
        "submitted",
        "frozen",
        "validation_failed",
        "validation_passed",
        "test_failed",
        "eligible_for_release",
        "rejected",
    ]


@dataclass(frozen=True)
class EvaluationRun:
    evaluation_id: str
    candidate_id: str
    split: Literal["validation", "test"]
    mode: Literal[
        "standalone", "residual", "ablation", "full_cascade", "fault_fallback", "latency_cost"
    ]
    request_order_digest: str
    ephemeral_request_id_salt_digest: str
    started_at: datetime
    completed_at: datetime | None = None


@dataclass(frozen=True)
class MetricSummary:
    accepted_count: int
    correct_accept_count: int
    wrong_accept_count: int
    precision: float | None
    coverage: float
    wrong_accept_rate: float
    precision_lower_bound: float | None = None
    wrong_accept_upper_bound: float | None = None


@dataclass(frozen=True)
class ReferenceSourceMetricSummary:
    reference_source: ReferenceSource
    sample_count: int
    metric: MetricSummary
    claim: Literal[
        "gold_correctness",
        "human_correctness",
        "l4_agreement",
        "verified_l4_correctness",
        "user_feedback_correctness",
    ]
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GeneralizationSummary:
    validation_precision: float | None
    test_precision: float | None
    validation_coverage: float
    test_coverage: float
    precision_drop: float | None
    coverage_retention: float | None
    cohort_floor: dict[str, Any] | None = None
    worst_slice: dict[str, Any] | None = None
    candidate_rank_stability: dict[str, Any] | None = None
    future_audit: dict[str, Any] | None = None
    min_accepted_sample_check: Literal["pass", "fail", "insufficient"] = "insufficient"
    min_slice_sample_check: Literal["pass", "fail", "insufficient"] = "insufficient"
    evidence_status: Literal["pass", "fail", "insufficient"] = "insufficient"


@dataclass(frozen=True)
class CostLedger:
    serving_l4_cost: float = 0.0
    serving_local_compute_cost: float = 0.0
    random_audit_cost: float = 0.0
    risk_audit_cost: float = 0.0
    compile_agent_cost: float = 0.0
    reference_labeling_cost: float = 0.0
    local_training_search_cost: float = 0.0
    compile_cost: float = 0.0
    saving_per_1000_requests: float | None = None
    estimated_payback_requests: int | None = None
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RequirementCheckResult:
    name: str
    status: Literal["pass", "fail", "insufficient"]
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateDecision:
    decision_id: str
    candidate_id: str
    target_name: str
    contract_hash: str
    snapshot_id: str
    baseline_release_id: str
    status: Literal["eligible_for_release", "rejected", "insufficient_evidence"]
    requirement_results: list[RequirementCheckResult]
    comparison_summary: dict[str, Any]
    selected_operating_point: dict[str, Any] | None
    release_blockers: list[str]
    created_at: datetime


@dataclass(frozen=True)
class Report:
    report_id: str
    report_stage: Literal["validation", "test", "final"]
    candidate_id: str
    target_name: str
    contract_hash: str
    snapshot_id: str
    baseline_release_id: str
    metrics: dict[str, Any]
    metrics_by_reference_source: list[ReferenceSourceMetricSummary]
    reference_qualification: ReferenceQualificationReport
    generalization: GeneralizationSummary
    latency: dict[str, Any]
    cost: CostLedger
    safety: dict[str, Any]
    holdout_consumption: ConsumedRowsManifest | None
    decision: CandidateDecision | None = None


@dataclass(frozen=True)
class AgentVisibleDecisionSummary:
    candidate_id: str
    status: Literal["eligible_for_release", "rejected", "insufficient_evidence"]
    headline_reason: str
    requirement_summary: dict[str, Any]
    comparison_summary: dict[str, Any]
    test_metrics_included: bool
    holdout_consumption: HoldoutConsumptionSummary | None = None


@dataclass(frozen=True)
class HoldoutConsumptionSummary:
    snapshot_id: str
    split: Literal["validation", "test"]
    reason_code: str
    consumed_at: datetime
    visible_to: Literal["user", "agent", "external_report"]
    replacement_required: bool
    record_count: int
    split_group_count: int
    manifest_digest: str


@dataclass(frozen=True)
class L4BaselineSummary:
    release_id: str
    target_name: str
    contract_hash: str
    quality_summary: dict[str, Any]
    quality_by_reference_source: list[ReferenceSourceMetricSummary]
    reference_qualification: ReferenceQualificationReport
    latency_summary: dict[str, Any]
    cost_summary: dict[str, Any]
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ReleaseBaseline:
    release: Release
    report: Report | None = None
    l4_baseline: L4BaselineSummary | None = None


@dataclass(frozen=True)
class AgentVisibleReport:
    report_id: str
    candidate_id: str
    target_name: str
    contract_hash: str
    snapshot_id: str
    baseline_release_id: str
    decision_summary: AgentVisibleDecisionSummary
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any] | None
    test_metrics_included: bool
    holdout_consumption: HoldoutConsumptionSummary | None
    generalization_summary: dict[str, Any]
    cost_summary: dict[str, Any]
    created_at: datetime


@dataclass(frozen=True)
class Release:
    release_id: str
    target_name: str
    contract_hash: str
    candidate_id: str | None
    snapshot_id: str | None
    snapshot_digest: str | None
    report_id: str | None
    created_at: datetime
    artifacts: dict[LayerName, ArtifactPackage | None]
    routing: RoutingSettings
    approval: ApprovalRecord | None
    status: Literal["created", "shadow", "canary", "stable", "retired", "rolled_back"] = "created"


@dataclass(frozen=True)
class ApprovalRecord:
    approval_id: str
    candidate_id: str
    report_id: str
    target_name: str
    contract_hash: str
    snapshot_id: str
    approved_at: datetime
    approved_by: Literal["user", "preapproved_policy"]


@dataclass(frozen=True)
class RuntimeRequest:
    request_id: str
    target_name: str
    input: dict[str, Any]
    tenant_key: str | None = None
    deadline_ms: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeResponse:
    request_id: str
    release_id: str
    status: Literal["ok", "error"]
    output: dict[str, Any] | None
    chosen_layer: Literal["cache", "L1", "L2", "L3", "L4"] | None
    error_type: (
        Literal["deadline_exceeded", "l4_fallback_failure", "no_valid_output", "runtime_error"]
        | None
    )
    public_error_message: str | None
    latency_ms: float
    trace_id: str


@dataclass(frozen=True)
class L4FallbackResult:
    input_raw: dict[str, Any]
    status: Literal["ok", "error"]
    output_raw: dict[str, Any] | None
    output_validated: dict[str, Any] | None
    reference_source: ReferenceSource | None
    cost: float
    latency_ms: float
    finish_status: str
    error_type: (
        Literal[
            "timeout",
            "rate_limited",
            "empty_response",
            "parse_failure",
            "validation_failure",
            "provider_error",
        ]
        | None
    ) = None
    error_message_raw: str | None = None
    error_message_hash: str | None = None


@dataclass(frozen=True)
class CascadeResult:
    release_id: str
    attempts: list[LayerAttemptResult]
    status: Literal["ok", "error"]
    chosen_layer: Literal["L1", "L2", "L3", "L4"] | None
    output: dict[str, Any] | None
    serving_cost: float
    latency_ms: float
    fallback_reason: str | None = None
    error_type: (
        Literal["deadline_exceeded", "l4_fallback_failure", "no_valid_output", "runtime_error"]
        | None
    ) = None
    error_message_hash: str | None = None
    l4_fallback_result: L4FallbackResult | None = None


@dataclass(frozen=True)
class CacheHit:
    output: dict[str, Any]
    cache_key: str


@dataclass(frozen=True)
class CacheMiss:
    cache_key: str


@dataclass(frozen=True)
class ServingResult:
    release_id: str
    path: Literal["cache", "cascade"]
    status: Literal["ok", "error"]
    cache_result: CacheHit | CacheMiss
    cascade_result: CascadeResult | None
    output: dict[str, Any] | None
    chosen_layer: Literal["cache", "L1", "L2", "L3", "L4"] | None
    serving_cost: float
    latency_ms: float
    error_type: (
        Literal["deadline_exceeded", "l4_fallback_failure", "no_valid_output", "runtime_error"]
        | None
    )
    error_message_hash: str | None
    l4_fallback_result: L4FallbackResult | None


@dataclass(frozen=True)
class RuntimeContext:
    request_id: str
    deadline_ms: int
    broker: ReferenceBroker


@dataclass
class ReleaseRegistry:
    releases: dict[str, Release] = field(default_factory=dict)
    channels: dict[tuple[str, str], str] = field(default_factory=dict)
    previous_channels: dict[tuple[str, str], str] = field(default_factory=dict)
    channel_options: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    circuit_breakers: dict[tuple[str, LayerName], dict[str, Any]] = field(
        default_factory=dict
    )


@dataclass
class ResultCache:
    entries: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class PreparedReleaseWorkers:
    workers: dict[LayerName, WorkerHandle]


@dataclass
class WorkerPool:
    workers: dict[str, WorkerHandle] = field(default_factory=dict)


@dataclass(frozen=True)
class TraceLayerAttempt:
    layer: LayerName
    artifact_id: str
    decision: Literal["accept", "abstain", "error", "timeout", "invalid_output", "protocol_error"]
    output_redacted: dict[str, Any] | None
    output_hash: str | None
    confidence: float | None
    reason_code: str | None
    latency_ms: float
    error_type: str | None = None
    error_message_hash: str | None = None


@dataclass(frozen=True)
class Trace:
    trace_id: str
    request_id_hash: str
    target_name: str
    contract_hash: str
    release_id: str
    input_redacted: dict[str, Any] | None
    input_hash: str
    attempts: list[TraceLayerAttempt]
    cache_hit: bool
    status: Literal["ok", "error"]
    chosen_layer: Literal["cache", "L1", "L2", "L3", "L4"] | None
    final_output_redacted: dict[str, Any] | None
    final_output_hash: str | None
    error_type: (
        Literal["deadline_exceeded", "l4_fallback_failure", "no_valid_output", "runtime_error"]
        | None
    )
    error_message_hash: str | None
    serving_cost: float
    latency_ms: float
    random_audit_probability: float
    risk_audit_flags: list[str]
    selected_audit_types: list[Literal["random", "risk"]]
    timestamp: datetime
    metadata_buckets: dict[str, Any]


@dataclass(frozen=True)
class RuntimeMetricWindow:
    target_name: str
    release_id: str
    contract_hash: str
    window_start: datetime
    window_end: datetime
    request_count: int
    local_coverage: float
    layer_attempt_counts: dict[str, int]
    layer_accept_counts: dict[str, int]
    layer_coverage: dict[str, float]
    configured_disabled_layers: list[LayerName]
    cache_hit_rate: float
    l4_fallback_rate: float
    random_audit_precision: float | None
    random_audit_precision_lower_bound: float | None
    random_audit_precision_confidence_level: float | None
    wrong_accept_estimate: float | None
    wrong_accept_rate_upper_bound: float | None
    random_audit_attempt_count: int
    random_audit_success_count: int
    random_audit_evaluable_count: int
    random_audit_correct_count: int
    random_audit_wrong_count: int
    random_audit_reference_failure_count: int
    random_audit_reference_failure_rate: float | None
    confidence_histogram: dict[str, int]
    reason_code_counts: dict[str, int]
    source_metadata_counts: dict[str, dict[str, int]]
    schema_failure_rate: float
    latency: dict[str, Any]
    cost: dict[str, Any]
    error_rates: dict[str, float]


@dataclass(frozen=True)
class RuntimeFailureDecision:
    target_name: str
    release_id: str
    contract_hash: str
    status: Literal["ok", "circuit_breaker", "rollback_recommended"]
    reasons: list[str]
    triggered_metrics: dict[str, Any]
    decided_at: datetime


@dataclass(frozen=True)
class DriftSignal:
    target_name: str
    release_id: str
    contract_hash: str
    status: Literal["none", "watch", "recompile_recommended"]
    signals: dict[str, Any]
    compared_to_report_id: str | None
    detected_at: datetime


@dataclass(frozen=True)
class OnlineQualitySummary:
    release_id: str
    metrics: RuntimeMetricWindow
    drift_status: str
    failure_status: str
    generated_at: datetime


@dataclass(frozen=True)
class AuditDecision:
    audit_type: Literal["random", "risk"]
    selected: bool
    sampling_probability: float | None
    risk_flags: list[str]
    reason: str | None = None


@dataclass(frozen=True)
class AuditDecisions:
    random: AuditDecision | None
    synchronous_risk: AuditDecision | None
    selected_audit_types: list[Literal["random", "risk"]]
    random_sampling_probability: float
    risk_flags: list[str]


@dataclass(frozen=True)
class SecureAuditPayload:
    trace_id: str
    release_id: str
    input_raw: dict[str, Any]
    lower_layer_output_raw: dict[str, Any] | None
    expires_at: datetime
    storage_policy: Literal["memory_only", "encrypted_short_lived"]


@dataclass(frozen=True)
class AuditRecord:
    audit_id: str
    trace_id: str
    target_name: str
    release_id: str
    contract_hash: str
    audit_type: Literal["random", "risk"]
    status: Literal["ok", "error"]
    sampling_probability: float | None
    reference_output_redacted: dict[str, Any] | None
    reference_output_hash: str | None
    reference_source: str | None
    lower_layer_output_redacted: dict[str, Any] | None
    lower_layer_output_hash: str | None
    is_correct: bool | None
    error_type: (
        Literal[
            "timeout",
            "rate_limited",
            "empty_response",
            "parse_failure",
            "validation_failure",
            "provider_error",
        ]
        | None
    )
    error_message_hash: str | None
    cost: float
    created_at: datetime
    risk_flags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AuditReferenceResult:
    status: Literal["ok", "error"]
    audit_record: AuditRecord
    reference_output_raw: dict[str, Any] | None
    reference_source: str | None
    cost: float
    latency_ms: float
    error_type: (
        Literal[
            "timeout",
            "rate_limited",
            "empty_response",
            "parse_failure",
            "validation_failure",
            "provider_error",
        ]
        | None
    )
    error_message_hash: str | None
    expires_at: datetime


@dataclass(frozen=True)
class RiskDiagnosticDecision:
    trace_id: str
    target_name: str
    release_id: str
    contract_hash: str
    selected: bool
    risk_flags: list[str]
    reason: str | None
    claim_original_output_comparison: bool
    allowed_next_actions: list[
        Literal["reference_call", "release_replay", "future_evidence_collection"]
    ] = field(default_factory=list)


@dataclass(frozen=True)
class AuditSummary:
    target_name: str
    release_id: str
    contract_hash: str
    window_start: datetime
    window_end: datetime
    random_attempt_count: int
    random_success_count: int
    random_reference_failure_count: int
    random_reference_failure_rate: float | None
    random_precision: float | None
    random_precision_lower_bound: float | None
    random_precision_confidence_level: float | None
    random_wrong_accept_rate_upper_bound: float | None
    risk_attempt_count: int
    risk_success_count: int
    risk_reference_failure_count: int
    risk_findings_by_flag: dict[str, int]
    cost: dict[str, Any]
    generated_at: datetime


@dataclass(frozen=True)
class PrivacyReviewRecord:
    review_id: str
    policy_version: str
    decision: Literal["approved", "rejected"]
    approved_for: list[SplitEligibility]
    payload_form: Literal["raw", "canonicalized", "redacted_not_trainable"]
    redactions_applied: list[str]
    reviewed_at: datetime
    reviewer: Literal["policy", "human"]
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ApprovedTelemetryEvidence:
    evidence_id: str
    trace_id: str | None
    release_id: str | None
    target_name: str
    contract_hash: str
    input_payload: dict[str, Any]
    reference_output_payload: dict[str, Any]
    reference_source: ReferenceSource
    source: Literal["l4_fallback", "random_audit", "risk_audit", "user_feedback"]
    approved_for: list[SplitEligibility]
    privacy_review: PrivacyReviewRecord
    source_event_at: datetime
    created_at: datetime


@dataclass(frozen=True)
class UserFeedbackRecord:
    feedback_id: str
    target_name: str
    contract_hash: str
    input_payload: dict[str, Any]
    corrected_output_payload: dict[str, Any]
    submitted_at: datetime
    source: Literal["human_review", "business_system", "customer_correction"]
    reviewer_id_hash: str | None
    requested_approved_for: list[SplitEligibility]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TelemetryDataSource:
    source_id: str
    target_name: str
    contract_hash: str
    cutoff_time: datetime
    records_uri: str
    default_split_eligibility: list[SplitEligibility]
    per_record_split_eligibility_uri: str | None
    included_sources: list[str]
    provenance_digest: str


@dataclass(frozen=True)
class RecompileReason:
    code: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RecompileRequest:
    target_name: str
    contract_hash: str
    reason: RecompileReason
    telemetry_source: TelemetryDataSource | None
    base_release_id: str
    budget_hint: CompileBudget | None
    created_at: datetime
    requested_by: Literal["user", "scheduler", "monitoring"]


@dataclass(frozen=True)
class AgentVisibleTelemetrySummary:
    target_name: str
    release_id: str
    contract_hash: str
    metrics_summary: dict[str, Any]
    drift_summary: dict[str, Any]
    telemetry_source_id: str | None
    redaction_policy_version: str
    generated_at: datetime


@dataclass(frozen=True)
class SchedulerPolicy:
    max_concurrent_compiles: int
    allow_monitoring_recompile: bool
    allow_scheduled_recompile: bool
    require_user_approval_for_insufficient_reference: bool
    default_compile_budget: CompileBudget = field(default_factory=CompileBudget)
    default_snapshot_options: SnapshotOptions = field(default_factory=SnapshotOptions)


@dataclass(frozen=True)
class CompileLaunchDecision:
    status: Literal["accepted", "deferred", "rejected"]
    target_name: str
    contract_hash: str
    base_release_id: str
    reason: str
    budget: CompileBudget | None
    telemetry_source_id: str | None
    snapshot_cutoff_time: datetime | None
    snapshot_options: SnapshotOptions | None
    created_at: datetime


@dataclass(frozen=True)
class TargetRuntimeContract:
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    validate_input_fn: Callable[[dict[str, Any]], dict[str, Any]]
    validate_output_fn: Callable[[dict[str, Any]], dict[str, Any]]
    is_correct_fn: Callable[[dict[str, Any], dict[str, Any]], bool]
    normalize_input_fn: Callable[[dict[str, Any]], str]
    split_group_fn: Callable[[SourceRecord], str]
    slice_tags_fn: Callable[[SourceRecord], list[str]]
    redact_for_trace_fn: Callable[[dict[str, Any]], dict[str, Any]]
    bucket_runtime_metadata_fn: Callable[[dict[str, Any]], dict[str, Any]]
    build_reference_request_fn: (
        Callable[[dict[str, Any], ReferenceContext], dict[str, Any]] | None
    ) = None
    parse_reference_response_fn: Callable[[ReferenceResponse], dict[str, Any]] | None = None
    contract_hash: str | None = None

    def validate_input(self, value: dict[str, Any]) -> dict[str, Any]:
        return self.validate_input_fn(value)

    def validate_output(self, value: dict[str, Any]) -> dict[str, Any]:
        return self.validate_output_fn(value)

    def is_correct(self, output: dict[str, Any], reference: dict[str, Any]) -> bool:
        result = self.is_correct_fn(output, reference)
        if type(result) is not bool:
            raise ValueError("is_correct must return bool")
        return result

    def normalize_input(self, input_value: dict[str, Any]) -> str:
        result = self.normalize_input_fn(input_value)
        if not isinstance(result, str) or not result:
            raise ValueError("normalize_input must return non-empty text")
        return result

    def split_group(self, record: SourceRecord) -> str:
        result = self.split_group_fn(record)
        if not isinstance(result, str) or not result:
            raise ValueError("split_group must return non-empty text")
        return result

    def slice_tags(self, record: SourceRecord) -> list[str]:
        result = self.slice_tags_fn(record)
        if not isinstance(result, list) or any(not isinstance(item, str) for item in result):
            raise ValueError("slice_tags must return list[str]")
        return result

    def redact_for_trace(self, value: dict[str, Any]) -> dict[str, Any]:
        result = self.redact_for_trace_fn(value)
        _check_bounded_dict(result, "redact_for_trace")
        return result

    def bucket_runtime_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        result = self.bucket_runtime_metadata_fn(metadata)
        _check_bounded_dict(result, "bucket_runtime_metadata")
        return result

    def build_reference_request(
        self, input_value: dict[str, Any], context: ReferenceContext
    ) -> dict[str, Any]:
        if self.build_reference_request_fn is None:
            return {"input": input_value, "context": context.metadata}
        result = self.build_reference_request_fn(input_value, context)
        if not isinstance(result, dict):
            raise ValueError("build_reference_request must return dict")
        return result

    def parse_reference_response(self, response: ReferenceResponse) -> dict[str, Any]:
        if self.parse_reference_response_fn is None:
            return self.validate_output(response.payload)
        return self.validate_output(self.parse_reference_response_fn(response))


def _check_bounded_dict(value: Any, name: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must return a dict")
    encoded = json.dumps(value, sort_keys=True, default=str)
    if len(encoded) > 2000:
        raise ValueError(f"{name} output is too large")
    for key, item in value.items():
        if not isinstance(key, str) or len(key) > 80:
            raise ValueError(f"{name} keys must be bounded strings")
        if isinstance(item, str) and len(item) > 200:
            raise ValueError(f"{name} string values must be bounded")
