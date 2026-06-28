from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from darjeeling.errors import TelemetryError
from darjeeling.model import (
    AgentVisibleTelemetrySummary,
    ApprovedTelemetryEvidence,
    AuditRecord,
    AuditReferenceResult,
    CompileBudget,
    DriftSignal,
    L4FallbackResult,
    PrivacyReviewRecord,
    RecompileReason,
    RecompileRequest,
    Release,
    RuntimeMetricWindow,
    SecureAuditPayload,
    TargetDefinition,
    TargetRuntimeContract,
    TelemetryDataSource,
    TelemetryPrivacyPolicy,
    Trace,
    UserFeedbackRecord,
)
from darjeeling.util import new_id, stable_hash, utcnow, write_json


def _review(
    source: str, requested: list[str], policy: TelemetryPrivacyPolicy
) -> PrivacyReviewRecord:
    if source not in policy.allowed_sources:
        return PrivacyReviewRecord(
            review_id=new_id("review"),
            policy_version=policy.policy_version,
            decision="rejected",
            approved_for=[],
            payload_form="redacted_not_trainable",
            redactions_applied=[],
            reviewed_at=utcnow(),
            reviewer="policy",
            notes=["source is not allowed by telemetry privacy policy"],
        )
    if source in policy.human_review_required_sources:
        return PrivacyReviewRecord(
            review_id=new_id("review"),
            policy_version=policy.policy_version,
            decision="rejected",
            approved_for=[],
            payload_form="redacted_not_trainable",
            redactions_applied=[],
            reviewed_at=utcnow(),
            reviewer="policy",
            notes=["source requires human privacy review"],
        )
    if not policy.raw_payload_allowed or policy.canonicalization_required:
        return PrivacyReviewRecord(
            review_id=new_id("review"),
            policy_version=policy.policy_version,
            decision="rejected",
            approved_for=[],
            payload_form="redacted_not_trainable",
            redactions_applied=[],
            reviewed_at=utcnow(),
            reviewer="policy",
            notes=["raw payload retention is not approved by telemetry privacy policy"],
        )
    approved = [
        role
        for role in requested
        if role in policy.default_approved_for_by_source.get(source, [])
    ]
    return PrivacyReviewRecord(
        review_id=new_id("review"),
        policy_version=policy.policy_version,
        decision="approved" if approved else "rejected",
        approved_for=approved,  # type: ignore[arg-type]
        payload_form="raw",
        redactions_applied=[],
        reviewed_at=utcnow(),
        reviewer="policy",
    )


def approve_telemetry_evidence(
    trace: Trace | None,
    secure_payload: SecureAuditPayload | None,
    audit_reference_result: AuditReferenceResult | None,
    l4_fallback_result: L4FallbackResult | None,
    audit_record: AuditRecord | None,
    user_feedback: UserFeedbackRecord | None,
    trace_id: str | None,
    release_id: str | None,
    source_event_at: datetime | None,
    target_name: str,
    contract_hash: str,
    contract: TargetRuntimeContract,
    privacy_policy: TelemetryPrivacyPolicy,
) -> ApprovedTelemetryEvidence | None:
    sources = [
        l4_fallback_result is not None,
        audit_reference_result is not None,
        user_feedback is not None,
    ]
    if sum(bool(value) for value in sources) != 1:
        raise TelemetryError("approve_telemetry_evidence accepts exactly one source path")
    if l4_fallback_result is not None:
        if secure_payload is not None or audit_record is not None:
            raise TelemetryError("unexpected carriers supplied for l4 fallback evidence")
        if trace is None:
            raise TelemetryError("l4 fallback evidence requires trace provenance")
        if (
            l4_fallback_result.status != "ok"
            or not trace_id
            or not release_id
            or source_event_at is None
        ):
            return None
        if (
            trace.trace_id != trace_id
            or trace.release_id != release_id
            or trace.target_name != target_name
            or trace.contract_hash != contract_hash
            or trace.status != "ok"
            or trace.chosen_layer != "L4"
        ):
            raise TelemetryError("l4 fallback evidence carriers do not describe the same request")
        if stable_hash(l4_fallback_result.input_raw) != trace.input_hash:
            raise TelemetryError("l4 fallback evidence carriers do not describe the same request")
        if (
            l4_fallback_result.output_validated is None
            or trace.final_output_hash != stable_hash(l4_fallback_result.output_validated)
        ):
            raise TelemetryError("l4 fallback evidence carriers do not describe the same request")
        source = "l4_fallback"
        input_payload = l4_fallback_result.input_raw
        output_payload = l4_fallback_result.output_validated or {}
        reference_source = l4_fallback_result.reference_source or "versioned_l4"
        approved_for = privacy_policy.default_approved_for_by_source.get(source, [])
    elif audit_reference_result is not None:
        if trace_id is not None or release_id is not None or source_event_at is not None:
            raise TelemetryError("unexpected carrier fields supplied for audit evidence")
        if not (trace and secure_payload and audit_record):
            raise TelemetryError(
                "audit evidence requires trace, secure payload, audit result, and audit record"
            )
        now = utcnow()
        if secure_payload.expires_at <= now or audit_reference_result.expires_at <= now:
            raise TelemetryError("audit evidence carrier expired")
        reference_audit_record = audit_reference_result.audit_record
        if (
            trace.trace_id != secure_payload.trace_id
            or trace.trace_id != audit_record.trace_id
            or trace.release_id != secure_payload.release_id
            or trace.release_id != audit_record.release_id
            or trace.target_name != target_name
            or trace.target_name != audit_record.target_name
            or trace.contract_hash != contract_hash
            or trace.contract_hash != audit_record.contract_hash
            or reference_audit_record.trace_id != trace.trace_id
            or reference_audit_record.release_id != trace.release_id
            or reference_audit_record.target_name != trace.target_name
            or reference_audit_record.contract_hash != trace.contract_hash
            or reference_audit_record.audit_id != audit_record.audit_id
            or reference_audit_record.audit_type != audit_record.audit_type
            or reference_audit_record.status != audit_record.status
            or reference_audit_record.status != audit_reference_result.status
            or audit_record.audit_type not in trace.selected_audit_types
        ):
            raise TelemetryError("audit evidence carriers do not describe the same request")
        if stable_hash(secure_payload.input_raw) != trace.input_hash:
            raise TelemetryError("audit evidence carriers do not describe the same request")
        if secure_payload.lower_layer_output_raw is None:
            raise TelemetryError("audit evidence requires lower-layer output payload")
        lower_output_hash = stable_hash(secure_payload.lower_layer_output_raw)
        if (
            trace.final_output_hash != lower_output_hash
            or audit_record.lower_layer_output_hash != lower_output_hash
            or reference_audit_record.lower_layer_output_hash != lower_output_hash
        ):
            raise TelemetryError("audit evidence carriers do not describe the same request")
        if (
            audit_reference_result.status != "ok"
            or audit_reference_result.reference_output_raw is None
        ):
            return None
        reference_output_hash = stable_hash(audit_reference_result.reference_output_raw)
        if (
            audit_record.reference_output_hash != reference_output_hash
            or reference_audit_record.reference_output_hash != reference_output_hash
            or audit_reference_result.reference_source != audit_record.reference_source
            or audit_reference_result.reference_source != reference_audit_record.reference_source
        ):
            raise TelemetryError("audit evidence carriers do not describe the same request")
        source = "random_audit" if audit_record.audit_type == "random" else "risk_audit"
        trace_id = trace.trace_id
        release_id = trace.release_id
        source_event_at = audit_record.created_at
        input_payload = secure_payload.input_raw
        output_payload = audit_reference_result.reference_output_raw
        reference_source = audit_reference_result.reference_source or "versioned_l4"
        approved_for = privacy_policy.default_approved_for_by_source.get(source, [])
    else:
        assert user_feedback is not None
        if (
            trace is not None
            or secure_payload is not None
            or audit_record is not None
            or trace_id is not None
            or release_id is not None
        ):
            raise TelemetryError("unexpected carriers supplied for user feedback evidence")
        if user_feedback.target_name != target_name or user_feedback.contract_hash != contract_hash:
            raise TelemetryError("user feedback scope mismatch")
        source = "user_feedback"
        trace_id = None
        release_id = None
        source_event_at = user_feedback.submitted_at
        input_payload = user_feedback.input_payload
        output_payload = user_feedback.corrected_output_payload
        reference_source = "user_feedback"
        approved_for = user_feedback.requested_approved_for
    review = _review(source, approved_for, privacy_policy)
    effective = [role for role in approved_for if role in review.approved_for]
    if (
        review.decision != "approved"
        or not effective
        or review.payload_form == "redacted_not_trainable"
    ):
        return None
    created_at = utcnow()
    if source_event_at is None:
        raise TelemetryError("source event timestamp is required")
    if created_at < source_event_at:
        created_at = source_event_at
    if source in {"l4_fallback", "random_audit", "risk_audit"} and not release_id:
        raise TelemetryError("runtime-derived evidence requires release provenance")
    return ApprovedTelemetryEvidence(
        evidence_id=new_id("evidence"),
        trace_id=trace_id,
        release_id=release_id,
        target_name=target_name,
        contract_hash=contract_hash,
        input_payload=contract.validate_input(input_payload),
        reference_output_payload=contract.validate_output(output_payload),
        reference_source=reference_source,  # type: ignore[arg-type]
        source=source,  # type: ignore[arg-type]
        approved_for=effective,  # type: ignore[arg-type]
        privacy_review=review,
        source_event_at=source_event_at,
        created_at=created_at,
    )


def _validate_approved_evidence(evidence: ApprovedTelemetryEvidence) -> None:
    if (
        evidence.privacy_review.decision != "approved"
        or not evidence.approved_for
        or not set(evidence.approved_for).issubset(set(evidence.privacy_review.approved_for))
        or evidence.privacy_review.payload_form == "redacted_not_trainable"
        or evidence.created_at < evidence.source_event_at
    ):
        raise TelemetryError("approved evidence is not eligible for telemetry source")
    if evidence.source in {"l4_fallback", "random_audit", "risk_audit"}:
        if not evidence.release_id:
            raise TelemetryError("runtime-derived evidence requires release provenance")


def build_telemetry_data_source(
    traces: Iterable[Trace],
    audits: Iterable[AuditRecord],
    approved_evidence: Iterable[ApprovedTelemetryEvidence],
    target_name: str,
    contract_hash: str,
    cutoff_time: datetime,
    contract: TargetRuntimeContract,
    output_dir: Path = Path(".darjeeling/telemetry"),
) -> TelemetryDataSource:
    trace_map = {trace.trace_id: trace for trace in traces}
    for trace in trace_map.values():
        if trace.target_name != target_name or trace.contract_hash != contract_hash:
            raise TelemetryError("trace provenance missing or mismatched")
    audits_by_trace: dict[str, list[AuditRecord]] = {}
    for audit in audits:
        if audit.target_name != target_name or audit.contract_hash != contract_hash:
            raise TelemetryError("audit provenance missing or mismatched")
        audits_by_trace.setdefault(audit.trace_id, []).append(audit)
    included = []
    for evidence in approved_evidence:
        if evidence.target_name != target_name or evidence.contract_hash != contract_hash:
            raise TelemetryError("approved evidence scope mismatch")
        if evidence.created_at > cutoff_time:
            continue
        _validate_approved_evidence(evidence)
        if evidence.trace_id is not None:
            trace = trace_map.get(evidence.trace_id)
            if (
                trace is None
                or trace.target_name != target_name
                or trace.contract_hash != contract_hash
            ):
                raise TelemetryError("trace provenance missing or mismatched")
            if evidence.release_id and trace.release_id != evidence.release_id:
                raise TelemetryError("trace release provenance mismatch")
            audit_rows = audits_by_trace.get(evidence.trace_id, [])
            matching_audit_source = False
            for audit in audit_rows:
                if audit.target_name != target_name or audit.contract_hash != contract_hash:
                    raise TelemetryError("audit provenance missing or mismatched")
                if evidence.release_id and audit.release_id != evidence.release_id:
                    raise TelemetryError("audit release provenance mismatch")
                if evidence.source == f"{audit.audit_type}_audit":
                    matching_audit_source = True
            if evidence.source in {"random_audit", "risk_audit"} and not matching_audit_source:
                raise TelemetryError("audit provenance missing or mismatched")
        included.append(evidence)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_id = new_id("telemetry")
    records_path = output_dir / f"{source_id}.json"
    rows = []
    for evidence in included:
        rows.append(
            {
                **asdict(evidence),
                "source_event_at": evidence.source_event_at.isoformat(),
                "created_at": evidence.created_at.isoformat(),
                "privacy_review": {
                    **asdict(evidence.privacy_review),
                    "reviewed_at": evidence.privacy_review.reviewed_at.isoformat(),
                },
            }
        )
    write_json(records_path, rows)
    unique_permissions = {tuple(e.approved_for) for e in included}
    default = (
        list(next(iter(unique_permissions))) if len(unique_permissions) == 1 and included else []
    )
    per_record_uri = None
    if included:
        per_record_path = output_dir / f"{source_id}-split-eligibility.json"
        write_json(per_record_path, {e.evidence_id: e.approved_for for e in included})
        per_record_uri = str(per_record_path)
    return TelemetryDataSource(
        source_id=source_id,
        target_name=target_name,
        contract_hash=contract_hash,
        cutoff_time=cutoff_time,
        records_uri=str(records_path),
        default_split_eligibility=default,  # type: ignore[arg-type]
        per_record_split_eligibility_uri=per_record_uri,
        included_sources=sorted({e.source for e in included}),
        provenance_digest=stable_hash(rows),
    )


def request_recompile(
    definition: TargetDefinition,
    base_release: Release,
    reason: RecompileReason,
    telemetry_source: TelemetryDataSource | None,
    budget_hint: CompileBudget | None,
    requested_by: str = "user",
) -> RecompileRequest:
    if (
        base_release.target_name != definition.name
        or base_release.contract_hash != definition.contract_hash
    ):
        raise TelemetryError("base release scope mismatch")
    if telemetry_source is not None and (
        telemetry_source.target_name != definition.name
        or telemetry_source.contract_hash != definition.contract_hash
    ):
        raise TelemetryError("telemetry source scope mismatch")
    if requested_by not in {"user", "scheduler", "monitoring"}:
        raise TelemetryError("requested_by is not allowed")
    return RecompileRequest(
        target_name=definition.name,
        contract_hash=definition.contract_hash,
        reason=reason,
        telemetry_source=telemetry_source,
        base_release_id=base_release.release_id,
        budget_hint=budget_hint,
        created_at=utcnow(),
        requested_by=requested_by,  # type: ignore[arg-type]
    )


def export_agent_visible_telemetry_summary(
    metrics: RuntimeMetricWindow,
    drift_signal: DriftSignal,
    redaction_policy: dict,
) -> AgentVisibleTelemetrySummary:
    if (
        metrics.target_name != drift_signal.target_name
        or metrics.release_id != drift_signal.release_id
        or metrics.contract_hash != drift_signal.contract_hash
    ):
        raise TelemetryError("metrics and drift signal scope mismatch")
    return AgentVisibleTelemetrySummary(
        target_name=metrics.target_name,
        release_id=metrics.release_id,
        contract_hash=metrics.contract_hash,
        metrics_summary={
            "request_count": metrics.request_count,
            "local_coverage": metrics.local_coverage,
            "l4_fallback_rate": metrics.l4_fallback_rate,
            "reason_code_counts": metrics.reason_code_counts,
        },
        drift_summary={"status": drift_signal.status, "signals": drift_signal.signals},
        telemetry_source_id=redaction_policy.get("telemetry_source_id"),
        redaction_policy_version=str(redaction_policy.get("policy_version", "unknown")),
        generated_at=utcnow(),
    )
