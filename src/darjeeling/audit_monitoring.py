from __future__ import annotations

import random
from collections import Counter
from collections.abc import Iterable
from datetime import timedelta
from typing import Literal

from darjeeling.errors import RuntimeErrorSafe
from darjeeling.model import (
    AuditDecision,
    AuditDecisions,
    AuditRecord,
    AuditReferenceResult,
    AuditSummary,
    ReferenceBroker,
    ReferenceContext,
    Release,
    RiskDiagnosticDecision,
    RuntimeRequest,
    SecureAuditPayload,
    ServingResult,
    TargetRuntimeContract,
    Trace,
)
from darjeeling.snapshot_reference import call_reference
from darjeeling.util import bounded_code, new_id, stable_hash, utcnow


def decide_random_audit(
    release: Release,
    request: RuntimeRequest,
    serving_result: ServingResult,
    audit_settings: dict,
) -> AuditDecision:
    probability = float(
        audit_settings.get("random_audit_rate", release.routing.audit.get("random_audit_rate", 0.0))
    )
    lower_layer_accept = (
        serving_result.chosen_layer in {"L1", "L2", "L3"} and serving_result.status == "ok"
    )
    selected = lower_layer_accept and random.random() < probability
    return AuditDecision(
        audit_type="random",
        selected=selected,
        sampling_probability=probability,
        risk_flags=[],
        reason="sampled" if selected else None,
    )


def decide_synchronous_risk_audit(
    release: Release,
    request: RuntimeRequest,
    serving_result: ServingResult,
    risk_rules: dict,
) -> AuditDecision:
    lower_layer_accept = (
        serving_result.status == "ok"
        and serving_result.chosen_layer in {"L1", "L2", "L3"}
        and serving_result.output is not None
    )
    if not lower_layer_accept:
        return AuditDecision(
            audit_type="risk",
            selected=False,
            sampling_probability=None,
            risk_flags=[],
            reason="no_lower_layer_output",
        )
    flags: list[str] = []
    for rule in risk_rules.get("flags", []):
        code = rule.get("code")
        metadata_key = rule.get("metadata_key")
        if code and metadata_key and request.metadata.get(metadata_key) == rule.get("equals"):
            if not bounded_code(code):
                raise RuntimeErrorSafe("risk flag code is not trace-safe")
            flags.append(code)
    selected = bool(flags)
    return AuditDecision(
        audit_type="risk",
        selected=selected,
        sampling_probability=None,
        risk_flags=flags,
        reason="risk_flag" if selected else None,
    )


def combine_audit_decisions(
    random_decision: AuditDecision | None,
    risk_decision: AuditDecision | None,
) -> AuditDecisions:
    selected: list[Literal["random", "risk"]] = []
    flags: list[str] = []
    if random_decision and random_decision.selected:
        selected.append("random")
    if risk_decision and risk_decision.selected:
        selected.append("risk")
        flags.extend(risk_decision.risk_flags)
    return AuditDecisions(
        random=random_decision,
        synchronous_risk=risk_decision,
        selected_audit_types=selected,
        random_sampling_probability=random_decision.sampling_probability
        if random_decision
        else 0.0,
        risk_flags=flags,
    )


def capture_secure_audit_payload(
    trace_id: str,
    runtime_request: RuntimeRequest,
    serving_result: ServingResult,
    audit_decisions: AuditDecisions,
    secure_store: dict[str, SecureAuditPayload],
) -> SecureAuditPayload | None:
    if not audit_decisions.selected_audit_types:
        return None
    if (
        serving_result.status != "ok"
        or serving_result.chosen_layer not in {"L1", "L2", "L3"}
        or serving_result.output is None
    ):
        raise RuntimeErrorSafe("selected audits require accepted lower-layer output")
    lower_output = (
        serving_result.output if serving_result.chosen_layer in {"L1", "L2", "L3"} else None
    )
    payload = SecureAuditPayload(
        trace_id=trace_id,
        release_id=serving_result.release_id,
        input_raw=runtime_request.input,
        lower_layer_output_raw=lower_output,
        expires_at=utcnow() + timedelta(minutes=15),
        storage_policy="memory_only",
    )
    secure_store[trace_id] = payload
    return payload


def _audit_decision_for_type(
    trace: Trace, audit_type: Literal["random", "risk"]
) -> tuple[float | None, list[str]]:
    if audit_type == "random":
        return trace.random_audit_probability, []
    return None, trace.risk_audit_flags


def run_audit_reference_call(
    trace: Trace,
    secure_payload: SecureAuditPayload,
    audit_type: Literal["random", "risk"],
    contract: TargetRuntimeContract,
    broker: ReferenceBroker,
    secure_store: dict[str, SecureAuditPayload] | None = None,
) -> AuditReferenceResult:
    try:
        return _run_audit_reference_call(trace, secure_payload, audit_type, contract, broker)
    finally:
        if secure_store is not None:
            secure_store.pop(secure_payload.trace_id, None)


def _run_audit_reference_call(
    trace: Trace,
    secure_payload: SecureAuditPayload,
    audit_type: Literal["random", "risk"],
    contract: TargetRuntimeContract,
    broker: ReferenceBroker,
) -> AuditReferenceResult:
    if trace.trace_id != secure_payload.trace_id or trace.release_id != secure_payload.release_id:
        raise RuntimeErrorSafe("trace and secure audit payload do not describe the same request")
    if secure_payload.expires_at <= utcnow():
        raise RuntimeErrorSafe("secure audit payload expired")
    if contract.contract_hash != trace.contract_hash:
        raise RuntimeErrorSafe("audit contract scope does not match trace")
    if audit_type not in trace.selected_audit_types:
        raise RuntimeErrorSafe("audit type was not selected for this trace")
    if audit_type == "risk" and (
        not trace.risk_audit_flags or any(not bounded_code(code) for code in trace.risk_audit_flags)
    ):
        raise RuntimeErrorSafe("risk audit requires trace-safe risk flags")
    if stable_hash(secure_payload.input_raw) != trace.input_hash:
        raise RuntimeErrorSafe("trace and secure audit payload do not describe the same request")
    if secure_payload.lower_layer_output_raw is None:
        raise RuntimeErrorSafe("selected audits require accepted lower-layer output")
    lower_output_hash = stable_hash(secure_payload.lower_layer_output_raw)
    if trace.final_output_hash != lower_output_hash:
        raise RuntimeErrorSafe("trace and secure audit payload do not describe the same request")
    accepted_attempt_hashes = [
        attempt.output_hash
        for attempt in trace.attempts
        if attempt.layer == trace.chosen_layer and attempt.decision == "accept"
    ]
    if accepted_attempt_hashes and lower_output_hash not in accepted_attempt_hashes:
        raise RuntimeErrorSafe("trace and secure audit payload do not describe the same request")
    sampling_probability, flags = _audit_decision_for_type(trace, audit_type)
    result = call_reference(
        contract,
        secure_payload.input_raw,
        broker,
        ReferenceContext(purpose=f"{audit_type}_audit", request_id=trace.trace_id),
    )
    if result.status == "ok" and result.output is not None:
        is_correct = None
        if secure_payload.lower_layer_output_raw is not None:
            is_correct = contract.is_correct(secure_payload.lower_layer_output_raw, result.output)
        record = AuditRecord(
            audit_id=new_id("audit"),
            trace_id=trace.trace_id,
            target_name=trace.target_name,
            release_id=trace.release_id,
            contract_hash=trace.contract_hash,
            audit_type=audit_type,
            status="ok",
            sampling_probability=sampling_probability,
            reference_output_redacted=contract.redact_for_trace(result.output),
            reference_output_hash=stable_hash(result.output),
            reference_source=result.reference_source,
            lower_layer_output_redacted=contract.redact_for_trace(
                secure_payload.lower_layer_output_raw
            )
            if secure_payload.lower_layer_output_raw
            else None,
            lower_layer_output_hash=stable_hash(secure_payload.lower_layer_output_raw)
            if secure_payload.lower_layer_output_raw
            else None,
            is_correct=is_correct,
            error_type=None,
            error_message_hash=None,
            cost=result.cost,
            created_at=utcnow(),
            risk_flags=list(flags) if audit_type == "risk" else [],
        )
        return AuditReferenceResult(
            status="ok",
            audit_record=record,
            reference_output_raw=result.output,
            reference_source=result.reference_source,
            cost=result.cost,
            latency_ms=result.latency_ms,
            error_type=None,
            error_message_hash=None,
            expires_at=utcnow() + timedelta(minutes=5),
        )
    record = AuditRecord(
        audit_id=new_id("audit"),
        trace_id=trace.trace_id,
        target_name=trace.target_name,
        release_id=trace.release_id,
        contract_hash=trace.contract_hash,
        audit_type=audit_type,
        status="error",
        sampling_probability=sampling_probability,
        reference_output_redacted=None,
        reference_output_hash=None,
        reference_source=None,
        lower_layer_output_redacted=contract.redact_for_trace(secure_payload.lower_layer_output_raw)
        if secure_payload.lower_layer_output_raw
        else None,
        lower_layer_output_hash=stable_hash(secure_payload.lower_layer_output_raw)
        if secure_payload.lower_layer_output_raw
        else None,
        is_correct=None,
        error_type=result.error_type or "provider_error",  # type: ignore[arg-type]
        error_message_hash=result.error_message_hash,
        cost=result.cost,
        created_at=utcnow(),
        risk_flags=list(flags) if audit_type == "risk" else [],
    )
    return AuditReferenceResult(
        status="error",
        audit_record=record,
        reference_output_raw=None,
        reference_source=None,
        cost=result.cost,
        latency_ms=result.latency_ms,
        error_type=result.error_type or "provider_error",  # type: ignore[arg-type]
        error_message_hash=result.error_message_hash,
        expires_at=utcnow() + timedelta(minutes=5),
    )


def decide_asynchronous_risk_diagnostic(
    trace: Trace, risk_rules: dict
) -> RiskDiagnosticDecision:
    offline_flags = []
    for code in risk_rules.get("offline_flags", []):
        if not bounded_code(code):
            raise RuntimeErrorSafe("risk flag code is not trace-safe")
        offline_flags.append(code)
    matched_flags = [code for code in trace.risk_audit_flags if code in offline_flags]
    selected = bool(matched_flags)
    return RiskDiagnosticDecision(
        trace_id=trace.trace_id,
        target_name=trace.target_name,
        release_id=trace.release_id,
        contract_hash=trace.contract_hash,
        selected=selected,
        risk_flags=matched_flags,
        reason="offline_risk_flag" if selected else None,
        claim_original_output_comparison=False,
        allowed_next_actions=[
            "reference_call",
            "release_replay",
            "future_evidence_collection",
        ]
        if selected
        else [],
    )


def build_audit_summary(
    audits: Iterable[AuditRecord],
    window: tuple,
    target_name: str,
    release_id: str,
    contract_hash: str,
) -> AuditSummary:
    selected = [audit for audit in audits if window[0] <= audit.created_at <= window[1]]
    for audit in selected:
        if (
            audit.target_name != target_name
            or audit.release_id != release_id
            or audit.contract_hash != contract_hash
        ):
            raise RuntimeErrorSafe("audit record scope mismatch")
    random_audits = [audit for audit in selected if audit.audit_type == "random"]
    risk_audits = [audit for audit in selected if audit.audit_type == "risk"]
    random_success = [audit for audit in random_audits if audit.status == "ok"]
    random_fail = [audit for audit in random_audits if audit.status == "error"]
    evaluable = [audit for audit in random_success if audit.is_correct is not None]
    correct = [audit for audit in evaluable if audit.is_correct is True]
    wrong = [audit for audit in evaluable if audit.is_correct is False]
    risk_findings = Counter()
    for audit in risk_audits:
        if audit.is_correct is False:
            for flag in audit.risk_flags or ["wrong_accept"]:
                risk_findings[flag] += 1
    precision = len(correct) / len(evaluable) if evaluable else None
    wrong_rate = len(wrong) / len(evaluable) if evaluable else None
    return AuditSummary(
        target_name=target_name,
        release_id=release_id,
        contract_hash=contract_hash,
        window_start=window[0],
        window_end=window[1],
        random_attempt_count=len(random_audits),
        random_success_count=len(random_success),
        random_reference_failure_count=len(random_fail),
        random_reference_failure_rate=len(random_fail) / len(random_audits)
        if random_audits
        else None,
        random_precision=precision,
        random_precision_lower_bound=max(0.0, precision - 0.05) if precision is not None else None,
        random_precision_confidence_level=0.95 if evaluable else None,
        random_wrong_accept_rate_upper_bound=min(1.0, wrong_rate + 0.05)
        if wrong_rate is not None
        else None,
        risk_attempt_count=len(risk_audits),
        risk_success_count=sum(1 for audit in risk_audits if audit.status == "ok"),
        risk_reference_failure_count=sum(1 for audit in risk_audits if audit.status == "error"),
        risk_findings_by_flag=dict(risk_findings),
        cost={"total": sum(audit.cost for audit in selected)},
        generated_at=utcnow(),
    )
