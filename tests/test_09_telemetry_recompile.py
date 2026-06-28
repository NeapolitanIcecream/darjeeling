from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest
from conftest import PrefixBroker

from darjeeling.audit_monitoring import (
    capture_secure_audit_payload,
    combine_audit_decisions,
    run_audit_reference_call,
)
from darjeeling.model import (
    AuditDecision,
    CacheMiss,
    DriftSignal,
    L4FallbackResult,
    RecompileReason,
    Release,
    RoutingSettings,
    RuntimeMetricWindow,
    RuntimeRequest,
    ServingResult,
    UserFeedbackRecord,
)
from darjeeling.runtime_trace_metrics import write_trace
from darjeeling.target_definition import load_checked_target
from darjeeling.telemetry_recompile import (
    approve_telemetry_evidence,
    build_telemetry_data_source,
    export_agent_visible_telemetry_summary,
    request_recompile,
)
from darjeeling.util import utcnow


def test_trace_or_audit_record_alone_cannot_become_snapshot_evidence(target_dir: Path) -> None:
    definition, contract, _ = load_checked_target(target_dir)
    with pytest.raises(Exception, match="exactly one source"):
        approve_telemetry_evidence(
            None,
            None,
            None,
            None,
            None,
            None,
            "trace",
            "rel",
            utcnow(),
            definition.name,
            definition.contract_hash,
            contract,
            definition.runtime_config.telemetry_privacy_policy,
        )


def test_privacy_policy_rejects_human_review_and_uncanonicalized_payloads(
    target_dir: Path,
) -> None:
    definition, contract, _ = load_checked_target(target_dir)
    feedback = UserFeedbackRecord(
        "fb1",
        definition.name,
        definition.contract_hash,
        {"text": "c:user"},
        {"label": "c"},
        utcnow(),
        "human_review",
        None,
        ["train"],
    )
    base_policy = definition.runtime_config.telemetry_privacy_policy

    no_raw_policy = replace(
        base_policy, raw_payload_allowed=False, canonicalization_required=True
    )
    assert (
        approve_telemetry_evidence(
            None,
            None,
            None,
            None,
            None,
            feedback,
            None,
            None,
            feedback.submitted_at,
            definition.name,
            definition.contract_hash,
            contract,
            no_raw_policy,
        )
        is None
    )

    human_review_policy = replace(
        base_policy, human_review_required_sources=["user_feedback"]
    )
    assert (
        approve_telemetry_evidence(
            None,
            None,
            None,
            None,
            None,
            feedback,
            None,
            None,
            feedback.submitted_at,
            definition.name,
            definition.contract_hash,
            contract,
            human_review_policy,
        )
        is None
    )

    no_user_feedback_grant_policy = replace(
        base_policy,
        default_approved_for_by_source={
            source: roles
            for source, roles in base_policy.default_approved_for_by_source.items()
            if source != "user_feedback"
        },
    )
    assert (
        approve_telemetry_evidence(
            None,
            None,
            None,
            None,
            None,
            feedback,
            None,
            None,
            feedback.submitted_at,
            definition.name,
            definition.contract_hash,
            contract,
            no_user_feedback_grant_policy,
        )
        is None
    )


def test_approval_rejects_stray_carriers_for_selected_source(target_dir: Path) -> None:
    definition, contract, _ = load_checked_target(target_dir)
    serving = ServingResult(
        "rel",
        "cascade",
        "ok",
        CacheMiss("k"),
        None,
        {"label": "a"},
        "L1",
        0.0,
        1.0,
        None,
        None,
        None,
    )
    request = RuntimeRequest("req", definition.name, {"text": "a:x"})
    decisions = combine_audit_decisions(AuditDecision("random", True, 1.0, []), None)
    trace = write_trace("trace-1", definition.contract_hash, contract, request, serving, decisions)
    secure_payload = capture_secure_audit_payload(
        "trace-1", request, serving, decisions, {}
    )
    assert secure_payload is not None
    audit_result = run_audit_reference_call(
        trace, secure_payload, "random", contract, PrefixBroker()
    )
    feedback = UserFeedbackRecord(
        "fb1",
        definition.name,
        definition.contract_hash,
        {"text": "c:user"},
        {"label": "c"},
        utcnow(),
        "human_review",
        None,
        ["train"],
    )
    with pytest.raises(Exception, match="unexpected carriers"):
        approve_telemetry_evidence(
            trace,
            None,
            None,
            None,
            None,
            feedback,
            None,
            None,
            feedback.submitted_at,
            definition.name,
            definition.contract_hash,
            contract,
            definition.runtime_config.telemetry_privacy_policy,
        )

    l4_result = L4FallbackResult(
        request.input,
        "ok",
        {"label": "a"},
        {"label": "a"},
        "versioned_l4",
        0.01,
        3.0,
        "stop",
    )
    with pytest.raises(Exception, match="unexpected carriers"):
        approve_telemetry_evidence(
            trace,
            None,
            None,
            l4_result,
            audit_result.audit_record,
            None,
            trace.trace_id,
            trace.release_id,
            trace.timestamp,
            definition.name,
            definition.contract_hash,
            contract,
            definition.runtime_config.telemetry_privacy_policy,
        )


def test_l4_fallback_evidence_requires_matching_trace_provenance(target_dir: Path) -> None:
    definition, contract, _ = load_checked_target(target_dir)
    request = RuntimeRequest("req", definition.name, {"text": "a:x"})
    l4_result = L4FallbackResult(
        request.input,
        "ok",
        {"label": "a"},
        {"label": "a"},
        "versioned_l4",
        0.01,
        3.0,
        "stop",
    )
    serving = ServingResult(
        "rel",
        "cascade",
        "ok",
        CacheMiss("k"),
        None,
        {"label": "a"},
        "L4",
        0.01,
        3.0,
        None,
        None,
        l4_result,
    )
    trace = write_trace(
        "trace-1",
        definition.contract_hash,
        contract,
        request,
        serving,
        combine_audit_decisions(None, None),
    )

    evidence = approve_telemetry_evidence(
        trace,
        None,
        None,
        l4_result,
        None,
        None,
        trace.trace_id,
        trace.release_id,
        trace.timestamp,
        definition.name,
        definition.contract_hash,
        contract,
        definition.runtime_config.telemetry_privacy_policy,
    )
    assert evidence is not None
    assert evidence.source == "l4_fallback"
    assert evidence.release_id == trace.release_id

    with pytest.raises(Exception, match="requires trace"):
        approve_telemetry_evidence(
            None,
            None,
            None,
            l4_result,
            None,
            None,
            trace.trace_id,
            trace.release_id,
            trace.timestamp,
            definition.name,
            definition.contract_hash,
            contract,
            definition.runtime_config.telemetry_privacy_policy,
        )
    with pytest.raises(Exception, match="same request"):
        approve_telemetry_evidence(
            trace,
            None,
            None,
            l4_result,
            None,
            None,
            trace.trace_id,
            "other-release",
            trace.timestamp,
            definition.name,
            definition.contract_hash,
            contract,
            definition.runtime_config.telemetry_privacy_policy,
        )
    with pytest.raises(Exception, match="same request"):
        approve_telemetry_evidence(
            trace,
            None,
            None,
            replace(l4_result, input_raw={"text": "b:tampered"}),
            None,
            None,
            trace.trace_id,
            trace.release_id,
            trace.timestamp,
            definition.name,
            definition.contract_hash,
            contract,
            definition.runtime_config.telemetry_privacy_policy,
        )


def test_audit_evidence_rejects_embedded_audit_scope_mismatch(target_dir: Path) -> None:
    definition, contract, _ = load_checked_target(target_dir)
    serving = ServingResult(
        "rel",
        "cascade",
        "ok",
        CacheMiss("k"),
        None,
        {"label": "a"},
        "L1",
        0.0,
        1.0,
        None,
        None,
        None,
    )
    request = RuntimeRequest("req", definition.name, {"text": "a:x"})
    decisions = combine_audit_decisions(AuditDecision("random", True, 1.0, []), None)
    trace = write_trace("trace-1", definition.contract_hash, contract, request, serving, decisions)
    secure_payload = capture_secure_audit_payload(
        "trace-1", request, serving, decisions, {}
    )
    assert secure_payload is not None
    audit_result = run_audit_reference_call(
        trace, secure_payload, "random", contract, PrefixBroker()
    )
    bad_embedded_record = replace(
        audit_result.audit_record, contract_hash="wrong-contract"
    )
    mismatched_result = replace(audit_result, audit_record=bad_embedded_record)

    with pytest.raises(Exception, match="same request"):
        approve_telemetry_evidence(
            trace,
            secure_payload,
            mismatched_result,
            None,
            audit_result.audit_record,
            None,
            None,
            None,
            None,
            definition.name,
            definition.contract_hash,
            contract,
            definition.runtime_config.telemetry_privacy_policy,
        )


def test_audit_evidence_rejects_inconsistent_audit_result_status(target_dir: Path) -> None:
    definition, contract, _ = load_checked_target(target_dir)
    serving = ServingResult(
        "rel",
        "cascade",
        "ok",
        CacheMiss("k"),
        None,
        {"label": "a"},
        "L1",
        0.0,
        1.0,
        None,
        None,
        None,
    )
    request = RuntimeRequest("req", definition.name, {"text": "a:x"})
    decisions = combine_audit_decisions(AuditDecision("random", True, 1.0, []), None)
    trace = write_trace("trace-1", definition.contract_hash, contract, request, serving, decisions)
    secure_payload = capture_secure_audit_payload(
        "trace-1", request, serving, decisions, {}
    )
    assert secure_payload is not None
    audit_result = run_audit_reference_call(
        trace, secure_payload, "random", contract, PrefixBroker()
    )
    error_record = replace(audit_result.audit_record, status="error")
    inconsistent_result = replace(audit_result, audit_record=error_record)

    with pytest.raises(Exception, match="same request"):
        approve_telemetry_evidence(
            trace,
            secure_payload,
            inconsistent_result,
            None,
            error_record,
            None,
            None,
            None,
            None,
            definition.name,
            definition.contract_hash,
            contract,
            definition.runtime_config.telemetry_privacy_policy,
        )


def test_audit_evidence_rejects_tampered_secure_payload_hashes(target_dir: Path) -> None:
    definition, contract, _ = load_checked_target(target_dir)
    serving = ServingResult(
        "rel",
        "cascade",
        "ok",
        CacheMiss("k"),
        None,
        {"label": "a"},
        "L1",
        0.0,
        1.0,
        None,
        None,
        None,
    )
    request = RuntimeRequest("req", definition.name, {"text": "a:x"})
    decisions = combine_audit_decisions(AuditDecision("random", True, 1.0, []), None)
    trace = write_trace("trace-1", definition.contract_hash, contract, request, serving, decisions)
    secure_payload = capture_secure_audit_payload(
        "trace-1", request, serving, decisions, {}
    )
    assert secure_payload is not None
    audit_result = run_audit_reference_call(
        trace, secure_payload, "random", contract, PrefixBroker()
    )

    tampered_input = replace(secure_payload, input_raw={"text": "b:tampered"})
    with pytest.raises(Exception, match="same request"):
        approve_telemetry_evidence(
            trace,
            tampered_input,
            audit_result,
            None,
            audit_result.audit_record,
            None,
            None,
            None,
            None,
            definition.name,
            definition.contract_hash,
            contract,
            definition.runtime_config.telemetry_privacy_policy,
        )

    tampered_output = replace(secure_payload, lower_layer_output_raw={"label": "b"})
    with pytest.raises(Exception, match="same request"):
        approve_telemetry_evidence(
            trace,
            tampered_output,
            audit_result,
            None,
            audit_result.audit_record,
            None,
            None,
            None,
            None,
            definition.name,
            definition.contract_hash,
            contract,
            definition.runtime_config.telemetry_privacy_policy,
        )


def test_audit_evidence_rejects_tampered_reference_output_or_source(
    target_dir: Path,
) -> None:
    definition, contract, _ = load_checked_target(target_dir)
    serving = ServingResult(
        "rel",
        "cascade",
        "ok",
        CacheMiss("k"),
        None,
        {"label": "a"},
        "L1",
        0.0,
        1.0,
        None,
        None,
        None,
    )
    request = RuntimeRequest("req", definition.name, {"text": "a:x"})
    decisions = combine_audit_decisions(AuditDecision("random", True, 1.0, []), None)
    trace = write_trace("trace-1", definition.contract_hash, contract, request, serving, decisions)
    secure_payload = capture_secure_audit_payload(
        "trace-1", request, serving, decisions, {}
    )
    assert secure_payload is not None
    audit_result = run_audit_reference_call(
        trace, secure_payload, "random", contract, PrefixBroker()
    )
    assert (
        approve_telemetry_evidence(
            trace,
            secure_payload,
            audit_result,
            None,
            audit_result.audit_record,
            None,
            None,
            None,
            None,
            definition.name,
            definition.contract_hash,
            contract,
            definition.runtime_config.telemetry_privacy_policy,
        )
        is not None
    )

    with pytest.raises(Exception, match="same request"):
        approve_telemetry_evidence(
            trace,
            secure_payload,
            replace(audit_result, reference_output_raw={"label": "b"}),
            None,
            audit_result.audit_record,
            None,
            None,
            None,
            None,
            definition.name,
            definition.contract_hash,
            contract,
            definition.runtime_config.telemetry_privacy_policy,
        )
    with pytest.raises(Exception, match="same request"):
        approve_telemetry_evidence(
            trace,
            secure_payload,
            replace(audit_result, reference_source="verified_l4"),
            None,
            audit_result.audit_record,
            None,
            None,
            None,
            None,
            definition.name,
            definition.contract_hash,
            contract,
            definition.runtime_config.telemetry_privacy_policy,
        )


def test_audit_evidence_rejects_audit_record_relabeling(target_dir: Path) -> None:
    definition, contract, _ = load_checked_target(target_dir)
    serving = ServingResult(
        "rel",
        "cascade",
        "ok",
        CacheMiss("k"),
        None,
        {"label": "a"},
        "L1",
        0.0,
        1.0,
        None,
        None,
        None,
    )
    request = RuntimeRequest("req", definition.name, {"text": "a:x"})
    decisions = combine_audit_decisions(AuditDecision("random", True, 1.0, []), None)
    trace = write_trace("trace-1", definition.contract_hash, contract, request, serving, decisions)
    secure_payload = capture_secure_audit_payload(
        "trace-1", request, serving, decisions, {}
    )
    assert secure_payload is not None
    audit_result = run_audit_reference_call(
        trace, secure_payload, "random", contract, PrefixBroker()
    )
    relabeled_record = replace(
        audit_result.audit_record, audit_id="risk-audit-1", audit_type="risk"
    )

    with pytest.raises(Exception, match="same request"):
        approve_telemetry_evidence(
            trace,
            secure_payload,
            audit_result,
            None,
            relabeled_record,
            None,
            None,
            None,
            None,
            definition.name,
            definition.contract_hash,
            contract,
            definition.runtime_config.telemetry_privacy_policy,
        )

    consistently_relabeled_record = replace(
        audit_result.audit_record, audit_id="risk-audit-2", audit_type="risk"
    )
    consistently_relabeled_result = replace(
        audit_result, audit_record=consistently_relabeled_record
    )
    with pytest.raises(Exception, match="same request"):
        approve_telemetry_evidence(
            trace,
            secure_payload,
            consistently_relabeled_result,
            None,
            consistently_relabeled_record,
            None,
            None,
            None,
            None,
            definition.name,
            definition.contract_hash,
            contract,
            definition.runtime_config.telemetry_privacy_policy,
        )


def test_audit_evidence_rejects_expired_raw_carriers(target_dir: Path) -> None:
    definition, contract, _ = load_checked_target(target_dir)
    serving = ServingResult(
        "rel",
        "cascade",
        "ok",
        CacheMiss("k"),
        None,
        {"label": "a"},
        "L1",
        0.0,
        1.0,
        None,
        None,
        None,
    )
    request = RuntimeRequest("req", definition.name, {"text": "a:x"})
    decisions = combine_audit_decisions(AuditDecision("random", True, 1.0, []), None)
    trace = write_trace("trace-1", definition.contract_hash, contract, request, serving, decisions)
    secure_payload = capture_secure_audit_payload(
        "trace-1", request, serving, decisions, {}
    )
    assert secure_payload is not None
    audit_result = run_audit_reference_call(
        trace, secure_payload, "random", contract, PrefixBroker()
    )
    expired_payload = replace(secure_payload, expires_at=utcnow() - timedelta(seconds=1))
    with pytest.raises(Exception, match="expired"):
        approve_telemetry_evidence(
            trace,
            expired_payload,
            audit_result,
            None,
            audit_result.audit_record,
            None,
            None,
            None,
            None,
            definition.name,
            definition.contract_hash,
            contract,
            definition.runtime_config.telemetry_privacy_policy,
        )
    expired_result = replace(audit_result, expires_at=utcnow() - timedelta(seconds=1))
    with pytest.raises(Exception, match="expired"):
        approve_telemetry_evidence(
            trace,
            secure_payload,
            expired_result,
            None,
            audit_result.audit_record,
            None,
            None,
            None,
            None,
            definition.name,
            definition.contract_hash,
            contract,
            definition.runtime_config.telemetry_privacy_policy,
        )


def test_telemetry_data_source_rejects_audit_row_scope_mismatch(
    target_dir: Path, tmp_path: Path
) -> None:
    definition, contract, _ = load_checked_target(target_dir)
    serving = ServingResult(
        "rel",
        "cascade",
        "ok",
        CacheMiss("k"),
        None,
        {"label": "a"},
        "L1",
        0.0,
        1.0,
        None,
        None,
        None,
    )
    request = RuntimeRequest("req", definition.name, {"text": "a:x"})
    decisions = combine_audit_decisions(AuditDecision("random", True, 1.0, []), None)
    trace = write_trace("trace-1", definition.contract_hash, contract, request, serving, decisions)
    secure_payload = capture_secure_audit_payload(
        "trace-1", request, serving, decisions, {}
    )
    assert secure_payload is not None
    audit_result = run_audit_reference_call(
        trace, secure_payload, "random", contract, PrefixBroker()
    )
    evidence = approve_telemetry_evidence(
        trace,
        secure_payload,
        audit_result,
        None,
        audit_result.audit_record,
        None,
        None,
        None,
        None,
        definition.name,
        definition.contract_hash,
        contract,
        definition.runtime_config.telemetry_privacy_policy,
    )
    assert evidence is not None
    rejected_evidence = replace(
        evidence,
        privacy_review=replace(evidence.privacy_review, decision="rejected"),
    )
    with pytest.raises(Exception, match="approved evidence"):
        build_telemetry_data_source(
            [trace],
            [audit_result.audit_record],
            [rejected_evidence],
            definition.name,
            definition.contract_hash,
            rejected_evidence.created_at + timedelta(seconds=1),
            contract,
            tmp_path,
        )
    runtime_evidence_without_release = replace(evidence, release_id=None)
    with pytest.raises(Exception, match="release provenance"):
        build_telemetry_data_source(
            [trace],
            [audit_result.audit_record],
            [runtime_evidence_without_release],
            definition.name,
            definition.contract_hash,
            runtime_evidence_without_release.created_at + timedelta(seconds=1),
            contract,
            tmp_path,
        )
    with pytest.raises(Exception, match="audit provenance"):
        build_telemetry_data_source(
            [trace],
            [],
            [evidence],
            definition.name,
            definition.contract_hash,
            evidence.created_at + timedelta(seconds=1),
            contract,
            tmp_path,
        )
    unrelated_bad_trace = replace(trace, trace_id="other-trace", target_name="other-target")
    with pytest.raises(Exception, match="trace provenance"):
        build_telemetry_data_source(
            [trace, unrelated_bad_trace],
            [audit_result.audit_record],
            [evidence],
            definition.name,
            definition.contract_hash,
            evidence.created_at + timedelta(seconds=1),
            contract,
            tmp_path,
        )
    unrelated_bad_audit = replace(
        audit_result.audit_record, trace_id="other-trace", target_name="other-target"
    )
    with pytest.raises(Exception, match="audit provenance"):
        build_telemetry_data_source(
            [trace],
            [audit_result.audit_record, unrelated_bad_audit],
            [evidence],
            definition.name,
            definition.contract_hash,
            evidence.created_at + timedelta(seconds=1),
            contract,
            tmp_path,
        )
    wrong_contract_audit = replace(
        audit_result.audit_record, contract_hash="wrong-contract"
    )

    with pytest.raises(Exception, match="audit provenance"):
        build_telemetry_data_source(
            [trace],
            [wrong_contract_audit],
            [evidence],
            definition.name,
            definition.contract_hash,
            evidence.created_at + timedelta(seconds=1),
            contract,
            tmp_path,
        )


def test_user_feedback_requires_privacy_approval_and_cutoff_filter(
    target_dir: Path, tmp_path: Path
) -> None:
    definition, contract, _ = load_checked_target(target_dir)
    feedback = UserFeedbackRecord(
        "fb1",
        definition.name,
        definition.contract_hash,
        {"text": "c:user"},
        {"label": "c"},
        utcnow(),
        "human_review",
        None,
        ["train", "validation_candidate"],
    )
    evidence = approve_telemetry_evidence(
        None,
        None,
        None,
        None,
        None,
        feedback,
        None,
        None,
        feedback.submitted_at,
        definition.name,
        definition.contract_hash,
        contract,
        definition.runtime_config.telemetry_privacy_policy,
    )
    assert evidence is not None
    source = build_telemetry_data_source(
        [],
        [],
        [evidence],
        definition.name,
        definition.contract_hash,
        evidence.created_at - timedelta(seconds=1),
        contract,
        tmp_path,
    )
    assert __import__("json").loads(Path(source.records_uri).read_text()) == []
    source = build_telemetry_data_source(
        [],
        [],
        [evidence],
        definition.name,
        definition.contract_hash,
        evidence.created_at + timedelta(seconds=1),
        contract,
        tmp_path,
    )
    assert len(__import__("json").loads(Path(source.records_uri).read_text())) == 1


def test_agent_visible_telemetry_summary_is_aggregate_and_scoped(target_dir: Path) -> None:
    definition, _, _ = load_checked_target(target_dir)
    window_start = utcnow()
    metrics = RuntimeMetricWindow(
        definition.name,
        "rel",
        definition.contract_hash,
        window_start,
        window_start + timedelta(minutes=5),
        10,
        0.6,
        {"L1": 6},
        {"L1": 5},
        {"L1": 0.5},
        [],
        0.1,
        0.3,
        None,
        None,
        None,
        None,
        None,
        0,
        0,
        0,
        0,
        0,
        0,
        None,
        {"low": 2, "high": 8},
        {"accept": 5, "fallback": 3},
        {"tenant": {"stable": 10}},
        0.0,
        {"p50_ms": 3.0},
        {"serving_cost_per_1000": 2.0},
        {"runtime_error": 0.0},
    )
    drift = DriftSignal(
        definition.name,
        "rel",
        definition.contract_hash,
        "watch",
        {"local_coverage_drop": 0.2},
        "report-1",
        utcnow(),
    )

    summary = export_agent_visible_telemetry_summary(
        metrics, drift, {"telemetry_source_id": "telemetry-1", "policy_version": "p1"}
    )

    assert summary.target_name == definition.name
    assert summary.release_id == "rel"
    assert summary.metrics_summary == {
        "request_count": 10,
        "local_coverage": 0.6,
        "l4_fallback_rate": 0.3,
        "reason_code_counts": {"accept": 5, "fallback": 3},
    }
    assert summary.drift_summary == {
        "status": "watch",
        "signals": {"local_coverage_drop": 0.2},
    }
    assert summary.telemetry_source_id == "telemetry-1"
    assert summary.redaction_policy_version == "p1"

    with pytest.raises(Exception, match="scope"):
        export_agent_visible_telemetry_summary(
            metrics,
            replace(drift, contract_hash="other-contract"),
            {"policy_version": "p1"},
        )


def test_recompile_request_scope_uses_current_release_even_for_cold_start(target_dir: Path) -> None:
    definition, _, _ = load_checked_target(target_dir)
    release = Release(
        "rel",
        definition.name,
        definition.contract_hash,
        None,
        None,
        None,
        None,
        utcnow(),
        {"L1": None, "L2": None, "L3": None},
        RoutingSettings(),
        None,
    )
    request = request_recompile(definition, release, RecompileReason("manual"), None, None)
    assert request.base_release_id == "rel"
    assert request.telemetry_source is None
    bad_release = __import__("dataclasses").replace(release, contract_hash="other")
    with pytest.raises(Exception, match="scope"):
        request_recompile(definition, bad_release, RecompileReason("manual"), None, None)
    with pytest.raises(Exception, match="requested_by"):
        request_recompile(
            definition,
            release,
            RecompileReason("manual"),
            None,
            None,
            requested_by="bot",
        )
