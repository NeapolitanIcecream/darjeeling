from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
from conftest import PrefixBroker

from darjeeling.audit_monitoring import (
    build_audit_summary,
    capture_secure_audit_payload,
    combine_audit_decisions,
    decide_asynchronous_risk_diagnostic,
    decide_random_audit,
    decide_synchronous_risk_audit,
    run_audit_reference_call,
)
from darjeeling.model import (
    AuditDecision,
    AuditDecisions,
    CacheMiss,
    Release,
    RoutingSettings,
    RuntimeRequest,
    ServingResult,
)
from darjeeling.runtime_trace_metrics import write_trace
from darjeeling.target_definition import load_checked_target
from darjeeling.util import utcnow


def test_secure_payload_must_match_trace_scope(target_dir: Path) -> None:
    definition, contract, _ = load_checked_target(target_dir)
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
    random_decision = decide_random_audit(
        release,
        RuntimeRequest("req", definition.name, {"text": "a:x"}),
        serving,
        {"random_audit_rate": 1.0},
    )
    decisions = combine_audit_decisions(random_decision, None)
    trace = write_trace(
        "trace-1",
        definition.contract_hash,
        contract,
        RuntimeRequest("req", definition.name, {"text": "a:x"}),
        serving,
        decisions,
    )
    store = {}
    payload = capture_secure_audit_payload(
        "trace-1",
        RuntimeRequest("req", definition.name, {"text": "a:x"}),
        serving,
        decisions,
        store,
    )
    assert payload is not None
    l4_serving = __import__("dataclasses").replace(serving, chosen_layer="L4")
    with pytest.raises(Exception, match="lower-layer output"):
        capture_secure_audit_payload(
            "trace-1",
            RuntimeRequest("req", definition.name, {"text": "a:x"}),
            l4_serving,
            decisions,
            {},
        )
    bad_payload = __import__("dataclasses").replace(payload, release_id="other")
    with pytest.raises(Exception, match="same request"):
        run_audit_reference_call(trace, bad_payload, "random", contract, PrefixBroker())
    bad_input_payload = __import__("dataclasses").replace(
        payload, input_raw={"text": "b:wrong"}
    )
    with pytest.raises(Exception, match="same request"):
        run_audit_reference_call(trace, bad_input_payload, "random", contract, PrefixBroker())
    bad_output_payload = __import__("dataclasses").replace(
        payload, lower_layer_output_raw={"label": "wrong"}
    )
    with pytest.raises(Exception, match="same request"):
        run_audit_reference_call(trace, bad_output_payload, "random", contract, PrefixBroker())
    expired_payload = __import__("dataclasses").replace(
        payload, expires_at=utcnow() - timedelta(seconds=1)
    )
    with pytest.raises(Exception, match="expired"):
        run_audit_reference_call(trace, expired_payload, "random", contract, PrefixBroker())
    bad_contract = __import__("dataclasses").replace(contract)
    object.__setattr__(bad_contract, "contract_hash", "other-contract")
    with pytest.raises(Exception, match="contract scope"):
        run_audit_reference_call(trace, payload, "random", bad_contract, PrefixBroker(fail=True))


def test_risk_audit_requires_lower_layer_accepted_output(target_dir: Path) -> None:
    definition, _contract, _ = load_checked_target(target_dir)
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
    request = RuntimeRequest("req", definition.name, {"text": "a:x"}, metadata={"risk": "yes"})
    risk_rules = {"flags": [{"code": "risk_case", "metadata_key": "risk", "equals": "yes"}]}
    l4_serving = ServingResult(
        "rel",
        "cascade",
        "ok",
        CacheMiss("k"),
        None,
        {"label": "a"},
        "L4",
        0.01,
        1.0,
        None,
        None,
        None,
    )
    assert not decide_synchronous_risk_audit(
        release, request, l4_serving, risk_rules
    ).selected
    local_serving = __import__("dataclasses").replace(l4_serving, chosen_layer="L1")
    decision = decide_synchronous_risk_audit(release, request, local_serving, risk_rules)
    assert decision.selected
    assert decision.risk_flags == ["risk_case"]


def test_audit_reference_requires_selected_audit_type(target_dir: Path) -> None:
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
    payload = capture_secure_audit_payload("trace-1", request, serving, decisions, {})
    assert payload is not None

    with pytest.raises(Exception, match="selected"):
        run_audit_reference_call(trace, payload, "risk", contract, PrefixBroker(fail=True))


def test_risk_audit_reference_requires_trace_safe_risk_flags(target_dir: Path) -> None:
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
    decisions = AuditDecisions(None, None, ["risk"], 0.0, [])
    trace = write_trace("trace-1", definition.contract_hash, contract, request, serving, decisions)
    payload = capture_secure_audit_payload("trace-1", request, serving, decisions, {})
    assert payload is not None
    with pytest.raises(Exception, match="risk flags"):
        run_audit_reference_call(trace, payload, "risk", contract, PrefixBroker(fail=True))

    bad_flag_trace = __import__("dataclasses").replace(
        trace, risk_audit_flags=["not safe"]
    )
    with pytest.raises(Exception, match="risk flags"):
        run_audit_reference_call(
            bad_flag_trace, payload, "risk", contract, PrefixBroker(fail=True)
        )


def test_risk_audit_summary_counts_selected_risk_flags(target_dir: Path) -> None:
    definition, contract, _ = load_checked_target(target_dir)
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
    request = RuntimeRequest("req", definition.name, {"text": "a:x"}, metadata={"risk": "yes"})
    serving = ServingResult(
        "rel",
        "cascade",
        "ok",
        CacheMiss("k"),
        None,
        {"label": "wrong"},
        "L1",
        0.0,
        1.0,
        None,
        None,
        None,
    )
    risk_decision = decide_synchronous_risk_audit(
        release,
        request,
        serving,
        {"flags": [{"code": "risk_case", "metadata_key": "risk", "equals": "yes"}]},
    )
    decisions = combine_audit_decisions(None, risk_decision)
    trace = write_trace("trace-1", definition.contract_hash, contract, request, serving, decisions)
    store = {}
    payload = capture_secure_audit_payload("trace-1", request, serving, decisions, store)
    assert payload is not None
    assert "trace-1" in store
    result = run_audit_reference_call(
        trace, payload, "risk", contract, PrefixBroker(), secure_store=store
    )
    summary = build_audit_summary(
        [result.audit_record],
        (utcnow() - timedelta(minutes=1), utcnow() + timedelta(minutes=1)),
        definition.name,
        "rel",
        definition.contract_hash,
    )
    assert result.audit_record.is_correct is False
    assert result.audit_record.risk_flags == ["risk_case"]
    assert summary.risk_findings_by_flag == {"risk_case": 1}
    assert store == {}


def test_audit_reference_failures_are_recorded_and_counted(target_dir: Path) -> None:
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
    decisions = combine_audit_decisions(AuditDecision("random", True, 1.0, []), None)
    trace = write_trace(
        "trace-1",
        definition.contract_hash,
        contract,
        RuntimeRequest("req", definition.name, {"text": "a:x"}),
        serving,
        decisions,
    )
    store = {}
    payload = capture_secure_audit_payload(
        "trace-1",
        RuntimeRequest("req", definition.name, {"text": "a:x"}),
        serving,
        decisions,
        store,
    )
    assert payload is not None
    assert "trace-1" in store
    result = run_audit_reference_call(
        trace, payload, "random", contract, PrefixBroker(fail=True), secure_store=store
    )
    summary = build_audit_summary(
        [result.audit_record],
        (utcnow() - timedelta(minutes=1), utcnow() + timedelta(minutes=1)),
        definition.name,
        "rel",
        definition.contract_hash,
    )
    assert result.status == "error"
    assert summary.random_reference_failure_count == 1
    assert result.audit_record.reference_output_redacted is None
    assert store == {}


def test_asynchronous_risk_diagnostic_is_trace_safe(target_dir: Path) -> None:
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
    trace = write_trace(
        "trace-1",
        definition.contract_hash,
        contract,
        RuntimeRequest("req", definition.name, {"text": "a:x"}),
        serving,
        AuditDecisions(None, None, ["risk"], 0.0, ["risk_case"]),
    )

    decision = decide_asynchronous_risk_diagnostic(
        trace, {"offline_flags": ["risk_case"]}
    )
    assert decision.selected
    assert decision.risk_flags == ["risk_case"]
    assert decision.claim_original_output_comparison is False
    assert "release_replay" in decision.allowed_next_actions

    unselected = decide_asynchronous_risk_diagnostic(trace, {"offline_flags": ["other_case"]})
    assert not unselected.selected
    assert unselected.allowed_next_actions == []

    with pytest.raises(Exception, match="trace-safe"):
        decide_asynchronous_risk_diagnostic(trace, {"offline_flags": ["not safe"]})
