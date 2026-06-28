from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml
from conftest import PrefixBroker, write_artifact

from darjeeling.artifact_worker import freeze_artifact_package, read_artifact_manifest
from darjeeling.errors import ReleaseError
from darjeeling.model import (
    ApprovalRecord,
    AuditDecision,
    AuditDecisions,
    PreparedReleaseWorkers,
    ReferenceContext,
    ReferenceResponse,
    Release,
    ReleaseRegistry,
    ResultCache,
    RoutingSettings,
    RuntimeContext,
    RuntimeRequest,
    TargetCheckReport,
    WorkerPool,
)
from darjeeling.release_runtime import (
    create_release_without_artifacts,
    load_release,
    prepare_workers,
    retire_release,
    rollback_release,
    run_cascade,
    run_shadow_request,
    select_release_for_request,
    serve_request,
    set_channel,
)
from darjeeling.runtime_trace_metrics import write_trace
from darjeeling.target_definition import load_checked_target


class SlowBroker:
    reference_version = "slow-v1"

    def __init__(self, sleep_seconds: float):
        self.sleep_seconds = sleep_seconds

    def call(self, request: dict[str, Any], context: ReferenceContext) -> ReferenceResponse:
        time.sleep(self.sleep_seconds)
        text = request["input"]["text"]
        return ReferenceResponse(
            payload={"label": text.split(":", 1)[0]},
            reference_source="versioned_l4",
            reference_version=self.reference_version,
            cost=0.01,
            latency_ms=self.sleep_seconds * 1000,
        )


class CancellableBroker:
    reference_version = "cancellable-v1"

    def __init__(self):
        self.cancel_seen = False
        self.late_mutations: list[dict[str, Any]] = []

    def call(self, request: dict[str, Any], context: ReferenceContext) -> ReferenceResponse:
        cancel_event = context.metadata["cancel_event"]
        deadline = context.metadata["deadline_at_monotonic"]
        while time.monotonic() < deadline + 0.05:
            if cancel_event.is_set():
                self.cancel_seen = True
                return ReferenceResponse(
                    payload={"label": request["input"]["text"].split(":", 1)[0]},
                    reference_source="versioned_l4",
                    reference_version=self.reference_version,
                    finish_status="cancelled",
                    latency_ms=0.0,
                )
            time.sleep(0.002)
        self.late_mutations.append(dict(request["input"]))
        return ReferenceResponse(
            payload={"label": request["input"]["text"].split(":", 1)[0]},
            reference_source="versioned_l4",
            reference_version=self.reference_version,
            latency_ms=50.0,
        )


class VersionedOutputBroker:
    def __init__(self, reference_version: str):
        self.reference_version = reference_version

    def call(self, request: dict[str, Any], context: ReferenceContext) -> ReferenceResponse:
        return ReferenceResponse(
            payload={"label": self.reference_version},
            reference_source="versioned_l4",
            reference_version=self.reference_version,
            cost=0.01,
            latency_ms=1.0,
        )


def _release_with_l1(
    definition, package, routing: RoutingSettings | None = None
) -> Release:
    return Release(
        "rel",
        definition.name,
        definition.contract_hash,
        "candidate-rel",
        "snapshot-rel",
        "snapshot-digest",
        "report-rel",
        __import__("darjeeling.util").util.utcnow(),
        {"L1": package, "L2": None, "L3": None},
        routing or RoutingSettings(enabled_layers=["L1"]),
        ApprovalRecord(
            "approval-rel",
            "candidate-rel",
            "report-rel",
            definition.name,
            definition.contract_hash,
            "snapshot-rel",
            __import__("darjeeling.util").util.utcnow(),
            "user",
        ),
    )


def test_release_without_artifacts_routes_cache_miss_to_l4(target_dir: Path) -> None:
    definition, contract, check = load_checked_target(target_dir)
    registry = ReleaseRegistry()
    release = create_release_without_artifacts(
        definition, contract, check, PrefixBroker(), RoutingSettings(), registry
    )
    result = run_cascade(
        release,
        PreparedReleaseWorkers({}),
        contract,
        {"text": "a:x"},
        RuntimeContext("r1", 1000, PrefixBroker()),
    )
    assert result.chosen_layer == "L4"
    assert result.output == {"label": "a"}


def test_disabled_layer_is_skipped_even_when_artifact_exists(
    target_dir: Path, tmp_path: Path
) -> None:
    definition, contract, check = load_checked_target(target_dir)
    artifact_dir = write_artifact(
        tmp_path / "artifact", definition.contract_hash, accept_prefixes=["a"]
    )
    package = freeze_artifact_package(
        artifact_dir, read_artifact_manifest(artifact_dir), tmp_path / "store", "snapshot-digest"
    )
    release = Release(
        "rel",
        definition.name,
        definition.contract_hash,
        None,
        None,
        None,
        None,
        __import__("darjeeling.util").util.utcnow(),
        {"L1": package, "L2": None, "L3": None},
        RoutingSettings(enabled_layers=[]),
        None,
    )
    workers = prepare_workers(release, WorkerPool())
    result = run_cascade(
        release, workers, contract, {"text": "a:x"}, RuntimeContext("r1", 1000, PrefixBroker())
    )
    assert result.attempts == []
    assert result.chosen_layer == "L4"


def test_prepare_workers_rejects_package_digest_mismatch(
    target_dir: Path, tmp_path: Path
) -> None:
    definition, _, _ = load_checked_target(target_dir)
    artifact_dir = write_artifact(tmp_path / "artifact", definition.contract_hash)
    package = freeze_artifact_package(
        artifact_dir, read_artifact_manifest(artifact_dir), tmp_path / "store", "snapshot-digest"
    )
    (package.package_path / "worker.py").write_text("print('changed')\n", encoding="utf-8")
    release = _release_with_l1(definition, package)
    with pytest.raises(ReleaseError, match="digest mismatch"):
        prepare_workers(release, WorkerPool())


def test_prepare_workers_runs_healthcheck_before_serving(
    target_dir: Path, tmp_path: Path
) -> None:
    definition, _, _ = load_checked_target(target_dir)
    artifact_dir = write_artifact(tmp_path / "artifact", definition.contract_hash)
    (artifact_dir / "healthcheck.py").write_text("raise SystemExit(1)\n", encoding="utf-8")
    manifest_path = artifact_dir / "artifact.yaml"
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    data["healthcheck_command"] = ["python3", "healthcheck.py"]
    manifest_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    package = freeze_artifact_package(
        artifact_dir, read_artifact_manifest(artifact_dir), tmp_path / "store", "snapshot-digest"
    )
    release = _release_with_l1(definition, package)
    with pytest.raises(ReleaseError, match="failed to prepare L1 worker"):
        prepare_workers(release, WorkerPool())


def test_prepare_workers_requires_healthcheck_for_warmup(
    target_dir: Path, tmp_path: Path
) -> None:
    definition, _, _ = load_checked_target(target_dir)
    artifact_dir = write_artifact(tmp_path / "artifact", definition.contract_hash)
    manifest_path = artifact_dir / "artifact.yaml"
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    data["healthcheck_command"] = None
    data["start_command"] = ["missing-worker-bin", "worker.py"]
    manifest_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    package = freeze_artifact_package(
        artifact_dir, read_artifact_manifest(artifact_dir), tmp_path / "store", "snapshot-digest"
    )
    release = _release_with_l1(definition, package)
    with pytest.raises(ReleaseError, match="healthcheck"):
        prepare_workers(release, WorkerPool())


def test_run_cascade_propagates_runtime_deadline_to_worker(
    target_dir: Path, tmp_path: Path
) -> None:
    definition, contract, _ = load_checked_target(target_dir)
    artifact_dir = write_artifact(tmp_path / "artifact", definition.contract_hash)
    (artifact_dir / "worker.py").write_text(
        """
import json
import sys
import time

request = json.loads(sys.stdin.readline())
time.sleep(0.2)
print(json.dumps({"decision": "accept", "output": {"label": "a"}, "reason": "prefix_match"}))
""".lstrip(),
        encoding="utf-8",
    )
    package = freeze_artifact_package(
        artifact_dir, read_artifact_manifest(artifact_dir), tmp_path / "store", "snapshot-digest"
    )
    release = _release_with_l1(
        definition, package, RoutingSettings(enabled_layers=["L1"], L1_timeout_ms=50)
    )
    workers = prepare_workers(release, WorkerPool())
    result = run_cascade(
        release,
        workers,
        contract,
        {"text": "a:x"},
        RuntimeContext("r1", 500, PrefixBroker()),
    )
    assert result.attempts[0].decision == "timeout"
    assert result.chosen_layer == "L4"


def test_run_cascade_enforces_total_deadline_before_l4(
    target_dir: Path, tmp_path: Path
) -> None:
    definition, contract, _ = load_checked_target(target_dir)
    artifact_dir = write_artifact(tmp_path / "artifact", definition.contract_hash)
    package = freeze_artifact_package(
        artifact_dir, read_artifact_manifest(artifact_dir), tmp_path / "store", "snapshot-digest"
    )
    release = _release_with_l1(definition, package)
    workers = prepare_workers(release, WorkerPool())

    result = run_cascade(
        release,
        workers,
        contract,
        {"text": "a:x"},
        RuntimeContext("r1", 0, PrefixBroker()),
    )

    assert result.status == "error"
    assert result.chosen_layer is None
    assert result.error_type == "deadline_exceeded"
    assert result.l4_fallback_result is None


def test_l4_fallback_failure_returns_public_error_and_trace(target_dir: Path) -> None:
    definition, contract, check = load_checked_target(target_dir)
    registry = ReleaseRegistry()
    release = create_release_without_artifacts(
        definition, contract, check, PrefixBroker(), RoutingSettings(), registry
    )
    set_channel(definition.name, "stable", release.release_id, {}, registry)
    traces = []
    response = serve_request(
        RuntimeRequest("req1", definition.name, {"text": "a:x"}),
        registry,
        lambda _name: contract,
        WorkerPool(),
        lambda: "trace-1",
        lambda *args: traces.append(write_trace(*args)),
        None,
        {},
        {},
        [],
        definition.runtime_config.telemetry_privacy_policy,
        [],
        PrefixBroker(fail=True),
        ResultCache(),
    )
    assert response.status == "error"
    assert response.public_error_message is not None
    assert traces[0].error_type == "l4_fallback_failure"
    assert "provider secret" not in repr(traces[0])


def test_serve_request_deadline_exceeded_writes_safe_trace(target_dir: Path) -> None:
    definition, contract, check = load_checked_target(target_dir)
    registry = ReleaseRegistry()
    release = create_release_without_artifacts(
        definition, contract, check, PrefixBroker(), RoutingSettings(), registry
    )
    set_channel(definition.name, "stable", release.release_id, {}, registry)
    traces = []

    response = serve_request(
        RuntimeRequest("req1", definition.name, {"text": "a:x"}, deadline_ms=0),
        registry,
        lambda _name: contract,
        WorkerPool(),
        lambda: "trace-1",
        lambda *args: traces.append(write_trace(*args)),
        None,
        {},
        {},
        [],
        definition.runtime_config.telemetry_privacy_policy,
        [],
        PrefixBroker(),
        ResultCache(),
    )

    assert response.status == "error"
    assert response.error_type == "deadline_exceeded"
    assert response.chosen_layer is None
    assert traces[0].error_type == "deadline_exceeded"


def test_slow_l4_fallback_respects_remaining_deadline(target_dir: Path) -> None:
    definition, contract, check = load_checked_target(target_dir)
    registry = ReleaseRegistry()
    release = create_release_without_artifacts(
        definition, contract, check, PrefixBroker(), RoutingSettings(), registry
    )
    set_channel(definition.name, "stable", release.release_id, {}, registry)
    traces = []

    response = serve_request(
        RuntimeRequest("req1", definition.name, {"text": "a:x"}, deadline_ms=10),
        registry,
        lambda _name: contract,
        WorkerPool(),
        lambda: "trace-1",
        lambda *args: traces.append(write_trace(*args)),
        None,
        {},
        {},
        [],
        definition.runtime_config.telemetry_privacy_policy,
        [],
        SlowBroker(0.05),
        ResultCache(),
    )

    assert response.status == "error"
    assert response.error_type == "deadline_exceeded"
    assert response.output is None
    assert traces[0].error_type == "deadline_exceeded"


def test_request_deadline_is_capped_by_release_total_deadline(target_dir: Path) -> None:
    definition, contract, check = load_checked_target(target_dir)
    registry = ReleaseRegistry()
    release = create_release_without_artifacts(
        definition,
        contract,
        check,
        PrefixBroker(),
        RoutingSettings(total_deadline_ms=10),
        registry,
    )
    set_channel(definition.name, "stable", release.release_id, {}, registry)
    traces = []

    response = serve_request(
        RuntimeRequest("req1", definition.name, {"text": "a:x"}, deadline_ms=1000),
        registry,
        lambda _name: contract,
        WorkerPool(),
        lambda: "trace-1",
        lambda *args: traces.append(write_trace(*args)),
        None,
        {},
        {},
        [],
        definition.runtime_config.telemetry_privacy_policy,
        [],
        SlowBroker(0.05),
        ResultCache(),
    )

    assert response.status == "error"
    assert response.error_type == "deadline_exceeded"
    assert traces[0].error_type == "deadline_exceeded"


def test_l4_timeout_sets_broker_cancellation_context(target_dir: Path) -> None:
    definition, contract, check = load_checked_target(target_dir)
    registry = ReleaseRegistry()
    release = create_release_without_artifacts(
        definition, contract, check, PrefixBroker(), RoutingSettings(), registry
    )
    set_channel(definition.name, "stable", release.release_id, {}, registry)
    broker = CancellableBroker()

    response = serve_request(
        RuntimeRequest("req1", definition.name, {"text": "a:x"}, deadline_ms=10),
        registry,
        lambda _name: contract,
        WorkerPool(),
        lambda: "trace-1",
        lambda *args: write_trace(*args),
        None,
        {},
        {},
        [],
        definition.runtime_config.telemetry_privacy_policy,
        [],
        broker,
        ResultCache(),
    )
    time.sleep(0.08)

    assert response.status == "error"
    assert response.error_type == "deadline_exceeded"
    assert broker.cancel_seen
    assert broker.late_mutations == []


def test_serve_request_rejects_loaded_contract_hash_mismatch(target_dir: Path) -> None:
    definition, contract, check = load_checked_target(target_dir)
    registry = ReleaseRegistry()
    release = create_release_without_artifacts(
        definition, contract, check, PrefixBroker(), RoutingSettings(), registry
    )
    set_channel(definition.name, "stable", release.release_id, {}, registry)
    bad_contract = replace(contract)
    object.__setattr__(bad_contract, "contract_hash", "other-contract")
    traces = []

    response = serve_request(
        RuntimeRequest("req1", definition.name, {"text": "a:x"}),
        registry,
        lambda _name: bad_contract,
        WorkerPool(),
        lambda: "trace-1",
        lambda *args: traces.append(write_trace(*args)),
        None,
        {},
        {},
        [],
        definition.runtime_config.telemetry_privacy_policy,
        [],
        PrefixBroker(),
        ResultCache(),
    )

    assert response.status == "error"
    assert response.output is None
    assert traces[0].error_type == "runtime_error"


def test_serve_request_writes_l4_evidence_without_l4_only_risk_audit_payload(
    target_dir: Path,
) -> None:
    definition, contract, check = load_checked_target(target_dir)
    registry = ReleaseRegistry()
    release = create_release_without_artifacts(
        definition, contract, check, PrefixBroker(), RoutingSettings(), registry
    )
    set_channel(definition.name, "stable", release.release_id, {}, registry)
    traces = []
    secure_store = {}
    audit_queue = []
    evidence_store = []

    def record_trace(*args):
        trace = write_trace(*args)
        traces.append(trace)
        return trace

    response = serve_request(
        RuntimeRequest("req1", definition.name, {"text": "a:x"}, metadata={"risk": "yes"}),
        registry,
        lambda _name: contract,
        WorkerPool(),
        lambda: "trace-1",
        record_trace,
        None,
        {"flags": [{"code": "risk_case", "metadata_key": "risk", "equals": "yes"}]},
        secure_store,
        audit_queue,
        definition.runtime_config.telemetry_privacy_policy,
        evidence_store,
        PrefixBroker(),
        ResultCache(),
    )

    assert response.status == "ok"
    assert response.chosen_layer == "L4"
    assert len(evidence_store) == 1
    assert evidence_store[0].source == "l4_fallback"
    assert secure_store == {}
    assert audit_queue == []
    assert traces[0].risk_audit_flags == []


def test_l4_evidence_requires_trace_writer_to_return_trace(target_dir: Path) -> None:
    definition, contract, check = load_checked_target(target_dir)
    registry = ReleaseRegistry()
    release = create_release_without_artifacts(
        definition, contract, check, PrefixBroker(), RoutingSettings(), registry
    )
    set_channel(definition.name, "stable", release.release_id, {}, registry)
    traces = []
    evidence_store = []

    response = serve_request(
        RuntimeRequest("req1", definition.name, {"text": "a:x"}),
        registry,
        lambda _name: contract,
        WorkerPool(),
        lambda: "trace-1",
        lambda *args: traces.append(write_trace(*args)),
        None,
        {},
        {},
        [],
        definition.runtime_config.telemetry_privacy_policy,
        evidence_store,
        PrefixBroker(),
        ResultCache(),
    )

    assert response.status == "error"
    assert response.error_type == "runtime_error"
    assert evidence_store == []
    assert traces[0].chosen_layer == "L4"
    assert traces[-1].error_type == "runtime_error"


def test_serve_request_does_not_enqueue_audit_without_persisted_secure_payload(
    target_dir: Path,
) -> None:
    definition, contract, check = load_checked_target(target_dir)
    registry = ReleaseRegistry()
    release = create_release_without_artifacts(
        definition, contract, check, PrefixBroker(), RoutingSettings(), registry
    )
    set_channel(definition.name, "stable", release.release_id, {}, registry)
    traces = []
    audit_queue = []

    def selected_random_audit(*_args):
        return AuditDecisions(
            AuditDecision("random", True, 1.0, []), None, ["random"], 1.0, []
        )

    def record_trace(*args):
        trace = write_trace(*args)
        traces.append(trace)
        return trace

    response = serve_request(
        RuntimeRequest("req1", definition.name, {"text": "a:x"}),
        registry,
        lambda _name: contract,
        WorkerPool(),
        lambda: "trace-1",
        record_trace,
        selected_random_audit,
        {},
        [],
        audit_queue,
        definition.runtime_config.telemetry_privacy_policy,
        [],
        PrefixBroker(),
        ResultCache(),
    )

    assert response.status == "ok"
    assert audit_queue == []
    assert traces[0].selected_audit_types == []


def test_result_cache_key_includes_reference_version(target_dir: Path) -> None:
    definition, contract, check = load_checked_target(target_dir)
    registry = ReleaseRegistry()
    release = create_release_without_artifacts(
        definition,
        contract,
        check,
        PrefixBroker(),
        RoutingSettings(cache_enabled=True),
        registry,
    )
    set_channel(definition.name, "stable", release.release_id, {}, registry)
    cache = ResultCache()

    first = serve_request(
        RuntimeRequest("req1", definition.name, {"text": "a:x"}),
        registry,
        lambda _name: contract,
        WorkerPool(),
        lambda: "trace-1",
        lambda *args: write_trace(*args),
        None,
        {},
        {},
        [],
        definition.runtime_config.telemetry_privacy_policy,
        [],
        VersionedOutputBroker("reference-v1"),
        cache,
    )
    second = serve_request(
        RuntimeRequest("req2", definition.name, {"text": "a:x"}),
        registry,
        lambda _name: contract,
        WorkerPool(),
        lambda: "trace-2",
        lambda *args: write_trace(*args),
        None,
        {},
        {},
        [],
        definition.runtime_config.telemetry_privacy_policy,
        [],
        VersionedOutputBroker("reference-v2"),
        cache,
    )

    assert first.output == {"label": "reference-v1"}
    assert second.output == {"label": "reference-v2"}
    assert len(cache.entries) == 2


def test_canary_channel_uses_stable_hash_routing(target_dir: Path) -> None:
    definition, contract, check = load_checked_target(target_dir)
    registry = ReleaseRegistry()
    stable = create_release_without_artifacts(
        definition, contract, check, PrefixBroker(), RoutingSettings(), registry
    )
    canary = create_release_without_artifacts(
        definition, contract, check, PrefixBroker(), RoutingSettings(), registry
    )
    set_channel(definition.name, "stable", stable.release_id, {}, registry)
    set_channel(definition.name, "canary", canary.release_id, {"traffic_fraction": 1.0}, registry)

    selected = select_release_for_request(
        definition.name,
        RuntimeRequest("req1", definition.name, {"text": "a:x"}, tenant_key="tenant-a"),
        registry,
    )

    assert selected.release_id == canary.release_id


def test_select_release_requires_active_stable_channel(target_dir: Path) -> None:
    definition, contract, check = load_checked_target(target_dir)
    registry = ReleaseRegistry()
    stable = create_release_without_artifacts(
        definition, contract, check, PrefixBroker(), RoutingSettings(), registry
    )
    canary = create_release_without_artifacts(
        definition, contract, check, PrefixBroker(), RoutingSettings(), registry
    )

    with pytest.raises(ReleaseError, match="no stable release channel"):
        select_release_for_request(
            definition.name,
            RuntimeRequest("req1", definition.name, {"text": "a:x"}),
            registry,
        )

    set_channel(definition.name, "canary", canary.release_id, {}, registry)
    with pytest.raises(ReleaseError, match="no stable release channel"):
        select_release_for_request(
            definition.name,
            RuntimeRequest("req2", definition.name, {"text": "a:x"}),
            registry,
        )

    set_channel(definition.name, "stable", stable.release_id, {}, registry)
    retire_release(stable.release_id, registry)
    with pytest.raises(ReleaseError, match="stable channel release is not active"):
        select_release_for_request(
            definition.name,
            RuntimeRequest("req3", definition.name, {"text": "a:x"}),
            registry,
        )


def test_set_channel_rejects_incompatible_active_contract(target_dir: Path) -> None:
    definition, contract, check = load_checked_target(target_dir)
    registry = ReleaseRegistry()
    stable = create_release_without_artifacts(
        definition, contract, check, PrefixBroker(), RoutingSettings(), registry
    )
    incompatible = replace(stable, release_id="rel-other-contract", contract_hash="other-contract")
    registry.releases[incompatible.release_id] = incompatible
    set_channel(definition.name, "stable", stable.release_id, {}, registry)

    with pytest.raises(ReleaseError, match="contract compatibility"):
        set_channel(definition.name, "canary", incompatible.release_id, {}, registry)


def test_rollback_rejects_retired_previous_stable(target_dir: Path) -> None:
    definition, contract, check = load_checked_target(target_dir)
    registry = ReleaseRegistry()
    previous = create_release_without_artifacts(
        definition, contract, check, PrefixBroker(), RoutingSettings(), registry
    )
    current = create_release_without_artifacts(
        definition, contract, check, PrefixBroker(), RoutingSettings(), registry
    )
    set_channel(definition.name, "stable", previous.release_id, {}, registry)
    set_channel(definition.name, "stable", current.release_id, {}, registry)
    retire_release(previous.release_id, registry)

    with pytest.raises(ReleaseError, match="previous stable release is not eligible"):
        rollback_release(definition.name, registry, {})
    assert registry.channels[(definition.name, "stable")] == current.release_id


def test_cold_start_release_requires_matching_target_check_and_usable_broker(
    target_dir: Path,
) -> None:
    definition, contract, check = load_checked_target(target_dir)
    registry = ReleaseRegistry()

    with pytest.raises(ReleaseError, match="target check scope"):
        create_release_without_artifacts(
            definition,
            contract,
            TargetCheckReport("other-target", definition.contract_hash, "pass"),
            PrefixBroker(),
            RoutingSettings(),
            registry,
        )

    with pytest.raises(ReleaseError, match="L4 fallback"):
        create_release_without_artifacts(
            definition,
            contract,
            check,
            PrefixBroker(fail=True),
            RoutingSettings(),
            registry,
        )


def test_cold_start_l4_probe_uses_target_owned_sample_input(target_dir: Path) -> None:
    contract_path = target_dir / "contract.py"
    contract_path.write_text(
        contract_path.read_text(encoding="utf-8").replace(
            "def validate_input(value):\n    return dict(value)\n",
            """
def validate_input(value):
    result = dict(value)
    if ":" not in result["text"]:
        raise ValueError("text must include target-owned separator")
    return result
""".lstrip(),
        ),
        encoding="utf-8",
    )
    definition, contract, check = load_checked_target(target_dir)
    registry = ReleaseRegistry()

    release = create_release_without_artifacts(
        definition, contract, check, PrefixBroker(), RoutingSettings(), registry
    )

    assert release.release_id in registry.releases


def test_load_release_rejects_tampered_artifact_package(
    target_dir: Path, tmp_path: Path
) -> None:
    definition, _, _ = load_checked_target(target_dir)
    artifact_dir = write_artifact(tmp_path / "artifact", definition.contract_hash)
    package = freeze_artifact_package(
        artifact_dir, read_artifact_manifest(artifact_dir), tmp_path / "store", "snapshot-digest"
    )
    release = _release_with_l1(definition, package)
    registry = ReleaseRegistry(releases={release.release_id: release})
    (package.package_path / "worker.py").write_text("print('changed')\n", encoding="utf-8")

    with pytest.raises(ReleaseError, match="digest mismatch"):
        load_release(release.release_id, registry)


def test_load_release_rejects_partial_and_unapproved_artifact_provenance(
    target_dir: Path, tmp_path: Path
) -> None:
    definition, _, _ = load_checked_target(target_dir)
    artifact_dir = write_artifact(tmp_path / "artifact", definition.contract_hash)
    package = freeze_artifact_package(
        artifact_dir, read_artifact_manifest(artifact_dir), tmp_path / "store", "snapshot-digest"
    )
    approved = _release_with_l1(definition, package)
    partial = replace(approved, release_id="rel-partial", report_id=None)
    unapproved_artifact = Release(
        "rel-unapproved-artifact",
        definition.name,
        definition.contract_hash,
        None,
        None,
        None,
        None,
        __import__("darjeeling.util").util.utcnow(),
        {"L1": package, "L2": None, "L3": None},
        RoutingSettings(enabled_layers=["L1"]),
        None,
    )
    registry = ReleaseRegistry(
        releases={
            partial.release_id: partial,
            unapproved_artifact.release_id: unapproved_artifact,
        }
    )

    with pytest.raises(ReleaseError, match="release atomicity"):
        load_release(partial.release_id, registry)
    with pytest.raises(ReleaseError, match="release atomicity"):
        load_release(unapproved_artifact.release_id, registry)


def test_serving_worker_prepare_failure_falls_back_to_l4(
    target_dir: Path, tmp_path: Path
) -> None:
    definition, contract, _check = load_checked_target(target_dir)
    artifact_dir = write_artifact(tmp_path / "artifact", definition.contract_hash)
    (artifact_dir / "healthcheck.py").write_text("raise SystemExit(1)\n", encoding="utf-8")
    package = freeze_artifact_package(
        artifact_dir, read_artifact_manifest(artifact_dir), tmp_path / "store", "snapshot-digest"
    )
    release = _release_with_l1(definition, package)
    registry = ReleaseRegistry(releases={release.release_id: release})
    set_channel(definition.name, "stable", release.release_id, {}, registry)
    traces = []

    def record_trace(*args):
        trace = write_trace(*args)
        traces.append(trace)
        return trace

    response = serve_request(
        RuntimeRequest("req1", definition.name, {"text": "a:x"}),
        registry,
        lambda _name: contract,
        WorkerPool(),
        lambda: "trace-1",
        record_trace,
        None,
        {},
        {},
        [],
        definition.runtime_config.telemetry_privacy_policy,
        [],
        PrefixBroker(),
        ResultCache(),
    )

    assert response.status == "ok"
    assert response.chosen_layer == "L4"
    assert traces[0].chosen_layer == "L4"
    assert traces[0].attempts[0].decision == "error"
    assert traces[0].attempts[0].reason_code == "health_failure"
    assert traces[0].attempts[0].error_type == "health_failure"


def test_circuit_breaker_temporarily_skips_failing_layer(
    target_dir: Path, tmp_path: Path
) -> None:
    definition, contract, _check = load_checked_target(target_dir)
    artifact_dir = write_artifact(tmp_path / "artifact", definition.contract_hash, bad_output=True)
    package = freeze_artifact_package(
        artifact_dir, read_artifact_manifest(artifact_dir), tmp_path / "store", "snapshot-digest"
    )
    release = _release_with_l1(
        definition,
        package,
        RoutingSettings(
            enabled_layers=["L1"],
            circuit_breaker={"failure_threshold": 1, "cooldown_requests": 2},
        ),
    )
    registry = ReleaseRegistry(releases={release.release_id: release})
    set_channel(definition.name, "stable", release.release_id, {}, registry)
    traces = []

    def record_trace(*args):
        trace = write_trace(*args)
        traces.append(trace)
        return trace

    for index in range(2):
        response = serve_request(
            RuntimeRequest(f"req{index}", definition.name, {"text": "a:x"}),
            registry,
            lambda _name: contract,
            WorkerPool(),
            lambda index=index: f"trace-{index}",
            record_trace,
            None,
            {},
            {},
            [],
            definition.runtime_config.telemetry_privacy_policy,
            [],
            PrefixBroker(),
            ResultCache(),
        )
        assert response.status == "ok"
        assert response.chosen_layer == "L4"

    assert traces[0].attempts[0].decision == "invalid_output"
    assert traces[1].attempts[0].decision == "error"
    assert traces[1].attempts[0].reason_code == "circuit_open"
    assert traces[1].attempts[0].error_type == "circuit_open"
    assert registry.circuit_breakers[(release.release_id, "L1")]["state"] == "open"


def test_run_shadow_request_compares_stable_and_shadow_paths(
    target_dir: Path, tmp_path: Path
) -> None:
    definition, contract, _check = load_checked_target(target_dir)
    stable_dir = write_artifact(
        tmp_path / "stable-artifact", definition.contract_hash, accept_prefixes=["a"]
    )
    shadow_dir = write_artifact(
        tmp_path / "shadow-artifact", definition.contract_hash, accept_prefixes=["b"]
    )
    stable_package = freeze_artifact_package(
        stable_dir, read_artifact_manifest(stable_dir), tmp_path / "stable-store", "snapshot-digest"
    )
    shadow_package = freeze_artifact_package(
        shadow_dir, read_artifact_manifest(shadow_dir), tmp_path / "shadow-store", "snapshot-digest"
    )
    stable = replace(_release_with_l1(definition, stable_package), release_id="stable-rel")
    shadow = replace(_release_with_l1(definition, shadow_package), release_id="shadow-rel")

    record = run_shadow_request(
        RuntimeRequest("req1", definition.name, {"text": "a:x"}),
        stable,
        shadow,
        contract,
        WorkerPool(),
        RuntimeContext("req1", 1000, PrefixBroker()),
        {},
    )

    assert record["stable"]["status"] == "ok"
    assert record["stable"]["chosen_layer"] == "L1"
    assert record["stable"]["attempts"][0]["decision"] == "accept"
    assert record["shadow"]["status"] == "ok"
    assert record["shadow"]["chosen_layer"] == "L4"
    assert record["shadow"]["attempts"][0]["decision"] == "abstain"
    assert record["shadow"]["l4_fallback_status"] == "ok"
    assert not record["chosen_layer_match"]
    assert record["output_hash_match"]


def test_run_shadow_request_shadow_failure_preserves_stable_result(
    target_dir: Path, tmp_path: Path
) -> None:
    definition, contract, _check = load_checked_target(target_dir)
    stable_dir = write_artifact(
        tmp_path / "stable-artifact", definition.contract_hash, accept_prefixes=["a"]
    )
    shadow_dir = write_artifact(
        tmp_path / "shadow-artifact", definition.contract_hash, accept_prefixes=["b"]
    )
    stable_package = freeze_artifact_package(
        stable_dir, read_artifact_manifest(stable_dir), tmp_path / "stable-store", "snapshot-digest"
    )
    shadow_package = freeze_artifact_package(
        shadow_dir, read_artifact_manifest(shadow_dir), tmp_path / "shadow-store", "snapshot-digest"
    )
    stable = replace(_release_with_l1(definition, stable_package), release_id="stable-rel")
    shadow = replace(_release_with_l1(definition, shadow_package), release_id="shadow-rel")

    record = run_shadow_request(
        RuntimeRequest("req1", definition.name, {"text": "a:x"}),
        stable,
        shadow,
        contract,
        WorkerPool(),
        RuntimeContext("req1", 1000, PrefixBroker(fail=True)),
        {},
    )

    assert record["stable"]["status"] == "ok"
    assert record["stable"]["chosen_layer"] == "L1"
    assert record["shadow"]["status"] == "error"
    assert record["shadow"]["chosen_layer"] == "L4"
    assert record["shadow"]["error_type"] == "l4_fallback_failure"
    assert not record["status_match"]
