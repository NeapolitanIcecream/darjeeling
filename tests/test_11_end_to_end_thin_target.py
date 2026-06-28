from __future__ import annotations

from pathlib import Path

from conftest import PrefixBroker, write_artifact

from darjeeling.agent_workspace import (
    close_agent_attempt,
    create_agent_workspace,
    create_compile_run,
    load_target_workspace,
    receive_candidate_submission,
)
from darjeeling.candidate_evaluation import (
    compare_candidates,
    evaluate_candidate_on_test,
    evaluate_candidate_on_validation,
    finalize_report,
)
from darjeeling.model import (
    AgentAttemptOptions,
    AgentUsageLedger,
    ApprovalRecord,
    CompileBudget,
    CompileOptions,
    L4BaselineSummary,
    ReleaseBaseline,
    ReleaseRegistry,
    ResultCache,
    RoutingSettings,
    RuntimeRequest,
    SnapshotOptions,
    WorkerPool,
    WorkspaceStore,
)
from darjeeling.release_runtime import (
    create_release,
    create_release_without_artifacts,
    serve_request,
    set_channel,
)
from darjeeling.runtime_trace_metrics import (
    aggregate_runtime_metrics,
    detect_runtime_failure,
    write_trace,
)
from darjeeling.snapshot_reference import build_snapshot
from darjeeling.target_definition import load_checked_target
from darjeeling.telemetry_recompile import request_recompile
from darjeeling.util import new_id, utcnow


def test_cold_start_to_compiled_release_to_runtime_feedback(
    target_dir: Path, tmp_path: Path, now
) -> None:
    definition, contract, check = load_checked_target(target_dir)
    registry = ReleaseRegistry()
    cold = create_release_without_artifacts(
        definition, contract, check, PrefixBroker(), RoutingSettings(), registry
    )
    set_channel(definition.name, "stable", cold.release_id, {}, registry)
    traces = []

    def record_trace(*args):
        trace = write_trace(*args)
        traces.append(trace)
        return trace

    response = serve_request(
        RuntimeRequest("req-cold", definition.name, {"text": "a:cold"}),
        registry,
        lambda _name: contract,
        WorkerPool(),
        lambda: new_id("trace"),
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
    assert response.chosen_layer == "L4"
    snapshot = build_snapshot(
        definition,
        contract,
        definition.data_config,
        None,
        [],
        PrefixBroker(),
        now,
        SnapshotOptions(storage_root=tmp_path / "snapshots"),
    )
    workspace = load_target_workspace(
        definition.name, definition.contract_hash, WorkspaceStore(tmp_path / "workspaces")
    )
    compile_run = create_compile_run(
        definition,
        check,
        snapshot.snapshot,
        cold,
        CompileBudget(),
        workspace,
        snapshot.reference_qualification,
        CompileOptions(),
    )
    attempt = create_agent_workspace(compile_run, workspace, AgentAttemptOptions())
    write_artifact(
        attempt.workspace_path / "submissions" / "c1" / "artifacts" / "l1",
        definition.contract_hash,
        accept_prefixes=["a", "b"],
    )
    submission = receive_candidate_submission(
        attempt, attempt.workspace_path / "submissions" / "c1"
    )
    validation = evaluate_candidate_on_validation(
        submission,
        definition,
        snapshot.snapshot,
        cold,
        snapshot.reference_qualification,
        AgentUsageLedger(),
        snapshot.reference_usage,
        None,
        None,
        {"serving_l4_cost": 1.0},
        {"contract": contract, "artifact_store": tmp_path / "artifacts"},
    )
    closed = close_agent_attempt(attempt, "ready_for_test")
    test = evaluate_candidate_on_test(
        validation["candidate"],
        closed,
        definition,
        snapshot.snapshot,
        cold,
        snapshot.reference_qualification,
        AgentUsageLedger(),
        snapshot.reference_usage,
        None,
        None,
        {"serving_l4_cost": 1.0},
        {"contract": contract, "test_results_visible": True},
    )
    decision = compare_candidates(
        [test["report"]],
        ReleaseBaseline(
            cold,
            l4_baseline=L4BaselineSummary(
                cold.release_id,
                definition.name,
                definition.contract_hash,
                {},
                [],
                snapshot.reference_qualification,
                {},
                {},
            ),
        ),
        {"requirements": definition.requirements},
    )
    final = finalize_report(test["report"], decision, validation["report"])
    approval = ApprovalRecord(
        new_id("approval"),
        validation["candidate"].candidate_id,
        final.report_id,
        definition.name,
        definition.contract_hash,
        snapshot.snapshot.snapshot_id,
        utcnow(),
        "user",
    )
    compiled = create_release(
        validation["candidate"], snapshot.snapshot, cold, final, approval, tmp_path / "artifacts"
    )
    registry.releases[compiled.release_id] = compiled
    set_channel(definition.name, "stable", compiled.release_id, {}, registry)
    response = serve_request(
        RuntimeRequest("req-compiled", definition.name, {"text": "a:compiled"}),
        registry,
        lambda _name: contract,
        WorkerPool(),
        lambda: new_id("trace"),
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
    assert response.chosen_layer == "L1"
    metrics = aggregate_runtime_metrics(
        [traces[-1]], [], (utcnow().replace(year=2020), utcnow().replace(year=2030)), compiled
    )
    failure = detect_runtime_failure(metrics, compiled, definition.requirements)
    assert failure.status in {"ok", "rollback_recommended"}
    recompile = request_recompile(
        definition,
        compiled,
        __import__("darjeeling.model").model.RecompileReason("manual"),
        None,
        None,
    )
    assert recompile.base_release_id == compiled.release_id
