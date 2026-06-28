from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from conftest import PrefixBroker

from darjeeling.compile_orchestration import plan_compile_launch, start_compile_launch
from darjeeling.errors import CompileLaunchError, WorkspaceError
from darjeeling.model import (
    AgentAttemptOptions,
    AgentVisibleDecisionSummary,
    AgentVisibleReport,
    AgentVisibleTelemetrySummary,
    CompileOptions,
    CompileRunStore,
    RecompileReason,
    RecompileRequest,
    ReleaseRegistry,
    RoutingSettings,
    SchedulerPolicy,
    SnapshotOptions,
    TelemetryDataSource,
    WorkspaceStore,
)
from darjeeling.release_runtime import create_release_without_artifacts
from darjeeling.target_definition import load_checked_target
from darjeeling.util import utcnow


def test_first_compile_after_cold_start_uses_recompile_request_path(
    target_dir: Path, tmp_path: Path
) -> None:
    definition, contract, check = load_checked_target(target_dir)
    registry = ReleaseRegistry()
    release = create_release_without_artifacts(
        definition, contract, check, PrefixBroker(), RoutingSettings(), registry
    )
    request = RecompileRequest(
        definition.name,
        definition.contract_hash,
        RecompileReason("manual"),
        None,
        release.release_id,
        None,
        utcnow(),
        "user",
    )
    policy = SchedulerPolicy(
        1,
        True,
        True,
        False,
        default_snapshot_options=SnapshotOptions(storage_root=tmp_path / "snapshots"),
    )
    decision = plan_compile_launch(definition, contract, check, request, release, [], [], policy)
    assert decision.status == "accepted"
    compile_run_store = CompileRunStore()
    outside_secret = tmp_path / "snapshot-secret.txt"
    outside_secret.write_text("secret\n", encoding="utf-8")
    agent_code = "\n".join(
        [
            "from pathlib import Path",
            "outside = Path(" + repr(str(outside_secret)) + ")",
            "try:",
            "    outside.read_text()",
            "    raise SystemExit(31)",
            "except PermissionError:",
            "    pass",
            "Path('journal/launched.txt').write_text('ok')",
        ]
    )
    report = AgentVisibleReport(
        "report-1",
        "candidate-1",
        definition.name,
        definition.contract_hash,
        "snapshot-prior",
        release.release_id,
        AgentVisibleDecisionSummary(
            "candidate-1", "rejected", "regression", {}, {}, False, None
        ),
        {"sample_count": 1},
        None,
        False,
        None,
        {},
        {},
        utcnow(),
    )
    telemetry = AgentVisibleTelemetrySummary(
        definition.name,
        release.release_id,
        definition.contract_hash,
        {"requests": 10},
        {},
        None,
        "v1",
        utcnow(),
    )
    compile_run, attempt, handle = start_compile_launch(
        decision,
        definition,
        contract,
        check,
        definition.data_config,
        request,
        release,
        [],
        PrefixBroker(),
        WorkspaceStore(tmp_path / "workspaces"),
        compile_run_store,
        CompileOptions(),
        AgentAttemptOptions(
            agent_command=[
                "/usr/bin/python3",
                "-c",
                agent_code,
            ]
        ),
        report_views=[report],
        telemetry_summaries=[telemetry],
    )
    mounted = "\n".join(
        path.read_text(errors="ignore")
        for path in attempt.workspace_path.rglob("*")
        if path.is_file()
    )
    assert compile_run.base_release_id == release.release_id
    assert compile_run_store.runs[compile_run.compile_id] == compile_run
    assert handle.status == "completed"
    deferred = plan_compile_launch(
        definition,
        contract,
        check,
        request,
        release,
        [],
        [replace(compile_run, status="closing")],
        policy,
    )
    assert deferred.status == "deferred"
    assert deferred.reason == "max concurrent compiles reached"
    assert (attempt.workspace_path / "journal" / "launched.txt").exists()
    assert "report-1" in (
        attempt.workspace_path / "readonly_inputs" / "agent_visible_reports.json"
    ).read_text()
    assert "requests" in (
        attempt.workspace_path / "readonly_inputs" / "agent_visible_telemetry.json"
    ).read_text()
    failed_store = CompileRunStore()
    with pytest.raises(WorkspaceError, match="telemetry mount"):
        start_compile_launch(
            decision,
            definition,
            contract,
            check,
            definition.data_config,
            request,
            release,
            [],
            PrefixBroker(),
            WorkspaceStore(tmp_path / "workspaces"),
            failed_store,
            CompileOptions(),
            AgentAttemptOptions(agent_command=["/usr/bin/python3", "-c", "print('no launch')"]),
            report_views=[report],
            telemetry_summaries=[replace(telemetry, contract_hash="other")],
        )
    assert failed_store.runs == {}
    assert "r2" not in mounted
    assert "r4" not in mounted


def test_plan_compile_launch_rejects_target_and_telemetry_scope_mismatch(
    target_dir: Path, tmp_path: Path
) -> None:
    definition, contract, check = load_checked_target(target_dir)
    registry = ReleaseRegistry()
    release = create_release_without_artifacts(
        definition, contract, check, PrefixBroker(), RoutingSettings(), registry
    )
    request = RecompileRequest(
        definition.name,
        definition.contract_hash,
        RecompileReason("manual"),
        None,
        release.release_id,
        None,
        utcnow(),
        "user",
    )
    policy = SchedulerPolicy(
        1,
        True,
        True,
        False,
        default_snapshot_options=SnapshotOptions(storage_root=tmp_path / "snapshots"),
    )

    decision = plan_compile_launch(
        definition,
        contract,
        replace(check, contract_hash="other-contract"),
        request,
        release,
        [],
        [],
        policy,
    )
    assert decision.status == "rejected"
    assert decision.reason == "target scope mismatch"

    bad_source = TelemetryDataSource(
        "telemetry-1",
        "other-target",
        definition.contract_hash,
        utcnow(),
        str(tmp_path / "records.json"),
        ["train"],
        None,
        ["user_feedback"],
        "digest",
    )
    decision = plan_compile_launch(
        definition,
        contract,
        check,
        replace(request, telemetry_source=bad_source),
        release,
        [],
        [],
        policy,
    )
    assert decision.status == "rejected"
    assert decision.reason == "telemetry source scope mismatch"


def test_plan_compile_launch_rejects_unsupported_recompile_trigger(
    target_dir: Path, tmp_path: Path
) -> None:
    definition, contract, check = load_checked_target(target_dir)
    registry = ReleaseRegistry()
    release = create_release_without_artifacts(
        definition, contract, check, PrefixBroker(), RoutingSettings(), registry
    )
    request = RecompileRequest(
        definition.name,
        definition.contract_hash,
        RecompileReason("manual"),
        None,
        release.release_id,
        None,
        utcnow(),
        "bot",  # type: ignore[arg-type]
    )

    decision = plan_compile_launch(
        definition,
        contract,
        check,
        request,
        release,
        [],
        [],
        SchedulerPolicy(
            1,
            True,
            True,
            False,
            default_snapshot_options=SnapshotOptions(storage_root=tmp_path / "snapshots"),
        ),
    )

    assert decision.status == "rejected"
    assert decision.reason == "unsupported recompile trigger"
    assert decision.budget is None
    assert decision.snapshot_cutoff_time is None
    assert decision.snapshot_options is None


def test_start_compile_launch_rejects_stale_accepted_decision_before_snapshot(
    target_dir: Path, tmp_path: Path
) -> None:
    definition, contract, check = load_checked_target(target_dir)
    registry = ReleaseRegistry()
    release = create_release_without_artifacts(
        definition, contract, check, PrefixBroker(), RoutingSettings(), registry
    )
    request = RecompileRequest(
        definition.name,
        definition.contract_hash,
        RecompileReason("manual"),
        None,
        release.release_id,
        None,
        utcnow(),
        "user",
    )
    decision = plan_compile_launch(
        definition,
        contract,
        check,
        request,
        release,
        [],
        [],
        SchedulerPolicy(
            1,
            True,
            True,
            False,
            default_snapshot_options=SnapshotOptions(storage_root=tmp_path / "snapshots"),
        ),
    )

    compile_run_store = CompileRunStore()
    with pytest.raises(CompileLaunchError, match="scope mismatch"):
        start_compile_launch(
            replace(decision, base_release_id="other-release"),
            definition,
            contract,
            check,
            definition.data_config,
            request,
            release,
            [],
            PrefixBroker(),
            WorkspaceStore(tmp_path / "workspaces"),
            compile_run_store,
            CompileOptions(),
            AgentAttemptOptions(agent_command=["/usr/bin/python3", "-c", "print('no launch')"]),
        )
    assert compile_run_store.runs == {}

    with pytest.raises(CompileLaunchError, match="target checks failed"):
        start_compile_launch(
            decision,
            definition,
            contract,
            replace(check, status="fail", failures=["stale target check failure"]),
            definition.data_config,
            request,
            release,
            [],
            PrefixBroker(),
            WorkspaceStore(tmp_path / "workspaces"),
            compile_run_store,
            CompileOptions(),
            AgentAttemptOptions(agent_command=["/usr/bin/python3", "-c", "print('no launch')"]),
        )
    assert compile_run_store.runs == {}
    assert not (tmp_path / "snapshots").exists()


def test_insufficient_reference_approval_reaches_snapshot_and_workspace(
    target_dir: Path, tmp_path: Path
) -> None:
    data = __import__("yaml").safe_load((target_dir / "data.yaml").read_text())
    for source in data["sources"]:
        for record in source["records"]:
            record.pop("reference_output", None)
            record.pop("reference_source", None)
    (target_dir / "data.yaml").write_text(__import__("yaml").safe_dump(data), encoding="utf-8")
    definition, contract, check = load_checked_target(target_dir)
    registry = ReleaseRegistry()
    release = create_release_without_artifacts(
        definition, contract, check, PrefixBroker(), RoutingSettings(), registry
    )
    request = RecompileRequest(
        definition.name,
        definition.contract_hash,
        RecompileReason("manual"),
        None,
        release.release_id,
        None,
        utcnow(),
        "user",
    )
    decision = plan_compile_launch(
        definition,
        contract,
        check,
        request,
        release,
        [],
        [],
        SchedulerPolicy(
            1,
            True,
            True,
            True,
            default_snapshot_options=SnapshotOptions(storage_root=tmp_path / "snapshots"),
        ),
    )

    with pytest.raises(CompileLaunchError, match="reference qualification insufficient"):
        start_compile_launch(
            decision,
            definition,
            contract,
            check,
            definition.data_config,
            request,
            release,
            [],
            PrefixBroker(),
            WorkspaceStore(tmp_path / "workspaces-no-approval"),
            CompileRunStore(),
            CompileOptions(),
            AgentAttemptOptions(agent_command=["/usr/bin/python3", "-c", "print('no launch')"]),
        )

    compile_run_store = CompileRunStore()
    compile_run, _, handle = start_compile_launch(
        decision,
        definition,
        contract,
        check,
        definition.data_config,
        request,
        release,
        [],
        PrefixBroker(),
        WorkspaceStore(tmp_path / "workspaces-approved"),
        compile_run_store,
        CompileOptions(allow_insufficient_reference_qualification=True),
        AgentAttemptOptions(
            agent_command=[
                "/usr/bin/python3",
                "-c",
                "from pathlib import Path; Path('journal/launched.txt').write_text('ok')",
            ]
        ),
    )
    assert handle.status == "completed"
    assert compile_run_store.runs[compile_run.compile_id] == compile_run
