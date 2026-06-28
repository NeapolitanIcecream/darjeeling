from __future__ import annotations

from dataclasses import replace
from typing import Any

from darjeeling.agent_workspace import (
    create_agent_workspace,
    create_compile_run,
    launch_target_adaptation_agent,
    load_target_workspace,
    mount_readonly_inputs,
    write_agent_brief,
)
from darjeeling.artifact_worker import build_protocol_docs
from darjeeling.errors import CompileLaunchError, SnapshotBuildError
from darjeeling.model import (
    AgentAttempt,
    AgentAttemptOptions,
    AgentSessionHandle,
    AgentViewOptions,
    CompileLaunchDecision,
    CompileOptions,
    CompileRun,
    CompileRunStore,
    ConsumedRowsManifest,
    DataConfig,
    RecompileRequest,
    ReferenceBroker,
    Release,
    SchedulerPolicy,
    TargetCheckReport,
    TargetDefinition,
    TargetRuntimeContract,
    WorkspaceStore,
)
from darjeeling.snapshot_reference import build_snapshot, export_train_view_for_agent
from darjeeling.target_definition import export_agent_readonly_target_view
from darjeeling.util import utcnow


def plan_compile_launch(
    definition: TargetDefinition,
    contract: TargetRuntimeContract,
    target_check: TargetCheckReport,
    request: RecompileRequest,
    base_release: Release,
    consumed_manifests: list[ConsumedRowsManifest],
    active_jobs: list[CompileRun],
    policy: SchedulerPolicy,
) -> CompileLaunchDecision:
    budget = request.budget_hint or policy.default_compile_budget
    if (
        contract.contract_hash != definition.contract_hash
        or target_check.target_name != definition.name
        or target_check.contract_hash != definition.contract_hash
    ):
        return CompileLaunchDecision(
            "rejected",
            definition.name,
            definition.contract_hash,
            base_release.release_id,
            "target scope mismatch",
            None,
            None,
            None,
            None,
            utcnow(),
        )
    if request.target_name != definition.name or request.contract_hash != definition.contract_hash:
        return CompileLaunchDecision(
            "rejected",
            definition.name,
            definition.contract_hash,
            base_release.release_id,
            "request scope mismatch",
            None,
            None,
            None,
            None,
            utcnow(),
        )
    if request.telemetry_source is not None and (
        request.telemetry_source.target_name != definition.name
        or request.telemetry_source.contract_hash != definition.contract_hash
    ):
        return CompileLaunchDecision(
            "rejected",
            definition.name,
            definition.contract_hash,
            base_release.release_id,
            "telemetry source scope mismatch",
            None,
            request.telemetry_source.source_id,
            None,
            None,
            utcnow(),
        )
    if (
        budget.max_agent_seconds < 0
        or budget.max_candidates <= 0
        or (budget.max_cost is not None and budget.max_cost < 0)
        or policy.max_concurrent_compiles <= 0
    ):
        return CompileLaunchDecision(
            "rejected",
            definition.name,
            definition.contract_hash,
            base_release.release_id,
            "invalid compile budget or scheduler policy",
            None,
            request.telemetry_source.source_id if request.telemetry_source else None,
            None,
            None,
            utcnow(),
        )
    if request.requested_by not in {"user", "scheduler", "monitoring"}:
        return CompileLaunchDecision(
            "rejected",
            definition.name,
            definition.contract_hash,
            base_release.release_id,
            "unsupported recompile trigger",
            None,
            request.telemetry_source.source_id if request.telemetry_source else None,
            None,
            None,
            utcnow(),
        )
    if (
        base_release.release_id != request.base_release_id
        or base_release.target_name != definition.name
        or base_release.contract_hash != definition.contract_hash
    ):
        return CompileLaunchDecision(
            "rejected",
            definition.name,
            definition.contract_hash,
            base_release.release_id,
            "base release scope mismatch",
            None,
            None,
            None,
            None,
            utcnow(),
        )
    if target_check.status != "pass":
        return CompileLaunchDecision(
            "rejected",
            definition.name,
            definition.contract_hash,
            base_release.release_id,
            "target checks failed",
            None,
            None,
            None,
            None,
            utcnow(),
        )
    if request.requested_by == "monitoring" and not policy.allow_monitoring_recompile:
        return CompileLaunchDecision(
            "deferred",
            definition.name,
            definition.contract_hash,
            base_release.release_id,
            "monitoring recompile disabled",
            None,
            None,
            None,
            None,
            utcnow(),
        )
    if request.requested_by == "scheduler" and not policy.allow_scheduled_recompile:
        return CompileLaunchDecision(
            "deferred",
            definition.name,
            definition.contract_hash,
            base_release.release_id,
            "scheduled recompile disabled",
            None,
            None,
            None,
            None,
            utcnow(),
        )
    running = [
        job
        for job in active_jobs
        if job.status in {"running", "closing"} and job.target_name == definition.name
    ]
    if len(running) >= policy.max_concurrent_compiles:
        return CompileLaunchDecision(
            "deferred",
            definition.name,
            definition.contract_hash,
            base_release.release_id,
            "max concurrent compiles reached",
            None,
            None,
            None,
            None,
            utcnow(),
        )
    cutoff = utcnow()
    if request.telemetry_source is not None and request.telemetry_source.cutoff_time > cutoff:
        return CompileLaunchDecision(
            "rejected",
            definition.name,
            definition.contract_hash,
            base_release.release_id,
            "telemetry source cutoff is later than snapshot cutoff",
            None,
            request.telemetry_source.source_id,
            None,
            None,
            utcnow(),
        )
    snapshot_options = replace(
        policy.default_snapshot_options,
        allow_insufficient_reference=(
            not policy.require_user_approval_for_insufficient_reference
        ),
    )
    return CompileLaunchDecision(
        status="accepted",
        target_name=definition.name,
        contract_hash=definition.contract_hash,
        base_release_id=base_release.release_id,
        reason="accepted",
        budget=budget,
        telemetry_source_id=request.telemetry_source.source_id
        if request.telemetry_source
        else None,
        snapshot_cutoff_time=cutoff,
        snapshot_options=snapshot_options,
        created_at=utcnow(),
    )


def _check_launch_scope(
    decision: CompileLaunchDecision,
    definition: TargetDefinition,
    contract: TargetRuntimeContract,
    target_check: TargetCheckReport,
    request: RecompileRequest,
    base_release: Release,
) -> None:
    if target_check.status != "pass":
        raise CompileLaunchError("target checks failed")
    telemetry_source_id = request.telemetry_source.source_id if request.telemetry_source else None
    if (
        decision.target_name != definition.name
        or decision.contract_hash != definition.contract_hash
        or decision.base_release_id != base_release.release_id
        or decision.telemetry_source_id != telemetry_source_id
        or contract.contract_hash != definition.contract_hash
        or target_check.target_name != definition.name
        or target_check.contract_hash != definition.contract_hash
        or request.target_name != definition.name
        or request.contract_hash != definition.contract_hash
        or request.base_release_id != base_release.release_id
        or base_release.target_name != definition.name
        or base_release.contract_hash != definition.contract_hash
    ):
        raise CompileLaunchError("compile launch decision scope mismatch")
    if request.telemetry_source is not None and (
        request.telemetry_source.target_name != definition.name
        or request.telemetry_source.contract_hash != definition.contract_hash
    ):
        raise CompileLaunchError("telemetry source scope mismatch")


def _effective_launch_options(
    snapshot_options: Any,
    compile_options: CompileOptions,
) -> tuple[Any, CompileOptions]:
    allow_insufficient = (
        snapshot_options.allow_insufficient_reference
        or compile_options.allow_insufficient_reference_qualification
    )
    return (
        replace(snapshot_options, allow_insufficient_reference=allow_insufficient),
        replace(
            compile_options,
            allow_insufficient_reference_qualification=allow_insufficient,
        ),
    )


def _record_compile_run(store: CompileRunStore, compile_run: CompileRun) -> None:
    if compile_run.compile_id in store.runs:
        raise CompileLaunchError("compile run already recorded")
    store.runs[compile_run.compile_id] = compile_run


def start_compile_launch(
    decision: CompileLaunchDecision,
    definition: TargetDefinition,
    contract: TargetRuntimeContract,
    target_check: TargetCheckReport,
    data_config: DataConfig,
    request: RecompileRequest,
    base_release: Release,
    consumed_manifests: list[ConsumedRowsManifest],
    broker: ReferenceBroker,
    workspace_store: WorkspaceStore,
    compile_run_store: CompileRunStore,
    compile_options: CompileOptions,
    agent_options: AgentAttemptOptions,
    report_views: list[Any] | None = None,
    telemetry_summaries: list[Any] | None = None,
) -> tuple[CompileRun, AgentAttempt, AgentSessionHandle]:
    if decision.status != "accepted":
        raise CompileLaunchError("compile launch decision is not accepted")
    if (
        decision.snapshot_cutoff_time is None
        or decision.snapshot_options is None
        or decision.budget is None
    ):
        raise CompileLaunchError(
            "accepted decision must include snapshot cutoff, options, and budget"
        )
    _check_launch_scope(decision, definition, contract, target_check, request, base_release)
    if (
        request.telemetry_source is not None
        and request.telemetry_source.cutoff_time > decision.snapshot_cutoff_time
    ):
        raise CompileLaunchError("telemetry source cutoff is later than accepted snapshot cutoff")
    snapshot_options, compile_options = _effective_launch_options(
        decision.snapshot_options, compile_options
    )
    try:
        snapshot_result = build_snapshot(
            definition,
            contract,
            data_config,
            request.telemetry_source,
            consumed_manifests,
            broker,
            decision.snapshot_cutoff_time,
            snapshot_options,
        )
    except SnapshotBuildError as exc:
        raise CompileLaunchError(str(exc)) from exc
    if snapshot_result.reference_qualification.status == "fail":
        raise CompileLaunchError("reference qualification failed")
    workspace = load_target_workspace(definition.name, definition.contract_hash, workspace_store)
    compile_run = create_compile_run(
        definition,
        target_check,
        snapshot_result.snapshot,
        base_release,
        decision.budget,
        workspace,
        snapshot_result.reference_qualification,
        compile_options,
    )
    attempt = create_agent_workspace(compile_run, workspace, agent_options)
    target_view = export_agent_readonly_target_view(
        definition, attempt.workspace_path / "readonly_source" / "target"
    )
    train_view = export_train_view_for_agent(
        snapshot_result.snapshot,
        contract,
        AgentViewOptions(),
        attempt.workspace_path / "readonly_source" / "train",
    )
    protocol_docs = build_protocol_docs("v1")
    mount_manifest = mount_readonly_inputs(
        attempt,
        target_view,
        train_view,
        base_release,
        report_views=report_views or [],
        telemetry_summaries=telemetry_summaries or [],
        protocol_docs=protocol_docs,
    )
    brief = write_agent_brief(attempt, compile_run, mount_manifest, compile_options.objective)
    timeout_seconds = agent_options.agent_timeout_seconds or (
        decision.budget.max_agent_seconds if decision.budget.max_agent_seconds > 0 else None
    )
    handle = launch_target_adaptation_agent(
        attempt,
        brief,
        {
            "command": agent_options.agent_command,
            "timeout_seconds": timeout_seconds,
        },
    )
    _record_compile_run(compile_run_store, compile_run)
    return compile_run, attempt, handle
