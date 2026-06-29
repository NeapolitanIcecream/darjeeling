from __future__ import annotations

import multiprocessing as mp
import os
import signal
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from darjeeling.agent_workspace import (
    _read_session_record_for_handle,
    candidate_submission_ready,
    close_agent_attempt,
    core_attempt_state_dir,
    create_agent_workspace,
    create_compile_run,
    launch_target_adaptation_agent,
    launch_target_adaptation_agent_async,
    load_target_workspace,
    mount_readonly_inputs,
    poll_agent_session,
    provide_validation_feedback,
    receive_candidate_submission,
    stop_agent_session,
    write_agent_brief,
)
from darjeeling.artifact_worker import build_protocol_docs
from darjeeling.candidate_evaluation import evaluate_candidate_on_validation
from darjeeling.errors import CompileLaunchError, SnapshotBuildError
from darjeeling.model import (
    AgentAttempt,
    AgentAttemptOptions,
    AgentFeedback,
    AgentSessionHandle,
    AgentUsageEvent,
    AgentUsageLedger,
    AgentViewOptions,
    CandidateSubmission,
    ClosedAgentAttempt,
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
from darjeeling.util import (
    file_digest,
    read_json,
    safe_public_error,
    stable_hash,
    utcnow,
    write_json_atomic,
)


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


def _submission_content_digest(path: Path) -> str:
    entries: list[tuple[str, str]] = []
    for item in sorted(path.rglob("*")):
        rel = item.relative_to(path).as_posix()
        if item.is_file() and not item.is_symlink():
            entries.append((rel, file_digest(item)))
        elif item.is_symlink():
            entries.append((rel, f"symlink:{os.readlink(item)}"))
    return stable_hash(entries)


def _submission_digest_failure_marker(submission_id: str) -> str:
    return f"digest_failed:{stable_hash(submission_id)}"


def _ledger_path(attempt: AgentAttempt) -> Path:
    return core_attempt_state_dir(attempt) / "evaluated_submissions.json"


def _core_agent_usage_path(attempt: AgentAttempt) -> Path:
    return core_attempt_state_dir(attempt) / "agent_usage.json"


def _load_submission_ledger(attempt: AgentAttempt) -> list[dict[str, Any]]:
    path = _ledger_path(attempt)
    if not path.exists():
        return []
    raw = read_json(path)
    if not isinstance(raw, list):
        raise CompileLaunchError("evaluated submission ledger must contain a list")
    return [entry for entry in raw if isinstance(entry, dict)]


def _write_submission_ledger(attempt: AgentAttempt, ledger: list[dict[str, Any]]) -> None:
    write_json_atomic(_ledger_path(attempt), ledger)


def _ledger_contains(
    ledger: list[dict[str, Any]], submission_id: str, submission_digest: str
) -> bool:
    return any(
        entry.get("submission_id") == submission_id
        and entry.get("submission_digest") == submission_digest
        and entry.get("validation_status")
        in {"feedback_written", "evaluation_failed", "skipped"}
        for entry in ledger
    )


def _count_evaluated_entries(ledger: list[dict[str, Any]]) -> int:
    return sum(
        1
        for entry in ledger
        if entry.get("validation_status") in {"feedback_written", "evaluation_failed"}
    )


def _ledger_cost(ledger: list[dict[str, Any]]) -> float:
    total = 0.0
    for entry in ledger:
        value = entry.get("total_compile_cost", entry.get("candidate_cost"))
        if isinstance(value, (int, float)):
            total = max(total, float(value))
    return total


def _agent_usage_ledger_from_raw(raw: Any) -> AgentUsageLedger | None:
    if not isinstance(raw, list):
        return None
    events: list[AgentUsageEvent] = []
    for event in raw:
        if not isinstance(event, dict):
            continue
        kind = event.get("kind")
        cost = event.get("cost", 0.0)
        metadata = event.get("metadata", {})
        if (
            not isinstance(kind, str)
            or isinstance(cost, bool)
            or not isinstance(cost, (int, float))
        ):
            continue
        cost_value = float(cost)
        if cost_value < 0:
            continue
        if not isinstance(metadata, dict):
            metadata = {}
        events.append(AgentUsageEvent(kind=kind, cost=cost_value, metadata=metadata))
    return AgentUsageLedger(events)


def _read_agent_usage_ledger_path(path: Path) -> AgentUsageLedger | None:
    if not path.exists():
        return AgentUsageLedger()
    if path.is_symlink() or not path.is_file():
        return None
    try:
        raw = read_json(path)
    except (OSError, TypeError, ValueError):
        return None
    return _agent_usage_ledger_from_raw(raw)


def _read_agent_usage_ledger(attempt: AgentAttempt) -> AgentUsageLedger:
    ledger = _read_agent_usage_ledger_path(
        attempt.workspace_path / "journal" / "agent_usage.json"
    )
    return ledger if ledger is not None else AgentUsageLedger()


def _read_core_agent_usage_ledger(attempt: AgentAttempt) -> AgentUsageLedger:
    ledger = _read_agent_usage_ledger_path(_core_agent_usage_path(attempt))
    if ledger is None:
        raise CompileLaunchError("core agent usage ledger is malformed")
    return ledger


def _write_core_agent_usage_ledger(
    attempt: AgentAttempt, ledger: AgentUsageLedger
) -> None:
    write_json_atomic(
        _core_agent_usage_path(attempt),
        [asdict(event) for event in ledger.events],
    )


def _sync_core_agent_usage_ledger(attempt: AgentAttempt) -> AgentUsageLedger:
    core_ledger = _read_core_agent_usage_ledger(attempt)
    observed_ledger = _read_agent_usage_ledger_path(
        attempt.workspace_path / "journal" / "agent_usage.json"
    )
    if observed_ledger is None:
        return core_ledger
    if observed_ledger.cost > core_ledger.cost:
        _write_core_agent_usage_ledger(attempt, observed_ledger)
        return observed_ledger
    return core_ledger


def _fixed_compile_cost(reference_usage: Any, local_training_search_usage: Any) -> float:
    compile_cost = getattr(reference_usage, "cost", 0.0)
    if isinstance(local_training_search_usage, dict):
        compile_cost += float(local_training_search_usage.get("cost", 0.0))
    return float(compile_cost)


def _live_compile_cost(
    agent_usage: AgentUsageLedger, fixed_compile_cost: float, reported_compile_cost: float
) -> float:
    return max(reported_compile_cost, fixed_compile_cost + agent_usage.cost)


def _safe_failure_feedback(submission_id: str, exc: Exception) -> AgentFeedback:
    return AgentFeedback(
        candidate_id=submission_id,
        summary={
            "status": "evaluation_failed",
            "error_class": exc.__class__.__name__,
            "safe_error_message": safe_public_error("runtime_error"),
        },
        requirement_results=[],
        metrics={},
        safe_slice_summaries=[],
        latency_cost_summary={},
        raw_rows_included=False,
    )


def _feedback_for_submission(
    submission: CandidateSubmission, evaluation: dict[str, Any]
) -> AgentFeedback:
    feedback = evaluation["feedback"]
    candidate = evaluation.get("candidate")
    summary = dict(feedback.summary)
    summary.setdefault("submission_id", submission.submission_id)
    if candidate is not None:
        summary.setdefault("candidate_id", candidate.candidate_id)
    return replace(feedback, candidate_id=submission.submission_id, summary=summary)


def _close_reason_for_agent_status(status: str) -> str:
    if status == "completed":
        return "ready_for_test"
    if status == "timed_out":
        return "time_limit"
    if status in {"failed", "stopped"}:
        return "failure"
    return "ready_for_test"


def _agent_session_timeout_seconds(handle: AgentSessionHandle) -> float | None:
    if handle.session_record_path is not None:
        record = _read_session_record_for_handle(handle)
        timeout_seconds = record.get("timeout_seconds")
        if isinstance(timeout_seconds, (int, float)) and timeout_seconds > 0:
            return float(timeout_seconds)
    timeout_seconds = handle.timeout_seconds
    if isinstance(timeout_seconds, (int, float)) and timeout_seconds > 0:
        return float(timeout_seconds)
    return None


def _agent_session_elapsed_seconds(
    handle: AgentSessionHandle, fallback_elapsed_seconds: float
) -> float:
    started_at = None
    if handle.session_record_path is not None:
        record = _read_session_record_for_handle(handle)
        raw_started = record.get("started_at")
        if isinstance(raw_started, str):
            try:
                started_at = datetime.fromisoformat(raw_started)
            except ValueError:
                started_at = None
    if started_at is None:
        started_at = handle.started_at
    if started_at is None:
        return fallback_elapsed_seconds
    return max(0.0, (utcnow() - started_at).total_seconds())


class ValidationProcessError(RuntimeError):
    pass


def _enter_validation_process_group() -> int | None:
    try:
        os.setsid()
    except (AttributeError, OSError):
        return None
    try:
        process_group_id = os.getpgrp()
    except (AttributeError, OSError):
        return None
    return process_group_id if process_group_id == os.getpid() else None


def _validation_process_group_id(process: mp.Process) -> int | None:
    pid = process.pid
    if pid is None:
        return None
    try:
        process_group_id = os.getpgid(pid)
    except (AttributeError, ProcessLookupError, OSError):
        return None
    if process_group_id != pid:
        return None
    return process_group_id


def _signal_validation_process_group(
    process_group_id: int, sig: signal.Signals
) -> bool:
    try:
        os.killpg(process_group_id, sig)
        return True
    except (AttributeError, ProcessLookupError, OSError):
        return False


def _stop_validation_process(
    process: mp.Process,
    process_group_id: int | None = None,
    timeout_seconds: float = 0.2,
) -> None:
    if process_group_id is None and process.is_alive():
        process_group_id = _validation_process_group_id(process)
    if not process.is_alive():
        process.join(timeout=0)
        if process_group_id is not None:
            _signal_validation_process_group(process_group_id, signal.SIGKILL)
        return
    signaled_process_group = process_group_id is not None and _signal_validation_process_group(
        process_group_id, signal.SIGTERM
    )
    if not signaled_process_group:
        process.terminate()
    process.join(timeout=timeout_seconds)
    if signaled_process_group and process_group_id is not None:
        _signal_validation_process_group(process_group_id, signal.SIGKILL)
        process.join(timeout=timeout_seconds)
    if not process.is_alive():
        return
    if process_group_id is None or not _signal_validation_process_group(
        process_group_id, signal.SIGKILL
    ):
        process.kill()
    process.join()


def _validation_process_main(
    connection: Any,
    args: tuple[Any, ...],
) -> None:
    process_group_id = _enter_validation_process_group()
    connection.send(("process_group", process_group_id))
    try:
        connection.send(("ok", evaluate_candidate_on_validation(*args)))
    except BaseException as exc:
        try:
            connection.send(("error", exc))
        except Exception:
            connection.send(("error_class", exc.__class__.__name__))
    finally:
        connection.close()


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
    *,
    launch_async: bool = False,
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
    launch = (
        launch_target_adaptation_agent_async
        if launch_async
        else launch_target_adaptation_agent
    )
    handle = launch(
        attempt,
        brief,
        {
            "command": agent_options.agent_command,
            "timeout_seconds": timeout_seconds,
        },
    )
    _record_compile_run(compile_run_store, compile_run)
    return compile_run, attempt, handle


def run_interactive_compile_loop(
    compile_run: CompileRun,
    attempt: AgentAttempt,
    agent_handle: AgentSessionHandle,
    definition: TargetDefinition,
    contract: TargetRuntimeContract,
    snapshot: Any,
    base_release: Release,
    reference_qualification: Any,
    reference_usage: Any,
    baseline_cost: dict[str, Any],
    evaluation_options: dict[str, Any],
    *,
    poll_interval_seconds: float = 0.05,
) -> dict[str, Any]:
    if attempt.compile_id != compile_run.compile_id:
        raise CompileLaunchError("agent attempt does not match compile run")
    if agent_handle.attempt_id != attempt.attempt_id:
        raise CompileLaunchError("agent handle does not match agent attempt")
    if poll_interval_seconds <= 0:
        raise CompileLaunchError("poll interval must be positive")
    ledger = _load_submission_ledger(attempt)
    evaluated_count = _count_evaluated_entries(ledger)
    feedback_count = sum(
        1 for entry in ledger if entry.get("validation_status") == "feedback_written"
    )
    failed_count = sum(
        1 for entry in ledger if entry.get("validation_status") == "evaluation_failed"
    )
    skipped_count = sum(1 for entry in ledger if entry.get("validation_status") == "skipped")
    total_candidate_cost = _ledger_cost(ledger)
    started = time.monotonic()
    agent_timeout_seconds = _agent_session_timeout_seconds(agent_handle)
    stop_reason: str | None = (
        "candidate_limit"
        if evaluated_count >= compile_run.budget.max_candidates
        else None
    )
    handle = agent_handle
    options = dict(evaluation_options)
    options["contract"] = contract
    fixed_compile_cost = _fixed_compile_cost(
        reference_usage, options.get("local_training_search_usage")
    )
    agent_usage_ledger = _sync_core_agent_usage_ledger(attempt)
    pending_stop_reason: str | None = None

    def refresh_agent_usage_ledger() -> AgentUsageLedger:
        nonlocal agent_usage_ledger
        agent_usage_ledger = _sync_core_agent_usage_ledger(attempt)
        return agent_usage_ledger

    def refresh_total_compile_cost() -> float:
        nonlocal total_candidate_cost
        total_candidate_cost = _live_compile_cost(
            refresh_agent_usage_ledger(), fixed_compile_cost, total_candidate_cost
        )
        return total_candidate_cost

    def current_stop_reason() -> str | None:
        elapsed = time.monotonic() - started
        agent_elapsed = _agent_session_elapsed_seconds(handle, elapsed)
        if agent_timeout_seconds is not None and agent_elapsed >= agent_timeout_seconds:
            return "time_limit"
        if (
            compile_run.budget.max_agent_seconds > 0
            and agent_elapsed >= compile_run.budget.max_agent_seconds
        ):
            return "time_limit"
        if (attempt.workspace_path / "journal" / "stop_compile").exists():
            return "user_stop"
        if (
            compile_run.budget.max_cost is not None
            and refresh_total_compile_cost() >= compile_run.budget.max_cost
        ):
            return "budget_exhausted"
        return None

    def stop_running_agent(reason: str) -> None:
        nonlocal handle, pending_stop_reason
        pending_stop_reason = pending_stop_reason or reason
        if handle.status == "running":
            handle = stop_agent_session(handle, reason=reason)

    def evaluate_candidate_with_budget_checks(
        submission: CandidateSubmission,
    ) -> dict[str, Any]:
        context = mp.get_context("fork")
        parent_connection, child_connection = context.Pipe(duplex=False)
        process = context.Process(
            target=_validation_process_main,
            args=(
                child_connection,
                (
                    submission,
                    definition,
                    snapshot,
                    base_release,
                    reference_qualification,
                    refresh_agent_usage_ledger(),
                    reference_usage,
                    options.get("audit_usage"),
                    options.get("local_training_search_usage"),
                    baseline_cost,
                    options,
                ),
            ),
        )
        process.start()
        child_connection.close()
        validation_process_group_id: int | None = None
        try:
            while True:
                if parent_connection.poll(poll_interval_seconds):
                    status, payload = parent_connection.recv()
                    if status == "process_group":
                        if isinstance(payload, int) and payload > 0:
                            validation_process_group_id = payload
                        continue
                    process.join(timeout=1.0)
                    reason = current_stop_reason()
                    if reason is not None:
                        _stop_validation_process(process, validation_process_group_id)
                        stop_running_agent(reason)
                        raise TimeoutError(f"validation stopped after {reason}")
                    if status == "ok":
                        return payload
                    if status == "error":
                        raise payload
                    raise ValidationProcessError(str(payload))
                reason = current_stop_reason()
                if reason is None:
                    continue
                _stop_validation_process(process, validation_process_group_id)
                stop_running_agent(reason)
                raise TimeoutError(f"validation stopped after {reason}")
        finally:
            parent_connection.close()
            _stop_validation_process(process, validation_process_group_id)

    def drain_ready_submissions() -> str | None:
        nonlocal evaluated_count, feedback_count, failed_count, total_candidate_cost
        if evaluated_count >= compile_run.budget.max_candidates:
            return "candidate_limit"
        submissions_dir = attempt.workspace_path / "submissions"
        if (
            not submissions_dir.exists()
            or submissions_dir.is_symlink()
            or not submissions_dir.is_dir()
        ):
            return None
        for submission_path in sorted(
            path
            for path in submissions_dir.iterdir()
            if path.is_dir() and candidate_submission_ready(path)
        ):
            if evaluated_count >= compile_run.budget.max_candidates:
                return "candidate_limit"
            candidate_cost: float | None = None
            submission_digest: str | None = None
            try:
                submission_digest = _submission_content_digest(submission_path)
                if _ledger_contains(ledger, submission_path.name, submission_digest):
                    continue
                submission = receive_candidate_submission(attempt, submission_path)
                pre_validation_stop = current_stop_reason()
                if pre_validation_stop is not None:
                    return pre_validation_stop
                evaluation = evaluate_candidate_with_budget_checks(submission)
                feedback = _feedback_for_submission(submission, evaluation)
                feedback_record = provide_validation_feedback(attempt, feedback)
                report = evaluation.get("report")
                report_cost = getattr(getattr(report, "cost", None), "compile_cost", None)
                if isinstance(report_cost, (int, float)):
                    candidate_cost = float(report_cost)
                    total_candidate_cost = max(total_candidate_cost, candidate_cost)
                ledger.append(
                    {
                        "submission_id": submission.submission_id,
                        "submission_digest": submission_digest,
                        "workspace_commit": submission.workspace_commit,
                        "validation_status": "feedback_written",
                        "candidate_id": evaluation["candidate"].candidate_id,
                        "feedback_path": str(feedback_record.path),
                        "total_compile_cost": candidate_cost,
                        "timestamp": utcnow(),
                    }
                )
                feedback_count += 1
            except Exception as exc:
                if submission_digest is None:
                    submission_digest = _submission_digest_failure_marker(submission_path.name)
                if _ledger_contains(ledger, submission_path.name, submission_digest):
                    continue
                feedback = _safe_failure_feedback(submission_path.name, exc)
                feedback_record = provide_validation_feedback(attempt, feedback)
                ledger.append(
                    {
                        "submission_id": submission_path.name,
                        "submission_digest": submission_digest,
                        "validation_status": "evaluation_failed",
                        "feedback_path": str(feedback_record.path),
                        "error_class": exc.__class__.__name__,
                        "safe_error_message": feedback.summary["safe_error_message"],
                        "timestamp": utcnow(),
                    }
                )
                failed_count += 1
            evaluated_count += 1
            _write_submission_ledger(attempt, ledger)
            if pending_stop_reason is not None:
                return pending_stop_reason
            if evaluated_count >= compile_run.budget.max_candidates:
                return "candidate_limit"
            if (
                compile_run.budget.max_cost is not None
                and refresh_total_compile_cost() >= compile_run.budget.max_cost
            ):
                return "budget_exhausted"
        return None

    while stop_reason is None:
        stop_reason = current_stop_reason()
        if stop_reason is not None:
            break
        stop_reason = drain_ready_submissions()
        if stop_reason is not None:
            break
        handle = poll_agent_session(handle)
        if handle.status in {"completed", "failed", "timed_out", "stopped"}:
            terminal_reason = _close_reason_for_agent_status(handle.status)
            stop_reason = drain_ready_submissions() or terminal_reason
            break
        time.sleep(poll_interval_seconds)

    elapsed_seconds = time.monotonic() - started
    total_candidate_cost = refresh_total_compile_cost()
    close_reason = stop_reason or "ready_for_test"
    closed_attempt: ClosedAgentAttempt = close_agent_attempt(
        attempt,
        close_reason,
        session_timeout_seconds=(
            int(agent_timeout_seconds) if agent_timeout_seconds is not None else None
        ),
    )
    return {
        "compile_id": compile_run.compile_id,
        "attempt_id": attempt.attempt_id,
        "evaluated_submission_count": evaluated_count,
        "feedback_count": feedback_count,
        "skipped_submission_count": skipped_count,
        "failed_submission_count": failed_count,
        "stop_reason": close_reason,
        "elapsed_seconds": elapsed_seconds,
        "total_candidate_cost": total_candidate_cost,
        "closed_attempt_status": closed_attempt.status,
        "closed_attempt_final_commit": closed_attempt.final_commit,
        "ledger_path": _ledger_path(attempt),
    }
