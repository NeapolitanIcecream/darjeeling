from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import pytest
from conftest import PrefixBroker, write_artifact

from darjeeling import agent_workspace as agent_workspace_module
from darjeeling.agent_workspace import (
    advance_target_workspace_baseline,
    candidate_submission_ready,
    close_agent_attempt,
    create_agent_workspace,
    create_compile_run,
    launch_target_adaptation_agent,
    launch_target_adaptation_agent_async,
    load_target_workspace,
    mount_readonly_inputs,
    poll_agent_session,
    provide_validation_feedback,
    receive_candidate_submission,
    run_compile_loop,
    stop_agent_session,
    write_agent_brief,
)
from darjeeling.artifact_worker import build_protocol_docs
from darjeeling.candidate_evaluation import (
    build_agent_visible_report,
    compare_candidates,
    evaluate_candidate_on_test,
    evaluate_candidate_on_validation,
    finalize_report,
)
from darjeeling.errors import WorkspaceError
from darjeeling.model import (
    AgentAttemptOptions,
    AgentFeedback,
    AgentSessionHandle,
    AgentUsageLedger,
    AgentVisibleTelemetrySummary,
    ApprovalRecord,
    CompileBudget,
    CompileOptions,
    L4BaselineSummary,
    ReleaseBaseline,
    RoutingSettings,
    SnapshotOptions,
    TrainViewManifest,
    WorkspaceStore,
)
from darjeeling.release_runtime import create_release, create_release_without_artifacts
from darjeeling.snapshot_reference import build_snapshot, export_train_view_for_agent
from darjeeling.target_definition import export_agent_readonly_target_view, load_checked_target
from darjeeling.util import file_digest, new_id, stable_hash, utcnow, write_json


def _remove_tree_with_write_permissions(path: Path) -> None:
    if not path.exists():
        return
    for item in sorted(path.rglob("*"), key=lambda child: len(child.parts), reverse=True):
        item.chmod(0o700 if item.is_dir() else 0o600)
    path.chmod(0o700)
    shutil.rmtree(path)


def _train_export_digest(
    snapshot_id: str,
    snapshot_digest: str,
    path: Path,
    record_count: int,
    redaction_level: str = "raw",
) -> str:
    return stable_hash(
        {
            "view_kind": "agent_train_export",
            "snapshot_id": snapshot_id,
            "snapshot_digest": snapshot_digest,
            "view_digest": file_digest(path),
            "record_count": record_count,
            "redaction_level": redaction_level,
        }
    )


def _wait_for_path(path: Path, timeout_seconds: float = 5.0) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if path.exists():
            return
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {path}")


def _process_alive(pid: int) -> bool:
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid)],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return False
    return result.returncode == 0


def _accepted_release_from_attempt(
    definition,
    contract,
    base_release,
    snapshot,
    attempt,
    tmp_path: Path,
):
    (attempt.workspace_path / "runtime" / "l1" / "notes.txt").write_text(
        "baseline source\n", encoding="utf-8"
    )
    (attempt.workspace_path / "AGENT_BRIEF.md").write_text(
        "attempt only\n", encoding="utf-8"
    )
    (attempt.workspace_path / "readonly_inputs").mkdir(parents=True, exist_ok=True)
    (attempt.workspace_path / "readonly_inputs" / "train.json").write_text(
        "[]\n", encoding="utf-8"
    )
    artifact_dir = attempt.workspace_path / "submissions" / "c1" / "artifacts" / "l1"
    write_artifact(artifact_dir, definition.contract_hash, accept_prefixes=["a", "b"])
    submission = receive_candidate_submission(
        attempt, attempt.workspace_path / "submissions" / "c1"
    )
    validation = evaluate_candidate_on_validation(
        submission,
        definition,
        snapshot.snapshot,
        base_release,
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
        base_release,
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
            base_release,
            l4_baseline=L4BaselineSummary(
                base_release.release_id,
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
    compiled_release = create_release(
        validation["candidate"],
        snapshot.snapshot,
        base_release,
        final,
        approval,
        tmp_path / "artifacts",
    )
    visible = build_agent_visible_report(final, include_test_metrics=True)
    return closed, validation["candidate"], compiled_release, visible


def test_agent_mount_contains_train_only_inputs(
    target_dir: Path, tmp_path: Path, now, monkeypatch
) -> None:
    definition, contract, check = load_checked_target(target_dir)
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
    registry = __import__("darjeeling.model").model.ReleaseRegistry()
    release = create_release_without_artifacts(
        definition, contract, check, PrefixBroker(), RoutingSettings(), registry
    )
    workspace = load_target_workspace(
        definition.name, definition.contract_hash, WorkspaceStore(tmp_path / "workspaces")
    )
    compile_run = create_compile_run(
        definition,
        check,
        snapshot.snapshot,
        release,
        CompileBudget(),
        workspace,
        snapshot.reference_qualification,
        CompileOptions(),
    )
    with pytest.raises(WorkspaceError, match="snapshot"):
        create_compile_run(
            definition,
            check,
            replace(snapshot.snapshot, contract_hash="other"),
            release,
            CompileBudget(),
            workspace,
            snapshot.reference_qualification,
            CompileOptions(),
        )
    with pytest.raises(WorkspaceError, match="target check"):
        create_compile_run(
            definition,
            replace(check, status="fail"),
            snapshot.snapshot,
            release,
            CompileBudget(),
            workspace,
            snapshot.reference_qualification,
            CompileOptions(),
        )
    with pytest.raises(WorkspaceError, match="reference qualification"):
        create_compile_run(
            definition,
            check,
            snapshot.snapshot,
            release,
            CompileBudget(),
            workspace,
            replace(snapshot.reference_qualification, contract_hash="other"),
            CompileOptions(),
        )
    insufficient_reference = replace(snapshot.reference_qualification, status="insufficient")
    with pytest.raises(WorkspaceError, match="insufficient reference"):
        create_compile_run(
            definition,
            check,
            snapshot.snapshot,
            release,
            CompileBudget(),
            workspace,
            insufficient_reference,
            CompileOptions(),
        )
    approved_insufficient = create_compile_run(
        definition,
        check,
        snapshot.snapshot,
        release,
        CompileBudget(),
        workspace,
        insufficient_reference,
        CompileOptions(allow_insufficient_reference_qualification=True),
    )
    assert approved_insufficient.snapshot_id == snapshot.snapshot.snapshot_id
    with pytest.raises(WorkspaceError, match="target workspace"):
        create_agent_workspace(
            compile_run,
            replace(workspace, target_name="other"),
            AgentAttemptOptions(),
        )
    preclone_drift = workspace.workspace_path / "runtime" / "l1" / "preclone.py"
    preclone_drift.write_text("# drift\n", encoding="utf-8")
    with pytest.raises(WorkspaceError, match="baseline changed"):
        create_agent_workspace(compile_run, workspace, AgentAttemptOptions())
    preclone_drift.unlink()
    outside_file = tmp_path / "outside-preclone.txt"
    outside_file.write_text("outside\n", encoding="utf-8")
    symlink_path = workspace.workspace_path / "runtime" / "l1" / "outside-link"
    symlink_path.symlink_to(outside_file)
    with pytest.raises(WorkspaceError, match="symlinks"):
        create_agent_workspace(compile_run, workspace, AgentAttemptOptions())
    symlink_path.unlink()
    attempt = create_agent_workspace(compile_run, workspace, AgentAttemptOptions())
    target_view = export_agent_readonly_target_view(
        definition, attempt.workspace_path / "target_view"
    )
    train_view = export_train_view_for_agent(
        snapshot.snapshot,
        contract,
        __import__("darjeeling.model").model.AgentViewOptions(),
        attempt.workspace_path / "train_view",
    )
    stale_view = replace(
        train_view,
        snapshot_id="snapshot-other",
        export_digest=_train_export_digest(
            "snapshot-other",
            train_view.snapshot_digest,
            train_view.view_path,
            train_view.record_count,
        ),
    )
    with pytest.raises(WorkspaceError, match="attempt snapshot"):
        mount_readonly_inputs(
            attempt, target_view, stale_view, release, [], [], build_protocol_docs("v1")
        )
    with pytest.raises(WorkspaceError, match="target view"):
        mount_readonly_inputs(
            attempt,
            replace(target_view, target_name="other"),
            train_view,
            release,
            [],
            [],
            build_protocol_docs("v1"),
        )
    manifest = mount_readonly_inputs(
        attempt, target_view, train_view, release, [], [], build_protocol_docs("v1")
    )
    mounted_text = "\n".join(
        path.read_text(errors="ignore") for path in manifest.mount_path.rglob("*") if path.is_file()
    )
    assert (manifest.mount_path / "target" / "tests" / "contract_cases.yaml").exists()
    assert "r2" not in mounted_text
    assert "r4" not in mounted_text

    brief = write_agent_brief(attempt, compile_run, manifest, {})
    with pytest.raises(WorkspaceError, match="command is required"):
        launch_target_adaptation_agent(attempt, brief, {"command": []})

    outside_secret = tmp_path / "outside-secret.txt"
    outside_secret.write_text("secret\n", encoding="utf-8")
    sandbox_check = "\n".join(
        [
            "from pathlib import Path",
            "outside = Path(" + repr(str(outside_secret)) + ")",
            "try:",
            "    Path('readonly_inputs/train.json').write_text('bad')",
            "    raise SystemExit(11)",
            "except PermissionError:",
            "    pass",
            "try:",
            "    outside.read_text()",
            "    raise SystemExit(12)",
            "except PermissionError:",
            "    pass",
            "Path('journal/sandbox-ok.txt').write_text('ok')",
        ]
    )
    handle = launch_target_adaptation_agent(
        attempt,
        brief,
        {
            "command": ["/usr/bin/python3", "-c", sandbox_check],
            "protected_paths": [str(outside_secret)],
        },
    )
    assert handle.status == "completed"
    assert (attempt.workspace_path / "journal" / "sandbox-ok.txt").exists()
    assert "bad" not in (manifest.mount_path / "train.json").read_text()

    with pytest.raises(WorkspaceError, match="holdout reconstruction"):
        provide_validation_feedback(
            attempt,
            AgentFeedback(
                candidate_id="candidate-1",
                summary={},
                requirement_results=[],
                metrics={"nested": {"row_ids": ["r2"]}},
                safe_slice_summaries=[],
                latency_cost_summary={},
                raw_rows_included=False,
            ),
        )
    with pytest.raises(WorkspaceError, match="holdout reconstruction"):
        provide_validation_feedback(
            attempt,
            AgentFeedback(
                candidate_id="candidate-1",
                summary={"snapshot_record_id": "snap-row-1"},
                requirement_results=[],
                metrics={},
                safe_slice_summaries=[],
                latency_cost_summary={},
                raw_rows_included=False,
            ),
        )
    feedback_path = attempt.workspace_path / "journal" / "feedback-atomic.json"
    original_write_json = agent_workspace_module.write_json

    def fail_after_partial_write(path: Path, value) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{", encoding="utf-8")
        raise RuntimeError("partial feedback write")

    monkeypatch.setattr(agent_workspace_module, "write_json", fail_after_partial_write)
    with pytest.raises(RuntimeError, match="partial feedback write"):
        provide_validation_feedback(
            attempt,
            AgentFeedback(
                candidate_id="atomic",
                summary={},
                requirement_results=[],
                metrics={},
                safe_slice_summaries=[],
                latency_cost_summary={},
                raw_rows_included=False,
            ),
        )
    assert not feedback_path.exists()
    assert not list(feedback_path.parent.glob(".feedback-atomic.json.*.tmp"))
    monkeypatch.setattr(agent_workspace_module, "write_json", original_write_json)
    with pytest.raises(WorkspaceError, match="holdout reconstruction"):
        mount_readonly_inputs(
            attempt,
            target_view,
            train_view,
            release,
            [],
            [
                AgentVisibleTelemetrySummary(
                    definition.name,
                    release.release_id,
                    definition.contract_hash,
                    {"request_ids": ["runtime-1"]},
                    {},
                    None,
                    "v1",
                    utcnow(),
                )
            ],
            build_protocol_docs("v1"),
        )
    with pytest.raises(WorkspaceError, match="base release"):
        mount_readonly_inputs(
            attempt,
            target_view,
            train_view,
            release,
            [],
            [
                AgentVisibleTelemetrySummary(
                    definition.name,
                    "release-other",
                    definition.contract_hash,
                    {"requests": 1},
                    {},
                    None,
                    "v1",
                    utcnow(),
                )
            ],
            build_protocol_docs("v1"),
        )

    write_artifact(
        attempt.workspace_path / "runtime" / "prototype" / "artifacts" / "l1",
        definition.contract_hash,
        accept_prefixes=["a"],
    )
    with pytest.raises(WorkspaceError, match="submissions"):
        receive_candidate_submission(attempt, attempt.workspace_path / "runtime" / "prototype")

    outside_submission = tmp_path / "outside-submission"
    write_artifact(
        outside_submission / "artifacts" / "l1",
        definition.contract_hash,
        accept_prefixes=["a"],
    )
    (outside_submission / "READY").write_text("ready\n", encoding="utf-8")
    linked_submission = attempt.workspace_path / "submissions" / "linked"
    linked_submission.symlink_to(outside_submission, target_is_directory=True)
    assert not candidate_submission_ready(linked_submission)
    with pytest.raises(WorkspaceError, match="symlink"):
        receive_candidate_submission(attempt, linked_submission)
    linked_submission.unlink()

    outside_submissions = tmp_path / "outside-submissions"
    write_artifact(
        outside_submissions / "loop" / "artifacts" / "l1",
        definition.contract_hash,
        accept_prefixes=["a"],
    )
    (outside_submissions / "loop" / "READY").write_text("ready\n", encoding="utf-8")
    submissions_root = attempt.workspace_path / "submissions"
    submissions_root.rmdir()
    submissions_root.symlink_to(outside_submissions, target_is_directory=True)
    symlinked_root = run_compile_loop(
        compile_run,
        [attempt],
        1,
        None,
        lambda submission: AgentFeedback(
            candidate_id=submission.submission_id,
            summary={},
            requirement_results=[],
            metrics={},
            safe_slice_summaries=[],
            latency_cost_summary={},
            raw_rows_included=False,
        ),
    )
    assert symlinked_root["submissions"] == []
    submissions_root.unlink()
    submissions_root.mkdir()

    write_artifact(
        attempt.workspace_path / "submissions" / "loop" / "artifacts" / "l1",
        definition.contract_hash,
        accept_prefixes=["a"],
    )
    not_ready = run_compile_loop(
        compile_run,
        [attempt],
        1,
        None,
        lambda submission: AgentFeedback(
            candidate_id=submission.submission_id,
            summary={},
            requirement_results=[],
            metrics={},
            safe_slice_summaries=[],
            latency_cost_summary={},
            raw_rows_included=False,
        ),
    )
    assert not_ready["submissions"] == []
    (attempt.workspace_path / "submissions" / "loop" / "READY").write_text(
        "ready\n", encoding="utf-8"
    )
    with pytest.raises(WorkspaceError, match="feedback callback"):
        run_compile_loop(compile_run, [attempt], 1, None)
    with pytest.raises(WorkspaceError, match="compile run"):
        run_compile_loop(
            compile_run,
            [replace(attempt, compile_id="other")],
            1,
            None,
            lambda submission: AgentFeedback(
                candidate_id=submission.submission_id,
                summary={},
                requirement_results=[],
                metrics={},
                safe_slice_summaries=[],
                latency_cost_summary={},
                raw_rows_included=False,
            ),
        )
    limited = run_compile_loop(
        replace(compile_run, budget=CompileBudget(max_candidates=0)),
        [attempt],
        1,
        None,
        lambda submission: AgentFeedback(
            candidate_id=submission.submission_id,
            summary={},
            requirement_results=[],
            metrics={},
            safe_slice_summaries=[],
            latency_cost_summary={},
            raw_rows_included=False,
        ),
    )
    assert limited["submissions"] == []
    timed_out = run_compile_loop(
        compile_run,
        [attempt],
        1,
        0,
        lambda submission: AgentFeedback(
            candidate_id=submission.submission_id,
            summary={},
            requirement_results=[],
            metrics={},
            safe_slice_summaries=[],
            latency_cost_summary={},
            raw_rows_included=False,
        ),
    )
    assert timed_out["submissions"] == []
    write_artifact(
        attempt.workspace_path / "submissions" / "later" / "artifacts" / "l1",
        definition.contract_hash,
        accept_prefixes=["a"],
    )
    (attempt.workspace_path / "submissions" / "later" / "READY").write_text(
        "ready\n", encoding="utf-8"
    )

    def slow_feedback(submission):
        time.sleep(0.02)
        return AgentFeedback(
            candidate_id=f"slow-{submission.submission_id}",
            summary={},
            requirement_results=[],
            metrics={},
            safe_slice_summaries=[],
            latency_cost_summary={},
            raw_rows_included=False,
        )

    deadline_limited = run_compile_loop(compile_run, [attempt], 2, 0.01, slow_feedback)
    assert len(deadline_limited["submissions"]) == 1
    shutil.rmtree(attempt.workspace_path / "submissions" / "later")
    loop = run_compile_loop(
        compile_run,
        [attempt],
        1,
        None,
        lambda submission: AgentFeedback(
            candidate_id=submission.submission_id,
            summary={"submission_id": submission.submission_id},
            requirement_results=[],
            metrics={"sample_count": 0},
            safe_slice_summaries=[],
            latency_cost_summary={},
            raw_rows_included=False,
        ),
    )
    assert loop["submissions"][0].submission_id == "loop"
    assert (attempt.workspace_path / "journal" / "feedback-loop.json").exists()
    with pytest.raises(WorkspaceError, match="close reason"):
        close_agent_attempt(attempt, "typo")

    with pytest.raises(WorkspaceError, match="base release"):
        mount_readonly_inputs(
            attempt,
            target_view,
            train_view,
            replace(release, contract_hash="other"),
            [],
            [],
            build_protocol_docs("v1"),
        )

    bad_train_path = tmp_path / "bad_train.json"
    write_json(
        bad_train_path,
        [
                {
                    "input": {"text": "a:hidden"},
                    "reference_output": {"label": "a"},
                    "split_eligibility": ["train"],
                    "source_provenance": {"split": "validation"},
                }
        ],
    )
    with pytest.raises(WorkspaceError, match="export digest"):
        mount_readonly_inputs(
            attempt,
            target_view,
            replace(train_view, view_path=bad_train_path, record_count=1),
            release,
            [],
            [],
            build_protocol_docs("v1"),
        )
    with pytest.raises(WorkspaceError, match="attempt snapshot"):
        mount_readonly_inputs(
            attempt,
            target_view,
            replace(train_view, snapshot_id="snapshot-other"),
            release,
            [],
            [],
            build_protocol_docs("v1"),
        )

    no_provenance_path = tmp_path / "no_provenance_train.json"
    write_json(
        no_provenance_path,
        [
            {
                "input": {"text": "a:hidden"},
                "reference_output": {"label": "a"},
                "split_eligibility": ["train"],
            }
        ],
    )
    no_provenance_view = TrainViewManifest(
        snapshot_id=snapshot.snapshot.snapshot_id,
        snapshot_digest=snapshot.snapshot.snapshot_digest,
        view_path=no_provenance_path,
        record_count=1,
        redaction_level="raw",
        export_digest=_train_export_digest(
            snapshot.snapshot.snapshot_id,
            snapshot.snapshot.snapshot_digest,
            no_provenance_path,
            1,
        ),
    )
    with pytest.raises(WorkspaceError, match="source provenance"):
        mount_readonly_inputs(
            attempt,
            target_view,
            no_provenance_view,
            release,
            [],
            [],
            build_protocol_docs("v1"),
        )

    redacted_holdout_path = tmp_path / "redacted_holdout_train.json"
    write_json(
        redacted_holdout_path,
        [
            {
                "snapshot_record_id": "redacted",
                "input": {"text": "<redacted>"},
                "reference_output": {"label": "<redacted>"},
                "reference_source": "gold",
                "reference_version": "v1",
                "split_eligibility": ["validation_candidate"],
            }
        ],
    )
    redacted_holdout_view = TrainViewManifest(
        snapshot_id=snapshot.snapshot.snapshot_id,
        snapshot_digest=snapshot.snapshot.snapshot_digest,
        view_path=redacted_holdout_path,
        record_count=1,
        redaction_level="redacted",
        export_digest=_train_export_digest(
            snapshot.snapshot.snapshot_id,
            snapshot.snapshot.snapshot_digest,
            redacted_holdout_path,
            1,
            "redacted",
        ),
    )
    with pytest.raises(WorkspaceError, match="train eligible"):
        mount_readonly_inputs(
            attempt,
            target_view,
            redacted_holdout_view,
            release,
            [],
            [],
            build_protocol_docs("v1"),
        )

    holdout_view = replace(
        no_provenance_view,
        view_path=bad_train_path,
        export_digest=_train_export_digest(
            snapshot.snapshot.snapshot_id,
            snapshot.snapshot.snapshot_digest,
            bad_train_path,
            1,
        ),
    )
    with pytest.raises(WorkspaceError, match="hidden holdout"):
        mount_readonly_inputs(
            attempt,
            target_view,
            holdout_view,
            release,
            [],
            [],
            build_protocol_docs("v1"),
        )


def test_close_agent_attempt_stops_persisted_async_pid_when_process_map_is_empty(
    target_dir: Path, tmp_path: Path, now
) -> None:
    definition, contract, check = load_checked_target(target_dir)
    registry = __import__("darjeeling.model").model.ReleaseRegistry()
    release = create_release_without_artifacts(
        definition, contract, check, PrefixBroker(), RoutingSettings(), registry
    )
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
        release,
        CompileBudget(max_agent_seconds=5, max_candidates=1),
        workspace,
        snapshot.reference_qualification,
        CompileOptions(),
    )
    attempt = create_agent_workspace(compile_run, workspace, AgentAttemptOptions())
    target_view = export_agent_readonly_target_view(
        definition, attempt.workspace_path / "target_view"
    )
    train_view = export_train_view_for_agent(
        snapshot.snapshot,
        contract,
        __import__("darjeeling.model").model.AgentViewOptions(),
        attempt.workspace_path / "train_view",
    )
    manifest = mount_readonly_inputs(
        attempt, target_view, train_view, release, [], [], build_protocol_docs("v1")
    )
    brief = write_agent_brief(attempt, compile_run, manifest, {})
    launch_target_adaptation_agent_async(
        attempt,
        brief,
        {"command": ["/usr/bin/python3", "-c", "import time; time.sleep(30)"]},
    )
    process = agent_workspace_module._LIVE_AGENT_PROCESSES.pop(attempt.attempt_id)
    journal_session = attempt.workspace_path / "journal" / "agent_session.json"
    tampered_session = json.loads(journal_session.read_text())
    tampered_session["pid"] = 999999999
    journal_session.write_text(json.dumps(tampered_session), encoding="utf-8")
    try:
        closed = close_agent_attempt(attempt, "time_limit")
        process.wait(timeout=2)
        session = json.loads(
            journal_session.read_text()
        )
        assert closed.status == "closed"
        assert process.poll() is not None
        assert session["status"] == "timed_out"
        assert session["stop_reason"] == "time_limit"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        agent_workspace_module._LIVE_AGENT_PROCESSES.pop(attempt.attempt_id, None)


def _launch_process_test_agent(
    target_dir: Path, tmp_path: Path, now, agent_code: str
):
    definition, contract, check = load_checked_target(target_dir)
    registry = __import__("darjeeling.model").model.ReleaseRegistry()
    release = create_release_without_artifacts(
        definition, contract, check, PrefixBroker(), RoutingSettings(), registry
    )
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
        release,
        CompileBudget(max_agent_seconds=5, max_candidates=1),
        workspace,
        snapshot.reference_qualification,
        CompileOptions(),
    )
    attempt = create_agent_workspace(compile_run, workspace, AgentAttemptOptions())
    target_view = export_agent_readonly_target_view(
        definition, attempt.workspace_path / "target_view"
    )
    train_view = export_train_view_for_agent(
        snapshot.snapshot,
        contract,
        __import__("darjeeling.model").model.AgentViewOptions(),
        attempt.workspace_path / "train_view",
    )
    manifest = mount_readonly_inputs(
        attempt, target_view, train_view, release, [], [], build_protocol_docs("v1")
    )
    brief = write_agent_brief(attempt, compile_run, manifest, {})
    handle = launch_target_adaptation_agent_async(
        attempt,
        brief,
        {"command": ["/usr/bin/python3", "-c", agent_code]},
    )
    return attempt, handle


def test_stop_agent_session_kills_agent_process_group_children(
    target_dir: Path, tmp_path: Path, now
) -> None:
    agent_code = "\n".join(
        [
            "from pathlib import Path",
            "import os",
            "import signal",
            "import time",
            "pid = os.fork()",
            "if pid == 0:",
            "    signal.signal(signal.SIGTERM, signal.SIG_IGN)",
            "    time.sleep(1.0)",
            "    Path('journal/child-survived.txt').write_text('survived', encoding='utf-8')",
            "    time.sleep(30)",
            "    os._exit(0)",
            "Path('journal/child.pid').write_text(str(pid), encoding='utf-8')",
            "time.sleep(30)",
        ]
    )
    attempt, handle = _launch_process_test_agent(target_dir, tmp_path, now, agent_code)
    child_pid_path = attempt.workspace_path / "journal" / "child.pid"
    _wait_for_path(child_pid_path)
    child_pid = int(child_pid_path.read_text(encoding="utf-8"))
    try:
        stop_agent_session(handle, reason="time_limit", timeout_seconds=0.2)
        time.sleep(1.2)
        assert not (attempt.workspace_path / "journal" / "child-survived.txt").exists()
    finally:
        if _process_alive(child_pid):
            subprocess.run(["kill", "-9", str(child_pid)], check=False)
        agent_workspace_module._LIVE_AGENT_PROCESSES.pop(attempt.attempt_id, None)


def test_poll_agent_session_stops_completed_parent_process_group_children(
    target_dir: Path, tmp_path: Path, now
) -> None:
    agent_code = "\n".join(
        [
            "from pathlib import Path",
            "import os",
            "import signal",
            "import time",
            "pid = os.fork()",
            "if pid == 0:",
            "    signal.signal(signal.SIGTERM, signal.SIG_IGN)",
            "    time.sleep(1.0)",
            "    Path('journal/child-survived.txt').write_text('survived', encoding='utf-8')",
            "    time.sleep(30)",
            "    os._exit(0)",
            "Path('journal/child.pid').write_text(str(pid), encoding='utf-8')",
        ]
    )
    attempt, handle = _launch_process_test_agent(target_dir, tmp_path, now, agent_code)
    child_pid_path = attempt.workspace_path / "journal" / "child.pid"
    _wait_for_path(child_pid_path)
    child_pid = int(child_pid_path.read_text(encoding="utf-8"))
    try:
        updated = poll_agent_session(handle)
        time.sleep(1.2)
        assert updated.status == "completed"
        assert not (attempt.workspace_path / "journal" / "child-survived.txt").exists()
    finally:
        if _process_alive(child_pid):
            subprocess.run(["kill", "-9", str(child_pid)], check=False)
        agent_workspace_module._LIVE_AGENT_PROCESSES.pop(attempt.attempt_id, None)


def test_poll_agent_session_stops_orphaned_recorded_process_group_on_resume(
    target_dir: Path, tmp_path: Path, now
) -> None:
    agent_code = "\n".join(
        [
            "from pathlib import Path",
            "import os",
            "import signal",
            "import time",
            "pid = os.fork()",
            "if pid == 0:",
            "    signal.signal(signal.SIGTERM, signal.SIG_IGN)",
            "    time.sleep(1.0)",
            "    Path('journal/child-survived.txt').write_text('survived', encoding='utf-8')",
            "    time.sleep(30)",
            "    os._exit(0)",
            "Path('journal/child.pid').write_text(str(pid), encoding='utf-8')",
        ]
    )
    attempt, handle = _launch_process_test_agent(target_dir, tmp_path, now, agent_code)
    child_pid_path = attempt.workspace_path / "journal" / "child.pid"
    _wait_for_path(child_pid_path)
    child_pid = int(child_pid_path.read_text(encoding="utf-8"))
    process = agent_workspace_module._LIVE_AGENT_PROCESSES.pop(attempt.attempt_id)
    try:
        process.wait(timeout=2.0)
        updated = poll_agent_session(handle)
        time.sleep(1.2)
        assert updated.status == "failed"
        assert not (attempt.workspace_path / "journal" / "child-survived.txt").exists()
        session = json.loads(
            (attempt.workspace_path / "journal" / "agent_session.json").read_text()
        )
        assert session["status"] == "failed"
    finally:
        if _process_alive(child_pid):
            subprocess.run(["kill", "-9", str(child_pid)], check=False)
        agent_workspace_module._LIVE_AGENT_PROCESSES.pop(attempt.attempt_id, None)


def test_agent_launch_rejects_existing_core_session_without_journal(
    target_dir: Path, tmp_path: Path, now
) -> None:
    agent_code = "import time; time.sleep(30)"
    attempt, handle = _launch_process_test_agent(target_dir, tmp_path, now, agent_code)
    original_process = agent_workspace_module._LIVE_AGENT_PROCESSES[attempt.attempt_id]
    journal_session = attempt.workspace_path / "journal" / "agent_session.json"
    journal_session.unlink()
    second_handle = None
    try:
        with pytest.raises(WorkspaceError, match="agent already launched"):
            second_handle = launch_target_adaptation_agent_async(
                attempt,
                attempt.workspace_path / "AGENT_BRIEF.md",
                {"command": ["/usr/bin/python3", "-c", agent_code]},
            )
    finally:
        if second_handle is not None:
            stop_agent_session(second_handle, reason="stopped", timeout_seconds=0.2)
        else:
            stop_agent_session(handle, reason="stopped", timeout_seconds=0.2)
        if original_process.poll() is None:
            original_process.kill()
            original_process.wait()
        agent_workspace_module._LIVE_AGENT_PROCESSES.pop(attempt.attempt_id, None)


def test_stop_agent_session_kills_recorded_group_after_agent_parent_exits(
    target_dir: Path, tmp_path: Path, now
) -> None:
    agent_code = "\n".join(
        [
            "from pathlib import Path",
            "import os",
            "import signal",
            "import time",
            "pid = os.fork()",
            "if pid == 0:",
            "    signal.signal(signal.SIGTERM, signal.SIG_IGN)",
            "    time.sleep(1.0)",
            "    Path('journal/child-survived.txt').write_text('survived', encoding='utf-8')",
            "    time.sleep(30)",
            "    os._exit(0)",
            "Path('journal/child.pid').write_text(str(pid), encoding='utf-8')",
        ]
    )
    attempt, handle = _launch_process_test_agent(target_dir, tmp_path, now, agent_code)
    child_pid_path = attempt.workspace_path / "journal" / "child.pid"
    _wait_for_path(child_pid_path)
    child_pid = int(child_pid_path.read_text(encoding="utf-8"))
    process = agent_workspace_module._LIVE_AGENT_PROCESSES[attempt.attempt_id]
    try:
        process.wait(timeout=2.0)
        stop_agent_session(handle, reason="time_limit", timeout_seconds=0.2)
        time.sleep(1.2)
        assert not (attempt.workspace_path / "journal" / "child-survived.txt").exists()
    finally:
        if _process_alive(child_pid):
            subprocess.run(["kill", "-9", str(child_pid)], check=False)
        agent_workspace_module._LIVE_AGENT_PROCESSES.pop(attempt.attempt_id, None)


def test_stop_agent_session_does_not_signal_mismatched_persisted_pid(
    tmp_path: Path,
) -> None:
    unrelated = subprocess.Popen(
        ["/usr/bin/python3", "-c", "import time; time.sleep(30)"]
    )
    attempt_id = "attempt-stale-pid"
    attempt_path = tmp_path / "attempt"
    journal_session = attempt_path / "journal" / "agent_session.json"
    core_session = attempt_path.parent / "_core" / attempt_id / "agent_session.json"
    log_path = attempt_path / "journal" / "agent.log"
    record = {
        "attempt_id": attempt_id,
        "command": ["/usr/bin/python3", "-c", "import time; time.sleep(30)"],
        "sandbox_profile": None,
        "sandbox_mode": "portable_python",
        "status": "running",
        "pid": unrelated.pid,
        "process_group_id": unrelated.pid,
        "process_start_token": "stale-process-start-token",
        "started_at": utcnow(),
        "log_path": log_path,
        "timeout_seconds": 10,
    }
    write_json(journal_session, record)
    write_json(core_session, record)
    handle = AgentSessionHandle(
        attempt_id=attempt_id,
        status="running",
        pid=unrelated.pid,
        session_record_path=journal_session,
        timeout_seconds=10,
    )
    try:
        stop_agent_session(handle, reason="time_limit", timeout_seconds=0.1)
        assert unrelated.poll() is None
        session = json.loads(journal_session.read_text())
        assert session["status"] == "timed_out"
    finally:
        if unrelated.poll() is None:
            unrelated.kill()
            unrelated.wait()


def test_poll_agent_session_marks_missing_persisted_pid_failed(tmp_path: Path) -> None:
    attempt_id = "attempt-missing-pid"
    attempt_path = tmp_path / "attempt"
    journal_session = attempt_path / "journal" / "agent_session.json"
    core_session = attempt_path.parent / "_core" / attempt_id / "agent_session.json"
    log_path = attempt_path / "journal" / "agent.log"
    record = {
        "attempt_id": attempt_id,
        "command": ["/usr/bin/python3", "-c", "raise SystemExit(0)"],
        "sandbox_profile": None,
        "sandbox_mode": "portable_python",
        "status": "running",
        "pid": 999999999,
        "started_at": utcnow(),
        "log_path": log_path,
        "timeout_seconds": 10,
    }
    write_json(journal_session, record)
    write_json(core_session, record)
    handle = AgentSessionHandle(
        attempt_id=attempt_id,
        status="running",
        pid=999999999,
        session_record_path=journal_session,
        timeout_seconds=10,
    )

    updated = poll_agent_session(handle)

    session = json.loads(journal_session.read_text())
    core_record = json.loads(core_session.read_text())
    assert updated.status == "failed"
    assert session["status"] == "failed"
    assert core_record["status"] == "failed"
    assert session["returncode"] is None


def test_poll_agent_session_writes_core_record_atomically(
    tmp_path: Path, monkeypatch
) -> None:
    attempt_id = "attempt-atomic-core-session"
    attempt_path = tmp_path / "attempt"
    journal_session = attempt_path / "journal" / "agent_session.json"
    core_session = attempt_path.parent / "_core" / attempt_id / "agent_session.json"
    log_path = attempt_path / "journal" / "agent.log"
    record = {
        "attempt_id": attempt_id,
        "command": ["/usr/bin/python3", "-c", "raise SystemExit(0)"],
        "sandbox_profile": None,
        "sandbox_mode": "portable_python",
        "status": "running",
        "pid": 999999999,
        "started_at": utcnow(),
        "log_path": log_path,
        "timeout_seconds": 10,
    }
    write_json(journal_session, record)
    write_json(core_session, record)
    handle = AgentSessionHandle(
        attempt_id=attempt_id,
        status="running",
        pid=999999999,
        session_record_path=journal_session,
        timeout_seconds=10,
    )
    original_write_text = Path.write_text

    def fail_in_place_core_write(path: Path, data: str, *args, **kwargs) -> int:
        if path == core_session:
            original_write_text(path, "{", *args, **kwargs)
            raise OSError("simulated interrupted core write")
        return original_write_text(path, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_in_place_core_write)

    updated = poll_agent_session(handle)

    core_record = json.loads(core_session.read_text())
    journal_record = json.loads(journal_session.read_text())
    assert updated.status == "failed"
    assert core_record["status"] == "failed"
    assert core_record["returncode"] is None
    assert journal_record["status"] == "failed"


def test_poll_agent_session_persists_core_when_journal_record_is_unwritable(
    tmp_path: Path,
) -> None:
    attempt_id = "attempt-unwritable-journal"
    attempt_path = tmp_path / "attempt"
    journal_session = attempt_path / "journal" / "agent_session.json"
    core_session = attempt_path.parent / "_core" / attempt_id / "agent_session.json"
    log_path = attempt_path / "journal" / "agent.log"
    record = {
        "attempt_id": attempt_id,
        "command": ["/usr/bin/python3", "-c", "raise SystemExit(0)"],
        "sandbox_profile": None,
        "sandbox_mode": "portable_python",
        "status": "running",
        "pid": 999999999,
        "started_at": utcnow(),
        "log_path": log_path,
        "timeout_seconds": 10,
    }
    journal_session.parent.mkdir(parents=True)
    journal_session.mkdir()
    write_json(core_session, record)
    handle = AgentSessionHandle(
        attempt_id=attempt_id,
        status="running",
        pid=999999999,
        session_record_path=journal_session,
        timeout_seconds=1000,
    )

    updated = poll_agent_session(handle)

    core_record = json.loads(core_session.read_text())
    assert updated.status == "failed"
    assert core_record["status"] == "failed"
    assert core_record["timeout_seconds"] == 10
    assert journal_session.is_dir()


def test_baseline_advances_only_with_accepted_release_or_explicit_carry_forward(
    target_dir: Path, tmp_path: Path, now
) -> None:
    definition, contract, check = load_checked_target(target_dir)
    registry = __import__("darjeeling.model").model.ReleaseRegistry()
    release = create_release_without_artifacts(
        definition, contract, check, PrefixBroker(), RoutingSettings(), registry
    )
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
    reloaded = load_target_workspace(
        definition.name, definition.contract_hash, WorkspaceStore(tmp_path / "workspaces")
    )
    assert reloaded.baseline_commit == workspace.baseline_commit
    drift_file = workspace.workspace_path / "runtime" / "l1" / "manual-edit.py"
    drift_file.write_text("# drift\n", encoding="utf-8")
    with pytest.raises(WorkspaceError, match="baseline drifted"):
        load_target_workspace(
            definition.name, definition.contract_hash, WorkspaceStore(tmp_path / "workspaces")
        )
    drift_file.unlink()
    compile_run = create_compile_run(
        definition,
        check,
        snapshot.snapshot,
        release,
        CompileBudget(),
        workspace,
        snapshot.reference_qualification,
        CompileOptions(),
    )
    attempt = create_agent_workspace(compile_run, workspace, AgentAttemptOptions())
    target_view = export_agent_readonly_target_view(
        definition, attempt.workspace_path / "target_view"
    )
    train_view = export_train_view_for_agent(
        snapshot.snapshot,
        contract,
        __import__("darjeeling.model").model.AgentViewOptions(),
        attempt.workspace_path / "train_view",
    )
    closed, candidate, compiled_release, visible_report = _accepted_release_from_attempt(
        definition, contract, release, snapshot, attempt, tmp_path
    )
    write_artifact(
        attempt.workspace_path / "submissions" / "late" / "artifacts" / "l1",
        definition.contract_hash,
        accept_prefixes=["a"],
    )
    with pytest.raises(WorkspaceError, match="submissions are closed"):
        receive_candidate_submission(attempt, attempt.workspace_path / "submissions" / "late")
    with pytest.raises(WorkspaceError, match="compiled release"):
        advance_target_workspace_baseline(workspace, closed, release, "accepted_release")
    with pytest.raises(WorkspaceError, match="closed attempt"):
        advance_target_workspace_baseline(
            workspace,
            replace(closed, attempt_id="other"),
            compiled_release,
            "accepted_release",
            accepted_candidate=candidate,
        )
    with pytest.raises(WorkspaceError, match="target workspace"):
        advance_target_workspace_baseline(
            workspace,
            replace(closed, target_name="other"),
            None,
            "explicit_carry_forward",
        )
    with pytest.raises(WorkspaceError, match="scope mismatch"):
        advance_target_workspace_baseline(
            workspace,
            closed,
            replace(compiled_release, contract_hash="other"),
            "accepted_release",
            accepted_candidate=candidate,
        )
    with pytest.raises(WorkspaceError, match="scope mismatch"):
        advance_target_workspace_baseline(
            workspace,
            closed,
            compiled_release,
            "accepted_release",
            accepted_candidate=replace(candidate, target_name="other"),
        )
    manifest = mount_readonly_inputs(
        attempt,
        target_view,
        train_view,
        release,
        [visible_report],
        [],
        build_protocol_docs("v1"),
    )
    mounted_reports = (manifest.mount_path / "agent_visible_reports.json").read_text()
    assert "test_metrics_included" in mounted_reports
    notes_path = attempt.workspace_path / "runtime" / "l1" / "notes.txt"
    original_notes = notes_path.read_text(encoding="utf-8")
    notes_path.write_text("post-close mutation\n", encoding="utf-8")
    with pytest.raises(WorkspaceError, match="final commit"):
        advance_target_workspace_baseline(
            workspace,
            closed,
            compiled_release,
            "accepted_release",
            accepted_candidate=candidate,
        )
    notes_path.write_text(original_notes, encoding="utf-8")
    update = advance_target_workspace_baseline(
        workspace,
        closed,
        compiled_release,
        "accepted_release",
        accepted_candidate=candidate,
    )
    advanced = load_target_workspace(
        definition.name, definition.contract_hash, WorkspaceStore(tmp_path / "workspaces")
    )
    assert update.previous_commit == workspace.baseline_commit
    assert advanced.baseline_commit == update.new_commit
    assert update.source_release_id == compiled_release.release_id
    assert (advanced.workspace_path / "runtime" / "l1" / "notes.txt").exists()
    assert not (advanced.workspace_path / "readonly_inputs").exists()
    assert not (advanced.workspace_path / "AGENT_BRIEF.md").exists()
    assert not (advanced.workspace_path / "submissions").exists()


def test_agent_launch_protects_core_paths_for_in_repo_workspace(
    target_dir: Path, now
) -> None:
    repo_root = Path.cwd()
    workspace_root = repo_root / ".test-workspaces" / new_id("workspace")
    try:
        definition, contract, check = load_checked_target(target_dir)
        registry = __import__("darjeeling.model").model.ReleaseRegistry()
        release = create_release_without_artifacts(
            definition, contract, check, PrefixBroker(), RoutingSettings(), registry
        )
        snapshot = build_snapshot(
            definition,
            contract,
            definition.data_config,
            None,
            [],
            PrefixBroker(),
            now,
            SnapshotOptions(storage_root=workspace_root / "snapshots"),
        )
        workspace = load_target_workspace(
            definition.name, definition.contract_hash, WorkspaceStore(workspace_root)
        )
        compile_run = create_compile_run(
            definition,
            check,
            snapshot.snapshot,
            release,
            CompileBudget(),
            workspace,
            snapshot.reference_qualification,
            CompileOptions(),
        )
        attempt = create_agent_workspace(compile_run, workspace, AgentAttemptOptions())
        target_view = export_agent_readonly_target_view(
            definition, attempt.workspace_path / "target_view"
        )
        train_view = export_train_view_for_agent(
            snapshot.snapshot,
            contract,
            __import__("darjeeling.model").model.AgentViewOptions(),
            attempt.workspace_path / "train_view",
        )
        manifest = mount_readonly_inputs(
            attempt, target_view, train_view, release, [], [], build_protocol_docs("v1")
        )
        brief = write_agent_brief(attempt, compile_run, manifest, {})
        core_file = repo_root / "pyproject.toml"
        sandbox_check = "\n".join(
            [
                "from pathlib import Path",
                "core_file = Path(" + repr(str(core_file)) + ")",
                "try:",
                "    core_file.read_text()",
                "    raise SystemExit(21)",
                "except PermissionError:",
                "    pass",
                "Path('journal/repo-sandbox-ok.txt').write_text('ok')",
            ]
        )
        handle = launch_target_adaptation_agent(
            attempt,
            brief,
            {"command": ["/usr/bin/python3", "-c", sandbox_check]},
        )
        assert handle.status == "completed"
        assert (attempt.workspace_path / "journal" / "repo-sandbox-ok.txt").exists()
    finally:
        _remove_tree_with_write_permissions(workspace_root)
        if workspace_root.parent.exists() and not any(workspace_root.parent.iterdir()):
            workspace_root.parent.rmdir()
