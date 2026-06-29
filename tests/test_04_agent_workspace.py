from __future__ import annotations

import shutil
import time
from dataclasses import replace
from pathlib import Path

import pytest
from conftest import PrefixBroker, write_artifact

from darjeeling.agent_workspace import (
    advance_target_workspace_baseline,
    close_agent_attempt,
    create_agent_workspace,
    create_compile_run,
    launch_target_adaptation_agent,
    load_target_workspace,
    mount_readonly_inputs,
    provide_validation_feedback,
    receive_candidate_submission,
    run_compile_loop,
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


def test_agent_mount_contains_train_only_inputs(target_dir: Path, tmp_path: Path, now) -> None:
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
