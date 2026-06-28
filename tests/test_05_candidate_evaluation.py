from __future__ import annotations

from dataclasses import asdict, is_dataclass, replace
from pathlib import Path

import pytest
from conftest import PrefixBroker, write_artifact

from darjeeling.agent_workspace import (
    close_agent_attempt,
    create_agent_workspace,
    create_compile_run,
    load_target_workspace,
    receive_candidate_submission,
)
from darjeeling.artifact_worker import freeze_artifact_package, read_artifact_manifest
from darjeeling.candidate_evaluation import (
    build_agent_visible_report,
    check_candidate_requirements,
    compare_candidates,
    compute_generalization_summary,
    evaluate_candidate_on_test,
    evaluate_candidate_on_validation,
    evaluate_changed_layer_ablation,
    evaluate_fault_fallback,
    evaluate_full_cascade,
    evaluate_residual_layer,
    evaluate_standalone_layer,
    finalize_report,
    freeze_candidate,
    validate_candidate_manifest,
)
from darjeeling.errors import ArtifactError, EvaluationError, ReleaseError
from darjeeling.model import (
    AgentAttemptOptions,
    AgentUsageLedger,
    CompileBudget,
    CompileOptions,
    L4BaselineSummary,
    MetricSummary,
    ReleaseBaseline,
    RoutingSettings,
    SnapshotOptions,
    SnapshotRecord,
    TargetRequirements,
    WorkspaceStore,
)
from darjeeling.release_runtime import create_release, create_release_without_artifacts
from darjeeling.snapshot_reference import build_snapshot, load_snapshot_records, load_snapshot_view
from darjeeling.target_definition import load_checked_target
from darjeeling.util import new_id, utcnow


def _submitted_candidate(target_dir: Path, tmp_path: Path, now):
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
        CompileBudget(),
        workspace,
        snapshot.reference_qualification,
        CompileOptions(),
    )
    attempt = create_agent_workspace(compile_run, workspace, AgentAttemptOptions())
    artifact_dir = attempt.workspace_path / "submissions" / "c1" / "artifacts" / "l1"
    write_artifact(artifact_dir, definition.contract_hash, accept_prefixes=["a", "b"])
    submission = receive_candidate_submission(
        attempt, attempt.workspace_path / "submissions" / "c1"
    )
    return definition, contract, release, snapshot, attempt, submission


def _cold_start_baseline(definition, release, snapshot) -> ReleaseBaseline:
    return ReleaseBaseline(
        release=release,
        l4_baseline=L4BaselineSummary(
            release.release_id,
            definition.name,
            definition.contract_hash,
            {},
            [],
            snapshot.reference_qualification,
            {},
            {},
        ),
    )


_FORBIDDEN_PUBLIC_REPORT_KEYS = {
    "record",
    "rows",
    "plan",
    "order",
    "request_id",
    "request_ids",
    "snapshot_record_id",
    "source_record_id",
    "split_group_key",
    "split_group_keys",
    "input",
    "normalized_input",
    "output",
    "reference_output",
    "fallback_output",
}


def _assert_no_private_evaluation_material(value) -> None:
    assert not isinstance(value, SnapshotRecord)
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        leaked = _FORBIDDEN_PUBLIC_REPORT_KEYS & set(value)
        assert leaked == set()
        for nested in value.values():
            _assert_no_private_evaluation_material(nested)
    elif isinstance(value, list | tuple | set):
        for nested in value:
            _assert_no_private_evaluation_material(nested)


def test_validation_and_test_reports_have_no_decision(
    target_dir: Path, tmp_path: Path, now
) -> None:
    definition, contract, release, snapshot, attempt, submission = _submitted_candidate(
        target_dir, tmp_path, now
    )
    validation = evaluate_candidate_on_validation(
        submission,
        definition,
        snapshot.snapshot,
        release,
        snapshot.reference_qualification,
        AgentUsageLedger(),
        snapshot.reference_usage,
        None,
        None,
        {"serving_l4_cost": 1.0},
        {"contract": contract, "artifact_store": tmp_path / "artifacts"},
    )
    assert validation["report"].report_stage == "validation"
    assert validation["report"].decision is None
    closed = close_agent_attempt(attempt, "ready_for_test")
    with pytest.raises(EvaluationError, match="candidate scope"):
        evaluate_candidate_on_test(
            replace(validation["candidate"], snapshot_id="other-snapshot"),
            closed,
            definition,
            snapshot.snapshot,
            release,
            snapshot.reference_qualification,
            AgentUsageLedger(),
            snapshot.reference_usage,
            None,
            None,
            {"serving_l4_cost": 1.0},
            {"contract": contract, "test_results_visible": True},
        )
    with pytest.raises(EvaluationError, match="test snapshot lineage"):
        evaluate_candidate_on_test(
            validation["candidate"],
            replace(closed, snapshot_id="other-snapshot"),
            definition,
            snapshot.snapshot,
            release,
            snapshot.reference_qualification,
            AgentUsageLedger(),
            snapshot.reference_usage,
            None,
            None,
            {"serving_l4_cost": 1.0},
            {"contract": contract, "test_results_visible": True},
        )
    with pytest.raises(EvaluationError, match="lineage"):
        evaluate_candidate_on_test(
            validation["candidate"],
            replace(closed, attempt_id="other"),
            definition,
            snapshot.snapshot,
            release,
            snapshot.reference_qualification,
            AgentUsageLedger(),
            snapshot.reference_usage,
            None,
            None,
            {"serving_l4_cost": 1.0},
            {"contract": contract, "test_results_visible": True},
        )
    post_close_file = attempt.workspace_path / "runtime" / "l1" / "post-close.py"
    post_close_file.write_text("# changed after close\n", encoding="utf-8")
    with pytest.raises(EvaluationError, match="changed after close"):
        evaluate_candidate_on_test(
            validation["candidate"],
            closed,
            definition,
            snapshot.snapshot,
            release,
            snapshot.reference_qualification,
            AgentUsageLedger(),
            snapshot.reference_usage,
            None,
            None,
            {"serving_l4_cost": 1.0},
            {"contract": contract, "test_results_visible": True},
        )
    post_close_file.unlink()
    test = evaluate_candidate_on_test(
        validation["candidate"],
        closed,
        definition,
        snapshot.snapshot,
        release,
        snapshot.reference_qualification,
        AgentUsageLedger(),
        snapshot.reference_usage,
        None,
        None,
        {"serving_l4_cost": 1.0},
        {"contract": contract, "test_results_visible": True},
    )
    assert test["report"].report_stage == "test"
    assert test["report"].decision is None
    assert test["report"].holdout_consumption is not None


def test_reference_qualification_scope_must_match_evaluation_scope(
    target_dir: Path, tmp_path: Path, now
) -> None:
    definition, contract, release, snapshot, attempt, submission = _submitted_candidate(
        target_dir, tmp_path, now
    )
    with pytest.raises(EvaluationError, match="reference qualification scope"):
        evaluate_candidate_on_validation(
            submission,
            definition,
            snapshot.snapshot,
            release,
            replace(snapshot.reference_qualification, target_name="other-target"),
            AgentUsageLedger(),
            snapshot.reference_usage,
            None,
            None,
            {"serving_l4_cost": 1.0},
            {"contract": contract, "artifact_store": tmp_path / "artifacts"},
        )

    validation = evaluate_candidate_on_validation(
        submission,
        definition,
        snapshot.snapshot,
        release,
        snapshot.reference_qualification,
        AgentUsageLedger(),
        snapshot.reference_usage,
        None,
        None,
        {"serving_l4_cost": 1.0},
        {"contract": contract, "artifact_store": tmp_path / "artifacts"},
    )
    closed = close_agent_attempt(attempt, "ready_for_test")
    with pytest.raises(EvaluationError, match="reference qualification scope"):
        evaluate_candidate_on_test(
            validation["candidate"],
            closed,
            definition,
            snapshot.snapshot,
            release,
            replace(snapshot.reference_qualification, contract_hash="other-contract"),
            AgentUsageLedger(),
            snapshot.reference_usage,
            None,
            None,
            {"serving_l4_cost": 1.0},
            {"contract": contract, "test_results_visible": True},
        )


def test_base_release_scope_must_match_evaluation_scope(
    target_dir: Path, tmp_path: Path, now
) -> None:
    definition, contract, release, snapshot, attempt, submission = _submitted_candidate(
        target_dir, tmp_path, now
    )
    with pytest.raises(EvaluationError, match="base release scope"):
        evaluate_candidate_on_validation(
            submission,
            definition,
            snapshot.snapshot,
            replace(release, target_name="other-target"),
            snapshot.reference_qualification,
            AgentUsageLedger(),
            snapshot.reference_usage,
            None,
            None,
            {"serving_l4_cost": 1.0},
            {"contract": contract, "artifact_store": tmp_path / "artifacts"},
        )

    validation = evaluate_candidate_on_validation(
        submission,
        definition,
        snapshot.snapshot,
        release,
        snapshot.reference_qualification,
        AgentUsageLedger(),
        snapshot.reference_usage,
        None,
        None,
        {"serving_l4_cost": 1.0},
        {"contract": contract, "artifact_store": tmp_path / "artifacts"},
    )
    closed = close_agent_attempt(attempt, "ready_for_test")
    with pytest.raises(EvaluationError, match="base release scope"):
        evaluate_candidate_on_test(
            validation["candidate"],
            closed,
            definition,
            snapshot.snapshot,
            replace(release, contract_hash="other-contract"),
            snapshot.reference_qualification,
            AgentUsageLedger(),
            snapshot.reference_usage,
            None,
            None,
            {"serving_l4_cost": 1.0},
            {"contract": contract, "test_results_visible": True},
        )


def test_validation_report_cannot_be_release_eligible_or_finalized_for_release(
    target_dir: Path, tmp_path: Path, now
) -> None:
    definition, contract, release, snapshot, _attempt, submission = _submitted_candidate(
        target_dir, tmp_path, now
    )
    validation = evaluate_candidate_on_validation(
        submission,
        definition,
        snapshot.snapshot,
        release,
        snapshot.reference_qualification,
        AgentUsageLedger(),
        snapshot.reference_usage,
        None,
        None,
        {"serving_l4_cost": 1.0},
        {"contract": contract, "artifact_store": tmp_path / "artifacts"},
    )

    decision = compare_candidates(
        [validation["report"]],
        _cold_start_baseline(definition, release, snapshot),
        {"requirements": definition.requirements},
    )

    assert decision.status == "insufficient_evidence"
    assert decision.release_blockers == ["test_report_required"]
    forced_eligible = replace(
        decision,
        status="eligible_for_release",
        release_blockers=[],
        selected_operating_point={"enabled_layers": ["L1", "L2", "L3"]},
    )
    with pytest.raises(EvaluationError, match="test report"):
        finalize_report(validation["report"], forced_eligible)

    final = finalize_report(validation["report"], decision)
    visible = build_agent_visible_report(final, include_test_metrics=False)
    assert visible.validation_metrics == validation["report"].metrics
    assert visible.test_metrics is None


def test_hidden_failed_test_report_consumes_holdout_rows(
    target_dir: Path, tmp_path: Path, now
) -> None:
    definition, contract, release, snapshot, attempt, submission = _submitted_candidate(
        target_dir, tmp_path, now
    )
    worker_path = submission.submission_path / "artifacts" / "l1" / "worker.py"
    worker_path.write_text(
        """
import json
import sys

json.loads(sys.stdin.readline())
print(json.dumps({"decision": "abstain", "confidence": 0.1, "reason": "outside"}))
""".lstrip(),
        encoding="utf-8",
    )
    validation = evaluate_candidate_on_validation(
        submission,
        definition,
        snapshot.snapshot,
        release,
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
        release,
        snapshot.reference_qualification,
        AgentUsageLedger(),
        snapshot.reference_usage,
        None,
        None,
        {"serving_l4_cost": 1.0},
        {"contract": contract, "test_results_visible": False},
    )

    metric = test["report"].metrics["local"]
    assert metric["accepted_count"] == 0
    assert test["report"].metrics["l4_fallback_share"] == 1.0
    assert test["report"].holdout_consumption is not None
    assert test["report"].holdout_consumption.visible_to == "external_report"
    assert len(test["report"].holdout_consumption.record_ids) == snapshot.snapshot.test_count
    decision = compare_candidates(
        [test["report"]],
        _cold_start_baseline(definition, release, snapshot),
        {"requirements": definition.requirements},
    )
    assert decision.status == "insufficient_evidence"
    assert "min_accepted_samples" in decision.release_blockers


def test_compiled_baseline_must_improve_selected_objective(
    target_dir: Path, tmp_path: Path, now
) -> None:
    definition, contract, release, snapshot, attempt, submission = _submitted_candidate(
        target_dir, tmp_path, now
    )
    write_artifact(submission.submission_path / "artifacts" / "l1", definition.contract_hash)
    validation = evaluate_candidate_on_validation(
        submission,
        definition,
        snapshot.snapshot,
        release,
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
        release,
        snapshot.reference_qualification,
        AgentUsageLedger(),
        snapshot.reference_usage,
        None,
        None,
        {"serving_l4_cost": 1.0},
        {"contract": contract, "test_results_visible": False},
    )
    report = test["report"]
    assert report.metrics["l4_fallback_share"] == 0.5
    assert report.latency["l4_fallback_cost"] > 0
    assert report.cost.serving_local_compute_cost == report.latency["serving_local_compute_cost"]
    assert report.latency["cascade_cost"] > report.cost.serving_local_compute_cost
    assert report.cost.saving_per_1000_requests == pytest.approx(
        1.0 - report.latency["cascade_cost"]
    )
    assert any("l4_fallback_cost" in note for note in report.cost.notes)
    stronger_baseline_report = replace(
        report,
        candidate_id="baseline-candidate",
        metrics={
            **report.metrics,
            "local": {
                **report.metrics["local"],
                "accepted_count": 2,
                "correct_accept_count": 2,
                "precision": 1.0,
                "coverage": 1.0,
            },
        },
    )
    compiled_release = replace(
        release,
        candidate_id="baseline-candidate",
        snapshot_id=stronger_baseline_report.snapshot_id,
        report_id=stronger_baseline_report.report_id,
    )

    coverage_decision = compare_candidates(
        [report],
        ReleaseBaseline(release=compiled_release, report=stronger_baseline_report),
        {"requirements": definition.requirements},
    )
    assert "baseline_coverage_improvement" in coverage_decision.release_blockers
    with pytest.raises(EvaluationError, match="baseline report"):
        compare_candidates(
            [report],
            ReleaseBaseline(
                release=replace(compiled_release, report_id="other-report"),
                report=stronger_baseline_report,
            ),
            {"requirements": definition.requirements},
        )
    cold_baseline = _cold_start_baseline(definition, release, snapshot)
    assert cold_baseline.l4_baseline is not None
    with pytest.raises(EvaluationError, match="baseline L4 summary"):
        compare_candidates(
            [report],
            ReleaseBaseline(
                release=release,
                l4_baseline=replace(cold_baseline.l4_baseline, release_id="other-release"),
            ),
            {"requirements": definition.requirements},
        )

    lower_latency_baseline = replace(stronger_baseline_report, latency={"p95_latency_ms": 0.0})
    latency_decision = compare_candidates(
        [report],
        ReleaseBaseline(release=compiled_release, report=lower_latency_baseline),
        {"requirements": definition.requirements, "optimize": "latency"},
    )
    assert "baseline_latency_improvement" in latency_decision.release_blockers

    lower_cost_baseline = replace(
        stronger_baseline_report,
        latency={**stronger_baseline_report.latency, "cascade_cost": 0.0},
        cost=replace(report.cost, serving_local_compute_cost=0.0),
    )
    cost_decision = compare_candidates(
        [report],
        ReleaseBaseline(release=compiled_release, report=lower_cost_baseline),
        {"requirements": definition.requirements, "optimize": "cost"},
    )
    assert "baseline_cost_improvement" in cost_decision.release_blockers


def test_compare_candidates_filters_failures_before_objective_selection(
    target_dir: Path, tmp_path: Path, now
) -> None:
    definition, contract, release, snapshot, attempt, submission = _submitted_candidate(
        target_dir, tmp_path, now
    )
    validation = evaluate_candidate_on_validation(
        submission,
        definition,
        snapshot.snapshot,
        release,
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
        release,
        snapshot.reference_qualification,
        AgentUsageLedger(),
        snapshot.reference_usage,
        None,
        None,
        {"serving_l4_cost": 1.0},
        {"contract": contract, "test_results_visible": False},
    )
    report = test["report"]
    passing_low_coverage = replace(
        report,
        candidate_id="passing-low-coverage",
        metrics={
            **report.metrics,
            "local": {
                **report.metrics["local"],
                "accepted_count": 1,
                "correct_accept_count": 1,
                "wrong_accept_count": 0,
                "precision": 1.0,
                "coverage": 0.5,
                "wrong_accept_rate": 0.0,
                "precision_lower_bound": 0.95,
                "wrong_accept_upper_bound": 0.05,
            },
        },
    )
    failing_high_coverage = replace(
        report,
        candidate_id="failing-high-coverage",
        metrics={
            **report.metrics,
            "local": {
                **report.metrics["local"],
                "accepted_count": 2,
                "correct_accept_count": 1,
                "wrong_accept_count": 1,
                "precision": 0.5,
                "coverage": 1.0,
                "wrong_accept_rate": 0.5,
                "precision_lower_bound": 0.45,
                "wrong_accept_upper_bound": 0.55,
            },
        },
    )

    decision = compare_candidates(
        [failing_high_coverage, passing_low_coverage],
        _cold_start_baseline(definition, release, snapshot),
        {"requirements": definition.requirements},
    )
    assert decision.status == "eligible_for_release"
    assert decision.candidate_id == "passing-low-coverage"
    with pytest.raises(EvaluationError, match="mixed target"):
        compare_candidates(
            [passing_low_coverage, replace(passing_low_coverage, snapshot_id="other")],
            _cold_start_baseline(definition, release, snapshot),
            {"requirements": definition.requirements},
        )


def test_compare_candidates_uses_requested_latency_and_cost_objectives(
    target_dir: Path, tmp_path: Path, now
) -> None:
    definition, contract, release, snapshot, attempt, submission = _submitted_candidate(
        target_dir, tmp_path, now
    )
    validation = evaluate_candidate_on_validation(
        submission,
        definition,
        snapshot.snapshot,
        release,
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
        release,
        snapshot.reference_qualification,
        AgentUsageLedger(),
        snapshot.reference_usage,
        None,
        None,
        {"serving_l4_cost": 1.0},
        {"contract": contract, "test_results_visible": False},
    )
    report = test["report"]
    slow_expensive = replace(
        report,
        candidate_id="slow-expensive",
        latency={**report.latency, "p95_latency_ms": 100.0, "cascade_cost": 10.0},
        cost=replace(report.cost, serving_local_compute_cost=10.0),
    )
    fast = replace(
        report,
        candidate_id="fast",
        latency={**report.latency, "p95_latency_ms": 1.0, "cascade_cost": 5.0},
        cost=replace(report.cost, serving_local_compute_cost=5.0),
    )
    cheap = replace(
        report,
        candidate_id="cheap",
        latency={**report.latency, "p95_latency_ms": 50.0, "cascade_cost": 1.0},
        cost=replace(report.cost, serving_local_compute_cost=1.0),
    )

    latency_decision = compare_candidates(
        [slow_expensive, fast],
        _cold_start_baseline(definition, release, snapshot),
        {"requirements": definition.requirements, "optimize": "latency"},
    )
    assert latency_decision.candidate_id == "fast"

    cost_decision = compare_candidates(
        [slow_expensive, cheap],
        _cold_start_baseline(definition, release, snapshot),
        {"requirements": definition.requirements, "optimize": "cost"},
    )
    assert cost_decision.candidate_id == "cheap"
    low_local_high_fallback = replace(
        report,
        candidate_id="low-local-high-fallback",
        latency={**report.latency, "cascade_cost": 10.0},
        cost=replace(report.cost, serving_local_compute_cost=0.1),
    )
    high_local_low_fallback = replace(
        report,
        candidate_id="high-local-low-fallback",
        latency={**report.latency, "cascade_cost": 2.0},
        cost=replace(report.cost, serving_local_compute_cost=1.0),
    )
    total_cost_decision = compare_candidates(
        [low_local_high_fallback, high_local_low_fallback],
        _cold_start_baseline(definition, release, snapshot),
        {"requirements": definition.requirements, "optimize": "cost"},
    )
    assert total_cost_decision.candidate_id == "high-local-low-fallback"


def test_candidate_digest_is_scoped_to_snapshot_and_base_release(
    target_dir: Path, tmp_path: Path, now
) -> None:
    definition, _contract, release, snapshot, _attempt, submission = _submitted_candidate(
        target_dir, tmp_path, now
    )
    first = freeze_candidate(
        submission,
        release,
        definition,
        tmp_path / "artifacts",
        snapshot.snapshot.snapshot_digest,
        snapshot.snapshot.snapshot_id,
    )
    second = freeze_candidate(
        submission,
        release,
        definition,
        tmp_path / "artifacts",
        "other-snapshot-digest",
        "other-snapshot",
    )
    other_base = replace(release, release_id="other-release")
    third = freeze_candidate(
        submission,
        other_base,
        definition,
        tmp_path / "artifacts",
        snapshot.snapshot.snapshot_digest,
        snapshot.snapshot.snapshot_id,
    )
    assert first.snapshot_id == snapshot.snapshot.snapshot_id
    assert first.digest != second.digest
    assert first.digest != third.digest


def test_residual_and_ablation_diagnostics_use_boundary_inputs(
    target_dir: Path, tmp_path: Path, now
) -> None:
    definition, contract, release, snapshot, _attempt, submission = _submitted_candidate(
        target_dir, tmp_path, now
    )
    candidate = freeze_candidate(
        submission,
        release,
        definition,
        tmp_path / "artifacts",
        snapshot.snapshot.snapshot_digest,
        snapshot.snapshot.snapshot_id,
    )
    records = load_snapshot_records(
        load_snapshot_view(snapshot.snapshot, "validation", "raw", requester="candidate_evaluation")
    )
    residual = evaluate_residual_layer(
        candidate,
        "L1",
        records,
        [{"record": records[0], "chosen_layer": "L1"}],
        contract,
    )
    assert residual["upstream_accepted_count"] == 1
    assert residual["residual_record_count"] == len(records) - 1
    assert residual["sample_count"] == len(records) - 1

    ablation = evaluate_changed_layer_ablation(candidate, release, ["L1"], records, contract)
    assert ablation["baseline_release_id"] == release.release_id
    assert set(ablation) >= {
        "changed_only",
        "with_baseline_unchanged",
        "cascade",
        "changed_layers",
    }


def test_generalization_transfer_and_slice_requirements_are_hard_gates() -> None:
    validation_result = {
        "rows": [
            {"chosen_layer": "L1", "is_correct": True},
            {"chosen_layer": "L1", "is_correct": True},
        ]
    }
    test_result = {
        "rows": [
            {"chosen_layer": "L1", "is_correct": True},
            {"chosen_layer": None, "is_correct": None},
        ]
    }
    requirements = TargetRequirements(
        validation_test_coverage_retention_min=0.75,
        min_slice_samples=2,
        critical_slices=["a"],
        critical_slice_precision_min=0.9,
    )
    generalization = compute_generalization_summary(
        validation_result,
        test_result,
        [{"slice": "a", "sample_count": 2, "accepted_count": 2, "precision": 0.5, "coverage": 1.0}],
        requirements,
    )
    checks = check_candidate_requirements(
        MetricSummary(1, 1, 0, 1.0, 0.5, 0.0),
        generalization,
        {},
        requirements,
    )
    statuses = {check.name: check.status for check in checks}
    assert statuses["coverage_retention"] == "fail"
    assert statuses["critical_slice_precision_min"] == "fail"
    assert statuses["generalization"] == "fail"

    insufficient = compute_generalization_summary(
        validation_result,
        test_result,
        [{"slice": "a", "sample_count": 1, "accepted_count": 1, "precision": 1.0, "coverage": 1.0}],
        TargetRequirements(min_slice_samples=2),
    )
    insufficient_checks = check_candidate_requirements(
        MetricSummary(1, 1, 0, 1.0, 0.5, 0.0),
        insufficient,
        {},
        TargetRequirements(min_slice_samples=2),
    )
    assert {check.name: check.status for check in insufficient_checks}[
        "min_slice_samples"
    ] == "insufficient"

    precision_drop = compute_generalization_summary(
        validation_result,
        {
            "rows": [
                {"chosen_layer": "L1", "is_correct": True},
                {"chosen_layer": "L1", "is_correct": False},
            ]
        },
        [{"slice": "a", "sample_count": 2, "accepted_count": 2, "precision": 0.5, "coverage": 1.0}],
        TargetRequirements(validation_test_precision_drop_max=0.25),
    )
    precision_checks = check_candidate_requirements(
        MetricSummary(2, 1, 1, 0.5, 1.0, 0.5),
        precision_drop,
        {},
        TargetRequirements(validation_test_precision_drop_max=0.25),
    )
    assert {check.name: check.status for check in precision_checks}["precision_drop"] == "fail"
    resource_checks = check_candidate_requirements(
        MetricSummary(1, 1, 0, 1.0, 0.5, 0.0),
        compute_generalization_summary(
            validation_result,
            test_result,
            [
                {
                    "slice": "a",
                    "sample_count": 2,
                    "accepted_count": 1,
                    "precision": 1.0,
                    "coverage": 0.5,
                }
            ],
            TargetRequirements(),
        ),
        {"memory_mb": 128, "throughput_per_second": 5.0, "cascade_cost": 0.2},
        TargetRequirements(
            memory_mb_max=64,
            throughput_per_second_min=10.0,
            serving_cost_per_1000_max=0.1,
        ),
    )
    resource_statuses = {check.name: check.status for check in resource_checks}
    assert resource_statuses["memory_mb_max"] == "fail"
    assert resource_statuses["throughput_per_second_min"] == "fail"
    assert resource_statuses["serving_cost_per_1000_max"] == "fail"


def test_test_generalization_requires_validation_evidence_for_transfer_gates(
    target_dir: Path, tmp_path: Path, now
) -> None:
    definition, contract, release, snapshot, attempt, submission = _submitted_candidate(
        target_dir, tmp_path, now
    )
    write_artifact(submission.submission_path / "artifacts" / "l1", definition.contract_hash)
    (submission.submission_path / "artifacts" / "l1" / "worker.py").write_text(
        """
import json
import sys

request = json.loads(sys.stdin.readline())
text = request["input"]["text"]
if text in {"a:sample-2", "b:sample-3", "a:sample-4"}:
    print(json.dumps({
        "decision": "accept",
        "output": {"label": text.split(":", 1)[0]},
        "confidence": 0.99,
        "reason": "prefix_match",
    }))
else:
    print(json.dumps({"decision": "abstain", "confidence": 0.1, "reason": "outside"}))
""".lstrip(),
        encoding="utf-8",
    )
    definition = replace(
        definition,
        requirements=replace(
            definition.requirements,
            validation_test_coverage_retention_min=0.75,
        ),
    )
    validation = evaluate_candidate_on_validation(
        submission,
        definition,
        snapshot.snapshot,
        release,
        snapshot.reference_qualification,
        AgentUsageLedger(),
        snapshot.reference_usage,
        None,
        None,
        {"serving_l4_cost": 1.0},
        {"contract": contract, "artifact_store": tmp_path / "artifacts"},
    )
    closed = close_agent_attempt(attempt, "ready_for_test")
    missing_validation = evaluate_candidate_on_test(
        validation["candidate"],
        closed,
        definition,
        snapshot.snapshot,
        release,
        snapshot.reference_qualification,
        AgentUsageLedger(),
        snapshot.reference_usage,
        None,
        None,
        {"serving_l4_cost": 1.0},
        {"contract": contract, "test_results_visible": False},
    )
    missing_decision = compare_candidates(
        [missing_validation["report"]],
        _cold_start_baseline(definition, release, snapshot),
        {"requirements": definition.requirements},
    )
    missing_statuses = {check.name: check.status for check in missing_decision.requirement_results}
    assert missing_statuses["coverage_retention"] == "insufficient"

    fake_validation_result = {
        "rows": [
            {"chosen_layer": "L1", "is_correct": True},
            {"chosen_layer": None, "is_correct": None},
        ]
    }
    with_validation = evaluate_candidate_on_test(
        validation["candidate"],
        closed,
        definition,
        snapshot.snapshot,
        release,
        snapshot.reference_qualification,
        AgentUsageLedger(),
        snapshot.reference_usage,
        None,
        None,
        {"serving_l4_cost": 1.0},
        {
            "contract": contract,
            "test_results_visible": False,
            "validation_result": fake_validation_result,
            "validation_report": validation["report"],
        },
    )
    transfer_decision = compare_candidates(
        [with_validation["report"]],
        _cold_start_baseline(definition, release, snapshot),
        {"requirements": definition.requirements},
    )
    transfer_statuses = {
        check.name: check.status for check in transfer_decision.requirement_results
    }
    assert transfer_statuses["coverage_retention"] == "fail"


def test_fault_fallback_checks_local_faults_and_fails_closed_for_l4(
    target_dir: Path, tmp_path: Path, now
) -> None:
    definition, contract, release, snapshot, _attempt, submission = _submitted_candidate(
        target_dir, tmp_path, now
    )
    candidate = freeze_candidate(
        submission,
        release,
        definition,
        tmp_path / "artifacts",
        snapshot.snapshot.snapshot_digest,
    )
    records = load_snapshot_records(
        load_snapshot_view(snapshot.snapshot, "validation", "raw", requester="candidate_evaluation")
    )
    local_faults = evaluate_fault_fallback(
        candidate,
        [{"kind": "malformed_response"}, {"kind": "invalid_output"}, {"kind": "timeout"}],
        records,
        contract,
    )
    assert local_faults["status"] == "pass"
    planned_record_id = local_faults["plan"]["order"][0]
    planned_request_id = local_faults["plan"]["request_ids"][planned_record_id]
    observed = {
        scenario["kind"]: scenario["observed_decision"]
        for scenario in local_faults["scenarios"]
    }
    observed_request_ids = {scenario["request_id"] for scenario in local_faults["scenarios"]}
    assert observed == {
        "malformed_response": "protocol_error",
        "invalid_output": "invalid_output",
        "timeout": "timeout",
    }
    assert observed_request_ids == {planned_request_id}
    assert planned_request_id.startswith("eval-")
    assert planned_request_id not in {record.snapshot_record_id for record in records}
    next_faults = evaluate_fault_fallback(
        candidate,
        [{"kind": "malformed_response"}],
        records,
        contract,
    )
    next_request_id = next_faults["scenarios"][0]["request_id"]
    assert next_request_id.startswith("eval-")
    assert next_request_id != planned_request_id
    default_faults = evaluate_fault_fallback(candidate, [], records, contract)
    assert default_faults["status"] == "pass"
    assert {scenario["kind"] for scenario in default_faults["scenarios"]} == {
        "malformed_response",
        "invalid_output",
        "timeout",
        "crash",
        "l4_fallback_failure",
    }

    l4_fault = evaluate_fault_fallback(
        candidate,
        [{"kind": "l4_fallback_failure"}],
        records,
        contract,
    )
    assert l4_fault["status"] == "pass"
    assert l4_fault["scenarios"][0]["reason"] == "safe_error_exercised"
    assert l4_fault["scenarios"][0]["fallback_status"] == "error"
    assert l4_fault["scenarios"][0]["error_message_hash"] is not None
    assert "provider secret" not in l4_fault["scenarios"][0]["public_error_message"]


@pytest.mark.parametrize(
    ("candidate_yaml", "match"),
    [
        ("routing:\n  enabled_layers: [L4]\n", "enabled_layers"),
        ("routing:\n  enabled_layers: L1\n", "enabled_layers"),
        ("routing:\n  L1_timeout_ms: 0\n", "L1_timeout_ms"),
        ("registry: {}\n", "Core-owned"),
    ],
)
def test_candidate_routing_validation_rejects_invalid_candidate_yaml(
    target_dir: Path, tmp_path: Path, now, candidate_yaml: str, match: str
) -> None:
    case_root = tmp_path / match.replace("_", "-")
    definition, contract, release, snapshot, _attempt, submission = _submitted_candidate(
        target_dir, case_root, now
    )
    (submission.submission_path / "candidate.yaml").write_text(candidate_yaml, encoding="utf-8")
    with pytest.raises(EvaluationError, match=match):
        evaluate_candidate_on_validation(
            submission,
            definition,
            snapshot.snapshot,
            release,
            snapshot.reference_qualification,
            AgentUsageLedger(),
            snapshot.reference_usage,
            None,
            None,
            {"serving_l4_cost": 1.0},
            {"contract": contract, "artifact_store": case_root / "artifacts"},
        )


@pytest.mark.parametrize(
    ("worker_source", "timeout_ms"),
    [
        ("print('not-json')\n", None),
        (
            """
import json
import sys

json.loads(sys.stdin.readline())
print(json.dumps({"decision": "accept", "output": {"bad": "shape"}, "reason": "prefix_match"}))
""".lstrip(),
            None,
        ),
        (
            """
import json
import sys
import time

json.loads(sys.stdin.readline())
time.sleep(2)
print(json.dumps({"decision": "abstain", "reason": "outside"}))
""".lstrip(),
            200,
        ),
        ("raise SystemExit(2)\n", None),
    ],
)
def test_protocol_preflight_rejects_healthcheck_passing_broken_artifacts(
    target_dir: Path,
    tmp_path: Path,
    now,
    worker_source: str,
    timeout_ms: int | None,
) -> None:
    definition, contract, release, snapshot, _attempt, submission = _submitted_candidate(
        target_dir, tmp_path, now
    )
    artifact_dir = submission.submission_path / "artifacts" / "l1"
    (artifact_dir / "worker.py").write_text(worker_source, encoding="utf-8")
    if timeout_ms is not None:
        manifest_path = artifact_dir / "artifact.yaml"
        manifest_path.write_text(
            manifest_path.read_text(encoding="utf-8").replace(
                "timeout_ms: 1000", f"timeout_ms: {timeout_ms}"
            ),
            encoding="utf-8",
        )

    with pytest.raises(EvaluationError, match="protocol preflight"):
        evaluate_candidate_on_validation(
            submission,
            definition,
            snapshot.snapshot,
            release,
            snapshot.reference_qualification,
            AgentUsageLedger(),
            snapshot.reference_usage,
            None,
            None,
            {"serving_l4_cost": 1.0},
            {"contract": contract, "artifact_store": tmp_path / "artifacts"},
        )


def test_official_evaluation_uses_candidate_routing_timeouts(
    target_dir: Path, tmp_path: Path, now
) -> None:
    definition, contract, release, snapshot, _attempt, submission = _submitted_candidate(
        target_dir, tmp_path, now
    )
    artifact_dir = submission.submission_path / "artifacts" / "l1"
    (artifact_dir / "worker.py").write_text(
        """
import json
import sys
import time

request = json.loads(sys.stdin.readline())
text = request["input"]["text"]
if text == "preflight":
    print(json.dumps({"decision": "abstain", "confidence": 0.1, "reason": "outside"}))
else:
    time.sleep(0.5)
    print(json.dumps({
        "decision": "accept",
        "output": {"label": text.split(":", 1)[0]},
        "confidence": 0.99,
        "reason": "prefix_match",
    }))
""".lstrip(),
        encoding="utf-8",
    )
    (submission.submission_path / "candidate.yaml").write_text(
        "routing:\n  enabled_layers: [L1]\n  L1_timeout_ms: 200\n  total_deadline_ms: 200\n",
        encoding="utf-8",
    )

    validation = evaluate_candidate_on_validation(
        submission,
        definition,
        snapshot.snapshot,
        release,
        snapshot.reference_qualification,
        AgentUsageLedger(),
        snapshot.reference_usage,
        None,
        None,
        {"serving_l4_cost": 1.0},
        {"contract": contract, "artifact_store": tmp_path / "artifacts"},
    )

    assert validation["report"].metrics["local"]["accepted_count"] == 0
    assert validation["report"].metrics["l4_fallback_share"] == 1.0


def test_disabled_artifacts_do_not_affect_official_preflight_or_resource_gates(
    target_dir: Path, tmp_path: Path, now
) -> None:
    definition, contract, release, snapshot, attempt, submission = _submitted_candidate(
        target_dir, tmp_path, now
    )
    definition = replace(
        definition,
        requirements=replace(definition.requirements, memory_mb_max=128),
    )
    l2_dir = submission.submission_path / "artifacts" / "l2"
    write_artifact(l2_dir, definition.contract_hash)
    (l2_dir / "worker.py").write_text("print('not-json')\n", encoding="utf-8")
    (l2_dir / "healthcheck.py").write_text("raise SystemExit(1)\n", encoding="utf-8")
    manifest_path = l2_dir / "artifact.yaml"
    manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8")
        .replace("layer: L1", "layer: L2")
        .replace("memory_mb: 64", "memory_mb: 999"),
        encoding="utf-8",
    )
    (submission.submission_path / "candidate.yaml").write_text(
        "routing:\n  enabled_layers: [L1]\n",
        encoding="utf-8",
    )
    submission = replace(submission, declared_layers=["L1", "L2"])

    validation = evaluate_candidate_on_validation(
        submission,
        definition,
        snapshot.snapshot,
        release,
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
        release,
        snapshot.reference_qualification,
        AgentUsageLedger(),
        snapshot.reference_usage,
        None,
        None,
        {"serving_l4_cost": 1.0},
        {"contract": contract, "test_results_visible": False},
    )
    decision = compare_candidates(
        [test["report"]],
        _cold_start_baseline(definition, release, snapshot),
        {"requirements": definition.requirements},
    )

    assert validation["candidate"].artifacts["L2"] is not None
    assert validation["candidate"].routing.enabled_layers == ["L1"]
    assert test["report"].latency["memory_mb"] == 64
    assert decision.status == "eligible_for_release"
    assert decision.selected_operating_point == {"enabled_layers": ["L1"]}


def test_candidate_manifest_rejects_invalid_layers_and_inherited_mismatch(
    target_dir: Path, tmp_path: Path, now
) -> None:
    definition, _contract, release, snapshot, _attempt, submission = _submitted_candidate(
        target_dir, tmp_path, now
    )
    candidate = freeze_candidate(
        submission,
        release,
        definition,
        tmp_path / "artifacts",
        snapshot.snapshot.snapshot_digest,
    )
    package = candidate.artifacts["L1"]
    assert package is not None

    bad_artifacts = dict(candidate.artifacts)
    bad_artifacts["LX"] = package  # type: ignore[assignment]
    layer_check = validate_candidate_manifest(
        replace(candidate, artifacts=bad_artifacts), definition, release
    )
    assert layer_check["status"] == "fail"
    assert any("exactly L1, L2, and L3" in failure for failure in layer_check["failures"])

    inherited_base = replace(
        release,
        artifacts={
            "L1": replace(package, digest="different"),
            "L2": None,
            "L3": None,
        },
    )
    inherited_check = validate_candidate_manifest(candidate, definition, inherited_base)
    assert inherited_check["status"] == "fail"
    assert any(
        "inherited artifact digest mismatch" in failure
        for failure in inherited_check["failures"]
    )


def test_final_report_required_before_release_and_agent_visible_strips_manifest(
    target_dir: Path, tmp_path: Path, now
) -> None:
    definition, contract, release, snapshot, attempt, submission = _submitted_candidate(
        target_dir, tmp_path, now
    )
    validation = evaluate_candidate_on_validation(
        submission,
        definition,
        snapshot.snapshot,
        release,
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
        release,
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
        _cold_start_baseline(definition, release, snapshot),
        {"requirements": definition.requirements},
    )
    mismatched_baseline = replace(release, release_id="other-release")
    with pytest.raises(EvaluationError, match="baseline release"):
        compare_candidates(
            [test["report"]],
            _cold_start_baseline(definition, mismatched_baseline, snapshot),
            {"requirements": definition.requirements},
        )
    with pytest.raises(EvaluationError, match="selected report"):
        finalize_report(
            replace(test["report"], report_id="other-report"),
            decision,
            validation["report"],
        )
    final = finalize_report(test["report"], decision, validation["report"])
    visible = build_agent_visible_report(final, include_test_metrics=True)
    hidden = build_agent_visible_report(final, include_test_metrics=False)
    assert final.decision is decision
    assert visible.validation_metrics == validation["report"].metrics
    assert visible.holdout_consumption is not None
    assert not hasattr(visible.holdout_consumption, "record_ids")
    assert hidden.validation_metrics == validation["report"].metrics
    assert hidden.test_metrics is None
    assert hidden.holdout_consumption is None
    assert hidden.decision_summary.comparison_summary == {}
    approval = __import__("darjeeling.model").model.ApprovalRecord(
        approval_id=new_id("approval"),
        candidate_id=validation["candidate"].candidate_id,
        report_id=final.report_id,
        target_name=definition.name,
        contract_hash=definition.contract_hash,
        snapshot_id=snapshot.snapshot.snapshot_id,
        approved_at=utcnow(),
        approved_by="user",
    )
    created = create_release(
        validation["candidate"], snapshot.snapshot, release, final, approval, tmp_path / "artifacts"
    )
    assert created.candidate_id == validation["candidate"].candidate_id
    with pytest.raises(ReleaseError, match="decision scope"):
        create_release(
            validation["candidate"],
            snapshot.snapshot,
            release,
            replace(final, decision=replace(decision, candidate_id="other-candidate")),
            approval,
            tmp_path / "artifacts",
        )
    with pytest.raises(ReleaseError, match="candidate snapshot"):
        create_release(
            replace(validation["candidate"], snapshot_id="other-snapshot"),
            snapshot.snapshot,
            release,
            final,
            approval,
            tmp_path / "artifacts",
        )
    with pytest.raises(ReleaseError, match="snapshot scope"):
        create_release(
            validation["candidate"],
            replace(snapshot.snapshot, target_name="other-target"),
            release,
            final,
            approval,
            tmp_path / "artifacts",
        )
    with pytest.raises(ReleaseError, match="snapshot scope"):
        create_release(
            validation["candidate"],
            replace(snapshot.snapshot, contract_hash="other-contract"),
            release,
            final,
            approval,
            tmp_path / "artifacts",
        )
    with pytest.raises(ReleaseError, match="approval target"):
        create_release(
            validation["candidate"],
            snapshot.snapshot,
            release,
            final,
            replace(approval, target_name="other-target"),
            tmp_path / "artifacts",
        )
    with pytest.raises(ReleaseError, match="base release identity"):
        create_release(
            validation["candidate"],
            snapshot.snapshot,
            replace(release, release_id="other-release"),
            final,
            approval,
            tmp_path / "artifacts",
        )
    package = validation["candidate"].artifacts["L1"]
    assert package is not None
    swapped_artifact_dir = write_artifact(
        tmp_path / "swapped-artifact", definition.contract_hash, accept_prefixes=["b"]
    )
    swapped_package = freeze_artifact_package(
        swapped_artifact_dir,
        read_artifact_manifest(swapped_artifact_dir),
        tmp_path / "swapped-store",
        snapshot.snapshot.snapshot_digest,
    )
    with pytest.raises(ReleaseError, match="candidate digest"):
        create_release(
            replace(
                validation["candidate"],
                artifacts={**validation["candidate"].artifacts, "L1": swapped_package},
            ),
            snapshot.snapshot,
            release,
            final,
            approval,
            tmp_path / "artifacts",
        )
    wrong_snapshot_package = replace(package, source_snapshot_digest="other-snapshot-digest")
    with pytest.raises(ReleaseError, match="snapshot digest"):
        create_release(
            replace(
                validation["candidate"],
                artifacts={**validation["candidate"].artifacts, "L1": wrong_snapshot_package},
            ),
            snapshot.snapshot,
            release,
            final,
            approval,
            tmp_path / "artifacts",
        )
    (package.package_path / "worker.py").write_text("print('changed')\n", encoding="utf-8")
    with pytest.raises(ReleaseError, match="digest mismatch"):
        create_release(
            validation["candidate"],
            snapshot.snapshot,
            release,
            final,
            approval,
            tmp_path / "artifacts",
        )
    with pytest.raises(ReleaseError):
        create_release(
            validation["candidate"],
            snapshot.snapshot,
            release,
            test["report"],
            approval,
            tmp_path / "artifacts",
        )


def test_final_test_report_recomputes_generalization_from_paired_validation_report(
    target_dir: Path, tmp_path: Path, now
) -> None:
    definition, contract, release, snapshot, attempt, submission = _submitted_candidate(
        target_dir, tmp_path, now
    )
    (submission.submission_path / "artifacts" / "l1" / "worker.py").write_text(
        """
import json
import sys

request = json.loads(sys.stdin.readline())
text = request["input"]["text"]
if text in {"a:sample-2", "b:sample-3", "a:sample-4"}:
    print(json.dumps({
        "decision": "accept",
        "output": {"label": text.split(":", 1)[0]},
        "confidence": 0.99,
        "reason": "prefix_match",
    }))
else:
    print(json.dumps({"decision": "abstain", "confidence": 0.1, "reason": "outside"}))
""".lstrip(),
        encoding="utf-8",
    )
    validation = evaluate_candidate_on_validation(
        submission,
        definition,
        snapshot.snapshot,
        release,
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
        release,
        snapshot.reference_qualification,
        AgentUsageLedger(),
        snapshot.reference_usage,
        None,
        None,
        {"serving_l4_cost": 1.0},
        {"contract": contract, "test_results_visible": True},
    )
    assert validation["report"].metrics["local"]["coverage"] == 1.0
    assert test["report"].metrics["local"]["coverage"] == 0.5

    decision = compare_candidates(
        [test["report"]],
        _cold_start_baseline(definition, release, snapshot),
        {"requirements": definition.requirements},
    )
    final = finalize_report(test["report"], decision, validation["report"])

    assert final.generalization.validation_precision == 1.0
    assert final.generalization.test_precision == 1.0
    assert final.generalization.validation_coverage == 1.0
    assert final.generalization.test_coverage == 0.5
    assert final.generalization.precision_drop == 0.0
    assert final.generalization.coverage_retention == 0.5


def test_report_safety_and_agent_visible_payloads_strip_private_evaluation_material(
    target_dir: Path, tmp_path: Path, now
) -> None:
    definition, contract, release, snapshot, attempt, submission = _submitted_candidate(
        target_dir, tmp_path, now
    )
    validation = evaluate_candidate_on_validation(
        submission,
        definition,
        snapshot.snapshot,
        release,
        snapshot.reference_qualification,
        AgentUsageLedger(),
        snapshot.reference_usage,
        None,
        None,
        {"serving_l4_cost": 1.0},
        {"contract": contract, "artifact_store": tmp_path / "artifacts"},
    )

    safety = validation["report"].safety
    assert safety["fault_fallback"]["status"] == "pass"
    assert safety["fault_fallback"]["scenario_count"] >= 1
    ablation = safety["diagnostics"]["changed_layer_ablation"]
    assert ablation["cascade"]["sample_count"] == validation["report"].metrics["sample_count"]
    assert "local" in ablation["cascade"]
    _assert_no_private_evaluation_material(safety)

    closed = close_agent_attempt(attempt, "ready_for_test")
    test = evaluate_candidate_on_test(
        validation["candidate"],
        closed,
        definition,
        snapshot.snapshot,
        release,
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
        _cold_start_baseline(definition, release, snapshot),
        {"requirements": definition.requirements},
    )
    final = finalize_report(test["report"], decision, validation["report"])
    visible = build_agent_visible_report(final, include_test_metrics=True)
    _assert_no_private_evaluation_material(visible)


def test_private_evaluation_uses_plan_order_and_request_ids(
    target_dir: Path, tmp_path: Path, now
) -> None:
    definition, contract, release, snapshot, _attempt, submission = _submitted_candidate(
        target_dir, tmp_path, now
    )
    worker_path = submission.submission_path / "artifacts" / "l1" / "worker.py"
    worker_path.write_text(
        """
import json
import sys

request = json.loads(sys.stdin.readline())
print(json.dumps({
    "decision": "accept",
    "output": {"label": request["request_id"]},
    "reason": "prefix_match",
}))
""".lstrip(),
        encoding="utf-8",
    )
    candidate = freeze_candidate(
        submission,
        release,
        definition,
        tmp_path / "artifacts",
        snapshot.snapshot.snapshot_digest,
    )
    records = []
    for split in ["validation", "test"]:
        view = load_snapshot_view(snapshot.snapshot, split, "raw", requester="candidate_evaluation")
        records.extend(load_snapshot_records(view))
    snapshot_order = [record.snapshot_record_id for record in records]

    standalone = evaluate_standalone_layer(candidate, "L1", records, contract)
    standalone_plan = standalone["plan"]
    standalone_result_order = [
        record.snapshot_record_id for record, _attempt in standalone["results"]
    ]
    standalone_request_ids = [
        attempt.output["label"] for _record, attempt in standalone["results"]
    ]
    assert standalone_plan["order"] != snapshot_order
    assert standalone_result_order == standalone_plan["order"]
    assert standalone_request_ids == [
        standalone_plan["request_ids"][snapshot_record_id]
        for snapshot_record_id in standalone_plan["order"]
    ]
    assert all(request_id.startswith("eval-") for request_id in standalone_request_ids)
    assert not set(standalone_request_ids) & set(snapshot_order)

    cascade = evaluate_full_cascade(candidate, records, contract)
    cascade_plan = cascade["plan"]
    cascade_result_order = [row["record"].snapshot_record_id for row in cascade["rows"]]
    cascade_request_ids = [row["output"]["label"] for row in cascade["rows"]]
    assert cascade_result_order == cascade_plan["order"]
    assert cascade_request_ids == [
        cascade_plan["request_ids"][snapshot_record_id]
        for snapshot_record_id in cascade_plan["order"]
    ]


def test_candidate_evaluation_rejects_mutated_frozen_package(
    target_dir: Path, tmp_path: Path, now
) -> None:
    definition, contract, release, snapshot, _attempt, submission = _submitted_candidate(
        target_dir, tmp_path, now
    )
    candidate = freeze_candidate(
        submission,
        release,
        definition,
        tmp_path / "artifacts",
        snapshot.snapshot.snapshot_digest,
    )
    package = candidate.artifacts["L1"]
    assert package is not None
    (package.package_path / "worker.py").write_text("print('changed')\n", encoding="utf-8")
    view = load_snapshot_view(
        snapshot.snapshot, "validation", "raw", requester="candidate_evaluation"
    )
    records = load_snapshot_records(view)
    with pytest.raises(ArtifactError, match="digest mismatch"):
        evaluate_standalone_layer(candidate, "L1", records, contract)
