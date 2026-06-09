import json
from pathlib import Path

from typer.testing import CliRunner

import darjeeling.compiler.l2_target_evolution as l2_target_evolution
from darjeeling.artifacts.store import ArtifactManifest, ArtifactStore
from darjeeling.cli import (
    _resolve_l2_target_agent_rounds,
    _resolve_l2_target_budget,
    _resolve_l2_target_local_search_cross_audit_top_k,
    _resolve_l2_target_visible_cross_audit_folds,
    _resolve_l2_target_visible_validation_folds,
    app,
)
from darjeeling.compiler.l2_target_evolution import (
    L2TargetEvolutionConfig,
    _adoption_decision,
    _selection_decision,
    evaluate_target_workspace,
    prepare_l2_target_workspace,
    run_l2_target_evolution,
    split_l2_target_traces,
)
from darjeeling.schemas import Frame, LayerResult, TraceRecord, traces_to_teacher_view


def _trace(index: int, *, intent: str, slots: dict[str, str]) -> TraceRecord:
    utterance = f"{intent.replace('_', ' ')} example {index}"
    frame = Frame(intent=intent, slots=slots)
    return TraceRecord(
        request_id=f"r{index}",
        utterance=utterance,
        gold_frame=frame,
        teacher_frame=frame,
        chosen_layer="L4",
        final_frame=frame,
        layer_results=[
            LayerResult(layer="L2", accepted=False, frame=None, latency_ms=1.0),
            LayerResult(layer="L4", accepted=True, frame=frame, latency_ms=1.0),
        ],
    )


def _traces() -> list[TraceRecord]:
    return [
        _trace(index, intent="alarm_set", slots={"time": f"{index} am"})
        if index % 2 == 0
        else _trace(index, intent="weather_query", slots={"location": f"city {index}"})
        for index in range(12)
    ]


def test_l2_target_evolution_runs_multiple_inner_rounds(tmp_path: Path) -> None:
    summary = run_l2_target_evolution(
        config=L2TargetEvolutionConfig(
            source_repo_dir=Path.cwd(),
            job_dir=tmp_path / "job",
            rounds=3,
            mode="dry-run",
            inner_patience_rounds=0,
        ),
        traces=traces_to_teacher_view(_traces()),
    )

    workspace = tmp_path / "job" / "workspace" / "l2_target"
    assert summary["schema_version"] == "l2-target-evolution-v1"
    assert summary["rounds_completed"] == 3
    assert summary["stop_reason"] == "round_budget_exhausted"
    assert summary["budget_policy"]["profile"] == "standard"
    assert summary["data_split_policy"] == {
        "schema_version": "l2-target-split-policy-v1",
        "policy": "chronological",
        "group_key": None,
        "split_counts": {
            "train": 7,
            "inner_validation": 2,
            "selection_holdout": 1,
            "promotion_holdout": 2,
        },
        "visible_validation_splits": ["inner_validation"],
        "visible_validation_folds": 1,
        "visible_validation_visibility": "agent_workspace_visible",
        "private_splits": ["selection_holdout", "promotion_holdout"],
        "private_split_visibility": "outer_harness_only",
    }
    assert summary["loop_cadence"] == {
        "kind": "fixed_trace_snapshot_inner_loop",
        "outer_replay_cadence_bound": False,
        "teacher_labeled_traces": 12,
        "note": (
            "target rounds reuse this fixed split; collecting another stream prefix "
            "is not part of the inner loop"
        ),
    }
    assert summary["target_code_policy"] == {
        "core_must_remain_dataset_independent": True,
        "target_dependent_code_allowed_in": "target/",
        "target_specific_code_is_not_rejected_for_dataset_dependence": True,
        "target_code_visibility_rule": (
            "target code may be derived from data/train.jsonl and "
            "visible data/inner_validation*.jsonl only"
        ),
        "private_holdout_visibility": (
            "selection/promotion holdouts remain outside the agent workspace"
        ),
        "adoption_authority": (
            "visible validation gate, private selection/promotion gates, and final outer replay"
        ),
    }
    assert summary["adoption_decision"]["adopted"] is False
    assert summary["best_adoptable_round"] is None
    assert summary["target_code_scope"] == "target/"
    assert summary["baseline"]["label"] == "baseline"
    assert summary["baseline"]["train_audit"]["split"] == "train_audit"
    assert summary["baseline"]["train_audit"]["gate_role"] == (
        "diagnostic_only_not_selection_or_adoption_gate"
    )
    assert (workspace / "target" / "target_l2.py").exists()
    assert (workspace / "system" / "darjeeling" / "src").exists()
    assert (workspace / "system" / "darjeeling" / "README.md").exists()
    assert not (workspace / "candidate").exists()
    assert not (workspace / "data" / "promotion_holdout.jsonl").exists()
    assert not (workspace / "data" / "selection_holdout.jsonl").exists()
    assert (tmp_path / "job" / "private" / "selection_holdout.jsonl").exists()
    assert (tmp_path / "job" / "private" / "promotion_holdout.jsonl").exists()
    assert (tmp_path / "job" / "rounds" / "round_003.json").exists()
    assert summary["rounds"][0]["target_snapshot"] == "rounds/round_001_target"
    assert (
        tmp_path
        / "job"
        / summary["rounds"][0]["target_snapshot"]
        / "target_l2.py"
    ).exists()

    manifest = json.loads((workspace / "workspace_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "l2-target-workspace-v1"
    assert manifest["target_dir"] == "target"
    assert manifest["system_repo_dir"] == "system/darjeeling"
    assert set(manifest["data_files"]) == {
        "inner_validation.jsonl",
        "train.jsonl",
    }
    assert manifest["private_data_files_not_in_workspace"] == [
        "selection_holdout.jsonl",
        "promotion_holdout.jsonl",
    ]
    assert set(manifest["visible_state_files"]) == {
        "commands.md",
        "objective.json",
        "round_state.json",
        "target_diagnostics.json",
    }
    assert (workspace / "data" / "objective.json").exists()
    assert (workspace / "data" / "target_diagnostics.json").exists()
    round_state = json.loads((workspace / "data" / "round_state.json").read_text())
    target_diagnostics = json.loads(
        (workspace / "data" / "target_diagnostics.json").read_text()
    )
    round_state_text = json.dumps(round_state)
    target_diagnostics_text = json.dumps(target_diagnostics)
    assert "promotion_holdout" not in round_state_text
    assert "selection_holdout" not in round_state_text
    assert "promotion_holdout" not in target_diagnostics_text
    assert "selection_holdout" not in target_diagnostics_text
    private_rows = [
        json.loads(line)
        for path in [
            tmp_path / "job" / "private" / "selection_holdout.jsonl",
            tmp_path / "job" / "private" / "promotion_holdout.jsonl",
        ]
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert all(row["request_id"] not in round_state_text for row in private_rows)
    assert all(row["request_id"] not in target_diagnostics_text for row in private_rows)
    objective = json.loads((workspace / "data" / "objective.json").read_text())
    program_text = (workspace / "program.md").read_text(encoding="utf-8")
    assert "candidate_selection_gate" in round_state
    assert "visible validation gate" in round_state["candidate_selection_gate"]
    assert "early_stop_policy" in round_state
    assert "does not stop the inner loop" in round_state["early_stop_policy"]
    assert "candidate_selection" in objective["gates"]
    assert objective["workspace_scope"]["candidate_code_writable_roots"] == ["target/"]
    assert objective["workspace_scope"]["scratch_writable_roots"] == ["runs/"]
    assert "system/darjeeling/" in objective["workspace_scope"]["protected_roots"]
    assert any(
        "near_miss_examples" in strategy
        for strategy in objective["allowed_strategies"]
    )
    assert "Private selection" in program_text
    assert "alone is not success" in program_text
    assert "outer selection signal" in program_text
    assert "inner-loop early-stop signal" in program_text
    assert "near_miss_examples" in program_text
    assert "target_diagnostics.json" in program_text
    assert "latest_safety_backlog" in program_text
    assert target_diagnostics["schema_version"] == "l2-target-diagnostics-v1"
    assert target_diagnostics["visibility"] == "visible_validation_only"
    assert target_diagnostics["baseline_inner_validation"]["split"] == "inner_validation"
    assert "families" in target_diagnostics["baseline_inner_validation"]
    assert target_diagnostics["baseline_safety_backlog"]["schema_version"] == (
        "l2-target-safety-backlog-v1"
    )
    assert target_diagnostics["baseline_safety_backlog"]["visibility"] == (
        "visible_validation_only"
    )
    assert target_diagnostics["baseline_train_audit"]["split"] == "train_audit"
    assert target_diagnostics["baseline_train_audit_safety_backlog"]["schema_version"] == (
        "l2-target-safety-backlog-v1"
    )
    assert "latest_train_audit_safety_backlog" in target_diagnostics
    assert "latest_safety_backlog" in target_diagnostics
    assert (
        summary["baseline"]["inner_validation"]["family_diagnostics"]["schema_version"]
        == "l2-target-family-diagnostics-v1"
    )
    assert (
        summary["baseline"]["inner_validation"]["safety_backlog"]["schema_version"]
        == "l2-target-safety-backlog-v1"
    )
    assert summary["agent_budget"]["mode"] == "dry-run"
    assert summary["agent_budget"]["applies_to_mode"] is False
    assert summary["agent_budget"]["local_search_consumes_llm"] is False
    assert summary["private_holdout_evidence"]["schema_version"] == (
        "l2-target-private-holdout-evidence-v1"
    )
    assert summary["private_holdout_evidence"]["visibility"] == (
        "outer_summary_only_not_agent_workspace"
    )
    assert "family_diagnostics" in round_state["baseline_inner_validation"]
    assert "safety_backlog" in round_state["baseline_inner_validation"]
    assert round_state["baseline_train_audit"]["split"] == "train_audit"
    assert round_state["baseline_train_audit"]["gate_role"] == (
        "diagnostic_only_not_selection_or_adoption_gate"
    )
    assert "candidate selection or adoption gate" in round_state["train_audit_policy"]
    assert summary["rounds"][0]["train_audit"]["gate_role"] == (
        "diagnostic_only_not_selection_or_adoption_gate"
    )
    assert round_state["agent_budget"]["mode"] == "dry-run"
    assert "private_holdout_evidence" not in round_state
    assert "not a" in program_text
    assert "Darjeeling-core dataset-independence violation" in program_text


def test_l2_target_family_diagnostics_expose_safety_backlog() -> None:
    risky_example = {
        "request_id": "visible-risk-1",
        "utterance": "tell me about the latest media trends",
        "teacher_frame": {"intent": "general_quirky", "slots": {}},
        "predicted_frame": {
            "intent": "general_quirky",
            "slots": {"date": "the latest media trends"},
        },
        "guard_probability": 0.99,
    }
    family_stats = {
        "coverage_opportunity": {
            "teacher_intent": "calendar_query",
            "total": 20,
            "accepted_correct": 0,
            "accepted_wrong": 0,
            "rejected_correct": 12,
            "rejected_wrong": 8,
            "vetoed_correct": 0,
            "vetoed_wrong": 0,
            "intent_correct_slot_wrong": 5,
            "predicted_intents": {"calendar_query": 20},
            "examples": {
                "accepted_wrong": [],
                "rejected_correct": [],
                "vetoed_correct": [],
                "intent_correct_slot_wrong": [],
            },
        },
        "accepted_wrong_risk": {
            "teacher_intent": "general_quirky",
            "total": 5,
            "accepted_correct": 1,
            "accepted_wrong": 2,
            "rejected_correct": 0,
            "rejected_wrong": 2,
            "vetoed_correct": 0,
            "vetoed_wrong": 0,
            "intent_correct_slot_wrong": 2,
            "predicted_intents": {"general_quirky": 5},
            "examples": {
                "accepted_wrong": [risky_example],
                "rejected_correct": [],
                "vetoed_correct": [],
                "intent_correct_slot_wrong": [risky_example],
            },
        },
    }

    payload = l2_target_evolution._family_diagnostics_payload(
        split="visible_validation",
        validation_size=25,
        family_stats=family_stats,
    )

    safety_backlog = payload["safety_backlog"]
    assert safety_backlog["schema_version"] == "l2-target-safety-backlog-v1"
    assert safety_backlog["priority"] == (
        "fix_visible_accepted_wrong_before_coverage_expansion"
    )
    assert safety_backlog["items"][0]["teacher_intent"] == "general_quirky"
    assert safety_backlog["items"][0]["accepted_wrong"] == 2
    assert safety_backlog["items"][0]["wrong_examples"] == [risky_example]
    assert "postprocess" in safety_backlog["items"][0]["recommended_action"]
    assert all(
        item["teacher_intent"] != "calendar_query"
        for item in safety_backlog["items"]
    )


def test_l2_target_intent_stratified_split_samples_private_splits() -> None:
    traces = [
        _trace(index, intent="alarm_set", slots={"time": f"{index} am"})
        for index in range(10)
    ] + [
        _trace(index + 10, intent="weather_query", slots={"location": f"city {index}"})
        for index in range(10)
    ]

    split = split_l2_target_traces(
        traces_to_teacher_view(traces),
        policy="intent-stratified",
    )

    assert {key: len(value) for key, value in split.items()} == {
        "train": 12,
        "inner_validation": 4,
        "selection_holdout": 2,
        "promotion_holdout": 2,
    }
    for split_name in ["inner_validation", "selection_holdout", "promotion_holdout"]:
        intents = {trace.teacher_frame.intent for trace in split[split_name]}
        assert intents == {"alarm_set", "weather_query"}


def test_l2_target_visible_validation_folds_stay_visible_not_private(
    tmp_path: Path,
) -> None:
    traces = [
        _trace(index, intent="alarm_set", slots={"time": f"{index} am"})
        if index % 3 == 0
        else _trace(index, intent="weather_query", slots={"location": f"city {index}"})
        for index in range(30)
    ]

    summary = run_l2_target_evolution(
        config=L2TargetEvolutionConfig(
            source_repo_dir=Path.cwd(),
            job_dir=tmp_path / "job",
            rounds=1,
            mode="dry-run",
            visible_validation_folds=3,
            inner_patience_rounds=0,
        ),
        traces=traces_to_teacher_view(traces),
    )

    workspace = tmp_path / "job" / "workspace" / "l2_target"
    manifest = json.loads((workspace / "workspace_manifest.json").read_text())
    round_state = json.loads((workspace / "data" / "round_state.json").read_text())
    visible_metric = summary["rounds"][0]["inner_validation"]

    assert summary["budget_policy"]["visible_validation_folds"] == 3
    assert summary["data_split_policy"]["visible_validation_splits"] == [
        "inner_validation",
        "inner_validation_shadow_1",
        "inner_validation_shadow_2",
    ]
    assert manifest["visible_validation_splits"] == [
        "inner_validation",
        "inner_validation_shadow_1",
        "inner_validation_shadow_2",
    ]
    assert set(manifest["data_files"]) >= {
        "inner_validation.jsonl",
        "inner_validation_shadow_1.jsonl",
        "inner_validation_shadow_2.jsonl",
        "train.jsonl",
    }
    assert not (workspace / "data" / "selection_holdout.jsonl").exists()
    assert not (workspace / "data" / "promotion_holdout.jsonl").exists()
    assert visible_metric["split"] == "visible_validation"
    assert visible_metric["visible_validation_splits"] == [
        "inner_validation",
        "inner_validation_shadow_1",
        "inner_validation_shadow_2",
    ]
    assert len(visible_metric["visible_validation_folds"]) == 3
    assert "visible validation gate" in round_state["candidate_selection_gate"]
    assert "selection_holdout" not in json.dumps(round_state)
    assert "promotion_holdout" not in json.dumps(round_state)


def test_l2_target_extra_visible_folds_do_not_keep_shrinking_train_split() -> None:
    traces = [
        _trace(index, intent="alarm_set", slots={"time": f"{index} am"})
        if index % 2 == 0
        else _trace(index, intent="weather_query", slots={"location": f"city {index}"})
        for index in range(100)
    ]

    three_fold = split_l2_target_traces(
        traces_to_teacher_view(traces),
        visible_validation_folds=3,
    )
    five_fold = split_l2_target_traces(
        traces_to_teacher_view(traces),
        visible_validation_folds=5,
    )

    assert len(three_fold["train"]) == 50
    assert len(five_fold["train"]) == 50
    assert len(three_fold["selection_holdout"]) == 10
    assert len(five_fold["selection_holdout"]) == 10
    assert len(three_fold["promotion_holdout"]) == 10
    assert len(five_fold["promotion_holdout"]) == 10
    assert sum(
        len(value)
        for key, value in three_fold.items()
        if key.startswith("inner_validation")
    ) == 30
    assert sum(
        len(value)
        for key, value in five_fold.items()
        if key.startswith("inner_validation")
    ) == 30
    assert {
        key for key in five_fold if key.startswith("inner_validation")
    } == {
        "inner_validation",
        "inner_validation_shadow_1",
        "inner_validation_shadow_2",
        "inner_validation_shadow_3",
        "inner_validation_shadow_4",
    }


def test_l2_target_evolution_applies_dry_run_patches_to_target_only(tmp_path: Path) -> None:
    patch_path = tmp_path / "target.patch"
    patch_path.write_text(
        "\n".join(
            [
                "diff --git a/target/target_l2.py b/target/target_l2.py",
                "--- a/target/target_l2.py",
                "+++ b/target/target_l2.py",
                "@@ -3,6 +3,8 @@",
                " import json",
                " from pathlib import Path",
                " from typing import Any",
                " ",
                "+TARGET_MARKER = 'patched'",
                "+",
                " ",
                " def config_overrides() -> dict[str, Any]:",
                "",
            ]
        ),
        encoding="utf-8",
    )

    summary = run_l2_target_evolution(
        config=L2TargetEvolutionConfig(
            source_repo_dir=Path.cwd(),
            job_dir=tmp_path / "job",
            rounds=2,
            mode="dry-run",
            dry_run_patches=(patch_path,),
        ),
        traces=traces_to_teacher_view(_traces()),
    )

    target_text = (
        tmp_path / "job" / "workspace" / "l2_target" / "target" / "target_l2.py"
    ).read_text(encoding="utf-8")
    assert "TARGET_MARKER = 'patched'" in target_text
    assert summary["rounds_completed"] == 2
    assert summary["workspace_scope_policy"] == {
        "schema_version": "l2-target-workspace-scope-v1",
        "candidate_code_writable_roots": ["target/"],
        "scratch_writable_roots": ["runs/"],
        "protected_roots": ["data/", "system/darjeeling/", "tools/", "program.md"],
        "ignored_generated_files": ["__pycache__/", ".pytest_cache/", "*.pyc", "*.pyo"],
        "enforcement": "checked_after_each_mutating_round_before_candidate_evaluation",
    }


def test_l2_target_evolution_rejects_protected_workspace_edits(
    tmp_path: Path,
) -> None:
    patch_path = tmp_path / "core.patch"
    patch_path.write_text(
        "\n".join(
            [
                "diff --git a/system/darjeeling/README.md b/system/darjeeling/README.md",
                "--- a/system/darjeeling/README.md",
                "+++ b/system/darjeeling/README.md",
                "@@ -1,4 +1,4 @@",
                "-# darjeeling",
                "+# patched darjeeling",
                " ",
                (
                    " Profile-guided edge intelligence runtime MVP for the NLU replay "
                    "demo described in"
                ),
                " [docs/mvp_demo_proposal.md](docs/mvp_demo_proposal.md).",
                "",
            ],
        ),
        encoding="utf-8",
    )

    summary = run_l2_target_evolution(
        config=L2TargetEvolutionConfig(
            source_repo_dir=Path.cwd(),
            job_dir=tmp_path / "job",
            rounds=1,
            mode="dry-run",
            dry_run_patches=(patch_path,),
        ),
        traces=traces_to_teacher_view(_traces()),
    )

    commands = [
        json.loads(line)
        for line in (tmp_path / "job" / "commands.jsonl").read_text().splitlines()
    ]
    assert summary["stop_reason"] == "workspace_scope_violation"
    assert summary["rounds_completed"] == 0
    assert commands[-1]["command"] == ["workspace-scope-check", "--round", "1"]
    violation = commands[-1]["workspace_scope_violation"]
    assert violation["modified_protected_files"] == ["system/darjeeling/README.md"]


def test_l2_target_evolution_stops_after_inner_patience(tmp_path: Path) -> None:
    summary = run_l2_target_evolution(
        config=L2TargetEvolutionConfig(
            source_repo_dir=Path.cwd(),
            job_dir=tmp_path / "job",
            rounds=5,
            mode="dry-run",
            inner_patience_rounds=1,
        ),
        traces=traces_to_teacher_view(_traces()),
    )

    assert summary["rounds_requested"] == 5
    assert summary["rounds_completed"] == 1
    assert summary["stop_reason"] == "inner_validation_patience_exhausted"
    assert summary["rounds"][0]["inner_improved"] is False
    assert summary["rounds"][0]["passes_private_selection_gate"] is False
    assert summary["rounds"][0]["passes_private_promotion_gate"] is False


def test_l2_target_evolution_does_not_stop_on_selection_gate_by_default(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def metric(split: str) -> dict:
        return {
            "split": split,
            "label": split,
            "train_size": 1,
            "validation_size": 1,
            "evaluated": 1,
            "accepted": 1,
            "correct_accepts": 1,
            "wrong_accepts": 0,
            "vetoed_accepts": 0,
            "coverage": 1.0,
            "accepted_accuracy": 1.0,
            "wrong_accept_rate": 0.0,
            "passes_gate": True,
            "config": {},
            "wrong_examples": [],
            "veto_examples": [],
            "near_miss_examples": [],
        }

    def fake_evaluate_candidate(**kwargs) -> dict:
        label = kwargs["label"]
        return {
            "label": label,
            "inner_validation": metric("inner_validation"),
            "selection_holdout": metric("selection_holdout"),
            "promotion_holdout": metric("promotion_holdout"),
        }

    monkeypatch.setattr(
        l2_target_evolution,
        "_evaluate_target_candidate",
        fake_evaluate_candidate,
    )

    summary = run_l2_target_evolution(
        config=L2TargetEvolutionConfig(
            source_repo_dir=Path.cwd(),
            job_dir=tmp_path / "job",
            rounds=3,
            mode="dry-run",
            inner_patience_rounds=0,
        ),
        traces=traces_to_teacher_view(_traces()),
    )

    assert summary["rounds_completed"] == 3
    assert summary["stop_reason"] == "round_budget_exhausted"
    assert summary["budget_policy"]["stop_on_selection_gate"] is False
    assert all(
        round_result["passes_candidate_selection_gate"]
        for round_result in summary["rounds"]
    )


def test_l2_target_evolution_local_search_uses_visible_workspace_only(
    tmp_path: Path,
) -> None:
    summary = run_l2_target_evolution(
        config=L2TargetEvolutionConfig(
            source_repo_dir=Path.cwd(),
            job_dir=tmp_path / "job",
            rounds=1,
            mode="local-search",
            local_search_trials=2,
            inner_patience_rounds=0,
        ),
        traces=traces_to_teacher_view(_traces()),
    )

    workspace = tmp_path / "job" / "workspace" / "l2_target"
    report_path = tmp_path / "job" / "rounds" / "round_001_local_search.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert summary["mode"] == "local-search"
    assert summary["agent_budget"]["mode"] == "local-search"
    assert summary["agent_budget"]["applies_to_mode"] is False
    assert summary["agent_budget"]["local_search_consumes_llm"] is False
    assert summary["agent_budget"]["agent_rounds_started"] == 0
    assert summary["rounds_completed"] == 1
    assert summary["rounds"][0]["local_search"]["schema_version"] == (
        "l2-target-local-search-v1"
    )
    assert report["trials_requested"] == 2
    assert report["cross_audit_rerank_enabled"] is False
    assert report["private_holdout_visibility"] == (
        "local search used only agent-visible train and validation-fold data"
    )
    assert "selection_holdout" not in report_path.read_text(encoding="utf-8")
    assert "promotion_holdout" not in report_path.read_text(encoding="utf-8")
    assert (workspace / "tools" / "search_config.py").exists()
    assert not (workspace / "data" / "selection_holdout.jsonl").exists()
    assert not (workspace / "data" / "promotion_holdout.jsonl").exists()


def test_l2_target_evolution_local_search_can_rerank_with_visible_cross_audit(
    tmp_path: Path,
) -> None:
    summary = run_l2_target_evolution(
        config=L2TargetEvolutionConfig(
            source_repo_dir=Path.cwd(),
            job_dir=tmp_path / "job",
            rounds=1,
            mode="local-search",
            local_search_trials=2,
            local_search_cross_audit_top_k=1,
            visible_cross_audit_folds=2,
            inner_patience_rounds=0,
        ),
        traces=traces_to_teacher_view(_traces()),
    )

    report_path = tmp_path / "job" / "rounds" / "round_001_local_search.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert summary["rounds_completed"] == 1
    assert report["cross_audit_rerank_enabled"] is True
    assert report["cross_audit_top_k"] == 1
    assert report["cross_audit_folds"] == 2
    assert report["current_visible_cross_audit"]["split"] == "visible_cross_audit"
    reranked = [
        trial for trial in report["trials"] if trial.get("visible_cross_audit") is not None
    ]
    assert len(reranked) == 1
    assert reranked[0]["visible_cross_audit"]["split"] == "visible_cross_audit"


def test_l2_target_evolution_respects_zero_agent_round_budget(tmp_path: Path) -> None:
    summary = run_l2_target_evolution(
        config=L2TargetEvolutionConfig(
            source_repo_dir=Path.cwd(),
            job_dir=tmp_path / "job",
            rounds=3,
            mode="codex-cli",
            max_agent_rounds=0,
            inner_patience_rounds=0,
        ),
        traces=traces_to_teacher_view(_traces()),
    )

    commands_path = tmp_path / "job" / "commands.jsonl"
    round_state = json.loads(
        (
            tmp_path / "job" / "workspace" / "l2_target" / "data" / "round_state.json"
        ).read_text(encoding="utf-8")
    )

    assert summary["mode"] == "codex-cli"
    assert summary["rounds_requested"] == 3
    assert summary["rounds_completed"] == 0
    assert summary["stop_reason"] == "agent_round_budget_exhausted"
    assert summary["budget_policy"]["max_agent_rounds"] == 0
    assert summary["agent_budget"]["applies_to_mode"] is True
    assert summary["agent_budget"]["max_agent_rounds"] == 0
    assert summary["agent_budget"]["agent_rounds_started"] == 0
    assert summary["agent_budget"]["agent_rounds_remaining"] == 0
    assert commands_path.read_text(encoding="utf-8") == ""
    assert round_state["state_kind"] == "final"
    assert round_state["agent_budget"]["max_agent_rounds"] == 0
    assert round_state["agent_budget"]["agent_rounds_remaining"] == 0


def test_l2_target_fixed_inner_budget_profile_resolves_long_loop_defaults() -> None:
    assert _resolve_l2_target_budget(
        budget_profile="standard",
        rounds=None,
        inner_patience_rounds=None,
        local_search_trials=None,
    ) == (12, 4, 96)
    assert _resolve_l2_target_budget(
        budget_profile="fixed-inner",
        rounds=None,
        inner_patience_rounds=None,
        local_search_trials=None,
    ) == (48, 0, 32)
    assert _resolve_l2_target_budget(
        budget_profile="fixed-inner",
        rounds=3,
        inner_patience_rounds=2,
        local_search_trials=5,
    ) == (3, 2, 5)
    assert _resolve_l2_target_agent_rounds(
        mode="codex-cli",
        budget_profile="standard",
        max_agent_rounds=None,
    ) == 3
    assert _resolve_l2_target_agent_rounds(
        mode="codex-cli",
        budget_profile="fixed-inner",
        max_agent_rounds=None,
    ) == 16
    assert _resolve_l2_target_agent_rounds(
        mode="codex-cli",
        budget_profile="smoke",
        max_agent_rounds=None,
    ) == 1
    assert _resolve_l2_target_agent_rounds(
        mode="codex-cli",
        budget_profile="fixed-inner",
        max_agent_rounds=0,
    ) == 0
    assert _resolve_l2_target_agent_rounds(
        mode="local-search",
        budget_profile="fixed-inner",
        max_agent_rounds=None,
    ) is None
    assert _resolve_l2_target_visible_validation_folds(
        budget_profile="standard",
        visible_validation_folds=None,
    ) == 1
    assert _resolve_l2_target_visible_validation_folds(
        budget_profile="fixed-inner",
        visible_validation_folds=None,
    ) == 5
    assert _resolve_l2_target_visible_validation_folds(
        budget_profile="fixed-inner",
        visible_validation_folds=3,
    ) == 3
    assert _resolve_l2_target_visible_cross_audit_folds(
        budget_profile="standard",
        visible_cross_audit_folds=None,
    ) == 0
    assert _resolve_l2_target_visible_cross_audit_folds(
        budget_profile="smoke",
        visible_cross_audit_folds=None,
    ) == 0
    assert _resolve_l2_target_visible_cross_audit_folds(
        budget_profile="fixed-inner",
        visible_cross_audit_folds=None,
    ) == 3
    assert _resolve_l2_target_visible_cross_audit_folds(
        budget_profile="fixed-inner",
        visible_cross_audit_folds=4,
    ) == 4
    assert _resolve_l2_target_local_search_cross_audit_top_k(
        budget_profile="standard",
        visible_cross_audit_folds=0,
        local_search_cross_audit_top_k=None,
    ) == 0
    assert _resolve_l2_target_local_search_cross_audit_top_k(
        budget_profile="fixed-inner",
        visible_cross_audit_folds=0,
        local_search_cross_audit_top_k=None,
    ) == 0
    assert _resolve_l2_target_local_search_cross_audit_top_k(
        budget_profile="fixed-inner",
        visible_cross_audit_folds=3,
        local_search_cross_audit_top_k=None,
    ) == 4
    assert _resolve_l2_target_local_search_cross_audit_top_k(
        budget_profile="fixed-inner",
        visible_cross_audit_folds=3,
        local_search_cross_audit_top_k=0,
    ) == 0
    assert l2_target_evolution._effective_max_agent_rounds(  # noqa: SLF001
        L2TargetEvolutionConfig(
            source_repo_dir=Path.cwd(),
            job_dir=Path("unused"),
            mode="codex-cli",
            budget_profile="fixed-inner",
        )
    ) == 16


def test_l2_target_accept_hook_can_veto_guard_accepts(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    prepare_l2_target_workspace(
        source_repo_dir=Path.cwd(),
        workspace_root=workspace,
        split=split_l2_target_traces(traces_to_teacher_view(_traces())),
    )
    (workspace / "target" / "target_l2.py").write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "",
                "from typing import Any",
                "",
                "",
                "def config_overrides() -> dict[str, Any]:",
                "    return {'accept_threshold': 0.0}",
                "",
                "",
                "def postprocess_frame(",
                "    utterance: str,",
                "    frame: dict[str, Any],",
                "    metadata: dict[str, Any],",
                ") -> dict[str, Any]:",
                "    del utterance, metadata",
                "    return frame",
                "",
                "",
                "def accept_prediction(",
                "    utterance: str,",
                "    frame: dict[str, Any],",
                "    metadata: dict[str, Any],",
                "    default_accept: bool,",
                ") -> bool | None:",
                "    del utterance, frame, metadata, default_accept",
                "    return False",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = evaluate_target_workspace(
        workspace_root=workspace,
        split="inner_validation",
    )

    assert result["accepted"] == 0
    assert result["vetoed_accepts"] == result["validation_size"]
    assert len(result["veto_examples"]) == result["validation_size"]
    assert result["veto_examples"][0]["predicted_frame"]


def test_l2_target_evaluator_reports_guard_near_misses(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    prepare_l2_target_workspace(
        source_repo_dir=Path.cwd(),
        workspace_root=workspace,
        split=split_l2_target_traces(traces_to_teacher_view(_traces())),
    )
    (workspace / "target" / "target_l2.py").write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "",
                "from typing import Any",
                "",
                "",
                "def config_overrides() -> dict[str, Any]:",
                "    return {'accept_threshold': 1.1}",
                "",
                "",
                "def postprocess_frame(",
                "    utterance: str,",
                "    frame: dict[str, Any],",
                "    metadata: dict[str, Any],",
                ") -> dict[str, Any]:",
                "    del utterance, metadata",
                "    return frame",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = evaluate_target_workspace(
        workspace_root=workspace,
        split="inner_validation",
    )

    assert result["accepted"] == 0
    assert 0 < len(result["near_miss_examples"]) <= 8
    probabilities = [
        example["guard_probability"] for example in result["near_miss_examples"]
    ]
    assert probabilities == sorted(probabilities, reverse=True)
    assert all("would_be_correct" in example for example in result["near_miss_examples"])


def test_l2_target_selection_requires_visible_validation_gate() -> None:
    round_result = {
        "round": 1,
        "inner_validation": {"passes_gate": False},
        "selection_holdout": {
            "passes_gate": True,
            "coverage": 0.1,
            "accepted_accuracy": 1.0,
            "wrong_accept_rate": 0.0,
        },
        "promotion_holdout": {
            "passes_gate": True,
            "coverage": 0.1,
            "accepted_accuracy": 1.0,
            "wrong_accept_rate": 0.0,
        },
    }

    assert _selection_decision([round_result])["selected"] is False
    assert _adoption_decision([round_result])["adopted"] is False


def test_l2_target_best_round_uses_inner_validation_as_tie_breaker() -> None:
    def metric(*, coverage: float, accepted_accuracy: float | None = None) -> dict:
        return {
            "passes_gate": False,
            "coverage": coverage,
            "accepted_accuracy": accepted_accuracy,
            "wrong_accept_rate": 0.0,
            "wrong_accepts": 0,
        }

    early_round = {
        "round": 1,
        "inner_validation": metric(coverage=0.05, accepted_accuracy=1.0),
        "selection_holdout": metric(coverage=0.0),
        "promotion_holdout": metric(coverage=0.0),
    }
    later_inner_improved_round = {
        "round": 2,
        "inner_validation": metric(coverage=0.30, accepted_accuracy=1.0),
        "selection_holdout": metric(coverage=0.0),
        "promotion_holdout": metric(coverage=0.0),
    }

    assert l2_target_evolution._best_round(  # noqa: SLF001
        [early_round, later_inner_improved_round]
    ) is later_inner_improved_round


def test_l2_target_private_holdout_evidence_reports_sparse_selection() -> None:
    def metric(*, accepted: int, correct: int, wrong: int, passes_gate: bool) -> dict:
        return {
            "passes_gate": passes_gate,
            "accepted": accepted,
            "correct_accepts": correct,
            "wrong_accepts": wrong,
            "coverage": accepted / 50,
            "accepted_accuracy": correct / accepted if accepted else None,
            "wrong_accept_rate": wrong / accepted if accepted else 0.0,
        }

    round_result = {
        "round": 3,
        "inner_validation": metric(accepted=4, correct=4, wrong=0, passes_gate=True),
        "selection_holdout": metric(accepted=0, correct=0, wrong=0, passes_gate=False),
        "promotion_holdout": metric(accepted=1, correct=1, wrong=0, passes_gate=True),
    }

    evidence = l2_target_evolution._private_holdout_evidence(  # noqa: SLF001
        [round_result]
    )

    assert evidence["best_round"] == 3
    assert evidence["best_round_selection"]["status"] == "zero_accepts"
    assert evidence["best_round_promotion"]["status"] == "passes_gate"
    assert evidence["inner_passing_rounds"] == 1
    assert evidence["inner_passing_selection_zero_accept_rounds"] == 1
    assert (
        evidence["selection_gate_diagnosis"]
        == "selection_zero_accepts_for_inner_passing_rounds"
    )
    assert evidence["adoption_gate_diagnosis"] == "selection_gate_not_passed"
    assert "larger/stratified target split" in evidence["recommendation"]


def test_l2_target_evolve_cli_writes_summary(tmp_path: Path) -> None:
    traces_path = tmp_path / "traces.jsonl"
    traces_path.write_text(
        "".join(trace.model_dump_json() + "\n" for trace in _traces()),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "l2",
            "target-evolve",
            "--traces",
            str(traces_path),
            "--out-dir",
            str(tmp_path / "target-run"),
            "--rounds",
            "2",
            "--budget-profile",
            "fixed-inner",
        ],
    )

    assert result.exit_code == 0, result.output
    summary = json.loads((tmp_path / "target-run" / "summary.json").read_text())
    assert summary["rounds_completed"] == 2
    assert summary["budget_policy"]["profile"] == "fixed-inner"
    assert summary["budget_policy"]["inner_patience_rounds"] == 0
    assert summary["budget_policy"]["local_search_trials"] == 32
    assert summary["budget_policy"]["local_search_cross_audit_top_k"] == 4
    assert summary["budget_policy"]["visible_validation_folds"] == 5
    assert summary["budget_policy"]["visible_cross_audit_folds"] == 3
    assert summary["data_split_policy"]["visible_validation_folds"] == 5
    assert summary["budget_policy"]["stop_on_selection_gate"] is False
    assert summary["budget_policy"]["max_agent_rounds"] is None
    assert summary["baseline"]["visible_cross_audit"]["gate_role"] == (
        "diagnostic_only_not_selection_or_adoption_gate"
    )
    assert summary["data_split"]["train"] > 0


def test_l2_target_evolve_cli_allows_zero_agent_round_budget(tmp_path: Path) -> None:
    traces_path = tmp_path / "traces.jsonl"
    traces_path.write_text(
        "".join(trace.model_dump_json() + "\n" for trace in _traces()),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "l2",
            "target-evolve",
            "--traces",
            str(traces_path),
            "--out-dir",
            str(tmp_path / "target-run"),
            "--mode",
            "codex-cli",
            "--rounds",
            "2",
            "--max-agent-rounds",
            "0",
        ],
    )

    assert result.exit_code == 0, result.output
    summary = json.loads((tmp_path / "target-run" / "summary.json").read_text())
    assert summary["stop_reason"] == "agent_round_budget_exhausted"
    assert summary["rounds_completed"] == 0
    assert summary["agent_budget"]["max_agent_rounds"] == 0
    assert summary["agent_budget"]["agent_rounds_started"] == 0


def test_l2_target_evolve_cli_accepts_intent_stratified_split_policy(
    tmp_path: Path,
) -> None:
    traces_path = tmp_path / "traces.jsonl"
    traces_path.write_text(
        "".join(trace.model_dump_json() + "\n" for trace in _traces()),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "l2",
            "target-evolve",
            "--traces",
            str(traces_path),
            "--out-dir",
            str(tmp_path / "target-run"),
            "--rounds",
            "1",
            "--split-policy",
            "intent-stratified",
        ],
    )

    assert result.exit_code == 0, result.output
    summary = json.loads((tmp_path / "target-run" / "summary.json").read_text())
    assert summary["data_split_policy"]["policy"] == "intent-stratified"
    assert summary["data_split_policy"]["group_key"] == "teacher_frame.intent"


def test_l2_target_evolve_cli_accepts_local_search_mode(tmp_path: Path) -> None:
    traces_path = tmp_path / "traces.jsonl"
    traces_path.write_text(
        "".join(trace.model_dump_json() + "\n" for trace in _traces()),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "l2",
            "target-evolve",
            "--traces",
            str(traces_path),
            "--out-dir",
            str(tmp_path / "target-run"),
            "--rounds",
            "1",
            "--mode",
            "local-search",
            "--local-search-trials",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    summary = json.loads((tmp_path / "target-run" / "summary.json").read_text())
    assert summary["mode"] == "local-search"
    assert summary["budget_policy"]["local_search_trials"] == 2
    assert summary["budget_policy"]["local_search_cross_audit_top_k"] == 0


def test_l2_promote_target_cli_writes_runtime_artifacts(tmp_path: Path) -> None:
    target_run = tmp_path / "target-run"
    workspace = target_run / "workspace" / "l2_target"
    target_dir = workspace / "target"
    snapshot_dir = target_run / "rounds" / "round_001_target"
    data_dir = workspace / "data"
    target_dir.mkdir(parents=True)
    snapshot_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    (data_dir / "train.jsonl").write_text(
        "".join(
            trace.model_dump_json() + "\n"
            for trace in traces_to_teacher_view(_traces())
        ),
        encoding="utf-8",
    )
    (target_dir / "target_l2.py").write_text(
        """
def config_overrides():
    return {"accept_threshold": 0.9, "min_examples": 4}

def postprocess_frame(utterance, frame, metadata):
    del utterance, metadata
    return frame

FINAL_WORKSPACE_MARKER = True
""",
        encoding="utf-8",
    )
    (snapshot_dir / "target_l2.py").write_text(
        """
def config_overrides():
    return {"accept_threshold": 0.0, "min_examples": 4}

def postprocess_frame(utterance, frame, metadata):
    del utterance, metadata
    return frame

SELECTED_SNAPSHOT_MARKER = True
""",
        encoding="utf-8",
    )
    (target_run / "summary.json").write_text(
        json.dumps(
            {
                "mode": "dry-run",
                "workspace": str(workspace),
                "data_split": {
                    "train": 8,
                    "inner_validation": 2,
                    "selection_holdout": 1,
                    "promotion_holdout": 1,
                },
                "selection_decision": {"selected": True, "round": 1},
                "adoption_decision": {"adopted": True, "round": 1},
                "loop_cadence": {
                    "kind": "fixed_trace_snapshot_inner_loop",
                    "outer_replay_cadence_bound": False,
                },
                "target_code_policy": {
                    "core_must_remain_dataset_independent": True,
                    "target_dependent_code_allowed_in": "target/",
                    "target_specific_code_is_not_rejected_for_dataset_dependence": True,
                },
                "workspace_scope_policy": {
                    "schema_version": "l2-target-workspace-scope-v1",
                    "candidate_code_writable_roots": ["target/"],
                    "scratch_writable_roots": ["runs/"],
                    "protected_roots": [
                        "data/",
                        "system/darjeeling/",
                        "tools/",
                        "program.md",
                    ],
                    "ignored_generated_files": [
                        "__pycache__/",
                        ".pytest_cache/",
                        "*.pyc",
                        "*.pyo",
                    ],
                    "enforcement": (
                        "checked_after_each_mutating_round_before_candidate_evaluation"
                    ),
                },
                "rounds": [
                    {
                        "round": 1,
                        "target_snapshot": "rounds/round_001_target",
                        "inner_validation": {"accepted": 1, "wrong_accepts": 0},
                        "selection_holdout": {"accepted": 1, "wrong_accepts": 0},
                        "promotion_holdout": {"accepted": 1, "wrong_accepts": 0},
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "replay-run"

    result = CliRunner().invoke(
        app,
        [
            "l2",
            "promote-target",
            "--target-run",
            str(target_run),
            "--run-dir",
            str(run_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    manifest = json.loads(
        (run_dir / "artifacts" / "manifest.current.json").read_text(encoding="utf-8")
    )
    assert manifest["promoted"] is True
    assert manifest["promotion_reason"] == "explicit L2 target adoption passed gates"
    assert manifest["artifact_paths"]["l2_student"].endswith("l2_student.joblib")
    assert manifest["artifact_paths"]["l2_target"].endswith("target/target_l2.py")
    assert (
        run_dir
        / "artifacts"
        / manifest["artifact_paths"]["l2_target"]
    ).exists()
    assert manifest["candidate_metrics"]["l2_target_runtime_promoted"] is True
    assert manifest["candidate_metrics"]["l2_target_inner_adopted"] is True
    assert manifest["candidate_metrics"]["l2_target_staged_for_outer_replay"] is False
    assert manifest["candidate_metrics"]["l2_target_loop_cadence"] == {
        "kind": "fixed_trace_snapshot_inner_loop",
        "outer_replay_cadence_bound": False,
    }
    assert manifest["candidate_metrics"]["l2_target_code_policy"] == {
        "core_must_remain_dataset_independent": True,
        "target_dependent_code_allowed_in": "target/",
        "target_specific_code_is_not_rejected_for_dataset_dependence": True,
    }
    assert manifest["candidate_metrics"]["l2_target_workspace_scope_policy"] == {
        "schema_version": "l2-target-workspace-scope-v1",
        "candidate_code_writable_roots": ["target/"],
        "scratch_writable_roots": ["runs/"],
        "protected_roots": ["data/", "system/darjeeling/", "tools/", "program.md"],
        "ignored_generated_files": ["__pycache__/", ".pytest_cache/", "*.pyc", "*.pyo"],
        "enforcement": "checked_after_each_mutating_round_before_candidate_evaluation",
    }
    assert manifest["candidate_metrics"]["l2_target_training_traces"] == 12
    assert manifest["candidate_metrics"]["l2_training_scope"] == "l2_target_workspace_train"
    assert manifest["candidate_metrics"]["l2_training_traces"] == 12
    promoted_target = (
        run_dir / "artifacts" / manifest["artifact_paths"]["l2_target"]
    ).read_text(encoding="utf-8")
    assert "SELECTED_SNAPSHOT_MARKER" in promoted_target
    assert "FINAL_WORKSPACE_MARKER" not in promoted_target
    assert manifest["candidate_metrics"]["l2_config"]["accept_threshold"] == 0.0


def test_l2_promote_target_cli_can_stage_non_adopted_candidate_for_outer_replay(
    tmp_path: Path,
) -> None:
    target_run = tmp_path / "target-run"
    workspace = target_run / "workspace" / "l2_target"
    target_dir = workspace / "target"
    snapshot_dir = target_run / "rounds" / "round_001_target"
    data_dir = workspace / "data"
    target_dir.mkdir(parents=True)
    snapshot_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    (data_dir / "train.jsonl").write_text(
        "".join(
            trace.model_dump_json() + "\n"
            for trace in traces_to_teacher_view(_traces())
        ),
        encoding="utf-8",
    )
    (target_dir / "target_l2.py").write_text(
        """
def config_overrides():
    return {"accept_threshold": 0.9, "min_examples": 4}

def postprocess_frame(utterance, frame, metadata):
    del utterance, metadata
    return frame

FINAL_WORKSPACE_MARKER = True
""",
        encoding="utf-8",
    )
    (snapshot_dir / "target_l2.py").write_text(
        """
def config_overrides():
    return {"accept_threshold": 0.0, "min_examples": 4}

def postprocess_frame(utterance, frame, metadata):
    del utterance, metadata
    return frame

SELECTED_SNAPSHOT_MARKER = True
""",
        encoding="utf-8",
    )
    (target_run / "summary.json").write_text(
        json.dumps(
            {
                "mode": "dry-run",
                "workspace": str(workspace),
                "data_split": {
                    "train": 8,
                    "inner_validation": 2,
                    "selection_holdout": 1,
                    "promotion_holdout": 1,
                },
                "selection_decision": {"selected": False, "round": None},
                "adoption_decision": {"adopted": False, "round": None},
                "private_holdout_evidence": {
                    "schema_version": "l2-target-private-holdout-evidence-v1",
                    "selection_gate_diagnosis": (
                        "selection_zero_accepts_for_inner_passing_rounds"
                    ),
                },
                "best_round": {
                    "round": 1,
                    "target_snapshot": "rounds/round_001_target",
                    "inner_validation": {"accepted": 1, "wrong_accepts": 0},
                    "selection_holdout": {"accepted": 0, "wrong_accepts": 0},
                    "promotion_holdout": {"accepted": 1, "wrong_accepts": 0},
                },
                "rounds": [
                    {
                        "round": 1,
                        "target_snapshot": "rounds/round_001_target",
                        "inner_validation": {"accepted": 1, "wrong_accepts": 0},
                        "selection_holdout": {"accepted": 0, "wrong_accepts": 0},
                        "promotion_holdout": {"accepted": 1, "wrong_accepts": 0},
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rejected = CliRunner().invoke(
        app,
        [
            "l2",
            "promote-target",
            "--target-run",
            str(target_run),
            "--run-dir",
            str(tmp_path / "rejected-run"),
        ],
    )
    assert rejected.exit_code == 2
    assert "not adopted" in rejected.output

    run_dir = tmp_path / "replay-run"
    result = CliRunner().invoke(
        app,
        [
            "l2",
            "promote-target",
            "--target-run",
            str(target_run),
            "--run-dir",
            str(run_dir),
            "--allow-non-adopted",
        ],
    )

    assert result.exit_code == 0, result.output
    manifest = json.loads(
        (run_dir / "artifacts" / "manifest.current.json").read_text(encoding="utf-8")
    )
    assert (
        manifest["promotion_reason"]
        == "explicit L2 target candidate staged for outer replay"
    )
    assert manifest["candidate_metrics"]["l2_target_inner_adopted"] is False
    assert manifest["candidate_metrics"]["l2_target_staged_for_outer_replay"] is True
    assert manifest["candidate_metrics"]["l2_target_adopted_round"] is None
    assert manifest["candidate_metrics"]["l2_target_staged_round"] == 1
    assert manifest["candidate_metrics"]["l2_target_private_holdout_evidence"] == {
        "schema_version": "l2-target-private-holdout-evidence-v1",
        "selection_gate_diagnosis": "selection_zero_accepts_for_inner_passing_rounds",
    }
    promoted_target = (
        run_dir / "artifacts" / manifest["artifact_paths"]["l2_target"]
    ).read_text(encoding="utf-8")
    assert "SELECTED_SNAPSHOT_MARKER" in promoted_target
    assert "FINAL_WORKSPACE_MARKER" not in promoted_target
    assert manifest["candidate_metrics"]["l2_config"]["accept_threshold"] == 0.0


def test_l2_replay_target_cli_compares_current_target_against_parent(
    tmp_path: Path,
) -> None:
    target_run = tmp_path / "target-run"
    workspace = target_run / "workspace" / "l2_target"
    target_dir = workspace / "target"
    data_dir = workspace / "data"
    target_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    traces = _traces()
    (data_dir / "train.jsonl").write_text(
        "".join(
            trace.model_dump_json() + "\n"
            for trace in traces_to_teacher_view(traces)
        ),
        encoding="utf-8",
    )
    (target_dir / "target_l2.py").write_text(
        """
def config_overrides():
    return {"accept_threshold": 0.0, "min_examples": 4}

def postprocess_frame(utterance, frame, metadata):
    del frame, metadata
    if utterance == "alarm set example 0":
        return {"intent": "alarm_set", "slots": {"time": "0 am"}}
    return {"intent": "weather_query", "slots": {"location": "city 1"}}
""",
        encoding="utf-8",
    )
    (target_run / "summary.json").write_text(
        json.dumps(
            {
                "mode": "dry-run",
                "workspace": str(workspace),
                "data_split": {
                    "train": 8,
                    "inner_validation": 2,
                    "selection_holdout": 1,
                    "promotion_holdout": 1,
                },
                "selection_decision": {"selected": True, "round": 1},
                "adoption_decision": {"adopted": True, "round": 1},
                "rounds": [
                    {
                        "round": 1,
                        "inner_validation": {"accepted": 1, "wrong_accepts": 0},
                        "selection_holdout": {"accepted": 1, "wrong_accepts": 0},
                        "promotion_holdout": {"accepted": 1, "wrong_accepts": 0},
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "replay-run"
    ArtifactStore(run_dir / "artifacts").promote(
        ArtifactManifest(
            artifact_set_id="gen_001_baseline",
            generation=1,
            promotion_reason="test fixture",
        )
    )
    promote = CliRunner().invoke(
        app,
        [
            "l2",
            "promote-target",
            "--target-run",
            str(target_run),
            "--run-dir",
            str(run_dir),
        ],
    )
    assert promote.exit_code == 0, promote.output
    traces_path = tmp_path / "traces.jsonl"
    traces_path.write_text(traces[0].model_dump_json() + "\n", encoding="utf-8")
    out = tmp_path / "target-replay.json"

    result = CliRunner().invoke(
        app,
        [
            "l2",
            "replay-target",
            "--run-dir",
            str(run_dir),
            "--traces",
            str(traces_path),
            "--out",
            str(out),
            "--no-include-default-l1",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "l2-target-outer-replay-v1"
    assert payload["status"] == "success"
    assert payload["candidate_inner_adopted"] is True
    assert payload["candidate_staged_for_outer_replay"] is False
    assert payload["decision"]["promoted"] is True
    assert payload["baseline"]["layer_counts"]["L4"] == 1
    assert payload["candidate"]["layer_counts"]["L2"] == 1
    assert payload["candidate"]["objective"]["frame_exact_match"] == 1.0
    assert payload["per_layer_deltas"]["L2"]["layer_share_delta"] == 1.0
    assert payload["per_layer_deltas"]["L4"]["layer_share_delta"] == -1.0


def test_l2_replay_target_cli_requires_target_artifact(tmp_path: Path) -> None:
    run_dir = tmp_path / "replay-run"
    ArtifactStore(run_dir / "artifacts").promote(
        ArtifactManifest(
            artifact_set_id="gen_001_baseline",
            generation=1,
            promotion_reason="test fixture",
        )
    )
    traces_path = tmp_path / "traces.jsonl"
    traces_path.write_text(_traces()[0].model_dump_json() + "\n", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "l2",
            "replay-target",
            "--run-dir",
            str(run_dir),
            "--traces",
            str(traces_path),
            "--out",
            str(tmp_path / "target-replay.json"),
            "--no-include-default-l1",
        ],
    )

    assert result.exit_code == 2
    assert "does not contain an l2_target" in result.output
