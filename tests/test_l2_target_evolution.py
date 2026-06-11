import json
import subprocess
import sys
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
    l2_target_traces_for_scope,
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


def _trace_with_utterance(
    index: int,
    *,
    utterance: str,
    intent: str,
    slots: dict[str, str],
) -> TraceRecord:
    trace = _trace(index, intent=intent, slots=slots)
    trace.utterance = utterance
    return trace


def _slot_cue_probe_workspace(tmp_path: Path, target_code: str) -> Path:
    workspace = tmp_path / "workspace"
    (workspace / "data").mkdir(parents=True)
    (workspace / "target").mkdir()
    (workspace / "data" / "train.jsonl").write_text("", encoding="utf-8")
    (workspace / "data" / "slot_cue_probes.json").write_text(
        json.dumps(
            {
                "schema_version": "l2-target-slot-cue-probe-specs-v1",
                "visibility": "visible_validation_only",
                "probes": [
                    {
                        "id": "repair_missing_slot_alpha",
                        "utterance": "alpha cue value red",
                        "input_frame": {"intent": "intent_alpha", "slots": {}},
                        "expectation": "must_match",
                        "expected_slots": {"slot_alpha": "red"},
                    },
                    {
                        "id": "veto_or_remove_slot_beta",
                        "utterance": "beta cue has generic filler",
                        "input_frame": {
                            "intent": "intent_beta",
                            "slots": {"slot_beta": "generic filler"},
                        },
                        "expectation": "abstain_or_match",
                        "forbidden_slot_keys": ["slot_beta"],
                    },
                    {
                        "id": "repair_intent_boundary",
                        "utterance": "gamma cue belongs elsewhere",
                        "input_frame": {"intent": "intent_beta", "slots": {}},
                        "expectation": "must_match",
                        "expected_intent": "intent_gamma",
                    },
                    {
                        "id": "veto_unrepaired_delta",
                        "utterance": "delta cue ambiguous",
                        "input_frame": {"intent": "intent_delta", "slots": {}},
                        "expectation": "abstain_or_match",
                        "required_slot_keys": ["slot_delta"],
                    },
                    {
                        "id": "must_abstain_epsilon",
                        "utterance": "epsilon cue must abstain",
                        "input_frame": {"intent": "intent_epsilon", "slots": {}},
                        "expectation": "must_abstain",
                    },
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (workspace / "target" / "target_l2.py").write_text(
        target_code.strip() + "\n",
        encoding="utf-8",
    )
    return workspace


def _traces() -> list[TraceRecord]:
    return [
        _trace(index, intent="alarm_set", slots={"time": f"{index} am"})
        if index % 2 == 0
        else _trace(index, intent="weather_query", slots={"location": f"city {index}"})
        for index in range(12)
    ]


def _trace_with_lower_result(
    index: int,
    *,
    intent: str,
    slots: dict[str, str],
    lower_layer: str | None,
) -> TraceRecord:
    trace = _trace(index, intent=intent, slots=slots)
    if lower_layer is None:
        return trace
    frame = Frame(intent=intent, slots=slots)
    trace.layer_results.insert(
        0,
        LayerResult(layer=lower_layer, accepted=True, frame=frame, latency_ms=0.1),
    )
    trace.chosen_layer = lower_layer
    return trace


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
    assert summary["budget_policy"]["profile_intent"] == {
        "schema_version": "l2-target-budget-profile-intent-v1",
        "profile": "standard",
        "profile_role": "cost_capped_default",
        "recommended_quality_profile": "fixed-inner",
        "guidance": (
            "The standard profile is cost-capped. For codex-cli it may launch "
            "only a few live agent rounds, so failure here is not evidence that "
            "L2 target evolution has been exhausted."
        ),
        "fixed_trace_snapshot_inner_loop": True,
        "outer_replay_cadence_bound": False,
        "rounds_are_l2_train_eval_iterations": True,
        "agent_session_controls_internal_loop": False,
        "local_search_consumes_llm": False,
        "codex_cli_rounds_consume_llm": False,
        "live_agent_session_consumes_llm": False,
        "effective_max_agent_rounds": None,
        "agent_round_cap_is_cost_control": False,
    }
    assert summary["evidence_policy"]["schema_version"] == (
        "l2-target-evidence-policy-v1"
    )
    assert summary["evidence_policy"]["evidence_class"] == "cost_capped_probe"
    assert summary["evidence_policy"]["quality_claim_supported"] is False
    assert summary["evidence_policy"]["quality_claim"] == "not_supported_by_this_run"
    assert summary["evidence_policy"]["fixed_trace_snapshot_inner_loop"] is True
    assert summary["evidence_policy"]["outer_replay_cadence_bound"] is False
    assert summary["evidence_policy"]["teacher_labeled_traces"] == 12
    assert summary["evidence_policy"]["required_for_quality_claim"][
        "min_teacher_labeled_traces"
    ] == 500
    assert "standard profile is cost-capped" in summary["evidence_policy"][
        "blocking_reasons"
    ][0]
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
        "visible_validation_ratio_requested": None,
        "visible_validation_ratio_effective": 2 / 12,
        "visible_validation_visibility": "agent_workspace_visible",
        "private_splits": ["selection_holdout", "promotion_holdout"],
        "private_split_visibility": "outer_harness_only",
    }
    assert summary["loop_cadence"] == {
        "kind": "fixed_trace_snapshot_inner_loop",
        "outer_replay_cadence_bound": False,
        "teacher_labeled_traces": 12,
        "scoped_teacher_labeled_traces": 12,
        "note": (
            "target rounds reuse this fixed split; collecting another stream prefix "
            "is not part of the inner loop"
        ),
    }
    assert summary["target_scope"] == {
        "schema_version": "l2-target-scope-v1",
        "scope": "teacher_train",
        "input_teacher_labeled_traces": 12,
        "scoped_teacher_labeled_traces": 12,
        "lower_layer_accepted_excluded": 0,
        "selection_basis": "all teacher-labeled traces",
    }
    assert summary["target_code_policy"] == {
        "core_must_remain_dataset_independent": True,
        "target_dependent_code_allowed_in": "target/",
        "target_specific_code_is_not_rejected_for_dataset_dependence": True,
        "target_code_visibility_rule": (
            "target code may be derived from data/train.jsonl and "
            "visible data/inner_validation*.jsonl only"
        ),
        "generalization_rule": (
            "avoid exact utterance exceptions from a single visible row; prefer "
            "pattern-level rules with multiple visible supports or clear schema semantics"
        ),
        "private_holdout_visibility": (
            "selection/promotion holdouts remain outside the agent workspace"
        ),
        "adoption_authority": (
            "visible validation/support/train-audit gates, "
            "private selection/promotion gates, and final outer replay"
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
    assert manifest["optional_target_probe_files"] == ["slot_cue_probes.json"]
    assert manifest["commands"]["inspect_context"] == "python3 tools/inspect_context.py"
    assert "uv run --project system/darjeeling" in manifest["commands"][
        "evaluate_visible_validation"
    ]
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
    assert all(
        "passes_candidate_selection_gate" not in entry
        for entry in round_state["round_history"]
    )
    assert all(
        "passes_visible_validation_gate" in entry
        for entry in round_state["round_history"]
    )
    assert all(
        "passes_train_audit_safety_gate" in entry
        for entry in round_state["round_history"]
    )
    assert "candidate_selection" in objective["gates"]
    assert objective["gates"]["train_audit_safety"] == (
        "zero accepted wrong on visible train audit"
    )
    assert objective["target_scope"]["scope"] == "teacher_train"
    assert objective["workspace_scope"]["candidate_code_writable_roots"] == ["target/"]
    assert objective["workspace_scope"]["scratch_writable_roots"] == ["runs/"]
    assert "system/darjeeling/" in objective["workspace_scope"]["protected_roots"]
    assert any(
        "near_miss_examples" in strategy
        for strategy in objective["allowed_strategies"]
    )
    assert any(
        "single-visible-row exact utterance" in strategy
        for strategy in objective["invalid_strategies"]
    )
    assert any(
        "lowering accept_threshold only to raise raw accepts" in strategy
        for strategy in objective["invalid_strategies"]
    )
    assert "multiple visible supports" in json.dumps(objective["allowed_strategies"])
    assert "Private selection" in program_text
    assert "alone is not success" in program_text
    assert "outer selection signal" in program_text
    assert "not a signal you can" in program_text
    assert "read during this session" in program_text
    assert "near_miss_examples" in program_text
    assert "target_diagnostics.json" in program_text
    assert "latest_safety_backlog" in program_text
    assert "Do not add exact utterance exceptions" in program_text
    assert "multiple visible examples" in program_text
    assert "target-supplied" in program_text
    assert "slot_cue_probes" in program_text
    assert "Once visible support passes" in program_text
    assert "target/config.json" in program_text
    assert target_diagnostics["schema_version"] == "l2-target-diagnostics-v1"
    assert target_diagnostics["visibility"] == "visible_validation_only"
    assert target_diagnostics["visible_slot_cue_summary"]["schema_version"] == (
        "l2-target-visible-slot-cue-summary-v1"
    )
    assert target_diagnostics["visible_slot_cue_summary"]["visibility"] == (
        "visible_validation_only"
    )
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
    assert target_diagnostics["baseline_slot_risk_backlog"]["schema_version"] == (
        "l2-target-slot-risk-backlog-v1"
    )
    assert target_diagnostics["baseline_train_audit_slot_risk_backlog"][
        "schema_version"
    ] == "l2-target-slot-risk-backlog-v1"
    assert "latest_train_audit_safety_backlog" in target_diagnostics
    assert "latest_slot_risk_backlog" in target_diagnostics
    assert "latest_train_audit_slot_risk_backlog" in target_diagnostics
    assert "latest_visible_cross_audit_slot_risk_backlog" in target_diagnostics
    assert "latest_intent_confusion_backlog" in target_diagnostics
    assert "latest_train_audit_intent_confusion_backlog" in target_diagnostics
    assert "latest_visible_cross_audit_intent_confusion_backlog" in target_diagnostics
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
    assert "safety gate" in round_state["train_audit_policy"]
    assert summary["rounds"][0]["train_audit"]["gate_role"] == (
        "diagnostic_only_not_selection_or_adoption_gate"
    )
    assert round_state["agent_budget"]["mode"] == "dry-run"
    assert round_state["target_scope"]["scope"] == "teacher_train"
    assert round_state["budget_policy"]["profile_intent"]["profile_role"] == (
        "cost_capped_default"
    )
    assert round_state["evidence_policy"]["evidence_class"] == "cost_capped_probe"
    assert round_state["evidence_policy"]["quality_claim_supported"] is False
    assert objective["budget_policy"]["profile_intent"]["profile_role"] == (
        "cost_capped_default"
    )
    assert objective["evidence_policy"]["evidence_class"] == "cost_capped_probe"
    assert objective["agent_budget"]["local_search_consumes_llm"] is False
    assert "private_holdout_evidence" not in round_state
    assert "not a" in program_text
    assert "Darjeeling-core dataset-independence violation" in program_text
    inspect_result = subprocess.run(
        [sys.executable, "tools/inspect_context.py"],
        cwd=workspace,
        text=True,
        capture_output=True,
        check=False,
    )
    assert inspect_result.returncode == 0, inspect_result.stderr
    assert "inner_validation.jsonl" in inspect_result.stdout
    assert "train.jsonl" in inspect_result.stdout
    assert "target_l2.py" in inspect_result.stdout
    assert "selection_holdout" not in inspect_result.stdout
    assert "promotion_holdout" not in inspect_result.stdout


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
    high_guard_slot_example = {
        "request_id": "visible-risk-2",
        "utterance": "play the morning show in the kitchen",
        "teacher_frame": {
            "intent": "play_radio",
            "slots": {"house_place": "kitchen"},
        },
        "predicted_frame": {"intent": "play_radio", "slots": {}},
        "guard_probability": 0.995,
    }
    intent_confusion_example = {
        "request_id": "visible-risk-3",
        "utterance": "play the newest science podcast",
        "teacher_frame": {
            "intent": "play_podcasts",
            "slots": {"podcast_name": "science"},
        },
        "predicted_frame": {"intent": "play_radio", "slots": {}},
        "guard_probability": 0.94,
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
            "missing_slot_keys": {},
            "extra_slot_keys": {"date": 1},
            "changed_slot_keys": {},
            "predicted_intents": {"general_quirky": 5},
            "examples": {
                "accepted_wrong": [risky_example],
                "rejected_correct": [],
                "vetoed_correct": [],
                "intent_correct_slot_wrong": [risky_example],
            },
        },
        "high_guard_slot_risk": {
            "teacher_intent": "play_radio",
            "total": 2,
            "accepted_correct": 0,
            "accepted_wrong": 0,
            "rejected_correct": 1,
            "rejected_wrong": 0,
            "vetoed_correct": 0,
            "vetoed_wrong": 0,
            "intent_correct_slot_wrong": 1,
            "missing_slot_keys": {"house_place": 1},
            "extra_slot_keys": {},
            "changed_slot_keys": {},
            "predicted_intents": {"play_radio": 2},
            "examples": {
                "accepted_wrong": [],
                "rejected_correct": [],
                "vetoed_correct": [],
                "intent_correct_slot_wrong": [high_guard_slot_example],
            },
        },
        "intent_confusion_risk": {
            "teacher_intent": "play_podcasts",
            "total": 3,
            "accepted_correct": 0,
            "accepted_wrong": 0,
            "rejected_correct": 0,
            "rejected_wrong": 3,
            "vetoed_correct": 0,
            "vetoed_wrong": 0,
            "intent_correct_slot_wrong": 0,
            "predicted_intents": {"play_radio": 3},
            "intent_confusions": {
                "play_radio": {
                    "teacher_intent": "play_podcasts",
                    "predicted_intent": "play_radio",
                    "total": 3,
                    "default_accepts": 1,
                    "accepted_wrong": 0,
                    "max_guard_probability": 0.94,
                    "examples": [intent_confusion_example],
                },
            },
            "examples": {
                "accepted_wrong": [],
                "rejected_correct": [],
                "vetoed_correct": [],
                "intent_correct_slot_wrong": [],
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
    slot_risk_backlog = payload["slot_risk_backlog"]
    assert slot_risk_backlog["schema_version"] == "l2-target-slot-risk-backlog-v1"
    assert slot_risk_backlog["priority"] == (
        "review_visible_slot_mismatches_after_accepted_wrong_backlog"
    )
    assert [item["teacher_intent"] for item in slot_risk_backlog["items"]] == [
        "calendar_query",
        "general_quirky",
        "play_radio",
    ]
    assert slot_risk_backlog["items"][0]["intent_correct_slot_wrong"] == 5
    assert slot_risk_backlog["items"][1]["slot_mismatch_examples"] == [risky_example]
    assert slot_risk_backlog["high_guard_item_limit"] == 8
    assert slot_risk_backlog["high_guard_items"][0]["teacher_intent"] == "play_radio"
    assert slot_risk_backlog["high_guard_items"][0]["slot_mismatch_examples"] == [
        high_guard_slot_example,
    ]
    assert slot_risk_backlog["high_guard_items"][0]["missing_slot_keys"] == [
        {"slot_key": "house_place", "count": 1},
    ]
    assert slot_risk_backlog["items"][1]["extra_slot_keys"] == [
        {"slot_key": "date", "count": 1},
    ]
    assert "postprocess" in slot_risk_backlog["items"][0]["recommended_action"]
    intent_confusion_backlog = payload["intent_confusion_backlog"]
    assert intent_confusion_backlog["schema_version"] == (
        "l2-target-intent-confusion-backlog-v1"
    )
    assert intent_confusion_backlog["items"][0]["teacher_intent"] == "play_podcasts"
    assert intent_confusion_backlog["items"][0]["predicted_intent"] == "play_radio"
    assert intent_confusion_backlog["items"][0]["examples"] == [
        intent_confusion_example,
    ]


def test_l2_target_private_holdout_safety_backlog_marks_outer_visibility() -> None:
    family_stats = {
        "accepted_wrong_risk": {
            "teacher_intent": "email_query",
            "total": 2,
            "accepted_correct": 0,
            "accepted_wrong": 1,
            "rejected_correct": 0,
            "rejected_wrong": 1,
            "vetoed_correct": 0,
            "vetoed_wrong": 0,
            "intent_correct_slot_wrong": 1,
            "predicted_intents": {"email_query": 2},
            "examples": {
                "accepted_wrong": [],
                "rejected_correct": [],
                "vetoed_correct": [],
                "intent_correct_slot_wrong": [],
            },
        },
    }

    payload = l2_target_evolution._family_diagnostics_payload(
        split="promotion_holdout",
        validation_size=10,
        family_stats=family_stats,
    )

    assert payload["safety_backlog"]["visibility"] == (
        "outer_summary_only_not_agent_workspace"
    )
    assert payload["slot_risk_backlog"]["visibility"] == (
        "outer_summary_only_not_agent_workspace"
    )
    assert payload["intent_confusion_backlog"]["visibility"] == (
        "outer_summary_only_not_agent_workspace"
    )


def test_l2_target_visible_slot_cue_summary_exposes_cross_intent_slot_values() -> None:
    traces = traces_to_teacher_view(
        [
            _trace_with_utterance(
                1,
                utterance="turn off the kitchen lights",
                intent="iot_hue_lightoff",
                slots={"house_place": "kitchen"},
            ),
            _trace_with_utterance(
                2,
                utterance="start vacuuming the bathroom",
                intent="iot_cleaning",
                slots={"house_place": "bathroom"},
            ),
            _trace_with_utterance(
                3,
                utterance="what is the weather in paris",
                intent="weather_query",
                slots={"place_name": "paris"},
            ),
        ]
    )

    payload = l2_target_evolution._visible_slot_cue_summary_payload(
        traces=traces,
        source_splits=["train", "inner_validation"],
    )

    assert payload["schema_version"] == "l2-target-visible-slot-cue-summary-v1"
    assert payload["visibility"] == "visible_validation_only"
    assert "slotless or missing-slot accepted frames" in payload["usage_hint"]
    house_place = next(
        item for item in payload["items"] if item["slot_key"] == "house_place"
    )
    assert house_place["total"] == 2
    assert house_place["slot_key_terms"] == ["house", "place"]
    assert house_place["top_values"] == [
        {"value": "bathroom", "count": 1},
        {"value": "kitchen", "count": 1},
    ]
    assert house_place["top_teacher_intents"] == [
        {"intent": "iot_cleaning", "count": 1},
        {"intent": "iot_hue_lightoff", "count": 1},
    ]
    assert house_place["examples"][0] == {
        "request_id": "r1",
        "utterance": "turn off the kitchen lights",
        "teacher_intent": "iot_hue_lightoff",
        "slot_value": "kitchen",
    }


def test_l2_target_visible_slot_cue_summary_keeps_low_frequency_schema_cues() -> None:
    trace_records: list[TraceRecord] = []
    for index in range(1, 46):
        trace_records.extend(
            [
                _trace_with_utterance(
                    index * 10,
                    utterance=f"visible cue {index} alpha",
                    intent=f"visible_intent_{index}",
                    slots={f"frequent_slot_{index:02d}": "alpha"},
                ),
                _trace_with_utterance(
                    index * 10 + 1,
                    utterance=f"visible cue {index} beta",
                    intent=f"visible_intent_{index}",
                    slots={f"frequent_slot_{index:02d}": "beta"},
                ),
            ]
        )
    trace_records.append(
        _trace_with_utterance(
            999,
            utterance="alpha item with rare cue",
            intent="intent_alpha",
            slots={"rare_item_name": "rare cue"},
        )
    )
    traces = traces_to_teacher_view(trace_records)

    payload = l2_target_evolution._visible_slot_cue_summary_payload(
        traces=traces,
        source_splits=["train", "inner_validation"],
    )

    rare_item_name = next(
        item for item in payload["items"] if item["slot_key"] == "rare_item_name"
    )
    assert rare_item_name["slot_key_terms"] == ["rare", "item", "name"]
    assert rare_item_name["top_teacher_intents"] == [
        {"intent": "intent_alpha", "count": 1}
    ]
    assert rare_item_name["examples"][0]["utterance"] == "alpha item with rare cue"


def test_l2_target_slot_cue_probes_fail_default_slotless_accepts(tmp_path: Path) -> None:
    workspace = _slot_cue_probe_workspace(
        tmp_path,
        """
def config_overrides():
    return {}


def postprocess_frame(utterance, frame, metadata):
    return frame


def accept_prediction(utterance, frame, metadata, default_accept):
    return default_accept
""",
    )

    payload = evaluate_target_workspace(
        workspace_root=workspace,
        split="slot_cue_probes",
    )

    assert payload["schema_version"] == "l2-target-slot-cue-probes-v1"
    assert payload["visibility"] == "visible_validation_only"
    assert payload["gate_role"] == "diagnostic_only_not_selection_or_adoption_gate"
    assert payload["passes_gate"] is False
    assert [item["id"] for item in payload["failed_checks"]] == [
        "repair_missing_slot_alpha",
        "veto_or_remove_slot_beta",
        "repair_intent_boundary",
        "veto_unrepaired_delta",
        "must_abstain_epsilon",
    ]


def test_l2_target_slot_cue_probes_pass_visible_vetoes(tmp_path: Path) -> None:
    workspace = _slot_cue_probe_workspace(
        tmp_path,
        """
def config_overrides():
    return {}


def postprocess_frame(utterance, frame, metadata):
    text = utterance.lower()
    intent = frame.get("intent")
    slots = frame.setdefault("slots", {})
    if intent == "intent_alpha" and "alpha cue value red" in text:
        slots["slot_alpha"] = "red"
    if intent == "intent_beta" and "generic filler" in text:
        slots.pop("slot_beta", None)
    if "gamma cue belongs elsewhere" in text:
        frame["intent"] = "intent_gamma"
    return frame


def accept_prediction(utterance, frame, metadata, default_accept):
    text = utterance.lower()
    if "delta cue ambiguous" in text:
        return False
    if "epsilon cue must abstain" in text:
        return False
    return default_accept
""",
    )

    payload = evaluate_target_workspace(
        workspace_root=workspace,
        split="slot_cue_probes",
    )

    assert payload["passes_gate"] is True
    assert payload["failed_checks"] == []
    assert payload["probe_count"] == 5


def test_l2_target_slot_cue_probes_are_empty_without_target_specs(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "data").mkdir(parents=True)
    (workspace / "target").mkdir()
    (workspace / "data" / "train.jsonl").write_text("", encoding="utf-8")
    (workspace / "target" / "target_l2.py").write_text(
        """
def config_overrides():
    return {}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    payload = evaluate_target_workspace(
        workspace_root=workspace,
        split="slot_cue_probes",
    )

    assert payload["probe_count"] == 0
    assert payload["passes_gate"] is True
    assert payload["empty_reason"] == "missing_slot_cue_probe_specs"


def test_l2_target_core_does_not_embed_default_application_probe_terms() -> None:
    source = Path(l2_target_evolution.__file__).read_text(encoding="utf-8")

    forbidden_terms = [
        "play_radio",
        "podcast_name",
        "general_joke",
        "calendar_remove",
        "recommendation_events",
        "house_place",
        "radio_name",
        "joke_type",
        "change_amount",
        "MASSIVE-specific",
    ]

    assert [term for term in forbidden_terms if term in source] == []


def test_l2_target_aggregate_slot_risk_backlog_keeps_high_guard_view() -> None:
    volume_example = {
        "request_id": "visible-volume-risk",
        "utterance": "add reminder for tomorrow",
        "teacher_frame": {
            "intent": "calendar_set",
            "slots": {"date": "tomorrow"},
        },
        "predicted_frame": {"intent": "calendar_set", "slots": {}},
        "guard_probability": 0.4,
    }
    high_guard_example = {
        "request_id": "visible-high-guard-risk",
        "utterance": "tell me a joke about airplanes",
        "teacher_frame": {
            "intent": "general_joke",
            "slots": {"joke_type": "airplanes"},
        },
        "predicted_frame": {"intent": "general_joke", "slots": {}},
        "guard_probability": 0.97,
    }

    payload = l2_target_evolution._aggregate_slot_risk_backlogs(
        split="visible_cross_audit",
        validation_size=100,
        backlogs=[
            {
                "items": [
                    {
                        "teacher_intent": "calendar_set",
                        "total": 20,
                        "accepted_correct": 0,
                        "accepted_wrong": 0,
                        "intent_correct_slot_wrong": 10,
                        "max_slot_mismatch_guard_probability": 0.4,
                        "top_predicted_intents": [
                            {"intent": "calendar_set", "count": 20},
                        ],
                        "missing_slot_keys": [
                            {"slot_key": "date", "count": 10},
                        ],
                        "extra_slot_keys": [],
                        "changed_slot_keys": [],
                        "slot_mismatch_examples": [volume_example],
                    },
                    {
                        "teacher_intent": "general_joke",
                        "total": 3,
                        "accepted_correct": 0,
                        "accepted_wrong": 0,
                        "intent_correct_slot_wrong": 1,
                        "max_slot_mismatch_guard_probability": 0.97,
                        "top_predicted_intents": [
                            {"intent": "general_joke", "count": 3},
                        ],
                        "missing_slot_keys": [
                            {"slot_key": "joke_type", "count": 1},
                        ],
                        "extra_slot_keys": [],
                        "changed_slot_keys": [],
                        "slot_mismatch_examples": [high_guard_example],
                    },
                ],
            },
        ],
    )

    assert payload["items"][0]["teacher_intent"] == "calendar_set"
    assert payload["high_guard_items"][0]["teacher_intent"] == "general_joke"
    assert payload["high_guard_items"][0]["slot_mismatch_examples"] == [
        high_guard_example,
    ]
    assert payload["high_guard_items"][0]["missing_slot_keys"] == [
        {"slot_key": "joke_type", "count": 1},
    ]


def test_l2_target_aggregate_intent_confusion_backlog_merges_pairs() -> None:
    confusion_example = {
        "request_id": "visible-confusion-risk",
        "utterance": "play the science podcast",
        "teacher_frame": {
            "intent": "play_podcasts",
            "slots": {"podcast_name": "science"},
        },
        "predicted_frame": {"intent": "play_radio", "slots": {}},
        "guard_probability": 0.96,
    }

    payload = l2_target_evolution._aggregate_intent_confusion_backlogs(
        split="visible_cross_audit",
        validation_size=100,
        backlogs=[
            {
                "items": [
                    {
                        "teacher_intent": "play_podcasts",
                        "predicted_intent": "play_radio",
                        "total": 2,
                        "default_accepts": 1,
                        "accepted_wrong": 0,
                        "max_guard_probability": 0.96,
                        "examples": [confusion_example],
                    },
                ],
            },
        ],
    )

    assert payload["schema_version"] == "l2-target-intent-confusion-backlog-v1"
    assert payload["visibility"] == "visible_validation_only"
    assert payload["items"][0]["teacher_intent"] == "play_podcasts"
    assert payload["items"][0]["predicted_intent"] == "play_radio"
    assert payload["items"][0]["default_accepts"] == 1
    assert payload["items"][0]["examples"] == [confusion_example]


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


def test_l2_target_lower_miss_scope_filters_lower_layer_accepts(tmp_path: Path) -> None:
    traces = [
        _trace_with_lower_result(
            index,
            intent="alarm_set" if index % 2 == 0 else "weather_query",
            slots=(
                {"time": f"{index} am"}
                if index % 2 == 0
                else {"location": f"city {index}"}
            ),
            lower_layer="L0" if index < 5 else ("L1" if index < 10 else None),
        )
        for index in range(24)
    ]
    teacher_traces = traces_to_teacher_view(traces)

    scoped = l2_target_traces_for_scope(teacher_traces, scope="lower_miss")
    assert [trace.request_id for trace in scoped] == [f"r{index}" for index in range(10, 24)]

    summary = run_l2_target_evolution(
        config=L2TargetEvolutionConfig(
            source_repo_dir=Path.cwd(),
            job_dir=tmp_path / "job",
            rounds=1,
            mode="dry-run",
            target_scope="lower_miss",
            inner_patience_rounds=0,
        ),
        traces=teacher_traces,
    )

    workspace = tmp_path / "job" / "workspace" / "l2_target"
    round_state = json.loads((workspace / "data" / "round_state.json").read_text())
    train_rows = [
        json.loads(line)
        for line in (workspace / "data" / "train.jsonl").read_text().splitlines()
    ]

    assert summary["target_scope"] == {
        "schema_version": "l2-target-scope-v1",
        "scope": "lower_miss",
        "input_teacher_labeled_traces": 24,
        "scoped_teacher_labeled_traces": 14,
        "lower_layer_accepted_excluded": 10,
        "selection_basis": "teacher-labeled traces where L0/L1 did not accept",
    }
    assert summary["loop_cadence"]["teacher_labeled_traces"] == 24
    assert summary["loop_cadence"]["scoped_teacher_labeled_traces"] == 14
    assert sum(summary["data_split"].values()) == 14
    assert round_state["target_scope"]["scope"] == "lower_miss"
    assert all(int(row["request_id"][1:]) >= 10 for row in train_rows)


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


def test_l2_target_visible_validation_ratio_expands_visible_pool() -> None:
    traces = [
        _trace(index, intent="alarm_set", slots={"time": f"{index} am"})
        if index % 2 == 0
        else _trace(index, intent="weather_query", slots={"location": f"city {index}"})
        for index in range(100)
    ]

    split = split_l2_target_traces(
        traces_to_teacher_view(traces),
        visible_validation_folds=5,
        visible_validation_ratio=0.40,
    )

    assert len(split["train"]) == 40
    assert len(split["selection_holdout"]) == 10
    assert len(split["promotion_holdout"]) == 10
    assert sum(
        len(value) for key, value in split.items() if key.startswith("inner_validation")
    ) == 40


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
        "enforcement": (
            "checked_after_each_mutating_round_or_session_before_candidate_evaluation"
        ),
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


def test_l2_target_local_search_vetoes_cross_audit_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "target").mkdir(parents=True)
    config = l2_target_evolution.L2StudentConfig().model_dump(mode="json")
    calls: list[str] = []

    def metric(split: str, *, passes_gate: bool, accepted_accuracy: float) -> dict:
        return {
            "split": split,
            "train_size": 10,
            "validation_size": 10,
            "accepted": 2,
            "correct_accepts": 2 if passes_gate else 1,
            "wrong_accepts": 0 if passes_gate else 1,
            "vetoed_accepts": 0,
            "coverage": 0.2,
            "accepted_accuracy": accepted_accuracy,
            "wrong_accept_rate": 0.0 if passes_gate else 0.1,
            "passes_gate": passes_gate,
            "config": config,
            "wrong_examples": [],
            "veto_examples": [],
            "near_miss_examples": [],
        }

    def fake_evaluate_target_workspace(**kwargs) -> dict:
        split = kwargs["split"]
        calls.append(split)
        if split == "visible_cross_audit":
            return metric(split, passes_gate=False, accepted_accuracy=0.5)
        return metric(split, passes_gate=True, accepted_accuracy=1.0)

    monkeypatch.setattr(
        l2_target_evolution,
        "evaluate_target_workspace",
        fake_evaluate_target_workspace,
    )

    report = l2_target_evolution.run_local_target_search(
        workspace_root=workspace,
        trials=1,
        cross_audit_folds=2,
        cross_audit_top_k=1,
    )

    assert report["cross_audit_rerank_enabled"] is True
    assert report["best_inner_validation"]["passes_gate"] is True
    assert report["best_visible_cross_audit"]["passes_gate"] is False
    assert report["cross_audit_safety_veto"] is True
    assert report["applied"] is False
    assert report["applied_reason"] == (
        "best visible/cross-audit config failed visible cross-audit safety gate"
    )
    assert not (workspace / "target" / "config.json").exists()
    assert calls.count("visible_cross_audit") >= 2


def test_l2_target_local_search_current_cross_audit_uses_original_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    target_dir = workspace / "target"
    target_dir.mkdir(parents=True)
    original_config = l2_target_evolution.L2StudentConfig(
        accept_threshold=0.42,
    ).model_dump(mode="json")
    (target_dir / "config.json").write_text(
        json.dumps(original_config),
        encoding="utf-8",
    )
    cross_audit_thresholds: list[float] = []

    def metric(split: str, config: dict) -> dict:
        return {
            "split": split,
            "train_size": 10,
            "validation_size": 10,
            "accepted": 2,
            "correct_accepts": 2,
            "wrong_accepts": 0,
            "vetoed_accepts": 0,
            "coverage": 0.2,
            "accepted_accuracy": 1.0,
            "wrong_accept_rate": 0.0,
            "passes_gate": True,
            "config": config,
            "wrong_examples": [],
            "veto_examples": [],
            "near_miss_examples": [],
        }

    def fake_evaluate_target_workspace(**kwargs) -> dict:
        config_path = workspace / "target" / "config.json"
        config = (
            json.loads(config_path.read_text(encoding="utf-8"))
            if config_path.exists()
            else l2_target_evolution.L2StudentConfig().model_dump(mode="json")
        )
        if kwargs["split"] == "visible_cross_audit":
            cross_audit_thresholds.append(float(config["accept_threshold"]))
        return metric(kwargs["split"], config)

    monkeypatch.setattr(
        l2_target_evolution,
        "evaluate_target_workspace",
        fake_evaluate_target_workspace,
    )

    l2_target_evolution.run_local_target_search(
        workspace_root=workspace,
        trials=1,
        cross_audit_folds=2,
        cross_audit_top_k=1,
    )

    assert cross_audit_thresholds
    assert cross_audit_thresholds[0] == 0.42


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


def test_l2_target_agent_session_launches_once_and_evaluates_candidate(
    tmp_path: Path,
) -> None:
    fake_codex = tmp_path / "fake_codex.py"
    fake_codex.write_text(
        """#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

args = sys.argv[1:]
workspace = Path(args[args.index("--cd") + 1])
report = Path(args[args.index("-o") + 1])
prompt = sys.stdin.read()
(workspace / "target" / "config.json").write_text(
    json.dumps({"accept_threshold": 0.98, "min_examples": 4}) + "\\n",
    encoding="utf-8",
)
(workspace / "runs").mkdir(exist_ok=True)
(workspace / "runs" / "agent_note.txt").write_text(
    "fake agent session completed\\n",
    encoding="utf-8",
)
report.write_text("fake agent session report\\n", encoding="utf-8")
print(json.dumps({"prompt": prompt, "workspace": str(workspace)}))
""",
        encoding="utf-8",
    )
    fake_codex.chmod(0o755)

    summary = run_l2_target_evolution(
        config=L2TargetEvolutionConfig(
            source_repo_dir=Path.cwd(),
            job_dir=tmp_path / "job",
            rounds=5,
            mode="agent-session",
            codex_command=str(fake_codex),
            codex_model=None,
            inner_patience_rounds=0,
        ),
        traces=traces_to_teacher_view(_traces()),
    )

    workspace = tmp_path / "job" / "workspace" / "l2_target"
    round_state = json.loads(
        (workspace / "data" / "round_state.json").read_text(encoding="utf-8")
    )
    transcript = (tmp_path / "job" / "transcripts" / "agent_session.jsonl").read_text(
        encoding="utf-8",
    )

    assert summary["mode"] == "agent-session"
    assert summary["rounds_requested"] == 5
    assert summary["rounds_completed"] == 1
    assert summary["stop_reason"] == "agent_session_completed"
    assert summary["agent_budget"]["applies_to_mode"] is True
    assert summary["agent_budget"]["max_agent_rounds"] == 1
    assert summary["agent_budget"]["agent_rounds_started"] == 1
    assert summary["agent_budget"]["agent_rounds_succeeded"] == 1
    assert summary["agent_budget"]["agent_rounds_remaining"] == 0
    assert summary["agent_budget"]["agent_session_scope"] == (
        "single_session_agent_controls_internal_loop"
    )
    assert summary["rounds"][0]["agent_session"] == {
        "schema_version": "l2-target-agent-session-v1",
        "session_scope": "single long-running L4 agent session",
        "internal_loop_control": "agent_decides_edit_evaluate_search_stop",
        "tool_policy": "agent may call visible tools/evaluate.py and tools/search_config.py",
        "private_holdout_visibility": (
            "selection and promotion holdouts are evaluated only after session exit"
        ),
    }
    assert "autonomous L2 target evolution session" in transcript
    assert (workspace / "target" / "config.json").exists()
    assert (workspace / "runs" / "agent_note.txt").exists()
    assert not (workspace / "data" / "selection_holdout.jsonl").exists()
    assert not (workspace / "data" / "promotion_holdout.jsonl").exists()
    assert round_state["state_kind"] == "final"
    assert round_state["agent_budget"]["agent_rounds_succeeded"] == 1


def test_l2_target_agent_session_failure_cannot_support_quality_claim(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fake_codex = tmp_path / "fake_codex.py"
    fake_codex.write_text(
        "#!/usr/bin/env python3\nimport sys\nsys.exit(7)\n",
        encoding="utf-8",
    )
    fake_codex.chmod(0o755)

    def metric(split: str) -> dict:
        return {
            "split": split,
            "train_size": 260,
            "validation_size": 50,
            "accepted": 0,
            "correct_accepts": 0,
            "wrong_accepts": 0,
            "vetoed_accepts": 0,
            "coverage": 0.0,
            "accepted_accuracy": 0.0,
            "wrong_accept_rate": 0.0,
            "passes_gate": False,
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

    traces = [
        _trace(index, intent="alarm_set", slots={"time": f"{index} am"})
        if index % 2 == 0
        else _trace(index, intent="weather_query", slots={"location": f"city {index}"})
        for index in range(520)
    ]
    summary = run_l2_target_evolution(
        config=L2TargetEvolutionConfig(
            source_repo_dir=Path.cwd(),
            job_dir=tmp_path / "job",
            rounds=48,
            mode="agent-session",
            budget_profile="fixed-inner",
            codex_command=str(fake_codex),
            codex_model=None,
            inner_patience_rounds=0,
        ),
        traces=traces_to_teacher_view(traces),
    )

    assert summary["stop_reason"] == "agent_session_failed"
    assert summary["rounds_completed"] == 0
    assert summary["evidence_policy"]["evidence_class"] == "incomplete_agent_session_probe"
    assert summary["evidence_policy"]["quality_claim_supported"] is False
    assert any(
        "did not complete one scoped candidate evaluation" in reason
        for reason in summary["evidence_policy"]["blocking_reasons"]
    )


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
        mode="agent-session",
        budget_profile="fixed-inner",
        max_agent_rounds=None,
    ) == 1
    assert _resolve_l2_target_agent_rounds(
        mode="agent-session",
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
    standard_codex_intent = l2_target_evolution._target_budget_policy_payload(  # noqa: SLF001
        L2TargetEvolutionConfig(
            source_repo_dir=Path.cwd(),
            job_dir=Path("unused"),
            mode="codex-cli",
            budget_profile="standard",
        )
    )["profile_intent"]
    fixed_inner_intent = l2_target_evolution._target_budget_policy_payload(  # noqa: SLF001
        L2TargetEvolutionConfig(
            source_repo_dir=Path.cwd(),
            job_dir=Path("unused"),
            mode="codex-cli",
            budget_profile="fixed-inner",
            rounds=48,
        )
    )["profile_intent"]
    standard_evidence = l2_target_evolution._target_evidence_policy_payload(  # noqa: SLF001
        L2TargetEvolutionConfig(
            source_repo_dir=Path.cwd(),
            job_dir=Path("unused"),
            mode="codex-cli",
            budget_profile="standard",
        )
    )
    fixed_inner_short_evidence = (  # noqa: SLF001
        l2_target_evolution._target_evidence_policy_payload(
            L2TargetEvolutionConfig(
                source_repo_dir=Path.cwd(),
                job_dir=Path("unused"),
                budget_profile="fixed-inner",
                rounds=2,
            )
        )
    )
    fixed_inner_small_snapshot_evidence = (  # noqa: SLF001
        l2_target_evolution._target_evidence_policy_payload(
            L2TargetEvolutionConfig(
                source_repo_dir=Path.cwd(),
                job_dir=Path("unused"),
                budget_profile="fixed-inner",
                rounds=48,
            ),
            teacher_labeled_traces=120,
        )
    )
    fixed_inner_quality_evidence = (  # noqa: SLF001
        l2_target_evolution._target_evidence_policy_payload(
            L2TargetEvolutionConfig(
                source_repo_dir=Path.cwd(),
                job_dir=Path("unused"),
                mode="codex-cli",
                budget_profile="fixed-inner",
                rounds=48,
            ),
            teacher_labeled_traces=500,
        )
    )
    assert standard_codex_intent["profile_role"] == "cost_capped_default"
    assert standard_codex_intent["effective_max_agent_rounds"] == 3
    assert standard_codex_intent["agent_round_cap_is_cost_control"] is True
    assert fixed_inner_intent["profile_role"] == "fixed_snapshot_research"
    assert standard_evidence["evidence_class"] == "cost_capped_probe"
    assert standard_evidence["quality_claim_supported"] is False
    assert fixed_inner_short_evidence["evidence_class"] == "short_fixed_snapshot_probe"
    assert fixed_inner_short_evidence["quality_claim_supported"] is False
    assert fixed_inner_small_snapshot_evidence["evidence_class"] == "small_snapshot_probe"
    assert fixed_inner_small_snapshot_evidence["quality_claim_supported"] is False
    assert fixed_inner_quality_evidence["evidence_class"] == "fixed_snapshot_research"
    assert fixed_inner_quality_evidence["quality_claim_supported"] is True
    assert fixed_inner_intent["effective_max_agent_rounds"] == 16
    assert fixed_inner_intent["outer_replay_cadence_bound"] is False


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


def test_l2_target_selection_requires_train_audit_safety_gate() -> None:
    round_result = {
        "round": 1,
        "inner_validation": {
            "passes_gate": True,
            "accepted": 2,
            "correct_accepts": 2,
            "validation_size": 10,
        },
        "train_audit": {"wrong_accepts": 1},
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
    evidence = l2_target_evolution._private_holdout_evidence(  # noqa: SLF001
        [round_result],
    )
    assert evidence["selection_gate_diagnosis"] == "train_audit_safety_gate_failed"
    assert evidence["inner_passing_train_audit_wrong_accept_rounds"] == 1


def test_l2_target_selection_requires_visible_support_gate() -> None:
    round_result = {
        "round": 1,
        "inner_validation": {
            "passes_gate": True,
            "accepted": 7,
            "correct_accepts": 7,
            "validation_size": 774,
            "coverage": 7 / 774,
            "accepted_accuracy": 1.0,
            "wrong_accepts": 0,
            "wrong_accept_rate": 0.0,
            "visible_validation_splits": [
                "inner_validation",
                "inner_validation_shadow_1",
                "inner_validation_shadow_2",
                "inner_validation_shadow_3",
                "inner_validation_shadow_4",
            ],
        },
        "train_audit": {"wrong_accepts": 0},
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
    evidence = l2_target_evolution._private_holdout_evidence(  # noqa: SLF001
        [round_result],
    )
    assert evidence["selection_gate_diagnosis"] == "visible_support_gate_failed"
    assert evidence["inner_passing_visible_support_failed_rounds"] == 1
    assert evidence["visible_support_passing_rounds"] == 0


def test_l2_target_private_evidence_keeps_train_audit_count_when_support_fails() -> None:
    round_result = {
        "round": 1,
        "inner_validation": {
            "passes_gate": True,
            "accepted": 1,
            "correct_accepts": 1,
            "validation_size": 100,
        },
        "train_audit": {"wrong_accepts": 1},
        "selection_holdout": {
            "passes_gate": False,
            "accepted": 0,
            "correct_accepts": 0,
            "wrong_accepts": 0,
            "coverage": 0.0,
            "accepted_accuracy": None,
            "wrong_accept_rate": 0.0,
        },
        "promotion_holdout": {
            "passes_gate": False,
            "accepted": 0,
            "correct_accepts": 0,
            "wrong_accepts": 0,
            "coverage": 0.0,
            "accepted_accuracy": None,
            "wrong_accept_rate": 0.0,
        },
    }

    evidence = l2_target_evolution._private_holdout_evidence(  # noqa: SLF001
        [round_result],
    )

    assert evidence["selection_gate_diagnosis"] == "visible_support_gate_failed"
    assert evidence["inner_passing_visible_support_failed_rounds"] == 1
    assert evidence["inner_passing_train_audit_wrong_accept_rounds"] == 1


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
            "--visible-validation-ratio",
            "0.4",
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
    assert summary["budget_policy"]["visible_validation_ratio"] == 0.4
    assert summary["budget_policy"]["visible_cross_audit_folds"] == 3
    assert summary["data_split_policy"]["visible_validation_folds"] == 5
    assert summary["data_split_policy"]["visible_validation_ratio_requested"] == 0.4
    assert summary["budget_policy"]["stop_on_selection_gate"] is False
    assert summary["budget_policy"]["max_agent_rounds"] is None
    assert summary["evidence_policy"]["evidence_class"] == "short_fixed_snapshot_probe"
    assert summary["evidence_policy"]["quality_claim_supported"] is False
    assert summary["baseline"]["visible_cross_audit"]["gate_role"] == (
        "diagnostic_only_not_selection_or_adoption_gate"
    )
    assert summary["data_split"]["train"] > 0


def test_l2_target_evolve_cli_accepts_lower_miss_target_scope(tmp_path: Path) -> None:
    traces = [
        _trace_with_lower_result(
            index,
            intent="alarm_set" if index % 2 == 0 else "weather_query",
            slots=(
                {"time": f"{index} am"}
                if index % 2 == 0
                else {"location": f"city {index}"}
            ),
            lower_layer="L0" if index < 6 else None,
        )
        for index in range(20)
    ]
    traces_path = tmp_path / "traces.jsonl"
    traces_path.write_text(
        "".join(trace.model_dump_json() + "\n" for trace in traces),
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
            "--target-scope",
            "lower_miss",
            "--rounds",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    summary = json.loads((tmp_path / "target-run" / "summary.json").read_text())
    assert summary["target_scope"]["scope"] == "lower_miss"
    assert summary["target_scope"]["input_teacher_labeled_traces"] == 20
    assert summary["target_scope"]["scoped_teacher_labeled_traces"] == 14
    assert summary["target_scope"]["lower_layer_accepted_excluded"] == 6


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


def test_l2_target_evolve_cli_accepts_agent_session_no_launch(tmp_path: Path) -> None:
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
            "agent-session",
            "--rounds",
            "2",
            "--max-agent-rounds",
            "0",
        ],
    )

    assert result.exit_code == 0, result.output
    summary = json.loads((tmp_path / "target-run" / "summary.json").read_text())
    assert summary["mode"] == "agent-session"
    assert summary["stop_reason"] == "agent_session_budget_exhausted"
    assert summary["rounds_completed"] == 0
    assert summary["agent_budget"]["applies_to_mode"] is True
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
