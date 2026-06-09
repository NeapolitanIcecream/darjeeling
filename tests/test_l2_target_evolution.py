import json
from pathlib import Path

from typer.testing import CliRunner

import darjeeling.compiler.l2_target_evolution as l2_target_evolution
from darjeeling.cli import app
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
    assert summary["adoption_decision"]["adopted"] is False
    assert summary["best_adoptable_round"] is None
    assert summary["target_code_scope"] == "target/"
    assert summary["baseline"]["label"] == "baseline"
    assert (workspace / "target" / "target_l2.py").exists()
    assert (workspace / "system" / "darjeeling" / "src").exists()
    assert (workspace / "system" / "darjeeling" / "README.md").exists()
    assert not (workspace / "candidate").exists()
    assert not (workspace / "data" / "promotion_holdout.jsonl").exists()
    assert not (workspace / "data" / "selection_holdout.jsonl").exists()
    assert (tmp_path / "job" / "private" / "selection_holdout.jsonl").exists()
    assert (tmp_path / "job" / "private" / "promotion_holdout.jsonl").exists()
    assert (tmp_path / "job" / "rounds" / "round_003.json").exists()

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
    }
    assert (workspace / "data" / "objective.json").exists()
    round_state = json.loads((workspace / "data" / "round_state.json").read_text())
    round_state_text = json.dumps(round_state)
    assert "promotion_holdout" not in round_state_text
    assert "selection_holdout" not in round_state_text
    private_rows = [
        json.loads(line)
        for path in [
            tmp_path / "job" / "private" / "selection_holdout.jsonl",
            tmp_path / "job" / "private" / "promotion_holdout.jsonl",
        ]
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert all(row["request_id"] not in round_state_text for row in private_rows)
    objective = json.loads((workspace / "data" / "objective.json").read_text())
    program_text = (workspace / "program.md").read_text(encoding="utf-8")
    assert "candidate_selection_gate" in round_state
    assert "visible inner validation gate" in round_state["candidate_selection_gate"]
    assert "early_stop_policy" in round_state
    assert "does not stop the inner loop" in round_state["early_stop_policy"]
    assert "candidate_selection" in objective["gates"]
    assert any(
        "near_miss_examples" in strategy
        for strategy in objective["allowed_strategies"]
    )
    assert "Private selection" in program_text
    assert "alone is not success" in program_text
    assert "outer selection signal" in program_text
    assert "inner-loop early-stop signal" in program_text
    assert "near_miss_examples" in program_text


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
    assert summary["rounds_completed"] == 1
    assert summary["rounds"][0]["local_search"]["schema_version"] == (
        "l2-target-local-search-v1"
    )
    assert report["trials_requested"] == 2
    assert report["private_holdout_visibility"] == (
        "local search used only visible train and inner_validation data"
    )
    assert "selection_holdout" not in report_path.read_text(encoding="utf-8")
    assert "promotion_holdout" not in report_path.read_text(encoding="utf-8")
    assert (workspace / "tools" / "search_config.py").exists()
    assert not (workspace / "data" / "selection_holdout.jsonl").exists()
    assert not (workspace / "data" / "promotion_holdout.jsonl").exists()


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


def test_l2_target_selection_requires_visible_inner_gate() -> None:
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
        ],
    )

    assert result.exit_code == 0, result.output
    summary = json.loads((tmp_path / "target-run" / "summary.json").read_text())
    assert summary["rounds_completed"] == 2
    assert summary["budget_policy"]["inner_patience_rounds"] == 4
    assert summary["budget_policy"]["stop_on_selection_gate"] is False
    assert summary["data_split"]["train"] > 0


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


def test_l2_promote_target_cli_writes_runtime_artifacts(tmp_path: Path) -> None:
    target_run = tmp_path / "target-run"
    workspace = target_run / "workspace" / "l2_target"
    target_dir = workspace / "target"
    data_dir = workspace / "data"
    target_dir.mkdir(parents=True)
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
    return {"accept_threshold": 0.0, "min_examples": 4}

def postprocess_frame(utterance, frame, metadata):
    del utterance, metadata
    return frame
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
    assert manifest["candidate_metrics"]["l2_target_training_traces"] == 12
    assert manifest["candidate_metrics"]["l2_training_scope"] == "l2_target_workspace_train"
    assert manifest["candidate_metrics"]["l2_training_traces"] == 12


def test_l2_promote_target_cli_can_stage_non_adopted_candidate_for_outer_replay(
    tmp_path: Path,
) -> None:
    target_run = tmp_path / "target-run"
    workspace = target_run / "workspace" / "l2_target"
    target_dir = workspace / "target"
    data_dir = workspace / "data"
    target_dir.mkdir(parents=True)
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
    return {"accept_threshold": 0.0, "min_examples": 4}

def postprocess_frame(utterance, frame, metadata):
    del utterance, metadata
    return frame
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
                "best_round": {
                    "round": 1,
                    "inner_validation": {"accepted": 1, "wrong_accepts": 0},
                    "selection_holdout": {"accepted": 0, "wrong_accepts": 0},
                    "promotion_holdout": {"accepted": 1, "wrong_accepts": 0},
                },
                "rounds": [
                    {
                        "round": 1,
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
