import json
from pathlib import Path

from typer.testing import CliRunner

from darjeeling.cli import app
from darjeeling.compiler.l2_target_evolution import (
    L2TargetEvolutionConfig,
    run_l2_target_evolution,
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
    assert "promotion_holdout" not in json.dumps(round_state)
    assert "selection_holdout" not in json.dumps(round_state)


def test_l2_target_evolution_applies_dry_run_patches_to_target_only(tmp_path: Path) -> None:
    patch_path = tmp_path / "target.patch"
    patch_path.write_text(
        "\n".join(
            [
                "diff --git a/target/target_l2.py b/target/target_l2.py",
                "--- a/target/target_l2.py",
                "+++ b/target/target_l2.py",
                "@@ -3,6 +3,8 @@",
                " from typing import Any",
                " ",
                " ",
                "+TARGET_MARKER = 'patched'",
                "+",
                " def config_overrides() -> dict[str, Any]:",
                "     \"\"\"Return target-specific L2StudentConfig overrides.\"\"\"",
                "     return {}",
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
    assert summary["budget_policy"]["inner_patience_rounds"] == 2
    assert summary["budget_policy"]["stop_on_selection_gate"] is True
    assert summary["data_split"]["train"] > 0
