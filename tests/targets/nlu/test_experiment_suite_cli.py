import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from darjeeling.targets.nlu import main_cli as cli
from darjeeling.targets.nlu.compiler.l2_tuner import L2TuneResult


def test_experiment_suite_builds_parallel_subprocess_plan(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured = {}

    def fake_run_suite(commands, *, parallel):
        captured["commands"] = commands
        captured["parallel"] = parallel
        return [
            {
                "experiment": command["experiment"],
                "command": command["command"],
                "run_dir": str(command["run_dir"]),
                "log_path": str(command["log_path"]),
                "return_code": 0,
            }
            for command in commands
        ]

    monkeypatch.setattr(cli, "_run_experiment_suite_commands", fake_run_suite)
    monkeypatch.setattr(cli, "_current_git_commit", lambda: "abc123def456")
    runner = CliRunner()

    result = runner.invoke(
        cli.app,
        [
            "experiment",
            "suite",
            "--run-root",
            str(tmp_path),
            "--experiment",
            "main-evolution",
            "--experiment",
            "l2-family",
            "--max-requests",
            "12",
            "--compile-every",
            "6",
            "--parallel",
            "2",
            "--no-compare",
        ],
    )

    assert result.exit_code == 0
    suite = json.loads((tmp_path / "suite.json").read_text(encoding="utf-8"))
    assert suite["experiments"] == ["main-evolution", "l2-family"]
    assert suite["max_requests"] == 12
    assert suite["compile_every"] == 6
    assert suite["commit_hash"] == "abc123def456"
    assert captured["parallel"] == 2
    assert [command["experiment"] for command in captured["commands"]] == [
        "main-evolution",
        "l2-family",
    ]
    first_command = captured["commands"][0]["command"]
    assert first_command[:3] == [first_command[0], "-m", "darjeeling.targets.nlu.main_cli"]
    assert "main-evolution" in first_command
    assert str(tmp_path / "main-evolution") in first_command


def test_experiment_suite_default_uses_post_refactor_matrix(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured = {}

    def fake_run_suite(commands, *, parallel):
        captured["commands"] = commands
        captured["parallel"] = parallel
        return [
            {
                "experiment": command["experiment"],
                "command": command["command"],
                "run_dir": str(command["run_dir"]),
                "log_path": str(command["log_path"]),
                "return_code": 0,
            }
            for command in commands
        ]

    monkeypatch.setattr(cli, "_run_experiment_suite_commands", fake_run_suite)
    runner = CliRunner()

    result = runner.invoke(
        cli.app,
        [
            "experiment",
            "suite",
            "--run-root",
            str(tmp_path),
            "--max-requests",
            "12",
            "--compile-every",
            "6",
            "--parallel",
            "1",
            "--no-compare",
        ],
    )

    assert result.exit_code == 0
    suite = json.loads((tmp_path / "suite.json").read_text(encoding="utf-8"))
    assert suite["experiments"] == list(cli.DEFAULT_EXPERIMENT_SUITE)
    assert "l3-guarded" not in suite["experiments"]
    assert [command["experiment"] for command in captured["commands"]] == list(
        cli.DEFAULT_EXPERIMENT_SUITE
    )


def test_experiment_suite_can_append_guarded_l3_after_preflight(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured = {}

    def fake_run_suite(commands, *, parallel):
        captured["commands"] = commands
        return [
            {
                "experiment": command["experiment"],
                "command": command["command"],
                "run_dir": str(command["run_dir"]),
                "log_path": str(command["log_path"]),
                "return_code": 0,
            }
            for command in commands
        ]

    monkeypatch.setattr(cli, "_run_experiment_suite_commands", fake_run_suite)
    runner = CliRunner()

    result = runner.invoke(
        cli.app,
        [
            "experiment",
            "suite",
            "--run-root",
            str(tmp_path),
            "--include-guarded-l3",
            "--max-requests",
            "12",
            "--compile-every",
            "6",
            "--no-compare",
        ],
    )

    assert result.exit_code == 0
    suite = json.loads((tmp_path / "suite.json").read_text(encoding="utf-8"))
    assert suite["experiments"] == [*cli.DEFAULT_EXPERIMENT_SUITE, "l3-guarded"]
    assert suite["include_guarded_l3"] is True
    assert captured["commands"][-1]["experiment"] == "l3-guarded"


@pytest.mark.parametrize(
    "command_name",
    [
        "no-audit",
        "l2-global-student",
        "l2-expert-bank",
        "l3-disabled",
        "l3-shadow",
        "l3-guarded",
    ],
)
def test_post_refactor_experiment_commands_dispatch_specs(
    command_name: str,
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured = {}

    def fake_run_single_experiment(experiment_name, **kwargs):
        captured["experiment_name"] = experiment_name
        captured["kwargs"] = kwargs

    monkeypatch.setattr(cli, "_run_single_experiment", fake_run_single_experiment)
    runner = CliRunner()

    result = runner.invoke(
        cli.app,
        [
            "experiment",
            command_name,
            "--run-dir",
            str(tmp_path),
            "--max-requests",
            "12",
            "--compile-every",
            "6",
            "--teacher",
            "cache",
        ],
    )

    assert result.exit_code == 0
    assert captured["experiment_name"] == command_name
    assert captured["kwargs"]["run_dir"] == tmp_path
    assert captured["kwargs"]["max_requests"] == 12
    assert captured["kwargs"]["compile_every"] == 6
    assert captured["kwargs"]["teacher"] == "cache"


def test_l2_mlp_experiment_command_dispatches_spec(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured = {}

    def fake_run_single_experiment(experiment_name, **kwargs):
        captured["experiment_name"] = experiment_name
        captured["kwargs"] = kwargs

    monkeypatch.setattr(cli, "_run_single_experiment", fake_run_single_experiment)
    runner = CliRunner()

    result = runner.invoke(
        cli.app,
        [
            "experiment",
            "l2-mlp",
            "--run-dir",
            str(tmp_path),
            "--max-requests",
            "12",
            "--compile-every",
            "6",
            "--teacher",
            "cache",
        ],
    )

    assert result.exit_code == 0
    assert captured["experiment_name"] == "l2-mlp"
    assert captured["kwargs"]["run_dir"] == tmp_path
    assert captured["kwargs"]["max_requests"] == 12
    assert captured["kwargs"]["compile_every"] == 6
    assert captured["kwargs"]["teacher"] == "cache"


def test_l2_tuned_experiment_command_dispatches_spec(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured = {}

    def fake_run_single_experiment(experiment_name, **kwargs):
        captured["experiment_name"] = experiment_name
        captured["kwargs"] = kwargs

    monkeypatch.setattr(cli, "_run_single_experiment", fake_run_single_experiment)
    runner = CliRunner()

    result = runner.invoke(
        cli.app,
        [
            "experiment",
            "l2-tuned",
            "--run-dir",
            str(tmp_path),
            "--max-requests",
            "12",
            "--compile-every",
            "6",
            "--teacher",
            "cache",
        ],
    )

    assert result.exit_code == 0
    assert captured["experiment_name"] == "l2-tuned"
    assert captured["kwargs"]["run_dir"] == tmp_path
    assert captured["kwargs"]["max_requests"] == 12
    assert captured["kwargs"]["compile_every"] == 6
    assert captured["kwargs"]["teacher"] == "cache"


def test_l2_tuned_lower_miss_experiment_command_dispatches_spec(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured = {}

    def fake_run_single_experiment(experiment_name, **kwargs):
        captured["experiment_name"] = experiment_name
        captured["kwargs"] = kwargs

    monkeypatch.setattr(cli, "_run_single_experiment", fake_run_single_experiment)
    runner = CliRunner()

    result = runner.invoke(
        cli.app,
        [
            "experiment",
            "l2-tuned-lower-miss",
            "--run-dir",
            str(tmp_path),
            "--max-requests",
            "12",
            "--compile-every",
            "6",
            "--teacher",
            "cache",
        ],
    )

    assert result.exit_code == 0
    assert captured["experiment_name"] == "l2-tuned-lower-miss"
    assert captured["kwargs"]["run_dir"] == tmp_path
    assert captured["kwargs"]["max_requests"] == 12
    assert captured["kwargs"]["compile_every"] == 6
    assert captured["kwargs"]["teacher"] == "cache"


def test_l2_tune_cli_writes_optuna_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    traces_path = tmp_path / "traces.jsonl"
    traces_path.write_text(
        json.dumps(
            {
                "request_id": "r1",
                "utterance": "beta request",
                "teacher_frame": {"intent": "intent_beta", "slots": {}},
                "chosen_layer": "L4",
                "final_frame": {"intent": "intent_beta", "slots": {}},
                "layer_results": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out_path = tmp_path / "l2_tuning.json"
    captured = {}

    def fake_tune_l2_student(traces, *, base_config, spec):
        captured["trace_count"] = len(traces)
        captured["base_config"] = base_config
        captured["spec"] = spec
        return L2TuneResult(
            train_size=1,
            validation_size=1,
            split_policy=spec.split_policy,
            n_trials_requested=spec.n_trials,
            n_trials_completed=1,
            best_trial_number=0,
            best_value=1.0,
            best_config=base_config.model_dump(mode="json"),
            best_metrics={"unguarded": {"accepted_accuracy": 1.0}},
            trials=[],
        )

    monkeypatch.setattr(cli, "tune_l2_student", fake_tune_l2_student)
    runner = CliRunner()

    result = runner.invoke(
        cli.app,
        [
            "l2",
            "tune",
            "--traces",
            str(traces_path),
            "--out",
            str(out_path),
            "--n-trials",
            "2",
            "--search-space",
            "wide",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "l2-tune-v1"
    assert payload["n_trials_requested"] == 2
    assert captured["trace_count"] == 1
    assert captured["spec"].search_space == "wide"
    assert captured["spec"].split_policy == "chronological"
