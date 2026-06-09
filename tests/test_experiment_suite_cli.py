import json
from pathlib import Path

from typer.testing import CliRunner

from darjeeling import cli


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
    assert captured["parallel"] == 2
    assert [command["experiment"] for command in captured["commands"]] == [
        "main-evolution",
        "l2-family",
    ]
    first_command = captured["commands"][0]["command"]
    assert first_command[:3] == [first_command[0], "-m", "darjeeling.cli"]
    assert "main-evolution" in first_command
    assert str(tmp_path / "main-evolution") in first_command


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
