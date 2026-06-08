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
