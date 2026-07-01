from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.main import get_command
from typer.testing import CliRunner

from darjeeling.cli import (
    _BudgetedReferenceBroker,
    _compile_reference_usage,
    _resolve_agent_command,
    _run_compile_command,
    app,
)
from darjeeling.errors import TargetDefinitionError
from darjeeling.model import ReferenceContext, ReferenceResponse, ReferenceUsageLedger


def test_thin_target_demo_reports_local_accept_and_fallback() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["demo", "thin-target"])

    assert result.exit_code == 0, result.output
    assert "This uses toy data and a simulated reference LLM." in result.output
    assert "does not call external APIs" in result.output
    assert "demo-cold -> simulated reference LLM fallback" in result.output
    assert "demo-local-a (a:known-local) -> local artifact accepted; output=a" in result.output
    assert "demo-fallback (z:unfamiliar) -> simulated reference LLM fallback; output=z" in (
        result.output
    )
    assert "precision: 100.0%" in result.output
    assert "local coverage: 66.7%" in result.output
    assert "fallback share: 33.3%" in result.output
    assert "estimated saving:" in result.output


def test_compile_cli_help_is_discoverable() -> None:
    runner = CliRunner()

    compile_help = runner.invoke(app, ["compile", "--help"])
    run_help = runner.invoke(app, ["compile", "run", "--help"])

    assert compile_help.exit_code == 0, compile_help.output
    assert run_help.exit_code == 0, run_help.output
    assert "Compile target-local artifacts." in compile_help.output
    run_command = get_command(app).commands["compile"].commands["run"]
    option_names = {
        option
        for parameter in run_command.params
        for option in getattr(parameter, "opts", [])
    }
    assert {"--reference-config", "--agent-command", "--l4-deadline-ms"} <= option_names


def test_compile_run_checks_sandbox_before_reference_setup(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("darjeeling.cli.shutil.which", lambda name: None)

    with pytest.raises(ValueError, match="No reference calls were made"):
        _run_compile_command(
            target_path=tmp_path / "missing-target",
            run_root=tmp_path / "run",
            reference_config=tmp_path / "missing-reference.json",
            agent_command='["python3", "-c", "pass"]',
            workspace_root=None,
            max_candidates=1,
            max_agent_seconds=1,
            max_cost=1.0,
            enabled_layers="L1",
            l4_deadline_ms=1000,
            agent_network=False,
            agent_dependency_install=False,
            allow_insufficient_reference=False,
        )


def test_compile_run_checks_agent_command_before_reference_setup(
    target_dir, tmp_path, monkeypatch
) -> None:
    def fake_which(name: str) -> str | None:
        return "/usr/bin/sandbox-exec" if name == "sandbox-exec" else None

    monkeypatch.setattr("darjeeling.cli.shutil.which", fake_which)

    with pytest.raises(ValueError, match="agent command executable was not found"):
        _run_compile_command(
            target_path=target_dir,
            run_root=tmp_path / "run",
            reference_config=tmp_path / "missing-reference.json",
            agent_command='["missing-darjeeling-agent", "-c", "pass"]',
            workspace_root=None,
            max_candidates=1,
            max_agent_seconds=1,
            max_cost=1.0,
            enabled_layers="L1",
            l4_deadline_ms=1000,
            agent_network=False,
            agent_dependency_install=False,
            allow_insufficient_reference=False,
        )


def test_compile_run_resolves_relative_interpreter_script_before_launch(
    tmp_path, monkeypatch
) -> None:
    script = tmp_path / "agent.py"
    script.write_text("raise SystemExit(0)\n", encoding="utf-8")

    def fake_which(name: str) -> str | None:
        return "/usr/bin/python3" if name == "python3" else None

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("darjeeling.cli.shutil.which", fake_which)

    command = _resolve_agent_command(["python3", "./agent.py", "--flag"])

    assert command == ["/usr/bin/python3", str(script.resolve()), "--flag"]


@pytest.mark.parametrize("protected_source", ["target", "repo"])
def test_compile_run_rejects_protected_agent_command_before_reference_setup(
    target_dir, tmp_path, monkeypatch, protected_source
) -> None:
    target_path = tmp_path / "source-target"
    shutil.copytree(target_dir, target_path)
    target_command = target_path / "agent-helper"
    target_command.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    repo_command = Path(__file__).resolve().parents[1] / "README.md"
    command_path = target_command if protected_source == "target" else repo_command

    def fake_which(name: str) -> str | None:
        if name == "sandbox-exec":
            return "/usr/bin/sandbox-exec"
        if name == str(command_path):
            return str(command_path)
        return f"/usr/bin/{name}"

    def fail_reference_setup(path):
        raise AssertionError("reference config should not be loaded")

    monkeypatch.setattr("darjeeling.cli.shutil.which", fake_which)
    monkeypatch.setattr(
        "darjeeling.cli.build_reference_broker_from_config", fail_reference_setup
    )

    with pytest.raises(ValueError, match="protected target/Core paths"):
        _run_compile_command(
            target_path=target_path,
            run_root=tmp_path / "run",
            reference_config=tmp_path / "missing-reference.json",
            agent_command=json.dumps([str(command_path)]),
            workspace_root=None,
            max_candidates=1,
            max_agent_seconds=1,
            max_cost=1.0,
            enabled_layers="L1",
            l4_deadline_ms=1000,
            agent_network=False,
            agent_dependency_install=False,
            allow_insufficient_reference=False,
        )


@pytest.mark.parametrize("protected_source", ["target", "repo"])
def test_compile_run_rejects_protected_interpreter_script_before_reference_setup(
    target_dir, tmp_path, monkeypatch, protected_source
) -> None:
    target_path = tmp_path / "source-target"
    shutil.copytree(target_dir, target_path)
    target_script = target_path / "agent.py"
    target_script.write_text("print('should not run')\n", encoding="utf-8")
    repo_script = Path(__file__).resolve().parents[1] / "README.md"
    script_path = target_script if protected_source == "target" else repo_script

    def fake_which(name: str) -> str | None:
        if name == "sandbox-exec":
            return "/usr/bin/sandbox-exec"
        if name == "python3":
            return "/usr/bin/python3"
        return f"/usr/bin/{name}"

    def fail_reference_setup(path):
        raise AssertionError("reference config should not be loaded")

    monkeypatch.setattr("darjeeling.cli.shutil.which", fake_which)
    monkeypatch.setattr(
        "darjeeling.cli.build_reference_broker_from_config", fail_reference_setup
    )

    with pytest.raises(ValueError, match="protected target/Core paths"):
        _run_compile_command(
            target_path=target_path,
            run_root=tmp_path / "run",
            reference_config=tmp_path / "missing-reference.json",
            agent_command=json.dumps(["python3", str(script_path)]),
            workspace_root=None,
            max_candidates=1,
            max_agent_seconds=1,
            max_cost=1.0,
            enabled_layers="L1",
            l4_deadline_ms=1000,
            agent_network=False,
            agent_dependency_install=False,
            allow_insufficient_reference=False,
        )


def test_compile_run_requires_target_reference_before_provider_setup(
    target_dir, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("darjeeling.cli.shutil.which", lambda name: "/usr/bin/sandbox-exec")
    target_path = tmp_path / "target-without-reference"
    shutil.copytree(target_dir, target_path)
    target_yaml = target_path / "target.yaml"
    target_yaml.write_text(
        "\n".join(
            line
            for line in target_yaml.read_text(encoding="utf-8").splitlines()
            if not line.startswith("reference:")
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(TargetDefinitionError, match="reference adapter is required"):
        _run_compile_command(
            target_path=target_path,
            run_root=tmp_path / "run",
            reference_config=tmp_path / "missing-reference.json",
            agent_command='["python3", "-c", "pass"]',
            workspace_root=None,
            max_candidates=1,
            max_agent_seconds=1,
            max_cost=1.0,
            enabled_layers="L1",
            l4_deadline_ms=1000,
            agent_network=False,
            agent_dependency_install=False,
            allow_insufficient_reference=False,
        )


def test_compile_run_protects_snapshots_reference_artifacts_and_resolves_agent_command(
    target_dir, tmp_path, monkeypatch
) -> None:
    class StopAfterLaunch(Exception):
        pass

    captured_runtime: dict[str, Any] = {}
    definition = SimpleNamespace(
        name="thin",
        contract_hash="contract-hash",
        data_config=SimpleNamespace(),
        requirements={},
    )
    snapshot_result = SimpleNamespace(
        snapshot=SimpleNamespace(snapshot_id="snapshot-id"),
        reference_qualification=SimpleNamespace(cost={}, latency={}),
        reference_usage=ReferenceUsageLedger(call_count=0, cost=0.0, errors={}),
    )
    attempt = SimpleNamespace(
        attempt_id="attempt-id",
        workspace_path=tmp_path / "external-workspaces" / "attempt-id",
    )
    agent_dir = tmp_path.parent / f"{tmp_path.name}-agent-bin"
    agent_dir.mkdir()
    relative_agent = agent_dir / "agent.sh"
    relative_agent.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    relative_agent.chmod(0o755)
    reference_cache = tmp_path / "reference-artifacts" / "cache.jsonl"
    reference_ledger = tmp_path / "reference-artifacts" / "usage-ledger.json"
    reference_cache.parent.mkdir()
    reference_cache.write_text("", encoding="utf-8")
    reference_ledger.write_text("{}", encoding="utf-8")

    original_which = shutil.which

    def fake_which(name: str) -> str | None:
        if name == "sandbox-exec":
            return "/usr/bin/sandbox-exec"
        return original_which(name)

    monkeypatch.chdir(agent_dir)
    monkeypatch.setattr("darjeeling.cli.shutil.which", fake_which)
    monkeypatch.setattr(
        "darjeeling.cli.load_checked_target",
        lambda target_path, require_reference=False: (
            definition,
            SimpleNamespace(),
            SimpleNamespace(),
        ),
    )
    monkeypatch.setattr(
        "darjeeling.cli.build_reference_broker_from_config",
        lambda path: SimpleNamespace(
            reference_version="reference",
            config=SimpleNamespace(
                cache_path=reference_cache,
                usage_ledger_path=reference_ledger,
            ),
        ),
    )
    monkeypatch.setattr(
        "darjeeling.cli.create_release_without_artifacts",
        lambda *args, **kwargs: SimpleNamespace(release_id="cold"),
    )
    monkeypatch.setattr(
        "darjeeling.cli.build_snapshot", lambda *args, **kwargs: snapshot_result
    )
    monkeypatch.setattr(
        "darjeeling.cli.load_target_workspace", lambda *args, **kwargs: SimpleNamespace()
    )
    monkeypatch.setattr(
        "darjeeling.cli.create_compile_run",
        lambda *args, **kwargs: SimpleNamespace(compile_id="compile-id"),
    )
    monkeypatch.setattr("darjeeling.cli.create_agent_workspace", lambda *args: attempt)
    monkeypatch.setattr(
        "darjeeling.cli.export_agent_readonly_target_view",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "darjeeling.cli.export_train_view_for_agent",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "darjeeling.cli.mount_readonly_inputs", lambda *args, **kwargs: SimpleNamespace()
    )
    monkeypatch.setattr(
        "darjeeling.cli.write_agent_brief", lambda *args, **kwargs: tmp_path / "brief.md"
    )

    def fake_launch(attempt, brief, runtime):
        captured_runtime.update(runtime)
        raise StopAfterLaunch

    monkeypatch.setattr("darjeeling.cli.launch_target_adaptation_agent_async", fake_launch)

    with pytest.raises(StopAfterLaunch):
        _run_compile_command(
            target_path=target_dir,
            run_root=tmp_path / "run",
            reference_config=tmp_path / "reference.json",
            agent_command='["./agent.sh"]',
            workspace_root=tmp_path / "external-workspaces",
            max_candidates=1,
            max_agent_seconds=1,
            max_cost=1.0,
            enabled_layers="L1",
            l4_deadline_ms=1000,
            agent_network=False,
            agent_dependency_install=False,
            allow_insufficient_reference=False,
        )

    assert captured_runtime["command"] == [str(relative_agent.resolve())]
    assert captured_runtime["protected_paths"] == [
        str(target_dir.resolve()),
        str((tmp_path / "run" / "snapshots").resolve()),
        str(reference_cache.resolve()),
        str(reference_ledger.resolve()),
    ]


def test_compile_run_rejects_budget_exhausted_before_final_test(
    target_dir, tmp_path, monkeypatch
) -> None:
    definition = SimpleNamespace(
        name="thin",
        contract_hash="contract-hash",
        data_config=SimpleNamespace(),
        requirements={},
    )
    snapshot_result = SimpleNamespace(
        snapshot=SimpleNamespace(snapshot_id="snapshot-id"),
        reference_qualification=SimpleNamespace(cost={}, latency={}),
        reference_usage=ReferenceUsageLedger(call_count=0, cost=0.0, errors={}),
    )
    attempt = SimpleNamespace(
        attempt_id="attempt-id",
        workspace_path=tmp_path / "workspaces" / "attempt-id",
    )

    monkeypatch.setattr("darjeeling.cli.shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        "darjeeling.cli.load_checked_target",
        lambda target_path, require_reference=False: (
            definition,
            SimpleNamespace(),
            SimpleNamespace(),
        ),
    )
    monkeypatch.setattr(
        "darjeeling.cli.build_reference_broker_from_config",
        lambda path: SimpleNamespace(reference_version="reference"),
    )
    monkeypatch.setattr(
        "darjeeling.cli.create_release_without_artifacts",
        lambda *args, **kwargs: SimpleNamespace(release_id="cold"),
    )
    monkeypatch.setattr(
        "darjeeling.cli.build_snapshot", lambda *args, **kwargs: snapshot_result
    )
    monkeypatch.setattr(
        "darjeeling.cli.load_target_workspace", lambda *args, **kwargs: SimpleNamespace()
    )
    monkeypatch.setattr(
        "darjeeling.cli.create_compile_run",
        lambda *args, **kwargs: SimpleNamespace(compile_id="compile-id"),
    )
    monkeypatch.setattr("darjeeling.cli.create_agent_workspace", lambda *args: attempt)
    monkeypatch.setattr(
        "darjeeling.cli.export_agent_readonly_target_view",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "darjeeling.cli.export_train_view_for_agent",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "darjeeling.cli.mount_readonly_inputs", lambda *args, **kwargs: SimpleNamespace()
    )
    monkeypatch.setattr(
        "darjeeling.cli.write_agent_brief", lambda *args, **kwargs: tmp_path / "brief.md"
    )
    monkeypatch.setattr(
        "darjeeling.cli.launch_target_adaptation_agent_async",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "darjeeling.cli.run_interactive_compile_loop",
        lambda *args, **kwargs: {
            "stop_reason": "budget_exhausted",
            "total_candidate_cost": 3.0,
            "selected_candidate": SimpleNamespace(),
            "validation_report": SimpleNamespace(),
            "closed_attempt": SimpleNamespace(),
        },
    )

    def fail_final_test(*args, **kwargs):
        raise AssertionError("final test should not run after budget exhaustion")

    monkeypatch.setattr("darjeeling.cli.evaluate_candidate_on_test", fail_final_test)

    with pytest.raises(RuntimeError, match="exhausted --max-cost before final test"):
        _run_compile_command(
            target_path=target_dir,
            run_root=tmp_path / "run",
            reference_config=tmp_path / "reference.json",
            agent_command='["python3", "-c", "pass"]',
            workspace_root=None,
            max_candidates=2,
            max_agent_seconds=1,
            max_cost=1.0,
            enabled_layers="L1",
            l4_deadline_ms=1000,
            agent_network=False,
            agent_dependency_install=False,
            allow_insufficient_reference=False,
        )


def test_compile_run_rejects_exhausted_reference_budget_before_agent_launch(
    target_dir, tmp_path, monkeypatch
) -> None:
    definition = SimpleNamespace(
        name="thin",
        contract_hash="contract-hash",
        data_config=SimpleNamespace(),
        requirements={},
    )
    snapshot_result = SimpleNamespace(
        snapshot=SimpleNamespace(snapshot_id="snapshot-id"),
        reference_qualification=SimpleNamespace(cost={}, latency={}),
        reference_usage=ReferenceUsageLedger(call_count=1, cost=1.25, errors={}),
    )

    monkeypatch.setattr("darjeeling.cli.shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        "darjeeling.cli.load_checked_target",
        lambda target_path, require_reference=False: (
            definition,
            SimpleNamespace(),
            SimpleNamespace(),
        ),
    )
    monkeypatch.setattr(
        "darjeeling.cli.build_reference_broker_from_config",
        lambda path: SimpleNamespace(reference_version="reference"),
    )
    monkeypatch.setattr(
        "darjeeling.cli.create_release_without_artifacts",
        lambda *args, **kwargs: SimpleNamespace(release_id="cold"),
    )
    monkeypatch.setattr(
        "darjeeling.cli.build_snapshot", lambda *args, **kwargs: snapshot_result
    )

    def fail_agent_setup(*args, **kwargs):
        raise AssertionError("agent workspace should not be prepared after budget exhaustion")

    monkeypatch.setattr("darjeeling.cli.load_target_workspace", fail_agent_setup)
    monkeypatch.setattr("darjeeling.cli.create_agent_workspace", fail_agent_setup)
    monkeypatch.setattr("darjeeling.cli.launch_target_adaptation_agent_async", fail_agent_setup)

    with pytest.raises(RuntimeError, match="before agent launch"):
        _run_compile_command(
            target_path=target_dir,
            run_root=tmp_path / "run",
            reference_config=tmp_path / "reference.json",
            agent_command='["python3", "-c", "pass"]',
            workspace_root=None,
            max_candidates=2,
            max_agent_seconds=1,
            max_cost=1.0,
            enabled_layers="L1",
            l4_deadline_ms=1000,
            agent_network=False,
            agent_dependency_install=False,
            allow_insufficient_reference=False,
        )


def test_compile_run_rejects_cold_start_reference_budget_before_snapshot(
    target_dir, tmp_path, monkeypatch
) -> None:
    definition = SimpleNamespace(
        name="thin",
        contract_hash="contract-hash",
        data_config=SimpleNamespace(),
        requirements={},
    )

    monkeypatch.setattr("darjeeling.cli.shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        "darjeeling.cli.load_checked_target",
        lambda target_path, require_reference=False: (
            definition,
            SimpleNamespace(),
            SimpleNamespace(),
        ),
    )
    monkeypatch.setattr(
        "darjeeling.cli.build_reference_broker_from_config",
        lambda path: SimpleNamespace(reference_version="reference"),
    )

    def spend_cold_start_budget(*args, **kwargs):
        broker = args[3]
        broker.cost = 1.0
        return SimpleNamespace(release_id="cold")

    def fail_snapshot(*args, **kwargs):
        raise AssertionError("snapshot should not be built after cold-start budget exhaustion")

    monkeypatch.setattr(
        "darjeeling.cli.create_release_without_artifacts", spend_cold_start_budget
    )
    monkeypatch.setattr("darjeeling.cli.build_snapshot", fail_snapshot)

    with pytest.raises(RuntimeError, match="before agent launch"):
        _run_compile_command(
            target_path=target_dir,
            run_root=tmp_path / "run",
            reference_config=tmp_path / "reference.json",
            agent_command='["python3", "-c", "pass"]',
            workspace_root=None,
            max_candidates=2,
            max_agent_seconds=1,
            max_cost=1.0,
            enabled_layers="L1",
            l4_deadline_ms=1000,
            agent_network=False,
            agent_dependency_install=False,
            allow_insufficient_reference=False,
        )


@pytest.mark.parametrize(
    ("run_root_name", "workspace_root_name"),
    [
        ("source-target/runs/compile", None),
        ("run", "source-target/workspaces"),
    ],
)
def test_compile_run_rejects_agent_workspace_inside_target_before_reference_setup(
    target_dir, tmp_path, monkeypatch, run_root_name, workspace_root_name
) -> None:
    monkeypatch.setattr("darjeeling.cli.shutil.which", lambda name: f"/usr/bin/{name}")
    target_path = tmp_path / "source-target"
    shutil.copytree(target_dir, target_path)
    run_root = tmp_path / run_root_name
    workspace_root = tmp_path / workspace_root_name if workspace_root_name else None

    with pytest.raises(ValueError, match="workspace root must not be inside"):
        _run_compile_command(
            target_path=target_path,
            run_root=run_root,
            reference_config=tmp_path / "missing-reference.json",
            agent_command='["python3", "-c", "pass"]',
            workspace_root=workspace_root,
            max_candidates=1,
            max_agent_seconds=1,
            max_cost=1.0,
            enabled_layers="L1",
            l4_deadline_ms=1000,
            agent_network=False,
            agent_dependency_install=False,
            allow_insufficient_reference=False,
        )

    assert not run_root.exists()


def test_compile_run_rejects_path_component_target_name_before_reference_setup(
    target_dir, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("darjeeling.cli.shutil.which", lambda name: f"/usr/bin/{name}")
    target_path = tmp_path / "source-target"
    shutil.copytree(target_dir, target_path)
    yaml = __import__("yaml")
    target_yaml = target_path / "target.yaml"
    target_data = yaml.safe_load(target_yaml.read_text(encoding="utf-8"))
    target_data["name"] = "../../source-target/generated/workspaces"
    target_yaml.write_text(yaml.safe_dump(target_data), encoding="utf-8")

    def fail_reference_setup(path):
        raise AssertionError("reference config should not be loaded")

    monkeypatch.setattr(
        "darjeeling.cli.build_reference_broker_from_config", fail_reference_setup
    )

    with pytest.raises(ValueError, match="single safe path segment"):
        _run_compile_command(
            target_path=target_path,
            run_root=tmp_path / "run",
            reference_config=tmp_path / "missing-reference.json",
            agent_command='["python3", "-c", "pass"]',
            workspace_root=None,
            max_candidates=1,
            max_agent_seconds=1,
            max_cost=1.0,
            enabled_layers="L1",
            l4_deadline_ms=1000,
            agent_network=False,
            agent_dependency_install=False,
            allow_insufficient_reference=False,
        )

    assert not (target_path / "generated").exists()


def test_reference_budget_blocks_zero_cost_before_provider_call() -> None:
    class CountingBroker:
        reference_version = "counting"

        def __init__(self) -> None:
            self.call_count = 0

        def call(
            self, request: dict[str, Any], context: ReferenceContext
        ) -> ReferenceResponse:
            self.call_count += 1
            return ReferenceResponse(payload={"label": "a"}, cost=0.01)

    broker = CountingBroker()
    budgeted = _BudgetedReferenceBroker(broker, max_cost=0.0)

    with pytest.raises(RuntimeError, match="before provider call"):
        budgeted.call({}, ReferenceContext(purpose="runtime_l4_fallback"))

    assert broker.call_count == 0


def test_reference_budget_keeps_response_that_crosses_budget() -> None:
    class PaidBroker:
        reference_version = "paid"

        def __init__(self) -> None:
            self.call_count = 0

        def call(
            self, request: dict[str, Any], context: ReferenceContext
        ) -> ReferenceResponse:
            self.call_count += 1
            return ReferenceResponse(payload={"label": "a"}, cost=0.25)

    broker = PaidBroker()
    budgeted = _BudgetedReferenceBroker(broker, max_cost=0.1)

    response = budgeted.call({}, ReferenceContext(purpose="snapshot_label"))

    assert response.payload == {"label": "a"}
    assert response.cost == 0.25
    assert broker.call_count == 1
    assert budgeted.call_count == 1
    assert budgeted.cost == 0.25


def test_compile_reference_usage_counts_prior_provider_spend() -> None:
    broker = SimpleNamespace(call_count=7, cost=9.0)
    snapshot_result = SimpleNamespace(
        reference_usage=ReferenceUsageLedger(
            call_count=2,
            cost=1.25,
            errors={"validation_failure": 1},
        )
    )

    usage = _compile_reference_usage(broker, snapshot_result)

    assert usage.call_count == 7
    assert usage.cost == 9.0
    assert usage.errors == {"validation_failure": 1}
