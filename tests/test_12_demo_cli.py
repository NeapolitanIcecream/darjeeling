from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from typer.main import get_command
from typer.testing import CliRunner

from darjeeling.cli import (
    _BudgetedReferenceBroker,
    _compile_reference_usage,
    _run_compile_command,
    app,
)
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
