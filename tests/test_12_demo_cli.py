from __future__ import annotations

from typing import Any

import pytest
from typer.testing import CliRunner

from darjeeling.cli import _BudgetedReferenceBroker, app
from darjeeling.model import ReferenceContext, ReferenceResponse


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

    compile_help = runner.invoke(app, ["compile", "--help"], env={"COLUMNS": "200"})
    run_help = runner.invoke(app, ["compile", "run", "--help"], env={"COLUMNS": "200"})

    assert compile_help.exit_code == 0, compile_help.output
    assert run_help.exit_code == 0, run_help.output
    assert "Compile target-local artifacts." in compile_help.output
    assert "--reference-config" in run_help.output
    assert "--agent-command" in run_help.output
    assert "--l4-deadline-ms" in run_help.output


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
