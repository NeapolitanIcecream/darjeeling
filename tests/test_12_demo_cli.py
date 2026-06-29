from __future__ import annotations

from typer.testing import CliRunner

from darjeeling.cli import app


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
