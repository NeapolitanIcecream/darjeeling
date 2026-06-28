from __future__ import annotations

from pathlib import Path

import typer

from darjeeling.model import TargetCheckOptions
from darjeeling.target_definition import check_target_definition

app = typer.Typer(help="Darjeeling CLI.")
target_app = typer.Typer(help="Target definition commands.")
app.add_typer(target_app, name="target")


@target_app.command("check")
def target_check(target_path: Path, require_reference: bool = False) -> None:
    report = check_target_definition(
        target_path, TargetCheckOptions(require_reference=require_reference)
    )
    if report.status == "pass":
        typer.echo(f"target check passed: {report.target_name} {report.contract_hash}")
        return
    for failure in report.failures:
        typer.echo(f"failure: {failure}", err=True)
    raise typer.Exit(1)
