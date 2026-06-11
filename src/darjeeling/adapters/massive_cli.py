from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from darjeeling.adapters.massive import prepare_massive_dataset
from darjeeling.settings import DEFAULT_PROCESSED_DATA_DIR

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.callback()
def main() -> None:
    """MASSIVE adapter CLI."""


@app.command()
def prepare(
    locale: Annotated[str, typer.Option(help="MASSIVE locale/config to prepare.")] = "en-US",
    out: Annotated[
        Path,
        typer.Option(help="Output directory for processed parquet/jsonl files."),
    ] = DEFAULT_PROCESSED_DATA_DIR,
) -> None:
    """Download and process MASSIVE records for Darjeeling replay."""

    result = prepare_massive_dataset(locale=locale, out_dir=out)
    console.print(f"prepared {result['records']} records in {out}")
