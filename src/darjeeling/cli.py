from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console

from darjeeling.targets import registry as target_registry

app = typer.Typer(no_args_is_help=True)
console = Console()
error_console = Console(stderr=True)
_settings_path: Path | None = None


@app.callback()
def main(
    settings: Annotated[
        Path | None,
        typer.Option("--settings", help="Optional target settings file."),
    ] = None,
) -> None:
    """Darjeeling core CLI."""

    global _settings_path
    _settings_path = settings


@app.command()
def run(
    stream: Annotated[
        str,
        typer.Option(help="Replay stream type, interpreted by the selected target."),
    ] = "zipf-heavy",
    max_requests: Annotated[int, typer.Option(min=1)] = 3000,
    compile_every: Annotated[int, typer.Option(min=1)] = 500,
    teacher: Annotated[
        str,
        typer.Option(help="Teacher mode: live, cache, or live-or-cache."),
    ] = "live-or-cache",
    run_dir: Annotated[Path, typer.Option(help="Run output directory.")] = Path("runs/latest"),
    data_dir: Annotated[
        Path,
        typer.Option(help="Target-owned processed data directory."),
    ] = Path("data/processed/default"),
    target: Annotated[
        str,
        typer.Option(help="Target name from the static target registry."),
    ] = target_registry.default_target_name(),
) -> None:
    """Run the selected target through the core entrypoint."""

    try:
        settings = _load_target_settings(target)
        summary = _execute_replay_run(
            stream=stream,
            max_requests=max_requests,
            compile_every=compile_every,
            teacher=teacher,
            run_dir=run_dir,
            data_dir=data_dir,
            target=target,
            settings=settings,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        error_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc

    console.print(
        f"ran {summary.requests} requests; traces={summary.traces_path}; "
        f"layers={summary.layer_counts}"
    )


@app.command()
def report(
    run_dir: Annotated[
        Path,
        typer.Option(help="Run directory to summarize."),
    ] = Path("runs/latest"),
    target: Annotated[
        str,
        typer.Option(help="Target name from the static target registry."),
    ] = target_registry.default_target_name(),
) -> None:
    """Generate report files for the selected target."""

    try:
        target_object = target_registry.get_target(target)
        result = _call_target_method(target_object, "generate_report", run_dir=run_dir)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        error_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc
    console.print(f"wrote {result.summary_path}")


def _load_target_settings(target: str) -> Any:
    target_object = target_registry.get_target(target)
    return _call_target_method(target_object, "load_settings", settings_path=_settings_path)


def _execute_replay_run(
    *,
    stream: str,
    max_requests: int,
    compile_every: int,
    teacher: str,
    run_dir: Path,
    data_dir: Path,
    target: str,
    settings: Any,
) -> Any:
    target_object = target_registry.get_target(target)
    return _call_target_method(
        target_object,
        "run_replay",
        stream=stream,
        max_requests=max_requests,
        compile_every=compile_every,
        teacher=teacher,
        run_dir=run_dir,
        data_dir=data_dir,
        settings=settings,
    )


def _call_target_method(target_object: object, method_name: str, **kwargs: Any) -> Any:
    method = getattr(target_object, method_name, None)
    if method is None:
        target_name = getattr(target_object, "name", target_object.__class__.__name__)
        raise ValueError(f"target {target_name!r} does not implement {method_name}")
    return method(**kwargs)
