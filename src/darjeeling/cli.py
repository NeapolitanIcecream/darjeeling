from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

import typer

from darjeeling.agent_workspace import (
    create_agent_workspace,
    create_compile_run,
    launch_target_adaptation_agent_async,
    load_target_workspace,
    mount_readonly_inputs,
    write_agent_brief,
)
from darjeeling.artifact_worker import build_protocol_docs
from darjeeling.candidate_evaluation import (
    compare_candidates,
    evaluate_candidate_on_test,
    finalize_report,
)
from darjeeling.compile_orchestration import run_interactive_compile_loop
from darjeeling.demos.thin_target import format_thin_target_report, run_thin_target_demo
from darjeeling.model import (
    AgentAttemptOptions,
    AgentSearchGuidance,
    AgentUsageLedger,
    AgentViewOptions,
    AgentWorkspacePermissions,
    CompileBudget,
    CompileOptions,
    L4BaselineSummary,
    ReferenceBroker,
    ReferenceBudget,
    ReferenceContext,
    ReferenceQualificationOptions,
    ReferenceResponse,
    ReferenceUsageLedger,
    ReleaseBaseline,
    ReleaseRegistry,
    RoutingSettings,
    SnapshotOptions,
    TargetCheckOptions,
    WorkspaceStore,
)
from darjeeling.reference_config import build_reference_broker_from_config
from darjeeling.release_runtime import create_release_without_artifacts
from darjeeling.snapshot_reference import build_snapshot, export_train_view_for_agent
from darjeeling.target_definition import (
    check_target_definition,
    export_agent_readonly_target_view,
    load_checked_target,
)
from darjeeling.util import utcnow, write_json

app = typer.Typer(help="Darjeeling CLI.")
target_app = typer.Typer(help="Target definition commands.")
demo_app = typer.Typer(help="Demo commands.")
compile_app = typer.Typer(help="Compile target-local artifacts.")
app.add_typer(target_app, name="target")
app.add_typer(demo_app, name="demo")
app.add_typer(compile_app, name="compile")


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


@demo_app.command("thin-target")
def demo_thin_target() -> None:
    """Run the no-network five-minute demo."""
    typer.echo(format_thin_target_report(run_thin_target_demo()))


@compile_app.command("run")
def compile_run(
    target_path: Annotated[
        Path,
        typer.Argument(help="Path to a target directory containing target.yaml."),
    ],
    run_root: Annotated[
        Path,
        typer.Option(
            "--run-root",
            help="Directory where manifests, reports, snapshots, and logs are written.",
        ),
    ],
    reference_config: Annotated[
        Path,
        typer.Option(
            "--reference-config",
            help="JSON/YAML OpenAI-compatible reference provider/cache config.",
        ),
    ],
    agent_command: Annotated[
        str,
        typer.Option(
            "--agent-command",
            help="JSON array command for the target-adaptation agent.",
        ),
    ],
    workspace_root: Annotated[
        Path | None,
        typer.Option(
            "--workspace-root",
            help="Agent workspace root. Defaults to <run-root>/workspaces.",
        ),
    ] = None,
    max_candidates: Annotated[
        int,
        typer.Option("--max-candidates", help="Maximum ready candidates to evaluate."),
    ] = 1,
    max_agent_seconds: Annotated[
        int,
        typer.Option("--max-agent-seconds", help="Target-adaptation agent wall limit."),
    ] = 300,
    max_cost: Annotated[
        float | None,
        typer.Option("--max-cost", help="Maximum compile/reference cost in USD."),
    ] = None,
    enabled_layers: Annotated[
        str,
        typer.Option(
            "--enabled-layers",
            help="Comma-separated lower layers enabled for the cold-start compile context.",
        ),
    ] = "L1,L2,L3",
    l4_deadline_ms: Annotated[
        int,
        typer.Option(
            "--l4-deadline-ms",
            help="Deadline for live L4/reference fallback and cold-start checks.",
        ),
    ] = 30_000,
    agent_network: Annotated[
        bool,
        typer.Option("--agent-network/--no-agent-network", help="Allow agent network research."),
    ] = False,
    agent_dependency_install: Annotated[
        bool,
        typer.Option(
            "--agent-dependency-install/--no-agent-dependency-install",
            help="Record authorization for workspace-local dependency installation.",
        ),
    ] = False,
    allow_insufficient_reference: Annotated[
        bool,
        typer.Option(
            "--allow-insufficient-reference",
            help="Allow compile to proceed when reference evidence is insufficient but not failed.",
        ),
    ] = False,
) -> None:
    """Run a target-independent compile attempt and final test handoff."""
    try:
        summary = _run_compile_command(
            target_path=target_path,
            run_root=run_root,
            reference_config=reference_config,
            agent_command=agent_command,
            workspace_root=workspace_root,
            max_candidates=max_candidates,
            max_agent_seconds=max_agent_seconds,
            max_cost=max_cost,
            enabled_layers=enabled_layers,
            l4_deadline_ms=l4_deadline_ms,
            agent_network=agent_network,
            agent_dependency_install=agent_dependency_install,
            allow_insufficient_reference=allow_insufficient_reference,
        )
    except Exception as exc:
        typer.echo(f"compile run failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(
        "compile run completed: "
        f"run_root={summary['run_root']} "
        f"selected_candidate={summary.get('selected_candidate_id')} "
        f"decision={summary.get('decision_status')}"
    )


def _run_compile_command(
    *,
    target_path: Path,
    run_root: Path,
    reference_config: Path,
    agent_command: str,
    workspace_root: Path | None,
    max_candidates: int,
    max_agent_seconds: int,
    max_cost: float | None,
    enabled_layers: str,
    l4_deadline_ms: int,
    agent_network: bool,
    agent_dependency_install: bool,
    allow_insufficient_reference: bool,
) -> dict[str, object]:
    run_root = run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "reports").mkdir(parents=True, exist_ok=True)
    layers = _parse_enabled_layers(enabled_layers)
    command = _parse_agent_command(agent_command)
    if max_candidates <= 0:
        raise ValueError("--max-candidates must be positive")
    if max_agent_seconds <= 0:
        raise ValueError("--max-agent-seconds must be positive")
    if l4_deadline_ms <= 0:
        raise ValueError("--l4-deadline-ms must be positive")
    if max_cost is not None and max_cost < 0:
        raise ValueError("--max-cost must be non-negative")
    if max_cost == 0:
        raise ValueError("--max-cost 0 leaves no budget for the required reference probe")
    _ensure_agent_execution_supported()
    definition, contract, target_check = load_checked_target(
        target_path, require_reference=True
    )
    broker = _BudgetedReferenceBroker(
        build_reference_broker_from_config(reference_config), max_cost
    )
    registry = ReleaseRegistry()
    routing = RoutingSettings(
        cache_enabled=False,
        enabled_layers=layers,
        total_deadline_ms=l4_deadline_ms,
    )
    cold_release = create_release_without_artifacts(
        definition,
        contract,
        target_check,
        broker,
        routing,
        registry,
    )
    snapshot_options = SnapshotOptions(
        allow_insufficient_reference=allow_insufficient_reference,
        qualification_options=ReferenceQualificationOptions(),
        reference_budget=ReferenceBudget(max_cost=max_cost),
        storage_root=run_root / "snapshots",
    )
    snapshot_result = build_snapshot(
        definition,
        contract,
        definition.data_config,
        None,
        [],
        broker,
        utcnow(),
        snapshot_options,
    )
    compile_reference_usage = _compile_reference_usage(broker, snapshot_result)
    workspace_store = WorkspaceStore(
        workspace_root.resolve() if workspace_root else run_root / "workspaces"
    )
    compile_options = CompileOptions(
        objective={
            "optimize": "coverage",
            "enabled_layers": layers,
            "l4_deadline_ms": l4_deadline_ms,
        },
        agent_guidance=AgentSearchGuidance(
            preferred_strategies=["simple_baseline_first", "iterate_from_core_feedback"],
            preferred_tools=["python"],
            extra_instructions=(
                "Submit L1/L2/L3 artifacts only when they can safely accept; "
                "use Core validation feedback only."
            ),
        ),
        allow_insufficient_reference_qualification=allow_insufficient_reference,
    )
    agent_options = AgentAttemptOptions(
        agent_model="cli",
        agent_command=command,
        agent_timeout_seconds=max_agent_seconds,
        permissions=AgentWorkspacePermissions(
            network_access=agent_network,
            dependency_install=agent_dependency_install,
        ),
    )
    workspace = load_target_workspace(
        definition.name,
        definition.contract_hash,
        workspace_store,
    )
    compile_run = create_compile_run(
        definition,
        target_check,
        snapshot_result.snapshot,
        cold_release,
        CompileBudget(
            max_agent_seconds=max_agent_seconds,
            max_candidates=max_candidates,
            max_cost=max_cost,
        ),
        workspace,
        snapshot_result.reference_qualification,
        compile_options,
    )
    attempt = create_agent_workspace(compile_run, workspace, agent_options)
    target_view = export_agent_readonly_target_view(
        definition,
        attempt.workspace_path / "readonly_source" / "target",
    )
    train_view = export_train_view_for_agent(
        snapshot_result.snapshot,
        contract,
        AgentViewOptions(),
        attempt.workspace_path / "readonly_source" / "train",
    )
    mount_manifest = mount_readonly_inputs(
        attempt,
        target_view,
        train_view,
        cold_release,
        report_views=[],
        telemetry_summaries=[],
        protocol_docs=build_protocol_docs("v1"),
    )
    brief = write_agent_brief(
        attempt,
        compile_run,
        mount_manifest,
        compile_options.objective,
        compile_options.agent_guidance,
        agent_options.permissions,
    )
    handle = launch_target_adaptation_agent_async(
        attempt,
        brief,
        {
            "command": command,
            "timeout_seconds": max_agent_seconds,
            "permissions": asdict(agent_options.permissions),
        },
    )
    loop_result = run_interactive_compile_loop(
        compile_run,
        attempt,
        handle,
        definition,
        contract,
        snapshot_result.snapshot,
        cold_release,
        snapshot_result.reference_qualification,
        compile_reference_usage,
        {"serving_l4_cost": _estimated_baseline_serving_cost(snapshot_result)},
        {"artifact_store": run_root / "artifacts"},
        poll_interval_seconds=0.05,
    )
    selected_candidate = loop_result.get("selected_candidate")
    validation_report = loop_result.get("validation_report")
    closed_attempt = loop_result.get("closed_attempt")
    if selected_candidate is None or validation_report is None or closed_attempt is None:
        raise RuntimeError(
            "interactive compile completed without a selected candidate for final test"
        )
    test_result = evaluate_candidate_on_test(
        selected_candidate,
        closed_attempt,
        definition,
        snapshot_result.snapshot,
        cold_release,
        snapshot_result.reference_qualification,
        loop_result.get("agent_usage_ledger", AgentUsageLedger()),
        compile_reference_usage,
        None,
        None,
        {"serving_l4_cost": _estimated_baseline_serving_cost(snapshot_result)},
        {
            "contract": contract,
            "validation_report": validation_report,
            "test_results_visible": True,
        },
    )
    decision = compare_candidates(
        [test_result["report"]],
        ReleaseBaseline(
            cold_release,
            l4_baseline=L4BaselineSummary(
                cold_release.release_id,
                definition.name,
                definition.contract_hash,
                {},
                [],
                snapshot_result.reference_qualification,
                {},
                {},
            ),
        ),
        {"requirements": definition.requirements},
    )
    final_report = finalize_report(test_result["report"], decision, validation_report)
    manifest = {
        "target_path": target_path.resolve(),
        "run_root": run_root,
        "reference_config": reference_config.resolve(),
        "workspace_root": workspace_store.root,
        "effective_l4_deadline_ms": l4_deadline_ms,
        "enabled_layers": layers,
        "compile_id": compile_run.compile_id,
        "attempt_id": attempt.attempt_id,
        "snapshot_id": snapshot_result.snapshot.snapshot_id,
        "base_release_id": cold_release.release_id,
        "interactive_result_path": loop_result["interactive_result_path"],
        "selected_candidate_id": loop_result["selected_candidate_id"],
        "selected_validation_report_id": loop_result["selected_validation_report_id"],
        "reference_timeout_ms": getattr(broker, "config", None).timeout_ms
        if hasattr(broker, "config")
        else None,
        "reference_call_count": broker.call_count,
        "reference_cost": broker.cost,
        "created_at": utcnow(),
    }
    summary = {
        "run_root": str(run_root),
        "compile_id": compile_run.compile_id,
        "attempt_id": attempt.attempt_id,
        "selected_candidate_id": loop_result["selected_candidate_id"],
        "selected_validation_report_id": loop_result["selected_validation_report_id"],
        "interactive_result_path": str(loop_result["interactive_result_path"]),
        "final_test_report_id": test_result["report"].report_id,
        "final_report_id": final_report.report_id,
        "decision_status": decision.status,
        "decision_blockers": list(decision.release_blockers),
        "effective_l4_deadline_ms": l4_deadline_ms,
        "reference_timeout_ms": manifest["reference_timeout_ms"],
        "reference_call_count": broker.call_count,
        "reference_cost": broker.cost,
        "cost": asdict(final_report.cost),
    }
    write_json(run_root / "manifest.json", manifest)
    write_json(run_root / "reports" / "compile_summary.json", summary)
    write_json(run_root / "reports" / "test_report.json", asdict(test_result["report"]))
    write_json(run_root / "reports" / "final_report.json", asdict(final_report))
    return summary


def _parse_agent_command(value: str) -> list[str]:
    try:
        command = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("--agent-command must be a JSON string array") from exc
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(part, str) or not part for part in command)
    ):
        raise ValueError("--agent-command must be a non-empty JSON string array")
    return command


def _parse_enabled_layers(value: str) -> list[str]:
    layers = [part.strip() for part in value.split(",") if part.strip()]
    allowed = {"L1", "L2", "L3"}
    if not layers or any(layer not in allowed for layer in layers):
        raise ValueError("--enabled-layers must contain only L1,L2,L3")
    if len(set(layers)) != len(layers):
        raise ValueError("--enabled-layers must not contain duplicates")
    return layers


def _estimated_baseline_serving_cost(snapshot_result) -> float:
    qualification = snapshot_result.reference_qualification
    usage = snapshot_result.reference_usage
    total = float(qualification.cost.get("total", 0.0)) + float(usage.cost)
    calls = max(
        int(qualification.latency.get("sample_count", 0)) + int(usage.call_count),
        1,
    )
    return total / calls


def _compile_reference_usage(broker: Any, snapshot_result: Any) -> ReferenceUsageLedger:
    usage = snapshot_result.reference_usage
    return ReferenceUsageLedger(
        call_count=max(
            int(getattr(broker, "call_count", 0)),
            int(getattr(usage, "call_count", 0)),
        ),
        cost=max(float(getattr(broker, "cost", 0.0)), float(getattr(usage, "cost", 0.0))),
        errors=dict(getattr(usage, "errors", {}) or {}),
    )


def _ensure_agent_execution_supported() -> None:
    if shutil.which("sandbox-exec") is None:
        raise ValueError(
            "agent execution requires macOS sandbox-exec; sandbox-exec was not found. "
            "No reference calls were made."
        )


class _BudgetedReferenceBroker:
    def __init__(self, broker: ReferenceBroker, max_cost: float | None):
        self._broker = broker
        self.max_cost = max_cost
        self.reference_version = broker.reference_version
        self.config = getattr(broker, "config", None)
        self.cost = 0.0
        self.call_count = 0

    def call(self, request: dict[str, Any], context: ReferenceContext) -> ReferenceResponse:
        if self.max_cost is not None and self.cost >= self.max_cost:
            raise RuntimeError("reference cost budget exhausted before provider call")
        response = self._broker.call(request, context)
        self.call_count += 1
        self.cost += max(float(response.cost), 0.0)
        if self.max_cost is not None and self.cost > self.max_cost:
            raise RuntimeError("reference cost budget exhausted after provider call")
        return response
