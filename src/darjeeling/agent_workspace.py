from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from darjeeling.errors import WorkspaceError
from darjeeling.model import (
    AgentAttempt,
    AgentAttemptOptions,
    AgentFeedback,
    AgentSessionHandle,
    AgentUsageEvent,
    AgentUsageLedger,
    AgentVisibleReport,
    AgentVisibleTelemetrySummary,
    Candidate,
    CandidateSubmission,
    ClosedAgentAttempt,
    CompileBudget,
    CompileOptions,
    CompileRun,
    FeedbackDeliveryRecord,
    JournalEntry,
    ProtocolDocs,
    ReferenceQualificationReport,
    Release,
    TargetChangeProposal,
    TargetCheckReport,
    TargetDefinition,
    TargetViewManifest,
    TargetWorkspace,
    TrainViewManifest,
    WorkspaceBaselineUpdate,
    WorkspaceMountManifest,
    WorkspaceStore,
)
from darjeeling.portable_sandbox import build_python_sandbox_command, is_python_command
from darjeeling.util import file_digest, new_id, read_json, stable_hash, utcnow, write_json

_BASELINE_DIRS = ["scaffolding", "runtime", "proposals", "journal", "tests"]
_ATTEMPT_DIRS = [*_BASELINE_DIRS, "submissions"]
_FORBIDDEN_REPORT_KEYS = {
    "record_id",
    "record_ids",
    "request_id",
    "request_ids",
    "row_id",
    "row_ids",
    "snapshot_record_id",
    "snapshot_record_ids",
    "source_record_id",
    "source_record_ids",
    "split_index",
    "split_indices",
    "normalized_input_key",
    "normalized_input_keys",
    "split_group_key",
    "split_group_keys",
    "input",
    "inputs",
    "raw_input",
    "raw_inputs",
    "expected_output",
    "expected_outputs",
    "reference_output",
    "reference_outputs",
    "validation_rows",
    "test_rows",
}
_LIVE_AGENT_PROCESSES: dict[str, subprocess.Popen[str]] = {}


def _ensure_workspace_layout(path: Path, *, include_submissions: bool = True) -> None:
    writable_dirs = _ATTEMPT_DIRS if include_submissions else _BASELINE_DIRS
    for name in writable_dirs:
        (path / name).mkdir(parents=True, exist_ok=True)
    for layer in ["l1", "l2", "l3"]:
        (path / "runtime" / layer).mkdir(parents=True, exist_ok=True)


def _workspace_tree_digest(
    path: Path,
    *,
    include_roots: list[str] | None = None,
    exclude_relpaths: set[str] | None = None,
) -> str:
    entries: list[tuple[str, str]] = []
    excludes = exclude_relpaths or set()
    roots = [path / root for root in include_roots] if include_roots else [path]
    for root in roots:
        if not root.exists():
            continue
        candidates = [root] if root.is_file() else sorted(root.rglob("*"))
        for item in candidates:
            rel = item.relative_to(path).as_posix()
            if rel == ".darjeeling_workspace.json" or rel in excludes:
                continue
            if item.is_file() and not item.is_symlink():
                entries.append((rel, file_digest(item)))
            elif item.is_symlink():
                raise WorkspaceError("target workspace must not contain symlinks")
    return stable_hash(entries)


def _baseline_content_digest(path: Path) -> str:
    return _workspace_tree_digest(
        path,
        include_roots=_BASELINE_DIRS,
        exclude_relpaths={"journal/closed.json"},
    )


def _assert_no_forbidden_report_material(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in _FORBIDDEN_REPORT_KEYS:
                raise WorkspaceError(
                    "agent-visible report contains holdout reconstruction material"
                )
            _assert_no_forbidden_report_material(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_forbidden_report_material(child)


def _assert_agent_visible_report_safe(report: AgentVisibleReport) -> None:
    _assert_no_forbidden_report_material(asdict(report))


def _path_contains(parent: Path, child: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _sandbox_path(value: Path) -> str:
    resolved = str(value.resolve())
    return '"' + resolved.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _make_readonly_tree(path: Path) -> None:
    if not path.exists():
        return
    for item in sorted(path.rglob("*"), reverse=True):
        if item.is_dir():
            item.chmod(0o555)
        elif item.is_file():
            item.chmod(0o444)
    path.chmod(0o555)


def _protect_siblings_from(root: Path, attempt_path: Path) -> list[Path]:
    protected: list[Path] = []
    current = root.resolve()
    attempt_resolved = attempt_path.resolve()
    while current != attempt_resolved and current.is_dir():
        try:
            rel_parts = attempt_resolved.relative_to(current).parts
        except ValueError:
            break
        if not rel_parts:
            break
        allowed_child = current / rel_parts[0]
        for child in current.iterdir():
            if child != allowed_child:
                protected.append(child)
        current = allowed_child
    return protected


def _default_protected_paths(attempt_path: Path, command: list[str]) -> list[Path]:
    repo_root = Path(__file__).resolve().parents[2]
    protected: list[Path] = []
    if _path_contains(repo_root, attempt_path):
        protected.extend(_protect_siblings_from(repo_root, attempt_path))
    else:
        protected.append(repo_root)
    if len(attempt_path.parents) >= 5:
        workspace_parent = attempt_path.parents[4]
        protected.extend(_protect_siblings_from(workspace_parent, attempt_path))
    command_path = shutil.which(command[0]) or command[0]
    command_resolved = Path(command_path).expanduser().resolve()
    for protected_path in protected:
        if _path_contains(protected_path, command_resolved):
            raise WorkspaceError("agent runtime command may not live inside protected Core paths")
    return protected


def _write_agent_sandbox_profile(
    attempt_path: Path, command: list[str], protected_paths: list[Path]
) -> Path:
    sandbox_exec = shutil.which("sandbox-exec")
    if sandbox_exec is None:
        raise WorkspaceError("sandbox-exec is required for agent runtime isolation")
    profile_path = attempt_path / "journal" / "agent_sandbox.sb"
    all_protected = _default_protected_paths(attempt_path, command)
    for path in protected_paths:
        if not _path_contains(path, attempt_path):
            all_protected.append(path)
    readonly_paths = [
        attempt_path / "readonly_inputs",
        attempt_path / "readonly_source",
    ]
    lines = [
        "(version 1)",
        "(allow default)",
        "(deny network*)",
    ]
    for path in readonly_paths:
        if path.exists():
            lines.append(f"(deny file-write* (subpath {_sandbox_path(path)}))")
    for path in all_protected:
        if path.exists():
            lines.append(f"(deny file-read* (subpath {_sandbox_path(path)}))")
            lines.append(f"(deny file-write* (subpath {_sandbox_path(path)}))")
    profile_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return profile_path


def _agent_portable_sandbox_command(
    attempt_path: Path, command: list[str], protected_paths: list[Path]
) -> list[str]:
    if not is_python_command(command):
        raise WorkspaceError(
            "sandbox-exec or a Python agent command is required for agent runtime isolation"
        )
    all_protected = _default_protected_paths(attempt_path, command)
    for path in protected_paths:
        if not _path_contains(path, attempt_path):
            all_protected.append(path)
    readonly_paths = [
        attempt_path / "readonly_inputs",
        attempt_path / "readonly_source",
    ]
    try:
        return build_python_sandbox_command(
            command,
            cwd=attempt_path,
            config_path=attempt_path / "journal" / "agent_python_sandbox.json",
            allowed_read_roots=[attempt_path],
            allowed_write_roots=[attempt_path],
            denied_read_roots=all_protected,
            denied_write_roots=[*readonly_paths, *all_protected],
        )
    except Exception as exc:
        raise WorkspaceError(str(exc)) from exc


def _expected_train_export_digest(train_view: TrainViewManifest) -> str:
    return stable_hash(
        {
            "view_kind": "agent_train_export",
            "snapshot_id": train_view.snapshot_id,
            "snapshot_digest": train_view.snapshot_digest,
            "view_digest": file_digest(train_view.view_path),
            "record_count": train_view.record_count,
            "redaction_level": train_view.redaction_level,
        }
    )


def load_target_workspace(
    target_name: str, contract_hash: str, workspace_store: WorkspaceStore
) -> TargetWorkspace:
    workspace_path = workspace_store.root / target_name / "main"
    workspace_path.mkdir(parents=True, exist_ok=True)
    _ensure_workspace_layout(workspace_path, include_submissions=False)
    manifest_path = workspace_path / ".darjeeling_workspace.json"
    if manifest_path.exists():
        import json

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["contract_hash"] != contract_hash and not manifest.get("migration_record"):
            raise WorkspaceError("target workspace contract hash changed without migration record")
        commit = _workspace_tree_digest(workspace_path)
        if commit != manifest["baseline_commit"]:
            raise WorkspaceError("target workspace baseline drifted without baseline update")
        baseline_commit = manifest["baseline_commit"]
        last_accepted_release_id = manifest.get("source_release_id")
    else:
        baseline_commit = _workspace_tree_digest(workspace_path)
        last_accepted_release_id = None
        write_json(
            manifest_path,
            {
                "target_name": target_name,
                "contract_hash": contract_hash,
                "baseline_commit": baseline_commit,
            },
        )
    return TargetWorkspace(
        target_name=target_name,
        workspace_path=workspace_path,
        baseline_commit=baseline_commit,
        contract_hash=contract_hash,
        last_accepted_release_id=last_accepted_release_id,
    )


def create_compile_run(
    definition: TargetDefinition,
    target_check: TargetCheckReport,
    snapshot: Any,
    base_release: Release,
    budget: CompileBudget,
    workspace: TargetWorkspace,
    reference_qualification: ReferenceQualificationReport,
    compile_options: CompileOptions,
) -> CompileRun:
    if target_check.status != "pass":
        raise WorkspaceError("target check must pass before compile run")
    if (
        target_check.target_name != definition.name
        or target_check.contract_hash != definition.contract_hash
    ):
        raise WorkspaceError("target check does not match target definition")
    if (
        definition.name != base_release.target_name
        or definition.contract_hash != base_release.contract_hash
    ):
        raise WorkspaceError("base release does not match target definition")
    if (
        definition.name != workspace.target_name
        or definition.contract_hash != workspace.contract_hash
    ):
        raise WorkspaceError("workspace does not match target definition")
    if reference_qualification.status not in {"pass", "insufficient"}:
        raise WorkspaceError("reference qualification is not acceptable")
    if (
        reference_qualification.status == "insufficient"
        and not compile_options.allow_insufficient_reference_qualification
    ):
        raise WorkspaceError("insufficient reference qualification requires explicit approval")
    if (
        snapshot.target_name != definition.name
        or snapshot.contract_hash != definition.contract_hash
    ):
        raise WorkspaceError("snapshot does not match target definition")
    if (
        reference_qualification.target_name != definition.name
        or reference_qualification.contract_hash != definition.contract_hash
    ):
        raise WorkspaceError("reference qualification does not match target definition")
    return CompileRun(
        compile_id=new_id("compile"),
        target_name=definition.name,
        contract_hash=definition.contract_hash,
        snapshot_id=snapshot.snapshot_id,
        snapshot_digest=snapshot.snapshot_digest,
        base_release_id=base_release.release_id,
        workspace_baseline_commit=workspace.baseline_commit,
        started_at=utcnow(),
        budget=budget,
        status="running",
    )


def create_agent_workspace(
    compile_run: CompileRun,
    target_workspace: TargetWorkspace,
    attempt_options: AgentAttemptOptions,
) -> AgentAttempt:
    if (
        compile_run.target_name != target_workspace.target_name
        or compile_run.contract_hash != target_workspace.contract_hash
    ):
        raise WorkspaceError("compile run does not match target workspace")
    if compile_run.workspace_baseline_commit != target_workspace.baseline_commit:
        raise WorkspaceError("compile run does not point at target workspace baseline")
    if _workspace_tree_digest(target_workspace.workspace_path) != target_workspace.baseline_commit:
        raise WorkspaceError("target workspace baseline changed before attempt clone")
    attempt_id = new_id("attempt")
    attempt_path = (
        target_workspace.workspace_path.parent / "attempts" / compile_run.compile_id / attempt_id
    )
    if attempt_path.exists():
        shutil.rmtree(attempt_path)
    shutil.copytree(target_workspace.workspace_path, attempt_path, symlinks=False)
    _ensure_workspace_layout(attempt_path)
    initial_commit = _workspace_tree_digest(attempt_path)
    return AgentAttempt(
        attempt_id=attempt_id,
        compile_id=compile_run.compile_id,
        target_name=compile_run.target_name,
        contract_hash=compile_run.contract_hash,
        snapshot_id=compile_run.snapshot_id,
        snapshot_digest=compile_run.snapshot_digest,
        workspace_path=attempt_path,
        source_workspace_commit=target_workspace.baseline_commit,
        initial_commit=initial_commit,
        final_commit=None,
        agent_model=attempt_options.agent_model,
        status="running",
    )


def advance_target_workspace_baseline(
    target_workspace: TargetWorkspace,
    closed_attempt: ClosedAgentAttempt,
    release: Release | None,
    reason: str,
    accepted_candidate: Candidate | None = None,
) -> WorkspaceBaselineUpdate:
    if reason not in {"accepted_release", "explicit_carry_forward"}:
        raise WorkspaceError("invalid baseline advance reason")
    if reason == "accepted_release" and release is None:
        raise WorkspaceError("accepted release baseline advancement requires a release")
    if reason == "accepted_release":
        if release is None or release.candidate_id is None or release.approval is None:
            raise WorkspaceError("accepted release baseline advancement requires compiled release")
        if accepted_candidate is None:
            raise WorkspaceError("accepted release baseline advancement requires candidate lineage")
        if (
            release.target_name != target_workspace.target_name
            or release.contract_hash != target_workspace.contract_hash
            or accepted_candidate.target_name != target_workspace.target_name
            or accepted_candidate.contract_hash != target_workspace.contract_hash
        ):
            raise WorkspaceError("accepted release baseline advancement scope mismatch")
        if release.candidate_id != accepted_candidate.candidate_id:
            raise WorkspaceError("accepted release does not match accepted candidate")
        if (
            accepted_candidate.compile_id != closed_attempt.compile_id
            or accepted_candidate.attempt_id != closed_attempt.attempt_id
        ):
            raise WorkspaceError("accepted candidate does not match closed attempt")
    if (
        closed_attempt.target_name != target_workspace.target_name
        or closed_attempt.contract_hash != target_workspace.contract_hash
    ):
        raise WorkspaceError("closed attempt does not match target workspace")
    if closed_attempt.source_workspace_commit != target_workspace.baseline_commit:
        raise WorkspaceError("closed attempt does not match current workspace baseline")
    if _baseline_content_digest(closed_attempt.workspace_path) != closed_attempt.final_commit:
        raise WorkspaceError("closed attempt final commit does not match current workspace")
    previous = target_workspace.baseline_commit
    if target_workspace.workspace_path.exists():
        shutil.rmtree(target_workspace.workspace_path)
    target_workspace.workspace_path.mkdir(parents=True, exist_ok=True)
    for name in _BASELINE_DIRS:
        source = closed_attempt.workspace_path / name
        destination = target_workspace.workspace_path / name
        if source.exists():
            shutil.copytree(source, destination, symlinks=False)
        else:
            destination.mkdir(parents=True, exist_ok=True)
    _ensure_workspace_layout(target_workspace.workspace_path, include_submissions=False)
    new_commit = _workspace_tree_digest(target_workspace.workspace_path)
    write_json(
        target_workspace.workspace_path / ".darjeeling_workspace.json",
        {
            "target_name": target_workspace.target_name,
            "contract_hash": target_workspace.contract_hash,
            "baseline_commit": new_commit,
            "source_attempt_id": closed_attempt.attempt_id,
            "source_release_id": release.release_id if release else None,
            "reason": reason,
        },
    )
    return WorkspaceBaselineUpdate(
        target_name=target_workspace.target_name,
        previous_commit=previous,
        new_commit=new_commit,
        source_attempt_id=closed_attempt.attempt_id,
        source_release_id=release.release_id if release else None,
        reason=reason,  # type: ignore[arg-type]
    )


def mount_readonly_inputs(
    attempt: AgentAttempt,
    target_view: TargetViewManifest,
    train_view: TrainViewManifest,
    base_release_view: Any,
    report_views: list[Any],
    telemetry_summaries: list[Any],
    protocol_docs: ProtocolDocs,
) -> WorkspaceMountManifest:
    expected_target = target_view.target_name
    expected_contract = target_view.contract_hash
    if expected_target != attempt.target_name or expected_contract != attempt.contract_hash:
        raise WorkspaceError("target view does not match agent attempt")
    if (
        getattr(base_release_view, "target_name", None) != expected_target
        or getattr(base_release_view, "contract_hash", None) != expected_contract
    ):
        raise WorkspaceError("base release view does not match target view")
    train_rows = read_json(train_view.view_path)
    if not isinstance(train_rows, list):
        raise WorkspaceError("train view must contain a row list")
    if train_view.view_kind != "agent_train_export":
        raise WorkspaceError("train view must come from the agent train export path")
    if (
        train_view.snapshot_id != attempt.snapshot_id
        or train_view.snapshot_digest != attempt.snapshot_digest
    ):
        raise WorkspaceError("train view does not match attempt snapshot")
    if train_view.record_count != len(train_rows):
        raise WorkspaceError("train view manifest row count does not match file")
    if train_view.export_digest != _expected_train_export_digest(train_view):
        raise WorkspaceError("train view export digest does not match file")
    for row in train_rows:
        if not isinstance(row, dict):
            raise WorkspaceError("train view rows must be objects")
        split_eligibility = row.get("split_eligibility")
        if not isinstance(split_eligibility, list) or "train" not in split_eligibility:
            raise WorkspaceError("train view row is not train eligible")
        if "source_provenance" not in row and train_view.redaction_level != "redacted":
            raise WorkspaceError("raw train view row is missing source provenance")
        provenance = row.get("source_provenance")
        if isinstance(provenance, dict) and provenance.get("split") in {"validation", "test"}:
            raise WorkspaceError("train view contains hidden holdout rows")
    for report in report_views:
        if not isinstance(report, AgentVisibleReport):
            raise WorkspaceError("report mounts must be AgentVisibleReport objects")
        if report.target_name != expected_target or report.contract_hash != expected_contract:
            raise WorkspaceError("report mount does not match target view")
        _assert_agent_visible_report_safe(report)
    for summary in telemetry_summaries:
        if not isinstance(summary, AgentVisibleTelemetrySummary):
            raise WorkspaceError("telemetry mounts must be AgentVisibleTelemetrySummary objects")
        if summary.target_name != expected_target or summary.contract_hash != expected_contract:
            raise WorkspaceError("telemetry mount does not match target view")
        if summary.release_id != getattr(base_release_view, "release_id", None):
            raise WorkspaceError("telemetry mount does not match base release")
        _assert_no_forbidden_report_material(
            {
                "metrics_summary": summary.metrics_summary,
                "drift_summary": summary.drift_summary,
            }
        )
    mount_path = attempt.workspace_path / "readonly_inputs"
    if mount_path.exists():
        shutil.rmtree(mount_path)
    mount_path.mkdir(parents=True)
    entries: list[str] = []
    shutil.copytree(target_view.view_path, mount_path / "target", symlinks=False)
    entries.append("target")
    shutil.copy2(train_view.view_path, mount_path / "train.json")
    entries.append("train.json")
    write_json(
        mount_path / "base_release.json",
        asdict(base_release_view)
        if hasattr(base_release_view, "__dataclass_fields__")
        else base_release_view,
    )
    entries.append("base_release.json")
    write_json(
        mount_path / "agent_visible_reports.json",
        [asdict(v) if hasattr(v, "__dataclass_fields__") else v for v in report_views],
    )
    entries.append("agent_visible_reports.json")
    write_json(
        mount_path / "agent_visible_telemetry.json",
        [asdict(v) if hasattr(v, "__dataclass_fields__") else v for v in telemetry_summaries],
    )
    entries.append("agent_visible_telemetry.json")
    (mount_path / "worker_protocol.md").write_text(protocol_docs.text, encoding="utf-8")
    entries.append("worker_protocol.md")
    forbidden = [
        "validation",
        "holdout",
        "registry_credentials",
        "production_secret",
    ]
    leaked = [
        path
        for path in mount_path.rglob("*")
        if any(term in path.name.lower() for term in forbidden)
    ]
    if leaked:
        raise WorkspaceError(
            f"agent mount contains forbidden holdout or credential material: {leaked[0]}"
        )
    return WorkspaceMountManifest(
        attempt_id=attempt.attempt_id, mount_path=mount_path, entries=entries
    )


def write_agent_brief(
    attempt: AgentAttempt,
    compile_run: CompileRun,
    mount_manifest: WorkspaceMountManifest,
    objective: dict[str, Any],
) -> Path:
    brief = attempt.workspace_path / "AGENT_BRIEF.md"
    brief.write_text(
        "\n".join(
            [
                "# Target Adaptation Brief",
                f"compile_id: {compile_run.compile_id}",
                f"attempt_id: {attempt.attempt_id}",
                f"target_name: {compile_run.target_name}",
                f"contract_hash: {compile_run.contract_hash}",
                f"budget: {asdict(compile_run.budget)}",
                "",
                "Writable directories: scaffolding/, runtime/, submissions/, "
                "proposals/, journal/, tests/.",
                "Readonly inputs are mounted under readonly_inputs/ and contain train data only.",
                "Do not access validation/test data, registry state, release "
                "internals, production secrets, L4 directly, or another "
                "autonomous coding agent.",
                "Submit candidates under submissions/<candidate>/artifacts/"
                "l1|l2|l3 with artifact.yaml files.",
                "After all candidate files are complete, create "
                "submissions/<candidate>/READY as the final atomic marker. "
                "Core evaluates only submissions with this marker.",
                "During an interactive compile, keep watching journal/ for "
                "official validation feedback files named feedback-<candidate>.json. "
                "Continue local search from aggregate feedback only.",
                "Core may stop the session when time, candidate, cost, or user-stop "
                "budgets are reached. Test evaluation and release decisions happen "
                "only after this attempt is closed.",
                f"objective: {objective}",
                f"readonly_entries: {mount_manifest.entries}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return brief


def _prepare_agent_launch(
    attempt: AgentAttempt,
    brief_path: Path,
    agent_runtime: dict[str, Any],
) -> dict[str, Any]:
    try:
        brief_path.resolve().relative_to(attempt.workspace_path.resolve())
    except ValueError as exc:
        raise WorkspaceError("agent brief path escapes attempt workspace") from exc
    if not brief_path.exists():
        raise WorkspaceError("agent brief path does not exist")
    session_record = attempt.workspace_path / "journal" / "agent_session.json"
    if session_record.exists():
        raise WorkspaceError("agent already launched for attempt")
    command = list(agent_runtime.get("command", []))
    if not command:
        raise WorkspaceError("agent runtime command is required")
    if not all(isinstance(part, str) and part for part in command):
        raise WorkspaceError("agent runtime command must be a non-empty string list")
    timeout_seconds = agent_runtime.get("timeout_seconds")
    if timeout_seconds is not None and (
        not isinstance(timeout_seconds, int) or timeout_seconds <= 0
    ):
        raise WorkspaceError("agent runtime timeout must be a positive integer")
    protected_paths = [
        Path(path) for path in agent_runtime.get("protected_paths", []) if str(path)
    ]
    _make_readonly_tree(attempt.workspace_path / "readonly_inputs")
    _make_readonly_tree(attempt.workspace_path / "readonly_source")
    sandbox_exec = shutil.which("sandbox-exec")
    if sandbox_exec is None:
        sandbox_profile: Path | None = None
        sandbox_mode = "portable_python"
        sandboxed_command = _agent_portable_sandbox_command(
            attempt.workspace_path, command, protected_paths
        )
    else:
        sandbox_profile = _write_agent_sandbox_profile(
            attempt.workspace_path, command, protected_paths
        )
        sandbox_mode = "sandbox_exec"
        sandboxed_command = [
            sandbox_exec,
            "-f",
            str(sandbox_profile),
            *command,
        ]
    started_at = utcnow()
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONUNBUFFERED": "1",
        "DARJEELING_ATTEMPT_ID": attempt.attempt_id,
        "DARJEELING_BRIEF_PATH": str(brief_path),
    }
    return {
        "command": command,
        "sandboxed_command": sandboxed_command,
        "sandbox_profile": sandbox_profile,
        "sandbox_mode": sandbox_mode,
        "started_at": started_at,
        "env": env,
        "timeout_seconds": timeout_seconds,
        "session_record": session_record,
        "log_path": attempt.workspace_path / "journal" / "agent.log",
    }


def _write_completed_session_record(
    *,
    attempt_id: str,
    command: list[str],
    sandbox_profile: Path | None,
    sandbox_mode: str,
    started_at: Any,
    session_record: Path,
    log_path: Path,
    status: str,
    returncode: int | None,
    timeout_seconds: int | None = None,
    stop_reason: str | None = None,
) -> None:
    record: dict[str, Any] = {
        "attempt_id": attempt_id,
        "command": command,
        "sandbox_profile": str(sandbox_profile) if sandbox_profile else None,
        "sandbox_mode": sandbox_mode,
        "status": status,
        "returncode": returncode,
        "started_at": started_at,
        "completed_at": utcnow(),
        "log_path": log_path,
    }
    if timeout_seconds is not None:
        record["timeout_seconds"] = timeout_seconds
    if stop_reason is not None:
        record["stop_reason"] = stop_reason
    write_json(session_record, record)


def launch_target_adaptation_agent(
    attempt: AgentAttempt,
    brief_path: Path,
    agent_runtime: dict[str, Any],
) -> AgentSessionHandle:
    launch = _prepare_agent_launch(attempt, brief_path, agent_runtime)
    command = launch["command"]
    sandbox_profile = launch["sandbox_profile"]
    sandbox_mode = launch["sandbox_mode"]
    started_at = launch["started_at"]
    session_record = launch["session_record"]
    log_path = launch["log_path"]
    timeout_seconds = launch["timeout_seconds"]
    try:
        completed = subprocess.run(
            launch["sandboxed_command"],
            cwd=attempt.workspace_path,
            env=launch["env"],
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = (
            exc.stdout.decode("utf-8", errors="replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        stderr = (
            exc.stderr.decode("utf-8", errors="replace")
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")
        )
        log_path.write_text(
            f"$ {' '.join(command)}\n\n[stdout]\n{stdout}\n[stderr]\n{stderr}\n",
            encoding="utf-8",
        )
        _write_completed_session_record(
            attempt_id=attempt.attempt_id,
            command=command,
            sandbox_profile=sandbox_profile,
            sandbox_mode=sandbox_mode,
            started_at=started_at,
            session_record=session_record,
            log_path=log_path,
            status="timed_out",
            returncode=None,
            timeout_seconds=timeout_seconds,
        )
        raise WorkspaceError("agent runtime command timed out") from exc
    log_path.write_text(
        f"$ {' '.join(command)}\n\n[stdout]\n{completed.stdout}\n[stderr]\n"
        f"{completed.stderr}\n",
        encoding="utf-8",
    )
    status = "completed" if completed.returncode == 0 else "failed"
    _write_completed_session_record(
        attempt_id=attempt.attempt_id,
        command=command,
        sandbox_profile=sandbox_profile,
        sandbox_mode=sandbox_mode,
        started_at=started_at,
        session_record=session_record,
        log_path=log_path,
        status=status,
        returncode=completed.returncode,
        timeout_seconds=timeout_seconds,
    )
    if completed.returncode != 0:
        raise WorkspaceError("agent runtime command failed")
    return AgentSessionHandle(
        attempt_id=attempt.attempt_id,
        status="completed",
        command=command,
        started_at=started_at,
        log_path=log_path,
        session_record_path=session_record,
        sandbox_mode=sandbox_mode,
    )


def launch_target_adaptation_agent_async(
    attempt: AgentAttempt,
    brief_path: Path,
    agent_runtime: dict[str, Any],
) -> AgentSessionHandle:
    launch = _prepare_agent_launch(attempt, brief_path, agent_runtime)
    command = launch["command"]
    sandbox_profile = launch["sandbox_profile"]
    sandbox_mode = launch["sandbox_mode"]
    started_at = launch["started_at"]
    session_record = launch["session_record"]
    log_path = launch["log_path"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"$ {' '.join(command)}\n\n")
        log_file.flush()
        try:
            process = subprocess.Popen(
                launch["sandboxed_command"],
                cwd=attempt.workspace_path,
                env=launch["env"],
                text=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as exc:
            raise WorkspaceError("agent runtime command could not be started") from exc
    _LIVE_AGENT_PROCESSES[attempt.attempt_id] = process
    write_json(
        session_record,
        {
            "attempt_id": attempt.attempt_id,
            "command": command,
            "sandbox_profile": str(sandbox_profile) if sandbox_profile else None,
            "sandbox_mode": sandbox_mode,
            "status": "running",
            "pid": process.pid,
            "started_at": started_at,
            "log_path": log_path,
            "timeout_seconds": launch["timeout_seconds"],
        },
    )
    return AgentSessionHandle(
        attempt_id=attempt.attempt_id,
        status="running",
        command=command,
        pid=process.pid,
        started_at=started_at,
        log_path=log_path,
        session_record_path=session_record,
        sandbox_mode=sandbox_mode,
    )


def poll_agent_session(handle: AgentSessionHandle) -> AgentSessionHandle:
    process = _LIVE_AGENT_PROCESSES.get(handle.attempt_id)
    if process is None:
        if handle.session_record_path is not None and handle.session_record_path.exists():
            record = read_json(handle.session_record_path)
            return replace(
                handle,
                status=record.get("status", handle.status),
                pid=record.get("pid", handle.pid),
            )
        return handle
    returncode = process.poll()
    if returncode is None:
        return replace(handle, status="running", pid=process.pid)
    _LIVE_AGENT_PROCESSES.pop(handle.attempt_id, None)
    status = "completed" if returncode == 0 else "failed"
    if handle.session_record_path is not None:
        record = (
            read_json(handle.session_record_path)
            if handle.session_record_path.exists()
            else {}
        )
        _write_completed_session_record(
            attempt_id=handle.attempt_id,
            command=list(record.get("command", handle.command)),
            sandbox_profile=Path(record["sandbox_profile"])
            if record.get("sandbox_profile")
            else None,
            sandbox_mode=record.get("sandbox_mode") or handle.sandbox_mode or "unknown",
            started_at=record.get("started_at") or handle.started_at or utcnow(),
            session_record=handle.session_record_path,
            log_path=Path(record.get("log_path") or handle.log_path or ""),
            status=status,
            returncode=returncode,
            timeout_seconds=record.get("timeout_seconds"),
        )
    return replace(handle, status=status, pid=process.pid)


def stop_agent_session(
    handle: AgentSessionHandle,
    reason: str = "stopped",
    timeout_seconds: float = 5.0,
) -> AgentSessionHandle:
    process = _LIVE_AGENT_PROCESSES.get(handle.attempt_id)
    status = "timed_out" if reason == "time_limit" else "stopped"
    if process is not None and process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except Exception:
            process.terminate()
        try:
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except Exception:
                process.kill()
            process.wait()
    returncode = process.returncode if process is not None else None
    _LIVE_AGENT_PROCESSES.pop(handle.attempt_id, None)
    if handle.session_record_path is not None:
        record = (
            read_json(handle.session_record_path)
            if handle.session_record_path.exists()
            else {}
        )
        log_path = Path(record.get("log_path") or handle.log_path or "")
        _write_completed_session_record(
            attempt_id=handle.attempt_id,
            command=list(record.get("command", handle.command)),
            sandbox_profile=Path(record["sandbox_profile"])
            if record.get("sandbox_profile")
            else None,
            sandbox_mode=record.get("sandbox_mode") or handle.sandbox_mode or "unknown",
            started_at=record.get("started_at") or handle.started_at or utcnow(),
            session_record=handle.session_record_path,
            log_path=log_path,
            status=status,
            returncode=returncode,
            timeout_seconds=record.get("timeout_seconds"),
            stop_reason=reason,
        )
    return replace(handle, status=status, pid=process.pid if process is not None else handle.pid)


def run_compile_loop(
    compile_run: CompileRun,
    attempts: list[AgentAttempt],
    candidate_limit: int,
    time_limit: Any,
    validation_feedback: Callable[[CandidateSubmission], AgentFeedback] | None = None,
) -> dict[str, Any]:
    if validation_feedback is None:
        raise WorkspaceError("validation feedback callback is required")
    effective_limit = min(candidate_limit, compile_run.budget.max_candidates)
    if effective_limit <= 0:
        return {
            "compile_id": compile_run.compile_id,
            "submissions": [],
            "feedback_records": [],
            "time_limit": time_limit,
        }
    if isinstance(time_limit, (int, float)) and time_limit <= 0:
        return {
            "compile_id": compile_run.compile_id,
            "submissions": [],
            "feedback_records": [],
            "time_limit": time_limit,
        }
    deadline = time.monotonic() + time_limit if isinstance(time_limit, (int, float)) else None
    submissions: list[CandidateSubmission] = []
    feedback_records: list[FeedbackDeliveryRecord] = []
    for attempt in attempts:
        if attempt.compile_id != compile_run.compile_id:
            raise WorkspaceError("agent attempt does not match compile run")
        submissions_dir = attempt.workspace_path / "submissions"
        if not submissions_dir.exists():
            continue
        for submission_path in sorted(path for path in submissions_dir.iterdir() if path.is_dir()):
            if deadline is not None and time.monotonic() >= deadline:
                break
            submission = receive_candidate_submission(attempt, submission_path)
            submissions.append(submission)
            feedback = validation_feedback(submission)
            feedback_records.append(provide_validation_feedback(attempt, feedback))
            if len(submissions) >= effective_limit:
                break
            if deadline is not None and time.monotonic() >= deadline:
                break
        if len(submissions) >= effective_limit:
            break
        if deadline is not None and time.monotonic() >= deadline:
            break
    return {
        "compile_id": compile_run.compile_id,
        "submissions": submissions[:effective_limit],
        "feedback_records": feedback_records,
        "time_limit": time_limit,
    }


def receive_candidate_submission(
    attempt: AgentAttempt, submission_path: Path
) -> CandidateSubmission:
    if (attempt.workspace_path / "journal" / "closed.json").exists():
        raise WorkspaceError("candidate submissions are closed for this attempt")
    submissions_root = attempt.workspace_path / "submissions"
    try:
        submission_path.resolve().relative_to(attempt.workspace_path.resolve())
    except ValueError as exc:
        raise WorkspaceError("candidate submission path escapes attempt workspace") from exc
    try:
        submission_rel = submission_path.resolve().relative_to(submissions_root.resolve())
    except ValueError as exc:
        raise WorkspaceError("candidate submission must be under submissions/") from exc
    if len(submission_rel.parts) != 1:
        raise WorkspaceError("candidate submission path must be submissions/<candidate>")
    layers: list[str] = []
    artifacts = submission_path / "artifacts"
    for layer_dir, layer_name in [("l1", "L1"), ("l2", "L2"), ("l3", "L3")]:
        if (artifacts / layer_dir).exists():
            layers.append(layer_name)
    if not layers:
        raise WorkspaceError("candidate submission declares no artifact layers")
    return CandidateSubmission(
        submission_id=submission_path.name,
        compile_id=attempt.compile_id,
        attempt_id=attempt.attempt_id,
        submission_path=submission_path,
        workspace_commit=_workspace_tree_digest(attempt.workspace_path),
        submitted_at=utcnow(),
        declared_layers=layers,  # type: ignore[arg-type]
    )


def provide_validation_feedback(
    attempt: AgentAttempt, feedback: AgentFeedback
) -> FeedbackDeliveryRecord:
    if feedback.raw_rows_included is not False:
        raise WorkspaceError("validation feedback may not include raw rows")
    _assert_no_forbidden_report_material(
        {
            "summary": feedback.summary,
            "requirement_results": feedback.requirement_results,
            "metrics": feedback.metrics,
            "safe_slice_summaries": feedback.safe_slice_summaries,
            "latency_cost_summary": feedback.latency_cost_summary,
        }
    )
    path = attempt.workspace_path / "journal" / f"feedback-{feedback.candidate_id}.json"
    write_json(path, asdict(feedback))
    return FeedbackDeliveryRecord(attempt_id=attempt.attempt_id, path=path, delivered_at=utcnow())


def close_agent_attempt(attempt: AgentAttempt, reason: str) -> ClosedAgentAttempt:
    if reason not in {
        "budget_exhausted",
        "candidate_limit",
        "time_limit",
        "user_stop",
        "ready_for_test",
        "failure",
    }:
        raise WorkspaceError("unsupported agent attempt close reason")
    session_record = attempt.workspace_path / "journal" / "agent_session.json"
    if attempt.attempt_id in _LIVE_AGENT_PROCESSES:
        stop_agent_session(
            AgentSessionHandle(
                attempt_id=attempt.attempt_id,
                status="running",
                session_record_path=session_record,
            ),
            reason=reason,
        )
    final_commit = _baseline_content_digest(attempt.workspace_path)
    write_json(
        attempt.workspace_path / "journal" / "closed.json",
        {"reason": reason, "final_commit": final_commit},
    )
    return ClosedAgentAttempt(
        attempt_id=attempt.attempt_id,
        compile_id=attempt.compile_id,
        target_name=attempt.target_name,
        contract_hash=attempt.contract_hash,
        snapshot_id=attempt.snapshot_id,
        snapshot_digest=attempt.snapshot_digest,
        source_workspace_commit=attempt.source_workspace_commit,
        workspace_path=attempt.workspace_path,
        final_commit=final_commit,
        status="failed" if reason == "failure" else "closed",
    )


def record_agent_usage(attempt: AgentAttempt, usage_event: AgentUsageEvent) -> AgentUsageLedger:
    path = attempt.workspace_path / "journal" / "agent_usage.json"
    events: list[AgentUsageEvent] = []
    if path.exists():
        import json

        for raw in json.loads(path.read_text(encoding="utf-8")):
            events.append(AgentUsageEvent(**raw))
    events.append(usage_event)
    write_json(path, [asdict(event) for event in events])
    return AgentUsageLedger(events)


def write_agent_journal_entry(attempt: AgentAttempt, entry: JournalEntry) -> Path:
    path = (
        attempt.workspace_path
        / "journal"
        / f"{stable_hash((entry.created_at, entry.title))[:12]}.md"
    )
    path.write_text(f"# {entry.title}\n\n{entry.body}\n", encoding="utf-8")
    return path


def collect_target_change_proposals(attempt: AgentAttempt) -> list[TargetChangeProposal]:
    proposals: list[TargetChangeProposal] = []
    for path in sorted((attempt.workspace_path / "proposals").glob("*.md")):
        first = (
            path.read_text(encoding="utf-8").splitlines()[0]
            if path.read_text(encoding="utf-8").splitlines()
            else path.name
        )
        proposals.append(TargetChangeProposal(path=path, summary=first.lstrip("# ")))
    return proposals
