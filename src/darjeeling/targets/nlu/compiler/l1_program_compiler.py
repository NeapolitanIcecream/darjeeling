from __future__ import annotations

import json
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from difflib import unified_diff
from pathlib import Path
from typing import Any, Literal

from darjeeling.targets.nlu.compiler.focus_tasks import (
    build_focus_tasks,
    focus_task_document_with_fields,
)
from darjeeling.targets.nlu.compiler.l4_context import assert_no_forbidden_context
from darjeeling.targets.nlu.layers.l1_program_bank import ProgramRule
from darjeeling.targets.nlu.schemas import TeacherTrace
from darjeeling.targets.nlu.settings import Settings


def validate_l1_candidates(candidates: list[dict]) -> list[ProgramRule]:
    return [ProgramRule.model_validate(candidate) for candidate in candidates]


L1AgentMode = Literal["dry-run", "codex-cli", "agent-session"]


@dataclass(frozen=True)
class L1CodingAgentJobConfig:
    mode: L1AgentMode
    source_crate_dir: Path
    job_dir: Path
    codex_command: str = "codex"
    codex_model: str | None = None
    timeout_s: float = 900.0
    sandbox: str = "workspace-write"
    approval_policy: str = "never"
    dry_run_patch: Path | None = None
    run_validation: bool = True


@dataclass(frozen=True)
class L1CodingAgentJobResult:
    mode: L1AgentMode
    job_dir: Path
    workspace_crate_dir: Path
    prompt_path: Path
    context_dir: Path
    transcript_path: Path
    diff_path: Path
    commands_path: Path
    provenance_path: Path
    report_path: Path
    return_code: int
    succeeded: bool
    command_results: list[dict[str, Any]] = field(default_factory=list)


class L1CodingAgentError(RuntimeError):
    pass


class L4CodingAgentAdapter:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def run_l1_job(
        self,
        *,
        job_dir: Path,
        source_crate_dir: Path,
        teacher_train: list[TeacherTrace],
        hard_cases: list[TeacherTrace] | None = None,
        current_metrics: dict[str, Any] | None = None,
        objective: dict[str, Any] | None = None,
        dry_run_patch: Path | None = None,
        run_validation: bool = True,
    ) -> L1CodingAgentJobResult:
        if self.settings.l1_agent_mode == "disabled":
            raise L1CodingAgentError("L1 agent mode is disabled")
        config = L1CodingAgentJobConfig(
            mode=self.settings.l1_agent_mode,
            source_crate_dir=source_crate_dir,
            job_dir=job_dir,
            codex_command=self.settings.l1_agent_codex_command,
            codex_model=self.settings.l1_agent_model,
            timeout_s=self.settings.l1_agent_timeout_s,
            sandbox=self.settings.l1_agent_sandbox,
            approval_policy=self.settings.l1_agent_approval_policy,
            dry_run_patch=dry_run_patch or self.settings.l1_agent_dry_run_patch,
            run_validation=run_validation,
        )
        return run_l1_coding_agent_job(
            config=config,
            teacher_train=teacher_train,
            hard_cases=hard_cases or [],
            current_metrics=current_metrics or {},
            objective=objective or {},
        )


def run_l1_coding_agent_job(
    *,
    config: L1CodingAgentJobConfig,
    teacher_train: list[TeacherTrace],
    hard_cases: list[TeacherTrace],
    current_metrics: dict[str, Any],
    objective: dict[str, Any],
) -> L1CodingAgentJobResult:
    job_dir = config.job_dir
    workspace_root = job_dir / "workspace"
    context_dir = workspace_root / "contexts"
    workspace_crate_dir = workspace_root / "l1_programbank"
    prompt_path = workspace_root / "program.md"
    transcript_path = job_dir / "transcript.jsonl"
    diff_path = job_dir / "diff.patch"
    commands_path = job_dir / "commands.jsonl"
    provenance_path = job_dir / "provenance.json"
    report_path = job_dir / "agent_report.md"

    job_dir.mkdir(parents=True, exist_ok=True)
    if workspace_root.exists():
        shutil.rmtree(workspace_root)
    workspace_root.mkdir(parents=True, exist_ok=True)
    _copy_crate(config.source_crate_dir, workspace_crate_dir)
    _write_context_files(
        context_dir=context_dir,
        teacher_train=teacher_train,
        hard_cases=hard_cases,
        current_metrics=current_metrics,
        objective=objective,
    )
    legacy_context_dir = job_dir / "contexts"
    if legacy_context_dir.exists():
        shutil.rmtree(legacy_context_dir)
    shutil.copytree(context_dir, legacy_context_dir)
    prompt_path.write_text(
        _build_l1_agent_prompt(context_dir=context_dir, workspace_crate_dir=workspace_crate_dir),
        encoding="utf-8",
    )
    _write_l1_workspace_manifest(
        mode=config.mode,
        workspace_root=workspace_root,
        workspace_crate_dir=workspace_crate_dir,
        context_dir=context_dir,
        program_path=prompt_path,
    )

    command_results: list[dict[str, Any]] = []
    if config.mode == "dry-run":
        result_code = _run_dry_run_job(
            config=config,
            workspace_crate_dir=workspace_crate_dir,
            transcript_path=transcript_path,
            report_path=report_path,
            command_results=command_results,
        )
    else:
        protected_snapshot = _protected_l1_workspace_snapshot(workspace_root)
        result_code = _run_codex_cli_job(
            config=config,
            workspace_root=workspace_root,
            workspace_crate_dir=workspace_crate_dir,
            prompt_path=prompt_path,
            transcript_path=transcript_path,
            report_path=report_path,
            command_results=command_results,
        )
        if config.mode == "agent-session":
            scope_report = _l1_workspace_scope_violation_report(
                workspace_root=workspace_root,
                before=protected_snapshot,
            )
            if scope_report is not None:
                command_results.append(
                    _l1_workspace_scope_violation_command_result(
                        workspace_root=workspace_root,
                        report=scope_report,
                    ),
                )
                result_code = 1

    if config.run_validation and result_code == 0:
        cargo_result = _run_command(
            ["cargo", "test"],
            cwd=workspace_crate_dir,
            timeout_s=config.timeout_s,
        )
        command_results.append(cargo_result)
        if cargo_result["return_code"] != 0 and result_code == 0:
            result_code = int(cargo_result["return_code"])

    diff_text = _crate_diff(config.source_crate_dir, workspace_crate_dir)
    diff_path.write_text(diff_text, encoding="utf-8")
    _write_jsonl(commands_path, command_results)
    _write_l1_agent_provenance(
        provenance_path=provenance_path,
        config=config,
        workspace_root=workspace_root,
        workspace_crate_dir=workspace_crate_dir,
        prompt_path=prompt_path,
        context_dir=context_dir,
        transcript_path=transcript_path,
        diff_path=diff_path,
        commands_path=commands_path,
        report_path=report_path,
        return_code=result_code,
        command_results=command_results,
        diff_text=diff_text,
    )

    return L1CodingAgentJobResult(
        mode=config.mode,
        job_dir=job_dir,
        workspace_crate_dir=workspace_crate_dir,
        prompt_path=prompt_path,
        context_dir=context_dir,
        transcript_path=transcript_path,
        diff_path=diff_path,
        commands_path=commands_path,
        provenance_path=provenance_path,
        report_path=report_path,
        return_code=result_code,
        succeeded=result_code == 0,
        command_results=command_results,
    )


def _write_context_files(
    *,
    context_dir: Path,
    teacher_train: list[TeacherTrace],
    hard_cases: list[TeacherTrace],
    current_metrics: dict[str, Any],
    objective: dict[str, Any],
) -> None:
    context_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "teacher_train.jsonl": [trace.model_dump(mode="json") for trace in teacher_train],
        "hard_cases.jsonl": [trace.model_dump(mode="json") for trace in hard_cases],
        "context_families.json": _context_families_payload(
            teacher_train=teacher_train,
            hard_cases=hard_cases,
        ),
        "focus_tasks.json": focus_task_document_with_fields(
            build_focus_tasks([*hard_cases, *teacher_train]),
            [*hard_cases, *teacher_train],
        ),
        "current_metrics.json": current_metrics,
        "objective.json": objective,
        "constraints.md": _constraints_text(),
        "commands.md": _commands_text(),
    }
    assert_no_forbidden_context(payloads)
    for name, payload in payloads.items():
        path = context_dir / name
        if name.endswith(".jsonl"):
            path.write_text(
                "".join(json.dumps(item, sort_keys=True) + "\n" for item in payload),
                encoding="utf-8",
            )
        elif name.endswith(".json"):
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        else:
            path.write_text(str(payload), encoding="utf-8")


def _build_l1_agent_prompt(*, context_dir: Path, workspace_crate_dir: Path) -> str:
    return "\n".join(
        [
            "# Darjeeling L1 Rust ProgramBank evolution job",
            "",
            "You are running as the L4 coding-agent compiler mode for L1.",
            "Edit only the Rust ProgramBank workspace provided for this job.",
            "Use the teacher-visible context files, objective, constraints, and command guide.",
            "Start from `focus_tasks.json`: it ranks local work by misses, wrong",
            "accepts, audit disagreement, and L4 pressure. Use `context_families.json`",
            "and raw JSONL files only for examples and boundary checks.",
            "Prefer native Rust code paths: if/else trees, tight loops, tables,",
            "small state machines, or validators.",
            "Default to abstain when uncertain. Do not change outer evaluator or promotion logic.",
            "",
            f"Workspace crate: `{workspace_crate_dir}`",
            f"Context directory: `{context_dir}`",
            "",
            "Required final response:",
            "- Summarize files changed.",
            "- Summarize commands run and results.",
            "- Identify any known risk or failed check.",
        ]
    )


def _write_l1_workspace_manifest(
    *,
    mode: L1AgentMode,
    workspace_root: Path,
    workspace_crate_dir: Path,
    context_dir: Path,
    program_path: Path,
) -> None:
    manifest = {
        "schema_version": "l1-agent-workspace-v1",
        "editable_roots": [workspace_crate_dir.relative_to(workspace_root).as_posix() + "/"],
        "scratch_roots": ["runs/"],
        "protected_roots": [
            context_dir.relative_to(workspace_root).as_posix() + "/",
            program_path.relative_to(workspace_root).as_posix(),
            "workspace_manifest.json",
        ],
        "tools": {
            "compile": "cargo check --manifest-path l1_programbank/Cargo.toml",
            "unit_test": "cargo test --manifest-path l1_programbank/Cargo.toml",
            "bench": (
                "edge-mvp-nlu l1 bench "
                "--crate-dir l1_programbank --out runs/l1_benchmark.json"
            ),
            "replay": "edge-mvp run ...",
        },
        "agent_session_policy": _l1_agent_session_policy(mode),
    }
    (workspace_root / "runs").mkdir(exist_ok=True)
    (workspace_root / "workspace_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _context_families_payload(
    *,
    teacher_train: list[TeacherTrace],
    hard_cases: list[TeacherTrace],
) -> dict[str, Any]:
    hard_case_ids = {trace.request_id for trace in hard_cases}
    grouped: dict[tuple[str, tuple[str, ...]], list[TeacherTrace]] = defaultdict(list)
    for trace in teacher_train:
        if trace.teacher_frame is None:
            continue
        signature = tuple(sorted(trace.teacher_frame.slots))
        grouped[(trace.teacher_frame.intent, signature)].append(trace)

    families = []
    for (intent, slot_signature), traces in sorted(
        grouped.items(),
        key=lambda item: (
            -sum(trace.request_id in hard_case_ids for trace in item[1]),
            -len(item[1]),
            item[0][0],
            item[0][1],
        ),
    ):
        l1_outcomes = Counter(_trace_l1_outcome(trace) for trace in traces)
        chosen_layers = Counter(trace.chosen_layer for trace in traces)
        token_counts = _content_token_counts(trace.utterance for trace in traces)
        examples = sorted(
            traces,
            key=lambda trace: (
                trace.request_id not in hard_case_ids,
                trace.request_id,
            ),
        )[:6]
        families.append(
            {
                "family_id": _family_id(intent, slot_signature),
                "intent": intent,
                "slot_signature": list(slot_signature),
                "support": len(traces),
                "hard_case_support": sum(trace.request_id in hard_case_ids for trace in traces),
                "chosen_layer_counts": dict(sorted(chosen_layers.items())),
                "l1_outcome_counts": dict(sorted(l1_outcomes.items())),
                "common_tokens": [token for token, _count in token_counts.most_common(8)],
                "examples": [
                    {
                        "request_id": trace.request_id,
                        "utterance": trace.utterance,
                        "teacher_frame": trace.teacher_frame.model_dump(mode="json")
                        if trace.teacher_frame is not None
                        else None,
                        "chosen_layer": trace.chosen_layer,
                        "l1_outcome": _trace_l1_outcome(trace),
                    }
                    for trace in examples
                ],
            }
        )

    payload = {
        "schema_version": "l1-context-families-v1",
        "teacher_train_count": len(teacher_train),
        "hard_case_count": len(hard_cases),
        "family_count": len(families),
        "families": families,
    }
    assert_no_forbidden_context(payload)
    return payload


def _family_id(intent: str, slot_signature: tuple[str, ...]) -> str:
    slots = ",".join(slot_signature) if slot_signature else "no_slots"
    return f"{intent}|{slots}"


def _trace_l1_outcome(trace: TeacherTrace) -> str:
    l1_result = next((result for result in trace.layer_results if result.layer == "L1"), None)
    if l1_result is None:
        return "not_run"
    if not l1_result.accepted:
        return "abstain"
    if l1_result.frame == trace.teacher_frame:
        return "correct_accept"
    return "wrong_accept"


def _content_token_counts(utterances: Any) -> Counter[str]:
    stop_words = {
        "a",
        "an",
        "and",
        "are",
        "for",
        "in",
        "is",
        "me",
        "my",
        "of",
        "on",
        "please",
        "the",
        "to",
        "what",
        "you",
    }
    counts: Counter[str] = Counter()
    for utterance in utterances:
        for token in str(utterance).lower().replace(".", " ").split():
            cleaned = "".join(char for char in token if char.isalnum() or char == "'")
            if len(cleaned) < 2 or cleaned in stop_words:
                continue
            counts[cleaned] += 1
    return counts


def _run_dry_run_job(
    *,
    config: L1CodingAgentJobConfig,
    workspace_crate_dir: Path,
    transcript_path: Path,
    report_path: Path,
    command_results: list[dict[str, Any]],
) -> int:
    if config.dry_run_patch is not None:
        patch_result = _run_command(
            ["git", "apply", str(config.dry_run_patch)],
            cwd=workspace_crate_dir,
            timeout_s=config.timeout_s,
        )
        command_results.append(patch_result)
        return_code = int(patch_result["return_code"])
    else:
        return_code = 0
    transcript_path.write_text(
        json.dumps(
            {
                "mode": "dry-run",
                "event": "completed",
                "patch": str(config.dry_run_patch) if config.dry_run_patch else None,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    report_path.write_text(
        "Dry-run L1 coding-agent job completed.\n",
        encoding="utf-8",
    )
    return return_code


def _run_codex_cli_job(
    *,
    config: L1CodingAgentJobConfig,
    workspace_root: Path,
    workspace_crate_dir: Path,
    prompt_path: Path,
    transcript_path: Path,
    report_path: Path,
    command_results: list[dict[str, Any]],
) -> int:
    cwd = workspace_root if config.mode == "agent-session" else workspace_crate_dir
    command = [config.codex_command]
    if config.codex_model:
        command.extend(["--model", config.codex_model])
    command.extend(
        [
            "--sandbox",
            config.sandbox,
            "-a",
            config.approval_policy,
            "exec",
            "--cd",
            str(cwd.resolve()),
            "--json",
            "-o",
            str(report_path.resolve()),
            "-",
        ]
    )
    prompt = (
        "\n".join(
            [
                "Read program.md and run one autonomous L1 ProgramBank evolution session.",
                "Edit only l1_programbank/. Use contexts/ and workspace_manifest.json",
                "for visible data, constraints, and tools. Stop when the visible",
                "objective is met, no safe progress remains, or budget/risk says to stop.",
            ]
        )
        if config.mode == "agent-session"
        else prompt_path.read_text(encoding="utf-8")
    )
    result = _run_command(
        command,
        cwd=cwd,
        timeout_s=config.timeout_s,
        stdin=prompt,
    )
    transcript_path.write_text(str(result["stdout"]), encoding="utf-8")
    command_results.append(result)
    return int(result["return_code"])


def _run_command(
    command: list[str],
    *,
    cwd: Path,
    timeout_s: float,
    stdin: str | None = None,
) -> dict[str, Any]:
    started_at = datetime.now(UTC).isoformat()
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            input=stdin,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
        return {
            "command": command,
            "cwd": str(cwd),
            "started_at": started_at,
            "return_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "cwd": str(cwd),
            "started_at": started_at,
            "return_code": 124,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or f"timed out after {timeout_s:.1f}s",
        }


def _write_l1_agent_provenance(
    *,
    provenance_path: Path,
    config: L1CodingAgentJobConfig,
    workspace_root: Path,
    workspace_crate_dir: Path,
    prompt_path: Path,
    context_dir: Path,
    transcript_path: Path,
    diff_path: Path,
    commands_path: Path,
    report_path: Path,
    return_code: int,
    command_results: list[dict[str, Any]],
    diff_text: str,
) -> None:
    payload = {
        "schema_version": "l1-agent-provenance-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "mode": config.mode,
        "succeeded": return_code == 0,
        "return_code": return_code,
        "codex_command": config.codex_command,
        "codex_model": config.codex_model,
        "sandbox": config.sandbox,
        "approval_policy": config.approval_policy,
        "paths": {
            "job_dir": str(config.job_dir),
            "workspace_root": str(workspace_root),
            "workspace_crate_dir": str(workspace_crate_dir),
            "prompt": str(prompt_path),
            "context_dir": str(context_dir),
            "transcript": str(transcript_path),
            "diff": str(diff_path),
            "commands": str(commands_path),
            "report": str(report_path),
        },
        "agent_session": _l1_agent_session_policy(config.mode),
        "workspace_scope_policy": _l1_workspace_scope_policy(),
        "transcript": _transcript_summary(transcript_path),
        "commands": [_command_summary(result) for result in command_results],
        "diff": _diff_summary(diff_text),
    }
    provenance_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _transcript_summary(transcript_path: Path) -> dict[str, Any]:
    if not transcript_path.exists():
        return {
            "line_count": 0,
            "json_event_count": 0,
            "parse_error_count": 0,
            "event_types": {},
            "sample_events": [],
        }

    line_count = 0
    parse_error_count = 0
    event_types: dict[str, int] = {}
    sample_events: list[dict[str, Any]] = []
    for line in transcript_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        line_count += 1
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            parse_error_count += 1
            continue
        if not isinstance(event, dict):
            parse_error_count += 1
            continue
        event_type = _event_type(event)
        event_types[event_type] = event_types.get(event_type, 0) + 1
        if len(sample_events) < 20:
            sample_events.append(_compact_json_event(event))
    return {
        "line_count": line_count,
        "json_event_count": sum(event_types.values()),
        "parse_error_count": parse_error_count,
        "event_types": dict(sorted(event_types.items())),
        "sample_events": sample_events,
    }


def _event_type(event: dict[str, Any]) -> str:
    for key in ["event", "type", "msg_type"]:
        value = event.get(key)
        if isinstance(value, str) and value:
            return value
    message = event.get("msg")
    if isinstance(message, dict):
        value = message.get("type")
        if isinstance(value, str) and value:
            return value
    return "unknown"


def _compact_json_event(event: dict[str, Any]) -> dict[str, Any]:
    return {key: _compact_json_value(value) for key, value in sorted(event.items())}


def _compact_json_value(value: Any) -> Any:
    if isinstance(value, str):
        return value if len(value) <= 240 else f"{value[:237]}..."
    if isinstance(value, int | float | bool) or value is None:
        return value
    if isinstance(value, dict):
        compact: dict[str, Any] = {}
        for key, nested_value in sorted(value.items())[:20]:
            compact[str(key)] = _compact_json_value(nested_value)
        return compact
    if isinstance(value, list):
        return [_compact_json_value(item) for item in value[:20]]
    return f"<{type(value).__name__}>"


def _command_summary(result: dict[str, Any]) -> dict[str, Any]:
    stdout = str(result.get("stdout", ""))
    stderr = str(result.get("stderr", ""))
    return {
        "command": result.get("command", []),
        "cwd": result.get("cwd", ""),
        "started_at": result.get("started_at", ""),
        "return_code": result.get("return_code"),
        "stdout_bytes": len(stdout.encode("utf-8")),
        "stderr_bytes": len(stderr.encode("utf-8")),
        "stdout_tail": stdout[-500:],
        "stderr_tail": stderr[-500:],
    }


def _diff_summary(diff_text: str) -> dict[str, Any]:
    changed_files: set[str] = set()
    additions = 0
    deletions = 0
    for line in diff_text.splitlines():
        if line.startswith("+++ b/"):
            changed_files.add(line.removeprefix("+++ b/"))
        elif line.startswith("--- a/"):
            changed_files.add(line.removeprefix("--- a/"))
        elif line.startswith("+") and not line.startswith("+++"):
            additions += 1
        elif line.startswith("-") and not line.startswith("---"):
            deletions += 1
    changed_files.discard("/dev/null")
    return {
        "changed_file_count": len(changed_files),
        "changed_files": sorted(changed_files),
        "additions": additions,
        "deletions": deletions,
    }


def _l1_agent_session_policy(mode: L1AgentMode | str) -> dict[str, Any]:
    return {
        "schema_version": "l1-agent-session-policy-v1",
        "applies_to_mode": mode == "agent-session",
        "session_scope": (
            "single long-running L4 agent session"
            if mode == "agent-session"
            else "legacy one-shot codex-cli or dry-run fixture"
        ),
        "internal_loop_control": (
            "agent_decides_edit_compile_test_bench_replay_stop"
            if mode == "agent-session"
            else None
        ),
        "editable_surface": "l1_programbank/",
        "tool_policy": "compile, unit test, bench, and replay are workspace tools",
        "adoption_authority": "outer replay and promotion policy",
    }


def _l1_workspace_scope_policy() -> dict[str, Any]:
    return {
        "schema_version": "l1-agent-workspace-scope-v1",
        "candidate_code_writable_roots": ["l1_programbank/"],
        "scratch_writable_roots": ["runs/"],
        "protected_roots": ["contexts/", "program.md", "workspace_manifest.json"],
        "ignored_generated_files": ["target/", "__pycache__/", ".pytest_cache/", "*.pyc"],
        "enforcement": "checked_after_agent_session_before_validation",
    }


def _protected_l1_workspace_snapshot(workspace_root: Path) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    if not workspace_root.exists():
        return snapshot
    for path in workspace_root.rglob("*"):
        if not path.is_file():
            continue
        rel_path = path.relative_to(workspace_root)
        if _is_l1_workspace_scope_ignored(rel_path) or _is_l1_workspace_scope_writable(rel_path):
            continue
        snapshot[rel_path.as_posix()] = _file_sha256(path)
    return snapshot


def _l1_workspace_scope_violation_report(
    *,
    workspace_root: Path,
    before: dict[str, str],
) -> dict[str, Any] | None:
    after = _protected_l1_workspace_snapshot(workspace_root)
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    modified = sorted(path for path in set(before) & set(after) if before[path] != after[path])
    if not added and not removed and not modified:
        return None
    return {
        "schema_version": "l1-agent-workspace-scope-violation-v1",
        "policy": _l1_workspace_scope_policy(),
        "added_protected_files": added,
        "removed_protected_files": removed,
        "modified_protected_files": modified,
        "message": (
            "L1 agent-session may change l1_programbank/ and write runs/ scratch "
            "outputs; protected workspace files changed before validation"
        ),
    }


def _l1_workspace_scope_violation_command_result(
    *,
    workspace_root: Path,
    report: dict[str, Any],
) -> dict[str, Any]:
    return {
        "command": ["l1-workspace-scope-check"],
        "cwd": str(workspace_root),
        "started_at": datetime.now(UTC).isoformat(),
        "return_code": 1,
        "stdout": "",
        "stderr": json.dumps(report, sort_keys=True),
        "workspace_scope_violation": report,
    }


def _is_l1_workspace_scope_writable(rel_path: Path) -> bool:
    parts = rel_path.parts
    return bool(parts and parts[0] in {"l1_programbank", "runs"})


def _is_l1_workspace_scope_ignored(rel_path: Path) -> bool:
    ignored_dirs = {"target", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
    if any(part in ignored_dirs for part in rel_path.parts):
        return True
    return rel_path.suffix in {".pyc", ".pyo"}


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_crate(source_crate_dir: Path, workspace_crate_dir: Path) -> None:
    if workspace_crate_dir.exists():
        shutil.rmtree(workspace_crate_dir)
    shutil.copytree(
        source_crate_dir,
        workspace_crate_dir,
        ignore=shutil.ignore_patterns("target", ".git"),
    )


def _crate_diff(source_crate_dir: Path, workspace_crate_dir: Path) -> str:
    diff_chunks: list[str] = []
    rel_paths = sorted(_diffable_files(source_crate_dir) | _diffable_files(workspace_crate_dir))
    for rel_path in rel_paths:
        source_path = source_crate_dir / rel_path
        workspace_path = workspace_crate_dir / rel_path
        source_text = _read_text_or_empty(source_path)
        workspace_text = _read_text_or_empty(workspace_path)
        if source_text == workspace_text:
            continue
        diff_chunks.extend(
            unified_diff(
                source_text.splitlines(keepends=True),
                workspace_text.splitlines(keepends=True),
                fromfile=f"a/{rel_path.as_posix()}",
                tofile=f"b/{rel_path.as_posix()}",
            )
        )
    return "".join(diff_chunks)


def _diffable_files(root: Path) -> set[Path]:
    paths: set[Path] = set()
    if not root.exists():
        return paths
    for path in root.rglob("*"):
        rel_path = path.relative_to(root)
        if "target" in rel_path.parts:
            continue
        if path.is_file():
            paths.add(rel_path)
    return paths


def _read_text_or_empty(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"<binary file: {path.name}>\n"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _constraints_text() -> str:
    return "\n".join(
        [
            "# L1 constraints",
            "",
            "- L1 runtime is Rust native code.",
            "- L1 must abstain when uncertain.",
            "- L1 may return a full frame or a precise intent-only/slot-only patch.",
            "- Do not use hidden evaluation labels or future labels.",
            "- Do not modify the outer evaluator, promotion logic, teacher cache,",
            "  or Python orchestration.",
            "- Candidate output is not self-certified; outer replay decides promotion.",
        ]
    )


def _commands_text() -> str:
    return "\n".join(
        [
            "# Allowed commands",
            "",
            "- `cargo test`",
            "- `cargo fmt --check`",
            "- local replay or benchmark commands documented in the workspace",
            "",
            "Do not run network commands from the L1 candidate workspace.",
        ]
    )
