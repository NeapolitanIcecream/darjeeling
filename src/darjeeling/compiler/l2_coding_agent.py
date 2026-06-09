from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from difflib import unified_diff
from pathlib import Path
from typing import Any, Literal

from darjeeling.compiler.l4_context import assert_no_forbidden_context
from darjeeling.schemas import TeacherTrace
from darjeeling.settings import Settings

L2AgentMode = Literal["dry-run", "codex-cli"]


@dataclass(frozen=True)
class L2CodingAgentJobConfig:
    mode: L2AgentMode
    source_repo_dir: Path
    job_dir: Path
    codex_command: str = "codex"
    codex_model: str | None = "gpt-5.5"
    timeout_s: float = 7200.0
    sandbox: str = "workspace-write"
    approval_policy: str = "never"
    ignore_user_config: bool = True
    ignore_rules: bool = True
    ephemeral: bool = True
    dry_run_patch: Path | None = None
    run_validation: bool = True


@dataclass(frozen=True)
class L2CodingAgentJobResult:
    mode: L2AgentMode
    job_dir: Path
    workspace_repo_dir: Path
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


class L2CodingAgentError(RuntimeError):
    pass


class L2CodingAgentAdapter:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def run_l2_job(
        self,
        *,
        job_dir: Path,
        source_repo_dir: Path,
        teacher_train: list[TeacherTrace],
        hard_cases: list[TeacherTrace] | None = None,
        current_metrics: dict[str, Any] | None = None,
        objective: dict[str, Any] | None = None,
        dry_run_patch: Path | None = None,
        run_validation: bool | None = None,
    ) -> L2CodingAgentJobResult:
        if self.settings.l2_agent_mode == "disabled":
            raise L2CodingAgentError("L2 agent mode is disabled")
        config = L2CodingAgentJobConfig(
            mode=self.settings.l2_agent_mode,
            source_repo_dir=source_repo_dir,
            job_dir=job_dir,
            codex_command=self.settings.l2_agent_codex_command,
            codex_model=self.settings.l2_agent_model,
            timeout_s=self.settings.l2_agent_timeout_s,
            sandbox=self.settings.l2_agent_sandbox,
            approval_policy=self.settings.l2_agent_approval_policy,
            ignore_user_config=self.settings.l2_agent_ignore_user_config,
            ignore_rules=self.settings.l2_agent_ignore_rules,
            ephemeral=self.settings.l2_agent_ephemeral,
            dry_run_patch=dry_run_patch or self.settings.l2_agent_dry_run_patch,
            run_validation=(
                self.settings.l2_agent_run_validation
                if run_validation is None
                else run_validation
            ),
        )
        return run_l2_coding_agent_job(
            config=config,
            teacher_train=teacher_train,
            hard_cases=hard_cases or [],
            current_metrics=current_metrics or {},
            objective=objective or {},
        )


def run_l2_coding_agent_job(
    *,
    config: L2CodingAgentJobConfig,
    teacher_train: list[TeacherTrace],
    hard_cases: list[TeacherTrace],
    current_metrics: dict[str, Any],
    objective: dict[str, Any],
) -> L2CodingAgentJobResult:
    job_dir = config.job_dir
    context_dir = job_dir / "contexts"
    workspace_repo_dir = job_dir / "workspace" / "l2_research"
    candidate_dir = workspace_repo_dir / "candidate"
    prompt_path = job_dir / "prompt.md"
    transcript_path = job_dir / "transcript.jsonl"
    diff_path = job_dir / "diff.patch"
    commands_path = job_dir / "commands.jsonl"
    provenance_path = job_dir / "provenance.json"
    report_path = job_dir / "agent_report.md"

    job_dir.mkdir(parents=True, exist_ok=True)
    _write_context_files(
        context_dir=context_dir,
        teacher_train=teacher_train,
        hard_cases=hard_cases,
        current_metrics=current_metrics,
        objective=objective,
    )
    _prepare_l2_research_workspace(
        source_repo_dir=config.source_repo_dir,
        workspace_root=workspace_repo_dir,
        context_dir=context_dir,
    )
    prompt_path.write_text(
        _build_l2_agent_prompt(
            workspace_repo_dir=workspace_repo_dir,
        ),
        encoding="utf-8",
    )

    command_results: list[dict[str, Any]] = []
    if config.mode == "dry-run":
        result_code = _run_dry_run_job(
            config=config,
            workspace_repo_dir=workspace_repo_dir,
            transcript_path=transcript_path,
            report_path=report_path,
            command_results=command_results,
        )
    else:
        result_code = _run_codex_cli_job(
            config=config,
            workspace_repo_dir=workspace_repo_dir,
            prompt_path=prompt_path,
            transcript_path=transcript_path,
            report_path=report_path,
            command_results=command_results,
        )

    if config.run_validation:
        for command in _validation_commands():
            validation_result = _run_command(
                command,
                cwd=workspace_repo_dir,
                timeout_s=config.timeout_s,
            )
            command_results.append(validation_result)
            if validation_result["return_code"] != 0 and result_code == 0:
                result_code = int(validation_result["return_code"])

    diff_text = _l2_candidate_diff(config.source_repo_dir, candidate_dir)
    if (
        config.mode == "dry-run"
        and config.dry_run_patch is not None
        and result_code == 0
        and not diff_text.strip()
    ):
        result_code = 1
        command_results.append(
            {
                "command": ["l2-agent", "diff-check"],
                "cwd": str(workspace_repo_dir),
                "started_at": datetime.now(UTC).isoformat(),
                "return_code": 1,
                "stdout": "",
                "stderr": "dry-run patch produced no L2-owned diff",
            }
        )
    diff_path.write_text(diff_text, encoding="utf-8")
    _write_jsonl(commands_path, command_results)
    _write_l2_agent_provenance(
        provenance_path=provenance_path,
        config=config,
        workspace_repo_dir=workspace_repo_dir,
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

    return L2CodingAgentJobResult(
        mode=config.mode,
        job_dir=job_dir,
        workspace_repo_dir=workspace_repo_dir,
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
    context_families = _context_families_payload(
        teacher_train=teacher_train,
        hard_cases=hard_cases,
    )
    slot_error_summary = _slot_error_summary_payload(
        teacher_train=teacher_train,
        hard_cases=hard_cases,
    )
    payloads = {
        "teacher_train.jsonl": [trace.model_dump(mode="json") for trace in teacher_train],
        "hard_cases.jsonl": [trace.model_dump(mode="json") for trace in hard_cases],
        "l2_context_families.json": context_families,
        "slot_error_summary.json": slot_error_summary,
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


def _build_l2_agent_prompt(
    *,
    workspace_repo_dir: Path,
) -> str:
    del workspace_repo_dir
    return "\n".join(
        [
            "Read `program.md` in this workspace and complete one bounded L2 research iteration.",
        ]
    )


def _compact_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


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
        l2_outcomes = Counter(_trace_l2_outcome(trace) for trace in traces)
        chosen_layers = Counter(trace.chosen_layer for trace in traces)
        examples = sorted(
            traces,
            key=lambda trace: (
                trace.request_id not in hard_case_ids,
                trace.request_id,
            ),
        )[:8]
        families.append(
            {
                "family_id": _family_id(intent, slot_signature),
                "intent": intent,
                "slot_signature": list(slot_signature),
                "support": len(traces),
                "hard_case_support": sum(trace.request_id in hard_case_ids for trace in traces),
                "chosen_layer_counts": dict(sorted(chosen_layers.items())),
                "l2_outcome_counts": dict(sorted(l2_outcomes.items())),
                "examples": [
                    {
                        "request_id": trace.request_id,
                        "utterance": trace.utterance,
                        "teacher_frame": trace.teacher_frame.model_dump(mode="json")
                        if trace.teacher_frame is not None
                        else None,
                        "chosen_layer": trace.chosen_layer,
                        "l2_outcome": _trace_l2_outcome(trace),
                    }
                    for trace in examples
                ],
            }
        )

    payload = {
        "schema_version": "l2-context-families-v1",
        "teacher_train_count": len(teacher_train),
        "hard_case_count": len(hard_cases),
        "family_count": len(families),
        "families": families,
    }
    assert_no_forbidden_context(payload)
    return payload


def _slot_error_summary_payload(
    *,
    teacher_train: list[TeacherTrace],
    hard_cases: list[TeacherTrace],
) -> dict[str, Any]:
    hard_case_ids = {trace.request_id for trace in hard_cases}
    traces_by_id = {trace.request_id: trace for trace in teacher_train}
    traces_by_id.update({trace.request_id: trace for trace in hard_cases})
    missing_slot_counts: Counter[str] = Counter()
    extra_slot_counts: Counter[str] = Counter()
    changed_slot_counts: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    wrong_accept_count = 0
    slot_mismatch_count = 0

    for trace in sorted(
        traces_by_id.values(),
        key=lambda item: (item.request_id not in hard_case_ids, item.request_id),
    ):
        if trace.teacher_frame is None:
            continue
        l2_result = next((result for result in trace.layer_results if result.layer == "L2"), None)
        if l2_result is None or not l2_result.accepted or l2_result.frame is None:
            continue
        if l2_result.frame == trace.teacher_frame:
            continue
        wrong_accept_count += 1
        if l2_result.frame.intent != trace.teacher_frame.intent:
            mismatch_type = "intent_mismatch"
        else:
            mismatch_type = "slot_mismatch"
            slot_mismatch_count += 1
        slot_delta = _slot_delta(
            expected_slots=trace.teacher_frame.slots,
            predicted_slots=l2_result.frame.slots,
        )
        missing_slot_counts.update(slot_delta["missing_slots"])
        extra_slot_counts.update(slot_delta["extra_slots"])
        changed_slot_counts.update(slot_delta["changed_slots"])
        if len(examples) < 12:
            examples.append(
                {
                    "request_id": trace.request_id,
                    "utterance": trace.utterance,
                    "is_hard_case": trace.request_id in hard_case_ids,
                    "mismatch_type": mismatch_type,
                    "teacher_frame": trace.teacher_frame.model_dump(mode="json"),
                    "l2_frame": l2_result.frame.model_dump(mode="json"),
                    "missing_slots": slot_delta["missing_slots"],
                    "extra_slots": slot_delta["extra_slots"],
                    "changed_slots": slot_delta["changed_slots"],
                    "l2_confidence": l2_result.confidence,
                    "l2_metadata": _l2_metadata_summary(l2_result.metadata),
                }
            )

    payload = {
        "schema_version": "l2-slot-error-summary-v1",
        "teacher_train_count": len(teacher_train),
        "hard_case_count": len(hard_cases),
        "l2_wrong_accept_count": wrong_accept_count,
        "l2_intent_correct_slot_mismatch_count": slot_mismatch_count,
        "missing_slot_counts": dict(sorted(missing_slot_counts.items())),
        "extra_slot_counts": dict(sorted(extra_slot_counts.items())),
        "changed_slot_counts": dict(sorted(changed_slot_counts.items())),
        "examples": examples,
    }
    assert_no_forbidden_context(payload)
    return payload


def _slot_delta(
    *,
    expected_slots: dict[str, str],
    predicted_slots: dict[str, str],
) -> dict[str, list[str]]:
    expected_names = set(expected_slots)
    predicted_names = set(predicted_slots)
    shared_names = expected_names & predicted_names
    return {
        "missing_slots": sorted(expected_names - predicted_names),
        "extra_slots": sorted(predicted_names - expected_names),
        "changed_slots": sorted(
            name for name in shared_names if expected_slots[name] != predicted_slots[name]
        ),
    }


def _l2_metadata_summary(metadata: dict[str, Any]) -> dict[str, Any]:
    allowed_keys = [
        "accept_threshold",
        "entropy",
        "frame_source",
        "frame_source_config",
        "guard_probability",
        "intent_model",
        "intent_support_margin",
        "margin",
        "predicted_has_slots",
        "predicted_intent_frame_accuracy",
        "predicted_intent_intent_accuracy",
        "predicted_signature_frame_accuracy",
        "predicted_signature_support",
        "predicted_slot_count",
        "retrieval_intent_matches_student",
        "retrieval_margin",
        "retrieval_similarity",
        "slot_avg_probability",
        "slot_invalid_bio",
        "slot_model",
        "top1_probability",
    ]
    return {key: metadata[key] for key in allowed_keys if key in metadata}


def _family_id(intent: str, slot_signature: tuple[str, ...]) -> str:
    slots = ",".join(slot_signature) if slot_signature else "no_slots"
    return f"{intent}|{slots}"


def _trace_l2_outcome(trace: TeacherTrace) -> str:
    l2_result = next((result for result in trace.layer_results if result.layer == "L2"), None)
    if l2_result is None:
        return "not_run"
    if not l2_result.accepted:
        return "abstain"
    if l2_result.frame == trace.teacher_frame:
        return "correct_accept"
    return "wrong_accept"


def _run_dry_run_job(
    *,
    config: L2CodingAgentJobConfig,
    workspace_repo_dir: Path,
    transcript_path: Path,
    report_path: Path,
    command_results: list[dict[str, Any]],
) -> int:
    if config.dry_run_patch is not None:
        patch_result = _run_command(
            ["git", "apply", str(config.dry_run_patch)],
            cwd=workspace_repo_dir,
            timeout_s=config.timeout_s,
            extra_env={
                "GIT_CEILING_DIRECTORIES": str(workspace_repo_dir.parent.resolve()),
            },
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
        "Dry-run L2 coding-agent job completed.\n",
        encoding="utf-8",
    )
    return return_code


def _run_codex_cli_job(
    *,
    config: L2CodingAgentJobConfig,
    workspace_repo_dir: Path,
    prompt_path: Path,
    transcript_path: Path,
    report_path: Path,
    command_results: list[dict[str, Any]],
) -> int:
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
        ]
    )
    if config.ignore_user_config:
        command.append("--ignore-user-config")
    if config.ignore_rules:
        command.append("--ignore-rules")
    if config.ephemeral:
        command.append("--ephemeral")
    command.extend(
        [
            "--skip-git-repo-check",
            "--cd",
            str(workspace_repo_dir.resolve()),
            "--json",
            "-o",
            str(report_path.resolve()),
            "-",
        ]
    )
    result = _run_command(
        command,
        cwd=workspace_repo_dir,
        timeout_s=config.timeout_s,
        stdin=prompt_path.read_text(encoding="utf-8"),
    )
    transcript_path.write_text(str(result["stdout"]), encoding="utf-8")
    command_results.append(result)
    return int(result["return_code"])


def _validation_commands() -> list[list[str]]:
    return [
        [
            "uv",
            "run",
            "--project",
            "system/darjeeling",
            "python",
            "tools/run_checks.py",
        ],
    ]


def _run_command(
    command: list[str],
    *,
    cwd: Path,
    timeout_s: float,
    stdin: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    started_at = datetime.now(UTC).isoformat()
    try:
        env = None
        if extra_env is not None:
            env = {**os.environ, **extra_env}
        completed = subprocess.run(
            command,
            cwd=cwd,
            input=stdin,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
            env=env,
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
            "stdout": _output_text(exc.stdout),
            "stderr": _output_text(exc.stderr) or f"timed out after {timeout_s:.1f}s",
        }


def _output_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _write_l2_agent_provenance(
    *,
    provenance_path: Path,
    config: L2CodingAgentJobConfig,
    workspace_repo_dir: Path,
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
        "schema_version": "l2-agent-provenance-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "mode": config.mode,
        "succeeded": return_code == 0,
        "return_code": return_code,
        "codex_command": config.codex_command,
        "codex_model": config.codex_model,
        "sandbox": config.sandbox,
        "approval_policy": config.approval_policy,
        "ignore_user_config": config.ignore_user_config,
        "ignore_rules": config.ignore_rules,
        "ephemeral": config.ephemeral,
        "runtime_patch_applied": False,
        "runtime_patch_reason": "Python L2 code changes require outer process apply/restart",
        "paths": {
            "job_dir": str(config.job_dir),
            "workspace_repo_dir": str(workspace_repo_dir),
            "prompt": str(prompt_path),
            "context_dir": str(context_dir),
            "transcript": str(transcript_path),
            "diff": str(diff_path),
            "commands": str(commands_path),
            "report": str(report_path),
        },
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
        return {
            str(key): _compact_json_value(nested_value)
            for key, nested_value in sorted(value.items())[:20]
        }
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


def _prepare_l2_research_workspace(
    *,
    source_repo_dir: Path,
    workspace_root: Path,
    context_dir: Path,
) -> None:
    if workspace_root.exists():
        shutil.rmtree(workspace_root)
    workspace_root.mkdir(parents=True, exist_ok=True)

    system_repo_dir = workspace_root / "system" / "darjeeling"
    candidate_dir = workspace_root / "candidate"
    data_dir = workspace_root / "data"
    tools_dir = workspace_root / "tools"

    _copy_l2_system_workspace(source_repo_dir, system_repo_dir)
    _copy_l2_candidate_files(source_repo_dir, candidate_dir)
    _copy_context_files(context_dir, data_dir)
    tools_dir.mkdir(parents=True, exist_ok=True)
    (workspace_root / "program.md").write_text(_l2_research_program_text(), encoding="utf-8")
    (tools_dir / "sync_candidate.py").write_text(
        _sync_candidate_tool_text(),
        encoding="utf-8",
    )
    (tools_dir / "run_checks.py").write_text(_run_checks_tool_text(), encoding="utf-8")
    (tools_dir / "inspect_context.py").write_text(
        _inspect_context_tool_text(),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "l2-research-workspace-v1",
        "candidate_dir": "candidate",
        "system_repo_dir": "system/darjeeling",
        "data_dir": "data",
        "tools_dir": "tools",
        "candidate_paths": [path.as_posix() for path in sorted(_diffable_l2_files(candidate_dir))],
        "data_files": sorted(path.name for path in data_dir.iterdir() if path.is_file()),
        "commands": {
            "inspect_context": "uv run --project system/darjeeling python tools/inspect_context.py",
            "run_checks": "uv run --project system/darjeeling python tools/run_checks.py",
        },
    }
    (workspace_root / "workspace_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _copy_l2_system_workspace(source_repo_dir: Path, workspace_repo_dir: Path) -> None:
    if workspace_repo_dir.exists():
        shutil.rmtree(workspace_repo_dir)
    workspace_repo_dir.mkdir(parents=True, exist_ok=True)
    for rel_path in [
        Path("src"),
        Path("tests"),
        Path("docs/design/modules"),
        Path("pyproject.toml"),
        Path("uv.lock"),
        Path("README.md"),
    ]:
        source_path = source_repo_dir / rel_path
        target_path = workspace_repo_dir / rel_path
        if not source_path.exists():
            continue
        if source_path.is_dir():
            shutil.copytree(
                source_path,
                target_path,
                ignore=shutil.ignore_patterns(
                    "__pycache__",
                    ".pytest_cache",
                    ".ruff_cache",
                    "*.pyc",
                ),
            )
        else:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)


def _copy_l2_candidate_files(source_repo_dir: Path, candidate_dir: Path) -> None:
    if candidate_dir.exists():
        shutil.rmtree(candidate_dir)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    for rel_path in sorted(_diffable_l2_files(source_repo_dir)):
        source_path = source_repo_dir / rel_path
        target_path = candidate_dir / rel_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)


def _copy_context_files(context_dir: Path, workspace_context_dir: Path) -> None:
    if workspace_context_dir.exists():
        shutil.rmtree(workspace_context_dir)
    shutil.copytree(context_dir, workspace_context_dir)


def _l2_research_program_text() -> str:
    return "\n".join(
        [
            "# L2 research program",
            "",
            "You are the L4 coding-agent for one bounded L2 research iteration.",
            "",
            "Workspace layout:",
            "- `candidate/` is the only editable research code area.",
            "- `system/darjeeling/` is the fixed Darjeeling system copy used for checks.",
            "- `data/` contains teacher-visible traces, hard cases, metrics, and objective.",
            "- `tools/` contains local inspection and validation commands.",
            "",
            "Rules:",
            "- Edit only files under `candidate/`.",
            "- Do not edit `system/`, `data/`, or `tools/`.",
            "- Do not use network commands.",
            "- Do not read MASSIVE gold labels, promotion holdout, future stream data,",
            "  teacher cache internals, or files outside this workspace.",
            "- Produce one small L2-owned patch, then stop.",
            "- Prefer Optuna or local deterministic tools for numeric tuning; use your",
            "  reasoning for code, feature, model-family, calibration, or search-space changes.",
            "",
            "Useful commands:",
            "- `uv run --project system/darjeeling python tools/inspect_context.py`",
            "- `uv run --project system/darjeeling python tools/run_checks.py`",
            "",
            "Evaluation contract:",
            "- `tools/run_checks.py` overlays `candidate/` into `system/darjeeling/`,",
            "  then runs focused L2 pytest and ruff checks.",
            "- Passing checks does not self-certify the patch. The outer Darjeeling",
            "  compiler/replay loop decides whether a patch is useful.",
            "",
            "Final response:",
            "- Summarize candidate files changed.",
            "- List commands run and results.",
            "- State expected impact on L2 coverage, exact-match accuracy, latency, and risks.",
        ]
    )


def _sync_candidate_tool_text() -> str:
    return """from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "candidate"
SYSTEM = ROOT / "system" / "darjeeling"


def sync_candidate() -> int:
    copied = 0
    for source in sorted(CANDIDATE.rglob("*")):
        if not source.is_file():
            continue
        rel_path = source.relative_to(CANDIDATE)
        target = SYSTEM / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied += 1
    print(f"synced {copied} candidate files into {SYSTEM}")
    return copied


if __name__ == "__main__":
    sync_candidate()
"""


def _run_checks_tool_text() -> str:
    return """from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "candidate"
SYSTEM = ROOT / "system" / "darjeeling"


def sync_candidate() -> None:
    for source in sorted(CANDIDATE.rglob("*")):
        if not source.is_file():
            continue
        rel_path = source.relative_to(CANDIDATE)
        target = SYSTEM / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())


def run(command: list[str]) -> int:
    print("+ " + " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=SYSTEM, check=False)
    return int(completed.returncode)


def main() -> int:
    sync_candidate()
    test_command = [
        "uv",
        "run",
        "pytest",
        "tests/test_l2_student_training.py",
        "tests/test_l2_tuner.py",
        "tests/test_l2_guard.py",
        "-q",
    ]
    test_code = run(test_command)
    if test_code != 0:
        return test_code
    l2_tests = sorted(
        str(path.relative_to(SYSTEM)) for path in (SYSTEM / "tests").glob("test_l2_*.py")
    )
    ruff_command = [
        "uv",
        "run",
        "ruff",
        "check",
        "src/darjeeling/layers/l2_student.py",
        "src/darjeeling/compiler/l2_tuner.py",
        "src/darjeeling/compiler/guard_optimizer.py",
        *l2_tests,
    ]
    return run(ruff_command)


if __name__ == "__main__":
    sys.exit(main())
"""


def _inspect_context_tool_text() -> str:
    return """from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def load_json(name: str):
    path = DATA / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    print("data files:")
    for path in sorted(DATA.iterdir()):
        if path.is_file():
            print(f"- {path.name}: {path.stat().st_size} bytes")
    objective = load_json("objective.json")
    if objective is not None:
        print("\\nobjective:")
        print(json.dumps(objective, indent=2, sort_keys=True))
    slot_summary = load_json("slot_error_summary.json")
    if slot_summary is not None:
        print("\\nslot error summary:")
        for key in [
            "l2_wrong_accept_count",
            "l2_intent_correct_slot_mismatch_count",
            "missing_slot_counts",
            "extra_slot_counts",
            "changed_slot_counts",
        ]:
            print(f"- {key}: {slot_summary.get(key)}")


if __name__ == "__main__":
    main()
"""


def _l2_candidate_diff(source_repo_dir: Path, candidate_dir: Path) -> str:
    diff_chunks: list[str] = []
    rel_paths = sorted(
        _diffable_l2_files(source_repo_dir) | _diffable_l2_files(candidate_dir)
    )
    for rel_path in rel_paths:
        source_path = source_repo_dir / rel_path
        workspace_path = candidate_dir / rel_path
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


def _diffable_l2_files(root: Path) -> set[Path]:
    paths: set[Path] = set()
    if not root.exists():
        return paths
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel_path = path.relative_to(root)
        if any(part in {"__pycache__", ".pytest_cache", ".ruff_cache"} for part in rel_path.parts):
            continue
        if _is_l2_diff_path(rel_path):
            paths.add(rel_path)
    return paths


def _is_l2_diff_path(rel_path: Path) -> bool:
    rel = rel_path.as_posix()
    return (
        rel == "src/darjeeling/layers/l2_student.py"
        or rel == "src/darjeeling/compiler/l2_tuner.py"
        or rel == "src/darjeeling/compiler/guard_optimizer.py"
        or rel == "src/darjeeling/compiler/l2_distiller.py"
        or rel.startswith("tests/test_l2_")
        or rel in {
            "docs/design/modules/l2_student.md",
            "docs/design/modules/l4_layer.md",
            "docs/design/modules/compiler.md",
            "docs/design/modules/settings.md",
            "docs/design/modules/cli.md",
        }
    )


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
            "# L2 constraints",
            "",
            "- L2 runtime is Python/scikit-learn code and serialized artifacts.",
            "- Optimize for residual requests that survive L0/L1 routing.",
            "- Do not use MASSIVE evaluation labels, gold labels, promotion holdout,",
            "  or labels from requests outside the compiler-visible train window.",
            "- Do not modify outer replay, promotion logic, teacher cache, settings",
            "  loading, or experiment reporting outside the L2-owned files.",
            "- Use Optuna or local deterministic tools for hyperparameters.",
            "- Candidate output is not self-certified; outer replay decides promotion.",
        ]
    )


def _commands_text() -> str:
    return "\n".join(
        [
            "# Allowed commands",
            "",
            "- Inspect available train-visible context with "
            "`uv run --project system/darjeeling python tools/inspect_context.py`.",
            "- Validate the current `candidate/` overlay with "
            "`uv run --project system/darjeeling python tools/run_checks.py`.",
            "- Use local deterministic tooling or Optuna for numeric tuning when useful; "
            "keep generated reports inside this workspace.",
            "",
            "Do not run network commands from the L2 research workspace.",
        ]
    )
