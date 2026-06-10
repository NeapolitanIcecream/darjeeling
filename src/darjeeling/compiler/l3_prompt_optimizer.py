from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from darjeeling.layers.l3_local_slm import (
    L3LocalSLMLayer,
    L3PromptArtifact,
    LocalSLMBackend,
    LocalSLMConfig,
)
from darjeeling.layers.l4_cloud_llm import TaskSchema
from darjeeling.schemas import Frame, TeacherTrace, TraceRecord, traces_to_teacher_view

L3_PROMPT_EVOLUTION_MODE = "agent-session"
L3PromptEvolutionMode = Literal["agent-session"]

L3_PROMPT_PROPOSAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["system_prompt"],
    "properties": {
        "system_prompt": {"type": "string", "minLength": 1},
        "confidence_threshold": {"type": ["number", "null"], "minimum": 0.0, "maximum": 1.0},
        "few_shot_trace_ids": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 8,
        },
    },
}


@dataclass(frozen=True)
class L3GuardCalibrationResult:
    threshold: float
    sample_count: int
    labeled_count: int
    accepted_count: int
    wrong_accept_count: int
    coverage: float
    accepted_accuracy: float | None
    wrong_accept_rate: float
    max_wrong_accept_rate: float


@dataclass(frozen=True)
class L3PromptEvolutionConfig:
    job_dir: Path
    mode: L3PromptEvolutionMode = L3_PROMPT_EVOLUTION_MODE
    codex_command: str = "codex"
    codex_model: str | None = None
    timeout_s: float = 7200.0
    sandbox: str = "workspace-write"
    approval_policy: str = "never"
    max_agent_sessions: int = 1
    skip_replay: bool = False
    min_accepted_accuracy: float = 0.90
    max_wrong_accept_rate: float = 0.05
    prompt_version: str = "l3-prompt-v1"


def l3_prompt_artifact_from_proposal(
    proposal: dict[str, Any],
    *,
    traces: list[TeacherTrace],
    prompt_version: str,
    max_few_shots: int = 8,
) -> L3PromptArtifact:
    system_prompt = proposal.get("system_prompt")
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError("L3 prompt proposal requires a non-empty system_prompt")

    confidence_threshold = proposal.get("confidence_threshold")
    if confidence_threshold is not None:
        if not isinstance(confidence_threshold, int | float):
            raise ValueError("confidence_threshold must be a number or null")
        if not 0.0 <= float(confidence_threshold) <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        confidence_threshold = float(confidence_threshold)

    selected_ids = proposal.get("few_shot_trace_ids") or []
    if not isinstance(selected_ids, list) or not all(
        isinstance(item, str) for item in selected_ids
    ):
        raise ValueError("few_shot_trace_ids must be a list of trace ids")

    trace_by_id = {trace.request_id: trace for trace in traces if trace.teacher_frame is not None}
    examples = []
    seen: set[str] = set()
    for trace_id in selected_ids:
        if trace_id in seen:
            continue
        seen.add(trace_id)
        trace = trace_by_id.get(trace_id)
        if trace is None:
            raise ValueError(f"few-shot trace id is not teacher-visible: {trace_id}")
        examples.append(
            {
                "trace_id": trace.request_id,
                "utterance": trace.utterance,
                "frame": trace.teacher_frame.model_dump(mode="json"),
            }
        )
        if len(examples) >= max_few_shots:
            break

    return L3PromptArtifact(
        prompt_version=prompt_version,
        system_prompt=system_prompt.strip(),
        confidence_threshold=confidence_threshold,
        few_shot_examples=examples,
    )


def l3_prompt_artifact_hash(prompt_artifact: L3PromptArtifact) -> str:
    payload = json.dumps(
        prompt_artifact.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_l3_prompt_evolution(
    *,
    config: L3PromptEvolutionConfig,
    traces: list[TraceRecord] | list[TeacherTrace],
    task_schema: TaskSchema,
    local_slm_config: LocalSLMConfig,
    backend: LocalSLMBackend | None = None,
) -> dict[str, Any]:
    if config.mode != L3_PROMPT_EVOLUTION_MODE:
        raise ValueError("L3 prompt evolution mode must be agent-session")
    if config.max_agent_sessions < 0:
        raise ValueError("max_agent_sessions must be non-negative")
    teacher_traces = _teacher_visible_traces(traces)
    split = _split_l3_prompt_traces(teacher_traces)

    job_dir = config.job_dir
    workspace_root = job_dir / "workspace" / "l3_prompt"
    private_dir = job_dir / "private"
    candidates_dir = job_dir / "candidates"
    transcript_dir = job_dir / "transcripts"
    summary_path = job_dir / "summary.json"
    commands_path = job_dir / "commands.jsonl"
    job_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir.mkdir(parents=True, exist_ok=True)
    transcript_dir.mkdir(parents=True, exist_ok=True)

    private_paths = {
        "selection_holdout": private_dir / "selection_holdout.jsonl",
        "promotion_holdout": private_dir / "promotion_holdout.jsonl",
    }
    for split_name, path in private_paths.items():
        _write_jsonl(path, [trace.model_dump(mode="json") for trace in split[split_name]])

    baseline_prompt = _baseline_l3_prompt_artifact(
        split["train"],
        prompt_version=config.prompt_version,
    )
    prepare_l3_prompt_workspace(
        workspace_root=workspace_root,
        split=split,
        task_schema=task_schema,
        prompt_artifact=baseline_prompt,
        config=config,
        local_slm_config=local_slm_config,
    )

    command_results: list[dict[str, Any]] = []
    baseline = (
        _evaluate_l3_prompt_candidate(
            prompt_artifact=baseline_prompt,
            split=split,
            config=config,
            task_schema=task_schema,
            local_slm_config=local_slm_config,
            backend=backend,
            label="baseline",
        )
        if not config.skip_replay
        else None
    )

    stop_reason = "agent_session_completed"
    agent_sessions_started = 0
    agent_sessions_succeeded = 0
    if config.max_agent_sessions == 0:
        stop_reason = "agent_session_budget_exhausted"
    else:
        protected_snapshot = _protected_l3_workspace_snapshot(workspace_root)
        transcript_path = transcript_dir / "agent_session.jsonl"
        report_path = candidates_dir / "agent_session_report.md"
        agent_sessions_started = 1
        command_results.append(
            _run_l3_agent_session(
                config=config,
                workspace_root=workspace_root,
                transcript_path=transcript_path,
                report_path=report_path,
            )
        )
        if command_results[-1]["return_code"] != 0:
            stop_reason = "agent_session_failed"
        else:
            scope_report = _l3_workspace_scope_violation_report(
                workspace_root=workspace_root,
                before=protected_snapshot,
            )
            if scope_report is not None:
                command_results.append(
                    _l3_workspace_scope_violation_command_result(
                        workspace_root=workspace_root,
                        report=scope_report,
                    ),
                )
                stop_reason = "workspace_scope_violation"
            else:
                agent_sessions_succeeded = 1

    candidate_prompt_path = workspace_root / "prompt" / "l3_prompt.json"
    candidate_prompt: L3PromptArtifact | None = None
    candidate: dict[str, Any] | None = None
    candidate_snapshot: str | None = None
    if stop_reason == "agent_session_completed":
        try:
            candidate_prompt = L3PromptArtifact.model_validate_json(
                candidate_prompt_path.read_text(encoding="utf-8"),
            )
        except ValueError as exc:
            stop_reason = "candidate_prompt_invalid"
            command_results.append(
                _l3_prompt_validation_command_result(
                    workspace_root=workspace_root,
                    error=str(exc),
                )
            )
        else:
            snapshot_path = candidates_dir / "candidate_l3_prompt.json"
            snapshot_path.write_text(
                candidate_prompt.model_dump_json(indent=2) + "\n",
                encoding="utf-8",
            )
            candidate_snapshot = snapshot_path.relative_to(job_dir).as_posix()
            if not config.skip_replay:
                candidate = _evaluate_l3_prompt_candidate(
                    prompt_artifact=candidate_prompt,
                    split=split,
                    config=config,
                    task_schema=task_schema,
                    local_slm_config=local_slm_config,
                    backend=backend,
                    label="candidate",
                )

    summary = {
        "schema_version": "l3-prompt-evolution-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "mode": config.mode,
        "workspace": str(workspace_root),
        "data_split": {name: len(rows) for name, rows in split.items()},
        "agent_session": {
            "schema_version": "l3-agent-session-v1",
            "session_scope": "single long-running L4 agent session",
            "internal_loop_control": (
                "agent_decides_prompt_context_guard_eval_bench_stop"
            ),
            "agent_sessions_started": agent_sessions_started,
            "agent_sessions_succeeded": agent_sessions_succeeded,
            "max_agent_sessions": config.max_agent_sessions,
        },
        "workspace_scope_policy": _l3_workspace_scope_policy(),
        "private_data_scope": (
            "selection and promotion holdouts are stored outside the agent workspace"
        ),
        "skip_replay": config.skip_replay,
        "stop_reason": stop_reason,
        "baseline": _candidate_evaluation_summary(baseline),
        "candidate": _candidate_evaluation_summary(candidate),
        "candidate_prompt_snapshot": candidate_snapshot,
        "candidate_prompt_hash": (
            l3_prompt_artifact_hash(candidate_prompt) if candidate_prompt is not None else None
        ),
        "selection_decision": _l3_selection_decision(candidate),
        "adoption_decision": _l3_adoption_decision(candidate),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_jsonl(commands_path, command_results)
    return summary


def prepare_l3_prompt_workspace(
    *,
    workspace_root: Path,
    split: dict[str, list[TeacherTrace]],
    task_schema: TaskSchema,
    prompt_artifact: L3PromptArtifact,
    config: L3PromptEvolutionConfig,
    local_slm_config: LocalSLMConfig,
) -> None:
    if workspace_root.exists():
        shutil.rmtree(workspace_root)
    (workspace_root / "prompt").mkdir(parents=True, exist_ok=True)
    (workspace_root / "contexts").mkdir(parents=True, exist_ok=True)
    (workspace_root / "tools").mkdir(parents=True, exist_ok=True)
    (workspace_root / "runs").mkdir(parents=True, exist_ok=True)

    (workspace_root / "prompt" / "l3_prompt.json").write_text(
        prompt_artifact.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    (workspace_root / "prompt" / "context_packing.json").write_text(
        json.dumps(
            {
                "schema_version": "l3-context-packing-v1",
                "max_few_shots": 8,
                "source": "teacher_visible_train_and_validation_only",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (workspace_root / "prompt" / "routing_guard.md").write_text(
        "L3 should abstain unless the generated frame is valid and confidence clears the guard.\n",
        encoding="utf-8",
    )
    for name in ["train", "visible_validation"]:
        _write_jsonl(
            workspace_root / "contexts" / f"{name}.jsonl",
            [trace.model_dump(mode="json") for trace in split[name]],
        )
    (workspace_root / "contexts" / "task_schema.json").write_text(
        json.dumps(
            {
                "intent_names": task_schema.intent_names,
                "slot_names": task_schema.slot_names,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (workspace_root / "contexts" / "objective.json").write_text(
        json.dumps(_l3_objective_payload(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (workspace_root / "contexts" / "local_slm_config.json").write_text(
        local_slm_config.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    (workspace_root / "program.md").write_text(_l3_program_text(), encoding="utf-8")
    (workspace_root / "tools" / "validate_prompt.py").write_text(
        _l3_validate_prompt_tool_text(),
        encoding="utf-8",
    )
    (workspace_root / "tools" / "evaluate_prompt.py").write_text(
        _l3_evaluate_prompt_tool_text(),
        encoding="utf-8",
    )
    (workspace_root / "tools" / "bench_prompt.py").write_text(
        _l3_bench_prompt_tool_text(),
        encoding="utf-8",
    )
    (workspace_root / "tools" / "latency_cost_eval.py").write_text(
        _l3_latency_cost_eval_tool_text(),
        encoding="utf-8",
    )
    (workspace_root / "workspace_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "l3-prompt-workspace-v1",
                "editable_roots": ["prompt/"],
                "scratch_roots": ["runs/"],
                "protected_roots": [
                    "contexts/",
                    "tools/",
                    "program.md",
                    "workspace_manifest.json",
                ],
                "visible_context_files": sorted(
                    path.name for path in (workspace_root / "contexts").iterdir()
                ),
                "private_data_files_not_in_workspace": [
                    "selection_holdout.jsonl",
                    "promotion_holdout.jsonl",
                ],
                "commands": {
                    "validate_prompt": "python3 tools/validate_prompt.py",
                    "evaluate_visible_prompt": (
                        "python3 tools/evaluate_prompt.py --split visible_validation "
                        "--out runs/visible_prompt_eval.json"
                    ),
                    "bench_prompt": "python3 tools/bench_prompt.py --out runs/l3_benchmark.json",
                    "latency_cost_eval": (
                        "python3 tools/latency_cost_eval.py "
                        "--eval runs/visible_prompt_eval.json "
                        "--out runs/latency_cost_eval.json"
                    ),
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _teacher_visible_traces(
    traces: list[TraceRecord] | list[TeacherTrace],
) -> list[TeacherTrace]:
    if not traces:
        return []
    first = traces[0]
    if isinstance(first, TeacherTrace):
        return cast(list[TeacherTrace], traces)
    return traces_to_teacher_view(cast(list[TraceRecord], traces))


def _split_l3_prompt_traces(
    traces: list[TeacherTrace],
) -> dict[str, list[TeacherTrace]]:
    labeled = [trace for trace in traces if trace.teacher_frame is not None]
    if len(labeled) < 4:
        raise ValueError("L3 prompt evolution requires at least 4 teacher-labeled traces")
    train_end = max(1, int(len(labeled) * 0.60))
    visible_end = max(train_end + 1, int(len(labeled) * 0.80))
    selection_end = max(visible_end + 1, int(len(labeled) * 0.90))
    selection_end = min(selection_end, len(labeled) - 1)
    visible_end = min(visible_end, selection_end - 1)
    return {
        "train": labeled[:train_end],
        "visible_validation": labeled[train_end:visible_end],
        "selection_holdout": labeled[visible_end:selection_end],
        "promotion_holdout": labeled[selection_end:],
    }


def _baseline_l3_prompt_artifact(
    train_traces: list[TeacherTrace],
    *,
    prompt_version: str,
) -> L3PromptArtifact:
    examples = [
        {
            "trace_id": trace.request_id,
            "utterance": trace.utterance,
            "frame": trace.teacher_frame.model_dump(mode="json"),
        }
        for trace in train_traces
        if trace.teacher_frame is not None
    ][:4]
    return L3PromptArtifact(
        prompt_version=prompt_version,
        system_prompt=(
            "You are Darjeeling L3, a local virtual-assistant NLU model. "
            "Return one strict JSON object only."
        ),
        few_shot_examples=examples,
    )


def _evaluate_l3_prompt_candidate(
    *,
    prompt_artifact: L3PromptArtifact,
    split: dict[str, list[TeacherTrace]],
    config: L3PromptEvolutionConfig,
    task_schema: TaskSchema,
    local_slm_config: LocalSLMConfig,
    backend: LocalSLMBackend | None,
    label: str,
) -> dict[str, Any]:
    visible = replay_l3_prompt_artifact(
        prompt_artifact=prompt_artifact,
        traces=split["visible_validation"],
        task_schema=task_schema,
        config=local_slm_config,
        backend=backend,
    )
    selection = replay_l3_prompt_artifact(
        prompt_artifact=prompt_artifact,
        traces=split["selection_holdout"],
        task_schema=task_schema,
        config=local_slm_config,
        backend=backend,
    )
    promotion = replay_l3_prompt_artifact(
        prompt_artifact=prompt_artifact,
        traces=split["promotion_holdout"],
        task_schema=task_schema,
        config=local_slm_config,
        backend=backend,
    )
    return {
        "label": label,
        "prompt_sha256": l3_prompt_artifact_hash(prompt_artifact),
        "visible_validation": _l3_replay_gate_payload(
            visible,
            min_accepted_accuracy=config.min_accepted_accuracy,
            max_wrong_accept_rate=config.max_wrong_accept_rate,
        ),
        "selection_holdout": _l3_replay_gate_payload(
            selection,
            min_accepted_accuracy=config.min_accepted_accuracy,
            max_wrong_accept_rate=config.max_wrong_accept_rate,
        ),
        "promotion_holdout": _l3_replay_gate_payload(
            promotion,
            min_accepted_accuracy=config.min_accepted_accuracy,
            max_wrong_accept_rate=config.max_wrong_accept_rate,
        ),
    }


def _l3_replay_gate_payload(
    payload: dict[str, Any],
    *,
    min_accepted_accuracy: float,
    max_wrong_accept_rate: float,
) -> dict[str, Any]:
    accepted_accuracy = payload.get("accepted_accuracy")
    accuracy_passes = isinstance(accepted_accuracy, int | float) and (
        accepted_accuracy >= min_accepted_accuracy
    )
    wrong_accept_rate = payload.get("wrong_accept_rate")
    wrong_accept_passes = isinstance(wrong_accept_rate, int | float) and (
        wrong_accept_rate <= max_wrong_accept_rate
    )
    would_accept_count = int(payload.get("would_accept_count") or 0)
    return {
        "schema_version": "l3-prompt-replay-gate-v1",
        "prompt_version": payload.get("prompt_version"),
        "prompt_sha256": payload.get("prompt_sha256"),
        "status": payload.get("status"),
        "requests": payload.get("requests"),
        "would_accept_count": would_accept_count,
        "correct_accept_count": payload.get("correct_accept_count"),
        "wrong_accept_count": payload.get("wrong_accept_count"),
        "coverage": payload.get("coverage"),
        "accepted_accuracy": accepted_accuracy,
        "wrong_accept_rate": wrong_accept_rate,
        "latency_p50_ms": payload.get("latency_p50_ms"),
        "latency_p95_ms": payload.get("latency_p95_ms"),
        "parse_failures": payload.get("parse_failures"),
        "repair_count": payload.get("repair_count"),
        "passes_gate": bool(
            payload.get("status") == "success"
            and would_accept_count > 0
            and accuracy_passes
            and wrong_accept_passes
        ),
    }


def _candidate_evaluation_summary(candidate: dict[str, Any] | None) -> dict[str, Any] | None:
    if candidate is None:
        return None
    return {
        "label": candidate["label"],
        "prompt_sha256": candidate["prompt_sha256"],
        "visible_validation": candidate["visible_validation"],
        "selection_holdout": candidate["selection_holdout"],
        "promotion_holdout": candidate["promotion_holdout"],
    }


def _l3_selection_decision(candidate: dict[str, Any] | None) -> dict[str, Any]:
    if candidate is None:
        return {
            "selected": False,
            "reason": "candidate was not replayed",
        }
    visible_passes = bool(candidate["visible_validation"]["passes_gate"])
    selection_passes = bool(candidate["selection_holdout"]["passes_gate"])
    return {
        "selected": bool(visible_passes and selection_passes),
        "reason": (
            "visible validation and private selection gates passed"
            if visible_passes and selection_passes
            else "visible validation and private selection gates did not both pass"
        ),
    }


def _l3_adoption_decision(candidate: dict[str, Any] | None) -> dict[str, Any]:
    selection = _l3_selection_decision(candidate)
    promotion_passes = (
        bool(candidate["promotion_holdout"]["passes_gate"]) if candidate is not None else False
    )
    return {
        "adopted": bool(selection["selected"] and promotion_passes),
        "reason": (
            "visible validation, private selection, and private promotion gates passed"
            if selection["selected"] and promotion_passes
            else "candidate did not pass all private gates"
        ),
    }


def _l3_objective_payload(config: L3PromptEvolutionConfig) -> dict[str, Any]:
    return {
        "schema_version": "l3-prompt-objective-v1",
        "primary_objective": "increase safe local SLM would-accepts without wrong accepts",
        "editable_surface": [
            "prompt/l3_prompt.json",
            "prompt/context_packing.json",
            "prompt/routing_guard.md",
        ],
        "workspace_tools": [
            "tools/validate_prompt.py",
            "tools/evaluate_prompt.py",
            "tools/bench_prompt.py",
            "tools/latency_cost_eval.py",
        ],
        "gates": {
            "min_accepted_accuracy": config.min_accepted_accuracy,
            "max_wrong_accept_rate": config.max_wrong_accept_rate,
            "candidate_selection": "visible validation gate AND private selection gate",
            "adoption": "visible validation gate AND private selection AND promotion gates",
        },
        "invalid_strategies": [
            "using private selection or promotion rows",
            "inventing few-shot labels",
            "changing contexts/, tools/, program.md, or workspace_manifest.json",
            "treating visible replay success as adoption",
        ],
        "agent_session_policy": {
            "session_closure": "one L4 agent session per prompt-evolve job",
            "internal_loop_control": "agent_decides_prompt_context_guard_eval_bench_stop",
            "adoption_authority": "outer_private_gates_and_outer_replay",
        },
    }


def _l3_program_text() -> str:
    return "\n".join(
        [
            "# L3 prompt evolution program",
            "",
            "You are evolving the L3 local SLM prompt artifact, not Darjeeling core.",
            "Edit candidate files only under `prompt/`.",
            "`runs/` is scratch output; it is not promoted.",
            "Do not modify `contexts/`, `tools/`, `program.md`, or `workspace_manifest.json`.",
            "",
            "This is one autonomous L4 agent session. Decide how many times to inspect",
            "context, edit prompt/context packing/routing guard files, validate, run",
            "prompt replay or local SLM bench when available, debug, and stop.",
            "Available tools include prompt structure validation, visible prompt eval,",
            "local SLM bench, and latency/cost estimation. Tool outputs belong under",
            "`runs/` and do not decide adoption.",
            "",
            "Visible data includes only train and visible validation rows. Private",
            "selection and promotion holdouts are outside this workspace; do not try",
            "to inspect parent directories. You may judge visible readiness, but outer",
            "private gates and outer replay decide adoption.",
        ]
    )


def _l3_validate_prompt_tool_text() -> str:
    return """from __future__ import annotations

import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
payload = json.loads((root / "prompt" / "l3_prompt.json").read_text(encoding="utf-8"))
if not isinstance(payload.get("system_prompt"), str) or not payload["system_prompt"].strip():
    raise SystemExit("system_prompt must be a non-empty string")
few_shots = payload.get("few_shot_examples", [])
if not isinstance(few_shots, list):
    raise SystemExit("few_shot_examples must be a list")
for example in few_shots:
    if not isinstance(example, dict):
        raise SystemExit("few-shot example must be an object")
    if not {"trace_id", "utterance", "frame"}.issubset(example):
        raise SystemExit("few-shot example must include trace_id, utterance, and frame")
print("l3 prompt candidate is structurally valid")
"""


def _l3_tool_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _l3_tool_prelude_text() -> str:
    repo_root = json.dumps(_l3_tool_repo_root().as_posix())
    return f"""from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path({repo_root})
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

root = Path(__file__).resolve().parents[1]

"""


def _l3_evaluate_prompt_tool_text() -> str:
    return (
        _l3_tool_prelude_text()
        + """from darjeeling.compiler.l3_prompt_optimizer import replay_l3_prompt_artifact
from darjeeling.layers.l3_local_slm import L3PromptArtifact, LocalSLMConfig
from darjeeling.layers.l4_cloud_llm import TaskSchema
from darjeeling.schemas import TeacherTrace


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_traces(path: Path) -> list[TeacherTrace]:
    rows: list[TeacherTrace] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(TeacherTrace.model_validate_json(line))
    return rows


parser = argparse.ArgumentParser(description="Evaluate the L3 prompt on visible rows only.")
parser.add_argument(
    "--split",
    choices=["train", "visible_validation"],
    default="visible_validation",
)
parser.add_argument("--out", default="runs/visible_prompt_eval.json")
parser.add_argument("--max-requests", type=int, default=None)
args = parser.parse_args()

prompt = L3PromptArtifact.model_validate_json(
    (root / "prompt" / "l3_prompt.json").read_text(encoding="utf-8")
)
task_schema_payload = read_json(root / "contexts" / "task_schema.json")
task_schema = TaskSchema(
    intent_names=task_schema_payload["intent_names"],
    slot_names=task_schema_payload["slot_names"],
)
config = LocalSLMConfig.model_validate(read_json(root / "contexts" / "local_slm_config.json"))
traces = read_traces(root / "contexts" / f"{args.split}.jsonl")

payload = replay_l3_prompt_artifact(
    prompt_artifact=prompt,
    traces=traces,
    task_schema=task_schema,
    config=config,
    max_requests=args.max_requests,
)
payload["workspace_tool"] = {
    "name": "evaluate_prompt",
    "split": args.split,
    "visible_only": True,
    "private_data_visible": False,
}
out = root / args.out
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
"""
    )


def _l3_bench_prompt_tool_text() -> str:
    return (
        _l3_tool_prelude_text()
        + """from darjeeling.compiler.l3_prompt_optimizer import l3_prompt_artifact_hash
from darjeeling.layers.l3_local_slm import (
    DEFAULT_L3_BENCHMARK_UTTERANCES,
    L3LocalSLMLayer,
    L3PromptArtifact,
    LocalSLMConfig,
    benchmark_l3_layer,
)
from darjeeling.layers.l4_cloud_llm import TaskSchema


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


parser = argparse.ArgumentParser(description="Benchmark the L3 prompt with the local SLM.")
parser.add_argument("--out", default="runs/l3_benchmark.json")
args = parser.parse_args()

prompt = L3PromptArtifact.model_validate_json(
    (root / "prompt" / "l3_prompt.json").read_text(encoding="utf-8")
)
task_schema_payload = read_json(root / "contexts" / "task_schema.json")
task_schema = TaskSchema(
    intent_names=task_schema_payload["intent_names"],
    slot_names=task_schema_payload["slot_names"],
)
config = LocalSLMConfig.model_validate(read_json(root / "contexts" / "local_slm_config.json"))
layer = L3LocalSLMLayer(config=config, task_schema=task_schema, prompt_artifact=prompt)
payload = {
    **benchmark_l3_layer(layer, DEFAULT_L3_BENCHMARK_UTTERANCES),
    "prompt_sha256": l3_prompt_artifact_hash(prompt),
    "workspace_tool": {
        "name": "bench_prompt",
        "visible_only": True,
        "private_data_visible": False,
    },
}
out = root / args.out
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
"""
    )


def _l3_latency_cost_eval_tool_text() -> str:
    return (
        _l3_tool_prelude_text()
        + """from darjeeling.layers.l3_local_slm import L3PromptArtifact, LocalSLMConfig
from darjeeling.layers.l4_cloud_llm import TaskSchema
from darjeeling.schemas import TeacherTrace


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_traces(path: Path) -> list[TeacherTrace]:
    rows: list[TeacherTrace] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(TeacherTrace.model_validate_json(line))
    return rows


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
    return float(ordered[idx])


parser = argparse.ArgumentParser(
    description="Estimate prompt size, latency, and local evaluation cost for visible rows."
)
parser.add_argument("--eval", default=None, help="Optional evaluate_prompt JSON output.")
parser.add_argument("--out", default="runs/latency_cost_eval.json")
args = parser.parse_args()

prompt = L3PromptArtifact.model_validate_json(
    (root / "prompt" / "l3_prompt.json").read_text(encoding="utf-8")
)
task_schema_payload = read_json(root / "contexts" / "task_schema.json")
task_schema = TaskSchema(
    intent_names=task_schema_payload["intent_names"],
    slot_names=task_schema_payload["slot_names"],
)
config = LocalSLMConfig.model_validate(read_json(root / "contexts" / "local_slm_config.json"))
traces = read_traces(root / "contexts" / "visible_validation.jsonl")
rendered_chars = [len(prompt.render(trace.utterance, task_schema)) for trace in traces]
eval_payload = None
if args.eval is not None and (root / args.eval).exists():
    eval_payload = read_json(root / args.eval)

payload = {
    "schema_version": "l3-workspace-latency-cost-eval-v1",
    "status": "success",
    "request_count": len(traces),
    "prompt_rendered_chars_p50": percentile([float(v) for v in rendered_chars], 50),
    "prompt_rendered_chars_p95": percentile([float(v) for v in rendered_chars], 95),
    "model_name": config.model_name,
    "max_new_tokens": config.max_new_tokens,
    "estimated_local_eval_cost_usd": 0.0,
    "eval_latency_p50_ms": eval_payload.get("latency_p50_ms") if eval_payload else None,
    "eval_latency_p95_ms": eval_payload.get("latency_p95_ms") if eval_payload else None,
    "workspace_tool": {
        "name": "latency_cost_eval",
        "visible_only": True,
        "private_data_visible": False,
    },
}
out = root / args.out
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
"""
    )


def _run_l3_agent_session(
    *,
    config: L3PromptEvolutionConfig,
    workspace_root: Path,
    transcript_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    command = [config.codex_command]
    if config.codex_model:
        command.extend(["--model", config.codex_model])
    command.extend(["--sandbox", config.sandbox, "-a", config.approval_policy, "exec"])
    command.extend(
        [
            "--skip-git-repo-check",
            "--cd",
            str(workspace_root.resolve()),
            "--json",
            "-o",
            str(report_path.resolve()),
            "-",
        ]
    )
    prompt = "\n".join(
        [
            "Read program.md and run one autonomous L3 prompt evolution session.",
            "Edit only prompt/. Use contexts/ and tools/ for visible data and validation.",
            "Stop when the visible objective is met, no safe progress remains, or",
            "budget/risk says to stop. Leave the final candidate in prompt/l3_prompt.json.",
        ]
    )
    result = _run_command(command, cwd=workspace_root, timeout_s=config.timeout_s, stdin=prompt)
    transcript_path.write_text(str(result["stdout"]), encoding="utf-8")
    return result


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


def _l3_workspace_scope_policy() -> dict[str, Any]:
    return {
        "schema_version": "l3-prompt-workspace-scope-v1",
        "candidate_code_writable_roots": ["prompt/"],
        "scratch_writable_roots": ["runs/"],
        "protected_roots": ["contexts/", "tools/", "program.md", "workspace_manifest.json"],
        "ignored_generated_files": ["__pycache__/", ".pytest_cache/", "*.pyc"],
        "enforcement": "checked_after_agent_session_before_candidate_validation",
    }


def _protected_l3_workspace_snapshot(workspace_root: Path) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for path in workspace_root.rglob("*"):
        if not path.is_file():
            continue
        rel_path = path.relative_to(workspace_root)
        if _is_l3_workspace_scope_ignored(rel_path) or _is_l3_workspace_scope_writable(rel_path):
            continue
        snapshot[rel_path.as_posix()] = _file_sha256(path)
    return snapshot


def _l3_workspace_scope_violation_report(
    *,
    workspace_root: Path,
    before: dict[str, str],
) -> dict[str, Any] | None:
    after = _protected_l3_workspace_snapshot(workspace_root)
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    modified = sorted(path for path in set(before) & set(after) if before[path] != after[path])
    if not added and not removed and not modified:
        return None
    return {
        "schema_version": "l3-prompt-workspace-scope-violation-v1",
        "policy": _l3_workspace_scope_policy(),
        "added_protected_files": added,
        "removed_protected_files": removed,
        "modified_protected_files": modified,
        "message": "L3 prompt evolution may change prompt/ and runs/ only",
    }


def _l3_workspace_scope_violation_command_result(
    *,
    workspace_root: Path,
    report: dict[str, Any],
) -> dict[str, Any]:
    return {
        "command": ["l3-workspace-scope-check"],
        "cwd": str(workspace_root),
        "started_at": datetime.now(UTC).isoformat(),
        "return_code": 1,
        "stdout": "",
        "stderr": json.dumps(report, sort_keys=True),
        "workspace_scope_violation": report,
    }


def _l3_prompt_validation_command_result(
    *,
    workspace_root: Path,
    error: str,
) -> dict[str, Any]:
    return {
        "command": ["l3-prompt-validate"],
        "cwd": str(workspace_root),
        "started_at": datetime.now(UTC).isoformat(),
        "return_code": 1,
        "stdout": "",
        "stderr": error,
    }


def _is_l3_workspace_scope_writable(rel_path: Path) -> bool:
    parts = rel_path.parts
    return bool(parts and parts[0] in {"prompt", "runs"})


def _is_l3_workspace_scope_ignored(rel_path: Path) -> bool:
    ignored_dirs = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
    if any(part in ignored_dirs for part in rel_path.parts):
        return True
    return rel_path.suffix in {".pyc", ".pyo"}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def calibrate_l3_confidence_threshold(
    traces: list[TraceRecord],
    *,
    max_wrong_accept_rate: float = 0.05,
) -> L3GuardCalibrationResult | None:
    samples = _l3_calibration_samples(traces)
    if not samples:
        return None

    thresholds = sorted({0.0, 1.0, *(sample["confidence"] for sample in samples)})
    best: L3GuardCalibrationResult | None = None
    for threshold in thresholds:
        accepted = [
            sample for sample in samples if sample["eligible"] and sample["confidence"] >= threshold
        ]
        accepted_count = len(accepted)
        wrong_accept_count = sum(1 for sample in accepted if not sample["correct"])
        wrong_accept_rate = wrong_accept_count / accepted_count if accepted_count else 0.0
        if wrong_accept_rate > max_wrong_accept_rate:
            continue
        accepted_accuracy = (
            (accepted_count - wrong_accept_count) / accepted_count if accepted_count else None
        )
        candidate = L3GuardCalibrationResult(
            threshold=float(threshold),
            sample_count=len(samples),
            labeled_count=len(samples),
            accepted_count=accepted_count,
            wrong_accept_count=wrong_accept_count,
            coverage=accepted_count / len(samples),
            accepted_accuracy=accepted_accuracy,
            wrong_accept_rate=wrong_accept_rate,
            max_wrong_accept_rate=max_wrong_accept_rate,
        )
        if best is None or (candidate.accepted_count, -candidate.threshold) > (
            best.accepted_count,
            -best.threshold,
        ):
            best = candidate
    return best


def replay_l3_prompt_artifact(
    *,
    prompt_artifact: L3PromptArtifact,
    traces: list[TraceRecord] | list[TeacherTrace],
    task_schema: TaskSchema,
    config: LocalSLMConfig,
    backend: LocalSLMBackend | None = None,
    max_requests: int | None = None,
) -> dict[str, Any]:
    replay_config = config.model_copy(update={"mode": "shadow"})
    layer = L3LocalSLMLayer(
        config=replay_config,
        task_schema=task_schema,
        prompt_artifact=prompt_artifact,
        backend=backend,
    )
    labeled = [
        trace
        for trace in traces
        if trace.teacher_frame is not None or getattr(trace, "gold_frame", None) is not None
    ]
    if max_requests is not None:
        labeled = labeled[:max_requests]

    request_results: list[dict[str, Any]] = []
    would_accept_count = 0
    correct_accept_count = 0
    wrong_accept_count = 0
    parse_failures = 0
    failures = 0
    repair_count = 0
    latencies_ms: list[float] = []

    for trace in labeled:
        expected = trace.teacher_frame or getattr(trace, "gold_frame", None)
        assert expected is not None
        result = layer.try_answer(trace.utterance)
        metadata = result.metadata or {}
        predicted = _frame_from_metadata(metadata.get("shadow_frame")) or result.frame
        would_accept = metadata.get("would_accept") is True
        correct = predicted == expected if predicted is not None else False
        would_accept_count += int(would_accept)
        correct_accept_count += int(would_accept and correct)
        wrong_accept_count += int(would_accept and not correct)
        parse_failures += int("parse failed" in result.reason)
        failures += int("failed" in result.reason)
        repair_count += int(metadata.get("repair_used") is True)
        latencies_ms.append(result.latency_ms)
        request_results.append(
            {
                "request_id": trace.request_id,
                "utterance": trace.utterance,
                "would_accept": would_accept,
                "correct": correct,
                "reason": result.reason,
                "latency_ms": result.latency_ms,
                "confidence": metadata.get("confidence", result.confidence),
                "predicted_frame": (
                    predicted.model_dump(mode="json") if predicted is not None else None
                ),
            }
        )

    labeled_count = len(labeled)
    coverage = would_accept_count / labeled_count if labeled_count else 0.0
    accepted_accuracy = correct_accept_count / would_accept_count if would_accept_count else None
    wrong_accept_rate = wrong_accept_count / labeled_count if labeled_count else 1.0
    return {
        "schema_version": "l3-prompt-replay-v1",
        "status": "success",
        "prompt_version": prompt_artifact.prompt_version,
        "prompt_sha256": l3_prompt_artifact_hash(prompt_artifact),
        "requests": labeled_count,
        "would_accept_count": would_accept_count,
        "correct_accept_count": correct_accept_count,
        "wrong_accept_count": wrong_accept_count,
        "coverage": coverage,
        "accepted_accuracy": accepted_accuracy,
        "wrong_accept_rate": wrong_accept_rate,
        "parse_failures": parse_failures,
        "failures": failures,
        "repair_count": repair_count,
        "latency_p50_ms": _percentile(latencies_ms, 50),
        "latency_p95_ms": _percentile(latencies_ms, 95),
        "backend": layer.backend.status(),
        "request_results": request_results,
    }


def _l3_calibration_samples(traces: list[TraceRecord]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for trace in traces:
        expected = trace.teacher_frame or trace.gold_frame
        if expected is None:
            continue
        for result in trace.layer_results:
            if result.layer != "L3" or not result.metadata:
                continue
            confidence = result.metadata.get("confidence")
            if not isinstance(confidence, int | float):
                continue
            predicted = result.frame or _frame_from_metadata(result.metadata.get("shadow_frame"))
            if predicted is None:
                continue
            validation_errors = result.metadata.get("validation_errors") or []
            eligible = (
                not predicted.is_abstain
                and isinstance(validation_errors, list)
                and not validation_errors
            )
            samples.append(
                {
                    "confidence": float(confidence),
                    "eligible": eligible,
                    "correct": predicted == expected,
                }
            )
    return samples


def _frame_from_metadata(payload: Any) -> Frame | None:
    if payload is None:
        return None
    try:
        return Frame.model_validate(payload)
    except ValueError:
        return None


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight
