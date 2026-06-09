from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

from darjeeling.layers.l2_student import (
    L2StudentConfig,
    train_l2_student,
    training_examples_from_teacher_traces,
)
from darjeeling.schemas import Frame, TeacherTrace

L2TargetEvolutionMode = Literal["dry-run", "codex-cli"]


@dataclass(frozen=True)
class L2TargetEvolutionConfig:
    source_repo_dir: Path
    job_dir: Path
    rounds: int = 3
    mode: L2TargetEvolutionMode = "dry-run"
    dry_run_patches: tuple[Path, ...] = ()
    codex_command: str = "codex"
    codex_model: str | None = "gpt-5.5"
    timeout_s: float = 7200.0
    sandbox: str = "workspace-write"
    approval_policy: str = "never"
    ignore_user_config: bool = True
    ignore_rules: bool = True
    ephemeral: bool = True
    min_accepted_accuracy: float = 0.93
    max_wrong_accept_rate: float = 0.05


def run_l2_target_evolution(
    *,
    config: L2TargetEvolutionConfig,
    traces: list[TeacherTrace],
) -> dict[str, Any]:
    if config.rounds < 1:
        raise ValueError("rounds must be at least 1")
    job_dir = config.job_dir
    workspace_root = job_dir / "workspace" / "l2_target"
    rounds_dir = job_dir / "rounds"
    commands_path = job_dir / "commands.jsonl"
    summary_path = job_dir / "summary.json"
    transcript_dir = job_dir / "transcripts"
    private_dir = job_dir / "private"
    job_dir.mkdir(parents=True, exist_ok=True)
    rounds_dir.mkdir(parents=True, exist_ok=True)
    transcript_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)

    split = split_l2_target_traces(traces)
    holdout_path = private_dir / "promotion_holdout.jsonl"
    _write_jsonl(
        holdout_path,
        [trace.model_dump(mode="json") for trace in split["promotion_holdout"]],
    )
    prepare_l2_target_workspace(
        source_repo_dir=config.source_repo_dir,
        workspace_root=workspace_root,
        split=split,
    )

    command_results: list[dict[str, Any]] = []
    round_results: list[dict[str, Any]] = []
    for round_index in range(1, config.rounds + 1):
        if config.mode == "dry-run":
            patch_index = round_index - 1
            if patch_index < len(config.dry_run_patches):
                command_results.append(
                    _run_command(
                        ["git", "apply", str(config.dry_run_patches[patch_index].resolve())],
                        cwd=workspace_root,
                        timeout_s=config.timeout_s,
                    )
                )
                if command_results[-1]["return_code"] != 0:
                    break
        else:
            transcript_path = transcript_dir / f"round_{round_index:03d}.jsonl"
            report_path = rounds_dir / f"round_{round_index:03d}_agent_report.md"
            command_results.append(
                _run_codex_round(
                    config=config,
                    workspace_root=workspace_root,
                    round_index=round_index,
                    transcript_path=transcript_path,
                    report_path=report_path,
                )
            )
            if command_results[-1]["return_code"] != 0:
                break

        inner_result = evaluate_target_workspace(
            workspace_root=workspace_root,
            split="inner_validation",
            min_accepted_accuracy=config.min_accepted_accuracy,
            max_wrong_accept_rate=config.max_wrong_accept_rate,
        )
        promotion_result = evaluate_target_workspace(
            workspace_root=workspace_root,
            split="promotion_holdout",
            holdout_path=holdout_path,
            min_accepted_accuracy=config.min_accepted_accuracy,
            max_wrong_accept_rate=config.max_wrong_accept_rate,
        )
        round_payload = {
            "round": round_index,
            "inner_validation": inner_result,
            "promotion_holdout": promotion_result,
        }
        (rounds_dir / f"round_{round_index:03d}.json").write_text(
            json.dumps(round_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        round_results.append(round_payload)

    summary = {
        "schema_version": "l2-target-evolution-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "mode": config.mode,
        "rounds_requested": config.rounds,
        "rounds_completed": len(round_results),
        "workspace": str(workspace_root),
        "data_split": {key: len(value) for key, value in split.items()},
        "rounds": round_results,
        "best_round": _best_round(round_results),
        "target_code_scope": "target/",
        "core_code_scope": "system/darjeeling/ is read-only evaluator/core",
        "private_data_scope": "promotion holdout is stored outside the agent workspace",
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_jsonl(commands_path, command_results)
    return summary


def split_l2_target_traces(traces: list[TeacherTrace]) -> dict[str, list[TeacherTrace]]:
    labeled = [trace for trace in traces if trace.teacher_frame is not None]
    if len(labeled) < 6:
        raise ValueError("target evolution requires at least 6 teacher-labeled traces")
    train_end = max(4, int(len(labeled) * 0.60))
    inner_end = max(train_end + 1, int(len(labeled) * 0.80))
    inner_end = min(inner_end, len(labeled) - 1)
    return {
        "train": labeled[:train_end],
        "inner_validation": labeled[train_end:inner_end],
        "promotion_holdout": labeled[inner_end:],
    }


def prepare_l2_target_workspace(
    *,
    source_repo_dir: Path,
    workspace_root: Path,
    split: dict[str, list[TeacherTrace]],
) -> None:
    if workspace_root.exists():
        shutil.rmtree(workspace_root)
    workspace_root.mkdir(parents=True, exist_ok=True)
    system_repo_dir = workspace_root / "system" / "darjeeling"
    _copy_system_workspace(source_repo_dir, system_repo_dir)
    (workspace_root / "target").mkdir(parents=True, exist_ok=True)
    (workspace_root / "data").mkdir(parents=True, exist_ok=True)
    (workspace_root / "tools").mkdir(parents=True, exist_ok=True)
    (workspace_root / "target" / "target_l2.py").write_text(
        _target_l2_template(),
        encoding="utf-8",
    )
    for name in ["train", "inner_validation"]:
        _write_jsonl(
            workspace_root / "data" / f"{name}.jsonl",
            [trace.model_dump(mode="json") for trace in split[name]],
        )
    (workspace_root / "program.md").write_text(_target_program_text(), encoding="utf-8")
    (workspace_root / "tools" / "evaluate.py").write_text(
        _evaluate_tool_text(),
        encoding="utf-8",
    )
    (workspace_root / "tools" / "inspect_context.py").write_text(
        _inspect_tool_text(),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "l2-target-workspace-v1",
        "target_dir": "target",
        "system_repo_dir": "system/darjeeling",
        "data_dir": "data",
        "tools_dir": "tools",
        "data_files": sorted(path.name for path in (workspace_root / "data").iterdir()),
        "private_data_files_not_in_workspace": ["promotion_holdout.jsonl"],
    }
    (workspace_root / "workspace_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def evaluate_target_workspace(
    *,
    workspace_root: Path,
    split: Literal["inner_validation", "promotion_holdout"],
    holdout_path: Path | None = None,
    min_accepted_accuracy: float = 0.93,
    max_wrong_accept_rate: float = 0.05,
) -> dict[str, Any]:
    train_traces = _read_teacher_jsonl(workspace_root / "data" / "train.jsonl")
    if split == "promotion_holdout":
        if holdout_path is None:
            raise ValueError("promotion_holdout evaluation requires a private holdout_path")
        validation_path = holdout_path
    else:
        validation_path = workspace_root / "data" / f"{split}.jsonl"
    validation_traces = _read_teacher_jsonl(validation_path)
    target_module = _load_target_module(workspace_root / "target" / "target_l2.py")
    overrides = _target_config_overrides(target_module)
    config = L2StudentConfig(**overrides)
    bundle = train_l2_student(training_examples_from_teacher_traces(train_traces), config)

    accepted = 0
    correct = 0
    wrong = 0
    examples: list[dict[str, Any]] = []
    for trace in validation_traces:
        if trace.teacher_frame is None:
            continue
        prediction = bundle.predict(trace.utterance)
        frame = _target_postprocess_frame(
            target_module,
            utterance=trace.utterance,
            frame=prediction.frame,
            metadata=prediction.model_dump(mode="json"),
        )
        should_accept = (
            bundle.config.runtime_enabled
            and prediction.guard_probability >= bundle.config.accept_threshold
        )
        if should_accept:
            accepted += 1
            if frame == trace.teacher_frame:
                correct += 1
            else:
                wrong += 1
                if len(examples) < 8:
                    examples.append(
                        {
                            "request_id": trace.request_id,
                            "utterance": trace.utterance,
                            "teacher_frame": trace.teacher_frame.model_dump(mode="json"),
                            "predicted_frame": frame.model_dump(mode="json"),
                            "guard_probability": prediction.guard_probability,
                        }
                    )

    total = len(validation_traces)
    accepted_accuracy = correct / accepted if accepted else None
    wrong_accept_rate = wrong / accepted if accepted else 0.0
    passes_gate = bool(
        accepted
        and accepted_accuracy is not None
        and accepted_accuracy >= min_accepted_accuracy
        and wrong_accept_rate <= max_wrong_accept_rate
    )
    return {
        "split": split,
        "train_size": len(train_traces),
        "validation_size": total,
        "accepted": accepted,
        "correct_accepts": correct,
        "wrong_accepts": wrong,
        "coverage": accepted / total if total else 0.0,
        "accepted_accuracy": accepted_accuracy,
        "wrong_accept_rate": wrong_accept_rate,
        "passes_gate": passes_gate,
        "config": bundle.config.model_dump(mode="json"),
        "wrong_examples": examples,
    }


def evaluate_target_workspace_cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--split", choices=["inner_validation"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-accepted-accuracy", type=float, default=0.93)
    parser.add_argument("--max-wrong-accept-rate", type=float, default=0.05)
    args = parser.parse_args(argv)
    payload = evaluate_target_workspace(
        workspace_root=args.workspace,
        split=args.split,
        min_accepted_accuracy=args.min_accepted_accuracy,
        max_wrong_accept_rate=args.max_wrong_accept_rate,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


def _copy_system_workspace(source_repo_dir: Path, system_repo_dir: Path) -> None:
    if system_repo_dir.exists():
        shutil.rmtree(system_repo_dir)
    system_repo_dir.mkdir(parents=True, exist_ok=True)
    for rel_path in [Path("src"), Path("tests"), Path("pyproject.toml"), Path("uv.lock")]:
        source_path = source_repo_dir / rel_path
        target_path = system_repo_dir / rel_path
        if not source_path.exists():
            continue
        if source_path.is_dir():
            shutil.copytree(
                source_path,
                target_path,
                ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc"),
            )
        else:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)


def _target_l2_template() -> str:
    return '''from __future__ import annotations

from typing import Any


def config_overrides() -> dict[str, Any]:
    """Return target-specific L2StudentConfig overrides."""
    return {}


def postprocess_frame(
    utterance: str,
    frame: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Return a target-specific frame dict after the core L2 model predicts."""
    del utterance, metadata
    return frame
'''


def _target_program_text() -> str:
    return "\n".join(
        [
            "# L2 target evolution program",
            "",
            "You are evolving target-dependent L2 runtime code, not Darjeeling core.",
            "Edit only files under `target/`.",
            "",
            "Workspace layout:",
            "- `target/` is editable target-specific L2 code.",
            "- `system/darjeeling/` is read-only Darjeeling core/evaluator code.",
            "- `data/train.jsonl` is visible training data.",
            "- `data/inner_validation.jsonl` is visible fast feedback data.",
            "- `tools/evaluate.py` trains/evaluates the target code in seconds.",
            "",
            "Optimize generalization from the visible train and inner-validation data.",
            "Promotion holdout is outside this workspace and only the outer harness can",
            "read it. It is acceptable for `target/` to contain target-specific code;",
            "do not move that code into core.",
        ]
    )


def _evaluate_tool_text() -> str:
    return """from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYSTEM_SRC = ROOT / "system" / "darjeeling" / "src"
sys.path.insert(0, str(SYSTEM_SRC))

from darjeeling.compiler.l2_target_evolution import evaluate_target_workspace_cli

if __name__ == "__main__":
    raise SystemExit(evaluate_target_workspace_cli())
"""


def _inspect_tool_text() -> str:
    return """from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in sorted((ROOT / "data").iterdir()):
    print(f"{path.name}: {path.stat().st_size} bytes")
"""


def _read_teacher_jsonl(path: Path) -> list[TeacherTrace]:
    return [
        TeacherTrace.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_target_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("target_l2", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import target module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _target_config_overrides(module: ModuleType) -> dict[str, Any]:
    function = getattr(module, "config_overrides", None)
    if function is None:
        return {}
    value = function()
    if not isinstance(value, dict):
        raise TypeError("target_l2.config_overrides() must return a dict")
    return value


def _target_postprocess_frame(
    module: ModuleType,
    *,
    utterance: str,
    frame: Frame,
    metadata: dict[str, Any],
) -> Frame:
    function = getattr(module, "postprocess_frame", None)
    if function is None:
        return frame
    value = function(utterance, frame.model_dump(mode="json"), metadata)
    return Frame.model_validate(value)


def _run_codex_round(
    *,
    config: L2TargetEvolutionConfig,
    workspace_root: Path,
    round_index: int,
    transcript_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    command = [config.codex_command]
    if config.codex_model:
        command.extend(["--model", config.codex_model])
    command.extend(["--sandbox", config.sandbox, "-a", config.approval_policy, "exec"])
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
            str(workspace_root.resolve()),
            "--json",
            "-o",
            str(report_path.resolve()),
            "-",
        ]
    )
    prompt = f"Read program.md and complete target L2 evolution round {round_index}."
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
            env={**os.environ, "GIT_CEILING_DIRECTORIES": str(cwd.parent.resolve())},
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
            "stderr": exc.stderr or "command timed out",
        }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _best_round(rounds: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rounds:
        return None
    return max(
        rounds,
        key=lambda item: (
            bool(item["promotion_holdout"]["passes_gate"]),
            item["promotion_holdout"]["coverage"],
            item["promotion_holdout"]["accepted_accuracy"] or 0.0,
            -item["promotion_holdout"]["wrong_accept_rate"],
        ),
    )
