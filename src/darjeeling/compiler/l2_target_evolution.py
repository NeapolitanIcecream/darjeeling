from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from darjeeling.layers.l2_student import (
    L2StudentConfig,
    train_l2_student,
    training_examples_from_teacher_traces,
)
from darjeeling.layers.l2_target import (
    load_target_module,
    target_accept_prediction,
    target_config_overrides,
    target_postprocess_frame,
)
from darjeeling.schemas import TeacherTrace

L2TargetEvolutionMode = Literal["dry-run", "codex-cli", "local-search"]
L2TargetSearchSpace = Literal["compact", "wide"]
L2TargetBudgetProfile = Literal["standard", "fixed-inner", "smoke"]

DEFAULT_TARGET_EVOLVE_ROUNDS = 12
DEFAULT_TARGET_LOCAL_SEARCH_TRIALS = 96
DEFAULT_TARGET_INNER_PATIENCE_ROUNDS = 4


@dataclass(frozen=True)
class L2TargetEvolutionConfig:
    source_repo_dir: Path
    job_dir: Path
    rounds: int = DEFAULT_TARGET_EVOLVE_ROUNDS
    mode: L2TargetEvolutionMode = "dry-run"
    dry_run_patches: tuple[Path, ...] = ()
    codex_command: str = "codex"
    codex_model: str | None = "gpt-5.5"
    timeout_s: float = 7200.0
    local_search_trials: int = DEFAULT_TARGET_LOCAL_SEARCH_TRIALS
    local_search_timeout_s: float | None = None
    local_search_space: L2TargetSearchSpace = "compact"
    budget_profile: L2TargetBudgetProfile = "standard"
    sandbox: str = "workspace-write"
    approval_policy: str = "never"
    ignore_user_config: bool = True
    ignore_rules: bool = True
    ephemeral: bool = True
    min_accepted_accuracy: float = 0.93
    max_wrong_accept_rate: float = 0.05
    inner_patience_rounds: int = DEFAULT_TARGET_INNER_PATIENCE_ROUNDS
    stop_on_selection_gate: bool = False


def run_l2_target_evolution(
    *,
    config: L2TargetEvolutionConfig,
    traces: list[TeacherTrace],
) -> dict[str, Any]:
    if config.rounds < 1:
        raise ValueError("rounds must be at least 1")
    if config.inner_patience_rounds < 0:
        raise ValueError("inner_patience_rounds must be non-negative")
    if config.local_search_trials < 1:
        raise ValueError("local_search_trials must be at least 1")
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
    private_paths = {
        "selection_holdout": private_dir / "selection_holdout.jsonl",
        "promotion_holdout": private_dir / "promotion_holdout.jsonl",
    }
    for split_name, path in private_paths.items():
        _write_jsonl(
            path,
            [trace.model_dump(mode="json") for trace in split[split_name]],
        )
    prepare_l2_target_workspace(
        source_repo_dir=config.source_repo_dir,
        workspace_root=workspace_root,
        split=split,
    )

    baseline = _evaluate_target_candidate(
        workspace_root=workspace_root,
        private_paths=private_paths,
        config=config,
        label="baseline",
    )
    (rounds_dir / "baseline.json").write_text(
        json.dumps(baseline, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    command_results: list[dict[str, Any]] = []
    round_results: list[dict[str, Any]] = []
    best_inner = baseline["inner_validation"]
    no_inner_improvement_rounds = 0
    stop_reason = "round_budget_exhausted"
    if (
        config.stop_on_selection_gate
        and baseline["inner_validation"]["passes_gate"]
        and baseline["selection_holdout"]["passes_gate"]
    ):
        stop_reason = "baseline_selection_gate_passed"

    for round_index in range(1, config.rounds + 1):
        if stop_reason != "round_budget_exhausted":
            break
        _write_target_state_files(
            workspace_root=workspace_root,
            config=config,
            round_index=round_index,
            state_kind="before_round",
            baseline=baseline,
            round_results=round_results,
            no_inner_improvement_rounds=no_inner_improvement_rounds,
        )
        if config.mode == "dry-run":
            local_search_report: dict[str, Any] | None = None
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
                    stop_reason = "dry_run_patch_failed"
                    break
        elif config.mode == "local-search":
            search_report_path = rounds_dir / f"round_{round_index:03d}_local_search.json"
            try:
                local_search_report = run_local_target_search(
                    workspace_root=workspace_root,
                    trials=config.local_search_trials,
                    search_space=config.local_search_space,
                    timeout_s=config.local_search_timeout_s,
                    min_accepted_accuracy=config.min_accepted_accuracy,
                    max_wrong_accept_rate=config.max_wrong_accept_rate,
                )
                search_report_path.write_text(
                    json.dumps(local_search_report, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                command_results.append(
                    _local_search_command_result(
                        workspace_root=workspace_root,
                        round_index=round_index,
                        report_path=search_report_path,
                        report=local_search_report,
                    )
                )
            except Exception as exc:
                command_results.append(
                    _failed_local_search_command_result(
                        workspace_root=workspace_root,
                        round_index=round_index,
                        error=str(exc),
                    )
                )
                stop_reason = "local_search_failed"
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
                stop_reason = "agent_command_failed"
                break
            local_search_report = None

        candidate = _evaluate_target_candidate(
            workspace_root=workspace_root,
            private_paths=private_paths,
            config=config,
            label=f"round_{round_index:03d}",
        )
        inner_result = candidate["inner_validation"]
        selection_result = candidate["selection_holdout"]
        promotion_result = candidate["promotion_holdout"]
        inner_improved = _is_inner_improvement(inner_result, best_inner)
        if inner_improved:
            best_inner = inner_result
            no_inner_improvement_rounds = 0
        else:
            no_inner_improvement_rounds += 1
        round_payload = {
            "round": round_index,
            "inner_improved": inner_improved,
            "passes_candidate_selection_gate": bool(
                inner_result["passes_gate"] and selection_result["passes_gate"]
            ),
            "passes_private_selection_gate": bool(selection_result["passes_gate"]),
            "passes_private_promotion_gate": bool(promotion_result["passes_gate"]),
            "inner_score": list(_inner_score(inner_result)),
            "inner_delta_vs_baseline": _metric_delta(
                inner_result,
                baseline["inner_validation"],
            ),
            "inner_validation": inner_result,
            "selection_delta_vs_baseline": _metric_delta(
                selection_result,
                baseline["selection_holdout"],
            ),
            "selection_holdout": selection_result,
            "promotion_delta_vs_baseline": _metric_delta(
                promotion_result,
                baseline["promotion_holdout"],
            ),
            "promotion_holdout": promotion_result,
        }
        if local_search_report is not None:
            round_payload["local_search"] = _visible_local_search_summary(
                local_search_report,
            )
        (rounds_dir / f"round_{round_index:03d}.json").write_text(
            json.dumps(round_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        round_results.append(round_payload)
        if (
            config.stop_on_selection_gate
            and inner_result["passes_gate"]
            and selection_result["passes_gate"]
        ):
            stop_reason = "selection_gate_passed"
            break
        if (
            round_index < config.rounds
            and config.inner_patience_rounds
            and no_inner_improvement_rounds >= config.inner_patience_rounds
        ):
            stop_reason = "inner_validation_patience_exhausted"
            break

    _write_target_state_files(
        workspace_root=workspace_root,
        config=config,
        round_index=len(round_results) + 1,
        state_kind="final",
        baseline=baseline,
        round_results=round_results,
        no_inner_improvement_rounds=no_inner_improvement_rounds,
    )

    summary = {
        "schema_version": "l2-target-evolution-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "mode": config.mode,
        "rounds_requested": config.rounds,
        "rounds_completed": len(round_results),
        "stop_reason": stop_reason,
        "budget_policy": {
            "inner_patience_rounds": config.inner_patience_rounds,
            "stop_on_selection_gate": config.stop_on_selection_gate,
            "local_search_trials": config.local_search_trials,
            "local_search_timeout_s": config.local_search_timeout_s,
            "local_search_space": config.local_search_space,
            "profile": config.budget_profile,
        },
        "loop_cadence": {
            "kind": "fixed_trace_snapshot_inner_loop",
            "outer_replay_cadence_bound": False,
            "teacher_labeled_traces": sum(
                1 for trace in traces if trace.teacher_frame is not None
            ),
            "note": (
                "target rounds reuse this fixed split; collecting another stream prefix "
                "is not part of the inner loop"
            ),
        },
        "workspace": str(workspace_root),
        "data_split": {key: len(value) for key, value in split.items()},
        "baseline": baseline,
        "rounds": round_results,
        "best_round": _best_round(round_results),
        "best_selection_round": _best_selection_round(round_results),
        "best_adoptable_round": _best_adoptable_round(round_results),
        "selection_decision": _selection_decision(round_results),
        "adoption_decision": _adoption_decision(round_results),
        "target_code_scope": "target/",
        "core_code_scope": "system/darjeeling/ is read-only evaluator/core",
        "target_code_policy": _target_code_policy_payload(),
        "private_data_scope": (
            "selection and promotion holdouts are stored outside the agent workspace"
        ),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_jsonl(commands_path, command_results)
    return summary


def split_l2_target_traces(traces: list[TeacherTrace]) -> dict[str, list[TeacherTrace]]:
    labeled = [trace for trace in traces if trace.teacher_frame is not None]
    if len(labeled) < 10:
        raise ValueError("target evolution requires at least 10 teacher-labeled traces")
    train_end = max(4, int(len(labeled) * 0.60))
    inner_end = max(train_end + 1, int(len(labeled) * 0.80))
    selection_end = max(inner_end + 1, int(len(labeled) * 0.90))
    selection_end = min(selection_end, len(labeled) - 1)
    inner_end = min(inner_end, selection_end - 1)
    return {
        "train": labeled[:train_end],
        "inner_validation": labeled[train_end:inner_end],
        "selection_holdout": labeled[inner_end:selection_end],
        "promotion_holdout": labeled[selection_end:],
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
    (workspace_root / "tools" / "search_config.py").write_text(
        _search_config_tool_text(),
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
        "private_data_files_not_in_workspace": [
            "selection_holdout.jsonl",
            "promotion_holdout.jsonl",
        ],
        "visible_state_files": [
            "objective.json",
            "round_state.json",
            "commands.md",
        ],
    }
    (workspace_root / "workspace_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def evaluate_target_workspace(
    *,
    workspace_root: Path,
    split: Literal["inner_validation", "selection_holdout", "promotion_holdout"],
    holdout_path: Path | None = None,
    min_accepted_accuracy: float = 0.93,
    max_wrong_accept_rate: float = 0.05,
) -> dict[str, Any]:
    train_traces = _read_teacher_jsonl(workspace_root / "data" / "train.jsonl")
    if split in {"selection_holdout", "promotion_holdout"}:
        if holdout_path is None:
            raise ValueError(f"{split} evaluation requires a private holdout_path")
        validation_path = holdout_path
    else:
        validation_path = workspace_root / "data" / f"{split}.jsonl"
    validation_traces = _read_teacher_jsonl(validation_path)
    target_module = load_target_module(workspace_root / "target" / "target_l2.py")
    overrides = target_config_overrides(target_module)
    config = L2StudentConfig(**overrides)
    bundle = train_l2_student(training_examples_from_teacher_traces(train_traces), config)

    accepted = 0
    correct = 0
    wrong = 0
    vetoed_accepts = 0
    examples: list[dict[str, Any]] = []
    veto_examples: list[dict[str, Any]] = []
    near_miss_examples: list[dict[str, Any]] = []
    for trace in validation_traces:
        if trace.teacher_frame is None:
            continue
        prediction = bundle.predict(trace.utterance)
        metadata = prediction.model_dump(mode="json")
        frame = target_postprocess_frame(
            target_module,
            utterance=trace.utterance,
            frame=prediction.frame,
            metadata=metadata,
        )
        default_accept = (
            bundle.config.runtime_enabled
            and prediction.guard_probability >= bundle.config.accept_threshold
        )
        should_accept = target_accept_prediction(
            target_module,
            utterance=trace.utterance,
            frame=frame,
            metadata=metadata,
            default_accept=default_accept,
        )
        if default_accept and not should_accept:
            vetoed_accepts += 1
            if len(veto_examples) < 8:
                veto_examples.append(
                    {
                        "request_id": trace.request_id,
                        "utterance": trace.utterance,
                        "teacher_frame": trace.teacher_frame.model_dump(mode="json"),
                        "predicted_frame": frame.model_dump(mode="json"),
                        "guard_probability": prediction.guard_probability,
                    }
                )
        if not default_accept:
            near_miss_examples.append(
                {
                    "request_id": trace.request_id,
                    "utterance": trace.utterance,
                    "teacher_frame": trace.teacher_frame.model_dump(mode="json"),
                    "predicted_frame": frame.model_dump(mode="json"),
                    "guard_probability": prediction.guard_probability,
                    "would_be_correct": frame == trace.teacher_frame,
                }
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
        "vetoed_accepts": vetoed_accepts,
        "coverage": accepted / total if total else 0.0,
        "accepted_accuracy": accepted_accuracy,
        "wrong_accept_rate": wrong_accept_rate,
        "passes_gate": passes_gate,
        "config": bundle.config.model_dump(mode="json"),
        "wrong_examples": examples,
        "veto_examples": veto_examples,
        "near_miss_examples": _top_guard_examples(near_miss_examples),
    }


def _evaluate_target_candidate(
    *,
    workspace_root: Path,
    private_paths: dict[str, Path],
    config: L2TargetEvolutionConfig,
    label: str,
) -> dict[str, Any]:
    inner_result = evaluate_target_workspace(
        workspace_root=workspace_root,
        split="inner_validation",
        min_accepted_accuracy=config.min_accepted_accuracy,
        max_wrong_accept_rate=config.max_wrong_accept_rate,
    )
    private_results = {
        split_name: evaluate_target_workspace(
            workspace_root=workspace_root,
            split=split_name,  # type: ignore[arg-type]
            holdout_path=path,
            min_accepted_accuracy=config.min_accepted_accuracy,
            max_wrong_accept_rate=config.max_wrong_accept_rate,
        )
        for split_name, path in private_paths.items()
    }
    return {
        "label": label,
        "inner_validation": inner_result,
        **private_results,
    }


def _write_target_state_files(
    *,
    workspace_root: Path,
    config: L2TargetEvolutionConfig,
    round_index: int,
    state_kind: Literal["before_round", "final"],
    baseline: dict[str, Any],
    round_results: list[dict[str, Any]],
    no_inner_improvement_rounds: int,
) -> None:
    data_dir = workspace_root / "data"
    objective = _target_objective_payload(config)
    state = {
        "schema_version": "l2-target-round-state-v1",
        "state_kind": state_kind,
        "next_round": round_index,
        "rounds_requested": config.rounds,
        "no_inner_improvement_rounds": no_inner_improvement_rounds,
        "budget_policy": {
            "inner_patience_rounds": config.inner_patience_rounds,
            "stop_on_selection_gate": config.stop_on_selection_gate,
            "local_search_trials": config.local_search_trials,
            "local_search_timeout_s": config.local_search_timeout_s,
            "local_search_space": config.local_search_space,
        },
        "candidate_selection_gate": (
            "visible inner validation gate and private selection holdout gate must both pass"
        ),
        "early_stop_policy": (
            "private selection is evaluated for outer candidate selection, but does not stop "
            "the inner loop unless stop_on_selection_gate is explicitly enabled"
        ),
        "baseline_inner_validation": _visible_metric_summary(baseline["inner_validation"]),
        "round_history": [_visible_round_summary(round_result) for round_result in round_results],
        "private_holdout_visibility": (
            "selection and promotion holdouts are not available in this workspace"
        ),
    }
    (data_dir / "objective.json").write_text(
        json.dumps(objective, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (data_dir / "round_state.json").write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (data_dir / "commands.md").write_text(_target_commands_text(), encoding="utf-8")


def _target_objective_payload(config: L2TargetEvolutionConfig) -> dict[str, Any]:
    return {
        "schema_version": "l2-target-objective-v1",
        "primary_objective": "increase safe L2 accepts on unseen target traffic",
        "gates": {
            "min_accepted_accuracy": config.min_accepted_accuracy,
            "max_wrong_accept_rate": config.max_wrong_accept_rate,
            "candidate_selection": (
                "visible inner validation gate AND private selection holdout gate"
            ),
            "adoption": (
                "visible inner validation gate AND private selection holdout gate "
                "AND private promotion holdout gate"
            ),
        },
        "optimization_order": [
            "zero or lower wrong accepts",
            "visible inner validation gate must pass before candidate selection",
            "accepted accuracy at or above gate",
            "coverage increase only after safety gates",
            "lower latency for equally safe behavior",
        ],
        "invalid_strategies": [
            "raw coverage increase with lower frame exactness",
            "lowering threshold when visible inner validation gate fails",
            "treating private selection success alone as candidate success",
            "changes outside target/",
            "using private holdout rows or aggregate feedback",
            "hardcoding MASSIVE-specific behavior from outside visible data",
        ],
        "allowed_strategies": [
            "target-dependent code derived from visible train/inner validation files",
            "config_overrides for bounded L2StudentConfig parameters",
            "local Optuna/config search over visible train and inner validation only",
            "postprocess_frame fixes that preserve exact frame correctness",
            "accept_prediction veto logic that abstains when uncertain",
            "near_miss_examples-driven mechanisms that still pass visible inner gate",
            "target-specific lexical or state-machine rules derived from visible target data",
        ],
    }


def _target_code_policy_payload() -> dict[str, Any]:
    return {
        "core_must_remain_dataset_independent": True,
        "target_dependent_code_allowed_in": "target/",
        "target_specific_code_is_not_rejected_for_dataset_dependence": True,
        "target_code_visibility_rule": (
            "target code may be derived from data/train.jsonl and "
            "data/inner_validation.jsonl only"
        ),
        "private_holdout_visibility": (
            "selection/promotion holdouts remain outside the agent workspace"
        ),
        "adoption_authority": (
            "visible inner gate, private selection/promotion gates, and final outer replay"
        ),
    }


def _target_commands_text() -> str:
    return "\n".join(
        [
            "# Commands",
            "",
            "Evaluate the visible inner-validation split:",
            "",
            "```bash",
            "uv run --project system/darjeeling python tools/evaluate.py \\",
            "  --split inner_validation \\",
            "  --out runs/inner_validation.json",
            "```",
            "",
            "If `uv` cannot use its dependency cache but a Python >=3.11 environment",
            "with Darjeeling dependencies is already active, use:",
            "",
            "```bash",
            "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=system/darjeeling/src \\",
            "  python tools/evaluate.py --split inner_validation --out runs/inner_validation.json",
            "```",
            "",
            "Run local Optuna config search on visible train/inner validation only:",
            "",
            "```bash",
            "uv run --project system/darjeeling python tools/search_config.py \\",
            f"  --trials {DEFAULT_TARGET_LOCAL_SEARCH_TRIALS} \\",
            "  --out runs/local_search.json",
            "```",
            "",
            "If Darjeeling dependencies are already active, avoid a nested uv env:",
            "",
            "```bash",
            "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=system/darjeeling/src \\",
            (
                "  python tools/search_config.py "
                f"--trials {DEFAULT_TARGET_LOCAL_SEARCH_TRIALS} "
                "--out runs/local_search.json"
            ),
            "```",
            "",
            "Inspect visible workspace context:",
            "",
            "```bash",
            "python3 tools/inspect_context.py",
            "```",
            "",
            "Only edit files under `target/`.",
        ]
    )


def _visible_round_summary(round_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "round": round_result["round"],
        "inner_improved": round_result["inner_improved"],
        "passes_candidate_selection_gate": round_result["passes_candidate_selection_gate"],
        "inner_score": round_result["inner_score"],
        "inner_delta_vs_baseline": round_result["inner_delta_vs_baseline"],
        "inner_validation": _visible_metric_summary(round_result["inner_validation"]),
    }


def _visible_metric_summary(metric: dict[str, Any]) -> dict[str, Any]:
    return {
        "split": metric["split"],
        "train_size": metric["train_size"],
        "validation_size": metric["validation_size"],
        "accepted": metric["accepted"],
        "correct_accepts": metric["correct_accepts"],
        "wrong_accepts": metric["wrong_accepts"],
        "vetoed_accepts": metric["vetoed_accepts"],
        "coverage": metric["coverage"],
        "accepted_accuracy": metric["accepted_accuracy"],
        "wrong_accept_rate": metric["wrong_accept_rate"],
        "passes_gate": metric["passes_gate"],
        "veto_examples": metric.get("veto_examples", []),
        "near_miss_examples": metric.get("near_miss_examples", []),
    }


def _metric_delta(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    return {
        "accepted": current["accepted"] - baseline["accepted"],
        "correct_accepts": current["correct_accepts"] - baseline["correct_accepts"],
        "wrong_accepts": current["wrong_accepts"] - baseline["wrong_accepts"],
        "vetoed_accepts": current["vetoed_accepts"] - baseline["vetoed_accepts"],
        "coverage": current["coverage"] - baseline["coverage"],
        "accepted_accuracy": _optional_float_delta(
            current["accepted_accuracy"],
            baseline["accepted_accuracy"],
        ),
        "wrong_accept_rate": current["wrong_accept_rate"] - baseline["wrong_accept_rate"],
    }


def _top_guard_examples(examples: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    return sorted(
        examples,
        key=lambda example: float(example["guard_probability"]),
        reverse=True,
    )[:limit]


def _optional_float_delta(current: float | None, baseline: float | None) -> float | None:
    if current is None or baseline is None:
        return None
    return current - baseline


def _is_inner_improvement(current: dict[str, Any], best: dict[str, Any]) -> bool:
    return _inner_score(current) > _inner_score(best)


def _inner_score(metric: dict[str, Any]) -> tuple[bool, int, float, float]:
    return (
        bool(metric["passes_gate"]),
        -int(metric["wrong_accepts"]),
        float(metric["accepted_accuracy"] or 0.0),
        float(metric["coverage"]),
    )


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


def run_local_target_search(
    *,
    workspace_root: Path,
    trials: int,
    search_space: L2TargetSearchSpace = "compact",
    timeout_s: float | None = None,
    min_accepted_accuracy: float = 0.93,
    max_wrong_accept_rate: float = 0.05,
) -> dict[str, Any]:
    """Tune target-owned L2 config using only visible train/inner validation data."""

    if trials < 1:
        raise ValueError("trials must be at least 1")
    if search_space not in {"compact", "wide"}:
        raise ValueError("search_space must be compact or wide")

    config_path = workspace_root / "target" / "config.json"
    original_config_text = config_path.read_text(encoding="utf-8") if config_path.exists() else None
    current_inner = evaluate_target_workspace(
        workspace_root=workspace_root,
        split="inner_validation",
        min_accepted_accuracy=min_accepted_accuracy,
        max_wrong_accept_rate=max_wrong_accept_rate,
    )
    current_config = L2StudentConfig(**current_inner["config"])

    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    trial_reports: dict[int, dict[str, Any]] = {}

    def objective(trial: Any) -> float:
        candidate_config = _sample_local_search_config(
            trial,
            base_config=current_config,
            search_space=search_space,
        )
        _write_target_config_json(config_path, candidate_config)
        try:
            metric = evaluate_target_workspace(
                workspace_root=workspace_root,
                split="inner_validation",
                min_accepted_accuracy=min_accepted_accuracy,
                max_wrong_accept_rate=max_wrong_accept_rate,
            )
            value = _local_search_objective_value(metric)
            trial_reports[trial.number] = {
                "number": trial.number,
                "state": "COMPLETE",
                "value": value,
                "params": dict(trial.params),
                "config": candidate_config.model_dump(mode="json"),
                "inner_validation": _visible_metric_summary(metric),
            }
            return value
        except Exception as exc:
            trial_reports[trial.number] = {
                "number": trial.number,
                "state": "FAIL",
                "value": -1_000_000.0,
                "params": dict(trial.params),
                "config": candidate_config.model_dump(mode="json"),
                "error": str(exc),
            }
            return -1_000_000.0

    try:
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=current_config.random_state),
        )
        study.optimize(objective, n_trials=trials, timeout=timeout_s)
        reports = [
            trial_reports.get(
                trial.number,
                {
                    "number": trial.number,
                    "state": trial.state.name,
                    "value": trial.value,
                    "params": dict(trial.params),
                },
            )
            for trial in study.trials
        ]
        completed = [
            report
            for report in reports
            if (
                report.get("state") == "COMPLETE"
                and report.get("config") is not None
                and report.get("inner_validation") is not None
            )
        ]
        best_report = max(
            completed,
            key=lambda report: (
                _inner_score(report["inner_validation"]),
                float(report["value"] or -1_000_000.0),
            ),
            default=None,
        )
        applied = False
        applied_reason = "no completed local-search trial"
        if best_report is not None:
            best_inner = best_report["inner_validation"]
            if _inner_score(best_inner) > _inner_score(current_inner):
                _write_target_config_json(
                    config_path,
                    L2StudentConfig(**best_report["config"]),
                )
                applied = True
                applied_reason = "best visible inner-validation config improved current target"
            else:
                _restore_target_config_json(config_path, original_config_text)
                applied_reason = (
                    "best visible inner-validation config did not improve current target"
                )
        else:
            _restore_target_config_json(config_path, original_config_text)
        return {
            "schema_version": "l2-target-local-search-v1",
            "search_space": search_space,
            "trials_requested": trials,
            "trials_completed": len(completed),
            "timeout_s": timeout_s,
            "current_inner_validation": _visible_metric_summary(current_inner),
            "best_trial_number": best_report["number"] if best_report is not None else None,
            "best_value": best_report["value"] if best_report is not None else None,
            "best_config": best_report["config"] if best_report is not None else None,
            "best_inner_validation": (
                best_report["inner_validation"] if best_report is not None else None
            ),
            "applied": applied,
            "applied_reason": applied_reason,
            "private_holdout_visibility": (
                "local search used only visible train and inner_validation data"
            ),
            "trials": reports,
        }
    except Exception:
        _restore_target_config_json(config_path, original_config_text)
        raise


def local_search_target_workspace_cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--trials", type=int, default=48)
    parser.add_argument("--search-space", choices=["compact", "wide"], default="compact")
    parser.add_argument("--timeout-s", type=float, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-accepted-accuracy", type=float, default=0.93)
    parser.add_argument("--max-wrong-accept-rate", type=float, default=0.05)
    args = parser.parse_args(argv)
    payload = run_local_target_search(
        workspace_root=args.workspace,
        trials=args.trials,
        search_space=args.search_space,
        timeout_s=args.timeout_s,
        min_accepted_accuracy=args.min_accepted_accuracy,
        max_wrong_accept_rate=args.max_wrong_accept_rate,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


def _sample_local_search_config(
    trial: Any,
    *,
    base_config: L2StudentConfig,
    search_space: L2TargetSearchSpace,
) -> L2StudentConfig:
    max_features_choices = (
        [1_000, 3_000, 5_000, 10_000]
        if search_space == "compact"
        else [1_000, 3_000, 5_000, 10_000, 25_000, 50_000]
    )
    max_iter_choices = [200, 300, 500] if search_space == "compact" else [200, 300, 500, 1000]
    threshold_choices = (
        [0.70, 0.80, 0.86, 0.90, 0.93, 0.96, 0.98]
        if search_space == "compact"
        else [0.50, 0.60, 0.70, 0.80, 0.86, 0.90, 0.93, 0.96, 0.98]
    )
    intent_model_family = trial.suggest_categorical(
        "intent_model_family",
        ["sgd_logreg"] if search_space == "compact" else ["sgd_logreg", "mlp"],
    )
    hidden_size_text = trial.suggest_categorical(
        "mlp_hidden_layer_sizes",
        ["32", "64", "128", "64,32"] if search_space == "wide" else ["32", "64"],
    )
    char_lower = trial.suggest_int("char_ngram_lower", 2, 4)
    char_upper = trial.suggest_int(
        "char_ngram_upper",
        char_lower,
        6 if search_space == "wide" else 5,
    )
    return base_config.model_copy(
        update={
            "accept_threshold": trial.suggest_categorical(
                "accept_threshold",
                threshold_choices,
            ),
            "runtime_enabled": True,
            "frame_source": trial.suggest_categorical(
                "frame_source",
                ["retrieval", "student"],
            ),
            "intent_model_family": intent_model_family,
            "slot_model_family": trial.suggest_categorical(
                "slot_model_family",
                ["token_sgd"] if search_space == "compact" else ["token_sgd", "none"],
            ),
            "max_features": int(
                trial.suggest_categorical("max_features", max_features_choices),
            ),
            "max_iter": int(trial.suggest_categorical("max_iter", max_iter_choices)),
            "word_ngram_range": (
                1,
                trial.suggest_int(
                    "word_ngram_upper",
                    1,
                    4 if search_space == "wide" else 3,
                ),
            ),
            "char_ngram_range": (char_lower, char_upper),
            "mlp_hidden_layer_sizes": tuple(
                int(part) for part in hidden_size_text.split(",") if part
            ),
            "mlp_alpha": trial.suggest_float("mlp_alpha", 1e-5, 1e-2, log=True),
            "mlp_early_stopping": False,
        },
    )


def _write_target_config_json(path: Path, config: L2StudentConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _restore_target_config_json(path: Path, original_text: str | None) -> None:
    if original_text is None:
        path.unlink(missing_ok=True)
        return
    path.write_text(original_text, encoding="utf-8")


def _local_search_objective_value(metric: dict[str, Any]) -> float:
    return (
        (100.0 if metric["passes_gate"] else 0.0)
        - 10.0 * float(metric["wrong_accepts"])
        + 2.0 * float(metric["accepted_accuracy"] or 0.0)
        + float(metric["coverage"])
    )


def _local_search_command_result(
    *,
    workspace_root: Path,
    round_index: int,
    report_path: Path,
    report: dict[str, Any],
) -> dict[str, Any]:
    return {
        "command": [
            "local-search",
            "--round",
            str(round_index),
            "--trials",
            str(report["trials_requested"]),
            "--search-space",
            str(report["search_space"]),
        ],
        "cwd": str(workspace_root),
        "started_at": datetime.now(UTC).isoformat(),
        "return_code": 0,
        "stdout": json.dumps(
            {
                "report": str(report_path),
                "trials_completed": report["trials_completed"],
                "applied": report["applied"],
                "best_trial_number": report["best_trial_number"],
            },
            sort_keys=True,
        ),
        "stderr": "",
    }


def _failed_local_search_command_result(
    *,
    workspace_root: Path,
    round_index: int,
    error: str,
) -> dict[str, Any]:
    return {
        "command": ["local-search", "--round", str(round_index)],
        "cwd": str(workspace_root),
        "started_at": datetime.now(UTC).isoformat(),
        "return_code": 1,
        "stdout": "",
        "stderr": error,
    }


def _visible_local_search_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": report["schema_version"],
        "search_space": report["search_space"],
        "trials_requested": report["trials_requested"],
        "trials_completed": report["trials_completed"],
        "best_trial_number": report["best_trial_number"],
        "best_value": report["best_value"],
        "best_inner_validation": report["best_inner_validation"],
        "applied": report["applied"],
        "applied_reason": report["applied_reason"],
        "private_holdout_visibility": report["private_holdout_visibility"],
    }


def _copy_system_workspace(source_repo_dir: Path, system_repo_dir: Path) -> None:
    if system_repo_dir.exists():
        shutil.rmtree(system_repo_dir)
    system_repo_dir.mkdir(parents=True, exist_ok=True)
    for rel_path in [
        Path("src"),
        Path("tests"),
        Path("pyproject.toml"),
        Path("uv.lock"),
        Path("README.md"),
    ]:
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

import json
from pathlib import Path
from typing import Any


def config_overrides() -> dict[str, Any]:
    """Return target-specific L2StudentConfig overrides."""
    config_path = Path(__file__).with_name("config.json")
    if config_path.exists():
        return json.loads(config_path.read_text(encoding="utf-8"))
    return {}


def postprocess_frame(
    utterance: str,
    frame: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Return a target-specific frame dict after the core L2 model predicts."""
    del utterance, metadata
    return frame


def accept_prediction(
    utterance: str,
    frame: dict[str, Any],
    metadata: dict[str, Any],
    default_accept: bool,
) -> bool | None:
    """Return False to veto a guard accept; True/None keep the default decision."""
    del utterance, frame, metadata
    return default_accept
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
            "- `target/config.json` is local-search-owned L2StudentConfig overrides.",
            "- `system/darjeeling/` is read-only Darjeeling core/evaluator code.",
            "- `data/train.jsonl` is visible training data.",
            "- `data/inner_validation.jsonl` is visible fast feedback data.",
            "- `data/objective.json` defines gates and invalid strategies.",
            "- `data/round_state.json` contains visible inner-validation history.",
            "- `data/commands.md` lists local commands.",
            "- `tools/evaluate.py` trains/evaluates the target code in seconds.",
            "- `tools/search_config.py` runs visible-data Optuna config search.",
            "",
            "Optimize generalization from the visible train and inner-validation data.",
            "Wrong accepts are worse than abstentions. A raw coverage increase is not",
            "useful if frame exactness or wrong-accept safety gets worse.",
            "A target round is selectable only if visible inner validation passes",
            "and the outer private selection holdout passes. Private selection",
            "alone is not success if visible inner validation has wrong accepts.",
            "Adoption also requires the private promotion holdout to pass.",
            "By default, private selection is an outer selection signal, not an",
            "inner-loop early-stop signal; keep improving target code until the",
            "round budget or visible inner-validation patience is exhausted.",
            "Use `near_miss_examples` from visible inner validation to find safe",
            "coverage opportunities, and use `wrong_examples` / `veto_examples`",
            "to tighten safety. Do not lower threshold globally unless the visible",
            "inner gate still passes.",
            "`target.accept_prediction` may veto uncertain guard accepts; it cannot",
            "force accepts that the core guard rejected.",
            "Private selection and promotion holdouts are outside this workspace and",
            "only the outer harness can read them; do not try to access parent",
            "directories to inspect them.",
            "",
            "It is acceptable for `target/` to contain target-dependent code derived",
            "from visible train and inner-validation data. This is not a",
            "Darjeeling-core dataset-independence violation. It becomes invalid only",
            "if it moves into core code, uses private holdout rows or aggregates, or",
            "uses MASSIVE/external dataset knowledge that is not visible here.",
            "Use local config search for cheap tuning; reserve code edits for changes",
            "that require target-specific design judgment.",
            "Do not move target-specific code into core.",
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


def _search_config_tool_text() -> str:
    return """from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYSTEM_SRC = ROOT / "system" / "darjeeling" / "src"
sys.path.insert(0, str(SYSTEM_SRC))

from darjeeling.compiler.l2_target_evolution import local_search_target_workspace_cli

if __name__ == "__main__":
    raise SystemExit(local_search_target_workspace_cli())
"""


def _inspect_tool_text() -> str:
    return """from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in sorted((ROOT / "data").iterdir()):
    if path.is_file():
        print(f"{path.name}: {path.stat().st_size} bytes")
print("")
print("Editable target files:")
for path in sorted((ROOT / "target").iterdir()):
    if path.is_file():
        print(f"{path.name}: {path.stat().st_size} bytes")
"""


def _read_teacher_jsonl(path: Path) -> list[TeacherTrace]:
    return [
        TeacherTrace.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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
    return _best_round_for_split(rounds, "selection_holdout")


def _best_round_for_split(rounds: list[dict[str, Any]], split: str) -> dict[str, Any] | None:
    if not rounds:
        return None
    return max(
        rounds,
        key=lambda item: (
            bool(item[split]["passes_gate"]),
            item[split]["coverage"],
            item[split]["accepted_accuracy"] or 0.0,
            -item[split]["wrong_accept_rate"],
        ),
    )


def _best_selection_round(rounds: list[dict[str, Any]]) -> dict[str, Any] | None:
    passing_rounds = [
        round_result
        for round_result in rounds
        if (
            round_result["inner_validation"]["passes_gate"]
            and round_result["selection_holdout"]["passes_gate"]
        )
    ]
    if not passing_rounds:
        return None
    return _best_round_for_split(passing_rounds, "selection_holdout")


def _best_adoptable_round(rounds: list[dict[str, Any]]) -> dict[str, Any] | None:
    passing_rounds = [
        round_result
        for round_result in rounds
        if (
            round_result["inner_validation"]["passes_gate"]
            and round_result["selection_holdout"]["passes_gate"]
            and round_result["promotion_holdout"]["passes_gate"]
        )
    ]
    if not passing_rounds:
        return None
    return _best_round_for_split(passing_rounds, "promotion_holdout")


def _selection_decision(rounds: list[dict[str, Any]]) -> dict[str, Any]:
    best_selection = _best_selection_round(rounds)
    if best_selection is None:
        return {
            "selected": False,
            "round": None,
            "reason": (
                "no target round passed both visible inner and private selection gates"
            ),
        }
    return {
        "selected": True,
        "round": best_selection["round"],
        "reason": "target round passed both visible inner and private selection gates",
    }


def _adoption_decision(rounds: list[dict[str, Any]]) -> dict[str, Any]:
    selection = _selection_decision(rounds)
    if not selection["selected"]:
        return {
            "adopted": False,
            "round": None,
            "reason": (
                "no target round passed both visible inner and private selection gates"
            ),
        }
    best_adoptable = _best_adoptable_round(rounds)
    if best_adoptable is None:
        return {
            "adopted": False,
            "round": selection["round"],
            "reason": (
                "best selected target round failed the private promotion holdout gate"
            ),
        }
    return {
        "adopted": True,
        "round": best_adoptable["round"],
        "reason": "target round passed both private selection and promotion gates",
    }
