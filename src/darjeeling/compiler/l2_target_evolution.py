from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
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
from darjeeling.schemas import Frame, TeacherTrace

L2TargetEvolutionMode = Literal[
    "dry-run",
    "codex-cli",
    "local-search",
    "agent-session",
]
L2TargetSearchSpace = Literal["compact", "wide"]
L2TargetBudgetProfile = Literal["standard", "fixed-inner", "smoke"]
L2TargetSplitPolicy = Literal["chronological", "intent-stratified"]
L2TargetScope = Literal["teacher_train", "lower_miss"]

DEFAULT_TARGET_EVOLVE_ROUNDS = 12
DEFAULT_TARGET_LOCAL_SEARCH_TRIALS = 96
DEFAULT_TARGET_INNER_PATIENCE_ROUNDS = 4
DEFAULT_TARGET_VISIBLE_CROSS_AUDIT_FOLDS = 3
DEFAULT_TARGET_LOCAL_SEARCH_CROSS_AUDIT_TOP_K = 4
MIN_VISIBLE_CORRECT_ACCEPTS_PER_VALIDATION_FOLD = 2


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
    local_search_cross_audit_top_k: int = 0
    budget_profile: L2TargetBudgetProfile = "standard"
    split_policy: L2TargetSplitPolicy = "chronological"
    target_scope: L2TargetScope = "teacher_train"
    visible_validation_folds: int = 1
    visible_validation_ratio: float | None = None
    visible_cross_audit_folds: int = 0
    max_agent_rounds: int | None = None
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
    if config.local_search_cross_audit_top_k < 0:
        raise ValueError("local_search_cross_audit_top_k must be non-negative")
    if config.visible_validation_folds < 1:
        raise ValueError("visible_validation_folds must be at least 1")
    if config.visible_validation_ratio is not None and not (
        0.0 < config.visible_validation_ratio < 0.80
    ):
        raise ValueError("visible_validation_ratio must be greater than 0 and less than 0.80")
    if config.visible_cross_audit_folds == 1 or config.visible_cross_audit_folds < 0:
        raise ValueError("visible_cross_audit_folds must be 0 or at least 2")
    if config.max_agent_rounds is not None and config.max_agent_rounds < 0:
        raise ValueError("max_agent_rounds must be non-negative")
    if config.mode not in {"dry-run", "local-search", "codex-cli", "agent-session"}:
        raise ValueError("mode must be dry-run, local-search, codex-cli, or agent-session")
    if config.target_scope not in {"teacher_train", "lower_miss"}:
        raise ValueError("target_scope must be teacher_train or lower_miss")
    max_agent_rounds = _effective_max_agent_rounds(config)
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

    teacher_labeled_trace_count = sum(
        1 for trace in traces if trace.teacher_frame is not None
    )
    target_traces = l2_target_traces_for_scope(traces, scope=config.target_scope)
    target_teacher_labeled_trace_count = sum(
        1 for trace in target_traces if trace.teacher_frame is not None
    )
    split = split_l2_target_traces(
        target_traces,
        policy=config.split_policy,
        visible_validation_folds=config.visible_validation_folds,
        visible_validation_ratio=config.visible_validation_ratio,
    )
    target_scope_payload = _target_scope_payload(
        scope=config.target_scope,
        input_teacher_labeled_traces=teacher_labeled_trace_count,
        scoped_teacher_labeled_traces=target_teacher_labeled_trace_count,
    )
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
    agent_rounds_started = 0
    agent_rounds_succeeded = 0
    if (
        config.stop_on_selection_gate
        and _passes_visible_selection_inputs(baseline)
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
            target_scope=target_scope_payload,
            baseline=baseline,
            round_results=round_results,
            no_inner_improvement_rounds=no_inner_improvement_rounds,
            agent_rounds_started=agent_rounds_started,
            agent_rounds_succeeded=agent_rounds_succeeded,
        )
        protected_snapshot = _protected_workspace_snapshot(workspace_root)
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
                scope_report = _workspace_scope_violation_report(
                    workspace_root=workspace_root,
                    before=protected_snapshot,
                )
                if scope_report is not None:
                    command_results.append(
                        _workspace_scope_violation_command_result(
                            workspace_root=workspace_root,
                            round_index=round_index,
                            report=scope_report,
                        ),
                    )
                    stop_reason = "workspace_scope_violation"
                    break
        elif config.mode == "local-search":
            search_report_path = rounds_dir / f"round_{round_index:03d}_local_search.json"
            try:
                local_search_report = run_local_target_search(
                    workspace_root=workspace_root,
                    trials=config.local_search_trials,
                    search_space=config.local_search_space,
                    timeout_s=config.local_search_timeout_s,
                    cross_audit_folds=config.visible_cross_audit_folds,
                    cross_audit_top_k=config.local_search_cross_audit_top_k,
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
                scope_report = _workspace_scope_violation_report(
                    workspace_root=workspace_root,
                    before=protected_snapshot,
                )
                if scope_report is not None:
                    command_results.append(
                        _workspace_scope_violation_command_result(
                            workspace_root=workspace_root,
                            round_index=round_index,
                            report=scope_report,
                        ),
                    )
                    stop_reason = "workspace_scope_violation"
                    break
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
        elif config.mode == "agent-session":
            if (
                max_agent_rounds is not None
                and agent_rounds_started >= max_agent_rounds
            ):
                stop_reason = "agent_session_budget_exhausted"
                break
            transcript_path = transcript_dir / "agent_session.jsonl"
            report_path = rounds_dir / "agent_session_report.md"
            agent_rounds_started += 1
            command_results.append(
                _run_agent_session(
                    config=config,
                    workspace_root=workspace_root,
                    transcript_path=transcript_path,
                    report_path=report_path,
                )
            )
            if command_results[-1]["return_code"] != 0:
                stop_reason = "agent_session_failed"
                break
            scope_report = _workspace_scope_violation_report(
                workspace_root=workspace_root,
                before=protected_snapshot,
            )
            if scope_report is not None:
                command_results.append(
                    _workspace_scope_violation_command_result(
                        workspace_root=workspace_root,
                        round_index=round_index,
                        report=scope_report,
                    ),
                )
                stop_reason = "workspace_scope_violation"
                break
            agent_rounds_succeeded += 1
            local_search_report = None
        else:
            if (
                max_agent_rounds is not None
                and agent_rounds_started >= max_agent_rounds
            ):
                stop_reason = "agent_round_budget_exhausted"
                break
            transcript_path = transcript_dir / f"round_{round_index:03d}.jsonl"
            report_path = rounds_dir / f"round_{round_index:03d}_agent_report.md"
            agent_rounds_started += 1
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
            scope_report = _workspace_scope_violation_report(
                workspace_root=workspace_root,
                before=protected_snapshot,
            )
            if scope_report is not None:
                command_results.append(
                    _workspace_scope_violation_command_result(
                        workspace_root=workspace_root,
                        round_index=round_index,
                        report=scope_report,
                    ),
                )
                stop_reason = "workspace_scope_violation"
                break
            agent_rounds_succeeded += 1
            local_search_report = None

        candidate = _evaluate_target_candidate(
            workspace_root=workspace_root,
            private_paths=private_paths,
            config=config,
            label=f"round_{round_index:03d}",
        )
        inner_result = candidate["inner_validation"]
        visible_support_gate = _visible_support_gate_payload(inner_result)
        train_audit_safety_passed = _passes_train_audit_safety_gate(candidate)
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
            "target_snapshot": _snapshot_target_dir(
                workspace_root=workspace_root,
                rounds_dir=rounds_dir,
                job_dir=job_dir,
                round_index=round_index,
            ),
            "inner_improved": inner_improved,
            "passes_visible_support_gate": visible_support_gate["passes_gate"],
            "visible_support_gate": visible_support_gate,
            "passes_train_audit_safety_gate": train_audit_safety_passed,
            "passes_candidate_selection_gate": _passes_candidate_selection_gate(
                candidate,
            ),
            "passes_private_selection_gate": bool(selection_result["passes_gate"]),
            "passes_private_promotion_gate": bool(promotion_result["passes_gate"]),
            "inner_score": list(_inner_score(inner_result)),
            "inner_delta_vs_baseline": _metric_delta(
                inner_result,
                baseline["inner_validation"],
            ),
            "inner_validation": inner_result,
            "train_audit": _candidate_train_audit(candidate),
            "visible_cross_audit": candidate.get("visible_cross_audit"),
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
        if config.mode == "agent-session":
            round_payload["agent_session"] = {
                "schema_version": "l2-target-agent-session-v1",
                "session_scope": "single long-running L4 agent session",
                "internal_loop_control": "agent_decides_edit_evaluate_search_stop",
                "tool_policy": (
                    "agent may call visible tools/evaluate.py and tools/search_config.py"
                ),
                "private_holdout_visibility": (
                    "selection and promotion holdouts are evaluated only after session exit"
                ),
            }
        (rounds_dir / f"round_{round_index:03d}.json").write_text(
            json.dumps(round_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        round_results.append(round_payload)
        if config.mode == "agent-session":
            stop_reason = "agent_session_completed"
            break
        if (
            config.stop_on_selection_gate
            and _passes_visible_selection_inputs(candidate)
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
        target_scope=target_scope_payload,
        baseline=baseline,
        round_results=round_results,
        no_inner_improvement_rounds=no_inner_improvement_rounds,
        agent_rounds_started=agent_rounds_started,
        agent_rounds_succeeded=agent_rounds_succeeded,
    )

    summary = {
        "schema_version": "l2-target-evolution-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "mode": config.mode,
        "rounds_requested": config.rounds,
        "rounds_completed": len(round_results),
        "stop_reason": stop_reason,
        "budget_policy": _target_budget_policy_payload(
            config,
            max_agent_rounds=max_agent_rounds,
        ),
        "evidence_policy": _target_evidence_policy_payload(
            config,
            max_agent_rounds=max_agent_rounds,
            rounds_completed=len(round_results),
            stop_reason=stop_reason,
            teacher_labeled_traces=target_teacher_labeled_trace_count,
        ),
        "agent_budget": _agent_budget_payload(
            config=config,
            agent_rounds_started=agent_rounds_started,
            agent_rounds_succeeded=agent_rounds_succeeded,
        ),
        "loop_cadence": {
            "kind": "fixed_trace_snapshot_inner_loop",
            "outer_replay_cadence_bound": False,
            "teacher_labeled_traces": teacher_labeled_trace_count,
            "scoped_teacher_labeled_traces": target_teacher_labeled_trace_count,
            "note": (
                "target rounds reuse this fixed split; collecting another stream prefix "
                "is not part of the inner loop"
            ),
        },
        "workspace": str(workspace_root),
        "target_scope": target_scope_payload,
        "data_split": {key: len(value) for key, value in split.items()},
        "data_split_policy": _target_split_policy_payload(
            config.split_policy,
            split,
            visible_validation_ratio=config.visible_validation_ratio,
        ),
        "baseline": baseline,
        "rounds": round_results,
        "best_round": _best_round(round_results),
        "best_selection_round": _best_selection_round(round_results),
        "best_adoptable_round": _best_adoptable_round(round_results),
        "selection_decision": _selection_decision(round_results),
        "adoption_decision": _adoption_decision(round_results),
        "private_holdout_evidence": _private_holdout_evidence(round_results),
        "target_code_scope": "target/",
        "core_code_scope": "system/darjeeling/ is read-only evaluator/core",
        "target_code_policy": _target_code_policy_payload(),
        "workspace_scope_policy": _workspace_scope_policy_payload(),
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


def split_l2_target_traces(
    traces: list[TeacherTrace],
    *,
    policy: L2TargetSplitPolicy = "chronological",
    visible_validation_folds: int = 1,
    visible_validation_ratio: float | None = None,
) -> dict[str, list[TeacherTrace]]:
    if visible_validation_folds < 1:
        raise ValueError("visible_validation_folds must be at least 1")
    if visible_validation_ratio is not None and not (0.0 < visible_validation_ratio < 0.80):
        raise ValueError(
            "visible_validation_ratio must be greater than 0 and less than 0.80",
        )
    labeled = [trace for trace in traces if trace.teacher_frame is not None]
    if len(labeled) < 10:
        raise ValueError("target evolution requires at least 10 teacher-labeled traces")
    if policy == "intent-stratified":
        return _split_l2_target_traces_by_intent(
            labeled,
            visible_validation_folds=visible_validation_folds,
            visible_validation_ratio=visible_validation_ratio,
        )
    if policy != "chronological":
        raise ValueError(f"unsupported target split policy: {policy}")
    return _split_l2_target_traces_chronological(
        labeled,
        visible_validation_folds=visible_validation_folds,
        visible_validation_ratio=visible_validation_ratio,
    )


def l2_target_traces_for_scope(
    traces: list[TeacherTrace],
    *,
    scope: L2TargetScope = "teacher_train",
) -> list[TeacherTrace]:
    if scope == "teacher_train":
        return traces
    if scope == "lower_miss":
        return [
            trace
            for trace in traces
            if trace.teacher_frame is not None and not _lower_layer_accepted(trace)
        ]
    raise ValueError(f"unsupported L2 target scope: {scope}")


def _lower_layer_accepted(trace: TeacherTrace) -> bool:
    return any(
        result.layer in {"L0", "L1"} and result.accepted and result.frame is not None
        for result in trace.layer_results
    )


def _target_scope_payload(
    *,
    scope: L2TargetScope,
    input_teacher_labeled_traces: int,
    scoped_teacher_labeled_traces: int,
) -> dict[str, Any]:
    return {
        "schema_version": "l2-target-scope-v1",
        "scope": scope,
        "input_teacher_labeled_traces": input_teacher_labeled_traces,
        "scoped_teacher_labeled_traces": scoped_teacher_labeled_traces,
        "lower_layer_accepted_excluded": (
            input_teacher_labeled_traces - scoped_teacher_labeled_traces
            if scope == "lower_miss"
            else 0
        ),
        "selection_basis": (
            "teacher-labeled traces where L0/L1 did not accept"
            if scope == "lower_miss"
            else "all teacher-labeled traces"
        ),
    }


def _split_l2_target_traces_chronological(
    labeled: list[TeacherTrace],
    *,
    visible_validation_folds: int = 1,
    visible_validation_ratio: float | None = None,
) -> dict[str, list[TeacherTrace]]:
    if visible_validation_folds == 1 and visible_validation_ratio is None:
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

    private_selection_count = max(1, int(len(labeled) * 0.10))
    private_promotion_count = max(1, int(len(labeled) * 0.10))
    visible_ratio = _resolve_visible_validation_ratio(
        visible_validation_folds=visible_validation_folds,
        visible_validation_ratio=visible_validation_ratio,
    )
    visible_total = max(visible_validation_folds, int(len(labeled) * visible_ratio))
    max_visible_total = len(labeled) - private_selection_count - private_promotion_count - 4
    if max_visible_total < visible_validation_folds:
        return _split_l2_target_traces_chronological(
            labeled,
            visible_validation_folds=1,
            visible_validation_ratio=visible_validation_ratio,
        )
    visible_total = min(visible_total, max_visible_total)
    train_end = len(labeled) - visible_total - private_selection_count - private_promotion_count
    visible_counts = _distribute_count(visible_total, visible_validation_folds)
    split: dict[str, list[TeacherTrace]] = {"train": labeled[:train_end]}
    cursor = train_end
    for split_name, count in zip(
        _visible_validation_split_names(visible_validation_folds),
        visible_counts,
        strict=True,
    ):
        split[split_name] = labeled[cursor : cursor + count]
        cursor += count
    selection_end = cursor + private_selection_count
    split["selection_holdout"] = labeled[cursor:selection_end]
    split["promotion_holdout"] = labeled[selection_end:]
    return split


def _split_l2_target_traces_by_intent(
    labeled: list[TeacherTrace],
    *,
    visible_validation_folds: int = 1,
    visible_validation_ratio: float | None = None,
) -> dict[str, list[TeacherTrace]]:
    split: dict[str, list[TeacherTrace]] = {
        "train": [],
    }
    for name in _visible_validation_split_names(visible_validation_folds):
        split[name] = []
    split["selection_holdout"] = []
    split["promotion_holdout"] = []
    grouped: dict[str, list[TeacherTrace]] = {}
    for trace in labeled:
        intent = trace.teacher_frame.intent if trace.teacher_frame is not None else ""
        grouped.setdefault(intent, []).append(trace)
    for group in grouped.values():
        counts = _intent_stratified_group_counts(
            len(group),
            visible_validation_folds=visible_validation_folds,
            visible_validation_ratio=visible_validation_ratio,
        )
        start = 0
        for split_name, count in zip(split, counts, strict=True):
            end = start + count
            split[split_name].extend(group[start:end])
            start = end
    private_or_validation_splits = [
        *_visible_validation_split_names(visible_validation_folds),
        "selection_holdout",
        "promotion_holdout",
    ]
    if any(not split[name] for name in private_or_validation_splits):
        return _split_l2_target_traces_chronological(
            labeled,
            visible_validation_folds=visible_validation_folds,
            visible_validation_ratio=visible_validation_ratio,
        )
    if len(split["train"]) < 4:
        return _split_l2_target_traces_chronological(
            labeled,
            visible_validation_folds=visible_validation_folds,
            visible_validation_ratio=visible_validation_ratio,
        )
    return split


def _intent_stratified_group_counts(
    size: int,
    *,
    visible_validation_folds: int = 1,
    visible_validation_ratio: float | None = None,
) -> tuple[int, ...]:
    split_count = 1 + visible_validation_folds + 2
    if size < split_count:
        return (size, *([0] * (split_count - 1)))
    private_ratio = 0.10
    visible_ratio = _resolve_visible_validation_ratio(
        visible_validation_folds=visible_validation_folds,
        visible_validation_ratio=visible_validation_ratio,
    )
    train_ratio = 1.0 - visible_ratio - (2 * private_ratio)
    ratios = [
        train_ratio,
        *([visible_ratio / visible_validation_folds] * visible_validation_folds),
        private_ratio,
        private_ratio,
    ]
    counts = [max(1, int(size * ratio)) for ratio in ratios]
    while sum(counts) > size:
        for index in range(len(counts)):
            if sum(counts) <= size:
                break
            if counts[index] > 1:
                counts[index] -= 1
    while sum(counts) < size:
        deficits = [
            (size * ratio) - count
            for ratio, count in zip(ratios, counts, strict=True)
        ]
        index = max(range(len(counts)), key=lambda item: deficits[item])
        counts[index] += 1
    return tuple(counts)


def _distribute_count(total: int, buckets: int) -> list[int]:
    base = total // buckets
    remainder = total % buckets
    return [base + (1 if index < remainder else 0) for index in range(buckets)]


def _resolve_visible_validation_ratio(
    *,
    visible_validation_folds: int,
    visible_validation_ratio: float | None,
) -> float:
    if visible_validation_ratio is not None:
        return visible_validation_ratio
    return 0.20 if visible_validation_folds == 1 else 0.30


def _visible_validation_split_names(count: int) -> list[str]:
    if count < 1:
        raise ValueError("visible validation split count must be at least 1")
    return [
        "inner_validation",
        *[f"inner_validation_shadow_{index}" for index in range(1, count)],
    ]


def _visible_validation_split_names_from_split(
    split: dict[str, list[TeacherTrace]],
) -> list[str]:
    return [
        key
        for key in split
        if key == "inner_validation" or key.startswith("inner_validation_shadow_")
    ]


def _target_split_policy_payload(
    policy: L2TargetSplitPolicy,
    split: dict[str, list[TeacherTrace]],
    *,
    visible_validation_ratio: float | None = None,
) -> dict[str, Any]:
    visible_validation_splits = _visible_validation_split_names_from_split(split)
    total = sum(len(value) for value in split.values())
    visible_total = sum(len(split[name]) for name in visible_validation_splits)
    return {
        "schema_version": "l2-target-split-policy-v1",
        "policy": policy,
        "group_key": "teacher_frame.intent" if policy == "intent-stratified" else None,
        "split_counts": {key: len(value) for key, value in split.items()},
        "visible_validation_splits": visible_validation_splits,
        "visible_validation_folds": len(visible_validation_splits),
        "visible_validation_ratio_requested": visible_validation_ratio,
        "visible_validation_ratio_effective": visible_total / total if total else 0.0,
        "visible_validation_visibility": "agent_workspace_visible",
        "private_splits": ["selection_holdout", "promotion_holdout"],
        "private_split_visibility": "outer_harness_only",
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
    visible_validation_splits = _visible_validation_split_names_from_split(split)
    for name in ["train", *visible_validation_splits]:
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
        "visible_validation_splits": visible_validation_splits,
        "private_data_files_not_in_workspace": [
            "selection_holdout.jsonl",
            "promotion_holdout.jsonl",
        ],
        "visible_state_files": [
            "objective.json",
            "round_state.json",
            "target_diagnostics.json",
            "commands.md",
        ],
        "commands": {
            "inspect_context": "python3 tools/inspect_context.py",
            "evaluate_visible_validation": (
                "uv run --project system/darjeeling python tools/evaluate.py "
                "--split visible_validation --out runs/visible_validation.json"
            ),
            "evaluate_train_audit": (
                "uv run --project system/darjeeling python tools/evaluate.py "
                "--split train_audit --out runs/train_audit.json"
            ),
            "evaluate_slot_cue_probes": (
                "uv run --project system/darjeeling python tools/evaluate.py "
                "--split slot_cue_probes --out runs/slot_cue_probes.json"
            ),
            "search_config": (
                "uv run --project system/darjeeling python tools/search_config.py "
                f"--trials {DEFAULT_TARGET_LOCAL_SEARCH_TRIALS} "
                "--out runs/local_search.json"
            ),
        },
    }
    (workspace_root / "workspace_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def evaluate_target_workspace(
    *,
    workspace_root: Path,
    split: str,
    holdout_path: Path | None = None,
    min_accepted_accuracy: float = 0.93,
    max_wrong_accept_rate: float = 0.05,
    visible_cross_audit_folds: int = DEFAULT_TARGET_VISIBLE_CROSS_AUDIT_FOLDS,
) -> dict[str, Any]:
    train_traces = _read_teacher_jsonl(workspace_root / "data" / "train.jsonl")
    target_module = load_target_module(workspace_root / "target" / "target_l2.py")
    overrides = target_config_overrides(target_module)
    config = L2StudentConfig(**overrides)
    if split == "slot_cue_probes":
        return _evaluate_slot_cue_probes(
            workspace_root=workspace_root,
            target_module=target_module,
        )
    if split == "visible_cross_audit":
        return _evaluate_visible_cross_audit(
            workspace_root=workspace_root,
            target_module=target_module,
            config=config,
            fold_count=visible_cross_audit_folds,
            min_accepted_accuracy=min_accepted_accuracy,
            max_wrong_accept_rate=max_wrong_accept_rate,
        )
    if split in {"selection_holdout", "promotion_holdout"}:
        if holdout_path is None:
            raise ValueError(f"{split} evaluation requires a private holdout_path")
        validation_sets = [(split, _read_teacher_jsonl(holdout_path))]
    elif split == "train_audit":
        validation_sets = [(split, train_traces)]
    elif split == "visible_validation":
        validation_sets = [
            (path.stem, _read_teacher_jsonl(path))
            for path in _visible_validation_paths(workspace_root)
        ]
    else:
        if not _is_visible_validation_split_name(split):
            raise ValueError(f"unsupported target evaluation split: {split}")
        validation_path = workspace_root / "data" / f"{split}.jsonl"
        validation_sets = [(split, _read_teacher_jsonl(validation_path))]
    bundle = train_l2_student(training_examples_from_teacher_traces(train_traces), config)
    validation_traces = [
        trace for _, traces_for_split in validation_sets for trace in traces_for_split
    ]
    payload = _evaluate_trained_target(
        bundle=bundle,
        target_module=target_module,
        split=split,
        train_size=len(train_traces),
        validation_traces=validation_traces,
        min_accepted_accuracy=min_accepted_accuracy,
        max_wrong_accept_rate=max_wrong_accept_rate,
    )
    if split == "train_audit":
        payload["gate_role"] = "diagnostic_only_not_selection_or_adoption_gate"
    if split == "visible_validation":
        fold_metrics = [
            _evaluate_trained_target(
                bundle=bundle,
                target_module=target_module,
                split=split_name,
                train_size=len(train_traces),
                validation_traces=split_traces,
                min_accepted_accuracy=min_accepted_accuracy,
                max_wrong_accept_rate=max_wrong_accept_rate,
            )
            for split_name, split_traces in validation_sets
        ]
        payload["visible_validation_splits"] = [name for name, _ in validation_sets]
        payload["visible_validation_folds"] = [
            _visible_metric_summary(metric) for metric in fold_metrics
        ]
    return payload


def _evaluate_slot_cue_probes(
    *,
    workspace_root: Path,
    target_module: Any,
) -> dict[str, Any]:
    diagnostics_path = workspace_root / "data" / "target_diagnostics.json"
    if not diagnostics_path.exists():
        return _slot_cue_probe_payload([], empty_reason="missing_target_diagnostics")
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    summary = diagnostics.get("visible_slot_cue_summary")
    if not isinstance(summary, dict):
        return _slot_cue_probe_payload([], empty_reason="missing_visible_slot_cue_summary")
    items = {
        str(item.get("slot_key")): item
        for item in summary.get("items", [])
        if isinstance(item, dict) and item.get("slot_key")
    }
    probes = _slot_cue_probe_specs(items)
    results = [_run_slot_cue_probe(target_module, probe) for probe in probes]
    empty_reason = None if results else "no_visible_slot_cue_probes_available"
    return _slot_cue_probe_payload(results, empty_reason=empty_reason)


def _slot_cue_probe_payload(
    results: list[dict[str, Any]],
    *,
    empty_reason: str | None,
) -> dict[str, Any]:
    failed = [result for result in results if not result["passed"]]
    return {
        "schema_version": "l2-target-slot-cue-probes-v1",
        "split": "slot_cue_probes",
        "visibility": "visible_validation_only",
        "gate_role": "diagnostic_only_not_selection_or_adoption_gate",
        "probe_count": len(results),
        "passed_count": len(results) - len(failed),
        "failed_count": len(failed),
        "passes_gate": not failed,
        "checks": results,
        "failed_checks": failed,
        "empty_reason": empty_reason,
    }


def _slot_cue_probe_specs(items: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    probes: list[dict[str, Any]] = []
    podcast_utterance = _slot_cue_example_utterance(
        [items.get("podcast_name"), items.get("podcast_descriptor")],
        required_text="podcast",
    )
    if podcast_utterance is not None:
        probes.append(
            {
                "id": "non_podcast_podcast_cue",
                "utterance": podcast_utterance,
                "input_frame": {"intent": "play_radio", "slots": {}},
                "expectation": "veto_or_repair_away_from_play_radio",
                "visible_support_slot_keys": [
                    key for key in ["podcast_name", "podcast_descriptor"] if key in items
                ],
            }
        )
    room_value = _slot_cue_top_value(
        items.get("house_place"),
        preferred=("kitchen", "living room", "bedroom", "bathroom", "room", "house"),
    )
    if room_value is not None:
        probes.append(
            {
                "id": "slotless_radio_room_cue",
                "utterance": f"play radio in the {room_value}",
                "input_frame": {"intent": "play_radio", "slots": {}},
                "expectation": "veto_or_add_house_place",
                "expected_slot_key": "house_place",
                "expected_slot_value": room_value,
                "visible_support_slot_keys": ["house_place"],
            }
        )
    if "radio_name" in items:
        probes.append(
            {
                "id": "play_radio_generic_station_name",
                "utterance": "play random radio station",
                "input_frame": {
                    "intent": "play_radio",
                    "slots": {"radio_name": "random radio station"},
                },
                "expectation": "veto_or_remove_radio_name",
                "forbidden_slot_key": "radio_name",
                "visible_support_slot_keys": ["radio_name"],
            }
        )
    if "media_type" in items:
        probes.append(
            {
                "id": "play_radio_music_media_type_cue",
                "utterance": "on the radio it is time for good music",
                "input_frame": {"intent": "play_radio", "slots": {}},
                "expectation": "veto_or_add_media_type",
                "expected_slot_key": "media_type",
                "visible_support_slot_keys": ["media_type"],
            }
        )
    if "date" in items:
        probes.append(
            {
                "id": "calendar_remove_today_date_cue",
                "utterance": "delete all the events of today",
                "input_frame": {"intent": "calendar_remove", "slots": {}},
                "expectation": "veto_or_add_date",
                "expected_slot_key": "date",
                "visible_support_slot_keys": ["date"],
            }
        )
    event_name_intents = {
        str(intent.get("intent"))
        for intent in items.get("event_name", {}).get("top_teacher_intents", [])
        if isinstance(intent, dict) and intent.get("intent")
    }
    if {"calendar_query", "recommendation_events"}.issubset(event_name_intents):
        probes.append(
            {
                "id": "recommendation_events_bare_upcoming_events",
                "utterance": "what are the upcoming events",
                "input_frame": {"intent": "recommendation_events", "slots": {}},
                "expectation": "veto_or_repair_away_from_recommendation_events",
                "visible_support_slot_keys": ["event_name"],
            }
        )
    joke_utterance = _slot_cue_example_utterance(
        [items.get("joke_type")],
        required_text="joke about",
    )
    if joke_utterance is None:
        joke_value = _slot_cue_top_value(items.get("joke_type"))
        if joke_value is not None:
            joke_utterance = f"tell me a joke about {joke_value}"
    if joke_utterance is not None:
        probes.append(
            {
                "id": "general_joke_missing_joke_type",
                "utterance": joke_utterance,
                "input_frame": {"intent": "general_joke", "slots": {}},
                "expectation": "veto_or_add_joke_type",
                "expected_slot_key": "joke_type",
                "visible_support_slot_keys": ["joke_type"],
            }
        )
    adjective_joke_utterance = _slot_cue_example_utterance(
        [items.get("joke_type")],
        required_text="joke",
        excluded_text=("joke about",),
    )
    if adjective_joke_utterance is not None:
        probes.append(
            {
                "id": "general_joke_adjective_missing_joke_type",
                "utterance": adjective_joke_utterance,
                "input_frame": {"intent": "general_joke", "slots": {}},
                "expectation": "veto_or_add_joke_type",
                "expected_slot_key": "joke_type",
                "visible_support_slot_keys": ["joke_type"],
            }
        )
    if "joke_type" in items:
        probes.append(
            {
                "id": "general_joke_superlative_missing_joke_type",
                "utterance": "what's the funniest joke",
                "input_frame": {"intent": "general_joke", "slots": {}},
                "expectation": "veto_or_add_joke_type",
                "expected_slot_key": "joke_type",
                "visible_support_slot_keys": ["joke_type"],
            }
        )
    if "change_amount" in items:
        probes.append(
            {
                "id": "audio_volume_spoken_amount_cue",
                "utterance": "change the volume level to nineteen please",
                "input_frame": {"intent": "audio_volume_up", "slots": {}},
                "expectation": "veto_or_add_change_amount",
                "expected_slot_key": "change_amount",
                "visible_support_slot_keys": ["change_amount"],
            }
        )
    return probes


def _slot_cue_example_utterance(
    items: list[dict[str, Any] | None],
    *,
    required_text: str,
    excluded_text: tuple[str, ...] = (),
) -> str | None:
    for item in items:
        if not isinstance(item, dict):
            continue
        for example in item.get("examples", []):
            if not isinstance(example, dict):
                continue
            utterance = str(example.get("utterance") or "")
            normalized = utterance.lower()
            if required_text in normalized and not any(
                excluded in normalized for excluded in excluded_text
            ):
                return utterance
    return None


def _slot_cue_top_value(
    item: dict[str, Any] | None,
    *,
    preferred: tuple[str, ...] = (),
) -> str | None:
    if not isinstance(item, dict):
        return None
    values = [
        str(value.get("value"))
        for value in item.get("top_values", [])
        if isinstance(value, dict) and value.get("value")
    ]
    for preferred_value in preferred:
        if preferred_value in values:
            return preferred_value
    return values[0] if values else None


def _run_slot_cue_probe(
    target_module: Any,
    probe: dict[str, Any],
) -> dict[str, Any]:
    input_frame = Frame.model_validate(
        {
            "intent": probe["input_frame"]["intent"],
            "slots": probe["input_frame"].get("slots", {}),
            "is_abstain": False,
        }
    )
    metadata = {
        "probe_id": probe["id"],
        "source": "visible_slot_cue_summary",
    }
    frame = target_postprocess_frame(
        target_module,
        utterance=str(probe["utterance"]),
        frame=input_frame,
        metadata=metadata,
    )
    accepted = target_accept_prediction(
        target_module,
        utterance=str(probe["utterance"]),
        frame=frame,
        metadata=metadata,
        default_accept=True,
    )
    passed = _slot_cue_probe_passed(probe, frame, accepted)
    return {
        **probe,
        "postprocessed_frame": frame.model_dump(mode="json"),
        "accepted_with_default_true": accepted,
        "passed": passed,
    }


def _slot_cue_probe_passed(
    probe: dict[str, Any],
    frame: Frame,
    accepted: bool,
) -> bool:
    if not accepted:
        return True
    expectation = probe.get("expectation")
    if expectation == "veto_or_repair_away_from_play_radio":
        return frame.intent != "play_radio"
    if expectation == "veto_or_add_house_place":
        return frame.slots.get("house_place") == probe.get("expected_slot_value")
    if expectation == "veto_or_remove_radio_name":
        return "radio_name" not in frame.slots
    if expectation == "veto_or_add_media_type":
        return "media_type" in frame.slots
    if expectation == "veto_or_add_date":
        return "date" in frame.slots
    if expectation == "veto_or_repair_away_from_recommendation_events":
        return frame.intent != "recommendation_events"
    if expectation == "veto_or_add_joke_type":
        return "joke_type" in frame.slots
    if expectation == "veto_or_add_change_amount":
        return "change_amount" in frame.slots
    return False


def _evaluate_visible_cross_audit(
    *,
    workspace_root: Path,
    target_module: Any,
    config: L2StudentConfig,
    fold_count: int,
    min_accepted_accuracy: float,
    max_wrong_accept_rate: float,
) -> dict[str, Any]:
    if fold_count < 2:
        raise ValueError("visible_cross_audit requires at least 2 folds")
    visible_traces = _visible_training_and_validation_traces(workspace_root)
    folds = _intent_stratified_cross_audit_folds(visible_traces, fold_count)
    fold_metrics: list[dict[str, Any]] = []
    for index, validation_traces in enumerate(folds):
        train_traces = [
            trace
            for fold_index, traces_for_fold in enumerate(folds)
            if fold_index != index
            for trace in traces_for_fold
        ]
        if len(train_traces) < 4 or not validation_traces:
            continue
        bundle = train_l2_student(
            training_examples_from_teacher_traces(train_traces),
            config,
        )
        fold_metrics.append(
            _evaluate_trained_target(
                bundle=bundle,
                target_module=target_module,
                split=f"visible_cross_audit_fold_{index + 1}",
                train_size=len(train_traces),
                validation_traces=validation_traces,
                min_accepted_accuracy=min_accepted_accuracy,
                max_wrong_accept_rate=max_wrong_accept_rate,
            ),
        )
    return _aggregate_visible_cross_audit_metrics(
        fold_metrics=fold_metrics,
        fold_count=fold_count,
        visible_size=len(visible_traces),
        min_accepted_accuracy=min_accepted_accuracy,
        max_wrong_accept_rate=max_wrong_accept_rate,
    )


def _visible_training_and_validation_traces(workspace_root: Path) -> list[TeacherTrace]:
    return [
        *_read_teacher_jsonl(workspace_root / "data" / "train.jsonl"),
        *[
            trace
            for path in _visible_validation_paths(workspace_root)
            for trace in _read_teacher_jsonl(path)
        ],
    ]


def _intent_stratified_cross_audit_folds(
    traces: list[TeacherTrace],
    fold_count: int,
) -> list[list[TeacherTrace]]:
    folds: list[list[TeacherTrace]] = [[] for _ in range(fold_count)]
    grouped: dict[str, list[TeacherTrace]] = {}
    for trace in traces:
        intent = trace.teacher_frame.intent if trace.teacher_frame is not None else ""
        grouped.setdefault(intent, []).append(trace)
    for group in grouped.values():
        for index, trace in enumerate(group):
            folds[index % fold_count].append(trace)
    if any(not fold for fold in folds):
        folds = [[] for _ in range(fold_count)]
        for index, trace in enumerate(traces):
            folds[index % fold_count].append(trace)
    return folds


def _aggregate_visible_cross_audit_metrics(
    *,
    fold_metrics: list[dict[str, Any]],
    fold_count: int,
    visible_size: int,
    min_accepted_accuracy: float,
    max_wrong_accept_rate: float,
) -> dict[str, Any]:
    accepted = sum(int(metric["accepted"]) for metric in fold_metrics)
    correct = sum(int(metric["correct_accepts"]) for metric in fold_metrics)
    wrong = sum(int(metric["wrong_accepts"]) for metric in fold_metrics)
    vetoed = sum(int(metric["vetoed_accepts"]) for metric in fold_metrics)
    validation_size = sum(int(metric["validation_size"]) for metric in fold_metrics)
    accepted_accuracy = correct / accepted if accepted else None
    wrong_accept_rate = wrong / accepted if accepted else 0.0
    passes_gate = bool(
        accepted
        and accepted_accuracy is not None
        and accepted_accuracy >= min_accepted_accuracy
        and wrong_accept_rate <= max_wrong_accept_rate
    )
    wrong_examples = _top_guard_examples(
        [
            example
            for metric in fold_metrics
            for example in metric.get("wrong_examples", [])
        ],
    )
    veto_examples = _top_guard_examples(
        [
            example
            for metric in fold_metrics
            for example in metric.get("veto_examples", [])
        ],
    )
    near_miss_examples = _top_guard_examples(
        [
            example
            for metric in fold_metrics
            for example in metric.get("near_miss_examples", [])
        ],
    )
    safety_backlog = _aggregate_safety_backlogs(
        split="visible_cross_audit",
        validation_size=validation_size,
        backlogs=[
            metric["safety_backlog"]
            for metric in fold_metrics
            if isinstance(metric.get("safety_backlog"), dict)
        ],
    )
    slot_risk_backlog = _aggregate_slot_risk_backlogs(
        split="visible_cross_audit",
        validation_size=validation_size,
        backlogs=[
            metric["family_diagnostics"]["slot_risk_backlog"]
            for metric in fold_metrics
            if isinstance(metric.get("family_diagnostics"), dict)
            and isinstance(
                metric["family_diagnostics"].get("slot_risk_backlog"),
                dict,
            )
        ],
    )
    intent_confusion_backlog = _aggregate_intent_confusion_backlogs(
        split="visible_cross_audit",
        validation_size=validation_size,
        backlogs=[
            metric["family_diagnostics"]["intent_confusion_backlog"]
            for metric in fold_metrics
            if isinstance(metric.get("family_diagnostics"), dict)
            and isinstance(
                metric["family_diagnostics"].get("intent_confusion_backlog"),
                dict,
            )
        ],
    )
    return {
        "split": "visible_cross_audit",
        "train_size": visible_size,
        "validation_size": validation_size,
        "accepted": accepted,
        "correct_accepts": correct,
        "wrong_accepts": wrong,
        "vetoed_accepts": vetoed,
        "coverage": accepted / validation_size if validation_size else 0.0,
        "accepted_accuracy": accepted_accuracy,
        "wrong_accept_rate": wrong_accept_rate,
        "passes_gate": passes_gate,
        "gate_role": "diagnostic_only_not_selection_or_adoption_gate",
        "visible_cross_audit_folds_requested": fold_count,
        "visible_cross_audit_folds_completed": len(fold_metrics),
        "wrong_examples": wrong_examples,
        "veto_examples": veto_examples,
        "near_miss_examples": near_miss_examples,
        "family_diagnostics": None,
        "safety_backlog": safety_backlog,
        "slot_risk_backlog": slot_risk_backlog,
        "intent_confusion_backlog": intent_confusion_backlog,
        "folds": [_visible_metric_summary(metric) for metric in fold_metrics],
    }


def _aggregate_safety_backlogs(
    *,
    split: str,
    validation_size: int,
    backlogs: list[dict[str, Any]],
    limit: int = 8,
) -> dict[str, Any]:
    by_intent: dict[str, dict[str, Any]] = {}
    for backlog in backlogs:
        for item in backlog.get("items", []):
            intent = str(item["teacher_intent"])
            aggregate = by_intent.setdefault(
                intent,
                {
                    "teacher_intent": intent,
                    "total": 0,
                    "accepted_correct": 0,
                    "accepted_wrong": 0,
                    "intent_correct_slot_wrong": 0,
                    "max_wrong_guard_probability": 0.0,
                    "top_predicted_intents": item.get("top_predicted_intents", []),
                    "wrong_examples": [],
                    "recommended_action": item.get(
                        "recommended_action",
                        (
                            "tighten accept_prediction or add exact postprocess "
                            "before any coverage expansion"
                        ),
                    ),
                },
            )
            aggregate["total"] += int(item.get("total") or 0)
            aggregate["accepted_correct"] += int(item.get("accepted_correct") or 0)
            aggregate["accepted_wrong"] += int(item.get("accepted_wrong") or 0)
            aggregate["intent_correct_slot_wrong"] += int(
                item.get("intent_correct_slot_wrong") or 0,
            )
            aggregate["max_wrong_guard_probability"] = max(
                float(aggregate["max_wrong_guard_probability"]),
                float(item.get("max_wrong_guard_probability") or 0.0),
            )
            aggregate["wrong_examples"].extend(item.get("wrong_examples", []))
    items = []
    for aggregate in by_intent.values():
        accepted_total = aggregate["accepted_correct"] + aggregate["accepted_wrong"]
        examples = _top_guard_examples(aggregate["wrong_examples"], limit=3)
        items.append(
            {
                **aggregate,
                "wrong_accept_share": aggregate["accepted_wrong"] / accepted_total
                if accepted_total
                else 0.0,
                "wrong_examples": examples,
            },
        )
    items.sort(
        key=lambda item: (
            int(item["accepted_wrong"]),
            float(item["wrong_accept_share"]),
            int(item["intent_correct_slot_wrong"]),
            float(item["max_wrong_guard_probability"]),
            item["teacher_intent"],
        ),
        reverse=True,
    )
    return {
        "schema_version": "l2-target-safety-backlog-v1",
        "split": split,
        "validation_size": validation_size,
        "visibility": _safety_backlog_visibility(split),
        "priority": "fix_visible_accepted_wrong_before_coverage_expansion",
        "item_limit": limit,
        "items": items[:limit],
        "empty_reason": None
        if items
        else _safety_backlog_empty_reason(split),
    }


def _aggregate_slot_risk_backlogs(
    *,
    split: str,
    validation_size: int,
    backlogs: list[dict[str, Any]],
    limit: int = 8,
) -> dict[str, Any]:
    by_intent: dict[str, dict[str, Any]] = {}
    for backlog in backlogs:
        for item in backlog.get("items", []):
            intent = str(item["teacher_intent"])
            aggregate = by_intent.setdefault(
                intent,
                {
                    "teacher_intent": intent,
                    "total": 0,
                    "accepted_correct": 0,
                    "accepted_wrong": 0,
                    "intent_correct_slot_wrong": 0,
                    "max_slot_mismatch_guard_probability": 0.0,
                    "top_predicted_intents": item.get("top_predicted_intents", []),
                    "slot_mismatch_examples": [],
                    "missing_slot_keys": {},
                    "extra_slot_keys": {},
                    "changed_slot_keys": {},
                    "recommended_action": item.get(
                        "recommended_action",
                        (
                            "add precise postprocess or abstain rules for visible "
                            "slot-risk patterns before broad coverage expansion"
                        ),
                    ),
                },
            )
            aggregate["total"] += int(item.get("total") or 0)
            aggregate["accepted_correct"] += int(item.get("accepted_correct") or 0)
            aggregate["accepted_wrong"] += int(item.get("accepted_wrong") or 0)
            aggregate["intent_correct_slot_wrong"] += int(
                item.get("intent_correct_slot_wrong") or 0,
            )
            aggregate["max_slot_mismatch_guard_probability"] = max(
                float(aggregate["max_slot_mismatch_guard_probability"]),
                float(item.get("max_slot_mismatch_guard_probability") or 0.0),
            )
            aggregate["slot_mismatch_examples"].extend(
                item.get("slot_mismatch_examples", []),
            )
            _merge_slot_key_counts(
                aggregate["missing_slot_keys"],
                item.get("missing_slot_keys", []),
            )
            _merge_slot_key_counts(
                aggregate["extra_slot_keys"],
                item.get("extra_slot_keys", []),
            )
            _merge_slot_key_counts(
                aggregate["changed_slot_keys"],
                item.get("changed_slot_keys", []),
            )
    items = []
    for aggregate in by_intent.values():
        items.append(
            {
                **aggregate,
                "slot_mismatch_examples": _top_guard_examples(
                    aggregate["slot_mismatch_examples"],
                    limit=3,
                ),
                "missing_slot_keys": _top_slot_key_counts(
                    aggregate["missing_slot_keys"],
                ),
                "extra_slot_keys": _top_slot_key_counts(
                    aggregate["extra_slot_keys"],
                ),
                "changed_slot_keys": _top_slot_key_counts(
                    aggregate["changed_slot_keys"],
                ),
            },
        )
    items.sort(
        key=lambda item: (
            int(item["intent_correct_slot_wrong"]),
            float(item["max_slot_mismatch_guard_probability"]),
            int(item["accepted_wrong"]),
            item["teacher_intent"],
        ),
        reverse=True,
    )
    high_guard_items = sorted(
        items,
        key=lambda item: (
            float(item["max_slot_mismatch_guard_probability"]),
            int(item["intent_correct_slot_wrong"]),
            int(item["accepted_wrong"]),
            item["teacher_intent"],
        ),
        reverse=True,
    )[:limit]
    return {
        "schema_version": "l2-target-slot-risk-backlog-v1",
        "split": split,
        "validation_size": validation_size,
        "visibility": _safety_backlog_visibility(split),
        "priority": "review_visible_slot_mismatches_after_accepted_wrong_backlog",
        "item_limit": limit,
        "items": items[:limit],
        "high_guard_item_limit": limit,
        "high_guard_items": high_guard_items,
        "empty_reason": None
        if items
        else _slot_risk_backlog_empty_reason(split),
    }


def _merge_slot_key_counts(
    counts: dict[str, int],
    rows: list[dict[str, Any]],
) -> None:
    for row in rows:
        slot_key = str(row.get("slot_key") or "")
        if not slot_key:
            continue
        counts[slot_key] = counts.get(slot_key, 0) + int(row.get("count") or 0)


def _aggregate_intent_confusion_backlogs(
    *,
    split: str,
    validation_size: int,
    backlogs: list[dict[str, Any]],
    limit: int = 8,
) -> dict[str, Any]:
    by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for backlog in backlogs:
        for item in backlog.get("items", []):
            teacher_intent = str(item["teacher_intent"])
            predicted_intent = str(item["predicted_intent"])
            pair = (teacher_intent, predicted_intent)
            aggregate = by_pair.setdefault(
                pair,
                {
                    "teacher_intent": teacher_intent,
                    "predicted_intent": predicted_intent,
                    "total": 0,
                    "default_accepts": 0,
                    "accepted_wrong": 0,
                    "max_guard_probability": 0.0,
                    "examples": [],
                    "recommended_action": item.get(
                        "recommended_action",
                        (
                            "add intent-specific abstain rules for high-guard "
                            "visible intent confusions before broad coverage expansion"
                        ),
                    ),
                },
            )
            aggregate["total"] += int(item.get("total") or 0)
            aggregate["default_accepts"] += int(item.get("default_accepts") or 0)
            aggregate["accepted_wrong"] += int(item.get("accepted_wrong") or 0)
            aggregate["max_guard_probability"] = max(
                float(aggregate["max_guard_probability"]),
                float(item.get("max_guard_probability") or 0.0),
            )
            aggregate["examples"].extend(item.get("examples", []))
    items = []
    for aggregate in by_pair.values():
        items.append(
            {
                **aggregate,
                "examples": _top_guard_examples(aggregate["examples"], limit=3),
            },
        )
    items.sort(
        key=lambda item: (
            int(item["accepted_wrong"]),
            int(item["default_accepts"]),
            float(item["max_guard_probability"]),
            int(item["total"]),
            item["teacher_intent"],
            item["predicted_intent"],
        ),
        reverse=True,
    )
    return {
        "schema_version": "l2-target-intent-confusion-backlog-v1",
        "split": split,
        "validation_size": validation_size,
        "visibility": _safety_backlog_visibility(split),
        "priority": "review_visible_intent_confusions_after_slot_risk",
        "item_limit": limit,
        "items": items[:limit],
        "empty_reason": None if items else _intent_confusion_empty_reason(split),
    }


def _intent_confusion_empty_reason(split: str) -> str:
    if split in {"selection_holdout", "promotion_holdout"}:
        return "no_private_intent_confusion_families"
    return "no_visible_intent_confusion_families"


def _visible_validation_paths(workspace_root: Path) -> list[Path]:
    data_dir = workspace_root / "data"
    paths = [
        path
        for path in data_dir.glob("inner_validation*.jsonl")
        if _is_visible_validation_split_name(path.stem)
    ]
    return sorted(paths, key=lambda path: _visible_validation_sort_key(path.stem))


def _visible_validation_sort_key(name: str) -> tuple[int, str]:
    if name == "inner_validation":
        return (0, name)
    return (1, name)


def _is_visible_validation_split_name(name: str) -> bool:
    return name == "inner_validation" or name.startswith("inner_validation_shadow_")


def _visible_gate_split(workspace_root: Path) -> str:
    paths = _visible_validation_paths(workspace_root)
    return "visible_validation" if len(paths) > 1 else "inner_validation"


def _evaluate_trained_target(
    *,
    bundle: Any,
    target_module: Any,
    split: str,
    train_size: int,
    validation_traces: list[TeacherTrace],
    min_accepted_accuracy: float,
    max_wrong_accept_rate: float,
) -> dict[str, Any]:

    accepted = 0
    correct = 0
    wrong = 0
    vetoed_accepts = 0
    examples: list[dict[str, Any]] = []
    veto_examples: list[dict[str, Any]] = []
    near_miss_examples: list[dict[str, Any]] = []
    family_stats: dict[str, dict[str, Any]] = {}
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
        _record_family_diagnostic(
            family_stats,
            trace=trace,
            frame=frame,
            guard_probability=prediction.guard_probability,
            default_accept=default_accept,
            should_accept=should_accept,
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
    family_diagnostics = _family_diagnostics_payload(
        split=split,
        validation_size=total,
        family_stats=family_stats,
    )
    return {
        "split": split,
        "train_size": train_size,
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
        "family_diagnostics": family_diagnostics,
        "safety_backlog": family_diagnostics["safety_backlog"],
    }


def _evaluate_target_candidate(
    *,
    workspace_root: Path,
    private_paths: dict[str, Path],
    config: L2TargetEvolutionConfig,
    label: str,
) -> dict[str, Any]:
    visible_gate_split = _visible_gate_split(workspace_root)
    inner_result = evaluate_target_workspace(
        workspace_root=workspace_root,
        split=visible_gate_split,
        min_accepted_accuracy=config.min_accepted_accuracy,
        max_wrong_accept_rate=config.max_wrong_accept_rate,
    )
    train_audit_result = evaluate_target_workspace(
        workspace_root=workspace_root,
        split="train_audit",
        min_accepted_accuracy=config.min_accepted_accuracy,
        max_wrong_accept_rate=config.max_wrong_accept_rate,
    )
    visible_cross_audit_result = (
        evaluate_target_workspace(
            workspace_root=workspace_root,
            split="visible_cross_audit",
            min_accepted_accuracy=config.min_accepted_accuracy,
            max_wrong_accept_rate=config.max_wrong_accept_rate,
            visible_cross_audit_folds=config.visible_cross_audit_folds,
        )
        if config.visible_cross_audit_folds >= 2
        else None
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
        "train_audit": train_audit_result,
        "visible_cross_audit": visible_cross_audit_result,
        **private_results,
    }


def _snapshot_target_dir(
    *,
    workspace_root: Path,
    rounds_dir: Path,
    job_dir: Path,
    round_index: int,
) -> str:
    snapshot_dir = rounds_dir / f"round_{round_index:03d}_target"
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    shutil.copytree(workspace_root / "target", snapshot_dir)
    return snapshot_dir.relative_to(job_dir).as_posix()


def _record_family_diagnostic(
    family_stats: dict[str, dict[str, Any]],
    *,
    trace: TeacherTrace,
    frame: Any,
    guard_probability: float,
    default_accept: bool,
    should_accept: bool,
) -> None:
    teacher_frame = trace.teacher_frame
    if teacher_frame is None:
        return
    teacher_intent = teacher_frame.intent
    predicted_intent = frame.intent
    stats = family_stats.setdefault(
        teacher_intent,
        {
            "teacher_intent": teacher_intent,
            "total": 0,
            "accepted_correct": 0,
            "accepted_wrong": 0,
            "rejected_correct": 0,
            "rejected_wrong": 0,
            "vetoed_correct": 0,
            "vetoed_wrong": 0,
            "intent_correct_slot_wrong": 0,
            "missing_slot_keys": {},
            "extra_slot_keys": {},
            "changed_slot_keys": {},
            "intent_confusions": {},
            "predicted_intents": {},
            "examples": {
                "accepted_wrong": [],
                "rejected_correct": [],
                "vetoed_correct": [],
                "intent_correct_slot_wrong": [],
            },
        },
    )
    stats["total"] += 1
    stats["predicted_intents"][predicted_intent] = (
        stats["predicted_intents"].get(predicted_intent, 0) + 1
    )
    correct = frame == teacher_frame
    if predicted_intent == teacher_intent and not correct:
        stats["intent_correct_slot_wrong"] += 1
        slot_key_deltas = _slot_key_deltas(
            teacher_slots=teacher_frame.slots,
            predicted_slots=frame.slots,
        )
        _increment_slot_key_counts(
            stats["missing_slot_keys"],
            slot_key_deltas["missing"],
        )
        _increment_slot_key_counts(
            stats["extra_slot_keys"],
            slot_key_deltas["extra"],
        )
        _increment_slot_key_counts(
            stats["changed_slot_keys"],
            slot_key_deltas["changed"],
        )
        _append_diagnostic_example(
            stats["examples"]["intent_correct_slot_wrong"],
            trace=trace,
            frame=frame,
            guard_probability=guard_probability,
        )
    if predicted_intent != teacher_intent:
        confusion = stats["intent_confusions"].setdefault(
            predicted_intent,
            {
                "teacher_intent": teacher_intent,
                "predicted_intent": predicted_intent,
                "total": 0,
                "default_accepts": 0,
                "accepted_wrong": 0,
                "max_guard_probability": 0.0,
                "examples": [],
            },
        )
        confusion["total"] += 1
        if default_accept:
            confusion["default_accepts"] += 1
        if should_accept:
            confusion["accepted_wrong"] += 1
        confusion["max_guard_probability"] = max(
            float(confusion["max_guard_probability"]),
            float(guard_probability),
        )
        _append_diagnostic_example(
            confusion["examples"],
            trace=trace,
            frame=frame,
            guard_probability=guard_probability,
        )
    if should_accept:
        if correct:
            stats["accepted_correct"] += 1
        else:
            stats["accepted_wrong"] += 1
            _append_diagnostic_example(
                stats["examples"]["accepted_wrong"],
                trace=trace,
                frame=frame,
                guard_probability=guard_probability,
            )
        return
    if default_accept:
        if correct:
            stats["vetoed_correct"] += 1
            _append_diagnostic_example(
                stats["examples"]["vetoed_correct"],
                trace=trace,
                frame=frame,
                guard_probability=guard_probability,
            )
        else:
            stats["vetoed_wrong"] += 1
        return
    if correct:
        stats["rejected_correct"] += 1
        _append_diagnostic_example(
            stats["examples"]["rejected_correct"],
            trace=trace,
            frame=frame,
            guard_probability=guard_probability,
        )
    else:
        stats["rejected_wrong"] += 1


def _slot_key_deltas(
    *,
    teacher_slots: dict[str, Any],
    predicted_slots: dict[str, Any],
) -> dict[str, list[str]]:
    teacher_keys = set(teacher_slots)
    predicted_keys = set(predicted_slots)
    shared_keys = teacher_keys & predicted_keys
    return {
        "missing": sorted(teacher_keys - predicted_keys),
        "extra": sorted(predicted_keys - teacher_keys),
        "changed": sorted(
            key
            for key in shared_keys
            if str(teacher_slots.get(key)) != str(predicted_slots.get(key))
        ),
    }


def _increment_slot_key_counts(counts: dict[str, int], slot_keys: list[str]) -> None:
    for slot_key in slot_keys:
        counts[slot_key] = counts.get(slot_key, 0) + 1


def _append_diagnostic_example(
    examples: list[dict[str, Any]],
    *,
    trace: TeacherTrace,
    frame: Any,
    guard_probability: float,
    limit: int = 3,
) -> None:
    example = {
        "request_id": trace.request_id,
        "utterance": trace.utterance,
        "teacher_frame": trace.teacher_frame.model_dump(mode="json")
        if trace.teacher_frame is not None
        else None,
        "predicted_frame": frame.model_dump(mode="json"),
        "guard_probability": guard_probability,
    }
    examples.append(example)
    examples.sort(key=lambda item: float(item["guard_probability"]), reverse=True)
    del examples[limit:]


def _family_diagnostics_payload(
    *,
    split: str,
    validation_size: int,
    family_stats: dict[str, dict[str, Any]],
    limit: int = 12,
) -> dict[str, Any]:
    families = [_finalize_family_diagnostic(stats) for stats in family_stats.values()]
    safety_backlog = _safety_backlog_payload(
        split=split,
        validation_size=validation_size,
        families=families,
    )
    slot_risk_backlog = _slot_risk_backlog_payload(
        split=split,
        validation_size=validation_size,
        families=families,
    )
    intent_confusion_backlog = _intent_confusion_backlog_payload(
        split=split,
        validation_size=validation_size,
        families=families,
    )
    families.sort(
        key=lambda item: (
            item["opportunity_score"],
            item["rejected_correct"],
            item["vetoed_correct"],
            -item["accepted_wrong"],
            item["teacher_intent"],
        ),
        reverse=True,
    )
    return {
        "schema_version": "l2-target-family-diagnostics-v1",
        "split": split,
        "validation_size": validation_size,
        "family_limit": limit,
        "families": families[:limit],
        "safety_backlog": safety_backlog,
        "slot_risk_backlog": slot_risk_backlog,
        "intent_confusion_backlog": intent_confusion_backlog,
    }


def _finalize_family_diagnostic(stats: dict[str, Any]) -> dict[str, Any]:
    predicted_intents = sorted(
        stats["predicted_intents"].items(),
        key=lambda item: (-int(item[1]), str(item[0])),
    )[:5]
    accepted_wrong = int(stats["accepted_wrong"])
    rejected_correct = int(stats["rejected_correct"])
    vetoed_correct = int(stats["vetoed_correct"])
    return {
        "teacher_intent": stats["teacher_intent"],
        "total": int(stats["total"]),
        "accepted_correct": int(stats["accepted_correct"]),
        "accepted_wrong": accepted_wrong,
        "rejected_correct": rejected_correct,
        "rejected_wrong": int(stats["rejected_wrong"]),
        "vetoed_correct": vetoed_correct,
        "vetoed_wrong": int(stats["vetoed_wrong"]),
        "intent_correct_slot_wrong": int(stats["intent_correct_slot_wrong"]),
        "missing_slot_keys": _top_slot_key_counts(stats.get("missing_slot_keys", {})),
        "extra_slot_keys": _top_slot_key_counts(stats.get("extra_slot_keys", {})),
        "changed_slot_keys": _top_slot_key_counts(stats.get("changed_slot_keys", {})),
        "intent_confusions": _intent_confusion_items_for_family(
            stats.get("intent_confusions", {}),
        ),
        "opportunity_score": rejected_correct + vetoed_correct - (10 * accepted_wrong),
        "top_predicted_intents": [
            {"intent": intent, "count": count} for intent, count in predicted_intents
        ],
        "examples": stats["examples"],
    }


def _safety_backlog_payload(
    *,
    split: str,
    validation_size: int,
    families: list[dict[str, Any]],
    limit: int = 8,
) -> dict[str, Any]:
    items = [
        _safety_backlog_item(family)
        for family in families
        if int(family["accepted_wrong"]) > 0
    ]
    items.sort(
        key=lambda item: (
            int(item["accepted_wrong"]),
            float(item["wrong_accept_share"]),
            int(item["intent_correct_slot_wrong"]),
            float(item["max_wrong_guard_probability"]),
            item["teacher_intent"],
        ),
        reverse=True,
    )
    return {
        "schema_version": "l2-target-safety-backlog-v1",
        "split": split,
        "validation_size": validation_size,
        "visibility": _safety_backlog_visibility(split),
        "priority": "fix_visible_accepted_wrong_before_coverage_expansion",
        "item_limit": limit,
        "items": items[:limit],
        "empty_reason": None
        if items
        else _safety_backlog_empty_reason(split),
    }


def _safety_backlog_visibility(split: str) -> str:
    if split in {"selection_holdout", "promotion_holdout"}:
        return "outer_summary_only_not_agent_workspace"
    return "visible_validation_only"


def _safety_backlog_empty_reason(split: str) -> str:
    if split in {"selection_holdout", "promotion_holdout"}:
        return "no_private_accepted_wrong_families"
    return "no_visible_accepted_wrong_families"


def _safety_backlog_item(family: dict[str, Any]) -> dict[str, Any]:
    accepted_wrong = int(family["accepted_wrong"])
    accepted_correct = int(family["accepted_correct"])
    accepted_total = accepted_wrong + accepted_correct
    wrong_examples = family["examples"].get("accepted_wrong", [])
    max_wrong_guard = max(
        (float(example["guard_probability"]) for example in wrong_examples),
        default=0.0,
    )
    return {
        "teacher_intent": family["teacher_intent"],
        "total": int(family["total"]),
        "accepted_correct": accepted_correct,
        "accepted_wrong": accepted_wrong,
        "wrong_accept_share": accepted_wrong / accepted_total
        if accepted_total
        else 0.0,
        "intent_correct_slot_wrong": int(family["intent_correct_slot_wrong"]),
        "max_wrong_guard_probability": max_wrong_guard,
        "top_predicted_intents": family["top_predicted_intents"],
        "wrong_examples": wrong_examples,
        "recommended_action": (
            "tighten accept_prediction or add exact postprocess before any coverage expansion"
        ),
    }


def _intent_confusion_items_for_family(
    confusions: dict[str, dict[str, Any]],
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    items = [
        {
            **confusion,
            "examples": _top_guard_examples(confusion["examples"], limit=3),
        }
        for confusion in confusions.values()
    ]
    items.sort(
        key=lambda item: (
            int(item["accepted_wrong"]),
            int(item["default_accepts"]),
            float(item["max_guard_probability"]),
            int(item["total"]),
            item["teacher_intent"],
            item["predicted_intent"],
        ),
        reverse=True,
    )
    return items[:limit]


def _intent_confusion_backlog_payload(
    *,
    split: str,
    validation_size: int,
    families: list[dict[str, Any]],
    limit: int = 8,
) -> dict[str, Any]:
    items = [
        {
            **confusion,
            "recommended_action": (
                "add intent-specific abstain rules for high-guard visible intent "
                "confusions before broad coverage expansion"
            ),
        }
        for family in families
        for confusion in family.get("intent_confusions", [])
    ]
    items.sort(
        key=lambda item: (
            int(item["accepted_wrong"]),
            int(item["default_accepts"]),
            float(item["max_guard_probability"]),
            int(item["total"]),
            item["teacher_intent"],
            item["predicted_intent"],
        ),
        reverse=True,
    )
    return {
        "schema_version": "l2-target-intent-confusion-backlog-v1",
        "split": split,
        "validation_size": validation_size,
        "visibility": _safety_backlog_visibility(split),
        "priority": "review_visible_intent_confusions_after_slot_risk",
        "item_limit": limit,
        "items": items[:limit],
        "empty_reason": None if items else _intent_confusion_empty_reason(split),
    }


def _slot_risk_backlog_payload(
    *,
    split: str,
    validation_size: int,
    families: list[dict[str, Any]],
    limit: int = 8,
) -> dict[str, Any]:
    items = [
        _slot_risk_backlog_item(family)
        for family in families
        if int(family["intent_correct_slot_wrong"]) > 0
    ]
    items.sort(
        key=lambda item: (
            int(item["intent_correct_slot_wrong"]),
            float(item["max_slot_mismatch_guard_probability"]),
            int(item["accepted_wrong"]),
            item["teacher_intent"],
        ),
        reverse=True,
    )
    high_guard_items = sorted(
        items,
        key=lambda item: (
            float(item["max_slot_mismatch_guard_probability"]),
            int(item["intent_correct_slot_wrong"]),
            int(item["accepted_wrong"]),
            item["teacher_intent"],
        ),
        reverse=True,
    )[:limit]
    return {
        "schema_version": "l2-target-slot-risk-backlog-v1",
        "split": split,
        "validation_size": validation_size,
        "visibility": _safety_backlog_visibility(split),
        "priority": "review_visible_slot_mismatches_after_accepted_wrong_backlog",
        "item_limit": limit,
        "items": items[:limit],
        "high_guard_item_limit": limit,
        "high_guard_items": high_guard_items,
        "empty_reason": None
        if items
        else _slot_risk_backlog_empty_reason(split),
    }


def _slot_risk_backlog_empty_reason(split: str) -> str:
    if split in {"selection_holdout", "promotion_holdout"}:
        return "no_private_slot_risk_families"
    return "no_visible_slot_risk_families"


def _slot_risk_backlog_item(family: dict[str, Any]) -> dict[str, Any]:
    slot_examples = family["examples"].get("intent_correct_slot_wrong", [])
    max_slot_guard = max(
        (float(example["guard_probability"]) for example in slot_examples),
        default=0.0,
    )
    return {
        "teacher_intent": family["teacher_intent"],
        "total": int(family["total"]),
        "accepted_correct": int(family["accepted_correct"]),
        "accepted_wrong": int(family["accepted_wrong"]),
        "intent_correct_slot_wrong": int(family["intent_correct_slot_wrong"]),
        "max_slot_mismatch_guard_probability": max_slot_guard,
        "top_predicted_intents": family["top_predicted_intents"],
        "missing_slot_keys": family.get("missing_slot_keys", []),
        "extra_slot_keys": family.get("extra_slot_keys", []),
        "changed_slot_keys": family.get("changed_slot_keys", []),
        "slot_mismatch_examples": slot_examples,
        "recommended_action": (
            "add precise postprocess or abstain rules for visible slot-risk patterns "
            "before broad coverage expansion"
        ),
    }


def _top_slot_key_counts(
    counts: dict[str, int],
    *,
    limit: int = 5,
) -> list[dict[str, int | str]]:
    return [
        {"slot_key": slot_key, "count": count}
        for slot_key, count in sorted(
            counts.items(),
            key=lambda item: (-int(item[1]), str(item[0])),
        )[:limit]
    ]


def _visible_slot_cue_summary(workspace_root: Path) -> dict[str, Any]:
    data_dir = workspace_root / "data"
    sources: list[tuple[str, Path]] = [("train", data_dir / "train.jsonl")]
    sources.extend((path.stem, path) for path in _visible_validation_paths(workspace_root))
    traces: list[TeacherTrace] = []
    source_splits: list[str] = []
    for split_name, path in sources:
        if not path.exists():
            continue
        source_splits.append(split_name)
        traces.extend(_read_teacher_jsonl(path))
    return _visible_slot_cue_summary_payload(
        traces=traces,
        source_splits=source_splits,
    )


def _visible_slot_cue_summary_payload(
    *,
    traces: list[TeacherTrace],
    source_splits: list[str],
    item_limit: int = 64,
) -> dict[str, Any]:
    by_slot: dict[str, dict[str, Any]] = {}
    for trace in traces:
        teacher_frame = trace.teacher_frame
        if teacher_frame is None:
            continue
        for slot_key, slot_value in teacher_frame.slots.items():
            value = str(slot_value)
            item = by_slot.setdefault(
                slot_key,
                {
                    "slot_key": slot_key,
                    "total": 0,
                    "teacher_intents": {},
                    "values": {},
                    "examples": [],
                },
            )
            item["total"] += 1
            item["teacher_intents"][teacher_frame.intent] = (
                item["teacher_intents"].get(teacher_frame.intent, 0) + 1
            )
            item["values"][value] = item["values"].get(value, 0) + 1
            if len(item["examples"]) < 3:
                item["examples"].append(
                    {
                        "request_id": trace.request_id,
                        "utterance": trace.utterance,
                        "teacher_intent": teacher_frame.intent,
                        "slot_value": value,
                    }
                )
    items = [
        {
            "slot_key": item["slot_key"],
            "slot_key_terms": _slot_key_terms(str(item["slot_key"])),
            "total": int(item["total"]),
            "top_teacher_intents": _top_intent_counts(item["teacher_intents"]),
            "top_values": _top_value_counts(item["values"]),
            "examples": item["examples"],
        }
        for item in by_slot.values()
    ]
    items.sort(key=lambda item: (-int(item["total"]), str(item["slot_key"])))
    return {
        "schema_version": "l2-target-visible-slot-cue-summary-v1",
        "visibility": "visible_validation_only",
        "usage_hint": (
            "For slotless or missing-slot accepted frames, use slot_key_terms, "
            "top_values, and examples as visible support for conservative "
            "postprocess or veto rules. This is diagnostic-only."
        ),
        "source_splits": source_splits,
        "item_limit": item_limit,
        "items": items[:item_limit],
        "empty_reason": None if items else "no_visible_slot_values",
    }


def _slot_key_terms(slot_key: str) -> list[str]:
    return [term for term in slot_key.lower().replace("-", "_").split("_") if term]


def _top_intent_counts(
    counts: dict[str, int],
    *,
    limit: int = 5,
) -> list[dict[str, int | str]]:
    return [
        {"intent": intent, "count": count}
        for intent, count in sorted(
            counts.items(),
            key=lambda item: (-int(item[1]), str(item[0])),
        )[:limit]
    ]


def _top_value_counts(
    counts: dict[str, int],
    *,
    limit: int = 8,
) -> list[dict[str, int | str]]:
    return [
        {"value": value, "count": count}
        for value, count in sorted(
            counts.items(),
            key=lambda item: (-int(item[1]), str(item[0])),
        )[:limit]
    ]


def _target_diagnostics_payload(
    *,
    state_kind: Literal["before_round", "final"],
    round_index: int,
    baseline: dict[str, Any],
    round_results: list[dict[str, Any]],
    visible_slot_cue_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    recent_rounds = round_results[-6:]
    latest_metric = (
        recent_rounds[-1]["inner_validation"]
        if recent_rounds
        else baseline["inner_validation"]
    )
    latest_train_audit = (
        _candidate_train_audit(recent_rounds[-1])
        if recent_rounds
        else _candidate_train_audit(baseline)
    )
    latest_cross_audit = (
        _candidate_visible_cross_audit(recent_rounds[-1])
        if recent_rounds
        else _candidate_visible_cross_audit(baseline)
    )
    return {
        "schema_version": "l2-target-diagnostics-v1",
        "visibility": "visible_validation_only",
        "state_kind": state_kind,
        "next_round": round_index,
        "round_history_window": "last_6",
        "visible_slot_cue_summary": visible_slot_cue_summary,
        "baseline_inner_validation": baseline["inner_validation"].get(
            "family_diagnostics",
        ),
        "baseline_safety_backlog": _metric_safety_backlog(baseline["inner_validation"]),
        "baseline_slot_risk_backlog": _metric_slot_risk_backlog(
            baseline["inner_validation"],
        ),
        "baseline_intent_confusion_backlog": _metric_intent_confusion_backlog(
            baseline["inner_validation"],
        ),
        "baseline_train_audit": _metric_family_diagnostics(
            _candidate_train_audit(baseline),
        ),
        "baseline_train_audit_safety_backlog": _metric_safety_backlog(
            _candidate_train_audit(baseline),
        ),
        "baseline_train_audit_slot_risk_backlog": _metric_slot_risk_backlog(
            _candidate_train_audit(baseline),
        ),
        "baseline_train_audit_intent_confusion_backlog": (
            _metric_intent_confusion_backlog(_candidate_train_audit(baseline))
        ),
        "latest_inner_validation": latest_metric.get("family_diagnostics"),
        "latest_safety_backlog": _metric_safety_backlog(latest_metric),
        "latest_slot_risk_backlog": _metric_slot_risk_backlog(latest_metric),
        "latest_intent_confusion_backlog": _metric_intent_confusion_backlog(
            latest_metric,
        ),
        "latest_train_audit": _metric_family_diagnostics(latest_train_audit),
        "latest_train_audit_safety_backlog": _metric_safety_backlog(
            latest_train_audit or {},
        ),
        "latest_train_audit_slot_risk_backlog": _metric_slot_risk_backlog(
            latest_train_audit or {},
        ),
        "latest_train_audit_intent_confusion_backlog": (
            _metric_intent_confusion_backlog(latest_train_audit or {})
        ),
        "baseline_visible_cross_audit": _metric_family_diagnostics(
            _candidate_visible_cross_audit(baseline),
        ),
        "baseline_visible_cross_audit_safety_backlog": _metric_safety_backlog(
            _candidate_visible_cross_audit(baseline) or {},
        ),
        "baseline_visible_cross_audit_slot_risk_backlog": _metric_slot_risk_backlog(
            _candidate_visible_cross_audit(baseline) or {},
        ),
        "baseline_visible_cross_audit_intent_confusion_backlog": (
            _metric_intent_confusion_backlog(
                _candidate_visible_cross_audit(baseline) or {},
            )
        ),
        "latest_visible_cross_audit": _metric_family_diagnostics(latest_cross_audit),
        "latest_visible_cross_audit_safety_backlog": _metric_safety_backlog(
            latest_cross_audit or {},
        ),
        "latest_visible_cross_audit_slot_risk_backlog": _metric_slot_risk_backlog(
            latest_cross_audit or {},
        ),
        "latest_visible_cross_audit_intent_confusion_backlog": (
            _metric_intent_confusion_backlog(latest_cross_audit or {})
        ),
        "round_history": [
            {
                "round": round_result["round"],
                "inner_improved": round_result["inner_improved"],
                "family_diagnostics": round_result["inner_validation"].get(
                    "family_diagnostics",
                ),
                "safety_backlog": _metric_safety_backlog(
                    round_result["inner_validation"],
                ),
                "slot_risk_backlog": _metric_slot_risk_backlog(
                    round_result["inner_validation"],
                ),
                "intent_confusion_backlog": _metric_intent_confusion_backlog(
                    round_result["inner_validation"],
                ),
                "train_audit_safety_backlog": _metric_safety_backlog(
                    _candidate_train_audit(round_result),
                ),
                "train_audit_slot_risk_backlog": _metric_slot_risk_backlog(
                    _candidate_train_audit(round_result),
                ),
                "train_audit_intent_confusion_backlog": (
                    _metric_intent_confusion_backlog(
                        _candidate_train_audit(round_result),
                    )
                ),
                "visible_cross_audit_safety_backlog": _metric_safety_backlog(
                    _candidate_visible_cross_audit(round_result) or {},
                ),
                "visible_cross_audit_slot_risk_backlog": _metric_slot_risk_backlog(
                    _candidate_visible_cross_audit(round_result) or {},
                ),
                "visible_cross_audit_intent_confusion_backlog": (
                    _metric_intent_confusion_backlog(
                        _candidate_visible_cross_audit(round_result) or {},
                    )
                ),
            }
            for round_result in recent_rounds
        ],
    }


def _candidate_train_audit(candidate: dict[str, Any]) -> dict[str, Any]:
    metric = candidate.get("train_audit")
    if isinstance(metric, dict):
        return metric
    fallback = dict(candidate["inner_validation"])
    fallback["split"] = "train_audit"
    fallback["gate_role"] = "diagnostic_only_not_selection_or_adoption_gate"
    return fallback


def _candidate_visible_cross_audit(candidate: dict[str, Any]) -> dict[str, Any] | None:
    metric = candidate.get("visible_cross_audit")
    return metric if isinstance(metric, dict) else None


def _metric_family_diagnostics(metric: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(metric, dict):
        return None
    family_diagnostics = metric.get("family_diagnostics")
    return family_diagnostics if isinstance(family_diagnostics, dict) else None


def _metric_safety_backlog(metric: dict[str, Any]) -> dict[str, Any] | None:
    safety_backlog = metric.get("safety_backlog")
    if isinstance(safety_backlog, dict):
        return safety_backlog
    family_diagnostics = metric.get("family_diagnostics")
    if not isinstance(family_diagnostics, dict):
        return None
    nested_backlog = family_diagnostics.get("safety_backlog")
    return nested_backlog if isinstance(nested_backlog, dict) else None


def _metric_slot_risk_backlog(metric: dict[str, Any]) -> dict[str, Any] | None:
    slot_risk_backlog = metric.get("slot_risk_backlog")
    if isinstance(slot_risk_backlog, dict):
        return slot_risk_backlog
    family_diagnostics = metric.get("family_diagnostics")
    if not isinstance(family_diagnostics, dict):
        return None
    nested_backlog = family_diagnostics.get("slot_risk_backlog")
    return nested_backlog if isinstance(nested_backlog, dict) else None


def _metric_intent_confusion_backlog(
    metric: dict[str, Any],
) -> dict[str, Any] | None:
    intent_confusion_backlog = metric.get("intent_confusion_backlog")
    if isinstance(intent_confusion_backlog, dict):
        return intent_confusion_backlog
    family_diagnostics = metric.get("family_diagnostics")
    if not isinstance(family_diagnostics, dict):
        return None
    nested_backlog = family_diagnostics.get("intent_confusion_backlog")
    return nested_backlog if isinstance(nested_backlog, dict) else None


def _write_target_state_files(
    *,
    workspace_root: Path,
    config: L2TargetEvolutionConfig,
    round_index: int,
    state_kind: Literal["before_round", "final"],
    target_scope: dict[str, Any],
    baseline: dict[str, Any],
    round_results: list[dict[str, Any]],
    no_inner_improvement_rounds: int,
    agent_rounds_started: int,
    agent_rounds_succeeded: int,
) -> None:
    data_dir = workspace_root / "data"
    visible_slot_cue_summary = _visible_slot_cue_summary(workspace_root)
    objective = _target_objective_payload(
        config,
        teacher_labeled_traces=target_scope["scoped_teacher_labeled_traces"],
        target_scope=target_scope,
    )
    state = {
        "schema_version": "l2-target-round-state-v1",
        "state_kind": state_kind,
        "next_round": round_index,
        "rounds_requested": config.rounds,
        "no_inner_improvement_rounds": no_inner_improvement_rounds,
        "target_scope": target_scope,
        "budget_policy": _target_budget_policy_payload(config),
        "evidence_policy": _target_evidence_policy_payload(
            config,
            rounds_completed=len(round_results),
            teacher_labeled_traces=target_scope["scoped_teacher_labeled_traces"],
        ),
        "agent_budget": _agent_budget_payload(
            config=config,
            agent_rounds_started=agent_rounds_started,
            agent_rounds_succeeded=agent_rounds_succeeded,
        ),
        "candidate_selection_gate": (
            "visible validation gate, visible support gate, visible train-audit "
            "safety gate, and private selection holdout gate must all pass"
        ),
        "early_stop_policy": (
            "private selection is evaluated for outer candidate selection, but does not stop "
            "the inner loop unless stop_on_selection_gate is explicitly enabled"
        ),
        "baseline_inner_validation": _visible_metric_summary(baseline["inner_validation"]),
        "baseline_train_audit": _visible_metric_summary(_candidate_train_audit(baseline)),
        "baseline_visible_cross_audit": _optional_visible_metric_summary(
            _candidate_visible_cross_audit(baseline),
        ),
        "train_audit_policy": (
            "visible train audit is a safety gate for accepted wrongs; it is not "
            "a coverage target and does not expose private holdouts"
        ),
        "visible_support_policy": (
            "before private selection, visible validation must retain at least "
            f"{MIN_VISIBLE_CORRECT_ACCEPTS_PER_VALIDATION_FOLD} correct accepts "
            "per visible validation fold"
        ),
        "visible_cross_audit_policy": (
            "visible cross-audit is diagnostic-only; it retrains on visible folds to "
            "simulate selection-like pressure without exposing private holdouts"
        ),
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
    (data_dir / "target_diagnostics.json").write_text(
        json.dumps(
            _target_diagnostics_payload(
                state_kind=state_kind,
                round_index=round_index,
                baseline=baseline,
                round_results=round_results,
                visible_slot_cue_summary=visible_slot_cue_summary,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (data_dir / "commands.md").write_text(_target_commands_text(), encoding="utf-8")


def _agent_budget_payload(
    *,
    config: L2TargetEvolutionConfig,
    agent_rounds_started: int,
    agent_rounds_succeeded: int,
) -> dict[str, Any]:
    max_rounds = _effective_max_agent_rounds(config)
    remaining = None if max_rounds is None else max(0, max_rounds - agent_rounds_started)
    live_agent_mode = config.mode in {"codex-cli", "agent-session"}
    return {
        "schema_version": "l2-target-agent-budget-v1",
        "applies_to_mode": live_agent_mode,
        "mode": config.mode,
        "codex_command": config.codex_command if live_agent_mode else None,
        "codex_model": config.codex_model if live_agent_mode else None,
        "timeout_s": config.timeout_s if live_agent_mode else None,
        "max_agent_rounds": max_rounds,
        "agent_rounds_started": agent_rounds_started,
        "agent_rounds_succeeded": agent_rounds_succeeded,
        "agent_rounds_remaining": remaining,
        "agent_session_scope": (
            "single_session_agent_controls_internal_loop"
            if config.mode == "agent-session"
            else "one_codex_process_per_outer_round"
            if config.mode == "codex-cli"
            else None
        ),
        "local_search_consumes_llm": False,
        "prompt_strategy": (
            "stable short stdin prompt; dynamic target context stays in workspace files"
        ),
        "cost_policy": (
            "live agent modes consume GPT budget; local-search/tool calls are cheap "
            "deterministic/Optuna work and do not count as live agent launches"
        ),
    }


def _target_budget_policy_payload(
    config: L2TargetEvolutionConfig,
    *,
    max_agent_rounds: int | None = None,
) -> dict[str, Any]:
    resolved_max_agent_rounds = (
        _effective_max_agent_rounds(config)
        if max_agent_rounds is None
        else max_agent_rounds
    )
    return {
        "inner_patience_rounds": config.inner_patience_rounds,
        "stop_on_selection_gate": config.stop_on_selection_gate,
        "local_search_trials": config.local_search_trials,
        "local_search_timeout_s": config.local_search_timeout_s,
        "local_search_space": config.local_search_space,
        "local_search_cross_audit_top_k": config.local_search_cross_audit_top_k,
        "max_agent_rounds": resolved_max_agent_rounds,
        "profile": config.budget_profile,
        "profile_intent": _target_budget_profile_intent_payload(
            config,
            max_agent_rounds=resolved_max_agent_rounds,
        ),
        "visible_validation_folds": config.visible_validation_folds,
        "visible_validation_ratio": config.visible_validation_ratio,
        "visible_cross_audit_folds": config.visible_cross_audit_folds,
    }


def _target_budget_profile_intent_payload(
    config: L2TargetEvolutionConfig,
    *,
    max_agent_rounds: int | None,
) -> dict[str, Any]:
    if config.budget_profile == "fixed-inner":
        profile_role = "fixed_snapshot_research"
        guidance = (
            "Use this profile for the main L2 target-evolution research loop; "
            "it is intentionally decoupled from outer replay cadence."
        )
    elif config.budget_profile == "smoke":
        profile_role = "connectivity_smoke"
        guidance = (
            "Use this profile only to check wiring; do not treat its result as "
            "evidence about L2 evolve quality."
        )
    else:
        profile_role = "cost_capped_default"
        guidance = (
            "The standard profile is cost-capped. For codex-cli it may launch "
            "only a few live agent rounds, so failure here is not evidence that "
            "L2 target evolution has been exhausted."
        )
    return {
        "schema_version": "l2-target-budget-profile-intent-v1",
        "profile": config.budget_profile,
        "profile_role": profile_role,
        "recommended_quality_profile": "fixed-inner",
        "guidance": guidance,
        "fixed_trace_snapshot_inner_loop": True,
        "outer_replay_cadence_bound": False,
        "rounds_are_l2_train_eval_iterations": config.mode != "agent-session",
        "agent_session_controls_internal_loop": config.mode == "agent-session",
        "local_search_consumes_llm": False,
        "codex_cli_rounds_consume_llm": config.mode == "codex-cli",
        "live_agent_session_consumes_llm": config.mode == "agent-session",
        "effective_max_agent_rounds": max_agent_rounds,
        "agent_round_cap_is_cost_control": (
            config.mode in {"codex-cli", "agent-session"} and max_agent_rounds is not None
        ),
    }


def _target_evidence_policy_payload(
    config: L2TargetEvolutionConfig,
    *,
    max_agent_rounds: int | None = None,
    rounds_completed: int | None = None,
    stop_reason: str | None = None,
    teacher_labeled_traces: int | None = None,
) -> dict[str, Any]:
    resolved_max_agent_rounds = (
        _effective_max_agent_rounds(config)
        if max_agent_rounds is None
        else max_agent_rounds
    )
    min_quality_rounds = 16
    min_quality_codex_agent_rounds = 8
    min_quality_teacher_labeled_traces = 500
    blocking_reasons: list[str] = []

    if config.budget_profile == "smoke":
        evidence_class = "connectivity_smoke"
        blocking_reasons.append("smoke profile only validates wiring")
    elif config.budget_profile == "standard":
        evidence_class = "cost_capped_probe"
        blocking_reasons.append(
            "standard profile is cost-capped and may launch only a few live agent rounds"
        )
    elif config.rounds < min_quality_rounds:
        evidence_class = "short_fixed_snapshot_probe"
        blocking_reasons.append(
            f"round budget {config.rounds} is below quality minimum {min_quality_rounds}"
        )
    elif (
        teacher_labeled_traces is not None
        and teacher_labeled_traces < min_quality_teacher_labeled_traces
    ):
        evidence_class = "small_snapshot_probe"
        blocking_reasons.append(
            "teacher-labeled snapshot size "
            f"{teacher_labeled_traces} is below quality minimum "
            f"{min_quality_teacher_labeled_traces}"
        )
    elif (
        config.mode == "codex-cli"
        and resolved_max_agent_rounds is not None
        and resolved_max_agent_rounds < min_quality_codex_agent_rounds
    ):
        evidence_class = "agent_budget_capped_fixed_snapshot"
        blocking_reasons.append(
            "codex-cli agent round cap is below the quality evidence minimum"
        )
    elif config.mode == "agent-session" and resolved_max_agent_rounds == 0:
        evidence_class = "agent_session_not_launched_probe"
        blocking_reasons.append("agent-session mode did not launch a live agent session")
    elif (
        config.mode == "agent-session"
        and stop_reason is not None
        and (
            stop_reason != "agent_session_completed"
            or rounds_completed is None
            or rounds_completed < 1
        )
    ):
        evidence_class = "incomplete_agent_session_probe"
        blocking_reasons.append(
            "agent-session did not complete one scoped candidate evaluation"
        )
    else:
        evidence_class = "fixed_snapshot_research"

    if (
        evidence_class == "fixed_snapshot_research"
        and config.mode != "agent-session"
        and stop_reason is not None
        and rounds_completed is not None
        and rounds_completed < min(config.rounds, min_quality_rounds)
    ):
        if stop_reason not in {"selection_gate_passed", "baseline_selection_gate_passed"}:
            evidence_class = "incomplete_fixed_snapshot_probe"
            blocking_reasons.append(
                f"completed {rounds_completed} rounds before reaching the requested evidence budget"
            )

    quality_claim_supported = evidence_class == "fixed_snapshot_research"
    return {
        "schema_version": "l2-target-evidence-policy-v1",
        "evidence_class": evidence_class,
        "quality_claim_supported": quality_claim_supported,
        "quality_claim": (
            "eligible_after_private_gates_and_outer_replay"
            if quality_claim_supported
            else "not_supported_by_this_run"
        ),
        "result_interpretation": (
            "May be used as L2 target-evolution quality evidence only after private "
            "selection/promotion gates and outer e2e replay also pass."
            if quality_claim_supported
            else (
                "Use this run for wiring, debugging, or bounded probing only; do not "
                "treat failure as evidence that L2 target evolution is exhausted."
            )
        ),
        "required_for_quality_claim": {
            "budget_profile": "fixed-inner",
            "min_rounds_requested": min_quality_rounds,
            "min_codex_cli_agent_rounds": min_quality_codex_agent_rounds,
            "agent_session_requires_one_completed_session": True,
            "min_teacher_labeled_traces": min_quality_teacher_labeled_traces,
            "requires_private_selection_gate": True,
            "requires_private_promotion_gate": True,
            "requires_outer_replay": True,
        },
        "blocking_reasons": blocking_reasons,
        "profile": config.budget_profile,
        "mode": config.mode,
        "rounds_requested": config.rounds,
        "rounds_completed": rounds_completed,
        "stop_reason": stop_reason,
        "teacher_labeled_traces": teacher_labeled_traces,
        "fixed_trace_snapshot_inner_loop": True,
        "outer_replay_cadence_bound": False,
        "effective_max_agent_rounds": resolved_max_agent_rounds,
        "agent_round_cap_is_cost_control": (
            config.mode in {"codex-cli", "agent-session"}
            and resolved_max_agent_rounds is not None
        ),
    }


def _effective_max_agent_rounds(config: L2TargetEvolutionConfig) -> int | None:
    if config.mode not in {"codex-cli", "agent-session"}:
        return config.max_agent_rounds
    if config.max_agent_rounds is not None:
        return config.max_agent_rounds
    if config.mode == "agent-session":
        return 1
    if config.budget_profile == "standard":
        return 3
    if config.budget_profile == "fixed-inner":
        return 16
    return 1


def _target_objective_payload(
    config: L2TargetEvolutionConfig,
    *,
    teacher_labeled_traces: int | None = None,
    target_scope: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "l2-target-objective-v1",
        "primary_objective": "increase safe L2 accepts on unseen target traffic",
        "target_scope": target_scope,
        "budget_policy": _target_budget_policy_payload(config),
        "evidence_policy": _target_evidence_policy_payload(
            config,
            teacher_labeled_traces=teacher_labeled_traces,
        ),
        "agent_budget": _agent_budget_payload(
            config=config,
            agent_rounds_started=0,
            agent_rounds_succeeded=0,
        ),
        "gates": {
            "min_accepted_accuracy": config.min_accepted_accuracy,
            "max_wrong_accept_rate": config.max_wrong_accept_rate,
            "candidate_selection": (
                "visible validation gate AND visible support gate AND visible "
                "train-audit safety gate AND private selection holdout gate"
            ),
            "adoption": (
                "visible validation gate AND visible support gate AND visible "
                "train-audit safety gate AND private selection holdout gate "
                "AND private promotion holdout gate"
            ),
            "visible_support": (
                "at least "
                f"{MIN_VISIBLE_CORRECT_ACCEPTS_PER_VALIDATION_FOLD} correct accepts "
                "per visible validation fold before private selection"
            ),
            "train_audit_safety": "zero accepted wrong on visible train audit",
            "visible_validation_folds": config.visible_validation_folds,
            "visible_validation_ratio": config.visible_validation_ratio,
            "visible_cross_audit_folds": config.visible_cross_audit_folds,
            "visible_cross_audit_role": "diagnostic_only_not_selection_or_adoption_gate",
        },
        "workspace_scope": _workspace_scope_policy_payload(),
        "optimization_order": [
            "zero or lower wrong accepts",
            "clear visible safety_backlog accepted-wrong families before coverage work",
            "clear visible train-audit accepted wrongs before private selection",
            "retain enough visible correct accepts before private selection",
            "visible validation gate must pass before candidate selection",
            "accepted accuracy at or above gate",
            "coverage increase only after safety gates",
            "lower latency for equally safe behavior",
        ],
        "invalid_strategies": [
            "raw coverage increase with lower frame exactness",
            "lowering threshold when visible validation gate fails",
            "lowering accept_threshold only to raise raw accepts after visible support passes",
            "expanding near-miss coverage while safety_backlog has accepted-wrong items",
            "single-visible-row exact utterance exceptions or request-id memorization",
            "treating a fixed edit/evaluate/search script as the agent plan",
            "treating private selection success alone as candidate success",
            "candidate code changes outside target/",
            "modifying data/, tools/, system/darjeeling/, or program.md",
            "using private holdout rows or aggregate feedback",
            "hardcoding MASSIVE-specific behavior from outside visible data",
        ],
        "allowed_strategies": [
            "target-dependent code derived from visible train and validation-fold files",
            (
                "config_overrides for bounded L2StudentConfig parameters when they "
                "are needed for support and remain safe under visible audits"
            ),
            (
                "agent-invoked Optuna/config search over visible train/validation "
                "with optional visible cross-audit top-k rerank"
            ),
            "visible cross-audit diagnostics for selection-like safety pressure",
            "postprocess_frame fixes that preserve exact frame correctness",
            "accept_prediction veto logic that abstains when uncertain",
            "near_miss_examples-driven mechanisms that still pass visible validation gate",
            (
                "target-specific lexical or state-machine rules derived from visible "
                "target data with multiple visible supports or clear schema semantics"
            ),
        ],
        "agent_session_policy": {
            "session_closure": "one L4 agent session per target-evolve job",
            "internal_loop_control": "agent_decides_edit_evaluate_optuna_stop",
            "outer_harness_role": (
                "prepare workspace, launch agent, check scope, evaluate private gates"
            ),
            "adoption_authority": "outer_private_gates_and_outer_replay",
        },
    }


def _target_code_policy_payload() -> dict[str, Any]:
    return {
        "core_must_remain_dataset_independent": True,
        "target_dependent_code_allowed_in": "target/",
        "target_specific_code_is_not_rejected_for_dataset_dependence": True,
        "target_code_visibility_rule": (
            "target code may be derived from data/train.jsonl and "
            "visible data/inner_validation*.jsonl only"
        ),
        "generalization_rule": (
            "avoid exact utterance exceptions from a single visible row; prefer "
            "pattern-level rules with multiple visible supports or clear schema semantics"
        ),
        "private_holdout_visibility": (
            "selection/promotion holdouts remain outside the agent workspace"
        ),
        "adoption_authority": (
            "visible validation/support/train-audit gates, "
            "private selection/promotion gates, and final outer replay"
        ),
    }


def _workspace_scope_policy_payload() -> dict[str, Any]:
    return {
        "schema_version": "l2-target-workspace-scope-v1",
        "candidate_code_writable_roots": ["target/"],
        "scratch_writable_roots": ["runs/"],
        "protected_roots": ["data/", "system/darjeeling/", "tools/", "program.md"],
        "ignored_generated_files": ["__pycache__/", ".pytest_cache/", "*.pyc", "*.pyo"],
        "enforcement": "checked_after_each_mutating_round_or_session_before_candidate_evaluation",
    }


def _protected_workspace_snapshot(workspace_root: Path) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    if not workspace_root.exists():
        return snapshot
    for path in workspace_root.rglob("*"):
        if not path.is_file():
            continue
        rel_path = path.relative_to(workspace_root)
        if _is_workspace_scope_ignored(rel_path) or _is_workspace_scope_writable(rel_path):
            continue
        snapshot[rel_path.as_posix()] = _file_sha256(path)
    return snapshot


def _workspace_scope_violation_report(
    *,
    workspace_root: Path,
    before: dict[str, str],
) -> dict[str, Any] | None:
    after = _protected_workspace_snapshot(workspace_root)
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    modified = sorted(path for path in set(before) & set(after) if before[path] != after[path])
    if not added and not removed and not modified:
        return None
    return {
        "schema_version": "l2-target-workspace-scope-violation-v1",
        "policy": _workspace_scope_policy_payload(),
        "added_protected_files": added,
        "removed_protected_files": removed,
        "modified_protected_files": modified,
        "message": (
            "target evolution rounds may change target/ and write runs/ scratch outputs; "
            "protected workspace files changed before candidate evaluation"
        ),
    }


def _is_workspace_scope_writable(rel_path: Path) -> bool:
    parts = rel_path.parts
    return bool(parts and parts[0] in {"target", "runs"})


def _is_workspace_scope_ignored(rel_path: Path) -> bool:
    ignored_dirs = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
    if any(part in ignored_dirs for part in rel_path.parts):
        return True
    return rel_path.suffix in {".pyc", ".pyo"}


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _workspace_scope_violation_command_result(
    *,
    workspace_root: Path,
    round_index: int,
    report: dict[str, Any],
) -> dict[str, Any]:
    return {
        "command": ["workspace-scope-check", "--round", str(round_index)],
        "cwd": str(workspace_root),
        "started_at": datetime.now(UTC).isoformat(),
        "return_code": 1,
        "stdout": "",
        "stderr": json.dumps(report, sort_keys=True),
        "workspace_scope_violation": report,
    }


def _target_commands_text() -> str:
    return "\n".join(
        [
            "# Commands",
            "",
            "Evaluate all visible validation folds:",
            "",
            "```bash",
            "uv run --project system/darjeeling python tools/evaluate.py \\",
            "  --split visible_validation \\",
            "  --out runs/visible_validation.json",
            "```",
            "",
            "Evaluate visible train audit diagnostics only:",
            "",
            "```bash",
            "uv run --project system/darjeeling python tools/evaluate.py \\",
            "  --split train_audit \\",
            "  --out runs/train_audit.json",
            "```",
            "",
            "If `uv` cannot use its dependency cache but a Python >=3.11 environment",
            "with Darjeeling dependencies is already active, use:",
            "",
            "```bash",
            "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=system/darjeeling/src \\",
            "  python tools/evaluate.py --split visible_validation \\",
            "  --out runs/visible_validation.json",
            "```",
            "",
            "The same fallback can evaluate train audit with `--split train_audit`.",
            "",
            "Run visible slot-cue probes for target-local veto/postprocess checks:",
            "",
            "```bash",
            "uv run --project system/darjeeling python tools/evaluate.py \\",
            "  --split slot_cue_probes \\",
            "  --out runs/slot_cue_probes.json",
            "```",
            "",
            "Evaluate visible cross-audit diagnostics when enabled:",
            "",
            "```bash",
            "uv run --project system/darjeeling python tools/evaluate.py \\",
            "  --split visible_cross_audit \\",
            f"  --visible-cross-audit-folds {DEFAULT_TARGET_VISIBLE_CROSS_AUDIT_FOLDS} \\",
            "  --out runs/visible_cross_audit.json",
            "```",
            "",
            "Run local Optuna config search on visible train/validation folds only:",
            "",
            "```bash",
            "uv run --project system/darjeeling python tools/search_config.py \\",
            f"  --trials {DEFAULT_TARGET_LOCAL_SEARCH_TRIALS} \\",
            (
                "  --cross-audit-folds "
                f"{DEFAULT_TARGET_VISIBLE_CROSS_AUDIT_FOLDS} \\"
            ),
            (
                "  --cross-audit-top-k "
                f"{DEFAULT_TARGET_LOCAL_SEARCH_CROSS_AUDIT_TOP_K} \\"
            ),
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
                f"--cross-audit-folds {DEFAULT_TARGET_VISIBLE_CROSS_AUDIT_FOLDS} "
                f"--cross-audit-top-k {DEFAULT_TARGET_LOCAL_SEARCH_CROSS_AUDIT_TOP_K} "
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
            "Inspect target family diagnostics:",
            "",
            "```bash",
            "python3 -m json.tool data/target_diagnostics.json | head -120",
            "```",
            "",
            "Inspect visible safety backlog first:",
            "",
            "```bash",
            (
                "python3 - <<'PY'\n"
                "import json\n"
                "d=json.load(open('data/target_diagnostics.json'))\n"
                "print(json.dumps(d.get('latest_safety_backlog'), indent=2))\n"
                "PY"
            ),
            "```",
            "",
            "Only edit candidate code under `target/`. `runs/` is scratch output.",
            "Do not modify `data/`, `tools/`, `system/darjeeling/`, or `program.md`.",
        ]
    )


def _visible_round_summary(round_result: dict[str, Any]) -> dict[str, Any]:
    visible_support_gate = round_result.get("visible_support_gate")
    if not isinstance(visible_support_gate, dict):
        visible_support_gate = _visible_support_gate_payload(
            round_result["inner_validation"],
        )
    return {
        "round": round_result["round"],
        "inner_improved": round_result["inner_improved"],
        "passes_visible_validation_gate": bool(
            round_result["inner_validation"]["passes_gate"],
        ),
        "passes_visible_support_gate": bool(visible_support_gate["passes_gate"]),
        "visible_support_gate": visible_support_gate,
        "passes_train_audit_safety_gate": _passes_train_audit_safety_gate(
            round_result,
        ),
        "inner_score": round_result["inner_score"],
        "inner_delta_vs_baseline": round_result["inner_delta_vs_baseline"],
        "inner_validation": _visible_metric_summary(round_result["inner_validation"]),
        "train_audit": _visible_metric_summary(_candidate_train_audit(round_result)),
        "visible_cross_audit": _optional_visible_metric_summary(
            _candidate_visible_cross_audit(round_result),
        ),
    }


def _optional_visible_metric_summary(metric: dict[str, Any] | None) -> dict[str, Any] | None:
    if metric is None:
        return None
    return _visible_metric_summary(metric)


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
        "gate_role": metric.get("gate_role", "metric_gate"),
        "veto_examples": metric.get("veto_examples", []),
        "near_miss_examples": metric.get("near_miss_examples", []),
        "family_diagnostics": metric.get("family_diagnostics"),
        "safety_backlog": _metric_safety_backlog(metric),
        "slot_risk_backlog": _metric_slot_risk_backlog(metric),
        "intent_confusion_backlog": _metric_intent_confusion_backlog(metric),
        "visible_cross_audit_folds_completed": metric.get(
            "visible_cross_audit_folds_completed",
        ),
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
    parser.add_argument(
        "--split",
        choices=[
            "inner_validation",
            "visible_validation",
            "train_audit",
            "visible_cross_audit",
            "slot_cue_probes",
        ],
        required=True,
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-accepted-accuracy", type=float, default=0.93)
    parser.add_argument("--max-wrong-accept-rate", type=float, default=0.05)
    parser.add_argument(
        "--visible-cross-audit-folds",
        type=int,
        default=DEFAULT_TARGET_VISIBLE_CROSS_AUDIT_FOLDS,
    )
    args = parser.parse_args(argv)
    payload = evaluate_target_workspace(
        workspace_root=args.workspace,
        split=args.split,
        min_accepted_accuracy=args.min_accepted_accuracy,
        max_wrong_accept_rate=args.max_wrong_accept_rate,
        visible_cross_audit_folds=args.visible_cross_audit_folds,
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
    cross_audit_folds: int = 0,
    cross_audit_top_k: int = 0,
    min_accepted_accuracy: float = 0.93,
    max_wrong_accept_rate: float = 0.05,
) -> dict[str, Any]:
    """Tune target-owned L2 config using only visible train/validation data."""

    if trials < 1:
        raise ValueError("trials must be at least 1")
    if search_space not in {"compact", "wide"}:
        raise ValueError("search_space must be compact or wide")
    if cross_audit_folds == 1 or cross_audit_folds < 0:
        raise ValueError("cross_audit_folds must be 0 or at least 2")
    if cross_audit_top_k < 0:
        raise ValueError("cross_audit_top_k must be non-negative")

    config_path = workspace_root / "target" / "config.json"
    original_config_text = config_path.read_text(encoding="utf-8") if config_path.exists() else None
    current_inner = evaluate_target_workspace(
        workspace_root=workspace_root,
        split=_visible_gate_split(workspace_root),
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
                split=_visible_gate_split(workspace_root),
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
        current_cross_audit = None
        cross_audit_rerank_enabled = cross_audit_folds >= 2 and cross_audit_top_k > 0
        if cross_audit_rerank_enabled:
            _restore_target_config_json(config_path, original_config_text)
            current_cross_audit = evaluate_target_workspace(
                workspace_root=workspace_root,
                split="visible_cross_audit",
                min_accepted_accuracy=min_accepted_accuracy,
                max_wrong_accept_rate=max_wrong_accept_rate,
                visible_cross_audit_folds=cross_audit_folds,
            )
            for report in _local_search_cross_audit_candidates(
                completed,
                limit=cross_audit_top_k,
            ):
                _write_target_config_json(
                    config_path,
                    L2StudentConfig(**report["config"]),
                )
                metric = evaluate_target_workspace(
                    workspace_root=workspace_root,
                    split="visible_cross_audit",
                    min_accepted_accuracy=min_accepted_accuracy,
                    max_wrong_accept_rate=max_wrong_accept_rate,
                    visible_cross_audit_folds=cross_audit_folds,
                )
                report["visible_cross_audit"] = _visible_metric_summary(metric)
                report["cross_audit_score"] = list(_inner_score(metric))
        best_report = max(
            completed,
            key=lambda report: _local_search_report_score(report),
            default=None,
        )
        applied = False
        cross_audit_safety_veto = False
        applied_reason = "no completed local-search trial"
        if best_report is not None:
            if (
                cross_audit_rerank_enabled
                and not _local_search_report_passes_cross_audit(best_report)
            ):
                _restore_target_config_json(config_path, original_config_text)
                cross_audit_safety_veto = True
                applied_reason = (
                    "best visible/cross-audit config failed visible cross-audit safety gate"
                )
            elif _local_search_report_score(best_report) > _local_search_current_score(
                current_inner=current_inner,
                current_cross_audit=current_cross_audit,
                cross_audit_enabled=cross_audit_rerank_enabled,
            ):
                _write_target_config_json(
                    config_path,
                    L2StudentConfig(**best_report["config"]),
                )
                applied = True
                applied_reason = (
                    "best visible/cross-audit config improved current target"
                    if cross_audit_rerank_enabled
                    else "best visible validation config improved current target"
                )
            else:
                _restore_target_config_json(config_path, original_config_text)
                applied_reason = (
                    "best visible/cross-audit config did not improve current target"
                    if cross_audit_rerank_enabled
                    else "best visible validation config did not improve current target"
                )
        else:
            _restore_target_config_json(config_path, original_config_text)
        return {
            "schema_version": "l2-target-local-search-v1",
            "search_space": search_space,
            "trials_requested": trials,
            "trials_completed": len(completed),
            "timeout_s": timeout_s,
            "cross_audit_folds": cross_audit_folds,
            "cross_audit_top_k": cross_audit_top_k,
            "cross_audit_rerank_enabled": cross_audit_rerank_enabled,
            "current_inner_validation": _visible_metric_summary(current_inner),
            "current_visible_cross_audit": _optional_visible_metric_summary(
                current_cross_audit,
            ),
            "best_trial_number": best_report["number"] if best_report is not None else None,
            "best_value": best_report["value"] if best_report is not None else None,
            "best_config": best_report["config"] if best_report is not None else None,
            "best_inner_validation": (
                best_report["inner_validation"] if best_report is not None else None
            ),
            "best_visible_cross_audit": (
                best_report.get("visible_cross_audit") if best_report is not None else None
            ),
            "applied": applied,
            "cross_audit_safety_veto": cross_audit_safety_veto,
            "applied_reason": applied_reason,
            "private_holdout_visibility": (
                "local search used only agent-visible train and validation-fold data"
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
    parser.add_argument("--cross-audit-folds", type=int, default=0)
    parser.add_argument("--cross-audit-top-k", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-accepted-accuracy", type=float, default=0.93)
    parser.add_argument("--max-wrong-accept-rate", type=float, default=0.05)
    args = parser.parse_args(argv)
    payload = run_local_target_search(
        workspace_root=args.workspace,
        trials=args.trials,
        search_space=args.search_space,
        timeout_s=args.timeout_s,
        cross_audit_folds=args.cross_audit_folds,
        cross_audit_top_k=args.cross_audit_top_k,
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


def _local_search_cross_audit_candidates(
    reports: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    return sorted(
        reports,
        key=lambda report: (
            _inner_score(report["inner_validation"]),
            float(report["value"] or -1_000_000.0),
        ),
        reverse=True,
    )[:limit]


def _local_search_report_score(
    report: dict[str, Any],
) -> tuple[tuple[bool, int, float, float], tuple[bool, int, float, float], float]:
    return (
        _inner_score(report["inner_validation"]),
        _optional_inner_score(report.get("visible_cross_audit")),
        float(report["value"] or -1_000_000.0),
    )


def _local_search_report_passes_cross_audit(report: dict[str, Any]) -> bool:
    metric = report.get("visible_cross_audit")
    return isinstance(metric, dict) and bool(metric.get("passes_gate"))


def _local_search_current_score(
    *,
    current_inner: dict[str, Any],
    current_cross_audit: dict[str, Any] | None,
    cross_audit_enabled: bool,
) -> tuple[tuple[bool, int, float, float], tuple[bool, int, float, float], float]:
    cross_score = (
        _inner_score(current_cross_audit)
        if cross_audit_enabled and current_cross_audit is not None
        else _empty_metric_score()
    )
    return (_inner_score(current_inner), cross_score, 0.0)


def _optional_inner_score(metric: Any) -> tuple[bool, int, float, float]:
    if isinstance(metric, dict):
        return _inner_score(metric)
    return _empty_metric_score()


def _empty_metric_score() -> tuple[bool, int, float, float]:
    return (False, -1_000_000_000, 0.0, 0.0)


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
        "cross_audit_folds": report.get("cross_audit_folds"),
        "cross_audit_top_k": report.get("cross_audit_top_k"),
        "cross_audit_rerank_enabled": report.get("cross_audit_rerank_enabled"),
        "best_trial_number": report["best_trial_number"],
        "best_value": report["best_value"],
        "best_inner_validation": report["best_inner_validation"],
        "best_visible_cross_audit": report.get("best_visible_cross_audit"),
        "applied": report["applied"],
        "cross_audit_safety_veto": report.get("cross_audit_safety_veto", False),
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
            "Edit candidate code only under `target/`.",
            "`runs/` is available for scratch command output.",
            "Do not modify `data/`, `tools/`, `system/darjeeling/`, or `program.md`.",
            "",
            "Workspace layout:",
            "- `target/` is editable target-specific L2 code.",
            "- `runs/` is scratch output; it is not promoted.",
            "- `target/config.json` contains target-specific L2StudentConfig overrides.",
            "- `system/darjeeling/` is read-only Darjeeling core/evaluator code.",
            "- `data/train.jsonl` is visible training data.",
            "- `data/inner_validation.jsonl` is visible fast feedback data.",
            "- `data/inner_validation_shadow_*.jsonl` are additional visible",
            "  validation folds when present.",
            "- `data/objective.json` defines gates and invalid strategies.",
            "- `data/round_state.json` contains visible validation history.",
            "- `data/target_diagnostics.json` summarizes visible validation",
            "  opportunities and risks by teacher intent family.",
            "  Read `latest_safety_backlog` first; if it has items, clear those",
            "  visible accepted-wrong families before working on near-miss coverage.",
            "  If accepted-wrong backlogs are empty, read `latest_slot_risk_backlog`",
            "  and related train/cross-audit slot-risk queues before stopping;",
            "  they show visible intent-correct slot mismatches that may become",
            "  accepted wrongs under broader coverage. Review both `items` and",
            "  `high_guard_items`; the latter catches lower-frequency risks near",
            "  the accept threshold. Use `missing_slot_keys`, `extra_slot_keys`,",
            "  and `changed_slot_keys` to choose precise postprocess or veto rules.",
            "  Then review `latest_intent_confusion_backlog` and related",
            "  train/cross-audit intent-confusion queues for high-guard teacher",
            "  intent / predicted intent mismatches.",
            "  `visible_slot_cue_summary` summarizes visible teacher slot keys",
            "  and values across train/validation rows; use it to generalize",
            "  clear slot cues such as room words without reading private rows.",
            "  Before stopping, check its `slot_key_terms`, `top_values`, and",
            "  examples against any slotless or missing-slot accepted frames;",
            "  visible cues such as podcast, radio, room, or joke-about terms",
            "  support conservative target-local vetoes or exact postprocess.",
            "  Mandatory cue checks when the corresponding visible slot keys are",
            "  present: non-podcast accepted intents containing a podcast cue;",
            "  slotless accepts containing visible room values such as kitchen,",
            "  bedroom, living room, bathroom, room, or house; generic radio",
            "  station phrases accepted as concrete `radio_name`; radio/music",
            "  utterances accepted without a visible media slot; calendar removes",
            "  with date cues accepted without `date`; bare upcoming events",
            "  accepted as `recommendation_events`; `general_joke` accepts with",
            "  joke adjectives, superlatives, or `joke about ...` but no",
            "  `joke_type` slot; and volume changes with spoken amounts but no",
            "  `change_amount` slot.",
            "  Run `tools/evaluate.py --split slot_cue_probes` after editing",
            "  target cue rules; this diagnostic is visible-only and not a private",
            "  selection/adoption gate.",
            "  `latest_train_audit_safety_backlog` is visible train feedback.",
            "  If it contains accepted wrongs, clear them before stopping; train",
            "  audit is a safety gate, not a coverage target.",
            "  `latest_visible_cross_audit_safety_backlog` retrains on visible",
            "  folds to expose selection-like risks without using private holdouts.",
            "- `data/commands.md` lists local commands.",
            "- `tools/evaluate.py` trains/evaluates the target code in seconds.",
            "- `tools/search_config.py` runs visible-data Optuna config search",
            "  when you decide tuning is useful.",
            "",
            "This is one autonomous L4 agent session. The outer harness does not",
            "prescribe an edit/evaluate/search script. You decide how many times to",
            "inspect context, edit `target/`, evaluate visible splits, run Optuna,",
            "debug, and stop. Stop when the visible objective is met, no safe",
            "progress remains, risk is too high, or budget is near exhaustion.",
            "",
            "Optimize generalization from the visible train and validation-fold data.",
            "Wrong accepts are worse than abstentions. A raw coverage increase is not",
            "useful if frame exactness or wrong-accept safety gets worse.",
            "A target round is selectable only if visible validation passes,",
            "visible validation keeps enough correct accepts, visible train audit",
            "has zero accepted wrongs, and the outer private selection holdout",
            "passes. Private selection alone is not success if visible validation",
            "or train audit has wrong accepts.",
            "Adoption also requires the private promotion holdout to pass.",
            "Private selection is an outer selection signal, not a signal you can",
            "read during this session. Adoption is decided after this session exits.",
            "Use `near_miss_examples` from visible validation to find safe",
            "coverage opportunities, and use `wrong_examples` / `veto_examples`",
            "to tighten safety. Do not lower threshold globally unless the visible",
            "inner gate still passes.",
            "Prefer `target_diagnostics.json` for choosing the next family to work on;",
            "it gives bounded family-level counts without exposing private holdouts.",
            "When `latest_safety_backlog.items` is non-empty, fix those visible",
            "accepted-wrong families before coverage expansion or broad threshold",
            "changes.",
            "When accepted-wrong backlogs are empty, review the visible slot-risk",
            "backlogs before stopping; inspect `high_guard_items` as well as the",
            "count-ranked `items`, and prefer precise postprocess or abstention",
            "rules based on the listed slot-key deltas over broad threshold changes.",
            "After slot-risk review, inspect intent-confusion backlogs for repeated",
            "high-guard wrong-intent pairs such as media intent boundary errors.",
            "Use `visible_slot_cue_summary` when a risk appears to need a slot",
            "cue that may be supported by other visible intents.",
            "Do not treat a slotless accepted frame as safe until you have checked",
            "`visible_slot_cue_summary` for visible slot cues that the frame omits;",
            "prefer a veto over accepting a high-guard frame with an obvious",
            "missing visible slot cue.",
            "Concretely, add conservative checks for podcast cues accepted as a",
            "non-podcast intent, room values accepted without a location slot, and",
            "joke cues accepted without `joke_type` when visible data supports",
            "those slot keys. Also handle generic radio-station names, radio",
            "music cues, calendar date cues, bare upcoming-events intent",
            "boundaries, and spoken volume amounts when the related visible slot",
            "keys are present.",
            "Use `uv run --project system/darjeeling python tools/evaluate.py",
            "--split slot_cue_probes --out runs/slot_cue_probes.json` or the",
            "documented fallback to verify those checks locally.",
            "When `latest_train_audit_safety_backlog.items` is non-empty, prefer",
            "abstention or target-local vetoes over accepting predictions that",
            "contradict visible teacher labels.",
            "Do not stop with a near-zero-coverage target that only passes by",
            "abstaining from almost everything; keep at least two correct accepts",
            "per visible validation fold before relying on private selection.",
            "Once visible support passes, do not lower `accept_threshold` or add",
            "`target/config.json` just to increase raw accepts. Prefer keeping",
            "target-local veto/postprocess rules and remove threshold-lowering",
            "config if it is only recovering coverage after safety vetoes.",
            "If visible validation backlog is empty but candidate selection still",
            "fails, use visible cross-audit and train-audit backlogs to design",
            "broader safety rules; do not inspect private holdout rows.",
            "`target.accept_prediction` may veto uncertain guard accepts; it cannot",
            "force accepts that the core guard rejected.",
            "Private selection and promotion holdouts are outside this workspace and",
            "only the outer harness can read them; do not try to access parent",
            "directories to inspect them.",
            "",
            "It is acceptable for `target/` to contain target-dependent code derived",
            "from visible train and validation-fold data. This is not a",
            "Darjeeling-core dataset-independence violation. It becomes invalid only",
            "if it moves into core code, uses private holdout rows or aggregates, or",
            "uses MASSIVE/external dataset knowledge that is not visible here.",
            "Do not add exact utterance exceptions from a single visible row or",
            "memorize request IDs. Prefer pattern-level lexical or slot-support",
            "rules backed by multiple visible examples or clear schema semantics.",
            "Use local config search for cheap tuning when useful; reserve code edits",
            "for changes that require target-specific design judgment.",
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


def _run_agent_session(
    *,
    config: L2TargetEvolutionConfig,
    workspace_root: Path,
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
    prompt = "\n".join(
        [
            "Read program.md and run one autonomous L2 target evolution session.",
            "You decide how many times to inspect, edit target/, evaluate, and run",
            "tools/search_config.py within the available budget. Stop when the",
            "visible objective is met, no safe progress remains, or budget/risk says",
            "to stop. Leave the final candidate in target/.",
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
            *_split_metric_score(item[split]),
            *_inner_tie_breaker_score(item.get("inner_validation", {})),
            int(item.get("round") or 0),
        ),
    )


def _split_metric_score(metric: dict[str, Any]) -> tuple[bool, float, float, float, int]:
    return (
        bool(metric["passes_gate"]),
        float(metric["coverage"]),
        float(metric["accepted_accuracy"] or 0.0),
        -float(metric["wrong_accept_rate"]),
        -int(metric.get("wrong_accepts") or 0),
    )


def _inner_tie_breaker_score(metric: dict[str, Any]) -> tuple[bool, int, float, float]:
    return (
        bool(metric.get("passes_gate")),
        -int(metric.get("wrong_accepts") or 0),
        float(metric.get("accepted_accuracy") or 0.0),
        float(metric.get("coverage") or 0.0),
    )


def _best_selection_round(rounds: list[dict[str, Any]]) -> dict[str, Any] | None:
    passing_rounds = [
        round_result
        for round_result in rounds
        if _passes_candidate_selection_gate(round_result)
    ]
    if not passing_rounds:
        return None
    return _best_round_for_split(passing_rounds, "selection_holdout")


def _best_adoptable_round(rounds: list[dict[str, Any]]) -> dict[str, Any] | None:
    passing_rounds = [
        round_result
        for round_result in rounds
        if _passes_adoption_gate(round_result)
    ]
    if not passing_rounds:
        return None
    return _best_round_for_split(passing_rounds, "promotion_holdout")


def _passes_train_audit_safety_gate(round_result: dict[str, Any]) -> bool:
    train_audit = _candidate_train_audit(round_result)
    return int(train_audit.get("wrong_accepts") or 0) == 0


def _visible_validation_fold_count(metric: dict[str, Any]) -> int:
    splits = metric.get("visible_validation_splits")
    if isinstance(splits, list) and splits:
        return len(splits)
    folds = metric.get("visible_validation_folds")
    if isinstance(folds, list) and folds:
        return len(folds)
    return 1


def _visible_support_gate_payload(metric: dict[str, Any]) -> dict[str, Any]:
    fold_count = _visible_validation_fold_count(metric)
    validation_size = int(metric.get("validation_size") or 0)
    min_correct_accepts = MIN_VISIBLE_CORRECT_ACCEPTS_PER_VALIDATION_FOLD * fold_count
    if validation_size > 0:
        min_correct_accepts = min(min_correct_accepts, validation_size)
    correct_accepts = int(metric.get("correct_accepts") or 0)
    return {
        "schema_version": "l2-target-visible-support-gate-v1",
        "visible_validation_folds": fold_count,
        "min_correct_accepts": min_correct_accepts,
        "correct_accepts": correct_accepts,
        "passes_gate": correct_accepts >= min_correct_accepts,
        "policy": (
            "retain at least "
            f"{MIN_VISIBLE_CORRECT_ACCEPTS_PER_VALIDATION_FOLD} correct accepts "
            "per visible validation fold before private selection"
        ),
    }


def _passes_visible_support_gate(round_result: dict[str, Any]) -> bool:
    support_gate = round_result.get("visible_support_gate")
    if not isinstance(support_gate, dict):
        support_gate = _visible_support_gate_payload(round_result["inner_validation"])
    return bool(support_gate["passes_gate"])


def _passes_visible_selection_inputs(round_result: dict[str, Any]) -> bool:
    return (
        bool(round_result["inner_validation"]["passes_gate"])
        and _passes_visible_support_gate(round_result)
        and _passes_train_audit_safety_gate(round_result)
    )


def _passes_candidate_selection_gate(round_result: dict[str, Any]) -> bool:
    return _passes_visible_selection_inputs(round_result) and bool(
        round_result["selection_holdout"]["passes_gate"],
    )


def _passes_adoption_gate(round_result: dict[str, Any]) -> bool:
    return _passes_candidate_selection_gate(round_result) and bool(
        round_result["promotion_holdout"]["passes_gate"],
    )


def _private_holdout_evidence(rounds: list[dict[str, Any]]) -> dict[str, Any]:
    best_round = _best_round(rounds)
    inner_passing_rounds = [
        round_result
        for round_result in rounds
        if bool(round_result.get("inner_validation", {}).get("passes_gate"))
    ]
    visible_support_rounds = [
        round_result
        for round_result in inner_passing_rounds
        if _passes_visible_support_gate(round_result)
    ]
    visible_selection_input_rounds = [
        round_result
        for round_result in visible_support_rounds
        if _passes_train_audit_safety_gate(round_result)
    ]
    selected_round = _best_selection_round(rounds)
    adoptable_round = _best_adoptable_round(rounds)
    return {
        "schema_version": "l2-target-private-holdout-evidence-v1",
        "visibility": "outer_summary_only_not_agent_workspace",
        "best_round": best_round.get("round") if best_round else None,
        "best_round_selection": (
            _holdout_split_evidence(best_round.get("selection_holdout", {}))
            if best_round
            else None
        ),
        "best_round_promotion": (
            _holdout_split_evidence(best_round.get("promotion_holdout", {}))
            if best_round
            else None
        ),
        "inner_passing_rounds": len(inner_passing_rounds),
        "visible_support_passing_rounds": len(visible_support_rounds),
        "visible_selection_input_passing_rounds": len(visible_selection_input_rounds),
        "inner_passing_visible_support_failed_rounds": (
            len(inner_passing_rounds) - len(visible_support_rounds)
        ),
        "inner_passing_train_audit_wrong_accept_rounds": sum(
            1
            for round_result in inner_passing_rounds
            if not _passes_train_audit_safety_gate(round_result)
        ),
        "inner_passing_selection_zero_accept_rounds": sum(
            1
            for round_result in visible_selection_input_rounds
            if int(round_result.get("selection_holdout", {}).get("accepted") or 0) == 0
        ),
        "inner_passing_selection_wrong_accept_rounds": sum(
            1
            for round_result in visible_selection_input_rounds
            if int(round_result.get("selection_holdout", {}).get("wrong_accepts") or 0) > 0
        ),
        "selection_gate_diagnosis": _selection_gate_diagnosis(rounds),
        "adoption_gate_diagnosis": _adoption_gate_diagnosis(
            selected_round=selected_round,
            adoptable_round=adoptable_round,
        ),
        "recommendation": _private_holdout_recommendation(rounds),
    }


def _holdout_split_evidence(metric: dict[str, Any]) -> dict[str, Any]:
    accepted = int(metric.get("accepted") or 0)
    wrong_accepts = int(metric.get("wrong_accepts") or 0)
    accepted_accuracy = metric.get("accepted_accuracy")
    return {
        "accepted": accepted,
        "correct_accepts": int(metric.get("correct_accepts") or 0),
        "wrong_accepts": wrong_accepts,
        "coverage": float(metric.get("coverage") or 0.0),
        "accepted_accuracy": accepted_accuracy,
        "passes_gate": bool(metric.get("passes_gate")),
        "status": _holdout_metric_status(metric),
    }


def _holdout_metric_status(metric: dict[str, Any]) -> str:
    accepted = int(metric.get("accepted") or 0)
    wrong_accepts = int(metric.get("wrong_accepts") or 0)
    if bool(metric.get("passes_gate")):
        return "passes_gate"
    if accepted == 0:
        return "zero_accepts"
    if wrong_accepts > 0:
        return "wrong_accepts"
    return "failed_gate"


def _selection_gate_diagnosis(rounds: list[dict[str, Any]]) -> str:
    if not rounds:
        return "no_rounds"
    if _best_selection_round(rounds) is not None:
        return "selection_gate_passed"
    inner_passing_rounds = [
        round_result
        for round_result in rounds
        if bool(round_result.get("inner_validation", {}).get("passes_gate"))
    ]
    if not inner_passing_rounds:
        return "visible_validation_gate_failed"
    visible_support_rounds = [
        round_result
        for round_result in inner_passing_rounds
        if _passes_visible_support_gate(round_result)
    ]
    if not visible_support_rounds:
        return "visible_support_gate_failed"
    visible_selection_input_rounds = [
        round_result
        for round_result in visible_support_rounds
        if _passes_train_audit_safety_gate(round_result)
    ]
    if not visible_selection_input_rounds:
        return "train_audit_safety_gate_failed"
    if all(
        int(round_result.get("selection_holdout", {}).get("accepted") or 0) == 0
        for round_result in visible_selection_input_rounds
    ):
        return "selection_zero_accepts_for_inner_passing_rounds"
    if any(
        int(round_result.get("selection_holdout", {}).get("wrong_accepts") or 0) > 0
        for round_result in visible_selection_input_rounds
    ):
        return "selection_wrong_accepts_for_inner_passing_rounds"
    return "selection_gate_failed_for_inner_passing_rounds"


def _adoption_gate_diagnosis(
    *,
    selected_round: dict[str, Any] | None,
    adoptable_round: dict[str, Any] | None,
) -> str:
    if adoptable_round is not None:
        return "adoption_gate_passed"
    if selected_round is None:
        return "selection_gate_not_passed"
    promotion = selected_round.get("promotion_holdout", {})
    return f"promotion_{_holdout_metric_status(promotion)}"


def _private_holdout_recommendation(rounds: list[dict[str, Any]]) -> str:
    diagnosis = _selection_gate_diagnosis(rounds)
    if diagnosis == "selection_zero_accepts_for_inner_passing_rounds":
        return (
            "selection holdout did not observe candidate accepts; keep the target "
            "non-adopted unless an explicit outer replay passes, or rerun with a "
            "larger/stratified target split"
        )
    if diagnosis == "train_audit_safety_gate_failed":
        return (
            "clear visible train-audit accepted wrongs before private selection; "
            "abstain rather than override visible teacher labels"
        )
    if diagnosis == "visible_support_gate_failed":
        return (
            "retain enough visible correct accepts before private selection; "
            "avoid targets that pass only by abstaining from almost everything"
        )
    if diagnosis == "selection_wrong_accepts_for_inner_passing_rounds":
        return "fix wrong accepts before staging; do not trade frame exactness for coverage"
    if diagnosis == "visible_validation_gate_failed":
        return "continue visible validation improvement before using private holdout evidence"
    return "use adoption_decision and outer replay before promoting target artifacts"


def _selection_decision(rounds: list[dict[str, Any]]) -> dict[str, Any]:
    best_selection = _best_selection_round(rounds)
    if best_selection is None:
        return {
            "selected": False,
            "round": None,
            "reason": (
                "no target round passed visible validation, visible support, visible "
                "train-audit safety, and private selection gates"
            ),
        }
    return {
        "selected": True,
        "round": best_selection["round"],
        "reason": (
            "target round passed visible validation, visible support, visible "
            "train-audit safety, and private selection gates"
        ),
    }


def _adoption_decision(rounds: list[dict[str, Any]]) -> dict[str, Any]:
    selection = _selection_decision(rounds)
    if not selection["selected"]:
        return {
            "adopted": False,
            "round": None,
            "reason": (
                "no target round passed visible validation, visible support, visible "
                "train-audit safety, and private selection gates"
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
