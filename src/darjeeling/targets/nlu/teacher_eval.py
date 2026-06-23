from __future__ import annotations

import csv
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from darjeeling.runtime.cost import replay_cost_model_from_settings
from darjeeling.targets.nlu.data import DataRecord
from darjeeling.targets.nlu.layers.l4_cloud_llm import (
    CloudLLMTeacher,
    TeacherCallResult,
)
from darjeeling.targets.nlu.replay import (
    load_processed_records,
    select_stream,
    task_schema_from_records,
)
from darjeeling.targets.nlu.schemas import Frame, TaskSchema
from darjeeling.targets.nlu.settings import Settings
from darjeeling.targets.nlu.teacher import (
    TEACHER_PROMPT_V1,
    TEACHER_PROMPT_V2_INTENT_FIRST,
    ensure_supported_teacher_prompt_version,
)

DEFAULT_TEACHER_PROMPT_COMPARISON = (
    TEACHER_PROMPT_V1,
    TEACHER_PROMPT_V2_INTENT_FIRST,
)
TEACHER_EVAL_SUMMARY_FILENAME = "teacher_live_vs_gold.summary.json"
TEACHER_EVAL_DETAILS_CSV_FILENAME = "teacher_live_vs_gold.details.csv"
TEACHER_EVAL_DETAILS_JSONL_FILENAME = "teacher_live_vs_gold.details.jsonl"
TEACHER_EVAL_RUN_MANIFEST_FILENAME = "teacher_live_vs_gold.run.json"
TEACHER_EVAL_COST_LEDGER_FILENAME = "teacher_live_vs_gold.cost_ledger.json"
TEACHER_PROMPT_COMPARISON_JSON_FILENAME = "teacher_prompt_comparison.json"
TEACHER_PROMPT_COMPARISON_CSV_FILENAME = "teacher_prompt_comparison.csv"


@dataclass(frozen=True)
class TeacherLiveEvalResult:
    summary: dict[str, Any]
    rows: list[dict[str, Any]]


@dataclass(frozen=True)
class TeacherLiveEvalArtifactResult:
    out_dir: Path
    summary_json_path: Path
    details_csv_path: Path
    details_jsonl_path: Path
    summary: dict[str, Any]
    cost_ledger_path: Path | None = None
    run_manifest_path: Path | None = None


@dataclass(frozen=True)
class TeacherPromptComparisonResult:
    out_dir: Path
    comparison_json_path: Path
    comparison_csv_path: Path
    prompt_results: list[TeacherLiveEvalArtifactResult]
    summary: dict[str, Any]


def run_teacher_live_vs_gold(
    *,
    data_dir: Path,
    split: str,
    stream: str,
    max_requests: int,
    prompt_version: str,
    settings: Settings,
    out_dir: Path,
    min_frame_exact_match: float = 0.0,
    teacher: Any | None = None,
) -> TeacherLiveEvalArtifactResult:
    records = load_processed_records(data_dir, split=split)
    task_schema = task_schema_from_records(records)
    sample_records = _sample_records(records, stream=stream, max_requests=max_requests)
    settings_for_prompt = settings.model_copy(update={"teacher_prompt_version": prompt_version})
    result = evaluate_live_teacher_vs_gold(
        records=sample_records,
        task_schema=task_schema,
        settings=settings_for_prompt,
        split=split,
        stream=stream,
        prompt_version=prompt_version,
        min_frame_exact_match=min_frame_exact_match,
        teacher=teacher,
    )
    return write_teacher_live_eval_artifacts(result, out_dir=out_dir)


def run_teacher_prompt_comparison(
    *,
    data_dir: Path,
    split: str,
    stream: str,
    max_requests: int,
    prompt_versions: Sequence[str],
    settings: Settings,
    out_dir: Path,
    min_frame_exact_match: float = 0.0,
    teacher_factory: Callable[[Settings], Any] | None = None,
) -> TeacherPromptComparisonResult:
    records = load_processed_records(data_dir, split=split)
    task_schema = task_schema_from_records(records)
    sample_records = _sample_records(records, stream=stream, max_requests=max_requests)
    prompt_results: list[TeacherLiveEvalArtifactResult] = []
    for prompt_version in prompt_versions:
        ensure_supported_teacher_prompt_version(prompt_version)
        settings_for_prompt = settings.model_copy(update={"teacher_prompt_version": prompt_version})
        teacher = teacher_factory(settings_for_prompt) if teacher_factory is not None else None
        eval_result = evaluate_live_teacher_vs_gold(
            records=sample_records,
            task_schema=task_schema,
            settings=settings_for_prompt,
            split=split,
            stream=stream,
            prompt_version=prompt_version,
            min_frame_exact_match=min_frame_exact_match,
            teacher=teacher,
        )
        prompt_results.append(
            write_teacher_live_eval_artifacts(
                eval_result,
                out_dir=out_dir / _prompt_dir_name(prompt_version),
            )
        )
    return write_teacher_prompt_comparison(
        prompt_results=prompt_results,
        out_dir=out_dir,
        split=split,
        stream=stream,
        max_requests=max_requests,
    )


def evaluate_live_teacher_vs_gold(
    *,
    records: Sequence[DataRecord],
    task_schema: TaskSchema,
    settings: Settings,
    split: str,
    stream: str,
    prompt_version: str,
    min_frame_exact_match: float = 0.0,
    teacher: Any | None = None,
) -> TeacherLiveEvalResult:
    ensure_supported_teacher_prompt_version(prompt_version)
    teacher = teacher or CloudLLMTeacher(settings)
    cost_model = replay_cost_model_from_settings(settings)
    rows: list[dict[str, Any]] = []

    for index, record in enumerate(records, start=1):
        started = perf_counter()
        try:
            call_result = teacher.answer(record.utterance, task_schema)
            latency_ms = (perf_counter() - started) * 1000.0
            rows.append(
                _teacher_eval_row(
                    index=index,
                    record=record,
                    task_schema=task_schema,
                    call_result=call_result,
                    latency_ms=latency_ms,
                    cost_usd=_observed_l4_cost_usd(cost_model, call_result.usage),
                    cost_model=cost_model,
                    prompt_version=prompt_version,
                    default_model=settings.openai_model,
                )
            )
        except Exception as exc:
            latency_ms = (perf_counter() - started) * 1000.0
            rows.append(
                _teacher_eval_error_row(
                    index=index,
                    record=record,
                    task_schema=task_schema,
                    error=exc,
                    latency_ms=latency_ms,
                    cost_model=cost_model,
                    prompt_version=prompt_version,
                    default_model=settings.openai_model,
                )
            )

    summary = _teacher_eval_summary(
        rows,
        settings=settings,
        split=split,
        stream=stream,
        prompt_version=prompt_version,
        min_frame_exact_match=min_frame_exact_match,
    )
    return TeacherLiveEvalResult(summary=summary, rows=rows)


def write_teacher_live_eval_artifacts(
    result: TeacherLiveEvalResult,
    *,
    out_dir: Path,
) -> TeacherLiveEvalArtifactResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / TEACHER_EVAL_SUMMARY_FILENAME
    details_csv_path = out_dir / TEACHER_EVAL_DETAILS_CSV_FILENAME
    details_jsonl_path = out_dir / TEACHER_EVAL_DETAILS_JSONL_FILENAME
    cost_ledger_path = out_dir / TEACHER_EVAL_COST_LEDGER_FILENAME
    run_manifest_path = out_dir / TEACHER_EVAL_RUN_MANIFEST_FILENAME
    sorted_rows = _sort_rows_by_index(result.rows)

    summary_path.write_text(
        json.dumps(result.summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_details_csv(details_csv_path, sorted_rows)
    details_jsonl_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in sorted_rows),
        encoding="utf-8",
    )
    write_teacher_live_eval_cost_ledger(
        rows=sorted_rows,
        summary=result.summary,
        path=cost_ledger_path,
    )
    return TeacherLiveEvalArtifactResult(
        out_dir=out_dir,
        summary_json_path=summary_path,
        details_csv_path=details_csv_path,
        details_jsonl_path=details_jsonl_path,
        cost_ledger_path=cost_ledger_path,
        summary=result.summary,
        run_manifest_path=run_manifest_path if run_manifest_path.exists() else None,
    )


def write_teacher_prompt_comparison(
    *,
    prompt_results: Sequence[TeacherLiveEvalArtifactResult],
    out_dir: Path,
    split: str,
    stream: str,
    max_requests: int,
) -> TeacherPromptComparisonResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [_comparison_row(result.summary) for result in prompt_results]
    summary = {
        "schema_version": "nlu-teacher-prompt-comparison-v1",
        "benchmark": "teacher-live-vs-gold",
        "split": split,
        "stream": stream,
        "max_requests": max_requests,
        "prompt_versions": [row["prompt_version"] for row in rows],
        "same_sample": _same_sample_request_ids(prompt_results),
        "sample_request_ids": (
            prompt_results[0].summary.get("request_ids", []) if prompt_results else []
        ),
        "rows": rows,
    }
    comparison_json_path = out_dir / TEACHER_PROMPT_COMPARISON_JSON_FILENAME
    comparison_csv_path = out_dir / TEACHER_PROMPT_COMPARISON_CSV_FILENAME
    comparison_json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_comparison_csv(comparison_csv_path, rows)
    return TeacherPromptComparisonResult(
        out_dir=out_dir,
        comparison_json_path=comparison_json_path,
        comparison_csv_path=comparison_csv_path,
        prompt_results=list(prompt_results),
        summary=summary,
    )


def _sample_records(
    records: Sequence[DataRecord],
    *,
    stream: str,
    max_requests: int,
) -> list[DataRecord]:
    return [
        item.record
        for item in select_stream(list(records), stream=stream, max_requests=max_requests)
    ]


def teacher_live_eval_run_identity(
    *,
    records: Sequence[DataRecord],
    task_schema: TaskSchema,
    settings: Settings,
    split: str,
    stream: str,
    prompt_version: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "nlu-teacher-live-eval-run-v1",
        "benchmark": "teacher-live-vs-gold",
        "split": split,
        "stream": stream,
        "prompt_version": prompt_version,
        "model": settings.openai_model,
        "task_schema_version": task_schema.schema_version,
        "sample_request_ids": [record.request_id for record in records],
        "request_count": len(records),
        **(extra or {}),
    }


def write_teacher_live_eval_run_manifest(
    *,
    out_dir: Path,
    run_identity: dict[str, Any],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / TEACHER_EVAL_RUN_MANIFEST_FILENAME
    path.write_text(
        json.dumps(run_identity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def load_teacher_live_eval_resume_rows(
    *,
    out_dir: Path,
    expected_run_identity: dict[str, Any],
) -> list[dict[str, Any]]:
    manifest_path = out_dir / TEACHER_EVAL_RUN_MANIFEST_FILENAME
    if not manifest_path.exists():
        raise ValueError(f"cannot resume without run manifest: {manifest_path}")
    observed_identity = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_teacher_live_eval_run_identity(
        observed=observed_identity,
        expected=expected_run_identity,
    )

    details_path = out_dir / TEACHER_EVAL_DETAILS_JSONL_FILENAME
    if not details_path.exists():
        return []
    rows = [
        json.loads(line)
        for line in details_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return _validate_resume_rows(
        rows,
        expected_request_ids=expected_run_identity["sample_request_ids"],
    )


def append_teacher_live_eval_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()


def write_teacher_live_eval_cost_ledger(
    *,
    rows: Sequence[dict[str, Any]],
    summary: dict[str, Any],
    path: Path,
) -> Path:
    sorted_rows = _sort_rows_by_index(rows)
    total_cost = sum(float(row.get("cost_usd", 0.0)) for row in sorted_rows)
    final_response_cost = sum(
        float(row.get("final_response_cost_usd", row.get("cost_usd", 0.0)))
        for row in sorted_rows
    )
    ledger = {
        "schema_version": "teacher-live-eval-cost-ledger-v1",
        "benchmark": summary.get("benchmark", "teacher-live-vs-gold"),
        "created_at": datetime.now(UTC).isoformat(),
        "split": summary.get("split", ""),
        "stream": summary.get("stream", ""),
        "prompt_version": summary.get("prompt_version", ""),
        "model": summary.get("model", ""),
        "requests": len(sorted_rows),
        "observed_attempt_cost_usd": total_cost,
        "observed_final_response_cost_usd": final_response_cost,
        "retry_overhead_cost_usd": total_cost - final_response_cost,
        "attempt_count": sum(int(row.get("attempt_count", 1)) for row in sorted_rows),
        "empty_response_attempts": sum(
            int(row.get("empty_response_attempts", 0))
            for row in sorted_rows
        ),
        "final_empty_response_failures": sum(
            1 for row in sorted_rows if row.get("final_empty_response_failure")
        ),
        "unknown_usage_attempts": sum(
            int(row.get("unknown_usage_attempts", 0))
            for row in sorted_rows
        ),
        "source_details_jsonl": str(path.with_name(TEACHER_EVAL_DETAILS_JSONL_FILENAME)),
        "request_costs": [
            {
                "index": row.get("index"),
                "request_id": row.get("request_id"),
                "parse_failure": row.get("parse_failure", False),
                "attempt_count": row.get("attempt_count", 1),
                "cost_usd": row.get("cost_usd", 0.0),
                "final_response_cost_usd": row.get(
                    "final_response_cost_usd",
                    row.get("cost_usd", 0.0),
                ),
                "attempt_cost_usd": row.get("attempt_cost_usd", row.get("cost_usd", 0.0)),
                "unknown_usage_attempts": row.get("unknown_usage_attempts", 0),
            }
            for row in sorted_rows
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _validate_teacher_live_eval_run_identity(
    *,
    observed: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    if observed != expected:
        mismatched = [
            key
            for key in sorted(set(observed) | set(expected))
            if observed.get(key) != expected.get(key)
        ]
        joined = ", ".join(mismatched)
        raise ValueError(f"cannot resume mismatched teacher eval run: {joined}")


def _validate_resume_rows(
    rows: list[dict[str, Any]],
    *,
    expected_request_ids: Sequence[str],
) -> list[dict[str, Any]]:
    expected = set(expected_request_ids)
    seen: set[str] = set()
    for row in rows:
        request_id = str(row.get("request_id", ""))
        if request_id not in expected:
            raise ValueError(f"cannot resume row outside run sample: {request_id}")
        if request_id in seen:
            raise ValueError(f"cannot resume duplicate completed row: {request_id}")
        seen.add(request_id)
    return rows


def _sort_rows_by_index(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: int(row.get("index", 0)))


def _teacher_eval_row(
    *,
    index: int,
    record: DataRecord,
    task_schema: TaskSchema,
    call_result: TeacherCallResult,
    latency_ms: float,
    cost_usd: float,
    cost_model: Any | None = None,
    prompt_version: str,
    default_model: str,
) -> dict[str, Any]:
    teacher_frame = call_result.frame
    gold_frame = record.gold_frame
    unsupported_slots = sorted(set(teacher_frame.slots) - set(task_schema.slot_names))
    invalid_intent = teacher_frame.intent not in task_schema.intent_names
    slot_stats = _slot_pair_stats(teacher_frame, gold_frame)
    attempt_diagnostics = _normalized_attempt_diagnostics(call_result.attempt_diagnostics)
    final_response_tokens = _usage_tokens(call_result.usage)
    attempt_tokens = _attempt_tokens(
        attempt_diagnostics,
        fallback_tokens=final_response_tokens,
    )
    attempt_cost_usd = _attempt_observed_cost_usd(
        attempt_diagnostics,
        cost_model=cost_model,
        fallback_cost_usd=cost_usd,
    )
    return {
        "index": index,
        "request_id": record.request_id,
        "prompt_version": prompt_version,
        "model": call_result.model or default_model,
        "utterance": record.utterance,
        "teacher_frame": teacher_frame.model_dump(mode="json"),
        "gold_frame": gold_frame.model_dump(mode="json"),
        "parse_failure": False,
        "error": "",
        "invalid_intent": invalid_intent,
        "invalid_slot_count": len(unsupported_slots),
        "invalid_slots": unsupported_slots,
        "is_abstain": teacher_frame.is_abstain,
        "frame_exact": teacher_frame == gold_frame,
        "intent_correct": teacher_frame.intent == gold_frame.intent,
        "slot_key_exact": set(teacher_frame.slots) == set(gold_frame.slots),
        **slot_stats,
        "latency_ms": latency_ms,
        "cost_usd": attempt_cost_usd,
        "final_response_cost_usd": cost_usd,
        "attempt_cost_usd": attempt_cost_usd,
        "tokens": attempt_tokens,
        "final_response_tokens": final_response_tokens,
        "attempt_tokens": attempt_tokens,
        "usage": call_result.usage,
        "context_hash": call_result.context_hash,
        "prompt_cache_key": call_result.prompt_cache_key,
        "attempt_count": _attempt_count(attempt_diagnostics, fallback=1),
        "attempt_diagnostics": attempt_diagnostics,
        "empty_response_attempts": _empty_response_attempts(attempt_diagnostics),
        "retry_recovered": _retry_recovered(attempt_diagnostics),
        "final_empty_response_failure": False,
        "unknown_usage_attempts": _unknown_usage_attempts(attempt_diagnostics),
    }


def _teacher_eval_error_row(
    *,
    index: int,
    record: DataRecord,
    task_schema: TaskSchema,
    error: Exception,
    latency_ms: float,
    cost_model: Any | None = None,
    prompt_version: str,
    default_model: str,
) -> dict[str, Any]:
    empty_frame = Frame(intent="", slots={})
    slot_stats = _slot_pair_stats(empty_frame, record.gold_frame)
    attempt_diagnostics = _normalized_attempt_diagnostics(
        getattr(error, "attempt_diagnostics", []),
    )
    usage = _error_usage(error, attempt_diagnostics)
    final_response_tokens = _usage_tokens(usage)
    final_response_cost_usd = _observed_l4_cost_usd(cost_model, usage)
    attempt_tokens = _attempt_tokens(
        attempt_diagnostics,
        fallback_tokens=final_response_tokens,
    )
    attempt_cost_usd = _attempt_observed_cost_usd(
        attempt_diagnostics,
        cost_model=cost_model,
        fallback_cost_usd=final_response_cost_usd,
    )
    return {
        "index": index,
        "request_id": record.request_id,
        "prompt_version": prompt_version,
        "model": _error_model(error, attempt_diagnostics, default_model),
        "utterance": record.utterance,
        "teacher_frame": None,
        "gold_frame": record.gold_frame.model_dump(mode="json"),
        "parse_failure": True,
        "error": str(error),
        "invalid_intent": False,
        "invalid_slot_count": 0,
        "invalid_slots": [],
        "is_abstain": False,
        "frame_exact": False,
        "intent_correct": False,
        "slot_key_exact": False,
        **slot_stats,
        "latency_ms": latency_ms,
        "cost_usd": attempt_cost_usd,
        "final_response_cost_usd": final_response_cost_usd,
        "attempt_cost_usd": attempt_cost_usd,
        "tokens": attempt_tokens,
        "final_response_tokens": final_response_tokens,
        "attempt_tokens": attempt_tokens,
        "usage": usage,
        "context_hash": "",
        "prompt_cache_key": "",
        "attempt_count": _attempt_count(
            attempt_diagnostics,
            fallback=1,
        ),
        "attempt_diagnostics": attempt_diagnostics,
        "empty_response_attempts": _empty_response_attempts(attempt_diagnostics),
        "retry_recovered": False,
        "final_empty_response_failure": _final_empty_response_failure(
            attempt_diagnostics,
            error=error,
        ),
        "unknown_usage_attempts": _unknown_usage_attempts(
            attempt_diagnostics,
            fallback=0 if _has_observed_usage(usage) else 1,
        ),
    }


def _teacher_eval_summary(
    rows: Sequence[dict[str, Any]],
    *,
    settings: Settings,
    split: str,
    stream: str,
    prompt_version: str,
    min_frame_exact_match: float,
) -> dict[str, Any]:
    request_count = len(rows)
    frame_exact = sum(1 for row in rows if row["frame_exact"])
    intent_correct = sum(1 for row in rows if row["intent_correct"])
    slot_key_exact = sum(1 for row in rows if row["slot_key_exact"])
    slot_pair_correct = sum(int(row["slot_pair_correct"]) for row in rows)
    slot_pair_predicted = sum(int(row["slot_pair_predicted"]) for row in rows)
    slot_pair_gold = sum(int(row["slot_pair_gold"]) for row in rows)
    precision = _slot_precision(slot_pair_correct, slot_pair_predicted, slot_pair_gold)
    recall = _slot_recall(slot_pair_correct, slot_pair_gold)
    frame_exact_match = _rate(frame_exact, request_count)
    latencies = [float(row["latency_ms"]) for row in rows]
    total_tokens = sum(float(row["tokens"]) for row in rows)
    total_cost = sum(float(row["cost_usd"]) for row in rows)
    final_response_tokens = sum(
        float(row.get("final_response_tokens", row["tokens"]))
        for row in rows
    )
    final_response_cost = sum(
        float(row.get("final_response_cost_usd", row["cost_usd"]))
        for row in rows
    )
    attempt_count = sum(int(row.get("attempt_count", 1)) for row in rows)
    empty_response_attempts = sum(int(row.get("empty_response_attempts", 0)) for row in rows)
    retry_recovered_rows = sum(1 for row in rows if row.get("retry_recovered"))
    final_empty_response_failures = sum(
        1 for row in rows if row.get("final_empty_response_failure")
    )
    unknown_usage_attempts = sum(int(row.get("unknown_usage_attempts", 0)) for row in rows)
    models = sorted({str(row["model"]) for row in rows if row.get("model")})
    return {
        "schema_version": "nlu-teacher-live-vs-gold-v1",
        "benchmark": "teacher-live-vs-gold",
        "split": split,
        "stream": stream,
        "prompt_version": prompt_version,
        "model": settings.openai_model,
        "models_observed": models,
        "requests": request_count,
        "request_ids": [row["request_id"] for row in rows],
        "parsed_requests": sum(1 for row in rows if not row["parse_failure"]),
        "parse_failure_count": sum(1 for row in rows if row["parse_failure"]),
        "attempt_count": attempt_count,
        "retry_recovered_rows": retry_recovered_rows,
        "empty_response_attempts": empty_response_attempts,
        "final_empty_response_failures": final_empty_response_failures,
        "unknown_usage_attempts": unknown_usage_attempts,
        "invalid_intent_count": sum(1 for row in rows if row["invalid_intent"]),
        "invalid_slot_count": sum(int(row["invalid_slot_count"]) for row in rows),
        "invalid_slot_request_count": sum(1 for row in rows if row["invalid_slot_count"]),
        "abstain_count": sum(1 for row in rows if row["is_abstain"]),
        "frame_exact_match": frame_exact_match,
        "intent_accuracy": _rate(intent_correct, request_count),
        "slot_key_exact_match": _rate(slot_key_exact, request_count),
        "slot_pair_correct": slot_pair_correct,
        "slot_pair_predicted": slot_pair_predicted,
        "slot_pair_gold": slot_pair_gold,
        "slot_pair_precision": precision,
        "slot_pair_recall": recall,
        "slot_pair_f1": _f1(precision, recall),
        "full_l4": {
            "calls": request_count,
            "api_attempts": attempt_count,
            "tokens": total_tokens,
            "tokens_per_request": total_tokens / request_count if request_count else 0.0,
            "cost_usd": total_cost,
            "cost_usd_per_100_requests": (
                total_cost / request_count * 100.0 if request_count else 0.0
            ),
            "attempt_tokens": total_tokens,
            "attempt_cost_usd": total_cost,
            "final_response_tokens": final_response_tokens,
            "final_response_cost_usd": final_response_cost,
            "latency_p50_ms": _percentile(latencies, 50),
            "latency_p95_ms": _percentile(latencies, 95),
        },
        "min_frame_exact_match": min_frame_exact_match,
        "passed": (
            frame_exact_match is not None and frame_exact_match >= min_frame_exact_match
        ),
    }


def _slot_pair_stats(teacher_frame: Frame, gold_frame: Frame) -> dict[str, int]:
    predicted_pairs = set(teacher_frame.slots.items())
    gold_pairs = set(gold_frame.slots.items())
    return {
        "slot_pair_correct": len(predicted_pairs & gold_pairs),
        "slot_pair_predicted": len(predicted_pairs),
        "slot_pair_gold": len(gold_pairs),
    }


def _usage_tokens(usage: Any) -> float:
    if not isinstance(usage, dict):
        return 0.0
    total = _numeric(usage.get("total_tokens"))
    if total is not None:
        return total
    prompt = _numeric(usage.get("prompt_tokens")) or _numeric(usage.get("input_tokens")) or 0.0
    completion = (
        _numeric(usage.get("completion_tokens")) or _numeric(usage.get("output_tokens")) or 0.0
    )
    return prompt + completion


def _normalized_attempt_diagnostics(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _observed_l4_cost_usd(cost_model: Any | None, usage: dict[str, Any]) -> float:
    if cost_model is None or not _has_observed_usage(usage):
        return 0.0
    return float(cost_model.layer_cost_usd("L4", usage))


def _attempt_observed_cost_usd(
    attempts: Sequence[dict[str, Any]],
    *,
    cost_model: Any | None,
    fallback_cost_usd: float,
) -> float:
    if not attempts:
        return fallback_cost_usd
    return sum(_observed_l4_cost_usd(cost_model, _attempt_usage(attempt)) for attempt in attempts)


def _attempt_tokens(
    attempts: Sequence[dict[str, Any]],
    *,
    fallback_tokens: float,
) -> float:
    if not attempts:
        return fallback_tokens
    return sum(_usage_tokens(_attempt_usage(attempt)) for attempt in attempts)


def _attempt_usage(attempt: dict[str, Any]) -> dict[str, Any]:
    usage = attempt.get("usage")
    return dict(usage) if isinstance(usage, dict) else {}


def _has_observed_usage(usage: dict[str, Any]) -> bool:
    return _usage_tokens(usage) > 0.0


def _attempt_count(attempts: Sequence[dict[str, Any]], *, fallback: int) -> int:
    return len(attempts) if attempts else fallback


def _empty_response_attempts(attempts: Sequence[dict[str, Any]]) -> int:
    return sum(1 for attempt in attempts if _is_empty_response_attempt(attempt))


def _is_empty_response_attempt(attempt: dict[str, Any]) -> bool:
    if attempt.get("visible_content_length") == 0:
        return True
    return attempt.get("error_message") == "teacher response content is empty"


def _retry_recovered(attempts: Sequence[dict[str, Any]]) -> bool:
    if len(attempts) <= 1:
        return False
    return bool(attempts[-1].get("success")) and any(
        not bool(attempt.get("success"))
        for attempt in attempts[:-1]
    )


def _final_empty_response_failure(
    attempts: Sequence[dict[str, Any]],
    *,
    error: Exception,
) -> bool:
    if attempts:
        return _is_empty_response_attempt(attempts[-1]) and not bool(attempts[-1].get("success"))
    return str(error) == "teacher response content is empty"


def _unknown_usage_attempts(
    attempts: Sequence[dict[str, Any]],
    *,
    fallback: int = 0,
) -> int:
    if not attempts:
        return fallback
    return sum(1 for attempt in attempts if not _has_observed_usage(_attempt_usage(attempt)))


def _error_usage(error: Exception, attempts: Sequence[dict[str, Any]]) -> dict[str, Any]:
    usage = getattr(error, "teacher_usage", None)
    if isinstance(usage, dict):
        return dict(usage)
    if attempts:
        return _attempt_usage(attempts[-1])
    return {}


def _error_model(
    error: Exception,
    attempts: Sequence[dict[str, Any]],
    default_model: str,
) -> str:
    model = getattr(error, "teacher_model", None)
    if model:
        return str(model)
    for attempt in reversed(attempts):
        response_model = attempt.get("response_model")
        if response_model:
            return str(response_model)
    return default_model


def _numeric(value: Any) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _slot_precision(correct: int, predicted: int, gold: int) -> float:
    if predicted:
        return correct / predicted
    return 1.0 if gold == 0 else 0.0


def _slot_recall(correct: int, gold: int) -> float:
    if gold:
        return correct / gold
    return 1.0


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _write_details_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fieldnames = [
        "index",
        "request_id",
        "prompt_version",
        "model",
        "utterance",
        "teacher_frame",
        "gold_frame",
        "parse_failure",
        "error",
        "invalid_intent",
        "invalid_slot_count",
        "invalid_slots",
        "is_abstain",
        "frame_exact",
        "intent_correct",
        "slot_key_exact",
        "slot_pair_correct",
        "slot_pair_predicted",
        "slot_pair_gold",
        "latency_ms",
        "cost_usd",
        "final_response_cost_usd",
        "attempt_cost_usd",
        "tokens",
        "final_response_tokens",
        "attempt_tokens",
        "usage",
        "context_hash",
        "prompt_cache_key",
        "attempt_count",
        "attempt_diagnostics",
        "empty_response_attempts",
        "retry_recovered",
        "final_empty_response_failure",
        "unknown_usage_attempts",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: (
                        json.dumps(row[field], ensure_ascii=False, sort_keys=True)
                        if isinstance(row.get(field), dict | list)
                        else row.get(field, "")
                    )
                    for field in fieldnames
                }
            )


def _comparison_row(summary: dict[str, Any]) -> dict[str, Any]:
    full_l4 = summary.get("full_l4") or {}
    return {
        "prompt_version": summary.get("prompt_version", ""),
        "model": summary.get("model", ""),
        "requests": summary.get("requests", 0),
        "passed": summary.get("passed", False),
        "frame_exact_match": summary.get("frame_exact_match"),
        "intent_accuracy": summary.get("intent_accuracy"),
        "slot_key_exact_match": summary.get("slot_key_exact_match"),
        "slot_pair_precision": summary.get("slot_pair_precision"),
        "slot_pair_recall": summary.get("slot_pair_recall"),
        "slot_pair_f1": summary.get("slot_pair_f1"),
        "parse_failure_count": summary.get("parse_failure_count", 0),
        "attempt_count": summary.get("attempt_count", 0),
        "retry_recovered_rows": summary.get("retry_recovered_rows", 0),
        "empty_response_attempts": summary.get("empty_response_attempts", 0),
        "final_empty_response_failures": summary.get("final_empty_response_failures", 0),
        "unknown_usage_attempts": summary.get("unknown_usage_attempts", 0),
        "invalid_intent_count": summary.get("invalid_intent_count", 0),
        "invalid_slot_count": summary.get("invalid_slot_count", 0),
        "abstain_count": summary.get("abstain_count", 0),
        "tokens": full_l4.get("tokens", 0.0),
        "cost_usd": full_l4.get("cost_usd", 0.0),
        "latency_p50_ms": full_l4.get("latency_p50_ms", 0.0),
        "latency_p95_ms": full_l4.get("latency_p95_ms", 0.0),
    }


def _write_comparison_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fieldnames = [
        "prompt_version",
        "model",
        "requests",
        "passed",
        "frame_exact_match",
        "intent_accuracy",
        "slot_key_exact_match",
        "slot_pair_precision",
        "slot_pair_recall",
        "slot_pair_f1",
        "parse_failure_count",
        "attempt_count",
        "retry_recovered_rows",
        "empty_response_attempts",
        "final_empty_response_failures",
        "unknown_usage_attempts",
        "invalid_intent_count",
        "invalid_slot_count",
        "abstain_count",
        "tokens",
        "cost_usd",
        "latency_p50_ms",
        "latency_p95_ms",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _same_sample_request_ids(prompt_results: Sequence[TeacherLiveEvalArtifactResult]) -> bool:
    request_ids_by_prompt = [
        tuple(result.summary.get("request_ids", []))
        for result in prompt_results
    ]
    if not request_ids_by_prompt:
        return True
    return all(request_ids == request_ids_by_prompt[0] for request_ids in request_ids_by_prompt)


def _prompt_dir_name(prompt_version: str) -> str:
    return prompt_version.replace("/", "_").replace(":", "_")
