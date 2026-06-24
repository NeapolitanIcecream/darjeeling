from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from darjeeling.runtime.cost import replay_cost_model_from_settings
from darjeeling.targets.nlu.adapters.clinc150 import CLINC150_OOS_INTENT
from darjeeling.targets.nlu.data import DataRecord
from darjeeling.targets.nlu.layers.l2_student import (
    L2StudentBundle,
    L2StudentConfig,
    L2TrainingExample,
    train_l2_student,
)
from darjeeling.targets.nlu.layers.l2_target import (
    load_target_module,
    target_accept_prediction,
    target_config_overrides,
    target_postprocess_frame,
)
from darjeeling.targets.nlu.layers.l4_cloud_llm import (
    MissingTeacherError,
    TeacherCallResult,
    _attach_teacher_error_context,
    _extract_chat_content,
    _extract_usage,
    create_chat_completion_with_retry,
)
from darjeeling.targets.nlu.replay import load_processed_records, select_stream
from darjeeling.targets.nlu.schemas import Frame, LayerResult, TaskSchema, TeacherTrace
from darjeeling.targets.nlu.settings import Settings
from darjeeling.targets.nlu.teacher import (
    CLINC150_PROMPT_V1,
    CLINC150_PROMPT_V2_LABEL_CARDS,
    build_clinc150_intent_system_prompt,
    ensure_supported_teacher_prompt_version,
    parse_clinc150_teacher_frame,
)
from darjeeling.targets.nlu.teacher_eval import (
    TEACHER_EVAL_DETAILS_JSONL_FILENAME,
    TeacherLiveEvalArtifactResult,
    TeacherLiveEvalResult,
    _observed_l4_cost_usd,
    _teacher_eval_error_row,
    _teacher_eval_row,
    _teacher_eval_summary,
    append_teacher_live_eval_jsonl_row,
    load_teacher_live_eval_resume_rows,
    teacher_live_eval_run_identity,
    write_teacher_live_eval_artifacts,
    write_teacher_live_eval_run_manifest,
)

DEFAULT_CLINC150_PROMPTS = (CLINC150_PROMPT_V1, CLINC150_PROMPT_V2_LABEL_CARDS)
CLINC150_COST_LEDGER_FILENAME = "cost_ledger.json"
DEFAULT_CLINC150_THRESHOLDS = (
    0.0,
    0.5,
    0.7,
    0.8,
    0.9,
    0.93,
    0.95,
    0.97,
    0.98,
    0.99,
    0.995,
)
CLINC150_CALIBRATION_REPAIR_THRESHOLDS = (
    0.97,
    0.98,
    0.982,
    0.985,
    0.987,
    0.99,
    0.992,
    0.995,
)
CLINC150_CALIBRATION_REPAIR_MARGIN_GRID = (0.05, 0.10, 0.15, 0.20, 0.25)
CLINC150_CALIBRATION_REPAIR_ENTROPY_GRID = (2.0, 2.5, 3.0, 3.5)
CLINC150_CALIBRATION_REPAIR_OOS_PROBABILITY_GRID = (0.001, 0.005, 0.01, 0.02, 0.05)
CLINC150_CALIBRATION_REPAIR_OOS_MARGIN_GRID = (0.10, 0.20, 0.30, 0.40, 0.50)
CLINC150_CALIBRATION_REPAIR_OOS_RANK_GRID = (2, 3, 5, 10)
CLINC150_AUTORESEARCH_INITIAL_L2_CONFIG = {
    "accept_threshold": 0.98,
    "frame_source": "student",
    "slot_model_family": "none",
    "intent_model_family": "sgd_logreg",
    "max_features": 50_000,
    "max_iter": 1000,
    "min_examples": 4,
    "runtime_enabled": True,
}


@dataclass(frozen=True)
class Clinc150TeacherEvalArtifact:
    prompt_version: str
    artifacts: TeacherLiveEvalArtifactResult
    clinc_metrics_path: Path
    clinc_metrics: dict[str, Any]


@dataclass(frozen=True)
class Clinc150L2TrainArtifact:
    bundle_path: Path
    summary_path: Path
    summary: dict[str, Any]


@dataclass(frozen=True)
class Clinc150L2EvalArtifact:
    summary_path: Path
    details_jsonl_path: Path | None
    cost_latency_path: Path
    summary: dict[str, Any]


@dataclass(frozen=True)
class Clinc150L4ReplayOracle:
    """Target-local benchmark replay oracle for observed CLINC150 L4 rows.

    This is experiment accounting, not runtime cache behavior: fallback rows
    still count as L4 calls and carry recorded L4 cost, tokens, and latency.
    """

    rows_by_request_id: dict[str, dict[str, Any]]
    source_path: Path | None = None
    duplicate_request_ids: tuple[str, ...] = ()

    @classmethod
    def from_rows(
        cls,
        rows: list[dict[str, Any]],
        *,
        source_path: Path | None = None,
    ) -> Clinc150L4ReplayOracle:
        rows_by_request_id: dict[str, dict[str, Any]] = {}
        duplicate_ids: list[str] = []
        for row in rows:
            request_id = str(row["request_id"])
            if request_id in rows_by_request_id:
                duplicate_ids.append(request_id)
                continue
            rows_by_request_id[request_id] = row
        return cls(
            rows_by_request_id=rows_by_request_id,
            source_path=source_path,
            duplicate_request_ids=tuple(sorted(set(duplicate_ids))),
        )

    def row_for(self, request_id: str) -> dict[str, Any] | None:
        return self.rows_by_request_id.get(request_id)

    def validate_coverage(self, request_ids: list[str]) -> dict[str, Any]:
        missing = sorted(
            {
                request_id
                for request_id in request_ids
                if request_id not in self.rows_by_request_id
            }
        )
        unique_request_ids = set(request_ids)
        return {
            "requested_rows": len(request_ids),
            "unique_requested_rows": len(unique_request_ids),
            "oracle_rows": len(self.rows_by_request_id),
            "covered_unique_rows": len(unique_request_ids) - len(missing),
            "missing_rows": len(missing),
            "missing_request_ids": missing,
            "duplicate_oracle_request_ids": list(self.duplicate_request_ids),
            "source_path": str(self.source_path) if self.source_path is not None else None,
            "accounting_semantics": (
                "target-local_l4_replay_oracle_counts_fallback_rows_as_l4_calls"
            ),
        }

    def baseline_metrics(self, prediction_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
        return _all_l4_baseline_metrics(
            prediction_rows,
            teacher_by_request_id=self.rows_by_request_id,
        )


def write_clinc150_teacher_cost_ledger(
    *,
    out_dir: Path,
    artifacts: list[Clinc150TeacherEvalArtifact],
    run_kind: str,
    budget_limit_usd: float = 10.0,
) -> Path:
    entries = []
    for artifact in artifacts:
        prompt_ledger = json.loads(
            artifact.artifacts.cost_ledger_path.read_text(encoding="utf-8")
        )
        entries.append(
            {
                "prompt_version": artifact.prompt_version,
                "summary_path": str(artifact.artifacts.summary_json_path),
                "details_jsonl_path": str(artifact.artifacts.details_jsonl_path),
                "cost_ledger_path": str(artifact.artifacts.cost_ledger_path),
                "observed_attempt_cost_usd": prompt_ledger.get(
                    "observed_attempt_cost_usd",
                    0.0,
                ),
                "observed_final_response_cost_usd": prompt_ledger.get(
                    "observed_final_response_cost_usd",
                    0.0,
                ),
                "retry_overhead_cost_usd": prompt_ledger.get("retry_overhead_cost_usd", 0.0),
                "attempt_count": prompt_ledger.get("attempt_count", 0),
                "empty_response_attempts": prompt_ledger.get("empty_response_attempts", 0),
                "final_empty_response_failures": prompt_ledger.get(
                    "final_empty_response_failures",
                    0,
                ),
                "unknown_usage_attempts": prompt_ledger.get("unknown_usage_attempts", 0),
            }
        )
    observed_spend = sum(float(entry["observed_attempt_cost_usd"]) for entry in entries)
    ledger = {
        "schema_version": "clinc150-teacher-reliability-cost-ledger-v1",
        "run_kind": run_kind,
        "run_root": str(out_dir),
        "budget_limit_usd": budget_limit_usd,
        "observed_spend_usd": observed_spend,
        "estimated_remaining_budget_usd": budget_limit_usd - observed_spend,
        "entries": entries,
    }
    path = out_dir / CLINC150_COST_LEDGER_FILENAME
    path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


class Clinc150IntentTeacher:
    def __init__(
        self,
        settings: Settings,
        *,
        label_cards: list[dict[str, object]] | None = None,
        client: Any | None = None,
    ) -> None:
        self.settings = settings
        self.label_cards = label_cards
        self._client = client

    def client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.settings.openai_api_key:
            raise MissingTeacherError("OPENAI_API_KEY is required for live CLINC150 teacher calls")
        from openai import OpenAI

        return OpenAI(
            api_key=self.settings.openai_api_key,
            base_url=self.settings.openai_base_url or None,
            timeout=self.settings.openai_timeout_s,
            max_retries=0,
        )

    def answer(self, utterance: str, task_schema: TaskSchema) -> TeacherCallResult:
        prompt_version = self.settings.teacher_prompt_version
        ensure_supported_teacher_prompt_version(prompt_version)
        stable_prefix = build_clinc150_intent_system_prompt(
            task_schema,
            prompt_version=prompt_version,
            label_cards=self.label_cards,
        )
        messages = [
            {"role": "system", "content": stable_prefix},
            {
                "role": "user",
                "content": json.dumps(
                    {"utterance": utterance},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            },
        ]
        attempt_diagnostics: list[dict[str, Any]] = []
        response = create_chat_completion_with_retry(
            self.client(),
            self.settings,
            response_check=_extract_chat_content,
            attempt_diagnostics=attempt_diagnostics,
            attempt_metadata={"call_kind": "clinc150_intent"},
            model=self.settings.openai_model,
            messages=messages,
            response_format={"type": "json_object"},
            max_completion_tokens=self.settings.teacher_max_tokens,
            prompt_cache_key=f"darjeeling:{prompt_version}:{task_schema.schema_version}",
            prompt_cache_retention=self.settings.prompt_cache_retention,
            timeout=self.settings.openai_timeout_s,
        )
        raw_response = _extract_chat_content(response)
        usage = _extract_usage(response)
        model = getattr(response, "model", self.settings.openai_model)
        try:
            frame = parse_clinc150_teacher_frame(raw_response, task_schema=task_schema)
        except Exception as exc:
            _attach_teacher_error_context(
                exc,
                attempt_diagnostics=attempt_diagnostics,
                usage=usage,
                model=model,
                raw_response=raw_response,
            )
            raise
        return TeacherCallResult(
            frame=frame,
            raw_response=raw_response,
            usage=usage,
            model=model,
            context_hash="",
            prompt_cache_key=f"darjeeling:{prompt_version}:{task_schema.schema_version}",
            attempt_diagnostics=attempt_diagnostics,
        )


def build_clinc150_gate_records(
    validation_records: list[DataRecord],
    *,
    per_intent: int = 3,
    oos_requests: int = 50,
) -> list[DataRecord]:
    by_intent: dict[str, list[DataRecord]] = defaultdict(list)
    for record in validation_records:
        by_intent[record.gold_frame.intent].append(record)

    selected: list[DataRecord] = []
    for intent in sorted(intent for intent in by_intent if intent != CLINC150_OOS_INTENT):
        examples = by_intent[intent]
        if len(examples) < per_intent:
            raise ValueError(
                f"CLINC150 validation has only {len(examples)} examples for {intent!r}; "
                f"{per_intent} required"
            )
        selected.extend(examples[:per_intent])
    oos_examples = by_intent.get(CLINC150_OOS_INTENT, [])
    if len(oos_examples) < oos_requests:
        raise ValueError(
            f"CLINC150 validation has only {len(oos_examples)} OOS examples; "
            f"{oos_requests} required"
        )
    selected.extend(oos_examples[:oos_requests])
    return selected


def build_clinc150_label_cards(
    train_records: list[DataRecord],
    *,
    examples_per_label: int = 2,
) -> list[dict[str, object]]:
    examples_by_intent: dict[str, list[str]] = defaultdict(list)
    for record in train_records:
        examples = examples_by_intent[record.gold_frame.intent]
        if len(examples) < examples_per_label:
            examples.append(record.utterance)
    return [
        {
            "intent": intent,
            "description": _label_description(intent),
            "examples": examples_by_intent[intent],
        }
        for intent in sorted(examples_by_intent)
    ]


def _label_cards_hash(label_cards: list[dict[str, object]] | None) -> str:
    payload = json.dumps(label_cards or [], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _label_description(intent: str) -> str:
    if intent == CLINC150_OOS_INTENT:
        return "unsupported or out-of-scope request"
    return intent.replace("_", " ")


def build_clinc150_stratified_records(
    records: list[DataRecord],
    *,
    max_requests: int,
) -> list[DataRecord]:
    if max_requests < 1:
        raise ValueError("max_requests must be at least 1")
    by_intent: dict[str, list[DataRecord]] = defaultdict(list)
    for record in records:
        by_intent[record.gold_frame.intent].append(record)

    selected: list[DataRecord] = []
    depth = 0
    intents = sorted(by_intent)
    while len(selected) < max_requests:
        added = False
        for intent in intents:
            examples = by_intent[intent]
            if depth >= len(examples):
                continue
            selected.append(examples[depth])
            added = True
            if len(selected) >= max_requests:
                break
        if not added:
            break
        depth += 1
    return selected


def sample_clinc150_records(
    records: list[DataRecord],
    *,
    stream: str,
    max_requests: int | None,
) -> list[DataRecord]:
    if max_requests is None:
        if stream != "sequential":
            max_requests = len(records)
        else:
            return records
    if stream == "stratified":
        return build_clinc150_stratified_records(records, max_requests=max_requests)
    return stream_clinc150_records(records, stream=stream, max_requests=max_requests)


def run_clinc150_teacher_live_eval(
    *,
    records: list[DataRecord],
    task_schema: TaskSchema,
    settings: Settings,
    split: str,
    stream: str,
    prompt_version: str,
    out_dir: Path,
    label_cards: list[dict[str, object]] | None = None,
    max_workers: int = 1,
    resume_existing: bool = False,
    min_overall_accuracy: float = 0.95,
    min_in_scope_accuracy: float = 0.97,
    max_parse_failure_rate: float = 0.005,
) -> Clinc150TeacherEvalArtifact:
    settings_for_prompt = settings.model_copy(update={"teacher_prompt_version": prompt_version})
    run_identity = teacher_live_eval_run_identity(
        records=records,
        task_schema=task_schema,
        settings=settings_for_prompt,
        split=split,
        stream=stream,
        prompt_version=prompt_version,
        extra={
            "target": "clinc150",
            "label_cards_hash": _label_cards_hash(label_cards),
            "artifact_schema_version": "clinc150-teacher-live-eval-v1",
        },
    )
    existing_rows = (
        load_teacher_live_eval_resume_rows(
            out_dir=out_dir,
            expected_run_identity=run_identity,
        )
        if resume_existing
        else []
    )
    write_teacher_live_eval_run_manifest(out_dir=out_dir, run_identity=run_identity)
    if not resume_existing:
        (out_dir / TEACHER_EVAL_DETAILS_JSONL_FILENAME).write_text("", encoding="utf-8")
    rows = _evaluate_clinc150_teacher_rows(
        records=records,
        task_schema=task_schema,
        settings=settings_for_prompt,
        prompt_version=prompt_version,
        label_cards=label_cards,
        max_workers=max_workers,
        out_dir=out_dir,
        existing_rows=existing_rows,
    )
    rows = sorted(rows, key=lambda row: int(row["index"]))
    summary = _teacher_eval_summary(
        rows,
        settings=settings_for_prompt,
        split=split,
        stream=stream,
        prompt_version=prompt_version,
        min_frame_exact_match=0.0,
    )
    clinc_metrics = clinc150_metrics_from_teacher_rows(
        rows,
        min_overall_accuracy=min_overall_accuracy,
        min_in_scope_accuracy=min_in_scope_accuracy,
        max_parse_failure_rate=max_parse_failure_rate,
    )
    summary["clinc150"] = clinc_metrics
    artifacts = write_teacher_live_eval_artifacts(
        TeacherLiveEvalResult(summary=summary, rows=rows),
        out_dir=out_dir,
    )
    metrics_path = out_dir / "clinc150_teacher_metrics.json"
    metrics_path.write_text(
        json.dumps(clinc_metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return Clinc150TeacherEvalArtifact(
        prompt_version=prompt_version,
        artifacts=artifacts,
        clinc_metrics_path=metrics_path,
        clinc_metrics=clinc_metrics,
    )


def _evaluate_clinc150_teacher_rows(
    *,
    records: list[DataRecord],
    task_schema: TaskSchema,
    settings: Settings,
    prompt_version: str,
    label_cards: list[dict[str, object]] | None,
    max_workers: int,
    out_dir: Path,
    existing_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    details_path = out_dir / TEACHER_EVAL_DETAILS_JSONL_FILENAME
    rows_by_request_id = {
        str(row["request_id"]): row
        for row in existing_rows or []
    }
    indexed = list(enumerate(records, start=1))
    missing_indexed = [
        (index, record)
        for index, record in indexed
        if record.request_id not in rows_by_request_id
    ]
    if not missing_indexed:
        return [rows_by_request_id[record.request_id] for _index, record in indexed]

    if max_workers <= 1:
        for index, record in missing_indexed:
            row = _clinc150_teacher_row(
                index=index,
                record=record,
                task_schema=task_schema,
                settings=settings,
                prompt_version=prompt_version,
                label_cards=label_cards,
            )
            rows_by_request_id[record.request_id] = row
            append_teacher_live_eval_jsonl_row(details_path, row)
        return [rows_by_request_id[record.request_id] for _index, record in indexed]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _clinc150_teacher_row,
                index=index,
                record=record,
                task_schema=task_schema,
                settings=settings,
                prompt_version=prompt_version,
                label_cards=label_cards,
            ): index
            for index, record in missing_indexed
        }
        for future in as_completed(futures):
            row = future.result()
            rows_by_request_id[str(row["request_id"])] = row
            append_teacher_live_eval_jsonl_row(details_path, row)
    return [rows_by_request_id[record.request_id] for _index, record in indexed]


def _clinc150_teacher_row(
    *,
    index: int,
    record: DataRecord,
    task_schema: TaskSchema,
    settings: Settings,
    prompt_version: str,
    label_cards: list[dict[str, object]] | None,
) -> dict[str, Any]:
    teacher = Clinc150IntentTeacher(settings, label_cards=label_cards)
    cost_model = replay_cost_model_from_settings(settings)
    started = perf_counter()
    try:
        call_result = teacher.answer(record.utterance, task_schema)
        latency_ms = (perf_counter() - started) * 1000.0
        return _teacher_eval_row(
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
    except Exception as exc:
        latency_ms = (perf_counter() - started) * 1000.0
        return _teacher_eval_error_row(
            index=index,
            record=record,
            task_schema=task_schema,
            error=exc,
            latency_ms=latency_ms,
            cost_model=cost_model,
            prompt_version=prompt_version,
            default_model=settings.openai_model,
        )


def clinc150_metrics_from_teacher_rows(
    rows: list[dict[str, Any]],
    *,
    min_overall_accuracy: float = 0.95,
    min_in_scope_accuracy: float = 0.97,
    max_parse_failure_rate: float = 0.005,
) -> dict[str, Any]:
    requests = len(rows)
    parsed = [row for row in rows if not row.get("parse_failure")]
    in_scope = [row for row in rows if _gold_intent(row) != CLINC150_OOS_INTENT]
    oos = [row for row in rows if _gold_intent(row) == CLINC150_OOS_INTENT]
    overall_accuracy = _rate(sum(1 for row in rows if row.get("frame_exact")), requests)
    in_scope_accuracy = _rate(
        sum(1 for row in in_scope if row.get("intent_correct")),
        len(in_scope),
    )
    parse_failure_rate = _rate(
        sum(1 for row in rows if row.get("parse_failure")),
        requests,
    )
    oos_counts = _oos_counts(rows)
    oos_precision = _rate(oos_counts["true_positive"], oos_counts["predicted_positive"])
    oos_recall = _rate(oos_counts["true_positive"], oos_counts["gold_positive"])
    attempt_count = sum(int(row.get("attempt_count", 1)) for row in rows)
    empty_response_attempts = sum(int(row.get("empty_response_attempts", 0)) for row in rows)
    retry_recovered_rows = sum(1 for row in rows if row.get("retry_recovered"))
    final_empty_response_failures = sum(
        1 for row in rows if row.get("final_empty_response_failure")
    )
    unknown_usage_attempts = sum(int(row.get("unknown_usage_attempts", 0)) for row in rows)
    return {
        "requests": requests,
        "parsed_requests": len(parsed),
        "in_scope_requests": len(in_scope),
        "oos_requests": len(oos),
        "overall_accuracy": overall_accuracy,
        "in_scope_accuracy": in_scope_accuracy,
        "parse_schema_failure_rate": parse_failure_rate,
        "attempt_count": attempt_count,
        "retry_recovered_rows": retry_recovered_rows,
        "empty_response_attempts": empty_response_attempts,
        "final_empty_response_failures": final_empty_response_failures,
        "unknown_usage_attempts": unknown_usage_attempts,
        "observed_attempt_cost_usd": sum(float(row.get("cost_usd", 0.0)) for row in rows),
        "observed_final_response_cost_usd": sum(
            float(row.get("final_response_cost_usd", row.get("cost_usd", 0.0)))
            for row in rows
        ),
        "oos_precision": oos_precision,
        "oos_recall": oos_recall,
        "oos_f1": _f1(oos_precision, oos_recall),
        "oos_counts": oos_counts,
        "gate_targets": {
            "min_overall_accuracy": min_overall_accuracy,
            "min_in_scope_accuracy": min_in_scope_accuracy,
            "max_parse_failure_rate": max_parse_failure_rate,
        },
        "passed_teacher_gate": (
            overall_accuracy is not None
            and in_scope_accuracy is not None
            and parse_failure_rate is not None
            and overall_accuracy >= min_overall_accuracy
            and in_scope_accuracy >= min_in_scope_accuracy
            and parse_failure_rate <= max_parse_failure_rate
        ),
    }


def compare_repeated_teacher_rows(
    first_rows: list[dict[str, Any]],
    second_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    second_by_id = {row["request_id"]: row for row in second_rows}
    comparable = []
    agreements = 0
    for first in first_rows:
        second = second_by_id.get(first["request_id"])
        if second is None:
            continue
        if first.get("parse_failure") or second.get("parse_failure"):
            continue
        comparable.append(first["request_id"])
        agreements += int(first.get("teacher_frame") == second.get("teacher_frame"))
    return {
        "requests": len(first_rows),
        "comparable_requests": len(comparable),
        "consistent_requests": agreements,
        "consistency": _rate(agreements, len(comparable)),
        "request_ids": comparable,
    }


def training_examples_from_gold_records(records: list[DataRecord]) -> list[L2TrainingExample]:
    return [
        L2TrainingExample(utterance=record.utterance, teacher_frame=record.gold_frame)
        for record in records
    ]


def training_examples_from_teacher_rows(rows: list[dict[str, Any]]) -> list[L2TrainingExample]:
    examples: list[L2TrainingExample] = []
    for row in rows:
        if row.get("parse_failure") or row.get("teacher_frame") is None:
            continue
        examples.append(
            L2TrainingExample(
                utterance=str(row["utterance"]),
                teacher_frame=Frame.model_validate(row["teacher_frame"]),
            )
        )
    return examples


def train_clinc150_l2(
    examples: list[L2TrainingExample],
    *,
    random_state: int = 17,
    accept_threshold: float = 0.99,
    config: L2StudentConfig | None = None,
) -> L2StudentBundle:
    l2_config = config or L2StudentConfig(
        accept_threshold=accept_threshold,
        random_state=random_state,
        min_examples=4,
        slot_model_family="none",
        frame_source="student",
        max_iter=1000,
    )
    return train_l2_student(
        examples,
        l2_config,
    )


def write_clinc150_l2_train_artifacts(
    *,
    bundle: L2StudentBundle,
    examples: list[L2TrainingExample],
    out_dir: Path,
    training_source: str,
    source_path: Path | None = None,
    split: str | None = None,
    sample_stream: str | None = None,
    max_examples: int | None = None,
) -> Clinc150L2TrainArtifact:
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = out_dir / "l2_student.joblib"
    bundle.save(bundle_path)
    intents: dict[str, int] = defaultdict(int)
    for example in examples:
        intents[example.teacher_frame.intent] += 1
    summary = {
        "schema_version": "clinc150-l2-train-v1",
        "training_source": training_source,
        "source_path": str(source_path) if source_path is not None else None,
        "split": split,
        "sample_stream": sample_stream,
        "max_examples": max_examples,
        "examples": len(examples),
        "intent_count": len(intents),
        "oos_examples": intents.get(CLINC150_OOS_INTENT, 0),
        "intents": dict(sorted(intents.items())),
        "bundle_path": str(bundle_path),
        "config": bundle.config.model_dump(mode="json"),
    }
    summary_path = out_dir / "clinc150_l2_train_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return Clinc150L2TrainArtifact(
        bundle_path=bundle_path,
        summary_path=summary_path,
        summary=summary,
    )


def evaluate_clinc150_l2(
    *,
    bundle: L2StudentBundle,
    records: list[DataRecord],
    teacher_rows: list[dict[str, Any]] | None = None,
    thresholds: tuple[float, ...] = DEFAULT_CLINC150_THRESHOLDS,
    include_prediction_rows: bool = False,
) -> dict[str, Any]:
    replay_oracle = (
        Clinc150L4ReplayOracle.from_rows(teacher_rows)
        if teacher_rows is not None
        else None
    )
    teacher_by_request_id = (
        replay_oracle.rows_by_request_id if replay_oracle is not None else {}
    )
    prediction_rows = clinc150_l2_prediction_rows(bundle=bundle, records=records)
    threshold_rows = [
        _l2_threshold_metrics(
            prediction_rows,
            teacher_by_request_id=teacher_by_request_id,
            threshold=threshold,
        )
        for threshold in thresholds
    ]
    result = {
        "schema_version": "clinc150-l2-eval-v1",
        "requests": len(prediction_rows),
        "measurement_path": "l2_only_shadow_l2_plus_l4_fallback",
        "l0_enabled": False,
        "l4_replay_oracle": (
            replay_oracle.validate_coverage([row["request_id"] for row in prediction_rows])
            if replay_oracle is not None
            else None
        ),
        "accuracy": _rate(
            sum(1 for row in prediction_rows if row["predicted_frame"] == row["gold_frame"]),
            len(prediction_rows),
        ),
        "in_scope_accuracy": _rate(
            sum(
                1
                for row in prediction_rows
                if row["gold_intent"] != CLINC150_OOS_INTENT
                and row["predicted_intent"] == row["gold_intent"]
            ),
            sum(1 for row in prediction_rows if row["gold_intent"] != CLINC150_OOS_INTENT),
        ),
        "oos_accuracy": _rate(
            sum(
                1
                for row in prediction_rows
                if row["gold_intent"] == CLINC150_OOS_INTENT
                and row["predicted_intent"] == CLINC150_OOS_INTENT
            ),
            sum(1 for row in prediction_rows if row["gold_intent"] == CLINC150_OOS_INTENT),
        ),
        "teacher_fallback_rows": sum(
            1 for row in prediction_rows if row["request_id"] in teacher_by_request_id
        ),
        "all_l4_baseline": _all_l4_baseline_metrics(
            prediction_rows,
            teacher_by_request_id=teacher_by_request_id,
        ),
        "thresholds": threshold_rows,
        "selected_threshold": select_l2_threshold(threshold_rows),
        "cost_latency_table": _cost_latency_table(threshold_rows),
    }
    if include_prediction_rows:
        result["prediction_rows"] = prediction_rows
    return result


def select_l2_threshold(
    threshold_rows: list[dict[str, Any]],
    *,
    min_precision: float = 0.99,
    max_oos_false_accept_rate: float = 0.02,
    min_accuracy_delta_vs_all_l4: float = -0.005,
) -> dict[str, Any] | None:
    candidates = [
        row
        for row in threshold_rows
        if (row["accepted_precision"] is not None and row["accepted_precision"] >= min_precision)
        and row["lower_layer_oos_false_accept_rate"] <= max_oos_false_accept_rate
        and (
            row.get("accuracy_delta_vs_all_l4") is None
            or row["accuracy_delta_vs_all_l4"] >= min_accuracy_delta_vs_all_l4
        )
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: (row["accepted_coverage"], -row["threshold"]))


def clinc150_calibration_guard_rules(
    *,
    thresholds: tuple[float, ...] = CLINC150_CALIBRATION_REPAIR_THRESHOLDS,
    margin_grid: tuple[float, ...] = CLINC150_CALIBRATION_REPAIR_MARGIN_GRID,
    entropy_grid: tuple[float, ...] = CLINC150_CALIBRATION_REPAIR_ENTROPY_GRID,
    oos_probability_grid: tuple[float, ...] = CLINC150_CALIBRATION_REPAIR_OOS_PROBABILITY_GRID,
    oos_margin_grid: tuple[float, ...] = CLINC150_CALIBRATION_REPAIR_OOS_MARGIN_GRID,
    oos_rank_grid: tuple[int, ...] = CLINC150_CALIBRATION_REPAIR_OOS_RANK_GRID,
    predicted_intent_vetoes: tuple[tuple[str, ...], ...] = (),
) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    for threshold in thresholds:
        rules.append(
            _guard_rule(
                family="threshold",
                threshold=threshold,
            )
        )
    for threshold in thresholds:
        for min_margin in margin_grid:
            rules.append(
                _guard_rule(
                    family="threshold_margin",
                    threshold=threshold,
                    min_margin=min_margin,
                )
            )
        for max_entropy in entropy_grid:
            rules.append(
                _guard_rule(
                    family="threshold_entropy",
                    threshold=threshold,
                    max_entropy=max_entropy,
                )
            )
        for max_oos_probability in oos_probability_grid:
            rules.append(
                _guard_rule(
                    family="threshold_oos_probability",
                    threshold=threshold,
                    max_oos_probability=max_oos_probability,
                )
            )
        for min_oos_margin in oos_margin_grid:
            rules.append(
                _guard_rule(
                    family="threshold_oos_margin",
                    threshold=threshold,
                    min_oos_margin=min_oos_margin,
                )
            )
        for min_oos_rank in oos_rank_grid:
            rules.append(
                _guard_rule(
                    family="threshold_oos_rank",
                    threshold=threshold,
                    min_oos_rank=min_oos_rank,
                )
            )
    for vetoes in predicted_intent_vetoes:
        if not vetoes:
            continue
        for threshold in thresholds:
            rules.append(
                _guard_rule(
                    family="threshold_predicted_intent_veto",
                    threshold=threshold,
                    veto_predicted_intents=tuple(sorted(vetoes)),
                )
            )
    return rules


def evaluate_clinc150_guard_rules(
    *,
    prediction_rows: list[dict[str, Any]],
    guard_rules: list[dict[str, Any]],
    replay_oracle: Clinc150L4ReplayOracle | None = None,
) -> list[dict[str, Any]]:
    return [
        evaluate_clinc150_guard_rule(
            prediction_rows=prediction_rows,
            guard_rule=guard_rule,
            replay_oracle=replay_oracle,
        )
        for guard_rule in guard_rules
    ]


def evaluate_clinc150_guard_rule(
    *,
    prediction_rows: list[dict[str, Any]],
    guard_rule: dict[str, Any],
    replay_oracle: Clinc150L4ReplayOracle | None = None,
    teacher_by_request_id: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    teacher_by_id = (
        replay_oracle.rows_by_request_id
        if replay_oracle is not None
        else teacher_by_request_id or {}
    )
    accepted_flags = [
        clinc150_guard_accepts(row, guard_rule=guard_rule)
        for row in prediction_rows
    ]
    return _evaluate_clinc150_acceptance_flags(
        prediction_rows=prediction_rows,
        accepted_flags=accepted_flags,
        teacher_by_request_id=teacher_by_id,
        guard_name=guard_rule["name"],
        guard_rule=guard_rule,
        threshold=guard_rule["threshold"],
    )


def _evaluate_clinc150_acceptance_flags(
    *,
    prediction_rows: list[dict[str, Any]],
    accepted_flags: list[bool],
    teacher_by_request_id: dict[str, dict[str, Any]],
    guard_name: str,
    guard_rule: dict[str, Any],
    threshold: float | None,
) -> dict[str, Any]:
    accepted_rows = [
        row for row, accepted in zip(prediction_rows, accepted_flags, strict=True) if accepted
    ]
    correct_accepted = [
        row for row in accepted_rows if row["predicted_frame"] == row["gold_frame"]
    ]
    oos_total = sum(1 for row in prediction_rows if row["gold_oos"])
    lower_oos_false_accepts = sum(
        1
        for row in accepted_rows
        if row["gold_oos"] and not row["predicted_oos"]
    )
    final_rows = [
        _cascade_row(
            row,
            teacher_by_request_id=teacher_by_request_id,
            accepted=accepted,
        )
        for row, accepted in zip(prediction_rows, accepted_flags, strict=True)
    ]
    final_correct = sum(1 for row in final_rows if row["final_correct"])
    all_l4_correct = sum(1 for row in final_rows if row["all_l4_correct"])
    latencies = [row["latency_ms"] for row in final_rows]
    l4_calls = sum(1 for row in final_rows if row["l4_called"])
    l4_tokens = sum(float(row["l4_tokens"]) for row in final_rows)
    l4_cost = sum(float(row["l4_cost_usd"]) for row in final_rows)
    all_l4_rows = [
        teacher_by_request_id.get(row["request_id"])
        for row in prediction_rows
    ]
    paired_teacher_rows = [row for row in all_l4_rows if row is not None]
    all_l4_tokens = sum(float(row.get("tokens", 0.0)) for row in paired_teacher_rows)
    all_l4_cost = sum(float(row.get("cost_usd", 0.0)) for row in paired_teacher_rows)
    all_l4_latencies = [
        float(row.get("latency_ms", 0.0))
        for row in paired_teacher_rows
    ]
    teacher_parse_failures = sum(
        1 for row in paired_teacher_rows if row.get("parse_failure")
    )
    oos_counts = _oos_counts_from_frames(
        gold_intents=[row["gold_intent"] for row in prediction_rows],
        predicted_intents=[row["final_intent"] for row in final_rows],
    )
    oos_precision = _rate(oos_counts["true_positive"], oos_counts["predicted_positive"])
    oos_recall = _rate(oos_counts["true_positive"], oos_counts["gold_positive"])
    requests = len(prediction_rows)
    accepted = len(accepted_rows)
    accepted_correct = len(correct_accepted)
    final_accuracy = _rate(final_correct, requests)
    all_l4_accuracy = _rate(all_l4_correct, requests) if teacher_by_request_id else None
    return {
        "guard_name": guard_name,
        "guard_rule": guard_rule,
        "threshold": threshold,
        "accepted": accepted,
        "accepted_correct": accepted_correct,
        "accepted_wrong": accepted - accepted_correct,
        "accepted_coverage": _rate(accepted, requests),
        "accepted_precision": _rate(accepted_correct, accepted),
        "accepted_precision_wilson_lower_95": _wilson_lower_bound(
            accepted_correct,
            accepted,
        ),
        "lower_layer_oos_false_accepts": lower_oos_false_accepts,
        "lower_layer_oos_false_accept_rate": _rate(lower_oos_false_accepts, oos_total) or 0.0,
        "all_l4_accuracy": all_l4_accuracy,
        "final_cascade_accuracy": final_accuracy,
        "accuracy_delta_vs_all_l4": (
            final_accuracy - all_l4_accuracy
            if final_accuracy is not None and all_l4_accuracy is not None
            else None
        ),
        "paired_teacher_rows": len(paired_teacher_rows),
        "paired_teacher_coverage": _rate(len(paired_teacher_rows), requests),
        "parse_schema_failures": teacher_parse_failures,
        "parse_schema_failure_rate": (
            _rate(teacher_parse_failures, len(paired_teacher_rows))
            if paired_teacher_rows
            else None
        ),
        "all_l4_calls_per_100_requests": 100.0 if teacher_by_request_id else None,
        "l4_calls_per_100_requests": (l4_calls / requests * 100.0) if requests else 0.0,
        "l4_call_reduction_rate": (
            1.0 - (l4_calls / requests)
            if teacher_by_request_id and requests
            else None
        ),
        "all_l4_tokens_per_request": (
            all_l4_tokens / requests
            if teacher_by_request_id and requests
            else None
        ),
        "l4_tokens_per_request": l4_tokens / requests if requests else 0.0,
        "l4_tokens_reduction_rate": (
            1.0 - (l4_tokens / all_l4_tokens)
            if all_l4_tokens > 0.0
            else None
        ),
        "all_l4_cost_usd_per_request": (
            all_l4_cost / requests
            if teacher_by_request_id and requests
            else None
        ),
        "l4_cost_usd_per_request": l4_cost / requests if requests else 0.0,
        "l4_cost_reduction_rate": (
            1.0 - (l4_cost / all_l4_cost)
            if all_l4_cost > 0.0
            else None
        ),
        "all_l4_latency_p50_ms": (
            _percentile(all_l4_latencies, 50)
            if teacher_by_request_id
            else None
        ),
        "all_l4_latency_p95_ms": (
            _percentile(all_l4_latencies, 95)
            if teacher_by_request_id
            else None
        ),
        "latency_p50_ms": _percentile(latencies, 50),
        "latency_p95_ms": _percentile(latencies, 95),
        "latency_p50_reduction_rate": _reduction_rate(
            _percentile(latencies, 50),
            _percentile(all_l4_latencies, 50) if all_l4_latencies else None,
        ),
        "latency_p95_reduction_rate": _reduction_rate(
            _percentile(latencies, 95),
            _percentile(all_l4_latencies, 95) if all_l4_latencies else None,
        ),
        "oos_precision": oos_precision,
        "oos_recall": oos_recall,
        "oos_f1": _f1(oos_precision, oos_recall),
    }


def clinc150_guard_accepts(row: dict[str, Any], *, guard_rule: dict[str, Any]) -> bool:
    if float(row["guard_probability"]) < float(guard_rule["threshold"]):
        return False
    min_margin = guard_rule.get("min_margin")
    if min_margin is not None and float(row.get("margin", 0.0)) < float(min_margin):
        return False
    max_entropy = guard_rule.get("max_entropy")
    if max_entropy is not None and float(row.get("entropy", 0.0)) > float(max_entropy):
        return False
    max_oos_probability = guard_rule.get("max_oos_probability")
    if max_oos_probability is not None and float(row.get("oos_probability", 0.0)) > float(
        max_oos_probability
    ):
        return False
    min_oos_margin = guard_rule.get("min_oos_margin")
    if min_oos_margin is not None and float(row.get("oos_margin", 1.0)) < float(
        min_oos_margin
    ):
        return False
    min_oos_rank = guard_rule.get("min_oos_rank")
    if min_oos_rank is not None and int(row.get("oos_rank", 1_000_000)) < int(min_oos_rank):
        return False
    vetoes = set(guard_rule.get("veto_predicted_intents") or [])
    return str(row.get("predicted_intent")) not in vetoes


def select_clinc150_calibration_guard(
    *,
    calibration_dev_results: list[dict[str, Any]],
    oos_heavy_results: list[dict[str, Any]],
    validation_results: list[dict[str, Any]],
    min_precision: float = 0.99,
    max_oos_false_accept_rate: float = 0.02,
    min_accuracy_delta_vs_all_l4: float = -0.005,
) -> dict[str, Any] | None:
    dev_by_name = {row["guard_name"]: row for row in calibration_dev_results}
    oos_by_name = {row["guard_name"]: row for row in oos_heavy_results}
    validation_by_name = {row["guard_name"]: row for row in validation_results}
    candidates = []
    for guard_name, validation in validation_by_name.items():
        dev = dev_by_name.get(guard_name)
        oos = oos_by_name.get(guard_name)
        if dev is None or oos is None:
            continue
        if not _guard_result_passes_constraints(
            dev,
            min_precision=min_precision,
            max_oos_false_accept_rate=max_oos_false_accept_rate,
            min_accuracy_delta_vs_all_l4=None,
        ):
            continue
        if oos["lower_layer_oos_false_accept_rate"] > max_oos_false_accept_rate:
            continue
        if not _guard_result_passes_constraints(
            validation,
            min_precision=min_precision,
            max_oos_false_accept_rate=max_oos_false_accept_rate,
            min_accuracy_delta_vs_all_l4=min_accuracy_delta_vs_all_l4,
        ):
            continue
        candidates.append(
            {
                "guard_name": guard_name,
                "guard_rule": validation["guard_rule"],
                "calibration_dev": dev,
                "oos_heavy": oos,
                "validation": validation,
            }
        )
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda candidate: (
            candidate["validation"]["accepted_coverage"] or 0.0,
            candidate["validation"]["accepted_precision"] or 0.0,
            -_guard_rule_complexity(candidate["guard_rule"]),
        ),
    )


def write_clinc150_l2_eval_artifacts(
    *,
    result: dict[str, Any],
    out_dir: Path,
) -> Clinc150L2EvalArtifact:
    out_dir.mkdir(parents=True, exist_ok=True)
    prediction_rows = result.get("prediction_rows")
    summary = {
        key: value
        for key, value in result.items()
        if key != "prediction_rows"
    }
    summary_path = out_dir / "clinc150_l2_eval_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    details_path = None
    if isinstance(prediction_rows, list):
        details_path = out_dir / "clinc150_l2_predictions.jsonl"
        details_path.write_text(
            "".join(
                json.dumps(row, sort_keys=True) + "\n"
                for row in prediction_rows
            ),
            encoding="utf-8",
        )
    cost_latency_path = out_dir / "clinc150_l2_cost_latency_table.json"
    cost_latency_path.write_text(
        json.dumps(summary.get("cost_latency_table", []), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return Clinc150L2EvalArtifact(
        summary_path=summary_path,
        details_jsonl_path=details_path,
        cost_latency_path=cost_latency_path,
        summary=summary,
    )


def clinc150_l2_prediction_rows(
    *,
    bundle: L2StudentBundle,
    records: list[DataRecord],
) -> list[dict[str, Any]]:
    return [
        _l2_prediction_row(bundle=bundle, record=record)
        for record in records
    ]


def load_clinc150_l2_prediction_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_clinc150_l2_prediction_rows(
    rows: list[dict[str, Any]],
    path: Path,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def build_clinc150_calibration_splits(
    teacher_rows: list[dict[str, Any]],
    *,
    calibration_fraction: float = 0.5,
    seed: str = "clinc150-calibration-repair-20260624",
) -> dict[str, Any]:
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be between 0 and 1")
    parsed_rows = [row for row in teacher_rows if not row.get("parse_failure")]
    by_intent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in parsed_rows:
        intent = _gold_intent(row)
        if intent is None:
            continue
        by_intent[intent].append(row)

    calibration_ids: list[str] = []
    dev_ids: list[str] = []
    intent_counts = {}
    for intent in sorted(by_intent):
        rows = sorted(
            by_intent[intent],
            key=lambda row: _stable_split_key(seed, str(row["request_id"])),
        )
        split_at = int(round(len(rows) * calibration_fraction))
        if len(rows) > 1:
            split_at = min(max(1, split_at), len(rows) - 1)
        else:
            split_at = len(rows)
        calibration_part = rows[:split_at]
        dev_part = rows[split_at:]
        calibration_ids.extend(str(row["request_id"]) for row in calibration_part)
        dev_ids.extend(str(row["request_id"]) for row in dev_part)
        intent_counts[intent] = {
            "total": len(rows),
            "calibration": len(calibration_part),
            "dev": len(dev_part),
        }

    calibration_ids = sorted(calibration_ids)
    dev_ids = sorted(dev_ids)
    return {
        "schema_version": "clinc150-calibration-splits-v1",
        "source": "parsed_teacher_train_rows_with_gold_labels",
        "seed": seed,
        "calibration_fraction": calibration_fraction,
        "parsed_rows": len(parsed_rows),
        "intent_count": len(intent_counts),
        "intents": intent_counts,
        "general_calibration": _split_payload(calibration_ids),
        "general_dev": _split_payload(dev_ids),
    }


def build_clinc150_oos_heavy_slice(
    *,
    teacher_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    teacher_by_id = {str(row["request_id"]): row for row in teacher_rows}
    selected: dict[str, dict[str, Any]] = {}
    reasons: dict[str, set[str]] = defaultdict(set)
    for row in teacher_rows:
        request_id = str(row["request_id"])
        if _gold_intent(row) == CLINC150_OOS_INTENT:
            selected[request_id] = row
            reasons[request_id].add("gold_oos")
        if _teacher_intent(row) == CLINC150_OOS_INTENT:
            selected[request_id] = row
            reasons[request_id].add("teacher_predicted_oos")
    for row in prediction_rows:
        request_id = str(row["request_id"])
        teacher_row = teacher_by_id.get(request_id)
        teacher_oos = (
            _teacher_intent(teacher_row) == CLINC150_OOS_INTENT
            if teacher_row is not None
            else False
        )
        if not row["predicted_oos"] and (row["gold_oos"] or teacher_oos):
            if teacher_row is not None:
                selected[request_id] = teacher_row
            reasons[request_id].add("l2_in_scope_with_oos_signal")
    request_ids = sorted(selected)
    reason_counts = Counter(
        reason
        for request_id in request_ids
        for reason in reasons[request_id]
    )
    return {
        "schema_version": "clinc150-oos-heavy-slice-v1",
        "source": "teacher_train_rows_plus_l2_train_predictions",
        "request_ids": request_ids,
        "requests": len(request_ids),
        "reason_counts": dict(sorted(reason_counts.items())),
        "duplicate_handling": "request_id_deduplicated_with_reason_counts_preserved",
    }


def clinc150_prediction_rows_for_request_ids(
    prediction_rows: list[dict[str, Any]],
    request_ids: list[str],
) -> list[dict[str, Any]]:
    by_id = {str(row["request_id"]): row for row in prediction_rows}
    return [by_id[request_id] for request_id in request_ids if request_id in by_id]


def summarize_clinc150_accepted_errors(
    *,
    prediction_rows: list[dict[str, Any]],
    guard_rule: dict[str, Any],
    replay_oracle: Clinc150L4ReplayOracle | None = None,
    max_examples_per_family: int = 5,
) -> dict[str, Any]:
    accepted_rows = [
        row for row in prediction_rows if clinc150_guard_accepts(row, guard_rule=guard_rule)
    ]
    wrong_rows = [
        row for row in accepted_rows if row["predicted_frame"] != row["gold_frame"]
    ]
    wrong_by_gold = Counter(str(row["gold_intent"]) for row in wrong_rows)
    wrong_by_predicted = Counter(str(row["predicted_intent"]) for row in wrong_rows)
    in_scope_confusions = Counter(
        f"{row['gold_intent']} -> {row['predicted_intent']}"
        for row in wrong_rows
        if not row["gold_oos"] and not row["predicted_oos"]
    )
    oos_false_accepts = [
        row for row in wrong_rows if row["gold_oos"] and not row["predicted_oos"]
    ]
    top_families = in_scope_confusions.most_common(10)
    examples_by_family = {
        family: _accepted_error_examples(
            [
                row
                for row in wrong_rows
                if f"{row['gold_intent']} -> {row['predicted_intent']}" == family
            ],
            replay_oracle=replay_oracle,
            limit=max_examples_per_family,
        )
        for family, _count in top_families
    }
    return {
        "schema_version": "clinc150-accepted-error-summary-v1",
        "guard_name": guard_rule["name"],
        "guard_rule": guard_rule,
        "accepted": len(accepted_rows),
        "accepted_wrong": len(wrong_rows),
        "accepted_wrong_by_gold_intent": dict(wrong_by_gold.most_common(20)),
        "accepted_wrong_by_predicted_intent": dict(wrong_by_predicted.most_common(20)),
        "accepted_wrong_in_scope_confusions": dict(top_families),
        "accepted_oos_false_accepts": len(oos_false_accepts),
        "accepted_distributions": _row_score_distributions(accepted_rows),
        "accepted_wrong_distributions": _row_score_distributions(wrong_rows),
        "accepted_wrong_examples": _accepted_error_examples(
            wrong_rows,
            replay_oracle=replay_oracle,
            limit=20,
        ),
        "examples_by_in_scope_confusion": examples_by_family,
    }


def clinc150_accepted_error_rows(
    *,
    prediction_rows: list[dict[str, Any]],
    guard_rule: dict[str, Any],
    replay_oracle: Clinc150L4ReplayOracle | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for row in prediction_rows:
        if not clinc150_guard_accepts(row, guard_rule=guard_rule):
            continue
        if row["predicted_frame"] == row["gold_frame"]:
            continue
        teacher_row = (
            replay_oracle.row_for(str(row["request_id"]))
            if replay_oracle is not None
            else None
        )
        rows.append(
            {
                "request_id": row["request_id"],
                "utterance": row.get("utterance"),
                "gold_intent": row.get("gold_intent"),
                "predicted_intent": row.get("predicted_intent"),
                "teacher_intent": _teacher_intent(teacher_row) if teacher_row else None,
                "gold_oos": row.get("gold_oos"),
                "predicted_oos": row.get("predicted_oos"),
                "guard_probability": row.get("guard_probability"),
                "top1_probability": row.get("top1_probability"),
                "margin": row.get("margin"),
                "entropy": row.get("entropy"),
                "oos_probability": row.get("oos_probability"),
                "oos_rank": row.get("oos_rank"),
                "oos_margin": row.get("oos_margin"),
            }
        )
    return rows


def write_clinc150_accepted_error_rows(
    *,
    rows: list[dict[str, Any]],
    path: Path,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def run_clinc150_calibration_repair(
    *,
    bundle_path: Path,
    data_dir: Path,
    out_dir: Path,
    train_teacher_details: Path,
    validation_teacher_details: Path,
    test_teacher_details: Path,
    train_predictions: Path | None = None,
    validation_predictions: Path | None = None,
    test_predictions: Path | None = None,
    validation_uniform_predictions: Path | None = None,
    validation_zipf_heavy_predictions: Path | None = None,
    thresholds: tuple[float, ...] = CLINC150_CALIBRATION_REPAIR_THRESHOLDS,
    selection_min_precision: float = 0.99,
    write_accepted_errors: bool = True,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = L2StudentBundle.load(bundle_path)
    train_oracle = load_clinc150_l4_replay_oracle(train_teacher_details)
    validation_oracle = load_clinc150_l4_replay_oracle(validation_teacher_details)

    train_records = sample_clinc150_records(
        load_processed_records(data_dir, split="train"),
        stream="stratified",
        max_requests=None,
    )
    validation_records = load_processed_records(data_dir, split="validation")
    generated_train_predictions_path = (
        out_dir / "predictions" / "train-full-stratified" / "clinc150_l2_predictions.jsonl"
    )
    train_prediction_rows = _load_or_generate_prediction_rows(
        bundle=bundle,
        records=train_records,
        source_path=train_predictions,
        generated_path=generated_train_predictions_path,
    )
    validation_prediction_rows = _load_or_generate_prediction_rows(
        bundle=bundle,
        records=validation_records,
        source_path=validation_predictions,
        generated_path=out_dir
        / "predictions"
        / "validation-sequential"
        / "clinc150_l2_predictions.jsonl",
    )
    validation_uniform_rows = _load_or_generate_prediction_rows(
        bundle=bundle,
        records=sample_clinc150_records(
            validation_records,
            stream="uniform",
            max_requests=len(validation_records),
        ),
        source_path=validation_uniform_predictions,
        generated_path=out_dir
        / "predictions"
        / "validation-uniform"
        / "clinc150_l2_predictions.jsonl",
    )
    validation_zipf_heavy_rows = _load_or_generate_prediction_rows(
        bundle=bundle,
        records=sample_clinc150_records(
            validation_records,
            stream="zipf-heavy",
            max_requests=len(validation_records),
        ),
        source_path=validation_zipf_heavy_predictions,
        generated_path=out_dir
        / "predictions"
        / "validation-zipf-heavy"
        / "clinc150_l2_predictions.jsonl",
    )

    split_summary = build_clinc150_calibration_splits(
        list(train_oracle.rows_by_request_id.values())
    )
    split_path = out_dir / "clinc150_calibration_splits.json"
    split_path.write_text(
        json.dumps(split_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    calibration_rows = clinc150_prediction_rows_for_request_ids(
        train_prediction_rows,
        split_summary["general_calibration"]["request_ids"],
    )
    calibration_dev_rows = clinc150_prediction_rows_for_request_ids(
        train_prediction_rows,
        split_summary["general_dev"]["request_ids"],
    )
    oos_heavy_slice = build_clinc150_oos_heavy_slice(
        teacher_rows=list(train_oracle.rows_by_request_id.values()),
        prediction_rows=train_prediction_rows,
    )
    oos_heavy_rows = clinc150_prediction_rows_for_request_ids(
        train_prediction_rows,
        oos_heavy_slice["request_ids"],
    )

    baseline_rule = _guard_rule(family="threshold", threshold=0.98)
    baseline_audit = {
        "train_calibration": _view_audit(
            prediction_rows=calibration_rows,
            guard_rule=baseline_rule,
            replay_oracle=train_oracle,
        ),
        "train_dev": _view_audit(
            prediction_rows=calibration_dev_rows,
            guard_rule=baseline_rule,
            replay_oracle=train_oracle,
        ),
        "oos_heavy": _view_audit(
            prediction_rows=oos_heavy_rows,
            guard_rule=baseline_rule,
            replay_oracle=train_oracle,
        ),
        "validation": _view_audit(
            prediction_rows=validation_prediction_rows,
            guard_rule=baseline_rule,
            replay_oracle=validation_oracle,
        ),
    }
    veto_candidates = _predicted_intent_veto_candidates(
        prediction_views=(calibration_dev_rows, validation_prediction_rows),
        guard_rule=baseline_rule,
        max_vetoes=3,
    )
    guard_rules = clinc150_calibration_guard_rules(
        thresholds=thresholds,
        predicted_intent_vetoes=veto_candidates,
    )
    calibration_dev_results = evaluate_clinc150_guard_rules(
        prediction_rows=calibration_dev_rows,
        guard_rules=guard_rules,
        replay_oracle=train_oracle,
    )
    oos_heavy_results = evaluate_clinc150_guard_rules(
        prediction_rows=oos_heavy_rows,
        guard_rules=guard_rules,
        replay_oracle=train_oracle,
    )
    validation_results = evaluate_clinc150_guard_rules(
        prediction_rows=validation_prediction_rows,
        guard_rules=guard_rules,
        replay_oracle=validation_oracle,
    )
    selected = select_clinc150_calibration_guard(
        calibration_dev_results=calibration_dev_results,
        oos_heavy_results=oos_heavy_results,
        validation_results=validation_results,
        min_precision=selection_min_precision,
    )
    selected_rule = selected["guard_rule"] if selected is not None else None
    test_oracle = None
    test_prediction_rows: list[dict[str, Any]] = []
    locked_test = None
    if selected_rule is not None:
        test_oracle = load_clinc150_l4_replay_oracle(test_teacher_details)
        test_prediction_rows = _load_or_generate_prediction_rows(
            bundle=bundle,
            records=load_processed_records(data_dir, split="test"),
            source_path=test_predictions,
            generated_path=out_dir
            / "predictions"
            / "test-sequential"
            / "clinc150_l2_predictions.jsonl",
        )
        locked_test = evaluate_clinc150_guard_rule(
            prediction_rows=test_prediction_rows,
            guard_rule=selected_rule,
            replay_oracle=test_oracle,
        )
    stream_confirmation = (
        {
            "validation_sequential": evaluate_clinc150_guard_rule(
                prediction_rows=validation_prediction_rows,
                guard_rule=selected_rule,
                replay_oracle=validation_oracle,
            ),
            "validation_uniform": evaluate_clinc150_guard_rule(
                prediction_rows=validation_uniform_rows,
                guard_rule=selected_rule,
                replay_oracle=validation_oracle,
            ),
            "validation_zipf_heavy": evaluate_clinc150_guard_rule(
                prediction_rows=validation_zipf_heavy_rows,
                guard_rule=selected_rule,
                replay_oracle=validation_oracle,
            ),
            "oos_heavy_diagnostic": evaluate_clinc150_guard_rule(
                prediction_rows=oos_heavy_rows,
                guard_rule=selected_rule,
                replay_oracle=train_oracle,
            ),
        }
        if selected_rule is not None
        else None
    )
    accepted_error_paths = {}
    if write_accepted_errors:
        error_views = {
            "baseline_train_dev": (calibration_dev_rows, baseline_rule, train_oracle),
            "baseline_validation": (validation_prediction_rows, baseline_rule, validation_oracle),
        }
        if selected_rule is not None:
            error_views.update(
                {
                    "selected_train_dev": (
                        calibration_dev_rows,
                        selected_rule,
                        train_oracle,
                    ),
                    "selected_validation": (
                        validation_prediction_rows,
                        selected_rule,
                        validation_oracle,
                    ),
                    "selected_locked_test": (
                        test_prediction_rows,
                        selected_rule,
                        test_oracle,
                    ),
                }
            )
        for name, (rows, rule, oracle) in error_views.items():
            path = write_clinc150_accepted_error_rows(
                rows=clinc150_accepted_error_rows(
                    prediction_rows=rows,
                    guard_rule=rule,
                    replay_oracle=oracle,
                ),
                path=out_dir / "accepted-errors" / f"{name}.jsonl",
            )
            accepted_error_paths[name] = str(path)

    result = {
        "schema_version": "clinc150-calibration-repair-v1",
        "measurement_path": "l2_only_shadow_l2_plus_l4_fallback",
        "l0_enabled": False,
        "new_paid_l4_calls": 0,
        "new_paid_spend_usd": 0.0,
        "reused_artifacts": {
            "bundle_path": str(bundle_path),
            "train_teacher_details": str(train_teacher_details),
            "validation_teacher_details": str(validation_teacher_details),
            "test_teacher_details": str(test_teacher_details),
            "train_predictions": str(train_predictions) if train_predictions is not None else None,
            "validation_predictions": str(validation_predictions)
            if validation_predictions is not None
            else None,
            "test_predictions": str(test_predictions) if test_predictions is not None else None,
            "validation_uniform_predictions": str(validation_uniform_predictions)
            if validation_uniform_predictions is not None
            else None,
            "validation_zipf_heavy_predictions": str(validation_zipf_heavy_predictions)
            if validation_zipf_heavy_predictions is not None
            else None,
        },
        "generated_artifacts": {
            "train_predictions": (
                None if train_predictions is not None else str(generated_train_predictions_path)
            ),
            "split_summary": str(split_path),
            "accepted_error_jsonl": accepted_error_paths,
        },
        "selection_policy": {
            "min_precision": selection_min_precision,
            "max_oos_false_accept_rate": 0.02,
            "min_accuracy_delta_vs_all_l4": -0.005,
            "locked_test_used_for_selection": False,
        },
        "l4_replay_oracle": {
            "train": train_oracle.validate_coverage(
                [row["request_id"] for row in train_prediction_rows]
            ),
            "validation": validation_oracle.validate_coverage(
                [row["request_id"] for row in validation_prediction_rows]
            ),
            "test": (
                test_oracle.validate_coverage(
                    [row["request_id"] for row in test_prediction_rows]
                )
                if test_oracle is not None
                else None
            ),
        },
        "splits": split_summary,
        "oos_heavy_slice": oos_heavy_slice,
        "baseline_rule": baseline_rule,
        "baseline_audit": baseline_audit,
        "veto_candidates": [list(candidate) for candidate in veto_candidates],
        "candidate_count": len(guard_rules),
        "candidate_families": dict(
            Counter(str(rule["family"]) for rule in guard_rules)
        ),
        "selection_inputs": {
            "calibration_dev": calibration_dev_results,
            "oos_heavy": oos_heavy_results,
            "validation": validation_results,
        },
        "selected": selected,
        "locked_test": locked_test,
        "stream_confirmation": stream_confirmation,
    }
    summary_path = out_dir / "clinc150_calibration_repair_summary.json"
    summary_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result["summary_path"] = str(summary_path)
    return result


def clinc150_autoresearch_traces_from_teacher_rows(
    rows: list[dict[str, Any]],
) -> list[TeacherTrace]:
    traces: list[TeacherTrace] = []
    for row in rows:
        if row.get("parse_failure") or row.get("teacher_frame") is None:
            continue
        teacher_frame = Frame.model_validate(row["teacher_frame"])
        traces.append(
            TeacherTrace(
                request_id=str(row["request_id"]),
                utterance=str(row["utterance"]),
                teacher_frame=teacher_frame,
                chosen_layer="L4",
                final_frame=teacher_frame,
                layer_results=[
                    LayerResult(
                        layer="L2",
                        accepted=False,
                        frame=None,
                        latency_ms=0.0,
                        reason="clinc150 autoresearch lower layers disabled",
                    ),
                    LayerResult(
                        layer="L4",
                        accepted=True,
                        frame=teacher_frame,
                        latency_ms=float(row.get("latency_ms", 0.0)),
                        cost_usd=float(row.get("cost_usd", 0.0)),
                        metadata={
                            "model": row.get("model"),
                            "tokens": row.get("tokens", 0),
                            "attempt_count": row.get("attempt_count", 1),
                            "retry_recovered": row.get("retry_recovered", False),
                            "replay_oracle_row": True,
                        },
                    ),
                ],
                l4_usage={
                    "tokens": row.get("tokens", 0),
                    "cost_usd": row.get("cost_usd", 0.0),
                    "latency_ms": row.get("latency_ms", 0.0),
                    "attempt_count": row.get("attempt_count", 1),
                    "unknown_usage_attempts": row.get("unknown_usage_attempts", 0),
                },
                metadata={
                    "source": "clinc150_teacher_details",
                    "teacher_visible_only": True,
                    "gold_frame_withheld": True,
                },
                timestamp=str(row.get("timestamp") or datetime.now(UTC).isoformat()),
            )
        )
    return traces


def run_clinc150_l2_autoresearch(
    *,
    data_dir: Path,
    out_dir: Path,
    source_repo_dir: Path,
    train_teacher_details: Path,
    validation_teacher_details: Path,
    test_teacher_details: Path,
    mode: str = "agent-session",
    rounds: int = 16,
    budget_profile: str = "fixed-inner",
    max_agent_rounds: int | None = None,
    timeout_s: float | None = None,
    local_search_trials: int = 32,
    visible_validation_folds: int = 5,
    visible_validation_ratio: float | None = 0.30,
    visible_cross_audit_folds: int = 3,
    local_search_cross_audit_top_k: int = 4,
    codex_command: str = "codex",
    codex_model: str | None = "gpt-5.5",
) -> dict[str, Any]:
    from darjeeling.targets.nlu.compiler.l2_target_evolution import (
        L2TargetEvolutionConfig,
        run_l2_target_evolution,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    train_rows = load_teacher_rows(train_teacher_details)
    train_traces = clinc150_autoresearch_traces_from_teacher_rows(train_rows)
    traces_path = out_dir / "clinc150_autoresearch_teacher_train_traces.jsonl"
    traces_path.write_text(
        "".join(trace.model_dump_json() + "\n" for trace in train_traces),
        encoding="utf-8",
    )
    target_context = _clinc150_autoresearch_target_context(
        train_rows=train_rows,
        train_teacher_details=train_teacher_details,
        validation_teacher_details=validation_teacher_details,
        test_teacher_details=test_teacher_details,
    )
    target_run_dir = out_dir / "target-evolution"
    target_summary = run_l2_target_evolution(
        config=L2TargetEvolutionConfig(
            source_repo_dir=source_repo_dir,
            job_dir=target_run_dir,
            rounds=rounds,
            mode=mode,  # type: ignore[arg-type]
            budget_profile=budget_profile,  # type: ignore[arg-type]
            split_policy="intent-stratified",
            target_scope="teacher_train",
            visible_validation_folds=visible_validation_folds,
            visible_validation_ratio=visible_validation_ratio,
            visible_cross_audit_folds=visible_cross_audit_folds,
            local_search_trials=local_search_trials,
            local_search_cross_audit_top_k=local_search_cross_audit_top_k,
            max_agent_rounds=max_agent_rounds,
            timeout_s=timeout_s if timeout_s is not None else 7200.0,
            codex_command=codex_command,
            codex_model=codex_model,
            min_accepted_accuracy=0.995,
            max_wrong_accept_rate=0.005,
            inner_patience_rounds=0,
            initial_target_config=CLINC150_AUTORESEARCH_INITIAL_L2_CONFIG,
            target_context=target_context,
        ),
        traces=train_traces,
    )

    candidate_round = target_summary.get("best_adoptable_round")
    candidate_role = "best_adoptable_round"
    if not isinstance(candidate_round, dict):
        candidate_round = target_summary.get("best_selection_round")
        candidate_role = "best_selection_round"
    diagnostic_round = candidate_round
    diagnostic_role = candidate_role
    if not isinstance(diagnostic_round, dict):
        diagnostic_round = target_summary.get("best_round")
        diagnostic_role = "best_round_diagnostic_only"

    validation_evaluation = None
    stream_confirmation = None
    locked_test = None
    selected_for_locked_test = False
    candidate_target_dir = None
    if isinstance(diagnostic_round, dict):
        candidate_target_dir = _clinc150_target_dir_from_round(
            target_run_dir=target_run_dir,
            round_payload=diagnostic_round,
        )
        module_path = candidate_target_dir / "target_l2.py"
        candidate_bundle = train_clinc150_target_bundle(
            train_rows=train_rows,
            target_module_path=module_path,
        )
        validation_records = load_processed_records(data_dir, split="validation")
        validation_rows = load_teacher_rows(validation_teacher_details)
        validation_evaluation = evaluate_clinc150_target_module(
            bundle=candidate_bundle,
            target_module_path=module_path,
            records=validation_records,
            teacher_rows=validation_rows,
            include_prediction_rows=False,
        )
        validation_uniform = evaluate_clinc150_target_module(
            bundle=candidate_bundle,
            target_module_path=module_path,
            records=sample_clinc150_records(
                validation_records,
                stream="uniform",
                max_requests=len(validation_records),
            ),
            teacher_rows=validation_rows,
            include_prediction_rows=False,
        )
        validation_zipf_heavy = evaluate_clinc150_target_module(
            bundle=candidate_bundle,
            target_module_path=module_path,
            records=sample_clinc150_records(
                validation_records,
                stream="zipf-heavy",
                max_requests=len(validation_records),
            ),
            teacher_rows=validation_rows,
            include_prediction_rows=False,
        )
        stream_confirmation = {
            "validation_sequential": validation_evaluation["metrics"],
            "validation_uniform": validation_uniform["metrics"],
            "validation_zipf_heavy": validation_zipf_heavy["metrics"],
        }
        selected_for_locked_test = (
            candidate_role == "best_adoptable_round"
            and _clinc150_autoresearch_visible_gates_pass(stream_confirmation)
        )
        if selected_for_locked_test:
            locked_test = evaluate_clinc150_target_module(
                bundle=candidate_bundle,
                target_module_path=module_path,
                records=load_processed_records(data_dir, split="test"),
                teacher_rows=load_teacher_rows(test_teacher_details),
                include_prediction_rows=False,
            )

    result = {
        "schema_version": "clinc150-l2-autoresearch-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "measurement_path": "l2_only_shadow_l2_plus_l4_fallback",
        "l0_enabled": False,
        "l1_enabled": False,
        "l3_enabled": False,
        "new_paid_l4_calls": 0,
        "new_paid_spend_usd": 0.0,
        "source_repo_dir": str(source_repo_dir),
        "run_root": str(out_dir),
        "target_evolution_run": str(target_run_dir),
        "target_evolution_summary_path": str(target_run_dir / "summary.json"),
        "teacher_train_traces": str(traces_path),
        "teacher_train_trace_count": len(train_traces),
        "target_context": target_context,
        "target_evolution": {
            "mode": target_summary.get("mode"),
            "rounds_requested": target_summary.get("rounds_requested"),
            "rounds_completed": target_summary.get("rounds_completed"),
            "stop_reason": target_summary.get("stop_reason"),
            "evidence_policy": target_summary.get("evidence_policy"),
            "selection_decision": target_summary.get("selection_decision"),
            "adoption_decision": target_summary.get("adoption_decision"),
            "best_round": target_summary.get("best_round"),
            "best_selection_round": target_summary.get("best_selection_round"),
            "best_adoptable_round": target_summary.get("best_adoptable_round"),
        },
        "candidate": {
            "role": diagnostic_role if diagnostic_round is not None else None,
            "round": diagnostic_round.get("round") if isinstance(diagnostic_round, dict) else None,
            "target_dir": str(candidate_target_dir) if candidate_target_dir is not None else None,
            "selected_for_locked_test": selected_for_locked_test,
            "locked_test_policy": (
                "official test evaluated only when best_adoptable candidate passes "
                "official validation and validation streams"
            ),
        },
        "validation_evaluation": validation_evaluation,
        "stream_confirmation": stream_confirmation,
        "locked_test": locked_test,
        "locked_test_exposures": 1 if locked_test is not None else 0,
    }
    summary_path = out_dir / "clinc150_l2_autoresearch_summary.json"
    summary_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result["summary_path"] = str(summary_path)
    return result


def train_clinc150_target_bundle(
    *,
    train_rows: list[dict[str, Any]],
    target_module_path: Path,
) -> L2StudentBundle:
    target_module = load_target_module(target_module_path)
    config_payload = {
        **CLINC150_AUTORESEARCH_INITIAL_L2_CONFIG,
        **target_config_overrides(target_module),
    }
    return train_clinc150_l2(
        training_examples_from_teacher_rows(train_rows),
        config=L2StudentConfig(**config_payload),
    )


def evaluate_clinc150_target_module(
    *,
    bundle: L2StudentBundle,
    target_module_path: Path,
    records: list[DataRecord],
    teacher_rows: list[dict[str, Any]],
    include_prediction_rows: bool = False,
) -> dict[str, Any]:
    target_module = load_target_module(target_module_path)
    replay_oracle = Clinc150L4ReplayOracle.from_rows(teacher_rows)
    prediction_rows: list[dict[str, Any]] = []
    accepted_flags: list[bool] = []
    for record in records:
        row, accepted = _l2_target_prediction_row(
            bundle=bundle,
            target_module=target_module,
            target_module_path=target_module_path,
            record=record,
        )
        prediction_rows.append(row)
        accepted_flags.append(accepted)
    metrics = _evaluate_clinc150_acceptance_flags(
        prediction_rows=prediction_rows,
        accepted_flags=accepted_flags,
        teacher_by_request_id=replay_oracle.rows_by_request_id,
        guard_name="target_l2_accept_prediction",
        guard_rule={
            "name": "target_l2_accept_prediction",
            "family": "target_l2",
            "threshold": bundle.config.accept_threshold,
            "target_module_path": str(target_module_path),
        },
        threshold=bundle.config.accept_threshold,
    )
    result = {
        "schema_version": "clinc150-l2-target-eval-v1",
        "requests": len(prediction_rows),
        "measurement_path": "l2_only_shadow_l2_plus_l4_fallback",
        "l0_enabled": False,
        "target_module_path": str(target_module_path),
        "config": bundle.config.model_dump(mode="json"),
        "l4_replay_oracle": replay_oracle.validate_coverage(
            [row["request_id"] for row in prediction_rows]
        ),
        "raw_l2_accuracy": _rate(
            sum(1 for row in prediction_rows if row["predicted_frame"] == row["gold_frame"]),
            len(prediction_rows),
        ),
        "metrics": metrics,
    }
    if include_prediction_rows:
        result["prediction_rows"] = prediction_rows
    return result


def _clinc150_autoresearch_target_context(
    *,
    train_rows: list[dict[str, Any]],
    train_teacher_details: Path,
    validation_teacher_details: Path,
    test_teacher_details: Path,
) -> dict[str, Any]:
    parsed_rows = [row for row in train_rows if not row.get("parse_failure")]
    teacher_intents = Counter(
        _teacher_intent(row)
        for row in parsed_rows
        if _teacher_intent(row) is not None
    )
    return {
        "schema_version": "clinc150-l2-autoresearch-context-v1",
        "visibility": "agent_visible_train_and_inner_validation_only",
        "target": "clinc150",
        "objective": (
            "repair high-confidence OOS and intent-boundary L2 accepts using "
            "target-local L2 config, postprocess, and accept_prediction logic"
        ),
        "strict_gates": {
            "official_validation_min_accepted_precision": 0.995,
            "official_validation_max_oos_false_accept_rate": 0.02,
            "official_validation_min_cascade_delta_vs_all_l4": -0.005,
            "official_validation_min_coverage_for_locked_test": 0.40,
            "locked_test_min_accepted_precision": 0.99,
            "locked_test_max_oos_false_accept_rate": 0.02,
            "locked_test_min_cascade_delta_vs_all_l4": -0.005,
            "practical_locked_test_coverage_target": 0.40,
        },
        "initial_l2_config": CLINC150_AUTORESEARCH_INITIAL_L2_CONFIG,
        "visible_teacher_train_rows": len(train_rows),
        "visible_parsed_teacher_train_rows": len(parsed_rows),
        "teacher_predicted_oos_rows": teacher_intents.get(CLINC150_OOS_INTENT, 0),
        "teacher_intent_count": len(teacher_intents),
        "reused_artifacts": {
            "train_teacher_details": str(train_teacher_details),
            "validation_teacher_details_for_outer_visible_eval": str(
                validation_teacher_details
            ),
            "test_teacher_details_withheld_until_selection": str(test_teacher_details),
        },
        "withheld_data_policy": {
            "official_test_labels_or_accepted_errors_in_workspace": False,
            "official_test_used_for_candidate_selection": False,
            "locked_test_only_after_best_adoptable_candidate_passes_validation": True,
        },
        "allowed_surface": [
            "target/config.json L2StudentConfig overrides",
            "target/target_l2.py postprocess_frame",
            "target/target_l2.py accept_prediction veto logic",
            "metadata.intent_probabilities and CLINC150 OOS-risk signals",
        ],
    }


def _clinc150_target_dir_from_round(
    *,
    target_run_dir: Path,
    round_payload: dict[str, Any],
) -> Path:
    snapshot = round_payload.get("target_snapshot")
    if isinstance(snapshot, str) and snapshot:
        snapshot_path = Path(snapshot)
        if not snapshot_path.is_absolute():
            snapshot_path = target_run_dir / snapshot_path
        return snapshot_path
    return target_run_dir / "workspace" / "l2_target" / "target"


def _clinc150_autoresearch_visible_gates_pass(
    stream_confirmation: dict[str, dict[str, Any]],
) -> bool:
    sequential = stream_confirmation["validation_sequential"]
    uniform = stream_confirmation["validation_uniform"]
    zipf_heavy = stream_confirmation["validation_zipf_heavy"]
    return (
        _guard_result_passes_constraints(
            sequential,
            min_precision=0.995,
            max_oos_false_accept_rate=0.02,
            min_accuracy_delta_vs_all_l4=-0.005,
        )
        and float(sequential.get("accepted_coverage") or 0.0) >= 0.40
        and _guard_result_passes_constraints(
            uniform,
            min_precision=0.99,
            max_oos_false_accept_rate=0.02,
            min_accuracy_delta_vs_all_l4=-0.005,
        )
        and _guard_result_passes_constraints(
            zipf_heavy,
            min_precision=0.99,
            max_oos_false_accept_rate=0.02,
            min_accuracy_delta_vs_all_l4=-0.005,
        )
    )


def _l2_target_prediction_row(
    *,
    bundle: L2StudentBundle,
    target_module: Any,
    target_module_path: Path,
    record: DataRecord,
) -> tuple[dict[str, Any], bool]:
    started = perf_counter()
    prediction = bundle.predict(record.utterance)
    metadata = prediction.model_dump(mode="json")
    frame = target_postprocess_frame(
        target_module,
        utterance=record.utterance,
        frame=prediction.frame,
        metadata=metadata,
    )
    runtime_enabled = getattr(bundle.config, "runtime_enabled", True)
    default_accept = (
        runtime_enabled and prediction.guard_probability >= bundle.config.accept_threshold
    )
    accepted = target_accept_prediction(
        target_module,
        utterance=record.utterance,
        frame=frame,
        metadata=metadata,
        default_accept=default_accept,
    )
    latency_ms = (perf_counter() - started) * 1000.0
    row = {
        "request_id": record.request_id,
        "utterance": record.utterance,
        "gold_frame": record.gold_frame.model_dump(mode="json"),
        "gold_intent": record.gold_frame.intent,
        "gold_oos": record.gold_frame.intent == CLINC150_OOS_INTENT,
        "raw_predicted_frame": prediction.frame.model_dump(mode="json"),
        "predicted_frame": frame.model_dump(mode="json"),
        "predicted_intent": frame.intent,
        "predicted_oos": frame.intent == CLINC150_OOS_INTENT,
        "guard_probability": prediction.guard_probability,
        "top1_probability": prediction.top1_probability,
        "margin": prediction.margin,
        "entropy": prediction.entropy,
        "intent_probabilities": prediction.intent_probabilities,
        "oos_probability": oos_probability_from_intent_probabilities(
            prediction.intent_probabilities
        ),
        "oos_rank": oos_rank_from_intent_probabilities(prediction.intent_probabilities),
        "oos_margin": oos_margin_from_intent_probabilities(
            prediction.intent_probabilities,
        ),
        "target_default_accept": default_accept,
        "target_accepted": accepted,
        "target_vetoed": bool(default_accept and not accepted),
        "target_postprocessed": frame != prediction.frame,
        "target_module": str(target_module_path),
        "latency_ms": latency_ms,
    }
    return row, accepted


def _load_or_generate_prediction_rows(
    *,
    bundle: L2StudentBundle,
    records: list[DataRecord],
    source_path: Path | None,
    generated_path: Path,
) -> list[dict[str, Any]]:
    if source_path is not None:
        return load_clinc150_l2_prediction_rows(source_path)
    rows = clinc150_l2_prediction_rows(bundle=bundle, records=records)
    write_clinc150_l2_prediction_rows(rows, generated_path)
    return rows


def _view_audit(
    *,
    prediction_rows: list[dict[str, Any]],
    guard_rule: dict[str, Any],
    replay_oracle: Clinc150L4ReplayOracle,
) -> dict[str, Any]:
    return {
        "metrics": evaluate_clinc150_guard_rule(
            prediction_rows=prediction_rows,
            guard_rule=guard_rule,
            replay_oracle=replay_oracle,
        ),
        "accepted_errors": summarize_clinc150_accepted_errors(
            prediction_rows=prediction_rows,
            guard_rule=guard_rule,
            replay_oracle=replay_oracle,
        ),
    }


def _predicted_intent_veto_candidates(
    *,
    prediction_views: tuple[list[dict[str, Any]], ...],
    guard_rule: dict[str, Any],
    max_vetoes: int,
) -> tuple[tuple[str, ...], ...]:
    counters = []
    for rows in prediction_views:
        counter: Counter[str] = Counter()
        for row in rows:
            if not clinc150_guard_accepts(row, guard_rule=guard_rule):
                continue
            if row["predicted_frame"] == row["gold_frame"]:
                continue
            counter[str(row["predicted_intent"])] += 1
        counters.append(counter)
    if not counters or any(not counter for counter in counters):
        return ()
    shared_intents = set(counters[0])
    for counter in counters[1:]:
        shared_intents &= set(counter)
    if not shared_intents:
        return ()
    ranked = sorted(
        shared_intents,
        key=lambda intent: (
            -sum(counter[intent] for counter in counters),
            intent,
        ),
    )[:max_vetoes]
    return tuple(tuple(ranked[:index]) for index in range(1, len(ranked) + 1))


def _l2_prediction_row(*, bundle: L2StudentBundle, record: DataRecord) -> dict[str, Any]:
    started = perf_counter()
    prediction = bundle.predict(record.utterance)
    latency_ms = (perf_counter() - started) * 1000.0
    return {
        "request_id": record.request_id,
        "utterance": record.utterance,
        "gold_frame": record.gold_frame.model_dump(mode="json"),
        "gold_intent": record.gold_frame.intent,
        "gold_oos": record.gold_frame.intent == CLINC150_OOS_INTENT,
        "predicted_frame": prediction.frame.model_dump(mode="json"),
        "predicted_intent": prediction.frame.intent,
        "predicted_oos": prediction.frame.intent == CLINC150_OOS_INTENT,
        "guard_probability": prediction.guard_probability,
        "top1_probability": prediction.top1_probability,
        "margin": prediction.margin,
        "entropy": prediction.entropy,
        "intent_probabilities": prediction.intent_probabilities,
        "oos_probability": oos_probability_from_intent_probabilities(
            prediction.intent_probabilities
        ),
        "oos_rank": oos_rank_from_intent_probabilities(prediction.intent_probabilities),
        "oos_margin": oos_margin_from_intent_probabilities(
            prediction.intent_probabilities,
        ),
        "latency_ms": latency_ms,
    }


def oos_probability_from_intent_probabilities(
    intent_probabilities: dict[str, float],
) -> float:
    return float(intent_probabilities.get(CLINC150_OOS_INTENT, 0.0))


def oos_rank_from_intent_probabilities(
    intent_probabilities: dict[str, float],
) -> int | None:
    if CLINC150_OOS_INTENT not in intent_probabilities:
        return None
    ranked = sorted(
        intent_probabilities.items(),
        key=lambda item: (-float(item[1]), item[0]),
    )
    for index, (intent, _probability) in enumerate(ranked, start=1):
        if intent == CLINC150_OOS_INTENT:
            return index
    return None


def oos_margin_from_intent_probabilities(
    intent_probabilities: dict[str, float],
) -> float:
    if not intent_probabilities:
        return 1.0
    top_probability = max(float(value) for value in intent_probabilities.values())
    return top_probability - oos_probability_from_intent_probabilities(intent_probabilities)


def _guard_rule(
    *,
    family: str,
    threshold: float,
    min_margin: float | None = None,
    max_entropy: float | None = None,
    max_oos_probability: float | None = None,
    min_oos_margin: float | None = None,
    min_oos_rank: int | None = None,
    veto_predicted_intents: tuple[str, ...] = (),
) -> dict[str, Any]:
    name_parts = [family, f"threshold_{_format_guard_float(threshold)}"]
    if min_margin is not None:
        name_parts.append(f"margin_{_format_guard_float(min_margin)}")
    if max_entropy is not None:
        name_parts.append(f"entropy_{_format_guard_float(max_entropy)}")
    if max_oos_probability is not None:
        name_parts.append(f"oosprob_{_format_guard_float(max_oos_probability)}")
    if min_oos_margin is not None:
        name_parts.append(f"oosmargin_{_format_guard_float(min_oos_margin)}")
    if min_oos_rank is not None:
        name_parts.append(f"oosrank_{min_oos_rank}")
    if veto_predicted_intents:
        digest = hashlib.sha256(
            json.dumps(veto_predicted_intents, sort_keys=True).encode("utf-8")
        ).hexdigest()[:8]
        name_parts.append(f"veto_{digest}")
    return {
        "name": "__".join(name_parts),
        "family": family,
        "threshold": threshold,
        "min_margin": min_margin,
        "max_entropy": max_entropy,
        "max_oos_probability": max_oos_probability,
        "min_oos_margin": min_oos_margin,
        "min_oos_rank": min_oos_rank,
        "veto_predicted_intents": list(veto_predicted_intents),
    }


def _format_guard_float(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def _stable_split_key(seed: str, request_id: str) -> str:
    return hashlib.sha256(f"{seed}:{request_id}".encode()).hexdigest()


def _split_payload(request_ids: list[str]) -> dict[str, Any]:
    return {
        "requests": len(request_ids),
        "request_ids": request_ids,
    }


def _row_score_distributions(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        field: _numeric_distribution(
            [float(row[field]) for row in rows if row.get(field) is not None]
        )
        for field in (
            "guard_probability",
            "top1_probability",
            "margin",
            "entropy",
            "oos_probability",
            "oos_rank",
            "oos_margin",
        )
    }


def _numeric_distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p10": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "max": None,
            "mean": None,
        }
    return {
        "count": len(values),
        "min": min(values),
        "p10": _percentile(values, 10),
        "p50": _percentile(values, 50),
        "p90": _percentile(values, 90),
        "p95": _percentile(values, 95),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def _accepted_error_examples(
    rows: list[dict[str, Any]],
    *,
    replay_oracle: Clinc150L4ReplayOracle | None,
    limit: int,
) -> list[dict[str, Any]]:
    examples = []
    for row in rows[:limit]:
        teacher_row = (
            replay_oracle.row_for(str(row["request_id"]))
            if replay_oracle is not None
            else None
        )
        examples.append(
            {
                "request_id": row["request_id"],
                "utterance": row.get("utterance"),
                "gold_intent": row.get("gold_intent"),
                "predicted_intent": row.get("predicted_intent"),
                "teacher_intent": _teacher_intent(teacher_row) if teacher_row else None,
                "guard_probability": row.get("guard_probability"),
                "top1_probability": row.get("top1_probability"),
                "margin": row.get("margin"),
                "entropy": row.get("entropy"),
                "oos_probability": row.get("oos_probability"),
                "oos_rank": row.get("oos_rank"),
                "oos_margin": row.get("oos_margin"),
            }
        )
    return examples


def _wilson_lower_bound(successes: int, total: int, *, z: float = 1.96) -> float | None:
    if total <= 0:
        return None
    p = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    centre = p + z2 / (2.0 * total)
    margin = z * ((p * (1.0 - p) + z2 / (4.0 * total)) / total) ** 0.5
    return (centre - margin) / denominator


def _guard_result_passes_constraints(
    row: dict[str, Any],
    *,
    min_precision: float,
    max_oos_false_accept_rate: float,
    min_accuracy_delta_vs_all_l4: float | None,
) -> bool:
    accepted_precision = row.get("accepted_precision")
    if accepted_precision is None or accepted_precision < min_precision:
        return False
    if row["lower_layer_oos_false_accept_rate"] > max_oos_false_accept_rate:
        return False
    if min_accuracy_delta_vs_all_l4 is None:
        return True
    accuracy_delta = row.get("accuracy_delta_vs_all_l4")
    return accuracy_delta is None or accuracy_delta >= min_accuracy_delta_vs_all_l4


def _guard_rule_complexity(guard_rule: dict[str, Any]) -> int:
    return (
        int(guard_rule.get("min_margin") is not None)
        + int(guard_rule.get("max_entropy") is not None)
        + int(guard_rule.get("max_oos_probability") is not None)
        + int(guard_rule.get("min_oos_margin") is not None)
        + int(guard_rule.get("min_oos_rank") is not None)
        + len(guard_rule.get("veto_predicted_intents") or [])
    )


def _l2_threshold_metrics(
    prediction_rows: list[dict[str, Any]],
    *,
    teacher_by_request_id: dict[str, dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    accepted_rows = [
        row for row in prediction_rows if float(row["guard_probability"]) >= threshold
    ]
    correct_accepted = [
        row for row in accepted_rows if row["predicted_frame"] == row["gold_frame"]
    ]
    oos_total = sum(1 for row in prediction_rows if row["gold_oos"])
    lower_oos_false_accepts = sum(
        1
        for row in accepted_rows
        if row["gold_oos"] and not row["predicted_oos"]
    )
    final_rows = [
        _cascade_row(
            row,
            teacher_by_request_id=teacher_by_request_id,
            accepted=row in accepted_rows,
        )
        for row in prediction_rows
    ]
    final_correct = sum(1 for row in final_rows if row["final_correct"])
    all_l4_correct = sum(1 for row in final_rows if row["all_l4_correct"])
    latencies = [row["latency_ms"] for row in final_rows]
    l4_calls = sum(1 for row in final_rows if row["l4_called"])
    l4_tokens = sum(float(row["l4_tokens"]) for row in final_rows)
    l4_cost = sum(float(row["l4_cost_usd"]) for row in final_rows)
    all_l4_rows = [
        teacher_by_request_id.get(row["request_id"])
        for row in prediction_rows
    ]
    paired_teacher_rows = [row for row in all_l4_rows if row is not None]
    all_l4_tokens = sum(float(row.get("tokens", 0.0)) for row in paired_teacher_rows)
    all_l4_cost = sum(float(row.get("cost_usd", 0.0)) for row in paired_teacher_rows)
    all_l4_latencies = [
        float(row.get("latency_ms", 0.0))
        for row in paired_teacher_rows
    ]
    teacher_parse_failures = sum(
        1 for row in paired_teacher_rows if row.get("parse_failure")
    )
    oos_counts = _oos_counts_from_frames(
        gold_intents=[row["gold_intent"] for row in prediction_rows],
        predicted_intents=[row["final_intent"] for row in final_rows],
    )
    oos_precision = _rate(oos_counts["true_positive"], oos_counts["predicted_positive"])
    oos_recall = _rate(oos_counts["true_positive"], oos_counts["gold_positive"])
    requests = len(prediction_rows)
    final_accuracy = _rate(final_correct, requests)
    all_l4_accuracy = _rate(all_l4_correct, requests) if teacher_by_request_id else None
    return {
        "threshold": threshold,
        "accepted": len(accepted_rows),
        "accepted_coverage": _rate(len(accepted_rows), requests),
        "accepted_precision": _rate(len(correct_accepted), len(accepted_rows)),
        "lower_layer_oos_false_accepts": lower_oos_false_accepts,
        "lower_layer_oos_false_accept_rate": _rate(lower_oos_false_accepts, oos_total) or 0.0,
        "all_l4_accuracy": all_l4_accuracy,
        "final_cascade_accuracy": final_accuracy,
        "accuracy_delta_vs_all_l4": (
            final_accuracy - all_l4_accuracy
            if final_accuracy is not None and all_l4_accuracy is not None
            else None
        ),
        "paired_teacher_rows": len(paired_teacher_rows),
        "paired_teacher_coverage": _rate(len(paired_teacher_rows), requests),
        "parse_schema_failures": teacher_parse_failures,
        "parse_schema_failure_rate": (
            _rate(teacher_parse_failures, len(paired_teacher_rows))
            if paired_teacher_rows
            else None
        ),
        "all_l4_calls_per_100_requests": 100.0 if teacher_by_request_id else None,
        "l4_calls_per_100_requests": (l4_calls / requests * 100.0) if requests else 0.0,
        "l4_call_reduction_rate": (
            1.0 - (l4_calls / requests)
            if teacher_by_request_id and requests
            else None
        ),
        "all_l4_tokens_per_request": (
            all_l4_tokens / requests
            if teacher_by_request_id and requests
            else None
        ),
        "l4_tokens_per_request": l4_tokens / requests if requests else 0.0,
        "l4_tokens_reduction_rate": (
            1.0 - (l4_tokens / all_l4_tokens)
            if all_l4_tokens > 0.0
            else None
        ),
        "all_l4_cost_usd_per_request": (
            all_l4_cost / requests
            if teacher_by_request_id and requests
            else None
        ),
        "l4_cost_usd_per_request": l4_cost / requests if requests else 0.0,
        "l4_cost_reduction_rate": (
            1.0 - (l4_cost / all_l4_cost)
            if all_l4_cost > 0.0
            else None
        ),
        "all_l4_latency_p50_ms": (
            _percentile(all_l4_latencies, 50)
            if teacher_by_request_id
            else None
        ),
        "all_l4_latency_p95_ms": (
            _percentile(all_l4_latencies, 95)
            if teacher_by_request_id
            else None
        ),
        "latency_p50_ms": _percentile(latencies, 50),
        "latency_p95_ms": _percentile(latencies, 95),
        "latency_p50_reduction_rate": _reduction_rate(
            _percentile(latencies, 50),
            _percentile(all_l4_latencies, 50) if all_l4_latencies else None,
        ),
        "latency_p95_reduction_rate": _reduction_rate(
            _percentile(latencies, 95),
            _percentile(all_l4_latencies, 95) if all_l4_latencies else None,
        ),
        "oos_precision": oos_precision,
        "oos_recall": oos_recall,
        "oos_f1": _f1(oos_precision, oos_recall),
    }


def _cascade_row(
    prediction_row: dict[str, Any],
    *,
    teacher_by_request_id: dict[str, dict[str, Any]],
    accepted: bool,
) -> dict[str, Any]:
    teacher_row = teacher_by_request_id.get(prediction_row["request_id"])
    teacher_frame = teacher_row.get("teacher_frame") if teacher_row else None
    teacher_parse_failed = bool(teacher_row and teacher_row.get("parse_failure"))
    if accepted:
        final_frame = prediction_row["predicted_frame"]
        l4_called = False
        l4_tokens = 0.0
        l4_cost_usd = 0.0
        l4_latency_ms = 0.0
    else:
        final_frame = teacher_frame
        l4_called = True
        l4_tokens = float(teacher_row.get("tokens", 0.0)) if teacher_row else 0.0
        l4_cost_usd = float(teacher_row.get("cost_usd", 0.0)) if teacher_row else 0.0
        l4_latency_ms = float(teacher_row.get("latency_ms", 0.0)) if teacher_row else 0.0
    final_intent = final_frame.get("intent") if isinstance(final_frame, dict) else None
    return {
        "request_id": prediction_row["request_id"],
        "final_frame": final_frame,
        "final_intent": final_intent,
        "final_correct": final_frame == prediction_row["gold_frame"],
        "all_l4_correct": (
            False
            if teacher_frame is None or teacher_parse_failed
            else teacher_frame == prediction_row["gold_frame"]
        ),
        "l4_called": l4_called,
        "l4_tokens": l4_tokens,
        "l4_cost_usd": l4_cost_usd,
        "latency_ms": float(prediction_row["latency_ms"]) + l4_latency_ms,
    }


def load_teacher_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_clinc150_l4_replay_oracle(path: Path) -> Clinc150L4ReplayOracle:
    return Clinc150L4ReplayOracle.from_rows(load_teacher_rows(path), source_path=path)


def teacher_details_path(out_dir: Path) -> Path:
    return out_dir / TEACHER_EVAL_DETAILS_JSONL_FILENAME


def load_clinc150_records(data_dir: Path, split: str) -> list[DataRecord]:
    return load_processed_records(data_dir, split=split)


def stream_clinc150_records(
    records: list[DataRecord],
    *,
    stream: str,
    max_requests: int,
) -> list[DataRecord]:
    return [
        item.record
        for item in select_stream(records, stream=stream, max_requests=max_requests)
    ]


def _all_l4_baseline_metrics(
    prediction_rows: list[dict[str, Any]],
    *,
    teacher_by_request_id: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    if not teacher_by_request_id:
        return None
    aligned_rows = [
        teacher_by_request_id.get(row["request_id"])
        for row in prediction_rows
    ]
    paired_rows = [row for row in aligned_rows if row is not None]
    requests = len(prediction_rows)
    correct = sum(
        1
        for prediction_row, teacher_row in zip(prediction_rows, aligned_rows, strict=True)
        if teacher_row is not None
        and not teacher_row.get("parse_failure")
        and teacher_row.get("teacher_frame") == prediction_row["gold_frame"]
    )
    parse_failures = sum(1 for row in paired_rows if row.get("parse_failure"))
    tokens = sum(float(row.get("tokens", 0.0)) for row in paired_rows)
    cost = sum(float(row.get("cost_usd", 0.0)) for row in paired_rows)
    latencies = [float(row.get("latency_ms", 0.0)) for row in paired_rows]
    return {
        "requests": requests,
        "paired_teacher_rows": len(paired_rows),
        "paired_teacher_coverage": _rate(len(paired_rows), requests),
        "accuracy": _rate(correct, requests),
        "parse_schema_failures": parse_failures,
        "parse_schema_failure_rate": _rate(parse_failures, len(paired_rows)),
        "tokens_per_request": tokens / requests if requests else 0.0,
        "cost_usd_per_request": cost / requests if requests else 0.0,
        "latency_p50_ms": _percentile(latencies, 50),
        "latency_p95_ms": _percentile(latencies, 95),
        "retry_recovered_rows": sum(1 for row in paired_rows if row.get("retry_recovered")),
        "unknown_usage_attempts": sum(
            int(row.get("unknown_usage_attempts", 0))
            for row in paired_rows
        ),
    }


def _cost_latency_table(threshold_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "threshold",
        "accepted_coverage",
        "accepted_precision",
        "final_cascade_accuracy",
        "accuracy_delta_vs_all_l4",
        "all_l4_calls_per_100_requests",
        "l4_calls_per_100_requests",
        "l4_call_reduction_rate",
        "all_l4_tokens_per_request",
        "l4_tokens_per_request",
        "l4_tokens_reduction_rate",
        "all_l4_cost_usd_per_request",
        "l4_cost_usd_per_request",
        "l4_cost_reduction_rate",
        "all_l4_latency_p50_ms",
        "latency_p50_ms",
        "latency_p50_reduction_rate",
        "all_l4_latency_p95_ms",
        "latency_p95_ms",
        "latency_p95_reduction_rate",
        "parse_schema_failure_rate",
    )
    return [
        {field: row.get(field) for field in fields}
        for row in threshold_rows
    ]


def _gold_intent(row: dict[str, Any]) -> str | None:
    gold_frame = row.get("gold_frame")
    return gold_frame.get("intent") if isinstance(gold_frame, dict) else None


def _teacher_intent(row: dict[str, Any]) -> str | None:
    teacher_frame = row.get("teacher_frame")
    return teacher_frame.get("intent") if isinstance(teacher_frame, dict) else None


def _oos_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return _oos_counts_from_frames(
        gold_intents=[_gold_intent(row) for row in rows],
        predicted_intents=[_teacher_intent(row) for row in rows],
    )


def _oos_counts_from_frames(
    *,
    gold_intents: list[str | None],
    predicted_intents: list[str | None],
) -> dict[str, int]:
    gold_positive = sum(1 for intent in gold_intents if intent == CLINC150_OOS_INTENT)
    predicted_positive = sum(
        1 for intent in predicted_intents if intent == CLINC150_OOS_INTENT
    )
    true_positive = sum(
        1
        for gold_intent, predicted_intent in zip(gold_intents, predicted_intents, strict=True)
        if gold_intent == CLINC150_OOS_INTENT and predicted_intent == CLINC150_OOS_INTENT
    )
    return {
        "gold_positive": gold_positive,
        "predicted_positive": predicted_positive,
        "true_positive": true_positive,
        "false_positive": predicted_positive - true_positive,
        "false_negative": gold_positive - true_positive,
    }


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _reduction_rate(new_value: float, baseline_value: float | None) -> float | None:
    if baseline_value is None or baseline_value <= 0.0:
        return None
    return 1.0 - (new_value / baseline_value)


def _percentile(values: list[float], percentile: float) -> float:
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
