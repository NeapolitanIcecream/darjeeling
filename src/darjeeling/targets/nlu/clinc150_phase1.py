from __future__ import annotations

import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
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
from darjeeling.targets.nlu.layers.l4_cloud_llm import (
    MissingTeacherError,
    TeacherCallResult,
    _extract_chat_content,
    _extract_usage,
    create_chat_completion_with_retry,
)
from darjeeling.targets.nlu.replay import load_processed_records, select_stream
from darjeeling.targets.nlu.schemas import Frame, TaskSchema
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
    _teacher_eval_error_row,
    _teacher_eval_row,
    _teacher_eval_summary,
    write_teacher_live_eval_artifacts,
)

DEFAULT_CLINC150_PROMPTS = (CLINC150_PROMPT_V1, CLINC150_PROMPT_V2_LABEL_CARDS)
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


@dataclass(frozen=True)
class Clinc150TeacherEvalArtifact:
    prompt_version: str
    artifacts: TeacherLiveEvalArtifactResult
    clinc_metrics_path: Path
    clinc_metrics: dict[str, Any]


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
        response = create_chat_completion_with_retry(
            self.client(),
            self.settings,
            response_check=_extract_chat_content,
            model=self.settings.openai_model,
            messages=messages,
            response_format={"type": "json_object"},
            max_completion_tokens=self.settings.teacher_max_tokens,
            prompt_cache_key=f"darjeeling:{prompt_version}:{task_schema.schema_version}",
            prompt_cache_retention=self.settings.prompt_cache_retention,
            timeout=self.settings.openai_timeout_s,
        )
        raw_response = _extract_chat_content(response)
        return TeacherCallResult(
            frame=parse_clinc150_teacher_frame(raw_response, task_schema=task_schema),
            raw_response=raw_response,
            usage=_extract_usage(response),
            model=getattr(response, "model", self.settings.openai_model),
            context_hash="",
            prompt_cache_key=f"darjeeling:{prompt_version}:{task_schema.schema_version}",
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


def _label_description(intent: str) -> str:
    if intent == CLINC150_OOS_INTENT:
        return "unsupported or out-of-scope request"
    return intent.replace("_", " ")


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
    min_overall_accuracy: float = 0.95,
    min_in_scope_accuracy: float = 0.97,
    max_parse_failure_rate: float = 0.005,
) -> Clinc150TeacherEvalArtifact:
    settings_for_prompt = settings.model_copy(update={"teacher_prompt_version": prompt_version})
    rows = _evaluate_clinc150_teacher_rows(
        records=records,
        task_schema=task_schema,
        settings=settings_for_prompt,
        prompt_version=prompt_version,
        label_cards=label_cards,
        max_workers=max_workers,
    )
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
) -> list[dict[str, Any]]:
    indexed = list(enumerate(records, start=1))
    if max_workers <= 1:
        return [
            _clinc150_teacher_row(
                index=index,
                record=record,
                task_schema=task_schema,
                settings=settings,
                prompt_version=prompt_version,
                label_cards=label_cards,
            )
            for index, record in indexed
        ]

    rows_by_index: dict[int, dict[str, Any]] = {}
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
            for index, record in indexed
        }
        for future in as_completed(futures):
            rows_by_index[futures[future]] = future.result()
    return [rows_by_index[index] for index, _record in indexed]


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
            cost_usd=cost_model.layer_cost_usd("L4", call_result.usage),
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
    return {
        "requests": requests,
        "parsed_requests": len(parsed),
        "in_scope_requests": len(in_scope),
        "oos_requests": len(oos),
        "overall_accuracy": overall_accuracy,
        "in_scope_accuracy": in_scope_accuracy,
        "parse_schema_failure_rate": parse_failure_rate,
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
) -> L2StudentBundle:
    return train_l2_student(
        examples,
        L2StudentConfig(
            accept_threshold=accept_threshold,
            random_state=random_state,
            min_examples=4,
            slot_model_family="none",
            frame_source="student",
            max_iter=1000,
        ),
    )


def evaluate_clinc150_l2(
    *,
    bundle: L2StudentBundle,
    records: list[DataRecord],
    teacher_rows: list[dict[str, Any]] | None = None,
    thresholds: tuple[float, ...] = DEFAULT_CLINC150_THRESHOLDS,
) -> dict[str, Any]:
    teacher_by_request_id = {
        row["request_id"]: row
        for row in teacher_rows or []
    }
    prediction_rows = [
        _l2_prediction_row(bundle=bundle, record=record)
        for record in records
    ]
    threshold_rows = [
        _l2_threshold_metrics(
            prediction_rows,
            teacher_by_request_id=teacher_by_request_id,
            threshold=threshold,
        )
        for threshold in thresholds
    ]
    return {
        "schema_version": "clinc150-l2-eval-v1",
        "requests": len(prediction_rows),
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
        "thresholds": threshold_rows,
        "selected_threshold": select_l2_threshold(threshold_rows),
    }


def select_l2_threshold(
    threshold_rows: list[dict[str, Any]],
    *,
    min_precision: float = 0.99,
    max_oos_false_accept_rate: float = 0.02,
) -> dict[str, Any] | None:
    candidates = [
        row
        for row in threshold_rows
        if (row["accepted_precision"] is not None and row["accepted_precision"] >= min_precision)
        and row["lower_layer_oos_false_accept_rate"] <= max_oos_false_accept_rate
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: (row["accepted_coverage"], -row["threshold"]))


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
        "latency_ms": latency_ms,
    }


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
        "l4_calls_per_100_requests": (l4_calls / requests * 100.0) if requests else 0.0,
        "l4_tokens_per_request": l4_tokens / requests if requests else 0.0,
        "l4_cost_usd_per_request": l4_cost / requests if requests else 0.0,
        "latency_p50_ms": _percentile(latencies, 50),
        "latency_p95_ms": _percentile(latencies, 95),
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
