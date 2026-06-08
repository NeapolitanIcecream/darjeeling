from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from darjeeling.layers.l3_local_slm import (
    L3LocalSLMLayer,
    L3PromptArtifact,
    LocalSLMBackend,
    LocalSLMConfig,
)
from darjeeling.layers.l4_cloud_llm import TaskSchema
from darjeeling.schemas import Frame, TeacherTrace, TraceRecord

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
    traces: list[TraceRecord],
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
        trace for trace in traces if trace.teacher_frame is not None or trace.gold_frame is not None
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
        expected = trace.teacher_frame or trace.gold_frame
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
