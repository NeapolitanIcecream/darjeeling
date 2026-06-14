from __future__ import annotations

from statistics import quantiles
from typing import Any

from darjeeling.runtime.cost import replay_cost_model_from_settings
from darjeeling.targets.nlu.compiler.replay import LAYER_LATENCY_MS
from darjeeling.targets.nlu.patches import accepted_field_keys, frame_patch_from_layer_result
from darjeeling.targets.nlu.schemas import TeacherTrace
from darjeeling.targets.nlu.settings import Settings

WEAK_BEFORE_L3 = {"L0", "L1", "L2"}


def evaluate_l3_residual_value(
    traces: list[TeacherTrace],
    *,
    settings: Settings,
) -> dict[str, Any]:
    residual_traces = [
        trace
        for trace in traces
        if trace.teacher_frame is not None and _is_l2_residual(trace)
    ]
    accepted = []
    correct = 0
    wrong = 0
    latencies = []
    for trace in residual_traces:
        result = _l3_result(trace)
        if result is None:
            continue
        latencies.append(result.latency_ms)
        if not result.accepted or result.frame is None:
            continue
        accepted.append(trace)
        if result.frame == trace.teacher_frame:
            correct += 1
        else:
            wrong += 1
    residual_count = len(residual_traces)
    accepted_count = len(accepted)
    accepted_accuracy = correct / accepted_count if accepted_count else None
    wrong_accept_rate = wrong / accepted_count if accepted_count else 0.0
    coverage = accepted_count / residual_count if residual_count else 0.0
    p95_latency_ms = _p95(latencies)
    cost_model = replay_cost_model_from_settings(settings)
    l3_cost = cost_model.layer_cost_usd("L3", {}) * accepted_count
    avoided_l4_cost = cost_model.layer_cost_usd("L4", {}) * correct
    expected_cost_value_usd_per_100 = (
        (avoided_l4_cost - l3_cost) / residual_count * 100.0 if residual_count else 0.0
    )
    expected_latency_value_ms_per_request = (
        coverage * max(0.0, LAYER_LATENCY_MS["L4"] - max(p95_latency_ms, LAYER_LATENCY_MS["L3"]))
        if residual_count
        else 0.0
    )
    passes_gate = bool(
        residual_count
        and accepted_count
        and accepted_accuracy is not None
        and accepted_accuracy >= 1.0 - settings.promotion_accuracy_epsilon
        and wrong_accept_rate <= settings.l2_max_wrong_accept_rate
        and (expected_cost_value_usd_per_100 > 0.0 or expected_latency_value_ms_per_request > 0.0)
    )
    reason = "residual value gate passed" if passes_gate else _skip_reason(
        residual_count=residual_count,
        accepted_count=accepted_count,
        accepted_accuracy=accepted_accuracy,
        wrong_accept_rate=wrong_accept_rate,
        expected_cost_value_usd_per_100=expected_cost_value_usd_per_100,
        expected_latency_value_ms_per_request=expected_latency_value_ms_per_request,
        settings=settings,
    )
    return {
        "schema_version": "l3-residual-value-gate-v1",
        "residual_requests": residual_count,
        "accepted": accepted_count,
        "coverage": coverage,
        "correct_accepts": correct,
        "wrong_accepts": wrong,
        "accepted_accuracy": accepted_accuracy,
        "wrong_accept_rate": wrong_accept_rate,
        "p95_latency_ms": p95_latency_ms,
        "expected_cost_value_usd_per_100": expected_cost_value_usd_per_100,
        "expected_latency_value_ms_per_request": expected_latency_value_ms_per_request,
        "passes_gate": passes_gate,
        "reason": reason,
    }


def _is_l2_residual(trace: TeacherTrace) -> bool:
    return not any(
        result.layer in WEAK_BEFORE_L3
        and result.accepted
        and accepted_field_keys(frame_patch_from_layer_result(result))
        for result in trace.layer_results
    )


def _l3_result(trace: TeacherTrace):
    return next((result for result in trace.layer_results if result.layer == "L3"), None)


def _skip_reason(
    *,
    residual_count: int,
    accepted_count: int,
    accepted_accuracy: float | None,
    wrong_accept_rate: float,
    expected_cost_value_usd_per_100: float,
    expected_latency_value_ms_per_request: float,
    settings: Settings,
) -> str:
    if residual_count == 0:
        return "no L2 residual evidence"
    if accepted_count == 0:
        return "L3 accepted no residual requests"
    if accepted_accuracy is None or accepted_accuracy < 1.0 - settings.promotion_accuracy_epsilon:
        return "L3 residual accuracy failed gate"
    if wrong_accept_rate > settings.l2_max_wrong_accept_rate:
        return "L3 residual wrong-accept rate failed gate"
    if expected_cost_value_usd_per_100 <= 0.0 and expected_latency_value_ms_per_request <= 0.0:
        return "L3 residual cost/latency value is not positive"
    return "L3 residual gate failed"


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    return float(quantiles(values, n=100, method="inclusive")[94])
