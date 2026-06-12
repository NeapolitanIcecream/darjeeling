from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectiveWeights:
    frame_exact_match: float = 100.0
    wrong_accept_rate: float = -200.0
    cost_usd_per_100_requests: float = -1.0
    p95_latency_ms: float = -0.01
    artifact_complexity: float = -0.001


@dataclass(frozen=True)
class ObjectiveMetrics:
    frame_exact_match: float
    wrong_accept_rate: float
    cost_usd_per_100_requests: float
    p95_latency_ms: float
    artifact_complexity: float = 0.0


def objective_score(
    metrics: ObjectiveMetrics,
    weights: ObjectiveWeights | None = None,
) -> float:
    weights = weights or ObjectiveWeights()
    return (
        weights.frame_exact_match * metrics.frame_exact_match
        + weights.wrong_accept_rate * metrics.wrong_accept_rate
        + weights.cost_usd_per_100_requests * metrics.cost_usd_per_100_requests
        + weights.p95_latency_ms * metrics.p95_latency_ms
        + weights.artifact_complexity * metrics.artifact_complexity
    )
