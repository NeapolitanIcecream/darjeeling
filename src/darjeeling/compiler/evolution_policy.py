from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

EvolutionRoundStatus = Literal[
    "completed",
    "failed",
    "timeout",
    "scope_violation",
    "validation_failed",
]


@dataclass(frozen=True)
class EvolutionRunPolicy:
    max_rounds: int = 1
    round_timeout_s: float | None = None
    patience_rounds: int = 0
    round_executor: str = "agent"

    def __post_init__(self) -> None:
        if self.max_rounds < 0:
            raise ValueError("max_rounds must be non-negative")
        if self.patience_rounds < 0:
            raise ValueError("patience_rounds must be non-negative")
        if self.round_timeout_s is not None and self.round_timeout_s <= 0:
            raise ValueError("round_timeout_s must be positive")
        if not self.round_executor:
            raise ValueError("round_executor must be non-empty")


@dataclass(frozen=True)
class EvolutionRoundResult:
    round_index: int
    status: str
    candidate_ref: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    improved: bool = False
    adoptable: bool = False
    stop_reason: str | None = None

    def __post_init__(self) -> None:
        if self.round_index < 1:
            raise ValueError("round_index must be one-based")
        if not self.status:
            raise ValueError("status must be non-empty")

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvolutionRunSummary:
    max_rounds: int
    rounds_completed: int
    stop_reason: str
    round_results: list[EvolutionRoundResult] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.max_rounds < 0:
            raise ValueError("max_rounds must be non-negative")
        if self.rounds_completed < 0:
            raise ValueError("rounds_completed must be non-negative")
        if self.rounds_completed > self.max_rounds:
            raise ValueError("rounds_completed cannot exceed max_rounds")
        if not self.stop_reason:
            raise ValueError("stop_reason must be non-empty")

    def to_payload(self) -> dict[str, Any]:
        return {
            "max_rounds": self.max_rounds,
            "rounds_completed": self.rounds_completed,
            "stop_reason": self.stop_reason,
            "round_results": [result.to_payload() for result in self.round_results],
        }


def evolution_run_summary_payload(
    *,
    policy: EvolutionRunPolicy,
    round_results: list[EvolutionRoundResult],
    stop_reason: str,
) -> dict[str, Any]:
    completed = sum(1 for result in round_results if result.status == "completed")
    return EvolutionRunSummary(
        max_rounds=policy.max_rounds,
        rounds_completed=completed,
        stop_reason=stop_reason,
        round_results=round_results,
    ).to_payload()
