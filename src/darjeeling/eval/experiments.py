from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from darjeeling.settings import Settings


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    default_stream: str = "zipf-heavy"
    default_max_requests: int = 3000
    default_compile_every: int = 500
    settings_overrides: dict[str, Any] = field(default_factory=dict)
    substreams: tuple[str, ...] = ()
    description: str = ""


EXPERIMENTS: dict[str, ExperimentSpec] = {
    "main-evolution": ExperimentSpec(
        name="main-evolution",
        description="Full cascade evolution curve on Zipf-heavy stream.",
    ),
    "direct-l4-optimization": ExperimentSpec(
        name="direct-l4-optimization",
        settings_overrides={"l4_proposal_mode": "live"},
        description="Enable direct L4 proposal calls for candidate config generation.",
    ),
    "l2-family": ExperimentSpec(
        name="l2-family",
        settings_overrides={"l4_proposal_mode": "live"},
        description="Run L2 candidate config proposal path and replay-selected guard.",
    ),
    "no-guard": ExperimentSpec(
        name="no-guard",
        settings_overrides={"l2_guard_mode": "always_accept"},
        description="Ablate the L2 guard by accepting every L2 prediction.",
    ),
    "no-l2": ExperimentSpec(
        name="no-l2",
        settings_overrides={"l2_enabled": False},
        description="Ablate L2 training, artifact promotion, and runtime routing.",
    ),
    "workload-locality": ExperimentSpec(
        name="workload-locality",
        substreams=("uniform", "zipf-mild", "zipf-heavy"),
        description="Run the same configuration across multiple locality streams.",
    ),
    "hard-buffer": ExperimentSpec(
        name="hard-buffer",
        settings_overrides={"hard_buffer_max_cases": 100},
        description="Exercise hard-buffer mining and replay pressure in the compiler loop.",
    ),
}


def experiment_spec(name: str) -> ExperimentSpec:
    try:
        return EXPERIMENTS[name]
    except KeyError as exc:
        available = ", ".join(sorted(EXPERIMENTS))
        raise ValueError(f"unknown experiment {name!r}; available: {available}") from exc


def apply_experiment_settings(settings: Settings, spec: ExperimentSpec) -> Settings:
    updated = settings.model_copy(deep=True)
    for field_name, value in spec.settings_overrides.items():
        if not hasattr(updated, field_name):
            raise ValueError(f"experiment {spec.name} overrides unknown setting: {field_name}")
        setattr(updated, field_name, value)
    return updated


def experiment_metadata(
    spec: ExperimentSpec,
    *,
    stream: str,
    max_requests: int,
    compile_every: int,
    teacher: str,
    data_dir: str,
) -> dict[str, Any]:
    return {
        "experiment": spec.name,
        "description": spec.description,
        "stream": stream,
        "max_requests": max_requests,
        "compile_every": compile_every,
        "teacher": teacher,
        "data_dir": data_dir,
        "settings_overrides": spec.settings_overrides,
        "substreams": list(spec.substreams),
    }
