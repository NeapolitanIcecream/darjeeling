from __future__ import annotations

from collections.abc import Callable

from darjeeling.contracts import TargetSpec
from darjeeling.targets.nlu.target import NluTarget

_TARGETS: dict[str, Callable[[], TargetSpec]] = {
    "nlu": NluTarget,
}
_DEFAULT_TARGET = "nlu"


def available_targets() -> tuple[str, ...]:
    return tuple(sorted(_TARGETS))


def default_target_name() -> str:
    return _DEFAULT_TARGET


def get_target(name: str) -> TargetSpec:
    try:
        factory = _TARGETS[name]
    except KeyError as exc:
        known = ", ".join(available_targets()) or "<none>"
        raise ValueError(f"unknown target {name!r}; available targets: {known}") from exc
    return factory()
