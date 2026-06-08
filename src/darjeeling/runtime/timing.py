from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from time import perf_counter


@contextmanager
def elapsed_ms() -> Iterator[Callable[[], float]]:
    start = perf_counter()

    def current() -> float:
        return (perf_counter() - start) * 1000.0

    yield current
