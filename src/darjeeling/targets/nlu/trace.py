from __future__ import annotations

from pathlib import Path

from darjeeling.targets.nlu.schemas import TraceRecord


class TraceWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, trace: TraceRecord) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(trace.model_dump_json() + "\n")


def read_traces(path: Path) -> list[TraceRecord]:
    if not path.exists():
        return []
    return [
        TraceRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
