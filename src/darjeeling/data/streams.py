from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from random import Random

from darjeeling.data.frames import normalize_utterance
from darjeeling.data.records import DataRecord


@dataclass(frozen=True)
class StreamItem:
    index: int
    record: DataRecord


def build_uniform_stream(
    records: list[DataRecord],
    max_requests: int,
    seed: int = 17,
) -> list[StreamItem]:
    rng = Random(seed)
    return [StreamItem(index=i, record=rng.choice(records)) for i in range(max_requests)]


def build_zipf_stream(
    records: list[DataRecord],
    max_requests: int,
    exponent: float,
    seed: int = 17,
) -> list[StreamItem]:
    rng = Random(seed)
    groups: dict[str, list[DataRecord]] = defaultdict(list)
    for record in records:
        groups[_record_workload_group_key(record)].append(record)

    ordered_groups = sorted(groups.values(), key=len, reverse=True)
    weights = [1.0 / ((rank + 1) ** exponent) for rank in range(len(ordered_groups))]
    return [
        StreamItem(
            index=i,
            record=rng.choice(rng.choices(ordered_groups, weights=weights, k=1)[0]),
        )
        for i in range(max_requests)
    ]


def _record_workload_group_key(record: DataRecord) -> str:
    if record.workload_group_key:
        return record.workload_group_key
    if record.template:
        return f"{record.gold_frame.intent}:{record.template}"
    return f"{record.gold_frame.intent}:{normalize_utterance(record.utterance)}"
