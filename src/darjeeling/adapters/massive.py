from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from darjeeling.data.frames import frame_from_annotated_utterance, normalized_template
from darjeeling.data.records import DataRecord


def _intent_name(dataset: Any, value: Any) -> str:
    feature = dataset.features.get("intent")
    if hasattr(feature, "int2str"):
        return feature.int2str(value)
    return str(value)


def _iter_split_records(locale: str, split: str) -> list[DataRecord]:
    from datasets import load_dataset

    dataset = load_dataset(
        "AmazonScience/massive",
        locale,
        split=split,
        trust_remote_code=True,
    )
    records: list[DataRecord] = []
    for idx, row in enumerate(dataset):
        utterance = row.get("utt") or row.get("utterance") or ""
        annotated = row.get("annot_utt") or row.get("annotated_utterance") or utterance
        template = normalized_template(annotated)
        intent = _intent_name(dataset, row["intent"])
        records.append(
            DataRecord(
                request_id=f"{split}-{idx}",
                locale=locale,
                split=split,
                utterance=utterance,
                annotated_utterance=annotated,
                template=template,
                workload_group_key=f"{intent}:{template}",
                gold_frame=frame_from_annotated_utterance(intent, annotated),
                metadata={"domain": row.get("domain")},
            )
        )
    return records


def prepare_massive_dataset(locale: str, out_dir: Path) -> dict[str, int]:
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)
    splits = ["train", "validation", "test"]
    all_records: list[DataRecord] = []

    for split in splits:
        split_records = _iter_split_records(locale, split)
        all_records.extend(split_records)
        split_jsonl = out_dir / f"{split}.jsonl"
        split_jsonl.write_text(
            "".join(record.model_dump_json() + "\n" for record in split_records),
            encoding="utf-8",
        )

    rows = [record.model_dump(mode="json") for record in all_records]
    pd.DataFrame(rows).to_parquet(out_dir / "records.parquet", index=False)
    (out_dir / "manifest.json").write_text(
        json.dumps({"locale": locale, "records": len(all_records)}, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"records": len(all_records)}
