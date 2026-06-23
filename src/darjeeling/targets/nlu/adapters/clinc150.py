from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any

from darjeeling.targets.nlu.data import DataRecord, normalize_utterance
from darjeeling.targets.nlu.schemas import Frame

CLINC150_SOURCE_COMMIT = "828f8093932c8fe6ca7936c3d2e52903b1c523de"
CLINC150_DATA_FULL_URL = (
    "https://raw.githubusercontent.com/clinc/oos-eval/"
    f"{CLINC150_SOURCE_COMMIT}/data/data_full.json"
)
CLINC150_DATA_FULL_SHA256 = (
    "36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0"
)
CLINC150_DATASET_NAME = "clinc150"
CLINC150_VARIANT = "data_full"
CLINC150_OOS_SOURCE_LABEL = "oos"
CLINC150_OOS_INTENT = "out_of_scope"
CLINC150_LICENSE_NOTE = (
    "The pinned clinc/oos-eval repository LICENSE is Creative Commons "
    "Attribution 3.0 Unported. The current UCI CLINC150 metadata page lists "
    "Creative Commons Attribution 4.0 International; this manifest records the "
    "pinned GitHub source used for these processed files."
)

_SPLIT_SOURCE_KEYS = {
    "train": ("train", "oos_train"),
    "validation": ("val", "oos_val"),
    "test": ("test", "oos_test"),
}


def prepare_clinc150_dataset(
    *,
    out_dir: Path,
    source: str | Path | None = None,
    expected_sha256: str | None = CLINC150_DATA_FULL_SHA256,
) -> dict[str, Any]:
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)
    source_ref = str(source or CLINC150_DATA_FULL_URL)
    source_bytes = _read_source_bytes(source_ref)
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if expected_sha256 is not None and source_sha256 != expected_sha256:
        raise ValueError(
            "CLINC150 data_full checksum mismatch: "
            f"expected {expected_sha256}, got {source_sha256}"
        )

    payload = json.loads(source_bytes.decode("utf-8"))
    split_records: dict[str, list[DataRecord]] = {
        split: _records_for_split(payload, split)
        for split in _SPLIT_SOURCE_KEYS
    }
    for split, records in split_records.items():
        (out_dir / f"{split}.jsonl").write_text(
            "".join(record.model_dump_json() + "\n" for record in records),
            encoding="utf-8",
        )

    all_records = [
        record
        for split in ("train", "validation", "test")
        for record in split_records[split]
    ]
    rows = [_parquet_row(record) for record in all_records]
    pd.DataFrame(rows).to_parquet(out_dir / "records.parquet", index=False)

    manifest = _manifest(
        source_ref=source_ref,
        source_sha256=source_sha256,
        split_records=split_records,
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "records": len(all_records),
        "source_sha256": source_sha256,
        "split_counts": manifest["split_counts"],
    }


def _read_source_bytes(source_ref: str) -> bytes:
    source_path = Path(source_ref)
    if source_path.exists():
        return source_path.read_bytes()
    with urllib.request.urlopen(source_ref, timeout=60) as response:
        return response.read()


def _parquet_row(record: DataRecord) -> dict[str, Any]:
    row = record.model_dump(mode="json")
    row["gold_frame"] = json.dumps(row["gold_frame"], sort_keys=True)
    row["metadata"] = json.dumps(row["metadata"], sort_keys=True)
    return row


def _records_for_split(payload: dict[str, Any], split: str) -> list[DataRecord]:
    records: list[DataRecord] = []
    source_keys = _SPLIT_SOURCE_KEYS[split]
    for source_key in source_keys:
        examples = payload.get(source_key)
        if not isinstance(examples, list):
            raise ValueError(f"CLINC150 source missing list split {source_key!r}")
        for item in examples:
            records.append(
                _record_from_source_item(
                    item,
                    split=split,
                    source_key=source_key,
                    index=len(records),
                )
            )
    return records


def _record_from_source_item(
    item: Any,
    *,
    split: str,
    source_key: str,
    index: int,
) -> DataRecord:
    if (
        not isinstance(item, list | tuple)
        or len(item) != 2
        or not isinstance(item[0], str)
        or not isinstance(item[1], str)
    ):
        raise ValueError(f"invalid CLINC150 source item in {source_key}: {item!r}")
    utterance, source_label = item
    intent = _mapped_intent(source_label)
    is_oos = source_label == CLINC150_OOS_SOURCE_LABEL
    return DataRecord(
        request_id=f"{split}-{index}",
        locale="en-US",
        split=split,
        utterance=utterance,
        annotated_utterance=utterance,
        template=normalize_utterance(utterance),
        workload_group_key=intent,
        gold_frame=Frame(intent=intent, slots={}, is_abstain=is_oos),
        metadata={
            "dataset": CLINC150_DATASET_NAME,
            "variant": CLINC150_VARIANT,
            "source_key": source_key,
            "source_label": source_label,
        },
    )


def _mapped_intent(source_label: str) -> str:
    if source_label == CLINC150_OOS_SOURCE_LABEL:
        return CLINC150_OOS_INTENT
    return source_label


def _manifest(
    *,
    source_ref: str,
    source_sha256: str,
    split_records: dict[str, list[DataRecord]],
) -> dict[str, Any]:
    split_counts = {split: len(records) for split, records in split_records.items()}
    in_scope_intents = sorted(
        {
            record.gold_frame.intent
            for records in split_records.values()
            for record in records
            if record.gold_frame.intent != CLINC150_OOS_INTENT
        }
    )
    oos_counts = {
        split: sum(1 for record in records if record.gold_frame.intent == CLINC150_OOS_INTENT)
        for split, records in split_records.items()
    }
    return {
        "schema_version": "nlu-clinc150-processed-v1",
        "dataset": CLINC150_DATASET_NAME,
        "variant": CLINC150_VARIANT,
        "source_url": CLINC150_DATA_FULL_URL,
        "source_ref": source_ref,
        "source_commit": CLINC150_SOURCE_COMMIT,
        "source_sha256": source_sha256,
        "license_note": CLINC150_LICENSE_NOTE,
        "split_source_keys": {
            split: list(source_keys)
            for split, source_keys in _SPLIT_SOURCE_KEYS.items()
        },
        "split_counts": split_counts,
        "oos_counts": oos_counts,
        "in_scope_intent_count": len(in_scope_intents),
        "intent_count_including_oos": len(in_scope_intents) + 1,
        "oos_mapping": {
            "source_label": CLINC150_OOS_SOURCE_LABEL,
            "intent": CLINC150_OOS_INTENT,
            "is_abstain": True,
        },
        "frame_shape": {"slots": {}, "slot_count": 0},
    }
