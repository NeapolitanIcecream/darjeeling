import json
from pathlib import Path

from darjeeling.targets.nlu.adapters.clinc150 import (
    CLINC150_DATA_FULL_SHA256,
    CLINC150_OOS_INTENT,
    prepare_clinc150_dataset,
)


def test_clinc150_prepare_preserves_splits_and_maps_oos(tmp_path: Path) -> None:
    source = tmp_path / "data_full.json"
    source.write_text(
        json.dumps(
            {
                "train": [["train alpha", "alpha_intent"]],
                "oos_train": [["train oos", "oos"]],
                "val": [["val alpha", "alpha_intent"]],
                "oos_val": [["val oos", "oos"]],
                "test": [["test alpha", "alpha_intent"]],
                "oos_test": [["test oos", "oos"]],
            }
        ),
        encoding="utf-8",
    )
    expected_sha256 = _sha256(source)

    result = prepare_clinc150_dataset(
        out_dir=tmp_path / "processed",
        source=source,
        expected_sha256=expected_sha256,
    )

    assert result["records"] == 6
    assert result["split_counts"] == {"train": 2, "validation": 2, "test": 2}
    train_rows = [
        json.loads(line)
        for line in (tmp_path / "processed" / "train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert train_rows[0]["gold_frame"] == {
        "intent": "alpha_intent",
        "slots": {},
        "is_abstain": False,
    }
    assert train_rows[1]["gold_frame"] == {
        "intent": CLINC150_OOS_INTENT,
        "slots": {},
        "is_abstain": True,
    }
    manifest = json.loads((tmp_path / "processed" / "manifest.json").read_text())
    assert manifest["schema_version"] == "nlu-clinc150-processed-v1"
    assert manifest["oos_mapping"]["intent"] == CLINC150_OOS_INTENT
    assert manifest["source_sha256"] == expected_sha256


def test_clinc150_prepare_rejects_checksum_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "data_full.json"
    source.write_text("{}", encoding="utf-8")

    try:
        prepare_clinc150_dataset(
            out_dir=tmp_path / "processed",
            source=source,
            expected_sha256=CLINC150_DATA_FULL_SHA256,
        )
    except ValueError as exc:
        assert "checksum mismatch" in str(exc)
    else:
        raise AssertionError("expected checksum mismatch")


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()
