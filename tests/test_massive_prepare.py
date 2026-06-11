import sys
from types import SimpleNamespace

from darjeeling.adapters.massive import prepare_massive_dataset


class _IntentFeature:
    def int2str(self, value):
        return "alarm_set" if value == 0 else str(value)


class _FakeDataset(list):
    features = {"intent": _IntentFeature()}


def test_prepare_massive_dataset_uses_noninteractive_trust_remote_code(
    tmp_path,
    monkeypatch,
) -> None:
    calls = []

    def load_dataset(path, locale, *, split, trust_remote_code):
        calls.append(
            {
                "path": path,
                "locale": locale,
                "split": split,
                "trust_remote_code": trust_remote_code,
            }
        )
        return _FakeDataset(
            [
                {
                    "utt": f"set alarm {split}",
                    "annot_utt": "set an alarm for [time : seven]",
                    "intent": 0,
                    "domain": "clock",
                }
            ]
        )

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=load_dataset),
    )

    result = prepare_massive_dataset("en-US", tmp_path)

    assert result == {"records": 3}
    assert {call["split"] for call in calls} == {"train", "validation", "test"}
    assert all(call["path"] == "AmazonScience/massive" for call in calls)
    assert all(call["locale"] == "en-US" for call in calls)
    assert all(call["trust_remote_code"] is True for call in calls)
    assert (tmp_path / "train.jsonl").exists()
    assert (tmp_path / "records.parquet").exists()
