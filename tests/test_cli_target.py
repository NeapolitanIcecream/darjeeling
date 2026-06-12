from pathlib import Path
from types import SimpleNamespace

import pytest

from darjeeling import cli
from darjeeling.targets.nlu.settings import load_settings


def test_execute_replay_run_writes_target_identity(tmp_path: Path, monkeypatch) -> None:
    captured = {}

    def fake_write_run_settings(path: Path, payload: dict) -> None:
        captured["settings_path"] = path
        captured["settings_payload"] = payload

    def fake_run_replay(**kwargs):
        captured["run_replay"] = kwargs
        return SimpleNamespace(
            requests=1,
            traces_path=tmp_path / "run" / "traces.jsonl",
            layer_counts={"L4": 1},
        )

    monkeypatch.setattr(cli, "require_live_or_cached_teacher", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "write_run_settings", fake_write_run_settings)
    monkeypatch.setattr(cli, "run_replay", fake_run_replay)

    summary = cli._execute_replay_run(
        stream="sequential",
        max_requests=1,
        compile_every=1,
        teacher="cache",
        run_dir=tmp_path / "run",
        data_dir=tmp_path / "data",
        target="nlu",
        settings=load_settings(),
    )

    assert summary.requests == 1
    assert captured["settings_path"] == tmp_path / "run" / "settings.json"
    assert captured["settings_payload"]["target_name"] == "nlu"
    assert captured["settings_payload"]["target_schema_version"] == "nlu-target-v1"
    assert captured["run_replay"]["run_dir"] == tmp_path / "run"


def test_execute_replay_run_rejects_unknown_target(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown target"):
        cli._execute_replay_run(
            stream="sequential",
            max_requests=1,
            compile_every=1,
            teacher="cache",
            run_dir=tmp_path / "run",
            data_dir=tmp_path / "data",
            target="missing",
            settings=load_settings(),
        )
