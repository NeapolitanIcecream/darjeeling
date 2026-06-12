from pathlib import Path
from types import SimpleNamespace

import pytest

from darjeeling import cli
from darjeeling.targets import registry


class _FakeTarget:
    name = "fake"
    schema_version = "fake-v1"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def load_settings(self, *, settings_path: Path | None = None):
        self.calls.append({"method": "load_settings", "settings_path": settings_path})
        return {"settings": "loaded"}

    def run_replay(self, **kwargs):
        self.calls.append({"method": "run_replay", **kwargs})
        return SimpleNamespace(
            requests=1,
            traces_path=kwargs["run_dir"] / "traces.jsonl",
            layer_counts={"L4": 1},
        )


def test_core_cli_dispatches_run_to_selected_target(tmp_path: Path, monkeypatch) -> None:
    fake_target = _FakeTarget()
    monkeypatch.setitem(registry._TARGETS, "fake", lambda: fake_target)

    summary = cli._execute_replay_run(
        stream="sequential",
        max_requests=1,
        compile_every=1,
        teacher="cache",
        run_dir=tmp_path / "run",
        data_dir=tmp_path / "data",
        target="fake",
        settings={"settings": "loaded"},
    )

    assert summary.requests == 1
    assert fake_target.calls == [
        {
            "method": "run_replay",
            "stream": "sequential",
            "max_requests": 1,
            "compile_every": 1,
            "teacher": "cache",
            "run_dir": tmp_path / "run",
            "data_dir": tmp_path / "data",
            "settings": {"settings": "loaded"},
        }
    ]


def test_core_cli_rejects_unknown_target(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown target"):
        cli._execute_replay_run(
            stream="sequential",
            max_requests=1,
            compile_every=1,
            teacher="cache",
            run_dir=tmp_path / "run",
            data_dir=tmp_path / "data",
            target="missing",
            settings={},
        )


def test_project_scripts_keep_core_and_nlu_entrypoints_separate() -> None:
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert 'edge-mvp = "darjeeling.cli:app"' in pyproject
    assert 'edge-mvp-nlu = "darjeeling.targets.nlu.main_cli:app"' in pyproject
