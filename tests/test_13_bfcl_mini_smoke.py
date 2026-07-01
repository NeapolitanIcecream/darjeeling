from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from darjeeling.target_definition import TargetCheckOptions, check_target_definition


def _load_smoke_module() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "bfcl_mini_user_journey"
        / "run_smoke.py"
    )
    spec = importlib.util.spec_from_file_location("bfcl_mini_run_smoke", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bfcl_mini_smoke_target_contract_cases_pass(tmp_path: Path) -> None:
    smoke = _load_smoke_module()
    target_root = tmp_path / "target"
    smoke.write_mini_target(target_root, tmp_path / "mini_data")

    report = check_target_definition(
        target_root, TargetCheckOptions(require_reference=True)
    )

    assert report.status == "pass", report.failures
