from pathlib import Path

import pytest

from darjeeling.settings import load_settings


def test_load_settings_reads_yaml_and_env_overrides_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(
        "\n".join(
            [
                "openai_model: yaml-model",
                "l4_proposal_mode: live",
                "local_slm_mode: shadow",
                "l1_rust_crate_dir: native/custom_l1",
                "l4_input_usd_per_million: 2.0",
                "l2_enabled: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_MODEL", "env-model")

    settings = load_settings(settings_path)

    assert settings.settings_file == settings_path
    assert settings.openai_model == "env-model"
    assert settings.l4_proposal_mode == "live"
    assert settings.local_slm_mode == "shadow"
    assert settings.l1_rust_crate_dir == Path("native/custom_l1")
    assert settings.l4_input_usd_per_million == 2.0
    assert settings.l2_enabled is False


def test_load_settings_uses_default_settings_yaml_from_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "settings.yaml").write_text("l2_guard_mode: always_accept\n", encoding="utf-8")

    settings = load_settings()

    assert settings.settings_file == Path("settings.yaml")
    assert settings.l2_guard_mode == "always_accept"


def test_default_prompt_cache_retention_matches_live_provider_requirement() -> None:
    settings = load_settings()

    assert settings.prompt_cache_retention == "24h"


def test_load_settings_fails_for_explicit_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="settings file not found"):
        load_settings(tmp_path / "missing.yaml")
