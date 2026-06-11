from pathlib import Path

from darjeeling.data.records import DataRecord
from darjeeling.layers.l4_cloud_llm import TaskSchema
from darjeeling.runtime.replay import load_processed_records
from darjeeling.settings import DEFAULT_PROCESSED_DATA_DIR, load_settings


def test_core_defaults_are_dataset_independent() -> None:
    settings = load_settings()
    schema = TaskSchema(intent_names=["intent_alpha"], slot_names=["slot_alpha"])

    assert schema.schema_version == "task-schema-v1"
    assert DEFAULT_PROCESSED_DATA_DIR == Path("data/processed/default")
    assert settings.l1_rust_crate_dir == Path("native/l1_empty_programbank")


def test_generic_data_record_is_not_owned_by_massive_adapter() -> None:
    assert DataRecord.__module__ == "darjeeling.data.records"


def test_processed_data_loader_error_is_dataset_independent(tmp_path: Path) -> None:
    missing_data_dir = tmp_path / "data"

    try:
        load_processed_records(missing_data_dir)
    except FileNotFoundError as exc:
        message = str(exc)
    else:
        raise AssertionError("load_processed_records should fail for a missing split")

    assert "processed data split not found" in message
    assert "MASSIVE" not in message


def test_core_runtime_replay_does_not_import_massive_adapter() -> None:
    source = Path("src/darjeeling/runtime/replay.py").read_text(encoding="utf-8")

    assert "darjeeling.adapters.massive" not in source
    assert "darjeeling.data.massive" not in source


def test_core_source_does_not_embed_bundled_dataset_or_demo_defaults() -> None:
    forbidden_terms = (
        "MASSIVE",
        "massive_en_us",
        "set an alarm for seven tomorrow morning",
        "what is the weather in san francisco",
        'Path("native/l1_programbank")',
    )

    for path in Path("src/darjeeling").rglob("*.py"):
        if "adapters" in path.parts:
            continue
        source = path.read_text(encoding="utf-8")
        for term in forbidden_terms:
            assert term not in source, f"{path} contains target-specific default {term!r}"


def test_massive_adapter_has_separate_cli_entrypoint() -> None:
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert 'edge-mvp-massive = "darjeeling.adapters.massive_cli:app"' in pyproject
