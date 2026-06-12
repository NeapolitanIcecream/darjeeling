from pathlib import Path

from darjeeling.data.records import DataRecord
from darjeeling.layers.l4_cloud_llm import TaskSchema
from darjeeling.runtime.replay import load_processed_records
from darjeeling.schemas import Frame
from darjeeling.settings import DEFAULT_PROCESSED_DATA_DIR, load_settings


def test_core_defaults_are_dataset_independent() -> None:
    settings = load_settings()
    schema = TaskSchema(intent_names=["intent_alpha"], slot_names=["slot_alpha"])

    assert schema.schema_version == "task-schema-v1"
    assert DEFAULT_PROCESSED_DATA_DIR == Path("data/processed/default")
    assert settings.l1_rust_crate_dir == Path("native/l1_empty_programbank")


def test_generic_data_record_is_not_owned_by_massive_adapter() -> None:
    assert DataRecord.__module__ == "darjeeling.data.records"
    record = DataRecord(
        request_id="r1",
        utterance="alpha request",
        gold_frame=Frame(intent="intent_alpha"),
    )
    assert record.workload_group_key is None
    assert record.annotated_utterance is None
    assert record.template is None


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
        "massive",
        "massive_en_us",
        "alpha request for seven tomorrow morning",
        "what is the gamma in san francisco",
        '"time"',
        '"date"',
        '"location"',
        "place_name",
        "artist_name",
        "beta_descriptor",
        "play_beta",
        "play queen",
        "gold-seven",
        "gold-tomorrow",
        "7:00",
        'Path("native/l1_programbank")',
        "alarm_set",
        "weather_query",
        "qa_factoid",
        "lists_query",
        "list_name",
        "iot_",
        "programs/alarm",
    )

    for path in Path("src/darjeeling").rglob("*.py"):
        if "adapters" in path.parts:
            continue
        source = path.read_text(encoding="utf-8")
        for term in forbidden_terms:
            assert term not in source, f"{path} contains target-specific default {term!r}"


def test_shared_core_tests_use_neutral_fixtures() -> None:
    adapter_or_demo_tests = {
        Path("tests/test_massive_prepare.py"),
        Path("tests/test_target_boundary.py"),
    }
    forbidden_terms = (
        "MASSIVE",
        "massive",
        "massive_en_us",
        "AmazonScience/massive",
        "native/l1_programbank",
        "alarm_set",
        "music_play",
        "weather_query",
        "qa_factoid",
        "lists_query",
        "list_name",
        "iot_",
        "programs/alarm",
        "set an alarm",
        "set alarm",
        "alarm at",
        '"time"',
        '"date"',
        '"location"',
        "B-time",
        "I-time",
        "place_name",
        "artist_name",
        "beta_descriptor",
        "play_beta",
        "play jazz",
        "play music",
        "play smooth jazz",
        "start smooth jazz",
        "play queen",
        "gold-seven",
        "gold-tomorrow",
        "7:00",
        "alpha request for seven",
        "alpha at eight",
        "alpha at nine",
        "set morning alpha",
        "set evening alpha",
        "gamma tomorrow",
        "what is the gamma",
        "tomorrow",
        '"paris"',
        " in paris",
        "jazz",
        "queen",
        "calendar",
        "joke",
        "radio",
        "podcast",
        "email",
        "emails",
        "kitchen",
        "bathroom",
        "carrie",
        "robert",
    )

    for path in Path("tests").glob("test_*.py"):
        if path in adapter_or_demo_tests:
            continue
        source = path.read_text(encoding="utf-8")
        for term in forbidden_terms:
            assert term not in source, f"{path} contains shared-test target fixture {term!r}"


def test_tracked_l1_native_fixtures_are_dataset_independent() -> None:
    assert not Path("native/l1_programbank").exists()

    forbidden_terms = (
        "alarm_set",
        "weather_query",
        "qa_factoid",
        "programs/alarm",
        "set an alarm",
        "weather in",
        "play some jazz",
        "carrie",
    )
    for path in Path("tests/fixtures/l1_neutral_programbank").rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in {".rs", ".toml", ".lock"}:
            continue
        source = path.read_text(encoding="utf-8")
        for term in forbidden_terms:
            assert term not in source.lower(), f"{path} contains target fixture {term!r}"


def test_current_architecture_doc_uses_dataset_independent_gold_label_terms() -> None:
    source = Path("docs/design/01_architecture.md").read_text(encoding="utf-8")

    assert "MASSIVE gold" not in source


def test_massive_adapter_has_separate_cli_entrypoint() -> None:
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert 'edge-mvp-massive = "darjeeling.adapters.massive_cli:app"' in pyproject
