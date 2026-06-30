from __future__ import annotations

from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

import pytest
from conftest import PrefixBroker

from darjeeling.errors import SnapshotBuildError, ValidationError
from darjeeling.model import (
    AgentViewOptions,
    ConsumedRowsManifest,
    DataConfig,
    ReferenceBudget,
    ReferenceContext,
    ReferenceQualificationOptions,
    ReferenceResponse,
    SnapshotOptions,
    SourceRecord,
    TelemetryDataSource,
)
from darjeeling.reference_config import build_reference_broker_from_config
from darjeeling.snapshot_reference import (
    build_snapshot,
    collect_source_records,
    export_train_view_for_agent,
    load_snapshot_records,
    load_snapshot_view,
    qualify_reference_baseline,
    reference_missing_outputs,
    validate_source_records,
)
from darjeeling.target_definition import load_checked_target
from darjeeling.util import write_json


def test_telemetry_scope_and_cutoff_are_enforced(target_dir: Path, tmp_path: Path, now) -> None:
    definition, contract, _ = load_checked_target(target_dir)
    records_path = tmp_path / "telemetry.json"
    write_json(records_path, [])
    mismatched = TelemetryDataSource(
        source_id="t1",
        target_name="other",
        contract_hash=definition.contract_hash,
        cutoff_time=now,
        records_uri=str(records_path),
        default_split_eligibility=["train"],
        per_record_split_eligibility_uri=None,
        included_sources=[],
        provenance_digest="x",
    )
    with pytest.raises(Exception, match="does not match"):
        collect_source_records(definition, definition.data_config, mismatched, now)
    missing_path_config = DataConfig(sources=[{"name": "missing", "path": "missing.json"}])
    with pytest.raises(SnapshotBuildError, match="does not match"):
        collect_source_records(definition, missing_path_config, mismatched, now)
    late = TelemetryDataSource(
        **{
            **asdict(mismatched),
            "target_name": definition.name,
            "cutoff_time": now + timedelta(seconds=1),
        }
    )
    with pytest.raises(Exception, match="later than snapshot cutoff"):
        collect_source_records(definition, definition.data_config, late, now)


def test_telemetry_requires_complete_per_record_split_eligibility(
    target_dir: Path,
    tmp_path: Path,
    now,
) -> None:
    definition, contract, _ = load_checked_target(target_dir)
    records_path = tmp_path / "telemetry.json"
    write_json(
        records_path,
        [
            {
                "evidence_id": "ev1",
                "created_at": now.isoformat(),
                "source_event_at": now.isoformat(),
                "input_payload": {"text": "a:telemetry"},
                "reference_output_payload": {"label": "a"},
                "reference_source": "user_feedback",
                "approved_for": ["train"],
                "release_id": "rel",
            }
        ],
    )
    source = TelemetryDataSource(
        source_id="t1",
        target_name=definition.name,
        contract_hash=definition.contract_hash,
        cutoff_time=now,
        records_uri=str(records_path),
        default_split_eligibility=["train"],
        per_record_split_eligibility_uri=None,
        included_sources=["user_feedback"],
        provenance_digest="x",
    )
    records = collect_source_records(definition, definition.data_config, source, now)
    telemetry_record = next(record for record in records if record.record_id == "ev1")
    assert telemetry_record.split_eligibility == ["train"]

    broadened_default = TelemetryDataSource(
        **{**asdict(source), "default_split_eligibility": ["train", "validation_candidate"]}
    )
    with pytest.raises(SnapshotBuildError, match="does not match row permissions"):
        collect_source_records(definition, definition.data_config, broadened_default, now)

    invalid_default = TelemetryDataSource(
        **{**asdict(source), "default_split_eligibility": []}
    )
    with pytest.raises(SnapshotBuildError, match="invalid split eligibility"):
        collect_source_records(definition, definition.data_config, invalid_default, now)

    per_record_path = tmp_path / "split.json"
    write_json(per_record_path, {"other": ["train"]})
    incomplete = TelemetryDataSource(
        **{**asdict(source), "per_record_split_eligibility_uri": str(per_record_path)}
    )
    with pytest.raises(SnapshotBuildError, match="exactly match"):
        collect_source_records(definition, definition.data_config, incomplete, now)

    write_json(per_record_path, {"ev1": ["validation_candidate"]})
    changed_permission = TelemetryDataSource(
        **{**asdict(source), "per_record_split_eligibility_uri": str(per_record_path)}
    )
    with pytest.raises(SnapshotBuildError, match="does not match row permissions"):
        collect_source_records(definition, definition.data_config, changed_permission, now)

    future_records_path = tmp_path / "future-telemetry.json"
    future_row = dict(__import__("json").loads(records_path.read_text())[0])
    future_row["source_event_at"] = (now + timedelta(seconds=1)).isoformat()
    write_json(future_records_path, [future_row])
    write_json(per_record_path, {"ev1": ["train"]})
    future_source = TelemetryDataSource(
        **{
            **asdict(source),
            "records_uri": str(future_records_path),
            "per_record_split_eligibility_uri": str(per_record_path),
        }
    )
    with pytest.raises(SnapshotBuildError, match="source event"):
        collect_source_records(definition, definition.data_config, future_source, now)

    invalid_reference_source_path = tmp_path / "invalid-reference-source.json"
    invalid_reference_source = dict(__import__("json").loads(records_path.read_text())[0])
    invalid_reference_source["reference_source"] = "teacher"
    write_json(invalid_reference_source_path, [invalid_reference_source])
    invalid_source = TelemetryDataSource(
        **{
            **asdict(source),
            "records_uri": str(invalid_reference_source_path),
            "per_record_split_eligibility_uri": str(per_record_path),
        }
    )
    records = collect_source_records(definition, definition.data_config, invalid_source, now)
    with pytest.raises(ValidationError, match="reference_source"):
        validate_source_records(contract, records)

    write_json(per_record_path, {"ev1": ["train"]})
    complete = TelemetryDataSource(
        **{**asdict(source), "per_record_split_eligibility_uri": str(per_record_path)}
    )
    records = collect_source_records(definition, definition.data_config, complete, now)
    telemetry_record = next(record for record in records if record.record_id == "ev1")
    assert telemetry_record.split_eligibility == ["train"]


def test_consumed_holdout_rows_cannot_reenter_hidden_holdout(target_dir: Path, now) -> None:
    definition, contract, _ = load_checked_target(target_dir)
    result = build_snapshot(
        definition,
        contract,
        definition.data_config,
        None,
        [],
        PrefixBroker(),
        now,
        SnapshotOptions(storage_root=target_dir / ".snapshots"),
    )
    validation = load_snapshot_view(
        result.snapshot, "validation", "raw", requester="candidate_evaluation"
    )
    assert validation.record_count > 0
    from darjeeling.snapshot_reference import load_snapshot_records, mark_consumed_holdout_rows

    record = load_snapshot_records(validation)[0]
    consumed = mark_consumed_holdout_rows(
        result.snapshot, "validation", [record.snapshot_record_id], "debug", "user"
    )
    with pytest.raises(SnapshotBuildError, match="visible_to"):
        mark_consumed_holdout_rows(
            result.snapshot, "validation", [record.snapshot_record_id], "debug", "bad"
        )
    with pytest.raises(SnapshotBuildError):
        build_snapshot(
            definition,
            contract,
            definition.data_config,
            None,
            [consumed],
            PrefixBroker(),
            now,
            SnapshotOptions(storage_root=target_dir / ".snapshots2"),
        )


def test_consumed_normalized_key_blocks_future_holdout(
    target_dir: Path,
    now,
) -> None:
    definition, contract, _ = load_checked_target(target_dir)
    result = build_snapshot(
        definition,
        contract,
        definition.data_config,
        None,
        [],
        PrefixBroker(),
        now,
        SnapshotOptions(storage_root=target_dir / ".snapshots"),
    )
    validation_view = load_snapshot_view(
        result.snapshot, "validation", "raw", requester="candidate_evaluation"
    )
    validation_record = load_snapshot_records(validation_view)[0]
    consumed = ConsumedRowsManifest(
        snapshot_id=result.snapshot.snapshot_id,
        split="validation",
        record_ids=[],
        normalized_input_keys=[validation_record.normalized_input_key],
        split_group_keys=["different-group-key"],
        reason="reported",
        consumed_at=now,
        visible_to="user",
        replacement_required=False,
    )
    with pytest.raises(SnapshotBuildError, match="consumed holdout enforcement"):
        build_snapshot(
            definition,
            contract,
            definition.data_config,
            None,
            [consumed],
            PrefixBroker(),
            now,
            SnapshotOptions(storage_root=target_dir / ".snapshots2"),
        )


def test_duplicate_conflicts_can_be_excluded_by_snapshot_option(
    target_dir: Path,
    now,
) -> None:
    definition, contract, _ = load_checked_target(target_dir)
    data_config = DataConfig(
        sources=[
            {
                "name": "duplicates",
                "records": [
                    {
                        "record_id": "dup-a",
                        "input": {"text": "a:duplicate"},
                        "reference_output": {"label": "a"},
                        "reference_source": "gold",
                        "split_eligibility": ["train"],
                    },
                    {
                        "record_id": "dup-b",
                        "input": {"text": "a:duplicate"},
                        "reference_output": {"label": "b"},
                        "reference_source": "gold",
                        "split_eligibility": ["train"],
                    },
                ],
            }
        ]
    )
    with pytest.raises(SnapshotBuildError, match="conflicting references"):
        build_snapshot(
            definition,
            contract,
            data_config,
            None,
            [],
            PrefixBroker(),
            now,
            SnapshotOptions(storage_root=target_dir / ".snapshots"),
        )
    excluded = build_snapshot(
        definition,
        contract,
        data_config,
        None,
        [],
        PrefixBroker(),
        now,
        SnapshotOptions(
            on_duplicate_conflict="exclude", storage_root=target_dir / ".snapshots-exclude"
        ),
    )
    assert excluded.snapshot.train_count == 0
    assert excluded.snapshot.validation_count == 0
    assert excluded.snapshot.test_count == 0


def test_duplicate_split_conflicts_can_be_excluded_by_snapshot_option(
    target_dir: Path,
    now,
) -> None:
    definition, contract, _ = load_checked_target(target_dir)
    data_config = DataConfig(
        sources=[
            {
                "name": "duplicates",
                "records": [
                    {
                        "record_id": "dup-train",
                        "input": {"text": "a:duplicate"},
                        "reference_output": {"label": "a"},
                        "reference_source": "gold",
                        "split_eligibility": ["train"],
                    },
                    {
                        "record_id": "dup-validation",
                        "input": {"text": "a:duplicate"},
                        "reference_output": {"label": "a"},
                        "reference_source": "gold",
                        "split_eligibility": ["validation_candidate"],
                    },
                ],
            }
        ]
    )
    with pytest.raises(SnapshotBuildError, match="no common legal split"):
        build_snapshot(
            definition,
            contract,
            data_config,
            None,
            [],
            PrefixBroker(),
            now,
            SnapshotOptions(storage_root=target_dir / ".snapshots"),
        )
    excluded = build_snapshot(
        definition,
        contract,
        data_config,
        None,
        [],
        PrefixBroker(),
        now,
        SnapshotOptions(
            on_duplicate_conflict="exclude", storage_root=target_dir / ".snapshots-exclude"
        ),
    )
    assert excluded.snapshot.train_count == 0
    assert excluded.snapshot.validation_count == 0
    assert excluded.snapshot.test_count == 0


def test_agent_workspace_cannot_load_holdout_or_redacted_snapshot_views(
    target_dir: Path, tmp_path: Path, now
) -> None:
    definition, contract, _ = load_checked_target(target_dir)
    result = build_snapshot(
        definition,
        contract,
        definition.data_config,
        None,
        [],
        PrefixBroker(),
        now,
        SnapshotOptions(storage_root=target_dir / ".snapshots"),
    )
    assert not hasattr(result, "validation_view")
    assert not hasattr(result, "test_view")
    with pytest.raises(SnapshotBuildError, match="agent workspace"):
        load_snapshot_view(result.snapshot, "validation", "raw", requester="agent_workspace")
    with pytest.raises(SnapshotBuildError, match="only returns raw"):
        load_snapshot_view(result.snapshot, "train", "redacted", requester="agent_workspace")
    train = load_snapshot_view(result.snapshot, "train", "raw", requester="agent_workspace")
    assert train.split == "train"
    manifest = export_train_view_for_agent(
        result.snapshot,
        contract,
        AgentViewOptions(redaction_level="redacted"),
        tmp_path / "agent-train",
    )
    rows = __import__("json").loads(manifest.view_path.read_text())
    assert manifest.view_path.name == "train.json"
    assert rows
    assert rows[0]["input"]["text"] == "<redacted>"
    assert "normalized_input_key" not in rows[0]
    assert "split_group_key" not in rows[0]
    assert "slice_tags" not in rows[0]
    assert "source_provenance" not in rows[0]


def test_invalid_reference_output_is_recorded_as_labeling_failure(target_dir: Path) -> None:
    definition, contract, _ = load_checked_target(target_dir)

    class BadOutputBroker(PrefixBroker):
        def call(self, request, context):
            return ReferenceResponse(
                payload={"bad": "shape"},
                reference_source="versioned_l4",
                reference_version=self.reference_version,
                usage={"completion_tokens": 3},
                cost=0.25,
                latency_ms=12.0,
            )

    record = SourceRecord(
        record_id="needs-reference",
        input={"text": "a:needs-reference"},
        reference_output=None,
        reference_source=None,
        split_eligibility=["train"],
        source_name="test",
    )
    labeled, ledger, failures = reference_missing_outputs(
        contract, [record], BadOutputBroker(), ReferenceBudget()
    )
    assert labeled == []
    assert ledger.cost == 0.25
    assert ledger.errors == {"validation_failure": 1}
    assert failures.failures == [
        {"record_id": "needs-reference", "error_type": "validation_failure"}
    ]


def test_source_record_provenance_literals_are_validated(target_dir: Path) -> None:
    _, contract, _ = load_checked_target(target_dir)
    bad_reference_source = SourceRecord(
        record_id="bad-source",
        input={"text": "a:bad-source"},
        reference_output={"label": "a"},
        reference_source="teacher",
        split_eligibility=["train"],
        source_name="test",
    )
    with pytest.raises(ValidationError, match="reference_source"):
        validate_source_records(contract, [bad_reference_source])

    bad_split = SourceRecord(
        record_id="bad-split",
        input={"text": "a:bad-split"},
        reference_output={"label": "a"},
        reference_source="gold",
        split_eligibility=["validation"],
        source_name="test",
    )
    with pytest.raises(ValidationError, match="split_eligibility"):
        validate_source_records(contract, [bad_split])


def test_invalid_broker_reference_source_is_labeling_failure(target_dir: Path) -> None:
    _, contract, _ = load_checked_target(target_dir)

    class BadReferenceSourceBroker(PrefixBroker):
        def call(self, request, context):
            return ReferenceResponse(
                payload={"label": "a"},
                reference_source="teacher",
                reference_version=self.reference_version,
            )

    record = SourceRecord(
        record_id="needs-reference",
        input={"text": "a:needs-reference"},
        reference_output=None,
        reference_source=None,
        split_eligibility=["train"],
        source_name="test",
    )
    labeled, ledger, failures = reference_missing_outputs(
        contract, [record], BadReferenceSourceBroker(), ReferenceBudget()
    )
    assert labeled == []
    assert ledger.errors == {"validation_failure": 1}
    assert failures.failures == [
        {"record_id": "needs-reference", "error_type": "validation_failure"}
    ]


def test_broker_labeled_snapshot_records_preserve_reference_version(
    target_dir: Path,
    now,
) -> None:
    data = __import__("yaml").safe_load((target_dir / "data.yaml").read_text())
    missing = data["sources"][0]["records"][0]
    missing.pop("reference_output", None)
    missing.pop("reference_source", None)
    (target_dir / "data.yaml").write_text(__import__("yaml").safe_dump(data))
    definition, contract, _ = load_checked_target(target_dir)
    result = build_snapshot(
        definition,
        contract,
        definition.data_config,
        None,
        [],
        PrefixBroker(),
        now,
        SnapshotOptions(storage_root=target_dir / ".snapshots"),
    )
    records = [
        *load_snapshot_records(result.train_view),
        *load_snapshot_records(
            load_snapshot_view(
                result.snapshot, "validation", "raw", requester="candidate_evaluation"
            )
        ),
        *load_snapshot_records(
            load_snapshot_view(result.snapshot, "test", "raw", requester="candidate_evaluation")
        ),
    ]
    labeled = next(record for record in records if record.input["text"] == missing["input"]["text"])
    assert labeled.reference_source == "versioned_l4"
    assert labeled.reference_version == PrefixBroker.reference_version


def test_build_snapshot_surfaces_reference_labeling_failure_report(
    target_dir: Path,
    now,
) -> None:
    data = __import__("yaml").safe_load((target_dir / "data.yaml").read_text())
    missing = data["sources"][0]["records"][0]
    missing.pop("reference_output", None)
    missing.pop("reference_source", None)
    (target_dir / "data.yaml").write_text(__import__("yaml").safe_dump(data))
    definition, contract, _ = load_checked_target(target_dir)

    class BadLabelBroker(PrefixBroker):
        def call(self, request, context):
            if context.purpose == "reference_qualification":
                return super().call(request, context)
            return ReferenceResponse(
                payload={"bad": "shape"},
                reference_source="versioned_l4",
                reference_version=self.reference_version,
                cost=0.25,
            )

    with pytest.raises(SnapshotBuildError, match="reference labeling failed") as exc:
        build_snapshot(
            definition,
            contract,
            definition.data_config,
            None,
            [],
            BadLabelBroker(),
            now,
            SnapshotOptions(storage_root=target_dir / ".snapshots"),
        )
    assert exc.value.reference_qualification.status == "pass"
    assert exc.value.reference_failure_report.failures == [
        {"record_id": missing["record_id"], "error_type": "validation_failure"}
    ]
    assert exc.value.reference_usage.errors == {"validation_failure": 1}
    assert exc.value.reference_usage.cost == 0.25


def test_reference_qualification_gold_precision_uses_only_gold_or_human(
    target_dir: Path,
    now,
) -> None:
    data = __import__("yaml").safe_load((target_dir / "data.yaml").read_text())
    for source in data["sources"]:
        for record in source["records"]:
            record["reference_source"] = "verified_l4"
    (target_dir / "data.yaml").write_text(__import__("yaml").safe_dump(data))
    definition, contract, _ = load_checked_target(target_dir)
    records = validate_source_records(
        contract, collect_source_records(definition, definition.data_config, None, now)
    )
    report = qualify_reference_baseline(
        definition,
        contract,
        records,
        PrefixBroker(),
        ReferenceQualificationOptions(min_gold_samples=1),
    )
    assert report.status == "insufficient"
    assert report.gold_sample_count == 0
    assert report.gold_precision is None


def test_reference_qualification_hard_failures_override_insufficient_gold(
    target_dir: Path,
    now,
) -> None:
    data = __import__("yaml").safe_load((target_dir / "data.yaml").read_text())
    for source in data["sources"]:
        for record in source["records"]:
            record["reference_source"] = "verified_l4"
    (target_dir / "data.yaml").write_text(__import__("yaml").safe_dump(data))
    definition, contract, _ = load_checked_target(target_dir)
    records = validate_source_records(
        contract, collect_source_records(definition, definition.data_config, None, now)
    )

    class BadOutputBroker(PrefixBroker):
        def call(self, request, context):
            return ReferenceResponse(
                payload={"bad": "shape"},
                reference_source="versioned_l4",
                reference_version=self.reference_version,
            )

    report = qualify_reference_baseline(
        definition,
        contract,
        records,
        BadOutputBroker(),
        ReferenceQualificationOptions(
            min_gold_samples=1,
            max_parse_failure_rate=0.0,
            max_schema_failure_rate=0.0,
        ),
    )
    assert report.status == "fail"
    assert report.schema_failure_rate > 0
    assert report.gold_sample_count == 0


def test_openai_compatible_reference_config_writes_cache_and_usage_ledger(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = tmp_path / "reference.json"
    cache_path = tmp_path / "reference_cache.jsonl"
    ledger_path = tmp_path / "reference_usage.json"
    config_path.write_text(
        __import__("json").dumps(
            {
                "provider": "openai_compatible",
                "base_url_env": "TEST_OPENAI_BASE_URL",
                "api_key_env": "TEST_OPENAI_API_KEY",
                "model": "test-model",
                "timeout_ms": 5000,
                "max_completion_tokens": 32,
                "price": {
                    "input_per_million": 1.0,
                    "output_per_million": 2.0,
                },
                "cache_path": cache_path.name,
                "usage_ledger_path": ledger_path.name,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TEST_OPENAI_BASE_URL", "https://provider.example/v1")
    monkeypatch.setenv("TEST_OPENAI_API_KEY", "secret")
    calls: list[dict] = []

    class FakeResponse:
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return __import__("json").dumps(
                {
                    "choices": [
                        {
                            "message": {"content": '{"label": "a"}'},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 10},
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        calls.append({"url": request.full_url, "timeout": timeout})
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    broker = build_reference_broker_from_config(config_path)
    context = ReferenceContext(
        purpose="snapshot_label",
        request_id="r1",
        metadata={"contract_hash": "c1", "normalized_input": "n1", "timeout_ms": 2500},
    )

    first = broker.call({"messages": [{"role": "user", "content": "x"}]}, context)
    second = broker.call({"messages": [{"role": "user", "content": "x"}]}, context)

    assert first.payload == {"label": "a"}
    assert first.cost == 0.00012
    assert second.payload == {"label": "a"}
    assert second.cost == 0.0
    assert len(calls) == 1
    assert calls[0]["url"] == "https://provider.example/v1/chat/completions"
    assert calls[0]["timeout"] == 2.5
    cache_lines = cache_path.read_text(encoding="utf-8").splitlines()
    assert len(cache_lines) == 1
    ledger = __import__("json").loads(ledger_path.read_text(encoding="utf-8"))
    assert ledger["totals"]["provider_call_count"] == 1
    assert ledger["totals"]["cache_hit_count"] == 1
    assert ledger["entries"][0]["cost_status"] == "estimated-from-token-usage"
    assert ledger["entries"][1]["cost_status"] == "cache-hit"


def test_openai_compatible_reference_config_defaults_openai_base_url(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = tmp_path / "reference.json"
    config_path.write_text(
        __import__("json").dumps(
            {
                "provider": "openai_compatible",
                "base_url_env": "OPENAI_BASE_URL",
                "api_key_env": "TEST_OPENAI_API_KEY",
                "model": "test-model",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("TEST_OPENAI_API_KEY", "secret")

    broker = build_reference_broker_from_config(config_path)

    assert broker.base_url == "https://api.openai.com/v1"


def test_path_backed_source_watermark_changes_with_file_contents(
    target_dir: Path,
    now,
) -> None:
    data = __import__("yaml").safe_load((target_dir / "data.yaml").read_text())
    records = data["sources"][0]["records"]
    records_path = target_dir / "records.json"
    write_json(records_path, records)
    data["sources"] = [{"name": "file-backed", "path": "records.json"}]
    (target_dir / "data.yaml").write_text(__import__("yaml").safe_dump(data))
    definition, contract, _ = load_checked_target(target_dir)
    first = build_snapshot(
        definition,
        contract,
        definition.data_config,
        None,
        [],
        PrefixBroker(),
        now,
        SnapshotOptions(storage_root=target_dir / ".snapshots"),
    )
    records[0]["metadata"]["watermark_probe"] = "changed"
    write_json(records_path, records)
    changed_definition, changed_contract, _ = load_checked_target(target_dir)
    second = build_snapshot(
        changed_definition,
        changed_contract,
        changed_definition.data_config,
        None,
        [],
        PrefixBroker(),
        now,
        SnapshotOptions(storage_root=target_dir / ".snapshots2"),
    )
    assert (
        first.snapshot.source_watermarks["file-backed"]
        != second.snapshot.source_watermarks["file-backed"]
    )


def test_telemetry_source_watermark_records_runtime_evidence_boundary(
    target_dir: Path,
    tmp_path: Path,
    now,
) -> None:
    definition, contract, _ = load_checked_target(target_dir)
    records_path = tmp_path / "telemetry.json"
    split_path = tmp_path / "telemetry-splits.json"
    write_json(
        records_path,
        [
            {
                "evidence_id": "ev1",
                "created_at": now.isoformat(),
                "source_event_at": now.isoformat(),
                "input_payload": {"text": "a:telemetry"},
                "reference_output_payload": {"label": "a"},
                "reference_source": "user_feedback",
                "approved_for": ["train"],
                "release_id": "rel",
            }
        ],
    )
    write_json(split_path, {"ev1": ["train"]})
    source = TelemetryDataSource(
        source_id="telemetry-test",
        target_name=definition.name,
        contract_hash=definition.contract_hash,
        cutoff_time=now,
        records_uri=str(records_path),
        default_split_eligibility=["train"],
        per_record_split_eligibility_uri=str(split_path),
        included_sources=["user_feedback"],
        provenance_digest="p1",
    )
    first = build_snapshot(
        definition,
        contract,
        definition.data_config,
        source,
        [],
        PrefixBroker(),
        now,
        SnapshotOptions(storage_root=target_dir / ".snapshots"),
    )
    changed_source = TelemetryDataSource(**{**asdict(source), "provenance_digest": "p2"})
    second = build_snapshot(
        definition,
        contract,
        definition.data_config,
        changed_source,
        [],
        PrefixBroker(),
        now,
        SnapshotOptions(storage_root=target_dir / ".snapshots2"),
    )
    key = "telemetry:telemetry-test"
    assert key in first.snapshot.source_watermarks
    assert first.snapshot.source_watermarks[key] != second.snapshot.source_watermarks[key]


def test_build_snapshot_enforces_reference_budget(target_dir: Path, now) -> None:
    data = __import__("yaml").safe_load((target_dir / "data.yaml").read_text())
    missing = data["sources"][0]["records"][0]
    missing.pop("reference_output", None)
    missing.pop("reference_source", None)
    (target_dir / "data.yaml").write_text(__import__("yaml").safe_dump(data))
    definition, contract, _ = load_checked_target(target_dir)
    with pytest.raises(SnapshotBuildError, match="reference budget exhausted"):
        build_snapshot(
            definition,
            contract,
            definition.data_config,
            None,
            [],
            PrefixBroker(),
            now,
            SnapshotOptions(
                reference_budget=ReferenceBudget(max_calls=0),
                storage_root=target_dir / ".snapshots",
            ),
        )


def test_reference_qualification_labels_l4_agreement_not_gold(target_dir: Path, now) -> None:
    definition, contract, _ = load_checked_target(target_dir)
    result = build_snapshot(
        definition,
        contract,
        definition.data_config,
        None,
        [],
        PrefixBroker(),
        now,
        SnapshotOptions(storage_root=target_dir / ".snapshots"),
    )
    assert result.reference_qualification.status == "pass"
    assert result.reference_qualification.l4_agreement_count >= 1


def test_reference_qualification_is_insufficient_without_comparable_evidence(
    target_dir: Path,
    now,
) -> None:
    data = __import__("yaml").safe_load((target_dir / "data.yaml").read_text())
    for source in data["sources"]:
        for record in source["records"]:
            record.pop("reference_output", None)
            record.pop("reference_source", None)
    (target_dir / "data.yaml").write_text(__import__("yaml").safe_dump(data))
    definition, contract, _ = load_checked_target(target_dir)
    records = validate_source_records(
        contract, collect_source_records(definition, definition.data_config, None, now)
    )
    report = qualify_reference_baseline(
        definition,
        contract,
        records,
        PrefixBroker(),
        SnapshotOptions().qualification_options,
    )
    assert report.status == "insufficient"
    with pytest.raises(SnapshotBuildError, match="reference qualification insufficient") as exc:
        build_snapshot(
            definition,
            contract,
            definition.data_config,
            None,
            [],
            PrefixBroker(),
            now,
            SnapshotOptions(storage_root=target_dir / ".snapshots"),
        )
    assert exc.value.reference_qualification.status == "insufficient"

    result = build_snapshot(
        definition,
        contract,
        definition.data_config,
        None,
        [],
        PrefixBroker(),
        now,
        SnapshotOptions(
            allow_insufficient_reference=True,
            storage_root=target_dir / ".snapshots-approved",
        ),
    )
    assert result.reference_qualification.status == "insufficient"
    assert result.reference_qualification.notes == [
        "no comparable reference evidence is available"
    ]
