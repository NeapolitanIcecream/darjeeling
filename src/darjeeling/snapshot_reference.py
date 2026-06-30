from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from darjeeling.errors import SnapshotBuildError, ValidationError
from darjeeling.model import (
    AgentViewOptions,
    ConsumedRowsManifest,
    DataConfig,
    ReferenceBroker,
    ReferenceBudget,
    ReferenceCallResult,
    ReferenceContext,
    ReferenceFailureReport,
    ReferenceQualificationOptions,
    ReferenceQualificationReport,
    ReferenceResponse,
    ReferenceUsageLedger,
    Snapshot,
    SnapshotBuildResult,
    SnapshotOptions,
    SnapshotRecord,
    SnapshotView,
    SourceRecord,
    SplitEligibility,
    SplitName,
    TargetDefinition,
    TargetRuntimeContract,
    TelemetryDataSource,
    TrainViewManifest,
)
from darjeeling.util import file_digest, new_id, read_json, stable_hash, utcnow, write_json

_ALLOWED_SPLIT_ELIGIBILITY = {"train", "validation_candidate", "test_candidate"}
_ALLOWED_REFERENCE_SOURCES = {"gold", "human", "versioned_l4", "verified_l4", "user_feedback"}


def load_data_config(definition: TargetDefinition) -> DataConfig:
    return definition.data_config


def _source_record_from_dict(
    raw: dict[str, Any], source_name: str, default_eligibility: list[SplitEligibility]
) -> SourceRecord:
    timestamp = raw.get("source_timestamp")
    parsed_timestamp = (
        datetime.fromisoformat(timestamp) if isinstance(timestamp, str) else timestamp
    )
    return SourceRecord(
        record_id=str(raw.get("record_id") or stable_hash(raw)),
        input=dict(raw["input"]),
        reference_output=raw.get("reference_output"),
        reference_source=raw.get("reference_source"),
        split_eligibility=list(raw.get("split_eligibility") or default_eligibility),
        source_name=source_name,
        source_timestamp=parsed_timestamp,
        metadata=dict(raw.get("metadata", {})),
    )


def collect_source_records(
    definition: TargetDefinition,
    data_config: DataConfig,
    telemetry_source: TelemetryDataSource | None,
    cutoff_time: datetime,
) -> list[SourceRecord]:
    if telemetry_source is not None:
        if (
            telemetry_source.target_name != definition.name
            or telemetry_source.contract_hash != definition.contract_hash
        ):
            raise SnapshotBuildError(
                "telemetry source target or contract does not match target definition"
            )
        if telemetry_source.cutoff_time > cutoff_time:
            raise SnapshotBuildError("telemetry source cutoff is later than snapshot cutoff")
    records: list[SourceRecord] = []
    for index, source in enumerate(data_config.sources):
        name = str(source.get("name") or f"source-{index}")
        default_eligibility = list(
            source.get("split_eligibility") or data_config.default_split_eligibility
        )
        for raw in source.get("records", []):
            record = _source_record_from_dict(raw, name, default_eligibility)
            if record.source_timestamp is None or record.source_timestamp <= cutoff_time:
                records.append(record)
        if source.get("path"):
            source_path = (definition.target_path / source["path"]).resolve()
            for raw in read_json(source_path):
                record = _source_record_from_dict(raw, name, default_eligibility)
                if record.source_timestamp is None or record.source_timestamp <= cutoff_time:
                    records.append(record)
    if telemetry_source is not None:
        telemetry_records = read_json(Path(telemetry_source.records_uri))
        if not isinstance(telemetry_records, list):
            raise SnapshotBuildError("telemetry records must be a list")
        active_telemetry_records: list[dict[str, Any]] = []
        for raw in telemetry_records:
            if not isinstance(raw, dict):
                raise SnapshotBuildError("telemetry records must be objects")
            created_at = datetime.fromisoformat(raw["created_at"])
            if created_at > telemetry_source.cutoff_time:
                continue
            source_event_at = datetime.fromisoformat(raw["source_event_at"])
            if source_event_at > telemetry_source.cutoff_time or source_event_at > cutoff_time:
                raise SnapshotBuildError("telemetry source event is after snapshot cutoff")
            active_telemetry_records.append(raw)
        per_record: dict[str, list[SplitEligibility]] = {}
        if active_telemetry_records:
            if telemetry_source.per_record_split_eligibility_uri:
                raw_per_record = read_json(Path(telemetry_source.per_record_split_eligibility_uri))
                if not isinstance(raw_per_record, dict):
                    raise SnapshotBuildError(
                        "per-record telemetry split eligibility must be a mapping"
                    )
                per_record = raw_per_record
                active_ids = {str(raw["evidence_id"]) for raw in active_telemetry_records}
                mapped_ids = set(per_record)
                missing = active_ids - mapped_ids
                extra = mapped_ids - active_ids
                if missing or extra:
                    raise SnapshotBuildError(
                        "per-record telemetry split eligibility must exactly match telemetry rows"
                    )
                for raw in active_telemetry_records:
                    row_permission = _telemetry_row_approved_for(raw)
                    sidecar_permission = _validate_telemetry_split_eligibility(
                        per_record[str(raw["evidence_id"])], str(raw["evidence_id"])
                    )
                    if (
                        row_permission is not None
                        and set(row_permission) != set(sidecar_permission)
                    ):
                        raise SnapshotBuildError(
                            "per-record telemetry split eligibility does not match row permissions"
                        )
            else:
                default_eligibility = _validate_telemetry_split_eligibility(
                    telemetry_source.default_split_eligibility, "default"
                )
                row_permissions = [
                    _telemetry_row_approved_for(raw) for raw in active_telemetry_records
                ]
                known_row_permissions = [
                    permission for permission in row_permissions if permission is not None
                ]
                if any(
                    set(permission) != set(default_eligibility)
                    for permission in known_row_permissions
                ):
                    raise SnapshotBuildError(
                        "telemetry default split eligibility does not match row permissions"
                    )
                per_record = {
                    str(raw["evidence_id"]): default_eligibility
                    for raw in active_telemetry_records
                }
        for raw in active_telemetry_records:
            eligibility = _validate_telemetry_split_eligibility(
                per_record.get(str(raw["evidence_id"])), str(raw["evidence_id"])
            )
            records.append(
                SourceRecord(
                    record_id=raw["evidence_id"],
                    input=dict(raw["input_payload"]),
                    reference_output=dict(raw["reference_output_payload"]),
                    reference_source=raw["reference_source"],
                    split_eligibility=list(eligibility),
                    source_name=telemetry_source.source_id,
                    source_timestamp=datetime.fromisoformat(raw["source_event_at"]),
                    metadata={
                        "telemetry_source_id": telemetry_source.source_id,
                        "release_id": raw.get("release_id"),
                    },
                )
            )
    return records


def _validate_telemetry_split_eligibility(
    value: Any, evidence_id: str
) -> list[SplitEligibility]:
    if (
        not isinstance(value, list)
        or not value
        or any(item not in _ALLOWED_SPLIT_ELIGIBILITY for item in value)
    ):
        raise SnapshotBuildError(
            f"telemetry evidence {evidence_id} has invalid split eligibility"
        )
    return list(value)


def _validate_record_split_eligibility(value: Any) -> list[SplitEligibility]:
    if (
        not isinstance(value, list)
        or not value
        or any(item not in _ALLOWED_SPLIT_ELIGIBILITY for item in value)
    ):
        raise ValidationError("split_eligibility must contain known split roles")
    return list(value)


def _validate_reference_source(value: Any) -> None:
    if value is None:
        return
    if value not in _ALLOWED_REFERENCE_SOURCES:
        raise ValidationError("reference_source must be a known reference source")


def _telemetry_row_approved_for(raw: dict[str, Any]) -> list[SplitEligibility] | None:
    value = raw.get("approved_for")
    if value is None and isinstance(raw.get("privacy_review"), dict):
        value = raw["privacy_review"].get("approved_for")
    if value is None:
        return None
    return _validate_telemetry_split_eligibility(value, str(raw["evidence_id"]))


def validate_source_records(
    contract: TargetRuntimeContract, records: Iterable[SourceRecord]
) -> list[SourceRecord]:
    validated: list[SourceRecord] = []
    for record in records:
        try:
            input_value = contract.validate_input(record.input)
            reference = (
                contract.validate_output(record.reference_output)
                if record.reference_output is not None
                else None
            )
            _validate_reference_source(record.reference_source)
            split_eligibility = _validate_record_split_eligibility(record.split_eligibility)
        except Exception as exc:
            raise ValidationError(f"invalid source record {record.record_id}: {exc}") from exc
        validated.append(
            SourceRecord(
                **{
                    **record.__dict__,
                    "input": input_value,
                    "reference_output": reference,
                    "split_eligibility": split_eligibility,
                }
            )
        )
    return validated


def call_reference(
    contract: TargetRuntimeContract,
    input_value: dict[str, Any],
    broker: ReferenceBroker,
    request_context: ReferenceContext,
) -> ReferenceCallResult:
    response: ReferenceResponse | None = None
    try:
        request = contract.build_reference_request(input_value, request_context)
        broker_context = replace(
            request_context,
            metadata={
                **request_context.metadata,
                "contract_hash": contract.contract_hash,
                "normalized_input": contract.normalize_input(input_value),
                "request_hash": stable_hash(request),
            },
        )
        response = broker.call(request, broker_context)
        _validate_reference_source(response.reference_source)
        output = contract.validate_output(contract.parse_reference_response(response))
        return ReferenceCallResult(
            status="ok",
            output=output,
            reference_source=response.reference_source,
            reference_version=response.reference_version or broker.reference_version,
            usage=response.usage,
            cost=response.cost,
            latency_ms=response.latency_ms,
            finish_status=response.finish_status,
        )
    except ValidationError as exc:
        return ReferenceCallResult(
            status="error",
            output=None,
            reference_source=None,
            reference_version=(
                response.reference_version or broker.reference_version
                if response is not None
                else getattr(broker, "reference_version", None)
            ),
            usage=response.usage if response is not None else {},
            cost=response.cost if response is not None else 0.0,
            latency_ms=response.latency_ms if response is not None else 0.0,
            finish_status="error",
            error_type="validation_failure",
            error_message_hash=stable_hash(str(exc)),
        )
    except Exception as exc:
        return ReferenceCallResult(
            status="error",
            output=None,
            reference_source=None,
            reference_version=getattr(broker, "reference_version", None),
            usage={},
            cost=0.0,
            latency_ms=0.0,
            finish_status="error",
            error_type="provider_error",
            error_message_hash=stable_hash(str(exc)),
        )


def reference_missing_outputs(
    contract: TargetRuntimeContract,
    records: Iterable[SourceRecord],
    broker: ReferenceBroker,
    budget: ReferenceBudget,
) -> tuple[list[SourceRecord], ReferenceUsageLedger, ReferenceFailureReport]:
    output: list[SourceRecord] = []
    failures: list[dict[str, Any]] = []
    calls = 0
    cost = 0.0
    errors: dict[str, int] = {}
    for record in records:
        if record.reference_output is not None:
            output.append(record)
            continue
        if budget.max_calls is not None and calls >= budget.max_calls:
            raise SnapshotBuildError("reference budget exhausted")
        result = call_reference(
            contract,
            record.input,
            broker,
            ReferenceContext(purpose="snapshot_label", request_id=record.record_id),
        )
        calls += 1
        cost += result.cost
        if budget.max_cost is not None and cost > budget.max_cost:
            raise SnapshotBuildError("reference cost budget exhausted")
        if (
            result.status == "ok"
            and result.output is not None
            and result.reference_source is not None
        ):
            metadata = dict(record.metadata)
            metadata["reference_version"] = result.reference_version
            output.append(
                SourceRecord(
                    **{
                        **record.__dict__,
                        "reference_output": result.output,
                        "reference_source": result.reference_source,
                        "metadata": metadata,
                    }
                )
            )
        else:
            errors[result.error_type or "unknown"] = (
                errors.get(result.error_type or "unknown", 0) + 1
            )
            failures.append({"record_id": record.record_id, "error_type": result.error_type})
    return (
        output,
        ReferenceUsageLedger(call_count=calls, cost=cost, errors=errors),
        ReferenceFailureReport(failures),
    )


def qualify_reference_baseline(
    definition: TargetDefinition,
    contract: TargetRuntimeContract,
    records: Iterable[SourceRecord],
    broker: ReferenceBroker,
    qualification_options: ReferenceQualificationOptions,
) -> ReferenceQualificationReport:
    records = list(records)
    comparable = [r for r in records if r.reference_output is not None]
    gold_like = [r for r in comparable if r.reference_source in {"gold", "human"}]
    l4_agreement_count = 0
    gold_correct_count = 0
    parse_failures = 0
    schema_failures = 0
    total_cost = 0.0
    total_latency = 0.0
    for record in comparable:
        result = call_reference(
            contract,
            record.input,
            broker,
            ReferenceContext(purpose="reference_qualification", request_id=record.record_id),
        )
        total_cost += result.cost
        total_latency += result.latency_ms
        if result.status != "ok" or result.output is None:
            if result.error_type == "validation_failure":
                schema_failures += 1
            else:
                parse_failures += 1
            continue
        if contract.is_correct(result.output, record.reference_output or {}):
            l4_agreement_count += 1
            if record in gold_like:
                gold_correct_count += 1
    sample_count = len(comparable)
    agreement_rate = l4_agreement_count / sample_count if sample_count else 0.0
    gold_precision = gold_correct_count / len(gold_like) if gold_like else None
    parse_rate = parse_failures / sample_count if sample_count else 0.0
    schema_rate = schema_failures / sample_count if sample_count else 0.0
    notes: list[str] = []
    if sample_count == 0:
        status: Literal["pass", "fail", "insufficient"] = "insufficient"
        notes.append("no comparable reference evidence is available")
    elif (
        parse_rate > qualification_options.max_parse_failure_rate
        or schema_rate > qualification_options.max_schema_failure_rate
    ):
        status = "fail"
        notes.append("reference parse or schema failure rate is too high")
    elif agreement_rate < qualification_options.min_agreement_rate:
        status = "fail"
        notes.append("reference agreement rate is below requirement")
    elif len(gold_like) < qualification_options.min_gold_samples:
        status = "insufficient"
        notes.append(
            "independent gold or human evidence is insufficient; "
            "versioned L4 rows are agreement only"
        )
    else:
        status = "pass"
    return ReferenceQualificationReport(
        target_name=definition.name,
        contract_hash=definition.contract_hash,
        reference_version=getattr(broker, "reference_version", "unknown"),
        gold_sample_count=len(gold_like),
        l4_agreement_count=l4_agreement_count,
        gold_correct_count=gold_correct_count if gold_like else None,
        agreement_rate=agreement_rate,
        gold_precision=gold_precision,
        parse_failure_rate=parse_rate,
        schema_failure_rate=schema_rate,
        latency={"total_ms": total_latency, "sample_count": sample_count},
        cost={"total": total_cost},
        status=status,
        notes=notes,
    )


def _intersect_split_eligibility(
    values: Iterable[list[SplitEligibility]],
) -> list[SplitEligibility]:
    iterator = iter(values)
    try:
        current = set(next(iterator))
    except StopIteration:
        return []
    for value in iterator:
        current &= set(value)
    ordered = ["train", "validation_candidate", "test_candidate"]
    return [v for v in ordered if v in current]


def deduplicate_records(
    contract: TargetRuntimeContract,
    records: Iterable[SourceRecord],
    on_duplicate_conflict: Literal["fail", "exclude"],
) -> list[SourceRecord]:
    grouped: dict[str, list[SourceRecord]] = defaultdict(list)
    for record in records:
        grouped[contract.normalize_input(record.input)].append(record)
    deduped: list[SourceRecord] = []
    for key, group in grouped.items():
        eligibility = _intersect_split_eligibility(record.split_eligibility for record in group)
        if not eligibility:
            if on_duplicate_conflict == "exclude":
                continue
            raise SnapshotBuildError(f"duplicate group {key} has no common legal split")
        references = {stable_hash((r.reference_output, r.reference_source)) for r in group}
        if len(references) > 1:
            if on_duplicate_conflict == "exclude":
                continue
            raise SnapshotBuildError(f"duplicate group {key} has conflicting references")
        chosen = group[0]
        provenance = {
            "duplicates": [record.record_id for record in group],
            "source_count": len(group),
        }
        metadata = {**chosen.metadata, "duplicate_provenance": provenance}
        deduped.append(
            SourceRecord(
                **{**chosen.__dict__, "split_eligibility": eligibility, "metadata": metadata}
            )
        )
    return deduped


def assign_split_groups(
    contract: TargetRuntimeContract, records: Iterable[SourceRecord]
) -> dict[str, list[SourceRecord]]:
    grouped: dict[str, list[SourceRecord]] = defaultdict(list)
    for record in records:
        grouped[contract.split_group(record)].append(record)
    return grouped


def build_split_plan(
    contract: TargetRuntimeContract,
    grouped_records: dict[str, list[SourceRecord]],
    split_options: SnapshotOptions,
    consumed_manifests: list[ConsumedRowsManifest],
) -> dict[str, SplitName]:
    consumed_keys = {
        key
        for manifest in consumed_manifests
        for key in [*manifest.normalized_input_keys, *manifest.split_group_keys]
    }
    groups = sorted(
        grouped_records.items(), key=lambda item: stable_hash((split_options.seed, item[0]))
    )
    split_plan: dict[str, SplitName] = {}
    rng = random.Random(split_options.seed)
    for group_key, records in groups:
        eligibility = _intersect_split_eligibility(record.split_eligibility for record in records)
        if not eligibility:
            if split_options.on_duplicate_conflict == "exclude":
                continue
            raise SnapshotBuildError(f"group {group_key} has no common legal split")
        normalized_keys = {contract.normalize_input(record.input) for record in records}
        hidden_allowed = group_key not in consumed_keys and not normalized_keys & consumed_keys
        choices: list[SplitName] = []
        if "train" in eligibility:
            choices.append("train")
        if hidden_allowed and "validation_candidate" in eligibility:
            choices.append("validation")
        if hidden_allowed and "test_candidate" in eligibility:
            choices.append("test")
        if not choices:
            raise SnapshotBuildError(
                f"group {group_key} has no legal split after consumed holdout enforcement"
            )
        roll = rng.random()
        desired = (
            "test"
            if roll < split_options.test_fraction
            else "validation"
            if roll < split_options.test_fraction + split_options.validation_fraction
            else "train"
        )
        split_plan[group_key] = desired if desired in choices else choices[0]
    return split_plan


def materialize_snapshot_records(
    contract: TargetRuntimeContract,
    records: Iterable[SourceRecord],
    split_manifest: dict[str, SplitName],
) -> list[SnapshotRecord]:
    result: list[SnapshotRecord] = []
    for record in records:
        group_key = contract.split_group(record)
        if group_key not in split_manifest:
            continue
        if record.reference_output is None or record.reference_source is None:
            raise SnapshotBuildError(f"record {record.record_id} has no reference output")
        try:
            _validate_reference_source(record.reference_source)
            split_eligibility = _validate_record_split_eligibility(record.split_eligibility)
        except ValidationError as exc:
            raise SnapshotBuildError(
                f"record {record.record_id} has invalid snapshot provenance: {exc}"
            ) from exc
        result.append(
            SnapshotRecord(
                snapshot_record_id=new_id("sr"),
                normalized_input_key=contract.normalize_input(record.input),
                split_group_key=group_key,
                input=record.input,
                reference_output=record.reference_output,
                reference_source=record.reference_source,
                split_eligibility=split_eligibility,
                reference_version=record.metadata.get("reference_version"),
                slice_tags=contract.slice_tags(record),
                source_provenance={
                    "source_name": record.source_name,
                    "record_id": record.record_id,
                    "source_timestamp": record.source_timestamp.isoformat()
                    if record.source_timestamp
                    else None,
                    "metadata": record.metadata,
                    "split": split_manifest[group_key],
                },
            )
        )
    return result


def write_snapshot(
    definition: TargetDefinition,
    records: list[SnapshotRecord],
    split_manifest: dict[str, SplitName],
    reference_ledger: ReferenceUsageLedger,
    cutoff_time: datetime,
    source_watermarks: dict[str, str],
    telemetry_source_id: str | None,
    storage: Path,
) -> Snapshot:
    snapshot_id = new_id("snapshot")
    snapshot_dir = storage / snapshot_id
    snapshot_dir.mkdir(parents=True, exist_ok=False)
    by_split: dict[SplitName, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for record in records:
        split = split_manifest[record.split_group_key]
        by_split[split].append(asdict(record))
    for split, split_records in by_split.items():
        write_json(snapshot_dir / f"{split}.json", split_records)
    write_json(snapshot_dir / "split_manifest.json", split_manifest)
    write_json(snapshot_dir / "reference_ledger.json", asdict(reference_ledger))
    records_digest = stable_hash([asdict(record) for record in records])
    split_digest = stable_hash(split_manifest)
    ledger_digest = stable_hash(reference_ledger)
    snapshot_digest = stable_hash(
        {
            "target_name": definition.name,
            "contract_hash": definition.contract_hash,
            "records_digest": records_digest,
            "split_manifest_digest": split_digest,
            "reference_ledger_digest": ledger_digest,
            "cutoff_time": cutoff_time,
        }
    )
    snapshot = Snapshot(
        snapshot_id=snapshot_id,
        snapshot_digest=snapshot_digest,
        target_name=definition.name,
        contract_hash=definition.contract_hash,
        created_at=utcnow(),
        cutoff_time=cutoff_time,
        source_watermarks=source_watermarks,
        telemetry_source_id=telemetry_source_id,
        records_digest=records_digest,
        split_manifest_digest=split_digest,
        reference_ledger_digest=ledger_digest,
        train_count=len(by_split["train"]),
        validation_count=len(by_split["validation"]),
        test_count=len(by_split["test"]),
        storage_uri=str(snapshot_dir),
    )
    write_json(snapshot_dir / "snapshot.json", asdict(snapshot))
    return snapshot


def build_snapshot(
    definition: TargetDefinition,
    contract: TargetRuntimeContract,
    data_config: DataConfig,
    telemetry_source: TelemetryDataSource | None,
    consumed_manifests: list[ConsumedRowsManifest],
    broker: ReferenceBroker,
    cutoff_time: datetime,
    snapshot_options: SnapshotOptions,
) -> SnapshotBuildResult:
    source_records = collect_source_records(definition, data_config, telemetry_source, cutoff_time)
    validated = validate_source_records(contract, source_records)
    qualified = qualify_reference_baseline(
        definition,
        contract,
        validated,
        broker,
        snapshot_options.qualification_options,
    )
    if qualified.status == "fail" or (
        qualified.status == "insufficient" and not snapshot_options.allow_insufficient_reference
    ):
        raise SnapshotBuildError(
            f"reference qualification {qualified.status}",
            reference_qualification=qualified,
        )
    labeled, ledger, failure_report = reference_missing_outputs(
        contract,
        validated,
        broker,
        snapshot_options.reference_budget,
    )
    if failure_report.failures:
        raise SnapshotBuildError(
            "reference labeling failed",
            reference_qualification=qualified,
            reference_failure_report=failure_report,
            reference_usage=ledger,
        )
    deduped = deduplicate_records(contract, labeled, snapshot_options.on_duplicate_conflict)
    grouped = assign_split_groups(contract, deduped)
    split_manifest = build_split_plan(contract, grouped, snapshot_options, consumed_manifests)
    snapshot_records = materialize_snapshot_records(contract, deduped, split_manifest)
    watermarks: dict[str, str] = {}
    for index, source in enumerate(data_config.sources):
        payload = dict(source)
        if source.get("path"):
            source_path = (definition.target_path / source["path"]).resolve()
            payload["path_digest"] = file_digest(source_path)
        watermarks[source.get("name", f"source-{index}")] = stable_hash(payload)
    if telemetry_source is not None:
        telemetry_payload: dict[str, Any] = {
            "source_id": telemetry_source.source_id,
            "cutoff_time": telemetry_source.cutoff_time.isoformat(),
            "records_digest": file_digest(Path(telemetry_source.records_uri)),
            "included_sources": telemetry_source.included_sources,
            "provenance_digest": telemetry_source.provenance_digest,
        }
        if telemetry_source.per_record_split_eligibility_uri:
            telemetry_payload["per_record_split_eligibility_digest"] = file_digest(
                Path(telemetry_source.per_record_split_eligibility_uri)
            )
        watermarks[f"telemetry:{telemetry_source.source_id}"] = stable_hash(telemetry_payload)
    snapshot = write_snapshot(
        definition,
        snapshot_records,
        split_manifest,
        ledger,
        cutoff_time,
        watermarks,
        telemetry_source.source_id if telemetry_source else None,
        snapshot_options.storage_root,
    )
    return SnapshotBuildResult(
        snapshot=snapshot,
        train_view=load_snapshot_view(snapshot, "train", "raw", requester="core"),
        reference_usage=ledger,
        reference_qualification=qualified,
        failure_report=failure_report,
    )


def load_snapshot_view(
    snapshot: Snapshot,
    split: SplitName,
    redaction_level: str,
    *,
    requester: Literal["core", "agent_workspace", "candidate_evaluation"],
) -> SnapshotView:
    if requester == "agent_workspace" and split != "train":
        raise SnapshotBuildError("agent workspace can only access train snapshot views")
    if requester == "candidate_evaluation" and split not in {"validation", "test"}:
        raise SnapshotBuildError("candidate evaluation can only access holdout snapshot views")
    if redaction_level != "raw":
        raise SnapshotBuildError("load_snapshot_view only returns raw snapshot views")
    path = Path(snapshot.storage_uri) / f"{split}.json"
    records = read_json(path)
    return SnapshotView(
        snapshot_id=snapshot.snapshot_id,
        split=split,
        records_uri=str(path),
        redaction_level=redaction_level,  # type: ignore[arg-type]
        record_count=len(records),
    )


def load_snapshot_records(view: SnapshotView) -> list[SnapshotRecord]:
    records = read_json(Path(view.records_uri))
    return [
        SnapshotRecord(
            snapshot_record_id=r["snapshot_record_id"],
            normalized_input_key=r["normalized_input_key"],
            split_group_key=r["split_group_key"],
            input=r["input"],
            reference_output=r["reference_output"],
            reference_source=r["reference_source"],
            split_eligibility=r["split_eligibility"],
            reference_version=r["reference_version"],
            slice_tags=r["slice_tags"],
            source_provenance=r["source_provenance"],
        )
        for r in records
    ]


def export_train_view_for_agent(
    snapshot: Snapshot,
    contract: TargetRuntimeContract,
    agent_view_options: AgentViewOptions,
    output_dir: Path,
) -> TrainViewManifest:
    view = load_snapshot_view(
        snapshot,
        "train",
        "raw",
        requester="agent_workspace",
    )
    records = load_snapshot_records(view)
    output_dir.mkdir(parents=True, exist_ok=True)
    output: list[dict[str, Any]] = []
    for record in records:
        if agent_view_options.redaction_level == "redacted":
            row = {
                "snapshot_record_id": record.snapshot_record_id,
                "input": contract.redact_for_trace(record.input),
                "reference_output": contract.redact_for_trace(record.reference_output),
                "reference_source": record.reference_source,
                "reference_version": record.reference_version,
                "split_eligibility": record.split_eligibility,
            }
        else:
            row = asdict(record)
        output.append(row)
    path = output_dir / "train.json"
    write_json(path, output)
    export_digest = stable_hash(
        {
            "view_kind": "agent_train_export",
            "snapshot_id": snapshot.snapshot_id,
            "snapshot_digest": snapshot.snapshot_digest,
            "view_digest": file_digest(path),
            "record_count": len(output),
            "redaction_level": agent_view_options.redaction_level,
        }
    )
    return TrainViewManifest(
        snapshot_id=snapshot.snapshot_id,
        snapshot_digest=snapshot.snapshot_digest,
        view_path=path,
        record_count=len(output),
        redaction_level=agent_view_options.redaction_level,
        export_digest=export_digest,
    )


def mark_consumed_holdout_rows(
    snapshot: Snapshot,
    split: str,
    records: list[str],
    reason: str,
    visible_to: str,
) -> ConsumedRowsManifest:
    if split not in {"validation", "test"}:
        raise SnapshotBuildError("only validation or test holdout rows can be consumed")
    if visible_to not in {"user", "agent", "external_report"}:
        raise SnapshotBuildError("visible_to must be user, agent, or external_report")
    all_records = {
        r.snapshot_record_id: r
        for r in load_snapshot_records(
            load_snapshot_view(snapshot, split, "raw", requester="core")
        )
    }
    missing = [record_id for record_id in records if record_id not in all_records]
    if missing:
        raise SnapshotBuildError(f"unknown snapshot records: {missing}")
    selected = [all_records[record_id] for record_id in records]
    return ConsumedRowsManifest(
        snapshot_id=snapshot.snapshot_id,
        split=split,  # type: ignore[arg-type]
        record_ids=[record.snapshot_record_id for record in selected],
        normalized_input_keys=[record.normalized_input_key for record in selected],
        split_group_keys=[record.split_group_key for record in selected],
        reason=reason,
        consumed_at=utcnow(),
        visible_to=visible_to,  # type: ignore[arg-type]
        replacement_required=True,
    )
