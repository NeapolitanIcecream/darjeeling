from __future__ import annotations

import random
import tempfile
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Literal

import yaml

from darjeeling.artifact_worker import (
    ArtifactWorkerClient,
    build_planned_private_worker_request,
    run_healthcheck,
    run_layer_attempt,
    start_worker,
)
from darjeeling.errors import EvaluationError
from darjeeling.model import (
    AgentFeedback,
    AgentUsageLedger,
    AgentVisibleDecisionSummary,
    AgentVisibleReport,
    ArtifactPackage,
    Candidate,
    CandidateDecision,
    CandidateSubmission,
    ClosedAgentAttempt,
    ConsumedRowsManifest,
    CostLedger,
    GeneralizationSummary,
    HoldoutConsumptionSummary,
    LayerName,
    MetricSummary,
    ReferenceQualificationReport,
    ReferenceSourceMetricSummary,
    Release,
    ReleaseBaseline,
    Report,
    RequirementCheckResult,
    RoutingSettings,
    RuntimeContext,
    Snapshot,
    SnapshotRecord,
    TargetDefinition,
    TargetRuntimeContract,
    WorkerLimits,
)
from darjeeling.snapshot_reference import (
    load_snapshot_records,
    load_snapshot_view,
    mark_consumed_holdout_rows,
)
from darjeeling.util import file_digest, new_id, safe_public_error, stable_hash, utcnow

_BASELINE_DIRS = ["scaffolding", "runtime", "proposals", "journal", "tests"]
_ALLOWED_LAYERS = {"L1", "L2", "L3"}
_FORBIDDEN_CANDIDATE_KEYS = {
    "contract",
    "contract_hash",
    "evaluator",
    "reference",
    "registry",
    "release",
    "snapshot",
    "split",
    "target",
}
_DEFAULT_FAULT_SCENARIOS = [
    {"kind": "malformed_response"},
    {"kind": "invalid_output"},
    {"kind": "timeout"},
    {"kind": "crash"},
    {"kind": "l4_fallback_failure"},
]


def _closed_attempt_baseline_digest(path: Path) -> str:
    entries: list[tuple[str, str]] = []
    for root_name in _BASELINE_DIRS:
        root = path / root_name
        if not root.exists():
            continue
        for item in sorted(root.rglob("*")):
            rel = item.relative_to(path).as_posix()
            if rel == "journal/closed.json":
                continue
            if item.is_file() and not item.is_symlink():
                entries.append((rel, file_digest(item)))
            elif item.is_symlink():
                raise EvaluationError("closed attempt workspace must not contain symlinks")
    return stable_hash(entries)


def _layer_dir(submission: CandidateSubmission, layer: LayerName) -> Path:
    return submission.submission_path / "artifacts" / layer.lower()


def _read_candidate_routing(submission: CandidateSubmission) -> RoutingSettings:
    path = submission.submission_path / "candidate.yaml"
    if not path.exists():
        return RoutingSettings()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise EvaluationError("candidate.yaml must contain a mapping")
    forbidden = sorted(_FORBIDDEN_CANDIDATE_KEYS & set(raw))
    if forbidden:
        raise EvaluationError(f"candidate.yaml may not set Core-owned fields: {forbidden}")
    routing = raw.get("routing", {})
    if not isinstance(routing, dict):
        raise EvaluationError("candidate routing must be a mapping")
    allowed_routing = set(RoutingSettings.__dataclass_fields__)
    unknown = sorted(set(routing) - allowed_routing)
    if unknown:
        raise EvaluationError(f"unknown candidate routing fields: {unknown}")
    cache_enabled = routing.get("cache_enabled", False)
    if not isinstance(cache_enabled, bool):
        raise EvaluationError("candidate routing cache_enabled must be boolean")
    enabled_layers = routing.get("enabled_layers", ["L1", "L2", "L3"])
    if not isinstance(enabled_layers, list) or any(
        not isinstance(layer, str) for layer in enabled_layers
    ):
        raise EvaluationError("candidate routing enabled_layers must be list[str]")
    if any(layer not in _ALLOWED_LAYERS for layer in enabled_layers):
        raise EvaluationError("candidate routing enabled_layers contains unknown layer")
    if len(set(enabled_layers)) != len(enabled_layers):
        raise EvaluationError("candidate routing enabled_layers must not contain duplicates")

    def _optional_positive_int(name: str) -> int | None:
        value = routing.get(name)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise EvaluationError(f"candidate routing {name} must be a positive integer")
        return value

    total_deadline_ms = routing.get("total_deadline_ms", 1000)
    if (
        isinstance(total_deadline_ms, bool)
        or not isinstance(total_deadline_ms, int)
        or total_deadline_ms <= 0
    ):
        raise EvaluationError("candidate routing total_deadline_ms must be a positive integer")
    circuit_breaker = routing.get("circuit_breaker", {})
    audit = routing.get("audit", {})
    if not isinstance(circuit_breaker, dict):
        raise EvaluationError("candidate routing circuit_breaker must be a mapping")
    if not isinstance(audit, dict):
        raise EvaluationError("candidate routing audit must be a mapping")
    return RoutingSettings(
        cache_enabled=cache_enabled,
        enabled_layers=enabled_layers,  # type: ignore[arg-type]
        L1_timeout_ms=_optional_positive_int("L1_timeout_ms"),
        L2_timeout_ms=_optional_positive_int("L2_timeout_ms"),
        L3_timeout_ms=_optional_positive_int("L3_timeout_ms"),
        total_deadline_ms=total_deadline_ms,
        circuit_breaker=dict(circuit_breaker),
        audit=dict(audit),
    )


def freeze_candidate(
    submission: CandidateSubmission,
    base_release: Release,
    definition: TargetDefinition,
    artifact_store: Path | ArtifactWorkerClient,
    source_snapshot_digest: str,
    source_snapshot_id: str = "",
) -> Candidate:
    worker_client = (
        artifact_store
        if isinstance(artifact_store, ArtifactWorkerClient)
        else ArtifactWorkerClient(artifact_store)
    )
    artifacts: dict[LayerName, ArtifactPackage | None] = dict(base_release.artifacts)
    for layer in submission.declared_layers:
        artifacts[layer] = worker_client.freeze(
            _layer_dir(submission, layer),
            layer,
            definition.contract_hash,
            source_snapshot_digest,
        )
    for layer in ["L1", "L2", "L3"]:
        artifacts.setdefault(layer, None)  # type: ignore[arg-type]
    routing = _read_candidate_routing(submission)
    candidate_id = new_id("candidate")
    digest = stable_hash(
        {
            "submission_id": submission.submission_id,
            "workspace_commit": submission.workspace_commit,
            "target_name": definition.name,
            "contract_hash": definition.contract_hash,
            "snapshot_id": source_snapshot_id,
            "source_snapshot_digest": source_snapshot_digest,
            "base_release_id": base_release.release_id,
            "artifacts": {
                layer: package.digest if package else None for layer, package in artifacts.items()
            },
            "routing": routing,
        }
    )
    return Candidate(
        candidate_id=candidate_id,
        submission_id=submission.submission_id,
        compile_id=submission.compile_id,
        attempt_id=submission.attempt_id,
        target_name=definition.name,
        contract_hash=definition.contract_hash,
        snapshot_id=source_snapshot_id,
        base_release_id=base_release.release_id,
        workspace_commit=submission.workspace_commit,
        artifacts=artifacts,
        routing=routing,
        digest=digest,
        status="frozen",
    )


def validate_candidate_manifest(
    candidate: Candidate, definition: TargetDefinition, base_release: Release
) -> dict[str, Any]:
    failures: list[str] = []
    if (
        candidate.target_name != definition.name
        or candidate.contract_hash != definition.contract_hash
    ):
        failures.append("candidate target or contract does not match active definition")
    if (
        base_release.target_name != definition.name
        or base_release.contract_hash != definition.contract_hash
    ):
        failures.append("base release scope mismatch")
    if candidate.base_release_id != base_release.release_id:
        failures.append("candidate base release does not match supplied base release")
    if set(candidate.artifacts) != _ALLOWED_LAYERS:
        failures.append("candidate artifacts must have exactly L1, L2, and L3 keys")
    for layer in candidate.routing.enabled_layers:
        if layer not in _ALLOWED_LAYERS:
            failures.append("candidate routing enabled_layers contains unknown layer")
    for timeout_name in ["L1_timeout_ms", "L2_timeout_ms", "L3_timeout_ms", "total_deadline_ms"]:
        timeout = getattr(candidate.routing, timeout_name)
        if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
            if timeout_name == "total_deadline_ms" or timeout is not None:
                failures.append(f"candidate routing {timeout_name} must be a positive integer")
    for layer, package in candidate.artifacts.items():
        if layer not in _ALLOWED_LAYERS:
            failures.append(f"unknown artifact layer key: {layer}")
            continue
        if package is not None and package.manifest.contract_hash != definition.contract_hash:
            failures.append(f"{layer} artifact contract hash mismatch")
        if package is not None and package.layer != layer:
            failures.append(f"{layer} artifact package layer mismatch")
        base_package = (
            base_release.artifacts.get(layer) if layer in base_release.artifacts else None
        )
        if (
            package is not None
            and base_package is not None
            and package.artifact_id == base_package.artifact_id
        ):
            inherited_fields = [
                "layer",
                "digest",
                "source_snapshot_digest",
                "package_path",
                "manifest",
            ]
            for field_name in inherited_fields:
                if getattr(package, field_name) != getattr(base_package, field_name):
                    failures.append(f"{layer} inherited artifact {field_name} mismatch")
    return {"status": "fail" if failures else "pass", "failures": failures}


def run_protocol_preflight(
    candidate: Candidate,
    contract: TargetRuntimeContract,
    artifact_worker: ArtifactWorkerClient | None = None,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    failures: list[str] = []
    try:
        preflight_input = contract.validate_input(_minimal_input_for_schema(contract.input_schema))
    except Exception as exc:
        return {
            "status": "fail",
            "results": {},
            "failures": [f"protocol preflight input generation failed: {exc}"],
        }
    for layer, package in candidate.artifacts.items():
        if layer not in candidate.routing.enabled_layers:
            continue
        if package is None:
            continue
        worker = start_worker(package, worker_limits=WorkerLimits())
        health = run_healthcheck(worker, package.manifest.timeout_ms)
        layer_result: dict[str, Any] = {"healthcheck": asdict(health)}
        if health.status != "pass":
            failures.append(f"{layer} healthcheck failed: {health.message}")
            results[layer] = layer_result
            continue
        request_id = f"eval-preflight-{stable_hash((candidate.candidate_id, layer))[:16]}"
        deadline_ms = min(
            getattr(candidate.routing, f"{layer}_timeout_ms")
            or package.manifest.timeout_ms,
            candidate.routing.total_deadline_ms,
            package.manifest.timeout_ms,
        )
        request = build_planned_private_worker_request(
            request_id,
            preflight_input,
            deadline_ms,
        )
        attempt = run_layer_attempt(worker, contract, request, layer)  # type: ignore[arg-type]
        layer_result["minimal_protocol_attempt"] = asdict(attempt)
        if attempt.decision not in {"accept", "abstain"}:
            failures.append(
                f"{layer} protocol preflight failed: {attempt.decision}"
                + (f" ({attempt.error})" if attempt.error else "")
            )
        results[layer] = layer_result
    return {"status": "fail" if failures else "pass", "results": results, "failures": failures}


def _minimal_input_for_schema(schema: dict[str, Any]) -> Any:
    if "const" in schema:
        return schema["const"]
    if "enum" in schema and isinstance(schema["enum"], list) and schema["enum"]:
        return schema["enum"][0]
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        non_null = [item for item in schema_type if item != "null"]
        schema_type = non_null[0] if non_null else "null"
    if schema_type == "object" or "properties" in schema:
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(properties, dict):
            properties = {}
        if not isinstance(required, list):
            required = []
        return {
            name: _minimal_input_for_schema(properties.get(name, {}))
            for name in required
            if isinstance(name, str)
        }
    if schema_type == "array":
        min_items = schema.get("minItems", 0)
        item_schema = schema.get("items", {})
        return [
            _minimal_input_for_schema(item_schema if isinstance(item_schema, dict) else {})
            for _ in range(min_items if isinstance(min_items, int) and min_items > 0 else 0)
        ]
    if schema_type == "integer":
        minimum = schema.get("minimum", 0)
        return int(minimum if isinstance(minimum, int | float) else 0)
    if schema_type == "number":
        minimum = schema.get("minimum", 0.0)
        return float(minimum if isinstance(minimum, int | float) else 0.0)
    if schema_type == "boolean":
        return False
    if schema_type == "null":
        return None
    return "preflight"


def build_private_evaluation_request_plan(
    records: Any, evaluation_id: str, random_seed: int
) -> dict[str, Any]:
    if not isinstance(records, list):
        records = load_snapshot_records(records)
    order = list(records)
    rng = random.Random(random_seed)
    rng.shuffle(order)
    salt = stable_hash((evaluation_id, random_seed, utcnow().isoformat()))
    request_ids = {
        record.snapshot_record_id: f"eval-{stable_hash((salt, index))[:20]}"
        for index, record in enumerate(order)
    }
    return {
        "evaluation_id": evaluation_id,
        "order": [record.snapshot_record_id for record in order],
        "request_ids": request_ids,
        "request_order_digest": stable_hash([record.snapshot_record_id for record in order]),
        "ephemeral_request_id_salt_digest": stable_hash(salt),
    }


def _run_layers_for_record(
    candidate: Candidate,
    record: SnapshotRecord,
    contract: TargetRuntimeContract,
    request_id: str,
) -> tuple[LayerName | None, dict[str, Any] | None, list[Any], float]:
    attempts = []
    total_latency = 0.0
    route_started_at = time.perf_counter()
    for layer in ["L1", "L2", "L3"]:
        if layer not in candidate.routing.enabled_layers:
            continue
        package = candidate.artifacts.get(layer)  # type: ignore[arg-type]
        if package is None:
            continue
        worker = start_worker(package, WorkerLimits())
        request = build_planned_private_worker_request(
            request_id,
            record.input,
            _candidate_layer_timeout_ms(candidate, layer, package, route_started_at),
        )
        attempt = run_layer_attempt(worker, contract, request, layer)  # type: ignore[arg-type]
        attempts.append(attempt)
        total_latency += attempt.latency_ms
        if attempt.decision == "accept" and attempt.output is not None:
            return layer, attempt.output, attempts, total_latency
    return None, None, attempts, total_latency


def _candidate_layer_timeout_ms(
    candidate: Candidate,
    layer: str,
    package: ArtifactPackage,
    route_started_at: float,
) -> int:
    elapsed_ms = int((time.perf_counter() - route_started_at) * 1000)
    remaining_route_ms = max(1, candidate.routing.total_deadline_ms - elapsed_ms)
    return min(
        remaining_route_ms,
        getattr(candidate.routing, f"{layer}_timeout_ms") or package.manifest.timeout_ms,
        package.manifest.timeout_ms,
    )


def evaluate_standalone_layer(
    candidate: Candidate,
    layer: LayerName,
    records: Any,
    contract: TargetRuntimeContract,
) -> dict[str, Any]:
    snapshot_records = load_snapshot_records(records) if not isinstance(records, list) else records
    package = candidate.artifacts.get(layer)
    if package is None:
        return {"layer": layer, "attempts": [], "sample_count": len(snapshot_records)}
    results = []
    plan = build_private_evaluation_request_plan(snapshot_records, new_id("eval"), 0)
    records_by_id = {record.snapshot_record_id: record for record in snapshot_records}
    for snapshot_record_id in plan["order"]:
        record = records_by_id[snapshot_record_id]
        worker = start_worker(package, WorkerLimits())
        route_started_at = time.perf_counter()
        request = build_planned_private_worker_request(
            plan["request_ids"][record.snapshot_record_id],
            record.input,
            _candidate_layer_timeout_ms(candidate, layer, package, route_started_at),
        )
        results.append((record, run_layer_attempt(worker, contract, request, layer)))
    return {"layer": layer, "results": results, "sample_count": len(snapshot_records), "plan": plan}


def evaluate_residual_layer(
    candidate: Candidate,
    layer: LayerName,
    residual_records: Any,
    upstream_results: list[Any],
    contract: TargetRuntimeContract,
) -> dict[str, Any]:
    snapshot_records = (
        load_snapshot_records(residual_records)
        if not isinstance(residual_records, list)
        else residual_records
    )
    accepted_record_ids: set[str] = set()
    for item in upstream_results:
        if isinstance(item, dict):
            record = item.get("record")
            if record is not None and item.get("chosen_layer") is not None:
                accepted_record_ids.add(record.snapshot_record_id)
        elif isinstance(item, tuple) and len(item) == 2:
            record, attempt = item
            if getattr(attempt, "decision", None) == "accept":
                accepted_record_ids.add(record.snapshot_record_id)
    records_reaching_layer = [
        record
        for record in snapshot_records
        if record.snapshot_record_id not in accepted_record_ids
    ]
    result = evaluate_standalone_layer(candidate, layer, records_reaching_layer, contract)
    result["upstream_accepted_count"] = len(accepted_record_ids)
    result["residual_record_count"] = len(records_reaching_layer)
    return result


def evaluate_full_cascade(
    candidate: Candidate,
    records: Any,
    contract: TargetRuntimeContract,
    reference_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reference_policy = reference_policy or {}
    fallback_cost = float(reference_policy.get("fallback_cost", 0.01))
    fallback_latency_ms = float(reference_policy.get("fallback_latency_ms", 3.0))
    snapshot_records = load_snapshot_records(records) if not isinstance(records, list) else records
    evaluation_id = new_id("eval")
    plan = build_private_evaluation_request_plan(snapshot_records, evaluation_id, 0)
    rows: list[dict[str, Any]] = []
    fallback_count = 0
    records_by_id = {record.snapshot_record_id: record for record in snapshot_records}
    for snapshot_record_id in plan["order"]:
        record = records_by_id[snapshot_record_id]
        chosen_layer, output, attempts, latency = _run_layers_for_record(
            candidate,
            record,
            contract,
            plan["request_ids"][record.snapshot_record_id],
        )
        is_correct = (
            contract.is_correct(output, record.reference_output) if output is not None else None
        )
        fallback_output = None
        row_fallback_cost = 0.0
        row_fallback_latency_ms = 0.0
        if chosen_layer is None:
            fallback_count += 1
            fallback_output = record.reference_output
            row_fallback_cost = fallback_cost
            row_fallback_latency_ms = fallback_latency_ms
            latency += row_fallback_latency_ms
        rows.append(
            {
                "record": record,
                "chosen_layer": chosen_layer,
                "output": output,
                "fallback_output": fallback_output,
                "fallback_status": "ok" if fallback_output is not None else None,
                "fallback_cost": row_fallback_cost,
                "fallback_latency_ms": row_fallback_latency_ms,
                "attempts": attempts,
                "is_correct": is_correct,
                "latency_ms": latency,
                "reference_source": record.reference_source,
            }
        )
    return {
        "evaluation_id": evaluation_id,
        "rows": rows,
        "sample_count": len(snapshot_records),
        "l4_fallback_count": fallback_count,
        "l4_fallback_share": fallback_count / len(snapshot_records)
        if snapshot_records
        else 0.0,
        "l4_fallback_cost": fallback_count * fallback_cost,
        "l4_fallback_latency_ms": fallback_count * fallback_latency_ms,
        "plan": plan,
    }


def evaluate_changed_layer_ablation(
    candidate: Candidate,
    baseline_release: Release,
    changed_layers: list[LayerName],
    records: Any,
    contract: TargetRuntimeContract,
) -> dict[str, Any]:
    changed = set(changed_layers)
    changed_only_artifacts = {
        layer: package if layer in changed else None
        for layer, package in candidate.artifacts.items()
    }
    with_baseline_artifacts = {
        layer: candidate.artifacts[layer]
        if layer in changed
        else baseline_release.artifacts.get(layer)
        for layer in ["L1", "L2", "L3"]
    }
    changed_only = replace(
        candidate,
        artifacts=changed_only_artifacts,  # type: ignore[arg-type]
        routing=replace(
            candidate.routing,
            enabled_layers=[layer for layer in ["L1", "L2", "L3"] if layer in changed],
        ),
    )
    with_baseline = replace(
        candidate,
        artifacts=with_baseline_artifacts,  # type: ignore[arg-type]
    )
    return {
        "changed_layers": changed_layers,
        "baseline_release_id": baseline_release.release_id,
        "changed_only": evaluate_full_cascade(changed_only, records, contract),
        "with_baseline_unchanged": evaluate_full_cascade(with_baseline, records, contract),
        "cascade": evaluate_full_cascade(candidate, records, contract),
    }


def _fault_worker_source(kind: str) -> str:
    if kind == "malformed_response":
        return "print('not-json')\n"
    if kind == "invalid_output":
        return (
            "import json\n"
            "import sys\n"
            "json.loads(sys.stdin.readline())\n"
            "print(json.dumps({'decision': 'accept', 'output': {'bad': 'shape'}, "
            "'reason': 'fault'}))\n"
        )
    if kind == "timeout":
        return (
            "import json\n"
            "import sys\n"
            "import time\n"
            "json.loads(sys.stdin.readline())\n"
            "time.sleep(0.2)\n"
            "print(json.dumps({'decision': 'abstain', 'reason': 'fault'}))\n"
        )
    if kind == "crash":
        return "raise SystemExit(2)\n"
    raise EvaluationError(f"unknown fault scenario kind: {kind}")


def _run_local_fault_scenario(
    candidate: Candidate,
    layer: LayerName,
    kind: str,
    record: SnapshotRecord,
    request_id: str,
    contract: TargetRuntimeContract,
) -> dict[str, Any]:
    expected = {
        "malformed_response": "protocol_error",
        "invalid_output": "invalid_output",
        "timeout": "timeout",
        "crash": "error",
    }[kind]
    base_package = candidate.artifacts[layer]
    if base_package is None:
        return {"kind": kind, "layer": layer, "status": "skipped", "reason": "no package"}
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        artifact_dir = root / "artifact"
        artifact_dir.mkdir()
        (artifact_dir / "worker.py").write_text(_fault_worker_source(kind), encoding="utf-8")
        (artifact_dir / "healthcheck.py").write_text("raise SystemExit(0)\n", encoding="utf-8")
        timeout_ms = 50 if kind == "timeout" else min(base_package.manifest.timeout_ms, 1000)
        (artifact_dir / "artifact.yaml").write_text(
            yaml.safe_dump(
                {
                    "api_version": "v1",
                    "layer": layer,
                    "start_command": ["python3", "worker.py"],
                    "healthcheck_command": ["python3", "healthcheck.py"],
                    "protocol": "jsonl",
                    "timeout_ms": timeout_ms,
                    "memory_mb": base_package.manifest.memory_mb,
                    "network": "disabled",
                    "contract_hash": candidate.contract_hash,
                    "allowed_reason_codes": ["fault"],
                }
            ),
            encoding="utf-8",
        )
        fault_package = ArtifactWorkerClient(root / "store").freeze(
            artifact_dir,
            layer,
            candidate.contract_hash,
            base_package.source_snapshot_digest,
        )
        artifacts = dict(candidate.artifacts)
        artifacts[layer] = fault_package
        fault_candidate = replace(
            candidate,
            artifacts=artifacts,
            routing=replace(candidate.routing, enabled_layers=[layer]),
        )
        chosen_layer, _output, attempts, _latency = _run_layers_for_record(
            fault_candidate,
            record,
            contract,
            request_id,
        )
    first_attempt = attempts[0] if attempts else None
    passed = (
        chosen_layer is None
        and first_attempt is not None
        and first_attempt.decision == expected
    )
    return {
        "kind": kind,
        "layer": layer,
        "status": "pass" if passed else "fail",
        "request_id": request_id,
        "expected_decision": expected,
        "observed_decision": first_attempt.decision if first_attempt else None,
    }


def evaluate_fault_fallback(
    candidate: Candidate,
    fault_scenarios: list[dict[str, Any]],
    records: Any,
    contract: TargetRuntimeContract,
) -> dict[str, Any]:
    snapshot_records = load_snapshot_records(records) if not isinstance(records, list) else records
    if not snapshot_records:
        return {"scenarios": fault_scenarios, "status": "fail", "failures": ["no records"]}
    scenarios = fault_scenarios or list(_DEFAULT_FAULT_SCENARIOS)
    results: list[dict[str, Any]] = []
    plan = build_private_evaluation_request_plan(snapshot_records, new_id("eval"), 0)
    planned_record_id = plan["order"][0]
    records_by_id = {record.snapshot_record_id: record for record in snapshot_records}
    record = records_by_id[planned_record_id]
    request_id = plan["request_ids"][planned_record_id]
    layers = [
        layer
        for layer in ["L1", "L2", "L3"]
        if layer in candidate.routing.enabled_layers and candidate.artifacts.get(layer) is not None
    ]
    for scenario in scenarios:
        kind = scenario.get("kind")
        if kind == "l4_fallback_failure":
            from darjeeling.release_runtime import call_l4_fallback

            class _FailingFallbackBroker:
                reference_version = "fault-fallback"

                def call(self, request: dict[str, Any], context: Any) -> Any:
                    raise RuntimeError("provider secret leaked")

            fallback = call_l4_fallback(
                contract,
                record.input,
                _FailingFallbackBroker(),
                RuntimeContext(request_id, 1, _FailingFallbackBroker()),
            )
            public_error = safe_public_error("l4_fallback_failure")
            passed = (
                fallback.status == "error"
                and fallback.error_message_hash is not None
                and "provider secret" not in public_error
            )
            results.append(
                {
                    "kind": kind,
                    "status": "pass" if passed else "fail",
                    "reason": "safe_error_exercised",
                    "fallback_status": fallback.status,
                    "error_message_hash": fallback.error_message_hash,
                    "public_error_message": public_error,
                }
            )
            continue
        if kind not in {"malformed_response", "invalid_output", "timeout", "crash"}:
            results.append({"kind": kind, "status": "fail", "reason": "unknown scenario"})
            continue
        if not layers:
            results.append({"kind": kind, "status": "skipped", "reason": "no local artifacts"})
            continue
        for layer in layers:
            results.append(
                _run_local_fault_scenario(candidate, layer, kind, record, request_id, contract)
            )
    failures = [
        result
        for result in results
        if result["status"] not in {"pass", "skipped"}
    ]
    return {
        "scenarios": results,
        "status": "fail" if failures else "pass",
        "failures": failures,
        "plan": plan,
    }



def measure_latency_and_cost(
    candidate: Candidate,
    records: Any,
    contract: TargetRuntimeContract,
    measurement_options: dict[str, Any],
) -> dict[str, Any]:
    cascade = evaluate_full_cascade(candidate, records, contract, measurement_options)
    latencies = [row["latency_ms"] for row in cascade["rows"]]
    sorted_latencies = sorted(latencies)
    p50 = sorted_latencies[int(0.50 * (len(latencies) - 1))] if latencies else 0.0
    p95 = sorted_latencies[int(0.95 * (len(latencies) - 1))] if latencies else 0.0
    p99 = sorted_latencies[int(0.99 * (len(latencies) - 1))] if latencies else 0.0
    local_compute_cost = len(latencies) * 0.000001
    fallback_cost = float(cascade.get("l4_fallback_cost", 0.0))
    cascade_cost = local_compute_cost + fallback_cost
    throughput = 1000.0 / p95 if p95 > 0 else None
    memory_values = [
        package.manifest.memory_mb or 0
        for layer, package in candidate.artifacts.items()
        if package is not None
        and layer in candidate.routing.enabled_layers
    ]
    return {
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
        "p99_latency_ms": p99,
        "throughput_per_second": throughput,
        "memory_mb": max(memory_values, default=0),
        "serving_local_compute_cost": local_compute_cost,
        "l4_fallback_share": cascade.get("l4_fallback_share", 0.0),
        "l4_fallback_cost": fallback_cost,
        "cascade_cost": cascade_cost,
        "sample_count": len(latencies),
    }


def build_cost_ledger(
    latency_cost: dict[str, Any],
    agent_usage: AgentUsageLedger,
    reference_usage: Any,
    audit_usage: Any | None,
    local_training_search_usage: Any | None,
    baseline_cost: dict[str, Any],
) -> CostLedger:
    compile_cost = (
        agent_usage.cost
        + getattr(reference_usage, "cost", 0.0)
        + float(
            (local_training_search_usage or {}).get("cost", 0.0)
            if isinstance(local_training_search_usage, dict)
            else 0.0
        )
    )
    serving_l4 = float(baseline_cost.get("serving_l4_cost", 0.0))
    local_compute_cost = float(latency_cost.get("serving_local_compute_cost", 0.0))
    fallback_cost = float(latency_cost.get("l4_fallback_cost", 0.0))
    candidate_serving_cost = float(
        latency_cost.get("cascade_cost", local_compute_cost + fallback_cost)
    )
    saving = serving_l4 - candidate_serving_cost if serving_l4 else None
    payback = int(compile_cost / saving * 1000) if saving and saving > 0 else None
    return CostLedger(
        serving_l4_cost=serving_l4,
        serving_local_compute_cost=local_compute_cost,
        random_audit_cost=float(
            (audit_usage or {}).get("random_audit_cost", 0.0)
            if isinstance(audit_usage, dict)
            else 0.0
        ),
        risk_audit_cost=float(
            (audit_usage or {}).get("risk_audit_cost", 0.0)
            if isinstance(audit_usage, dict)
            else 0.0
        ),
        compile_agent_cost=agent_usage.cost,
        reference_labeling_cost=getattr(reference_usage, "cost", 0.0),
        local_training_search_cost=float(
            (local_training_search_usage or {}).get("cost", 0.0)
            if isinstance(local_training_search_usage, dict)
            else 0.0
        ),
        compile_cost=compile_cost,
        saving_per_1000_requests=saving,
        estimated_payback_requests=payback,
        notes=[
            f"local_compute_cost={local_compute_cost}",
            f"l4_fallback_cost={fallback_cost}",
        ],
    )


def compute_metric_summary(
    evaluation_result: dict[str, Any], confidence_options: dict[str, Any] | None = None
) -> MetricSummary:
    rows = evaluation_result["rows"]
    accepted = [row for row in rows if row["chosen_layer"] is not None]
    correct = [row for row in accepted if row["is_correct"] is True]
    wrong = [row for row in accepted if row["is_correct"] is False]
    sample_count = len(rows)
    precision = len(correct) / len(accepted) if accepted else None
    coverage = len(accepted) / sample_count if sample_count else 0.0
    wrong_rate = len(wrong) / len(accepted) if accepted else 0.0
    return MetricSummary(
        accepted_count=len(accepted),
        correct_accept_count=len(correct),
        wrong_accept_count=len(wrong),
        precision=precision,
        coverage=coverage,
        wrong_accept_rate=wrong_rate,
        precision_lower_bound=max(0.0, (precision or 0.0) - 0.05) if accepted else None,
        wrong_accept_upper_bound=min(1.0, wrong_rate + 0.05),
    )


def compute_reference_source_metrics(
    evaluation_result: dict[str, Any],
    reference_provenance: Any,
    confidence_options: dict[str, Any] | None = None,
) -> list[ReferenceSourceMetricSummary]:
    by_source: dict[str, list[dict[str, Any]]] = {}
    for row in evaluation_result["rows"]:
        by_source.setdefault(row["reference_source"], []).append(row)
    claim_map = {
        "gold": "gold_correctness",
        "human": "human_correctness",
        "versioned_l4": "l4_agreement",
        "verified_l4": "verified_l4_correctness",
        "user_feedback": "user_feedback_correctness",
    }
    summaries: list[ReferenceSourceMetricSummary] = []
    for source, rows in by_source.items():
        metric = compute_metric_summary({"rows": rows}, confidence_options)
        summaries.append(
            ReferenceSourceMetricSummary(
                reference_source=source,  # type: ignore[arg-type]
                sample_count=len(rows),
                metric=metric,
                claim=claim_map[source],  # type: ignore[arg-type]
                notes=["versioned L4 rows are agreement, not gold correctness"]
                if source == "versioned_l4"
                else [],
            )
        )
    return summaries


def _compute_slice_results(evaluation_result: dict[str, Any]) -> list[dict[str, Any]]:
    by_slice: dict[str, dict[str, Any]] = {}
    rows = evaluation_result["rows"]
    for row in rows:
        record = row["record"]
        for tag in record.slice_tags:
            stats = by_slice.setdefault(
                tag,
                {"slice": tag, "sample_count": 0, "accepted_count": 0, "correct_accept_count": 0},
            )
            stats["sample_count"] += 1
            if row["chosen_layer"] is not None:
                stats["accepted_count"] += 1
                if row["is_correct"] is True:
                    stats["correct_accept_count"] += 1
    results: list[dict[str, Any]] = []
    for stats in by_slice.values():
        accepted = stats["accepted_count"]
        sample_count = stats["sample_count"]
        precision = stats["correct_accept_count"] / accepted if accepted else None
        coverage = accepted / sample_count if sample_count else 0.0
        results.append(
            {
                "slice": stats["slice"],
                "sample_count": sample_count,
                "accepted_count": accepted,
                "precision": precision,
                "coverage": coverage,
            }
        )
    return results


def compute_generalization_summary(
    validation_result: dict[str, Any],
    test_result: dict[str, Any] | None,
    slice_results: list[Any],
    requirements: Any,
) -> GeneralizationSummary:
    validation_metric = compute_metric_summary(validation_result)
    test_metric = compute_metric_summary(test_result) if test_result else None
    precision_drop = None
    coverage_retention = None
    if (
        validation_metric.precision is not None
        and test_metric
        and test_metric.precision is not None
    ):
        precision_drop = validation_metric.precision - test_metric.precision
    if validation_metric.coverage:
        coverage_retention = (
            test_metric.coverage if test_metric else 0.0
        ) / validation_metric.coverage
    min_accept = test_metric.accepted_count if test_metric else validation_metric.accepted_count
    min_check = "pass" if min_accept >= requirements.min_accepted_samples else "insufficient"
    if not slice_results:
        slice_results = _compute_slice_results(test_result or validation_result)
    min_slice_count = (
        min((int(item.get("sample_count", 0)) for item in slice_results), default=0)
        if requirements.min_slice_samples > 0
        else 0
    )
    slice_check = (
        "pass"
        if requirements.min_slice_samples == 0
        or (slice_results and min_slice_count >= requirements.min_slice_samples)
        else "insufficient"
    )
    worst_slice = None
    if slice_results:
        worst = min(
            slice_results,
            key=lambda item: (
                item.get("precision") if item.get("precision") is not None else -1.0,
                item.get("coverage", 0.0),
            ),
        )
        worst_slice = {
            "sample_count": int(worst.get("sample_count", 0)),
            "precision": worst.get("precision"),
            "coverage": worst.get("coverage", 0.0),
        }
    critical_stats = [
        item for item in slice_results if item.get("slice") in requirements.critical_slices
    ]
    critical_missing_count = len(
        set(requirements.critical_slices) - {item["slice"] for item in critical_stats}
    )
    critical_min_sample_count = min(
        (int(item.get("sample_count", 0)) for item in critical_stats),
        default=0 if requirements.critical_slices else requirements.min_slice_samples,
    )
    critical_min_precision = min(
        (
            item.get("precision")
            for item in critical_stats
            if item.get("precision") is not None
        ),
        default=None,
    )
    critical_min_coverage = min(
        (float(item.get("coverage", 0.0)) for item in critical_stats),
        default=None,
    )
    cohort_floor = {
        "min_slice_sample_count": min_slice_count,
        "critical_slice_count": len(critical_stats),
        "critical_slice_missing_count": critical_missing_count,
        "critical_slice_min_sample_count": critical_min_sample_count,
        "critical_slice_min_precision": critical_min_precision,
        "critical_slice_min_coverage": critical_min_coverage,
    }
    evidence = "pass"
    if min_check == "insufficient" or slice_check == "insufficient":
        evidence = "insufficient"
    if test_metric is not None and test_metric.wrong_accept_count > 0:
        evidence = "fail"
    if (
        requirements.validation_test_precision_drop_max is not None
        and precision_drop is not None
        and precision_drop > requirements.validation_test_precision_drop_max
    ):
        evidence = "fail"
    if (
        requirements.validation_test_coverage_retention_min is not None
        and coverage_retention is not None
        and coverage_retention < requirements.validation_test_coverage_retention_min
    ):
        evidence = "fail"
    return GeneralizationSummary(
        validation_precision=validation_metric.precision,
        test_precision=test_metric.precision if test_metric else None,
        validation_coverage=validation_metric.coverage,
        test_coverage=test_metric.coverage if test_metric else 0.0,
        precision_drop=precision_drop,
        coverage_retention=coverage_retention,
        cohort_floor=cohort_floor,
        worst_slice=worst_slice,
        min_accepted_sample_check=min_check,  # type: ignore[arg-type]
        min_slice_sample_check=slice_check,  # type: ignore[arg-type]
        evidence_status=evidence,  # type: ignore[arg-type]
    )


def check_candidate_requirements(
    metrics: MetricSummary,
    generalization: GeneralizationSummary,
    latency_cost: dict[str, Any],
    requirements: Any,
) -> list[RequirementCheckResult]:
    results: list[RequirementCheckResult] = []
    precision = (
        metrics.precision_lower_bound
        if metrics.precision_lower_bound is not None
        else metrics.precision
    )
    if metrics.accepted_count < requirements.min_accepted_samples:
        results.append(
            RequirementCheckResult(
                "min_accepted_samples", "insufficient", {"accepted": metrics.accepted_count}
            )
        )
    elif precision is None or precision < requirements.precision_min:
        results.append(
            RequirementCheckResult(
                "precision_min",
                "fail",
                {"precision": precision, "required": requirements.precision_min},
            )
        )
    else:
        results.append(RequirementCheckResult("precision_min", "pass", {"precision": precision}))
    if (
        requirements.wrong_accept_rate_max is not None
        and metrics.wrong_accept_upper_bound is not None
    ):
        status = (
            "fail"
            if metrics.wrong_accept_upper_bound > requirements.wrong_accept_rate_max
            else "pass"
        )
        results.append(
            RequirementCheckResult(
                "wrong_accept_rate_max", status, {"upper_bound": metrics.wrong_accept_upper_bound}
            )
        )
    if requirements.p95_latency_ms_max is not None:
        status = (
            "fail"
            if latency_cost.get("p95_latency_ms", 0) > requirements.p95_latency_ms_max
            else "pass"
        )
        results.append(
            RequirementCheckResult(
                "p95_latency_ms_max", status, {"p95": latency_cost.get("p95_latency_ms")}
            )
        )
    if requirements.memory_mb_max is not None:
        memory_mb = latency_cost.get("memory_mb")
        status = (
            "fail"
            if memory_mb is None or memory_mb > requirements.memory_mb_max
            else "pass"
        )
        results.append(
            RequirementCheckResult(
                "memory_mb_max", status, {"memory_mb": memory_mb}
            )
        )
    if requirements.throughput_per_second_min is not None:
        throughput = latency_cost.get("throughput_per_second")
        status = (
            "fail"
            if throughput is None or throughput < requirements.throughput_per_second_min
            else "pass"
        )
        results.append(
            RequirementCheckResult(
                "throughput_per_second_min",
                status,
                {"throughput_per_second": throughput},
            )
        )
    if requirements.serving_cost_per_1000_max is not None:
        serving_cost = latency_cost.get("cascade_cost")
        status = (
            "fail"
            if serving_cost is None or serving_cost > requirements.serving_cost_per_1000_max
            else "pass"
        )
        results.append(
            RequirementCheckResult(
                "serving_cost_per_1000_max",
                status,
                {"serving_cost_per_1000": serving_cost},
            )
        )
    if requirements.validation_test_precision_drop_max is not None:
        if generalization.precision_drop is None:
            results.append(RequirementCheckResult("precision_drop", "insufficient", {}))
        else:
            status = (
                "fail"
                if generalization.precision_drop > requirements.validation_test_precision_drop_max
                else "pass"
            )
            results.append(
                RequirementCheckResult(
                    "precision_drop",
                    status,
                    {
                        "precision_drop": generalization.precision_drop,
                        "max": requirements.validation_test_precision_drop_max,
                    },
                )
            )
    if requirements.validation_test_coverage_retention_min is not None:
        if generalization.coverage_retention is None:
            results.append(RequirementCheckResult("coverage_retention", "insufficient", {}))
        else:
            status = (
                "fail"
                if generalization.coverage_retention
                < requirements.validation_test_coverage_retention_min
                else "pass"
            )
            results.append(
                RequirementCheckResult(
                    "coverage_retention",
                    status,
                    {
                        "coverage_retention": generalization.coverage_retention,
                        "min": requirements.validation_test_coverage_retention_min,
                    },
                )
            )
    if generalization.min_slice_sample_check != "pass":
        results.append(
            RequirementCheckResult(
                "min_slice_samples",
                generalization.min_slice_sample_check,
                generalization.cohort_floor or {},
            )
        )
    critical = generalization.cohort_floor or {}
    if requirements.critical_slices:
        if critical.get("critical_slice_missing_count", 0) > 0:
            results.append(
                RequirementCheckResult("critical_slices", "insufficient", dict(critical))
            )
        elif critical.get("critical_slice_min_sample_count", 0) < requirements.min_slice_samples:
            results.append(
                RequirementCheckResult("critical_slices", "insufficient", dict(critical))
            )
        else:
            if requirements.critical_slice_precision_min is not None:
                min_precision = critical.get("critical_slice_min_precision")
                status = (
                    "fail"
                    if min_precision is None
                    or min_precision < requirements.critical_slice_precision_min
                    else "pass"
                )
                results.append(
                    RequirementCheckResult(
                        "critical_slice_precision_min",
                        status,
                        dict(critical),
                    )
                )
            if requirements.critical_slice_coverage_min is not None:
                min_coverage = critical.get("critical_slice_min_coverage")
                status = (
                    "fail"
                    if min_coverage is None
                    or min_coverage < requirements.critical_slice_coverage_min
                    else "pass"
                )
                results.append(
                    RequirementCheckResult(
                        "critical_slice_coverage_min",
                        status,
                        dict(critical),
                    )
                )
    if generalization.evidence_status == "insufficient":
        results.append(RequirementCheckResult("generalization", "insufficient", {}))
    elif generalization.evidence_status == "fail":
        results.append(RequirementCheckResult("generalization", "fail", {}))
    else:
        results.append(RequirementCheckResult("generalization", "pass", {}))
    return results


def _build_report(
    stage: Literal["validation", "test"],
    candidate: Candidate,
    definition: TargetDefinition,
    snapshot: Snapshot,
    base_release: Release,
    reference_qualification: ReferenceQualificationReport,
    cascade: dict[str, Any],
    generalization: GeneralizationSummary,
    latency_cost: dict[str, Any],
    cost: CostLedger,
    holdout: ConsumedRowsManifest | None,
    safety: dict[str, Any],
    diagnostics: dict[str, Any],
) -> Report:
    metrics = compute_metric_summary(cascade)
    return Report(
        report_id=new_id("report"),
        report_stage=stage,
        candidate_id=candidate.candidate_id,
        target_name=definition.name,
        contract_hash=definition.contract_hash,
        snapshot_id=snapshot.snapshot_id,
        baseline_release_id=base_release.release_id,
        metrics={
            "local": asdict(metrics),
            "sample_count": cascade["sample_count"],
            "l4_fallback_share": cascade.get("l4_fallback_share", 0.0),
            "l4_fallback_count": cascade.get("l4_fallback_count", 0),
            "routing_enabled_layers": list(candidate.routing.enabled_layers),
        },
        metrics_by_reference_source=compute_reference_source_metrics(cascade, None),
        reference_qualification=reference_qualification,
        generalization=generalization,
        latency=latency_cost,
        cost=cost,
        safety={
            "fault_fallback": _summarize_fault_fallback(safety),
            "diagnostics": _summarize_diagnostics(diagnostics),
        },
        holdout_consumption=holdout,
        decision=None,
    )


def _validate_reference_qualification_scope(
    reference_qualification: ReferenceQualificationReport,
    definition: TargetDefinition,
    snapshot: Snapshot,
) -> None:
    if (
        snapshot.target_name != definition.name
        or snapshot.contract_hash != definition.contract_hash
    ):
        raise EvaluationError("snapshot scope mismatch")
    if (
        reference_qualification.target_name != definition.name
        or reference_qualification.contract_hash != definition.contract_hash
    ):
        raise EvaluationError("reference qualification scope mismatch")


def _validate_base_release_scope(base_release: Release, definition: TargetDefinition) -> None:
    if (
        base_release.target_name != definition.name
        or base_release.contract_hash != definition.contract_hash
    ):
        raise EvaluationError("base release scope mismatch")


def _validate_paired_validation_report(
    validation_report: Report,
    candidate: Candidate,
    definition: TargetDefinition,
    snapshot: Snapshot,
    base_release: Release,
) -> None:
    if validation_report.report_stage != "validation" or validation_report.decision is not None:
        raise EvaluationError("validation report required for test generalization")
    if (
        validation_report.candidate_id != candidate.candidate_id
        or validation_report.target_name != definition.name
        or validation_report.contract_hash != definition.contract_hash
        or validation_report.snapshot_id != snapshot.snapshot_id
        or validation_report.baseline_release_id != base_release.release_id
    ):
        raise EvaluationError("validation report scope mismatch")


def _summarize_fault_fallback(safety: dict[str, Any]) -> dict[str, Any]:
    scenarios = [
        _summarize_fault_scenario(scenario)
        for scenario in safety.get("scenarios", [])
        if isinstance(scenario, dict)
    ]
    failures = [
        _summarize_fault_scenario(failure)
        for failure in safety.get("failures", [])
        if isinstance(failure, dict)
    ]
    return {
        "status": safety.get("status", "fail"),
        "scenario_count": len(scenarios),
        "failure_count": len(failures),
        "scenarios": scenarios,
        "failures": failures,
    }


def _summarize_fault_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
    public_keys = {
        "kind",
        "layer",
        "status",
        "reason",
        "expected_decision",
        "observed_decision",
        "fallback_status",
        "error_message_hash",
        "public_error_message",
    }
    return {key: scenario[key] for key in public_keys if key in scenario}


def _summarize_diagnostics(diagnostics: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    ablation = diagnostics.get("changed_layer_ablation")
    if isinstance(ablation, dict):
        summary["changed_layer_ablation"] = {
            "changed_layers": list(ablation.get("changed_layers", [])),
            "baseline_release_id": ablation.get("baseline_release_id"),
            "changed_only": _summarize_cascade_diagnostic(ablation.get("changed_only", {})),
            "with_baseline_unchanged": _summarize_cascade_diagnostic(
                ablation.get("with_baseline_unchanged", {})
            ),
            "cascade": _summarize_cascade_diagnostic(ablation.get("cascade", {})),
        }
    return summary


def _summarize_cascade_diagnostic(cascade: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(cascade, dict):
        return {}
    local_metrics = asdict(compute_metric_summary(cascade)) if "rows" in cascade else {}
    return {
        "sample_count": cascade.get("sample_count", 0),
        "local": local_metrics,
        "l4_fallback_count": cascade.get("l4_fallback_count", 0),
        "l4_fallback_share": cascade.get("l4_fallback_share", 0.0),
        "l4_fallback_cost": cascade.get("l4_fallback_cost", 0.0),
        "l4_fallback_latency_ms": cascade.get("l4_fallback_latency_ms", 0.0),
    }


def _changed_candidate_layers(candidate: Candidate, base_release: Release) -> list[LayerName]:
    changed: list[LayerName] = []
    for layer in ["L1", "L2", "L3"]:
        package = candidate.artifacts.get(layer)  # type: ignore[arg-type]
        base_package = base_release.artifacts.get(layer)  # type: ignore[arg-type]
        if package is not None and (base_package is None or package.digest != base_package.digest):
            changed.append(layer)  # type: ignore[arg-type]
    return changed


def evaluate_candidate_on_validation(
    submission: CandidateSubmission,
    definition: TargetDefinition,
    snapshot: Snapshot,
    base_release: Release,
    reference_qualification: ReferenceQualificationReport,
    agent_usage: AgentUsageLedger,
    reference_usage: Any,
    audit_usage: Any | None,
    local_training_search_usage: Any | None,
    baseline_cost: dict[str, Any],
    evaluation_options: dict[str, Any],
) -> dict[str, Any]:
    _validate_reference_qualification_scope(reference_qualification, definition, snapshot)
    _validate_base_release_scope(base_release, definition)
    artifact_store = Path(evaluation_options.get("artifact_store", ".darjeeling/artifacts"))
    candidate = freeze_candidate(
        submission,
        base_release,
        definition,
        artifact_store,
        snapshot.snapshot_digest,
        snapshot.snapshot_id,
    )
    if candidate.snapshot_id != snapshot.snapshot_id:
        raise EvaluationError("candidate snapshot scope mismatch")
    check = validate_candidate_manifest(candidate, definition, base_release)
    if check["status"] != "pass":
        raise EvaluationError("; ".join(check["failures"]))
    preflight = run_protocol_preflight(candidate, evaluation_options["contract"])
    if preflight["status"] != "pass":
        raise EvaluationError("; ".join(preflight["failures"]))
    view = load_snapshot_view(
        snapshot, "validation", "raw", requester="candidate_evaluation"
    )
    cascade = evaluate_full_cascade(candidate, view, evaluation_options["contract"])
    latency_cost = measure_latency_and_cost(candidate, view, evaluation_options["contract"], {})
    generalization = compute_generalization_summary(cascade, None, [], definition.requirements)
    safety = evaluate_fault_fallback(
        candidate,
        evaluation_options.get("fault_scenarios", []),
        view,
        evaluation_options["contract"],
    )
    diagnostics = {
        "changed_layer_ablation": evaluate_changed_layer_ablation(
            candidate,
            base_release,
            _changed_candidate_layers(candidate, base_release),
            view,
            evaluation_options["contract"],
        )
    }
    cost = build_cost_ledger(
        latency_cost,
        agent_usage,
        reference_usage,
        audit_usage,
        local_training_search_usage,
        baseline_cost,
    )
    report = _build_report(
        "validation",
        candidate,
        definition,
        snapshot,
        base_release,
        reference_qualification,
        cascade,
        generalization,
        latency_cost,
        cost,
        None,
        safety,
        diagnostics,
    )
    feedback = build_agent_feedback(report, {})
    return {"candidate": candidate, "report": report, "feedback": feedback}


def evaluate_candidate_on_test(
    candidate: Candidate,
    closed_attempt: ClosedAgentAttempt,
    definition: TargetDefinition,
    snapshot: Snapshot,
    base_release: Release,
    reference_qualification: ReferenceQualificationReport,
    agent_usage: AgentUsageLedger,
    reference_usage: Any,
    audit_usage: Any | None,
    local_training_search_usage: Any | None,
    baseline_cost: dict[str, Any],
    evaluation_options: dict[str, Any],
) -> dict[str, Any]:
    _validate_reference_qualification_scope(reference_qualification, definition, snapshot)
    _validate_base_release_scope(base_release, definition)
    if closed_attempt.status != "closed":
        raise EvaluationError("agent attempt must be closed before test evaluation")
    if (
        closed_attempt.compile_id != candidate.compile_id
        or closed_attempt.attempt_id != candidate.attempt_id
    ):
        raise EvaluationError("closed attempt does not match candidate lineage")
    if (
        candidate.target_name != definition.name
        or candidate.contract_hash != definition.contract_hash
        or candidate.snapshot_id != snapshot.snapshot_id
        or candidate.base_release_id != base_release.release_id
    ):
        raise EvaluationError("candidate scope does not match test snapshot")
    if (
        closed_attempt.target_name != candidate.target_name
        or closed_attempt.contract_hash != candidate.contract_hash
        or closed_attempt.snapshot_id != snapshot.snapshot_id
        or closed_attempt.snapshot_digest != snapshot.snapshot_digest
    ):
        raise EvaluationError("closed attempt does not match test snapshot lineage")
    if (
        _closed_attempt_baseline_digest(closed_attempt.workspace_path)
        != closed_attempt.final_commit
    ):
        raise EvaluationError("closed attempt workspace changed after close")
    view = load_snapshot_view(snapshot, "test", "raw", requester="candidate_evaluation")
    cascade = evaluate_full_cascade(candidate, view, evaluation_options["contract"])
    validation_cascade = evaluation_options.get("validation_result")
    validation_report = evaluation_options.get("validation_report")
    if validation_cascade is not None and validation_report is None:
        raise EvaluationError("validation report required for validation transfer evidence")
    if validation_report is not None:
        _validate_paired_validation_report(
            validation_report, candidate, definition, snapshot, base_release
        )
    latency_cost = measure_latency_and_cost(candidate, view, evaluation_options["contract"], {})
    generalization = compute_generalization_summary(cascade, cascade, [], definition.requirements)
    if validation_report is not None:
        generalization = _reconcile_generalization_with_metrics(
            generalization,
            MetricSummary(**validation_report.metrics["local"]),
            compute_metric_summary(cascade),
            definition.requirements,
        )
    elif (
        definition.requirements.validation_test_precision_drop_max is not None
        or definition.requirements.validation_test_coverage_retention_min is not None
    ):
        generalization = replace(
            generalization,
            precision_drop=None,
            coverage_retention=None,
            evidence_status="insufficient",
        )
    safety = evaluate_fault_fallback(
        candidate,
        evaluation_options.get("fault_scenarios", []),
        view,
        evaluation_options["contract"],
    )
    diagnostics = {
        "changed_layer_ablation": evaluate_changed_layer_ablation(
            candidate,
            base_release,
            _changed_candidate_layers(candidate, base_release),
            view,
            evaluation_options["contract"],
        )
    }
    cost = build_cost_ledger(
        latency_cost,
        agent_usage,
        reference_usage,
        audit_usage,
        local_training_search_usage,
        baseline_cost,
    )
    visible = bool(evaluation_options.get("test_results_visible", False))
    test_records = load_snapshot_records(view)
    holdout = mark_consumed_holdout_rows(
        snapshot,
        "test",
        [record.snapshot_record_id for record in test_records],
        "test_results_visible" if visible else "test_release_gate",
        "user" if visible else "external_report",
    )
    report = _build_report(
        "test",
        candidate,
        definition,
        snapshot,
        base_release,
        reference_qualification,
        cascade,
        generalization,
        latency_cost,
        cost,
        holdout,
        safety,
        diagnostics,
    )
    return {"candidate": candidate, "report": report}


def _baseline_values(baseline: ReleaseBaseline) -> dict[str, float | None]:
    if baseline.report is not None:
        baseline_metric = MetricSummary(**baseline.report.metrics["local"])
        return {
            "coverage": baseline_metric.coverage,
            "p95_latency_ms": baseline.report.latency.get("p95_latency_ms"),
            "serving_cost": _report_candidate_serving_cost(baseline.report),
        }
    summary = baseline.l4_baseline
    if summary is None:
        return {"coverage": None, "p95_latency_ms": None, "serving_cost": None}
    quality_summary = summary.quality_summary or {}
    local_quality = quality_summary.get("local", quality_summary)
    cost_summary = summary.cost_summary or {}
    return {
        "coverage": local_quality.get("coverage") if isinstance(local_quality, dict) else None,
        "p95_latency_ms": (summary.latency_summary or {}).get("p95_latency_ms"),
        "serving_cost": cost_summary.get(
            "serving_local_compute_cost",
            cost_summary.get("serving_cost_per_1000"),
        ),
    }


def _report_candidate_serving_cost(report: Report) -> float:
    return float(report.latency.get("cascade_cost", report.cost.serving_local_compute_cost))


def _check_baseline_improvement(
    report: Report,
    metric: MetricSummary,
    baseline: ReleaseBaseline,
    objective: dict[str, Any],
    requirements: Any | None,
) -> list[RequirementCheckResult]:
    values = _baseline_values(baseline)
    optimize = objective.get("optimize", "coverage")
    results: list[RequirementCheckResult] = []
    coverage_objective = getattr(requirements, "coverage_objective", "maximize")
    baseline_coverage = values["coverage"]
    if (
        optimize == "coverage"
        and coverage_objective == "maximize"
        and baseline_coverage is not None
        and metric.coverage <= baseline_coverage
    ):
        results.append(
            RequirementCheckResult(
                "baseline_coverage_improvement",
                "fail",
                {"candidate": metric.coverage, "baseline": baseline_coverage},
            )
        )
    if optimize == "latency" and values["p95_latency_ms"] is not None:
        candidate_latency = report.latency.get("p95_latency_ms")
        if candidate_latency is None or candidate_latency >= values["p95_latency_ms"]:
            results.append(
                RequirementCheckResult(
                    "baseline_latency_improvement",
                    "fail",
                    {"candidate": candidate_latency, "baseline": values["p95_latency_ms"]},
                )
            )
    if optimize == "cost" and values["serving_cost"] is not None:
        candidate_cost = _report_candidate_serving_cost(report)
        if candidate_cost >= values["serving_cost"]:
            results.append(
                RequirementCheckResult(
                    "baseline_cost_improvement",
                    "fail",
                    {"candidate": candidate_cost, "baseline": values["serving_cost"]},
                )
            )
    return results


def _check_report_safety(report: Report) -> list[RequirementCheckResult]:
    fault_fallback = report.safety.get("fault_fallback")
    if fault_fallback == "pass":
        return []
    if isinstance(fault_fallback, dict) and fault_fallback.get("status") == "pass":
        return []
    return [
        RequirementCheckResult(
            "fault_fallback",
            "fail",
            fault_fallback if isinstance(fault_fallback, dict) else {"status": fault_fallback},
        )
    ]


def _candidate_requirement_results(
    report: Report,
    metric: MetricSummary,
    baseline: ReleaseBaseline,
    objective: dict[str, Any],
    requirements: Any | None,
) -> list[RequirementCheckResult]:
    if requirements is None:
        blockers = (
            []
            if metric.wrong_accept_count == 0 and metric.accepted_count > 0
            else ["candidate has wrong accepts or no accepts"]
        )
        results = [
            RequirementCheckResult("default_quality", "pass" if not blockers else "fail", {})
        ]
    else:
        results = check_candidate_requirements(
            metric, report.generalization, report.latency, requirements
        )
    results.extend(_check_baseline_improvement(report, metric, baseline, objective, requirements))
    results.extend(_check_report_safety(report))
    return results


def _candidate_objective_key(
    report: Report, metric: MetricSummary, objective: dict[str, Any]
) -> float:
    optimize = objective.get("optimize", "coverage")
    if optimize == "latency":
        latency = report.latency.get("p95_latency_ms")
        return -(float(latency) if latency is not None else float("inf"))
    if optimize == "cost":
        return -_report_candidate_serving_cost(report)
    return metric.coverage


def compare_candidates(
    candidate_reports: list[Report],
    baseline: ReleaseBaseline,
    objective: dict[str, Any],
) -> CandidateDecision:
    if baseline.release.candidate_id is None and baseline.l4_baseline is None:
        raise EvaluationError("cold-start baseline requires L4BaselineSummary, not a fake Report")
    if baseline.release.candidate_id is not None and baseline.report is None:
        raise EvaluationError("compiled baseline requires baseline Report")
    if not candidate_reports:
        raise EvaluationError("no candidate reports supplied")
    invalid_reports = [
        report.report_id
        for report in candidate_reports
        if report.report_stage not in {"validation", "test"} or report.decision is not None
    ]
    if invalid_reports:
        raise EvaluationError(
            "compare_candidates requires validation/test reports without decisions"
        )
    baseline_mismatches = [
        report.report_id
        for report in candidate_reports
        if report.baseline_release_id != baseline.release.release_id
    ]
    if baseline_mismatches:
        raise EvaluationError("candidate reports do not match baseline release")
    scope = {
        (report.target_name, report.contract_hash, report.snapshot_id)
        for report in candidate_reports
    }
    if len(scope) != 1:
        raise EvaluationError("candidate reports have mixed target, contract, or snapshot scope")
    target_name, contract_hash, _snapshot_id = next(iter(scope))
    if (
        target_name != baseline.release.target_name
        or contract_hash != baseline.release.contract_hash
    ):
        raise EvaluationError("candidate reports do not match baseline target scope")
    if baseline.report is not None and (
        baseline.report.target_name != baseline.release.target_name
        or baseline.report.contract_hash != baseline.release.contract_hash
    ):
        raise EvaluationError("baseline report scope mismatch")
    if baseline.report is not None and (
        baseline.release.report_id != baseline.report.report_id
        or baseline.release.candidate_id != baseline.report.candidate_id
        or (
            baseline.release.snapshot_id is not None
            and baseline.release.snapshot_id != baseline.report.snapshot_id
        )
    ):
        raise EvaluationError("baseline report does not belong to baseline release")
    if baseline.l4_baseline is not None and (
        baseline.l4_baseline.target_name != baseline.release.target_name
        or baseline.l4_baseline.contract_hash != baseline.release.contract_hash
    ):
        raise EvaluationError("baseline L4 summary scope mismatch")
    if (
        baseline.l4_baseline is not None
        and baseline.l4_baseline.release_id != baseline.release.release_id
    ):
        raise EvaluationError("baseline L4 summary does not belong to baseline release")
    test_reports = [report for report in candidate_reports if report.report_stage == "test"]
    if not test_reports:
        best = max(candidate_reports, key=lambda report: report.metrics["local"]["coverage"])
        requirement_results = [
            RequirementCheckResult(
                "test_report_required",
                "insufficient",
                {"supplied_stages": sorted({report.report_stage for report in candidate_reports})},
            )
        ]
        return CandidateDecision(
            decision_id=new_id("decision"),
            candidate_id=best.candidate_id,
            target_name=best.target_name,
            contract_hash=best.contract_hash,
            snapshot_id=best.snapshot_id,
            baseline_release_id=best.baseline_release_id,
            status="insufficient_evidence",
            requirement_results=requirement_results,
            comparison_summary={
                "selected_report_id": best.report_id,
                "objective": objective,
                "coverage": best.metrics["local"]["coverage"],
                "release_evidence_stage": best.report_stage,
            },
            selected_operating_point=None,
            release_blockers=["test_report_required"],
            created_at=utcnow(),
        )
    reqs = objective.get("requirements")
    evaluated: list[tuple[Report, MetricSummary, list[RequirementCheckResult]]] = []
    for report in test_reports:
        metric = MetricSummary(**report.metrics["local"])
        requirement_results = _candidate_requirement_results(
            report, metric, baseline, objective, reqs
        )
        evaluated.append((report, metric, requirement_results))
    passing = [
        item
        for item in evaluated
        if not any(result.status in {"fail", "insufficient"} for result in item[2])
    ]
    if passing:
        best, metric, requirement_results = max(
            passing, key=lambda item: _candidate_objective_key(item[0], item[1], objective)
        )
    else:
        best, metric, requirement_results = max(
            evaluated, key=lambda item: _candidate_objective_key(item[0], item[1], objective)
        )
    blockers = [r.name for r in requirement_results if r.status in {"fail", "insufficient"}]
    has_fail = any(result.status == "fail" for result in requirement_results)
    has_insufficient = any(result.status == "insufficient" for result in requirement_results)
    if not blockers:
        status = "eligible_for_release"
    elif has_insufficient and not has_fail:
        status = "insufficient_evidence"
    else:
        status = "rejected"
    return CandidateDecision(
        decision_id=new_id("decision"),
        candidate_id=best.candidate_id,
        target_name=best.target_name,
        contract_hash=best.contract_hash,
        snapshot_id=best.snapshot_id,
        baseline_release_id=best.baseline_release_id,
        status=status,  # type: ignore[arg-type]
        requirement_results=requirement_results,
        comparison_summary={
            "selected_report_id": best.report_id,
            "objective": objective,
            "coverage": metric.coverage,
            "release_evidence_stage": best.report_stage,
            "routing_enabled_layers": best.metrics.get("routing_enabled_layers", []),
        },
        selected_operating_point={
            "enabled_layers": best.metrics.get("routing_enabled_layers", [])
        }
        if status == "eligible_for_release"
        else None,
        release_blockers=blockers,
        created_at=utcnow(),
    )


def finalize_report(
    report: Report, decision: CandidateDecision, validation_report: Report | None = None
) -> Report:
    if (
        report.candidate_id != decision.candidate_id
        or report.target_name != decision.target_name
        or report.contract_hash != decision.contract_hash
        or report.snapshot_id != decision.snapshot_id
        or report.baseline_release_id != decision.baseline_release_id
    ):
        raise EvaluationError("report and decision scope mismatch")
    if decision.status == "eligible_for_release" and report.report_stage != "test":
        raise EvaluationError("eligible release decisions require a test report")
    selected_report_id = decision.comparison_summary.get("selected_report_id")
    if selected_report_id is not None and selected_report_id != report.report_id:
        raise EvaluationError("selected report does not match decision evidence")
    final_metrics = dict(report.metrics)
    if report.report_stage == "test":
        if validation_report is None or validation_report.report_stage != "validation":
            raise EvaluationError("final test reports require matching validation metrics")
        if (
            validation_report.candidate_id != report.candidate_id
            or validation_report.target_name != report.target_name
            or validation_report.contract_hash != report.contract_hash
            or validation_report.snapshot_id != report.snapshot_id
            or validation_report.baseline_release_id != report.baseline_release_id
        ):
            raise EvaluationError("validation and test report scope mismatch")
        final_metrics["validation"] = validation_report.metrics
        final_generalization = _generalization_from_paired_reports(report, validation_report)
    else:
        final_generalization = report.generalization
    return replace(
        report,
        report_id=new_id("report-final"),
        report_stage="final",
        metrics=final_metrics,
        generalization=final_generalization,
        decision=decision,
    )


def _generalization_from_paired_reports(
    test_report: Report, validation_report: Report
) -> GeneralizationSummary:
    return _reconcile_generalization_with_metrics(
        test_report.generalization,
        MetricSummary(**validation_report.metrics["local"]),
        MetricSummary(**test_report.metrics["local"]),
        None,
    )


def _reconcile_generalization_with_metrics(
    base: GeneralizationSummary,
    validation_metric: MetricSummary,
    test_metric: MetricSummary,
    requirements: Any | None,
) -> GeneralizationSummary:
    precision_drop = None
    if validation_metric.precision is not None and test_metric.precision is not None:
        precision_drop = validation_metric.precision - test_metric.precision
    coverage_retention = None
    if validation_metric.coverage:
        coverage_retention = test_metric.coverage / validation_metric.coverage
    evidence = base.evidence_status
    if requirements is not None:
        if requirements.validation_test_precision_drop_max is not None:
            if precision_drop is None:
                evidence = "insufficient" if evidence != "fail" else evidence
            elif precision_drop > requirements.validation_test_precision_drop_max:
                evidence = "fail"
        if requirements.validation_test_coverage_retention_min is not None:
            if coverage_retention is None:
                evidence = "insufficient" if evidence != "fail" else evidence
            elif coverage_retention < requirements.validation_test_coverage_retention_min:
                evidence = "fail"
    return replace(
        base,
        validation_precision=validation_metric.precision,
        test_precision=test_metric.precision,
        validation_coverage=validation_metric.coverage,
        test_coverage=test_metric.coverage,
        precision_drop=precision_drop,
        coverage_retention=coverage_retention,
        evidence_status=evidence,  # type: ignore[arg-type]
    )


def build_agent_feedback(report: Report, feedback_policy: dict[str, Any]) -> AgentFeedback:
    if report.report_stage != "validation":
        raise EvaluationError("only validation reports can be fed back to the active agent")
    return AgentFeedback(
        candidate_id=report.candidate_id,
        summary={"report_id": report.report_id, "stage": report.report_stage},
        requirement_results=[],
        metrics=report.metrics,
        safe_slice_summaries=[],
        latency_cost_summary={"latency": report.latency, "cost": asdict(report.cost)},
        raw_rows_included=False,
    )


def summarize_holdout_consumption(manifest: ConsumedRowsManifest) -> HoldoutConsumptionSummary:
    return HoldoutConsumptionSummary(
        snapshot_id=manifest.snapshot_id,
        split=manifest.split,
        reason_code=manifest.reason,
        consumed_at=manifest.consumed_at,
        visible_to=manifest.visible_to,
        replacement_required=manifest.replacement_required,
        record_count=len(manifest.record_ids),
        split_group_count=len(set(manifest.split_group_keys)),
        manifest_digest=stable_hash(asdict(manifest)),
    )


def build_agent_visible_decision_summary(
    decision: CandidateDecision,
    include_test_metrics: bool,
    holdout_consumption: ConsumedRowsManifest | None,
) -> AgentVisibleDecisionSummary:
    summary = summarize_holdout_consumption(holdout_consumption) if holdout_consumption else None
    return AgentVisibleDecisionSummary(
        candidate_id=decision.candidate_id,
        status=decision.status,
        headline_reason=decision.release_blockers[0]
        if decision.release_blockers
        else decision.status,
        requirement_summary={result.name: result.status for result in decision.requirement_results},
        comparison_summary=decision.comparison_summary if include_test_metrics else {},
        test_metrics_included=include_test_metrics,
        holdout_consumption=summary,
    )


def build_agent_visible_report(report: Report, include_test_metrics: bool) -> AgentVisibleReport:
    if report.report_stage != "final" or report.decision is None:
        raise EvaluationError("only final reports with decisions can become agent-visible reports")
    if include_test_metrics and report.holdout_consumption is None:
        raise EvaluationError(
            "test metrics may be visible only after holdout consumption is summarized"
        )
    decision_summary = build_agent_visible_decision_summary(
        report.decision,
        include_test_metrics,
        report.holdout_consumption if include_test_metrics else None,
    )
    holdout_summary = (
        summarize_holdout_consumption(report.holdout_consumption)
        if include_test_metrics and report.holdout_consumption
        else None
    )
    return AgentVisibleReport(
        report_id=report.report_id,
        candidate_id=report.candidate_id,
        target_name=report.target_name,
        contract_hash=report.contract_hash,
        snapshot_id=report.snapshot_id,
        baseline_release_id=report.baseline_release_id,
        decision_summary=decision_summary,
        validation_metrics=_agent_visible_validation_metrics(report),
        test_metrics={k: v for k, v in report.metrics.items() if k != "validation"}
        if include_test_metrics
        else None,
        test_metrics_included=include_test_metrics,
        holdout_consumption=holdout_summary,
        generalization_summary=asdict(report.generalization),
        cost_summary=asdict(report.cost),
        created_at=utcnow(),
    )


def _agent_visible_validation_metrics(report: Report) -> dict[str, Any]:
    if "validation" in report.metrics:
        return report.metrics["validation"]
    if report.holdout_consumption is None:
        return report.metrics
    return {}
