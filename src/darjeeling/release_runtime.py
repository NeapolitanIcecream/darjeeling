from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import replace
from threading import Event
from typing import Any, Literal

from darjeeling.artifact_worker import (
    build_worker_request,
    run_healthcheck,
    run_layer_attempt,
    start_worker,
    verify_artifact_package_digest,
)
from darjeeling.audit_monitoring import (
    capture_secure_audit_payload,
    combine_audit_decisions,
    decide_random_audit,
    decide_synchronous_risk_audit,
)
from darjeeling.errors import ArtifactError, ReleaseError
from darjeeling.model import (
    ApprovalRecord,
    ArtifactPackage,
    AuditDecisions,
    CacheHit,
    CacheMiss,
    Candidate,
    CascadeResult,
    L4FallbackResult,
    LayerAttemptResult,
    LayerName,
    PreparedReleaseWorkers,
    ReferenceBroker,
    ReferenceContext,
    Release,
    ReleaseRegistry,
    Report,
    ResultCache,
    RoutingSettings,
    RuntimeContext,
    RuntimeRequest,
    RuntimeResponse,
    ServingResult,
    Snapshot,
    TargetCheckReport,
    TargetDefinition,
    TargetRuntimeContract,
    Trace,
    WorkerLimits,
    WorkerPool,
)
from darjeeling.snapshot_reference import call_reference
from darjeeling.telemetry_recompile import approve_telemetry_evidence
from darjeeling.util import (
    new_id,
    read_json,
    safe_public_error,
    scoped_hash,
    stable_hash,
    utcnow,
)


def _empty_artifacts() -> dict[LayerName, ArtifactPackage | None]:
    return {"L1": None, "L2": None, "L3": None}


def create_release_without_artifacts(
    definition: TargetDefinition,
    contract: TargetRuntimeContract,
    target_check: TargetCheckReport,
    reference_broker: ReferenceBroker,
    routing: RoutingSettings,
    registry: ReleaseRegistry,
) -> Release:
    if target_check.status != "pass":
        raise ReleaseError("target checks must pass before cold-start release")
    if (
        target_check.target_name != definition.name
        or target_check.contract_hash != definition.contract_hash
    ):
        raise ReleaseError("target check scope mismatch")
    probe = call_l4_fallback(
        contract,
        contract.validate_input(_target_owned_probe_input(definition)),
        reference_broker,
        RuntimeContext(new_id("coldstart"), routing.total_deadline_ms, reference_broker),
    )
    if probe.status != "ok":
        raise ReleaseError("L4 fallback is not usable for cold-start release")
    release = Release(
        release_id=new_id("release"),
        target_name=definition.name,
        contract_hash=definition.contract_hash,
        candidate_id=None,
        snapshot_id=None,
        snapshot_digest=None,
        report_id=None,
        created_at=utcnow(),
        artifacts=_empty_artifacts(),
        routing=routing,
        approval=None,
        status="created",
    )
    registry.releases[release.release_id] = release
    return release


def _target_owned_probe_input(definition: TargetDefinition) -> dict[str, Any]:
    for source in definition.data_config.sources:
        for record in source.get("records", []):
            if isinstance(record, dict) and isinstance(record.get("input"), dict):
                return dict(record["input"])
        path_value = source.get("path")
        if not path_value:
            continue
        for record in read_json((definition.target_path / path_value).resolve()):
            if isinstance(record, dict) and isinstance(record.get("input"), dict):
                return dict(record["input"])
    raise ReleaseError("target-owned sample input is required for cold-start release")


def create_release(
    candidate: Candidate,
    snapshot: Snapshot,
    base_release: Release,
    report: Report,
    approval: ApprovalRecord,
    artifact_store: Any,
) -> Release:
    if report.report_stage != "final" or report.decision is None:
        raise ReleaseError("create_release requires a final report with a decision")
    if report.decision.status != "eligible_for_release":
        raise ReleaseError("candidate decision is not eligible for release")
    if approval.candidate_id != candidate.candidate_id or approval.report_id != report.report_id:
        raise ReleaseError("approval does not reference exact candidate and report")
    if (
        approval.target_name != candidate.target_name
        or approval.contract_hash != candidate.contract_hash
    ):
        raise ReleaseError("approval target or contract scope mismatch")
    expected = {
        "candidate_id": candidate.candidate_id,
        "target_name": candidate.target_name,
        "contract_hash": candidate.contract_hash,
        "snapshot_id": snapshot.snapshot_id,
    }
    if (
        snapshot.target_name != candidate.target_name
        or snapshot.contract_hash != candidate.contract_hash
    ):
        raise ReleaseError("snapshot scope mismatch")
    actual = {
        "candidate_id": report.candidate_id,
        "target_name": report.target_name,
        "contract_hash": report.contract_hash,
        "snapshot_id": report.snapshot_id,
    }
    if expected != actual:
        raise ReleaseError("candidate, report, and snapshot scope mismatch")
    decision_actual = {
        "candidate_id": report.decision.candidate_id,
        "target_name": report.decision.target_name,
        "contract_hash": report.decision.contract_hash,
        "snapshot_id": report.decision.snapshot_id,
    }
    if expected != decision_actual:
        raise ReleaseError("decision scope mismatch")
    if candidate.snapshot_id != snapshot.snapshot_id:
        raise ReleaseError("candidate snapshot scope mismatch")
    if approval.snapshot_id != snapshot.snapshot_id:
        raise ReleaseError("approval snapshot scope mismatch")
    if (
        base_release.target_name != candidate.target_name
        or base_release.contract_hash != candidate.contract_hash
    ):
        raise ReleaseError("base release scope mismatch")
    if (
        candidate.base_release_id != base_release.release_id
        or report.baseline_release_id != base_release.release_id
        or report.decision.baseline_release_id != base_release.release_id
    ):
        raise ReleaseError("base release identity mismatch")
    if _recompute_candidate_digest(candidate, snapshot.snapshot_digest) != candidate.digest:
        raise ReleaseError("candidate digest mismatch")
    for package in candidate.artifacts.values():
        if package is None:
            continue
        if package.source_snapshot_digest != snapshot.snapshot_digest:
            raise ReleaseError("candidate artifact snapshot digest mismatch")
        try:
            verify_artifact_package_digest(package)
        except ArtifactError as exc:
            raise ReleaseError("candidate artifact digest mismatch") from exc
    release = Release(
        release_id=new_id("release"),
        target_name=candidate.target_name,
        contract_hash=candidate.contract_hash,
        candidate_id=candidate.candidate_id,
        snapshot_id=snapshot.snapshot_id,
        snapshot_digest=snapshot.snapshot_digest,
        report_id=report.report_id,
        created_at=utcnow(),
        artifacts=candidate.artifacts,
        routing=candidate.routing,
        approval=approval,
        status="created",
    )
    return release


def _recompute_candidate_digest(candidate: Candidate, source_snapshot_digest: str) -> str:
    return stable_hash(
        {
            "submission_id": candidate.submission_id,
            "workspace_commit": candidate.workspace_commit,
            "target_name": candidate.target_name,
            "contract_hash": candidate.contract_hash,
            "snapshot_id": candidate.snapshot_id,
            "source_snapshot_digest": source_snapshot_digest,
            "base_release_id": candidate.base_release_id,
            "artifacts": {
                layer: package.digest if package else None
                for layer, package in candidate.artifacts.items()
            },
            "routing": candidate.routing,
        }
    )


def _release_has_local_artifacts(release: Release) -> bool:
    return any(package is not None for package in release.artifacts.values())


def _validate_release_atomicity(release: Release) -> None:
    provenance_values = [
        release.candidate_id,
        release.snapshot_id,
        release.snapshot_digest,
        release.report_id,
        release.approval,
    ]
    has_any_provenance = any(value is not None for value in provenance_values)
    has_all_provenance = all(value is not None for value in provenance_values)
    if _release_has_local_artifacts(release) and not has_any_provenance:
        raise ReleaseError("release atomicity violation: artifact release requires provenance")
    if has_any_provenance and not has_all_provenance:
        raise ReleaseError("release atomicity violation: partial compile provenance")
    if release.approval is not None:
        if (
            release.approval.candidate_id != release.candidate_id
            or release.approval.report_id != release.report_id
            or release.approval.target_name != release.target_name
            or release.approval.contract_hash != release.contract_hash
            or release.approval.snapshot_id != release.snapshot_id
        ):
            raise ReleaseError("release atomicity violation: approval scope mismatch")
    for layer, package in release.artifacts.items():
        if package is None:
            continue
        if package.layer != layer:
            raise ReleaseError("release atomicity violation: artifact layer mismatch")
        if package.manifest.contract_hash != release.contract_hash:
            raise ReleaseError("release atomicity violation: artifact contract mismatch")
        if (
            release.snapshot_digest is not None
            and package.source_snapshot_digest != release.snapshot_digest
        ):
            raise ReleaseError("release atomicity violation: artifact snapshot mismatch")


def load_release(release_id: str, registry: ReleaseRegistry) -> Release:
    try:
        release = registry.releases[release_id]
    except KeyError as exc:
        raise ReleaseError(f"unknown release: {release_id}") from exc
    if release.release_id != release_id:
        raise ReleaseError("release registry metadata mismatch")
    _validate_release_atomicity(release)
    for package in release.artifacts.values():
        if package is None:
            continue
        try:
            verify_artifact_package_digest(package)
        except ArtifactError as exc:
            raise ReleaseError("release artifact digest mismatch") from exc
    return release


def set_channel(
    target_name: str,
    channel: Literal["shadow", "canary", "stable"],
    release_id: str,
    channel_options: dict[str, Any],
    registry: ReleaseRegistry,
) -> dict[str, Any]:
    release = load_release(release_id, registry)
    if release.target_name != target_name:
        raise ReleaseError("release target does not match channel target")
    for (channel_target, existing_channel), existing_release_id in registry.channels.items():
        if channel_target != target_name or existing_release_id == release_id:
            continue
        existing = load_release(existing_release_id, registry)
        if existing.contract_hash != release.contract_hash:
            raise ReleaseError(
                f"channel contract compatibility mismatch: {existing_channel}"
            )
    key = (target_name, channel)
    previous = registry.channels.get(key)
    if previous:
        registry.previous_channels[key] = previous
    registry.channels[key] = release_id
    registry.channel_options[key] = dict(channel_options)
    registry.releases[release_id] = replace(release, status=channel)
    return {
        "target_name": target_name,
        "channel": channel,
        "release_id": release_id,
        "previous_release_id": previous,
    }


def select_release_for_request(
    target_name: str, request: RuntimeRequest, registry: ReleaseRegistry
) -> Release:
    stable_id = registry.channels.get((target_name, "stable"))
    canary_id = registry.channels.get((target_name, "canary"))
    if stable_id is None:
        raise ReleaseError(f"no stable release channel for target {target_name}")
    stable_release = load_release(stable_id, registry)
    if stable_release.status != "stable":
        raise ReleaseError("stable channel release is not active")
    if stable_id is not None and canary_id is not None:
        canary_release = load_release(canary_id, registry)
        if canary_release.status != "canary":
            raise ReleaseError("canary channel release is not active")
        options = registry.channel_options.get((target_name, "canary"), {})
        fraction = float(options.get("traffic_fraction", options.get("traffic", 0.0)))
        if fraction > 0.0:
            routing_key = request.tenant_key or request.request_id
            bucket = int(stable_hash((target_name, routing_key))[:12], 16) / float(
                0xFFFFFFFFFFFF
            )
            if bucket < min(1.0, max(0.0, fraction)):
                return canary_release
    return stable_release


def _cache_key(
    release: Release,
    contract: TargetRuntimeContract,
    input_value: dict[str, Any],
    reference_version: str | None,
) -> str:
    return stable_hash(
        (
            release.contract_hash,
            release.release_id,
            reference_version,
            contract.normalize_input(input_value),
        )
    )


def check_result_cache(
    release: Release,
    contract: TargetRuntimeContract,
    input_value: dict[str, Any],
    cache_policy: ResultCache | None,
    reference_version: str | None = None,
) -> CacheHit | CacheMiss:
    key = _cache_key(release, contract, input_value, reference_version)
    if cache_policy is not None and release.routing.cache_enabled and key in cache_policy.entries:
        return CacheHit(output=cache_policy.entries[key], cache_key=key)
    return CacheMiss(cache_key=key)


def write_result_cache(
    release: Release,
    contract: TargetRuntimeContract,
    input_value: dict[str, Any],
    output: dict[str, Any],
    cache_policy: ResultCache | None,
    reference_version: str | None = None,
) -> dict[str, Any]:
    key = _cache_key(release, contract, input_value, reference_version)
    if cache_policy is not None and release.routing.cache_enabled:
        cache_policy.entries[key] = output
        return {"status": "written", "cache_key": key}
    return {"status": "skipped", "cache_key": key}


_CIRCUIT_FAILURE_DECISIONS = {"timeout", "error", "protocol_error", "invalid_output"}
_CIRCUIT_FAILURE_KINDS = {
    "digest_failure",
    "health_failure",
    "invalid_output",
    "protocol_error",
    "start_failure",
    "timeout",
    "worker_crash",
}


def _circuit_key(release: Release, layer: LayerName) -> tuple[str, LayerName]:
    return (release.release_id, layer)


def _circuit_setting_int(
    release: Release, name: str, default: int, *, minimum: int = 1
) -> int:
    value = release.routing.circuit_breaker.get(name, default)
    if type(value) is not int:
        return default
    return max(minimum, value)


def _circuit_state(
    release: Release,
    layer: LayerName,
    state_store: dict[tuple[str, LayerName], dict[str, Any]],
) -> dict[str, Any]:
    return state_store.setdefault(
        _circuit_key(release, layer),
        {"state": "closed", "failures": 0, "disabled_remaining": 0, "last_event": None},
    )


def _is_layer_circuit_open(
    release: Release,
    layer: LayerName,
    state_store: dict[tuple[str, LayerName], dict[str, Any]] | None,
    *,
    consume_skip: bool,
) -> bool:
    if state_store is None:
        return False
    state = state_store.get(_circuit_key(release, layer))
    if state is None or state.get("state") != "open":
        return False
    remaining = int(state.get("disabled_remaining", 0))
    if remaining <= 0:
        state.update({"state": "closed", "failures": 0, "disabled_remaining": 0})
        return False
    if consume_skip:
        state["disabled_remaining"] = remaining - 1
        if state["disabled_remaining"] <= 0:
            state.update({"state": "closed", "failures": 0, "disabled_remaining": 0})
    return True


def update_circuit_breaker(
    release: Release,
    layer: LayerName,
    event: dict[str, Any],
    state_store: dict[tuple[str, LayerName], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    local_store: dict[tuple[str, LayerName], dict[str, Any]] = {}
    store = state_store if state_store is not None else local_store
    state = _circuit_state(release, layer, store)
    is_failure = (
        event.get("decision") in _CIRCUIT_FAILURE_DECISIONS
        or event.get("kind") in _CIRCUIT_FAILURE_KINDS
    )
    if not is_failure:
        state.update(
            {"state": "closed", "failures": 0, "disabled_remaining": 0, "last_event": event}
        )
        return {"release_id": release.release_id, "layer": layer, **state}
    failures = int(state.get("failures", 0)) + 1
    threshold = _circuit_setting_int(release, "failure_threshold", 3)
    if failures >= threshold:
        state.update(
            {
                "state": "open",
                "failures": failures,
                "disabled_remaining": _circuit_setting_int(
                    release, "cooldown_requests", 1
                ),
                "last_event": event,
            }
        )
    else:
        state.update(
            {
                "state": "closed",
                "failures": failures,
                "disabled_remaining": 0,
                "last_event": event,
            }
        )
    return {"release_id": release.release_id, "layer": layer, **state}


def prepare_workers(release: Release, worker_pool: WorkerPool) -> PreparedReleaseWorkers:
    workers = {}
    for layer in ["L1", "L2", "L3"]:
        package = release.artifacts.get(layer)  # type: ignore[arg-type]
        if package is None or layer not in release.routing.enabled_layers:
            continue
        try:
            verify_artifact_package_digest(package)
            worker = worker_pool.workers.get(package.artifact_id)
            if worker is None:
                worker = start_worker(package, WorkerLimits())
            health = run_healthcheck(worker, package.manifest.timeout_ms)
        except Exception as exc:
            raise ReleaseError(f"failed to prepare {layer} worker: {exc}") from exc
        if health.status != "pass":
            raise ReleaseError(f"failed to prepare {layer} worker: {health.message}")
        worker_pool.workers[package.artifact_id] = worker
        workers[layer] = worker
    return PreparedReleaseWorkers(workers=workers)  # type: ignore[arg-type]


def _prepare_workers_for_serving(
    release: Release,
    worker_pool: WorkerPool,
    circuit_breaker_state: dict[tuple[str, LayerName], dict[str, Any]] | None = None,
) -> tuple[PreparedReleaseWorkers, list[LayerAttemptResult]]:
    workers = {}
    skip_attempts = []
    for layer in ["L1", "L2", "L3"]:
        package = release.artifacts.get(layer)  # type: ignore[arg-type]
        if package is None or layer not in release.routing.enabled_layers:
            continue
        if _is_layer_circuit_open(
            release, layer, circuit_breaker_state, consume_skip=False
        ):
            continue
        try:
            verify_artifact_package_digest(package)
            worker = worker_pool.workers.get(package.artifact_id)
            if worker is None:
                worker = start_worker(package, WorkerLimits())
            health = run_healthcheck(worker, package.manifest.timeout_ms)
        except Exception as exc:
            update_circuit_breaker(
                release,
                layer,
                {"kind": "health_failure", "message_hash": stable_hash(str(exc))},
                circuit_breaker_state,
            )
            skip_attempts.append(
                _skip_layer_attempt(layer, package.artifact_id, "health_failure", str(exc))
            )
            continue
        if health.status != "pass":
            update_circuit_breaker(
                release,
                layer,
                {"kind": "health_failure", "message_hash": stable_hash(health.message)},
                circuit_breaker_state,
            )
            skip_attempts.append(
                _skip_layer_attempt(
                    layer, package.artifact_id, "health_failure", health.message
                )
            )
            continue
        worker_pool.workers[package.artifact_id] = worker
        workers[layer] = worker
    return PreparedReleaseWorkers(workers=workers), skip_attempts  # type: ignore[arg-type]


def _skip_layer_attempt(
    layer: LayerName, artifact_id: str, reason: str, error: str | None = None
) -> LayerAttemptResult:
    return LayerAttemptResult(
        layer=layer,
        artifact_id=artifact_id,
        decision="error",
        output=None,
        confidence=None,
        reason=reason,
        latency_ms=0.0,
        error=error,
    )


def call_l4_fallback(
    contract: TargetRuntimeContract,
    input_value: dict[str, Any],
    broker: ReferenceBroker,
    runtime_context: RuntimeContext,
) -> L4FallbackResult:
    started = time.perf_counter()
    if runtime_context.deadline_ms <= 0:
        return _l4_timeout_result(input_value, started)
    reference_context = ReferenceContext(
        purpose="runtime_l4_fallback", request_id=runtime_context.request_id
    )
    result = _call_reference_with_timeout(
        contract, input_value, broker, reference_context, runtime_context.deadline_ms
    )
    if result is None:
        return _l4_timeout_result(input_value, started)
    if result.status == "ok" and result.output is not None:
        return L4FallbackResult(
            input_raw=input_value,
            status="ok",
            output_raw=result.output,
            output_validated=result.output,
            reference_source=result.reference_source,
            cost=result.cost,
            latency_ms=result.latency_ms,
            finish_status=result.finish_status,
        )
    return L4FallbackResult(
        input_raw=input_value,
        status="error",
        output_raw=None,
        output_validated=None,
        reference_source=None,
        cost=result.cost,
        latency_ms=result.latency_ms,
        finish_status=result.finish_status,
        error_type=result.error_type or "provider_error",  # type: ignore[arg-type]
        error_message_hash=result.error_message_hash,
    )


def _call_reference_with_timeout(
    contract: TargetRuntimeContract,
    input_value: dict[str, Any],
    broker: ReferenceBroker,
    reference_context: ReferenceContext,
    timeout_ms: int,
):
    cancel_event = Event()
    context = replace(
        reference_context,
        metadata={
            **reference_context.metadata,
            "cancel_event": cancel_event,
            "deadline_at_monotonic": time.monotonic() + (max(timeout_ms, 1) / 1000),
            "timeout_ms": max(timeout_ms, 1),
        },
    )
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="darjeeling-l4")
    future = executor.submit(call_reference, contract, input_value, broker, context)
    try:
        return future.result(timeout=max(timeout_ms, 1) / 1000)
    except FutureTimeoutError:
        cancel_event.set()
        future.cancel()
        return None
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _l4_timeout_result(input_value: dict[str, Any], started: float) -> L4FallbackResult:
    return L4FallbackResult(
        input_raw=input_value,
        status="error",
        output_raw=None,
        output_validated=None,
        reference_source=None,
        cost=0.0,
        latency_ms=(time.perf_counter() - started) * 1000,
        finish_status="timeout",
        error_type="timeout",
        error_message_hash=stable_hash("deadline_exceeded"),
    )


def _remaining_deadline_ms(started: float, deadline_ms: int) -> int:
    return deadline_ms - int((time.perf_counter() - started) * 1000)


def _deadline_exceeded_cascade_result(
    release: Release,
    attempts: list[Any],
    started: float,
) -> CascadeResult:
    return CascadeResult(
        release_id=release.release_id,
        attempts=attempts,
        status="error",
        chosen_layer=None,
        output=None,
        serving_cost=0.0,
        latency_ms=(time.perf_counter() - started) * 1000,
        fallback_reason="deadline_exceeded",
        error_type="deadline_exceeded",
        error_message_hash=stable_hash("deadline_exceeded"),
        l4_fallback_result=None,
    )


def run_cascade(
    release: Release,
    workers: PreparedReleaseWorkers,
    contract: TargetRuntimeContract,
    input_value: dict[str, Any],
    runtime_context: RuntimeContext,
    circuit_breaker_state: dict[tuple[str, LayerName], dict[str, Any]] | None = None,
    initial_attempts: list[LayerAttemptResult] | None = None,
) -> CascadeResult:
    started = time.perf_counter()
    attempts = list(initial_attempts or [])
    for layer in ["L1", "L2", "L3"]:
        if layer not in release.routing.enabled_layers:
            continue
        if _remaining_deadline_ms(started, runtime_context.deadline_ms) <= 0:
            return _deadline_exceeded_cascade_result(release, attempts, started)
        if _is_layer_circuit_open(
            release, layer, circuit_breaker_state, consume_skip=True
        ):
            package = release.artifacts.get(layer)  # type: ignore[arg-type]
            artifact_id = package.artifact_id if package is not None else "missing"
            attempts.append(_skip_layer_attempt(layer, artifact_id, "circuit_open"))
            continue
        worker = workers.workers.get(layer)  # type: ignore[arg-type]
        if worker is None:
            continue
        remaining_request_ms = _remaining_deadline_ms(started, runtime_context.deadline_ms)
        if remaining_request_ms <= 0:
            return _deadline_exceeded_cascade_result(release, attempts, started)
        timeout = min(
            remaining_request_ms,
            getattr(release.routing, f"{layer}_timeout_ms") or worker.package.manifest.timeout_ms,
            worker.package.manifest.timeout_ms,
        )
        request = build_worker_request(
            runtime_context.request_id, input_value, timeout, "runtime_stable"
        )
        attempt = run_layer_attempt(worker, contract, request, layer)  # type: ignore[arg-type]
        attempts.append(attempt)
        update_circuit_breaker(
            release,
            layer,
            {
                "decision": attempt.decision,
                "error_message_hash": stable_hash(attempt.error) if attempt.error else None,
            },
            circuit_breaker_state,
        )
        if attempt.decision == "accept" and attempt.output is not None:
            return CascadeResult(
                release_id=release.release_id,
                attempts=attempts,
                status="ok",
                chosen_layer=layer,  # type: ignore[arg-type]
                output=attempt.output,
                serving_cost=0.0,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
    if _remaining_deadline_ms(started, runtime_context.deadline_ms) <= 0:
        return _deadline_exceeded_cascade_result(release, attempts, started)
    remaining_ms = _remaining_deadline_ms(started, runtime_context.deadline_ms)
    fallback_context = replace(runtime_context, deadline_ms=remaining_ms)
    fallback = call_l4_fallback(contract, input_value, runtime_context.broker, fallback_context)
    latency = (time.perf_counter() - started) * 1000
    if fallback.status == "ok":
        return CascadeResult(
            release_id=release.release_id,
            attempts=attempts,
            status="ok",
            chosen_layer="L4",
            output=fallback.output_validated,
            serving_cost=fallback.cost,
            latency_ms=latency,
            fallback_reason="no_local_accept",
            l4_fallback_result=fallback,
        )
    if fallback.error_type == "timeout":
        return CascadeResult(
            release_id=release.release_id,
            attempts=attempts,
            status="error",
            chosen_layer="L4",
            output=None,
            serving_cost=fallback.cost,
            latency_ms=latency,
            fallback_reason="deadline_exceeded",
            error_type="deadline_exceeded",
            error_message_hash=fallback.error_message_hash,
            l4_fallback_result=fallback,
        )
    return CascadeResult(
        release_id=release.release_id,
        attempts=attempts,
        status="error",
        chosen_layer="L4",
        output=None,
        serving_cost=fallback.cost,
        latency_ms=latency,
        fallback_reason="no_local_accept",
        error_type="l4_fallback_failure",
        error_message_hash=fallback.error_message_hash,
        l4_fallback_result=fallback,
    )


def build_cache_serving_result(
    release: Release,
    cache_hit: CacheHit,
    contract: TargetRuntimeContract,
    runtime_context: RuntimeContext,
) -> ServingResult:
    output = contract.validate_output(cache_hit.output)
    return ServingResult(
        release_id=release.release_id,
        path="cache",
        status="ok",
        cache_result=cache_hit,
        cascade_result=None,
        output=output,
        chosen_layer="cache",
        serving_cost=0.0,
        latency_ms=0.0,
        error_type=None,
        error_message_hash=None,
        l4_fallback_result=None,
    )


def _trace_audit_none() -> AuditDecisions:
    return AuditDecisions(
        random=None,
        synchronous_risk=None,
        selected_audit_types=[],
        random_sampling_probability=0.0,
        risk_flags=[],
    )


def serve_request(
    request: RuntimeRequest,
    registry: ReleaseRegistry,
    contract_loader: Callable[[str], TargetRuntimeContract],
    worker_pool: WorkerPool,
    trace_id_generator: Callable[[], str],
    trace_writer: Callable[..., Any],
    audit_decider: Callable[..., AuditDecisions] | None,
    risk_rules: Any,
    secure_audit_store: Any,
    audit_queue: Any,
    telemetry_privacy_policy: Any,
    approved_evidence_store: Any,
    reference_broker: ReferenceBroker,
    result_cache: ResultCache | None = None,
) -> RuntimeResponse:
    trace_id = trace_id_generator()
    release = select_release_for_request(request.target_name, request, registry)
    contract = contract_loader(release.target_name)
    started = time.perf_counter()
    try:
        if getattr(contract, "contract_hash", None) != release.contract_hash:
            raise ReleaseError("loaded contract hash does not match selected release")
        input_value = contract.validate_input(request.input)
        reference_version = getattr(reference_broker, "reference_version", None)
        cache_result = check_result_cache(
            release, contract, input_value, result_cache, reference_version
        )
        deadline_ms = release.routing.total_deadline_ms
        if request.deadline_ms is not None:
            deadline_ms = min(request.deadline_ms, release.routing.total_deadline_ms)
        runtime_context = RuntimeContext(
            request_id=request.request_id,
            deadline_ms=deadline_ms,
            broker=reference_broker,
        )
        if isinstance(cache_result, CacheHit):
            serving = build_cache_serving_result(release, cache_result, contract, runtime_context)
        else:
            workers, preparation_attempts = _prepare_workers_for_serving(
                release, worker_pool, registry.circuit_breakers
            )
            cascade = run_cascade(
                release,
                workers,
                contract,
                input_value,
                runtime_context,
                registry.circuit_breakers,
                preparation_attempts,
            )
            serving = ServingResult(
                release_id=release.release_id,
                path="cascade",
                status=cascade.status,
                cache_result=cache_result,
                cascade_result=cascade,
                output=cascade.output,
                chosen_layer=cascade.chosen_layer,
                serving_cost=cascade.serving_cost,
                latency_ms=cascade.latency_ms,
                error_type=cascade.error_type,
                error_message_hash=cascade.error_message_hash,
                l4_fallback_result=cascade.l4_fallback_result,
            )
            if serving.status == "ok" and serving.output is not None:
                write_result_cache(
                    release,
                    contract,
                    input_value,
                    serving.output,
                    result_cache,
                    reference_version,
                )
        audit_decisions = _build_runtime_audit_decisions(
            release, request, serving, audit_decider, risk_rules
        )
        if audit_decisions.selected_audit_types:
            if isinstance(secure_audit_store, dict):
                payload = capture_secure_audit_payload(
                    trace_id, request, serving, audit_decisions, secure_audit_store
                )
                if payload is None:
                    audit_decisions = _trace_audit_none()
            else:
                audit_decisions = _trace_audit_none()
        trace = trace_writer(
            trace_id, release.contract_hash, contract, request, serving, audit_decisions
        )
        _append_l4_fallback_evidence(
            trace if isinstance(trace, Trace) else None,
            release,
            contract,
            serving,
            telemetry_privacy_policy,
            approved_evidence_store,
        )
        _enqueue_selected_audits(trace_id, release, audit_decisions, audit_queue)
        latency = (time.perf_counter() - started) * 1000
        if serving.status == "ok":
            return RuntimeResponse(
                request_id=request.request_id,
                release_id=release.release_id,
                status="ok",
                output=serving.output,
                chosen_layer=serving.chosen_layer,
                error_type=None,
                public_error_message=None,
                latency_ms=latency,
                trace_id=trace_id,
            )
        error_type = serving.error_type or "runtime_error"
        return RuntimeResponse(
            request_id=request.request_id,
            release_id=release.release_id,
            status="error",
            output=None,
            chosen_layer=serving.chosen_layer,
            error_type=error_type,  # type: ignore[arg-type]
            public_error_message=safe_public_error(error_type),
            latency_ms=latency,
            trace_id=trace_id,
        )
    except Exception as exc:
        error_hash = stable_hash(str(exc))
        serving = ServingResult(
            release_id=release.release_id,
            path="cascade",
            status="error",
            cache_result=CacheMiss(cache_key="not_checked"),
            cascade_result=None,
            output=None,
            chosen_layer=None,
            serving_cost=0.0,
            latency_ms=(time.perf_counter() - started) * 1000,
            error_type="runtime_error",
            error_message_hash=error_hash,
            l4_fallback_result=None,
        )
        trace_writer(
            trace_id, release.contract_hash, contract, request, serving, _trace_audit_none()
        )
        return RuntimeResponse(
            request_id=request.request_id,
            release_id=release.release_id,
            status="error",
            output=None,
            chosen_layer=None,
            error_type="runtime_error",
            public_error_message=safe_public_error("runtime_error"),
            latency_ms=serving.latency_ms,
            trace_id=trace_id,
        )


def _build_runtime_audit_decisions(
    release: Release,
    request: RuntimeRequest,
    serving: ServingResult,
    audit_decider: Callable[..., AuditDecisions] | None,
    risk_rules: Any,
) -> AuditDecisions:
    if audit_decider is not None:
        return audit_decider(release, request, serving)
    random_decision = decide_random_audit(release, request, serving, release.routing.audit)
    risk_decision = decide_synchronous_risk_audit(
        release, request, serving, risk_rules if isinstance(risk_rules, dict) else {}
    )
    return combine_audit_decisions(random_decision, risk_decision)


def _append_l4_fallback_evidence(
    trace: Trace | None,
    release: Release,
    contract: TargetRuntimeContract,
    serving: ServingResult,
    telemetry_privacy_policy: Any,
    approved_evidence_store: Any,
) -> None:
    if serving.l4_fallback_result is None or serving.l4_fallback_result.status != "ok":
        return
    if approved_evidence_store is None:
        return
    if trace is None:
        raise ReleaseError("trace writer must return Trace to persist L4 fallback evidence")
    evidence = approve_telemetry_evidence(
        trace,
        None,
        None,
        serving.l4_fallback_result,
        None,
        None,
        trace.trace_id,
        release.release_id,
        trace.timestamp,
        release.target_name,
        release.contract_hash,
        contract,
        telemetry_privacy_policy,
    )
    if evidence is None:
        return
    if hasattr(approved_evidence_store, "append"):
        approved_evidence_store.append(evidence)
    elif isinstance(approved_evidence_store, dict):
        approved_evidence_store[evidence.evidence_id] = evidence
    else:
        raise ReleaseError("approved evidence store must support append or mapping writes")


def _enqueue_selected_audits(
    trace_id: str,
    release: Release,
    audit_decisions: AuditDecisions,
    audit_queue: Any,
) -> None:
    if not audit_decisions.selected_audit_types or not hasattr(audit_queue, "append"):
        return
    audit_queue.append(
        {
            "trace_id": trace_id,
            "release_id": release.release_id,
            "audit_types": list(audit_decisions.selected_audit_types),
        }
    )


def run_shadow_request(
    request: RuntimeRequest,
    stable_release: Release,
    shadow_release: Release,
    contract: TargetRuntimeContract,
    worker_pool: WorkerPool,
    runtime_context: RuntimeContext,
    circuit_breaker_state: dict[tuple[str, LayerName], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if (
        stable_release.target_name != request.target_name
        or shadow_release.target_name != request.target_name
    ):
        raise ReleaseError("shadow release target mismatch")
    if stable_release.contract_hash != shadow_release.contract_hash:
        raise ReleaseError("shadow release contract mismatch")
    if getattr(contract, "contract_hash", None) != stable_release.contract_hash:
        raise ReleaseError("shadow contract hash mismatch")
    input_value = contract.validate_input(request.input)
    stable_result = _run_shadow_cascade(
        stable_release, contract, input_value, worker_pool, runtime_context, circuit_breaker_state
    )
    shadow_result = _run_shadow_cascade(
        shadow_release, contract, input_value, worker_pool, runtime_context, circuit_breaker_state
    )
    return {
        "request_id_hash": scoped_hash("shadow", request.request_id),
        "stable_release_id": stable_release.release_id,
        "shadow_release_id": shadow_release.release_id,
        "stable": _shadow_cascade_summary(stable_result),
        "shadow": _shadow_cascade_summary(shadow_result),
        "chosen_layer_match": stable_result.chosen_layer == shadow_result.chosen_layer,
        "output_hash_match": (
            stable_result.output is not None
            and shadow_result.output is not None
            and stable_hash(stable_result.output) == stable_hash(shadow_result.output)
        ),
        "status_match": stable_result.status == shadow_result.status,
    }


def _run_shadow_cascade(
    release: Release,
    contract: TargetRuntimeContract,
    input_value: dict[str, Any],
    worker_pool: WorkerPool,
    runtime_context: RuntimeContext,
    circuit_breaker_state: dict[tuple[str, LayerName], dict[str, Any]] | None,
) -> CascadeResult:
    deadline_ms = min(runtime_context.deadline_ms, release.routing.total_deadline_ms)
    release_context = replace(runtime_context, deadline_ms=deadline_ms)
    workers, preparation_attempts = _prepare_workers_for_serving(
        release, worker_pool, circuit_breaker_state
    )
    return run_cascade(
        release,
        workers,
        contract,
        input_value,
        release_context,
        circuit_breaker_state,
        preparation_attempts,
    )


def _shadow_cascade_summary(result: CascadeResult) -> dict[str, Any]:
    return {
        "release_id": result.release_id,
        "status": result.status,
        "chosen_layer": result.chosen_layer,
        "output_hash": stable_hash(result.output) if result.output is not None else None,
        "fallback_reason": result.fallback_reason,
        "error_type": result.error_type,
        "latency_ms": result.latency_ms,
        "l4_fallback_status": result.l4_fallback_result.status
        if result.l4_fallback_result is not None
        else None,
        "attempts": [
            {
                "layer": attempt.layer,
                "artifact_id": attempt.artifact_id,
                "decision": attempt.decision,
                "reason_code": attempt.reason,
                "error_type": _shadow_attempt_error_type(attempt),
            }
            for attempt in result.attempts
        ],
    }


def _shadow_attempt_error_type(attempt: LayerAttemptResult) -> str | None:
    if attempt.decision == "error" and attempt.reason in {"health_failure", "circuit_open"}:
        return attempt.reason
    if attempt.decision in {"error", "timeout", "invalid_output", "protocol_error"}:
        return attempt.decision
    return None


def rollback_release(
    target_name: str, registry: ReleaseRegistry, rollback_options: dict[str, Any]
) -> dict[str, Any]:
    key = (target_name, "stable")
    previous = registry.previous_channels.get(key)
    if previous is None:
        raise ReleaseError("no previous stable release for rollback")
    previous_release = load_release(previous, registry)
    if previous_release.target_name != target_name or previous_release.status != "stable":
        raise ReleaseError("previous stable release is not eligible for rollback")
    current = registry.channels.get(key)
    current_release = load_release(current, registry) if current else None
    registry.channels[key] = previous
    if current_release is not None:
        registry.releases[current] = replace(current_release, status="rolled_back")
    registry.releases[previous] = replace(previous_release, status="stable")
    return {"target_name": target_name, "rolled_back_from": current, "stable_release_id": previous}


def retire_release(release_id: str, registry: ReleaseRegistry) -> dict[str, Any]:
    release = load_release(release_id, registry)
    registry.releases[release_id] = replace(release, status="retired")
    return {"release_id": release_id, "status": "retired"}
