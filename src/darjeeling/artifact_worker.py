from __future__ import annotations

import json
import math
import os
import selectors
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import yaml

try:
    import resource
except ImportError:  # pragma: no cover - non-Unix fallback
    resource = None  # type: ignore[assignment]

from darjeeling.errors import ArtifactError, ValidationError
from darjeeling.model import (
    ArtifactCheckReport,
    ArtifactManifest,
    ArtifactPackage,
    HealthcheckResult,
    LayerAttemptResult,
    PackagePolicy,
    ProtocolDocs,
    ProtocolError,
    RawWorkerCallResult,
    TargetRuntimeContract,
    WorkerHandle,
    WorkerLimits,
    WorkerRequest,
    WorkerResponse,
    WorkerStopResult,
)
from darjeeling.portable_sandbox import build_python_sandbox_command, is_python_command
from darjeeling.util import bounded_code, file_digest, new_id, stable_hash, tree_digest, write_json

_PACKAGE_METADATA_NAME = ".darjeeling-package.json"
_SANDBOX_PROFILE_NAME = ".worker_sandbox.sb"
_MAX_STDOUT_BYTES = 64 * 1024
_MAX_STDERR_BYTES = 16 * 1024
_FORBIDDEN_EXACT_PATH_PARTS = {
    "validation_rows",
    "validation_records",
    "validation_data",
    "test_fixtures",
    "test_rows",
    "test_records",
    "test_data",
    "train_data",
    "train_rows",
    "train_records",
    "training_data",
    "registry_credentials",
    "broker_credentials",
    "search_scripts",
}
_FORBIDDEN_PATH_TOKENS = {
    "validation",
    "holdout",
    "registry",
    "credentials",
    "credential",
    "broker",
    "scaffolding",
    "journal",
}
_DATA_TOKENS = {"data", "dataset", "datasets", "rows", "records", "examples", "fixtures"}


def _required_string_list(raw: dict[str, Any], field: str) -> list[str]:
    if field not in raw:
        raise KeyError(field)
    value = raw[field]
    if not isinstance(value, list):
        raise ArtifactError(f"{field} must be a list of strings")
    if not value:
        raise ArtifactError(f"{field} must be a non-empty string list")
    if not all(isinstance(part, str) and part for part in value):
        raise ArtifactError(f"{field} must be a non-empty string list")
    return list(value)


def _optional_string_list(
    raw: dict[str, Any], field: str, *, allow_empty: bool = False
) -> list[str] | None:
    if field not in raw or raw[field] is None:
        return None
    value = raw[field]
    if not isinstance(value, list):
        raise ArtifactError(f"{field} must be a list of strings")
    if not allow_empty and not value:
        raise ArtifactError(f"{field} must be a non-empty string list")
    if not all(isinstance(part, str) and part for part in value):
        raise ArtifactError(f"{field} must be a string list")
    return list(value)


def _required_string(raw: dict[str, Any], field: str) -> str:
    if field not in raw:
        raise KeyError(field)
    value = raw[field]
    if not isinstance(value, str):
        raise ArtifactError(f"{field} must be a string")
    return value


def _required_int(raw: dict[str, Any], field: str) -> int:
    if field not in raw:
        raise KeyError(field)
    value = raw[field]
    if type(value) is not int:
        raise ArtifactError(f"{field} must be an integer")
    return value


def _optional_positive_int(raw: dict[str, Any], field: str) -> int | None:
    if field not in raw or raw[field] is None:
        return None
    value = raw[field]
    if type(value) is not int:
        raise ArtifactError(f"{field} must be an integer")
    if value <= 0:
        raise ArtifactError(f"{field} must be positive when present")
    return value


def read_artifact_manifest(artifact_dir: Path) -> ArtifactManifest:
    path = artifact_dir / "artifact.yaml"
    if not path.exists():
        raise ArtifactError(f"missing artifact manifest: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ArtifactError("artifact.yaml must contain a mapping")
    try:
        manifest = ArtifactManifest(
            api_version=_required_string(raw, "api_version"),
            layer=_required_string(raw, "layer"),
            start_command=_required_string_list(raw, "start_command"),
            healthcheck_command=_optional_string_list(raw, "healthcheck_command"),
            protocol=_required_string(raw, "protocol"),
            timeout_ms=_required_int(raw, "timeout_ms"),
            memory_mb=_optional_positive_int(raw, "memory_mb"),
            network=_required_string(raw, "network"),
            contract_hash=_required_string(raw, "contract_hash"),
            artifact_id=raw.get("artifact_id"),
            allowed_reason_codes=_optional_string_list(
                raw, "allowed_reason_codes", allow_empty=True
            ),
        )
    except KeyError as exc:
        raise ArtifactError(f"artifact manifest missing field: {exc.args[0]}") from exc
    if manifest.api_version != "v1":
        raise ArtifactError("unsupported artifact api_version")
    if manifest.layer not in {"L1", "L2", "L3"}:
        raise ArtifactError("artifact layer must be L1, L2, or L3")
    if manifest.protocol != "jsonl":
        raise ArtifactError("only jsonl protocol is implemented by the reboot worker adapter")
    if manifest.network != "disabled":
        raise ArtifactError("artifact network must be disabled")
    if manifest.timeout_ms <= 0:
        raise ArtifactError("timeout_ms must be positive")
    if manifest.artifact_id is not None and not isinstance(manifest.artifact_id, str):
        raise ArtifactError("artifact_id must be a string when present")
    if manifest.allowed_reason_codes is not None:
        if len(manifest.allowed_reason_codes) > 128:
            raise ArtifactError("too many allowed reason codes")
        bad = [code for code in manifest.allowed_reason_codes if not bounded_code(code)]
        if bad:
            raise ArtifactError(f"invalid reason codes: {bad}")
    return manifest


def _normalized_tokens(path: Path) -> tuple[set[str], set[str]]:
    parts: set[str] = set()
    tokens: set[str] = set()
    for part in path.parts:
        normalized = part.lower().replace("-", "_").replace(".", "_")
        parts.add(normalized)
        tokens.update(piece for piece in normalized.split("_") if piece)
    return parts, tokens


def _forbidden_runtime_material(path: Path) -> str | None:
    parts, tokens = _normalized_tokens(path)
    if parts & _FORBIDDEN_EXACT_PATH_PARTS:
        return "forbidden runtime package path"
    if tokens & _FORBIDDEN_PATH_TOKENS:
        return "forbidden runtime package path"
    if tokens & {"train", "training"} and tokens & _DATA_TOKENS:
        return "forbidden runtime package data path"
    if "test" in tokens and tokens & _DATA_TOKENS:
        return "forbidden runtime package data path"
    return None


def validate_artifact_package(
    artifact_dir: Path,
    expected_layer: Literal["L1", "L2", "L3"],
    contract_hash: str,
    package_policy: PackagePolicy,
) -> ArtifactCheckReport:
    failures: list[str] = []
    manifest: ArtifactManifest | None = None
    try:
        manifest = read_artifact_manifest(artifact_dir)
        if manifest.layer != expected_layer:
            failures.append("manifest layer does not match expected layer")
        if manifest.contract_hash != contract_hash:
            failures.append("manifest contract hash mismatch")
        for path in artifact_dir.rglob("*"):
            rel_path = path.relative_to(artifact_dir)
            rel_parts = {part.lower() for part in rel_path.parts}
            if rel_parts & set(package_policy.forbidden_names):
                failures.append(f"forbidden runtime package path: {rel_path}")
            forbidden = _forbidden_runtime_material(rel_path)
            if forbidden is not None:
                failures.append(f"{forbidden}: {rel_path}")
            if path.is_symlink():
                target = path.resolve()
                if (
                    artifact_dir.resolve() not in target.parents
                    and target != artifact_dir.resolve()
                ):
                    failures.append(f"symlink escapes package: {path.relative_to(artifact_dir)}")
    except Exception as exc:
        failures.append(str(exc))
    return ArtifactCheckReport(
        artifact_dir=artifact_dir,
        expected_layer=expected_layer,
        status="fail" if failures else "pass",
        manifest=manifest,
        failures=failures,
    )


def freeze_artifact_package(
    artifact_dir: Path,
    manifest: ArtifactManifest,
    artifact_store: Path,
    source_snapshot_digest: str,
) -> ArtifactPackage:
    digest = tree_digest(artifact_dir)
    artifact_id = f"{manifest.layer.lower()}-{digest[:16]}"
    package_path = artifact_store / artifact_id
    if not package_path.exists():
        shutil.copytree(artifact_dir, package_path, symlinks=False)
    elif _runtime_package_digest(package_path) != digest:
        raise ArtifactError("existing artifact package digest mismatch")
    package = ArtifactPackage(
        artifact_id=artifact_id,
        layer=manifest.layer,
        package_path=package_path,
        manifest=manifest,
        digest=digest,
        source_snapshot_digest=source_snapshot_digest,
        build_provenance={"artifact_dir": str(artifact_dir)},
    )
    write_json(package_path / _PACKAGE_METADATA_NAME, asdict(package))
    write_json(
        package_path / f".darjeeling-package-{stable_hash(source_snapshot_digest)[:16]}.json",
        asdict(package),
    )
    return package


def _runtime_package_digest(package_path: Path) -> str:
    entries: list[tuple[str, str]] = []
    for item in sorted(package_path.rglob("*")):
        rel = item.relative_to(package_path).as_posix()
        if item.name == _SANDBOX_PROFILE_NAME or item.name.startswith(".darjeeling-package"):
            continue
        if item.is_file() and not item.is_symlink():
            entries.append((rel, file_digest(item)))
        elif item.is_symlink():
            entries.append((rel, f"symlink:{os.readlink(item)}"))
    return stable_hash(entries)


def _copy_runtime_package(source: Path, destination: Path) -> None:
    shutil.copytree(
        source,
        destination,
        symlinks=False,
        ignore=lambda _dir, names: [
            name
            for name in names
            if name == _SANDBOX_PROFILE_NAME or name.startswith(".darjeeling-package")
        ],
    )


def verify_artifact_package_digest(package: ArtifactPackage) -> None:
    if _runtime_package_digest(package.package_path) != package.digest:
        raise ArtifactError("artifact package digest mismatch")


def start_worker(package: ArtifactPackage, worker_limits: WorkerLimits) -> WorkerHandle:
    verify_artifact_package_digest(package)
    if package.manifest.network != "disabled":
        raise ArtifactError("artifact worker network must be disabled")
    if worker_limits.timeout_ms is not None and worker_limits.timeout_ms <= 0:
        raise ArtifactError("worker timeout limit must be positive when present")
    if worker_limits.memory_mb is not None and worker_limits.memory_mb <= 0:
        raise ArtifactError("worker memory limit must be positive when present")
    return WorkerHandle(package=package, limits=worker_limits)


def _path_contains(parent: Path, child: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _sandbox_path(value: Path) -> str:
    resolved = str(value.resolve())
    return '"' + resolved.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _protect_siblings_from(root: Path, package_path: Path) -> list[Path]:
    protected: list[Path] = []
    current = root.resolve()
    package_resolved = package_path.resolve()
    while current != package_resolved and current.is_dir():
        try:
            rel_parts = package_resolved.relative_to(current).parts
        except ValueError:
            break
        if not rel_parts:
            break
        allowed_child = current / rel_parts[0]
        for child in current.iterdir():
            if child != allowed_child:
                protected.append(child)
        current = allowed_child
    return protected


def _append_protected_path(protected: list[Path], path: Path) -> None:
    resolved = path.resolve()
    if not any(existing.resolve() == resolved for existing in protected):
        protected.append(path)


def _worker_sandbox_command(
    worker: WorkerHandle,
    runtime_package_path: Path,
    scratch_path: Path,
    command: list[str] | None = None,
) -> list[str]:
    sandbox_exec = shutil.which("sandbox-exec")
    package_path = runtime_package_path
    repo_root = Path(__file__).resolve().parents[2]
    protected: list[Path] = []
    for path in [
        repo_root,
        Path.home(),
        Path("/tmp"),
        Path("/private/tmp"),
        Path("/var/tmp"),
        Path("/private/var/tmp"),
        Path("/etc"),
        Path("/private/etc"),
        worker.package.package_path,
    ]:
        if not _path_contains(path, package_path) and not _path_contains(path, scratch_path):
            _append_protected_path(protected, path)
    if len(worker.package.package_path.parents) >= 2:
        for path in _protect_siblings_from(
            worker.package.package_path.parents[1], worker.package.package_path
        ):
            if not _path_contains(path, package_path) and not _path_contains(path, scratch_path):
                _append_protected_path(protected, path)
    sandboxed_command = list(command or worker.package.manifest.start_command)
    executable = (
        "/usr/bin/python3"
        if sandboxed_command[0] in {"python", "python3"} and Path("/usr/bin/python3").exists()
        else shutil.which(sandboxed_command[0])
    )
    if executable:
        sandboxed_command[0] = executable
    if sandbox_exec is None:
        if not is_python_command(sandboxed_command):
            raise ArtifactError(
                "sandbox-exec or a Python artifact command is required for worker isolation"
            )
        return build_python_sandbox_command(
            sandboxed_command,
            cwd=package_path,
            config_path=scratch_path / "python_sandbox.json",
            allowed_read_roots=[package_path, scratch_path],
            allowed_write_roots=[scratch_path],
            denied_read_roots=protected,
            denied_write_roots=[package_path, *protected],
        )
    profile_path = scratch_path / _SANDBOX_PROFILE_NAME
    lines = ["(version 1)", "(allow default)", "(deny network*)"]
    lines.append(f"(deny file-write* (subpath {_sandbox_path(package_path)}))")
    for path in protected:
        if path.exists():
            lines.append(f"(deny file-read* (subpath {_sandbox_path(path)}))")
            lines.append(f"(deny file-write* (subpath {_sandbox_path(path)}))")
    profile_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [sandbox_exec, "-f", str(profile_path), *sandboxed_command]


def _worker_environment(scratch_path: Path) -> dict[str, str]:
    scratch = str(scratch_path)
    return {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HOME": scratch,
        "TMPDIR": scratch,
        "TMP": scratch,
        "TEMP": scratch,
    }


def _worker_resource_preexec(memory_mb: int | None, timeout_ms: int):
    if resource is None:
        return None

    def apply_limits() -> None:
        cpu_seconds = max(1, math.ceil(timeout_ms / 1000) + 1)
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        except (OSError, ValueError):
            pass
        if memory_mb is None:
            return
        memory_bytes = memory_mb * 1024 * 1024
        for limit_name in ["RLIMIT_AS", "RLIMIT_DATA"]:
            limit = getattr(resource, limit_name, None)
            if limit is None:
                continue
            try:
                resource.setrlimit(limit, (memory_bytes, memory_bytes))
            except (OSError, ValueError):
                pass

    return apply_limits


def _bounded_subprocess_run(
    command: list[str],
    cwd: Path,
    input_bytes: bytes | None,
    timeout_ms: int,
    env: dict[str, str],
    memory_mb: int | None,
) -> tuple[int, bytes, bytes, bool]:
    started = time.perf_counter()
    proc = subprocess.Popen(
        command,
        cwd=cwd,
        stdin=subprocess.PIPE if input_bytes is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        preexec_fn=_worker_resource_preexec(memory_mb, timeout_ms),
    )
    if input_bytes is not None and proc.stdin is not None:
        proc.stdin.write(input_bytes)
        proc.stdin.close()
    selector = selectors.DefaultSelector()
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    limits = {"stdout": _MAX_STDOUT_BYTES, "stderr": _MAX_STDERR_BYTES}
    assert proc.stdout is not None
    assert proc.stderr is not None
    for name, pipe in {"stdout": proc.stdout, "stderr": proc.stderr}.items():
        os.set_blocking(pipe.fileno(), False)
        selector.register(pipe, selectors.EVENT_READ, name)
    output_limited = False
    try:
        while selector.get_map():
            remaining = timeout_ms / 1000 - (time.perf_counter() - started)
            if remaining <= 0:
                proc.kill()
                proc.wait()
                raise subprocess.TimeoutExpired(command, timeout_ms / 1000)
            for key, _ in selector.select(remaining):
                stream_name = key.data
                try:
                    chunk = os.read(key.fileobj.fileno(), 8192)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(key.fileobj)
                    key.fileobj.close()
                    continue
                buffer = buffers[stream_name]
                limit = limits[stream_name]
                if len(buffer) + len(chunk) > limit:
                    remaining_bytes = max(0, limit + 1 - len(buffer))
                    buffer.extend(chunk[:remaining_bytes])
                    output_limited = True
                    proc.kill()
                    break
                buffer.extend(chunk)
            if output_limited:
                break
        try:
            proc.wait(timeout=max(0.0, timeout_ms / 1000 - (time.perf_counter() - started)))
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise
    finally:
        selector.close()
        for pipe in [proc.stdout, proc.stderr]:
            if pipe is not None and not pipe.closed:
                pipe.close()
    return proc.returncode or 0, bytes(buffers["stdout"]), bytes(buffers["stderr"]), output_limited


def run_healthcheck(worker: WorkerHandle, timeout_ms: int) -> HealthcheckResult:
    command = worker.package.manifest.healthcheck_command
    if not command:
        return HealthcheckResult(status="fail", message="healthcheck command is required")
    try:
        with tempfile.TemporaryDirectory(prefix="darjeeling-worker-") as run_root_raw:
            run_root = Path(run_root_raw)
            runtime_package = run_root / "package"
            scratch = run_root / "scratch"
            scratch.mkdir()
            _copy_runtime_package(worker.package.package_path, runtime_package)
            returncode, _stdout, stderr, output_limited = _bounded_subprocess_run(
                _worker_sandbox_command(worker, runtime_package, scratch, command),
                runtime_package,
                None,
                timeout_ms,
                _worker_environment(scratch),
                worker.limits.memory_mb or worker.package.manifest.memory_mb,
            )
    except Exception as exc:
        return HealthcheckResult(status="fail", message=str(exc))
    if output_limited:
        return HealthcheckResult(status="fail", message="worker output exceeded limit")
    if returncode != 0:
        return HealthcheckResult(
            status="fail", message=stderr.decode("utf-8", "replace")[:400]
        )
    return HealthcheckResult(status="pass", message="ok")


def build_worker_request(
    request_id: str,
    input_value: dict[str, Any],
    deadline_ms: int,
    request_id_policy: Literal["runtime_stable", "private_eval_ephemeral"],
) -> WorkerRequest:
    if request_id_policy == "private_eval_ephemeral":
        request_id = new_id("eval")
    return WorkerRequest(request_id=request_id, input=input_value, deadline_ms=deadline_ms)


def build_planned_private_worker_request(
    request_id: str,
    input_value: dict[str, Any],
    deadline_ms: int,
) -> WorkerRequest:
    if not request_id.startswith("eval-") or not bounded_code(request_id, max_len=80):
        raise ArtifactError("planned private evaluation request id must be opaque eval id")
    return WorkerRequest(request_id=request_id, input=input_value, deadline_ms=deadline_ms)


def call_worker(
    worker: WorkerHandle, request: WorkerRequest, call_timeout_ms: int
) -> RawWorkerCallResult:
    started = time.perf_counter()
    payload = json.dumps(asdict(request), separators=(",", ":")) + "\n"
    try:
        with tempfile.TemporaryDirectory(prefix="darjeeling-worker-") as run_root_raw:
            run_root = Path(run_root_raw)
            runtime_package = run_root / "package"
            scratch = run_root / "scratch"
            scratch.mkdir()
            _copy_runtime_package(worker.package.package_path, runtime_package)
            returncode, stdout, stderr, output_limited = _bounded_subprocess_run(
                _worker_sandbox_command(worker, runtime_package, scratch),
                runtime_package,
                payload.encode("utf-8"),
                call_timeout_ms,
                _worker_environment(scratch),
                worker.limits.memory_mb or worker.package.manifest.memory_mb,
            )
    except subprocess.TimeoutExpired:
        return RawWorkerCallResult(
            status="timeout",
            response_bytes=None,
            latency_ms=(time.perf_counter() - started) * 1000,
            error="timeout",
        )
    except (OSError, ArtifactError) as exc:
        return RawWorkerCallResult(
            status="error",
            response_bytes=None,
            latency_ms=(time.perf_counter() - started) * 1000,
            error=str(exc),
        )
    latency = (time.perf_counter() - started) * 1000
    if output_limited:
        return RawWorkerCallResult(
            status="ok",
            response_bytes=None,
            latency_ms=latency,
            error="worker output exceeded limit",
        )
    if returncode != 0:
        return RawWorkerCallResult(
            status="error",
            response_bytes=stdout,
            latency_ms=latency,
            error=stderr.decode("utf-8", "replace")[:400],
        )
    return RawWorkerCallResult(status="ok", response_bytes=stdout, latency_ms=latency)


def parse_worker_response(raw_result: RawWorkerCallResult) -> WorkerResponse | ProtocolError:
    if raw_result.status != "ok" or raw_result.response_bytes is None:
        return ProtocolError(raw_result.error or raw_result.status)
    lines = [
        line
        for line in raw_result.response_bytes.decode("utf-8", "replace").splitlines()
        if line.strip()
    ]
    if len(lines) != 1:
        return ProtocolError("worker must emit exactly one JSON response line")
    try:
        payload = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        return ProtocolError(f"invalid JSON response: {exc}")
    if not isinstance(payload, dict):
        return ProtocolError("worker response must be a JSON object")
    decision = payload.get("decision")
    if decision not in {"accept", "abstain"}:
        return ProtocolError("decision must be accept or abstain")
    output = payload.get("output")
    if decision == "accept" and not isinstance(output, dict):
        return ProtocolError("accept responses require object output")
    if decision == "abstain" and output is not None:
        return ProtocolError("abstain responses must not include output")
    confidence = payload.get("confidence")
    if confidence is not None and (
        isinstance(confidence, bool) or not isinstance(confidence, int | float)
    ):
        return ProtocolError("confidence must be numeric")
    reason = payload.get("reason")
    if reason is not None and not isinstance(reason, str):
        return ProtocolError("reason must be string")
    return WorkerResponse(
        decision=decision,
        output=output,
        confidence=float(confidence) if confidence is not None else None,
        reason=reason,
    )


def validate_accept_output(
    contract: TargetRuntimeContract, response: WorkerResponse
) -> dict[str, Any]:
    if response.decision != "accept" or response.output is None:
        raise ValidationError("only accept responses have output")
    return contract.validate_output(response.output)


def run_layer_attempt(
    worker: WorkerHandle,
    contract: TargetRuntimeContract,
    request: WorkerRequest,
    layer: Literal["L1", "L2", "L3"],
) -> LayerAttemptResult:
    effective_timeout_ms = min(
        value
        for value in [
            worker.package.manifest.timeout_ms,
            request.deadline_ms,
            worker.limits.timeout_ms,
        ]
        if value is not None
    )
    raw = call_worker(worker, request, effective_timeout_ms)
    if raw.status == "timeout":
        return LayerAttemptResult(
            layer,
            worker.package.artifact_id,
            "timeout",
            None,
            None,
            None,
            raw.latency_ms,
            raw.error,
        )
    if raw.status == "error":
        return LayerAttemptResult(
            layer, worker.package.artifact_id, "error", None, None, None, raw.latency_ms, raw.error
        )
    parsed = parse_worker_response(raw)
    if isinstance(parsed, ProtocolError):
        return LayerAttemptResult(
            layer,
            worker.package.artifact_id,
            "protocol_error",
            None,
            None,
            None,
            raw.latency_ms,
            parsed.message,
        )
    if parsed.reason is not None:
        if not bounded_code(parsed.reason):
            return LayerAttemptResult(
                layer,
                worker.package.artifact_id,
                "protocol_error",
                None,
                parsed.confidence,
                None,
                raw.latency_ms,
                "invalid reason code",
            )
        allowed = worker.package.manifest.allowed_reason_codes
        if allowed is not None and parsed.reason not in allowed:
            return LayerAttemptResult(
                layer,
                worker.package.artifact_id,
                "protocol_error",
                None,
                parsed.confidence,
                None,
                raw.latency_ms,
                "reason code is not allowed",
            )
    if parsed.decision == "abstain":
        return LayerAttemptResult(
            layer,
            worker.package.artifact_id,
            "abstain",
            None,
            parsed.confidence,
            parsed.reason,
            raw.latency_ms,
        )
    try:
        output = validate_accept_output(contract, parsed)
    except Exception as exc:
        return LayerAttemptResult(
            layer,
            worker.package.artifact_id,
            "invalid_output",
            None,
            parsed.confidence,
            parsed.reason,
            raw.latency_ms,
            str(exc),
        )
    return LayerAttemptResult(
        layer,
        worker.package.artifact_id,
        "accept",
        output,
        parsed.confidence,
        parsed.reason,
        raw.latency_ms,
    )


def stop_worker(worker: WorkerHandle, reason: str) -> WorkerStopResult:
    return WorkerStopResult(stopped=True)


def build_protocol_docs(protocol_version: str) -> ProtocolDocs:
    text = """# Darjeeling Worker Protocol

Workers receive one JSON line with request_id, input, and deadline_ms. They emit one JSON
line with decision=accept and an output object, or decision=abstain. The reason field is
a bounded reason code, not free-form text. Runtime artifacts must not call L4, other
layers, registries, or validation/test data.
"""
    return ProtocolDocs(protocol_version=protocol_version, path=None, text=text)


class ArtifactWorkerClient:
    def __init__(self, store: Path):
        self.store = store

    def freeze(
        self,
        artifact_dir: Path,
        expected_layer: Literal["L1", "L2", "L3"],
        contract_hash: str,
        source_snapshot_digest: str,
    ) -> ArtifactPackage:
        report = validate_artifact_package(
            artifact_dir, expected_layer, contract_hash, PackagePolicy()
        )
        if report.status != "pass" or report.manifest is None:
            raise ArtifactError("; ".join(report.failures))
        return freeze_artifact_package(
            artifact_dir, report.manifest, self.store, source_snapshot_digest
        )
