from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from conftest import write_artifact

from darjeeling.artifact_worker import (
    build_worker_request,
    freeze_artifact_package,
    read_artifact_manifest,
    run_healthcheck,
    run_layer_attempt,
    start_worker,
    validate_artifact_package,
)
from darjeeling.errors import ArtifactError
from darjeeling.model import PackagePolicy, WorkerLimits
from darjeeling.target_definition import load_checked_target
from darjeeling.util import stable_hash


def test_private_evaluation_request_ids_are_ephemeral(target_dir: Path) -> None:
    _, contract, _ = load_checked_target(target_dir)
    first = build_worker_request("stable-row-id", {"text": "a:x"}, 10, "private_eval_ephemeral")
    second = build_worker_request("stable-row-id", {"text": "a:x"}, 10, "private_eval_ephemeral")
    third = build_worker_request(
        "eval-source-row-id", {"text": "a:x"}, 10, "private_eval_ephemeral"
    )
    assert first.request_id != "stable-row-id"
    assert second.request_id != "stable-row-id"
    assert first.request_id != second.request_id
    assert third.request_id != "eval-source-row-id"
    assert third.request_id.startswith("eval-")


def test_invalid_worker_accept_output_falls_back_safely(target_dir: Path, tmp_path: Path) -> None:
    definition, contract, _ = load_checked_target(target_dir)
    artifact_dir = write_artifact(tmp_path / "artifact", definition.contract_hash, bad_output=True)
    manifest = read_artifact_manifest(artifact_dir)
    package = freeze_artifact_package(
        artifact_dir, manifest, tmp_path / "store", "snapshot-digest"
    )
    worker = start_worker(package, WorkerLimits())
    request = build_worker_request("runtime-1", {"text": "a:x"}, 1000, "runtime_stable")
    result = run_layer_attempt(worker, contract, request, "L1")
    assert result.decision == "invalid_output"
    assert result.output is None


def test_package_policy_rejects_network_and_holdout_material(
    target_dir: Path, tmp_path: Path
) -> None:
    definition, _, _ = load_checked_target(target_dir)
    artifact_dir = write_artifact(tmp_path / "artifact", definition.contract_hash)
    (artifact_dir / "validation").mkdir()
    report = validate_artifact_package(
        artifact_dir, "L1", definition.contract_hash, PackagePolicy()
    )
    assert report.status == "fail"


def test_package_policy_rejects_realistic_forbidden_material(
    target_dir: Path, tmp_path: Path
) -> None:
    definition, _, _ = load_checked_target(target_dir)
    artifact_dir = write_artifact(tmp_path / "artifact", definition.contract_hash)
    (artifact_dir / "validation_rows.json").write_text("[]\n", encoding="utf-8")
    (artifact_dir / "registry_credentials.env").write_text("TOKEN=secret\n", encoding="utf-8")
    (artifact_dir / "scaffolding").mkdir()
    (artifact_dir / "scaffolding" / "notes.txt").write_text("compile\n", encoding="utf-8")
    report = validate_artifact_package(
        artifact_dir, "L1", definition.contract_hash, PackagePolicy()
    )
    assert report.status == "fail"
    assert any("validation_rows" in failure for failure in report.failures)
    assert any("registry_credentials" in failure for failure in report.failures)
    assert any("scaffolding" in failure for failure in report.failures)


def test_manifest_validation_rejects_bad_api_layer_and_command(
    target_dir: Path, tmp_path: Path
) -> None:
    definition, _, _ = load_checked_target(target_dir)
    artifact_dir = write_artifact(tmp_path / "artifact", definition.contract_hash)
    manifest_path = artifact_dir / "artifact.yaml"

    original = manifest_path.read_text(encoding="utf-8")
    manifest_path.write_text(original.replace("api_version: v1", "api_version: old"))
    with pytest.raises(ArtifactError, match="api_version"):
        read_artifact_manifest(artifact_dir)

    manifest_path.write_text(original.replace("layer: L1", "layer: L9"))
    with pytest.raises(ArtifactError, match="layer"):
        read_artifact_manifest(artifact_dir)

    manifest_path.write_text(
        original.replace("start_command:\n- python3\n- worker.py", "start_command: []")
    )
    with pytest.raises(ArtifactError, match="start_command"):
        read_artifact_manifest(artifact_dir)

    data = yaml.safe_load(original)
    data["memory_mb"] = 0
    manifest_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ArtifactError, match="memory_mb"):
        read_artifact_manifest(artifact_dir)

    data["memory_mb"] = "many"
    manifest_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ArtifactError, match="memory_mb"):
        read_artifact_manifest(artifact_dir)

    data["memory_mb"] = "64"
    manifest_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ArtifactError, match="memory_mb"):
        read_artifact_manifest(artifact_dir)

    data = yaml.safe_load(original)
    data["timeout_ms"] = "1000"
    manifest_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ArtifactError, match="timeout_ms"):
        read_artifact_manifest(artifact_dir)

    data = yaml.safe_load(original)
    data["contract_hash"] = 123
    manifest_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ArtifactError, match="contract_hash"):
        read_artifact_manifest(artifact_dir)

    data = yaml.safe_load(original)
    data["artifact_id"] = 456
    manifest_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ArtifactError, match="artifact_id"):
        read_artifact_manifest(artifact_dir)

    data = yaml.safe_load(original)
    data["start_command"] = "python3"
    manifest_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ArtifactError, match="start_command"):
        read_artifact_manifest(artifact_dir)


def test_empty_reason_allowlist_rejects_worker_reason(
    target_dir: Path, tmp_path: Path
) -> None:
    definition, contract, _ = load_checked_target(target_dir)
    artifact_dir = write_artifact(tmp_path / "artifact", definition.contract_hash)
    manifest_path = artifact_dir / "artifact.yaml"
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    data["allowed_reason_codes"] = []
    manifest_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    manifest = read_artifact_manifest(artifact_dir)
    assert manifest.allowed_reason_codes == []
    package = freeze_artifact_package(artifact_dir, manifest, tmp_path / "store", "snap-1")
    worker = start_worker(package, WorkerLimits())
    request = build_worker_request("runtime-1", {"text": "a:x"}, 1000, "runtime_stable")
    result = run_layer_attempt(worker, contract, request, "L1")
    assert result.decision == "protocol_error"
    assert result.error == "reason code is not allowed"


def test_missing_worker_executable_returns_error(target_dir: Path, tmp_path: Path) -> None:
    definition, contract, _ = load_checked_target(target_dir)
    artifact_dir = write_artifact(tmp_path / "artifact", definition.contract_hash)
    manifest_path = artifact_dir / "artifact.yaml"
    manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8").replace("python3", "missing-worker-bin")
    )
    manifest = read_artifact_manifest(artifact_dir)
    package = freeze_artifact_package(artifact_dir, manifest, tmp_path / "store", "snap-1")
    worker = start_worker(package, WorkerLimits())
    request = build_worker_request("runtime-1", {"text": "a:x"}, 1000, "runtime_stable")
    result = run_layer_attempt(worker, contract, request, "L1")
    assert result.decision == "error"
    assert result.output is None


def test_freeze_is_content_addressed_and_preserves_snapshot_digest(
    target_dir: Path, tmp_path: Path
) -> None:
    definition, _, _ = load_checked_target(target_dir)
    artifact_dir = write_artifact(tmp_path / "artifact", definition.contract_hash)
    manifest_path = artifact_dir / "artifact.yaml"
    manifest_path.write_text(manifest_path.read_text(encoding="utf-8") + "artifact_id: fixed\n")
    first = freeze_artifact_package(
        artifact_dir, read_artifact_manifest(artifact_dir), tmp_path / "store", "snap-1"
    )
    (first.package_path / "marker.txt").write_text("keep\n", encoding="utf-8")
    with pytest.raises(ArtifactError, match="digest mismatch"):
        freeze_artifact_package(
            artifact_dir, read_artifact_manifest(artifact_dir), tmp_path / "store", "snap-1"
        )
    (artifact_dir / "worker.py").write_text(
        (artifact_dir / "worker.py").read_text(encoding="utf-8") + "\n# changed\n",
        encoding="utf-8",
    )
    second = freeze_artifact_package(
        artifact_dir, read_artifact_manifest(artifact_dir), tmp_path / "store", "snap-2"
    )
    assert first.artifact_id != "fixed"
    assert first.artifact_id != second.artifact_id
    assert first.source_snapshot_digest == "snap-1"
    assert second.source_snapshot_digest == "snap-2"
    assert (first.package_path / "marker.txt").exists()


def test_reused_package_metadata_tracks_latest_source_snapshot_digest(
    target_dir: Path, tmp_path: Path
) -> None:
    definition, _, _ = load_checked_target(target_dir)
    artifact_dir = write_artifact(tmp_path / "artifact", definition.contract_hash)
    manifest = read_artifact_manifest(artifact_dir)
    first = freeze_artifact_package(artifact_dir, manifest, tmp_path / "store", "snap-1")
    second = freeze_artifact_package(artifact_dir, manifest, tmp_path / "store", "snap-2")
    assert first.artifact_id == second.artifact_id
    metadata = json.loads(
        (second.package_path / ".darjeeling-package.json").read_text(encoding="utf-8")
    )
    assert second.source_snapshot_digest == "snap-2"
    assert metadata["source_snapshot_digest"] == "snap-2"
    sidecar = second.package_path / f".darjeeling-package-{stable_hash('snap-2')[:16]}.json"
    assert sidecar.exists()


def test_worker_sandbox_blocks_unmounted_files(target_dir: Path, tmp_path: Path) -> None:
    definition, contract, _ = load_checked_target(target_dir)
    outside = tmp_path / "secret.txt"
    outside.write_text("secret\n", encoding="utf-8")
    artifact_dir = write_artifact(tmp_path / "artifact", definition.contract_hash)
    (artifact_dir / "worker.py").write_text(
        f"""
import json
from pathlib import Path
import sys

request = json.loads(sys.stdin.readline())
try:
    Path({str(outside)!r}).read_text()
    print(json.dumps({{"decision": "accept", "output": {{"label": "leak"}}}}))
except PermissionError:
    print(json.dumps({{"decision": "abstain", "reason": "outside"}}))
""".lstrip(),
        encoding="utf-8",
    )
    manifest = read_artifact_manifest(artifact_dir)
    package = freeze_artifact_package(artifact_dir, manifest, tmp_path / "store", "snap-1")
    worker = start_worker(package, WorkerLimits())
    request = build_worker_request("runtime-1", {"text": "a:x"}, 1000, "runtime_stable")
    result = run_layer_attempt(worker, contract, request, "L1")
    assert result.decision == "abstain"
    text = (
        (artifact_dir / "artifact.yaml")
        .read_text()
        .replace("network: disabled", "network: enabled")
    )
    (artifact_dir / "artifact.yaml").write_text(text)
    report = validate_artifact_package(
        artifact_dir, "L1", definition.contract_hash, PackagePolicy()
    )
    assert report.status == "fail"
    assert any("network" in failure for failure in report.failures)


def test_healthcheck_runs_inside_worker_sandbox(target_dir: Path, tmp_path: Path) -> None:
    definition, _, _ = load_checked_target(target_dir)
    outside = tmp_path / "secret.txt"
    outside.write_text("secret\n", encoding="utf-8")
    artifact_dir = write_artifact(tmp_path / "artifact", definition.contract_hash)
    (artifact_dir / "healthcheck.py").write_text(
        f"""
from pathlib import Path
import sys

try:
    Path({str(outside)!r}).read_text()
except PermissionError:
    sys.exit(0)
else:
    sys.exit(1)
""".lstrip(),
        encoding="utf-8",
    )
    manifest_path = artifact_dir / "artifact.yaml"
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    data["healthcheck_command"] = ["python3", "healthcheck.py"]
    manifest_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    manifest = read_artifact_manifest(artifact_dir)
    package = freeze_artifact_package(artifact_dir, manifest, tmp_path / "store", "snap-1")
    worker = start_worker(package, WorkerLimits())
    assert run_healthcheck(worker, 1000).status == "pass"


def test_request_deadline_caps_worker_timeout(target_dir: Path, tmp_path: Path) -> None:
    definition, contract, _ = load_checked_target(target_dir)
    artifact_dir = write_artifact(tmp_path / "artifact", definition.contract_hash)
    (artifact_dir / "worker.py").write_text(
        """
import time

time.sleep(1)
""".lstrip(),
        encoding="utf-8",
    )
    manifest = read_artifact_manifest(artifact_dir)
    package = freeze_artifact_package(artifact_dir, manifest, tmp_path / "store", "snap-1")
    worker = start_worker(package, WorkerLimits())
    request = build_worker_request("runtime-1", {"text": "a:x"}, 50, "runtime_stable")
    result = run_layer_attempt(worker, contract, request, "L1")
    assert result.decision == "timeout"


def test_worker_cannot_persist_state_to_package(target_dir: Path, tmp_path: Path) -> None:
    definition, contract, _ = load_checked_target(target_dir)
    artifact_dir = write_artifact(tmp_path / "artifact", definition.contract_hash)
    (artifact_dir / "worker.py").write_text(
        """
import json
from pathlib import Path

try:
    Path("state.txt").write_text("persisted")
except PermissionError:
    print(json.dumps({"decision": "abstain", "reason": "outside"}))
else:
    print(json.dumps({"decision": "accept", "output": {"label": "leak"}}))
""".lstrip(),
        encoding="utf-8",
    )
    manifest = read_artifact_manifest(artifact_dir)
    package = freeze_artifact_package(artifact_dir, manifest, tmp_path / "store", "snap-1")
    worker = start_worker(package, WorkerLimits())
    request = build_worker_request("runtime-1", {"text": "a:x"}, 1000, "runtime_stable")
    result = run_layer_attempt(worker, contract, request, "L1")
    assert result.decision == "abstain"
    assert not (package.package_path / "state.txt").exists()


def test_oversized_worker_output_becomes_protocol_error(
    target_dir: Path, tmp_path: Path
) -> None:
    definition, contract, _ = load_checked_target(target_dir)
    artifact_dir = write_artifact(tmp_path / "artifact", definition.contract_hash)
    (artifact_dir / "worker.py").write_text(
        """
print("x" * 70000)
""".lstrip(),
        encoding="utf-8",
    )
    manifest = read_artifact_manifest(artifact_dir)
    package = freeze_artifact_package(artifact_dir, manifest, tmp_path / "store", "snap-1")
    worker = start_worker(package, WorkerLimits())
    request = build_worker_request("runtime-1", {"text": "a:x"}, 1000, "runtime_stable")
    result = run_layer_attempt(worker, contract, request, "L1")
    assert result.decision == "protocol_error"
    assert result.error == "worker output exceeded limit"


def test_non_object_worker_json_becomes_protocol_error(
    target_dir: Path, tmp_path: Path
) -> None:
    definition, contract, _ = load_checked_target(target_dir)
    artifact_dir = write_artifact(tmp_path / "artifact", definition.contract_hash)
    (artifact_dir / "worker.py").write_text("print('[]')\n", encoding="utf-8")
    manifest = read_artifact_manifest(artifact_dir)
    package = freeze_artifact_package(artifact_dir, manifest, tmp_path / "store", "snap-1")
    worker = start_worker(package, WorkerLimits())
    request = build_worker_request("runtime-1", {"text": "a:x"}, 1000, "runtime_stable")
    result = run_layer_attempt(worker, contract, request, "L1")
    assert result.decision == "protocol_error"
    assert result.error == "worker response must be a JSON object"


def test_boolean_confidence_becomes_protocol_error(
    target_dir: Path, tmp_path: Path
) -> None:
    definition, contract, _ = load_checked_target(target_dir)
    artifact_dir = write_artifact(tmp_path / "artifact", definition.contract_hash)
    (artifact_dir / "worker.py").write_text(
        """
import json

print(json.dumps({
    "decision": "accept",
    "output": {"label": "a"},
    "confidence": True,
    "reason": "prefix_match",
}))
""".lstrip(),
        encoding="utf-8",
    )
    manifest = read_artifact_manifest(artifact_dir)
    package = freeze_artifact_package(artifact_dir, manifest, tmp_path / "store", "snap-1")
    worker = start_worker(package, WorkerLimits())
    request = build_worker_request("runtime-1", {"text": "a:x"}, 1000, "runtime_stable")
    result = run_layer_attempt(worker, contract, request, "L1")
    assert result.decision == "protocol_error"
    assert result.error == "confidence must be numeric"


def test_worker_memory_limit_returns_safe_error(target_dir: Path, tmp_path: Path) -> None:
    definition, contract, _ = load_checked_target(target_dir)
    artifact_dir = write_artifact(tmp_path / "artifact", definition.contract_hash)
    (artifact_dir / "worker.py").write_text(
        """
_value = bytearray(256 * 1024 * 1024)
print("unreachable")
""".lstrip(),
        encoding="utf-8",
    )
    manifest_path = artifact_dir / "artifact.yaml"
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    data["memory_mb"] = 64
    manifest_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    manifest = read_artifact_manifest(artifact_dir)
    package = freeze_artifact_package(artifact_dir, manifest, tmp_path / "store", "snap-1")
    worker = start_worker(package, WorkerLimits())
    request = build_worker_request("runtime-1", {"text": "a:x"}, 1000, "runtime_stable")
    result = run_layer_attempt(worker, contract, request, "L1")
    assert result.decision in {"error", "protocol_error", "timeout"}
    assert result.output is None
