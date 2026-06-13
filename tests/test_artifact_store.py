import json

import pytest
from pydantic import ValidationError

from darjeeling.artifacts.store import ArtifactManifest, ArtifactStore, LayerDelta


def test_artifact_store_promotes_manifest_atomically(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    manifest = ArtifactManifest(
        artifact_set_id="gen_001_abcd",
        generation=1,
        target_name="neutral",
        target_schema_version="neutral-v1",
        parent_artifact_set_id="gen_000_prev",
        artifact_paths={"l1": "generations/gen_001/l1"},
        per_layer_deltas={"L1": LayerDelta(coverage_delta=0.1, accepted_accuracy_delta=0.02)},
        promotion_reason="objective improved within gates",
        l3_mode="disabled",
    )

    current_path = store.promote(manifest)
    loaded = store.load_current_manifest()

    assert current_path == store.current_manifest_path
    assert loaded is not None
    assert loaded.promoted
    assert loaded.artifact_set_id == "gen_001_abcd"
    assert loaded.target_name == "neutral"
    assert loaded.target_schema_version == "neutral-v1"
    assert loaded.per_layer_deltas["L1"].coverage_delta == 0.1
    assert (tmp_path / "artifacts" / "generations" / "gen_001" / "manifest.json").exists()


def test_artifact_store_returns_none_without_current_manifest(tmp_path) -> None:
    assert ArtifactStore(tmp_path / "artifacts").load_current_manifest() is None


def test_artifact_manifest_rejects_payload_without_target_identity() -> None:
    with pytest.raises(ValidationError):
        ArtifactManifest.model_validate(
            {
                "artifact_set_id": "gen_001_missing_target",
                "generation": 1,
            }
        )


def test_artifact_store_rejects_current_manifest_without_target_identity(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    store.current_manifest_path.write_text(
        json.dumps(
            {
                "artifact_set_id": "gen_001_missing_target",
                "generation": 1,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        store.load_current_manifest()
