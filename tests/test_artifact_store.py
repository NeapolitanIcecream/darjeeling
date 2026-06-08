from darjeeling.artifacts.store import ArtifactManifest, ArtifactStore, LayerDelta


def test_artifact_store_promotes_manifest_atomically(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    manifest = ArtifactManifest(
        artifact_set_id="gen_001_abcd",
        generation=1,
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
    assert loaded.per_layer_deltas["L1"].coverage_delta == 0.1
    assert (tmp_path / "artifacts" / "generations" / "gen_001" / "manifest.json").exists()


def test_artifact_store_returns_none_without_current_manifest(tmp_path) -> None:
    assert ArtifactStore(tmp_path / "artifacts").load_current_manifest() is None
