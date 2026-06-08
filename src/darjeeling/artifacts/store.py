from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class LayerDelta(BaseModel):
    coverage_delta: float = 0.0
    accepted_accuracy_delta: float = 0.0
    wrong_accept_delta: float = 0.0
    p95_latency_ms_delta: float = 0.0
    cost_delta: float = 0.0
    layer_share_delta: float = 0.0


class ArtifactManifest(BaseModel):
    artifact_set_id: str
    generation: int
    parent_artifact_set_id: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    schema_versions: dict[str, str] = Field(default_factory=dict)
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    candidate_metrics: dict[str, Any] = Field(default_factory=dict)
    per_layer_deltas: dict[str, LayerDelta] = Field(default_factory=dict)
    promoted_with_layer_regression: bool = False
    regressed_layers: list[str] = Field(default_factory=list)
    promoted: bool = False
    promotion_reason: str = ""
    l3_mode: str = "disabled"


class ArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def current_manifest_path(self) -> Path:
        return self.root / "manifest.current.json"

    def generation_dir(self, generation: int) -> Path:
        return self.root / "generations" / f"gen_{generation:03d}"

    def write_json(self, name: str, payload: dict[str, Any]) -> Path:
        path = self.root / name
        self._atomic_write_json(path, payload)
        return path

    def write_generation_json(
        self,
        generation: int,
        relative_path: str,
        payload: dict[str, Any],
    ) -> Path:
        path = self.generation_dir(generation) / relative_path
        self._atomic_write_json(path, payload)
        return path

    def write_generation_manifest(self, manifest: ArtifactManifest) -> Path:
        path = self.generation_dir(manifest.generation) / "manifest.json"
        self._atomic_write_json(path, manifest.model_dump(mode="json"))
        return path

    def promote(self, manifest: ArtifactManifest) -> Path:
        promoted_manifest = manifest.model_copy(update={"promoted": True})
        self.write_generation_manifest(promoted_manifest)
        self._atomic_write_json(
            self.current_manifest_path,
            promoted_manifest.model_dump(mode="json"),
        )
        return self.current_manifest_path

    def load_current_manifest(self) -> ArtifactManifest | None:
        if not self.current_manifest_path.exists():
            return None
        return ArtifactManifest.model_validate_json(
            self.current_manifest_path.read_text(encoding="utf-8")
        )

    def _atomic_write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.tmp")
        tmp_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp_path.replace(path)
