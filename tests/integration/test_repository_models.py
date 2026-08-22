from __future__ import annotations

from pathlib import Path

from number_plate_recognition.model_registry import (
    load_manifest,
    verify_manifest_artifacts,
)


def test_repository_model_artifacts_match_manifest(project_root: Path) -> None:
    model_dir = project_root / "models"
    manifest = load_manifest(model_dir / "manifest.json")

    verify_manifest_artifacts(manifest, model_dir)
