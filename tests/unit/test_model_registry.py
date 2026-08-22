from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from number_plate_recognition.errors import ModelIntegrityError
from number_plate_recognition.model_registry import (
    calculate_sha256,
    load_manifest,
    verify_artifact,
)


def _write_manifest(path: Path, *, digest: str, size: int, filename: str = "model.pt") -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "models": [
                    {
                        "name": "test-model",
                        "role": "vehicle",
                        "filename": filename,
                        "sha256": digest,
                        "size_bytes": size,
                        "task": "detect",
                        "expected_classes": {"2": "car"},
                        "output_map": {},
                    },
                    {
                        "name": "test-plate-model",
                        "role": "plate",
                        "filename": "plate.pt",
                        "sha256": digest,
                        "size_bytes": size,
                        "task": "detect",
                        "expected_classes": {"0": "plate"},
                        "output_map": {},
                    },
                    {
                        "name": "test-character-model",
                        "role": "character",
                        "filename": "character.pt",
                        "sha256": digest,
                        "size_bytes": size,
                        "task": "detect",
                        "expected_classes": {"0": "0"},
                        "output_map": {},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def test_loads_and_verifies_manifest_artifact(tmp_path: Path) -> None:
    payload = b"trusted test model"
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, digest=digest, size=len(payload))

    artifact = load_manifest(manifest_path).artifact_for_role("vehicle")

    assert verify_artifact(artifact, tmp_path) == model_path
    assert calculate_sha256(model_path) == digest


def test_rejects_checksum_mismatch(tmp_path: Path) -> None:
    (tmp_path / "model.pt").write_bytes(b"different")
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, digest="0" * 64, size=len(b"different"))

    artifact = load_manifest(manifest_path).artifact_for_role("vehicle")
    with pytest.raises(ModelIntegrityError, match="checksum"):
        verify_artifact(artifact, tmp_path)


def test_rejects_path_traversal(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, digest="0" * 64, size=1, filename="../model.pt")

    with pytest.raises(ModelIntegrityError, match="Unsafe"):
        load_manifest(manifest_path)


def test_rejects_invalid_manifest(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text('{"schema_version": 3}', encoding="utf-8")

    with pytest.raises(ModelIntegrityError, match="models"):
        load_manifest(manifest_path)


@pytest.mark.parametrize("missing_role", ["vehicle", "plate", "character"])
def test_manifest_requires_every_model_role(tmp_path: Path, missing_role: str) -> None:
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, digest="0" * 64, size=1)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["models"] = [model for model in payload["models"] if model["role"] != missing_role]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ModelIntegrityError, match=rf"missing:.*{missing_role}"):
        load_manifest(manifest_path)


def test_manifest_rejects_duplicate_model_role(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, digest="0" * 64, size=1)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    duplicate = dict(payload["models"][0])
    duplicate.update(name="second-vehicle", filename="second-vehicle.pt")
    payload["models"].append(duplicate)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ModelIntegrityError, match="filenames and roles must be unique"):
        load_manifest(manifest_path)


def test_manifest_rejects_output_map_not_in_class_contract(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, digest="0" * 64, size=1)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    model = payload["models"][0]
    model["role"] = "character"
    model["output_map"] = {"10": "A"}
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ModelIntegrityError, match="expected class"):
        load_manifest(manifest_path)


def test_manifest_rejects_obsolete_or_unknown_fields(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, digest="0" * 64, size=1)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["models"][0]["unexpected_field"] = True
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ModelIntegrityError, match="Unsupported model artifact fields"):
        load_manifest(manifest_path)


def test_manifest_rejects_unsupported_schema(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, digest="0" * 64, size=1)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["schema_version"] = 2
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ModelIntegrityError, match="Unsupported model manifest schema"):
        load_manifest(manifest_path)


def test_manifest_rejects_multi_character_raw_output(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, digest="0" * 64, size=1)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    model = payload["models"][0]
    model["role"] = "character"
    model["expected_classes"] = {"2": "10"}
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ModelIntegrityError, match="ASCII plate symbol"):
        load_manifest(manifest_path)
