"""Model manifest parsing and cryptographic artifact verification."""

from __future__ import annotations

import hashlib
import json
import string
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from number_plate_recognition.errors import ModelIntegrityError


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    name: str
    role: str
    filename: str
    sha256: str
    size_bytes: int
    source: str | None
    download_url: str | None
    license: str
    license_status: str
    provenance_status: str
    production_approved: bool
    task: str
    expected_classes: Mapping[int, str]
    output_map: Mapping[str, str]

    def path_in(self, model_dir: Path) -> Path:
        path = (model_dir / self.filename).resolve()
        try:
            path.relative_to(model_dir.resolve())
        except ValueError as exc:
            raise ModelIntegrityError(f"Unsafe model filename: {self.filename}") from exc
        return path


@dataclass(frozen=True, slots=True)
class ModelManifest:
    schema_version: int
    artifacts: tuple[ModelArtifact, ...]

    def artifact_for_role(self, role: str) -> ModelArtifact:
        matches = [artifact for artifact in self.artifacts if artifact.role == role]
        if len(matches) != 1:
            raise ModelIntegrityError(
                f"Expected exactly one model for role '{role}', found {len(matches)}"
            )
        return matches[0]


def _required_string(item: dict[str, Any], field: str) -> str:
    value = item.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ModelIntegrityError(f"Model manifest field '{field}' must be a string")
    return value.strip()


def _optional_string(item: dict[str, Any], field: str) -> str | None:
    value = item.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ModelIntegrityError(
            f"Model manifest field '{field}' must be a non-empty string or null"
        )
    return value.strip()


def _expected_classes(item: dict[str, Any]) -> Mapping[int, str]:
    raw = item.get("expected_classes")
    if not isinstance(raw, dict) or not raw:
        raise ModelIntegrityError("Model expected_classes must be a non-empty object")
    parsed: dict[int, str] = {}
    for raw_class_id, raw_label in raw.items():
        if not isinstance(raw_class_id, str) or not raw_class_id.isdecimal():
            raise ModelIntegrityError(
                "Model expected_classes keys must be non-negative integer strings"
            )
        class_id = int(raw_class_id)
        if str(class_id) != raw_class_id:
            raise ModelIntegrityError(
                "Model expected_classes keys must use canonical integer strings"
            )
        if not isinstance(raw_label, str) or not raw_label.strip():
            raise ModelIntegrityError("Model expected_classes values must be non-empty strings")
        parsed[class_id] = raw_label.strip()
    return MappingProxyType(parsed)


def _output_map(item: dict[str, Any]) -> Mapping[str, str]:
    raw = item.get("output_map")
    if not isinstance(raw, dict):
        raise ModelIntegrityError("Model output_map must be an object")
    parsed: dict[str, str] = {}
    for raw_label, raw_value in raw.items():
        if (
            not isinstance(raw_label, str)
            or not raw_label.strip()
            or not isinstance(raw_value, str)
            or not raw_value.strip()
        ):
            raise ModelIntegrityError("Model output_map keys and values must be non-empty strings")
        label = raw_label.strip()
        if label in parsed:
            raise ModelIntegrityError("Model output_map contains normalized duplicate keys")
        parsed[label] = raw_value.strip()
    return MappingProxyType(parsed)


def load_manifest(path: Path) -> ModelManifest:
    """Load and validate a versioned JSON model manifest."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ModelIntegrityError(f"Model manifest not found: {path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ModelIntegrityError(f"Cannot read model manifest: {path}") from exc

    if not isinstance(raw, dict) or raw.get("schema_version") != 2:
        raise ModelIntegrityError("Unsupported model manifest schema")
    raw_models = raw.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise ModelIntegrityError("Model manifest must contain a non-empty models list")

    artifacts: list[ModelArtifact] = []
    for raw_model in raw_models:
        if not isinstance(raw_model, dict):
            raise ModelIntegrityError("Each model manifest entry must be an object")
        size = raw_model.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ModelIntegrityError("Model size_bytes must be a positive integer")
        sha256 = _required_string(raw_model, "sha256").lower()
        if len(sha256) != 64 or any(char not in "0123456789abcdef" for char in sha256):
            raise ModelIntegrityError("Model sha256 must contain 64 hexadecimal characters")
        license_status = _required_string(raw_model, "license_status").casefold()
        if license_status not in {"unverified", "approved"}:
            raise ModelIntegrityError("Model license_status must be 'unverified' or 'approved'")
        production_approved = raw_model.get("production_approved")
        if not isinstance(production_approved, bool):
            raise ModelIntegrityError("Model production_approved must be a boolean")
        task = _required_string(raw_model, "task").casefold()
        if task != "detect":
            raise ModelIntegrityError("Only detection model artifacts are supported")
        role = _required_string(raw_model, "role").casefold()
        if role not in {"vehicle", "plate", "character"}:
            raise ModelIntegrityError(f"Unsupported model role: {role}")
        provenance_status = _required_string(raw_model, "provenance_status").casefold()
        if provenance_status not in {"unverified", "verified"}:
            raise ModelIntegrityError("Model provenance_status must be 'unverified' or 'verified'")
        expected_classes = _expected_classes(raw_model)
        output_map = _output_map(raw_model)
        if not set(output_map).issubset(expected_classes.values()):
            raise ModelIntegrityError(
                "Model output_map keys must be declared expected class labels"
            )
        if role != "character" and output_map:
            raise ModelIntegrityError("Only character models may declare output_map")
        if role == "character":
            supported_symbols = frozenset(string.digits + string.ascii_uppercase)
            decoded_labels = (
                output_map.get(label, label).strip().upper() for label in expected_classes.values()
            )
            if any(len(value) != 1 or value not in supported_symbols for value in decoded_labels):
                raise ModelIntegrityError(
                    "Every accepted character class must decode to one ASCII plate symbol"
                )
        artifacts.append(
            ModelArtifact(
                name=_required_string(raw_model, "name"),
                role=role,
                filename=_required_string(raw_model, "filename"),
                sha256=sha256,
                size_bytes=size,
                source=_optional_string(raw_model, "source"),
                download_url=_optional_string(raw_model, "download_url"),
                license=_required_string(raw_model, "license"),
                license_status=license_status,
                provenance_status=provenance_status,
                production_approved=production_approved,
                task=task,
                expected_classes=expected_classes,
                output_map=output_map,
            )
        )

    filenames = [artifact.filename.casefold() for artifact in artifacts]
    roles = [artifact.role for artifact in artifacts]
    if len(filenames) != len(set(filenames)) or len(roles) != len(set(roles)):
        raise ModelIntegrityError("Model filenames and roles must be unique")
    return ModelManifest(schema_version=2, artifacts=tuple(artifacts))


def calculate_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as model_file:
            for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ModelIntegrityError(f"Cannot read model artifact: {path}") from exc
    return digest.hexdigest()


def verify_artifact(artifact: ModelArtifact, model_dir: Path) -> Path:
    """Verify a model's path, byte length, and SHA-256 checksum."""

    path = artifact.path_in(model_dir)
    try:
        stat = path.stat()
    except FileNotFoundError as exc:
        raise ModelIntegrityError(f"Model artifact not found: {path}") from exc
    except OSError as exc:
        raise ModelIntegrityError(f"Cannot inspect model artifact: {path}") from exc
    if not path.is_file():
        raise ModelIntegrityError(f"Model artifact is not a regular file: {path}")
    if stat.st_size != artifact.size_bytes:
        raise ModelIntegrityError(f"Model size does not match manifest: {path.name}")
    if calculate_sha256(path) != artifact.sha256:
        raise ModelIntegrityError(f"Model checksum does not match manifest: {path.name}")
    return path


def require_production_approval(artifact: ModelArtifact) -> None:
    """Reject artifacts whose provenance or license is not production-approved."""

    errors: list[str] = []
    if not artifact.production_approved:
        errors.append("production approval is not recorded")
    if artifact.provenance_status.casefold() != "verified":
        errors.append("provenance is not verified")
    if not artifact.source:
        errors.append("source is not documented")
    if artifact.license_status != "approved":
        errors.append("license is not approved")
    if errors:
        raise ModelIntegrityError(
            f"Model '{artifact.name}' is not approved for production: " + "; ".join(errors)
        )


def verify_manifest_artifacts(manifest: ModelManifest, model_dir: Path) -> None:
    for artifact in manifest.artifacts:
        verify_artifact(artifact, model_dir)
