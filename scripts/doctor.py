#!/usr/bin/env python3
"""Validate the local development environment and model registry."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import re
import string
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
DEVICE_PATTERN = re.compile(r"^(?:auto|cpu|mps|cuda(?::\d+)?)$")
VALID_LOG_LEVELS = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}

# Keep dependency import checks offline and isolate Ultralytics' generated
# settings from user profiles and the repository root.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("YOLO_OFFLINE", "true")
os.environ.setdefault(
    "YOLO_CONFIG_DIR",
    str(REPOSITORY_ROOT / ".runtime" / "ultralytics"),
)


class ManifestError(ValueError):
    """Raised when the model manifest is malformed."""


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """A validated model artifact declaration."""

    name: str
    role: str
    filename: str
    sha256: str
    size_bytes: int
    task: str
    license_text: str
    provenance_status: str
    license_status: str
    production_approved: bool
    expected_classes: tuple[tuple[str, str], ...]
    output_map: tuple[tuple[str, str], ...]
    source: str | None = None
    download_url: str | None = None


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One diagnostic result."""

    name: str
    status: str
    message: str


def repository_path(value: str | Path) -> Path:
    """Resolve NPR_APP_ROOT using the application's current-directory semantics."""
    path = Path(value).expanduser()
    return path.resolve()


def app_path(value: str | Path, app_root: Path) -> Path:
    """Resolve a configured application path against NPR_APP_ROOT."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (app_root / path).resolve()


def _required_string(entry: dict[str, Any], key: str, index: int) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"models[{index}].{key} must be a non-empty string")
    return value.strip()


def _string_mapping(
    entry: dict[str, Any],
    key: str,
    index: int,
    *,
    allow_empty: bool,
    numeric_keys: bool,
) -> tuple[tuple[str, str], ...]:
    value = entry.get(key)
    if not isinstance(value, dict) or (not allow_empty and not value):
        qualifier = "an object" if allow_empty else "a non-empty object"
        raise ManifestError(f"models[{index}].{key} must be {qualifier}")
    normalized: list[tuple[str, str]] = []
    normalized_keys: set[str] = set()
    for raw_key, raw_label in value.items():
        if not isinstance(raw_key, str) or not raw_key.strip():
            raise ManifestError(f"models[{index}].{key} keys must be non-empty strings")
        if numeric_keys:
            if not raw_key.isdecimal() or str(int(raw_key)) != raw_key:
                raise ManifestError(
                    f"models[{index}].{key} keys must be canonical non-negative integer strings"
                )
            normalized_key = raw_key
        else:
            normalized_key = raw_key.strip()
        if normalized_key in normalized_keys:
            raise ManifestError(f"models[{index}].{key} contains normalized duplicate keys")
        if not isinstance(raw_label, str) or not raw_label.strip():
            raise ManifestError(f"models[{index}].{key} values must be non-empty strings")
        normalized_keys.add(normalized_key)
        normalized.append((normalized_key, raw_label.strip()))
    return tuple(normalized)


def load_manifest(path: Path) -> list[ModelSpec]:
    """Parse and validate a version-2 model manifest."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManifestError(f"manifest does not exist: {path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read manifest {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ManifestError("manifest root must be a JSON object")
    if raw.get("schema_version") != 2:
        raise ManifestError("schema_version must be 2")

    entries = raw.get("models")
    if not isinstance(entries, list) or not entries:
        raise ManifestError("models must be a non-empty list")

    models: list[ModelSpec] = []
    roles: set[str] = set()
    filenames: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ManifestError(f"models[{index}] must be a JSON object")

        name = _required_string(entry, "name", index)
        role = _required_string(entry, "role", index).casefold()
        if role not in {"vehicle", "plate", "character"}:
            raise ManifestError(f"models[{index}].role is unsupported: {role}")
        filename = _required_string(entry, "filename", index)
        digest = _required_string(entry, "sha256", index)
        task = _required_string(entry, "task", index).casefold()
        if task != "detect":
            raise ManifestError(f"models[{index}].task must be 'detect'")
        size_bytes = entry.get("size_bytes")
        download_url = entry.get("download_url")
        source = entry.get("source")
        license_text = _required_string(entry, "license", index)
        provenance_status = _required_string(entry, "provenance_status", index).casefold()
        if provenance_status not in {"unverified", "verified"}:
            raise ManifestError(f"models[{index}].provenance_status must be unverified or verified")
        license_status = _required_string(entry, "license_status", index).lower()
        production_approved = entry.get("production_approved")
        expected_classes = _string_mapping(
            entry,
            "expected_classes",
            index,
            allow_empty=False,
            numeric_keys=True,
        )
        output_map = _string_mapping(
            entry,
            "output_map",
            index,
            allow_empty=True,
            numeric_keys=False,
        )
        expected_labels = {label for _, label in expected_classes}
        output_labels = {label for label, _ in output_map}
        if not output_labels.issubset(expected_labels):
            raise ManifestError(
                f"models[{index}].output_map keys must be declared expected class labels"
            )
        if role != "character" and output_map:
            raise ManifestError(f"models[{index}].output_map is only valid for character models")
        if role == "character":
            output_mapping = dict(output_map)
            supported_symbols = frozenset(string.digits + string.ascii_uppercase)
            decoded_labels = (
                output_mapping.get(label, label).strip().upper() for label in expected_labels
            )
            if any(len(value) != 1 or value not in supported_symbols for value in decoded_labels):
                raise ManifestError(
                    f"models[{index}] character classes must decode to one ASCII plate symbol"
                )

        artifact_path = Path(filename)
        if artifact_path.is_absolute() or ".." in artifact_path.parts:
            raise ManifestError(f"models[{index}].filename must be a safe relative path")
        if artifact_path == Path("."):
            raise ManifestError(f"models[{index}].filename must name a file")
        if not SHA256_PATTERN.fullmatch(digest):
            raise ManifestError(f"models[{index}].sha256 must contain 64 hexadecimal characters")
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes <= 0:
            raise ManifestError(f"models[{index}].size_bytes must be a positive integer")
        if download_url is not None and (
            not isinstance(download_url, str) or not download_url.strip()
        ):
            raise ManifestError(f"models[{index}].download_url must be null or a non-empty string")
        if source is not None and (not isinstance(source, str) or not source.strip()):
            raise ManifestError(f"models[{index}].source must be null or a non-empty string")
        if license_status not in {"unverified", "approved"}:
            raise ManifestError(f"models[{index}].license_status must be unverified or approved")
        if not isinstance(production_approved, bool):
            raise ManifestError(f"models[{index}].production_approved must be a boolean")
        if role in roles:
            raise ManifestError(f"duplicate model role: {role}")
        if filename.casefold() in filenames:
            raise ManifestError(f"duplicate model filename: {filename}")

        roles.add(role)
        filenames.add(filename.casefold())
        models.append(
            ModelSpec(
                name=name,
                role=role,
                filename=filename,
                sha256=digest.lower(),
                size_bytes=size_bytes,
                task=task,
                license_text=license_text,
                provenance_status=provenance_status,
                license_status=license_status,
                production_approved=production_approved,
                expected_classes=expected_classes,
                output_map=output_map,
                source=source.strip() if isinstance(source, str) else None,
                download_url=(download_url.strip() if isinstance(download_url, str) else None),
            )
        )

    return models


def model_path(model_dir: Path, model: ModelSpec) -> Path:
    """Resolve an artifact path while preventing manifest path traversal."""
    resolved_dir = model_dir.resolve()
    resolved_path = (resolved_dir / model.filename).resolve()
    try:
        resolved_path.relative_to(resolved_dir)
    except ValueError as exc:
        raise ManifestError(f"model path escapes the model directory: {model.filename}") from exc
    return resolved_path


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model_file(path: Path, model: ModelSpec) -> str | None:
    """Return an error message when an artifact differs from its manifest entry."""
    if not path.is_file():
        return f"missing artifact: {path}"
    try:
        actual_size = path.stat().st_size
    except OSError as exc:
        return f"cannot inspect {path}: {exc}"
    if actual_size != model.size_bytes:
        return f"size mismatch for {path.name}: expected {model.size_bytes}, found {actual_size}"
    try:
        actual_hash = sha256_file(path)
    except OSError as exc:
        return f"cannot hash {path}: {exc}"
    if actual_hash != model.sha256:
        return f"SHA-256 mismatch for {path.name}"
    return None


def production_model_checks(models: list[ModelSpec]) -> list[CheckResult]:
    """Enforce release-blocking provenance and licensing metadata."""
    by_role = {model.role: model for model in models}
    results: list[CheckResult] = []
    for role in ("vehicle", "plate", "character"):
        model = by_role.get(role)
        if model is None:
            results.append(
                CheckResult(f"production-model:{role}", "FAIL", "required role is missing")
            )
            continue
        errors: list[str] = []
        if model.provenance_status.casefold() != "verified":
            errors.append("provenance_status must be verified")
        if not model.source or not model.source.strip():
            errors.append("source is missing")
        if model.license_status != "approved":
            errors.append("license_status must be approved")
        if not model.production_approved:
            errors.append("production_approved must be true")
        results.append(
            CheckResult(
                f"production-model:{role}",
                "FAIL" if errors else "PASS",
                "; ".join(errors) if errors else "provenance and license metadata are verified",
            )
        )
    return results


def _check_environment_values() -> list[str]:
    errors: list[str] = []
    device = os.getenv("NPR_DEVICE", "cpu").strip().lower()
    if not DEVICE_PATTERN.fullmatch(device):
        errors.append("NPR_DEVICE must be cpu, auto, mps, cuda, or cuda:<index>")

    confidence_defaults = {
        "NPR_VEHICLE_CONFIDENCE": "0.40",
        "NPR_PLATE_CONFIDENCE": "0.35",
        "NPR_CHARACTER_CONFIDENCE": "0.35",
        "NPR_CHARACTER_OVERLAP_IOU": "0.50",
        "NPR_PLATE_DEDUP_IOU": "0.50",
    }
    for name, default in confidence_defaults.items():
        try:
            value = float(os.getenv(name, default))
        except ValueError:
            errors.append(f"{name} must be a number")
            continue
        if not 0.0 <= value <= 1.0:
            errors.append(f"{name} must be between 0.0 and 1.0")

    integer_defaults = {
        "NPR_INFERENCE_MAX_DIMENSION": "1024",
        "NPR_MAX_UPLOAD_BYTES": "10485760",
        "NPR_MAX_IMAGE_PIXELS": "25000000",
        "NPR_MAX_FILES": "10",
        "NPR_MAX_VEHICLES": "20",
        "NPR_MAX_PLATES_PER_VEHICLE": "2",
        "NPR_MAX_CHARACTERS_PER_PLATE": "12",
        "STREAMLIT_SERVER_MAX_UPLOAD_SIZE": "10",
    }
    for name, default in integer_defaults.items():
        try:
            value = int(os.getenv(name, default))
        except ValueError:
            errors.append(f"{name} must be an integer")
            continue
        if value <= 0:
            errors.append(f"{name} must be greater than zero")

    raw_classes = os.getenv("NPR_VEHICLE_CLASSES", "2,3,5,7")
    try:
        vehicle_classes = [int(value.strip()) for value in raw_classes.split(",") if value.strip()]
    except ValueError:
        errors.append("NPR_VEHICLE_CLASSES must be a comma-separated list of integers")
    else:
        if not vehicle_classes or any(value < 0 for value in vehicle_classes):
            errors.append("NPR_VEHICLE_CLASSES must contain non-negative class IDs")

    log_level = os.getenv("NPR_LOG_LEVEL", "INFO").strip().upper()
    if log_level not in VALID_LOG_LEVELS:
        errors.append(f"NPR_LOG_LEVEL must be one of {', '.join(sorted(VALID_LOG_LEVELS))}")

    verify_models = os.getenv("NPR_VERIFY_MODELS", "true").strip().lower()
    if verify_models not in {"1", "0", "true", "false", "yes", "no", "on", "off"}:
        errors.append("NPR_VERIFY_MODELS must be a boolean value")
    environment = os.getenv("NPR_ENVIRONMENT", "development").strip().lower()
    if environment not in {"development", "test", "production"}:
        errors.append("NPR_ENVIRONMENT must be development, test, or production")
    if environment == "production" and verify_models in {"0", "false", "no", "off"}:
        errors.append("NPR_VERIFY_MODELS cannot be false in production")
    try:
        re.compile(
            os.getenv(
                "NPR_PLATE_PATTERN",
                r"^[0-9]{1,5}[ABEDH][0-9]{1,2}$",
            )
        )
    except re.error:
        errors.append("NPR_PLATE_PATTERN must be a valid regular expression")

    try:
        max_upload_bytes = int(os.getenv("NPR_MAX_UPLOAD_BYTES", "10485760"))
        streamlit_upload_mb = int(os.getenv("STREAMLIT_SERVER_MAX_UPLOAD_SIZE", "10"))
    except ValueError:
        pass  # The integer checks above already report the invalid value.
    else:
        if max_upload_bytes > streamlit_upload_mb * 1024 * 1024:
            errors.append("NPR_MAX_UPLOAD_BYTES exceeds STREAMLIT_SERVER_MAX_UPLOAD_SIZE")
    return errors


def _dependency_checks(skip_imports: bool) -> list[CheckResult]:
    results: list[CheckResult] = []
    distributions = {
        "numpy": "numpy",
        "opencv-python-headless": "OpenCV headless",
        "pillow": "Pillow",
        "streamlit": "Streamlit",
        "torch": "PyTorch",
        "torchvision": "torchvision",
        "ultralytics-opencv-headless": "Ultralytics headless",
    }
    for distribution_name, label in distributions.items():
        try:
            version = importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            results.append(CheckResult(f"dependency:{label}", "FAIL", "not installed"))
        else:
            results.append(CheckResult(f"dependency:{label}", "PASS", version))

    conflicting_distributions = ("opencv-python", "opencv-contrib-python", "ultralytics")
    installed_conflicts: list[str] = []
    for distribution_name in conflicting_distributions:
        try:
            version = importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            continue
        installed_conflicts.append(f"{distribution_name}=={version}")
    if installed_conflicts:
        results.append(
            CheckResult(
                "dependency-conflicts",
                "FAIL",
                "remove GUI/duplicate distributions: " + ", ".join(installed_conflicts),
            )
        )
    else:
        results.append(CheckResult("dependency-conflicts", "PASS", "none detected"))

    if skip_imports:
        return results

    for module_name in ("cv2", "numpy", "PIL", "streamlit", "torch", "ultralytics"):
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # Import failures can originate in native dependencies.
            results.append(CheckResult(f"import:{module_name}", "FAIL", str(exc)))
        else:
            results.append(CheckResult(f"import:{module_name}", "PASS", "imported"))
    return results


def _device_check() -> CheckResult:
    requested = os.getenv("NPR_DEVICE", "cpu").strip().lower()
    if not requested.startswith("cuda"):
        return CheckResult("inference-device", "PASS", f"configured as {requested}")
    try:
        torch = importlib.import_module("torch")
        available = bool(torch.cuda.is_available())
    except Exception as exc:
        return CheckResult("inference-device", "FAIL", f"cannot inspect CUDA: {exc}")
    if not available:
        return CheckResult("inference-device", "FAIL", "CUDA requested but unavailable")
    return CheckResult("inference-device", "PASS", str(torch.cuda.get_device_name(0)))


def run_checks(
    manifest_path: Path,
    model_dir: Path,
    app_root: Path,
    image_dir: Path,
    *,
    skip_imports: bool,
    skip_model_files: bool,
    production: bool,
) -> list[CheckResult]:
    """Run the complete environment diagnostic suite."""
    results: list[CheckResult] = []
    python_ok = sys.version_info[:2] == (3, 12)
    results.append(
        CheckResult(
            "python",
            "PASS" if python_ok else "FAIL",
            f"{sys.version.split()[0]} (requires 3.12.x)",
        )
    )
    results.append(
        CheckResult(
            "virtual-environment",
            "PASS" if sys.prefix != sys.base_prefix else "WARN",
            sys.prefix if sys.prefix != sys.base_prefix else "not active; use `uv run`",
        )
    )

    expected_paths = (
        REPOSITORY_ROOT / "pyproject.toml",
        REPOSITORY_ROOT / "src" / "number_plate_recognition",
        REPOSITORY_ROOT / "app" / "streamlit_app.py",
    )
    missing_paths = [
        str(path.relative_to(REPOSITORY_ROOT)) for path in expected_paths if not path.exists()
    ]
    results.append(
        CheckResult(
            "repository-layout",
            "FAIL" if missing_paths else "PASS",
            "missing: " + ", ".join(missing_paths) if missing_paths else "expected paths exist",
        )
    )

    configured_paths = {
        "app_root": app_root,
        "model_dir": model_dir,
        "image_dir": image_dir,
        "manifest": manifest_path,
    }
    missing_configured_paths = [
        name for name, path in configured_paths.items() if not path.exists()
    ]
    results.append(
        CheckResult(
            "application-paths",
            "FAIL" if missing_configured_paths else "PASS",
            "; ".join(f"{name}={path}" for name, path in configured_paths.items())
            + (
                "; missing=" + ",".join(missing_configured_paths)
                if missing_configured_paths
                else ""
            ),
        )
    )

    environment_errors = _check_environment_values()
    results.append(
        CheckResult(
            "configuration",
            "FAIL" if environment_errors else "PASS",
            "; ".join(environment_errors) if environment_errors else "values are valid",
        )
    )
    results.extend(_dependency_checks(skip_imports))
    results.append(_device_check())

    try:
        models = load_manifest(manifest_path)
    except ManifestError as exc:
        results.append(CheckResult("model-manifest", "FAIL", str(exc)))
        return results
    results.append(CheckResult("model-manifest", "PASS", f"{len(models)} entries"))
    if production:
        results.extend(production_model_checks(models))

    if skip_model_files:
        return results
    for model in models:
        try:
            path = model_path(model_dir, model)
        except ManifestError as exc:
            results.append(CheckResult(f"model:{model.role}", "FAIL", str(exc)))
            continue
        error = verify_model_file(path, model)
        results.append(
            CheckResult(
                f"model:{model.role}",
                "FAIL" if error else "PASS",
                error or f"verified {path.name}",
            )
        )
    return results


def _print_results(results: list[CheckResult], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps([asdict(result) for result in results], indent=2))
        return
    for result in results:
        print(f"[{result.status}] {result.name}: {result.message}")
    passed = sum(result.status == "PASS" for result in results)
    warnings = sum(result.status == "WARN" for result in results)
    failed = sum(result.status == "FAIL" for result in results)
    print(f"Summary: {passed} passed, {warnings} warnings, {failed} failed")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="manifest path (default: NPR_MODEL_MANIFEST or models/manifest.json)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="artifact directory (default: NPR_MODEL_DIR or models)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--manifest-only",
        action="store_true",
        help="validate only the manifest schema; do not inspect the environment or artifacts",
    )
    mode.add_argument(
        "--models-only",
        action="store_true",
        help="verify only manifest schema and local model size/checksums (stdlib-only)",
    )
    parser.add_argument("--skip-imports", action="store_true", help="skip runtime import checks")
    parser.add_argument(
        "--skip-model-files",
        action="store_true",
        help="validate the registry but do not hash local model artifacts",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--production",
        action="store_true",
        help="also require verified provenance, source, and license metadata for production roles",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run diagnostics and return a process exit code."""
    arguments = build_parser().parse_args(argv)
    configured_app_root = repository_path(os.getenv("NPR_APP_ROOT", str(REPOSITORY_ROOT)))
    configured_model_dir = arguments.model_dir or Path(os.getenv("NPR_MODEL_DIR", "models"))
    model_dir = app_path(configured_model_dir, configured_app_root)
    configured_manifest = arguments.manifest or Path(
        os.getenv("NPR_MODEL_MANIFEST", str(model_dir / "manifest.json"))
    )
    manifest_path = app_path(configured_manifest, configured_app_root)
    image_dir = app_path(os.getenv("NPR_IMAGE_DIR", "images"), configured_app_root)
    production = arguments.production or (
        os.getenv("NPR_ENVIRONMENT", "development").strip().lower() == "production"
    )

    if arguments.manifest_only:
        try:
            models = load_manifest(manifest_path)
        except ManifestError as exc:
            results = [CheckResult("model-manifest", "FAIL", str(exc))]
        else:
            results = [CheckResult("model-manifest", "PASS", f"{len(models)} entries")]
            if production:
                results.extend(production_model_checks(models))
    elif arguments.models_only:
        try:
            models = load_manifest(manifest_path)
        except ManifestError as exc:
            results = [CheckResult("model-manifest", "FAIL", str(exc))]
        else:
            results = [CheckResult("model-manifest", "PASS", f"{len(models)} entries")]
            if production:
                results.extend(production_model_checks(models))
            for model in models:
                try:
                    path = model_path(model_dir, model)
                except ManifestError as exc:
                    results.append(CheckResult(f"model:{model.role}", "FAIL", str(exc)))
                    continue
                error = verify_model_file(path, model)
                results.append(
                    CheckResult(
                        f"model:{model.role}",
                        "FAIL" if error else "PASS",
                        error or f"verified {path.name}",
                    )
                )
    else:
        results = run_checks(
            manifest_path,
            model_dir,
            configured_app_root,
            image_dir,
            skip_imports=arguments.skip_imports,
            skip_model_files=arguments.skip_model_files,
            production=production,
        )

    _print_results(results, as_json=arguments.json)
    return 1 if any(result.status == "FAIL" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
