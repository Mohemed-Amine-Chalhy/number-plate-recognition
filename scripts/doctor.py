#!/usr/bin/env python3
"""Validate the local runtime, configuration, and bundled model artifacts."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from number_plate_recognition.config import AppConfig
from number_plate_recognition.errors import ConfigurationError, ModelIntegrityError
from number_plate_recognition.model_registry import ModelManifest, load_manifest, verify_artifact

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

# Dependency checks stay offline and do not write Ultralytics settings to a
# user profile or the repository root.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("YOLO_OFFLINE", "true")
os.environ.setdefault(
    "YOLO_CONFIG_DIR",
    str(REPOSITORY_ROOT / ".runtime" / "ultralytics"),
)


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One diagnostic result."""

    name: str
    status: str
    message: str


def _app_path(value: str | Path, app_root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (app_root / path).resolve()


def _configured_paths(arguments: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    app_root = Path(os.getenv("NPR_APP_ROOT", str(REPOSITORY_ROOT))).expanduser().resolve()
    model_dir = _app_path(
        arguments.model_dir or os.getenv("NPR_MODEL_DIR", "models"),
        app_root,
    )
    manifest_path = _app_path(
        arguments.manifest or os.getenv("NPR_MODEL_MANIFEST", str(model_dir / "manifest.json")),
        app_root,
    )
    image_dir = _app_path(os.getenv("NPR_IMAGE_DIR", "images"), app_root)
    return app_root, model_dir, manifest_path, image_dir


def _configuration_check() -> CheckResult:
    try:
        AppConfig.from_env()
    except ConfigurationError as exc:
        return CheckResult("configuration", "FAIL", str(exc))
    return CheckResult("configuration", "PASS", "values are valid")


def _dependency_checks(*, skip_imports: bool) -> list[CheckResult]:
    results: list[CheckResult] = []
    distributions = {
        "numpy": "NumPy",
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
    if requested != "mps" and not requested.startswith("cuda"):
        return CheckResult("inference-device", "PASS", f"configured as {requested}")
    try:
        torch = importlib.import_module("torch")
        if requested == "mps":
            if not bool(torch.backends.mps.is_available()):
                return CheckResult("inference-device", "FAIL", "MPS requested but unavailable")
            return CheckResult("inference-device", "PASS", "MPS available")

        device_index = int(requested.partition(":")[2] or "0")
        if not bool(torch.cuda.is_available()):
            return CheckResult("inference-device", "FAIL", "CUDA requested but unavailable")
        device_count = int(torch.cuda.device_count())
        if device_index >= device_count:
            return CheckResult(
                "inference-device",
                "FAIL",
                f"CUDA device {device_index} requested but only {device_count} available",
            )
        device_name = str(torch.cuda.get_device_name(device_index))
    except Exception as exc:
        backend = "MPS" if requested == "mps" else "CUDA"
        return CheckResult("inference-device", "FAIL", f"cannot inspect {backend}: {exc}")
    return CheckResult(
        "inference-device",
        "PASS",
        f"cuda:{device_index} ({device_name})",
    )


def _load_manifest(path: Path) -> tuple[ModelManifest | None, CheckResult]:
    try:
        manifest = load_manifest(path)
    except ModelIntegrityError as exc:
        return None, CheckResult("model-manifest", "FAIL", str(exc))
    return manifest, CheckResult(
        "model-manifest",
        "PASS",
        f"schema {manifest.schema_version}, {len(manifest.artifacts)} entries",
    )


def _artifact_checks(manifest: ModelManifest, model_dir: Path) -> list[CheckResult]:
    results: list[CheckResult] = []
    for artifact in manifest.artifacts:
        try:
            path = verify_artifact(artifact, model_dir)
        except ModelIntegrityError as exc:
            results.append(CheckResult(f"model:{artifact.role}", "FAIL", str(exc)))
        else:
            results.append(
                CheckResult(
                    f"model:{artifact.role}",
                    "PASS",
                    f"verified {path.name}",
                )
            )
    return results


def _runtime_checks(
    *,
    app_root: Path,
    model_dir: Path,
    manifest_path: Path,
    image_dir: Path,
    skip_imports: bool,
    skip_model_files: bool,
) -> list[CheckResult]:
    python_ok = sys.version_info[:2] == (3, 12)
    results = [
        CheckResult(
            "python",
            "PASS" if python_ok else "FAIL",
            f"{sys.version.split()[0]} (requires 3.12.x)",
        ),
        CheckResult(
            "virtual-environment",
            "PASS" if sys.prefix != sys.base_prefix else "WARN",
            sys.prefix if sys.prefix != sys.base_prefix else "not active; use `uv run`",
        ),
    ]

    expected_paths = (
        REPOSITORY_ROOT / "pyproject.toml",
        REPOSITORY_ROOT / "src" / "number_plate_recognition",
        REPOSITORY_ROOT / "app" / "streamlit_app.py",
    )
    missing_layout = [
        str(path.relative_to(REPOSITORY_ROOT)) for path in expected_paths if not path.exists()
    ]
    results.append(
        CheckResult(
            "repository-layout",
            "FAIL" if missing_layout else "PASS",
            "missing: " + ", ".join(missing_layout) if missing_layout else "expected paths exist",
        )
    )

    configured_paths = {
        "app_root": app_root,
        "model_dir": model_dir,
        "image_dir": image_dir,
        "manifest": manifest_path,
    }
    missing_configured = [name for name, path in configured_paths.items() if not path.exists()]
    results.append(
        CheckResult(
            "application-paths",
            "FAIL" if missing_configured else "PASS",
            "; ".join(f"{name}={path}" for name, path in configured_paths.items())
            + ("; missing=" + ",".join(missing_configured) if missing_configured else ""),
        )
    )

    results.append(_configuration_check())
    results.extend(_dependency_checks(skip_imports=skip_imports))
    results.append(_device_check())
    manifest, manifest_result = _load_manifest(manifest_path)
    results.append(manifest_result)
    if manifest is not None and not skip_model_files:
        results.extend(_artifact_checks(manifest, model_dir))
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
        help="validate only the model manifest",
    )
    mode.add_argument(
        "--models-only",
        action="store_true",
        help="validate the manifest and verify bundled model files",
    )
    parser.add_argument("--skip-imports", action="store_true", help="skip runtime import checks")
    parser.add_argument(
        "--skip-model-files",
        action="store_true",
        help="validate the manifest without hashing model artifacts",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    app_root, model_dir, manifest_path, image_dir = _configured_paths(arguments)

    if arguments.manifest_only:
        _, manifest_result = _load_manifest(manifest_path)
        results = [manifest_result]
    elif arguments.models_only:
        manifest, manifest_result = _load_manifest(manifest_path)
        results = [manifest_result]
        if manifest is not None:
            results.extend(_artifact_checks(manifest, model_dir))
    else:
        results = _runtime_checks(
            app_root=app_root,
            model_dir=model_dir,
            manifest_path=manifest_path,
            image_dir=image_dir,
            skip_imports=arguments.skip_imports,
            skip_model_files=arguments.skip_model_files,
        )

    _print_results(results, as_json=arguments.json)
    return 1 if any(result.status == "FAIL" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
