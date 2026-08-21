"""Validated application configuration loaded from environment variables."""

from __future__ import annotations

import logging
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from number_plate_recognition.errors import ConfigurationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEVICE_PATTERN = re.compile(r"^(?:auto|cpu|mps|cuda(?::\d+)?)$")
DEFAULT_PLATE_PATTERN = r"^[0-9]{1,5}[ABEDH][0-9]{1,2}$"


def _discover_app_root() -> Path:
    configured = os.getenv("NPR_APP_ROOT")
    if configured:
        path = Path(configured).expanduser()
        return (Path.cwd() / path).resolve() if not path.is_absolute() else path.resolve()

    candidates = (Path.cwd().resolve(), PROJECT_ROOT)
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file() and (candidate / "models").is_dir():
            return candidate
    return Path.cwd().resolve()


def _resolve_app_path(raw: str | Path, app_root: Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = app_root / path
    return path.resolve()


def _read_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer") from exc
    if value < minimum:
        raise ConfigurationError(f"{name} must be at least {minimum}")
    return value


def _read_probability(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number") from exc
    if not 0.0 <= value <= 1.0:
        raise ConfigurationError(f"{name} must be between 0 and 1")
    return value


def _read_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"{name} must be true or false")


def _read_vehicle_classes(default: tuple[int, ...]) -> tuple[int, ...]:
    raw = os.getenv("NPR_VEHICLE_CLASSES")
    if raw is None:
        return default
    try:
        values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    except ValueError as exc:
        raise ConfigurationError(
            "NPR_VEHICLE_CLASSES must be a comma-separated list of integers"
        ) from exc
    if not values or any(value < 0 for value in values):
        raise ConfigurationError("NPR_VEHICLE_CLASSES must contain non-negative IDs")
    return values


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Immutable runtime configuration shared by the UI and inference pipeline."""

    app_root: Path = field(default_factory=_discover_app_root)
    model_dir: Path = Path("models")
    image_dir: Path = Path("images")
    manifest_path: Path = Path()
    device: str = "cpu"
    inference_max_dimension: int = 1024
    max_upload_bytes: int = 10 * 1024 * 1024
    max_image_pixels: int = 25_000_000
    max_files: int = 10
    max_vehicles: int = 20
    max_plates_per_vehicle: int = 2
    max_characters_per_plate: int = 12
    vehicle_confidence: float = 0.40
    plate_confidence: float = 0.35
    character_confidence: float = 0.35
    character_overlap_iou: float = 0.50
    plate_dedup_iou: float = 0.50
    vehicle_classes: tuple[int, ...] = (2, 3, 5, 7)
    plate_pattern: str = DEFAULT_PLATE_PATTERN
    environment: str = "development"
    log_level: str = "INFO"
    verify_models: bool = True

    def __post_init__(self) -> None:
        app_root = Path(self.app_root).expanduser()
        if not app_root.is_absolute():
            app_root = Path.cwd() / app_root
        app_root = app_root.resolve()
        object.__setattr__(self, "app_root", app_root)
        for field_name in ("model_dir", "image_dir"):
            path = Path(getattr(self, field_name)).expanduser()
            if not path.is_absolute():
                path = app_root / path
            object.__setattr__(self, field_name, path.resolve())
        manifest_path = self.manifest_path
        if manifest_path == Path():
            manifest_path = self.model_dir / "manifest.json"
        else:
            manifest_path = manifest_path.expanduser()
            if not manifest_path.is_absolute():
                manifest_path = app_root / manifest_path
        object.__setattr__(self, "manifest_path", manifest_path.resolve())

        device = self.device.lower().strip()
        if not _DEVICE_PATTERN.fullmatch(device):
            raise ConfigurationError("device must be auto, cpu, mps, cuda, or cuda:<index>")
        object.__setattr__(self, "device", device)

        environment = self.environment.lower().strip()
        if environment not in {"development", "test", "production"}:
            raise ConfigurationError("environment must be development, test, or production")
        object.__setattr__(self, "environment", environment)

        log_level = self.log_level.upper().strip()
        if log_level not in logging.getLevelNamesMapping():
            raise ConfigurationError(f"Unsupported log level: {log_level}")
        object.__setattr__(self, "log_level", log_level)

        integer_fields = (
            "inference_max_dimension",
            "max_upload_bytes",
            "max_image_pixels",
            "max_files",
            "max_vehicles",
            "max_plates_per_vehicle",
            "max_characters_per_plate",
        )
        for field_name in integer_fields:
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ConfigurationError(f"{field_name} must be a positive integer")

        probability_fields = (
            "vehicle_confidence",
            "plate_confidence",
            "character_confidence",
            "character_overlap_iou",
            "plate_dedup_iou",
        )
        for field_name in probability_fields:
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ConfigurationError(f"{field_name} must be between 0 and 1")

        vehicle_classes = tuple(self.vehicle_classes)
        if not vehicle_classes or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in vehicle_classes
        ):
            raise ConfigurationError("vehicle_classes must contain non-negative integers")
        object.__setattr__(self, "vehicle_classes", vehicle_classes)
        if not isinstance(self.plate_pattern, str) or not self.plate_pattern:
            raise ConfigurationError("plate_pattern must be a non-empty regex")
        try:
            re.compile(self.plate_pattern)
        except re.error as exc:
            raise ConfigurationError("plate_pattern must be a valid regex") from exc
        if not isinstance(self.verify_models, bool):
            raise ConfigurationError("verify_models must be a boolean")
        if environment == "production" and not self.verify_models:
            raise ConfigurationError(
                "verify_models cannot be disabled in the production environment"
            )

    @classmethod
    def from_env(cls) -> AppConfig:
        """Build configuration from NPR_* environment variables."""

        app_root = _discover_app_root()
        model_dir = _resolve_app_path(os.getenv("NPR_MODEL_DIR", "models"), app_root)
        image_dir = _resolve_app_path(os.getenv("NPR_IMAGE_DIR", "images"), app_root)
        manifest_path = _resolve_app_path(
            os.getenv("NPR_MODEL_MANIFEST", str(model_dir / "manifest.json")), app_root
        )
        device = os.getenv("NPR_DEVICE", "cpu").strip()
        if not _DEVICE_PATTERN.fullmatch(device.lower()):
            raise ConfigurationError("NPR_DEVICE must be auto, cpu, mps, cuda, or cuda:<index>")

        log_level = os.getenv("NPR_LOG_LEVEL", "INFO").upper().strip()
        if log_level not in logging.getLevelNamesMapping():
            raise ConfigurationError(f"Unsupported NPR_LOG_LEVEL: {log_level}")
        environment = os.getenv("NPR_ENVIRONMENT", "development").lower().strip()
        if environment not in {"development", "test", "production"}:
            raise ConfigurationError("NPR_ENVIRONMENT must be development, test, or production")
        verify_models = _read_bool("NPR_VERIFY_MODELS", True)
        if environment == "production" and not verify_models:
            raise ConfigurationError(
                "NPR_VERIFY_MODELS cannot be disabled in the production environment"
            )
        plate_pattern = os.getenv("NPR_PLATE_PATTERN", DEFAULT_PLATE_PATTERN)
        try:
            re.compile(plate_pattern)
        except re.error as exc:
            raise ConfigurationError("NPR_PLATE_PATTERN must be a valid regex") from exc
        max_upload_bytes = _read_int("NPR_MAX_UPLOAD_BYTES", 10 * 1024 * 1024)
        streamlit_upload_mb = _read_int("STREAMLIT_SERVER_MAX_UPLOAD_SIZE", 10)
        if max_upload_bytes > streamlit_upload_mb * 1024 * 1024:
            raise ConfigurationError(
                "NPR_MAX_UPLOAD_BYTES exceeds STREAMLIT_SERVER_MAX_UPLOAD_SIZE"
            )

        return cls(
            app_root=app_root,
            model_dir=model_dir,
            image_dir=image_dir,
            manifest_path=manifest_path,
            device=device.lower(),
            inference_max_dimension=_read_int("NPR_INFERENCE_MAX_DIMENSION", 1024),
            max_upload_bytes=max_upload_bytes,
            max_image_pixels=_read_int("NPR_MAX_IMAGE_PIXELS", 25_000_000),
            max_files=_read_int("NPR_MAX_FILES", 10),
            max_vehicles=_read_int("NPR_MAX_VEHICLES", 20),
            max_plates_per_vehicle=_read_int("NPR_MAX_PLATES_PER_VEHICLE", 2),
            max_characters_per_plate=_read_int("NPR_MAX_CHARACTERS_PER_PLATE", 12),
            vehicle_confidence=_read_probability("NPR_VEHICLE_CONFIDENCE", 0.40),
            plate_confidence=_read_probability("NPR_PLATE_CONFIDENCE", 0.35),
            character_confidence=_read_probability("NPR_CHARACTER_CONFIDENCE", 0.35),
            character_overlap_iou=_read_probability("NPR_CHARACTER_OVERLAP_IOU", 0.50),
            plate_dedup_iou=_read_probability("NPR_PLATE_DEDUP_IOU", 0.50),
            vehicle_classes=_read_vehicle_classes((2, 3, 5, 7)),
            plate_pattern=plate_pattern,
            environment=environment,
            log_level=log_level,
            verify_models=verify_models,
        )
