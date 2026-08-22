from __future__ import annotations

from pathlib import Path

import pytest

from number_plate_recognition.config import PROJECT_ROOT, AppConfig
from number_plate_recognition.errors import ConfigurationError


def test_config_reads_and_validates_runtime_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("NPR_APP_ROOT", raising=False)
    monkeypatch.setenv("NPR_MODEL_DIR", str(tmp_path / "weights"))
    monkeypatch.setenv("NPR_IMAGE_DIR", str(tmp_path / "samples"))
    monkeypatch.setenv("NPR_DEVICE", "cuda:0")
    monkeypatch.setenv("NPR_INFERENCE_MAX_DIMENSION", "800")
    monkeypatch.setenv("NPR_MAX_UPLOAD_BYTES", "1024")
    monkeypatch.setenv("NPR_MAX_IMAGE_PIXELS", "2000")
    monkeypatch.setenv("NPR_MAX_FILES", "3")
    monkeypatch.setenv("NPR_MAX_VEHICLES", "4")
    monkeypatch.setenv("NPR_MAX_PLATES_PER_VEHICLE", "2")
    monkeypatch.setenv("NPR_MAX_CHARACTERS_PER_PLATE", "9")
    monkeypatch.setenv("NPR_VEHICLE_CONFIDENCE", "0.6")
    monkeypatch.setenv("NPR_PLATE_CONFIDENCE", "0.55")
    monkeypatch.setenv("NPR_CHARACTER_CONFIDENCE", "0.5")
    monkeypatch.setenv("NPR_CHARACTER_OVERLAP_IOU", "0.4")
    monkeypatch.setenv("NPR_PLATE_DEDUP_IOU", "0.6")
    monkeypatch.setenv("NPR_VEHICLE_CLASSES", "2, 7")
    monkeypatch.setenv("NPR_PLATE_PATTERN", "^[0-9]+A[0-9]+$")
    monkeypatch.setenv("NPR_LOG_LEVEL", "debug")
    monkeypatch.setenv("NPR_VERIFY_MODELS", "false")

    config = AppConfig.from_env()

    assert config.model_dir == (tmp_path / "weights").resolve()
    assert config.app_root == PROJECT_ROOT
    assert config.manifest_path == (tmp_path / "weights" / "manifest.json").resolve()
    assert config.image_dir == (tmp_path / "samples").resolve()
    assert config.device == "cuda:0"
    assert config.inference_max_dimension == 800
    assert config.max_upload_bytes == 1024
    assert config.max_image_pixels == 2000
    assert config.max_files == 3
    assert config.max_vehicles == 4
    assert config.max_plates_per_vehicle == 2
    assert config.max_characters_per_plate == 9
    assert config.vehicle_confidence == 0.6
    assert config.plate_confidence == 0.55
    assert config.character_confidence == 0.5
    assert config.character_overlap_iou == 0.4
    assert config.plate_dedup_iou == 0.6
    assert config.vehicle_classes == (2, 7)
    assert config.plate_pattern == "^[0-9]+A[0-9]+$"
    assert config.log_level == "DEBUG"
    assert config.verify_models is False


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("NPR_MAX_FILES", "zero"),
        ("NPR_MAX_FILES", "0"),
        ("NPR_VEHICLE_CONFIDENCE", "1.1"),
        ("NPR_PLATE_CONFIDENCE", "-0.1"),
        ("NPR_VEHICLE_CLASSES", "car,bus"),
        ("NPR_VERIFY_MODELS", "sometimes"),
        ("NPR_LOG_LEVEL", "LOUD"),
        ("NPR_DEVICE", "gpu"),
        ("NPR_PLATE_PATTERN", "["),
        ("NPR_CHARACTER_OVERLAP_IOU", "2"),
        ("NPR_PLATE_DEDUP_IOU", "-1"),
        ("NPR_MAX_VEHICLES", "0"),
    ],
)
def test_config_rejects_invalid_environment(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ConfigurationError):
        AppConfig.from_env()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"vehicle_confidence": float("nan")},
        {"max_vehicles": 0},
        {"plate_pattern": "["},
        {"vehicle_classes": ()},
    ],
)
def test_direct_construction_is_validated(kwargs: dict[str, object]) -> None:
    with pytest.raises(ConfigurationError):
        AppConfig(**kwargs)  # type: ignore[arg-type]


def test_upload_limit_cannot_exceed_streamlit_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NPR_MAX_UPLOAD_BYTES", str(11 * 1024 * 1024))
    monkeypatch.setenv("STREAMLIT_SERVER_MAX_UPLOAD_SIZE", "10")

    with pytest.raises(ConfigurationError, match="STREAMLIT"):
        AppConfig.from_env()


def test_relative_paths_are_resolved_from_project_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("NPR_APP_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NPR_MODEL_DIR", "custom-models")
    monkeypatch.setenv("NPR_IMAGE_DIR", "custom-images")
    monkeypatch.setenv("NPR_MODEL_MANIFEST", "config/models.json")

    config = AppConfig.from_env()

    assert config.model_dir == (PROJECT_ROOT / "custom-models").resolve()
    assert config.image_dir == (PROJECT_ROOT / "custom-images").resolve()
    assert config.manifest_path == (PROJECT_ROOT / "config/models.json").resolve()


def test_explicit_app_root_supports_non_editable_deployments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("NPR_APP_ROOT", str(tmp_path))
    monkeypatch.setenv("NPR_MODEL_DIR", "artifacts")
    monkeypatch.setenv("NPR_IMAGE_DIR", "examples")

    config = AppConfig.from_env()

    assert config.app_root == tmp_path.resolve()
    assert config.model_dir == (tmp_path / "artifacts").resolve()
    assert config.image_dir == (tmp_path / "examples").resolve()
    assert config.manifest_path == (tmp_path / "artifacts/manifest.json").resolve()


def test_direct_construction_derives_paths_from_app_root(tmp_path: Path) -> None:
    config = AppConfig(app_root=tmp_path)

    assert config.model_dir == (tmp_path / "models").resolve()
    assert config.image_dir == (tmp_path / "images").resolve()
    assert config.manifest_path == (tmp_path / "models/manifest.json").resolve()


def test_direct_construction_derives_manifest_from_custom_model_dir(
    tmp_path: Path,
) -> None:
    config = AppConfig(app_root=tmp_path, model_dir=Path("weights"))

    assert config.model_dir == (tmp_path / "weights").resolve()
    assert config.manifest_path == (tmp_path / "weights/manifest.json").resolve()
