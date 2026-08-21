from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from number_plate_recognition.adapters.ultralytics import load_model_bundle
from number_plate_recognition.config import AppConfig
from number_plate_recognition.pipeline import RecognitionPipeline


@pytest.mark.model
def test_verified_models_load_and_run_on_synthetic_frame(
    project_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("YOLO_CONFIG_DIR", str(tmp_path / "ultralytics"))
    monkeypatch.setenv("YOLO_AUTOINSTALL", "false")
    monkeypatch.setenv("YOLO_OFFLINE", "true")
    config = AppConfig(
        model_dir=project_root / "models",
        image_dir=project_root / "images",
        manifest_path=project_root / "models" / "manifest.json",
        device="cpu",
        inference_max_dimension=640,
    )
    image = np.zeros((320, 320, 3), dtype=np.uint8)

    models = load_model_bundle(config)
    assert (
        models.vehicle.predict(
            image,
            confidence=config.vehicle_confidence,
            max_detections=config.max_vehicles,
        )
        == ()
    )
    models.plate.predict(
        image,
        confidence=config.plate_confidence,
        max_detections=config.max_plates_per_vehicle,
    )
    models.character.predict(
        image,
        confidence=config.character_confidence,
        max_detections=config.max_characters_per_plate,
        agnostic_nms=True,
    )

    result = RecognitionPipeline(models, config).process(image)

    assert result.annotated_image_rgb.ndim == 3
    assert result.annotated_image_rgb.shape[2] == 3
    assert result.vehicle_count == 0
    assert result.plates == ()
    assert result.timings_ms["total"] > 0
    assert set(result.model_versions) == {"vehicle", "plate", "character"}
