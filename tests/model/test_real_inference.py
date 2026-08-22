from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from number_plate_recognition.adapters.ultralytics import load_model_bundle
from number_plate_recognition.config import AppConfig
from number_plate_recognition.imaging import decode_image
from number_plate_recognition.pipeline import RecognitionPipeline


@pytest.mark.model
def test_verified_models_run_the_complete_demo_pipeline(
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
    models = load_model_bundle(config)
    pipeline = RecognitionPipeline(models, config)
    blank_image = np.zeros((320, 320, 3), dtype=np.uint8)
    assert (
        models.vehicle.predict(
            blank_image,
            confidence=config.vehicle_confidence,
            max_detections=config.max_vehicles,
        )
        == ()
    )
    models.plate.predict(
        blank_image,
        confidence=config.plate_confidence,
        max_detections=config.max_plates_per_vehicle,
    )
    models.character.predict(
        blank_image,
        confidence=config.character_confidence,
        max_detections=config.max_characters_per_plate,
        agnostic_nms=True,
    )

    blank_result = pipeline.process(blank_image)

    assert blank_result.vehicle_count == 0
    assert blank_result.plates == ()

    expected_plates = {
        "Car1.jpg": "90120A72",
        "Car2.jpg": "1678E1",
        "Car3.jpg": "45296B6",
    }
    for filename, expected_plate in expected_plates.items():
        payload = (project_root / "images" / filename).read_bytes()
        image = decode_image(
            payload,
            max_bytes=config.max_upload_bytes,
            max_pixels=config.max_image_pixels,
        )

        result = pipeline.process(image)

        assert result.annotated_image_rgb.ndim == 3
        assert result.annotated_image_rgb.shape[2] == 3
        assert result.vehicle_count >= 1
        assert [plate.text for plate in result.plates] == [expected_plate]
        assert result.plates[0].format_valid is True
        assert result.timings_ms["total"] > 0
        assert set(result.model_versions) == {"vehicle", "plate", "character"}
