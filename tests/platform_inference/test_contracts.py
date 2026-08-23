from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from number_plate_recognition.domain import BoundingBox, InferenceResult, PlateResult
from services.inference_worker.contracts import RecognitionObservation


def _result() -> InferenceResult:
    plate = PlateResult(
        text="90120A72",
        detection_confidence=0.92345678,
        recognition_confidence=0.88765432,
        box=BoundingBox(10.12345, 20.5, 110.5, 60.0),
        vehicle_box=BoundingBox(0, 0, 200, 100),
        characters=(),
        format_valid=True,
    )
    return InferenceResult(
        annotated_image_rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        plates=(plate,),
        vehicle_count=1,
        timings_ms={"total": 12.34567},
        model_versions={"vehicle": "vehicle@sha256:abc"},
    )


def test_observation_is_json_safe_and_versioned() -> None:
    timestamp = datetime(2026, 8, 23, 12, 0, tzinfo=UTC)

    observation = RecognitionObservation.from_result(
        job_id="job-1",
        capture_id="capture-1",
        captured_at=timestamp,
        completed_at=timestamp,
        result=_result(),
    )

    payload = observation.to_dict()
    assert payload["schema_version"] == 1
    assert payload["captured_at"] == "2026-08-23T12:00:00Z"
    assert payload["timings_ms"] == {"total": 12.346}
    assert payload["plates"][0]["text"] == "90120A72"  # type: ignore[index]


def test_observation_rejects_naive_timestamps() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        RecognitionObservation.from_result(
            job_id="job-1",
            capture_id="capture-1",
            captured_at=datetime(2026, 8, 23),
            completed_at=datetime.now(UTC),
            result=_result(),
        )
