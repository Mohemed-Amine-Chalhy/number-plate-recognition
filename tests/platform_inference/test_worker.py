from __future__ import annotations

from datetime import UTC, datetime

import cv2 as cv
import numpy as np
import pytest

from number_plate_recognition.config import AppConfig
from number_plate_recognition.domain import ImageArray, InferenceResult
from number_plate_recognition.errors import ImageValidationError
from services.inference_worker.worker import RecognitionWorker


class FakePipeline:
    def __init__(self) -> None:
        self.calls = 0

    def process(self, image_bgr: ImageArray) -> InferenceResult:
        self.calls += 1
        return InferenceResult(
            annotated_image_rgb=image_bgr,
            plates=(),
            vehicle_count=0,
            timings_ms={"total": 1.0},
            model_versions={"vehicle": "fake"},
        )


def _png_bytes() -> bytes:
    encoded, payload = cv.imencode(".png", np.zeros((8, 8, 3), dtype=np.uint8))
    assert encoded
    return payload.tobytes()


def test_worker_lazily_builds_and_reuses_pipeline() -> None:
    pipeline = FakePipeline()
    factory_calls = 0

    def factory() -> FakePipeline:
        nonlocal factory_calls
        factory_calls += 1
        return pipeline

    worker = RecognitionWorker(
        AppConfig(max_image_pixels=1_000, max_upload_bytes=10_000),
        pipeline_factory=factory,
    )
    ready_before_inference = worker.ready

    first = worker.recognize_bytes(
        _png_bytes(),
        capture_id="capture-1",
        job_id="job-1",
        captured_at=datetime(2026, 8, 23, tzinfo=UTC),
    )
    second = worker.recognize_bytes(_png_bytes(), capture_id="capture-2", job_id="job-2")

    ready_after_inference = worker.ready
    assert (ready_before_inference, ready_after_inference) == (False, True)
    assert factory_calls == 1
    assert pipeline.calls == 2
    assert first.capture_id == "capture-1"
    assert second.job_id == "job-2"


@pytest.mark.parametrize("capture_id", ["", "   "])
def test_worker_rejects_empty_capture_id(capture_id: str) -> None:
    worker = RecognitionWorker(
        AppConfig(max_image_pixels=1_000, max_upload_bytes=10_000),
        pipeline_factory=FakePipeline,
    )

    with pytest.raises(ValueError, match="capture_id"):
        worker.recognize_bytes(_png_bytes(), capture_id=capture_id)


def test_worker_validates_capture_before_loading_models() -> None:
    factory_called = False

    def factory() -> FakePipeline:
        nonlocal factory_called
        factory_called = True
        return FakePipeline()

    worker = RecognitionWorker(AppConfig(), pipeline_factory=factory)

    with pytest.raises(ImageValidationError):
        worker.recognize_bytes(b"not-an-image", capture_id="capture-1")
    assert not factory_called
