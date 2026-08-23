"""Lazy, process-local inference worker used by queue and CLI adapters."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from threading import Lock
from typing import Protocol
from uuid import uuid4

from number_plate_recognition.adapters.ultralytics import load_model_bundle
from number_plate_recognition.config import AppConfig
from number_plate_recognition.domain import ImageArray, InferenceResult
from number_plate_recognition.imaging import decode_image
from number_plate_recognition.pipeline import RecognitionPipeline
from services.inference_worker.contracts import RecognitionObservation


class Pipeline(Protocol):
    """Small boundary implemented by the production recognition pipeline."""

    def process(self, image_bgr: ImageArray) -> InferenceResult:
        """Recognize an already validated image."""


type PipelineFactory = Callable[[], Pipeline]


class RecognitionWorker:
    """Load one model bundle per process and emit transport-safe observations."""

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        pipeline_factory: PipelineFactory | None = None,
    ) -> None:
        self._config = config or AppConfig.from_env()
        self._pipeline_factory = pipeline_factory or self._build_pipeline
        self._pipeline: Pipeline | None = None
        self._startup_lock = Lock()

    @property
    def ready(self) -> bool:
        """Return whether the model bundle has been warmed in this process."""

        return self._pipeline is not None

    def warm(self) -> None:
        """Load and validate the configured model bundle exactly once."""

        self._get_pipeline()

    def recognize_bytes(
        self,
        payload: bytes,
        *,
        capture_id: str,
        job_id: str | None = None,
        captured_at: datetime | None = None,
    ) -> RecognitionObservation:
        """Validate one capture, run inference, and return a v1 observation."""

        normalized_capture_id = capture_id.strip()
        if not normalized_capture_id:
            raise ValueError("capture_id must not be empty")
        normalized_job_id = (job_id or str(uuid4())).strip()
        if not normalized_job_id:
            raise ValueError("job_id must not be empty")
        captured = captured_at or datetime.now(UTC)
        image = decode_image(
            payload,
            max_bytes=self._config.max_upload_bytes,
            max_pixels=self._config.max_image_pixels,
        )
        result = self._get_pipeline().process(image)
        return RecognitionObservation.from_result(
            job_id=normalized_job_id,
            capture_id=normalized_capture_id,
            captured_at=captured,
            completed_at=datetime.now(UTC),
            result=result,
        )

    def _get_pipeline(self) -> Pipeline:
        pipeline = self._pipeline
        if pipeline is not None:
            return pipeline
        with self._startup_lock:
            if self._pipeline is None:
                self._pipeline = self._pipeline_factory()
            return self._pipeline

    def _build_pipeline(self) -> RecognitionPipeline:
        return RecognitionPipeline(load_model_bundle(self._config), self._config)
