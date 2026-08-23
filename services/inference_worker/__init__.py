"""Wire-safe adapter around the existing number-plate recognition pipeline."""

from services.inference_worker.contracts import RecognitionObservation
from services.inference_worker.worker import RecognitionWorker

__all__ = ["RecognitionObservation", "RecognitionWorker"]
