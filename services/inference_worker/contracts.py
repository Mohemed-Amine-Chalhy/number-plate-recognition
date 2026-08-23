"""Serializable contracts emitted by the central inference worker."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Self

from number_plate_recognition.domain import BoundingBox, InferenceResult, PlateResult


@dataclass(frozen=True, slots=True)
class BoxObservation:
    """Pixel-space bounding box with exclusive bottom-right coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    @classmethod
    def from_domain(cls, box: BoundingBox) -> Self:
        """Convert a pipeline bounding box without leaking NumPy values."""

        return cls(
            x1=round(float(box.x1), 3),
            y1=round(float(box.y1), 3),
            x2=round(float(box.x2), 3),
            y2=round(float(box.y2), 3),
        )


@dataclass(frozen=True, slots=True)
class PlateObservation:
    """One immutable plate candidate emitted by a model bundle."""

    text: str
    detection_confidence: float
    recognition_confidence: float
    format_valid: bool
    box: BoxObservation
    vehicle_box: BoxObservation

    @classmethod
    def from_domain(cls, plate: PlateResult) -> Self:
        """Convert a plate result into a stable transport value."""

        return cls(
            text=plate.text,
            detection_confidence=round(float(plate.detection_confidence), 6),
            recognition_confidence=round(float(plate.recognition_confidence), 6),
            format_valid=plate.format_valid,
            box=BoxObservation.from_domain(plate.box),
            vehicle_box=BoxObservation.from_domain(plate.vehicle_box),
        )


@dataclass(frozen=True, slots=True)
class RecognitionObservation:
    """JSON-safe, immutable result crossing the inference/control-plane boundary."""

    schema_version: int
    job_id: str
    capture_id: str
    captured_at: str
    completed_at: str
    vehicle_count: int
    plates: tuple[PlateObservation, ...]
    timings_ms: dict[str, float]
    model_versions: dict[str, str]

    @classmethod
    def from_result(
        cls,
        *,
        job_id: str,
        capture_id: str,
        captured_at: datetime,
        completed_at: datetime,
        result: InferenceResult,
    ) -> Self:
        """Build the v1 transport contract from the in-memory pipeline result."""

        return cls(
            schema_version=1,
            job_id=job_id,
            capture_id=capture_id,
            captured_at=_as_utc(captured_at),
            completed_at=_as_utc(completed_at),
            vehicle_count=result.vehicle_count,
            plates=tuple(PlateObservation.from_domain(plate) for plate in result.plates),
            timings_ms={
                str(name): round(float(value), 3)
                for name, value in sorted(result.timings_ms.items())
            },
            model_versions={
                str(name): str(version) for name, version in sorted(result.model_versions.items())
            },
        )

    def to_dict(self) -> dict[str, object]:
        """Return a value accepted by standard JSON encoders."""

        return asdict(self)


def _as_utc(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("observation timestamps must be timezone-aware")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
