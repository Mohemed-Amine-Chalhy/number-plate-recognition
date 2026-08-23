"""Translate the existing ANPR result shape into control-plane observations.

The adapter is structural on purpose: the API does not import or install the heavyweight
computer-vision runtime. A worker that already owns `number_plate_recognition.InferenceResult`
can pass it here, serialize the returned payload, and call the recognition endpoint.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from control_api.schemas import RecognitionCreate, RecognitionStatus


class PlateResultLike(Protocol):
    """Subset of the current package's PlateResult needed by the wire adapter."""

    @property
    def text(self) -> str: ...

    @property
    def detection_confidence(self) -> float: ...

    @property
    def recognition_confidence(self) -> float: ...

    @property
    def format_valid(self) -> bool: ...


class InferenceResultLike(Protocol):
    """Subset of the current package's InferenceResult needed by the wire adapter."""

    @property
    def plates(self) -> Sequence[PlateResultLike]: ...

    @property
    def model_versions(self) -> Mapping[str, str]: ...


def recognition_payloads(
    result: InferenceResultLike,
    *,
    evidence_label: str = "Synthetic composite - no real person or operational record",
) -> tuple[RecognitionCreate, ...]:
    """Map local inference results without coupling recognition to authorization."""

    version = "/".join(
        f"{role}:{model_version}" for role, model_version in sorted(result.model_versions.items())
    )
    if not result.plates:
        return (
            RecognitionCreate(
                status=RecognitionStatus.UNREADABLE,
                model_version=version or "unknown",
                source="central_worker",
                evidence_label=evidence_label,
            ),
        )
    return tuple(
        RecognitionCreate(
            status=(
                RecognitionStatus.RECOGNIZED
                if plate.text and plate.format_valid
                else RecognitionStatus.UNCERTAIN
            ),
            plate_text=plate.text or None,
            detection_confidence=plate.detection_confidence,
            recognition_confidence=plate.recognition_confidence,
            format_valid=plate.format_valid,
            model_version=version or "unknown",
            source="central_worker",
            evidence_label=evidence_label,
        )
        for plate in result.plates
    )
