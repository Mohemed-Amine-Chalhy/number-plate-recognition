"""Typed domain objects shared by detector adapters and the recognition pipeline."""

from __future__ import annotations

import math
from _thread import RLock
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

type ImageArray = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Axis-aligned box using exclusive bottom-right coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    def clamp(self, width: int, height: int) -> BoundingBox | None:
        """Clamp the box to image bounds, returning None when it becomes empty."""

        clamped = BoundingBox(
            x1=max(0.0, min(float(width), self.x1)),
            y1=max(0.0, min(float(height), self.y1)),
            x2=max(0.0, min(float(width), self.x2)),
            y2=max(0.0, min(float(height), self.y2)),
        )
        return clamped if clamped.area > 0.0 else None

    def translated(self, x_offset: float, y_offset: float) -> BoundingBox:
        return BoundingBox(
            self.x1 + x_offset,
            self.y1 + y_offset,
            self.x2 + x_offset,
            self.y2 + y_offset,
        )

    def as_int_tuple(self) -> tuple[int, int, int, int]:
        return (
            math.floor(self.x1),
            math.floor(self.y1),
            math.ceil(self.x2),
            math.ceil(self.y2),
        )

    def iou(self, other: BoundingBox) -> float:
        """Return intersection-over-union with another valid box."""

        intersection_width = max(0.0, min(self.x2, other.x2) - max(self.x1, other.x1))
        intersection_height = max(0.0, min(self.y2, other.y2) - max(self.y1, other.y1))
        intersection = intersection_width * intersection_height
        union = self.area + other.area - intersection
        return intersection / union if union > 0.0 else 0.0


@dataclass(frozen=True, slots=True)
class RawDetection:
    """Framework-neutral detector output."""

    box: BoundingBox
    confidence: float
    class_id: int
    label: str


class Detector(Protocol):
    """Minimal interface implemented by concrete object detectors."""

    @property
    def version(self) -> str:
        """Return a stable identifier for the loaded artifact."""

    def predict(
        self,
        image: ImageArray,
        *,
        confidence: float,
        classes: Sequence[int] | None = None,
        max_detections: int | None = None,
        agnostic_nms: bool = False,
    ) -> Sequence[RawDetection]:
        """Return detections for an image."""


@dataclass(frozen=True, slots=True)
class ModelBundle:
    vehicle: Detector
    plate: Detector
    character: Detector
    character_map: Mapping[str, str]
    inference_lock: RLock = field(default_factory=RLock, repr=False, compare=False)

    @property
    def versions(self) -> Mapping[str, str]:
        return MappingProxyType(
            {
                "vehicle": self.vehicle.version,
                "plate": self.plate.version,
                "character": self.character.version,
            }
        )


@dataclass(frozen=True, slots=True)
class CharacterResult:
    value: str
    raw_label: str
    confidence: float
    box: BoundingBox


@dataclass(frozen=True, slots=True)
class PlateResult:
    text: str
    detection_confidence: float
    recognition_confidence: float
    box: BoundingBox
    vehicle_box: BoundingBox
    characters: tuple[CharacterResult, ...]
    format_valid: bool


@dataclass(frozen=True, slots=True)
class InferenceResult:
    annotated_image_rgb: ImageArray
    plates: tuple[PlateResult, ...]
    vehicle_count: int
    timings_ms: Mapping[str, float] = field(default_factory=dict)
    model_versions: Mapping[str, str] = field(default_factory=dict)
