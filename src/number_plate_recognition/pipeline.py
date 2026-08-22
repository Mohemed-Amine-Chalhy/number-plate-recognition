"""Framework-independent, typed number-plate recognition pipeline."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import cv2 as cv

from number_plate_recognition.config import AppConfig
from number_plate_recognition.domain import (
    BoundingBox,
    ImageArray,
    InferenceResult,
    ModelBundle,
    PlateResult,
    RawDetection,
)
from number_plate_recognition.errors import InferenceError
from number_plate_recognition.imaging import resize_for_inference
from number_plate_recognition.postprocessing import reconstruct_characters


def _valid_detections(
    detections: Sequence[RawDetection],
    *,
    confidence: float,
    width: int,
    height: int,
    allowed_classes: Sequence[int] | None = None,
) -> tuple[RawDetection, ...]:
    allowed = set(allowed_classes) if allowed_classes is not None else None
    valid: list[RawDetection] = []
    for detection in detections:
        values = (
            detection.confidence,
            detection.box.x1,
            detection.box.y1,
            detection.box.x2,
            detection.box.y2,
        )
        if not all(math.isfinite(value) for value in values):
            continue
        if not 0.0 <= detection.confidence <= 1.0:
            continue
        if detection.confidence < confidence:
            continue
        if allowed is not None and detection.class_id not in allowed:
            continue
        box = detection.box.clamp(width, height)
        if box is None:
            continue
        valid.append(
            RawDetection(
                box=box,
                confidence=detection.confidence,
                class_id=detection.class_id,
                label=detection.label,
            )
        )
    return tuple(valid)


def _suppress_overlapping_detections(
    detections: Sequence[RawDetection], *, iou_threshold: float
) -> tuple[RawDetection, ...]:
    """Apply class-agnostic overlap suppression, retaining higher confidence."""

    ranked = sorted(
        enumerate(detections),
        key=lambda item: (-item[1].confidence, item[0]),
    )
    kept: list[tuple[int, RawDetection]] = []
    for original_index, candidate in ranked:
        if any(candidate.box.iou(existing.box) > iou_threshold for _, existing in kept):
            continue
        kept.append((original_index, candidate))
    kept.sort(key=lambda item: item[0])
    return tuple(detection for _, detection in kept)


def _rank_and_limit_detections(
    detections: Sequence[RawDetection], *, limit: int
) -> tuple[RawDetection, ...]:
    ranked = sorted(
        enumerate(detections),
        key=lambda item: (-item[1].confidence, item[0]),
    )[:limit]
    ranked.sort(key=lambda item: item[0])
    return tuple(detection for _, detection in ranked)


@dataclass(frozen=True, slots=True)
class _PlateCandidate:
    detection: RawDetection
    vehicle_box: BoundingBox


def _deduplicate_plate_candidates(
    candidates: Sequence[_PlateCandidate], *, iou_threshold: float
) -> tuple[_PlateCandidate, ...]:
    """Deduplicate absolute plate boxes before expensive character inference."""

    ranked = sorted(
        enumerate(candidates),
        key=lambda item: (-item[1].detection.confidence, item[0]),
    )
    kept: list[tuple[int, _PlateCandidate]] = []
    for original_index, candidate in ranked:
        if any(
            candidate.detection.box.iou(existing.detection.box) > iou_threshold
            for _, existing in kept
        ):
            continue
        kept.append((original_index, candidate))
    kept.sort(key=lambda item: item[0])
    return tuple(candidate for _, candidate in kept)


def _translate_detection(detection: RawDetection, x_offset: float, y_offset: float) -> RawDetection:
    return RawDetection(
        box=detection.box.translated(x_offset, y_offset),
        confidence=detection.confidence,
        class_id=detection.class_id,
        label=detection.label,
    )


def _draw_box(
    image: ImageArray,
    box: BoundingBox,
    color: tuple[int, int, int],
    *,
    width: int,
) -> None:
    x1, y1, x2, y2 = box.as_int_tuple()
    cv.rectangle(image, (x1, y1), (x2, y2), color, width)


class RecognitionPipeline:
    """Run vehicle, plate, and character detectors as a bounded pipeline."""

    def __init__(self, models: ModelBundle, config: AppConfig) -> None:
        self._models = models
        self._config = config

    def process(self, image_bgr: ImageArray) -> InferenceResult:
        """Serialize a complete request across the shared model bundle."""

        request_started = time.perf_counter()
        with self._models.inference_lock:
            queue_ms = (time.perf_counter() - request_started) * 1000
            return self._process_locked(
                image_bgr,
                request_started=request_started,
                queue_ms=queue_ms,
            )

    def _process_locked(
        self,
        image_bgr: ImageArray,
        *,
        request_started: float,
        queue_ms: float,
    ) -> InferenceResult:
        image = resize_for_inference(image_bgr, self._config.inference_max_dimension)
        annotated = image.copy()
        height, width = image.shape[:2]

        vehicle_started = time.perf_counter()
        vehicle_predictions = self._models.vehicle.predict(
            image,
            confidence=self._config.vehicle_confidence,
            classes=self._config.vehicle_classes,
            max_detections=self._config.max_vehicles,
        )
        vehicles = _rank_and_limit_detections(
            _valid_detections(
                vehicle_predictions,
                confidence=self._config.vehicle_confidence,
                width=width,
                height=height,
                allowed_classes=self._config.vehicle_classes,
            ),
            limit=self._config.max_vehicles,
        )
        vehicle_ms = (time.perf_counter() - vehicle_started) * 1000

        plate_ms = 0.0
        character_ms = 0.0
        plate_candidates: list[_PlateCandidate] = []

        for vehicle in vehicles:
            _draw_box(annotated, vehicle.box, (0, 0, 255), width=2)
            x1_car, y1_car, x2_car, y2_car = vehicle.box.as_int_tuple()
            vehicle_roi = image[y1_car:y2_car, x1_car:x2_car]
            if vehicle_roi.size == 0:
                continue

            stage_started = time.perf_counter()
            plate_predictions = self._models.plate.predict(
                vehicle_roi,
                confidence=self._config.plate_confidence,
                max_detections=self._config.max_plates_per_vehicle,
            )
            plate_ms += (time.perf_counter() - stage_started) * 1000
            roi_height, roi_width = vehicle_roi.shape[:2]
            plates = _rank_and_limit_detections(
                _valid_detections(
                    plate_predictions,
                    confidence=self._config.plate_confidence,
                    width=roi_width,
                    height=roi_height,
                ),
                limit=self._config.max_plates_per_vehicle,
            )

            for plate in plates:
                absolute_plate = _translate_detection(plate, x1_car, y1_car)
                absolute_box = absolute_plate.box.clamp(width, height)
                if absolute_box is None:
                    continue
                plate_candidates.append(
                    _PlateCandidate(
                        detection=RawDetection(
                            box=absolute_box,
                            confidence=absolute_plate.confidence,
                            class_id=absolute_plate.class_id,
                            label=absolute_plate.label,
                        ),
                        vehicle_box=vehicle.box,
                    )
                )

        plate_candidates = list(
            _deduplicate_plate_candidates(
                plate_candidates,
                iou_threshold=self._config.plate_dedup_iou,
            )
        )
        plate_results: list[PlateResult] = []
        for candidate in plate_candidates:
            absolute_plate = candidate.detection
            x1_plate, y1_plate, x2_plate, y2_plate = absolute_plate.box.as_int_tuple()
            plate_roi = image[y1_plate:y2_plate, x1_plate:x2_plate]
            if plate_roi.size == 0:
                continue

            stage_started = time.perf_counter()
            character_predictions = self._models.character.predict(
                plate_roi,
                confidence=self._config.character_confidence,
                max_detections=self._config.max_characters_per_plate,
                agnostic_nms=True,
            )
            character_ms += (time.perf_counter() - stage_started) * 1000
            plate_height, plate_width = plate_roi.shape[:2]
            local_characters = _valid_detections(
                character_predictions,
                confidence=self._config.character_confidence,
                width=plate_width,
                height=plate_height,
            )
            local_characters = _rank_and_limit_detections(
                _suppress_overlapping_detections(
                    local_characters,
                    iou_threshold=self._config.character_overlap_iou,
                ),
                limit=self._config.max_characters_per_plate,
            )
            absolute_characters = tuple(
                _translate_detection(character, x1_plate, y1_plate)
                for character in local_characters
            )
            text, characters, recognition_confidence, format_valid = reconstruct_characters(
                absolute_characters,
                mapping=self._models.character_map,
                plate_pattern=self._config.plate_pattern,
            )
            plate_results.append(
                PlateResult(
                    text=text,
                    detection_confidence=absolute_plate.confidence,
                    recognition_confidence=recognition_confidence,
                    box=absolute_plate.box,
                    vehicle_box=candidate.vehicle_box,
                    characters=characters,
                    format_valid=format_valid,
                )
            )

        deduplicated_plates = tuple(plate_results)
        for plate_result in deduplicated_plates:
            _draw_box(annotated, plate_result.box, (0, 255, 0), width=2)
            x1_plate, y1_plate, _, _ = plate_result.box.as_int_tuple()
            for character in plate_result.characters:
                _draw_box(annotated, character.box, (255, 0, 0), width=1)
                x1_char, y1_char, _, _ = character.box.as_int_tuple()
                cv.putText(
                    annotated,
                    character.value,
                    (x1_char, max(12, y1_char - 5)),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2,
                )
            if plate_result.text:
                cv.putText(
                    annotated,
                    plate_result.text,
                    (x1_plate, max(18, y1_plate - 8)),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

        if any(value < 0 for value in (queue_ms, vehicle_ms, plate_ms, character_ms)):
            raise InferenceError("Inference timing failed")
        total_ms = (time.perf_counter() - request_started) * 1000
        annotated_rgb = cast(ImageArray, cv.cvtColor(annotated, cv.COLOR_BGR2RGB))
        return InferenceResult(
            annotated_image_rgb=annotated_rgb,
            plates=deduplicated_plates,
            vehicle_count=len(vehicles),
            timings_ms={
                "queue": queue_ms,
                "vehicle": vehicle_ms,
                "plate": plate_ms,
                "character": character_ms,
                "total": total_ms,
            },
            model_versions=self._models.versions,
        )
