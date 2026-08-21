from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from number_plate_recognition.config import AppConfig
from number_plate_recognition.domain import (
    BoundingBox,
    ImageArray,
    ModelBundle,
    RawDetection,
)
from number_plate_recognition.pipeline import RecognitionPipeline


class FakeDetector:
    def __init__(self, *outputs: Sequence[RawDetection], version: str = "fake-v1") -> None:
        self._outputs = list(outputs)
        self._version = version
        self.calls: list[tuple[tuple[int, ...], float, Sequence[int] | None, int | None, bool]] = []

    @property
    def version(self) -> str:
        return self._version

    def predict(
        self,
        image: ImageArray,
        *,
        confidence: float,
        classes: Sequence[int] | None = None,
        max_detections: int | None = None,
        agnostic_nms: bool = False,
    ) -> Sequence[RawDetection]:
        self.calls.append((image.shape, confidence, classes, max_detections, agnostic_nms))
        return self._outputs.pop(0) if self._outputs else ()


def _bundle(vehicle: FakeDetector, plate: FakeDetector, character: FakeDetector) -> ModelBundle:
    return ModelBundle(
        vehicle,
        plate,
        character,
        character_map={"10": "A", "11": "B", "12": "E", "13": "D", "14": "H"},
    )


def _detection(
    box: tuple[float, float, float, float],
    *,
    class_id: int,
    label: str,
    confidence: float = 0.9,
) -> RawDetection:
    return RawDetection(BoundingBox(*box), confidence, class_id, label)


def test_pipeline_runs_all_stages_and_reconstructs_plate() -> None:
    vehicle = FakeDetector(
        [_detection((10, 10, 190, 90), class_id=2, label="car")], version="vehicle-v1"
    )
    plate = FakeDetector(
        [_detection((20, 30, 160, 70), class_id=0, label="plate")], version="plate-v1"
    )
    character = FakeDetector(
        [
            _detection((50, 1, 60, 20), class_id=1, label="1", confidence=0.8),
            _detection((10, 1, 20, 20), class_id=10, label="10", confidence=0.7),
            _detection((30, 1, 40, 20), class_id=2, label="2", confidence=0.9),
            _detection((1, 1, 8, 20), class_id=1, label="1", confidence=0.8),
        ],
        version="character-v1",
    )
    pipeline = RecognitionPipeline(
        _bundle(vehicle, plate, character),
        AppConfig(inference_max_dimension=200),
    )

    result = pipeline.process(np.zeros((100, 200, 3), dtype=np.uint8))

    assert result.vehicle_count == 1
    assert len(result.plates) == 1
    recognized = result.plates[0]
    assert recognized.text == "1A21"
    assert recognized.box == BoundingBox(30, 40, 170, 80)
    assert recognized.format_valid is True
    assert recognized.recognition_confidence == 0.8
    assert result.annotated_image_rgb.shape == (100, 200, 3)
    assert result.model_versions == {
        "vehicle": "vehicle-v1",
        "plate": "plate-v1",
        "character": "character-v1",
    }
    assert vehicle.calls[0][2] == (2, 3, 5, 7)
    assert vehicle.calls[0][3] == 20


def test_pipeline_filters_non_vehicle_classes_before_downstream_inference() -> None:
    vehicle = FakeDetector([_detection((0, 0, 20, 20), class_id=0, label="person")])
    plate = FakeDetector([_detection((1, 1, 5, 5), class_id=0, label="plate")])
    character = FakeDetector()
    pipeline = RecognitionPipeline(
        _bundle(vehicle, plate, character),
        AppConfig(inference_max_dimension=100),
    )

    result = pipeline.process(np.zeros((30, 30, 3), dtype=np.uint8))

    assert result.vehicle_count == 0
    assert result.plates == ()
    assert plate.calls == []
    assert character.calls == []


def test_pipeline_clamps_boxes_and_skips_empty_regions() -> None:
    vehicle = FakeDetector(
        [
            _detection((-10, -10, 20, 20), class_id=2, label="car"),
            _detection((50, 50, 70, 70), class_id=2, label="car"),
        ]
    )
    plate = FakeDetector((), ())
    character = FakeDetector()
    pipeline = RecognitionPipeline(
        _bundle(vehicle, plate, character),
        AppConfig(inference_max_dimension=100),
    )

    result = pipeline.process(np.zeros((40, 40, 3), dtype=np.uint8))

    assert result.vehicle_count == 1
    assert len(plate.calls) == 1
    assert plate.calls[0][0] == (20, 20, 3)


def test_pipeline_downscales_before_inference() -> None:
    vehicle = FakeDetector(())
    pipeline = RecognitionPipeline(
        _bundle(vehicle, FakeDetector(), FakeDetector()),
        AppConfig(inference_max_dimension=100),
    )

    result = pipeline.process(np.zeros((100, 200, 3), dtype=np.uint8))

    assert vehicle.calls[0][0] == (50, 100, 3)
    assert result.annotated_image_rgb.shape == (50, 100, 3)


def test_pipeline_rejects_non_finite_and_out_of_range_detections() -> None:
    vehicle = FakeDetector(
        [
            _detection((0, 0, 10, 10), class_id=2, label="car", confidence=float("nan")),
            _detection((0, 0, float("inf"), 10), class_id=2, label="car"),
            _detection((0, 0, 10, 10), class_id=2, label="car", confidence=1.2),
        ]
    )
    pipeline = RecognitionPipeline(
        _bundle(vehicle, FakeDetector(), FakeDetector()),
        AppConfig(inference_max_dimension=100),
    )

    result = pipeline.process(np.zeros((20, 20, 3), dtype=np.uint8))

    assert result.vehicle_count == 0


def test_pipeline_suppresses_overlapping_character_classes() -> None:
    vehicle = FakeDetector([_detection((0, 0, 100, 50), class_id=2, label="car")])
    plate = FakeDetector([_detection((0, 0, 100, 30), class_id=0, label="plate")])
    character = FakeDetector(
        [
            _detection((1, 1, 10, 20), class_id=1, label="1"),
            _detection((20, 1, 30, 20), class_id=10, label="10", confidence=0.95),
            _detection((21, 1, 31, 20), class_id=11, label="11", confidence=0.60),
            _detection((40, 1, 50, 20), class_id=2, label="2"),
        ]
    )
    pipeline = RecognitionPipeline(
        _bundle(vehicle, plate, character),
        AppConfig(inference_max_dimension=100),
    )

    result = pipeline.process(np.zeros((50, 100, 3), dtype=np.uint8))

    assert result.plates[0].text == "1A2"
    assert [character.raw_label for character in result.plates[0].characters] == ["1", "10", "2"]
    assert character.calls[0][4] is True


def test_pipeline_deduplicates_same_absolute_plate_across_vehicles() -> None:
    vehicle = FakeDetector(
        [
            _detection((0, 0, 100, 60), class_id=2, label="car"),
            _detection((5, 0, 105, 60), class_id=2, label="car"),
        ]
    )
    plate = FakeDetector(
        [_detection((20, 20, 80, 40), class_id=0, label="plate", confidence=0.8)],
        [_detection((15, 20, 75, 40), class_id=0, label="plate", confidence=0.9)],
    )
    character_outputs = [
        _detection((1, 1, 5, 10), class_id=1, label="1"),
        _detection((10, 1, 15, 10), class_id=10, label="10"),
        _detection((20, 1, 25, 10), class_id=2, label="2"),
    ]
    character = FakeDetector(character_outputs, character_outputs)
    pipeline = RecognitionPipeline(
        _bundle(vehicle, plate, character),
        AppConfig(inference_max_dimension=120),
    )

    result = pipeline.process(np.zeros((70, 120, 3), dtype=np.uint8))

    assert len(result.plates) == 1
    assert result.plates[0].detection_confidence == 0.9
    assert len(character.calls) == 1


def test_pipeline_enforces_cascade_detection_limits() -> None:
    vehicle_predictions = [
        _detection((index * 10, 0, index * 10 + 8, 40), class_id=2, label="car")
        for index in range(8)
    ]
    plate_predictions = [
        _detection((index * 2, 5, index * 2 + 2, 20), class_id=0, label="plate")
        for index in range(5)
    ]
    character_predictions = [
        _detection(
            (index, 1, index + 0.8, 10),
            class_id=index % 10,
            label=str(index % 10),
            confidence=0.5 + index / 100,
        )
        for index in range(10)
    ]
    vehicle = FakeDetector(vehicle_predictions)
    plate = FakeDetector(plate_predictions, plate_predictions)
    character = FakeDetector(
        character_predictions,
        character_predictions,
        character_predictions,
        character_predictions,
    )
    pipeline = RecognitionPipeline(
        _bundle(vehicle, plate, character),
        AppConfig(
            inference_max_dimension=100,
            max_vehicles=2,
            max_plates_per_vehicle=2,
            max_characters_per_plate=3,
            character_overlap_iou=1.0,
        ),
    )

    result = pipeline.process(np.zeros((50, 100, 3), dtype=np.uint8))

    assert result.vehicle_count == 2
    assert len(plate.calls) == 2
    assert len(character.calls) <= 4
    assert all(len(result_plate.characters) <= 3 for result_plate in result.plates)
    assert vehicle.calls[0][3] == 2
    assert all(call[3] == 2 for call in plate.calls)
    assert all(call[3] == 3 for call in character.calls)
