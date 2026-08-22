from __future__ import annotations

import pytest

from number_plate_recognition.domain import BoundingBox, RawDetection
from number_plate_recognition.postprocessing import reconstruct_characters

CHARACTER_MAP = {"10": "A", "11": "B", "12": "E", "13": "D", "14": "H"}


def _character(x: int, label: str, confidence: float = 0.8) -> RawDetection:
    return RawDetection(
        box=BoundingBox(x, 0, x + 5, 10),
        confidence=confidence,
        class_id=int(label) if label.isdigit() else 99,
        label=label,
    )


def test_reconstructs_left_to_right_and_maps_letters() -> None:
    text, characters, confidence, valid = reconstruct_characters(
        [
            _character(40, "4", 0.9),
            _character(20, "10", 0.6),
            _character(30, "2", 0.75),
            _character(10, "1", 0.75),
        ],
        mapping=CHARACTER_MAP,
    )

    assert text == "1A24"
    assert [character.raw_label for character in characters] == ["1", "10", "2", "4"]
    assert confidence == pytest.approx(0.75)
    assert valid is True


def test_marks_unknown_format_for_manual_review() -> None:
    text, _, _, valid = reconstruct_characters([_character(1, "?")], mapping=CHARACTER_MAP)

    assert text == ""
    assert valid is False


def test_filters_undeclared_multi_character_class_labels() -> None:
    text, characters, _, valid = reconstruct_characters(
        [_character(1, "15"), _character(10, "1")],
        mapping=CHARACTER_MAP,
    )

    assert text == "1"
    assert [character.raw_label for character in characters] == ["1"]
    assert valid is False


def test_handles_no_characters() -> None:
    assert reconstruct_characters([], mapping=CHARACTER_MAP) == ("", (), 0.0, False)


@pytest.mark.parametrize("labels", [["1"], ["10", "11", "13"], ["1", "2", "3"]])
def test_rejects_incomplete_or_malformed_plate_grammar(labels: list[str]) -> None:
    detections = [_character(index * 10, label) for index, label in enumerate(labels)]

    assert reconstruct_characters(detections, mapping=CHARACTER_MAP)[3] is False
