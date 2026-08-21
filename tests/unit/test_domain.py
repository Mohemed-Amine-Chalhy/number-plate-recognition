from __future__ import annotations

from number_plate_recognition.domain import BoundingBox


def test_bounding_box_clamps_to_image() -> None:
    box = BoundingBox(-4, 3, 15, 20).clamp(width=10, height=12)

    assert box == BoundingBox(0, 3, 10, 12)
    assert box.area == 90


def test_bounding_box_rejects_empty_intersection() -> None:
    assert BoundingBox(20, 20, 30, 30).clamp(width=10, height=10) is None
    assert BoundingBox(5, 5, 5, 8).clamp(width=10, height=10) is None


def test_bounding_box_translation_is_immutable() -> None:
    original = BoundingBox(1, 2, 3, 4)

    assert original.translated(10, 20) == BoundingBox(11, 22, 13, 24)
    assert original == BoundingBox(1, 2, 3, 4)


def test_integer_bounds_keep_fractional_edge_pixels() -> None:
    assert BoundingBox(1.9, 2.1, 3.1, 4.01).as_int_tuple() == (1, 2, 4, 5)


def test_bounding_box_iou() -> None:
    assert BoundingBox(0, 0, 10, 10).iou(BoundingBox(5, 0, 15, 10)) == 1 / 3
    assert BoundingBox(0, 0, 2, 2).iou(BoundingBox(3, 3, 4, 4)) == 0.0
