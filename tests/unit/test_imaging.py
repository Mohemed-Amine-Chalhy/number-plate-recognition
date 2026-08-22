from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from number_plate_recognition.errors import ImageValidationError
from number_plate_recognition.imaging import decode_image, resize_for_inference


def _jpeg_payload(width: int = 8, height: int = 4) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (width, height), color=(255, 0, 0)).save(output, format="JPEG")
    return output.getvalue()


def test_decode_image_returns_bgr_array() -> None:
    decoded = decode_image(_jpeg_payload(), max_bytes=10_000, max_pixels=100)

    assert decoded.shape == (4, 8, 3)
    assert decoded.dtype == np.uint8
    assert int(decoded[0, 0, 2]) > 240  # red is in the BGR array's final channel


@pytest.mark.parametrize("payload", [b"", b"not an image"])
def test_decode_image_rejects_invalid_payload(payload: bytes) -> None:
    with pytest.raises(ImageValidationError):
        decode_image(payload, max_bytes=100, max_pixels=100)


def test_decode_image_enforces_byte_and_pixel_limits() -> None:
    payload = _jpeg_payload(width=20, height=20)
    with pytest.raises(ImageValidationError, match="upload limit"):
        decode_image(payload, max_bytes=2, max_pixels=1_000)
    with pytest.raises(ImageValidationError, match="pixel safety limit"):
        decode_image(payload, max_bytes=10_000, max_pixels=100)


def test_resize_only_downscales() -> None:
    wide = np.zeros((50, 200, 3), dtype=np.uint8)
    tall = np.zeros((200, 50, 3), dtype=np.uint8)
    small = np.zeros((20, 30, 3), dtype=np.uint8)

    assert resize_for_inference(wide, 100).shape == (25, 100, 3)
    assert resize_for_inference(tall, 100).shape == (100, 25, 3)
    assert resize_for_inference(small, 100).shape == small.shape
    assert resize_for_inference(small, 100) is not small


def test_resize_rejects_non_color_array() -> None:
    with pytest.raises(ImageValidationError):
        resize_for_inference(np.zeros((10, 10), dtype=np.uint8), 100)
