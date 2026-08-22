"""Safe image decoding and resizing helpers."""

from __future__ import annotations

import io
import warnings
from typing import cast

import cv2 as cv
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from number_plate_recognition.domain import ImageArray
from number_plate_recognition.errors import ImageValidationError

ALLOWED_IMAGE_FORMATS = frozenset({"JPEG", "PNG"})


def decode_image(
    payload: bytes,
    *,
    max_bytes: int,
    max_pixels: int,
) -> ImageArray:
    """Decode a bounded JPEG/PNG payload into a contiguous OpenCV BGR array."""

    if not payload:
        raise ImageValidationError("The uploaded image is empty")
    if len(payload) > max_bytes:
        raise ImageValidationError(
            f"The image exceeds the {max_bytes // (1024 * 1024)} MiB upload limit"
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(payload)) as source:
                if source.format not in ALLOWED_IMAGE_FORMATS:
                    raise ImageValidationError("Only JPEG and PNG images are supported")
                width, height = source.size
                if width <= 0 or height <= 0 or width * height > max_pixels:
                    raise ImageValidationError(
                        f"The image exceeds the {max_pixels:,}-pixel safety limit"
                    )
                oriented = ImageOps.exif_transpose(source)
                rgb = np.asarray(oriented.convert("RGB"), dtype=np.uint8)
    except ImageValidationError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ImageValidationError("The image dimensions are unsafe") from exc
    except (OSError, SyntaxError, UnidentifiedImageError, ValueError) as exc:
        raise ImageValidationError("The uploaded file is not a valid image") from exc

    return np.ascontiguousarray(cv.cvtColor(rgb, cv.COLOR_RGB2BGR))


def resize_for_inference(image: ImageArray, max_dimension: int) -> ImageArray:
    """Bound both image dimensions while preserving aspect ratio; never upscale."""

    if image.ndim != 3 or image.shape[2] != 3:
        raise ImageValidationError("Expected a three-channel color image")
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ImageValidationError("The image has invalid dimensions")
    longest_side = max(width, height)
    if longest_side <= max_dimension:
        return image.copy()
    scale = max_dimension / longest_side
    target_width = max(1, round(width * scale))
    target_height = max(1, round(height * scale))
    return cast(
        ImageArray,
        cv.resize(image, (target_width, target_height), interpolation=cv.INTER_AREA),
    )
