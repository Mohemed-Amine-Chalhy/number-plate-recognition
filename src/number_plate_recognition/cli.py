"""Command-line adapter for batch image recognition."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections.abc import Sequence
from pathlib import Path

import cv2 as cv

from number_plate_recognition.adapters.ultralytics import load_model_bundle
from number_plate_recognition.config import AppConfig
from number_plate_recognition.domain import BoundingBox, InferenceResult, PlateResult
from number_plate_recognition.errors import ImageValidationError, NumberPlateRecognitionError
from number_plate_recognition.imaging import decode_image
from number_plate_recognition.observability import configure_logging
from number_plate_recognition.pipeline import RecognitionPipeline

LOGGER = logging.getLogger("number_plate_recognition.cli")
_SAFE_STEM = re.compile(r"[^A-Za-z0-9._-]+")


class _InputReadError(Exception):
    """Raised when a CLI input cannot be read within the configured bound."""


def _read_bounded(path: Path, max_bytes: int) -> bytes:
    try:
        if not path.is_file():
            raise _InputReadError("Input is not a readable file")
        if path.stat().st_size > max_bytes:
            raise ImageValidationError(
                f"The image exceeds the {max_bytes // (1024 * 1024)} MiB upload limit"
            )
        with path.open("rb") as image_file:
            payload = image_file.read(max_bytes + 1)
    except OSError as exc:
        raise _InputReadError("Input is not a readable file") from exc
    if len(payload) > max_bytes:
        raise ImageValidationError(
            f"The image exceeds the {max_bytes // (1024 * 1024)} MiB upload limit"
        )
    return payload


def _box_payload(box: BoundingBox) -> dict[str, float]:
    return {
        "x1": round(box.x1, 3),
        "x2": round(box.x2, 3),
        "y1": round(box.y1, 3),
        "y2": round(box.y2, 3),
    }


def _plate_payload(plate: PlateResult) -> dict[str, object]:
    return {
        "box": _box_payload(plate.box),
        "character_confidence": round(plate.recognition_confidence, 6),
        "detection_confidence": round(plate.detection_confidence, 6),
        "format_valid": plate.format_valid,
        "text": plate.text,
    }


def _success_payload(
    image_path: Path,
    result: InferenceResult,
    annotated_path: Path | None,
) -> dict[str, object]:
    return {
        "annotated_image": str(annotated_path) if annotated_path is not None else None,
        "image": str(image_path),
        "model_versions": dict(sorted(result.model_versions.items())),
        "plates": [_plate_payload(plate) for plate in result.plates],
        "status": "ok",
        "vehicle_count": result.vehicle_count,
    }


def _error_payload(image_path: Path, *, code: str, message: str) -> dict[str, object]:
    return {
        "error": {"code": code, "message": message},
        "image": str(image_path),
        "status": "error",
    }


def _output_path(output_dir: Path, image_path: Path, index: int) -> Path:
    stem = _SAFE_STEM.sub("-", image_path.stem).strip("-._") or "image"
    return output_dir / f"{index:03d}-{stem}.annotated.png"


def _write_annotated_png(path: Path, result: InferenceResult) -> None:
    bgr = cv.cvtColor(result.annotated_image_rgb, cv.COLOR_RGB2BGR)
    encoded, png = cv.imencode(".png", bgr)
    if not encoded:
        raise OSError("OpenCV could not encode the annotated image")
    path.write_bytes(png.tobytes())


def _emit(results: list[dict[str, object]]) -> None:
    json.dump({"results": results}, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(
        prog="npr-recognize",
        description="Recognize Moroccan number plates in one or more JPEG/PNG images.",
    )
    parser.add_argument("images", nargs="+", type=Path, metavar="IMAGE")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="write annotated PNG files to this directory",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Recognize every requested image and emit one deterministic JSON document."""

    arguments = build_parser().parse_args(argv)
    image_paths: list[Path] = arguments.images
    output_dir: Path | None = arguments.output_dir
    results: list[dict[str, object]] = []

    if output_dir is not None:
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            results.extend(
                _error_payload(
                    image_path,
                    code="output_error",
                    message="The output directory could not be created",
                )
                for image_path in image_paths
            )
            _emit(results)
            return 1

    try:
        config = AppConfig.from_env()
        configure_logging(config.log_level)
        pipeline = RecognitionPipeline(load_model_bundle(config), config)
    except Exception:
        LOGGER.error("recognition pipeline initialization failed")
        results.extend(
            _error_payload(
                image_path,
                code="pipeline_unavailable",
                message="The recognition pipeline could not be initialized",
            )
            for image_path in image_paths
        )
        _emit(results)
        return 1

    failed = False
    for index, image_path in enumerate(image_paths, start=1):
        try:
            payload = _read_bounded(image_path, config.max_upload_bytes)
            image = decode_image(
                payload,
                max_bytes=config.max_upload_bytes,
                max_pixels=config.max_image_pixels,
            )
            inference = pipeline.process(image)
            annotated_path = (
                _output_path(output_dir, image_path, index) if output_dir is not None else None
            )
            if annotated_path is not None:
                _write_annotated_png(annotated_path, inference)
        except _InputReadError as exc:
            failed = True
            results.append(_error_payload(image_path, code="read_error", message=str(exc)))
        except ImageValidationError as exc:
            failed = True
            results.append(_error_payload(image_path, code="invalid_image", message=str(exc)))
        except NumberPlateRecognitionError:
            failed = True
            LOGGER.error("recognition failed for input %d", index)
            results.append(
                _error_payload(
                    image_path,
                    code="recognition_error",
                    message="Recognition failed for this image",
                )
            )
        except OSError:
            failed = True
            LOGGER.error("annotated output could not be written for input %d", index)
            results.append(
                _error_payload(
                    image_path,
                    code="output_error",
                    message="The annotated image could not be written",
                )
            )
        except Exception:
            failed = True
            LOGGER.error("unexpected recognition failure for input %d", index)
            results.append(
                _error_payload(
                    image_path,
                    code="unexpected_error",
                    message="Recognition failed unexpectedly for this image",
                )
            )
        else:
            results.append(_success_payload(image_path, inference, annotated_path))

    _emit(results)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
