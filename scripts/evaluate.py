#!/usr/bin/env python3
"""Evaluate the production pipeline against a labeled image manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from number_plate_recognition.adapters.ultralytics import load_model_bundle
from number_plate_recognition.config import AppConfig
from number_plate_recognition.errors import NumberPlateRecognitionError
from number_plate_recognition.evaluation import EvaluationSample, evaluate_samples
from number_plate_recognition.imaging import decode_image
from number_plate_recognition.pipeline import RecognitionPipeline


class GroundTruthError(ValueError):
    """Raised when the evaluation manifest is invalid or unsafe."""


def _required_string(item: dict[str, Any], key: str, index: int) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise GroundTruthError(f"samples[{index}].{key} must be a non-empty string")
    return value.strip()


def load_ground_truth(path: Path, image_root: Path) -> list[tuple[str, Path, tuple[str, ...]]]:
    """Load a ground-truth file and resolve images inside the configured root."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GroundTruthError(f"ground-truth file does not exist: {path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GroundTruthError(f"cannot read ground-truth file {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise GroundTruthError("ground-truth root must be a JSON object")
    samples = raw.get("samples")
    if not isinstance(samples, list) or not samples:
        raise GroundTruthError("samples must be a non-empty list")

    resolved_root = image_root.resolve()
    parsed: list[tuple[str, Path, tuple[str, ...]]] = []
    sample_ids: set[str] = set()
    for index, item in enumerate(samples):
        if not isinstance(item, dict):
            raise GroundTruthError(f"samples[{index}] must be a JSON object")
        sample_id = _required_string(item, "id", index)
        image_name = _required_string(item, "image", index)
        expected_value = item.get("expected")
        if not isinstance(expected_value, list) or not expected_value:
            raise GroundTruthError(f"samples[{index}].expected must be a non-empty list")
        if any(not isinstance(value, str) or not value.strip() for value in expected_value):
            raise GroundTruthError(f"samples[{index}].expected must contain only non-empty strings")
        if sample_id in sample_ids:
            raise GroundTruthError(f"duplicate sample id: {sample_id}")

        relative_image = Path(image_name)
        if relative_image.is_absolute() or ".." in relative_image.parts:
            raise GroundTruthError(f"samples[{index}].image must be a safe relative path")
        resolved_image = (resolved_root / relative_image).resolve()
        try:
            resolved_image.relative_to(resolved_root)
        except ValueError as exc:
            raise GroundTruthError(
                f"samples[{index}].image escapes the configured image root"
            ) from exc

        sample_ids.add(sample_id)
        parsed.append(
            (
                sample_id,
                resolved_image,
                tuple(value.strip() for value in expected_value),
            )
        )
    return parsed


def _read_bounded_image(path: Path, max_bytes: int) -> bytes:
    if not path.is_file():
        raise GroundTruthError(f"image does not exist: {path}")
    try:
        declared_size = path.stat().st_size
    except OSError as exc:
        raise GroundTruthError(f"cannot inspect image {path}: {exc}") from exc
    if declared_size > max_bytes:
        raise GroundTruthError(f"image exceeds the configured byte limit: {path}")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise GroundTruthError(f"cannot read image {path}: {exc}") from exc
    if len(payload) > max_bytes:
        raise GroundTruthError(f"image exceeds the configured byte limit: {path}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ground_truth", type=Path, help="ground-truth JSON file")
    parser.add_argument(
        "--image-root",
        type=Path,
        help="base for sample image paths (default: ground-truth file directory)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run model evaluation and print metrics as JSON."""
    arguments = build_parser().parse_args(argv)
    ground_truth_path = arguments.ground_truth.expanduser().resolve()
    image_root = (
        arguments.image_root.expanduser().resolve()
        if arguments.image_root is not None
        else ground_truth_path.parent
    )

    try:
        labeled_samples = load_ground_truth(ground_truth_path, image_root)
        config = AppConfig.from_env()
        pipeline = RecognitionPipeline(load_model_bundle(config), config)
        evaluation_samples: list[EvaluationSample] = []
        for sample_id, image_path, expected in labeled_samples:
            payload = _read_bounded_image(image_path, config.max_upload_bytes)
            image = decode_image(
                payload,
                max_bytes=config.max_upload_bytes,
                max_pixels=config.max_image_pixels,
            )
            inference = pipeline.process(image)
            evaluation_samples.append(
                EvaluationSample(
                    sample_id=sample_id,
                    expected=expected,
                    predicted=tuple(plate.text for plate in inference.plates),
                )
            )
        metrics = evaluate_samples(evaluation_samples)
    except (GroundTruthError, NumberPlateRecognitionError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1
    except Exception as exc:  # Normalize unexpected third-party model/runtime failures.
        print(json.dumps({"error": f"evaluation failed: {exc}"}), file=sys.stderr)
        return 1

    print(json.dumps(metrics.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
