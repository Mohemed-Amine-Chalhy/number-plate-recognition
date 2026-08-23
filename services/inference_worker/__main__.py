"""Run one image through the central worker contract and print JSON."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from number_plate_recognition.config import AppConfig
from services.inference_worker.worker import RecognitionWorker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recognize a capture and emit the platform's wire-safe observation schema."
    )
    parser.add_argument("image", type=Path)
    parser.add_argument("--capture-id", default="capture-local")
    parser.add_argument("--job-id")
    parser.add_argument("--compact", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    config = AppConfig.from_env()
    try:
        if arguments.image.stat().st_size > config.max_upload_bytes:
            raise ValueError("image exceeds the configured upload limit")
        payload = arguments.image.read_bytes()
        observation = RecognitionWorker(config).recognize_bytes(
            payload,
            capture_id=arguments.capture_id,
            job_id=arguments.job_id,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(
        json.dumps(
            observation.to_dict(),
            indent=None if arguments.compact else 2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
