from __future__ import annotations

import io
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from number_plate_recognition import cli
from number_plate_recognition.config import AppConfig
from number_plate_recognition.domain import BoundingBox, InferenceResult, PlateResult
from number_plate_recognition.errors import InferenceError


def _png_bytes(color: tuple[int, int, int] = (20, 40, 60)) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (24, 12), color).save(output, format="PNG")
    return output.getvalue()


def _result(text: str = "123A45") -> InferenceResult:
    return InferenceResult(
        annotated_image_rgb=np.full((12, 24, 3), 127, dtype=np.uint8),
        plates=(
            PlateResult(
                text=text,
                detection_confidence=0.91234567,
                recognition_confidence=0.82345678,
                box=BoundingBox(1.23456, 2.34567, 20.45678, 10.56789),
                vehicle_box=BoundingBox(0, 0, 24, 12),
                characters=(),
                format_valid=True,
            ),
        ),
        vehicle_count=1,
        timings_ms={"total": 12.3},
        model_versions={"vehicle": "vehicle-v1", "character": "character-v1"},
    )


class FakePipeline:
    def __init__(
        self,
        outcomes: Sequence[InferenceResult | Exception],
    ) -> None:
        self._outcomes = iter(outcomes)
        self.calls = 0

    def process(self, image: Any) -> InferenceResult:
        assert image.shape == (12, 24, 3)
        self.calls += 1
        outcome = next(self._outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    pipeline: FakePipeline,
) -> list[object]:
    config = AppConfig(app_root=tmp_path, max_upload_bytes=1024 * 1024)
    loaded_bundles: list[object] = []

    def fake_load_model_bundle(received_config: AppConfig) -> object:
        assert received_config is config
        bundle = object()
        loaded_bundles.append(bundle)
        return bundle

    def fake_pipeline(bundle: object, received_config: AppConfig) -> FakePipeline:
        assert bundle is loaded_bundles[0]
        assert received_config is config
        return pipeline

    monkeypatch.setattr(AppConfig, "from_env", lambda: config)
    monkeypatch.setattr(cli, "configure_logging", lambda _level: None)
    monkeypatch.setattr(cli, "load_model_bundle", fake_load_model_bundle)
    monkeypatch.setattr(cli, "RecognitionPipeline", fake_pipeline)
    return loaded_bundles


def _stdout_json(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    captured = capsys.readouterr()
    assert captured.err == ""
    parsed: dict[str, Any] = json.loads(captured.out)
    assert captured.out == json.dumps(parsed, indent=2, sort_keys=True) + "\n"
    return parsed


def test_cli_recognizes_multiple_images_with_one_pipeline_and_writes_pngs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    first = tmp_path / "first image.png"
    second = tmp_path / "second.png"
    first.write_bytes(_png_bytes())
    second.write_bytes(_png_bytes((80, 100, 120)))
    output_dir = tmp_path / "annotated"
    pipeline = FakePipeline([_result(), _result("77B8")])
    loaded_bundles = _install_fakes(monkeypatch, tmp_path, pipeline)

    exit_code = cli.main([str(first), str(second), "--output-dir", str(output_dir)])

    payload = _stdout_json(capsys)
    assert exit_code == 0
    assert pipeline.calls == 2
    assert len(loaded_bundles) == 1
    results = payload["results"]
    assert [item["status"] for item in results] == ["ok", "ok"]
    assert results[0]["plates"] == [
        {
            "box": {"x1": 1.235, "x2": 20.457, "y1": 2.346, "y2": 10.568},
            "character_confidence": 0.823457,
            "detection_confidence": 0.912346,
            "format_valid": True,
            "text": "123A45",
        }
    ]
    assert results[0]["model_versions"] == {
        "character": "character-v1",
        "vehicle": "vehicle-v1",
    }
    assert "timings_ms" not in results[0]

    output_paths = [Path(item["annotated_image"]) for item in results]
    assert [path.name for path in output_paths] == [
        "001-first-image.annotated.png",
        "002-second.annotated.png",
    ]
    for output_path in output_paths:
        with Image.open(output_path) as rendered:
            assert rendered.format == "PNG"
            assert rendered.size == (24, 12)


def test_cli_reports_invalid_and_inference_errors_and_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    invalid = tmp_path / "invalid.png"
    failing = tmp_path / "failing.png"
    unexpected = tmp_path / "unexpected.png"
    valid = tmp_path / "valid.png"
    invalid.write_bytes(b"not an image")
    failing.write_bytes(_png_bytes())
    unexpected.write_bytes(_png_bytes())
    valid.write_bytes(_png_bytes())
    pipeline = FakePipeline(
        [InferenceError("backend secret"), RuntimeError("runtime secret"), _result()]
    )
    _install_fakes(monkeypatch, tmp_path, pipeline)

    exit_code = cli.main([str(invalid), str(failing), str(unexpected), str(valid)])

    payload = _stdout_json(capsys)
    assert exit_code == 1
    assert pipeline.calls == 3
    assert [item["status"] for item in payload["results"]] == [
        "error",
        "error",
        "error",
        "ok",
    ]
    assert [item["error"]["code"] for item in payload["results"][:3]] == [
        "invalid_image",
        "recognition_error",
        "unexpected_error",
    ]
    assert "backend secret" not in json.dumps(payload)
    assert "runtime secret" not in json.dumps(payload)
    assert payload["results"][3]["annotated_image"] is None


def test_cli_normalizes_missing_input_and_pipeline_initialization_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing = tmp_path / "missing.png"
    oversized = tmp_path / "oversized.png"
    existing = tmp_path / "existing.png"
    oversized.write_bytes(b"0" * (1024 * 1024 + 1))
    existing.write_bytes(_png_bytes())
    pipeline = FakePipeline([_result()])
    _install_fakes(monkeypatch, tmp_path, pipeline)

    assert cli.main([str(missing), str(oversized), str(existing)]) == 1
    first_payload = _stdout_json(capsys)
    assert first_payload["results"][0]["error"] == {
        "code": "read_error",
        "message": "Input is not a readable file",
    }
    assert first_payload["results"][1]["error"]["code"] == "invalid_image"
    assert first_payload["results"][2]["status"] == "ok"

    def fail_to_load(_config: AppConfig) -> object:
        raise InferenceError("private model path")

    monkeypatch.setattr(cli, "load_model_bundle", fail_to_load)

    assert cli.main([str(existing), str(missing)]) == 1
    second_payload = _stdout_json(capsys)
    assert [item["error"]["code"] for item in second_payload["results"]] == [
        "pipeline_unavailable",
        "pipeline_unavailable",
    ]
    assert "private model path" not in json.dumps(second_payload)


def test_cli_reports_output_directory_and_write_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    image_path = tmp_path / "car.png"
    image_path.write_bytes(_png_bytes())
    output_file = tmp_path / "not-a-directory"
    output_file.write_text("occupied", encoding="utf-8")
    pipeline = FakePipeline([_result()])
    loaded_bundles = _install_fakes(monkeypatch, tmp_path, pipeline)

    assert cli.main([str(image_path), "--output-dir", str(output_file)]) == 1
    directory_payload = _stdout_json(capsys)
    assert directory_payload["results"][0]["error"]["code"] == "output_error"
    assert not loaded_bundles

    output_dir = tmp_path / "output"

    def fail_to_write(_path: Path, _result: InferenceResult) -> None:
        raise OSError("disk detail")

    monkeypatch.setattr(cli, "_write_annotated_png", fail_to_write)

    assert cli.main([str(image_path), "--output-dir", str(output_dir)]) == 1
    write_payload = _stdout_json(capsys)
    assert write_payload["results"][0]["error"] == {
        "code": "output_error",
        "message": "The annotated image could not be written",
    }
    assert "disk detail" not in json.dumps(write_payload)
