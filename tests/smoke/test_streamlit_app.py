from __future__ import annotations

import io
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from number_plate_recognition.domain import BoundingBox, InferenceResult, PlateResult
from number_plate_recognition.errors import InferenceError


def _png_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (32, 16), (120, 130, 140)).save(output, format="PNG")
    return output.getvalue()


def _result(*, format_valid: bool = True) -> InferenceResult:
    plate = PlateResult(
        text="123A45" if format_valid else "UNKNOWN",
        detection_confidence=0.9,
        recognition_confidence=0.8,
        box=BoundingBox(1, 1, 20, 10),
        vehicle_box=BoundingBox(0, 0, 30, 15),
        characters=(),
        format_valid=format_valid,
    )
    return InferenceResult(
        annotated_image_rgb=np.zeros((16, 32, 3), dtype=np.uint8),
        plates=(plate,),
        vehicle_count=1,
        timings_ms={"queue": 0.0, "total": 10.0},
        model_versions={
            "vehicle": "vehicle@unverified",
            "plate": "plate@sha256:bbbbbbbbbbbb",
            "character": "character@sha256:cccccccccccc",
        },
    )


class FakePipeline:
    def __init__(
        self,
        *,
        result: InferenceResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.calls = 0

    def process(self, image: Any) -> InferenceResult:
        self.calls += 1
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


def _submit_upload(app: AppTest, files: list[tuple[str, bytes, str]]) -> AppTest:
    app.file_uploader[0].set_value(files)
    submit = next(button for button in app.button if button.label == "Run recognition")
    submit.click()
    return app.run(timeout=30)


def test_streamlit_app_starts_without_loading_models(
    project_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NPR_APP_ROOT", str(tmp_path))
    app_path = project_root / "app" / "streamlit_app.py"

    app = AppTest.from_file(str(app_path)).run(timeout=30)

    assert not app.exception
    assert app.title[0].value == "Moroccan Number-Plate Recognition"
    assert app.file_uploader
    assert app.selectbox[0].label == "Demo image"
    assert next(button for button in app.button if button.label == "Run demo image").disabled
    assert not app.info
    assert any("How it works" in caption.value for caption in app.caption)


def test_streamlit_app_processes_a_valid_upload(
    project_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NPR_APP_ROOT", str(tmp_path))
    pipeline = FakePipeline(result=_result())
    app_path = project_root / "app" / "streamlit_app.py"

    with (
        patch(
            "number_plate_recognition.adapters.ultralytics.load_model_bundle",
            return_value=object(),
        ),
        patch(
            "number_plate_recognition.pipeline.RecognitionPipeline",
            return_value=pipeline,
        ),
    ):
        app = AppTest.from_file(str(app_path)).run(timeout=30)
        app = _submit_upload(
            app,
            [("[car](evil).png", _png_bytes(), "image/png")],
        )

    assert not app.exception
    assert pipeline.calls == 1
    assert any("123A45" in success.value for success in app.success)
    assert len(app.image) == 2
    assert app.dataframe
    assert {button.label for button in app.download_button} == {
        "Download annotated image",
        "Download JSON result",
    }
    assert any(text.value.startswith("File: ") for text in app.text)
    assert all("![" not in text.value for text in app.text)
    assert any(code.value == "vehicle: vehicle@unverified" for code in app.code)


def test_streamlit_app_preselects_and_runs_first_demo_image(
    project_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "z-last.png").write_bytes(_png_bytes())
    first_demo = image_dir / "a-first.png"
    first_demo.write_bytes(_png_bytes())
    monkeypatch.setenv("NPR_APP_ROOT", str(tmp_path))
    pipeline = FakePipeline(result=_result())
    app_path = project_root / "app" / "streamlit_app.py"

    with (
        patch(
            "number_plate_recognition.adapters.ultralytics.load_model_bundle",
            return_value=object(),
        ),
        patch(
            "number_plate_recognition.pipeline.RecognitionPipeline",
            return_value=pipeline,
        ),
    ):
        app = AppTest.from_file(str(app_path)).run(timeout=30)
        selected_demo = app.selectbox[0].value
        assert selected_demo is not None
        assert Path(str(selected_demo)).name == "a-first.png"
        run_demo = next(button for button in app.button if button.label == "Run demo image")
        assert not run_demo.disabled
        run_demo.click()
        app = app.run(timeout=30)

    assert not app.exception
    assert pipeline.calls == 1
    assert any("123A45" in success.value for success in app.success)
    assert len(app.download_button) == 2


def test_streamlit_app_rejects_invalid_image_before_inference(
    project_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NPR_APP_ROOT", str(tmp_path))
    pipeline = FakePipeline(result=_result())
    app_path = project_root / "app" / "streamlit_app.py"

    with (
        patch(
            "number_plate_recognition.adapters.ultralytics.load_model_bundle",
            return_value=object(),
        ),
        patch(
            "number_plate_recognition.pipeline.RecognitionPipeline",
            return_value=pipeline,
        ),
    ):
        app = AppTest.from_file(str(app_path)).run(timeout=30)
        app = _submit_upload(app, [("bad.png", b"not an image", "image/png")])

    assert not app.exception
    assert pipeline.calls == 0
    assert any("not a valid image" in error.value for error in app.error)


def test_streamlit_app_reports_model_failure_without_details(
    project_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NPR_APP_ROOT", str(tmp_path))
    pipeline = FakePipeline(error=InferenceError("sensitive backend details"))
    app_path = project_root / "app" / "streamlit_app.py"

    with (
        patch(
            "number_plate_recognition.adapters.ultralytics.load_model_bundle",
            return_value=object(),
        ),
        patch(
            "number_plate_recognition.pipeline.RecognitionPipeline",
            return_value=pipeline,
        ),
    ):
        app = AppTest.from_file(str(app_path)).run(timeout=30)
        app = _submit_upload(app, [("car.png", _png_bytes(), "image/png")])

    assert not app.exception
    assert pipeline.calls == 1
    assert any("service is unavailable" in error.value for error in app.error)
    assert all("sensitive" not in error.value for error in app.error)


def test_streamlit_app_enforces_batch_limit_before_inference(
    project_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NPR_APP_ROOT", str(tmp_path))
    monkeypatch.setenv("NPR_MAX_FILES", "1")
    pipeline = FakePipeline(result=_result())
    app_path = project_root / "app" / "streamlit_app.py"

    with (
        patch(
            "number_plate_recognition.adapters.ultralytics.load_model_bundle",
            return_value=object(),
        ),
        patch(
            "number_plate_recognition.pipeline.RecognitionPipeline",
            return_value=pipeline,
        ),
    ):
        app = AppTest.from_file(str(app_path)).run(timeout=30)
        app = _submit_upload(
            app,
            [
                ("one.png", _png_bytes(), "image/png"),
                ("two.png", _png_bytes(), "image/png"),
            ],
        )

    assert not app.exception
    assert pipeline.calls == 0
    assert any("no more than 1" in error.value for error in app.error)
