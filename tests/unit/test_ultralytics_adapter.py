from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import number_plate_recognition.adapters.ultralytics as ultralytics_adapter
from number_plate_recognition.adapters.ultralytics import (
    UltralyticsDetector,
    _validate_model_contract,
)
from number_plate_recognition.config import AppConfig
from number_plate_recognition.errors import InferenceError, ModelIntegrityError
from number_plate_recognition.model_registry import ModelArtifact


class TensorStub:
    def __init__(self, values: list[Any]) -> None:
        self._values = np.asarray(values)

    def detach(self) -> TensorStub:
        return self

    def cpu(self) -> TensorStub:
        return self

    def numpy(self) -> np.ndarray[Any, Any]:
        return self._values


class BoxesStub:
    def __init__(self) -> None:
        self.xyxy = TensorStub([[1, 2, 10, 20]])
        self.conf = TensorStub([0.75])
        self.cls = TensorStub([2])

    def __len__(self) -> int:
        return 1


class ResultStub:
    def __init__(self, boxes: Any | None = None) -> None:
        self.boxes = boxes if boxes is not None else BoxesStub()
        self.names = {2: "car"}


class ModelStub:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    def predict(self, **kwargs: Any) -> list[ResultStub]:
        self.kwargs = kwargs
        return [ResultStub()]


def _artifact() -> ModelArtifact:
    return ModelArtifact(
        name="vehicle",
        role="vehicle",
        filename="vehicle.pt",
        sha256="a" * 64,
        size_bytes=1,
        task="detect",
        expected_classes={2: "car"},
        output_map={},
    )


def test_adapter_normalizes_predictions() -> None:
    model = ModelStub()
    detector = UltralyticsDetector(
        model, artifact=_artifact(), device="cpu", integrity_verified=True
    )
    image = np.zeros((30, 40, 3), dtype=np.uint8)

    detections = detector.predict(
        image,
        confidence=0.5,
        classes=(2, 7),
        max_detections=5,
        agnostic_nms=True,
    )

    assert len(detections) == 1
    assert detections[0].box.as_int_tuple() == (1, 2, 10, 20)
    assert detections[0].confidence == 0.75
    assert detections[0].class_id == 2
    assert detections[0].label == "car"
    assert model.kwargs["source"] is image
    assert model.kwargs["classes"] == [2]
    assert model.kwargs["max_det"] == 5
    assert model.kwargs["agnostic_nms"] is True
    assert detector.version == "vehicle@sha256:aaaaaaaaaaaa"


def test_adapter_does_not_claim_a_digest_when_integrity_is_not_verified() -> None:
    detector = UltralyticsDetector(
        ModelStub(), artifact=_artifact(), device="cpu", integrity_verified=False
    )

    assert detector.version == "vehicle@unverified"


def test_adapter_defaults_inference_to_manifest_class_contract() -> None:
    model = ModelStub()
    detector = UltralyticsDetector(
        model, artifact=_artifact(), device="cpu", integrity_verified=True
    )

    detector.predict(np.zeros((5, 5, 3), dtype=np.uint8), confidence=0.5)

    assert model.kwargs["classes"] == [2]


def test_adapter_drops_backend_output_outside_requested_classes() -> None:
    class MixedBoxes:
        def __init__(self) -> None:
            self.xyxy = TensorStub([[1, 2, 10, 20], [2, 3, 11, 21]])
            self.conf = TensorStub([0.75, 0.99])
            self.cls = TensorStub([2, 7])

        def __len__(self) -> int:
            return 2

    class IgnoringModel(ModelStub):
        def predict(self, **kwargs: Any) -> list[ResultStub]:
            self.kwargs = kwargs
            return [ResultStub(MixedBoxes())]

    detector = UltralyticsDetector(
        IgnoringModel(), artifact=_artifact(), device="cpu", integrity_verified=True
    )

    detections = detector.predict(
        np.zeros((30, 40, 3), dtype=np.uint8), confidence=0.5, classes=(2,)
    )

    assert [detection.class_id for detection in detections] == [2]


def test_adapter_wraps_external_runtime_errors() -> None:
    class BrokenModel:
        def predict(self, **kwargs: Any) -> None:
            raise RuntimeError("private runtime details")

    detector = UltralyticsDetector(
        BrokenModel(), artifact=_artifact(), device="cpu", integrity_verified=True
    )

    with pytest.raises(InferenceError, match="Inference failed"):
        detector.predict(np.zeros((5, 5, 3), dtype=np.uint8), confidence=0.5)


def test_adapter_accepts_numpy_outputs_and_rejects_mismatched_lengths() -> None:
    class NumpyBoxes:
        def __init__(self) -> None:
            self.xyxy = np.asarray([[1, 2, 3, 4]], dtype=np.float32)
            self.conf = np.asarray([0.8, 0.7], dtype=np.float32)
            self.cls = np.asarray([2], dtype=np.float32)

        def __len__(self) -> int:
            return 1

    class MismatchedModel:
        def predict(self, **kwargs: Any) -> list[ResultStub]:
            return [ResultStub(NumpyBoxes())]

    detector = UltralyticsDetector(
        MismatchedModel(), artifact=_artifact(), device="cpu", integrity_verified=True
    )

    with pytest.raises(InferenceError, match="Inference failed"):
        detector.predict(np.zeros((5, 5, 3), dtype=np.uint8), confidence=0.5)


def test_adapter_rejects_fractional_class_ids() -> None:
    class FractionalBoxes:
        def __init__(self) -> None:
            self.xyxy = np.asarray([[1, 2, 3, 4]], dtype=np.float32)
            self.conf = np.asarray([0.8], dtype=np.float32)
            self.cls = np.asarray([2.9], dtype=np.float32)

        def __len__(self) -> int:
            return 1

    class FractionalModel:
        def predict(self, **kwargs: Any) -> list[ResultStub]:
            return [ResultStub(FractionalBoxes())]

    detector = UltralyticsDetector(
        FractionalModel(), artifact=_artifact(), device="cpu", integrity_verified=True
    )

    with pytest.raises(InferenceError, match="Inference failed"):
        detector.predict(np.zeros((5, 5, 3), dtype=np.uint8), confidence=0.5)


def test_loaded_model_must_match_semantic_class_contract() -> None:
    class LoadedModel:
        def __init__(self) -> None:
            self.task = "detect"
            self.names = {2: "person"}

    with pytest.raises(ModelIntegrityError, match="class names"):
        _validate_model_contract(LoadedModel(), _artifact())


@pytest.mark.parametrize("task", [None, "segment"])
def test_loaded_model_must_expose_matching_task(task: str | None) -> None:
    class LoadedModel:
        def __init__(self) -> None:
            self.task = task
            self.names = {2: "car"}

    with pytest.raises(ModelIntegrityError, match="task"):
        _validate_model_contract(LoadedModel(), _artifact())


def test_loaded_model_accepts_matching_semantic_contract() -> None:
    class LoadedModel:
        def __init__(self) -> None:
            self.task = "detect"
            self.names = {2: "car", 99: "unused"}

    _validate_model_contract(LoadedModel(), _artifact())


def test_bundle_verifies_every_artifact_before_loading_any_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifacts = {
        "vehicle": _artifact(),
        "plate": replace(
            _artifact(),
            name="plate",
            role="plate",
            filename="plate.pt",
            expected_classes={0: "license_plate"},
        ),
        "character": replace(
            _artifact(),
            name="character",
            role="character",
            filename="character.pt",
            expected_classes={0: "0"},
        ),
    }

    class ManifestStub:
        def artifact_for_role(self, role: str) -> ModelArtifact:
            return artifacts[role]

    loaded_paths: list[Path] = []

    def verify(artifact: ModelArtifact, model_dir: Path) -> Path:
        if artifact.role == "plate":
            raise ModelIntegrityError("plate checksum mismatch")
        return model_dir / artifact.filename

    def load(path: Path) -> object:
        loaded_paths.append(path)
        return object()

    monkeypatch.setattr(ultralytics_adapter, "load_manifest", lambda path: ManifestStub())
    monkeypatch.setattr(ultralytics_adapter, "verify_artifact", verify)
    monkeypatch.setattr(ultralytics_adapter, "_load_yolo", load)

    with pytest.raises(ModelIntegrityError, match="checksum"):
        ultralytics_adapter.load_model_bundle(
            AppConfig(
                app_root=tmp_path,
                model_dir=tmp_path,
                manifest_path=tmp_path / "manifest.json",
                vehicle_classes=(2,),
            )
        )

    assert loaded_paths == []


def test_disabled_checksum_verification_still_requires_local_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifacts = {
        "vehicle": _artifact(),
        "plate": replace(
            _artifact(),
            role="plate",
            filename="plate.pt",
            expected_classes={0: "license_plate"},
        ),
        "character": replace(
            _artifact(),
            role="character",
            filename="character.pt",
            expected_classes={0: "0"},
        ),
    }

    class ManifestStub:
        def artifact_for_role(self, role: str) -> ModelArtifact:
            return artifacts[role]

    loaded_paths: list[Path] = []
    monkeypatch.setattr(ultralytics_adapter, "load_manifest", lambda path: ManifestStub())
    monkeypatch.setattr(
        ultralytics_adapter,
        "_load_yolo",
        lambda path: loaded_paths.append(path),
    )

    with pytest.raises(ModelIntegrityError, match="not found"):
        ultralytics_adapter.load_model_bundle(
            AppConfig(
                app_root=tmp_path,
                model_dir=tmp_path,
                manifest_path=tmp_path / "manifest.json",
                vehicle_classes=(2,),
                verify_models=False,
            )
        )

    assert loaded_paths == []


def test_ultralytics_runtime_defaults_to_ignored_application_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("YOLO_CONFIG_DIR", raising=False)
    config = AppConfig(app_root=tmp_path)

    ultralytics_adapter._configure_ultralytics_runtime(config)

    expected = (tmp_path / ".runtime" / "ultralytics").resolve()
    assert Path(os.environ["YOLO_CONFIG_DIR"]) == expected
    assert expected.is_dir()
    assert os.environ["YOLO_AUTOINSTALL"] == "false"
    assert os.environ["YOLO_OFFLINE"] == "true"
    assert os.environ["YOLO_VERBOSE"] == "false"
