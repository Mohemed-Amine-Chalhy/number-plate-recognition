"""Thread-safe adapter around Ultralytics YOLO detectors."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np
from numpy.typing import NDArray

from number_plate_recognition.config import AppConfig
from number_plate_recognition.domain import (
    BoundingBox,
    ImageArray,
    ModelBundle,
    RawDetection,
)
from number_plate_recognition.errors import InferenceError, ModelIntegrityError
from number_plate_recognition.model_registry import (
    ModelArtifact,
    load_manifest,
    verify_artifact,
)


class UltralyticsDetector:
    """Normalize YOLO outputs and serialize access to a model instance."""

    def __init__(
        self,
        model: Any,
        *,
        artifact: ModelArtifact,
        device: str,
        integrity_verified: bool,
    ) -> None:
        self._model = model
        self._device = device
        self._lock = RLock()
        self._version = (
            f"{artifact.name}@sha256:{artifact.sha256[:12]}"
            if integrity_verified
            else f"{artifact.name}@unverified"
        )
        self._allowed_class_ids = frozenset(artifact.expected_classes)
        self._class_names = dict(artifact.expected_classes)

    @property
    def version(self) -> str:
        return self._version

    def predict(
        self,
        image: ImageArray,
        *,
        confidence: float,
        classes: Sequence[int] | None = None,
        max_detections: int | None = None,
        agnostic_nms: bool = False,
    ) -> Sequence[RawDetection]:
        kwargs: dict[str, Any] = {
            "source": image,
            "conf": confidence,
            "verbose": False,
        }
        if self._device != "auto":
            kwargs["device"] = self._device
        requested_classes = (
            self._allowed_class_ids
            if classes is None
            else self._allowed_class_ids.intersection(classes)
        )
        if not requested_classes:
            return ()
        kwargs["classes"] = sorted(requested_classes)
        if max_detections is not None:
            kwargs["max_det"] = max_detections
        if agnostic_nms:
            kwargs["agnostic_nms"] = True
        try:
            with self._lock:
                results = self._model.predict(**kwargs)
            if not results:
                return ()
            result = results[0]
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                return ()
            coordinates = _as_numpy(boxes.xyxy)
            confidences = _as_numpy(boxes.conf).reshape(-1)
            class_ids = _as_numpy(boxes.cls).reshape(-1)
            if coordinates.ndim != 2 or coordinates.shape[1] < 4:
                raise ValueError("detector coordinates must have shape (n, >=4)")
            if not (len(coordinates) == len(confidences) == len(class_ids)):
                raise ValueError("detector output arrays have different lengths")

            detections: list[RawDetection] = []
            for coordinate, score, class_id_value in zip(
                coordinates, confidences, class_ids, strict=True
            ):
                numeric_class_id = float(class_id_value)
                if not math.isfinite(numeric_class_id) or not numeric_class_id.is_integer():
                    raise ValueError("detector class IDs must be finite integers")
                class_id = int(numeric_class_id)
                if class_id not in requested_classes:
                    continue
                label = self._class_names[class_id]
                x1, y1, x2, y2 = (float(value) for value in coordinate[:4])
                detections.append(
                    RawDetection(
                        box=BoundingBox(x1, y1, x2, y2),
                        confidence=float(score),
                        class_id=class_id,
                        label=label,
                    )
                )
            return tuple(detections)
        except Exception as exc:  # external runtime failures are normalized here
            raise InferenceError(f"Inference failed for {self._version}") from exc


def _as_numpy(value: Any) -> NDArray[Any]:
    """Normalize either a torch-like tensor or an existing NumPy array."""

    if isinstance(value, np.ndarray):
        return value
    detached = value.detach() if hasattr(value, "detach") else value
    on_cpu = detached.cpu() if hasattr(detached, "cpu") else detached
    if hasattr(on_cpu, "numpy"):
        return np.asarray(on_cpu.numpy())
    return np.asarray(on_cpu)


def _configure_ultralytics_runtime(config: AppConfig) -> None:
    """Keep third-party settings local unless the caller provides an override."""

    configured_dir = Path(
        os.environ.setdefault(
            "YOLO_CONFIG_DIR",
            str(config.app_root / ".runtime" / "ultralytics"),
        )
    ).expanduser()
    if not configured_dir.is_absolute():
        configured_dir = config.app_root / configured_dir
    configured_dir = configured_dir.resolve()
    try:
        configured_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise InferenceError("Cannot prepare the Ultralytics runtime directory") from exc
    os.environ["YOLO_CONFIG_DIR"] = str(configured_dir)
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_VERBOSE"] = "false"


def _load_yolo(path: Path) -> Any:
    try:
        from ultralytics import YOLO  # type: ignore[attr-defined]
    except ImportError as exc:
        raise InferenceError(
            "Ultralytics is not installed; run the environment bootstrap script"
        ) from exc
    try:
        return YOLO(str(path), task="detect")
    except Exception as exc:
        raise InferenceError(f"Cannot load model artifact: {path.name}") from exc


def _model_class_names(model: Any) -> dict[int, str]:
    names = getattr(model, "names", None)
    if isinstance(names, Mapping):
        normalized: dict[int, str] = {}
        try:
            for class_id, label in names.items():
                normalized[int(class_id)] = str(label)
        except (TypeError, ValueError) as exc:
            raise ModelIntegrityError("Loaded model has an invalid class-name schema") from exc
        return normalized
    if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        return {class_id: str(label) for class_id, label in enumerate(names)}
    raise ModelIntegrityError("Loaded model does not expose a class-name schema")


def _validate_model_contract(model: Any, artifact: ModelArtifact) -> None:
    """Ensure a checksum-valid model still has the semantics the pipeline expects."""

    actual_task = getattr(model, "task", None)
    if actual_task is None or str(actual_task).casefold() != artifact.task:
        raise ModelIntegrityError(
            f"Model '{artifact.name}' task does not match its manifest contract"
        )
    actual_names = _model_class_names(model)
    mismatches = {
        class_id: (expected_label, actual_names.get(class_id))
        for class_id, expected_label in artifact.expected_classes.items()
        if actual_names.get(class_id) != expected_label
    }
    if mismatches:
        raise ModelIntegrityError(
            f"Model '{artifact.name}' class names do not match its manifest contract"
        )


def load_model_bundle(config: AppConfig) -> ModelBundle:
    """Load all configured detectors after optional manifest verification."""

    manifest = load_manifest(config.manifest_path)
    roles = ("vehicle", "plate", "character")
    artifacts = {role: manifest.artifact_for_role(role) for role in roles}
    missing_vehicle_classes = set(config.vehicle_classes) - set(
        artifacts["vehicle"].expected_classes
    )
    if missing_vehicle_classes:
        raise ModelIntegrityError(
            "Configured vehicle classes are absent from the vehicle model contract"
        )

    # Complete every static integrity check before deserializing any
    # pickle-backed checkpoint. A bundle therefore fails as one unit.
    paths: dict[str, Path] = {}
    for role, artifact in artifacts.items():
        path = (
            verify_artifact(artifact, config.model_dir)
            if config.verify_models
            else artifact.path_in(config.model_dir)
        )
        if not config.verify_models and not path.is_file():
            raise ModelIntegrityError(f"Model artifact not found: {path}")
        paths[role] = path

    _configure_ultralytics_runtime(config)
    detectors: dict[str, UltralyticsDetector] = {}
    for role in roles:
        artifact = artifacts[role]
        model = _load_yolo(paths[role])
        _validate_model_contract(model, artifact)
        detectors[role] = UltralyticsDetector(
            model,
            artifact=artifact,
            device=config.device,
            integrity_verified=config.verify_models,
        )
    return ModelBundle(
        vehicle=detectors["vehicle"],
        plate=detectors["plate"],
        character=detectors["character"],
        character_map=artifacts["character"].output_map,
    )
