"""Streamlit user interface for the number-plate recognition package."""

from __future__ import annotations

import json
import logging
import math
import uuid
from pathlib import Path

import cv2 as cv
import streamlit as st

from number_plate_recognition.adapters.ultralytics import load_model_bundle
from number_plate_recognition.config import AppConfig
from number_plate_recognition.domain import InferenceResult
from number_plate_recognition.errors import ImageValidationError, NumberPlateRecognitionError
from number_plate_recognition.imaging import decode_image, resize_for_inference
from number_plate_recognition.observability import configure_logging
from number_plate_recognition.pipeline import RecognitionPipeline

st.set_page_config(page_title="Moroccan Plate Recognition", page_icon="🚘", layout="wide")
LOGGER = logging.getLogger("number_plate_recognition.ui")
_SAFE_FILENAME_CHARACTERS = frozenset(" ._-()")


@st.cache_resource(show_spinner="Loading recognition models…")
def _cached_pipeline(config: AppConfig) -> RecognitionPipeline:
    """Load and reuse one thread-safe model bundle per configuration."""

    return RecognitionPipeline(load_model_bundle(config), config)


def _result_rows(result: InferenceResult) -> list[dict[str, str]]:
    return [
        {
            "Plate": plate.text or "Not reconstructed",
            "Plate confidence": f"{plate.detection_confidence:.1%}",
            "Character confidence": f"{plate.recognition_confidence:.1%}",
            "Pattern": "Match" if plate.format_valid else "Review",
        }
        for plate in result.plates
    ]


def _result_json(name: str, result: InferenceResult) -> bytes:
    payload = {
        "file": _display_name(name),
        "model_versions": dict(sorted(result.model_versions.items())),
        "plates": [
            {
                "character_confidence": round(plate.recognition_confidence, 6),
                "detection_confidence": round(plate.detection_confidence, 6),
                "format_valid": plate.format_valid,
                "text": plate.text,
            }
            for plate in result.plates
        ],
        "timings_ms": {key: round(value, 3) for key, value in sorted(result.timings_ms.items())},
        "vehicle_count": result.vehicle_count,
    }
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _annotated_png(result: InferenceResult) -> bytes:
    annotated_bgr = cv.cvtColor(result.annotated_image_rgb, cv.COLOR_RGB2BGR)
    encoded, png = cv.imencode(".png", annotated_bgr)
    if not encoded:
        raise NumberPlateRecognitionError("Could not encode the annotated image")
    return png.tobytes()


def _display_name(name: str) -> str:
    """Return a bounded plain-text basename suitable for untrusted uploads."""

    basename = Path(name).name[:128]
    cleaned = "".join(
        character
        if character.isascii() and (character.isalnum() or character in _SAFE_FILENAME_CHARACTERS)
        else "_"
        for character in basename
    ).strip()
    return cleaned or "uploaded-image"


def _render_result(name: str, payload: bytes, config: AppConfig) -> None:
    request_id = uuid.uuid4().hex
    st.subheader("Recognition result")
    st.text(f"File: {_display_name(name)}")
    try:
        original_bgr = decode_image(
            payload,
            max_bytes=config.max_upload_bytes,
            max_pixels=config.max_image_pixels,
        )
        with st.spinner("Running vehicle, plate, and character detection…"):
            result = _cached_pipeline(config).process(original_bgr)
        annotated_png = _annotated_png(result)
        result_json = _result_json(name, result)
    except ImageValidationError as exc:
        LOGGER.info(
            "image rejected by input validation",
            extra={"event": "image_rejected", "request_id": request_id},
        )
        st.error(str(exc))
        return
    except NumberPlateRecognitionError:
        LOGGER.exception(
            "recognition service failure",
            extra={"event": "recognition_failed", "request_id": request_id},
        )
        st.error("The recognition service is unavailable. Contact the operator.")
        return
    except Exception:
        LOGGER.exception(
            "unexpected recognition failure",
            extra={"event": "recognition_failed", "request_id": request_id},
        )
        st.error("Recognition failed unexpectedly. Check the server logs for details.")
        return

    preview_bgr = resize_for_inference(original_bgr, config.inference_max_dimension)
    original_rgb = cv.cvtColor(preview_bgr, cv.COLOR_BGR2RGB)
    original_column, result_column = st.columns(2)
    with original_column:
        st.image(original_rgb, caption="Original image", width="stretch")
    with result_column:
        st.image(
            result.annotated_image_rgb,
            caption="Recognition result",
            width="stretch",
        )

    rows = _result_rows(result)
    pattern_matches = list(
        dict.fromkeys(plate.text for plate in result.plates if plate.text and plate.format_valid)
    )
    review_required = list(
        dict.fromkeys(
            plate.text for plate in result.plates if plate.text and not plate.format_valid
        )
    )
    if pattern_matches:
        st.success(f"Detected plate(s): {', '.join(pattern_matches)}")
    if review_required:
        st.warning(
            "Prediction(s) require review because they do not match the configured "
            f"plate pattern: {', '.join(review_required)}"
        )
    elif result.plates and not pattern_matches:
        st.warning("Plate regions were detected, but their text could not be reconstructed.")
    elif not result.plates:
        st.warning("No license plate was detected.")
    if rows:
        st.dataframe(rows, width="stretch", hide_index=True)

    st.caption(
        f"Vehicles: {result.vehicle_count} · "
        f"Total inference: {result.timings_ms['total']:.0f} ms · "
        f"Device: {config.device}"
    )
    download_stem = Path(_display_name(name)).stem or "recognition"
    image_download, json_download = st.columns(2)
    with image_download:
        st.download_button(
            "Download annotated image",
            data=annotated_png,
            file_name=f"{download_stem}.annotated.png",
            mime="image/png",
            key=f"annotated-{request_id}",
            on_click="ignore",
            width="stretch",
        )
    with json_download:
        st.download_button(
            "Download JSON result",
            data=result_json,
            file_name=f"{download_stem}.json",
            mime="application/json",
            key=f"json-{request_id}",
            on_click="ignore",
            width="stretch",
        )
    with st.expander("Model versions"):
        for role, version in result.model_versions.items():
            st.code(f"{role}: {version}")
    LOGGER.info(
        "recognition completed",
        extra={
            "event": "recognition_completed",
            "request_id": request_id,
            "vehicle_count": result.vehicle_count,
            "plate_region_count": len(result.plates),
            "total_inference_ms": round(result.timings_ms["total"], 2),
        },
    )


def _demo_images(image_dir: Path) -> tuple[Path, ...]:
    if not image_dir.is_dir():
        return ()
    return tuple(
        sorted(
            (
                path
                for path in image_dir.iterdir()
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ),
            key=lambda path: path.name.lower(),
        )
    )


def _read_demo(path: Path, max_bytes: int) -> bytes:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ImageValidationError("The selected demo image cannot be inspected") from exc
    if size > max_bytes:
        raise ImageValidationError("The selected demo image exceeds the upload limit")
    try:
        with path.open("rb") as demo_file:
            payload = demo_file.read(max_bytes + 1)
    except OSError as exc:
        raise ImageValidationError("The selected demo image cannot be read") from exc
    if len(payload) > max_bytes:
        raise ImageValidationError("The selected demo image exceeds the upload limit")
    return payload


def main() -> None:
    try:
        config = AppConfig.from_env()
    except NumberPlateRecognitionError:
        LOGGER.exception("Invalid server configuration")
        st.error("The server configuration is invalid. Contact the operator.")
        st.stop()

    configure_logging(config.log_level)

    st.title("Moroccan Number-Plate Recognition")
    st.write(
        "Upload a JPEG or PNG image to detect vehicles, locate their "
        "license plates, and reconstruct visible plate characters."
    )
    st.caption(
        "How it works: vehicle detection → plate localization → character detection "
        "→ left-to-right reconstruction."
    )

    with st.sidebar:
        st.header("Recognition settings")
        st.write(f"Inference device: `{config.device}`")
        st.write(f"Maximum files: {config.max_files}")
        st.write(f"Upload limit: {config.max_upload_bytes // (1024 * 1024)} MiB each")
        st.caption(
            "Confidence thresholds and model locations are controlled by server "
            "environment variables."
        )

        demo_images = _demo_images(config.image_dir)
        selected_demo = st.selectbox(
            "Demo image",
            demo_images,
            index=0 if demo_images else None,
            format_func=lambda path: path.name,
            disabled=not demo_images,
            placeholder="No demo images available",
        )
        run_demo = st.button(
            "Run demo image",
            disabled=selected_demo is None,
            width="stretch",
        )

    with st.form("recognition-upload-form"):
        uploaded_files = st.file_uploader(
            "Upload one or more vehicle images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            max_upload_size=max(1, math.ceil(config.max_upload_bytes / (1024 * 1024))),
            help=(
                f"Up to {config.max_files} files; each file must be no larger than "
                f"{config.max_upload_bytes // (1024 * 1024)} MiB."
            ),
        )
        run_uploads = st.form_submit_button(
            "Run recognition",
            type="primary",
            width="stretch",
        )

    processed_any = False
    if run_demo and selected_demo is not None:
        try:
            payload = _read_demo(selected_demo, config.max_upload_bytes)
        except ImageValidationError as exc:
            LOGGER.warning("Demo image rejected", exc_info=True)
            st.error(str(exc))
        else:
            _render_result(selected_demo.name, payload, config)
            processed_any = True

    if run_uploads:
        if len(uploaded_files) > config.max_files:
            st.error(f"Upload no more than {config.max_files} images at once.")
        else:
            for uploaded_file in uploaded_files:
                _render_result(uploaded_file.name, uploaded_file.getvalue(), config)
                processed_any = True

    if not processed_any:
        st.caption("Choose a demo image or upload an image to begin.")


if __name__ == "__main__":
    main()
