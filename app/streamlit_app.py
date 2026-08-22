"""Streamlit user interface for the production recognition package."""

from __future__ import annotations

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
            "Format": "Supported" if plate.format_valid else "Review required",
        }
        for plate in result.plates
    ]


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
        st.image(original_rgb, caption="Original image", use_container_width=True)
    with result_column:
        st.image(
            result.annotated_image_rgb,
            caption="Recognition result",
            use_container_width=True,
        )

    rows = _result_rows(result)
    supported = list(
        dict.fromkeys(plate.text for plate in result.plates if plate.text and plate.format_valid)
    )
    review_required = list(
        dict.fromkeys(
            plate.text for plate in result.plates if plate.text and not plate.format_valid
        )
    )
    if supported:
        st.success(f"Supported plate prediction(s): {', '.join(supported)}")
    if review_required:
        st.warning(
            "Prediction(s) require review because they do not match the configured "
            f"plate pattern: {', '.join(review_required)}"
        )
    elif result.plates and not supported:
        st.warning("Plate regions were detected, but their text could not be reconstructed.")
    elif not result.plates:
        st.warning("No supported license plate was detected.")
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)

    st.caption(
        f"Vehicles: {result.vehicle_count} · "
        f"Total inference: {result.timings_ms['total']:.0f} ms · "
        f"Device: {config.device}"
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


def _sample_images(image_dir: Path) -> tuple[Path, ...]:
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


def _read_sample(path: Path, max_bytes: int) -> bytes:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ImageValidationError("The selected example cannot be inspected") from exc
    if size > max_bytes:
        raise ImageValidationError("The selected example exceeds the upload limit")
    try:
        with path.open("rb") as sample_file:
            payload = sample_file.read(max_bytes + 1)
    except OSError as exc:
        raise ImageValidationError("The selected example cannot be read") from exc
    if len(payload) > max_bytes:
        raise ImageValidationError("The selected example exceeds the upload limit")
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
        "Upload a JPEG or PNG image to detect supported vehicles, locate their "
        "license plates, and reconstruct visible plate characters."
    )
    st.info(
        "This application does not intentionally persist uploads; deployment "
        "infrastructure may have its own retention policy. Do not upload images "
        "unless you are authorized to process their vehicle registration data."
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

        samples = _sample_images(config.image_dir)
        selected_sample = st.selectbox(
            "Approved example",
            samples,
            format_func=lambda path: path.name,
            disabled=not samples,
            placeholder="No approved examples available",
        )
        run_sample = st.button(
            "Run selected example",
            disabled=selected_sample is None,
            use_container_width=True,
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
            use_container_width=True,
        )

    processed_any = False
    if run_sample and selected_sample is not None:
        try:
            payload = _read_sample(selected_sample, config.max_upload_bytes)
        except ImageValidationError as exc:
            LOGGER.warning("Approved sample rejected", exc_info=True)
            st.error(str(exc))
        else:
            _render_result(f"Example: {selected_sample.name}", payload, config)
            processed_any = True

    if run_uploads:
        if len(uploaded_files) > config.max_files:
            st.error(f"Upload no more than {config.max_files} images at once.")
        else:
            for uploaded_file in uploaded_files:
                _render_result(uploaded_file.name, uploaded_file.getvalue(), config)
                processed_any = True

    if not processed_any:
        st.caption("Choose an approved example or upload an image to begin.")


if __name__ == "__main__":
    main()
