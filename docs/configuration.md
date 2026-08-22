# Configuration

The application reads process environment variables. Copy `.env.example` to `.env` for local overrides; the supplied run scripts load it automatically. Direct `uv run streamlit ...` commands use only the current process environment unless an env file is passed explicitly.

Invalid values fail during configuration loading instead of silently falling back.

## Runtime and paths

| Setting | Default | Description |
| --- | --- | --- |
| `NPR_APP_ROOT` | Repository root | Base directory for relative project paths. |
| `NPR_MODEL_DIR` | `models` | Directory containing the three checkpoints. |
| `NPR_MODEL_MANIFEST` | `models/manifest.json` | Schema-v3 model inventory. |
| `NPR_IMAGE_DIR` | `images` | Directory scanned for local JPEG/PNG examples. |
| `NPR_DEVICE` | `cpu` | Ultralytics device value: `auto`, `cpu`, `mps`, `cuda`, or `cuda:<index>`. |
| `NPR_LOG_LEVEL` | `INFO` | Standard Python logging level. |
| `NPR_VERIFY_MODELS` | `true` | Verify model sizes and SHA-256 values before loading. |

Relative model, manifest, and image paths resolve from `NPR_APP_ROOT`, not from whichever directory happens to launch Streamlit. The supplied scripts change to the repository root before running.

## Inference policy

| Setting | Default | Contract |
| --- | --- | --- |
| `NPR_VEHICLE_CONFIDENCE` | `0.40` | Vehicle confidence threshold in `[0, 1]`. |
| `NPR_PLATE_CONFIDENCE` | `0.35` | Plate confidence threshold in `[0, 1]`. |
| `NPR_CHARACTER_CONFIDENCE` | `0.35` | Character confidence threshold in `[0, 1]`. |
| `NPR_VEHICLE_CLASSES` | `2,3,5,7` | Non-empty comma-separated IDs declared by the vehicle manifest entry. |
| `NPR_CHARACTER_OVERLAP_IOU` | `0.50` | Class-agnostic character overlap-suppression threshold. |
| `NPR_PLATE_DEDUP_IOU` | `0.50` | Absolute-coordinate plate candidate de-duplication threshold. |
| `NPR_PLATE_PATTERN` | `^[0-9]{1,5}[ABEDH][0-9]{1,2}$` | Full-match expression used to classify reconstructed strings. |

Confidence and IoU values must be finite numbers in `[0, 1]`. The plate pattern classifies pipeline output; it does not alter predictions or fill missing characters.

## Input and cascade bounds

| Setting | Default | Contract |
| --- | --- | --- |
| `NPR_INFERENCE_MAX_DIMENSION` | `1024` | Maximum longest side used for inference and preview, without upscaling. |
| `NPR_MAX_UPLOAD_BYTES` | `10485760` | Exact encoded-byte limit per file. |
| `NPR_MAX_IMAGE_PIXELS` | `25000000` | Maximum decoded pixel count. |
| `NPR_MAX_FILES` | `10` | Maximum files per submitted batch. |
| `NPR_MAX_VEHICLES` | `20` | Maximum retained vehicle detections. |
| `NPR_MAX_PLATES_PER_VEHICLE` | `2` | Maximum retained plates per vehicle crop. |
| `NPR_MAX_CHARACTERS_PER_PLATE` | `12` | Maximum retained characters per plate crop. |
| `STREAMLIT_SERVER_MAX_UPLOAD_SIZE` | `10` | Streamlit transport limit in MiB; it must cover `NPR_MAX_UPLOAD_BYTES`. |

All integer limits must be positive. Detector caps are passed to Ultralytics and enforced again on normalized project detections.

## Ultralytics runtime controls

| Setting | Enforced value/location | Purpose |
| --- | --- | --- |
| `YOLO_AUTOINSTALL` | `false` | Prevent inference-time dependency installation. |
| `YOLO_OFFLINE` | `true` | Prevent runtime checkpoint fetching. |
| `YOLO_CONFIG_DIR` | `.runtime/ultralytics` locally; `/tmp/ultralytics` in the container | Isolate writable framework state. |

The run scripts and adapter establish these values. All checkpoint files must exist locally and match the manifest.

## Streamlit and Compose

`.streamlit/config.toml` provides the checked-in server baseline: `0.0.0.0:8501`, headless mode, upload/message limits, XSRF protection, disabled usage telemetry, and disabled file watching.

Common Streamlit overrides follow its environment naming convention:

```bash
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

Compose reads an optional `.env` and publishes `${NPR_PORT:-8501}`. `NPR_PORT` controls only host-port interpolation; the Python application does not read it.

## Model manifest

`models/manifest.json` uses schema version 3 and declares exactly one artifact for each pipeline role. It contains technical identity, integrity, task/class, and output-mapping fields. See [Models and evaluation](models.md) for the full contract.

## Complete local template

```dotenv
NPR_APP_ROOT=.
NPR_MODEL_DIR=models
NPR_MODEL_MANIFEST=models/manifest.json
NPR_IMAGE_DIR=images
NPR_DEVICE=cpu
NPR_VEHICLE_CONFIDENCE=0.40
NPR_PLATE_CONFIDENCE=0.35
NPR_CHARACTER_CONFIDENCE=0.35
NPR_CHARACTER_OVERLAP_IOU=0.50
NPR_PLATE_DEDUP_IOU=0.50
NPR_VEHICLE_CLASSES=2,3,5,7
NPR_PLATE_PATTERN=^[0-9]{1,5}[ABEDH][0-9]{1,2}$
NPR_INFERENCE_MAX_DIMENSION=1024
NPR_MAX_UPLOAD_BYTES=10485760
NPR_MAX_IMAGE_PIXELS=25000000
NPR_MAX_FILES=10
NPR_MAX_VEHICLES=20
NPR_MAX_PLATES_PER_VEHICLE=2
NPR_MAX_CHARACTERS_PER_PLATE=12
STREAMLIT_SERVER_MAX_UPLOAD_SIZE=10
NPR_LOG_LEVEL=INFO
NPR_VERIFY_MODELS=true
YOLO_AUTOINSTALL=false
YOLO_OFFLINE=true
```

Update this guide and `.env.example` together when the configuration interface changes.
