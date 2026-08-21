# Configuration

The application reads process environment variables. `.env.example` is the authoritative template; the Python process does not parse `.env` itself. The supplied `run_app` scripts pass the repository `.env` to `uv run --env-file` when it exists, Compose reads it when present, and `docker run` can receive it with `--env-file .env`. Existing process environment settings remain available.

For a local shell override:

```powershell
$env:NPR_DEVICE = "cpu"
.\scripts\run_app.ps1
```

```bash
export NPR_DEVICE=cpu
bash scripts/run_app.sh
```

Production values should come from the deployment platform. Do not commit `.env` or credentials. Run `uv run python scripts/doctor.py` to validate configuration, dependencies, paths, device availability, manifest metadata, and local artifact integrity.

## Runtime and paths

| Setting | Default | Contract |
| --- | --- | --- |
| `NPR_APP_ROOT` | Auto-discovered; `.` in `.env.example` | Base for relative model, manifest, and image paths. An explicit relative value resolves from the process working directory. Without it, the app searches the working directory and package project root for `pyproject.toml` plus `models/`. The container fixes it to `/opt/app`. |
| `NPR_ENVIRONMENT` | `development` | One of `development`, `test`, or `production`. Production forces model verification and requires every production-role artifact to have approved machine-readable metadata. |
| `NPR_MODEL_DIR` | `models` | Artifact directory relative to `NPR_APP_ROOT`, or an absolute path. |
| `NPR_MODEL_MANIFEST` | `<NPR_MODEL_DIR>/manifest.json` | Schema-v2 manifest path relative to `NPR_APP_ROOT`, or absolute. |
| `NPR_IMAGE_DIR` | `images` | Operator-provisioned **Approved examples** directory relative to `NPR_APP_ROOT`, or absolute. The repository ships only [its policy file](../images/README.md), not photographs. |
| `NPR_DEVICE` | `cpu` | `auto`, `cpu`, `mps`, `cuda`, or `cuda:<index>`. CPU is the reproducible baseline. |
| `NPR_LOG_LEVEL` | `INFO` | `CRITICAL`, `ERROR`, `WARNING`, `INFO`, or `DEBUG`. User image and plate payloads are not application log fields. |
| `NPR_VERIFY_MODELS` | `true` | Verifies byte size and SHA-256 before loading. It may be disabled only in development/test and should normally remain enabled; production rejects `false`. When disabled, displayed model versions use `@unverified` instead of claiming a digest was checked. |

Relative application paths resolve against `NPR_APP_ROOT`, not whichever directory happens to contain a model file. The launch scripts change to the repository root and the container pins `/opt/app`, making the normal entry points independent of the caller's working directory.

## Inference policy

| Setting | Default | Contract |
| --- | --- | --- |
| `NPR_VEHICLE_CONFIDENCE` | `0.40` | Vehicle-stage threshold in `[0, 1]`. |
| `NPR_PLATE_CONFIDENCE` | `0.35` | Plate-stage threshold in `[0, 1]`. |
| `NPR_CHARACTER_CONFIDENCE` | `0.35` | Character-stage threshold in `[0, 1]`. |
| `NPR_VEHICLE_CLASSES` | `2,3,5,7` | Non-empty comma-separated non-negative class IDs. Every configured ID must be present in the vehicle artifact's `expected_classes` contract. |
| `NPR_CHARACTER_OVERLAP_IOU` | `0.50` | Class-agnostic character overlap-suppression threshold in `[0, 1]`; higher-confidence detections win when overlap exceeds it. |
| `NPR_PLATE_DEDUP_IOU` | `0.50` | Deduplicates overlapping absolute plate candidates, including the same plate found through overlapping vehicle boxes, before character inference. |
| `NPR_PLATE_PATTERN` | `^[0-9]{1,5}[ABEDH][0-9]{1,2}$` | Full-match regular expression used to label reconstructed text as “Supported” or “Review required.” The default is a configurable one-letter heuristic, not regulatory validation and not proof that a plate is genuine or correctly recognized. |

Thresholds and the pattern require evaluation on authorized, representative data. A pattern match must never turn an uncertain prediction into an authoritative identity.

## Input and cascade bounds

| Setting | Default | Contract |
| --- | --- | --- |
| `NPR_INFERENCE_MAX_DIMENSION` | `1024` | Bounds the longest image side while preserving aspect ratio and never upscaling. The UI original-image preview uses the same bound. |
| `NPR_MAX_UPLOAD_BYTES` | `10485760` | Exact per-file encoded-byte limit enforced during decode and for operator-provisioned Approved examples. It cannot exceed the Streamlit transport cap. |
| `NPR_MAX_IMAGE_PIXELS` | `25000000` | Decoded pixel-count/decompression-bomb guard. |
| `NPR_MAX_FILES` | `10` | Maximum files accepted by one submitted upload batch. An oversized batch is rejected before inference. |
| `NPR_MAX_VEHICLES` | `20` | Maximum retained vehicle detections and detector `max_det` for the first stage. |
| `NPR_MAX_PLATES_PER_VEHICLE` | `2` | Maximum retained plate detections and detector `max_det` per vehicle crop. |
| `NPR_MAX_CHARACTERS_PER_PLATE` | `12` | Maximum retained character detections and detector `max_det` per plate crop. |
| `STREAMLIT_SERVER_MAX_UPLOAD_SIZE` | `10` | Streamlit per-widget transport limit in MiB. It must be at least `ceil(NPR_MAX_UPLOAD_BYTES / 1 MiB)`. The uploader also receives the computed per-widget cap. |

All integer limits must be positive. Confidence and IoU values must be finite numbers in `[0, 1]`.

## Ultralytics runtime controls

| Setting | Enforced value/location | Contract |
| --- | --- | --- |
| `YOLO_AUTOINSTALL` | `false` | The adapter and run scripts disable Ultralytics dependency auto-install behavior. |
| `YOLO_OFFLINE` | `true` | The adapter and run scripts require offline model operation. Runtime inference never fetches a missing weight. |
| `YOLO_CONFIG_DIR` | `.runtime/ultralytics` locally; `/tmp/ultralytics` in the container | The run scripts isolate writable Ultralytics state under an ignored runtime directory; the container uses its writable temporary filesystem. |

`YOLO_AUTOINSTALL=false` and `YOLO_OFFLINE=true` are fixed safety policy, not supported opt-outs. Artifacts must be provisioned separately and pass manifest size/hash checks. `ULTRALYTICS_SAFE_LOAD` is not enabled because compatibility with the current artifacts has not been validated; treat pickle-backed `.pt` loading as a residual risk and do not toggle it without compatibility and accuracy testing.

## Streamlit server

`.streamlit/config.toml` supplies the checked-in server baseline: address `0.0.0.0`, port `8501`, 10 MiB upload limit, 25 MiB message limit, headless mode, XSRF protection, CORS, disabled usage telemetry, and disabled file watching/run-on-save.

Streamlit supports environment mappings such as:

```bash
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

The application validates only the transport relationship exposed by `STREAMLIT_SERVER_MAX_UPLOAD_SIZE`; test any other override with the ingress and UI. Do not place credentials in client-visible Streamlit state.

`compose.yaml` optionally reads `.env`, publishes `${NPR_PORT:-8501}`, and then pins container filesystem paths plus `NPR_ENVIRONMENT`. Explicit Compose `environment` entries take precedence over values from `env_file`.

`NPR_PORT` is Compose-only host-port interpolation and is not read by the Python application. The run scripts are the preferred local entry point because they load `.env` if present and establish the offline runtime directory/policy. A direct `uv run streamlit ...` invocation sees only its process environment unless `uv --env-file` is supplied.

## Model schema and production policy

`models/manifest.json` is schema version 2 and contains exactly the current `vehicle`, `plate`, and `character` artifacts. It records integrity, semantic contracts, provenance/licensing state, and explicit production approval. See [models](models.md) for every field.

In `production`:

- `NPR_VERIFY_MODELS` must be true;
- all three artifacts must have `provenance_status: "verified"`;
- each must have a non-null authoritative `source`;
- each must have `license_status: "approved"`;
- each must have `production_approved: true`;
- sizes, SHA-256 hashes, task, and required class-name subsets must still match.

The current manifest intentionally does not meet those external-evidence fields, so production-mode diagnostics/container startup fail closed. Development-mode, checksum-verified local operation remains supported.

## Template parity

The complete current template is:

```dotenv
NPR_APP_ROOT=.
NPR_ENVIRONMENT=development
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

Update this guide and `.env.example` together whenever the configuration interface changes.
