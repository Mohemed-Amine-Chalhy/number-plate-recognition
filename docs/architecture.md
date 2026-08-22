# Architecture

## Scope

The production target is a stateless Moroccan number-plate recognition web application. It accepts image inputs and returns annotated images plus structured detection results. It is not a physical gate controller and contains no authorization rules, actuator integration, safety interlocks, or vehicle registry.

If gate actuation is ever added, implement it as a separately reviewed service with authenticated commands, explicit policy, tamper-resistant audit records, hardware feedback, fail-safe behavior, manual override, and a safety assessment. Recognition output alone must never actuate a barrier.

## Components

```text
Browser
  |
  v
Streamlit adapter (app/streamlit_app.py)
  |  decodes/validates input and renders output
  v
Application pipeline (src/number_plate_recognition/pipeline.py)
  |-- vehicle detector adapter
  |-- plate detector adapter
  |-- character detector adapter
  |-- geometry and image helpers
  `-- Moroccan plate post-processing
        |
        v
Typed InferenceResult
  |-- detections and confidences
  |-- reconstructed text + pattern-review flag
  |-- queue, stage, and total timings
  `-- model identifiers/checksums
```

The UI is deliberately thin. Core inference must be callable without Streamlit so it can be unit-tested, batch-tested, or exposed through another adapter later.

## Package boundaries

- `app/streamlit_app.py` owns upload controls, session presentation, and user-safe error messages.
- `src/number_plate_recognition/config.py` loads and validates environment-based configuration.
- `src/number_plate_recognition/domain.py` defines typed boxes, detections, plate results, and inference results.
- `src/number_plate_recognition/imaging.py` validates JPEG/PNG payloads, corrects EXIF orientation, converts color space, and bounds the longest image side without upscaling.
- `src/number_plate_recognition/pipeline.py` serializes complete requests, enforces cascade limits, filters/clips detections, deduplicates overlaps, and orchestrates the three stages without UI dependencies.
- `src/number_plate_recognition/postprocessing.py` orders characters, applies the manifest output map, and full-matches the configured review heuristic.
- `src/number_plate_recognition/model_registry.py` parses schema v2, verifies artifact integrity, and enforces machine-readable production approval.
- `src/number_plate_recognition/adapters/ultralytics.py` isolates third-party result shapes/calls, requires loaded task/class subset matches, and filters predictions to manifest-declared IDs.
- `src/number_plate_recognition/evaluation.py` provides deterministic end-to-end exact-match and character-similarity metrics.
- `src/number_plate_recognition/observability.py` emits structured logs without recognition payloads by default.
- `src/number_plate_recognition/errors.py` defines user-safe project error categories at external boundaries.

Exact modules may be consolidated while the code remains small, but the direction of dependencies must remain UI → application/domain → adapters. Domain and post-processing code must not import Streamlit or Ultralytics.

## Request lifecycle

1. Uploaded files live inside an explicit Streamlit form. Selecting files does not run inference; the user submits with **Run recognition**. If an operator has provisioned authorized files under `NPR_IMAGE_DIR`, the sidebar labels them **Approved examples** and runs the selected example through a separate explicit button. No photographs ship in the repository.
2. Streamlit receives a per-widget upload cap equal to `ceil(NPR_MAX_UPLOAD_BYTES / 1 MiB)`. Configuration rejects an exact byte limit above `STREAMLIT_SERVER_MAX_UPLOAD_SIZE`; a submitted batch above `NPR_MAX_FILES` is rejected before inference.
3. The imaging layer enforces exact encoded bytes and decoded pixels, rejects unsupported/corrupt/decompression-bomb input, normalizes EXIF orientation, converts to BGR, and never trusts the filename/MIME alone. The UI reduces an upload name to a bounded, sanitized basename and renders it as plain text. Approved examples receive a bounded pre-read and the same decoder checks.
4. The request waits for the model bundle's inference lock. `queue` timing measures this wait; `total` begins before it and therefore includes queueing.
5. Once admitted, the longest image side is bounded by `NPR_INFERENCE_MAX_DIMENSION`, preserving aspect ratio and never upscaling. The original-image UI preview uses the same resize helper.
6. The vehicle detector receives the configured class IDs and `NPR_MAX_VEHICLES`. The adapter intersects caller-supplied IDs with its manifest allowlist; stages without a caller subset default to the complete role allowlist. It also filters returned IDs and assigns trusted manifest labels. Non-finite, out-of-range, low-confidence, unsupported-class, and empty-after-clamp detections are discarded; retained results are confidence-ranked and capped.
7. The plate detector runs in each vehicle crop with `NPR_MAX_PLATES_PER_VEHICLE`. Boxes are translated/clipped to absolute coordinates, then overlapping candidates are confidence-ranked and deduplicated using `NPR_PLATE_DEDUP_IOU` before character inference.
8. The character detector receives `NPR_MAX_CHARACTERS_PER_PLATE` with class-agnostic NMS. Project-level class-agnostic overlap suppression applies `NPR_CHARACTER_OVERLAP_IOU`, and surviving detections are capped and translated.
9. Characters are sorted left-to-right and mapped using the character artifact's schema-v2 `output_map`. The text is full-matched against `NPR_PLATE_PATTERN` to classify it as **Supported** or **Review required**. This is a configurable one-letter heuristic, never regulatory validation or proof of correctness.
10. The pipeline returns an annotated RGB image, plate/character confidences and boxes, vehicle count, model version identifiers, and `queue`/`vehicle`/`plate`/`character`/`total` milliseconds. A verified load reports a short manifest SHA-256; a development load with `NPR_VERIFY_MODELS=false` is explicitly labeled `@unverified`. The UI shows bounded previews, a results table, review messages, total time, device, and model versions.

Empty detections remain successful results. No post-processing step invents a missing character to satisfy the regex.

## State and concurrency

The service has no intended durable application state. Streamlit maintains browser-session/widget state and upload buffers for its framework lifecycle, but project code does not write user images or results to a database or disk. Operators can separately provision local Approved examples; the repository ignores them and the application only reads them. Model objects are cached in-process because loading weights is expensive; user payloads/results are not added to that global cache.

A bundle-wide re-entrant lock serializes the **complete three-stage request**, preventing cross-request model interleaving. Each detector adapter also protects its concrete model call. `queue_ms` makes contention visible instead of blending it into detector latency. Load-test this one-request-per-process boundary; scale with isolated processes/replicas if the serialized design misses throughput objectives.

Multiple replicas can run behind a load balancer because there is no required shared durable state; verify Streamlit session affinity and websocket behavior at the ingress. The current entrypoint verifies artifacts/policy before startup, while `/_stcore/health` is process liveness and model objects still load on first inference. A deployment that requires semantic model-aware readiness must add a warm-up/readiness probe before admitting traffic.

## Failure model

- Invalid user input is a normal request error and must not terminate the process.
- The container entrypoint rejects missing/corrupt artifacts before Streamlit. In `production`, it also rejects absent provenance/source/license/approval metadata. Loaded task/class semantic mismatches are rejected when models initialize.
- Core loading and launch/deployment configuration force Ultralytics offline mode and disable auto-install. Missing artifacts fail instead of triggering a runtime download.
- CUDA out-of-memory, dependency errors, and unexpected inference exceptions are logged with a request ID and surfaced as generic user-safe errors.
- Empty detections are successful results, not exceptions.
- Downstream stages are never called with empty or out-of-bounds crops.

## Trust boundaries

The main boundaries are the untrusted uploaded file, third-party parsers and ML runtimes, model artifacts, runtime configuration, and any reverse proxy or identity provider. Controls and residual risks are documented in [privacy and security](privacy-security.md).

## Architecture decisions to record

Create an ADR when any of these choices is finalized:

- Ultralytics AGPL compliance, Enterprise licensing, or replacement.
- Whether weights ship in the image or are fetched from an artifact store.
- CPU-only versus a supported CUDA matrix.
- Retention and redaction policy for production telemetry.
- Any API adapter, persistence, access-control policy, or physical-device integration.
- Any change from complete-request serialization to parallel model access.
