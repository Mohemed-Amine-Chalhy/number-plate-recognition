# Testing and quality gates

## Local commands

Run the supported default gate:

```bash
uv run python scripts/quality.py check
```

It checks Ruff formatting/linting, strict mypy, manifest schema v2, and the default pytest suite. `pyproject.toml` configures pytest with `-m "not model"`, branch coverage, and an 85% threshold, so normal code checks do not load the real weights.

Apply configured safe Ruff fixes/formatting, then review the diff:

```bash
uv run python scripts/quality.py fix
```

Individual commands:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest
uv run pre-commit run --all-files
```

Run the real-model CPU smoke explicitly after changing a weight, manifest semantic contract, inference dependency, adapter, or pipeline boundary:

```bash
uv run pytest tests/model/test_real_inference.py -m model --no-cov
```

Use `--help`/`--markers` as the command-line authority:

```bash
uv run python scripts/quality.py --help
uv run pytest --markers
```

## Current production gate

```bash
uv run python scripts/doctor.py --production
```

This currently fails by design for all three roles. Production requires `provenance_status=verified`, non-null `source`, `license_status=approved`, and `production_approved=true`, in addition to valid schema, size, and SHA-256. Setting `NPR_ENVIRONMENT=production` activates the same metadata policy automatically. Representative quality acceptance remains separate human release evidence.

## Test layers

### Unit

Unit tests use synthetic arrays and fake detectors; they do not load production weights. Current coverage includes:

- box clipping, empty intersection, fractional-edge floor/ceil conversion, immutable translation, area, and IoU;
- JPEG/PNG decoding, encoded-byte/pixel guards, invalid input, BGR conversion, and longest-side downscaling for wide/tall images;
- every environment setting, explicit/auto application-root path behavior, direct-construction validation, production verification policy, regex syntax, and Streamlit transport-limit consistency;
- invalid/non-finite detections, class/confidence filtering, coordinate translation, empty crops, longest-side resize, cascade caps, character overlap suppression, plate deduplication across overlapping vehicles, deterministic ordering, mapping, confidence, and the one-letter default grammar;
- schema-v2 parsing, path traversal, hash mismatch, explicit production fields, output-map/class-contract consistency, and optional metadata validation;
- exact-match/character-similarity evaluation, duplicate/surplus predictions, empty inputs, and optimal order-independent character matching;
- privacy-safe structured logging.

The regex tests establish application behavior only. A passing pattern is a configurable review heuristic, not regulatory validation.

### Adapter contract

Fake NumPy and torch-like results verify third-party output normalization, detector arguments (`classes`, `max_det`, class-agnostic NMS), fractional/mismatched output rejection, safe error wrapping, verified versus `@unverified` version identifiers, required task presence/match, exact class-label subset checks, and filtering of predictions outside the manifest allowlist. Extra loaded classes may exist, but cannot enter project detections.

### Repository integration

The integration test parses the live schema-v2 manifest and verifies all three repository artifacts by size and SHA-256. It does not establish provenance or quality.

### Streamlit AppTest

`tests/smoke/test_streamlit_app.py` uses Streamlit `AppTest` with a fake pipeline. It currently verifies:

- startup without loading model artifacts;
- explicit form submission of a valid upload, one pipeline call, two rendered images, supported prediction, and results table;
- corrupt image rejection before inference;
- sanitization of model/backend failures so sensitive details are not displayed;
- plain-text rendering of a crafted upload filename and an explicit `@unverified` model identifier;
- rejection of an oversized submitted file batch before inference.

The upload flow is intentionally submit-driven: changing file selection does not trigger inference. The live uploader also receives a per-widget MiB cap calculated from `NPR_MAX_UPLOAD_BYTES`; exact bytes/pixels are enforced by the decoder. Operator-provisioned Approved-example submission and the browser/framework's own transport rejection are not currently direct AppTest assertions.

### Real-model smoke

The `model`-marked test verifies all three artifacts and loads the actual models on CPU with offline/no-auto-install settings. It directly sends one deterministic blank in-memory frame through each of the vehicle, plate, and character adapters so every model API is exercised even when the first cascade stage is empty, then runs the complete bounded pipeline. It requires no vehicle false positive, an empty pipeline result, valid output shape/timing, and all three version keys.

This proves current artifact integrity, deserialization, task/class compatibility, and invocation compatibility only. A blank synthetic frame cannot measure recognition accuracy, domain performance, calibration, or fairness; those require an approved external labeled dataset.

### Model regression

On a versioned, representative, legally usable holdout, measure:

- vehicle/plate precision, recall, and false-positive/negative rates;
- per-character accuracy/confusions and confidence calibration;
- exact end-to-end plate match;
- lighting, distance, angle, blur, occlusion, plate style, camera/domain, no-plate, and non-Moroccan slices;
- CPU and target-GPU p50/p95/p99 queue, stage, and total latency, throughput, peak memory, and failure rate;
- sensitivity to confidence, cascade, overlap/dedup IoU, resize, and pattern policy.

Choose thresholds/regex policy without the final holdout. Pattern-match rate is not regulatory validity or accuracy.

The evaluator expects a non-empty `samples` array with unique IDs, safe image paths relative to the manifest directory (or `--image-root`), and non-empty exact expected strings:

```json
{
  "samples": [
    {"id": "daylight-001", "image": "daylight-001.jpg", "expected": ["123A45"]}
  ]
}
```

Run only on an authorized, versioned dataset:

```bash
uv run python scripts/evaluate.py path/to/ground-truth.json
```

The evaluator reports exact-match precision/recall/F1, exact sample rate, and mean normalized character similarity. It does not yet calculate detector mAP, calibration, slices, confidence intervals, or latency percentiles; those remain production evidence.

## CI and pre-commit

The workflow currently provides:

- Ubuntu quality: lock check, default development sync, schema v2, Ruff, strict mypy, non-model pytest/coverage, and all commit-stage hooks;
- dependency audit: `pip-audit`, advisory on pull requests and blocking on protected-branch/manual runs;
- Windows/Python 3.12 compatibility: doctor plus fast non-model tests;
- real-model CPU smoke on non-pull-request runs;
- container build/start/liveness after the real-model job, then SPDX SBOM generation and a blocking scan for fixable high-or-worse vulnerabilities.

The container probe checks Streamlit liveness. The entrypoint already preflights metadata/integrity, but semantic model initialization still occurs at first inference; model-aware readiness/warm-up remains a deployment concern.

Pre-commit additionally checks syntax/structured files, conflicts/case issues, private keys, large new files, whitespace/line endings, strips notebook output, scans committed secrets, validates the manifest, and runs type/tests at pre-push. Existing manifest-pinned weights have a narrow large-file exception; new binaries remain blocked.

## Coverage and determinism

The configured threshold is at least 85% line and branch coverage for project code. Coverage does not replace model evaluation.

- Pin dependencies with `uv.lock`.
- Pin artifacts by byte size and SHA-256.
- Validate loaded task and required class subsets.
- Seed supported randomness and use documented float tolerances.
- Keep structured expected results separate from rendered-image goldens.
- Review semantic differences before updating any expected output.

## Test data rules

Use synthetic, explicitly consented, public-domain, or otherwise authorized fixtures. Mask real plate values unless exact values are necessary and approved. Never place production uploads in bug reports/tests. The repository's `images/` directory is for ignored local Approved examples, not the evaluation corpus. Keep representative labeled data in a separate controlled location and record source, rights/authority, transformations, retention, and deletion for every sample/evaluation dataset.
