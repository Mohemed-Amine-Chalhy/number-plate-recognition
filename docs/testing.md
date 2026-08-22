# Testing

The test strategy keeps deterministic application logic fast while reserving expensive checkpoint loading for an explicit smoke tier.

## Local commands

Run the supported default gate:

```bash
uv run python scripts/quality.py check
```

It checks Ruff formatting/linting, strict mypy, the schema-v3 model manifest, and the default pytest suite. Pytest excludes the `model` marker by default and enforces at least 85% line and branch coverage for project code.

Individual commands:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest
uv run pre-commit run --all-files
```

Run the real-checkpoint CPU smoke after changing a model, inference dependency, adapter, or pipeline boundary:

```bash
uv run pytest tests/model/test_real_inference.py -m model --no-cov
```

## Test matrix

| Layer | Uses real weights | What it verifies |
| --- | --- | --- |
| Unit | No | Geometry, decoding, configuration, model-manifest parsing, filtering, post-processing, evaluation, and error behavior. |
| Adapter contract | No | Normalization of NumPy/torch-like Ultralytics results, call arguments, semantic checks, and allowlist filtering. |
| Repository integration | No runtime load | The checked-in schema-v3 manifest and all checkpoint sizes/SHA-256 values. |
| CLI | No | Batch reuse, deterministic JSON, optional PNG output, bounded input, partial failures, and sanitized errors. |
| Streamlit AppTest | No | Startup, explicit submission, upload validation, pipeline integration, rendering, and safe errors. |
| Real-model smoke | Yes | Deserialization, task/class compatibility, all three model APIs, and the complete cascade. |
| Dataset evaluation | Yes | End-to-end string metrics for a supplied labeled dataset. |

## Fast tests

Unit tests use generated arrays and fake detectors. Coverage includes:

- box clipping, translation, intersection, area, and IoU;
- JPEG/PNG decoding, EXIF orientation, byte/pixel limits, corrupt input, color conversion, and longest-side resizing;
- every environment setting and invalid configuration boundary;
- detection normalization, finite/range/class/confidence filtering, crop validation, cascade limits, plate de-duplication, and character overlap suppression;
- deterministic character ordering, output mapping, pattern classification, and confidence aggregation;
- schema-v3 parsing, safe paths, hash mismatch, required roles, task/classes, and output-map consistency;
- one model-bundle initialization per CLI batch, deterministic output, annotated-image writes, and per-file failure isolation;
- exact-match and character-similarity evaluation behavior;
- structured, bounded error/log output.

Adapter tests use fake NumPy and torch-like result objects to verify third-party shapes without importing real checkpoints. Extra classes may exist in a loaded model, but predictions outside the manifest allowlist cannot enter project detections.

## Streamlit AppTest

`tests/smoke/test_streamlit_app.py` runs the page with a fake pipeline. It verifies startup without model loading, explicit form submission, image/result rendering, corrupt and oversized input handling, filename rendering, backend error sanitization, and manifest-version display behavior.

Keeping AppTest above the core pipeline boundary makes UI checks fast and deterministic while pipeline behavior remains covered by unit/integration tests.

## Real-model smoke

The `model`-marked test verifies all three artifact hashes and loads each checkpoint on CPU with offline/no-auto-install behavior. It invokes every detector on a deterministic blank frame, then runs the complete pipeline over the three tracked demo images.

It asserts the exact reconstructed strings `90120A72`, `1678E1`, and `45296B6`. This establishes a narrow, reproducible end-to-end regression path; three examples are not a statistically meaningful quality benchmark.

## Dataset evaluation

Run the evaluator on a labeled set:

```bash
uv run python scripts/evaluate.py path/to/ground-truth.json
```

The evaluator reports exact-match precision/recall/F1, exact sample rate, and normalized character similarity. See [Models and evaluation](models.md) for the expected JSON shape and useful future metrics.

The tracked `images/Car1.jpg` through `images/Car3.jpg` files are demonstration inputs, not a statistically meaningful benchmark.

## CI and hooks

The GitHub Actions workflow provides:

- Ubuntu lock, manifest, formatting, lint, strict type, test/coverage, package-build, and pre-commit checks;
- Windows/Python 3.12 diagnostics and fast tests;
- real-model CPU smoke on non-pull-request runs;
- container build, startup, and liveness after the real-model job.

Pre-commit checks syntax/structured files, conflict markers, case collisions, private keys, large new files, whitespace/line endings, notebook output, secrets, and the model manifest. Pre-push runs the complete quality command.

## Determinism

- Pin Python dependencies with `uv.lock`.
- Pin checkpoints by byte size and SHA-256.
- Validate loaded task and required class subsets.
- Use generated arrays/fakes for normal tests and fixed in-memory input for the model smoke.
- Keep structured expectations separate from rendered-image goldens.
- Review behavioral differences before updating expected results.
