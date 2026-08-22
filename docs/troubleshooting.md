# Troubleshooting

Start from the repository root with:

```bash
uv run python scripts/doctor.py
```

Use `uv run` for project commands so they execute in the locked environment.

## Wrong Python version

The supported runtime is Python 3.12:

```bash
uv python find 3.12
uv run python --version
```

Install/select it and resynchronize if necessary:

```bash
uv python install 3.12
uv sync --locked
```

## `uv` is not found

Install `uv` using its official instructions, restart the shell so the install directory is on `PATH`, then rerun the bootstrap script. A global `pip` environment will not reproduce `uv.lock`.

## Lockfile or dependency mismatch

```bash
uv lock --check
uv sync --locked
```

If `pyproject.toml` intentionally changed, update `uv.lock` in the same change and run the complete quality gate.

## Missing or invalid models

```bash
uv run python scripts/doctor.py --models-only
```

The schema-v3 manifest must contain one `vehicle`, one `plate`, and one `character` role. Check each declared filename, byte size, SHA-256, and path relative to `NPR_APP_ROOT`/`NPR_MODEL_DIR`.

If size/hash verification passes but the first request fails, run the real-model smoke to check deserialization and task/class compatibility:

```bash
uv run pytest tests/model/test_real_inference.py -m model --no-cov
```

Extra loaded classes are allowed, but the adapter filters predictions to manifest-declared IDs. A missing declared class, changed label, or mismatched task is an integrity error.

If a model version ends in `@unverified`, `NPR_VERIFY_MODELS=false` disabled byte/hash checks. Restore verification to get a manifest-digest identifier.

## App starts only from one directory

Use the supplied run script or launch from the repository root:

```bash
uv run streamlit run app/streamlit_app.py
```

Relative model, manifest, and image paths resolve against `NPR_APP_ROOT`. Set an explicit absolute application root for a relocated installation.

## Streamlit page does not load

The default URL is <http://localhost:8501>. For a container:

```bash
docker ps
docker logs <container-name-or-id>
```

Check port publication, firewall rules, proxy websocket support, and the entrypoint model check. `/_stcore/health` proves process liveness; checkpoints still load on first inference unless a warm-up is configured.

## Upload is rejected

Inference starts only after **Run recognition**. Check:

- `STREAMLIT_SERVER_MAX_UPLOAD_SIZE` for the per-widget transport limit;
- `NPR_MAX_UPLOAD_BYTES` for exact bytes per file;
- `NPR_MAX_FILES` for batch count;
- `NPR_MAX_IMAGE_PIXELS` for decoded image size;
- any reverse-proxy body limit.

Restart Streamlit after changing environment variables.

## Demo images do not appear

Confirm that `NPR_IMAGE_DIR` points to the repository `images/` directory and contains `Car1.jpg`, `Car2.jpg`, and `Car3.jpg`. Supported extensions are `.jpg`, `.jpeg`, and `.png`.

## `libGL` or GUI-library error

The project uses headless OpenCV. Confirm that a GUI-enabled OpenCV wheel was not installed alongside it, then resynchronize from `uv.lock` or rebuild the container.

## CUDA is unavailable

CPU is the default. If GPU inference is intentional, verify the driver, container runtime, CUDA runtime, PyTorch build, and device visibility:

```bash
nvidia-smi
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Use a pinned GPU profile instead of installing an unrelated PyTorch wheel into the locked environment.

## Out of memory or slow inference

Inspect uploaded dimensions, `NPR_INFERENCE_MAX_DIMENSION`, cascade maxima, queue time, process count, and model memory. One process serializes complete requests, so total time includes lock wait. Use profiling before reducing image/detail limits or adding workers.

## Corrupt image crashes the app

This is a bug: invalid input should produce a bounded message while the process continues serving. Capture the exception category and a minimal reproducer.

## No detections or poor recognition

- Verify manifest hashes and loaded task/class contracts.
- Check all three confidence thresholds and `NPR_VEHICLE_CLASSES`.
- Inspect vehicle, plate, and character cascade limits.
- Review `NPR_PLATE_DEDUP_IOU` and `NPR_CHARACTER_OVERLAP_IOU` for aggressive suppression.
- Inspect annotated boxes and structured per-stage results.
- Reproduce the case through `scripts/evaluate.py` before tuning thresholds.

The plate pattern classifies reconstructed text only; it does not change detector output.

## Pre-commit differs from CI

```bash
uv sync --locked
uv run pre-commit clean
uv run pre-commit install --install-hooks
uv run pre-commit run --all-files
uv run python scripts/quality.py check
```

Confirm the same source revision, Python 3.12, `uv.lock`, and hook configuration. CI is authoritative.

## Getting help

For ordinary bugs, include the platform, Python version, source revision, doctor output, exact command, and a minimized reproduction. Report security issues through [SECURITY.md](../SECURITY.md).
