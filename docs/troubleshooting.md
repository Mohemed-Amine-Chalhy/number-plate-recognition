# Troubleshooting

Start with the environment diagnostic from the repository root:

```bash
uv run python scripts/doctor.py
```

Use `uv run` for project commands so they execute in the locked environment.

## Wrong Python version

The supported version is Python 3.12. Confirm what `uv` selected:

```bash
uv python find 3.12
uv run python --version
```

Install/select it and resynchronize:

```bash
uv python install 3.12
uv sync --locked
```

## `uv` is not found

Install `uv` using the official instructions, restart the shell so its install directory is on `PATH`, then run the bootstrap script again. Avoid falling back to an unrelated global `pip` environment because it will not match `uv.lock`.

## Lockfile or dependency mismatch

For a normal checkout:

```bash
uv lock --check
uv sync --locked
```

Do not regenerate the lockfile merely to suppress an error. If `pyproject.toml` intentionally changed, update the lockfile in the same reviewed change and run the complete quality gate.

## Missing or invalid models

```bash
uv run python scripts/doctor.py --models-only
uv run python scripts/fetch_models.py --verify-only
```

The schema-v2 manifest must contain exactly one each of the `vehicle`, `plate`, and `character` roles. Check the declared filename, positive `size_bytes`, and SHA-256, and remember that paths resolve from `NPR_APP_ROOT`/`NPR_MODEL_DIR`. If a reviewed entry later supplies an absolute HTTPS `download_url`:

```bash
uv run python scripts/fetch_models.py
```

All current `download_url` values are null, so missing files must come through an approved provisioning channel. Core loading and the run scripts force `YOLO_OFFLINE=true` and `YOLO_AUTOINSTALL=false`; the application will not ask Ultralytics to download or install anything. Do not copy a similarly named model from an unknown source or disable verification. A passing artifact check establishes local integrity only; current weights are not production-approved.

Size/hash diagnostics do not load Ultralytics. If the first recognition request fails after the preflight passed, run the explicit real-model smoke to check required `detect` tasks and the schema-v2 required class-name subsets:

```bash
uv run pytest tests/model/test_real_inference.py -m model --no-cov
```

Extra classes exposed by a model are allowed, but the adapter filters predictions to manifest-declared IDs. A missing declared ID, changed declared label, or absent/mismatched task is an integrity error.

If model versions display `@unverified`, `NPR_VERIFY_MODELS=false` was used in development/test. Restore verification and fix the artifact mismatch; never treat the manifest digest as checked in that mode, and production refuses it.

## Production-mode startup intentionally fails

```bash
uv run python scripts/doctor.py --production
```

This is expected with the current three entries: `source` is null, `provenance_status` and `license_status` are unverified, and `production_approved` is false. Use `NPR_ENVIRONMENT=development` only for local/dev operation. Do not bypass the gate for a production deployment or edit approval fields without authoritative provenance, rights approval, and representative quality evidence.

## App starts only from a particular directory

Use the repository-root entry point:

```bash
uv run streamlit run app/streamlit_app.py
```

The supplied run scripts change to the repository root. For a relocated/non-editable launch, prefer an explicit absolute `NPR_APP_ROOT`; relative model, manifest, and image overrides resolve against it. An explicit relative `NPR_APP_ROOT` itself resolves from the current working directory.

The run scripts also load repository `.env` through `uv` when it exists and isolate Ultralytics state under `.runtime/ultralytics`. A direct `uv run streamlit ...` command does not parse `.env`; export the settings or pass the file explicitly.

## Streamlit page does not load

Confirm the process is listening and review privacy-safe startup logs. By default the local URL is <http://localhost:8501>. If running a container:

```bash
docker ps
docker logs <container-name-or-id>
```

Check port publication, the entrypoint artifact/policy result, proxy websocket support, and firewall rules. `/_stcore/health` proves Streamlit liveness only; concrete model initialization and task/class validation happen on first inference unless the deployment adds a warm-up/model-aware readiness probe. Never post logs containing secrets, images, or plate values in a public issue.

## Upload is rejected before the app sees it

The uploader is inside an explicit form and inference begins only after **Run recognition**. `STREAMLIT_SERVER_MAX_UPLOAD_SIZE` is a per-widget MiB transport cap, while `NPR_MAX_UPLOAD_BYTES` is the exact per-file byte cap and must not exceed it. `NPR_MAX_FILES` limits a submitted batch, and `NPR_MAX_IMAGE_PIXELS` applies during decode. Align proxy/body limits with these settings and restart Streamlit after changing process environment variables.

## No Approved examples appear

That is the clean-checkout default: the repository and container include no vehicle photographs. Follow [images/README.md](../images/README.md) to provision only authorized JPEG/PNG examples locally under `NPR_IMAGE_DIR`. They are intentionally ignored by Git and Docker; do not force-add them or use that directory as an evaluation dataset.

## `libGL` or GUI-library error in a container

Production uses a headless OpenCV package. Ensure a GUI-enabled OpenCV distribution was not introduced alongside it and rebuild from the locked production dependencies. Multiple OpenCV wheels in one environment can conflict.

## CUDA is unavailable

CPU is the default. Confirm configuration is not requesting CUDA unintentionally. For GPU deployment, compare the installed NVIDIA driver, container runtime, CUDA runtime, PyTorch build, and device visibility against the release's tested matrix:

```bash
nvidia-smi
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Do not install an arbitrary PyTorch wheel into the locked environment. Use the maintained GPU profile and rerun all tests/benchmarks.

## Out of memory or very slow inference

Check uploaded pixel dimensions, `NPR_INFERENCE_MAX_DIMENSION`, cascade maxima, queue time, number of loaded processes, and measured model memory. The longest side is resized without upscaling, and one process serializes complete requests; total timing includes time waiting for that lock. Reduce admission concurrency or reviewed limits based on profiling. Do not silently lower image quality or switch devices without re-running model evaluation.

## Corrupt image crashes the app

This is a bug. The UI should reject invalid images with a safe message and keep serving other requests. Capture the exception category and a minimal non-sensitive/synthetic reproducer. Do not attach a real user's plate image without authorization.

## No detections or poor recognition

- Confirm the exact manifest-declared hashes, required task/class subsets, and character `output_map`; do not call current development artifacts approved.
- Check per-stage confidence thresholds, `NPR_VEHICLE_CLASSES`, and all three cascade maxima.
- Review `NPR_PLATE_DEDUP_IOU` and `NPR_CHARACTER_OVERLAP_IOU`; overly aggressive overlap suppression can remove real candidates.
- Review the annotated boxes and stage-level structured results.
- Compare the case to known slices in the model card.
- Reproduce with an authorized fixture in the model-regression suite.

The default `NPR_PLATE_PATTERN` requires one of `A/B/E/D/H` between digit groups. It is only a configurable review-routing heuristic. Do not tune it or thresholds from a single image, force a pattern match from uncertain characters, or interpret a match as regulatory validity.

## Pre-commit differs from CI

```bash
uv sync --locked
uv run pre-commit clean
uv run pre-commit install --install-hooks
uv run pre-commit run --all-files
uv run python scripts/quality.py check
```

Confirm the same source revision, Python 3.12, `uv.lock`, and hook configuration. CI is authoritative for merge readiness.

## Getting help

For security or privacy issues, follow [SECURITY.md](../SECURITY.md). For ordinary bugs, include platform, Python version, source revision, redacted doctor output, exact command, and a synthetic/minimized reproduction. Never include credentials, raw production images, full plate values, or unapproved model artifacts.
