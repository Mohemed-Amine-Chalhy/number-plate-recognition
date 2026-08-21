# Moroccan Number Plate Recognition

A stateless Streamlit application that detects vehicles, locates Moroccan number plates, and reconstructs plate text from uploaded images. The application is a recognition tool: it does **not** operate barriers, authorize vehicles, retain an access list, or provide the safety controls required for a physical gate.

> [!IMPORTANT]
> The bundled custom weights are development artifacts. Their training-data rights, authorship, model metrics, and redistribution terms have not yet been recorded. Complete the [model provenance checklist](docs/models.md#production-release-gate) and resolve the [Ultralytics licensing decision](#licensing-decision) before any production or commercial release.

Current status: local CPU operation is supported with the three bundled artifacts, whose byte sizes, SHA-256 hashes, detection tasks, and required class-name subsets are checked against the schema-v2 `models/manifest.json`. The repository's intentional production-release gate is the external evidence those checks cannot provide: model/dataset provenance, approved usage and redistribution rights, and representative quality acceptance. An internet-facing deployment must also supply its environment-specific TLS, identity, observability, capacity, and retention controls.

## What it does

The inference pipeline runs three bounded stages:

1. Detect supported vehicle classes in an image.
2. Detect and deduplicate plate candidates inside bounded vehicle crops.
3. Detect, overlap-suppress, map, and order plate characters, then return structured results and an annotated image.

Every cascade stage has a configurable maximum. Complete requests share one inference lock; results report queue plus per-stage/total timings. Uploads run only after the explicit form submission, have both framework and exact byte/pixel/count caps, and are previewed at the same bounded longest-side size used for inference.

The default plate regex requires digits, exactly one mapped letter, then digits. It is a configurable review-routing heuristic—never regulatory validation, identity proof, or an accuracy guarantee.

The implementation is designed for Moroccan plates and has not been validated for general OCR, surveillance, vehicle identity, or access-control decisions. Recognition output can be wrong; a human or an independently validated policy layer must handle consequential decisions.

## Requirements

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- PowerShell on Windows, or Bash on Linux/macOS
- Streamlit 1.56 or newer within the locked `<2` range (installed by bootstrap)
- Integrity-declared model artifacts described by `models/manifest.json` (production additionally requires provenance approval)

CPU inference is the supported reproducible default. GPU deployment is optional and must be tested against the selected PyTorch/CUDA combination.

## Quick start

Clone the repository, then run the platform bootstrap script from the repository root.

PowerShell:

```powershell
.\scripts\bootstrap.ps1
.\scripts\run_app.ps1
```

Bash:

```bash
bash scripts/bootstrap.sh
bash scripts/run_app.sh
```

The run scripts load the repository `.env` through `uv` when it exists, preserve existing process environment settings, keep Ultralytics runtime state under the ignored `.runtime/ultralytics` directory, and force model auto-install/download behavior off. The three manifest-pinned artifacts must already exist locally and verify.

The app is normally available at <http://localhost:8501>. You can also launch it directly:

```bash
uv run streamlit run app/streamlit_app.py
```

Run the environment diagnostic if startup fails:

```bash
uv run python scripts/doctor.py
```

The stricter release diagnostic intentionally fails while model provenance is unresolved:

```bash
uv run python scripts/doctor.py --production
```

See [development setup](docs/development.md) and [troubleshooting](docs/troubleshooting.md) for platform-specific details.

## Repository layout

```text
app/                         Streamlit presentation adapter
src/number_plate_recognition/  Typed inference/domain package
tests/                       Unit, integration, and smoke tests
models/manifest.json         Model integrity/provenance inventory
images/README.md             Local approved-example policy (no photos ship)
scripts/                     Bootstrap, diagnostics, evaluation, quality, and run tools
docs/                        Architecture, model, deployment, and operations guides
```

## Configuration

`.env.example` documents the supported environment variables and can be copied to `.env`. The supplied run scripts load that file automatically; Compose reads it when present, and `docker run` can receive it with `--env-file`. The Python application itself reads process environment variables and does not parse dotenv files. Do not commit `.env` or secrets.

Application root/environment, model locations, confidence/IoU/pattern policy, cascade/input limits, execution device, logging, and Streamlit transport settings are documented in [configuration](docs/configuration.md). Invalid values fail early rather than silently falling back.

## Quality checks

The same checks are intended to run locally, in pre-commit, and in CI:

```bash
uv run python scripts/quality.py check
uv run pytest
uv run mypy
uv run ruff check .
uv run ruff format --check .
uv run pre-commit run --all-files
```

The default test configuration excludes the slow real-model test. Run it explicitly when changing models, inference dependencies, or adapters:

```bash
uv run pytest tests/model/test_real_inference.py -m model --no-cov
```

That smoke uses a deterministic blank in-memory frame to load and invoke all three real models without committing identifiable plate imagery. It checks integrity/load/API compatibility and absence of a vehicle false positive on that frame; it is not an accuracy evaluation.

Install the hooks after a manual environment setup with:

```bash
uv run pre-commit install --install-hooks
uv run pre-commit install --hook-type pre-push
```

See [testing](docs/testing.md) for test tiers, markers, and model-regression expectations.

## Model artifacts

Exactly three production roles are tracked through schema-v2 `models/manifest.json`; code does not select arbitrary files by filename alone. To verify local artifacts:

```bash
uv run python scripts/fetch_models.py --verify-only
```

If approved download URLs are configured, obtain missing weights with:

```bash
uv run python scripts/fetch_models.py
```

Do not add large, unreviewed weights to Git. Every released artifact needs integrity fields, `task`, a required `expected_classes` subset, any character `output_map`, provenance/license status, explicit production approval, and an evaluation record. See [models and model cards](docs/models.md).

The repository ships no vehicle photographs. Operators may add only approved, authorized JPEG/PNG examples under `images/` for local use; Git and Docker ignore those files. Read [the local example policy](images/README.md), and keep representative labeled quality datasets in a separate controlled location.

## Container deployment

Build the CPU-first image from the repository root:

```bash
docker build --tag number-plate-recognition:local .
docker run --rm --publish 8501:8501 number-plate-recognition:local
```

To apply reviewed overrides, add `--env-file .env`; `compose.yaml` also loads an optional `.env` and provides a hardened local exercise. The container entrypoint verifies manifest policy, sizes, and hashes before Streamlit starts.

The image is only one layer of a production deployment. Put it behind TLS and authentication, impose request limits, keep the runtime filesystem read-only where practical, and configure observability without logging raw images or full plate values. See [deployment](docs/deployment.md) and [operations](docs/operations.md).

## Documentation

- [Architecture](docs/architecture.md)
- [Configuration](docs/configuration.md)
- [Development](docs/development.md)
- [Testing and quality gates](docs/testing.md)
- [Models, provenance, and evaluation](docs/models.md)
- [Current model cards](docs/model-cards/README.md)
- [Deployment](docs/deployment.md)
- [Operations and rollback](docs/operations.md)
- [Privacy, security, and threat model](docs/privacy-security.md)
- [Limitations](docs/limitations.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Contributing](CONTRIBUTING.md)
- [Security policy](SECURITY.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)

## Production readiness

This repository is ready for production only when all of the following are true:

- `scripts/doctor.py` and `scripts/quality.py check` pass in a clean, locked environment.
- CI passes linting, formatting, strict type checking, tests, coverage, security checks, and the container smoke test.
- The exact deployed model hashes and semantic contracts are approved and their provenance/evaluation records are complete.
- End-to-end accuracy, failure-rate, latency, and memory targets are met on representative Moroccan data and target hardware.
- Authentication, TLS, rate limits, upload limits, monitoring, incident response, retention, and rollback are configured for the deployment.
- A privacy review has approved the handling of plate images and recognized values.
- The owner has assessed whether previously removed plate photographs still reachable in Git history or remote caches require a coordinated purge/cache cleanup before public release.
- The dependency and model licensing decision has been documented and approved.

The local development gate is supported. `scripts/doctor.py --production` remains deliberately red because each current artifact has a null source, unverified provenance/license status, and `production_approved: false`. Representative quality acceptance remains a separate release decision; do not flip those fields without its evidence.

## Licensing decision

The repository source is offered under the [MIT License](LICENSE). That does not automatically grant rights to training data, model weights, or third-party dependencies.

This project currently uses Ultralytics software and model formats. Ultralytics describes AGPL-3.0 and Enterprise licensing options for its software and trained models; the correct option depends on how this application is distributed and operated. Before production or commercial use, obtain a qualified review and either comply with all applicable open-source obligations, obtain the required commercial rights, or replace the dependency with an approved alternative. See the [Ultralytics license page](https://www.ultralytics.com/license) and [third-party notices](THIRD_PARTY_NOTICES.md).

This is project documentation, not legal advice.

## License

Repository-authored source code is licensed under MIT unless a file states otherwise. See [LICENSE](LICENSE). Third-party components and artifacts retain their own terms.
