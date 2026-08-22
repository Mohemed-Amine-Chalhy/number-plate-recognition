# Development

## Supported environment

Use Python 3.12 and `uv`. The committed `uv.lock` is the reproducibility boundary; do not hand-edit it or install unrecorded packages into the project environment. The runtime contract pins Streamlit to `>=1.56,<2`; the lower bound is required by the tested uploader/AppTest interface.

Check prerequisites:

```powershell
python --version
uv --version
```

```bash
python3 --version
uv --version
```

## Bootstrap

From the repository root:

```powershell
.\scripts\bootstrap.ps1
```

```bash
bash scripts/bootstrap.sh
```

Bootstrap should verify Python/uv, synchronize from the lockfile, install Git hooks, and run the environment doctor. It must be safe to run repeatedly.

Equivalent manual setup:

```bash
uv sync --locked
uv run pre-commit install --install-hooks
uv run pre-commit install --hook-type pre-push
uv run python scripts/doctor.py
```

Verify the three current artifacts; if a future approved manifest supplies HTTPS download URLs, the same tool can provision missing files:

```bash
uv run python scripts/fetch_models.py --verify-only
uv run python scripts/fetch_models.py
```

All current `download_url` fields are null, so the second command only confirms already-valid files and reports a missing artifact rather than downloading it. Never add an arbitrary URL or bypass a checksum to make development proceed.

## Run the app

```powershell
.\scripts\run_app.ps1
```

```bash
bash scripts/run_app.sh
```

Or directly:

```bash
uv run streamlit run app/streamlit_app.py
```

Run documented commands from the repository root. The launch scripts do this automatically, load the repository `.env` when present, force `YOLO_AUTOINSTALL=false`/`YOLO_OFFLINE=true`, and place Ultralytics state under ignored `.runtime/ultralytics`. Runtime-relative paths resolve against `NPR_APP_ROOT`; set it explicitly for a non-editable/relocated deployment. The direct command above does not parse `.env`; export settings first or pass a reviewed file through `uv run --env-file`.

## Custom script reference

| Script | Supported interface |
| --- | --- |
| `bootstrap.ps1` | `[-RuntimeOnly | -AllGroups] [-NoHooks]` |
| `bootstrap.sh` | `[--runtime-only | --all-groups] [--no-hooks]` |
| `doctor.py` | `--manifest-only`, `--models-only`, `--production`, `--skip-imports`, `--skip-model-files`, and `--json` diagnostics |
| `fetch_models.py` | `--verify-only`, `--force`, repeatable `--role ROLE`, and path/timeout overrides |
| `evaluate.py` | Required ground-truth JSON plus optional `--image-root DIR` |
| `quality.py` | `check` or `fix`, with optional `--skip-tests` and `--keep-going` |
| `run_app.ps1` / `run_app.sh` | Loads repository `.env` when present, forces offline/no-auto-install policy, isolates Ultralytics state, and passes remaining arguments to Streamlit after fixed safe launcher options |

Use `uv run python scripts/<name>.py --help` or the shell script's `--help` for the exact interface. The PowerShell bootstrap switches are case-insensitive.

Bootstrap installs runtime plus the default development group. `--runtime-only`/`-RuntimeOnly` omits development tools and hooks. `--all-groups`/`-AllGroups` additionally installs every optional group currently declared in `pyproject.toml` (the `notebook` group); it is mutually exclusive with runtime-only mode.

## Working on code

Keep UI, inference orchestration, domain logic, and the Ultralytics adapter separated as described in [architecture](architecture.md). In particular:

- Add type annotations to all project code.
- Convert third-party result objects to project-owned typed values at the adapter boundary.
- Keep deterministic logic independent of model weights so it can be tested quickly.
- Validate external data before using it in array slicing or model calls.
- Avoid wildcard imports, debug `print` calls, global request data, and implicit current-directory paths.
- Do not add a dependency without documenting why it is direct and production- or development-only.

## Dependency changes

Add a runtime dependency:

```bash
uv add package-name
```

Add a development dependency:

```bash
uv add --group dev package-name
```

Then run:

```bash
uv lock --check
uv run python scripts/quality.py check
```

Review transitive changes, licenses, known vulnerabilities, platform wheels, and container size. Commit `pyproject.toml` and `uv.lock` together.

## Quality workflow

Fast targeted checks while editing:

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy
uv run pytest tests/unit
```

Complete local gate before opening a pull request:

```bash
uv run python scripts/quality.py check
uv run pre-commit run --all-files
```

The default pytest configuration excludes the real-model marker. Run it when changing artifacts or the inference boundary:

```bash
uv run pytest tests/model/test_real_inference.py -m model --no-cov
```

Use `uv run python scripts/quality.py fix` for configured safe fixes, then review every automated change.

## Data and notebooks

Notebooks are exploratory, not a production execution path. The retained legacy notebook references a public dataset listing, but that reference is not provenance for any model or removed image and is not production-quality evidence. Strip outputs before committing, pin notebook-only dependencies in an optional group, and move reusable logic into `src/`.

The tracked `images/` directory contains only [policy](../images/README.md). Local Approved examples are ignored by Git/Docker and must have recorded source, authorization, allowed purpose, retention, and deletion ownership. Never commit identifiable plate photos, private datasets, credentials, or training paths. Keep evaluation datasets outside the repository in an access-controlled workflow and record their provenance/authority in the model card.

## Adding or replacing a model

Follow [models](models.md). A model change is a production code change: update all schema-v2 integrity, `task`, required `expected_classes`, character `output_map`, provenance/license, and explicit approval fields; update its model card; run semantic/real-model/evaluation suites; review privacy/licensing; measure queue/resource use; and provide rollback instructions. Do not set production approval fields without linked evidence.

## Pull requests and commits

Keep commits focused and reversible. A useful sequence is environment/tooling, application refactor, tests, deployment, and documentation. Do not combine generated weight files with source changes. See [CONTRIBUTING.md](../CONTRIBUTING.md) for the review checklist.
