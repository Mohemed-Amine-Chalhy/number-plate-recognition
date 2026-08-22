# Development

## Supported environment

Use Python 3.12 and `uv`. The committed `uv.lock` is the reproducibility boundary; update it through `uv`, not by hand.

```bash
python --version
uv --version
```

## Bootstrap

From the repository root:

PowerShell:

```powershell
.\scripts\bootstrap.ps1
```

Bash:

```bash
bash scripts/bootstrap.sh
```

Bootstrap verifies Python and `uv`, synchronizes the locked environment, installs Git hooks, and runs the environment doctor. It is safe to run repeatedly.

Equivalent manual setup:

```bash
uv sync --locked
uv run pre-commit install --install-hooks
uv run pre-commit install --hook-type pre-push
uv run python scripts/doctor.py
```

## Run the application

```powershell
.\scripts\run_app.ps1
```

```bash
bash scripts/run_app.sh
```

Or launch Streamlit directly:

```bash
uv run streamlit run app/streamlit_app.py
```

The supplied scripts load a repository `.env` when present, use the repository root, disable Ultralytics auto-install/online behavior, and isolate framework state under `.runtime/ultralytics`.

## Project scripts

| Script | Purpose |
| --- | --- |
| `bootstrap.ps1` / `bootstrap.sh` | Create a locked development or runtime environment and optionally install hooks. |
| `doctor.py` | Validate Python, dependencies, paths, configuration, and schema-v3 model artifacts. |
| `evaluate.py` | Run the structured end-to-end string evaluator against ground-truth JSON. |
| `quality.py` | Run or safely fix the configured format, lint, type, manifest, and test checks. |
| `run_app.ps1` / `run_app.sh` | Establish the runtime environment and launch Streamlit. |

Use `uv run python scripts/<name>.py --help` for exact Python interfaces. Useful doctor modes include `--manifest-only`, `--models-only`, `--skip-imports`, `--skip-model-files`, and `--json`.

Bootstrap installs runtime plus the default development group. `--runtime-only`/`-RuntimeOnly` omits developer tools and hooks. `--all-groups`/`-AllGroups` also installs optional notebook tooling.

## Code boundaries

- Keep UI behavior in `app/streamlit_app.py`.
- Keep deterministic domain, imaging, geometry, post-processing, and evaluation code independent of Streamlit and Ultralytics.
- Convert third-party results into project-owned typed values in the adapter.
- Validate configuration, payloads, model metadata, and predictions at their boundaries.
- Avoid implicit current-directory paths, wildcard imports, debug `print` calls, and global request state.
- Add complete annotations and preserve strict mypy.

See [Architecture](architecture.md) for module responsibilities and request flow.

## Dependency changes

Runtime dependency:

```bash
uv add package-name
```

Development dependency:

```bash
uv add --group dev package-name
```

Then verify the lock and full project:

```bash
uv lock --check
uv run python scripts/quality.py check
```

Review transitive changes, platform wheels, import-time behavior, known vulnerabilities, and container size. Commit `pyproject.toml` and `uv.lock` together.

## Quality workflow

Fast targeted checks while editing:

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy
uv run pytest tests/unit
```

Complete local checks before opening a pull request:

```bash
uv run python scripts/quality.py check
uv run pre-commit run --all-files
```

Use `uv run python scripts/quality.py fix` for configured Ruff fixes and formatting, then review the resulting diff.

The default pytest run excludes the real-model marker. Run the checkpoint smoke after changing model files, inference dependencies, the Ultralytics adapter, or pipeline boundaries:

```bash
uv run pytest tests/model/test_real_inference.py -m model --no-cov
```

## Examples and notebooks

The tracked `images/Car1.jpg` through `images/Car3.jpg` files provide a quick interactive demo. Unit tests should continue to prefer small generated arrays and fakes so the fast suite remains deterministic.

Notebooks are exploratory, not an application entry point. Strip outputs before committing, keep notebook-only dependencies in their optional group, and move reusable logic into `src/` with tests.

## Replacing a model

1. Add the new artifact without overwriting an existing checkpoint identity.
2. Update its schema-v3 filename, byte size, SHA-256, task, expected classes, and output mapping.
3. Run manifest/integrity checks and the real-model smoke.
4. Compare structured results with `scripts/evaluate.py` when behavior changes.
5. Measure model load, inference latency, and memory when architecture or input size changes.
6. Update [Models and evaluation](models.md) and any affected configuration.

## Pull requests and commits

Keep commits focused and reversible. A useful sequence is implementation, tests, tooling, and documentation. Do not mix generated checkpoint binaries with unrelated source changes. See [CONTRIBUTING.md](../CONTRIBUTING.md) for the review checklist.
