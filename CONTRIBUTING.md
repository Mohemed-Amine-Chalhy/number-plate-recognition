# Contributing

Contributions should keep the project reproducible, typed, testable, and easy to review.

## Setup

Use Python 3.12 and the committed `uv.lock`.

PowerShell:

```powershell
.\scripts\bootstrap.ps1
```

Bash:

```bash
bash scripts/bootstrap.sh
```

Create a focused branch and keep unrelated cleanup out of the change.

## Design rules

- Keep Streamlit concerns in `app/streamlit_app.py`; domain and inference code belongs under `src/number_plate_recognition/`.
- Convert Ultralytics objects to project-owned typed values at the adapter boundary.
- Add complete type annotations and keep strict mypy green.
- Validate files, configuration, model contracts, and detector output at their boundaries.
- Keep deterministic geometry, imaging, and post-processing logic independent of model weights.
- Add tests for success, empty, boundary, and failure paths.
- Preserve input and cascade bounds unless measurements justify a change.
- Update documentation and `.env.example` whenever public behavior or configuration changes.

## Model changes

A checkpoint change must include:

- an updated schema-v3 manifest entry with exact byte size and SHA-256;
- verified task, expected-class, and character-output contracts;
- the real-model smoke test;
- an evaluator comparison when prediction behavior changes;
- measured latency and memory when the model architecture or input size changes.

Do not overwrite a checkpoint while retaining its old identity or hash. Keep large exploratory artifacts and generated notebook output out of commits.

## Run the gate

```bash
uv run python scripts/quality.py check
uv run pre-commit run --all-files
```

For changes to models, inference dependencies, or adapters, also run:

```bash
uv run pytest tests/model/test_real_inference.py -m model --no-cov
```

Container-related changes should include a successful image build and health check.

## Commits and pull requests

Use small, descriptive commits that each leave the repository coherent. A pull request should explain:

- the problem and user-visible impact;
- the chosen design and meaningful trade-offs;
- tests and manual verification performed;
- configuration, compatibility, performance, or deployment effects;
- screenshots when the interface changes.

Before requesting review, confirm that the diff is focused, CI is green, behavior is tested, and documentation matches the implementation.
