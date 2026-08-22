# Contributing

Thank you for improving the project. Contributions should keep the application reproducible, testable, privacy-preserving, and explicit about model uncertainty.

## Before starting

- Read the [architecture](docs/architecture.md), [development guide](docs/development.md), [testing guide](docs/testing.md), and [privacy/security model](docs/privacy-security.md).
- Open an issue or design discussion before changing the public result schema, model stack, artifact format, data collection, deployment boundary, or licensing posture.
- Never submit confidential data, personal plate images, credentials, proprietary datasets, or weights you do not have permission to redistribute.

## Setup

```powershell
.\scripts\bootstrap.ps1
```

```bash
bash scripts/bootstrap.sh
```

Create a focused branch. Keep unrelated cleanup out of the change.

## Implementation expectations

- Support Python 3.12 and the committed `uv.lock`.
- Add complete type annotations and preserve strict type checking.
- Keep Streamlit concerns out of domain and inference logic.
- Validate untrusted files and model outputs at boundaries.
- Add tests for success, empty, boundary, and failure cases.
- Keep user images/results request-scoped; do not add telemetry or persistence without review.
- Use structured logging with no raw image or full plate value.
- Update documentation whenever behavior or configuration changes.
- Do not weaken a security, type, lint, test, checksum, or coverage gate to make a change pass.

## Models and data

A model change must include an immutable checksum, updated schema-v2 manifest, completed model card, provenance and license evidence, evaluation comparison, performance measurements, privacy/security review, and rollback target. Preserve the required `detect` task, required `expected_classes` subset/runtime allowlist, role-appropriate `output_map`, and explicit approval fields. See [docs/models.md](docs/models.md).

Fixtures must be synthetic, explicitly consented, public-domain, or otherwise authorized. Record their exact source, authority/rights, transformations, retention, and deletion owner. Prefer masked or fictional plate values. Never force-add local `images/` photographs; follow [images/README.md](images/README.md), and keep representative evaluation datasets outside the repository.

## Run the gate

```bash
uv run python scripts/quality.py check
uv run pre-commit run --all-files
```

Run relevant model and container tests for changes that touch inference, dependencies, artifacts, or deployment.

The default pytest configuration excludes real weights. For a model or inference-contract change, also run:

```bash
uv run pytest tests/model/test_real_inference.py -m model --no-cov
```

## Commits and pull requests

Use small, descriptive commits that each leave the repository coherent. In the pull request, explain:

- the user-visible or operational problem;
- the chosen design and important alternatives;
- tests and manual verification performed;
- privacy, security, licensing, compatibility, and performance impact;
- configuration, migration, deployment, and rollback steps;
- screenshots only when useful and only with non-sensitive data.

The author checklist is:

- [ ] Scope is focused and no unrelated user work was changed.
- [ ] Formatting, Ruff, strict mypy, pytest/coverage, pre-commit, and required scans pass.
- [ ] New/changed behavior has tests, including failure paths.
- [ ] Documentation and examples match the implementation.
- [ ] No secrets, sensitive plate data, notebook output, or unapproved binary artifacts were added.
- [ ] Dependency/model licenses and provenance were reviewed when applicable.
- [ ] Deployment and rollback effects are documented.

Review approval and green CI are required before merge. Security-sensitive fixes may follow a private process described in [SECURITY.md](SECURITY.md).

## License

By contributing repository-authored source or documentation, you agree that it may be distributed under the repository's MIT license. You must have the right to contribute every submitted artifact. Third-party code, data, and weights require explicit provenance and compatible terms.
