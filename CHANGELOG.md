# Changelog

All notable changes will be documented in this file. The project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and intends to adopt [Semantic Versioning](https://semver.org/) when the first supported release is published.

## [Unreleased]

### Added

- Production-readiness foundation: typed package architecture, reproducible Python 3.12/`uv` environment, custom bootstrap/diagnostic/quality/model scripts, tests, pre-commit checks, CI, containerization, and operational documentation.
- Schema-v2 model inventory for the three vehicle, plate, and character roles, including task/class subset contracts, character output mapping, integrity fields, and explicit provenance/license/production state.
- Bounded three-stage cascade controls, plate deduplication, character overlap suppression, complete-request queue timing, and longest-side inference/preview resizing.
- Submit-driven multi-file Streamlit form with framework and exact byte/pixel/batch limits, safe errors, structured results, and AppTest smoke coverage.
- Operator-provisioned **Approved example** workflow with an empty tracked image directory, local authorization/retention policy, and Git/Docker exclusion for photographs.

### Changed

- Product scope clarified as a stateless Moroccan number-plate recognition application, not a physical gate controller.
- Model predictions are restricted to manifest-declared class IDs; every accepted character class must decode to one ASCII digit or uppercase letter.
- Plate-pattern matching is documented and presented as a configurable one-letter review heuristic, never regulatory validation.
- The real-model smoke now invokes all three artifacts with a deterministic blank in-memory frame; representative accuracy evidence remains an external release gate.
- Local launch scripts load repository `.env` when present, isolate Ultralytics runtime state, and force offline/no-auto-install behavior.
- Documentation now treats model provenance, evaluation, privacy, security, and licensing as release gates.

### Security

- Added bounded uploads/cascade work, verified model artifacts, privacy-safe telemetry and errors, non-root/read-only container operation, dependency/container scanning, and responsible vulnerability reporting.
- Removed unprovenanced plate photographs from the current tree/container; historical Git/cache copies require an owner assessment and any cleanup must be coordinated.
- Model versions are labeled `@unverified` when development-only checksum verification is disabled, avoiding a false integrity claim.

No production release has been declared. Checksum-verified local operation is supported; the custom model artifacts remain blocked from production release pending provenance, licensing, and representative evaluation approval.

<!-- Add release links here once version tags exist. -->
