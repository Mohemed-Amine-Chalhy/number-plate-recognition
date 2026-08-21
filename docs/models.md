# Models, contracts, provenance, and evaluation

Model artifacts are executable inputs with security, privacy, quality, and licensing implications. A valid hash proves only that a file matches the reviewed inventory; it does not prove authorship, data rights, safety, accuracy, or production permission.

## Current artifacts

The production pipeline has exactly three required roles:

| Role | File | SHA-256 | Local status | Production status |
| --- | --- | --- | --- | --- |
| Vehicle | `yolov10n.pt` | `11287ed0735678e7ba1ac2a9b3098c049155b3fde123992e724c1264bcc16b6f` | Size/hash and semantic subset contract verified. | Blocked pending authoritative source, provenance verification, license approval, explicit approval, and representative evaluation. |
| Plate | `license_plate_detector.pt` | `8ec3b254a6c87610f037a90957462cafa11a9c03224e33a28c6a1d1ac2ac51b0` | Size/hash and semantic subset contract verified. | Blocked pending custom-model/data provenance, rights, explicit approval, and representative evaluation. |
| Character | `PlateReaderyolo.pt` | `adaddb32e801f59e0d18c2bede2d893b2cf2419d66e922846a789263da889425` | Size/hash, accepted class subset, and single-symbol decoding contract verified. | Blocked pending custom-model/data provenance, rights, domain review, explicit approval, and representative evaluation. |

Checksum-verified local CPU inference is supported. None of these artifacts is currently approved for production or redistribution.

## Manifest schema v2

`models/manifest.json` has `schema_version: 2` and a non-empty `models` array. Roles and case-insensitive filenames must be unique, and only `vehicle`, `plate`, and `character` roles are supported.

Every entry contains:

| Field | Contract |
| --- | --- |
| `name` | Stable non-empty artifact identifier used in displayed model-version strings. |
| `role` | One of `vehicle`, `plate`, or `character`. |
| `filename` | Safe relative artifact path; absolute paths and traversal are rejected. |
| `sha256` | Exactly 64 hexadecimal characters. |
| `size_bytes` | Positive integer byte size. |
| `source` | Authoritative provenance reference or `null`. It is required for production approval. |
| `download_url` | Optional absolute HTTPS fetch source or `null`. It is currently null for all artifacts. |
| `license` | Non-empty human-readable license record; it does not itself mark approval. |
| `license_status` | `unverified` or `approved`. Production requires `approved`. |
| `provenance_status` | `unverified` or `verified`. Production requires `verified`. |
| `production_approved` | Explicit boolean. Production requires `true`; approval must not be inferred from other fields. |
| `task` | Currently only `detect` is supported. The loaded model must expose a matching task; absence is rejected. |
| `expected_classes` | Non-empty mapping from canonical non-negative integer strings to exact class labels. |
| `output_map` | Label-to-display-symbol mapping. It must be empty outside the character role. |

### Required class subset contract

`expected_classes` is both a required semantic **subset** and the runtime allowlist. Each declared ID must exist in the loaded model and have the exact declared label. The loaded ontology may contain extra classes, but the adapter discards every prediction whose ID is not declared. This prevents a checksum-valid but semantically incompatible or unsupported class from entering the cascade.

The configured `NPR_VEHICLE_CLASSES` must also be a subset of the vehicle artifact's declared IDs. The current required subsets are:

- Vehicle: `2=car`, `3=motorcycle`, `5=bus`, `7=truck`.
- Plate: `0=license_plate`.
- Character: IDs `0,1,2,3,4,5,6,9,10,11,12,13,14,15,16` with exact labels that decode to digits `0`–`9` or letters `A/B/E/D/H`. Loaded IDs `7` and `8` (raw labels `15` and `16`) are deliberately not accepted.

For character artifacts, every `output_map` key must be one of the declared expected class labels. After applying the map (or retaining an unmapped raw label), **every accepted class must decode to exactly one ASCII digit or uppercase letter**. The runtime passes this mapping into post-processing; it is not a separate hard-coded production mapping.

The current accepted `class ID → raw model label → reconstructed symbol` contract is:

| Class ID | Raw label | Symbol | Class ID | Raw label | Symbol |
| ---: | --- | --- | ---: | --- | --- |
| 0 | `0` | `0` | 9 | `2` | `2` |
| 1 | `1` | `1` | 10 | `3` | `3` |
| 2 | `10` | `A` | 11 | `4` | `4` |
| 3 | `11` | `B` | 12 | `5` | `5` |
| 4 | `12` | `E` | 13 | `6` | `6` |
| 5 | `13` | `D` | 14 | `7` | `7` |
| 6 | `14` | `H` | 15 | `8` | `8` |
|  |  |  | 16 | `9` | `9` |

`output_map` keys are raw label strings, not class IDs. Raw labels `15` and `16` are excluded because they would not decode to one plate symbol.

## Verification and fetching

Verify schema, byte sizes, and hashes without importing the ML runtime:

```bash
uv run python scripts/doctor.py --models-only
```

The equivalent artifact command is:

```bash
uv run python scripts/fetch_models.py --verify-only
```

`fetch_models.py` supports repeated `--role ROLE`, `--force`, path overrides, and a timeout. It downloads only from an absolute HTTPS `download_url`, bounds the received bytes by `size_bytes`, verifies size/hash in a same-directory temporary file, and atomically replaces the target after success. No current entry has a download URL, so a missing file must be provisioned through an approved channel.

At model load, the adapter requires the task and required class subset to match, then restricts predictions to manifest-declared IDs. Setting `NPR_VERIFY_MODELS=false` skips file size/hash verification only in development/test; it does not skip schema or semantic contract validation, and production refuses that setting. Unverified development loads use model version identifiers ending in `@unverified` rather than displaying a manifest digest as if it had been checked.

Core loading and the supported run/deployment entry points set `YOLO_OFFLINE=true` and `YOLO_AUTOINSTALL=false`. Inference never asks Ultralytics to fetch a missing model or dependency; provision artifacts through `fetch_models.py` or another approved channel before startup.

## Current production gate

Run the same gate explicitly:

```bash
uv run python scripts/doctor.py --production
```

Production checks the three required roles and fails each unless all of these are true:

- `provenance_status` is `verified`;
- `source` is present;
- `license_status` is `approved`;
- `production_approved` is `true`.

The runtime enforces the same policy when `NPR_ENVIRONMENT=production`. The container entrypoint runs `doctor.py --models-only` before Streamlit; that command automatically applies production checks when the environment is production. The current entries use null sources, unverified provenance/licenses, and false approvals, so production startup fails closed by design.

This machine-readable gate cannot establish model quality. A representative, authorized evaluation report and acceptance decision remain separate release evidence. Do not set the fields to passing values without the referenced provenance, license, model-card, and evaluation records.

## Model card requirements

Create a versioned card for each changed/custom artifact with:

1. **Identity:** name, version, date, owner, contacts, immutable hash.
2. **Purpose:** intended use, out-of-scope use, users, and deployment context.
3. **Architecture:** base model, framework versions, input shape, task, complete classes, and preprocessing.
4. **Training data:** sources, dates, geography, consent/authority, licenses, filtering, labels, splits, retention, and leakage checks.
5. **Training:** code revision, parameters, compute, seeds, augmentations, and reproducibility artifacts.
6. **Evaluation:** dataset version, thresholds, exact-match/detection/class metrics, slice analysis, calibration, latency/memory, and comparison with the prior release.
7. **Risks:** failure modes, privacy impact, domain gaps, abuse scenarios, and mitigations.
8. **Licensing:** code, base-model, weight, dataset, and redistribution terms reviewed.
9. **Operations:** monitoring, drift criteria, rollback hash, and deprecation date.
10. **Approval:** technical, privacy, security, product, and licensing sign-off.

Do not fill unknown fields with assumptions. Mark them `TODO — release blocking`, assign an owner, and keep `production_approved` false. Current records are in [model-cards/README.md](model-cards/README.md).

## Evaluation protocol

Use a versioned holdout representing intended Moroccan conditions and never use it for training or threshold/pattern selection. Report raw counts and confidence intervals where possible. Include daylight/night, glare, shadow, oblique angles, blur, partial occlusion, multiple/overlapping vehicles, small plates, no-plate images, non-Moroccan plates, and adversarial/printed images.

End-to-end exact plate match is the main user-visible metric. Also report vehicle/plate detector precision and recall, per-character results/confusions, confidence calibration, false positives/negatives, and CPU/target-GPU latency, queue time, throughput, and memory. The configurable plate regex is only a routing heuristic; pattern-match rate is not regulatory validity or accuracy.

See [testing](testing.md) for the evaluator interface and current metric gaps.

## Safe serialization

PyTorch `.pt` files may use pickle-based loading. Treat them as untrusted executable content: accept only controlled, hash-pinned artifacts and load them in a restricted build/runtime. Prefer a safer constrained format after validating compatibility and accuracy. A scanner or hash cannot make an unknown model trustworthy.

`ULTRALYTICS_SAFE_LOAD` is not enabled because compatibility with the current artifacts has not been established. Enabling that or migrating serialization requires a dedicated load/semantic/accuracy comparison; offline mode and a checksum do not remove pickle execution risk.

## Updating a model

1. Create a new immutable artifact identity; never overwrite a released version in place.
2. Complete provenance, data/base-model rights, schema-v2 contract, and model card.
3. Evaluate against the fixed holdout and compare with the current production hash.
4. Run code, security, real-model, load, and container checks.
5. Update manifest size/hash/semantic fields and application compatibility.
6. Obtain approvals and only then set the machine-readable production fields.
7. Canary by immutable application and model identifiers; retain the prior approved set for rollback.

## Production release gate

The current artifacts remain release-blocking until evidence establishes:

- author/owner and authoritative source or training job;
- training/evaluation dataset provenance, authority/consent, and licenses;
- base-model, dependency, weight, and redistribution terms;
- complete ontology, mappings, preprocessing, and compatible runtime versions;
- representative quality, calibration, latency, capacity, and failure results;
- privacy, security, technical, and licensing approval;
- explicit immutable approval records reflected in schema v2.

The repository MIT license does not resolve artifact/dependency rights. The Ultralytics AGPL-3.0 versus Enterprise decision is unresolved; see [third-party notices](../THIRD_PARTY_NOTICES.md). This is not legal advice.
