# Current model cards

These provisional cards cover the three artifacts in `models/manifest.json`. They record known integrity and runtime contracts without inventing missing provenance. Every `TODO — release blocking` item needs evidence and approval before production or redistribution.

The schema-v2 manifest is authoritative for machine-readable identity, hashes, task/class contracts, mapping, and approval state. A model update must change its immutable identity/hash and this record together.

## Vehicle detector: `yolov10n-vehicle-detector`

| Field | Current record |
| --- | --- |
| Role / task | `vehicle` / `detect` |
| File | `yolov10n.pt` |
| SHA-256 | `11287ed0735678e7ba1ac2a9b3098c049155b3fde123992e724c1264bcc16b6f` |
| Size | 5,860,383 bytes |
| Accepted class subset | `2=car`, `3=motorcycle`, `5=bus`, `7=truck`; extra loaded classes may exist but their predictions are filtered out. |
| Intended use | Bound the first cascade stage to configured supported vehicle classes. |
| Source / provenance | `null` / `unverified` |
| License / status | “Ultralytics AGPL-3.0 or applicable commercial license; verify before production” / `unverified` |
| Production approved | `false` |

Release-blocking TODOs:

- Establish exact upstream source/release, author, acquisition date, and whether the file is unmodified.
- Resolve AGPL-3.0 compliance versus sufficient commercial rights for the intended deployment/distribution.
- Archive the complete ontology, preprocessing, framework/runtime compatibility, and acquisition record.
- Evaluate vehicle recall/precision and downstream exact-plate impact on the approved Moroccan holdout and slices.
- Record calibration, CPU/GPU latency, queue behavior, memory, concurrency, owner, approvals, and rollback artifact.

## Plate detector: `moroccan-license-plate-detector`

| Field | Current record |
| --- | --- |
| Role / task | `plate` / `detect` |
| File | `license_plate_detector.pt` |
| SHA-256 | `8ec3b254a6c87610f037a90957462cafa11a9c03224e33a28c6a1d1ac2ac51b0` |
| Size | 6,241,454 bytes |
| Accepted class subset | `0=license_plate`; extra loaded classes may exist but their predictions are filtered out. |
| Intended use | Detect plate candidates inside bounded vehicle crops before absolute-coordinate deduplication. |
| Source / provenance | `null` / `unverified` |
| License / status | “Unknown custom-model license; verify before production” / `unverified` |
| Production approved | `false` |

Release-blocking TODOs:

- Identify author/owner, source repository or training job, base architecture/version, and acquisition date.
- Document training/evaluation datasets, geography/dates, labels, splits, consent/authority, licenses, retention, and leakage checks.
- Establish code, base-weight, output-weight, dataset, deployment, and redistribution rights.
- Record input/preprocessing, complete ontology, training parameters, framework versions, seeds, and reproducibility artifacts.
- Evaluate precision/recall and false positives across lighting, angle, distance, occlusion, plate style, no-plate, and non-Moroccan cases.
- Record calibration, cascade/IoU sensitivity, CPU/GPU latency/memory, owner, approvals, and rollback artifact.

## Character detector: `moroccan-plate-character-detector`

| Field | Current record |
| --- | --- |
| Role / task | `character` / `detect` |
| File | `PlateReaderyolo.pt` |
| SHA-256 | `adaddb32e801f59e0d18c2bede2d893b2cf2419d66e922846a789263da889425` |
| Size | 16,533,563 bytes |
| Accepted class subset | IDs `0,1,2,3,4,5,6,9,10,11,12,13,14,15,16` with exact labels from schema v2. Loaded IDs `7/8` (raw labels `15/16`) are filtered out. |
| Output mapping | Raw label strings `10→A`, `11→B`, `12→E`, `13→D`, `14→H`; accepted unmapped labels are single digits. These keys are labels, not class IDs. Every accepted class must decode to one ASCII digit/uppercase letter. |
| Intended use | Detect, overlap-suppress, limit, left-to-right order, and reconstruct characters within a plate crop. |
| Pattern behavior | Default full-match heuristic: `^[0-9]{1,5}[ABEDH][0-9]{1,2}$`. It requires one mapped letter and is configurable; it is not regulatory validation or proof of correctness. |
| Source / provenance | `null` / `unverified` |
| License / status | “Unknown custom-model license; verify before production” / `unverified` |
| Production approved | `false` |

Release-blocking TODOs:

- Identify author/owner, source repository or training job, base architecture/version, and acquisition date.
- Document dataset provenance/rights, complete ontology, label quality/balance, augmentations, splits, and leakage checks.
- Establish code, base-weight, output-weight, dataset, deployment, and redistribution rights.
- Confirm the accepted-class allowlist and letter output mapping with Moroccan domain expertise; retain evidence for excluding raw labels `15` and `16`.
- Define evidence-based plate-pattern policy, uncertain/unknown handling, and human-review behavior without claiming regulatory validity.
- Report per-class metrics/confusions, character accuracy, exact-plate match, calibration, slice failures, and cascade/IoU sensitivity.
- Record compatibility, reproducibility, CPU/GPU latency/memory, owner, approvals, and rollback artifact.

## Approval record

No current artifact has production approval. Each future approval record must include:

- artifact name, role, SHA-256, source, and complete model-card revision;
- technical/model-quality approver and date;
- dataset/privacy approver and date;
- security approver and date;
- licensing/compliance approver and date;
- application versions/environments approved;
- evaluation report and explicit acceptance thresholds;
- expiry/review date and rollback target.

Only after those records exist should `provenance_status`, `license_status`, and `production_approved` be updated. This documentation does not grant permission and is not legal advice.
