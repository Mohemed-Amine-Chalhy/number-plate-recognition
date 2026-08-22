# Model contracts and evaluation

The pipeline uses three local YOLO checkpoints. The schema-v3 manifest makes their integrity and runtime semantics explicit so a renamed, corrupted, or incompatible file cannot enter inference silently.

## Current artifacts

| Role | File | Size | SHA-256 | Runtime contract |
| --- | --- | ---: | --- | --- |
| Vehicle | `yolov10n.pt` | 5,860,383 bytes | `11287ed0735678e7ba1ac2a9b3098c049155b3fde123992e724c1264bcc16b6f` | `detect`; supported vehicle class subset. |
| Plate | `license_plate_detector.pt` | 6,241,454 bytes | `8ec3b254a6c87610f037a90957462cafa11a9c03224e33a28c6a1d1ac2ac51b0` | `detect`; `0=license_plate`. |
| Character | `PlateReaderyolo.pt` | 16,533,563 bytes | `adaddb32e801f59e0d18c2bede2d893b2cf2419d66e922846a789263da889425` | `detect`; accepted symbol subset plus output mapping. |

## Schema v3

`models/manifest.json` contains `schema_version: 3` and exactly one entry for each `vehicle`, `plate`, and `character` role. Roles and case-insensitive filenames must be unique.

Every model entry contains:

| Field | Contract |
| --- | --- |
| `name` | Stable non-empty artifact identifier used in model-version strings. |
| `role` | `vehicle`, `plate`, or `character`. |
| `filename` | Safe relative artifact path; absolute paths and traversal are rejected. |
| `sha256` | Exactly 64 hexadecimal characters. |
| `size_bytes` | Positive integer byte size. |
| `task` | The expected Ultralytics task; currently `detect`. |
| `expected_classes` | Non-empty mapping from canonical non-negative integer IDs to exact class labels. |
| `output_map` | Raw-label-to-symbol mapping; empty outside the character role. |

### Required class subset

`expected_classes` is both a semantic subset check and a prediction allowlist. The loaded model may expose extra classes, but only declared IDs can enter the project pipeline. Each declared ID must exist and have the exact declared label.

The current required subsets are:

- Vehicle: `2=car`, `3=motorcycle`, `5=bus`, `7=truck`.
- Plate: `0=license_plate`.
- Character: IDs `0,1,2,3,4,5,6,9,10,11,12,13,14,15,16`, decoded to digits `0`–`9` or letters `A/B/E/D/H`.

The configured `NPR_VEHICLE_CLASSES` must be a subset of the vehicle entry. Predictions with undeclared IDs are discarded even if the loaded ontology contains them.

### Character output mapping

The character checkpoint uses numeric raw labels for some letter classes. `output_map` keys are raw label strings, not class IDs:

| Raw label | Symbol | Raw label | Symbol |
| --- | --- | --- | --- |
| `10` | `A` | `13` | `D` |
| `11` | `B` | `14` | `H` |
| `12` | `E` |  |  |

Unmapped accepted labels are single digits. Every accepted class must decode to exactly one ASCII digit or uppercase letter. Raw labels `15` and `16` are excluded because they do not represent one supported output symbol.

## Verification

Verify manifest structure, file sizes, and hashes without loading the ML runtime:

```bash
uv run python scripts/doctor.py --models-only
```

At model initialization, the Ultralytics adapter also verifies task and required class labels, then filters all predictions through the declared allowlist. Core loading and the supported run/container entry points force offline mode and disable auto-installation.

Setting `NPR_VERIFY_MODELS=false` skips byte-size/hash verification for a deliberate local experiment; schema and loaded semantic checks still run. Model identifiers from that mode end in `@unverified` so results do not imply that the checkpoint digest was checked.

## Checkpoint loading

PyTorch `.pt` files may use pickle-based deserialization and can execute code while loading. Only load the repository's hash-pinned local artifacts. Do not point the manifest at an arbitrary downloaded file or enable inference-time fetching.

A future migration to a constrained serialization format should compare load compatibility, output semantics, accuracy, latency, and memory before replacing the current artifacts.

## Evaluation protocol

The real-model smoke proves that checkpoints deserialize, match their contracts, and can be invoked. Recognition quality requires a labeled image set that represents the intended camera conditions.

For an evaluation set, report:

- end-to-end exact plate match;
- vehicle and plate precision/recall;
- per-character accuracy and confusion pairs;
- false-positive/negative cases;
- results by lighting, distance, angle, blur, occlusion, and plate style;
- queue/stage/total latency, throughput, and peak memory on named hardware.

The evaluator accepts ground-truth JSON with unique sample IDs and one or more valid expected strings:

```json
{
  "samples": [
    {"id": "daylight-001", "image": "daylight-001.jpg", "expected": ["123A45"]}
  ]
}
```

Run it with:

```bash
uv run python scripts/evaluate.py path/to/ground-truth.json
```

It reports exact-match precision/recall/F1, exact sample rate, and mean normalized character similarity. Detector mAP, calibration, slice aggregation, confidence intervals, and latency percentiles are useful future extensions.

## Updating a checkpoint

1. Give the artifact a new immutable filename/identity.
2. Update size, SHA-256, task, expected classes, and output mapping in the manifest.
3. Run the default test suite and real-model smoke.
4. Compare the new and current checkpoints on the same evaluation set.
5. Record behavior, latency, and memory differences in the pull request.
6. Update this document when model semantics or limitations change.
