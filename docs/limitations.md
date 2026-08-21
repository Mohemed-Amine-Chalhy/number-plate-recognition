# Limitations and responsible use

## Intended use

The application demonstrates image-based detection and transcription of Moroccan number plates. It may support operator review, dataset evaluation, or a carefully governed workflow after validation.

## Not intended for

- autonomous opening or closing of a physical barrier;
- deciding access, penalties, law-enforcement action, identity, ownership, or legal status;
- covert or indiscriminate surveillance;
- tracking people or vehicles across time or locations;
- facial recognition or inference of protected/sensitive attributes;
- general OCR or reliable recognition outside the evaluated Moroccan domain;
- safety-critical, emergency, or high-availability control.

## Accuracy limitations

Output is probabilistic and can be incomplete or incorrect. Likely failure conditions include small/distant plates, motion blur, glare, darkness, rain/dust, oblique viewpoints, occlusion, damaged or nonstandard plates, multiple overlapping vehicles, unusual fonts, image compression, adversarial signs/prints, and domains absent from training data.

A string that matches the configured one-letter plate-pattern heuristic is not proof of a correct or regulatorily valid plate. Confidence values are not guaranteed probabilities unless calibrated on representative data. Overlap suppression, de-duplication, and pattern classification must not hide uncertainty or create missing characters.

No production accuracy claim is currently supported because the bundled custom weights lack a completed provenance and evaluation record. See [models](models.md).

The real-model smoke uses a deterministic blank synthetic frame. It establishes deserialization/API compatibility and a narrow no-false-positive assertion only; the repository ships neither representative photographs nor a labeled accuracy dataset. Production quality claims require separately approved external data.

## Domain and fairness

Performance can differ across geography, camera type, environment, vehicle type, plate age/style, and other slices. Evaluate intended conditions and relevant subgroups with authorized data. Aggregate accuracy can conceal systematic failures. Operational drift must trigger review, not unreviewed automatic retraining.

## Privacy and human oversight

Plate images and values can reveal movements and associations. Minimize collection, restrict access, avoid retention, and provide appropriate transparency and redress. A trained operator must verify outputs before consequential use, and the workflow must make uncertainty and failures visible.

## Operational limitations

- CPU inference prioritizes portability over maximum throughput.
- GPU behavior depends on a specifically tested driver/CUDA/PyTorch matrix.
- Streamlit is the presentation adapter, not a complete public security perimeter.
- In-process model caching requires concurrency and memory testing. A bundle-wide lock serializes complete requests; queue wait is reported but application-level queue depth and timeouts are not bounded.
- Cascade caps, confidence thresholds, plate deduplication, character overlap suppression, and longest-side resizing deliberately trade work bounds against possible missed detail/detections; validate them on the intended domain.
- `.pt` artifacts are sensitive executable inputs and must be trusted and checksum-verified.
- Offline/no-auto-install settings prevent surprise runtime fetching but do not make pickle-backed `.pt` deserialization safe; the optional Ultralytics safe-load mode has not been compatibility-validated.
- Statelessness does not prevent proxies, platforms, or telemetry systems from retaining data; those layers require separate configuration.

## Licensing and artifact limitations

The MIT repository license does not grant rights to third-party software, base models, weights, or datasets. The custom weights' provenance and redistribution rights are incomplete. The Ultralytics AGPL/Enterprise choice must be reviewed before production or commercial use. See [third-party notices](../THIRD_PARTY_NOTICES.md). This is not legal advice.
