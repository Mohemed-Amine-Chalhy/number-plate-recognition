# Operations

## Service-level indicators

Define objectives from real requirements before launch. At minimum observe:

- request success and rejected-input rates;
- inference error and timeout rates by stage;
- p50/p95/p99 `queue`, `vehicle`, `plate`, `character`, and `total` latency;
- request concurrency, CPU/GPU use, memory, and restarts;
- Streamlit liveness, deployment-defined model-aware readiness, and model-load duration;
- model and application versions;
- aggregate no-detection and low-confidence rates;
- input size/dimension distributions in coarse buckets.

Do not use raw images, full plate text, filenames, or user-supplied metadata as metric labels or routine log fields. Avoid high-cardinality labels.

The pipeline returns queue and stage timings, but the current UI caption and completion log expose only total time. A production telemetry adapter may export the other numeric fields with bounded labels; it must not export image or reconstructed-plate content. The Streamlit `/_stcore/health` endpoint is process liveness, not proof that the lazily loaded models satisfy their task/class contracts.

## Logging

Use structured logs with timestamp, severity, service version, model hashes or short approved identifiers, stage, duration, outcome, and a generated request/correlation ID. Log validation categories rather than user content. Exceptions belong in restricted telemetry with credentials and sensitive values scrubbed.

Full plate values and image bytes are disabled by default. If a narrowly justified investigation requires sensitive capture, use a separately approved, access-controlled, encrypted, time-limited workflow with audit logs and automatic deletion—not a debug flag left enabled.

## Alerts

Page or notify an owner for sustained unavailability, model-aware readiness failure, model checksum/load failure, high error/timeout rate, restart loop, resource exhaustion, or a security signal. Alert separately on sustained queue time because complete requests are serialized within a process. Use lower-urgency alerts for latency degradation and significant shifts in aggregate confidence/no-detection distributions. Model drift alerts require investigation; they do not authorize automatic retraining or deployment.

## Release record

Every production release should record:

- source revision and immutable container digest;
- dependency lockfile revision and SBOM;
- each model artifact ID and SHA-256;
- the schema-v2 manifest and its `task`, `expected_classes`, `output_map`, provenance/license status, and explicit approval fields;
- configuration/policy version, excluding secret values;
- code-test, AppTest, real-model, security-scan, and representative model-evaluation results;
- approved external evaluation-dataset identity/provenance and confirmation that no local example photographs entered the release image;
- the recorded owner decision for historical Git/cache copies of removed photographs;
- approvers, rollout time, and rollback target.

## Rollback

Application code, manifest, and model artifacts form one compatibility unit. Roll them back together unless compatibility is proven.

1. Stop traffic promotion and preserve privacy-safe diagnostic context.
2. Route traffic to the last known-good immutable image and approved model set.
3. Verify readiness and the fixed smoke fixture.
4. Confirm error, latency, and resource indicators recover.
5. Record the incident and invalidate a bad artifact in the registry; do not overwrite it.

Exercise this procedure before first production use and periodically afterward.

The current manifest is suitable for checksum-verified development but intentionally fails `uv run python scripts/doctor.py --production`: every role still has a null source, unverified provenance/license status, and `production_approved: false`. Do not treat a green local hash check as a production approval or change those fields without the corresponding external evidence.

## Incident playbooks

### Model checksum or load failure

- Keep the replica unready.
- Confirm manifest and mounted artifact versions without logging secrets.
- Restore the last approved immutable artifact; never disable verification.
- Investigate artifact-store integrity and access logs.

### Elevated inference errors or memory exhaustion

- Limit or drain traffic to affected replicas.
- Compare input-size buckets, application/model revision, and resource metrics.
- Roll back when correlated with a release.
- Preserve only privacy-approved diagnostic material.

### Suspected data exposure

- Restrict access and stop the source of capture without destroying evidence.
- Follow the organization's incident-response and notification process.
- Identify affected storage, logs, backups, users, fields, and retention windows.
- Rotate exposed credentials and remove unauthorized data through an audited process.

### Accuracy regression or drift

- Do not silently lower confidence thresholds.
- Compare against the versioned evaluation suite and slice metrics.
- Disable consequential downstream use if safety or rights could be affected.
- Retraining requires the full data, model, privacy, licensing, and release review.

## Capacity and availability

Run load tests with safe synthetic/authorized images at representative dimensions and cascade caps. Size workers based on measured memory per loaded model plus peak decode/inference buffers. A bundle-wide lock serializes each complete three-stage request in a process; the application reports queue wait but does not itself impose a queue-depth limit or service-busy timeout. Bound admission and queueing at the ingress/platform, or add a separately tested backpressure layer, so overload becomes an explicit rate-limit/service-busy response instead of unbounded waiting or process failure.

The service has no intended durable user application data. Back up deployment configuration, signed release records, approved manifests/model cards, and immutable model artifacts. Test restoration. Exclude Streamlit/session/upload buffers and temporary inference data from backups. Operator-provisioned Approved examples are governed deployment data: do not back them up unless the recorded authority and retention policy explicitly require it.

## Dependency and model maintenance

- Review dependency and base-image updates on a regular cadence.
- Re-lock, scan, test, and benchmark; never auto-promote an update.
- Re-evaluate models when the target domain, camera population, plate standard, or preprocessing changes.
- Define deprecation dates and owners for old artifacts.
- Keep rollback artifacts according to the release-retention policy.

## Shutdown and deletion

Graceful shutdown should stop accepting new work, finish or cancel bounded in-flight requests, release model/device resources, and remove temporary files. Validate that platform snapshots, crash dumps, access logs, and observability exports comply with the documented privacy retention policy.

Removal from a current checkout/container does not remove data from Git history, clones, mirrors, CI artifacts, or caches. Any history rewrite/cache cleanup must be an owner-approved coordinated operation with downstream communication and verification, not an ad hoc maintenance command.
