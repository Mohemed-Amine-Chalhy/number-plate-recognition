# Deployment

## Supported baseline

The supported baseline is a single stateless Linux container running CPU inference. Scale replicas horizontally only after measuring memory, model-loading time, and safe concurrency. GPU is an optional separately pinned profile, not an automatic production default.

Do not deploy until the [model release gate](models.md#production-release-gate) and licensing decision are complete.

Local `development` container operation with the three checksum/contract-declared bundled artifacts is supported. A production release is intentionally blocked on model provenance, rights, explicit approval, and representative quality evidence that cannot be inferred from repository code. The remaining ingress, identity, telemetry, retention, and capacity controls in this guide are deployment-environment responsibilities.

## Build and run locally

```bash
docker build --tag number-plate-recognition:local .
docker run --rm --publish 8501:8501 number-plate-recognition:local
```

Add `--env-file .env` after reviewing/copying `.env.example` when overrides are required. Alternatively, use the hardened local Compose service; it reads `.env` when present:

```bash
docker compose up --build
```

Open <http://localhost:8501>. The container runs as UID/GID 10001 and exposes only port 8501. Before Streamlit starts, `docker/entrypoint.sh` runs `doctor.py --models-only`, validating schema v2 and all three sizes/hashes; if `NPR_ENVIRONMENT=production`, it also enforces provenance/source/license/approval metadata. The current production setting therefore exits before serving traffic, by design.

The health check queries Streamlit's `/_stcore/health`, which establishes process liveness. Concrete model initialization and task/class subset validation remain lazy until first inference, so an orchestrator requiring semantic model-aware readiness needs a warm-up/readiness strategy.

For a stricter local exercise, supply a writable temporary mount while keeping the root filesystem read-only:

```bash
docker run --rm --read-only --tmpfs /tmp:rw,noexec,nosuid,size=512m --publish 8501:8501 number-plate-recognition:local
```

The shipped Compose service already applies a read-only root, 512 MiB `/tmp` tmpfs, dropped capabilities, and `no-new-privileges`. Adjust temporary memory only from measured peak use.

## Image requirements

- Pin the Python base image by supported version and, for releases, immutable digest.
- Synchronize only production dependencies from `uv.lock`.
- Exclude documentation, notebooks, local environments, Git/CI data, caches, secrets, tests, and legacy root scripts via `.dockerignore`.
- Run under an unprivileged UID/GID with no shell requirement at runtime.
- Use a read-only root filesystem where the platform permits it.
- Run the entrypoint artifact/policy preflight; add semantic model warm-up before declaring readiness when required.
- Generate an SPDX SBOM and scan the final image. CI currently blocks fixable high-or-worse container findings after its smoke test.
- Label the image with source revision, application version, and model manifest revision.

## Model delivery

The current image copies the manifest and three bundled weights into `/opt/app`, then verifies artifacts at entry. It copies only `images/README.md`; `.dockerignore` excludes all operator-provisioned images, so no vehicle photographs ship. This artifact arrangement is acceptable for local development only while approval is unresolved. A production release should choose one reviewed model-delivery approach:

1. Copy approved weights during a controlled image build, verifying hashes before the final layer.
2. Mount a read-only, versioned artifact volume.
3. Use an init step that downloads from an authenticated artifact store, verifies hashes, and then exposes a read-only directory.

The runtime forces `YOLO_OFFLINE=true` and `YOLO_AUTOINSTALL=false`; a missing artifact fails rather than invoking Ultralytics download/install behavior. If a reviewed init/build step provisions artifacts, it must finish before the offline app starts. Never download a model during a live request, use a mutable `latest` reference, or rely on framework auto-download. Keep the previous application image and model hashes available for rollback.

## Edge and identity

Streamlit should sit behind a production reverse proxy or managed ingress that provides:

- TLS and secure modern cipher configuration;
- authentication and, if needed, authorization;
- request-body, connection, and rate limits;
- security headers and access logs with privacy-safe fields;
- timeouts aligned with measured inference latency;
- denial-of-service protections appropriate to the exposure.

Do not expose the container directly to the public internet. If browser sessions are authenticated at the proxy, test websocket/session behavior and ensure identity headers cannot be supplied directly by clients.

## Health and rollout

Liveness answers whether the process can respond. Readiness answers whether configuration/policy, artifacts, and loaded semantic contracts are usable. The entrypoint covers policy/integrity and the shipped Streamlit probe covers liveness; neither triggers model initialization. Keep these states distinct.

Use a rolling or canary deployment:

1. Deploy by immutable image digest and model hashes.
2. Wait for readiness and warm-up.
3. Run a non-sensitive smoke fixture with an expected structured result.
4. Shift a small traffic fraction and watch failures, latency, memory, and detection distributions.
5. Promote only if release thresholds hold; otherwise roll back image and model together.

## Resource sizing

Benchmark on the exact instance type and image. Complete requests are serialized by a bundle-wide lock, so record queue time separately from vehicle/plate/character/total time, along with model load time, steady/peak memory, CPU saturation, and throughput at representative image sizes/cascade caps. Set CPU/memory requests and limits with measured headroom. An out-of-memory kill must not result in a partial success response.

## GPU profile

For NVIDIA deployment:

- maintain a separate image/profile with a tested driver, CUDA runtime, PyTorch wheel, and NVIDIA Container Toolkit matrix;
- pin all versions and record them in release notes;
- run the full model regression and load suite on the target GPU;
- expose only required devices and avoid privileged containers;
- define CPU fallback behavior explicitly—silent fallback can violate latency expectations.

Follow the current official PyTorch and NVIDIA compatibility guidance when creating the profile. Do not infer compatibility from a developer workstation.

## Production checklist

- All three schema-v2 entries have verified provenance, authoritative source, approved license status, explicit production approval, and linked quality evidence.
- Ultralytics/dependency, privacy, and security decisions complete.
- Immutable application and model identifiers recorded.
- Secrets injected and rotation tested; none present in image layers.
- TLS, authentication, request limits, upload limits, and rate limiting tested.
- Liveness, readiness, metrics, logs, alerting, and dashboards verified.
- Restore and rollback exercised.
- Dependency audit, SPDX SBOM, and image scan reviewed with no unresolved release-blocking finding.
- Representative performance and failure tests passed.
- Representative accuracy evidence came from an approved external labeled dataset; the synthetic real-model smoke was not treated as quality evidence.
- No unapproved local examples are packaged, and the owner has assessed whether removed photographs still reachable in Git history/remote caches require a coordinated purge and cache cleanup.
- On-call owner, runbook, and incident contacts published.
