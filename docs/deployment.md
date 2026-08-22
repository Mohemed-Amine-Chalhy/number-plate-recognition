# Deployment

The supported baseline is a single stateless Linux container running CPU inference. GPU deployment and horizontal scaling are optional extensions that should be driven by measurements.

## Build and run

```bash
docker build --tag number-plate-recognition:local .
docker run --rm --publish 8501:8501 number-plate-recognition:local
```

For reviewed environment overrides:

```bash
docker run --rm --env-file .env --publish 8501:8501 number-plate-recognition:local
```

The Compose profile reads `.env` when present:

```bash
docker compose up --build
```

Open <http://localhost:8501>. The image runs as UID/GID 10001 and exposes port 8501. Before Streamlit starts, `docker/entrypoint.sh` validates the model manifest and verifies all three checkpoint sizes/hashes.

## Container properties

The shipped image is designed to:

- install only locked runtime dependencies;
- run as an unprivileged user;
- work with a read-only root filesystem and writable `/tmp` tmpfs;
- drop Linux capabilities and set `no-new-privileges` in Compose;
- keep Ultralytics offline with auto-install disabled;
- exclude source-control data, notebooks, tests, caches, local environments, and secrets from the build context;
- expose a process health check at `/_stcore/health`.

Exercise the read-only profile directly with:

```bash
docker run --rm --read-only --tmpfs /tmp:rw,noexec,nosuid,size=512m \
  --publish 8501:8501 number-plate-recognition:local
```

Adjust temporary memory only after measuring peak decode and inference use.

## Models

The current image copies the manifest and three pinned checkpoints into `/opt/app`. The entrypoint verifies their size and SHA-256 before launching Streamlit; the adapter verifies task/class semantics when each model initializes.

For a deployment that manages checkpoints separately, mount a read-only versioned artifact directory and point `NPR_MODEL_DIR`/`NPR_MODEL_MANIFEST` at it. Keep the app offline and finish provisioning before the process starts.

## Health and warm-up

`/_stcore/health` answers whether Streamlit can respond. Models load lazily on the first recognition request, so liveness does not prove that every checkpoint has initialized successfully.

An orchestrated deployment can add model-aware readiness by running a deterministic warm-up request before admitting traffic. Keep liveness and readiness separate so a temporary model initialization failure does not hide process diagnostics.

## Network boundary

For internet exposure, place Streamlit behind a reverse proxy or managed ingress that provides TLS, authentication, body/rate limits, websocket support, and timeouts aligned with measured inference latency. Prevent clients from reaching the container around that boundary.

The application already enforces exact file bytes, decoded pixels, batch count, resize, and per-stage cascade limits. Ingress limits should reject excess work before it reaches the Python process.

## Capacity

Complete requests are serialized by a bundle-wide lock. Benchmark the exact instance type and record:

- model load time;
- p50/p95 queue and stage/total latency;
- steady and peak memory;
- CPU saturation;
- throughput at representative image sizes and cascade limits.

If queue time dominates, add isolated worker processes or replicas rather than sharing one model bundle concurrently. Account for the full model memory cost in every process.

## GPU profile

For NVIDIA hardware:

- maintain a separate image/profile with pinned driver, CUDA, PyTorch, and container-toolkit versions;
- run the complete test, real-model, and evaluator suites on the target GPU;
- expose only required devices and avoid privileged containers;
- define whether a missing GPU should fail startup or fall back to CPU.

Silent fallback can hide a large latency regression, so make device selection visible in results and logs.

## Deployment verification

- Model manifest, sizes, hashes, task, and classes verify.
- Container starts as non-root with the intended read-only/tmpfs settings.
- Liveness and optional model warm-up pass.
- Upload/body/rate limits behave as configured.
- Queue, inference, memory, and restart metrics are observable.
- A rollback keeps application code, manifest, and checkpoints as one tested unit.
- Representative latency and recognition metrics meet the deployment's target.
