# Privacy, security, and threat model

## Data classification and default behavior

Vehicle images and recognized plate values can identify or be linked to people, locations, and movements. Treat them as sensitive personal data where applicable. The service is designed to process data in memory for the current request and not intentionally persist uploads or results.

No vehicle photographs ship in the current tree or container. The `images/` directory contains only [operator guidance](../images/README.md); any locally provisioned Approved example is deployment data and must have recorded authority, purpose, retention, and deletion ownership.

The deployer is responsible for establishing a lawful basis, clear purpose, notice, access controls, retention/deletion policy, data-subject handling, processor agreements, and any required impact assessment for the applicable jurisdiction. This documentation is not legal advice.

## Data flow

1. In the required production topology, a browser sends an authenticated, TLS-protected image upload. The development server does not supply that perimeter.
2. The edge enforces body/rate/admission limits and forwards the request to Streamlit.
3. The user explicitly submits the upload form. The application enforces batch count, exact encoded bytes and decoded pixels, then performs local bounded-cascade model inference.
4. The result is returned to that session.
5. Project code creates no durable upload/result record; Streamlit, browser, proxy, process-memory, and platform buffer lifecycles still require deployment verification. Routine application telemetry contains no raw image or full plate value.

An operator-selected Approved example enters at step 3 from the local filesystem rather than the browser and receives the same byte/pixel/decoder checks. The application does not create or manage those files.

Any object storage, database, external API, analytics SDK, error-reporting attachment, or training capture is a material architecture and privacy change requiring review and documentation.

## Threats and controls

| Threat | Required controls | Residual risk |
| --- | --- | --- |
| Malicious/corrupt image exploits a decoder | Allowlisted formats, magic/decoder validation, current libraries, encoded/pixel/dimension limits, isolated non-root container, read-only filesystem, time/resource bounds | Native image/ML dependencies remain a large attack surface. |
| Decompression bomb or inference denial of service | Framework/widget and exact byte limits, pixel/count/cascade limits, longest-side inference resize, ingress rate/admission limits, timeouts, queue bounds, resource limits | Valid inputs can still be expensive; the application serializes complete requests but does not bound the surrounding request queue. |
| Malicious model artifact executes code | Controlled source, immutable SHA-256, restricted build/runtime, forced offline/no-auto-install behavior, no request-time downloads, artifact review; prefer safer serialization after validation | Pickle-based `.pt` loading can execute code; `ULTRALYTICS_SAFE_LOAD` is not enabled because compatibility is unvalidated. |
| Unauthorized image or result access | TLS, authentication, least privilege, session isolation, no public container endpoint, tested proxy headers | Streamlit session/proxy configuration errors may leak access. |
| Sensitive data leaks to logs/metrics | No payload/full-plate logging, structured allowlisted fields, redaction, restricted telemetry, short retention | Stack traces and third-party defaults require continuous review. |
| Fabricated/adversarial plate causes wrong result | Confidence reporting, no automatic consequential action, representative evaluation, human review, abuse monitoring | ML output remains probabilistic and attackable. |
| Dependency or container compromise | Locked dependencies, review/scan/SBOM, signed immutable images where supported, minimal runtime, patch cadence | Scanners cannot find every issue. |
| Model theft or unauthorized redistribution | Access-controlled artifact store, least-privilege credentials, distribution/license review, audit logs | A runtime able to load weights can potentially expose them. |
| Cross-user data mixing | No global request state, per-session results, concurrency tests, cache only immutable models/config | Framework or application defects can violate isolation. |
| Physical surveillance misuse | Purpose limitation, authorization, minimization, access audit, no covert retention or face analysis | Governance is deployment-specific and cannot be enforced by model code alone. |

## Input policy

- Accept only the documented JPEG/PNG formats after successful decoding; do not trust filename extensions or MIME headers alone.
- Reject empty files, invalid dimensions, excessive encoded bytes/pixels, excessive upload count, and truncated/corrupt data.
- Reduce any displayed upload name to a bounded sanitized basename and render it as text, never as HTML or a path.
- Apply EXIF orientation, convert to RGB/BGR pixels, and do not return or intentionally persist source metadata such as EXIF location data.
- Clamp all coordinates before slicing, reject empty crops, and bound output rendering.
- Never interpolate user-controlled values into shell commands, paths, HTML, or log templates.
- Keep local Approved examples and evaluation datasets outside Git/container layers and follow [the image policy](../images/README.md).

## Authentication and network

The Streamlit process provides no application-specific authentication/authorization and is not the production identity perimeter. Deploy behind a trusted TLS ingress or identity-aware proxy. Prevent clients from bypassing it, sanitize identity headers, use secure cookies, and verify websocket behavior. Apply least-privilege network policies: the running app should not need general outbound internet access.

## Secrets and artifacts

No secret belongs in Git, an image layer, notebook output, model metadata, or frontend state. Use the platform secret store, short-lived artifact credentials, rotation, and audit logs. Secret scanning is required but is not a substitute for review.

## Retention

Project code configures no durable retention for uploads or recognized values. That is not a guarantee of immediate erasure: browser/Streamlit sessions, proxies, process memory, temporary storage, crash dumps, and platform services may keep buffers or copies. Verify and document their lifecycles. Define explicit short retention for edge access logs, application logs, security events, backups, and crash dumps, including who can access each store, why, encryption, deletion behavior, and whether backups honor deletion.

Never reuse production uploads for training, evaluation, debugging, or demonstrations without an independently documented lawful basis/authority, minimization, access control, and retention process.

Deleting a photograph from the current tree prevents it from shipping in future checkouts/images, but does not erase prior Git objects, forks, clones, mirrors, CI artifacts, or remote caches. Before public release, the repository owner must assess whether the previously removed plate photographs require a coordinated history purge and cache cleanup. Do not perform a casual rewrite: it is disruptive, requires coordination, and needs a separately reviewed retention/incident plan.

## Security verification

Before launch and after material changes:

- review the data flow and trust boundaries;
- test invalid images, decompression bombs, high concurrency, timeouts, and session isolation;
- scan source, secrets, dependencies, container, and model delivery paths;
- verify non-root/read-only runtime and restricted egress;
- verify the container contains no local Approved-example photographs and the runtime fails rather than auto-downloading models;
- exercise incident response, deletion, artifact rollback, and credential rotation;
- record the owner decision on historical Git/cache copies of removed photographs;
- perform an application security review or penetration test appropriate to exposure.

Report suspected vulnerabilities using [SECURITY.md](../SECURITY.md), not a public issue.
