# ADR-0001: Modular control plane with separate edge and AI workers

- Status: Accepted
- Date: 2026-08-23
- Owners: Platform engineering case study

← [ADR index](README.md) · [Architecture](../architecture.md)

## Context

The product spans tenant/topology CRUD, access workflows, passage/incident state, camera LAN
connectivity, high-dependency GPU/CPU inference, and browser presentation. Splitting every domain into
a microservice would add deployment, schema, tracing, and transaction complexity before scale/team
boundaries are measured. Running cameras/models inside the public API would couple incompatible
network placement, dependencies, startup time, and failure modes.

## Decision

Build the control plane as a modular monolith with one transactional database at first. Keep modules
and repository contracts explicit. Deploy these components separately:

- site edge agent, because it lives on the camera LAN and must buffer offline;
- AI worker, because it owns model/GPU lifecycle and scales independently;
- media gateway/object storage, because continuous media is not an HTTP CRUD concern;
- static console, because browser delivery and API runtime are separable even when served together
  in the local prototype.

## Consequences

### Positive

- Simple transactions for request/grant/passage/decision/event state.
- Faster vertical-slice development and debugging.
- No ML/camera dependency in the public API process.
- Clear future extraction seams based on measured load/ownership.

### Negative

- Control-plane modules share deployment/database failure domain.
- Requires discipline to prevent cross-module imports and “big ball of mud.”
- Independent module scaling is not available until extraction/read-model work.

## Alternatives considered

- Microservice per domain: rejected as premature operational complexity.
- Single all-in-one process including Streamlit, API, models, and cameras: rejected for dependency,
  security, scaling, and edge-network reasons.
- Serverless function per route: rejected for model/camera connection lifecycle and local/offline
  requirements.

## Validation

- Deliver one vertical slice: request → grant → passage → recognition → decision → event → console.
- Load-test API/database and worker separately.
- Extract only when a measured scaling, availability, or team-ownership constraint exists.
