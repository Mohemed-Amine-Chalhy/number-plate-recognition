# ADR-0006: At-least-once delivery with idempotent consumers

- Status: Proposed target
- Date: 2026-08-23
- Owners: Platform and edge/worker engineering

← [ADR index](README.md) · [Event envelope](../data-and-workflows.md#event-envelope)

## Context

Edge agents, object uploads, brokers, AI workers, and the control plane can fail between receiving,
committing, and acknowledging work. Claiming network-wide exactly-once delivery would hide these
failure windows. Lost captures/decisions are unacceptable; duplicates are manageable when identities
and side effects are explicit.

## Decision

- Deliver edge/job/result/events at least once.
- Assign stable message/capture/job/event IDs plus edge boot ID/sequence.
- Make consumers idempotent with inbox/deduplication state and uniqueness constraints.
- Publish committed domain changes through a transactional outbox.
- Keep event feeds append-oriented with a monotonic cursor.
- Place media in object storage; messages carry immutable references/checksums.
- Expired physical commands are never replayed, even if an at-least-once channel redelivers them.

## Consequences

### Positive

- Honest failure semantics and recoverable retries.
- Durable offline store-forward and worker retry.
- Easier replay/read-model construction with explicit identities.

### Negative

- Every consumer must design idempotent side effects.
- Inbox/outbox retention and monitoring add storage/operations.
- Global event sequence gaps may exist in tenant-filtered views and need clear client semantics.

## Alternatives considered

- Best-effort fire-and-forget: rejected because outages lose operational evidence.
- Distributed exactly-once claim: rejected as brittle/unverifiable across database, broker, edge, and
  object storage.
- Put image bytes in broker messages: rejected for broker size/backpressure and media lifecycle.

## Validation

- Inject duplicate, delayed, reordered, and post-commit/pre-ack failures.
- Verify one canonical observation/decision/event side effect per idempotency identity.
- Reconnect an edge spool and confirm expired commands are discarded rather than replayed.
