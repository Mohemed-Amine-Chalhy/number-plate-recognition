# Architecture decision records

← [Platform documentation index](../README.md)

ADRs record consequential choices and trade-offs. “Accepted” means accepted for this case-study
architecture or prototype; it does not imply a production deployment has validated the choice.

| ADR | Status | Decision |
| --- | --- | --- |
| [ADR-0001](0001-modular-control-plane.md) | Accepted | Modular control plane with separate edge and AI workers |
| [ADR-0002](0002-separate-recognition-authorization.md) | Accepted | Recognition observations are separate from authorization and actuation |
| [ADR-0003](0003-sqlite-prototype-postgresql-production.md) | Accepted | SQLite for the self-contained prototype; PostgreSQL for replicated production |
| [ADR-0004](0004-edge-owned-camera-connectivity.md) | Proposed target | Outbound site edge agent owns ONVIF/RTSP connectivity and secrets |
| [ADR-0005](0005-white-label-deterministic-demo.md) | Accepted | White-label configuration plus explicitly labeled deterministic demo mode |
| [ADR-0006](0006-at-least-once-events.md) | Proposed target | At-least-once event delivery with idempotent consumers and inbox/outbox |
| [ADR-0007](0007-bounded-agent-runtime.md) | Accepted | Bounded planner/tool runtime with external policy, durable traces, and human-approved incident actions |

## Status definitions

- **Proposed target**: designed but not implemented end to end.
- **Accepted**: current architecture/prototype should follow it.
- **Superseded**: retained for history with a link to the replacing ADR.
- **Rejected**: considered and intentionally not selected.

## Change rule

Do not silently rewrite the decision of an accepted ADR after implementation. Add a superseding ADR
when constraints or evidence change. Small clarifications and links may be added without changing the
decision.
