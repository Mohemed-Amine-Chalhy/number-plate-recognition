# ADR-0002: Separate recognition, authorization, and actuation

- Status: Accepted
- Date: 2026-08-23
- Owners: Product, platform, and site-safety design

← [ADR index](README.md) · [Data workflows](../data-and-workflows.md)

## Context

The recognition pipeline estimates plate text and confidence. Entry authority also depends on an
active grant, site/gate/time scope, device/data freshness, local policy, and sometimes an operator.
A high-confidence plate can still be unauthorized; a valid visitor can have an unreadable plate.
Physical actuation has additional lane-safety and replay constraints.

## Decision

Represent three distinct records/actions:

1. **Recognition observation**: what a camera/model reported, with source/version/confidence.
2. **Authorization decision**: allowed/review/denied/no-match, with reason/source/actor.
3. **Physical command**: optional future expiring request to an edge controller with acknowledgement.

No confidence threshold directly opens a barrier. Missing/stale dependencies do not imply approval.
The initial pilot contains no automated physical command.

## Consequences

### Positive

- Explainable decisions and model comparison without rewriting history.
- Safer degraded behavior and independent policy evolution.
- Manual correction does not falsify the original model output.
- Actuator replay/acknowledgement can be secured separately.

### Negative

- More records and UI states than a binary allow/deny system.
- Passage correlation and policy recommendation require explicit implementation.
- Operators need clear reason hierarchy to avoid added cognitive load.

## Alternatives considered

- `if confidence > threshold: open`: rejected as unsafe and conceptually incorrect.
- Store only corrected plate/decision: rejected because it destroys model evidence.
- Put authorization inside the AI worker: rejected because worker/model lifecycle is not policy
  authority.

## Validation

- Contract tests prove adding recognition creates no authorization decision.
- Role tests prove edge/worker identity cannot make an operator-only decision unless explicitly
  granted a policy capability.
- Shadow pilot compares recommendations with existing procedure before assisted use.
