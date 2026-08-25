# ADR-0007: Bounded agent runtime with external policy and human approval

- Status: Accepted
- Date: 2026-08-25
- Owners: Platform engineering case study

← [ADR index](README.md) · [Agentic architecture](../agentic-ai.md)

## Context

Gate-health triage benefits from assembling topology, device-health, and incident context before an
operator responds. A general chat agent with broad API or database access would make that flow quick
to prototype, but it would blur five boundaries: what the planner may request, which tenant/gate it
may inspect, which tools mutate state, whether an action was approved, and what survives for replay
or incident review.

The repository also needs a deterministic local path. Review and policy tests must not depend on a
hosted model, network availability, or non-repeatable prose.

## Decision

Implement the operations agent inside the modular control plane with these boundaries:

- expose a typed run/decision API and one versioned `gate_health_triage` intent;
- define `AgentPlanner` as a replaceable provider seam, with a deterministic offline planner as the
  implemented provider;
- have that intent select one fixed typed trajectory while retaining the objective only as operator
  context; the implemented planner does not interpret or decompose it;
- accept only typed plans whose tools and risk classifications match a closed server registry;
- derive organization and gate scope from authenticated run state, then re-check it at tool
  execution rather than trusting the objective or planner output;
- execute read-only topology, health, and incident tools automatically;
- derive health readiness from enabled configured cameras, exact camera/gate identity, online state,
  five-minute freshness, and one-minute future tolerance;
- pause before `create_incident` or `start_incident_investigation` and require an authenticated
  approve/reject decision with a reason;
- persist run, plan, steps, inputs/outputs, policy checks, trace/correlation identifiers, approval,
  failures, and audit events;
- require separate idempotency keys for run creation and human decision;
- commit an approved incident effect, completed step/run state, and audit events in one transaction;
- revalidate tenant/gate/resource health and duplicate-action preconditions in that transaction;
- return a persisted `running` trace, without replaying its pending reads, when the same actor retries
  the identical create key; automatic recovery requires an explicit lease owner;
- expose no shell, arbitrary network/database, camera-secret, or physical-actuation tool.

A future model-backed planner may replace the deterministic planner only behind the same tool,
policy, scope, approval, persistence, and evaluation boundary.

## Consequences

### Positive

- Planner quality and operational authority can be tested and changed independently.
- Read-only context gathering is automated without silently granting write authority.
- A reviewer sees the proposed effect, evidence, policy checks, and skipped branch before deciding.
- Retries and human decisions are reconstructable from durable structured state.
- The same deterministic fixtures support API, policy, UI, and trajectory evaluation.

### Negative

- The first intent is deliberately narrow and does not provide open-ended conversation.
- Every new tool needs schema, risk, policy, UI, idempotency, audit, and negative-scope work.
- Human approval adds latency even when the proposal is obvious.
- SQLite persists prototype traces but does not provide distributed leases or automatic recovery.
- A model-backed planner still requires adversarial and trajectory evaluation before shadow use.

## Alternatives considered

- General-purpose assistant with direct API/SQL access: rejected because prompt text would sit too
  close to authority and tenant data.
- Deterministic background automation without a visible plan: rejected because it would be harder to
  review, extend, and compare with a future planner.
- Approve an entire plan before any evidence is read: rejected because approval would not cover the
  actual observations or final tool arguments.
- Give all tools the same autonomy level: rejected because a read and an incident mutation have
  different consequences.
- Deploy the runtime as a separate service immediately: rejected until workload, scaling, or team
  ownership demonstrates an extraction boundary.

## Validation

- Assert exact plan/tool ordering and risk classification.
- Exercise healthy, degraded, missing-health, and existing-incident branches.
- Prove no incident mutation occurs before approval or after rejection.
- Replay create/decision idempotency keys and assert at most one effect.
- Race identical/different create and approve/reject requests; bind each key to the winning request
  and retain exactly one terminal decision/effect.
- Inject a failure after an approved domain insert and assert the effect rolls back while a terminal
  failure remains inspectable and non-replayable.
- Change incident/health state during approval and assert stale proposals fail without side effects.
- Reject wrong-gate/phantom camera reports, require coverage for every enabled configured camera,
  classify stale/future evidence, and ensure out-of-order history cannot regress current state.
- Force a partially persisted running read trajectory, retry its original create idempotency key,
  and verify that pending reads do not replay without recovery ownership.
- Attempt cross-organization and wrong-gate reads/writes.
- Verify structured policy, failure, human-decision, and audit data survives persistence/reload.
- Keep manual incident workflows available when the agent path fails.
