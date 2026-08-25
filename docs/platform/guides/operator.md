# Operator guide

← [Platform documentation index](../README.md)

## Purpose and scope

This guide covers the command-center/security-operator workflow. The current console is a
**prototype** using Live API, Partial API, or deterministic Reference scenario resources. Follow
local site safety and gate procedures first; the platform does not replace an attendant's physical
verification, intercom, barrier safety loop, or emergency process.

## Start-of-shift checklist

1. Confirm the tenant, campus, locale, time zone, and signed-in role.
2. Read the source indicator:
   - **Live API**: every expected resource loaded from the API.
   - **Partial API**: only some resources loaded; identify which panels may use reference records.
   - **Reference scenario**: no operational resource is live. Its internal source-state key is
     `demo`; never use reference records for real access decisions.
   - **Offline fallback**: the browser is offline and the deterministic reference snapshot is active.
3. Review gate states and queue estimates.
4. Review degraded/offline devices and the age of their last health signal.
5. Review open critical/warning incidents and current owner.
6. Review agent runs awaiting a human decision; confirm their gate and evidence freshness.
7. Confirm the local manual fallback and escalation contact for each active gate.

## Triage order

Use this order unless local emergency procedure overrides it:

1. physical safety/emergency at a gate;
2. critical incident or suspicious passage;
3. vehicle currently waiting for a decision;
4. growing queue or degraded gate;
5. pending future access request;
6. routine administration/analytics.

## Review an arrival

Open the passage/arrival and read the evidence in this order:

1. gate, direction, and occurred time;
2. data freshness and camera/edge health;
3. recognition status (`recognized`, `uncertain`, or `unreadable`);
4. plate candidate and detection/recognition confidence;
5. model/source/evidence label;
6. matching active grant, time window, subject, and gate/site scope;
7. prior decisions and any linked incident;
8. policy recommendation and reason.

Confidence is supporting evidence, not permission. Check that the grant is active now and applies to
the site/gate. When evidence is stale, contradictory, unreadable, or missing, use review/manual
procedure rather than guessing.

## Record a decision

| Outcome | Use when | Required record |
| --- | --- | --- |
| Allow | Identity/access context is confirmed under current policy | Reason/source and linked grant where applicable |
| Review required | More information or another actor is needed | Specific missing/contradictory element and owner |
| Deny | Current policy clearly denies access | Bounded reason; follow site communication procedure |
| No match | No eligible grant/record matches | Search method and next safe step |

The API records the authenticated actor and time. Do not use a colleague's session or put sensitive
free-form details into the reason field.

If a future deployment integrates barrier commands, confirming a decision and executing a physical
command remain separate actions. Confirm the lane is safe and never replay an old command.

## Handle a pending request

1. Confirm requester/host and subject name.
2. Check site, preferred gate, start/end time, purpose, and plate if used.
3. Check for overlap, duplicate, revoked subject/vehicle, or an incident note.
4. Approve or reject through the decision action, with a reason.
5. Verify that an approved request produced the expected grant and event.

Hosts can submit; only roles with access-decision permission approve/reject.

## Create and manage an incident

Create an incident when work must persist beyond one immediate decision, for example:

- possible tailgating;
- recurring camera packet loss;
- repeated unmatched arrivals;
- incorrect topology/policy affecting a gate;
- evidence or service outage requiring follow-up.

Include gate/passage link, concise title, severity, observable facts, and current owner. Do not turn an
uncertain model result into an accusation. Use status:

```mermaid
stateDiagram-v2
    [*] --> Open
    Open --> Investigating: owner accepts
    Investigating --> Resolved: outcome and follow-up recorded
    Resolved --> Investigating: verified recurrence
```

## Run bounded gate-health triage

The Agent workspace gathers gate, latest device-health, and open-incident context through read-only
tools. It does not control a camera or barrier. Use it as a traceable triage aid:

1. confirm the console is using **Live API** before expecting a mutation; Reference scenario runs
   are computed demonstrations;
2. select the gate and enter a narrow objective as operator context; the deterministic planner does
   not interpret it or change the fixed `gate_health_triage` trajectory;
3. start the `gate_health_triage` run and verify its organization, gate, planner/policy versions,
   and trace identifier;
4. review all three read steps, their structured observations, freshness, and policy checks;
   configured-camera health needs attention when no enabled camera is configured or a required
   report is missing, older than five minutes, more than one minute in the future, invalid, or
   non-online;
5. confirm whether the proposed consequential step targets an existing incident or a new incident;
6. inspect the exact incident inputs, then approve or reject with a reason;
7. after approval, confirm the incident/result in Operations; after rejection, confirm that no
   incident state changed.

The workspace's **Evidence coverage** percentage measures read/tool/check coverage from zero and is
**Unavailable** without read steps; it is not the probability that the proposed action is correct.
It does not replace reviewing the underlying observations. Unknown tool, risk, or policy data fails
closed and removes decision controls.

The runtime rechecks health, incident state, scope, and duplicates at commit time. An approval can
therefore finish as a safe failure when the world changed during review; read the failure and refresh
current state instead of treating the earlier proposal as authority.

An `awaiting_approval` run has not executed the proposed incident action. A `completed` run may mean
either an approved action succeeded or healthy evidence required no action; read the steps and audit
events rather than inferring from the status alone. A `failed` run is not permission to assume the
proposal was safe—use the manual incident workflow and preserve the run ID for diagnosis.

If a request times out and the returned run remains `running`, retrying the same create key only
returns that persisted trace; it does not replay pending reads. Use the manual path and escalate the
run for recovery rather than inventing a new key to force execution.

Do not approve from the plan summary alone. The approval authorizes the concrete staged tool call in
that run, including its selected gate and incident arguments. See
[Agentic AI architecture and operations](../agentic-ai.md).

## Degraded operation

### Camera degraded/offline

- Confirm which role/stream is affected and last good frame/heartbeat.
- Check whether another approved camera/attendant procedure covers the lane.
- Do not interpret absence of detections as absence of vehicles.
- Create/assign an incident if the fault exceeds the shift threshold.

### API or WAN unavailable

- Treat the console as stale/unavailable even if cached layout remains visible.
- Use the documented local manual procedure.
- Record decisions in the approved offline log if one exists; reconcile after service recovery with
  clear original timestamps/source.
- Never use deterministic demo records as offline operational records.

### AI worker delayed

- Check queue age and passage status.
- Do not repeatedly resubmit the same capture; idempotent retry belongs to the service.
- Use manual plate/identity verification when the waiting-time threshold is reached.

## End-of-shift handoff

- unresolved critical/warning incidents and owners;
- gates/devices in degraded, offline, maintenance, or disabled state;
- queue/backlog and oldest pending passage/request;
- temporary manual procedures in effect;
- configuration/model change or rollback during the shift;
- agent runs awaiting approval or failed, plus trace IDs and responsible reviewer;
- any offline records that still require reconciliation.

Do not export or message screenshots with real evidence unless the approved incident workflow
requires it.

## Quick “do / do not”

| Do | Do not |
| --- | --- |
| Verify source freshness and grant window | Approve from confidence alone |
| Record a concise reason and owner | Put passwords or unnecessary personal details in notes |
| Follow manual fallback when dependencies are stale | Treat no response/no detection as allow |
| Link incidents to gate/passage | Duplicate an arrival for every retry/frame |
| Review agent tool evidence and exact proposed effect | Treat an agent summary as evidence or approval |
| Reject a stale/wrong-scope proposal and record why | Retry with a new key merely because the response was slow |
| Escalate physical safety first | Assume a delivered future command was executed |

## Related documents

- [Host guide](host.md)
- [Administrator guide](admin.md)
- [Troubleshooting](../troubleshooting.md)
- [Security and privacy](../security-and-privacy.md)
- [Data workflows](../data-and-workflows.md#arrival-and-decision-workflow)
- [Agentic AI architecture and operations](../agentic-ai.md)
