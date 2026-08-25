# Agentic AI architecture and operations

← [Platform documentation index](README.md)

## Purpose

Campus Access includes a bounded operations agent for gate-health triage. The supported
`gate_health_triage` intent selects one fixed, typed five-step trajectory for a selected gate. The
request's objective is retained and displayed as operator context; the deterministic planner does
not interpret or decompose it. The runtime calls a small allowlist of domain tools, records what it
observed, and stops for a human decision before any consequential tool can run. This makes the agent
useful for gathering scattered operational context without allowing model confidence or a free-form
instruction to become an unreviewed campus action.

The agent complements two existing layers rather than replacing them:

- the ANPR pipeline produces perception evidence from an image;
- the control plane owns tenant-scoped operational truth and state transitions;
- the agent coordinates approved control-plane tools and leaves a reviewable execution trace.

The implemented planner is deterministic and offline. No LLM is required to run or test the
workflow. A future model-backed planner can propose within the same boundary, but it must not gain a
broader tool set, tenant scope, or approval authority by changing the planning implementation.

Terminology matters: the **operations agent** documented here plans and invokes control-plane tools.
The separately documented **site edge agent** is a target device process for camera connectivity and
offline buffering; it is not implemented by this runtime and is not an AI planner.

## Implementation status

| Capability | Repository status | Evidence or boundary |
| --- | --- | --- |
| `gate_health_triage` intent | **Implemented** | The intent selects one deterministic, fixed five-step gate-health plan; the objective remains context |
| Typed run and decision API | **Implemented** | `POST/GET /api/v1/agent/runs` and `POST /api/v1/agent/runs/{run_id}/decisions` |
| Read-only tool execution | **Implemented** | Gate, latest device-health, and open-incident tools execute through an allowlist |
| Camera-health evidence policy | **Implemented** | Readiness requires at least one enabled camera at the gate and fresh online coverage for each, with a five-minute max age and one-minute future tolerance; missing/stale/future/invalid/non-online evidence requires attention |
| Consequential tool staging | **Implemented** | Incident creation/investigation is proposed but cannot execute before a human decision |
| Human approve/reject decision | **Implemented** | An authenticated decision executes or terminates the staged action |
| Tenant and gate scope | **Implemented prototype** | Tool queries are constrained to the run's organization and selected gate |
| Idempotent run/decision requests | **Implemented** | Caller-provided keys prevent retry-created duplicate effects |
| Idempotent running-run replay | **Implemented** | Repeating the same create request returns the persisted `running` trace without executing pending reads |
| Durable trace and audit state | **Implemented prototype** | Plan, steps, policy checks, tool results, decision, failure, and audit metadata persist in SQLite |
| Machine-readable agent evaluation | **Implemented** | Six isolated deterministic scenarios report versioned JSON and fail non-zero on a regression |
| Model-backed planning | **Follow-up** | Evaluate a structured-output planner against the same policy/tool boundary before adoption |
| Crash recovery and distributed execution | **Follow-up** | No read step is resumed without explicit lease/recovery ownership; durable queue, leases, heartbeats, and recovery workers are not implemented |
| Production identity and storage | **Follow-up** | Replace demo tokens/SQLite with OIDC-derived scope and the production data topology |

`Implemented prototype` means the behavior is runnable and tested in this repository; it is not a
claim of a deployed, independently certified production system.

## The bounded agent loop

```mermaid
sequenceDiagram
    actor O as Security operator
    participant API as Control API
    participant A as Agent runtime
    participant P as Policy boundary
    participant T as Allowlisted tools
    participant C as Control-plane state

    O->>API: POST run(objective, gate, intent, idempotency key)
    API->>A: Authenticated organization, actor, and validated request
    A->>P: Validate organization, gate, intent, tool order, and risk
    P-->>A: Approved read-only plan
    A->>T: get_gate
    T->>C: Tenant- and gate-scoped query
    C-->>T: Gate state
    T-->>A: Gate observation
    A->>T: get_latest_device_health
    T->>C: Tenant- and gate-scoped query
    C-->>T: Health state
    T-->>A: Health observation
    A->>T: list_open_gate_incidents
    T->>C: Tenant- and gate-scoped query
    C-->>T: Incident state
    T-->>A: Existing incident observations
    A->>P: Propose create_incident or start_incident_investigation
    P-->>A: awaiting_approval
    A-->>API: Plan, observations, policy checks, proposed action
    API-->>O: Typed run awaiting human decision
    O->>API: POST approve/reject with reason and idempotency key
    API->>A: Authenticated decision
    alt approved
        A->>T: Execute the selected incident tool
        T->>C: Explicit tenant-scoped mutation
        C-->>A: Incident result
    else rejected
        A->>A: Record rejection; execute no mutation
    end
    A-->>API: Final trace and audit metadata
    API-->>O: Typed terminal run
```

This is deliberately narrower than a general-purpose chatbot. The free-form objective supplies
operator context only. The `intent` selects the server-owned planner and its fixed typed trajectory;
the implemented planner neither parses nor decomposes the objective. The runtime can invoke only
registered tools, and every tool receives scope from authenticated server state rather than from
prose.

## Contract surface

Create a run:

```http
POST /api/v1/agent/runs
Authorization: Bearer demo-operator
X-Organization-ID: org-atlas
Content-Type: application/json

{
  "objective": "Check the north gate device state and prepare an incident if attention is required",
  "gate_id": "gate-atlas-north",
  "intent": "gate_health_triage",
  "idempotency_key": "shift-a:north:health:001"
}
```

Read runs or one complete trace:

```http
GET /api/v1/agent/runs?gate_id=gate-atlas-north&limit=20
GET /api/v1/agent/runs/AGENT_RUN_ID
```

Resolve the pending action:

```http
POST /api/v1/agent/runs/AGENT_RUN_ID/decisions
Authorization: Bearer demo-operator
X-Organization-ID: org-atlas
Content-Type: application/json

{
  "decision": "approved",
  "reason": "Device health and existing incidents reviewed",
  "idempotency_key": "shift-a:north:health:001:decision"
}
```

Use `"decision": "rejected"` to close the proposal without changing incident state. The generated
OpenAPI document is the executable source of truth for exact response schemas and validation. The
API strips surrounding decision-reason whitespace, validates the canonical reason at 3–500
characters, and uses that stored value for exact idempotency binding; whitespace-only input is
rejected.

### Lifecycle and permissions

| Run status | Meaning |
| --- | --- |
| `running` | Plan/read execution is active or was interrupted; a duplicate create call inspects this state but does not recover it |
| `awaiting_approval` | Reads completed; one incident mutation is staged but has not run |
| `completed` | Approved action succeeded, or the healthy branch completed without an action |
| `rejected` | A human rejected the proposal; no consequential tool ran |
| `failed` | A plan/tool/policy path failed and the bounded failure is persisted |

Organization administrators and security operators hold the prototype's `agent_run` and
`agent_approve` permissions. Platform administrators hold all permissions. List/detail reads use the
existing read capability and remain organization scoped; host and edge roles cannot start or approve
runs. These are prototype capability assignments, not a substitute for production identity design.

### Console contract

The `/#/agent` workspace renders the server contract; the browser never executes a tool. It shows
the retained objective context and fixed gate scope, intent-selected five-step plan, execution
evidence, read/consequential risk, policy checks, planner/policy provenance, terminal audit items,
and the exact pending branch. One of the two conditional incident tools is paused while the other is
visibly `skipped`.

**Live API run** means create/decision actions use a confirmed session and the exact selected live
gate ID. If either is unconfirmed, the form says that submission will use the **Reference
trajectory**, derived from the checked-in deterministic snapshot; it is not an operational
observation or mutation. Approval and rejection open a localized confirmation surface requiring a
3–500 character reason. The canonical trimmed reason and actor render with the completed decision.
Create and decision retries reuse their idempotency keys, and a decision timeout reconciles by
reading the run.

The displayed **Evidence coverage** is a 0-based coverage signal over successful read steps,
non-empty evidence, and passed policy checks; it is **Unavailable** when there are no read steps. It
is not a probability that the proposal is correct, an ANPR confidence score, or permission to
approve. Policy chips expand to their recorded detail. An unknown tool, risk, or policy outcome
fails closed and removes decision controls.

## Tool contracts and authority

| Tool | Risk class | What it may read or change | Autonomy policy |
| --- | --- | --- | --- |
| `get_gate` | Read only | The selected gate's metadata and operating state | May run after server-side scope checks |
| `get_latest_device_health` | Read only | Latest configured-camera reports plus coverage, status, and freshness evidence for the selected gate | May run after server-side scope checks |
| `list_open_gate_incidents` | Read only | Open incidents for the selected gate | May run after server-side scope checks |
| `create_incident` | Consequential | Adds one incident for the selected gate | Must remain pending until an authenticated human approves |
| `start_incident_investigation` | Consequential | Moves an existing open gate incident into investigation and assigns the approving operator | Must remain pending until an authenticated human approves |

The runtime has no shell, arbitrary HTTP, database, camera-credential, or physical-actuation tool.
Adding a tool is therefore an authority change, not a prompt edit: it requires a typed input/output
contract, explicit risk class, policy rule, audit treatment, negative-scope tests, and an operator
presentation that makes the proposed effect understandable.

## Policy invariants

The following constraints stay outside planning logic:

1. **Authentication is authoritative.** UI visibility never grants agent permissions.
2. **Organization scope is server derived.** An ordinary principal cannot switch it through an
   objective, organization header, or planner/tool argument.
3. **The gate is validated and fixed.** The request selects a gate inside the authenticated
   organization; the persisted run pins it, so later tool arguments cannot inspect or mutate another
   gate.
4. **Intent, tools, and order are allowlisted.** For `gate_health_triage`, the runtime validates the
   exact registered sequence and risk classes; text cannot reorder a write ahead of evidence, name a
   new intent, import code, or call an unregistered capability.
5. **Reads and writes have different policy.** Read-only triage may proceed; either incident
   mutation pauses.
6. **A human decision is data.** Actor, reason, time, and outcome are retained with the run.
7. **Retries do not expand effects.** Run creation and approval use separate idempotency keys.
8. **Effect and completion are atomic.** An approved incident mutation and successful run/step/audit
   transition commit together; a failure rolls back the domain mutation before recording failure.
9. **Approval is not stale-plan authority.** Tenant, gate, incident state, device/gate health, and
   duplicate-incident preconditions are revalidated inside the commit transaction.
10. **Health coverage follows topology.** The gate needs at least one enabled configured camera, and
    every such camera needs one fresh, online report. Wrong-gate or unregistered camera reports are
    rejected, and non-camera device rows never satisfy camera coverage. Missing, older-than-five-
    minute, more-than-one-minute-future, invalid, and non-online evidence requires attention.
11. **Recognition is evidence, not authority.** Neither ANPR confidence nor agent output controls a
   physical barrier.

These rules are enforced by the API/runtime boundary. They must remain true if a probabilistic
planner is introduced later.

## Traceability model

An agent response is not only a final sentence. The persisted run exposes enough structure to
reconstruct why an action was proposed:

- objective, intent, authenticated actor, organization, gate, and lifecycle timestamps;
- a typed plan and ordered steps;
- each tool name, risk class, status, bounded input, and structured output;
- policy checks and the approval state of consequential work;
- human decision, reason, actor, and resulting incident reference when applicable;
- failure code/detail when a run cannot complete;
- audit events that preserve lifecycle transitions.

Tool results should remain operationally useful but minimal. Secrets, raw credentials, unrestricted
media, and unrelated tenant records do not belong in the trace. A production telemetry pipeline
should correlate the agent run with request/trace identifiers while preserving the same redaction
rules used by the control plane.

## Failure semantics

| Failure or ambiguity | Current behavior or required invariant | Production follow-up |
| --- | --- | --- |
| Unknown intent or malformed request | Typed validation rejects the request before tools run | Version intent schemas and publish compatibility tests |
| Gate outside the organization | Tenant-scoped lookup does not expose or use it | Add OIDC/RLS and systematic cross-tenant contract tests |
| Missing camera health or no configured camera | Compare reports with enabled cameras configured at the gate; an empty expected set or missing coverage requires a reviewed incident proposal | Make the five-minute policy tenant/site configurable only with versioned policy and tests |
| Stale, future, or invalid health timestamp | Accept stale history without regressing the latest camera state; reject ingestion more than one minute in the future; treat any stale/future/invalid evidence reaching evaluation as attention and retain its state/evaluated time | Monitor clock skew and calibrate versioned thresholds from site telemetry |
| Wrong-gate or unregistered camera report | Reject camera ingestion; configured-camera readiness ignores any phantom camera record, and non-camera device rows never satisfy camera coverage | Alert on repeated invalid ingestion and reconcile topology |
| Existing open incident | Propose investigation only for an actionable unassigned incident; if all unresolved work is assigned, skip reassignment and duplicate creation | Evaluate explicit merge/reassignment policy with operators |
| Duplicate create request | Return the idempotent run rather than create a second trajectory | Define retention and caller-key namespace policy |
| Duplicate human decision | Bind the key to the actor, decision, and canonical trimmed reason; preserve one decision/effect and reject conflicting reuse | Exercise concurrent retries against the production database |
| Human rejects the proposal | Record the reason and perform no consequential tool call | Feed rejection categories into planner evaluation, not automatic retraining |
| World changes during human review | Revalidate at commit; recovered health, a resolved/already-assigned target, or a newly opened incident fails safely without the stale effect | Define refresh/expiry UX and policy-specific proposal TTLs |
| Tool or policy failure | Persist a failed/blocked trace instead of presenting a successful action; an approved domain mutation rolls back if run completion fails | Add retry classes, leases, and dead-letter operations |
| Process restart during reads | The same create key returns the persisted `running` trace but does not execute pending reads implicitly | Add explicit lease ownership, heartbeat/expiry, retry classes, and tested crash recovery before enabling resume |
| Prompt injection in operational text | Current deterministic planner does not interpret text as code or tool authority | Treat future model inputs/results as untrusted data and run adversarial evals |
| Model or planner unavailable | Control-plane and manual workflows remain usable | Define timeouts, fallback UX, and an agent-specific availability objective |

Failures are first-class run outcomes. “No action” and “needs human review” are valid results; the
runtime must never fill missing evidence with a plausible-looking operational claim.

## Evaluation strategy

The deterministic planner makes policy and trajectory tests repeatable. The current backend suite
covers boundaries that matter more than fluent output:

- stable plan/tool ordering for `gate_health_triage`;
- rejection of reordered or risk-reclassified provider plans before persistence/execution;
- healthy, degraded/offline, and existing-incident fixtures;
- no consequential call before approval and no call after rejection;
- actionable unassigned incidents route to investigation, already-assigned unresolved work produces
  no duplicate/reassignment, and attention evidence without an incident routes to creation;
- approval creates or transitions at most one incident across retries;
- same-key create retry returns a persisted running trace without replaying pending reads;
- configured-camera health coverage with five-minute max age, one-minute future tolerance, and
  missing/stale/future/invalid evidence states;
- commit-time rejection of stale proposals after a competing incident or changed target state;
- role permissions and cross-organization/gate isolation;
- concurrent create-key binding, exact decision replay, actor conflicts, and approve/reject races;
- wrong-gate/unregistered camera-health rejection, configured-camera-only coverage, and out-of-order
  history that cannot regress current state;
- trace, policy-check, decision, durable failure, and audit serialization;
- sanitized planning failure persisted as one terminal failure;
- atomic rollback when a failure occurs after an approved domain insert but before run completion.

Run the checked-in evaluation matrix from the repository root:

```bash
uv run --project services/control_api --frozen python -m control_api.agent_evals
```

The runner creates a fresh seeded SQLite database per scenario and emits a stable JSON report with
schema, planner, and policy versions. Its six current scenarios assert healthy no-action,
degraded/offline approval gating, existing-incident reuse, tenant-escape denial, and duplicate-
decision safety. The backend suite calls the same runner, so the executable evaluation cannot drift
silently from CI coverage.

This deterministic matrix is a regression baseline, not evidence of field usefulness or model
quality. Add concurrent decision races, health-policy boundary/clock-skew cases, explicit
lease-based restart recovery, and sensitive-output cases as the production topology becomes
concrete. The next layer is trajectory evaluation on representative and adversarial cases.

Before enabling a model-backed planner, add a versioned evaluation set with expected tools, allowed
scope, expected escalation class, and forbidden actions. Measure trajectory correctness separately
from answer quality:

| Evaluation dimension | Example signal | Promotion gate |
| --- | --- | --- |
| Tool selection | Required/forbidden tool calls by scenario | No forbidden or out-of-scope call |
| Argument grounding | Gate and organization match trusted run context | No model-derived scope widening |
| Policy compliance | Consequential calls stop at the approval boundary | Zero bypasses under normal and adversarial prompts |
| Evidence faithfulness | Claims are supported by recorded tool outputs | Unsupported operational claims block promotion |
| Retry safety | Replayed run/decision keys preserve one effect | No duplicate incident under concurrency tests |
| Human usefulness | Acceptance, rejection, edit reason, and time-to-review | Establish in shadow mode; do not assume a target |
| Reliability | Tool latency, failure class, trace completeness | Define an SLO from measured pilot behavior |
| Cost and latency | Tokens/tool calls/wall time if a model is introduced | Budget per intent before broader rollout |

Evaluation values are criteria to collect, not achieved metrics. Shadow operation should compare
agent proposals with normal operator work before any new consequential tool is considered.

## Operating the agent

1. Confirm the active organization, role, gate, API source, and data freshness.
2. Select the supported `gate_health_triage` intent and record a narrow objective as operator
   context; it does not change the fixed tool trajectory.
3. Review the plan, each read-only observation, and the policy checks.
4. If an incident is proposed, verify the gate, device evidence, existing open incidents, title,
   severity, and description shown by the console.
5. Approve or reject with a concise operational reason.
6. Confirm the final run status and, after approval, the resulting incident in Operations.
7. For handoff, share the run identifier rather than reconstructing the trajectory from memory.

The operator remains responsible for the consequential decision. Approval is not a generic
“continue” button: it authorizes the staged tool call represented in that run.

## Extending safely

Use the following sequence for a new intent or tool:

1. define the operational job, trusted scope, stop conditions, and unavailable-data behavior;
2. add typed tool input/output without giving the planner raw repository or network access;
3. classify the tool as read-only or consequential and document its worst credible effect;
4. implement policy outside the planner and keep tenant context server-derived;
5. add idempotency for every effect and an audit representation for every outcome;
6. render proposed arguments and evidence clearly enough for a reviewer to decide;
7. test success, denial, cross-tenant access, duplicate delivery, partial failure, and restart;
8. ship in observe-only mode and evaluate trajectories before expanding authority.

Physical actuation remains a separate integration and safety project. It should not be introduced by
adding an `open_gate` tool to this runtime without independent interlocks, command expiry,
acknowledgement, fail-safe behavior, and on-site validation.

## Related documents

- [Product overview](product-overview.md)
- [Platform architecture](architecture.md)
- [Control API overview](api-overview.md)
- [Data model and workflows](data-and-workflows.md)
- [Security and privacy](security-and-privacy.md)
- [Operator guide](guides/operator.md)
- [Design evolution](design-evolution.md)
- [ADR-0007: bounded agent runtime](adrs/0007-bounded-agent-runtime.md)
- [Pilot and rollout plan](pilot-rollout.md)
