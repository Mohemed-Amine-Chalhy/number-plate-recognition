# Campus Control API

This service is the self-contained control plane for the campus-access portfolio demo. It uses
FastAPI and SQLite, seeds two fictional organizations, and serves the static console from
`web/console` at the same origin. The existing computer-vision runtime stays outside the API.

Recognition and authorization are deliberately different records:

1. A site edge agent creates a passage.
2. An inference worker attaches a recognition observation describing what it saw.
3. A policy engine or operator records a separate authorization decision.

This prevents a high-confidence plate prediction from being treated as permission to enter.

## Seeded primary-campus topology

The Atlas main-campus fixture exposes a complete six-gate control surface. Existing gate and
camera identifiers remain stable for API clients; the South and Sports stacks use deterministic
new identifiers. Every gate has a camera and a current device-health report, so the topology and
dashboard endpoints return the same operational inventory.

On an existing demo database, stable IDs take precedence over presentation fields. The seed
relocates legacy service records without replacing operator-edited names, directions, coordinates,
camera profiles, or runtime status. If a preferred gate or camera code is already in use, the seed
selects a deterministic `-DEMO` code instead of changing the existing resource or failing startup.

| Access point | Gate ID | Camera ID |
| --- | --- | --- |
| North | `gate-atlas-north` | `camera-atlas-north-anpr` |
| North-East / Innovation | `gate-atlas-research` | `camera-atlas-research-overview` |
| East / Logistics | `gate-atlas-service` | `camera-atlas-service-anpr` |
| South-East | `gate-atlas-residence` | `camera-atlas-residence-anpr` |
| Main / South | `gate-atlas-south` | `camera-atlas-south-anpr` |
| Sports / West | `gate-atlas-sports` | `camera-atlas-sports-anpr` |

## Run

From the repository root:

```powershell
uv sync --project services/control_api --frozen
uv run --project services/control_api --frozen python -m control_api
```

Open `http://127.0.0.1:8000/` for the console or `/docs` for interactive OpenAPI documentation.
The default database is `.runtime/campus-control.sqlite3`. Configuration variables are described
in `.env.example`. The development-safe network default is `127.0.0.1:8000`; set
`CONTROL_API_HOST=0.0.0.0` for a container and override `CONTROL_API_PORT` when needed.

## Demo roles

`GET /api/v1/demo-identities` lists the intentional demo bearer tokens. Representative tokens:

| Token | Capability |
| --- | --- |
| `demo-admin` | Manage Atlas topology, access, and approval-gated operations agents |
| `demo-operator` | Review passages, handle incidents, and run/approve operations agents |
| `demo-host` | Submit and maintain visitor requests |
| `demo-viewer` | Read dashboards and operations |
| `demo-edge` | Submit passages, recognitions, and health reports |
| `demo-rif-admin` | Demonstrate tenant isolation |
| `demo-platform` | Switch tenant using `X-Organization-ID` |

Demo bearer tokens are a product-demo boundary, not a substitute for OIDC in a deployment.

## Vertical slice

Create a capture as the edge agent:

```http
POST /api/v1/passages
Authorization: Bearer demo-edge
Content-Type: application/json

{
  "site_id": "site-atlas-main",
  "gate_id": "gate-atlas-north",
  "camera_id": "camera-atlas-north-anpr",
  "direction": "inbound",
  "occurred_at": "2026-08-23T10:10:00+00:00",
  "evidence_label": "Synthetic composite - simulator frame"
}
```

Attach the worker result using the returned passage ID:

```http
POST /api/v1/passages/{passage_id}/recognitions
Authorization: Bearer demo-edge
Content-Type: application/json

{
  "status": "recognized",
  "plate_text": "12345-A-6",
  "detection_confidence": 0.96,
  "recognition_confidence": 0.93,
  "format_valid": true,
  "model_version": "vehicle:v1/plate:v1/characters:v1",
  "source": "central_worker",
  "evidence_label": "Synthetic composite - simulator frame"
}
```

Then record an independent operator decision:

```http
POST /api/v1/passages/{passage_id}/authorization-decisions
Authorization: Bearer demo-operator
Content-Type: application/json

{
  "outcome": "allowed",
  "reason": "Operator verified the active staff grant",
  "source": "operator",
  "grant_id": "grant-atlas-staff"
}
```

The console can poll `GET /api/v1/events?after_sequence=0&limit=50` and pass the returned
`next_sequence` on the next request. `GET /api/v1/dashboard` is the initial command-center read
model; `GET /api/v1/passages/{passage_id}` returns both independent timelines.

## Inspectable operations agent

The API includes a truthful, offline reference implementation of an agentic gate-health loop. It
does not call an LLM. `gate_health_triage` selects one fixed typed five-step trajectory; the
objective is persisted as operator context and the current planner neither interprets nor
decomposes it. `AgentPlanner` is an explicit provider protocol, and the installed
`deterministic_gate_health_planner@1.0.0` returns that plan for executor validation and persistence.
A future planner can implement the same protocol, but it cannot expand the runtime's closed tool
registry or change registered tool risk.

The allowlist contains three read tools and two conditional consequential tools:

| Tool | Risk | Behavior |
| --- | --- | --- |
| `get_gate` | read-only | Reads the selected tenant-scoped gate |
| `get_latest_device_health` | read-only | Evaluates configured-camera coverage, status, and report freshness for the gate |
| `list_open_gate_incidents` | read-only | Finds unresolved gate incidents |
| `start_incident_investigation` | consequential | Reuses and assigns an existing incident |
| `create_incident` | consequential | Creates one incident when unhealthy evidence has none |

Start a run as an organization administrator or security operator:

```http
POST /api/v1/agent/runs
Authorization: Bearer demo-operator
Content-Type: application/json

{
  "objective": "Inspect East gate health and prepare the safest operational response",
  "gate_id": "gate-atlas-service",
  "intent": "gate_health_triage",
  "idempotency_key": "east-triage-shift-20260825"
}
```

The response includes the versioned planner/policy trace, immutable plan, step inputs and
structured outputs, policy checks, timing and failure fields, the pending handoff, and ordered
audit events. A healthy gate completes with both action branches skipped. An actionable unassigned
incident causes `start_incident_investigation` to await approval; if all unresolved work is already
assigned, both reassignment and duplicate creation are skipped. Unhealthy evidence without an
incident causes `create_incident` to await approval. Neither tool executes while the run is
`awaiting_approval`.

Approve or reject the pending step explicitly:

```http
POST /api/v1/agent/runs/{run_id}/decisions
Authorization: Bearer demo-operator
Content-Type: application/json

{
  "decision": "approved",
  "reason": "The inspected camera evidence warrants a tracked investigation",
  "idempotency_key": "east-triage-approval-20260825"
}
```

Run status is one of `running`, `awaiting_approval`, `completed`, `rejected`, or `failed`; step
status is one of `pending`, `running`, `awaiting_approval`, `succeeded`, `skipped`, or `failed`.
Read-only users can inspect `GET /api/v1/agent/runs` and `GET /api/v1/agent/runs/{run_id}`, but
only organization administrators, security operators, and platform administrators can start or
decide runs. Every query and effect is pinned to the authenticated organization and selected gate.
Decision reasons are stripped, validated at 3–500 characters, and stored canonically for exact
idempotency binding; whitespace-only input is rejected.

Run creation and human decisions have scoped idempotency keys. For approved actions, the decision,
incident effect, terminal run state, and audit entries share one SQLite transaction: a failure
cannot commit an incident without its corresponding completed trace. Expected tool failures become
durable `failed` results rather than unstructured server responses. The commit path also rechecks
resource scope and state, suppressing a stale proposal if health recovered, an incident was
resolved, or another operator created an incident during the approval handoff.

Configured camera health is ready only when the gate has at least one enabled camera and every such
camera has an online report no more than five minutes old and no more than one minute in the future.
Missing, stale, future, invalid, and unhealthy evidence states remain explicit in the tool output;
wrong-gate or unregistered camera reports are rejected, and non-camera device rows do not satisfy
camera coverage. If a run is persisted as `running`, a duplicate create key returns that trace
without replaying pending reads. Lease-owned crash recovery remains a follow-up.

Execute the six-scenario, network-free evaluation matrix (healthy, degraded, offline, existing
incident, tenant escape, and duplicate approval) as machine-readable JSON:

```powershell
uv run --project services/control_api --frozen python -m control_api.agent_evals
```

## Quality checks

Run from `services/control_api` so the nested project configuration is authoritative. Give pytest
a unique workspace-local base directory on restricted Windows environments:

```powershell
uv run --project . --frozen ruff format --check control_api ../../tests/platform_backend
uv run --project . --frozen ruff check control_api ../../tests/platform_backend
uv run --project . --frozen mypy control_api ../../tests/platform_backend
uv run --project . --frozen pytest -q --basetemp ../../.runtime/pytest-control-api-<unique-id>
```

SQLite is appropriate for this single-process demo. The repository boundary and organization ID
on every tenant row keep a later PostgreSQL migration straightforward.
