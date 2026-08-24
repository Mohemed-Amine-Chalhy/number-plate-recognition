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
| `demo-admin` | Manage Atlas topology, requests, and grants |
| `demo-operator` | Review passages, authorize, and handle incidents |
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
