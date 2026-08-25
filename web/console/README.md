# Campus Access Console

A dependency-free, white-label operations console for the campus control API. It is intentionally
static: no package installation or build step is required, and seeded data keeps every screen usable
when the API is absent.

## Run it

From the repository root:

```powershell
python -m http.server 4173 --directory web/console
```

Open `http://localhost:4173`. Serving the files over HTTP is recommended because browser modules do
not reliably load from `file://` URLs.

The optional Node scripts have no third-party dependencies:

```powershell
cd web/console
npm run check
```

## Configuration

`config.mjs` is the deployment seam. Replace the tenant name, logo path, color tokens, API base URL,
organization ID, and role-token mapping there. The configured logo is optional; the UI displays a
text mark if it cannot load. `branding.logoUrl` deliberately makes no assumptions about image size
or aspect ratio.

The included bearer credentials are the control API's intentional demo identities:

| Console role | Demo token | API role |
| --- | --- | --- |
| Security operator | `demo-operator` | `security_operator` |
| Gate attendant | `demo-operator` | `security_operator` |
| Campus admin | `demo-admin` | `org_admin` |
| Operations viewer | `demo-viewer` | `viewer` |

Changing the role changes the token only while the current token is one of these configured demo
values. A manually supplied token is never overwritten. The token is stored in this browser's
`localStorage`; that behavior is suitable for this demo, not a production identity session. A real
deployment should inject a short-lived token from its authentication client and remove the demo
role-token mapping.

Every protected request sends both `Authorization: Bearer <token>` and the configured
`X-Organization-ID`. The client consumes `/api/v1/dashboard`, `/session`, `/organizations`, `/sites`,
`/gates`, `/cameras`, `/access-requests`, `/access-grants`, `/passages`, `/events`, `/incidents`, and
`/device-health`. Access decisions use `POST /access-requests/{id}/decision` and translate the UI's
"denied" state to the API's `rejected` enum.

The current API has no physical barrier-command route. Gate commands therefore stay as an explicit
simulation. When an actuator service exists, set `api.gateCommandPath` to a path template containing
`{gateId}`; the adapter will then POST `{ "command": "..." }`.

### Agent operations contract

`/#/agent` is a human-supervised gate-health triage workspace. It does not expose a general-purpose
chat box or run browser-side tools. Instead, the console sends one typed intent to
`POST /agent/runs` with an objective, one `gate_id`, `gate_health_triage`, and an idempotency key.
The fixed intent selects the versioned five-step trajectory. The objective is retained as operator
context and is not interpreted or decomposed by the current deterministic planner. The service
returns the plan, executed steps, structured inputs and outputs, policy checks, trace metadata, and
audit events.

The allowlist contains three read-only tools and two mutually exclusive consequential branches:

- `get_gate`
- `get_latest_device_health`
- `list_open_gate_incidents`
- `start_incident_investigation` — requires a human decision
- `create_incident` — requires a human decision

If an actionable unresolved incident is unassigned, the planner proposes starting its investigation
and skips incident creation. If every unresolved incident is already assigned, both action branches
are skipped rather than reassigning or duplicating work. If no incident exists but device health
needs attention, the planner proposes creation and skips the existing-incident branch. The API
pauses either mutation at `awaiting_approval`; the UI submits an explicit `approved` or `rejected`
decision to `POST /agent/runs/{run_id}/decisions`. No agent path contains barrier, lane, or physical
actuator commands.

Every decision requires a visible operator-entered reason. The reason, actor, and decision are
rendered back from the run record. A run creation or decision keeps one idempotency key across
retries; decision timeouts are reconciled by reading the run before the UI offers another retry.

When the API is absent, the same page renders a deterministic **Reference trajectory** derived from
the checked-in snapshot. Its badge stays distinct from **Live API run**, including when the rest of
the console is connected. A hybrid snapshot can start a live run only for a gate returned alongside
a confirmed live session; seed-only gates stay explicit reference trajectories. **Evidence
coverage** starts at zero and measures completed, inspectable reads and passing policy checks. It is
shown as unavailable when the runtime provides no read steps and is not a model-confidence score.
Policy chips expand to the recorded detail. Unknown tools, risks, or policy outcomes fail closed and
withhold decision controls.

## Data-source states

The sidebar and top bar always display one of four explicit states:

- **Live API**: every configured resource responded.
- **Partial API**: available resources are merged over the deterministic seed snapshot.
- **Reference scenario**: no API resource responded; the complete version-controlled snapshot is
  active. The internal source-state key remains `demo`.
- **Offline fallback**: the browser reports no network; API requests are skipped.

This keeps local walkthroughs resilient without presenting reference values as live operations.

## Structure

- `app.mjs` — hash-routed shell, views, interaction state, keyboard/modal behavior.
- `agentic.mjs` — normalized agent-run contract, evidence metric, idempotent request builders, and
  deterministic reference trajectory.
- `api.mjs` — authenticated v1 client and typed-contract-to-view-model normalization.
- `config.mjs` — replaceable tenant and integration configuration.
- `demo-data.mjs` — deterministic six-gate campus snapshot.
- `campus-map.mjs` — tenant-configured gate placement and coordinate projection.
- `i18n.mjs` — English, French, and Arabic interface dictionaries.
- `styles.css` — responsive tokens, light/dark themes, RTL-aware layout, print rules.
- `tests/` — dependency-free Node unit and contract tests.

## Accessibility and browser support

The console uses semantic landmarks, a skip link, visible keyboard focus, native controls, labelled
tables, a focus-trapped modal with Escape handling, live status regions, reduced-motion support, and
logical CSS properties for RTL. It targets current evergreen browsers with ES modules,
`structuredClone`, `Intl`, and CSS `color-mix()` support.
