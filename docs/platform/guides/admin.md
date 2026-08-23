# Administrator guide

← [Platform documentation index](../README.md)

## Purpose

Administrators configure tenant presentation, organization/site/gate topology, roles, access
workflows, and devices. The prototype's demo administrator proves these workflows locally; real
deployment requires OIDC, reviewed permissions, and change control described in
[Security and privacy](../security-and-privacy.md).

## Change discipline

For every administrative change:

1. state the intended organization/site/gate scope;
2. preview or validate references and state transition;
3. record a reason/ticket when operational impact is material;
4. apply during an agreed window if it affects an active gate;
5. verify desired versus applied state;
6. keep a known rollback configuration/version.

Prefer disable/archive/revoke over deleting records needed to explain prior events.

## Organization and site setup

1. Create the organization slug/name and display time zone.
2. Create sites with unique codes, time zone, address, and optional map coordinates.
3. Confirm ordinary organization users cannot view another organization.
4. Add gates to the correct site; set stable code, name, direction, coordinates, and initial
   disabled/offline state.
5. Add camera metadata only after an edge/network owner is assigned.

Use UTC for stored event instants and the site time zone for operator display. Daylight/time-zone
changes should never rewrite existing timestamps.

## Gate lifecycle

| Status | Meaning | Administrative action |
| --- | --- | --- |
| Operational | Expected dependencies healthy | Monitor normal workflow |
| Congested | Queue threshold exceeded | Coordinate staffing/traffic; do not hide device faults |
| Degraded | Some dependency unavailable | Publish fallback and owner |
| Offline | Gate not providing service/data | Route traffic and investigate |
| Disabled | Intentionally unavailable | Require explicit enable after validation |

Changing a gate status does not itself control a physical barrier.

## Access configuration

- Define who may submit, decide, create a direct grant, revoke, or view.
- Use explicit validity start/end and optional gate scope.
- Require a decision reason for rejection and a revocation reason.
- Review long-lived grants and avoid wildcard/all-gate permissions by default.
- Test late/early arrival, changed plate, cancelled visit, duplicate request, and expired grant.
- Keep recognition confidence out of grant policy; it belongs to evidence quality.

## Roles

| Role | Typical assignment | Review concern |
| --- | --- | --- |
| Platform admin | Product/platform operations | Cross-organization power; very limited membership |
| Organization admin | Campus access administration | Topology and grant lifecycle |
| Security operator | Command center/gate operations | Decisions and incidents, not broad tenant setup |
| Host | Department/reception coordinator | Own request submission, not approval by default |
| Viewer | Analyst/auditor/observer | Read-only and evidence access still scoped |
| Edge agent | Workload identity | Only ingest/health capabilities, never interactive login |

Review effective permissions, not only role labels. Remove or rotate access promptly when ownership
changes.

## Branding, language, and time zone

The demo tenant configuration is isolated in `web/console/config.mjs`. It includes tenant/campus ID,
brand names, logo, alternative text, fallback mark, accent palette, support label, API timing, locale,
theme, role, and time zone.

For a white-label deployment:

1. supply an authorized logo and correct alternative text;
2. verify contrast in light/dark theme and focus/error states;
3. replace full/short/product/support names;
4. set an approved accent palette without relying on color alone for status;
5. choose supported locale and site time zone;
6. have English, French, and Arabic content reviewed by fluent users;
7. test Arabic RTL order, tables, icons, plate strings, dates, and keyboard navigation;
8. replace all deterministic tenant data and disclosures;
9. configure production API URL and disable demo fallback for live operation.

Branding authorization for the case-study demo does not make synthetic records real or imply a
production endorsement. See [Demo-data disclosure](../video/demo-data-disclosure.md).

## Camera and edge setup

Do not paste credentials into the central camera record. Coordinate the site survey and edge
enrollment, then follow [Camera and edge onboarding](../camera-edge-onboarding.md). Keep desired and
applied config versions visible; a failed apply retains last-known-good configuration.

Disable a camera before moving it between gates. Recalibrate ROI/trigger/inference profile after a
physical move, firmware change, stream-profile change, or material lighting change.

## Model and inference rollout

The model manifest pins artifacts, but production rollout also needs a versioned assignment:

- model bundle ID and manifest digest;
- inference profile/threshold version;
- camera/gate/site scope;
- start time and rollback version;
- warm-up/readiness result;
- shadow/canary comparison.

Do not replace a model in place while preserving its identity. Promote after deterministic smoke,
representative camera evaluation, shadow output comparison, and measured latency/memory.

## Data and retention settings

- Assign retention by media/metadata category and storage budget.
- Keep incident-pinned evidence explicit and expiring.
- Restrict exports and signed media duration.
- Verify deletion jobs and backup retention; a UI setting without an executed job is not retention.
- Keep synthetic demo and live tenant stores separated.

## Administrative verification

After a topology/policy/device change:

- [ ] API response and subsequent read show the intended state.
- [ ] Event trail records actor, scope, and time.
- [ ] Unrelated organization/site/gate is unchanged.
- [ ] Edge desired/applied versions converge where relevant.
- [ ] Health and operator view reflect the change.
- [ ] Rollback has been tested or remains available.
- [ ] Shift owner has been informed if live operations are affected.

## Related documents

- [Operator guide](operator.md)
- [Host guide](host.md)
- [Camera and edge onboarding](../camera-edge-onboarding.md)
- [Deployment runbook](../deployment-runbook.md)
- [Security and privacy](../security-and-privacy.md)
