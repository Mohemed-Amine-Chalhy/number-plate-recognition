# Product media

← [Platform documentation index](../README.md) · [Video package](../video/README.md)

These images are deterministic captures of the local seeded demonstration. All people, vehicles,
plate-like values, events, and operational metrics shown in the interface are synthetic. UM6P
is the configured tenant identity for this reference scenario; the product keeps tenant identity,
campus topology, and map art replaceable. See the [full demo-data disclosure](../video/demo-data-disclosure.md).

## Screenshot set

### Command center

![Desktop command center showing the visible Reference scenario badge, operational summary, six-gate map, and generated arrivals](command-center.png)

Use: product overview and the opening/closing video frame. The visible **Reference scenario** badge
is part of the evidence boundary and must not be cropped out.

### Gate workspace

![Desktop gate workspace showing generated camera evidence, matched access context, six gate tabs, and a visible Reference scenario badge](gate-workspace.png)

Use: explain the observation/access/actuation separation. The camera panel's `SCENARIO` treatment is
generated UI state inside a page clearly marked **Reference scenario**. The
**Open barrier** control is displayed but was not pressed, and no actuator endpoint is configured.

### Access and approvals

![Desktop access approvals queue showing generated requests and the visible Reference scenario badge](access-approvals.png)

Use: show typed requests and bounded review actions. Every name, request, plate-like value, metric,
and activity item is synthetic.

### Operations and health

![Desktop operations view showing generated incidents, device health, metrics, and the visible Reference scenario badge](operations.png)

Use: explain persistent incident ownership and device-health context. Latency, uptime, delivery,
incident, and camera/barrier values are fixtures, not measured service levels or live telemetry.

### Campus setup

![Desktop campus setup view showing configurable organization and API fields with the visible Reference scenario badge](campus-setup.png)

Use: show the tenant-configuration seam. The masked integration-token field is browser-local in this
reference console; deployments connect the API to the organization identity system. Do not replace
the mask with a visible token in published media.

### Mobile Arabic/RTL

![Mobile Arabic right-to-left command-center layout with all six map gates using generated records](mobile-rtl.png)

Use: demonstrate responsive behavior, Arabic localization, right-to-left layout, and six-gate map
geometry that does not mirror in RTL. Keep the reference-scenario disclosure with the image whenever
it appears outside a page that already carries the disclosure.

## Regeneration and publication

Regenerate captures after intentional visual or fixture changes by following the
[deterministic recording guide](../video/recording-guide.md). Preserve a clean browser profile,
fixed viewport, seeded state, and the disclosure/source badge. Before publishing, verify that no
token, credential, notification, real name, real plate, or unrelated browser/desktop context is
visible.
