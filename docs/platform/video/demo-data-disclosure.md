# Demo-data and branding disclosure

← [Video package](README.md) · [Platform documentation index](../README.md)

## Short on-screen disclosure

> Engineering prototype · synthetic demo data · not a live campus deployment

Use this on the opening card and final description. Keep the UI's **Demo data** badge visible during
operational shots.

## Full disclosure

The campus console and control API include deterministic fictional/composite organizations, sites,
gates, people, vehicles, plate-like values, access requests, grants, passages, recognition results,
incidents, device health, metrics, coordinates, timestamps, and operator names. They are created for
software demonstration, testing, screenshots, and video reproducibility. They do not represent real
UM6P operational records or research participants.

UM6P branding is used as authorized demonstration branding for this repository. Its presence does
not claim that the prototype is deployed by UM6P, endorsed as a production service, or evaluated by
UM6P staff/students. The product is white-label and tenant presentation is configuration.

Stakeholder journeys, interview dates, and prototype feedback in the case study are explicitly
labeled **illustrative/composite**. The author-provided project motivation is not independently
verified field evidence. See [Research and evidence](../research-and-evidence.md).

## Deterministic sources

- Console fixture: `web/console/demo-data.mjs`
- Console tenant configuration: `web/console/config.mjs`
- API fixture: `services/control_api/control_api/seed.py`
- API evidence label: `Synthetic composite - generated for the platform demo`
- API metadata disclosure: `/api/v1/meta`

The console's fixture timestamp and relative-minute fields are intentionally stable. API-seeded
timestamps are also deterministic, but live projection may calculate relative age from the recording
date; the main UI recording therefore uses static demo mode.

## Demo identities

Named bearer tokens such as `demo-operator`, `demo-admin`, and `demo-host` are intentional local test
fixtures, not secure accounts. Do not show token lists in the video, expose the service publicly, or
reuse these values in a live environment.

## Plate and identity interpretation

- Plate-like strings are fabricated and must not be used to infer a real owner.
- A recognized plate is a vehicle observation, not proof of a person's identity.
- Confidence and model output in fixtures illustrate UI/data shape, not measured model performance.
- Names, emails, organizations, incidents, and metrics are synthetic even when plausible.

## White-label replacement checklist

A different deployment or recorded case study must replace/review:

- tenant/campus IDs and display names;
- authorized logo, alternative text, palette, and support label;
- locale/time zone and translated content;
- every demo record and organization/site/gate identifier;
- API fixture/evidence label and demo tokens;
- screenshots, narration, captions, repository/video description;
- production switch that disables demo fallback and demo authentication.

## Suggested video-description text

> This two-minute engineering case study shows a local Campus Access prototype built around a typed
> number-plate recognition pipeline. All displayed people, plates, events, incidents, metrics, and
> device records are synthetic/composite demonstration data. UM6P branding is used with authorization
> for the demo; the video does not claim a live UM6P deployment or verified UM6P user research. Edge
> camera integration and asynchronous worker infrastructure are presented as target architecture.

## Related documents

- [Research methodology](../research-and-evidence.md)
- [Product overview](../product-overview.md)
- [Architecture implementation status](../architecture.md#implementation-status)
- [Recording guide](recording-guide.md)
