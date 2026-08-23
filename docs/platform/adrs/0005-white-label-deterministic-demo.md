# ADR-0005: White-label configuration and deterministic demo mode

- Status: Accepted
- Date: 2026-08-23
- Owners: Product and frontend engineering

← [ADR index](README.md) · [Demo-data disclosure](../video/demo-data-disclosure.md)

## Context

The case study needs an immediately reviewable console and repeatable video even when an API/model is
not running. UM6P demo branding is authorized for this project, but the product should not hard-code
one institution into domain behavior. Demo records can become misleading if they look live.

## Decision

- Keep tenant branding, locale, time zone, API timing/location, and defaults in explicit
  configuration.
- Keep deterministic synthetic demo data version controlled with stable IDs/timestamps/values.
- Load console resources independently and report `live`, `hybrid`, or `demo` source mode.
- Label synthetic/composite evidence and publish a demo-data disclosure.
- Disable demo fallback and demo bearer identities for a live production tenant.

## Consequences

### Positive

- Reproducible screenshots, tests, and video.
- Reviewer can explore UI without model/camera infrastructure.
- White-label boundary and localization are visible engineering concerns.
- Partial backend work remains inspectable through hybrid mode.

### Negative

- Hybrid data can be misunderstood without prominent source labels.
- Fixtures and API projections can drift and need contract tests/adapters.
- A production build needs an explicit hard-off switch, not a hidden fallback.

## Alternatives considered

- Blank/error page when API is absent: rejected for portfolio review and deterministic recording.
- Institution strings scattered through templates: rejected for white-label maintenance.
- Randomly generated demo data: rejected because screenshots/tests would be nondeterministic.
- Pretend demo data is a live integration: rejected as misleading.

## Validation

- Unit tests for locale/RTL, merge behavior, source mode, escaping, route resolution, and stable data.
- Recording checklist visibly confirms demo source and disclosure.
