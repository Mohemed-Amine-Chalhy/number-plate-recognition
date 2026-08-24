# Reference-scenario data note

← [Video package](README.md) · [Recording guide](recording-guide.md)

## On-screen treatment

Use one quiet label during operational UI scenes:

> Reference scenario · generated operational data

It should remain readable at normal playback size without competing with gate state, evidence, or
workflow controls. Do not add repeated disclosure slates between scenes.

## Data source

The walkthrough uses deterministic, version-controlled records created for the software experience:

- organizations, campus sites, gates, and map coordinates;
- people, vehicles, plate-like values, requests, and access records;
- arrivals, recognition results, incidents, device health, and event history;
- queue, timing, confidence, throughput, and availability values;
- fixture timestamps, names, operator labels, and camera states.

These records keep local runs, screenshots, tests, captions, and video edits reproducible. Interface
values demonstrate product behavior and data shape; the narration does not present them as measured
real-world outcomes.

## Versioned sources

- Console fixture: `web/console/demo-data.mjs`
- Tenant presentation: `web/console/config.mjs`
- Control API fixture: `services/control_api/control_api/seed.py`
- API metadata: `/api/v1/meta`
- Video timeline: `scripts/build_demo_video.py`
- Narration timings: `docs/platform/video/captions.vtt`

The console fixture timestamp and relative-minute fields remain stable for repeatable captures. Use
static reference mode for the main UI sequence so local database state and wall-clock calculations
cannot alter the take.

## Recording hygiene

- Keep bearer tokens, credentials, private URLs, notifications, and unrelated browser content out of
  the frame.
- Treat plate-like strings and displayed identities as generated records.
- Treat confidence values as interface fixtures, not an accuracy benchmark.
- Keep camera imagery visibly within the reference scenario.
- Describe edge devices and physical controllers as integration seams unless the repository gains a
  tested implementation for them.

## Reusable tenant presentation

Tenant name, logo, palette, locale, time zone, topology, and support label are configuration. When
creating another tenant cut, replace those values and regenerate the screenshot/video set while
preserving the scene and caption timing contract.

## Suggested video description

> Campus Access is a two-minute walkthrough of a configurable, multi-gate vehicle-access platform
> built around a typed Moroccan number-plate recognition pipeline. The video follows the command
> center, gate workspace, structured access, operations, localization, and modular system design.
> Displayed operational records are generated as a deterministic reference scenario.
