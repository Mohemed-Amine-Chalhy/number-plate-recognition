# Two-minute storyboard, script, and shot list

← [Video package](README.md) · [Platform documentation index](../README.md)

## Timed storyboard

The narration below is written for a calm professional pace of roughly 120–130 words per minute.
Pause briefly at view transitions; do not speed-read to compensate for late edits.

| Time | Picture/action | Voiceover | On-screen callout |
| --- | --- | --- | --- |
| 00:00–00:07 | Clean title card, then fade into command center with Demo data badge visible | “This is Campus Access: a white-label engineering prototype built around a typed Moroccan number-plate recognition pipeline.” | `Campus Access` / `Engineering prototype · synthetic demo data` |
| 00:07–00:20 | Hold command-center overview; slow cursor across entries, pending reviews, wait, and device health | “The starting model could read a still image. The product question was larger: how should several gates, invitations, incidents, and uncertain observations become one safe operating workflow?” | `From model demo → operational system` |
| 00:20–00:35 | Select Residential Gate on map; attention panel and live-arrivals list remain visible | “The command view prioritizes what needs attention. Operators can see queue pressure, gate state, recent arrivals, and degraded equipment without treating missing data as a successful decision.” | `Exceptions first` / `Freshness and degraded state are explicit` |
| 00:35–00:51 | Open Gate workspace; show camera placeholder, recognized plate, confidence, and matched-access context; do not press gate command | “Inside a gate workspace, recognition is evidence—not authority. Plate candidates, confidence, camera health, and an access match are shown together, while authorization remains a separate, traceable action.” | `Observation ≠ authorization ≠ actuation` |
| 00:51–01:07 | Go to Access & approvals; switch to Campus admin if needed; approve one deterministic pending request | “Hosts submit a typed request with a site and time window. An authorized reviewer can approve or reject it with context, creating a bounded grant instead of passing free-form messages between teams.” | `Typed request → explicit decision → bounded grant` |
| 01:07–01:22 | Go to Operations; pan from incidents to device-health cards and acknowledge one synthetic incident | “Operations brings incidents and device heartbeat into the same model. A possible tailgating signal and an intermittent camera stay assigned, time-stamped, and linked to follow-up rather than disappearing after the queue moves.” | `Incidents have owners` / `Device health is operational context` |
| 01:22–01:35 | Use language selector: French, then Arabic; hold RTL layout for three seconds, return to English | “The interface is white-label and supports English, French, and Arabic with right-to-left layout. Branding, locale, time zone, and API location are configuration—not campus-specific business logic.” | `EN · FR · AR / RTL` / `White-label configuration` |
| 01:35–01:49 | Cut to OpenAPI UI, then architecture Mermaid rendered in documentation; highlight control API, AI worker, and edge agent | “Behind the interface, a typed FastAPI control plane persists tenant-scoped workflows. The target architecture keeps model workers separate and places an outbound edge agent beside ONVIF and RTSP cameras.” | `Prototype: FastAPI + SQLite` / `Target: edge + durable worker plane` |
| 01:49–02:00 | Return to command view, then end card with repository/case-study statement | “The result is deliberately honest: deterministic synthetic data today, explicit failure and recovery paths, and a staged shadow pilot before automation. It demonstrates product judgment as well as working code.” | `Working prototype · documented path to pilot` |

## Continuous voiceover script

> This is Campus Access: a white-label engineering prototype built around a typed Moroccan
> number-plate recognition pipeline.
>
> The starting model could read a still image. The product question was larger: how should several
> gates, invitations, incidents, and uncertain observations become one safe operating workflow?
>
> The command view prioritizes what needs attention. Operators can see queue pressure, gate state,
> recent arrivals, and degraded equipment without treating missing data as a successful decision.
>
> Inside a gate workspace, recognition is evidence—not authority. Plate candidates, confidence,
> camera health, and an access match are shown together, while authorization remains a separate,
> traceable action.
>
> Hosts submit a typed request with a site and time window. An authorized reviewer can approve or
> reject it with context, creating a bounded grant instead of passing free-form messages between
> teams.
>
> Operations brings incidents and device heartbeat into the same model. A possible tailgating signal
> and an intermittent camera stay assigned, time-stamped, and linked to follow-up rather than
> disappearing after the queue moves.
>
> The interface is white-label and supports English, French, and Arabic with right-to-left layout.
> Branding, locale, time zone, and API location are configuration—not campus-specific business logic.
>
> Behind the interface, a typed FastAPI control plane persists tenant-scoped workflows. The target
> architecture keeps model workers separate and places an outbound edge agent beside ONVIF and RTSP
> cameras.
>
> The result is deliberately honest: deterministic synthetic data today, explicit failure and
> recovery paths, and a staged shadow pilot before automation. It demonstrates product judgment as
> well as working code.

## Detailed shot list

| Shot ID | Route/source | Setup/action | Duration | Edit note |
| --- | --- | --- | --- | --- |
| S01 | Title graphic | Product name, subtitle, synthetic-data disclosure | 4 s | Simple fade, no animated logo claim |
| S02 | `/#/command` | Wait for `Demo data`; no cursor movement for first beat | 8 s | Establish hierarchy |
| S03 | `/#/command` | Move across metrics and map; click Residential Gate pin | 15 s | Ease crop 100% → 108% |
| S04 | Residential attention | Hold degraded status and arrivals | 8 s | Keep source badge visible |
| S05 | `/#/gates` | Select Residential Gate tab; point to recognition and match | 16 s | Do not click Open barrier |
| S06 | `/#/access` | Switch role to Campus admin; approve first pending request | 16 s | Capture success toast; data is demo-only |
| S07 | `/#/operations` | Show incidents, then device health; acknowledge synthetic incident | 15 s | No rapid scrolling |
| S08 | Any stable view | EN → FR → AR, hold RTL, return EN | 13 s | Hard cuts between locales are acceptable |
| S09 | `http://127.0.0.1:8000/docs` | Show API tags, not demo tokens/details | 7 s | Crop to title/resource groups |
| S10 | Rendered architecture doc | Highlight console/API/worker/edge | 7 s | Label “Target architecture” |
| S11 | `/#/command` + end card | Return overview; dissolve to final statement | 11 s | End exactly at 02:00 |

## B-roll alternatives

If a UI interaction is unstable, replace it with:

- a static crop of the deterministic view;
- the data/workflow Mermaid diagram;
- OpenAPI resource tags;
- a short code crop of tenant configuration or recognition/authorization schema separation.

Do not substitute stock security-camera footage that could imply a real deployment.

## Caption plan

- Use [the supplied WebVTT file](captions.vtt) as the master timing track.
- Maximum two lines per cue and approximately 42 characters per line where practical.
- Preserve technical terms `FastAPI`, `ONVIF`, and `RTSP`.
- Use an em dash in “evidence—not authority” only if the caption renderer supports it reliably.
- Place captions in the lower safe area, but move them above toasts/dialog actions when necessary.
- Use high-contrast solid/semitransparent backing; never rely on a drop shadow alone.
- Do not caption decorative on-screen callouts unless they are spoken.
- Re-time captions after the final voice recording; do not time-stretch narration to the draft VTT.

## Audio and motion

- Prefer clean narration without music. If music is used, it must be licensed/cleared and remain at
  least 16–20 dB below narration.
- Use hard cuts or short dissolves; avoid cinematic camera moves over an operations UI.
- Keep cursor acceleration off where possible and use one deliberate click per beat.
- Remove notification sounds and operating-system alerts.
- Do not animate charts or numbers in a way that implies live telemetry.

## Truthfulness review

Before export, compare the video against
[Demo-data disclosure](demo-data-disclosure.md),
[Implementation status](../architecture.md#implementation-status), and
[Research methodology](../research-and-evidence.md#methodology-disclosure).
