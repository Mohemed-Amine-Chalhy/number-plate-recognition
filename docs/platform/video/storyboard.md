# Two-minute product walkthrough

← [Video package](README.md) · [Recording guide](recording-guide.md)

The cut follows the product from campus-wide awareness to one gate, structured access,
operational follow-through, configuration, and system boundaries. Scene lengths and narration beats
are the same timing contract used by `scripts/build_demo_video.py` and `captions.vtt`.

## Scene contract

| Scene | Time | Source | On-screen callout | Product beat |
| --- | --- | --- | --- | --- |
| S01 | 00:00–00:08 | Generated title card | `Campus Access` | Establish one coordinated workflow for every gate. |
| S02 | 00:08–00:24 | `command-center.png` | `Control every gate from one view` | Move from campus map to gate status, arrivals, reviews, and device health. |
| S03 | 00:24–00:40 | `gate-workspace.png` | `Turn recognition into usable context` | Connect plate evidence, camera state, access context, and deliberate gate controls. |
| S04 | 00:40–00:56 | `access-approvals.png` | `Replace inbox searching with structured access` | Show a bounded, searchable request and review workflow. |
| S05 | 00:56–01:12 | `operations.png` | `Make exceptions operational` | Keep incidents, degraded devices, ownership, and recency in the operating picture. |
| S06 | 01:12–01:27 | `campus-setup.png` | `Configure the campus, not a product fork` | Show tenant identity, topology, devices, locale, time zone, and API as configuration. |
| S07 | 01:27–01:39 | `mobile-rtl.png` | `Support the workflow wherever it happens` | Demonstrate responsive behavior and multilingual right-to-left support. |
| S08 | 01:39–01:53 | Generated architecture card | `Keep every system boundary replaceable` | Explain the typed control plane, inference boundary, and edge integration seams. |
| S09 | 01:53–02:00 | Generated end card | `A complete platform slice` | Close on the working product surface and its engineering delivery system. |

During UI scenes, use one quiet source note in the upper safe area:
`Reference scenario · generated operational data`.

## Timed narration

These rows match the WebVTT cues exactly. Keep each sentence inside its assigned scene when replacing
the draft voice track.

| Beat | Time | Narration |
| --- | --- | --- |
| N01 | 00:00–00:04 | Campus Access coordinates vehicle entry across every gate from one system. |
| N02 | 00:04–00:08 | It connects hosts, security teams, and administrators around one shared workflow. |
| N03 | 00:08–00:16 | The command center brings gate status, queues, arrivals, reviews, and device health together. |
| N04 | 00:16–00:24 | An interactive campus map makes each entry point directly accessible while exceptions stay visible. |
| N05 | 00:24–00:32 | At each gate, plate candidates and confidence sit beside camera state and matching access context. |
| N06 | 00:32–00:40 | Operators can inspect the evidence, understand the match, and keep the gate action deliberate. |
| N07 | 00:40–00:48 | Hosts create structured requests with vehicle, site, gate, purpose, and a bounded time window. |
| N08 | 00:48–00:56 | Reviewers work from a prioritized queue, so a decision becomes a searchable record—not a lost email. |
| N09 | 00:56–01:04 | The operations view connects incidents and device health to gate, severity, owner, and recency. |
| N10 | 01:04–01:12 | Degraded cameras and unresolved events remain visible without pushing active arrivals out of view. |
| N11 | 01:12–01:20 | Administrators configure tenant identity, campus topology, gate devices, locale, time zone, and API. |
| N12 | 01:20–01:27 | Those choices remain configuration, keeping the product reusable without campus-specific forks. |
| N13 | 01:27–01:33 | The same console adapts to booth and mobile widths without splitting the workflow across separate apps. |
| N14 | 01:33–01:39 | English, French, and Arabic right-to-left layouts are built into the interface. |
| N15 | 01:39–01:46 | A typed FastAPI control plane persists tenant-scoped workflows behind the console. |
| N16 | 01:46–01:53 | The versioned inference boundary and explicit edge integration seams keep each responsibility replaceable. |
| N17 | 01:53–01:57 | The result is a runnable platform slice: |
| N18 | 01:57–02:00 | multi-gate operations, structured access, AI evidence, tests, runbooks, and reproducible delivery. |

## Edit direction

- Favor the product hierarchy over decorative motion: a restrained 3.5% crop drift and short
  dissolves are enough.
- Keep the command-center map readable during S02; do not crop away gate pins, health state, or the
  selected-gate panel.
- Keep the gate-workspace evidence and access context visible together during S03.
- Use the checked-in light-theme desktop captures for the main flow and the dark Arabic capture only
  for S07.
- Keep captions to two lines, above the lower callout panel, with a solid or semitransparent backing.
- Use clean narration without notification sounds. If music is added, keep it clearly below speech
  and use only licensed material.

## Content boundary

The walkthrough describes product behavior visible in the repository. Physical controllers and
site edge hardware remain explicit integration seams on the architecture card. Interface numbers
are part of the generated reference scenario and are not narrated as measured outcomes.

Before export, verify the scene and cue ranges against the automated timeline test, then watch once
muted to confirm that the product story remains understandable without narration.
