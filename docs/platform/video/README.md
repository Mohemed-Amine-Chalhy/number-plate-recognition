# Two-minute product video package

← [Platform documentation index](../README.md)

This package builds a repeatable 120-second walkthrough of Campus Access as an agentic operations
platform. The narrative starts with the campus-wide operating context, follows one bounded agent
trajectory through typed tools and a human decision boundary, and closes with the reusable system
boundaries behind the interface.

## Watch or rebuild

[Watch the checked-in two-minute MP4](campus-access-case-study-2m-v1.mp4).

The render target is 1920×1080 H.264 at 30 fps with 48 kHz AAC audio. Its source of truth is:

- the scene and narration-beat durations in `scripts/build_demo_video.py`;
- the exact spoken text and cue ranges in [captions.vtt](captions.vtt);
- the matching visual plan in [storyboard.md](storyboard.md);
- the browser-verified product captures in [../assets](../assets/README.md).

After a successful render, the builder atomically writes `build-manifest.json`. The sidecar records
the SHA-256 digest of the builder, captions, storyboard, optional title-card logo, and every
screenshot that contributes pixels, then binds that source snapshot to the completed MP4's size and
SHA-256 digest.
The video unit test recomputes both sides and fails when copy, captures, render code, or the MP4
changes without a matching rebuild. A missing manifest is also a failure, so a binary cannot be
published from an untracked set of inputs.

On Windows, rebuild the system-voice draft narration with:

```powershell
uv run --group media --frozen python scripts/build_demo_video.py
```

For a reviewed voice recording, provide a WAV file:

```powershell
uv run --group media --frozen python scripts/build_demo_video.py `
  --narration-wav path/to/reviewed-narration.wav
```

Use `--no-narration` for a silent review copy. Encoding happens in ignored runtime files, replaces
the requested output only after the complete render succeeds, and publishes the manifest last. A
custom `--output` gets an adjacent manifest; use `--manifest` to select another sidecar path.

## Narrative structure

| Time | Product focus |
| --- | --- |
| 00:00–00:24 | Introduce agentic Campus Access, select one gate, and retain a narrow objective as operator context. |
| 00:24–00:44 | Inspect the fixed, intent-selected plan, allowlisted tool trajectory, policy results, evidence, and human handoff. |
| 00:44–01:08 | Connect the bounded agent to gate evidence and the structured access workflow. |
| 01:08–01:35 | Show configurable authority, then review Agent Operations in a mobile Arabic right-to-left layout. |
| 01:35–02:00 | Explain the agent boundary and close on the runnable, evaluated platform slice. |

Operational footage carries one understated note:
`Reference scenario · generated operational data`. It keeps the interface readable while making the
source of displayed people, vehicles, events, and metrics clear.

## Deliverables

- [Generated two-minute reference MP4](campus-access-case-study-2m-v1.mp4)
- [Scene plan and timed narration](storyboard.md)
- [Deterministic capture and recording guide](recording-guide.md)
- [WebVTT captions](captions.vtt)
- [Reference-scenario data note](demo-data-disclosure.md)
- [Product screenshot set](../assets/README.md)
- `build-manifest.json`, generated only after the MP4 is complete

## Output specification

| Property | Setting |
| --- | --- |
| Duration | Exactly 120 seconds |
| Canvas | 1920×1080, 16:9 |
| Source viewport | 1600×900 at 100% zoom for desktop captures |
| Frame rate | 30 fps constant |
| Audio | 48 kHz AAC, normalized consistently, no clipping |
| Captions | Sidecar WebVTT matching every narration beat |
| UI coverage | Light desktop flow plus one dark Arabic right-to-left Agent Operations view |
| Agent coverage | Retained objective context, fixed intent-selected plan, typed tools, evidence, policy/trace versions, approval boundary, and audit state |
| Data treatment | Deterministic reference scenario with one unobtrusive note |

## Final review

- [ ] Scene durations total 120 seconds and match the storyboard ranges.
- [ ] Every WebVTT cue matches the narration-beat contract enforced by tests.
- [ ] `build-manifest.json` matches both the current render sources and the checked-in MP4.
- [ ] The command-center map and gate pins remain readable at normal playback size.
- [ ] Both Agent Operations scenes keep retained objective context, fixed plan, evidence, trace, and
  human decision controls readable together at desktop and mobile widths.
- [ ] The reference-scenario note appears consistently without competing with product callouts.
- [ ] Captions use no more than two lines and do not cover controls or status.
- [ ] No token, credential, browser notification, or unrelated desktop content is visible.
- [ ] The architecture card distinguishes the planner, tool/policy boundary, human authority,
  inference worker, and deployment integration seams.
- [ ] No physical gate action or numeric interface value is described as an observed outcome.
- [ ] Muted playback still communicates the workflow and system shape.
