# Two-minute product video package

← [Platform documentation index](../README.md)

This package builds a repeatable 120-second walkthrough of Campus Access. The narrative starts with
the operating problem, follows the workflow across the command center and individual gates, and
closes with the reusable system boundaries behind the interface.

## Watch or rebuild

[Watch the checked-in two-minute MP4](campus-access-case-study-2m-v1.mp4).

The render target is 1920×1080 H.264 at 30 fps with 48 kHz AAC audio. Its source of truth is:

- the scene and narration-beat durations in `scripts/build_demo_video.py`;
- the exact spoken text and cue ranges in [captions.vtt](captions.vtt);
- the matching visual plan in [storyboard.md](storyboard.md);
- the browser-verified product captures in [../assets](../assets/README.md).

On Windows, rebuild the deterministic draft narration with:

```powershell
uv run --group media --frozen python scripts/build_demo_video.py
```

For a reviewed voice recording, provide a WAV file:

```powershell
uv run --group media --frozen python scripts/build_demo_video.py `
  --narration-wav path/to/reviewed-narration.wav
```

Use `--no-narration` for a silent review copy. Encoding happens in ignored runtime files and replaces
the requested output only after the complete render succeeds.

## Narrative structure

| Time | Product focus |
| --- | --- |
| 00:00–00:24 | Introduce Campus Access and the campus-wide command view. |
| 00:24–00:56 | Move from plate evidence at one gate to a structured access decision. |
| 00:56–01:27 | Show incident ownership, device health, and campus configuration. |
| 01:27–01:53 | Demonstrate responsive localization and modular system boundaries. |
| 01:53–02:00 | Close on the complete runnable platform slice. |

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

## Output specification

| Property | Setting |
| --- | --- |
| Duration | Exactly 120 seconds |
| Canvas | 1920×1080, 16:9 |
| Source viewport | 1600×900 at 100% zoom for desktop captures |
| Frame rate | 30 fps constant |
| Audio | 48 kHz AAC, normalized consistently, no clipping |
| Captions | Sidecar WebVTT matching every narration beat |
| UI coverage | Light desktop flow plus one dark Arabic right-to-left view |
| Data treatment | Deterministic reference scenario with one unobtrusive note |

## Final review

- [ ] Scene durations total 120 seconds and match the storyboard ranges.
- [ ] Every WebVTT cue matches the narration-beat contract enforced by tests.
- [ ] The command-center map and gate pins remain readable at normal playback size.
- [ ] The reference-scenario note appears consistently without competing with product callouts.
- [ ] Captions use no more than two lines and do not cover controls or status.
- [ ] No token, credential, browser notification, or unrelated desktop content is visible.
- [ ] The architecture card distinguishes built components from integration seams.
- [ ] No physical gate action or numeric interface value is described as an observed outcome.
- [ ] Muted playback still communicates the workflow and system shape.
