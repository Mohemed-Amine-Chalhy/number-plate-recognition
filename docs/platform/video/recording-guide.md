# Deterministic capture and recording guide

← [Video package](README.md) · [Storyboard](storyboard.md)

Use this workflow whenever the interface, agent contract, campus map, generated data, narration, or
scene copy changes. The goal is a sharp, reproducible product cut whose stills, captions, and audio
share one 120-second timeline.

## 1. Verify the workspace

Run the repository checks before capturing media:

```powershell
.\scripts\bootstrap_platform.ps1
uv run --frozen python scripts/platform_quality.py check
```

Close applications that may show notifications or unrelated content. Use a dedicated browser
profile with no extensions, saved passwords, or personal autofill.

## 2. Start the deterministic console

Serve the static console so its version-controlled reference records remain stable:

```powershell
uv run --frozen python -m http.server 4173 --bind 127.0.0.1 --directory web/console
```

Open <http://127.0.0.1:4173/#/command> and wait for the source indicator to settle before capture.

## 3. Reset browser state

Open developer tools, run the following reset, and close developer tools again:

```javascript
for (const key of ["campus.locale", "campus.theme", "campus.role", "campus.apiToken"]) {
  localStorage.removeItem(key);
}
location.hash = "/command";
location.reload();
```

Confirm the starting state:

- Command center route;
- English, left-to-right, light theme;
- Security operator role;
- initial gate selected;
- Agent Operations opens with the deterministic reference trajectory and a visible approval gate;
- no dialog, toast, browser find box, or open developer tools.

Reloading the static page restores deterministic requests and incidents after interactive review.

## 4. Lock capture conditions

Use the same settings for every desktop still:

- viewport: 1600×900 CSS pixels;
- browser zoom: 100%;
- consistent operating-system display scale;
- no browser chrome where possible;
- hidden scrollbars only when they do not obscure scroll position;
- reduced-motion preference enabled;
- no automatic color-temperature or theme switching.

Use a 390×844 CSS-pixel viewport for the mobile Arabic capture. Export final desktop stills at a
consistent 16:9 composition and inspect text at 100% before using them as video sources.

## 5. Capture the product sequence

Capture the source images in storyboard order:

| Scene | Route/state | Capture requirement |
| --- | --- | --- |
| S02 | `/#/command` | Full command hierarchy, illustrated campus footprint, six gate pins, source state, and selected-gate detail. |
| S03 | `/#/agent` | Retained objective context, fixed intent-selected plan, tool allowlist, completed evidence steps, policy/trace versions, and pending human decision in one frame. |
| S04 | `/#/gates` | Plate candidates, camera state, matching access context, and gate control in one frame. |
| S05 | `/#/access` | Prioritized requests plus the fields and review actions that make a request inspectable. |
| S06 | `/#/setup` | Tenant, topology, device, locale, time-zone, and API configuration surface. |
| S07 | `/#/agent` | Mobile width, dark theme, Arabic locale, right-to-left Agent Operations layout, visible run scope/evidence, and pending human decision. |

Use these filenames exactly:

```text
docs/platform/assets/command-center.png
docs/platform/assets/agent-operations.png
docs/platform/assets/gate-workspace.png
docs/platform/assets/access-approvals.png
docs/platform/assets/operations.png
docs/platform/assets/campus-setup.png
docs/platform/assets/mobile-rtl.png
```

For the command center, verify that the illustrated map loads before capture and that every pin
remains aligned with its landmark at both normal and reduced widths.

## 6. Prepare narration

Use [captions.vtt](captions.vtt) as the spoken-text and timing master:

- keep each narration beat inside its cue range;
- record at 48 kHz with consistent microphone distance and gain;
- capture one complete read plus pickups for product and protocol names;
- leave clean pauses at all nine scene boundaries;
- normalize consistently and inspect for clipping;
- keep music, when used, clearly below speech.

The Windows system voice is an editing draft and may vary with installed voice/runtime versions. A
reviewed WAV can replace it without changing the visual timeline.

## 7. Build the reference cut

After the final stills and caption text are in place, run:

```powershell
uv run --group media --frozen python scripts/build_demo_video.py
```

Or provide a narration recording:

```powershell
uv run --group media --frozen python scripts/build_demo_video.py `
  --narration-wav path/to/reviewed-narration.wav
```

The builder validates that scene durations total 120 seconds and that every WebVTT cue matches its
configured narration beat before encoding begins. It snapshots the render script, captions,
storyboard, optional title-card logo, and screenshot digests before encoding, then writes
`build-manifest.json`
only after the completed MP4 is in place. Run the video unit test after the build; it rejects a
missing manifest, changed source, or changed binary:

```powershell
uv run --frozen pytest tests/unit/test_demo_video.py -q --no-cov `
  --basetemp .runtime/pytest-video
```

## 8. Review the export

- [ ] File is 1920×1080, constant 30 fps, and exactly 120 seconds.
- [ ] Scene boundaries match [storyboard.md](storyboard.md).
- [ ] Captions match the final audio and remain within a two-line safe area.
- [ ] `Reference scenario · generated operational data` is readable but visually secondary.
- [ ] The campus map, gate pins, status, and selected-gate details survive the video crop.
- [ ] The agent run is visibly a reference trajectory and its evidence, scope, trace metadata, and
  approval boundary survive the crop.
- [ ] The mobile Arabic frame is Agent Operations—not the command map—and keeps right-to-left run
  evidence and the human decision boundary legible.
- [ ] The source/output digests in `build-manifest.json` pass the video freshness test.
- [ ] No credentials, private URLs, notifications, or unrelated desktop content appear.
- [ ] Planner, tools, policy, human decision, inference, and edge seams remain distinct on the
  architecture card.
- [ ] Audio is clear, consistently leveled, and free from alert sounds.
- [ ] Muted playback still communicates the complete workflow.

## Take log

| Field | Value |
| --- | --- |
| Git commit |  |
| Capture date/time zone |  |
| Browser/version |  |
| Viewport/zoom/display scale |  |
| Console source state |  |
| Narration take |  |
| Known deviation |  |
| Reviewer |  |
