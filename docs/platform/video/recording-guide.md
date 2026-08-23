# Deterministic recording guide

← [Video package](README.md) · [Storyboard](storyboard.md)

## Recording mode

Record the console from its deterministic static-demo mode and run the control API separately for the
OpenAPI shot. This prevents a partially available API, mutable local database, or wall-clock-relative
live projection from changing the main UI take.

The source badge must say **Demo data**. That is a feature of the case-study recording, not something
to hide.

## 1. Prepare a clean environment

Close applications that may show notifications, messages, credentials, personal browser profiles,
or unrelated files. Use a dedicated browser profile/private window with no extensions.

Run repository checks before recording:

```powershell
.\scripts\bootstrap_platform.ps1
uv run --frozen python scripts/platform_quality.py check
```

The bootstrap synchronizes both locked Python environments, installs the repository hooks, and runs
the deterministic platform/model diagnostics. See the
[deployment runbook](../deployment-runbook.md#recommended-bootstrap) for Bash and deliberate
skip/no-hook options. The quality orchestrator then exercises root, service, console, and script
boundaries using the same integrated gate as CI/pre-push. If a check fails, fix or disclose it; do
not record around a known broken path.

## 2. Start the deterministic console

Serve only the static console on port 4173. API calls to the same static origin will fail cleanly and
the UI will use its version-controlled demo snapshot.

```powershell
uv run --project services/control_api --frozen python -m http.server 4173 --bind 127.0.0.1 --directory web/console
```

Open <http://127.0.0.1:4173/#/command>. Wait until the source badge changes from **Connecting** to
**Demo data** before recording.

## 3. Start the API for the OpenAPI shot

Use a new dedicated database path; do not delete or reuse an unknown local database.

```powershell
$env:CONTROL_API_DB_PATH = ".runtime/video/control-api-take.sqlite3"
$env:CONTROL_API_SEED_DEMO = "true"
uv run --project services/control_api --frozen python -m control_api
```

If that filename already contains a prior take, choose a new explicit filename such as
`control-api-take-02.sqlite3`. Open <http://127.0.0.1:8000/docs> and collapse any payload/examples
that expose demo tokens. Show only title, version, and resource groups.

## 4. Reset browser state

In the dedicated recording profile, open developer tools before capture and run:

```javascript
for (const key of ["campus.locale", "campus.theme", "campus.role", "campus.apiToken"]) {
  localStorage.removeItem(key);
}
location.hash = "/command";
location.reload();
```

Then close developer tools. Expected initial state:

- route: Command center;
- locale: English;
- direction: left-to-right;
- theme: light;
- role: Security operator;
- source: Demo data;
- selected gate: initial deterministic default;
- no modal/toast/open browser find dialog.

Reloading resets in-memory demo mutations such as an approved request or acknowledged incident.

## 5. Fix visual conditions

Set:

- viewport: 1600×900 CSS pixels;
- browser zoom: 100%;
- operating-system display scaling: recorded in take notes and unchanged;
- capture: 1920×1080, 30 fps constant;
- pointer size: default, high contrast only if needed;
- reduced motion: consistent across takes;
- no browser chrome if possible, or identical crop in every shot;
- no automatic dark mode/color-temperature shift.

Do a five-second test and inspect text sharpness, RTL layout, status colors, focus ring, and caption
safe area before the final take.

## 6. Rehearse deterministic actions

Use this sequence and do not improvise during the clean take:

1. `/#/command`: wait for Demo data.
2. Select Residential Gate on the map.
3. Open Gate workspace; select Residential Gate tab.
4. Point to recognition and access context; do **not** confirm a gate command.
5. Open Access & approvals; switch active role to Campus admin.
6. Approve the first pending synthetic request; wait for the success toast.
7. Open Operations; acknowledge the first synthetic incident.
8. Change EN → FR → AR; hold Arabic RTL for three seconds; return EN.
9. Cut to API docs at port 8000.
10. Cut to the rendered architecture diagram.
11. Return/reload `/#/command` for the closing shot.

If a take fails after step 6 or 7, reload the static page to restore demo data before starting again.

## 7. Capture architecture material

Render [Architecture](../architecture.md) in a Markdown viewer that supports Mermaid. Capture only the
target container/system diagram and keep “Target architecture” in frame. If Mermaid rendering differs
between tools, export once to a fixed 1920×1080 composition and reuse it; do not modify the diagram's
meaning for visual symmetry.

## 8. Record audio

- Record narration in a quiet room at 48 kHz.
- Keep mouth-to-microphone distance and gain constant.
- Capture 10 seconds of room tone.
- Record two complete reads plus pickups for technical terms.
- Choose the clearest natural read, then edit picture to it.
- Do not use voice synthesis that implies the author personally conducted the composite interviews.

The final spoken words must remain consistent with [the disclosure](demo-data-disclosure.md).

## 9. Edit and caption

1. Place the selected voiceover and mark every storyboard boundary.
2. Fit UI shots to narration without speeding pointer motion.
3. Add concise on-screen callouts from the storyboard.
4. Add opening disclosure and label target architecture.
5. Import [captions.vtt](captions.vtt), then adjust cues to the final voice waveform.
6. Check captions at normal size and on a smaller player.
7. Normalize audio consistently and inspect peaks.
8. Export a review copy, watch once muted and once with captions only.

## 10. Determinism and disclosure check

- [ ] Demo source visible in every operational UI sequence or stated on the persistent overlay.
- [ ] UI numbers/names/plates match version-controlled demo data.
- [ ] No local live API data leaked into the static take.
- [ ] No gate command was presented as executed.
- [ ] API shot is labeled prototype and does not show demo tokens.
- [ ] Architecture shot is labeled target.
- [ ] Runtime is exactly 120 seconds.
- [ ] Captions match final audio.
- [ ] Export metadata/file name does not contain personal/customer information.

## Suggested file names

```text
campus-access-case-study-2m-v1.mp4
campus-access-case-study-2m-v1.vtt
campus-access-case-study-2m-v1-transcript.txt
campus-access-case-study-2m-v1-disclosure.txt
```

Keep raw recordings outside the repository if they contain browser/desktop context. Commit only
reviewed, intentionally published media.

## Take log

Record for each take:

| Field | Value |
| --- | --- |
| Git commit |  |
| Date/time/time zone |  |
| Browser/version |  |
| Viewport/zoom/display scale |  |
| Console source mode | Demo data |
| API database path |  |
| Voiceover take |  |
| Known deviation |  |
| Reviewer |  |
