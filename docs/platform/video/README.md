# Two-minute case-study video package

← [Platform documentation index](../README.md)

This package produces a professional, repeatable 120-second walkthrough without claiming that
synthetic records are live, composite feedback is verified research, or target camera/edge services
are already deployed.

## Generated reference video

[Watch the checked-in two-minute MP4](campus-access-case-study-2m-v1.mp4).

The reference render is 1920×1080 H.264 at 30 fps with a 48 kHz AAC draft narration track. It was
generated from the reviewed screenshots, the transcript represented by `captions.vtt`, and the
explicit implementation/target labels in `scripts/build_demo_video.py`. The Windows system voice is
a reproducible draft, not a voice clone or evidence of a real interview. Replace it with a reviewed
recording when publishing a personal portfolio cut:

```bash
uv run --group media --frozen python scripts/build_demo_video.py \
  --narration-wav path/to/reviewed-narration.wav
```

Running the command without `--narration-wav` on Windows regenerates the deterministic draft. Use
`--no-narration` for a silent review copy. The generator writes through ignored runtime temporaries
and only replaces the requested output after encoding succeeds.

## Deliverables

- [Generated two-minute reference MP4](campus-access-case-study-2m-v1.mp4)
- [Storyboard, timed voiceover, and shot list](storyboard.md)
- [Deterministic recording guide](recording-guide.md)
- [WebVTT captions](captions.vtt)
- [Demo-data and branding disclosure](demo-data-disclosure.md)
- [Deterministic reference screenshots](../assets/README.md)

## Reference captures

The checked-in [product-media set](../assets/README.md) provides deterministic visual references for
the command center, gate workspace, access review, operations/health, white-label setup, and mobile
Arabic/RTL layout. Use those stills as approved B-roll or framing references when a UI interaction
is not required. Keep the Demo data badge or an equivalent persistent disclosure in frame; the
mobile crop needs an added **Synthetic demo data** label.

## Editorial position

The video should communicate three things:

1. the repository evolved from a working recognition pipeline into a broader product/system case
   study;
2. the prototype makes operator, access, tenancy, degraded-state, and engineering boundaries
   tangible;
3. edge-camera connectivity and asynchronous production infrastructure are target architecture, not
   presented as completed deployment.

Use the words **prototype**, **synthetic demo data**, and **target architecture** on screen or in
voiceover. Do not use “in production,” “deployed at UM6P,” “validated with users,” or invented
performance/accuracy claims.

## Output specification

| Property | Setting |
| --- | --- |
| Duration | Exactly 2:00, excluding optional platform player slate |
| Canvas | 1920×1080, 16:9 |
| Capture viewport | 1600×900 at 100% zoom, scaled uniformly to canvas |
| Frame rate | 30 fps constant |
| Audio | 48 kHz, mono or stereo; normalized consistently, no clipping |
| Captions | Sidecar WebVTT plus optional burned-in review copy |
| UI theme/locale | Light, English for main take; brief Arabic RTL shot |
| Data mode | Deterministic demo, visibly labeled |

## Review checklist

- [ ] Runtime is 120 seconds.
- [ ] Opening or first UI shot says “Prototype · synthetic demo data.”
- [ ] Source badge reads Demo data during UI shots.
- [ ] No real name, plate, email, token, credential, URL secret, or notification is captured.
- [ ] No gate command is confirmed or described as physically executed.
- [ ] Composite research is not described as completed interviews.
- [ ] Target edge/worker architecture is called target/future architecture.
- [ ] UM6P demo branding follows the disclosure and is not framed as deployment endorsement.
- [ ] Captions match final voiceover and do not cover controls/status.
- [ ] A silent viewing still communicates product, prototype status, and outcome.
