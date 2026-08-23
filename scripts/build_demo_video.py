#!/usr/bin/env python3
"""Build the deterministic two-minute Campus Access case-study video.

The generator uses checked-in, browser-verified screenshots and burns concise
disclosure/callout text into every scene. On Windows it can synthesize a
replaceable narration track through the operating-system speech engine; on any
platform an externally recorded WAV file can be supplied instead.
"""

from __future__ import annotations

import argparse
import importlib
import math
import os
import shutil
import subprocess
import sys
import textwrap
import uuid
import wave
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, cast

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MEDIA_ROOT = REPOSITORY_ROOT / "docs" / "platform"
ASSET_ROOT = MEDIA_ROOT / "assets"
VIDEO_ROOT = MEDIA_ROOT / "video"
CANVAS = (1920, 1080)
FPS = 30
TOTAL_SECONDS = 120
ACCENT = (190, 24, 58)
INK = (25, 25, 24)
PAPER = (246, 244, 240)


@dataclass(frozen=True, slots=True)
class Scene:
    """One fixed-duration video scene."""

    kind: str
    duration_seconds: int
    title: str
    subtitle: str
    image_name: str | None = None


SCENES = (
    Scene(
        "intro",
        7,
        "Campus Access",
        "From a lost approval email to a multi-gate platform",
    ),
    Scene(
        "desktop",
        18,
        "One operating picture",
        "Queues, arrivals, reviews, and device health across every gate",
        "command-center.png",
    ),
    Scene(
        "desktop",
        20,
        "Recognition is evidence—not authority",
        "Observation, grant matching, authorization, and actuation stay separate",
        "gate-workspace.png",
    ),
    Scene(
        "desktop",
        18,
        "Typed, bounded approvals",
        "A vehicle, site, gate, and time window replace free-form email searching",
        "access-approvals.png",
    ),
    Scene(
        "desktop",
        18,
        "Operational exceptions have owners",
        "Incidents and device heartbeat remain visible after the queue moves",
        "operations.png",
    ),
    Scene(
        "desktop",
        14,
        "White-label campus setup",
        "Brand, topology, devices, locale, time zone, and API are configuration",
        "campus-setup.png",
    ),
    Scene(
        "mobile",
        10,
        "Built for the people at the gate",
        "Responsive navigation plus English, French, and Arabic RTL",
        "mobile-rtl.png",
    ),
    Scene(
        "architecture",
        9,
        "A clear path from prototype to pilot",
        "Runnable control plane today; edge, durable jobs, and actuators are target integrations",
    ),
    Scene(
        "outro",
        6,
        "Working prototype · documented path to pilot",
        "Product judgment, typed boundaries, tests, runbooks, and reproducible delivery",
    ),
)


def _narration_text() -> str:
    """Return the exact spoken transcript represented by the WebVTT cues."""

    lines = (VIDEO_ROOT / "captions.vtt").read_text(encoding="utf-8").splitlines()
    spoken_lines = [
        line.strip()
        for line in lines
        if line.strip() and line.strip() != "WEBVTT" and "-->" not in line
    ]
    if not spoken_lines:
        raise ValueError("The caption file does not contain a narration transcript")
    return " ".join(spoken_lines)


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
        Path(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ),
        Path("/System/Library/Fonts/SFNS.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _pil_to_bgr(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def _load_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read video source image: {path}")
    return image


def _title_card(*, outro: bool = False) -> np.ndarray:
    width, height = CANVAS
    canvas = Image.new("RGB", CANVAS, PAPER)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 28, height), fill=ACCENT)
    draw.ellipse((width - 540, -320, width + 160, 380), fill=(238, 221, 224))
    draw.ellipse((-220, height - 300, 420, height + 300), fill=(232, 231, 225))

    logo_path = ASSET_ROOT / "tenant-logo.png"
    if logo_path.is_file():
        logo = Image.open(logo_path).convert("RGBA")
        logo.thumbnail((270, 120), Image.Resampling.LANCZOS)
        canvas.paste(logo, (130, 100), logo)

    kicker = "ENGINEERING CASE STUDY" if not outro else "CAMPUS ACCESS"
    title = "Working prototype.\nDocumented path to pilot." if outro else "Campus Access"
    subtitle = (
        "Product judgment · typed boundaries · tests · runbooks · reproducible delivery"
        if outro
        else "A white-label, AI-assisted platform for multi-gate vehicle access"
    )
    draw.text((130, 305), kicker, font=_font(27, bold=True), fill=ACCENT, spacing=4)
    draw.multiline_text(
        (130, 360),
        title,
        font=_font(84, bold=True),
        fill=INK,
        spacing=14,
    )
    draw.multiline_text(
        (135, 590),
        "\n".join(textwrap.wrap(subtitle, width=58)),
        font=_font(36),
        fill=(72, 70, 67),
        spacing=12,
    )
    disclosure = (
        "Prototype · synthetic demonstration data · target integrations labeled"
        if not outro
        else "Recognition ≠ authorization ≠ physical actuation"
    )
    draw.rounded_rectangle((130, 885, 1120, 970), radius=22, fill=(255, 255, 255))
    draw.ellipse((165, 917, 181, 933), fill=ACCENT)
    draw.text((205, 908), disclosure, font=_font(25, bold=True), fill=INK)
    return _pil_to_bgr(canvas)


def _architecture_card() -> np.ndarray:
    canvas = Image.new("RGB", CANVAS, PAPER)
    draw = ImageDraw.Draw(canvas)
    draw.text((105, 70), "Target deployment path", font=_font(54, bold=True), fill=INK)
    draw.rounded_rectangle((1475, 74, 1815, 130), radius=20, fill=(255, 233, 238))
    draw.text((1520, 87), "TARGET ARCHITECTURE", font=_font(20, bold=True), fill=ACCENT)

    boxes = (
        ((105, 270, 405, 455), "Security console", "Implemented"),
        ((520, 270, 850, 455), "Control API", "Implemented"),
        ((985, 165, 1320, 350), "Central AI worker", "Local contract"),
        ((985, 520, 1320, 705), "Gate edge agent", "Target"),
        ((1460, 520, 1795, 705), "Camera + barrier", "Target"),
    )
    for bounds, label, status in boxes:
        fill = (255, 255, 255) if status != "Target" else (247, 235, 238)
        draw.rounded_rectangle(bounds, radius=28, fill=fill, outline=(214, 210, 204), width=3)
        draw.text((bounds[0] + 28, bounds[1] + 42), label, font=_font(31, bold=True), fill=INK)
        status_color = (29, 128, 88) if status == "Implemented" else ACCENT
        draw.text(
            (bounds[0] + 28, bounds[1] + 112),
            status.upper(),
            font=_font(20, bold=True),
            fill=status_color,
        )

    connectors = (
        ((405, 362), (520, 362)),
        ((850, 330), (985, 260)),
        ((850, 390), (985, 612)),
        ((1320, 612), (1460, 612)),
        ((1150, 350), (1150, 520)),
    )
    for start, end in connectors:
        draw.line((*start, *end), fill=(117, 113, 107), width=5)
        angle = math.atan2(end[1] - start[1], end[0] - start[0])
        arrow = (
            end,
            (
                end[0] - 18 * math.cos(angle - 0.55),
                end[1] - 18 * math.sin(angle - 0.55),
            ),
            (
                end[0] - 18 * math.cos(angle + 0.55),
                end[1] - 18 * math.sin(angle + 0.55),
            ),
        )
        draw.polygon(arrow, fill=(117, 113, 107))

    draw.rounded_rectangle((105, 835, 1815, 985), radius=28, fill=(31, 31, 30))
    draw.text(
        (150, 870),
        "Recognition evidence",
        font=_font(28, bold=True),
        fill=(255, 255, 255),
    )
    draw.text((555, 870), "→", font=_font(34, bold=True), fill=ACCENT)
    draw.text(
        (650, 870),
        "authorization decision",
        font=_font(28, bold=True),
        fill=(255, 255, 255),
    )
    draw.text((1130, 870), "→", font=_font(34, bold=True), fill=ACCENT)
    draw.text(
        (1220, 870),
        "actuator command",
        font=_font(28, bold=True),
        fill=(255, 255, 255),
    )
    draw.text(
        (150, 925),
        "Three records · three trust boundaries · one traceable passage",
        font=_font(23),
        fill=(190, 188, 183),
    )
    return _pil_to_bgr(canvas)


def _cover_frame(source: np.ndarray, progress: float) -> np.ndarray:
    width, height = CANVAS
    source_height, source_width = source.shape[:2]
    base_scale = max(width / source_width, height / source_height)
    zoom = 1.0 + 0.035 * progress
    scaled_width = max(width, round(source_width * base_scale * zoom))
    scaled_height = max(height, round(source_height * base_scale * zoom))
    scaled = cv2.resize(source, (scaled_width, scaled_height), interpolation=cv2.INTER_LANCZOS4)
    available_x = scaled_width - width
    available_y = scaled_height - height
    offset_x = round(available_x * (0.25 + 0.35 * progress))
    offset_y = round(available_y * (0.45 - 0.15 * progress))
    return scaled[offset_y : offset_y + height, offset_x : offset_x + width].copy()


def _mobile_frame(source: np.ndarray, background: np.ndarray, progress: float) -> np.ndarray:
    frame = _cover_frame(background, progress * 0.35)
    frame = cv2.GaussianBlur(frame, (0, 0), sigmaX=18)
    frame = cv2.addWeighted(frame, 0.35, np.full_like(frame, 18), 0.65, 0)
    target_height = 840
    scale = target_height / source.shape[0]
    target_width = round(source.shape[1] * scale)
    phone = cv2.resize(source, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
    x = int(1140 + 14 * math.sin(progress * math.pi))
    y = (CANVAS[1] - target_height) // 2
    cv2.rectangle(
        frame,
        (x - 18, y - 18),
        (x + target_width + 18, y + target_height + 18),
        (8, 8, 8),
        -1,
    )
    frame[y : y + target_height, x : x + target_width] = phone
    return frame


def _overlay(frame: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    canvas = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", CANVAS, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rounded_rectangle((70, 815, 1375, 1015), radius=28, fill=(18, 18, 18, 222))
    draw.rectangle((70, 815, 82, 1015), fill=(*ACCENT, 255))
    draw.text((125, 850), title, font=_font(39, bold=True), fill=(255, 255, 255, 255))
    wrapped = "\n".join(textwrap.wrap(subtitle, width=78))
    draw.multiline_text(
        (128, 912),
        wrapped,
        font=_font(25),
        fill=(218, 216, 211, 255),
        spacing=6,
    )
    draw.rounded_rectangle((1495, 905, 1848, 976), radius=20, fill=(255, 255, 255, 238))
    draw.ellipse((1530, 932, 1548, 950), fill=(*ACCENT, 255))
    draw.text(
        (1572, 922),
        "SYNTHETIC DEMO",
        font=_font(20, bold=True),
        fill=(*INK, 255),
    )
    return _pil_to_bgr(Image.alpha_composite(canvas, overlay))


def _scene_base(scene: Scene, progress: float, sources: dict[str, np.ndarray]) -> np.ndarray:
    if scene.kind == "intro":
        return _title_card()
    if scene.kind == "outro":
        return _title_card(outro=True)
    if scene.kind == "architecture":
        return _architecture_card()
    if scene.image_name is None:
        raise ValueError(f"Scene {scene.title!r} has no source image")
    source = sources[scene.image_name]
    if scene.kind == "mobile":
        return _mobile_frame(source, sources["command-center.png"], progress)
    return _cover_frame(source, progress)


def _scene_frame(scene: Scene, progress: float, sources: dict[str, np.ndarray]) -> np.ndarray:
    frame = _scene_base(scene, progress, sources)
    if scene.kind in {"intro", "outro", "architecture"}:
        return frame
    return _overlay(frame, scene.title, scene.subtitle)


def _smoothstep(value: float) -> float:
    clipped = min(1.0, max(0.0, value))
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _ffmpeg_executable() -> str:
    try:
        module = importlib.import_module("imageio_ffmpeg")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The optional media group is missing. Run with "
            "'uv run --group media --frozen python scripts/build_demo_video.py'."
        ) from exc
    getter = cast(Callable[[], str], module.get_ffmpeg_exe)
    return getter()


def _supports_automatic_narration() -> bool:
    """Keep the OS branch runtime-driven so every platform type-checks both paths."""

    return sys.platform == "win32"


def _write_video(silent_path: Path, *, ffmpeg: str) -> None:
    source_names = {scene.image_name for scene in SCENES if scene.image_name is not None} | {
        "command-center.png"
    }
    sources = {name: _load_bgr(ASSET_ROOT / name) for name in source_names}
    arguments = (
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{CANVAS[0]}x{CANVAS[1]}",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "22",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(silent_path),
    )
    process = subprocess.Popen(  # noqa: S603 - executable and argv are internally resolved
        arguments,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stream = cast(BinaryIO, process.stdin)
    stderr_stream = cast(BinaryIO, process.stderr)
    transition_frames = int(FPS * 0.65)
    previous_end: np.ndarray | None = None
    try:
        for scene in SCENES:
            frame_count = scene.duration_seconds * FPS
            for index in range(frame_count):
                progress = index / max(1, frame_count - 1)
                frame = _scene_frame(scene, progress, sources)
                if previous_end is not None and index < transition_frames:
                    alpha = _smoothstep((index + 1) / transition_frames)
                    frame = cv2.addWeighted(previous_end, 1.0 - alpha, frame, alpha, 0)
                stream.write(np.ascontiguousarray(frame).tobytes())
            previous_end = _scene_frame(scene, 1.0, sources)
    except BrokenPipeError as exc:
        stderr = stderr_stream.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"FFmpeg stopped while encoding: {stderr}") from exc
    finally:
        stream.close()
    stderr = stderr_stream.read().decode("utf-8", errors="replace")
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"FFmpeg video encoding failed ({return_code}): {stderr}")


def _synthesize_windows_narration(path: Path, *, voice: str) -> None:
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if powershell is None:
        raise RuntimeError(
            "No Windows speech engine was found; provide --narration-wav or use --no-narration."
        )
    command = (
        "Add-Type -AssemblyName System.Speech; "
        "$speaker = [System.Speech.Synthesis.SpeechSynthesizer]::new(); "
        "try { "
        "$speaker.SelectVoice($env:CAMPUS_VIDEO_VOICE); "
        "$speaker.Rate = -1; "
        "$speaker.Volume = 100; "
        "$speaker.SetOutputToWaveFile($env:CAMPUS_VIDEO_WAV); "
        "$speaker.Speak($env:CAMPUS_VIDEO_NARRATION); "
        "} finally { $speaker.Dispose() }"
    )
    environment = os.environ.copy()
    environment.update(
        {
            "CAMPUS_VIDEO_VOICE": voice,
            "CAMPUS_VIDEO_WAV": str(path),
            "CAMPUS_VIDEO_NARRATION": _narration_text(),
        }
    )
    completed = subprocess.run(  # noqa: S603 - fixed script and explicit argv; no shell
        (powershell, "-NoProfile", "-NonInteractive", "-Command", command),
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"Windows narration synthesis failed: {detail}")


def _wave_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as audio:
        return audio.getnframes() / audio.getframerate()


def _atempo_filter(ratio: float) -> str:
    factors: list[float] = []
    remaining = ratio
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    factors.append(remaining)
    return ",".join(f"atempo={factor:.6f}" for factor in factors)


def _mux_narration(
    silent_path: Path,
    narration_path: Path,
    output_path: Path,
    *,
    ffmpeg: str,
) -> None:
    narration_seconds = _wave_duration(narration_path)
    target_speech_seconds = TOTAL_SECONDS - 5
    tempo_ratio = narration_seconds / target_speech_seconds
    audio_filter = (
        f"{_atempo_filter(tempo_ratio)},"
        f"apad=pad_dur={TOTAL_SECONDS},atrim=duration={TOTAL_SECONDS},"
        "afade=t=out:st=118:d=2"
    )
    arguments = (
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(silent_path),
        "-i",
        str(narration_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-filter:a",
        audio_filter,
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-ar",
        "48000",
        "-t",
        str(TOTAL_SECONDS),
        "-movflags",
        "+faststart",
        "-metadata",
        "title=Campus Access - engineering case study",
        "-metadata",
        "comment=Prototype using synthetic demonstration data",
        str(output_path),
    )
    completed = subprocess.run(  # noqa: S603 - executable and argv are internally resolved
        arguments,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(f"FFmpeg audio mux failed: {completed.stderr.strip()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=VIDEO_ROOT / "campus-access-case-study-2m-v1.mp4",
    )
    parser.add_argument("--narration-wav", type=Path, help="use a reviewed external WAV narration")
    parser.add_argument(
        "--no-narration",
        action="store_true",
        help="emit a silent video; the checked-in WebVTT remains available",
    )
    parser.add_argument(
        "--voice",
        default=os.getenv("CAMPUS_DEMO_VOICE", "Microsoft Zira Desktop"),
        help="Windows speech voice used for the reproducible draft narration",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.no_narration and arguments.narration_wav is not None:
        raise SystemExit("--no-narration and --narration-wav are mutually exclusive")
    if sum(scene.duration_seconds for scene in SCENES) != TOTAL_SECONDS:
        raise SystemExit("Video scene durations must total exactly 120 seconds")

    output_path = arguments.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_root = REPOSITORY_ROOT / ".runtime" / "video"
    runtime_root.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    silent_path = runtime_root / f"silent-{run_id}.mp4"
    muxed_path = runtime_root / f"muxed-{run_id}.mp4"
    synthesized_path = runtime_root / f"narration-{run_id}.wav"
    ffmpeg = _ffmpeg_executable()

    print("Encoding 1920x1080, 30 fps, two-minute visual track...", flush=True)
    try:
        _write_video(silent_path, ffmpeg=ffmpeg)
        if arguments.no_narration:
            silent_path.replace(output_path)
        else:
            narration_path = arguments.narration_wav
            if narration_path is None:
                if not _supports_automatic_narration():
                    raise RuntimeError(
                        "Automatic narration is Windows-only; provide --narration-wav "
                        "or use --no-narration."
                    )
                print(f"Synthesizing draft narration with {arguments.voice!r}...", flush=True)
                _synthesize_windows_narration(synthesized_path, voice=arguments.voice)
                narration_path = synthesized_path
            narration_path = narration_path.expanduser().resolve()
            if not narration_path.is_file():
                raise FileNotFoundError(f"Narration WAV does not exist: {narration_path}")
            _mux_narration(silent_path, narration_path, muxed_path, ffmpeg=ffmpeg)
            muxed_path.replace(output_path)
    finally:
        silent_path.unlink(missing_ok=True)
        muxed_path.unlink(missing_ok=True)
        synthesized_path.unlink(missing_ok=True)

    size_megabytes = output_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {output_path} ({size_megabytes:.1f} MiB, {TOTAL_SECONDS}s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
