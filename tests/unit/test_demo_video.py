from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

from scripts import build_demo_video as video


def _clock_seconds(value: str) -> int:
    minutes, seconds = value.split(":")
    return int(minutes) * 60 + int(seconds)


def _storyboard_rows(prefix: str) -> list[tuple[int, int, str]]:
    storyboard = (video.VIDEO_ROOT / "storyboard.md").read_text(encoding="utf-8")
    pattern = re.compile(
        rf"^\| {prefix}\d{{2}} \| (\d{{2}}:\d{{2}})\u2013(\d{{2}}:\d{{2}}) \| (.+?) \|$",
        re.MULTILINE,
    )
    return [
        (_clock_seconds(start), _clock_seconds(end), content)
        for start, end, content in pattern.findall(storyboard)
    ]


def test_scenes_captions_and_storyboard_share_one_120_second_timeline() -> None:
    assert sum(scene.duration_seconds for scene in video.SCENES) == video.TOTAL_SECONDS == 120

    expected_scenes = video._scene_ranges()
    storyboard_scenes = [(start, end) for start, end, _ in _storyboard_rows("S")]
    assert storyboard_scenes == list(expected_scenes)

    expected_cues = video._expected_caption_ranges()
    caption_cues = video._caption_cues()
    actual_cues = [(cue.start_seconds, cue.end_seconds) for cue in caption_cues]
    assert actual_cues == list(expected_cues)

    storyboard_beats = _storyboard_rows("N")
    storyboard_cues = [(float(start), float(end)) for start, end, _ in storyboard_beats]
    assert storyboard_cues == list(expected_cues)
    assert [content for _, _, content in storyboard_beats] == [cue.text for cue in caption_cues]

    video._validate_timeline()


def test_narration_is_loaded_from_the_reviewed_captions() -> None:
    transcript = video._narration_text()

    assert transcript.startswith("Campus Access runs a fixed, typed gate-health trajectory")
    assert transcript.endswith("tests, evaluations, runbooks, and reproducible delivery.")
    for agentic_boundary in (
        "typed steps",
        "server-registered, gate-scoped tools",
        "human handoff",
        "scenario evaluations",
    ):
        assert agentic_boundary in transcript
    assert "retains that context without interpreting it" in transcript
    assert "-->" not in transcript
    for unsupported_claim in (
        "decomposes that goal",
        "goal-driven",
        "sold to",
        "deployed at",
        "interviewed",
        "measured reduction",
    ):
        assert unsupported_claim not in transcript.lower()


def test_reference_scenario_note_is_concise_and_stable() -> None:
    assert video.REFERENCE_NOTE == "Reference scenario · generated operational data"


@pytest.mark.parametrize("ratio", [0.2, 0.75, 1.0, 1.6, 4.5])
def test_atempo_chain_stays_in_ffmpeg_bounds(ratio: float) -> None:
    factors = [
        float(item.removeprefix("atempo=")) for item in video._atempo_filter(ratio).split(",")
    ]

    assert all(0.5 <= factor <= 2.0 for factor in factors)
    assert math.prod(factors) == pytest.approx(ratio, abs=1e-5)


def test_generated_cards_match_delivery_canvas() -> None:
    assert video._title_card().shape == (1080, 1920, 3)
    assert video._architecture_card().shape == (1080, 1920, 3)


def test_narration_flags_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit, match="mutually exclusive"):
        video.main(["--no-narration", "--narration-wav", "narration.wav"])


def test_source_snapshot_covers_every_render_input() -> None:
    snapshot = video._source_snapshot()
    sources = snapshot["sources"]

    assert isinstance(sources, dict)
    expected = {
        "scripts/build_demo_video.py",
        "docs/platform/video/captions.vtt",
        "docs/platform/video/storyboard.md",
        *(f"docs/platform/assets/{name}" for name in video._source_image_names()),
    }
    if (video.ASSET_ROOT / "tenant-logo.png").is_file():
        expected.add("docs/platform/assets/tenant-logo.png")
    assert expected == set(sources)


def test_build_manifest_detects_a_changed_source(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    output = tmp_path / "walkthrough.mp4"
    manifest = tmp_path / "build-manifest.json"
    source.write_text("first render", encoding="utf-8")
    output.write_bytes(b"encoded-video")

    original_snapshot = video._source_snapshot((source,))
    payload = video._manifest_payload(
        output,
        source_snapshot=original_snapshot,
        narration={"mode": "silent"},
    )
    video._write_build_manifest(manifest, payload)
    video._validate_build_manifest(
        manifest,
        output,
        current_snapshot=original_snapshot,
    )

    source.write_text("changed after render", encoding="utf-8")
    with pytest.raises(ValueError, match="source digest is stale"):
        video._validate_build_manifest(
            manifest,
            output,
            current_snapshot=video._source_snapshot((source,)),
        )


def test_build_manifest_detects_a_changed_binary(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    output = tmp_path / "walkthrough.mp4"
    manifest = tmp_path / "build-manifest.json"
    source.write_text("stable source", encoding="utf-8")
    output.write_bytes(b"encoded-video")

    snapshot = video._source_snapshot((source,))
    video._write_build_manifest(
        manifest,
        video._manifest_payload(
            output,
            source_snapshot=snapshot,
            narration={"mode": "silent"},
        ),
    )
    output.write_bytes(b"encoded-vide0")

    with pytest.raises(ValueError, match="output digest does not match"):
        video._validate_build_manifest(
            manifest,
            output,
            current_snapshot=snapshot,
        )


def test_checked_in_video_matches_its_build_manifest() -> None:
    video._validate_build_manifest()
