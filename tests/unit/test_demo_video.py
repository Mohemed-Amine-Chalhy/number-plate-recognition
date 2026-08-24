from __future__ import annotations

import math
import re

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

    assert transcript.startswith("Campus Access coordinates vehicle entry")
    assert transcript.endswith("tests, runbooks, and reproducible delivery.")
    assert "-->" not in transcript
    for unsupported_claim in ("sold to", "deployed at", "interviewed", "measured reduction"):
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
