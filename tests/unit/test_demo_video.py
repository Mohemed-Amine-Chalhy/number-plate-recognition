from __future__ import annotations

import math

import pytest

from scripts import build_demo_video as video


def test_storyboard_is_exactly_two_minutes() -> None:
    assert sum(scene.duration_seconds for scene in video.SCENES) == video.TOTAL_SECONDS == 120


def test_narration_is_loaded_from_the_reviewed_captions() -> None:
    transcript = video._narration_text()

    assert transcript.startswith("This is Campus Access")
    assert transcript.endswith("as well as working code.")
    assert "-->" not in transcript


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
