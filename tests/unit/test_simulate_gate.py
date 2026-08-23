from __future__ import annotations

import pytest

from scripts.simulate_gate import JsonValue, RecognitionInput, _policy_decision


def _recognition(*, confidence: float = 0.95, valid: bool = True) -> RecognitionInput:
    return RecognitionInput(
        plate_text="12345A6",
        detection_confidence=confidence,
        recognition_confidence=confidence,
        format_valid=valid,
        model_version="test",
        status="recognized",
    )


def test_policy_allows_separator_independent_exact_match() -> None:
    decision = _policy_decision(
        _recognition(),
        [
            {
                "id": "grant-1",
                "plate_text": "12345-A-6",
                "status": "active",
                "gate_id": None,
            }
        ],
        gate_id="gate-main",
        confidence_threshold=0.8,
    )

    assert decision["outcome"] == "allowed"
    assert decision["grant_id"] == "grant-1"


@pytest.mark.parametrize(
    ("recognition", "grants", "outcome"),
    [
        (_recognition(confidence=0.6), [], "review_required"),
        (_recognition(valid=False), [], "review_required"),
        (_recognition(), [], "no_match"),
        (
            _recognition(),
            [
                {
                    "id": "grant-1",
                    "plate_text": "12345-A-6",
                    "status": "active",
                    "gate_id": "gate-other",
                }
            ],
            "review_required",
        ),
        (
            _recognition(),
            [
                {
                    "id": "grant-1",
                    "plate_text": "12345-A-6",
                    "status": "revoked",
                    "gate_id": None,
                }
            ],
            "denied",
        ),
    ],
)
def test_policy_routes_non_automatic_cases(
    recognition: RecognitionInput,
    grants: list[dict[str, JsonValue]],
    outcome: str,
) -> None:
    decision = _policy_decision(
        recognition,
        grants,
        gate_id="gate-main",
        confidence_threshold=0.8,
    )

    assert decision["outcome"] == outcome
