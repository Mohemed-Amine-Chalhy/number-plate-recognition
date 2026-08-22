"""Pure post-processing rules for detected license-plate characters."""

from __future__ import annotations

import re
import string
from collections.abc import Mapping, Sequence

from number_plate_recognition.config import DEFAULT_PLATE_PATTERN
from number_plate_recognition.domain import CharacterResult, RawDetection


def decode_character(label: str, mapping: Mapping[str, str]) -> str | None:
    """Translate a model class name to its display value."""

    value = mapping.get(label, label).strip().upper()
    supported_symbols = frozenset(string.digits + string.ascii_uppercase)
    return value if len(value) == 1 and value in supported_symbols else None


def reconstruct_characters(
    detections: Sequence[RawDetection],
    *,
    mapping: Mapping[str, str],
    plate_pattern: str = DEFAULT_PLATE_PATTERN,
) -> tuple[str, tuple[CharacterResult, ...], float, bool]:
    """Sort detections left-to-right and reconstruct a plate string."""

    ordered = sorted(detections, key=lambda detection: detection.box.x1)
    decoded = ((detection, decode_character(detection.label, mapping)) for detection in ordered)
    characters = tuple(
        CharacterResult(
            value=value,
            raw_label=detection.label,
            confidence=detection.confidence,
            box=detection.box,
        )
        for detection, value in decoded
        if value is not None
    )
    text = "".join(character.value for character in characters)
    confidence = (
        sum(character.confidence for character in characters) / len(characters)
        if characters
        else 0.0
    )
    format_valid = bool(text and re.fullmatch(plate_pattern, text))
    return text, characters, confidence, format_valid
