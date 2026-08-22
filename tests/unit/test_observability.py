from __future__ import annotations

import json
import logging

from number_plate_recognition.observability import JsonFormatter


def test_json_formatter_includes_safe_structured_fields() -> None:
    record = logging.LogRecord(
        name="npr.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="request complete",
        args=(),
        exc_info=None,
    )
    record.__dict__.update(
        {
            "request_id": "request-1",
            "vehicle_count": 2,
            "plate": "sensitive-value",
            "image_path": "/private/image.jpg",
        }
    )

    payload = json.loads(JsonFormatter().format(record))

    assert payload["level"] == "INFO"
    assert payload["message"] == "request complete"
    assert payload["request_id"] == "request-1"
    assert payload["vehicle_count"] == 2
    assert "plate" not in payload
    assert "image_path" not in payload
    assert "timestamp" in payload
