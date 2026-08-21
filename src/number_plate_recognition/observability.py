"""Small structured-logging helpers that avoid recognition data by default."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

_STANDARD_FIELDS = frozenset(logging.makeLogRecord({}).__dict__)
_ALLOWED_OPERATIONAL_FIELDS = frozenset(
    {
        "event",
        "request_id",
        "vehicle_count",
        "plate_region_count",
        "total_inference_ms",
        "model_role",
        "model_version",
    }
)


class JsonFormatter(logging.Formatter):
    """Render log records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(
            {
                key: value
                for key, value in record.__dict__.items()
                if key in _ALLOWED_OPERATIONAL_FIELDS and key not in _STANDARD_FIELDS
            }
        )
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str) -> None:
    """Configure the root logger once for process-level JSON output."""

    root = logging.getLogger()
    root.setLevel(level)
    if root.handlers:
        for handler in root.handlers:
            handler.setFormatter(JsonFormatter())
        return
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
