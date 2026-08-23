"""Isolated FastAPI/SQLite fixtures for the control-plane service."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from control_api.app import create_app
from control_api.config import Settings
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    settings = Settings(
        database_path=tmp_path / "control.sqlite3",
        seed_demo_data=True,
        console_dir=None,
        cors_origins=(),
    )
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def auth(token: str, *, organization_id: str | None = None) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {token}"}
    if organization_id:
        headers["X-Organization-ID"] = organization_id
    return headers
