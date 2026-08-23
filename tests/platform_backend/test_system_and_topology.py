"""Health, documentation, tenancy, roles, and campus-topology tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import uvicorn
from conftest import auth
from control_api import __main__ as control_main
from control_api.app import create_app
from control_api.config import Settings
from fastapi.testclient import TestClient


def test_health_docs_and_demo_session(client: TestClient) -> None:
    assert client.get("/health/live").json() == {
        "status": "ok",
        "service": "campus-control-api",
        "schema_version": None,
    }
    assert client.get("/health/ready").json()["schema_version"] == 1

    openapi = client.get("/openapi.json").json()
    assert openapi["info"]["title"] == "Campus Access Control API"
    assert "/api/v1/passages/{passage_id}/recognitions" in openapi["paths"]
    assert "/api/v1/passages/{passage_id}/authorization-decisions" in openapi["paths"]

    assert client.get("/api/v1/sites").status_code == 401
    session = client.get("/api/v1/session", headers=auth("demo-viewer"))
    assert session.status_code == 200
    assert session.json()["roles"] == ["viewer"]
    identities = client.get("/api/v1/demo-identities").json()
    assert {identity["token"] for identity in identities} >= {"demo-admin", "demo-edge"}


def test_network_settings_are_environment_driven_and_validated(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("CONTROL_API_DB_PATH", str(tmp_path / "settings.sqlite3"))
    monkeypatch.setenv("CONTROL_API_HOST", "0.0.0.0")  # noqa: S104 - container setting
    monkeypatch.setenv("CONTROL_API_PORT", "8181")
    settings = Settings.from_environment()
    assert settings.host == "0.0.0.0"  # noqa: S104 - validates explicit opt-in
    assert settings.port == 8181

    monkeypatch.setenv("CONTROL_API_PORT", "70000")
    with pytest.raises(ValueError, match="between 1 and 65535"):
        Settings.from_environment()


def test_module_runner_uses_configured_network_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(app_name: str, *, host: str, port: int, reload: bool) -> None:
        captured.update(app_name=app_name, host=host, port=port, reload=reload)

    monkeypatch.setenv("CONTROL_API_HOST", "127.0.0.2")
    monkeypatch.setenv("CONTROL_API_PORT", "8182")
    monkeypatch.setattr(uvicorn, "run", fake_run)
    control_main.main()
    assert captured == {
        "app_name": "control_api.app:app",
        "host": "127.0.0.2",
        "port": 8182,
        "reload": False,
    }


def test_static_console_is_mounted_without_shadowing_api(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    settings = Settings(
        database_path=tmp_path / "mounted.sqlite3",
        seed_demo_data=True,
        console_dir=repository_root / "web" / "console",
        cors_origins=(),
    )
    with TestClient(create_app(settings)) as mounted:
        assert "text/html" in mounted.get("/").headers["content-type"]
        assert mounted.get("/health/live").status_code == 200
        assert mounted.get("/docs").status_code == 200


def test_organization_isolation_and_platform_switching(client: TestClient) -> None:
    atlas_sites = client.get("/api/v1/sites", headers=auth("demo-admin"))
    assert atlas_sites.status_code == 200
    assert {site["organization_id"] for site in atlas_sites.json()} == {"org-atlas"}

    hidden = client.get(
        "/api/v1/sites/site-atlas-main",
        headers=auth("demo-rif-admin"),
    )
    assert hidden.status_code == 404
    assert hidden.headers["content-type"].startswith("application/problem+json")

    switched = client.get(
        "/api/v1/sites",
        headers=auth("demo-platform", organization_id="org-rif"),
    )
    assert [site["id"] for site in switched.json()] == ["site-rif-main"]

    forbidden_switch = client.get(
        "/api/v1/sites",
        headers=auth("demo-admin", organization_id="org-rif"),
    )
    assert forbidden_switch.status_code == 403


def test_admin_can_manage_topology_but_viewer_cannot(client: TestClient) -> None:
    site_payload = {
        "code": "HEALTH",
        "name": "Health Sciences Annex",
        "timezone": "Africa/Casablanca",
        "address": "Synthetic campus address",
        "latitude": 31.62,
        "longitude": -7.98,
    }
    assert (
        client.post("/api/v1/sites", headers=auth("demo-viewer"), json=site_payload).status_code
        == 403
    )
    site = client.post("/api/v1/sites", headers=auth("demo-admin"), json=site_payload)
    assert site.status_code == 201
    site_id = site.json()["id"]

    gate = client.post(
        "/api/v1/gates",
        headers=auth("demo-admin"),
        json={
            "site_id": site_id,
            "code": "WEST",
            "name": "West Clinical Gate",
            "direction": "bidirectional",
            "latitude": 31.6201,
            "longitude": -7.9801,
        },
    )
    assert gate.status_code == 201
    gate_id = gate.json()["id"]

    camera = client.post(
        "/api/v1/cameras",
        headers=auth("demo-admin"),
        json={
            "site_id": site_id,
            "gate_id": gate_id,
            "code": "WEST-ANPR-01",
            "name": "West ANPR Camera",
            "role": "anpr",
            "stream_profile": "plate-closeup-h264",
        },
    )
    assert camera.status_code == 201
    camera_id = camera.json()["id"]

    updated = client.patch(
        f"/api/v1/gates/{gate_id}",
        headers=auth("demo-admin"),
        json={"queue_estimate": 4, "status": "congested"},
    )
    assert updated.json()["queue_estimate"] == 4

    assert (
        client.delete(f"/api/v1/cameras/{camera_id}", headers=auth("demo-admin")).status_code == 204
    )
    assert (
        client.get(f"/api/v1/cameras/{camera_id}", headers=auth("demo-admin")).json()["status"]
        == "disabled"
    )
