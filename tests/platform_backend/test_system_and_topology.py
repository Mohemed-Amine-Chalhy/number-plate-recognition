"""Health, documentation, tenancy, roles, and campus-topology tests."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
import uvicorn
from conftest import auth
from control_api import __main__ as control_main
from control_api.app import create_app
from control_api.config import Settings
from fastapi.testclient import TestClient

_PRIMARY_CAMPUS_GATES = {
    "gate-atlas-north": "North Gate",
    "gate-atlas-research": "North-East / Innovation Gate",
    "gate-atlas-service": "East / Logistics Gate",
    "gate-atlas-residence": "South-East Gate",
    "gate-atlas-south": "Main / South Gate",
    "gate-atlas-sports": "Sports / West Gate",
}


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


def test_primary_campus_seeds_six_complete_tenant_scoped_gate_stacks(
    client: TestClient,
) -> None:
    gates = client.get(
        "/api/v1/gates?site_id=site-atlas-main",
        headers=auth("demo-viewer"),
    )
    assert gates.status_code == 200
    assert {gate["id"]: gate["name"] for gate in gates.json()} == _PRIMARY_CAMPUS_GATES
    assert {gate["site_id"] for gate in gates.json()} == {"site-atlas-main"}

    cameras = client.get(
        "/api/v1/cameras?site_id=site-atlas-main",
        headers=auth("demo-viewer"),
    )
    assert cameras.status_code == 200
    assert {camera["gate_id"] for camera in cameras.json()} == set(_PRIMARY_CAMPUS_GATES)

    health = client.get(
        "/api/v1/device-health?site_id=site-atlas-main",
        headers=auth("demo-viewer"),
    )
    assert health.status_code == 200
    assert {report["gate_id"] for report in health.json()} == set(_PRIMARY_CAMPUS_GATES)

    # A tenant-scoped identity cannot discover either of the newly seeded gate IDs.
    for gate_id in ("gate-atlas-south", "gate-atlas-sports"):
        assert (
            client.get(
                f"/api/v1/gates/{gate_id}",
                headers=auth("demo-rif-admin"),
            ).status_code
            == 404
        )
    rif_gates = client.get("/api/v1/gates", headers=auth("demo-rif-admin"))
    assert [gate["id"] for gate in rif_gates.json()] == ["gate-rif-east"]


def test_seed_resolves_legitimate_gate_and_camera_code_collisions(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "colliding-topology.sqlite3"
    settings = Settings(
        database_path=database_path,
        seed_demo_data=True,
        console_dir=None,
        cors_origins=(),
    )
    with TestClient(create_app(settings)):
        pass

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "DELETE FROM device_health WHERE id IN (?, ?)",
            ("health-atlas-south", "health-atlas-sports"),
        )
        connection.execute(
            "DELETE FROM cameras WHERE id IN (?, ?)",
            ("camera-atlas-south-anpr", "camera-atlas-sports-anpr"),
        )
        connection.execute(
            "DELETE FROM gates WHERE id IN (?, ?)",
            ("gate-atlas-south", "gate-atlas-sports"),
        )
        connection.executemany(
            "INSERT INTO gates "
            "(id, organization_id, site_id, code, name, direction, latitude, longitude, "
            "status, queue_estimate, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    "gate-user-south",
                    "org-atlas",
                    "site-atlas-main",
                    "SOUTH",
                    "Existing South Operations Gate",
                    "bidirectional",
                    31.623,
                    -7.988,
                    "operational",
                    0,
                    "2026-08-24T08:00:00+00:00",
                ),
                (
                    "gate-user-sports",
                    "org-atlas",
                    "site-atlas-main",
                    "SPORTS",
                    "Existing Sports Operations Gate",
                    "bidirectional",
                    31.624,
                    -7.992,
                    "operational",
                    0,
                    "2026-08-24T08:00:00+00:00",
                ),
            ],
        )
        connection.executemany(
            "INSERT INTO cameras "
            "(id, organization_id, site_id, gate_id, code, name, role, stream_profile, "
            "status, last_seen_at, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    "camera-user-south",
                    "org-atlas",
                    "site-atlas-main",
                    "gate-user-south",
                    "SOUTH-ANPR-01",
                    "Existing South Camera",
                    "anpr",
                    "custom-primary",
                    "online",
                    "2026-08-24T08:00:00+00:00",
                    "2026-08-24T08:00:00+00:00",
                ),
                (
                    "camera-user-sports",
                    "org-atlas",
                    "site-atlas-main",
                    "gate-user-sports",
                    "SPORTS-ANPR-01",
                    "Existing Sports Camera",
                    "anpr",
                    "custom-primary",
                    "online",
                    "2026-08-24T08:00:00+00:00",
                    "2026-08-24T08:00:00+00:00",
                ),
            ],
        )
        connection.commit()

    expected_codes = {
        "gate-atlas-south": "SOUTH-DEMO",
        "gate-atlas-sports": "SPORTS-DEMO",
    }
    expected_camera_codes = {
        "camera-atlas-south-anpr": "SOUTH-ANPR-01-DEMO",
        "camera-atlas-sports-anpr": "SPORTS-ANPR-01-DEMO",
    }
    with TestClient(create_app(settings)) as upgraded:
        assert {
            gate_id: upgraded.get(
                f"/api/v1/gates/{gate_id}",
                headers=auth("demo-viewer"),
            ).json()["code"]
            for gate_id in expected_codes
        } == expected_codes
        assert {
            camera_id: upgraded.get(
                f"/api/v1/cameras/{camera_id}",
                headers=auth("demo-viewer"),
            ).json()["code"]
            for camera_id in expected_camera_codes
        } == expected_camera_codes
        reports = upgraded.get(
            "/api/v1/device-health?site_id=site-atlas-main",
            headers=auth("demo-viewer"),
        ).json()
        assert {report["id"] for report in reports} >= {
            "health-atlas-south",
            "health-atlas-sports",
        }
        assert (
            upgraded.get(
                "/api/v1/gates/gate-user-south",
                headers=auth("demo-viewer"),
            ).json()["code"]
            == "SOUTH"
        )

    # A second startup must retain the same deterministic codes.
    with TestClient(create_app(settings)) as restarted:
        assert {
            gate_id: restarted.get(
                f"/api/v1/gates/{gate_id}",
                headers=auth("demo-viewer"),
            ).json()["code"]
            for gate_id in expected_codes
        } == expected_codes


def test_legacy_service_relocation_preserves_custom_fields_and_aligns_dependents(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "legacy-topology.sqlite3"
    settings = Settings(
        database_path=database_path,
        seed_demo_data=True,
        console_dir=None,
        cors_origins=(),
    )
    with TestClient(create_app(settings)):
        pass

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE gates SET site_id = ?, code = ?, name = ?, direction = ?, latitude = ?, "
            "longitude = ?, status = ?, queue_estimate = ? WHERE id = ?",
            (
                "site-atlas-innovation",
                "SERVICE",
                "Custom East Dock",
                "outbound",
                31.111,
                -7.222,
                "offline",
                17,
                "gate-atlas-service",
            ),
        )
        connection.execute(
            "UPDATE cameras SET site_id = ?, code = ?, name = ?, role = ?, stream_profile = ?, "
            "status = ?, last_seen_at = ? WHERE id = ?",
            (
                "site-atlas-innovation",
                "SERVICE-ANPR-01",
                "Custom East Dock Camera",
                "thermal-overview",
                "custom-h265",
                "offline",
                "2026-08-24T07:15:00+00:00",
                "camera-atlas-service-anpr",
            ),
        )
        connection.execute(
            "INSERT INTO gates "
            "(id, organization_id, site_id, code, name, direction, latitude, longitude, "
            "status, queue_estimate, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "gate-user-east",
                "org-atlas",
                "site-atlas-main",
                "EAST",
                "Existing East Gate",
                "inbound",
                31.626,
                -7.985,
                "operational",
                0,
                "2026-08-24T08:00:00+00:00",
            ),
        )
        connection.execute(
            "INSERT INTO cameras "
            "(id, organization_id, site_id, gate_id, code, name, role, stream_profile, "
            "status, last_seen_at, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "camera-user-east",
                "org-atlas",
                "site-atlas-main",
                "gate-user-east",
                "EAST-ANPR-01",
                "Existing East Camera",
                "anpr",
                "custom-primary",
                "online",
                "2026-08-24T08:00:00+00:00",
                "2026-08-24T08:00:00+00:00",
            ),
        )
        for table_name, row_id in (
            ("access_requests", "request-atlas-supplier-approved"),
            ("access_grants", "grant-atlas-supplier"),
            ("incidents", "incident-atlas-camera"),
            ("device_health", "health-atlas-service"),
        ):
            connection.execute(
                f"UPDATE {table_name} SET site_id = ? WHERE id = ?",  # noqa: S608
                ("site-atlas-innovation", row_id),
            )
        connection.commit()

    with TestClient(create_app(settings)) as upgraded:
        service_gate = upgraded.get(
            "/api/v1/gates/gate-atlas-service",
            headers=auth("demo-viewer"),
        ).json()
        assert (service_gate["site_id"], service_gate["code"], service_gate["name"]) == (
            "site-atlas-main",
            "EAST-DEMO",
            "Custom East Dock",
        )
        assert (
            service_gate["direction"],
            service_gate["latitude"],
            service_gate["longitude"],
            service_gate["status"],
            service_gate["queue_estimate"],
        ) == ("outbound", 31.111, -7.222, "offline", 17)
        service_camera = upgraded.get(
            "/api/v1/cameras/camera-atlas-service-anpr",
            headers=auth("demo-viewer"),
        ).json()
        assert (service_camera["site_id"], service_camera["code"], service_camera["name"]) == (
            "site-atlas-main",
            "EAST-ANPR-01-DEMO",
            "Custom East Dock Camera",
        )
        assert (
            service_camera["role"],
            service_camera["stream_profile"],
            service_camera["status"],
        ) == (
            "thermal-overview",
            "custom-h265",
            "offline",
        )
        assert service_camera["last_seen_at"].startswith("2026-08-24T07:15:00")
        assert (
            upgraded.get(
                "/api/v1/access-requests/request-atlas-supplier-approved",
                headers=auth("demo-viewer"),
            ).json()["site_id"]
            == "site-atlas-main"
        )
        assert (
            upgraded.get(
                "/api/v1/access-grants/grant-atlas-supplier",
                headers=auth("demo-viewer"),
            ).json()["site_id"]
            == "site-atlas-main"
        )
        assert (
            upgraded.get(
                "/api/v1/incidents/incident-atlas-camera",
                headers=auth("demo-viewer"),
            ).json()["site_id"]
            == "site-atlas-main"
        )
        service_health = upgraded.get(
            "/api/v1/device-health?site_id=site-atlas-main",
            headers=auth("demo-viewer"),
        ).json()
        assert any(report["id"] == "health-atlas-service" for report in service_health)


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
