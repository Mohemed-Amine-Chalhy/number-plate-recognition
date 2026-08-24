"""End-to-end capture, recognition, authorization, event, and health tests."""

from __future__ import annotations

from dataclasses import dataclass

from conftest import auth
from control_api.anpr_adapter import recognition_payloads
from fastapi.testclient import TestClient


def test_recognition_and_authorization_are_independent_records(client: TestClient) -> None:
    passage = client.post(
        "/api/v1/passages",
        headers=auth("demo-edge"),
        json={
            "site_id": "site-atlas-main",
            "gate_id": "gate-atlas-north",
            "camera_id": "camera-atlas-north-anpr",
            "direction": "inbound",
            "occurred_at": "2026-08-23T10:10:00+00:00",
            "evidence_label": "Synthetic composite - vertical slice fixture",
        },
    )
    assert passage.status_code == 201
    passage_id = passage.json()["id"]

    recognition = client.post(
        f"/api/v1/passages/{passage_id}/recognitions",
        headers=auth("demo-edge"),
        json={
            "status": "recognized",
            "plate_text": "12345-A-6",
            "detection_confidence": 0.96,
            "recognition_confidence": 0.93,
            "format_valid": True,
            "model_version": "vehicle:v1/plate:v1/characters:v1",
            "source": "central_worker",
            "evidence_label": "Synthetic composite - vertical slice fixture",
        },
    )
    assert recognition.status_code == 201
    detail_before = client.get(
        f"/api/v1/passages/{passage_id}", headers=auth("demo-operator")
    ).json()
    assert len(detail_before["recognitions"]) == 1
    assert detail_before["authorization_decisions"] == []
    assert detail_before["status"] == "open"

    authorization = client.post(
        f"/api/v1/passages/{passage_id}/authorization-decisions",
        headers=auth("demo-operator"),
        json={
            "outcome": "allowed",
            "reason": "Operator verified the active staff grant",
            "source": "operator",
            "grant_id": "grant-atlas-staff",
        },
    )
    assert authorization.status_code == 201
    detail_after = client.get(
        f"/api/v1/passages/{passage_id}", headers=auth("demo-operator")
    ).json()
    assert detail_after["status"] == "completed"
    assert detail_after["authorization_decisions"][0]["outcome"] == "allowed"


def test_event_feed_has_stable_cursor_and_is_tenant_scoped(client: TestClient) -> None:
    first = client.get(
        "/api/v1/events?after_sequence=0&limit=2",
        headers=auth("demo-viewer"),
    ).json()
    assert len(first["items"]) == 2
    assert first["has_more"] is True
    assert first["next_sequence"] == first["items"][-1]["sequence"]

    second = client.get(
        f"/api/v1/events?after_sequence={first['next_sequence']}&limit=20",
        headers=auth("demo-viewer"),
    ).json()
    assert all(item["sequence"] > first["next_sequence"] for item in second["items"])
    rif = client.get("/api/v1/events", headers=auth("demo-rif-admin")).json()
    assert {item["organization_id"] for item in rif["items"]} == {"org-rif"}


def test_policy_cannot_allow_a_revoked_grant(client: TestClient) -> None:
    passage_id = client.post(
        "/api/v1/passages",
        headers=auth("demo-edge"),
        json={
            "site_id": "site-atlas-main",
            "gate_id": "gate-atlas-research",
            "camera_id": "camera-atlas-research-overview",
            "direction": "inbound",
            "occurred_at": "2026-08-23T10:15:00+00:00",
            "evidence_label": "Synthetic composite - revoked-grant fixture",
        },
    ).json()["id"]
    client.post(
        f"/api/v1/passages/{passage_id}/recognitions",
        headers=auth("demo-edge"),
        json={
            "status": "recognized",
            "plate_text": "90421-C-6",
            "detection_confidence": 0.97,
            "recognition_confidence": 0.94,
            "format_valid": True,
            "model_version": "simulator:v1",
            "evidence_label": "Synthetic composite - revoked-grant fixture",
        },
    )
    invalid_allow = client.post(
        f"/api/v1/passages/{passage_id}/authorization-decisions",
        headers=auth("demo-admin"),
        json={
            "outcome": "allowed",
            "reason": "Invalid policy result used for regression coverage",
            "source": "policy",
            "grant_id": "grant-atlas-revoked",
        },
    )
    assert invalid_allow.status_code == 422
    denied = client.post(
        f"/api/v1/passages/{passage_id}/authorization-decisions",
        headers=auth("demo-admin"),
        json={
            "outcome": "denied",
            "reason": "Matching grant is revoked",
            "source": "policy",
            "grant_id": "grant-atlas-revoked",
        },
    )
    assert denied.status_code == 201


def test_incident_health_and_dashboard_vertical_slice(client: TestClient) -> None:
    incident = client.post(
        "/api/v1/incidents",
        headers=auth("demo-operator"),
        json={
            "site_id": "site-atlas-main",
            "gate_id": "gate-atlas-north",
            "title": "Manual lane review requested",
            "severity": "warning",
            "description": "Synthetic drill raised by the test operator.",
        },
    )
    assert incident.status_code == 201
    incident_id = incident.json()["id"]
    resolved = client.patch(
        f"/api/v1/incidents/{incident_id}",
        headers=auth("demo-operator"),
        json={"status": "resolved", "assigned_to": "operator-omar"},
    )
    assert resolved.json()["resolved_at"] is not None

    health = client.post(
        "/api/v1/device-health",
        headers=auth("demo-edge"),
        json={
            "site_id": "site-atlas-main",
            "gate_id": "gate-atlas-north",
            "camera_id": "camera-atlas-north-anpr",
            "device_id": "camera-atlas-north-anpr",
            "device_type": "camera",
            "status": "degraded",
            "latency_ms": 380.5,
            "detail": "Synthetic packet-loss drill",
            "reported_at": "2026-08-23T10:20:00+00:00",
        },
    )
    assert health.status_code == 201
    camera = client.get(
        "/api/v1/cameras/camera-atlas-north-anpr",
        headers=auth("demo-viewer"),
    )
    assert camera.json()["status"] == "degraded"

    dashboard = client.get("/api/v1/dashboard", headers=auth("demo-viewer"))
    assert dashboard.status_code == 200
    assert dashboard.json()["counts"]["gates"] == 6
    assert dashboard.json()["counts"]["cameras"] == 6
    assert {gate["id"] for gate in dashboard.json()["gates"]} == {
        "gate-atlas-north",
        "gate-atlas-service",
        "gate-atlas-research",
        "gate-atlas-residence",
        "gate-atlas-south",
        "gate-atlas-sports",
    }
    assert {item["device_id"] for item in dashboard.json()["device_health"]} == {
        "camera-atlas-north-anpr",
        "camera-atlas-service-anpr",
        "camera-atlas-research-overview",
        "camera-atlas-residence-anpr",
        "camera-atlas-south-anpr",
        "camera-atlas-sports-anpr",
    }
    assert dashboard.json()["recent_events"]


@dataclass(frozen=True)
class _Plate:
    text: str
    detection_confidence: float
    recognition_confidence: float
    format_valid: bool


@dataclass(frozen=True)
class _Result:
    plates: tuple[_Plate, ...]
    model_versions: dict[str, str]


def test_existing_anpr_result_shape_maps_to_worker_payload() -> None:
    payloads = recognition_payloads(
        _Result(
            plates=(_Plate("12345-A-6", 0.95, 0.91, True),),
            model_versions={"plate": "p1", "vehicle": "v1", "character": "c1"},
        )
    )
    assert len(payloads) == 1
    assert payloads[0].status == "recognized"
    assert payloads[0].model_version == "character:c1/plate:p1/vehicle:v1"
