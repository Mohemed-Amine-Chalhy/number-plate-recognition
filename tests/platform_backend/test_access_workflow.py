"""Access requests, approvals, grants, and role-boundary tests."""

from __future__ import annotations

from conftest import auth
from fastapi.testclient import TestClient


def _request_payload() -> dict[str, object]:
    return {
        "site_id": "site-atlas-main",
        "requested_for_name": "Leila Mansouri",
        "subject_kind": "visitor",
        "purpose": "Guest lecture",
        "plate_text": " 55771-a-6 ",
        "valid_from": "2026-08-26T09:00:00+00:00",
        "valid_until": "2026-08-26T13:00:00+00:00",
        "preferred_gate_id": "gate-atlas-north",
    }


def test_host_submits_and_admin_approves_request(client: TestClient) -> None:
    created = client.post(
        "/api/v1/access-requests",
        headers=auth("demo-host"),
        json=_request_payload(),
    )
    assert created.status_code == 201
    request_id = created.json()["id"]
    assert created.json()["plate_text"] == "55771-A-6"
    assert created.json()["requested_by"] == "host-salma"

    assert (
        client.post(
            f"/api/v1/access-requests/{request_id}/decision",
            headers=auth("demo-host"),
            json={"decision": "approved", "reason": "Self approval"},
        ).status_code
        == 403
    )

    decision = client.post(
        f"/api/v1/access-requests/{request_id}/decision",
        headers=auth("demo-admin"),
        json={
            "decision": "approved",
            "reason": "Host and visit window verified",
            "gate_id": "gate-atlas-north",
        },
    )
    assert decision.status_code == 200
    body = decision.json()
    assert body["request"]["status"] == "approved"
    assert body["grant"]["status"] == "active"
    assert body["grant"]["source_request_id"] == request_id

    repeated = client.post(
        f"/api/v1/access-requests/{request_id}/decision",
        headers=auth("demo-admin"),
        json={"decision": "rejected", "reason": "Duplicate decision"},
    )
    assert repeated.status_code == 422


def test_host_only_lists_and_changes_own_pending_requests(client: TestClient) -> None:
    host_requests = client.get("/api/v1/access-requests", headers=auth("demo-host")).json()
    assert host_requests
    assert {item["requested_by"] for item in host_requests} == {"host-salma"}

    created = client.post(
        "/api/v1/access-requests",
        headers=auth("demo-host"),
        json=_request_payload(),
    ).json()
    request_id = created["id"]
    updated = client.patch(
        f"/api/v1/access-requests/{request_id}",
        headers=auth("demo-host"),
        json={"purpose": "Updated guest lecture and lab tour"},
    )
    assert updated.status_code == 200
    assert updated.json()["purpose"].startswith("Updated")
    assert (
        client.delete(
            f"/api/v1/access-requests/{request_id}", headers=auth("demo-host")
        ).status_code
        == 204
    )


def test_grant_can_be_created_and_revoked(client: TestClient) -> None:
    created = client.post(
        "/api/v1/access-grants",
        headers=auth("demo-admin"),
        json={
            "site_id": "site-atlas-main",
            "gate_id": "gate-atlas-north",
            "subject_name": "Temporary Researcher",
            "subject_kind": "researcher",
            "plate_text": "63001-B-6",
            "valid_from": "2026-08-24T08:00:00+00:00",
            "valid_until": "2026-08-30T18:00:00+00:00",
        },
    )
    assert created.status_code == 201
    grant_id = created.json()["id"]
    revoked = client.post(
        f"/api/v1/access-grants/{grant_id}/revoke",
        headers=auth("demo-admin"),
        json={"reason": "Visit completed early"},
    )
    assert revoked.json()["status"] == "revoked"
