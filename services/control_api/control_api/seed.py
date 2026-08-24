"""Deterministic, fictional campus data used by the portfolio demo."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence

from control_api.database import Database

_CREATED = "2026-08-20T08:00:00+00:00"
_DEMO_EVIDENCE = "Synthetic composite - generated for the platform demo"
_CODE_MAX_LENGTH = 30

GateSeed = tuple[str, str, str, str, str, str, float, float, str, int, str]
CameraSeed = tuple[str, str, str, str, str, str, str, str, str, str, str]


def seed_database(database: Database) -> None:
    """Insert a realistic multi-organization scenario without real operational records."""

    with database.connect() as connection:
        connection.executemany(
            "INSERT OR IGNORE INTO organizations "
            "(id, name, slug, timezone, status, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    "org-atlas",
                    "Atlas Technical University",
                    "atlas-technical-university",
                    "Africa/Casablanca",
                    "active",
                    _CREATED,
                ),
                (
                    "org-rif",
                    "Rif Applied Sciences Institute",
                    "rif-applied-sciences",
                    "Africa/Casablanca",
                    "active",
                    _CREATED,
                ),
            ],
        )
        connection.executemany(
            "INSERT OR IGNORE INTO sites "
            "(id, organization_id, code, name, timezone, address, latitude, longitude, "
            "status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    "site-atlas-main",
                    "org-atlas",
                    "MAIN",
                    "Main Academic Campus",
                    "Africa/Casablanca",
                    "Fictional university district, Morocco",
                    31.6267,
                    -7.9891,
                    "active",
                    _CREATED,
                ),
                (
                    "site-atlas-innovation",
                    "org-atlas",
                    "INNOV",
                    "Innovation Annex",
                    "Africa/Casablanca",
                    "Fictional innovation quarter, Morocco",
                    31.6301,
                    -7.9824,
                    "degraded",
                    _CREATED,
                ),
                (
                    "site-rif-main",
                    "org-rif",
                    "MAIN",
                    "Rif Main Campus",
                    "Africa/Casablanca",
                    "Fictional sciences campus, Morocco",
                    35.1687,
                    -5.2636,
                    "active",
                    _CREATED,
                ),
            ],
        )
        _seed_gates(
            connection,
            [
                (
                    "gate-atlas-north",
                    "org-atlas",
                    "site-atlas-main",
                    "NORTH",
                    "North Gate",
                    "inbound",
                    31.6295,
                    -7.9898,
                    "operational",
                    3,
                    _CREATED,
                ),
                (
                    "gate-atlas-service",
                    "org-atlas",
                    "site-atlas-main",
                    "EAST",
                    "East / Logistics Gate",
                    "bidirectional",
                    31.6272,
                    -7.9854,
                    "degraded",
                    1,
                    _CREATED,
                ),
                (
                    "gate-atlas-research",
                    "org-atlas",
                    "site-atlas-main",
                    "NORTH-EAST",
                    "North-East / Innovation Gate",
                    "bidirectional",
                    31.629,
                    -7.9865,
                    "congested",
                    8,
                    _CREATED,
                ),
                (
                    "gate-atlas-residence",
                    "org-atlas",
                    "site-atlas-main",
                    "SOUTH-EAST",
                    "South-East Gate",
                    "bidirectional",
                    31.6249,
                    -7.9859,
                    "operational",
                    2,
                    _CREATED,
                ),
                (
                    "gate-atlas-south",
                    "org-atlas",
                    "site-atlas-main",
                    "SOUTH",
                    "Main / South Gate",
                    "bidirectional",
                    31.6236,
                    -7.9888,
                    "operational",
                    4,
                    _CREATED,
                ),
                (
                    "gate-atlas-sports",
                    "org-atlas",
                    "site-atlas-main",
                    "SPORTS",
                    "Sports / West Gate",
                    "bidirectional",
                    31.6247,
                    -7.9921,
                    "operational",
                    2,
                    _CREATED,
                ),
                (
                    "gate-rif-east",
                    "org-rif",
                    "site-rif-main",
                    "EAST",
                    "East Academic Gate",
                    "bidirectional",
                    35.1692,
                    -5.2629,
                    "operational",
                    2,
                    _CREATED,
                ),
            ],
        )
        _relocate_legacy_service_gate(connection)
        _seed_cameras(
            connection,
            [
                (
                    "camera-atlas-north-anpr",
                    "org-atlas",
                    "site-atlas-main",
                    "gate-atlas-north",
                    "NORTH-ANPR-01",
                    "North ANPR Camera",
                    "anpr",
                    "plate-closeup-h264",
                    "online",
                    "2026-08-23T09:29:56+00:00",
                    _CREATED,
                ),
                (
                    "camera-atlas-service-anpr",
                    "org-atlas",
                    "site-atlas-main",
                    "gate-atlas-service",
                    "EAST-ANPR-01",
                    "East / Logistics ANPR Camera",
                    "anpr",
                    "plate-closeup-h265",
                    "degraded",
                    "2026-08-23T09:25:10+00:00",
                    _CREATED,
                ),
                (
                    "camera-atlas-research-overview",
                    "org-atlas",
                    "site-atlas-main",
                    "gate-atlas-research",
                    "NORTH-EAST-OVERVIEW-01",
                    "North-East / Innovation Overview Camera",
                    "overview",
                    "overview-h264",
                    "online",
                    "2026-08-23T09:29:51+00:00",
                    _CREATED,
                ),
                (
                    "camera-atlas-residence-anpr",
                    "org-atlas",
                    "site-atlas-main",
                    "gate-atlas-residence",
                    "SOUTH-EAST-ANPR-01",
                    "South-East ANPR Camera",
                    "anpr",
                    "plate-closeup-h264",
                    "online",
                    "2026-08-23T09:29:53+00:00",
                    _CREATED,
                ),
                (
                    "camera-atlas-south-anpr",
                    "org-atlas",
                    "site-atlas-main",
                    "gate-atlas-south",
                    "SOUTH-ANPR-01",
                    "Main / South ANPR Camera",
                    "anpr",
                    "plate-closeup-h264",
                    "online",
                    "2026-08-23T09:29:55+00:00",
                    _CREATED,
                ),
                (
                    "camera-atlas-sports-anpr",
                    "org-atlas",
                    "site-atlas-main",
                    "gate-atlas-sports",
                    "SPORTS-ANPR-01",
                    "Sports / West ANPR Camera",
                    "anpr",
                    "plate-closeup-h264",
                    "online",
                    "2026-08-23T09:29:54+00:00",
                    _CREATED,
                ),
                (
                    "camera-rif-east-anpr",
                    "org-rif",
                    "site-rif-main",
                    "gate-rif-east",
                    "EAST-ANPR-01",
                    "East ANPR Camera",
                    "anpr",
                    "plate-closeup-h264",
                    "online",
                    "2026-08-23T09:29:58+00:00",
                    _CREATED,
                ),
            ],
        )
        _relocate_legacy_service_camera(connection)
        connection.executemany(
            "INSERT OR IGNORE INTO access_requests "
            "(id, organization_id, site_id, requested_by, requested_for_name, subject_kind, "
            "purpose, plate_text, valid_from, valid_until, preferred_gate_id, status, "
            "decision_reason, decided_by, decided_at, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    "request-atlas-visitor-pending",
                    "org-atlas",
                    "site-atlas-main",
                    "host-salma",
                    "Nadia Benali",
                    "visitor",
                    "Research collaboration meeting",
                    "24680-B-6",
                    "2026-08-24T08:30:00+00:00",
                    "2026-08-24T16:00:00+00:00",
                    "gate-atlas-north",
                    "pending",
                    None,
                    None,
                    None,
                    "2026-08-22T14:10:00+00:00",
                ),
                (
                    "request-atlas-supplier-approved",
                    "org-atlas",
                    "site-atlas-main",
                    "host-salma",
                    "Atlas Laboratory Supplies",
                    "supplier",
                    "Scheduled equipment delivery",
                    "78123-D-6",
                    "2026-08-23T08:00:00+00:00",
                    "2026-08-23T14:00:00+00:00",
                    "gate-atlas-service",
                    "approved",
                    "Delivery order verified",
                    "admin-amal",
                    "2026-08-22T15:00:00+00:00",
                    "2026-08-22T12:40:00+00:00",
                ),
                (
                    "request-rif-visitor-pending",
                    "org-rif",
                    "site-rif-main",
                    "host-rif",
                    "Omar El Fassi",
                    "visitor",
                    "Admissions appointment",
                    "11357-A-40",
                    "2026-08-25T09:00:00+00:00",
                    "2026-08-25T11:00:00+00:00",
                    "gate-rif-east",
                    "pending",
                    None,
                    None,
                    None,
                    "2026-08-22T16:00:00+00:00",
                ),
            ],
        )
        connection.executemany(
            "INSERT OR IGNORE INTO access_grants "
            "(id, organization_id, site_id, gate_id, source_request_id, subject_name, "
            "subject_kind, plate_text, valid_from, valid_until, status, created_by, "
            "created_at, revoked_at, revocation_reason) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    "grant-atlas-staff",
                    "org-atlas",
                    "site-atlas-main",
                    None,
                    None,
                    "Youssef Amrani",
                    "staff",
                    "12345-A-6",
                    "2026-01-01T00:00:00+00:00",
                    "2026-12-31T23:59:59+00:00",
                    "active",
                    "admin-amal",
                    _CREATED,
                    None,
                    None,
                ),
                (
                    "grant-atlas-supplier",
                    "org-atlas",
                    "site-atlas-main",
                    "gate-atlas-service",
                    "request-atlas-supplier-approved",
                    "Atlas Laboratory Supplies",
                    "supplier",
                    "78123-D-6",
                    "2026-08-23T08:00:00+00:00",
                    "2026-08-23T14:00:00+00:00",
                    "active",
                    "admin-amal",
                    "2026-08-22T15:00:00+00:00",
                    None,
                    None,
                ),
                (
                    "grant-atlas-revoked",
                    "org-atlas",
                    "site-atlas-main",
                    "gate-atlas-research",
                    None,
                    "Former contractor",
                    "contractor",
                    "90421-C-6",
                    "2026-01-01T00:00:00+00:00",
                    "2026-12-31T23:59:59+00:00",
                    "revoked",
                    "admin-amal",
                    _CREATED,
                    "2026-08-21T10:00:00+00:00",
                    "Contract completed",
                ),
            ],
        )
        connection.executemany(
            "INSERT OR IGNORE INTO passages "
            "(id, organization_id, site_id, gate_id, camera_id, direction, status, "
            "occurred_at, completed_at, evidence_label, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    "passage-atlas-allowed",
                    "org-atlas",
                    "site-atlas-main",
                    "gate-atlas-north",
                    "camera-atlas-north-anpr",
                    "inbound",
                    "completed",
                    "2026-08-23T09:26:12+00:00",
                    "2026-08-23T09:26:14+00:00",
                    _DEMO_EVIDENCE,
                    "2026-08-23T09:26:12+00:00",
                ),
                (
                    "passage-atlas-review",
                    "org-atlas",
                    "site-atlas-main",
                    "gate-atlas-research",
                    "camera-atlas-research-overview",
                    "inbound",
                    "review_required",
                    "2026-08-23T09:27:48+00:00",
                    None,
                    _DEMO_EVIDENCE,
                    "2026-08-23T09:27:48+00:00",
                ),
                (
                    "passage-atlas-denied",
                    "org-atlas",
                    "site-atlas-main",
                    "gate-atlas-research",
                    "camera-atlas-research-overview",
                    "inbound",
                    "completed",
                    "2026-08-23T09:28:31+00:00",
                    "2026-08-23T09:28:33+00:00",
                    _DEMO_EVIDENCE,
                    "2026-08-23T09:28:31+00:00",
                ),
            ],
        )
        connection.executemany(
            "INSERT OR IGNORE INTO recognition_observations "
            "(id, organization_id, passage_id, status, plate_text, detection_confidence, "
            "recognition_confidence, format_valid, model_version, source, evidence_label, "
            "occurred_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    "recognition-atlas-allowed",
                    "org-atlas",
                    "passage-atlas-allowed",
                    "recognized",
                    "12345-A-6",
                    0.97,
                    0.94,
                    1,
                    "vehicle-v1/plate-v1/characters-v1",
                    "central_worker",
                    _DEMO_EVIDENCE,
                    "2026-08-23T09:26:13+00:00",
                ),
                (
                    "recognition-atlas-review",
                    "org-atlas",
                    "passage-atlas-review",
                    "uncertain",
                    "2468?-B-6",
                    0.71,
                    0.58,
                    0,
                    "vehicle-v1/plate-v1/characters-v1",
                    "central_worker",
                    _DEMO_EVIDENCE,
                    "2026-08-23T09:27:49+00:00",
                ),
                (
                    "recognition-atlas-denied",
                    "org-atlas",
                    "passage-atlas-denied",
                    "recognized",
                    "90421-C-6",
                    0.96,
                    0.92,
                    1,
                    "vehicle-v1/plate-v1/characters-v1",
                    "central_worker",
                    _DEMO_EVIDENCE,
                    "2026-08-23T09:28:32+00:00",
                ),
            ],
        )
        connection.executemany(
            "INSERT OR IGNORE INTO authorization_decisions "
            "(id, organization_id, passage_id, outcome, reason, source, grant_id, "
            "decided_by, occurred_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    "authorization-atlas-allowed",
                    "org-atlas",
                    "passage-atlas-allowed",
                    "allowed",
                    "Active staff grant matched the recognized plate",
                    "policy",
                    "grant-atlas-staff",
                    "policy-engine-demo",
                    "2026-08-23T09:26:14+00:00",
                ),
                (
                    "authorization-atlas-review",
                    "org-atlas",
                    "passage-atlas-review",
                    "review_required",
                    "Recognition confidence is below the assisted-decision threshold",
                    "policy",
                    None,
                    "policy-engine-demo",
                    "2026-08-23T09:27:50+00:00",
                ),
                (
                    "authorization-atlas-denied",
                    "org-atlas",
                    "passage-atlas-denied",
                    "denied",
                    "Matched grant is revoked",
                    "policy",
                    "grant-atlas-revoked",
                    "policy-engine-demo",
                    "2026-08-23T09:28:33+00:00",
                ),
            ],
        )
        _seed_events(connection)
        connection.executemany(
            "INSERT OR IGNORE INTO incidents "
            "(id, organization_id, site_id, gate_id, passage_id, title, severity, status, "
            "description, assigned_to, created_by, created_at, resolved_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    "incident-atlas-review",
                    "org-atlas",
                    "site-atlas-main",
                    "gate-atlas-research",
                    "passage-atlas-review",
                    "Recognition requires operator review",
                    "warning",
                    "investigating",
                    "Synthetic composite produced an ambiguous character candidate.",
                    "operator-omar",
                    "policy-engine-demo",
                    "2026-08-23T09:27:51+00:00",
                    None,
                ),
                (
                    "incident-atlas-camera",
                    "org-atlas",
                    "site-atlas-main",
                    "gate-atlas-service",
                    None,
                    "East logistics camera packet loss",
                    "warning",
                    "open",
                    "Demo health signal reports intermittent stream latency.",
                    None,
                    "edge-agent-demo",
                    "2026-08-23T09:25:20+00:00",
                    None,
                ),
            ],
        )
        connection.executemany(
            "INSERT OR IGNORE INTO device_health "
            "(id, organization_id, site_id, gate_id, camera_id, device_id, device_type, "
            "status, latency_ms, detail, reported_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    "health-atlas-north",
                    "org-atlas",
                    "site-atlas-main",
                    "gate-atlas-north",
                    "camera-atlas-north-anpr",
                    "camera-atlas-north-anpr",
                    "camera",
                    "online",
                    42.0,
                    "Stream and ONVIF heartbeat healthy",
                    "2026-08-23T09:29:56+00:00",
                ),
                (
                    "health-atlas-research",
                    "org-atlas",
                    "site-atlas-main",
                    "gate-atlas-research",
                    "camera-atlas-research-overview",
                    "camera-atlas-research-overview",
                    "camera",
                    "online",
                    55.0,
                    "Overview stream healthy",
                    "2026-08-23T09:29:51+00:00",
                ),
                (
                    "health-atlas-service",
                    "org-atlas",
                    "site-atlas-main",
                    "gate-atlas-service",
                    "camera-atlas-service-anpr",
                    "camera-atlas-service-anpr",
                    "camera",
                    "degraded",
                    410.0,
                    "Intermittent packet loss in synthetic East logistics telemetry",
                    "2026-08-23T09:25:10+00:00",
                ),
                (
                    "health-atlas-residence",
                    "org-atlas",
                    "site-atlas-main",
                    "gate-atlas-residence",
                    "camera-atlas-residence-anpr",
                    "camera-atlas-residence-anpr",
                    "camera",
                    "online",
                    47.0,
                    "South-East stream and ONVIF heartbeat healthy",
                    "2026-08-23T09:29:53+00:00",
                ),
                (
                    "health-atlas-south",
                    "org-atlas",
                    "site-atlas-main",
                    "gate-atlas-south",
                    "camera-atlas-south-anpr",
                    "camera-atlas-south-anpr",
                    "camera",
                    "online",
                    44.0,
                    "Main / South stream and ONVIF heartbeat healthy",
                    "2026-08-23T09:29:55+00:00",
                ),
                (
                    "health-atlas-sports",
                    "org-atlas",
                    "site-atlas-main",
                    "gate-atlas-sports",
                    "camera-atlas-sports-anpr",
                    "camera-atlas-sports-anpr",
                    "camera",
                    "online",
                    46.0,
                    "Sports / West stream and ONVIF heartbeat healthy",
                    "2026-08-23T09:29:54+00:00",
                ),
                (
                    "health-rif-east",
                    "org-rif",
                    "site-rif-main",
                    "gate-rif-east",
                    "camera-rif-east-anpr",
                    "camera-rif-east-anpr",
                    "camera",
                    "online",
                    38.0,
                    "Stream healthy",
                    "2026-08-23T09:29:58+00:00",
                ),
            ],
        )
        _relocate_legacy_service_dependents(connection)
        connection.commit()


def _seed_gates(connection: sqlite3.Connection, fixtures: Sequence[GateSeed]) -> None:
    """Insert stable gate IDs while preserving existing rows and resolving code collisions."""

    statement = (
        "INSERT INTO gates "
        "(id, organization_id, site_id, code, name, direction, latitude, longitude, status, "
        "queue_estimate, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )
    for fixture in fixtures:
        (
            gate_id,
            organization_id,
            site_id,
            preferred_code,
            name,
            direction,
            latitude,
            longitude,
            status,
            queue_estimate,
            created_at,
        ) = fixture
        existing = connection.execute(
            "SELECT organization_id FROM gates WHERE id = ?",
            (gate_id,),
        ).fetchone()
        if existing is not None:
            _require_seed_owner("gate", gate_id, organization_id, str(existing["organization_id"]))
            continue
        code = _available_gate_code(
            connection,
            organization_id=organization_id,
            site_id=site_id,
            preferred=preferred_code,
            gate_id=gate_id,
        )
        connection.execute(
            statement,
            (
                gate_id,
                organization_id,
                site_id,
                code,
                name,
                direction,
                latitude,
                longitude,
                status,
                queue_estimate,
                created_at,
            ),
        )


def _seed_cameras(connection: sqlite3.Connection, fixtures: Sequence[CameraSeed]) -> None:
    """Insert stable camera IDs while preserving existing rows and resolving code collisions."""

    statement = (
        "INSERT INTO cameras "
        "(id, organization_id, site_id, gate_id, code, name, role, stream_profile, status, "
        "last_seen_at, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )
    for fixture in fixtures:
        (
            camera_id,
            organization_id,
            site_id,
            gate_id,
            preferred_code,
            name,
            role,
            stream_profile,
            status,
            last_seen_at,
            created_at,
        ) = fixture
        existing = connection.execute(
            "SELECT organization_id FROM cameras WHERE id = ?",
            (camera_id,),
        ).fetchone()
        if existing is not None:
            _require_seed_owner(
                "camera",
                camera_id,
                organization_id,
                str(existing["organization_id"]),
            )
            continue
        code = _available_camera_code(
            connection,
            organization_id=organization_id,
            preferred=preferred_code,
            camera_id=camera_id,
        )
        connection.execute(
            statement,
            (
                camera_id,
                organization_id,
                site_id,
                gate_id,
                code,
                name,
                role,
                stream_profile,
                status,
                last_seen_at,
                created_at,
            ),
        )


def _require_seed_owner(kind: str, seed_id: str, expected: str, actual: str) -> None:
    """Reject an impossible cross-tenant stable-ID collision with an actionable error."""

    if actual != expected:
        raise RuntimeError(
            f"Seed {kind} ID {seed_id!r} belongs to {actual!r}; expected {expected!r}"
        )


def _collision_code(preferred: str, attempt: int) -> str:
    """Return a deterministic, contract-sized alternative for a colliding topology code."""

    suffix = "-DEMO" if attempt == 1 else f"-D{attempt}"
    return f"{preferred[: _CODE_MAX_LENGTH - len(suffix)]}{suffix}"


def _available_gate_code(
    connection: sqlite3.Connection,
    *,
    organization_id: str,
    site_id: str,
    preferred: str,
    gate_id: str,
) -> str:
    attempt = 0
    while True:
        candidate = preferred if attempt == 0 else _collision_code(preferred, attempt)
        conflict = connection.execute(
            "SELECT id FROM gates WHERE organization_id = ? AND site_id = ? AND code = ? "
            "AND id != ?",
            (organization_id, site_id, candidate, gate_id),
        ).fetchone()
        if conflict is None:
            return candidate
        attempt += 1


def _available_camera_code(
    connection: sqlite3.Connection,
    *,
    organization_id: str,
    preferred: str,
    camera_id: str,
) -> str:
    attempt = 0
    while True:
        candidate = preferred if attempt == 0 else _collision_code(preferred, attempt)
        conflict = connection.execute(
            "SELECT id FROM cameras WHERE organization_id = ? AND code = ? AND id != ?",
            (organization_id, candidate, camera_id),
        ).fetchone()
        if conflict is None:
            return candidate
        attempt += 1


def _relocate_legacy_service_gate(connection: sqlite3.Connection) -> None:
    """Move the stable legacy service gate without changing API-editable fields."""

    gate_id = "gate-atlas-service"
    organization_id = "org-atlas"
    legacy_site_id = "site-atlas-innovation"
    target_site_id = "site-atlas-main"
    existing = connection.execute(
        "SELECT site_id FROM gates WHERE id = ? AND organization_id = ?",
        (gate_id, organization_id),
    ).fetchone()
    if existing is None or str(existing["site_id"]) != legacy_site_id:
        return
    code = _available_gate_code(
        connection,
        organization_id=organization_id,
        site_id=target_site_id,
        preferred="EAST",
        gate_id=gate_id,
    )
    connection.execute(
        "UPDATE gates SET site_id = ?, code = ? "
        "WHERE id = ? AND organization_id = ? AND site_id = ?",
        (target_site_id, code, gate_id, organization_id, legacy_site_id),
    )


def _relocate_legacy_service_camera(connection: sqlite3.Connection) -> None:
    """Move the stable legacy service camera without changing API-editable fields."""

    camera_id = "camera-atlas-service-anpr"
    organization_id = "org-atlas"
    legacy_site_id = "site-atlas-innovation"
    target_site_id = "site-atlas-main"
    existing = connection.execute(
        "SELECT site_id FROM cameras WHERE id = ? AND organization_id = ?",
        (camera_id, organization_id),
    ).fetchone()
    if existing is None or str(existing["site_id"]) != legacy_site_id:
        return
    code = _available_camera_code(
        connection,
        organization_id=organization_id,
        preferred="EAST-ANPR-01",
        camera_id=camera_id,
    )
    connection.execute(
        "UPDATE cameras SET site_id = ?, code = ? "
        "WHERE id = ? AND organization_id = ? AND site_id = ?",
        (target_site_id, code, camera_id, organization_id, legacy_site_id),
    )


def _relocate_legacy_service_dependents(connection: sqlite3.Connection) -> None:
    """Keep stable service fixtures aligned after moving their gate to the main site."""

    for table_name, row_id in (
        ("access_requests", "request-atlas-supplier-approved"),
        ("access_grants", "grant-atlas-supplier"),
        ("incidents", "incident-atlas-camera"),
        ("device_health", "health-atlas-service"),
    ):
        connection.execute(
            f"UPDATE {table_name} SET site_id = ? "  # noqa: S608 - closed table-name allowlist
            "WHERE id = ? AND organization_id = ? AND site_id = ?",
            ("site-atlas-main", row_id, "org-atlas", "site-atlas-innovation"),
        )


def _seed_events(connection: sqlite3.Connection) -> None:
    """Insert event fixtures while keeping the main seed function readable."""

    execute_many = connection.executemany
    execute_many(
        "INSERT OR IGNORE INTO events "
        "(id, organization_id, site_id, gate_id, passage_id, source, event_type, severity, "
        "summary, evidence_label, metadata_json, occurred_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                "event-atlas-allowed",
                "org-atlas",
                "site-atlas-main",
                "gate-atlas-north",
                "passage-atlas-allowed",
                "authorization",
                "authorization.allowed",
                "info",
                "Active staff grant matched",
                _DEMO_EVIDENCE,
                json.dumps({"outcome": "allowed", "grant_id": "grant-atlas-staff"}),
                "2026-08-23T09:26:14+00:00",
            ),
            (
                "event-atlas-review",
                "org-atlas",
                "site-atlas-main",
                "gate-atlas-research",
                "passage-atlas-review",
                "recognition",
                "recognition.review_required",
                "warning",
                "Low-confidence recognition requires review",
                _DEMO_EVIDENCE,
                json.dumps({"confidence": 0.58, "format_valid": False}),
                "2026-08-23T09:27:50+00:00",
            ),
            (
                "event-atlas-denied",
                "org-atlas",
                "site-atlas-main",
                "gate-atlas-research",
                "passage-atlas-denied",
                "authorization",
                "authorization.denied",
                "critical",
                "Revoked grant matched",
                _DEMO_EVIDENCE,
                json.dumps({"outcome": "denied", "reason": "revoked_grant"}),
                "2026-08-23T09:28:33+00:00",
            ),
            (
                "event-rif-camera-online",
                "org-rif",
                "site-rif-main",
                "gate-rif-east",
                None,
                "device",
                "device.online",
                "info",
                "East ANPR camera is online",
                None,
                json.dumps({"device_id": "camera-rif-east-anpr"}),
                "2026-08-23T09:29:58+00:00",
            ),
        ],
    )
