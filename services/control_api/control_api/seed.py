"""Deterministic, fictional campus data used by the portfolio demo."""

from __future__ import annotations

import json
import sqlite3

from control_api.database import Database

_CREATED = "2026-08-20T08:00:00+00:00"
_DEMO_EVIDENCE = "Synthetic composite - generated for the platform demo"


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
        connection.executemany(
            "INSERT OR IGNORE INTO gates "
            "(id, organization_id, site_id, code, name, direction, latitude, longitude, "
            "status, queue_estimate, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    "gate-atlas-north",
                    "org-atlas",
                    "site-atlas-main",
                    "NORTH",
                    "North Arrival Gate",
                    "inbound",
                    31.6281,
                    -7.9899,
                    "operational",
                    3,
                    _CREATED,
                ),
                (
                    "gate-atlas-research",
                    "org-atlas",
                    "site-atlas-main",
                    "RESEARCH",
                    "Research Gate",
                    "bidirectional",
                    31.6259,
                    -7.9862,
                    "congested",
                    8,
                    _CREATED,
                ),
                (
                    "gate-atlas-service",
                    "org-atlas",
                    "site-atlas-innovation",
                    "SERVICE",
                    "Service Entrance",
                    "inbound",
                    31.631,
                    -7.9818,
                    "degraded",
                    1,
                    _CREATED,
                ),
                (
                    "gate-atlas-residence",
                    "org-atlas",
                    "site-atlas-main",
                    "RESIDENCE",
                    "Residence West Gate",
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
        connection.executemany(
            "INSERT OR IGNORE INTO cameras "
            "(id, organization_id, site_id, gate_id, code, name, role, stream_profile, "
            "status, last_seen_at, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                    "camera-atlas-research-overview",
                    "org-atlas",
                    "site-atlas-main",
                    "gate-atlas-research",
                    "RESEARCH-OVERVIEW-01",
                    "Research Overview Camera",
                    "overview",
                    "overview-h264",
                    "online",
                    "2026-08-23T09:29:51+00:00",
                    _CREATED,
                ),
                (
                    "camera-atlas-service-anpr",
                    "org-atlas",
                    "site-atlas-innovation",
                    "gate-atlas-service",
                    "SERVICE-ANPR-01",
                    "Service ANPR Camera",
                    "anpr",
                    "plate-closeup-h265",
                    "degraded",
                    "2026-08-23T09:25:10+00:00",
                    _CREATED,
                ),
                (
                    "camera-atlas-residence-anpr",
                    "org-atlas",
                    "site-atlas-main",
                    "gate-atlas-residence",
                    "RESIDENCE-ANPR-01",
                    "Residence West ANPR Camera",
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
                    "site-atlas-innovation",
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
                    "site-atlas-innovation",
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
                    "site-atlas-innovation",
                    "gate-atlas-service",
                    None,
                    "Service camera packet loss",
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
                    "site-atlas-innovation",
                    "gate-atlas-service",
                    "camera-atlas-service-anpr",
                    "camera-atlas-service-anpr",
                    "camera",
                    "degraded",
                    410.0,
                    "Intermittent packet loss in synthetic demo telemetry",
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
                    "Residence stream and ONVIF heartbeat healthy",
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
        connection.commit()


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
