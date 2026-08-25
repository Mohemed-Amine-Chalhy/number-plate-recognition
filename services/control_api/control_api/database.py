"""Small SQLite persistence boundary for the self-contained platform demo."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SCHEMA_VERSION = 2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_metadata (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS organizations (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    slug TEXT NOT NULL UNIQUE,
    timezone TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('active', 'suspended', 'archived')),
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sites (
    id TEXT PRIMARY KEY,
    organization_id TEXT NOT NULL REFERENCES organizations(id),
    code TEXT NOT NULL,
    name TEXT NOT NULL,
    timezone TEXT NOT NULL,
    address TEXT NOT NULL,
    latitude REAL,
    longitude REAL,
    status TEXT NOT NULL CHECK (status IN ('active', 'degraded', 'offline', 'archived')),
    created_at TEXT NOT NULL,
    UNIQUE (organization_id, code)
);

CREATE TABLE IF NOT EXISTS gates (
    id TEXT PRIMARY KEY,
    organization_id TEXT NOT NULL REFERENCES organizations(id),
    site_id TEXT NOT NULL REFERENCES sites(id),
    code TEXT NOT NULL,
    name TEXT NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('inbound', 'outbound', 'bidirectional')),
    latitude REAL,
    longitude REAL,
    status TEXT NOT NULL CHECK (
        status IN ('operational', 'congested', 'degraded', 'offline', 'disabled')
    ),
    queue_estimate INTEGER NOT NULL DEFAULT 0 CHECK (queue_estimate >= 0),
    created_at TEXT NOT NULL,
    UNIQUE (organization_id, site_id, code)
);

CREATE TABLE IF NOT EXISTS cameras (
    id TEXT PRIMARY KEY,
    organization_id TEXT NOT NULL REFERENCES organizations(id),
    site_id TEXT NOT NULL REFERENCES sites(id),
    gate_id TEXT NOT NULL REFERENCES gates(id),
    code TEXT NOT NULL,
    name TEXT NOT NULL,
    role TEXT NOT NULL,
    stream_profile TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('online', 'degraded', 'offline', 'disabled')),
    last_seen_at TEXT,
    created_at TEXT NOT NULL,
    UNIQUE (organization_id, code)
);

CREATE TABLE IF NOT EXISTS access_requests (
    id TEXT PRIMARY KEY,
    organization_id TEXT NOT NULL REFERENCES organizations(id),
    site_id TEXT NOT NULL REFERENCES sites(id),
    requested_by TEXT NOT NULL,
    requested_for_name TEXT NOT NULL,
    subject_kind TEXT NOT NULL,
    purpose TEXT NOT NULL,
    plate_text TEXT,
    valid_from TEXT NOT NULL,
    valid_until TEXT NOT NULL,
    preferred_gate_id TEXT REFERENCES gates(id),
    status TEXT NOT NULL CHECK (status IN ('pending', 'approved', 'rejected', 'cancelled')),
    decision_reason TEXT,
    decided_by TEXT,
    decided_at TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS access_grants (
    id TEXT PRIMARY KEY,
    organization_id TEXT NOT NULL REFERENCES organizations(id),
    site_id TEXT NOT NULL REFERENCES sites(id),
    gate_id TEXT REFERENCES gates(id),
    source_request_id TEXT REFERENCES access_requests(id),
    subject_name TEXT NOT NULL,
    subject_kind TEXT NOT NULL,
    plate_text TEXT,
    valid_from TEXT NOT NULL,
    valid_until TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('active', 'revoked', 'expired')),
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    revoked_at TEXT,
    revocation_reason TEXT
);

CREATE TABLE IF NOT EXISTS passages (
    id TEXT PRIMARY KEY,
    organization_id TEXT NOT NULL REFERENCES organizations(id),
    site_id TEXT NOT NULL REFERENCES sites(id),
    gate_id TEXT NOT NULL REFERENCES gates(id),
    camera_id TEXT REFERENCES cameras(id),
    direction TEXT NOT NULL CHECK (direction IN ('inbound', 'outbound', 'bidirectional')),
    status TEXT NOT NULL CHECK (status IN ('open', 'completed', 'review_required')),
    occurred_at TEXT NOT NULL,
    completed_at TEXT,
    evidence_label TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS recognition_observations (
    id TEXT PRIMARY KEY,
    organization_id TEXT NOT NULL REFERENCES organizations(id),
    passage_id TEXT NOT NULL REFERENCES passages(id),
    status TEXT NOT NULL CHECK (status IN ('recognized', 'uncertain', 'unreadable')),
    plate_text TEXT,
    detection_confidence REAL,
    recognition_confidence REAL,
    format_valid INTEGER NOT NULL CHECK (format_valid IN (0, 1)),
    model_version TEXT NOT NULL,
    source TEXT NOT NULL,
    evidence_label TEXT NOT NULL,
    occurred_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS authorization_decisions (
    id TEXT PRIMARY KEY,
    organization_id TEXT NOT NULL REFERENCES organizations(id),
    passage_id TEXT NOT NULL REFERENCES passages(id),
    outcome TEXT NOT NULL CHECK (
        outcome IN ('allowed', 'review_required', 'denied', 'no_match')
    ),
    reason TEXT NOT NULL,
    source TEXT NOT NULL CHECK (source IN ('policy', 'operator', 'system')),
    grant_id TEXT REFERENCES access_grants(id),
    decided_by TEXT NOT NULL,
    occurred_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    organization_id TEXT NOT NULL REFERENCES organizations(id),
    site_id TEXT REFERENCES sites(id),
    gate_id TEXT REFERENCES gates(id),
    passage_id TEXT REFERENCES passages(id),
    source TEXT NOT NULL,
    event_type TEXT NOT NULL,
    severity TEXT NOT NULL CHECK (severity IN ('info', 'warning', 'critical')),
    summary TEXT NOT NULL,
    evidence_label TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    occurred_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS incidents (
    id TEXT PRIMARY KEY,
    organization_id TEXT NOT NULL REFERENCES organizations(id),
    site_id TEXT NOT NULL REFERENCES sites(id),
    gate_id TEXT REFERENCES gates(id),
    passage_id TEXT REFERENCES passages(id),
    title TEXT NOT NULL,
    severity TEXT NOT NULL CHECK (severity IN ('info', 'warning', 'critical')),
    status TEXT NOT NULL CHECK (status IN ('open', 'investigating', 'resolved')),
    description TEXT NOT NULL,
    assigned_to TEXT,
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    resolved_at TEXT
);

CREATE TABLE IF NOT EXISTS device_health (
    id TEXT PRIMARY KEY,
    organization_id TEXT NOT NULL REFERENCES organizations(id),
    site_id TEXT NOT NULL REFERENCES sites(id),
    gate_id TEXT REFERENCES gates(id),
    camera_id TEXT REFERENCES cameras(id),
    device_id TEXT NOT NULL,
    device_type TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('online', 'degraded', 'offline', 'unknown')),
    latency_ms REAL,
    detail TEXT NOT NULL,
    reported_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_runs (
    id TEXT PRIMARY KEY,
    organization_id TEXT NOT NULL REFERENCES organizations(id),
    site_id TEXT NOT NULL REFERENCES sites(id),
    gate_id TEXT NOT NULL REFERENCES gates(id),
    objective TEXT NOT NULL,
    intent TEXT NOT NULL CHECK (intent IN ('gate_health_triage')),
    status TEXT NOT NULL CHECK (
        status IN ('running', 'awaiting_approval', 'completed', 'rejected', 'failed')
    ),
    created_by TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    trace_id TEXT NOT NULL,
    correlation_id TEXT NOT NULL,
    planner_name TEXT NOT NULL,
    planner_version TEXT NOT NULL,
    policy_name TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    plan_summary TEXT NOT NULL,
    failure_code TEXT,
    failure_detail TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (organization_id, created_by, idempotency_key)
);

CREATE TABLE IF NOT EXISTS agent_steps (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES agent_runs(id),
    organization_id TEXT NOT NULL REFERENCES organizations(id),
    sequence INTEGER NOT NULL CHECK (sequence > 0),
    tool_name TEXT NOT NULL CHECK (
        tool_name IN (
            'get_gate',
            'get_latest_device_health',
            'list_open_gate_incidents',
            'start_incident_investigation',
            'create_incident'
        )
    ),
    risk TEXT NOT NULL CHECK (risk IN ('read_only', 'consequential')),
    status TEXT NOT NULL CHECK (
        status IN ('pending', 'running', 'awaiting_approval', 'succeeded', 'skipped', 'failed')
    ),
    rationale TEXT NOT NULL,
    input_json TEXT NOT NULL DEFAULT '{}',
    output_json TEXT,
    policy_checks_json TEXT NOT NULL DEFAULT '[]',
    started_at TEXT,
    completed_at TEXT,
    error_code TEXT,
    error_detail TEXT,
    UNIQUE (run_id, sequence)
);

CREATE TABLE IF NOT EXISTS agent_approvals (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL UNIQUE REFERENCES agent_runs(id),
    organization_id TEXT NOT NULL REFERENCES organizations(id),
    step_id TEXT NOT NULL REFERENCES agent_steps(id),
    decision TEXT NOT NULL CHECK (decision IN ('approved', 'rejected')),
    reason TEXT NOT NULL,
    decided_by TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    decided_at TEXT NOT NULL,
    UNIQUE (run_id, idempotency_key)
);

CREATE TABLE IF NOT EXISTS agent_audit_events (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    run_id TEXT NOT NULL REFERENCES agent_runs(id),
    organization_id TEXT NOT NULL REFERENCES organizations(id),
    step_id TEXT REFERENCES agent_steps(id),
    event_type TEXT NOT NULL,
    actor_type TEXT NOT NULL,
    actor_id TEXT NOT NULL,
    summary TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    occurred_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sites_org ON sites(organization_id, status);
CREATE INDEX IF NOT EXISTS idx_gates_org_site ON gates(organization_id, site_id, status);
CREATE INDEX IF NOT EXISTS idx_cameras_org_gate ON cameras(organization_id, gate_id, status);
CREATE INDEX IF NOT EXISTS idx_requests_org_status
    ON access_requests(organization_id, status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_grants_org_plate
    ON access_grants(organization_id, plate_text, status, valid_until);
CREATE INDEX IF NOT EXISTS idx_passages_org_time
    ON passages(organization_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_recognition_passage
    ON recognition_observations(organization_id, passage_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_authorization_passage
    ON authorization_decisions(organization_id, passage_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_events_org_sequence ON events(organization_id, sequence);
CREATE INDEX IF NOT EXISTS idx_incidents_org_status
    ON incidents(organization_id, status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_health_org_device
    ON device_health(organization_id, device_id, reported_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_runs_org_created
    ON agent_runs(organization_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_runs_org_gate
    ON agent_runs(organization_id, gate_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_steps_run ON agent_steps(run_id, sequence);
CREATE INDEX IF NOT EXISTS idx_agent_audit_run ON agent_audit_events(run_id, sequence);
"""


class Database:
    """Open short-lived SQLite connections suitable for FastAPI thread workers."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Yield one configured connection and close it deterministically."""

        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Commit a successful unit of work or roll it back on any failure."""

        with self.connect() as connection:
            try:
                yield connection
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    @contextmanager
    def immediate_transaction(self) -> Iterator[sqlite3.Connection]:
        """Serialize a read-check-write state transition before inspecting its current state."""

        with self.connect() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                yield connection
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def initialize(self, *, seed: bool) -> None:
        """Create the schema and optionally install deterministic demo fixtures."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(_SCHEMA)
            connection.execute(
                "INSERT OR IGNORE INTO schema_metadata(version, applied_at) "
                "VALUES (?, CURRENT_TIMESTAMP)",
                (SCHEMA_VERSION,),
            )
            connection.commit()
        if seed:
            from control_api.seed import seed_database

            seed_database(self)

    def is_ready(self) -> bool:
        """Return whether the expected schema can answer a trivial query."""

        try:
            with self.connect() as connection:
                row = connection.execute(
                    "SELECT version FROM schema_metadata WHERE version = ?", (SCHEMA_VERSION,)
                ).fetchone()
            return row is not None
        except sqlite3.Error:
            return False
