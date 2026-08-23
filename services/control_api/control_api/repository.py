"""Organization-scoped persistence and transactional control-plane use cases."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from typing import Any, cast
from uuid import uuid4

from pydantic import BaseModel

from control_api.database import Database
from control_api.errors import ConflictError, InvalidStateError, ResourceNotFoundError
from control_api.schemas import (
    AccessGrantCreate,
    AccessGrantRead,
    AccessGrantStatus,
    AccessRequestCreate,
    AccessRequestDecision,
    AccessRequestRead,
    AccessRequestStatus,
    AccessRequestUpdate,
    AuthorizationDecisionCreate,
    AuthorizationDecisionRead,
    AuthorizationOutcome,
    CameraCreate,
    CameraRead,
    CameraStatus,
    CameraUpdate,
    DecisionSource,
    DeviceHealthCreate,
    DeviceHealthRead,
    EventPage,
    EventRead,
    EventSeverity,
    GateCreate,
    GateRead,
    GateStatus,
    GateUpdate,
    IncidentCreate,
    IncidentRead,
    IncidentStatus,
    IncidentUpdate,
    OrganizationCreate,
    OrganizationRead,
    OrganizationStatus,
    OrganizationUpdate,
    PassageCreate,
    PassageDetail,
    PassageRead,
    PassageStatus,
    RecognitionCreate,
    RecognitionRead,
    RecognitionStatus,
    SiteCreate,
    SiteRead,
    SiteStatus,
    SiteUpdate,
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex}"


def _canonical_plate(value: str | None) -> str:
    return "".join(character for character in (value or "").upper() if character.isalnum())


def _row_data(row: sqlite3.Row) -> dict[str, Any]:
    data: dict[str, Any] = dict(row)
    if "format_valid" in data:
        data["format_valid"] = bool(data["format_valid"])
    if "metadata_json" in data:
        raw_metadata = data.pop("metadata_json")
        data["metadata"] = json.loads(raw_metadata)
    return data


def _model[ModelT: BaseModel](model_type: type[ModelT], row: sqlite3.Row) -> ModelT:
    return model_type.model_validate(_row_data(row))


def _payload(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="json", exclude_unset=True)


class Repository:
    """Keep SQL and domain state transitions out of HTTP route handlers."""

    def __init__(self, database: Database) -> None:
        self.database = database

    # Organizations -----------------------------------------------------
    def list_organizations(
        self, organization_id: str, *, platform_admin: bool
    ) -> list[OrganizationRead]:
        with self.database.connect() as connection:
            if platform_admin:
                rows = connection.execute(
                    "SELECT * FROM organizations WHERE status != 'archived' ORDER BY name"
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM organizations WHERE id = ? AND status != 'archived'",
                    (organization_id,),
                ).fetchall()
        return [_model(OrganizationRead, row) for row in rows]

    def get_organization(self, organization_id: str) -> OrganizationRead:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM organizations WHERE id = ?", (organization_id,)
            ).fetchone()
        if row is None:
            raise ResourceNotFoundError("Organization was not found")
        return _model(OrganizationRead, row)

    def create_organization(self, payload: OrganizationCreate) -> OrganizationRead:
        organization_id = _new_id("org")
        try:
            with self.database.transaction() as connection:
                connection.execute(
                    "INSERT INTO organizations "
                    "(id, name, slug, timezone, status, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        organization_id,
                        payload.name,
                        payload.slug,
                        payload.timezone,
                        OrganizationStatus.ACTIVE,
                        _now(),
                    ),
                )
        except sqlite3.IntegrityError as error:
            raise ConflictError("Organization slug already exists") from error
        return self.get_organization(organization_id)

    def update_organization(
        self, organization_id: str, payload: OrganizationUpdate
    ) -> OrganizationRead:
        self.get_organization(organization_id)
        self._update_row("organizations", organization_id, None, _payload(payload))
        return self.get_organization(organization_id)

    # Sites -------------------------------------------------------------
    def list_sites(self, organization_id: str, *, include_archived: bool = False) -> list[SiteRead]:
        sql = "SELECT * FROM sites WHERE organization_id = ?"
        params: list[Any] = [organization_id]
        if not include_archived:
            sql += " AND status != 'archived'"
        sql += " ORDER BY name"
        with self.database.connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [_model(SiteRead, row) for row in rows]

    def get_site(self, organization_id: str, site_id: str) -> SiteRead:
        return _model(SiteRead, self._get_scoped_row("sites", organization_id, site_id))

    def create_site(self, organization_id: str, payload: SiteCreate) -> SiteRead:
        self.get_organization(organization_id)
        site_id = _new_id("site")
        try:
            with self.database.transaction() as connection:
                connection.execute(
                    "INSERT INTO sites "
                    "(id, organization_id, code, name, timezone, address, latitude, longitude, "
                    "status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        site_id,
                        organization_id,
                        payload.code,
                        payload.name,
                        payload.timezone,
                        payload.address,
                        payload.latitude,
                        payload.longitude,
                        SiteStatus.ACTIVE,
                        _now(),
                    ),
                )
        except sqlite3.IntegrityError as error:
            raise ConflictError("Site code already exists in this organization") from error
        return self.get_site(organization_id, site_id)

    def update_site(self, organization_id: str, site_id: str, payload: SiteUpdate) -> SiteRead:
        self.get_site(organization_id, site_id)
        self._update_row("sites", site_id, organization_id, _payload(payload))
        return self.get_site(organization_id, site_id)

    def archive_site(self, organization_id: str, site_id: str) -> None:
        self.get_site(organization_id, site_id)
        with self.database.transaction() as connection:
            connection.execute(
                "UPDATE sites SET status = ? WHERE id = ? AND organization_id = ?",
                (SiteStatus.ARCHIVED, site_id, organization_id),
            )
            connection.execute(
                "UPDATE gates SET status = ? WHERE site_id = ? AND organization_id = ?",
                (GateStatus.DISABLED, site_id, organization_id),
            )
            connection.execute(
                "UPDATE cameras SET status = ? WHERE site_id = ? AND organization_id = ?",
                (CameraStatus.DISABLED, site_id, organization_id),
            )

    # Gates -------------------------------------------------------------
    def list_gates(
        self,
        organization_id: str,
        *,
        site_id: str | None = None,
        include_disabled: bool = False,
    ) -> list[GateRead]:
        sql = "SELECT * FROM gates WHERE organization_id = ?"
        params: list[Any] = [organization_id]
        if site_id is not None:
            self.get_site(organization_id, site_id)
            sql += " AND site_id = ?"
            params.append(site_id)
        if not include_disabled:
            sql += " AND status != 'disabled'"
        sql += " ORDER BY name"
        with self.database.connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [_model(GateRead, row) for row in rows]

    def get_gate(self, organization_id: str, gate_id: str) -> GateRead:
        return _model(GateRead, self._get_scoped_row("gates", organization_id, gate_id))

    def create_gate(self, organization_id: str, payload: GateCreate) -> GateRead:
        self.get_site(organization_id, payload.site_id)
        gate_id = _new_id("gate")
        try:
            with self.database.transaction() as connection:
                connection.execute(
                    "INSERT INTO gates "
                    "(id, organization_id, site_id, code, name, direction, latitude, longitude, "
                    "status, queue_estimate, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        gate_id,
                        organization_id,
                        payload.site_id,
                        payload.code,
                        payload.name,
                        payload.direction,
                        payload.latitude,
                        payload.longitude,
                        GateStatus.OPERATIONAL,
                        0,
                        _now(),
                    ),
                )
        except sqlite3.IntegrityError as error:
            raise ConflictError("Gate code already exists at this site") from error
        return self.get_gate(organization_id, gate_id)

    def update_gate(self, organization_id: str, gate_id: str, payload: GateUpdate) -> GateRead:
        self.get_gate(organization_id, gate_id)
        self._update_row("gates", gate_id, organization_id, _payload(payload))
        return self.get_gate(organization_id, gate_id)

    def disable_gate(self, organization_id: str, gate_id: str) -> None:
        self.get_gate(organization_id, gate_id)
        with self.database.transaction() as connection:
            connection.execute(
                "UPDATE gates SET status = ? WHERE id = ? AND organization_id = ?",
                (GateStatus.DISABLED, gate_id, organization_id),
            )
            connection.execute(
                "UPDATE cameras SET status = ? WHERE gate_id = ? AND organization_id = ?",
                (CameraStatus.DISABLED, gate_id, organization_id),
            )

    # Cameras -----------------------------------------------------------
    def list_cameras(
        self,
        organization_id: str,
        *,
        site_id: str | None = None,
        gate_id: str | None = None,
        include_disabled: bool = False,
    ) -> list[CameraRead]:
        sql = "SELECT * FROM cameras WHERE organization_id = ?"
        params: list[Any] = [organization_id]
        if site_id is not None:
            self.get_site(organization_id, site_id)
            sql += " AND site_id = ?"
            params.append(site_id)
        if gate_id is not None:
            self.get_gate(organization_id, gate_id)
            sql += " AND gate_id = ?"
            params.append(gate_id)
        if not include_disabled:
            sql += " AND status != 'disabled'"
        sql += " ORDER BY name"
        with self.database.connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [_model(CameraRead, row) for row in rows]

    def get_camera(self, organization_id: str, camera_id: str) -> CameraRead:
        return _model(CameraRead, self._get_scoped_row("cameras", organization_id, camera_id))

    def create_camera(self, organization_id: str, payload: CameraCreate) -> CameraRead:
        site = self.get_site(organization_id, payload.site_id)
        gate = self.get_gate(organization_id, payload.gate_id)
        if gate.site_id != site.id:
            raise InvalidStateError("Camera gate must belong to the selected site")
        camera_id = _new_id("camera")
        try:
            with self.database.transaction() as connection:
                connection.execute(
                    "INSERT INTO cameras "
                    "(id, organization_id, site_id, gate_id, code, name, role, stream_profile, "
                    "status, last_seen_at, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        camera_id,
                        organization_id,
                        payload.site_id,
                        payload.gate_id,
                        payload.code,
                        payload.name,
                        payload.role,
                        payload.stream_profile,
                        CameraStatus.OFFLINE,
                        None,
                        _now(),
                    ),
                )
        except sqlite3.IntegrityError as error:
            raise ConflictError("Camera code already exists in this organization") from error
        return self.get_camera(organization_id, camera_id)

    def update_camera(
        self, organization_id: str, camera_id: str, payload: CameraUpdate
    ) -> CameraRead:
        self.get_camera(organization_id, camera_id)
        self._update_row("cameras", camera_id, organization_id, _payload(payload))
        return self.get_camera(organization_id, camera_id)

    def disable_camera(self, organization_id: str, camera_id: str) -> None:
        self.get_camera(organization_id, camera_id)
        self._update_row("cameras", camera_id, organization_id, {"status": CameraStatus.DISABLED})

    # Access requests and grants ---------------------------------------
    def list_access_requests(
        self,
        organization_id: str,
        *,
        request_status: AccessRequestStatus | None = None,
        site_id: str | None = None,
        requested_by: str | None = None,
    ) -> list[AccessRequestRead]:
        sql = "SELECT * FROM access_requests WHERE organization_id = ?"
        params: list[Any] = [organization_id]
        if request_status is not None:
            sql += " AND status = ?"
            params.append(request_status)
        if site_id is not None:
            self.get_site(organization_id, site_id)
            sql += " AND site_id = ?"
            params.append(site_id)
        if requested_by is not None:
            sql += " AND requested_by = ?"
            params.append(requested_by)
        sql += " ORDER BY created_at DESC, id DESC"
        with self.database.connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [_model(AccessRequestRead, row) for row in rows]

    def get_access_request(self, organization_id: str, request_id: str) -> AccessRequestRead:
        return _model(
            AccessRequestRead,
            self._get_scoped_row("access_requests", organization_id, request_id),
        )

    def create_access_request(
        self,
        organization_id: str,
        requested_by: str,
        payload: AccessRequestCreate,
    ) -> AccessRequestRead:
        self.get_site(organization_id, payload.site_id)
        if payload.preferred_gate_id:
            gate = self.get_gate(organization_id, payload.preferred_gate_id)
            if gate.site_id != payload.site_id:
                raise InvalidStateError("Preferred gate must belong to the request site")
        request_id = _new_id("request")
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT INTO access_requests "
                "(id, organization_id, site_id, requested_by, requested_for_name, subject_kind, "
                "purpose, plate_text, valid_from, valid_until, preferred_gate_id, status, "
                "decision_reason, decided_by, decided_at, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    request_id,
                    organization_id,
                    payload.site_id,
                    requested_by,
                    payload.requested_for_name,
                    payload.subject_kind,
                    payload.purpose,
                    payload.plate_text,
                    payload.valid_from.isoformat(),
                    payload.valid_until.isoformat(),
                    payload.preferred_gate_id,
                    AccessRequestStatus.PENDING,
                    None,
                    None,
                    None,
                    _now(),
                ),
            )
            self._insert_event(
                connection,
                organization_id=organization_id,
                site_id=payload.site_id,
                gate_id=payload.preferred_gate_id,
                passage_id=None,
                source="access",
                event_type="access_request.submitted",
                severity=EventSeverity.INFO,
                summary=f"Access request submitted for {payload.requested_for_name}",
                evidence_label=None,
                metadata={"request_id": request_id},
            )
        return self.get_access_request(organization_id, request_id)

    def update_access_request(
        self,
        organization_id: str,
        request_id: str,
        payload: AccessRequestUpdate,
    ) -> AccessRequestRead:
        current = self.get_access_request(organization_id, request_id)
        if current.status is not AccessRequestStatus.PENDING:
            raise InvalidStateError("Only pending access requests can be changed")
        valid_from = payload.valid_from or current.valid_from
        valid_until = payload.valid_until or current.valid_until
        if valid_until <= valid_from:
            raise InvalidStateError("valid_until must be after valid_from")
        changes = _payload(payload)
        preferred_gate = changes.get("preferred_gate_id")
        if preferred_gate:
            gate = self.get_gate(organization_id, str(preferred_gate))
            if gate.site_id != current.site_id:
                raise InvalidStateError("Preferred gate must belong to the request site")
        self._update_row("access_requests", request_id, organization_id, changes)
        return self.get_access_request(organization_id, request_id)

    def cancel_access_request(self, organization_id: str, request_id: str) -> None:
        current = self.get_access_request(organization_id, request_id)
        if current.status is not AccessRequestStatus.PENDING:
            raise InvalidStateError("Only pending access requests can be cancelled")
        self._update_row(
            "access_requests",
            request_id,
            organization_id,
            {"status": AccessRequestStatus.CANCELLED},
        )

    def decide_access_request(
        self,
        organization_id: str,
        request_id: str,
        decided_by: str,
        payload: AccessRequestDecision,
    ) -> tuple[AccessRequestRead, AccessGrantRead | None]:
        request = self.get_access_request(organization_id, request_id)
        if request.status is not AccessRequestStatus.PENDING:
            raise InvalidStateError("Access request has already been decided")
        selected_gate = payload.gate_id or request.preferred_gate_id
        if selected_gate:
            gate = self.get_gate(organization_id, selected_gate)
            if gate.site_id != request.site_id:
                raise InvalidStateError("Selected gate must belong to the request site")

        grant_id = _new_id("grant") if payload.decision is AccessRequestStatus.APPROVED else None
        now = _now()
        with self.database.transaction() as connection:
            update = connection.execute(
                "UPDATE access_requests SET status = ?, decision_reason = ?, decided_by = ?, "
                "decided_at = ? WHERE id = ? AND organization_id = ? AND status = 'pending'",
                (payload.decision, payload.reason, decided_by, now, request_id, organization_id),
            )
            if update.rowcount != 1:
                raise ConflictError("Access request was decided concurrently")
            if grant_id:
                connection.execute(
                    "INSERT INTO access_grants "
                    "(id, organization_id, site_id, gate_id, source_request_id, subject_name, "
                    "subject_kind, plate_text, valid_from, valid_until, status, created_by, "
                    "created_at, revoked_at, revocation_reason) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        grant_id,
                        organization_id,
                        request.site_id,
                        selected_gate,
                        request.id,
                        request.requested_for_name,
                        request.subject_kind,
                        request.plate_text,
                        request.valid_from.isoformat(),
                        request.valid_until.isoformat(),
                        AccessGrantStatus.ACTIVE,
                        decided_by,
                        now,
                        None,
                        None,
                    ),
                )
            self._insert_event(
                connection,
                organization_id=organization_id,
                site_id=request.site_id,
                gate_id=selected_gate,
                passage_id=None,
                source="access",
                event_type=f"access_request.{payload.decision}",
                severity=EventSeverity.INFO,
                summary=f"Access request {payload.decision} for {request.requested_for_name}",
                evidence_label=None,
                metadata={"request_id": request_id, "grant_id": grant_id},
            )
        updated = self.get_access_request(organization_id, request_id)
        grant = self.get_access_grant(organization_id, grant_id) if grant_id else None
        return updated, grant

    def list_access_grants(
        self,
        organization_id: str,
        *,
        grant_status: AccessGrantStatus | None = None,
        plate_text: str | None = None,
    ) -> list[AccessGrantRead]:
        sql = "SELECT * FROM access_grants WHERE organization_id = ?"
        params: list[Any] = [organization_id]
        if grant_status is not None:
            sql += " AND status = ?"
            params.append(grant_status)
        if plate_text is not None:
            sql += " AND plate_text = ?"
            params.append(plate_text.strip().upper())
        sql += " ORDER BY created_at DESC, id DESC"
        with self.database.connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [_model(AccessGrantRead, row) for row in rows]

    def get_access_grant(self, organization_id: str, grant_id: str) -> AccessGrantRead:
        return _model(
            AccessGrantRead,
            self._get_scoped_row("access_grants", organization_id, grant_id),
        )

    def create_access_grant(
        self,
        organization_id: str,
        created_by: str,
        payload: AccessGrantCreate,
    ) -> AccessGrantRead:
        self.get_site(organization_id, payload.site_id)
        if payload.gate_id:
            gate = self.get_gate(organization_id, payload.gate_id)
            if gate.site_id != payload.site_id:
                raise InvalidStateError("Grant gate must belong to the selected site")
        if payload.source_request_id:
            self.get_access_request(organization_id, payload.source_request_id)
        grant_id = _new_id("grant")
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT INTO access_grants "
                "(id, organization_id, site_id, gate_id, source_request_id, subject_name, "
                "subject_kind, plate_text, valid_from, valid_until, status, created_by, "
                "created_at, revoked_at, revocation_reason) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    grant_id,
                    organization_id,
                    payload.site_id,
                    payload.gate_id,
                    payload.source_request_id,
                    payload.subject_name,
                    payload.subject_kind,
                    payload.plate_text,
                    payload.valid_from.isoformat(),
                    payload.valid_until.isoformat(),
                    AccessGrantStatus.ACTIVE,
                    created_by,
                    _now(),
                    None,
                    None,
                ),
            )
        return self.get_access_grant(organization_id, grant_id)

    def revoke_access_grant(
        self, organization_id: str, grant_id: str, *, reason: str
    ) -> AccessGrantRead:
        current = self.get_access_grant(organization_id, grant_id)
        if current.status is not AccessGrantStatus.ACTIVE:
            raise InvalidStateError("Only active grants can be revoked")
        self._update_row(
            "access_grants",
            grant_id,
            organization_id,
            {
                "status": AccessGrantStatus.REVOKED,
                "revoked_at": _now(),
                "revocation_reason": reason,
            },
        )
        return self.get_access_grant(organization_id, grant_id)

    # Passage / recognition / authorization ---------------------------
    def list_passages(
        self,
        organization_id: str,
        *,
        site_id: str | None = None,
        gate_id: str | None = None,
        passage_status: PassageStatus | None = None,
        limit: int = 100,
    ) -> list[PassageRead]:
        sql = "SELECT * FROM passages WHERE organization_id = ?"
        params: list[Any] = [organization_id]
        if site_id:
            self.get_site(organization_id, site_id)
            sql += " AND site_id = ?"
            params.append(site_id)
        if gate_id:
            self.get_gate(organization_id, gate_id)
            sql += " AND gate_id = ?"
            params.append(gate_id)
        if passage_status:
            sql += " AND status = ?"
            params.append(passage_status)
        sql += " ORDER BY occurred_at DESC, id DESC LIMIT ?"
        params.append(limit)
        with self.database.connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [_model(PassageRead, row) for row in rows]

    def get_passage(self, organization_id: str, passage_id: str) -> PassageRead:
        return _model(
            PassageRead,
            self._get_scoped_row("passages", organization_id, passage_id),
        )

    def get_passage_detail(self, organization_id: str, passage_id: str) -> PassageDetail:
        passage = self.get_passage(organization_id, passage_id)
        with self.database.connect() as connection:
            recognition_rows = connection.execute(
                "SELECT * FROM recognition_observations "
                "WHERE organization_id = ? AND passage_id = ? ORDER BY occurred_at, id",
                (organization_id, passage_id),
            ).fetchall()
            authorization_rows = connection.execute(
                "SELECT * FROM authorization_decisions "
                "WHERE organization_id = ? AND passage_id = ? ORDER BY occurred_at, id",
                (organization_id, passage_id),
            ).fetchall()
        return PassageDetail(
            **passage.model_dump(),
            recognitions=[_model(RecognitionRead, row) for row in recognition_rows],
            authorization_decisions=[
                _model(AuthorizationDecisionRead, row) for row in authorization_rows
            ],
        )

    def create_passage(self, organization_id: str, payload: PassageCreate) -> PassageRead:
        site = self.get_site(organization_id, payload.site_id)
        gate = self.get_gate(organization_id, payload.gate_id)
        if gate.site_id != site.id:
            raise InvalidStateError("Passage gate must belong to the selected site")
        if payload.camera_id:
            camera = self.get_camera(organization_id, payload.camera_id)
            if camera.gate_id != gate.id:
                raise InvalidStateError("Passage camera must belong to the selected gate")
        passage_id = _new_id("passage")
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT INTO passages "
                "(id, organization_id, site_id, gate_id, camera_id, direction, status, "
                "occurred_at, completed_at, evidence_label, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    passage_id,
                    organization_id,
                    payload.site_id,
                    payload.gate_id,
                    payload.camera_id,
                    payload.direction,
                    PassageStatus.OPEN,
                    payload.occurred_at.isoformat(),
                    None,
                    payload.evidence_label,
                    _now(),
                ),
            )
            self._insert_event(
                connection,
                organization_id=organization_id,
                site_id=payload.site_id,
                gate_id=payload.gate_id,
                passage_id=passage_id,
                source="capture",
                event_type="passage.captured",
                severity=EventSeverity.INFO,
                summary="Vehicle passage captured",
                evidence_label=payload.evidence_label,
                metadata={"camera_id": payload.camera_id},
            )
        return self.get_passage(organization_id, passage_id)

    def add_recognition(
        self,
        organization_id: str,
        passage_id: str,
        payload: RecognitionCreate,
    ) -> RecognitionRead:
        passage = self.get_passage(organization_id, passage_id)
        recognition_id = _new_id("recognition")
        occurred_at = _now()
        passage_status = (
            PassageStatus.REVIEW_REQUIRED
            if payload.status in {RecognitionStatus.UNCERTAIN, RecognitionStatus.UNREADABLE}
            else PassageStatus.OPEN
        )
        severity = (
            EventSeverity.WARNING
            if passage_status is PassageStatus.REVIEW_REQUIRED
            else EventSeverity.INFO
        )
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT INTO recognition_observations "
                "(id, organization_id, passage_id, status, plate_text, detection_confidence, "
                "recognition_confidence, format_valid, model_version, source, evidence_label, "
                "occurred_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    recognition_id,
                    organization_id,
                    passage_id,
                    payload.status,
                    payload.plate_text,
                    payload.detection_confidence,
                    payload.recognition_confidence,
                    int(payload.format_valid),
                    payload.model_version,
                    payload.source,
                    payload.evidence_label,
                    occurred_at,
                ),
            )
            connection.execute(
                "UPDATE passages SET status = ? WHERE id = ? AND organization_id = ?",
                (passage_status, passage_id, organization_id),
            )
            self._insert_event(
                connection,
                organization_id=organization_id,
                site_id=passage.site_id,
                gate_id=passage.gate_id,
                passage_id=passage_id,
                source="recognition",
                event_type=f"recognition.{payload.status}",
                severity=severity,
                summary=(
                    "Recognition completed"
                    if payload.status is RecognitionStatus.RECOGNIZED
                    else "Recognition requires operator review"
                ),
                evidence_label=payload.evidence_label,
                metadata={
                    "recognition_id": recognition_id,
                    "confidence": payload.recognition_confidence,
                    "format_valid": payload.format_valid,
                },
            )
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM recognition_observations WHERE id = ? AND organization_id = ?",
                (recognition_id, organization_id),
            ).fetchone()
        if row is None:
            raise ResourceNotFoundError("Recognition observation was not found")
        return _model(RecognitionRead, row)

    def add_authorization_decision(
        self,
        organization_id: str,
        passage_id: str,
        decided_by: str,
        payload: AuthorizationDecisionCreate,
    ) -> AuthorizationDecisionRead:
        passage = self.get_passage(organization_id, passage_id)
        grant: AccessGrantRead | None = None
        if payload.grant_id:
            grant = self.get_access_grant(organization_id, payload.grant_id)
            if grant.site_id != passage.site_id:
                raise InvalidStateError("Authorization grant belongs to another site")
        if (
            payload.outcome is AuthorizationOutcome.ALLOWED
            and payload.grant_id is None
            and payload.source.value != "operator"
        ):
            raise InvalidStateError("Automated allow decisions require an access grant")
        if (
            payload.outcome is AuthorizationOutcome.ALLOWED
            and payload.source is DecisionSource.POLICY
            and grant is not None
        ):
            if grant.status is not AccessGrantStatus.ACTIVE:
                raise InvalidStateError("Automated allow decisions require an active grant")
            if not grant.valid_from <= passage.occurred_at <= grant.valid_until:
                raise InvalidStateError("Access grant is outside its validity window")
            if grant.gate_id is not None and grant.gate_id != passage.gate_id:
                raise InvalidStateError("Access grant is assigned to another gate")
            with self.database.connect() as connection:
                latest_recognition = connection.execute(
                    "SELECT plate_text FROM recognition_observations "
                    "WHERE organization_id = ? AND passage_id = ? "
                    "ORDER BY occurred_at DESC, id DESC LIMIT 1",
                    (organization_id, passage_id),
                ).fetchone()
            recognized_plate = (
                str(latest_recognition["plate_text"])
                if latest_recognition is not None and latest_recognition["plate_text"]
                else None
            )
            if grant.plate_text and _canonical_plate(recognized_plate) != _canonical_plate(
                grant.plate_text
            ):
                raise InvalidStateError("Recognized plate does not match the access grant")
        decision_id = _new_id("authorization")
        occurred_at = _now()
        passage_status = (
            PassageStatus.REVIEW_REQUIRED
            if payload.outcome is AuthorizationOutcome.REVIEW_REQUIRED
            else PassageStatus.COMPLETED
        )
        severity = {
            AuthorizationOutcome.ALLOWED: EventSeverity.INFO,
            AuthorizationOutcome.REVIEW_REQUIRED: EventSeverity.WARNING,
            AuthorizationOutcome.NO_MATCH: EventSeverity.WARNING,
            AuthorizationOutcome.DENIED: EventSeverity.CRITICAL,
        }[payload.outcome]
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT INTO authorization_decisions "
                "(id, organization_id, passage_id, outcome, reason, source, grant_id, "
                "decided_by, occurred_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    decision_id,
                    organization_id,
                    passage_id,
                    payload.outcome,
                    payload.reason,
                    payload.source,
                    payload.grant_id,
                    decided_by,
                    occurred_at,
                ),
            )
            connection.execute(
                "UPDATE passages SET status = ?, completed_at = ? "
                "WHERE id = ? AND organization_id = ?",
                (
                    passage_status,
                    None if passage_status is PassageStatus.REVIEW_REQUIRED else occurred_at,
                    passage_id,
                    organization_id,
                ),
            )
            self._insert_event(
                connection,
                organization_id=organization_id,
                site_id=passage.site_id,
                gate_id=passage.gate_id,
                passage_id=passage_id,
                source="authorization",
                event_type=f"authorization.{payload.outcome}",
                severity=severity,
                summary=f"Authorization outcome: {payload.outcome}",
                evidence_label=passage.evidence_label,
                metadata={"decision_id": decision_id, "grant_id": payload.grant_id},
            )
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM authorization_decisions WHERE id = ? AND organization_id = ?",
                (decision_id, organization_id),
            ).fetchone()
        if row is None:
            raise ResourceNotFoundError("Authorization decision was not found")
        return _model(AuthorizationDecisionRead, row)

    # Event polling -----------------------------------------------------
    def list_events(
        self,
        organization_id: str,
        *,
        after_sequence: int,
        limit: int,
        site_id: str | None = None,
        severity: EventSeverity | None = None,
    ) -> EventPage:
        sql = "SELECT * FROM events WHERE organization_id = ? AND sequence > ?"
        params: list[Any] = [organization_id, after_sequence]
        if site_id:
            self.get_site(organization_id, site_id)
            sql += " AND site_id = ?"
            params.append(site_id)
        if severity:
            sql += " AND severity = ?"
            params.append(severity)
        sql += " ORDER BY sequence ASC LIMIT ?"
        params.append(limit + 1)
        with self.database.connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        has_more = len(rows) > limit
        selected = rows[:limit]
        items = [_model(EventRead, row) for row in selected]
        next_sequence = items[-1].sequence if items else after_sequence
        return EventPage(items=items, next_sequence=next_sequence, has_more=has_more)

    def recent_events(self, organization_id: str, *, limit: int) -> list[EventRead]:
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM events WHERE organization_id = ? ORDER BY sequence DESC LIMIT ?",
                (organization_id, limit),
            ).fetchall()
        return [_model(EventRead, row) for row in rows]

    # Incidents ---------------------------------------------------------
    def list_incidents(
        self,
        organization_id: str,
        *,
        incident_status: IncidentStatus | None = None,
        site_id: str | None = None,
    ) -> list[IncidentRead]:
        sql = "SELECT * FROM incidents WHERE organization_id = ?"
        params: list[Any] = [organization_id]
        if incident_status:
            sql += " AND status = ?"
            params.append(incident_status)
        if site_id:
            self.get_site(organization_id, site_id)
            sql += " AND site_id = ?"
            params.append(site_id)
        sql += " ORDER BY created_at DESC, id DESC"
        with self.database.connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [_model(IncidentRead, row) for row in rows]

    def get_incident(self, organization_id: str, incident_id: str) -> IncidentRead:
        return _model(
            IncidentRead,
            self._get_scoped_row("incidents", organization_id, incident_id),
        )

    def create_incident(
        self,
        organization_id: str,
        created_by: str,
        payload: IncidentCreate,
    ) -> IncidentRead:
        self.get_site(organization_id, payload.site_id)
        if payload.gate_id:
            self.get_gate(organization_id, payload.gate_id)
        if payload.passage_id:
            self.get_passage(organization_id, payload.passage_id)
        incident_id = _new_id("incident")
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT INTO incidents "
                "(id, organization_id, site_id, gate_id, passage_id, title, severity, status, "
                "description, assigned_to, created_by, created_at, resolved_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    incident_id,
                    organization_id,
                    payload.site_id,
                    payload.gate_id,
                    payload.passage_id,
                    payload.title,
                    payload.severity,
                    IncidentStatus.OPEN,
                    payload.description,
                    None,
                    created_by,
                    _now(),
                    None,
                ),
            )
            self._insert_event(
                connection,
                organization_id=organization_id,
                site_id=payload.site_id,
                gate_id=payload.gate_id,
                passage_id=payload.passage_id,
                source="incident",
                event_type="incident.opened",
                severity=EventSeverity(payload.severity),
                summary=payload.title,
                evidence_label=None,
                metadata={"incident_id": incident_id},
            )
        return self.get_incident(organization_id, incident_id)

    def update_incident(
        self,
        organization_id: str,
        incident_id: str,
        payload: IncidentUpdate,
    ) -> IncidentRead:
        current = self.get_incident(organization_id, incident_id)
        changes = _payload(payload)
        if changes.get("status") == IncidentStatus.RESOLVED:
            changes["resolved_at"] = _now()
        elif current.status is IncidentStatus.RESOLVED and changes.get("status"):
            changes["resolved_at"] = None
        self._update_row("incidents", incident_id, organization_id, changes)
        return self.get_incident(organization_id, incident_id)

    # Device health -----------------------------------------------------
    def list_device_health(
        self,
        organization_id: str,
        *,
        site_id: str | None = None,
        latest_only: bool = True,
        limit: int = 100,
    ) -> list[DeviceHealthRead]:
        params: list[Any] = [organization_id]
        if site_id:
            self.get_site(organization_id, site_id)
            params.append(site_id)
        if latest_only:
            if site_id:
                sql = (
                    "SELECT health.* FROM device_health AS health "
                    "WHERE health.organization_id = ? AND health.site_id = ? "
                    "AND health.reported_at = ("
                    "SELECT MAX(newer.reported_at) FROM device_health AS newer "
                    "WHERE newer.organization_id = health.organization_id "
                    "AND newer.device_id = health.device_id) "
                    "ORDER BY health.reported_at DESC, health.id DESC LIMIT ?"
                )
            else:
                sql = (
                    "SELECT health.* FROM device_health AS health "
                    "WHERE health.organization_id = ? "
                    "AND health.reported_at = ("
                    "SELECT MAX(newer.reported_at) FROM device_health AS newer "
                    "WHERE newer.organization_id = health.organization_id "
                    "AND newer.device_id = health.device_id) "
                    "ORDER BY health.reported_at DESC, health.id DESC LIMIT ?"
                )
        else:
            if site_id:
                sql = (
                    "SELECT health.* FROM device_health AS health "
                    "WHERE health.organization_id = ? AND health.site_id = ? "
                    "ORDER BY health.reported_at DESC, health.id DESC LIMIT ?"
                )
            else:
                sql = (
                    "SELECT health.* FROM device_health AS health "
                    "WHERE health.organization_id = ? "
                    "ORDER BY health.reported_at DESC, health.id DESC LIMIT ?"
                )
        params.append(limit)
        with self.database.connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [_model(DeviceHealthRead, row) for row in rows]

    def report_device_health(
        self, organization_id: str, payload: DeviceHealthCreate
    ) -> DeviceHealthRead:
        self.get_site(organization_id, payload.site_id)
        if payload.gate_id:
            self.get_gate(organization_id, payload.gate_id)
        if payload.camera_id:
            camera = self.get_camera(organization_id, payload.camera_id)
            if camera.site_id != payload.site_id:
                raise InvalidStateError("Health camera must belong to the selected site")
        health_id = _new_id("health")
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT INTO device_health "
                "(id, organization_id, site_id, gate_id, camera_id, device_id, device_type, "
                "status, latency_ms, detail, reported_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    health_id,
                    organization_id,
                    payload.site_id,
                    payload.gate_id,
                    payload.camera_id,
                    payload.device_id,
                    payload.device_type,
                    payload.status,
                    payload.latency_ms,
                    payload.detail,
                    payload.reported_at.isoformat(),
                ),
            )
            if payload.camera_id:
                camera_status = {
                    "online": CameraStatus.ONLINE,
                    "degraded": CameraStatus.DEGRADED,
                    "offline": CameraStatus.OFFLINE,
                    "unknown": CameraStatus.OFFLINE,
                }[payload.status]
                connection.execute(
                    "UPDATE cameras SET status = ?, last_seen_at = ? "
                    "WHERE id = ? AND organization_id = ?",
                    (
                        camera_status,
                        payload.reported_at.isoformat(),
                        payload.camera_id,
                        organization_id,
                    ),
                )
            severity = (
                EventSeverity.INFO if payload.status.value == "online" else EventSeverity.WARNING
            )
            self._insert_event(
                connection,
                organization_id=organization_id,
                site_id=payload.site_id,
                gate_id=payload.gate_id,
                passage_id=None,
                source="device",
                event_type=f"device.{payload.status}",
                severity=severity,
                summary=f"{payload.device_type} {payload.device_id} is {payload.status}",
                evidence_label=None,
                metadata={"device_id": payload.device_id, "latency_ms": payload.latency_ms},
            )
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM device_health WHERE id = ? AND organization_id = ?",
                (health_id, organization_id),
            ).fetchone()
        if row is None:
            raise ResourceNotFoundError("Device health report was not found")
        return _model(DeviceHealthRead, row)

    # Aggregate ---------------------------------------------------------
    def counts(self, organization_id: str) -> dict[str, int]:
        today_prefix = datetime.now(UTC).date().isoformat()
        queries = {
            "sites": "SELECT COUNT(*) FROM sites WHERE organization_id = ? AND status != 'archived'",
            "gates": "SELECT COUNT(*) FROM gates WHERE organization_id = ? AND status != 'disabled'",
            "cameras": "SELECT COUNT(*) FROM cameras WHERE organization_id = ? AND status != 'disabled'",
            "open_incidents": "SELECT COUNT(*) FROM incidents WHERE organization_id = ? AND status != 'resolved'",
            "pending_access_requests": "SELECT COUNT(*) FROM access_requests WHERE organization_id = ? AND status = 'pending'",
            "passages_today": "SELECT COUNT(*) FROM passages WHERE organization_id = ? AND occurred_at LIKE ?",
        }
        result: dict[str, int] = {}
        with self.database.connect() as connection:
            for key, query in queries.items():
                params = (
                    (organization_id, f"{today_prefix}%")
                    if key == "passages_today"
                    else (organization_id,)
                )
                row = connection.execute(query, params).fetchone()
                result[key] = int(row[0]) if row is not None else 0
        return result

    # Internal helpers --------------------------------------------------
    def _get_scoped_row(self, table: str, organization_id: str, row_id: str) -> sqlite3.Row:
        queries = {
            "sites": "SELECT * FROM sites WHERE id = ? AND organization_id = ?",
            "gates": "SELECT * FROM gates WHERE id = ? AND organization_id = ?",
            "cameras": "SELECT * FROM cameras WHERE id = ? AND organization_id = ?",
            "access_requests": (
                "SELECT * FROM access_requests WHERE id = ? AND organization_id = ?"
            ),
            "access_grants": ("SELECT * FROM access_grants WHERE id = ? AND organization_id = ?"),
            "passages": "SELECT * FROM passages WHERE id = ? AND organization_id = ?",
            "incidents": "SELECT * FROM incidents WHERE id = ? AND organization_id = ?",
        }
        query = queries.get(table)
        if query is None:
            raise ValueError("Unsupported table")
        with self.database.connect() as connection:
            row = connection.execute(query, (row_id, organization_id)).fetchone()
        if row is None:
            raise ResourceNotFoundError(
                f"{table.removesuffix('s').replace('_', ' ')} was not found"
            )
        return cast(sqlite3.Row, row)

    def _update_row(
        self,
        table: str,
        row_id: str,
        organization_id: str | None,
        changes: dict[str, Any],
    ) -> None:
        allowed_columns = {
            "organizations": {"name", "timezone", "status"},
            "sites": {"name", "timezone", "address", "latitude", "longitude", "status"},
            "gates": {"name", "direction", "latitude", "longitude", "status", "queue_estimate"},
            "cameras": {"name", "role", "stream_profile", "status", "last_seen_at"},
            "access_requests": {
                "requested_for_name",
                "purpose",
                "plate_text",
                "valid_from",
                "valid_until",
                "preferred_gate_id",
                "status",
            },
            "access_grants": {"status", "revoked_at", "revocation_reason"},
            "incidents": {
                "title",
                "severity",
                "status",
                "description",
                "assigned_to",
                "resolved_at",
            },
        }
        if table not in allowed_columns:
            raise ValueError("Unsupported table")
        invalid = set(changes) - allowed_columns[table]
        if invalid:
            raise ValueError(f"Unsupported update columns: {sorted(invalid)}")
        if not changes:
            return
        assignments = ", ".join(f"{column} = ?" for column in changes)
        values = list(changes.values())
        sql = f"UPDATE {table} SET {assignments} WHERE id = ?"  # noqa: S608
        values.append(row_id)
        if organization_id is not None:
            sql += " AND organization_id = ?"
            values.append(organization_id)
        try:
            with self.database.transaction() as connection:
                cursor = connection.execute(sql, values)
                if cursor.rowcount != 1:
                    raise ResourceNotFoundError("Resource was not found")
        except sqlite3.IntegrityError as error:
            raise ConflictError("Update conflicts with an existing resource") from error

    @staticmethod
    def _insert_event(
        connection: sqlite3.Connection,
        *,
        organization_id: str,
        site_id: str | None,
        gate_id: str | None,
        passage_id: str | None,
        source: str,
        event_type: str,
        severity: EventSeverity,
        summary: str,
        evidence_label: str | None,
        metadata: dict[str, Any],
    ) -> None:
        connection.execute(
            "INSERT INTO events "
            "(id, organization_id, site_id, gate_id, passage_id, source, event_type, severity, "
            "summary, evidence_label, metadata_json, occurred_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                _new_id("event"),
                organization_id,
                site_id,
                gate_id,
                passage_id,
                source,
                event_type,
                severity,
                summary,
                evidence_label,
                json.dumps(metadata, sort_keys=True),
                _now(),
            ),
        )
