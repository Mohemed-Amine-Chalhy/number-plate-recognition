"""Typed API contracts for the campus control plane."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Identifier = Annotated[str, Field(min_length=1, max_length=80, pattern=r"^[a-zA-Z0-9_-]+$")]
Name = Annotated[str, Field(min_length=1, max_length=160)]


class ApiModel(BaseModel):
    """Reject unknown fields so clients notice contract mistakes early."""

    model_config = ConfigDict(extra="forbid")


class Role(StrEnum):
    PLATFORM_ADMIN = "platform_admin"
    ORG_ADMIN = "org_admin"
    SECURITY_OPERATOR = "security_operator"
    HOST = "host"
    VIEWER = "viewer"
    EDGE_AGENT = "edge_agent"


class OrganizationStatus(StrEnum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    ARCHIVED = "archived"


class SiteStatus(StrEnum):
    ACTIVE = "active"
    DEGRADED = "degraded"
    OFFLINE = "offline"
    ARCHIVED = "archived"


class GateStatus(StrEnum):
    OPERATIONAL = "operational"
    CONGESTED = "congested"
    DEGRADED = "degraded"
    OFFLINE = "offline"
    DISABLED = "disabled"


class CameraStatus(StrEnum):
    ONLINE = "online"
    DEGRADED = "degraded"
    OFFLINE = "offline"
    DISABLED = "disabled"


class GateDirection(StrEnum):
    INBOUND = "inbound"
    OUTBOUND = "outbound"
    BIDIRECTIONAL = "bidirectional"


class AccessRequestStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class AccessGrantStatus(StrEnum):
    ACTIVE = "active"
    REVOKED = "revoked"
    EXPIRED = "expired"


class PassageStatus(StrEnum):
    OPEN = "open"
    COMPLETED = "completed"
    REVIEW_REQUIRED = "review_required"


class RecognitionStatus(StrEnum):
    RECOGNIZED = "recognized"
    UNCERTAIN = "uncertain"
    UNREADABLE = "unreadable"


class AuthorizationOutcome(StrEnum):
    ALLOWED = "allowed"
    REVIEW_REQUIRED = "review_required"
    DENIED = "denied"
    NO_MATCH = "no_match"


class DecisionSource(StrEnum):
    POLICY = "policy"
    OPERATOR = "operator"
    SYSTEM = "system"


class IncidentSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class IncidentStatus(StrEnum):
    OPEN = "open"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"


class DeviceStatus(StrEnum):
    ONLINE = "online"
    DEGRADED = "degraded"
    OFFLINE = "offline"
    UNKNOWN = "unknown"


class EventSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class OrganizationCreate(ApiModel):
    name: Name
    slug: Annotated[str, Field(min_length=2, max_length=80, pattern=r"^[a-z0-9-]+$")]
    timezone: str = Field(default="Africa/Casablanca", min_length=1, max_length=80)


class OrganizationUpdate(ApiModel):
    name: Name | None = None
    timezone: str | None = Field(default=None, min_length=1, max_length=80)
    status: OrganizationStatus | None = None


class OrganizationRead(ApiModel):
    id: Identifier
    name: str
    slug: str
    timezone: str
    status: OrganizationStatus
    created_at: datetime


class SiteCreate(ApiModel):
    code: Annotated[str, Field(min_length=1, max_length=30)]
    name: Name
    timezone: str = Field(default="Africa/Casablanca", min_length=1, max_length=80)
    address: str = Field(default="", max_length=240)
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)


class SiteUpdate(ApiModel):
    name: Name | None = None
    timezone: str | None = Field(default=None, min_length=1, max_length=80)
    address: str | None = Field(default=None, max_length=240)
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    status: SiteStatus | None = None


class SiteRead(ApiModel):
    id: Identifier
    organization_id: Identifier
    code: str
    name: str
    timezone: str
    address: str
    latitude: float | None
    longitude: float | None
    status: SiteStatus
    created_at: datetime


class GateCreate(ApiModel):
    site_id: Identifier
    code: Annotated[str, Field(min_length=1, max_length=30)]
    name: Name
    direction: GateDirection
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)


class GateUpdate(ApiModel):
    name: Name | None = None
    direction: GateDirection | None = None
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    status: GateStatus | None = None
    queue_estimate: int | None = Field(default=None, ge=0, le=999)


class GateRead(ApiModel):
    id: Identifier
    organization_id: Identifier
    site_id: Identifier
    code: str
    name: str
    direction: GateDirection
    latitude: float | None
    longitude: float | None
    status: GateStatus
    queue_estimate: int
    created_at: datetime


class CameraCreate(ApiModel):
    site_id: Identifier
    gate_id: Identifier
    code: Annotated[str, Field(min_length=1, max_length=30)]
    name: Name
    role: str = Field(default="anpr", min_length=1, max_length=40)
    stream_profile: str = Field(default="primary", min_length=1, max_length=80)


class CameraUpdate(ApiModel):
    name: Name | None = None
    role: str | None = Field(default=None, min_length=1, max_length=40)
    stream_profile: str | None = Field(default=None, min_length=1, max_length=80)
    status: CameraStatus | None = None
    last_seen_at: datetime | None = None


class CameraRead(ApiModel):
    id: Identifier
    organization_id: Identifier
    site_id: Identifier
    gate_id: Identifier
    code: str
    name: str
    role: str
    stream_profile: str
    status: CameraStatus
    last_seen_at: datetime | None
    created_at: datetime


class AccessRequestCreate(ApiModel):
    site_id: Identifier
    requested_for_name: Name
    subject_kind: str = Field(min_length=1, max_length=40)
    purpose: str = Field(min_length=1, max_length=400)
    plate_text: str | None = Field(default=None, max_length=32)
    valid_from: datetime
    valid_until: datetime
    preferred_gate_id: Identifier | None = None

    @field_validator("plate_text")
    @classmethod
    def normalize_plate(cls, value: str | None) -> str | None:
        return value.strip().upper() if value else None

    @model_validator(mode="after")
    def validate_window(self) -> AccessRequestCreate:
        if self.valid_until <= self.valid_from:
            raise ValueError("valid_until must be after valid_from")
        return self


class AccessRequestUpdate(ApiModel):
    requested_for_name: Name | None = None
    purpose: str | None = Field(default=None, min_length=1, max_length=400)
    plate_text: str | None = Field(default=None, max_length=32)
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    preferred_gate_id: Identifier | None = None

    @field_validator("plate_text")
    @classmethod
    def normalize_plate(cls, value: str | None) -> str | None:
        return value.strip().upper() if value else None


class AccessRequestRead(ApiModel):
    id: Identifier
    organization_id: Identifier
    site_id: Identifier
    requested_by: str
    requested_for_name: str
    subject_kind: str
    purpose: str
    plate_text: str | None
    valid_from: datetime
    valid_until: datetime
    preferred_gate_id: str | None
    status: AccessRequestStatus
    decision_reason: str | None
    decided_by: str | None
    decided_at: datetime | None
    created_at: datetime


class AccessRequestDecision(ApiModel):
    decision: Annotated[AccessRequestStatus, Field(description="approved or rejected")]
    reason: str = Field(min_length=1, max_length=400)
    gate_id: Identifier | None = None

    @model_validator(mode="after")
    def validate_decision(self) -> AccessRequestDecision:
        if self.decision not in {AccessRequestStatus.APPROVED, AccessRequestStatus.REJECTED}:
            raise ValueError("decision must be approved or rejected")
        return self


class AccessGrantCreate(ApiModel):
    site_id: Identifier
    gate_id: Identifier | None = None
    subject_name: Name
    subject_kind: str = Field(min_length=1, max_length=40)
    plate_text: str | None = Field(default=None, max_length=32)
    valid_from: datetime
    valid_until: datetime
    source_request_id: Identifier | None = None

    @field_validator("plate_text")
    @classmethod
    def normalize_plate(cls, value: str | None) -> str | None:
        return value.strip().upper() if value else None

    @model_validator(mode="after")
    def validate_window(self) -> AccessGrantCreate:
        if self.valid_until <= self.valid_from:
            raise ValueError("valid_until must be after valid_from")
        return self


class AccessGrantRead(ApiModel):
    id: Identifier
    organization_id: Identifier
    site_id: Identifier
    gate_id: str | None
    source_request_id: str | None
    subject_name: str
    subject_kind: str
    plate_text: str | None
    valid_from: datetime
    valid_until: datetime
    status: AccessGrantStatus
    created_by: str
    created_at: datetime
    revoked_at: datetime | None
    revocation_reason: str | None


class AccessRequestDecisionResult(ApiModel):
    request: AccessRequestRead
    grant: AccessGrantRead | None


class GrantRevoke(ApiModel):
    reason: str = Field(min_length=1, max_length=400)


class PassageCreate(ApiModel):
    site_id: Identifier
    gate_id: Identifier
    camera_id: Identifier | None = None
    direction: GateDirection
    occurred_at: datetime
    evidence_label: str = Field(
        default="Synthetic composite - no real person or operational record",
        min_length=1,
        max_length=200,
    )


class PassageRead(ApiModel):
    id: Identifier
    organization_id: Identifier
    site_id: Identifier
    gate_id: Identifier
    camera_id: str | None
    direction: GateDirection
    status: PassageStatus
    occurred_at: datetime
    completed_at: datetime | None
    evidence_label: str
    created_at: datetime


class RecognitionCreate(ApiModel):
    status: RecognitionStatus
    plate_text: str | None = Field(default=None, max_length=32)
    detection_confidence: float | None = Field(default=None, ge=0, le=1)
    recognition_confidence: float | None = Field(default=None, ge=0, le=1)
    format_valid: bool = False
    model_version: str = Field(min_length=1, max_length=200)
    source: str = Field(default="central_worker", min_length=1, max_length=40)
    evidence_label: str = Field(
        default="Synthetic composite - no real person or operational record",
        min_length=1,
        max_length=200,
    )

    @field_validator("plate_text")
    @classmethod
    def normalize_plate(cls, value: str | None) -> str | None:
        return value.strip().upper() if value else None


class RecognitionRead(ApiModel):
    id: Identifier
    organization_id: Identifier
    passage_id: Identifier
    status: RecognitionStatus
    plate_text: str | None
    detection_confidence: float | None
    recognition_confidence: float | None
    format_valid: bool
    model_version: str
    source: str
    evidence_label: str
    occurred_at: datetime


class AuthorizationDecisionCreate(ApiModel):
    outcome: AuthorizationOutcome
    reason: str = Field(min_length=1, max_length=400)
    source: DecisionSource = DecisionSource.OPERATOR
    grant_id: Identifier | None = None


class AuthorizationDecisionRead(ApiModel):
    id: Identifier
    organization_id: Identifier
    passage_id: Identifier
    outcome: AuthorizationOutcome
    reason: str
    source: DecisionSource
    grant_id: str | None
    decided_by: str
    occurred_at: datetime


class PassageDetail(PassageRead):
    recognitions: list[RecognitionRead]
    authorization_decisions: list[AuthorizationDecisionRead]


class EventRead(ApiModel):
    sequence: int
    id: Identifier
    organization_id: Identifier
    site_id: str | None
    gate_id: str | None
    passage_id: str | None
    source: str
    event_type: str
    severity: EventSeverity
    summary: str
    evidence_label: str | None
    metadata: dict[str, Any]
    occurred_at: datetime


class EventPage(ApiModel):
    items: list[EventRead]
    next_sequence: int
    has_more: bool


class IncidentCreate(ApiModel):
    site_id: Identifier
    gate_id: Identifier | None = None
    passage_id: Identifier | None = None
    title: Name
    severity: IncidentSeverity
    description: str = Field(min_length=1, max_length=1000)


class IncidentUpdate(ApiModel):
    title: Name | None = None
    severity: IncidentSeverity | None = None
    status: IncidentStatus | None = None
    description: str | None = Field(default=None, min_length=1, max_length=1000)
    assigned_to: str | None = Field(default=None, max_length=120)


class IncidentRead(ApiModel):
    id: Identifier
    organization_id: Identifier
    site_id: Identifier
    gate_id: str | None
    passage_id: str | None
    title: str
    severity: IncidentSeverity
    status: IncidentStatus
    description: str
    assigned_to: str | None
    created_by: str
    created_at: datetime
    resolved_at: datetime | None


class DeviceHealthCreate(ApiModel):
    site_id: Identifier
    gate_id: Identifier | None = None
    camera_id: Identifier | None = None
    device_id: Identifier
    device_type: str = Field(min_length=1, max_length=40)
    status: DeviceStatus
    latency_ms: float | None = Field(default=None, ge=0)
    detail: str = Field(default="", max_length=400)
    reported_at: datetime


class DeviceHealthRead(ApiModel):
    id: Identifier
    organization_id: Identifier
    site_id: Identifier
    gate_id: str | None
    camera_id: str | None
    device_id: str
    device_type: str
    status: DeviceStatus
    latency_ms: float | None
    detail: str
    reported_at: datetime


class PrincipalRead(ApiModel):
    subject: str
    display_name: str
    organization_id: str
    roles: list[Role]


class DemoIdentity(ApiModel):
    token: str
    display_name: str
    organization_id: str
    roles: list[Role]
    use_case: str


class DashboardCounts(ApiModel):
    sites: int
    gates: int
    cameras: int
    open_incidents: int
    pending_access_requests: int
    passages_today: int


class DashboardRead(ApiModel):
    counts: DashboardCounts
    gates: list[GateRead]
    recent_events: list[EventRead]
    open_incidents: list[IncidentRead]
    device_health: list[DeviceHealthRead]


class HealthRead(ApiModel):
    status: str
    service: str
    schema_version: int | None = None


class ErrorDetail(ApiModel):
    type: str
    title: str
    status: int
    detail: str
    instance: str
