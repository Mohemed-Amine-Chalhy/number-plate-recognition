"""Passage, recognition, authorization, event, incident, and device-health routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, status

from control_api.dependencies import (
    AuthorizationPrincipal,
    HealthPrincipal,
    IncidentPrincipal,
    PassagePrincipal,
    ReadPrincipal,
    RecognitionPrincipal,
    RepositoryDependency,
    TenantDependency,
)
from control_api.schemas import (
    AuthorizationDecisionCreate,
    AuthorizationDecisionRead,
    DeviceHealthCreate,
    DeviceHealthRead,
    EventPage,
    EventSeverity,
    IncidentCreate,
    IncidentRead,
    IncidentStatus,
    IncidentUpdate,
    PassageCreate,
    PassageDetail,
    PassageRead,
    PassageStatus,
    RecognitionCreate,
    RecognitionRead,
)

router = APIRouter(prefix="/api/v1", tags=["operations"])


@router.get("/passages", response_model=list[PassageRead])
def list_passages(
    repository: RepositoryDependency,
    tenant: TenantDependency,
    _principal: ReadPrincipal,
    site_id: str | None = None,
    gate_id: str | None = None,
    passage_status: PassageStatus | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> list[PassageRead]:
    return repository.list_passages(
        tenant.organization_id,
        site_id=site_id,
        gate_id=gate_id,
        passage_status=passage_status,
        limit=limit,
    )


@router.post("/passages", response_model=PassageRead, status_code=status.HTTP_201_CREATED)
def create_passage(
    payload: PassageCreate,
    repository: RepositoryDependency,
    tenant: TenantDependency,
    _principal: PassagePrincipal,
) -> PassageRead:
    return repository.create_passage(tenant.organization_id, payload)


@router.get("/passages/{passage_id}", response_model=PassageDetail)
def get_passage(
    passage_id: str,
    repository: RepositoryDependency,
    tenant: TenantDependency,
    _principal: ReadPrincipal,
) -> PassageDetail:
    return repository.get_passage_detail(tenant.organization_id, passage_id)


@router.post(
    "/passages/{passage_id}/recognitions",
    response_model=RecognitionRead,
    status_code=status.HTTP_201_CREATED,
    summary="Attach a recognition observation without making an access decision",
)
def add_recognition(
    passage_id: str,
    payload: RecognitionCreate,
    repository: RepositoryDependency,
    tenant: TenantDependency,
    _principal: RecognitionPrincipal,
) -> RecognitionRead:
    return repository.add_recognition(tenant.organization_id, passage_id, payload)


@router.post(
    "/passages/{passage_id}/authorization-decisions",
    response_model=AuthorizationDecisionRead,
    status_code=status.HTTP_201_CREATED,
    summary="Record a policy or operator decision independently from recognition",
)
def add_authorization_decision(
    passage_id: str,
    payload: AuthorizationDecisionCreate,
    repository: RepositoryDependency,
    tenant: TenantDependency,
    principal: AuthorizationPrincipal,
) -> AuthorizationDecisionRead:
    return repository.add_authorization_decision(
        tenant.organization_id,
        passage_id,
        principal.subject,
        payload,
    )


@router.get(
    "/events",
    response_model=EventPage,
    summary="Poll the ordered organization event feed",
)
def list_events(
    repository: RepositoryDependency,
    tenant: TenantDependency,
    _principal: ReadPrincipal,
    after_sequence: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    site_id: str | None = None,
    severity: EventSeverity | None = None,
) -> EventPage:
    return repository.list_events(
        tenant.organization_id,
        after_sequence=after_sequence,
        limit=limit,
        site_id=site_id,
        severity=severity,
    )


@router.get("/incidents", response_model=list[IncidentRead])
def list_incidents(
    repository: RepositoryDependency,
    tenant: TenantDependency,
    _principal: ReadPrincipal,
    incident_status: IncidentStatus | None = None,
    site_id: str | None = None,
) -> list[IncidentRead]:
    return repository.list_incidents(
        tenant.organization_id,
        incident_status=incident_status,
        site_id=site_id,
    )


@router.post("/incidents", response_model=IncidentRead, status_code=status.HTTP_201_CREATED)
def create_incident(
    payload: IncidentCreate,
    repository: RepositoryDependency,
    tenant: TenantDependency,
    principal: IncidentPrincipal,
) -> IncidentRead:
    return repository.create_incident(
        tenant.organization_id,
        principal.subject,
        payload,
    )


@router.get("/incidents/{incident_id}", response_model=IncidentRead)
def get_incident(
    incident_id: str,
    repository: RepositoryDependency,
    tenant: TenantDependency,
    _principal: ReadPrincipal,
) -> IncidentRead:
    return repository.get_incident(tenant.organization_id, incident_id)


@router.patch("/incidents/{incident_id}", response_model=IncidentRead)
def update_incident(
    incident_id: str,
    payload: IncidentUpdate,
    repository: RepositoryDependency,
    tenant: TenantDependency,
    _principal: IncidentPrincipal,
) -> IncidentRead:
    return repository.update_incident(tenant.organization_id, incident_id, payload)


@router.get("/device-health", response_model=list[DeviceHealthRead])
def list_device_health(
    repository: RepositoryDependency,
    tenant: TenantDependency,
    _principal: ReadPrincipal,
    site_id: str | None = None,
    latest_only: bool = True,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> list[DeviceHealthRead]:
    return repository.list_device_health(
        tenant.organization_id,
        site_id=site_id,
        latest_only=latest_only,
        limit=limit,
    )


@router.post(
    "/device-health",
    response_model=DeviceHealthRead,
    status_code=status.HTTP_201_CREATED,
)
def report_device_health(
    payload: DeviceHealthCreate,
    repository: RepositoryDependency,
    tenant: TenantDependency,
    _principal: HealthPrincipal,
) -> DeviceHealthRead:
    return repository.report_device_health(tenant.organization_id, payload)
