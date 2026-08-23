"""Visitor/staff access request and grant workflows."""

from __future__ import annotations

from fastapi import APIRouter, Response, status

from control_api.dependencies import (
    AccessDecisionPrincipal,
    AccessRequestPrincipal,
    GrantPrincipal,
    ReadPrincipal,
    RepositoryDependency,
    TenantDependency,
)
from control_api.errors import ResourceNotFoundError
from control_api.schemas import (
    AccessGrantCreate,
    AccessGrantRead,
    AccessGrantStatus,
    AccessRequestCreate,
    AccessRequestDecision,
    AccessRequestDecisionResult,
    AccessRequestRead,
    AccessRequestStatus,
    AccessRequestUpdate,
    GrantRevoke,
    Role,
)

router = APIRouter(prefix="/api/v1", tags=["access"])


@router.get("/access-requests", response_model=list[AccessRequestRead])
def list_access_requests(
    repository: RepositoryDependency,
    tenant: TenantDependency,
    principal: ReadPrincipal,
    request_status: AccessRequestStatus | None = None,
    site_id: str | None = None,
) -> list[AccessRequestRead]:
    requested_by = principal.subject if principal.roles == frozenset({Role.HOST}) else None
    return repository.list_access_requests(
        tenant.organization_id,
        request_status=request_status,
        site_id=site_id,
        requested_by=requested_by,
    )


@router.post(
    "/access-requests",
    response_model=AccessRequestRead,
    status_code=status.HTTP_201_CREATED,
)
def create_access_request(
    payload: AccessRequestCreate,
    repository: RepositoryDependency,
    tenant: TenantDependency,
    principal: AccessRequestPrincipal,
) -> AccessRequestRead:
    return repository.create_access_request(
        tenant.organization_id,
        principal.subject,
        payload,
    )


@router.get("/access-requests/{request_id}", response_model=AccessRequestRead)
def get_access_request(
    request_id: str,
    repository: RepositoryDependency,
    tenant: TenantDependency,
    principal: ReadPrincipal,
) -> AccessRequestRead:
    access_request = repository.get_access_request(tenant.organization_id, request_id)
    if (
        principal.roles == frozenset({Role.HOST})
        and access_request.requested_by != principal.subject
    ):
        raise ResourceNotFoundError("Access request was not found")
    return access_request


@router.patch("/access-requests/{request_id}", response_model=AccessRequestRead)
def update_access_request(
    request_id: str,
    payload: AccessRequestUpdate,
    repository: RepositoryDependency,
    tenant: TenantDependency,
    principal: AccessRequestPrincipal,
) -> AccessRequestRead:
    current = repository.get_access_request(tenant.organization_id, request_id)
    if principal.roles == frozenset({Role.HOST}) and current.requested_by != principal.subject:
        raise ResourceNotFoundError("Access request was not found")
    return repository.update_access_request(tenant.organization_id, request_id, payload)


@router.delete("/access-requests/{request_id}", status_code=status.HTTP_204_NO_CONTENT)
def cancel_access_request(
    request_id: str,
    repository: RepositoryDependency,
    tenant: TenantDependency,
    principal: AccessRequestPrincipal,
) -> Response:
    current = repository.get_access_request(tenant.organization_id, request_id)
    if principal.roles == frozenset({Role.HOST}) and current.requested_by != principal.subject:
        raise ResourceNotFoundError("Access request was not found")
    repository.cancel_access_request(tenant.organization_id, request_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/access-requests/{request_id}/decision",
    response_model=AccessRequestDecisionResult,
)
def decide_access_request(
    request_id: str,
    payload: AccessRequestDecision,
    repository: RepositoryDependency,
    tenant: TenantDependency,
    principal: AccessDecisionPrincipal,
) -> AccessRequestDecisionResult:
    access_request, grant = repository.decide_access_request(
        tenant.organization_id,
        request_id,
        principal.subject,
        payload,
    )
    return AccessRequestDecisionResult(request=access_request, grant=grant)


@router.get("/access-grants", response_model=list[AccessGrantRead])
def list_access_grants(
    repository: RepositoryDependency,
    tenant: TenantDependency,
    _principal: ReadPrincipal,
    grant_status: AccessGrantStatus | None = None,
    plate_text: str | None = None,
) -> list[AccessGrantRead]:
    return repository.list_access_grants(
        tenant.organization_id,
        grant_status=grant_status,
        plate_text=plate_text,
    )


@router.post(
    "/access-grants",
    response_model=AccessGrantRead,
    status_code=status.HTTP_201_CREATED,
)
def create_access_grant(
    payload: AccessGrantCreate,
    repository: RepositoryDependency,
    tenant: TenantDependency,
    principal: GrantPrincipal,
) -> AccessGrantRead:
    return repository.create_access_grant(
        tenant.organization_id,
        principal.subject,
        payload,
    )


@router.get("/access-grants/{grant_id}", response_model=AccessGrantRead)
def get_access_grant(
    grant_id: str,
    repository: RepositoryDependency,
    tenant: TenantDependency,
    _principal: ReadPrincipal,
) -> AccessGrantRead:
    return repository.get_access_grant(tenant.organization_id, grant_id)


@router.post("/access-grants/{grant_id}/revoke", response_model=AccessGrantRead)
def revoke_access_grant(
    grant_id: str,
    payload: GrantRevoke,
    repository: RepositoryDependency,
    tenant: TenantDependency,
    _principal: GrantPrincipal,
) -> AccessGrantRead:
    return repository.revoke_access_grant(
        tenant.organization_id,
        grant_id,
        reason=payload.reason,
    )
