"""Organization, site, gate, camera, and dashboard routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, Response, status

from control_api.dependencies import (
    PlatformPrincipal,
    ReadPrincipal,
    RepositoryDependency,
    TenantDependency,
    TopologyPrincipal,
)
from control_api.schemas import (
    CameraCreate,
    CameraRead,
    CameraUpdate,
    DashboardCounts,
    DashboardRead,
    GateCreate,
    GateRead,
    GateUpdate,
    IncidentStatus,
    OrganizationCreate,
    OrganizationRead,
    OrganizationStatus,
    OrganizationUpdate,
    Role,
    SiteCreate,
    SiteRead,
    SiteUpdate,
)

router = APIRouter(prefix="/api/v1", tags=["campus topology"])


@router.get("/organizations", response_model=list[OrganizationRead])
def list_organizations(
    repository: RepositoryDependency,
    tenant: TenantDependency,
    principal: ReadPrincipal,
) -> list[OrganizationRead]:
    return repository.list_organizations(
        tenant.organization_id,
        platform_admin=Role.PLATFORM_ADMIN in principal.roles,
    )


@router.post(
    "/organizations",
    response_model=OrganizationRead,
    status_code=status.HTTP_201_CREATED,
)
def create_organization(
    payload: OrganizationCreate,
    repository: RepositoryDependency,
    _principal: PlatformPrincipal,
) -> OrganizationRead:
    return repository.create_organization(payload)


@router.get("/organizations/{organization_id}", response_model=OrganizationRead)
def get_organization(
    organization_id: str,
    repository: RepositoryDependency,
    tenant: TenantDependency,
    _principal: ReadPrincipal,
) -> OrganizationRead:
    # The selected tenant is authoritative. Returning 404 avoids leaking another tenant.
    if organization_id != tenant.organization_id:
        from control_api.errors import ResourceNotFoundError

        raise ResourceNotFoundError("Organization was not found")
    return repository.get_organization(organization_id)


@router.patch("/organizations/{organization_id}", response_model=OrganizationRead)
def update_organization(
    organization_id: str,
    payload: OrganizationUpdate,
    repository: RepositoryDependency,
    _principal: PlatformPrincipal,
) -> OrganizationRead:
    return repository.update_organization(organization_id, payload)


@router.delete("/organizations/{organization_id}", status_code=status.HTTP_204_NO_CONTENT)
def archive_organization(
    organization_id: str,
    repository: RepositoryDependency,
    _principal: PlatformPrincipal,
) -> Response:
    repository.update_organization(
        organization_id,
        OrganizationUpdate(status=OrganizationStatus.ARCHIVED),
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/sites", response_model=list[SiteRead])
def list_sites(
    repository: RepositoryDependency,
    tenant: TenantDependency,
    _principal: ReadPrincipal,
    include_archived: bool = False,
) -> list[SiteRead]:
    return repository.list_sites(tenant.organization_id, include_archived=include_archived)


@router.post("/sites", response_model=SiteRead, status_code=status.HTTP_201_CREATED)
def create_site(
    payload: SiteCreate,
    repository: RepositoryDependency,
    tenant: TenantDependency,
    _principal: TopologyPrincipal,
) -> SiteRead:
    return repository.create_site(tenant.organization_id, payload)


@router.get("/sites/{site_id}", response_model=SiteRead)
def get_site(
    site_id: str,
    repository: RepositoryDependency,
    tenant: TenantDependency,
    _principal: ReadPrincipal,
) -> SiteRead:
    return repository.get_site(tenant.organization_id, site_id)


@router.patch("/sites/{site_id}", response_model=SiteRead)
def update_site(
    site_id: str,
    payload: SiteUpdate,
    repository: RepositoryDependency,
    tenant: TenantDependency,
    _principal: TopologyPrincipal,
) -> SiteRead:
    return repository.update_site(tenant.organization_id, site_id, payload)


@router.delete("/sites/{site_id}", status_code=status.HTTP_204_NO_CONTENT)
def archive_site(
    site_id: str,
    repository: RepositoryDependency,
    tenant: TenantDependency,
    _principal: TopologyPrincipal,
) -> Response:
    repository.archive_site(tenant.organization_id, site_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/gates", response_model=list[GateRead])
def list_gates(
    repository: RepositoryDependency,
    tenant: TenantDependency,
    _principal: ReadPrincipal,
    site_id: str | None = None,
    include_disabled: bool = False,
) -> list[GateRead]:
    return repository.list_gates(
        tenant.organization_id,
        site_id=site_id,
        include_disabled=include_disabled,
    )


@router.post("/gates", response_model=GateRead, status_code=status.HTTP_201_CREATED)
def create_gate(
    payload: GateCreate,
    repository: RepositoryDependency,
    tenant: TenantDependency,
    _principal: TopologyPrincipal,
) -> GateRead:
    return repository.create_gate(tenant.organization_id, payload)


@router.get("/gates/{gate_id}", response_model=GateRead)
def get_gate(
    gate_id: str,
    repository: RepositoryDependency,
    tenant: TenantDependency,
    _principal: ReadPrincipal,
) -> GateRead:
    return repository.get_gate(tenant.organization_id, gate_id)


@router.patch("/gates/{gate_id}", response_model=GateRead)
def update_gate(
    gate_id: str,
    payload: GateUpdate,
    repository: RepositoryDependency,
    tenant: TenantDependency,
    _principal: TopologyPrincipal,
) -> GateRead:
    return repository.update_gate(tenant.organization_id, gate_id, payload)


@router.delete("/gates/{gate_id}", status_code=status.HTTP_204_NO_CONTENT)
def disable_gate(
    gate_id: str,
    repository: RepositoryDependency,
    tenant: TenantDependency,
    _principal: TopologyPrincipal,
) -> Response:
    repository.disable_gate(tenant.organization_id, gate_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/cameras", response_model=list[CameraRead])
def list_cameras(
    repository: RepositoryDependency,
    tenant: TenantDependency,
    _principal: ReadPrincipal,
    site_id: str | None = None,
    gate_id: str | None = None,
    include_disabled: bool = False,
) -> list[CameraRead]:
    return repository.list_cameras(
        tenant.organization_id,
        site_id=site_id,
        gate_id=gate_id,
        include_disabled=include_disabled,
    )


@router.post("/cameras", response_model=CameraRead, status_code=status.HTTP_201_CREATED)
def create_camera(
    payload: CameraCreate,
    repository: RepositoryDependency,
    tenant: TenantDependency,
    _principal: TopologyPrincipal,
) -> CameraRead:
    return repository.create_camera(tenant.organization_id, payload)


@router.get("/cameras/{camera_id}", response_model=CameraRead)
def get_camera(
    camera_id: str,
    repository: RepositoryDependency,
    tenant: TenantDependency,
    _principal: ReadPrincipal,
) -> CameraRead:
    return repository.get_camera(tenant.organization_id, camera_id)


@router.patch("/cameras/{camera_id}", response_model=CameraRead)
def update_camera(
    camera_id: str,
    payload: CameraUpdate,
    repository: RepositoryDependency,
    tenant: TenantDependency,
    _principal: TopologyPrincipal,
) -> CameraRead:
    return repository.update_camera(tenant.organization_id, camera_id, payload)


@router.delete("/cameras/{camera_id}", status_code=status.HTTP_204_NO_CONTENT)
def disable_camera(
    camera_id: str,
    repository: RepositoryDependency,
    tenant: TenantDependency,
    _principal: TopologyPrincipal,
) -> Response:
    repository.disable_camera(tenant.organization_id, camera_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/dashboard", response_model=DashboardRead, tags=["dashboard"])
def dashboard(
    repository: RepositoryDependency,
    tenant: TenantDependency,
    _principal: ReadPrincipal,
    recent_event_limit: Annotated[int, Query(ge=1, le=50)] = 12,
) -> DashboardRead:
    counts = repository.counts(tenant.organization_id)
    return DashboardRead(
        counts=DashboardCounts.model_validate(counts),
        gates=repository.list_gates(tenant.organization_id),
        recent_events=repository.recent_events(
            tenant.organization_id,
            limit=recent_event_limit,
        ),
        open_incidents=repository.list_incidents(
            tenant.organization_id,
            incident_status=IncidentStatus.OPEN,
        )
        + repository.list_incidents(
            tenant.organization_id,
            incident_status=IncidentStatus.INVESTIGATING,
        ),
        device_health=repository.list_device_health(
            tenant.organization_id,
            latest_only=True,
            limit=50,
        ),
    )
