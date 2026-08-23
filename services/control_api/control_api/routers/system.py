"""Service health, API metadata, and demo-session routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from control_api.auth import demo_identities
from control_api.database import SCHEMA_VERSION
from control_api.dependencies import ReadPrincipal, RepositoryDependency
from control_api.schemas import DemoIdentity, HealthRead, PrincipalRead

router = APIRouter(tags=["system"])


@router.get("/health/live", response_model=HealthRead, operation_id="liveness")
def liveness() -> HealthRead:
    """Report process liveness without depending on downstream resources."""

    return HealthRead(status="ok", service="campus-control-api")


@router.get("/health/ready", response_model=HealthRead, operation_id="readiness")
def readiness(repository: RepositoryDependency) -> HealthRead:
    """Report readiness only after the SQLite schema is available."""

    if not repository.database.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database not ready"
        )
    return HealthRead(
        status="ready",
        service="campus-control-api",
        schema_version=SCHEMA_VERSION,
    )


@router.get(
    "/api/v1/demo-identities",
    response_model=list[DemoIdentity],
    summary="List intentional demo-only bearer identities",
)
def list_demo_identities() -> list[DemoIdentity]:
    """Support a role switcher while making the demo-only boundary explicit."""

    return demo_identities()


@router.get("/api/v1/session", response_model=PrincipalRead)
def session(principal: ReadPrincipal) -> PrincipalRead:
    """Return the authenticated role and organization used by the UI shell."""

    return PrincipalRead(
        subject=principal.subject,
        display_name=principal.display_name,
        organization_id=principal.organization_id,
        roles=sorted(principal.roles, key=str),
    )


@router.get("/api/v1/meta", response_model=dict[str, object])
def metadata(request: Request) -> dict[str, object]:
    """Expose generic product metadata without institution-specific branding."""

    return {
        "service": request.app.title,
        "api_version": "v1",
        "product_name": "Campus Access Control",
        "demo_data": True,
        "evidence_policy": "All seeded evidence is synthetic or composite-labelled.",
    }
