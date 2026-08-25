"""Explicit demo authentication and role authorization dependencies."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated

from fastapi import Header, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from control_api.errors import ForbiddenError
from control_api.schemas import DemoIdentity, Role


class Permission(StrEnum):
    READ = "read"
    PLATFORM_MANAGE = "platform_manage"
    TOPOLOGY_WRITE = "topology_write"
    ACCESS_REQUEST_WRITE = "access_request_write"
    ACCESS_DECIDE = "access_decide"
    GRANT_WRITE = "grant_write"
    PASSAGE_INGEST = "passage_ingest"
    RECOGNITION_INGEST = "recognition_ingest"
    AUTHORIZATION_DECIDE = "authorization_decide"
    INCIDENT_WRITE = "incident_write"
    HEALTH_REPORT = "health_report"
    AGENT_RUN = "agent_run"
    AGENT_APPROVE = "agent_approve"


@dataclass(frozen=True, slots=True)
class Principal:
    """Authenticated demo actor."""

    subject: str
    display_name: str
    organization_id: str
    roles: frozenset[Role]


@dataclass(frozen=True, slots=True)
class TenantContext:
    """Organization scope selected after authenticating the actor."""

    organization_id: str
    principal: Principal


_DEMO_PRINCIPALS: dict[str, Principal] = {
    "demo-platform": Principal(
        "platform-admin",
        "Demo Platform Administrator",
        "org-atlas",
        frozenset({Role.PLATFORM_ADMIN}),
    ),
    "demo-admin": Principal(
        "admin-amal",
        "Amal - Access Administrator",
        "org-atlas",
        frozenset({Role.ORG_ADMIN}),
    ),
    "demo-operator": Principal(
        "operator-omar",
        "Omar - Security Operator",
        "org-atlas",
        frozenset({Role.SECURITY_OPERATOR}),
    ),
    "demo-host": Principal(
        "host-salma",
        "Salma - Campus Host",
        "org-atlas",
        frozenset({Role.HOST}),
    ),
    "demo-viewer": Principal(
        "viewer-demo",
        "Demo Read-only Analyst",
        "org-atlas",
        frozenset({Role.VIEWER}),
    ),
    "demo-edge": Principal(
        "edge-agent-demo",
        "Demo Site Edge Agent",
        "org-atlas",
        frozenset({Role.EDGE_AGENT}),
    ),
    "demo-rif-admin": Principal(
        "admin-rif",
        "Rif Demo Administrator",
        "org-rif",
        frozenset({Role.ORG_ADMIN}),
    ),
}

_ROLE_PERMISSIONS: dict[Role, frozenset[Permission]] = {
    Role.PLATFORM_ADMIN: frozenset(Permission),
    Role.ORG_ADMIN: frozenset(
        {
            Permission.READ,
            Permission.TOPOLOGY_WRITE,
            Permission.ACCESS_REQUEST_WRITE,
            Permission.ACCESS_DECIDE,
            Permission.GRANT_WRITE,
            Permission.PASSAGE_INGEST,
            Permission.RECOGNITION_INGEST,
            Permission.AUTHORIZATION_DECIDE,
            Permission.INCIDENT_WRITE,
            Permission.HEALTH_REPORT,
            Permission.AGENT_RUN,
            Permission.AGENT_APPROVE,
        }
    ),
    Role.SECURITY_OPERATOR: frozenset(
        {
            Permission.READ,
            Permission.AUTHORIZATION_DECIDE,
            Permission.INCIDENT_WRITE,
            Permission.HEALTH_REPORT,
            Permission.AGENT_RUN,
            Permission.AGENT_APPROVE,
        }
    ),
    Role.HOST: frozenset({Permission.READ, Permission.ACCESS_REQUEST_WRITE}),
    Role.VIEWER: frozenset({Permission.READ}),
    Role.EDGE_AGENT: frozenset(
        {
            Permission.PASSAGE_INGEST,
            Permission.RECOGNITION_INGEST,
            Permission.HEALTH_REPORT,
        }
    ),
}

_bearer = HTTPBearer(
    auto_error=False,
    scheme_name="DemoBearer",
    description="Use a token returned by GET /api/v1/demo-identities.",
)


def demo_identities() -> list[DemoIdentity]:
    """Describe intentional demo credentials for the UI role switcher."""

    use_cases = {
        "demo-platform": "Switch organizations and manage the platform",
        "demo-admin": "Manage Atlas topology, access, and approval-gated agent runs",
        "demo-operator": "Review passages, resolve incidents, and supervise agent runs",
        "demo-host": "Submit and track visitor access requests",
        "demo-viewer": "Inspect dashboards without changing state",
        "demo-edge": "Report device health and recognition observations",
        "demo-rif-admin": "Demonstrate organization isolation",
    }
    return [
        DemoIdentity(
            token=token,
            display_name=principal.display_name,
            organization_id=principal.organization_id,
            roles=sorted(principal.roles, key=str),
            use_case=use_cases[token],
        )
        for token, principal in _DEMO_PRINCIPALS.items()
    ]


def authenticate(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Security(_bearer)],
) -> Principal:
    """Resolve a bearer credential without pretending demo tokens are production auth."""

    principal = _DEMO_PRINCIPALS.get(credentials.credentials) if credentials else None
    if principal is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="A valid demo bearer token is required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return principal


def require_permission(permission: Permission) -> Callable[[Principal], Principal]:
    """Return a FastAPI dependency that checks a named capability."""

    def dependency(principal: Annotated[Principal, Security(authenticate)]) -> Principal:
        permissions = frozenset().union(*(_ROLE_PERMISSIONS[role] for role in principal.roles))
        if permission not in permissions:
            raise ForbiddenError(f"Role does not grant '{permission}'")
        return principal

    return dependency


def tenant_context(
    principal: Annotated[Principal, Security(authenticate)],
    organization_header: Annotated[
        str | None,
        Header(alias="X-Organization-ID", max_length=80),
    ] = None,
) -> TenantContext:
    """Prevent clients from escaping their authenticated organization scope."""

    selected = organization_header or principal.organization_id
    if selected != principal.organization_id and Role.PLATFORM_ADMIN not in principal.roles:
        raise ForbiddenError("Only a platform administrator can switch organizations")
    return TenantContext(selected, principal)
