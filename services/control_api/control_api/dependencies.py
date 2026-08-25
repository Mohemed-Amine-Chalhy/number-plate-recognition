"""Shared FastAPI dependency aliases."""

from __future__ import annotations

from typing import Annotated, cast

from fastapi import Depends, Request

from control_api.agentic import AgentWorkflowService
from control_api.auth import (
    Permission,
    Principal,
    TenantContext,
    require_permission,
    tenant_context,
)
from control_api.repository import Repository


def repository_from_request(request: Request) -> Repository:
    """Return the repository installed by the application factory."""

    return cast(Repository, request.app.state.repository)


def agent_service_from_request(request: Request) -> AgentWorkflowService:
    """Return the bounded agent workflow installed by the application factory."""

    return cast(AgentWorkflowService, request.app.state.agent_service)


RepositoryDependency = Annotated[Repository, Depends(repository_from_request)]
AgentServiceDependency = Annotated[AgentWorkflowService, Depends(agent_service_from_request)]
TenantDependency = Annotated[TenantContext, Depends(tenant_context)]
ReadPrincipal = Annotated[Principal, Depends(require_permission(Permission.READ))]
PlatformPrincipal = Annotated[Principal, Depends(require_permission(Permission.PLATFORM_MANAGE))]
TopologyPrincipal = Annotated[Principal, Depends(require_permission(Permission.TOPOLOGY_WRITE))]
AccessRequestPrincipal = Annotated[
    Principal, Depends(require_permission(Permission.ACCESS_REQUEST_WRITE))
]
AccessDecisionPrincipal = Annotated[
    Principal, Depends(require_permission(Permission.ACCESS_DECIDE))
]
GrantPrincipal = Annotated[Principal, Depends(require_permission(Permission.GRANT_WRITE))]
PassagePrincipal = Annotated[Principal, Depends(require_permission(Permission.PASSAGE_INGEST))]
RecognitionPrincipal = Annotated[
    Principal, Depends(require_permission(Permission.RECOGNITION_INGEST))
]
AuthorizationPrincipal = Annotated[
    Principal, Depends(require_permission(Permission.AUTHORIZATION_DECIDE))
]
IncidentPrincipal = Annotated[Principal, Depends(require_permission(Permission.INCIDENT_WRITE))]
HealthPrincipal = Annotated[Principal, Depends(require_permission(Permission.HEALTH_REPORT))]
AgentRunPrincipal = Annotated[Principal, Depends(require_permission(Permission.AGENT_RUN))]
AgentApprovalPrincipal = Annotated[Principal, Depends(require_permission(Permission.AGENT_APPROVE))]
