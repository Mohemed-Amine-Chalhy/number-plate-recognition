"""Inspectable, approval-gated operations-agent routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, status

from control_api.dependencies import (
    AgentApprovalPrincipal,
    AgentRunPrincipal,
    AgentServiceDependency,
    ReadPrincipal,
    TenantDependency,
)
from control_api.schemas import AgentApprovalDecisionCreate, AgentRunCreate, AgentRunRead

router = APIRouter(prefix="/api/v1/agent", tags=["agent operations"])


@router.post(
    "/runs",
    response_model=AgentRunRead,
    status_code=status.HTTP_201_CREATED,
    summary="Run deterministic gate triage until completion or a human approval boundary",
)
def create_agent_run(
    payload: AgentRunCreate,
    agent_service: AgentServiceDependency,
    tenant: TenantDependency,
    principal: AgentRunPrincipal,
) -> AgentRunRead:
    return agent_service.create_run(tenant.organization_id, principal.subject, payload)


@router.get("/runs", response_model=list[AgentRunRead])
def list_agent_runs(
    agent_service: AgentServiceDependency,
    tenant: TenantDependency,
    _principal: ReadPrincipal,
    gate_id: str | None = None,
    limit: Annotated[int, Query(ge=1, le=50)] = 20,
) -> list[AgentRunRead]:
    return agent_service.list_runs(tenant.organization_id, gate_id=gate_id, limit=limit)


@router.get("/runs/{run_id}", response_model=AgentRunRead)
def get_agent_run(
    run_id: str,
    agent_service: AgentServiceDependency,
    tenant: TenantDependency,
    _principal: ReadPrincipal,
) -> AgentRunRead:
    return agent_service.get_run(tenant.organization_id, run_id)


@router.post(
    "/runs/{run_id}/decisions",
    response_model=AgentRunRead,
    summary="Approve or reject the one pending consequential tool invocation",
)
def decide_agent_run(
    run_id: str,
    payload: AgentApprovalDecisionCreate,
    agent_service: AgentServiceDependency,
    tenant: TenantDependency,
    principal: AgentApprovalPrincipal,
) -> AgentRunRead:
    return agent_service.decide(
        tenant.organization_id,
        run_id,
        principal.subject,
        payload,
    )
