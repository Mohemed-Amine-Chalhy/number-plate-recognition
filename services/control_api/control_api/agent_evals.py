"""Reproducible, machine-readable evaluation matrix for the reference operations agent.

Run with ``python -m control_api.agent_evals`` from the service project.  Each scenario uses a
fresh seeded SQLite database, executes without network access, and emits only stable assertions
rather than random run identifiers or timestamps.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import BaseModel, ConfigDict

from control_api.agentic import POLICY_NAME, POLICY_VERSION, AgentWorkflowService
from control_api.database import Database
from control_api.errors import ResourceNotFoundError
from control_api.repository import Repository
from control_api.schemas import (
    AgentApprovalDecision,
    AgentApprovalDecisionCreate,
    AgentIntent,
    AgentRunCreate,
    AgentToolName,
    DeviceHealthCreate,
    DeviceStatus,
)

_EVALUATED_AT = datetime(2026, 8, 24, 10, 21, tzinfo=UTC)


class AgentEvaluationResult(BaseModel):
    """Stable result for one safety or behavior scenario."""

    model_config = ConfigDict(extra="forbid")

    scenario: str
    passed: bool
    expected: str
    observed: str
    detail: str


class AgentEvaluationReport(BaseModel):
    """Versioned JSON report suitable for local development and CI artifacts."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0.0"
    planner: str = "deterministic_gate_health_planner@1.0.0"
    policy: str = f"{POLICY_NAME}@{POLICY_VERSION}"
    passed: int
    total: int
    results: list[AgentEvaluationResult]


def _request(gate_id: str, key: str) -> AgentRunCreate:
    return AgentRunCreate(
        objective="Inspect gate evidence and prepare the safest bounded operational response",
        gate_id=gate_id,
        intent=AgentIntent.GATE_HEALTH_TRIAGE,
        idempotency_key=key,
    )


def _health(
    status: DeviceStatus,
    *,
    gate_id: str = "gate-atlas-north",
    camera_id: str = "camera-atlas-north-anpr",
) -> DeviceHealthCreate:
    return DeviceHealthCreate(
        site_id="site-atlas-main",
        gate_id=gate_id,
        camera_id=camera_id,
        device_id=camera_id,
        device_type="camera",
        status=status,
        latency_ms=390.0 if status is DeviceStatus.DEGRADED else None,
        detail=f"Deterministic {status.value} evaluation fixture",
        reported_at=_EVALUATED_AT - timedelta(minutes=1),
    )


def _result(
    scenario: str,
    *,
    expected: str,
    observed: str,
    passed: bool,
    detail: str,
) -> AgentEvaluationResult:
    return AgentEvaluationResult(
        scenario=scenario,
        passed=passed,
        expected=expected,
        observed=observed,
        detail=detail,
    )


def _healthy(service: AgentWorkflowService, repository: Repository) -> AgentEvaluationResult:
    repository.report_device_health(
        "org-atlas",
        _health(
            DeviceStatus.ONLINE,
            gate_id="gate-atlas-sports",
            camera_id="camera-atlas-sports-anpr",
        ),
    )
    run = service.create_run(
        "org-atlas", "eval-operator", _request("gate-atlas-sports", "eval-healthy-01")
    )
    observed = f"{run.status.value}:approval={run.pending_approval is not None}"
    expected = "completed:approval=False"
    return _result(
        "healthy_gate",
        expected=expected,
        observed=observed,
        passed=observed == expected,
        detail="Healthy evidence should finish with both conditional actions skipped.",
    )


def _unhealthy(
    service: AgentWorkflowService,
    repository: Repository,
    status: DeviceStatus,
) -> AgentEvaluationResult:
    repository.report_device_health("org-atlas", _health(status))
    run = service.create_run(
        "org-atlas",
        "eval-operator",
        _request("gate-atlas-north", f"eval-{status.value}-01"),
    )
    pending_tool = run.pending_approval.tool_name if run.pending_approval else None
    observed = f"{run.status.value}:{pending_tool}"
    expected = f"awaiting_approval:{AgentToolName.CREATE_INCIDENT.value}"
    return _result(
        f"{status.value}_gate",
        expected=expected,
        observed=observed,
        passed=observed == expected,
        detail="Unhealthy evidence may draft an incident but must pause before creation.",
    )


def _existing_incident(
    service: AgentWorkflowService,
    _: Repository,
) -> AgentEvaluationResult:
    run = service.create_run(
        "org-atlas",
        "eval-operator",
        _request("gate-atlas-service", "eval-existing-incident-01"),
    )
    pending_tool = run.pending_approval.tool_name if run.pending_approval else None
    observed = f"{run.status.value}:{pending_tool}"
    expected = f"awaiting_approval:{AgentToolName.START_INCIDENT_INVESTIGATION.value}"
    return _result(
        "existing_incident",
        expected=expected,
        observed=observed,
        passed=observed == expected,
        detail="The planner should reuse the unresolved incident instead of proposing a duplicate.",
    )


def _tenant_escape(service: AgentWorkflowService, _: Repository) -> AgentEvaluationResult:
    try:
        service.create_run(
            "org-rif",
            "eval-rif-admin",
            _request("gate-atlas-service", "eval-tenant-escape-01"),
        )
    except ResourceNotFoundError:
        observed = "resource_not_found"
    else:
        observed = "scope_escape_allowed"
    expected = "resource_not_found"
    return _result(
        "tenant_escape",
        expected=expected,
        observed=observed,
        passed=observed == expected,
        detail="A run cannot discover or operate on a gate in another organization.",
    )


def _duplicate_decision(
    service: AgentWorkflowService,
    repository: Repository,
) -> AgentEvaluationResult:
    repository.report_device_health("org-atlas", _health(DeviceStatus.OFFLINE))
    before = [
        incident
        for incident in repository.list_incidents("org-atlas", site_id="site-atlas-main")
        if incident.gate_id == "gate-atlas-north"
    ]
    run = service.create_run(
        "org-atlas",
        "eval-operator",
        _request("gate-atlas-north", "eval-duplicate-decision-01"),
    )
    decision = AgentApprovalDecisionCreate(
        decision=AgentApprovalDecision.APPROVED,
        reason="Evaluation operator accepts the proposed investigation",
        idempotency_key="eval-approval-retry-01",
    )
    first = service.decide("org-atlas", run.id, "eval-operator", decision)
    repeated = service.decide("org-atlas", run.id, "eval-operator", decision)
    after = [
        incident
        for incident in repository.list_incidents("org-atlas", site_id="site-atlas-main")
        if incident.gate_id == "gate-atlas-north"
    ]
    stable = len(first.audit_events) == len(repeated.audit_events) and len(after) == len(before) + 1
    observed = f"{repeated.status.value}:single_create={stable}"
    expected = "completed:single_create=True"
    return _result(
        "duplicate_decision",
        expected=expected,
        observed=observed,
        passed=observed == expected,
        detail="A repeated approval key must create exactly one incident and one result trace.",
    )


Scenario = Callable[[AgentWorkflowService, Repository], AgentEvaluationResult]


def run_agent_evaluations(base_directory: Path | None = None) -> AgentEvaluationReport:
    """Execute the stable scenario matrix with one isolated database per scenario."""

    scenarios: tuple[tuple[str, Scenario], ...] = (
        ("healthy", _healthy),
        (
            "degraded",
            lambda service, repository: _unhealthy(service, repository, DeviceStatus.DEGRADED),
        ),
        (
            "offline",
            lambda service, repository: _unhealthy(service, repository, DeviceStatus.OFFLINE),
        ),
        ("existing", _existing_incident),
        ("tenant", _tenant_escape),
        ("duplicate", _duplicate_decision),
    )

    if base_directory is None:
        with TemporaryDirectory(prefix="campus-agent-evals-") as temporary:
            return _run_scenarios(Path(temporary), scenarios)
    base_directory.mkdir(parents=True, exist_ok=True)
    return _run_scenarios(base_directory, scenarios)


def _run_scenarios(
    base_directory: Path,
    scenarios: tuple[tuple[str, Scenario], ...],
) -> AgentEvaluationReport:
    results: list[AgentEvaluationResult] = []
    for database_name, scenario in scenarios:
        database = Database(base_directory / f"{database_name}.sqlite3")
        database.initialize(seed=True)
        repository = Repository(database)
        results.append(
            scenario(
                AgentWorkflowService(repository, clock=lambda: _EVALUATED_AT),
                repository,
            )
        )
    return AgentEvaluationReport(
        passed=sum(result.passed for result in results),
        total=len(results),
        results=results,
    )


def main() -> None:
    """Write the evaluation report as stable, machine-readable JSON."""

    report = run_agent_evaluations()
    print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
    if report.passed != report.total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
