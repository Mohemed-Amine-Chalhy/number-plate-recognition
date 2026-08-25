"""Bounded, inspectable operations-agent workflow.

The reference planner is intentionally deterministic and offline.  The ``AgentPlanner``
protocol is the provider seam: a future model-backed planner can implement it without changing
the tool registry, policy checks, persistence, or approval boundary in this module.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol, cast
from uuid import uuid4

from control_api.errors import (
    ConflictError,
    ControlApiError,
    InvalidStateError,
    ResourceNotFoundError,
)
from control_api.repository import Repository
from control_api.schemas import (
    AgentApprovalDecision,
    AgentApprovalDecisionCreate,
    AgentApprovalRead,
    AgentAuditEventRead,
    AgentIntent,
    AgentPendingApproval,
    AgentPlan,
    AgentPlannedStep,
    AgentPolicyCheck,
    AgentPolicyOutcome,
    AgentRunCreate,
    AgentRunRead,
    AgentRunStatus,
    AgentStepRead,
    AgentStepStatus,
    AgentToolName,
    AgentToolRisk,
    AgentTraceMetadata,
    GateRead,
    IncidentCreate,
    IncidentSeverity,
    IncidentStatus,
    IncidentUpdate,
)

POLICY_NAME = "campus_operations_guardrails"
POLICY_VERSION = "1.0.0"
HEALTH_MAX_AGE = timedelta(minutes=5)
HEALTH_FUTURE_TOLERANCE = timedelta(minutes=1)
LOGGER = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex}"


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """One tool in the closed runtime registry."""

    name: AgentToolName
    risk: AgentToolRisk
    description: str


TOOL_REGISTRY: dict[AgentToolName, ToolDefinition] = {
    AgentToolName.GET_GATE: ToolDefinition(
        AgentToolName.GET_GATE,
        AgentToolRisk.READ_ONLY,
        "Read the tenant-scoped gate record.",
    ),
    AgentToolName.GET_LATEST_DEVICE_HEALTH: ToolDefinition(
        AgentToolName.GET_LATEST_DEVICE_HEALTH,
        AgentToolRisk.READ_ONLY,
        "Read current health reports for devices attached to the selected gate.",
    ),
    AgentToolName.LIST_OPEN_GATE_INCIDENTS: ToolDefinition(
        AgentToolName.LIST_OPEN_GATE_INCIDENTS,
        AgentToolRisk.READ_ONLY,
        "Read unresolved incidents attached to the selected gate.",
    ),
    AgentToolName.START_INCIDENT_INVESTIGATION: ToolDefinition(
        AgentToolName.START_INCIDENT_INVESTIGATION,
        AgentToolRisk.CONSEQUENTIAL,
        "Move an existing incident into investigation and assign the approving operator.",
    ),
    AgentToolName.CREATE_INCIDENT: ToolDefinition(
        AgentToolName.CREATE_INCIDENT,
        AgentToolRisk.CONSEQUENTIAL,
        "Create a gate-scoped incident from the inspected evidence.",
    ),
}
ALLOWED_AGENT_TOOLS = frozenset(TOOL_REGISTRY)


class AgentPlanner(Protocol):
    """Replaceable planner/provider contract; orchestration remains policy controlled."""

    name: str
    version: str

    def plan(self, request: AgentRunCreate, gate: GateRead) -> AgentPlan:
        """Return a typed plan containing only registered tool names."""


class DeterministicGateHealthPlanner:
    """Offline reference planner with an auditable, versioned plan."""

    name = "deterministic_gate_health_planner"
    version = "1.0.0"

    def plan(self, request: AgentRunCreate, gate: GateRead) -> AgentPlan:
        if request.intent is not AgentIntent.GATE_HEALTH_TRIAGE:
            raise InvalidStateError(f"Unsupported agent intent: {request.intent}")
        return AgentPlan(
            summary=(
                f"Inspect {gate.name}, compare device health with unresolved incidents, "
                "and prepare at most one approval-gated response."
            ),
            steps=[
                AgentPlannedStep(
                    sequence=1,
                    tool_name=AgentToolName.GET_GATE,
                    risk=AgentToolRisk.READ_ONLY,
                    rationale="Ground the run in the selected organization's current gate record.",
                ),
                AgentPlannedStep(
                    sequence=2,
                    tool_name=AgentToolName.GET_LATEST_DEVICE_HEALTH,
                    risk=AgentToolRisk.READ_ONLY,
                    rationale="Collect current gate-device evidence before proposing an action.",
                ),
                AgentPlannedStep(
                    sequence=3,
                    tool_name=AgentToolName.LIST_OPEN_GATE_INCIDENTS,
                    risk=AgentToolRisk.READ_ONLY,
                    rationale="Avoid creating a duplicate incident for an active investigation.",
                ),
                AgentPlannedStep(
                    sequence=4,
                    tool_name=AgentToolName.START_INCIDENT_INVESTIGATION,
                    risk=AgentToolRisk.CONSEQUENTIAL,
                    rationale="Use an existing unresolved incident when one already covers the gate.",
                ),
                AgentPlannedStep(
                    sequence=5,
                    tool_name=AgentToolName.CREATE_INCIDENT,
                    risk=AgentToolRisk.CONSEQUENTIAL,
                    rationale="Create one incident only when evidence needs action and none exists.",
                ),
            ],
        )


class AgentWorkflowService:
    """Plan, execute, pause, and resume a durable gate-operations workflow."""

    def __init__(
        self,
        repository: Repository,
        planner: AgentPlanner | None = None,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.repository = repository
        self.planner = planner or DeterministicGateHealthPlanner()
        self.clock = clock or _utc_now

    def create_run(
        self,
        organization_id: str,
        actor_id: str,
        request: AgentRunCreate,
    ) -> AgentRunRead:
        gate = self.repository.get_gate(organization_id, request.gate_id)
        existing = self._find_idempotent_run(
            organization_id,
            actor_id,
            request.idempotency_key,
        )
        if existing is not None:
            self._assert_run_binding(existing, organization_id, actor_id, request)
            existing_id = str(existing["id"])
            return self.get_run(organization_id, existing_id)

        plan = self.planner.plan(request, gate)
        self._validate_plan(plan)
        run_id = _new_id("agent-run")
        trace_id = _new_id("trace")
        correlation_id = _new_id("corr")
        timestamp = _now()
        try:
            with self.repository.database.transaction() as connection:
                connection.execute(
                    "INSERT INTO agent_runs "
                    "(id, organization_id, site_id, gate_id, objective, intent, status, "
                    "created_by, idempotency_key, trace_id, correlation_id, planner_name, "
                    "planner_version, policy_name, policy_version, plan_summary, created_at, "
                    "updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        run_id,
                        organization_id,
                        gate.site_id,
                        gate.id,
                        request.objective,
                        request.intent,
                        AgentRunStatus.RUNNING,
                        actor_id,
                        request.idempotency_key,
                        trace_id,
                        correlation_id,
                        self.planner.name,
                        self.planner.version,
                        POLICY_NAME,
                        POLICY_VERSION,
                        plan.summary,
                        timestamp,
                        timestamp,
                    ),
                )
                for planned in plan.steps:
                    connection.execute(
                        "INSERT INTO agent_steps "
                        "(id, run_id, organization_id, sequence, tool_name, risk, status, "
                        "rationale, input_json, policy_checks_json) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, '{}', '[]')",
                        (
                            _new_id("agent-step"),
                            run_id,
                            organization_id,
                            planned.sequence,
                            planned.tool_name,
                            planned.risk,
                            AgentStepStatus.PENDING,
                            planned.rationale,
                        ),
                    )
                self._append_audit(
                    connection,
                    run_id=run_id,
                    organization_id=organization_id,
                    event_type="run.created",
                    actor_type="human",
                    actor_id=actor_id,
                    summary="Operations agent run requested",
                    metadata={"intent": request.intent, "gate_id": gate.id},
                )
                self._append_audit(
                    connection,
                    run_id=run_id,
                    organization_id=organization_id,
                    event_type="plan.created",
                    actor_type="planner",
                    actor_id=self.planner.name,
                    summary="Versioned deterministic plan persisted",
                    metadata={
                        "planner_version": self.planner.version,
                        "policy_version": POLICY_VERSION,
                        "tool_count": len(plan.steps),
                    },
                )
        except sqlite3.IntegrityError as error:
            concurrent = self._find_idempotent_run(
                organization_id,
                actor_id,
                request.idempotency_key,
            )
            if concurrent is not None:
                self._assert_run_binding(concurrent, organization_id, actor_id, request)
                concurrent_id = str(concurrent["id"])
                return self.get_run(organization_id, concurrent_id)
            raise ConflictError("Agent run could not be created") from error

        self._execute_inspection(run_id, organization_id)
        return self.get_run(organization_id, run_id)

    def list_runs(
        self,
        organization_id: str,
        *,
        gate_id: str | None,
        limit: int,
    ) -> list[AgentRunRead]:
        params: list[object] = [organization_id]
        sql = "SELECT id FROM agent_runs WHERE organization_id = ?"
        if gate_id is not None:
            self.repository.get_gate(organization_id, gate_id)
            sql += " AND gate_id = ?"
            params.append(gate_id)
        sql += " ORDER BY created_at DESC, id DESC LIMIT ?"
        params.append(limit)
        with self.repository.database.connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [self.get_run(organization_id, str(row["id"])) for row in rows]

    def get_run(self, organization_id: str, run_id: str) -> AgentRunRead:
        with self.repository.database.connect() as connection:
            run = connection.execute(
                "SELECT * FROM agent_runs WHERE id = ? AND organization_id = ?",
                (run_id, organization_id),
            ).fetchone()
            if run is None:
                raise ResourceNotFoundError("Agent run was not found")
            step_rows = connection.execute(
                "SELECT * FROM agent_steps WHERE run_id = ? AND organization_id = ? "
                "ORDER BY sequence",
                (run_id, organization_id),
            ).fetchall()
            approval_row = connection.execute(
                "SELECT * FROM agent_approvals WHERE run_id = ? AND organization_id = ?",
                (run_id, organization_id),
            ).fetchone()
            audit_rows = connection.execute(
                "SELECT * FROM agent_audit_events WHERE run_id = ? AND organization_id = ? "
                "ORDER BY sequence",
                (run_id, organization_id),
            ).fetchall()

        steps = [self._step_model(row) for row in step_rows]
        pending_step = next(
            (step for step in steps if step.status is AgentStepStatus.AWAITING_APPROVAL),
            None,
        )
        pending = (
            AgentPendingApproval(
                step_id=pending_step.id,
                tool_name=pending_step.tool_name,
                reason="A human operator must approve this consequential tool invocation.",
                requested_at=pending_step.started_at,
            )
            if pending_step is not None and pending_step.started_at is not None
            else None
        )
        approval = (
            AgentApprovalRead(
                id=approval_row["id"],
                step_id=approval_row["step_id"],
                decision=approval_row["decision"],
                reason=approval_row["reason"],
                decided_by=approval_row["decided_by"],
                decided_at=approval_row["decided_at"],
            )
            if approval_row is not None
            else None
        )
        audit_events = [
            AgentAuditEventRead(
                sequence=row["sequence"],
                id=row["id"],
                step_id=row["step_id"],
                event_type=row["event_type"],
                actor_type=row["actor_type"],
                actor_id=row["actor_id"],
                summary=row["summary"],
                metadata=json.loads(row["metadata_json"]),
                occurred_at=row["occurred_at"],
            )
            for row in audit_rows
        ]
        return AgentRunRead(
            id=run["id"],
            organization_id=run["organization_id"],
            site_id=run["site_id"],
            gate_id=run["gate_id"],
            objective=run["objective"],
            intent=run["intent"],
            status=run["status"],
            created_by=run["created_by"],
            created_at=run["created_at"],
            updated_at=run["updated_at"],
            trace=AgentTraceMetadata(
                trace_id=run["trace_id"],
                correlation_id=run["correlation_id"],
                planner_name=run["planner_name"],
                planner_version=run["planner_version"],
                policy_name=run["policy_name"],
                policy_version=run["policy_version"],
            ),
            plan=AgentPlan(
                summary=run["plan_summary"],
                steps=[
                    AgentPlannedStep(
                        sequence=step.sequence,
                        tool_name=step.tool_name,
                        risk=step.risk,
                        rationale=step.rationale,
                    )
                    for step in steps
                ],
            ),
            steps=steps,
            pending_approval=pending,
            approval=approval,
            audit_events=audit_events,
            failure_code=run["failure_code"],
            failure_detail=run["failure_detail"],
        )

    def decide(
        self,
        organization_id: str,
        run_id: str,
        actor_id: str,
        request: AgentApprovalDecisionCreate,
    ) -> AgentRunRead:
        snapshot = self.get_run(organization_id, run_id)
        with self.repository.database.immediate_transaction() as connection:
            current = connection.execute(
                "SELECT * FROM agent_runs WHERE id = ? AND organization_id = ?",
                (run_id, organization_id),
            ).fetchone()
            if current is None:
                raise ResourceNotFoundError("Agent run was not found")
            existing = connection.execute(
                "SELECT * FROM agent_approvals WHERE run_id = ? AND organization_id = ?",
                (run_id, organization_id),
            ).fetchone()
            if existing is not None:
                self._assert_decision_binding(existing, actor_id, request)
            else:
                if current["status"] != AgentRunStatus.AWAITING_APPROVAL:
                    raise InvalidStateError("Agent run is not awaiting a human decision")
                step_row = connection.execute(
                    "SELECT * FROM agent_steps WHERE run_id = ? AND organization_id = ? "
                    "AND status = ?",
                    (run_id, organization_id, AgentStepStatus.AWAITING_APPROVAL),
                ).fetchone()
                if step_row is None:
                    raise InvalidStateError("Agent run has no approval-gated step")
                step = self._step_model(cast(sqlite3.Row, step_row))
                if request.decision is AgentApprovalDecision.REJECTED:
                    self._record_rejection_in_transaction(
                        connection,
                        snapshot,
                        step,
                        actor_id,
                        request,
                    )
                else:
                    self._record_approval_and_effect_in_transaction(
                        connection,
                        snapshot,
                        step,
                        actor_id,
                        request,
                    )
        return self.get_run(organization_id, run_id)

    def _validate_plan(self, plan: AgentPlan) -> None:
        sequences = [step.sequence for step in plan.steps]
        if sequences != list(range(1, len(plan.steps) + 1)):
            raise InvalidStateError("Planner returned non-contiguous step sequences")
        unknown = [step.tool_name for step in plan.steps if step.tool_name not in TOOL_REGISTRY]
        if unknown:
            raise InvalidStateError(f"Planner returned tools outside the allowlist: {unknown}")
        expected_tools = tuple(AgentToolName)
        actual_tools = tuple(step.tool_name for step in plan.steps)
        if actual_tools != expected_tools:
            raise InvalidStateError("Planner changed the required gate_health_triage tool sequence")
        for step in plan.steps:
            if TOOL_REGISTRY[step.tool_name].risk is not step.risk:
                raise InvalidStateError(f"Planner changed the registered risk for {step.tool_name}")

    def _execute_inspection(self, run_id: str, organization_id: str) -> None:
        for sequence in (1, 2, 3):
            step = self._step_row(run_id, organization_id, sequence)
            if step["status"] == AgentStepStatus.SUCCEEDED:
                continue
            if step["status"] == AgentStepStatus.FAILED:
                return
            if not self._execute_read_step(run_id, organization_id, sequence):
                return
        current = self._run_row(organization_id, run_id)
        if current["status"] != AgentRunStatus.RUNNING:
            return
        try:
            self._prepare_consequential_step(run_id, organization_id)
        except Exception as error:
            LOGGER.exception("Agent evidence evaluation failed safely")
            code, detail = self._public_failure(error, phase="planning")
            self._fail_run(
                run_id,
                organization_id,
                code=code,
                detail=detail,
            )

    def _execute_read_step(self, run_id: str, organization_id: str, sequence: int) -> bool:
        run = self._run_row(organization_id, run_id)
        step = self._step_row(run_id, organization_id, sequence)
        tool_name = AgentToolName(step["tool_name"])
        definition = TOOL_REGISTRY.get(tool_name)
        if definition is None or definition.risk is not AgentToolRisk.READ_ONLY:
            self._fail_step(
                run_id,
                organization_id,
                str(step["id"]),
                code="tool_not_allowed",
                detail="Planner requested an unregistered read tool",
            )
            return False
        inputs = {"organization_id": organization_id, "gate_id": run["gate_id"]}
        checks = self._policy_checks(tool_name, organization_id, str(run["gate_id"]))
        started_at = _now()
        with self.repository.database.transaction() as connection:
            connection.execute(
                "UPDATE agent_steps SET status = ?, input_json = ?, policy_checks_json = ?, "
                "started_at = ? WHERE id = ? AND run_id = ? AND organization_id = ?",
                (
                    AgentStepStatus.RUNNING,
                    _json(inputs),
                    _json([check.model_dump(mode="json") for check in checks]),
                    started_at,
                    step["id"],
                    run_id,
                    organization_id,
                ),
            )
            self._append_audit(
                connection,
                run_id=run_id,
                organization_id=organization_id,
                step_id=str(step["id"]),
                event_type="tool.started",
                actor_type="agent_runtime",
                actor_id=str(run["planner_name"]),
                summary=f"Read tool {tool_name.value} started",
                metadata={"risk": definition.risk},
            )
        try:
            output = self._invoke_read_tool(tool_name, organization_id, str(run["gate_id"]))
        except Exception as error:
            LOGGER.exception("Read-only agent tool failed safely")
            code, detail = self._public_failure(error, phase="read")
            self._fail_step(
                run_id,
                organization_id,
                str(step["id"]),
                code=code,
                detail=detail,
            )
            return False

        completed_at = _now()
        with self.repository.database.transaction() as connection:
            connection.execute(
                "UPDATE agent_steps SET status = ?, output_json = ?, completed_at = ? "
                "WHERE id = ? AND run_id = ? AND organization_id = ?",
                (
                    AgentStepStatus.SUCCEEDED,
                    _json(output),
                    completed_at,
                    step["id"],
                    run_id,
                    organization_id,
                ),
            )
            connection.execute(
                "UPDATE agent_runs SET updated_at = ? WHERE id = ? AND organization_id = ?",
                (completed_at, run_id, organization_id),
            )
            self._append_audit(
                connection,
                run_id=run_id,
                organization_id=organization_id,
                step_id=str(step["id"]),
                event_type="tool.succeeded",
                actor_type="agent_runtime",
                actor_id=str(run["planner_name"]),
                summary=f"Read tool {tool_name.value} completed",
                metadata={"output_keys": sorted(output)},
            )
        return True

    def _invoke_read_tool(
        self,
        tool_name: AgentToolName,
        organization_id: str,
        gate_id: str,
    ) -> dict[str, Any]:
        gate = self.repository.get_gate(organization_id, gate_id)
        if tool_name is AgentToolName.GET_GATE:
            return {"gate": gate.model_dump(mode="json")}
        if tool_name is AgentToolName.GET_LATEST_DEVICE_HEALTH:
            with self.repository.database.connect() as connection:
                return self._gate_health_evidence(
                    connection,
                    organization_id=organization_id,
                    site_id=gate.site_id,
                    gate_id=gate_id,
                )
        if tool_name is AgentToolName.LIST_OPEN_GATE_INCIDENTS:
            incidents = [
                incident
                for incident in self.repository.list_incidents(
                    organization_id,
                    site_id=gate.site_id,
                )
                if incident.gate_id == gate_id and incident.status is not IncidentStatus.RESOLVED
            ]
            return {
                "count": len(incidents),
                "incidents": [incident.model_dump(mode="json") for incident in incidents],
            }
        raise InvalidStateError(f"Read tool is not registered: {tool_name}")

    def _gate_health_evidence(
        self,
        connection: sqlite3.Connection,
        *,
        organization_id: str,
        site_id: str,
        gate_id: str,
    ) -> dict[str, Any]:
        """Measure fresh health coverage against configured cameras, ignoring phantom reports."""

        evaluated_at = self.clock()
        if evaluated_at.tzinfo is None:
            evaluated_at = evaluated_at.replace(tzinfo=UTC)
        else:
            evaluated_at = evaluated_at.astimezone(UTC)
        camera_rows = connection.execute(
            "SELECT id, code, name, status FROM cameras "
            "WHERE organization_id = ? AND site_id = ? AND gate_id = ? "
            "AND status != 'disabled' ORDER BY id",
            (organization_id, site_id, gate_id),
        ).fetchall()
        expected_devices = [
            {
                "device_id": str(camera["id"]),
                "camera_id": str(camera["id"]),
                "code": str(camera["code"]),
                "name": str(camera["name"]),
            }
            for camera in camera_rows
        ]
        reports: list[dict[str, Any]] = []
        missing_device_ids: list[str] = []
        stale_device_ids: list[str] = []
        future_device_ids: list[str] = []
        invalid_timestamp_device_ids: list[str] = []
        unhealthy_device_ids: list[str] = []
        fresh_device_ids: list[str] = []
        for camera in camera_rows:
            device_id = str(camera["id"])
            report = connection.execute(
                "SELECT * FROM device_health WHERE organization_id = ? AND site_id = ? "
                "AND gate_id = ? AND camera_id = ? AND device_id = ? AND device_type = 'camera' "
                "ORDER BY julianday(reported_at) DESC, id DESC LIMIT 1",
                (organization_id, site_id, gate_id, device_id, device_id),
            ).fetchone()
            if report is None:
                missing_device_ids.append(device_id)
                continue
            evidence_state = self._health_timestamp_state(
                str(report["reported_at"]),
                evaluated_at,
            )
            report_data = dict(report)
            report_data["evidence_state"] = evidence_state
            reports.append(report_data)
            if evidence_state == "stale":
                stale_device_ids.append(device_id)
            elif evidence_state == "future":
                future_device_ids.append(device_id)
            elif evidence_state == "invalid_timestamp":
                invalid_timestamp_device_ids.append(device_id)
            else:
                fresh_device_ids.append(device_id)
                if report["status"] != "online":
                    unhealthy_device_ids.append(device_id)

        attention_reasons: list[str] = []
        if not expected_devices:
            attention_reasons.append("no_configured_camera")
        if missing_device_ids:
            attention_reasons.append("missing_health")
        if stale_device_ids:
            attention_reasons.append("stale_health")
        if future_device_ids:
            attention_reasons.append("future_health")
        if invalid_timestamp_device_ids:
            attention_reasons.append("invalid_health_timestamp")
        if unhealthy_device_ids:
            attention_reasons.append("unhealthy_status")
        ready = bool(expected_devices) and not attention_reasons
        return {
            "count": len(reports),
            "expected_count": len(expected_devices),
            "fresh_count": len(fresh_device_ids),
            "ready": ready,
            "evaluated_at": evaluated_at.isoformat(),
            "max_age_seconds": int(HEALTH_MAX_AGE.total_seconds()),
            "future_tolerance_seconds": int(HEALTH_FUTURE_TOLERANCE.total_seconds()),
            "expected_devices": expected_devices,
            "reports": reports,
            "missing_device_ids": missing_device_ids,
            "stale_device_ids": stale_device_ids,
            "future_device_ids": future_device_ids,
            "invalid_timestamp_device_ids": invalid_timestamp_device_ids,
            "unhealthy_device_ids": unhealthy_device_ids,
            "attention_reasons": attention_reasons,
        }

    @staticmethod
    def _health_timestamp_state(reported_at: str, evaluated_at: datetime) -> str:
        try:
            parsed = datetime.fromisoformat(reported_at)
        except ValueError:
            return "invalid_timestamp"
        if parsed.tzinfo is None:
            return "invalid_timestamp"
        normalized = parsed.astimezone(UTC)
        if normalized > evaluated_at + HEALTH_FUTURE_TOLERANCE:
            return "future"
        if normalized < evaluated_at - HEALTH_MAX_AGE:
            return "stale"
        return "fresh"

    def _prepare_consequential_step(self, run_id: str, organization_id: str) -> None:
        run = self._run_row(organization_id, run_id)
        gate_output = self._step_output(run_id, organization_id, 1)
        health_output = self._step_output(run_id, organization_id, 2)
        incident_output = self._step_output(run_id, organization_id, 3)
        gate = gate_output["gate"]
        reports = health_output["reports"]
        incidents = incident_output["incidents"]
        unhealthy = not bool(health_output["ready"])
        unhealthy = unhealthy or gate["status"] in {"congested", "degraded", "offline"}

        if incidents:
            selected = next(
                (
                    incident
                    for incident in incidents
                    if incident["status"] in {IncidentStatus.OPEN, IncidentStatus.INVESTIGATING}
                    and not incident["assigned_to"]
                ),
                None,
            )
            if selected is not None:
                self._set_step_awaiting_approval(
                    run,
                    sequence=4,
                    inputs={
                        "organization_id": organization_id,
                        "gate_id": run["gate_id"],
                        "incident_id": selected["id"],
                        "target_status": IncidentStatus.INVESTIGATING,
                        "assigned_to": "$approving_operator",
                    },
                    summary="Human approval requested to start the existing incident investigation",
                )
                self._skip_step(
                    run_id,
                    organization_id,
                    sequence=5,
                    reason=(
                        "An actionable unresolved incident exists; duplicate creation was suppressed."
                    ),
                )
                return
            self._skip_step(
                run_id,
                organization_id,
                sequence=4,
                reason="All unresolved incidents are already assigned; reassignment was suppressed.",
            )
            self._skip_step(
                run_id,
                organization_id,
                sequence=5,
                reason="An active gate incident already exists; duplicate creation was suppressed.",
            )
            self._complete_without_action(run_id, organization_id)
            return

        self._skip_step(
            run_id,
            organization_id,
            sequence=4,
            reason="No unresolved incident exists to advance.",
        )
        if unhealthy:
            statuses = sorted({str(report["status"]) for report in reports}) or ["missing"]
            evidence_reasons = list(health_output["attention_reasons"])
            if gate["status"] in {"congested", "degraded", "offline"}:
                evidence_reasons.append(f"gate_{gate['status']}")
            severity = (
                IncidentSeverity.CRITICAL
                if "offline" in statuses or gate["status"] == "offline"
                else IncidentSeverity.WARNING
            )
            self._set_step_awaiting_approval(
                run,
                sequence=5,
                inputs={
                    "site_id": run["site_id"],
                    "gate_id": run["gate_id"],
                    "title": f"Agent-assisted health triage: {gate['name']}",
                    "severity": severity,
                    "description": (
                        "The bounded operations workflow found gate or device health requiring "
                        f"review. Evidence attention: {', '.join(evidence_reasons)}. "
                        f"Observed device states: {', '.join(statuses)}. "
                        f"Agent run: {run_id}."
                    ),
                },
                summary="Human approval requested to create a gate-health incident",
            )
            return

        self._skip_step(
            run_id,
            organization_id,
            sequence=5,
            reason="Gate and current device reports are healthy; no consequential action proposed.",
        )
        self._complete_without_action(run_id, organization_id)

    def _policy_checks(
        self,
        tool_name: AgentToolName,
        organization_id: str,
        gate_id: str,
        *,
        consequential: bool = False,
    ) -> list[AgentPolicyCheck]:
        checks = [
            AgentPolicyCheck(
                code="tool_allowlisted",
                outcome=(
                    AgentPolicyOutcome.ALLOW
                    if tool_name in ALLOWED_AGENT_TOOLS
                    else AgentPolicyOutcome.DENY
                ),
                detail=f"{tool_name.value} is registered in the closed operations tool registry.",
                policy_name=POLICY_NAME,
                policy_version=POLICY_VERSION,
            ),
            AgentPolicyCheck(
                code="organization_scope",
                outcome=AgentPolicyOutcome.ALLOW,
                detail=f"Tool input is pinned to authenticated organization {organization_id}.",
                policy_name=POLICY_NAME,
                policy_version=POLICY_VERSION,
            ),
            AgentPolicyCheck(
                code="gate_scope",
                outcome=AgentPolicyOutcome.ALLOW,
                detail=f"Tool input is pinned to selected gate {gate_id}.",
                policy_name=POLICY_NAME,
                policy_version=POLICY_VERSION,
            ),
        ]
        if consequential:
            checks.append(
                AgentPolicyCheck(
                    code="human_approval",
                    outcome=AgentPolicyOutcome.APPROVAL_REQUIRED,
                    detail="The runtime is paused; this tool cannot execute without a decision.",
                    policy_name=POLICY_NAME,
                    policy_version=POLICY_VERSION,
                )
            )
        return checks

    def _set_step_awaiting_approval(
        self,
        run: sqlite3.Row,
        *,
        sequence: int,
        inputs: dict[str, object],
        summary: str,
    ) -> None:
        step = self._step_row(str(run["id"]), str(run["organization_id"]), sequence)
        tool_name = AgentToolName(step["tool_name"])
        checks = self._policy_checks(
            tool_name,
            str(run["organization_id"]),
            str(run["gate_id"]),
            consequential=True,
        )
        timestamp = _now()
        with self.repository.database.transaction() as connection:
            connection.execute(
                "UPDATE agent_steps SET status = ?, input_json = ?, policy_checks_json = ?, "
                "started_at = ? WHERE id = ? AND run_id = ? AND organization_id = ?",
                (
                    AgentStepStatus.AWAITING_APPROVAL,
                    _json(inputs),
                    _json([check.model_dump(mode="json") for check in checks]),
                    timestamp,
                    step["id"],
                    run["id"],
                    run["organization_id"],
                ),
            )
            connection.execute(
                "UPDATE agent_runs SET status = ?, updated_at = ? "
                "WHERE id = ? AND organization_id = ?",
                (
                    AgentRunStatus.AWAITING_APPROVAL,
                    timestamp,
                    run["id"],
                    run["organization_id"],
                ),
            )
            self._append_audit(
                connection,
                run_id=str(run["id"]),
                organization_id=str(run["organization_id"]),
                step_id=str(step["id"]),
                event_type="approval.requested",
                actor_type="policy_engine",
                actor_id=POLICY_NAME,
                summary=summary,
                metadata={"tool_name": tool_name, "policy_version": POLICY_VERSION},
            )

    def _skip_step(
        self,
        run_id: str,
        organization_id: str,
        *,
        sequence: int,
        reason: str,
    ) -> None:
        step = self._step_row(run_id, organization_id, sequence)
        if step["status"] == AgentStepStatus.SKIPPED:
            return
        run = self._run_row(organization_id, run_id)
        tool_name = AgentToolName(step["tool_name"])
        timestamp = _now()
        with self.repository.database.transaction() as connection:
            connection.execute(
                "UPDATE agent_steps SET status = ?, output_json = ?, completed_at = ? "
                "WHERE id = ? AND run_id = ? AND organization_id = ?",
                (
                    AgentStepStatus.SKIPPED,
                    _json({"reason": reason}),
                    timestamp,
                    step["id"],
                    run_id,
                    organization_id,
                ),
            )
            self._append_audit(
                connection,
                run_id=run_id,
                organization_id=organization_id,
                step_id=str(step["id"]),
                event_type="tool.skipped",
                actor_type="agent_runtime",
                actor_id=str(run["planner_name"]),
                summary=f"Conditional tool {tool_name.value} skipped",
                metadata={"reason": reason},
            )

    def _complete_without_action(self, run_id: str, organization_id: str) -> None:
        run = self._run_row(organization_id, run_id)
        timestamp = _now()
        with self.repository.database.transaction() as connection:
            connection.execute(
                "UPDATE agent_runs SET status = ?, updated_at = ? "
                "WHERE id = ? AND organization_id = ?",
                (AgentRunStatus.COMPLETED, timestamp, run_id, organization_id),
            )
            self._append_audit(
                connection,
                run_id=run_id,
                organization_id=organization_id,
                event_type="run.completed",
                actor_type="agent_runtime",
                actor_id=str(run["planner_name"]),
                summary="Inspection completed with no consequential action",
                metadata={"action_taken": False},
            )

    def _record_rejection_in_transaction(
        self,
        connection: sqlite3.Connection,
        run: AgentRunRead,
        step: AgentStepRead,
        actor_id: str,
        request: AgentApprovalDecisionCreate,
    ) -> None:
        timestamp = _now()
        self._insert_approval(connection, run, step, actor_id, request, timestamp)
        connection.execute(
            "UPDATE agent_steps SET status = ?, output_json = ?, completed_at = ? "
            "WHERE id = ? AND run_id = ? AND organization_id = ?",
            (
                AgentStepStatus.SKIPPED,
                _json({"reason": "Human operator rejected the proposed action."}),
                timestamp,
                step.id,
                run.id,
                run.organization_id,
            ),
        )
        connection.execute(
            "UPDATE agent_runs SET status = ?, updated_at = ? WHERE id = ? AND organization_id = ?",
            (AgentRunStatus.REJECTED, timestamp, run.id, run.organization_id),
        )
        self._append_audit(
            connection,
            run_id=run.id,
            organization_id=run.organization_id,
            step_id=step.id,
            event_type="approval.rejected",
            actor_type="human",
            actor_id=actor_id,
            summary="Human operator rejected the proposed action",
            metadata={"reason": request.reason},
        )

    def _record_approval_and_effect_in_transaction(
        self,
        connection: sqlite3.Connection,
        run: AgentRunRead,
        step: AgentStepRead,
        actor_id: str,
        request: AgentApprovalDecisionCreate,
    ) -> None:
        timestamp = _now()
        checks = self._approved_policy_checks(step, actor_id)
        self._insert_approval(connection, run, step, actor_id, request, timestamp)
        connection.execute(
            "UPDATE agent_steps SET status = ?, policy_checks_json = ? "
            "WHERE id = ? AND run_id = ? AND organization_id = ?",
            (
                AgentStepStatus.RUNNING,
                _json([check.model_dump(mode="json") for check in checks]),
                step.id,
                run.id,
                run.organization_id,
            ),
        )
        self._append_audit(
            connection,
            run_id=run.id,
            organization_id=run.organization_id,
            step_id=step.id,
            event_type="approval.approved",
            actor_type="human",
            actor_id=actor_id,
            summary="Human operator approved the proposed action",
            metadata={"reason": request.reason, "tool_name": step.tool_name},
        )

        connection.execute("SAVEPOINT agent_effect")
        try:
            output = self._execute_consequential_tool(connection, run, step, actor_id)
            checks.append(
                AgentPolicyCheck(
                    code="effect_preconditions_revalidated",
                    outcome=AgentPolicyOutcome.ALLOW,
                    detail=(
                        "Tenant, gate, evidence freshness, resource state, and duplicate-action "
                        "guards passed at commit time."
                    ),
                    policy_name=POLICY_NAME,
                    policy_version=POLICY_VERSION,
                )
            )
            self._complete_action(
                connection,
                run,
                step,
                actor_id,
                output,
                timestamp,
                checks,
            )
        except Exception as error:
            connection.execute("ROLLBACK TO agent_effect")
            connection.execute("RELEASE agent_effect")
            LOGGER.exception("Approved agent tool failed safely")
            code, detail = self._public_failure(error, phase="effect")
            self._record_effect_failure_in_transaction(
                connection,
                run,
                step,
                code=code,
                detail=detail,
                timestamp=timestamp,
                checks=checks,
            )
            return
        connection.execute("RELEASE agent_effect")

    def _record_effect_failure_in_transaction(
        self,
        connection: sqlite3.Connection,
        run: AgentRunRead,
        step: AgentStepRead,
        *,
        code: str,
        detail: str,
        timestamp: str,
        checks: list[AgentPolicyCheck],
    ) -> None:
        connection.execute(
            "UPDATE agent_steps SET status = ?, policy_checks_json = ?, error_code = ?, "
            "error_detail = ?, completed_at = ? "
            "WHERE id = ? AND run_id = ? AND organization_id = ?",
            (
                AgentStepStatus.FAILED,
                _json([check.model_dump(mode="json") for check in checks]),
                code,
                detail,
                timestamp,
                step.id,
                run.id,
                run.organization_id,
            ),
        )
        connection.execute(
            "UPDATE agent_runs SET status = ?, failure_code = ?, failure_detail = ?, "
            "updated_at = ? WHERE id = ? AND organization_id = ?",
            (
                AgentRunStatus.FAILED,
                code,
                detail,
                timestamp,
                run.id,
                run.organization_id,
            ),
        )
        self._append_audit(
            connection,
            run_id=run.id,
            organization_id=run.organization_id,
            step_id=step.id,
            event_type="tool.failed",
            actor_type="agent_runtime",
            actor_id=run.trace.planner_name,
            summary="Approved tool failed without committing its effect",
            metadata={"error_code": code, "error_detail": detail},
        )
        self._append_audit(
            connection,
            run_id=run.id,
            organization_id=run.organization_id,
            event_type="run.failed",
            actor_type="agent_runtime",
            actor_id=run.trace.planner_name,
            summary="Agent run failed after the approved effect rolled back",
            metadata={"error_code": code, "error_detail": detail},
        )

    @staticmethod
    def _approved_policy_checks(
        step: AgentStepRead,
        actor_id: str,
    ) -> list[AgentPolicyCheck]:
        checks = [*step.policy_checks]
        checks.append(
            AgentPolicyCheck(
                code="human_approval_recorded",
                outcome=AgentPolicyOutcome.ALLOW,
                detail=f"Approval was recorded for operator {actor_id}.",
                policy_name=POLICY_NAME,
                policy_version=POLICY_VERSION,
            )
        )
        return checks

    @staticmethod
    def _assert_decision_binding(
        row: sqlite3.Row,
        actor_id: str,
        request: AgentApprovalDecisionCreate,
    ) -> None:
        """Return an earlier decision only for an exact actor/request retry."""

        if (
            row["idempotency_key"] != request.idempotency_key
            or row["decided_by"] != actor_id
            or row["decision"] != request.decision.value
            or row["reason"] != request.reason
        ):
            raise ConflictError("Agent run is bound to a different human decision")

    @staticmethod
    def _public_failure(error: Exception, *, phase: str) -> tuple[str, str]:
        """Map internal exceptions to stable, non-sensitive workflow failure contracts."""

        if phase == "effect":
            if isinstance(error, InvalidStateError):
                return (
                    "effect_precondition_failed",
                    "The proposed action no longer satisfies commit-time safety checks.",
                )
            if isinstance(error, ResourceNotFoundError):
                return (
                    "effect_resource_changed",
                    "A scoped resource changed or is no longer available.",
                )
            if isinstance(error, sqlite3.Error):
                return (
                    "effect_persistence_failed",
                    "The persistence layer could not commit the approved tool safely.",
                )
            return (
                "effect_execution_failed",
                "The approved tool could not complete safely.",
            )
        if phase == "read":
            if isinstance(error, ResourceNotFoundError):
                return (
                    "read_resource_changed",
                    "A scoped resource changed while the read tool was running.",
                )
            if isinstance(error, sqlite3.Error):
                return (
                    "read_persistence_failed",
                    "The persistence layer could not complete the read tool.",
                )
            if isinstance(error, ControlApiError):
                return (
                    "read_policy_failed",
                    "The read tool no longer satisfies its scoped policy checks.",
                )
            return (
                "read_execution_failed",
                "The read-only agent tool could not complete safely.",
            )
        return (
            "planning_evidence_failed",
            "The agent could not evaluate the inspected evidence safely.",
        )

    def _execute_consequential_tool(
        self,
        connection: sqlite3.Connection,
        run: AgentRunRead,
        step: AgentStepRead,
        actor_id: str,
    ) -> dict[str, Any]:
        if (
            step.tool_name not in ALLOWED_AGENT_TOOLS
            or step.risk is not AgentToolRisk.CONSEQUENTIAL
        ):
            raise InvalidStateError("Consequential tool is not allowlisted")
        gate = connection.execute(
            "SELECT site_id, status FROM gates WHERE id = ? AND organization_id = ?",
            (run.gate_id, run.organization_id),
        ).fetchone()
        if gate is None:
            raise ResourceNotFoundError("Run gate was not found")
        if gate["site_id"] != run.site_id:
            raise InvalidStateError("Run gate no longer belongs to the recorded site")
        if step.tool_name is AgentToolName.START_INCIDENT_INVESTIGATION:
            incident_id = str(step.input["incident_id"])
            incident = connection.execute(
                "SELECT gate_id, status, assigned_to FROM incidents "
                "WHERE id = ? AND organization_id = ?",
                (incident_id, run.organization_id),
            ).fetchone()
            if incident is None:
                raise ResourceNotFoundError("Proposed incident was not found")
            if incident["gate_id"] != run.gate_id:
                raise InvalidStateError("Incident escaped the run's gate scope")
            if incident["status"] == IncidentStatus.RESOLVED:
                raise InvalidStateError("Incident was resolved after the agent prepared its plan")
            if incident["assigned_to"]:
                raise InvalidStateError("Incident was assigned after the agent prepared its plan")
            if incident["status"] not in {IncidentStatus.OPEN, IncidentStatus.INVESTIGATING}:
                raise InvalidStateError("Incident is no longer actionable")
            updated = self.repository.update_incident_in_transaction(
                connection,
                run.organization_id,
                incident_id,
                IncidentUpdate(
                    status=IncidentStatus.INVESTIGATING,
                    assigned_to=actor_id,
                ),
            )
            return {"incident": updated.model_dump(mode="json")}
        if step.tool_name is AgentToolName.CREATE_INCIDENT:
            payload = IncidentCreate.model_validate(step.input)
            if payload.gate_id != run.gate_id or payload.site_id != run.site_id:
                raise InvalidStateError("Incident proposal escaped the run scope")
            duplicate = connection.execute(
                "SELECT id FROM incidents WHERE organization_id = ? AND gate_id = ? "
                "AND status != 'resolved' LIMIT 1",
                (run.organization_id, run.gate_id),
            ).fetchone()
            if duplicate is not None:
                raise InvalidStateError(
                    "An unresolved gate incident appeared after the agent prepared its plan"
                )
            health_evidence = self._gate_health_evidence(
                connection,
                organization_id=run.organization_id,
                site_id=run.site_id,
                gate_id=run.gate_id,
            )
            if health_evidence["ready"] and gate["status"] == "operational":
                raise InvalidStateError("Gate health recovered after the agent prepared its plan")
            created = self.repository.create_incident_in_transaction(
                connection,
                run.organization_id,
                actor_id,
                payload,
            )
            return {"incident": created.model_dump(mode="json")}
        raise InvalidStateError(f"Unsupported consequential tool: {step.tool_name}")

    def _complete_action(
        self,
        connection: sqlite3.Connection,
        run: AgentRunRead,
        step: AgentStepRead,
        actor_id: str,
        output: dict[str, Any],
        timestamp: str,
        checks: list[AgentPolicyCheck],
    ) -> None:
        connection.execute(
            "UPDATE agent_steps SET status = ?, output_json = ?, policy_checks_json = ?, "
            "completed_at = ? "
            "WHERE id = ? AND run_id = ? AND organization_id = ?",
            (
                AgentStepStatus.SUCCEEDED,
                _json(output),
                _json([check.model_dump(mode="json") for check in checks]),
                timestamp,
                step.id,
                run.id,
                run.organization_id,
            ),
        )
        connection.execute(
            "UPDATE agent_runs SET status = ?, updated_at = ? WHERE id = ? AND organization_id = ?",
            (AgentRunStatus.COMPLETED, timestamp, run.id, run.organization_id),
        )
        self._append_audit(
            connection,
            run_id=run.id,
            organization_id=run.organization_id,
            step_id=step.id,
            event_type="tool.succeeded",
            actor_type="agent_runtime",
            actor_id=run.trace.planner_name,
            summary=f"Approved tool {step.tool_name.value} completed",
            metadata={
                "approved_by": actor_id,
                "output_keys": sorted(output),
            },
        )
        self._append_audit(
            connection,
            run_id=run.id,
            organization_id=run.organization_id,
            event_type="run.completed",
            actor_type="agent_runtime",
            actor_id=run.trace.planner_name,
            summary="Operations agent run completed after human approval",
            metadata={"action_taken": True, "tool_name": step.tool_name},
        )

    def _fail_step(
        self,
        run_id: str,
        organization_id: str,
        step_id: str,
        *,
        code: str,
        detail: str,
    ) -> None:
        safe_detail = detail[:1000]
        timestamp = _now()
        run = self._run_row(organization_id, run_id)
        with self.repository.database.transaction() as connection:
            current = connection.execute(
                "SELECT status FROM agent_runs WHERE id = ? AND organization_id = ?",
                (run_id, organization_id),
            ).fetchone()
            if current is not None and current["status"] == AgentRunStatus.FAILED:
                return
            connection.execute(
                "UPDATE agent_steps SET status = ?, error_code = ?, error_detail = ?, "
                "completed_at = ? WHERE id = ? AND run_id = ? AND organization_id = ?",
                (
                    AgentStepStatus.FAILED,
                    code,
                    safe_detail,
                    timestamp,
                    step_id,
                    run_id,
                    organization_id,
                ),
            )
            connection.execute(
                "UPDATE agent_runs SET status = ?, failure_code = ?, failure_detail = ?, "
                "updated_at = ? WHERE id = ? AND organization_id = ?",
                (
                    AgentRunStatus.FAILED,
                    code,
                    safe_detail,
                    timestamp,
                    run_id,
                    organization_id,
                ),
            )
            self._append_audit(
                connection,
                run_id=run_id,
                organization_id=organization_id,
                step_id=step_id,
                event_type="tool.failed",
                actor_type="agent_runtime",
                actor_id=str(run["planner_name"]),
                summary="Tool execution failed gracefully",
                metadata={"error_code": code, "error_detail": safe_detail},
            )
            self._append_audit(
                connection,
                run_id=run_id,
                organization_id=organization_id,
                event_type="run.failed",
                actor_type="agent_runtime",
                actor_id=str(run["planner_name"]),
                summary="Agent run failed after a read-only tool error",
                metadata={"error_code": code, "error_detail": safe_detail},
            )

    def _fail_run(
        self,
        run_id: str,
        organization_id: str,
        *,
        code: str,
        detail: str,
    ) -> None:
        safe_detail = detail[:1000]
        timestamp = _now()
        run = self._run_row(organization_id, run_id)
        with self.repository.database.transaction() as connection:
            current = connection.execute(
                "SELECT status FROM agent_runs WHERE id = ? AND organization_id = ?",
                (run_id, organization_id),
            ).fetchone()
            if current is not None and current["status"] == AgentRunStatus.FAILED:
                return
            connection.execute(
                "UPDATE agent_runs SET status = ?, failure_code = ?, failure_detail = ?, "
                "updated_at = ? WHERE id = ? AND organization_id = ?",
                (
                    AgentRunStatus.FAILED,
                    code,
                    safe_detail,
                    timestamp,
                    run_id,
                    organization_id,
                ),
            )
            self._append_audit(
                connection,
                run_id=run_id,
                organization_id=organization_id,
                event_type="run.failed",
                actor_type="agent_runtime",
                actor_id=str(run["planner_name"]),
                summary="Agent run failed gracefully",
                metadata={"error_code": code, "error_detail": safe_detail},
            )

    def _insert_approval(
        self,
        connection: sqlite3.Connection,
        run: AgentRunRead,
        step: AgentStepRead,
        actor_id: str,
        request: AgentApprovalDecisionCreate,
        timestamp: str,
    ) -> None:
        connection.execute(
            "INSERT INTO agent_approvals "
            "(id, run_id, organization_id, step_id, decision, reason, decided_by, "
            "idempotency_key, decided_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                _new_id("agent-approval"),
                run.id,
                run.organization_id,
                step.id,
                request.decision,
                request.reason,
                actor_id,
                request.idempotency_key,
                timestamp,
            ),
        )

    def _find_idempotent_run(
        self,
        organization_id: str,
        actor_id: str,
        idempotency_key: str,
    ) -> sqlite3.Row | None:
        with self.repository.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM agent_runs WHERE organization_id = ? AND created_by = ? "
                "AND idempotency_key = ?",
                (organization_id, actor_id, idempotency_key),
            ).fetchone()
        return cast(sqlite3.Row | None, row)

    @staticmethod
    def _assert_run_binding(
        row: sqlite3.Row,
        organization_id: str,
        actor_id: str,
        request: AgentRunCreate,
    ) -> None:
        """Bind an idempotency key to the complete original request and actor."""

        if (
            row["organization_id"] != organization_id
            or row["created_by"] != actor_id
            or row["idempotency_key"] != request.idempotency_key
            or row["gate_id"] != request.gate_id
            or row["objective"] != request.objective
            or row["intent"] != request.intent.value
        ):
            raise ConflictError("Agent run idempotency key is bound to a different request")

    def _run_row(self, organization_id: str, run_id: str) -> sqlite3.Row:
        with self.repository.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM agent_runs WHERE id = ? AND organization_id = ?",
                (run_id, organization_id),
            ).fetchone()
        if row is None:
            raise ResourceNotFoundError("Agent run was not found")
        return cast(sqlite3.Row, row)

    def _step_row(self, run_id: str, organization_id: str, sequence: int) -> sqlite3.Row:
        with self.repository.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM agent_steps WHERE run_id = ? AND organization_id = ? "
                "AND sequence = ?",
                (run_id, organization_id, sequence),
            ).fetchone()
        if row is None:
            raise ResourceNotFoundError("Agent step was not found")
        return cast(sqlite3.Row, row)

    def _step_output(self, run_id: str, organization_id: str, sequence: int) -> dict[str, Any]:
        row = self._step_row(run_id, organization_id, sequence)
        if row["status"] != AgentStepStatus.SUCCEEDED or row["output_json"] is None:
            raise InvalidStateError(f"Agent step {sequence} has no successful output")
        value: dict[str, Any] = json.loads(row["output_json"])
        return value

    @staticmethod
    def _step_model(row: sqlite3.Row) -> AgentStepRead:
        return AgentStepRead(
            id=row["id"],
            sequence=row["sequence"],
            tool_name=row["tool_name"],
            risk=row["risk"],
            status=row["status"],
            rationale=row["rationale"],
            input=json.loads(row["input_json"]),
            output=json.loads(row["output_json"]) if row["output_json"] is not None else None,
            policy_checks=json.loads(row["policy_checks_json"]),
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            error_code=row["error_code"],
            error_detail=row["error_detail"],
        )

    @staticmethod
    def _append_audit(
        connection: sqlite3.Connection,
        *,
        run_id: str,
        organization_id: str,
        event_type: str,
        actor_type: str,
        actor_id: str,
        summary: str,
        metadata: dict[str, object],
        step_id: str | None = None,
    ) -> None:
        connection.execute(
            "INSERT INTO agent_audit_events "
            "(id, run_id, organization_id, step_id, event_type, actor_type, actor_id, "
            "summary, metadata_json, occurred_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                _new_id("agent-audit"),
                run_id,
                organization_id,
                step_id,
                event_type,
                actor_type,
                actor_id,
                summary,
                _json(metadata),
                _now(),
            ),
        )
