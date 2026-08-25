"""Behavior, policy, tenancy, approval, and trace tests for operations agents."""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from threading import Barrier
from typing import Any, cast

import pytest
from conftest import auth
from control_api.agent_evals import run_agent_evaluations
from control_api.agentic import AgentWorkflowService, DeterministicGateHealthPlanner
from control_api.errors import ConflictError, InvalidStateError
from control_api.repository import Repository
from control_api.schemas import (
    AgentApprovalDecision,
    AgentApprovalDecisionCreate,
    AgentIntent,
    AgentPlan,
    AgentRunCreate,
    AgentToolName,
    GateRead,
    IncidentCreate,
    IncidentRead,
)
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _create_run(
    client: TestClient,
    *,
    gate_id: str,
    key: str,
    identity: str = "demo-operator",
    objective: str = "Inspect current gate health and prepare the safest operational response",
) -> Any:
    return client.post(
        "/api/v1/agent/runs",
        headers=auth(identity),
        json={
            "objective": objective,
            "gate_id": gate_id,
            "intent": "gate_health_triage",
            "idempotency_key": key,
        },
    )


def _report_camera_health(
    client: TestClient,
    *,
    gate_id: str,
    camera_id: str,
    status: str = "online",
    reported_at: datetime | None = None,
) -> Any:
    timestamp = reported_at or datetime.now(UTC)
    return client.post(
        "/api/v1/device-health",
        headers=auth("demo-edge"),
        json={
            "site_id": "site-atlas-main",
            "gate_id": gate_id,
            "camera_id": camera_id,
            "device_id": camera_id,
            "device_type": "camera",
            "status": status,
            "detail": f"Agent workflow {status} health fixture",
            "reported_at": timestamp.isoformat(),
        },
    )


def test_existing_incident_is_reused_and_requires_approval(client: TestClient) -> None:
    created = _create_run(
        client,
        gate_id="gate-atlas-service",
        key="triage-east-existing-01",
    )
    assert created.status_code == 201
    run = created.json()
    assert run["status"] == "awaiting_approval"
    assert run["trace"] == {
        "trace_id": run["trace"]["trace_id"],
        "correlation_id": run["trace"]["correlation_id"],
        "planner_name": "deterministic_gate_health_planner",
        "planner_version": "1.0.0",
        "policy_name": "campus_operations_guardrails",
        "policy_version": "1.0.0",
    }
    assert [step["tool_name"] for step in run["plan"]["steps"]] == [
        "get_gate",
        "get_latest_device_health",
        "list_open_gate_incidents",
        "start_incident_investigation",
        "create_incident",
    ]
    assert [step["status"] for step in run["steps"]] == [
        "succeeded",
        "succeeded",
        "succeeded",
        "awaiting_approval",
        "skipped",
    ]
    pending = run["pending_approval"]
    assert pending["tool_name"] == "start_incident_investigation"
    action = run["steps"][3]
    assert action["input"]["incident_id"] == "incident-atlas-camera"
    assert {check["code"] for check in action["policy_checks"]} >= {
        "tool_allowlisted",
        "organization_scope",
        "gate_scope",
        "human_approval",
    }
    assert (
        next(check for check in action["policy_checks"] if check["code"] == "human_approval")[
            "outcome"
        ]
        == "approval_required"
    )

    # Planning and read tools must not change the incident.
    before = client.get(
        "/api/v1/incidents/incident-atlas-camera",
        headers=auth("demo-viewer"),
    ).json()
    assert before["status"] == "open"
    assert before["assigned_to"] is None

    decision_payload = {
        "decision": "approved",
        "reason": "Camera packet-loss evidence warrants investigation",
        "idempotency_key": "approve-east-existing-01",
    }
    approved = client.post(
        f"/api/v1/agent/runs/{run['id']}/decisions",
        headers=auth("demo-operator"),
        json=decision_payload,
    )
    assert approved.status_code == 200
    completed = approved.json()
    assert completed["status"] == "completed"
    assert completed["approval"]["decision"] == "approved"
    assert completed["steps"][3]["status"] == "succeeded"
    assert completed["steps"][3]["output"]["incident"]["status"] == "investigating"
    assert any(
        check["code"] == "effect_preconditions_revalidated"
        for check in completed["steps"][3]["policy_checks"]
    )
    assert {event["event_type"] for event in completed["audit_events"]} >= {
        "run.created",
        "plan.created",
        "approval.requested",
        "approval.approved",
        "run.completed",
    }
    after = client.get(
        "/api/v1/incidents/incident-atlas-camera",
        headers=auth("demo-viewer"),
    ).json()
    assert after["status"] == "investigating"
    assert after["assigned_to"] == "operator-omar"

    # A transport retry returns the original decision without executing twice.
    retried = client.post(
        f"/api/v1/agent/runs/{run['id']}/decisions",
        headers=auth("demo-operator"),
        json=decision_payload,
    )
    assert retried.status_code == 200
    assert retried.json()["audit_events"] == completed["audit_events"]
    different_actor = client.post(
        f"/api/v1/agent/runs/{run['id']}/decisions",
        headers=auth("demo-admin"),
        json=decision_payload,
    )
    assert different_actor.status_code == 409


def test_degraded_gate_without_incident_drafts_then_creates_one(client: TestClient) -> None:
    health = client.post(
        "/api/v1/device-health",
        headers=auth("demo-edge"),
        json={
            "site_id": "site-atlas-main",
            "gate_id": "gate-atlas-north",
            "camera_id": "camera-atlas-north-anpr",
            "device_id": "camera-atlas-north-anpr",
            "device_type": "camera",
            "status": "offline",
            "latency_ms": None,
            "detail": "Deterministic agent evaluation signal",
            "reported_at": "2026-08-24T10:20:00+00:00",
        },
    )
    assert health.status_code == 201

    run = _create_run(
        client,
        gate_id="gate-atlas-north",
        key="triage-north-offline-01",
    ).json()
    assert run["status"] == "awaiting_approval"
    assert [step["status"] for step in run["steps"]][-2:] == [
        "skipped",
        "awaiting_approval",
    ]
    assert run["pending_approval"]["tool_name"] == "create_incident"
    assert run["steps"][4]["input"]["severity"] == "critical"
    before = client.get(
        "/api/v1/incidents?site_id=site-atlas-main",
        headers=auth("demo-viewer"),
    ).json()

    completed = client.post(
        f"/api/v1/agent/runs/{run['id']}/decisions",
        headers=auth("demo-admin"),
        json={
            "decision": "approved",
            "reason": "Open a tracked response for the offline gate camera",
            "idempotency_key": "approve-north-offline-01",
        },
    ).json()
    incident = completed["steps"][4]["output"]["incident"]
    assert completed["status"] == "completed"
    assert incident["gate_id"] == "gate-atlas-north"
    assert incident["severity"] == "critical"
    after = client.get(
        "/api/v1/incidents?site_id=site-atlas-main",
        headers=auth("demo-viewer"),
    ).json()
    assert len(after) == len(before) + 1


def test_healthy_gate_completes_without_requesting_approval(client: TestClient) -> None:
    assert (
        _report_camera_health(
            client,
            gate_id="gate-atlas-sports",
            camera_id="camera-atlas-sports-anpr",
        ).status_code
        == 201
    )
    run = _create_run(
        client,
        gate_id="gate-atlas-sports",
        key="triage-sports-healthy-01",
    ).json()
    assert run["status"] == "completed"
    assert run["pending_approval"] is None
    assert run["approval"] is None
    assert [step["status"] for step in run["steps"]][-2:] == ["skipped", "skipped"]
    assert run["audit_events"][-1]["metadata"]["action_taken"] is False


def test_rejection_is_audited_and_never_executes_action(client: TestClient) -> None:
    run = _create_run(
        client,
        gate_id="gate-atlas-service",
        key="triage-east-reject-01",
    ).json()
    rejected = client.post(
        f"/api/v1/agent/runs/{run['id']}/decisions",
        headers=auth("demo-operator"),
        json={
            "decision": "rejected",
            "reason": "Keep the incident open for the next shift",
            "idempotency_key": "reject-east-existing-01",
        },
    ).json()
    assert rejected["status"] == "rejected"
    assert rejected["approval"]["decision"] == "rejected"
    assert rejected["steps"][3]["status"] == "skipped"
    incident = client.get(
        "/api/v1/incidents/incident-atlas-camera",
        headers=auth("demo-viewer"),
    ).json()
    assert incident["status"] == "open"
    assert any(event["event_type"] == "approval.rejected" for event in rejected["audit_events"])


def test_decision_reason_is_trimmed_nonblank_and_retry_canonical(client: TestClient) -> None:
    run = _create_run(
        client,
        gate_id="gate-atlas-service",
        key="canonical-decision-reason-run-01",
    ).json()
    endpoint = f"/api/v1/agent/runs/{run['id']}/decisions"
    blank = client.post(
        endpoint,
        headers=auth("demo-operator"),
        json={
            "decision": "rejected",
            "reason": "    ",
            "idempotency_key": "canonical-decision-reason-01",
        },
    )
    assert blank.status_code == 422

    canonical = {
        "decision": "rejected",
        "reason": "Keep this investigation with the current shift",
        "idempotency_key": "canonical-decision-reason-01",
    }
    decided = client.post(
        endpoint,
        headers=auth("demo-operator"),
        json={**canonical, "reason": f"  {canonical['reason']}  "},
    )
    assert decided.status_code == 200
    assert decided.json()["approval"]["reason"] == canonical["reason"]
    retried = client.post(endpoint, headers=auth("demo-operator"), json=canonical)
    assert retried.status_code == 200
    assert retried.json() == decided.json()


def test_agent_permissions_tenant_scope_and_run_idempotency(client: TestClient) -> None:
    assert (
        _create_run(
            client,
            gate_id="gate-atlas-sports",
            key="viewer-cannot-run-01",
            identity="demo-viewer",
        ).status_code
        == 403
    )
    assert (
        _create_run(
            client,
            gate_id="gate-atlas-service",
            key="rif-tenant-escape-01",
            identity="demo-rif-admin",
        ).status_code
        == 404
    )

    first = _create_run(
        client,
        gate_id="gate-atlas-service",
        key="idempotent-agent-run-01",
    )
    repeated = _create_run(
        client,
        gate_id="gate-atlas-service",
        key="idempotent-agent-run-01",
    )
    assert repeated.json() == first.json()
    conflict = _create_run(
        client,
        gate_id="gate-atlas-sports",
        key="idempotent-agent-run-01",
    )
    assert conflict.status_code == 409

    run_id = first.json()["id"]
    assert (
        client.post(
            f"/api/v1/agent/runs/{run_id}/decisions",
            headers=auth("demo-viewer"),
            json={
                "decision": "approved",
                "reason": "Viewer must not be able to approve",
                "idempotency_key": "viewer-cannot-approve-01",
            },
        ).status_code
        == 403
    )
    hidden = client.get(
        f"/api/v1/agent/runs/{run_id}",
        headers=auth("demo-rif-admin"),
    )
    assert hidden.status_code == 404


def test_read_tool_failure_is_returned_as_a_durable_failed_run(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = cast(FastAPI, client.app)
    service = cast(AgentWorkflowService, application.state.agent_service)
    original = service._invoke_read_tool

    def fail_health(tool_name: AgentToolName, organization_id: str, gate_id: str) -> dict[str, Any]:
        if str(tool_name) == "get_latest_device_health":
            raise RuntimeError("simulated provider timeout")
        return original(tool_name, organization_id, gate_id)

    monkeypatch.setattr(service, "_invoke_read_tool", fail_health)
    response = _create_run(
        client,
        gate_id="gate-atlas-sports",
        key="graceful-read-failure-01",
    )
    assert response.status_code == 201
    run = response.json()
    assert run["status"] == "failed"
    assert run["failure_code"] == "read_execution_failed"
    assert run["failure_detail"] == "The read-only agent tool could not complete safely."
    assert run["steps"][1]["status"] == "failed"
    assert run["steps"][1]["error_detail"] == run["failure_detail"]
    assert "simulated provider timeout" not in str(run)
    persisted = client.get(
        f"/api/v1/agent/runs/{run['id']}",
        headers=auth("demo-viewer"),
    ).json()
    assert [event["event_type"] for event in persisted["audit_events"]].count("run.failed") == 1
    assert persisted["audit_events"][-1]["event_type"] == "run.failed"


def test_planning_failure_is_sanitized_and_terminal_once(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application = cast(FastAPI, client.app)
    service = cast(AgentWorkflowService, application.state.agent_service)

    def fail_evidence_evaluation(run_id: str, organization_id: str) -> None:
        del run_id, organization_id
        raise RuntimeError("private planner/provider diagnostic")

    monkeypatch.setattr(service, "_prepare_consequential_step", fail_evidence_evaluation)
    response = _create_run(
        client,
        gate_id="gate-atlas-sports",
        key="graceful-planning-failure-01",
    )
    assert response.status_code == 201
    run = response.json()
    assert run["status"] == "failed"
    assert run["failure_code"] == "planning_evidence_failed"
    assert run["failure_detail"] == ("The agent could not evaluate the inspected evidence safely.")
    assert "private planner/provider diagnostic" not in str(run)
    assert [event["event_type"] for event in run["audit_events"]].count("run.failed") == 1
    assert run["audit_events"][-1]["event_type"] == "run.failed"


def test_agent_evaluation_matrix_is_reproducible_and_all_scenarios_pass(
    tmp_path: Path,
) -> None:
    report = run_agent_evaluations(tmp_path / "agent-evals")
    assert report.passed == report.total == 6
    assert {result.scenario for result in report.results} == {
        "healthy_gate",
        "degraded_gate",
        "offline_gate",
        "existing_incident",
        "tenant_escape",
        "duplicate_decision",
    }
    assert all(result.passed for result in report.results)


def test_reordered_provider_plan_is_rejected_before_persistence(client: TestClient) -> None:
    application = cast(FastAPI, client.app)
    repository = cast(Repository, application.state.repository)

    class ReorderedPlanner(DeterministicGateHealthPlanner):
        name = "hostile_reordered_test_planner"

        def plan(self, request: AgentRunCreate, gate: GateRead) -> AgentPlan:
            valid = super().plan(request, gate)
            valid.steps[0], valid.steps[1] = valid.steps[1], valid.steps[0]
            valid.steps[0].sequence, valid.steps[1].sequence = 1, 2
            return valid

    service = AgentWorkflowService(repository, planner=ReorderedPlanner())
    with pytest.raises(InvalidStateError, match="required gate_health_triage tool sequence"):
        service.create_run(
            "org-atlas",
            "test-operator",
            AgentRunCreate(
                objective="Inspect this gate with a deliberately reordered provider plan",
                gate_id="gate-atlas-sports",
                intent=AgentIntent.GATE_HEALTH_TRIAGE,
                idempotency_key="hostile-reorder-01",
            ),
        )


def test_approved_effect_and_run_completion_are_atomic(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    health = client.post(
        "/api/v1/device-health",
        headers=auth("demo-edge"),
        json={
            "site_id": "site-atlas-main",
            "gate_id": "gate-atlas-north",
            "camera_id": "camera-atlas-north-anpr",
            "device_id": "camera-atlas-north-anpr",
            "device_type": "camera",
            "status": "offline",
            "detail": "Atomicity evaluation signal",
            "reported_at": "2026-08-24T11:20:00+00:00",
        },
    )
    assert health.status_code == 201
    run = _create_run(
        client,
        gate_id="gate-atlas-north",
        key="atomic-effect-run-01",
    ).json()
    before = client.get(
        "/api/v1/incidents?site_id=site-atlas-main",
        headers=auth("demo-viewer"),
    ).json()

    application = cast(FastAPI, client.app)
    repository = cast(Repository, application.state.repository)
    original = repository.create_incident_in_transaction

    def fail_after_insert(
        connection: sqlite3.Connection,
        organization_id: str,
        created_by: str,
        payload: IncidentCreate,
    ) -> IncidentRead:
        original(connection, organization_id, created_by, payload)
        raise RuntimeError("simulated process failure after domain insert")

    monkeypatch.setattr(repository, "create_incident_in_transaction", fail_after_insert)
    decision = {
        "decision": "approved",
        "reason": "Exercise the atomic approval and effect transaction",
        "idempotency_key": "atomic-effect-approval-01",
    }
    failed = client.post(
        f"/api/v1/agent/runs/{run['id']}/decisions",
        headers=auth("demo-operator"),
        json=decision,
    ).json()
    assert failed["status"] == "failed"
    assert failed["approval"]["decision"] == "approved"
    assert failed["failure_code"] == "effect_execution_failed"
    assert failed["failure_detail"] == "The approved tool could not complete safely."
    assert "simulated process failure" not in str(failed)
    assert [event["event_type"] for event in failed["audit_events"]].count("run.failed") == 1
    after = client.get(
        "/api/v1/incidents?site_id=site-atlas-main",
        headers=auth("demo-viewer"),
    ).json()
    assert len(after) == len(before)

    # Same-key retry returns the terminal failure and cannot replay the rolled-back insert.
    repeated = client.post(
        f"/api/v1/agent/runs/{run['id']}/decisions",
        headers=auth("demo-operator"),
        json=decision,
    ).json()
    assert repeated == failed


def test_same_run_key_does_not_replay_a_running_checkpoint(client: TestClient) -> None:
    _report_camera_health(
        client,
        gate_id="gate-atlas-sports",
        camera_id="camera-atlas-sports-anpr",
    )
    first = _create_run(
        client,
        gate_id="gate-atlas-sports",
        key="resume-running-checkpoint-01",
    ).json()
    assert first["status"] == "completed"

    application = cast(FastAPI, client.app)
    repository = cast(Repository, application.state.repository)
    with repository.database.transaction() as connection:
        connection.execute(
            "UPDATE agent_runs SET status = 'running' WHERE id = ?",
            (first["id"],),
        )
        connection.execute(
            "UPDATE agent_steps SET status = 'pending', output_json = NULL, started_at = NULL, "
            "completed_at = NULL WHERE run_id = ? AND sequence >= 2",
            (first["id"],),
        )

    resumed = _create_run(
        client,
        gate_id="gate-atlas-sports",
        key="resume-running-checkpoint-01",
    ).json()
    assert resumed["id"] == first["id"]
    assert resumed["status"] == "running"
    assert [step["status"] for step in resumed["steps"]] == [
        "succeeded",
        "pending",
        "pending",
        "pending",
        "pending",
    ]


def test_commit_time_guard_suppresses_a_new_duplicate_incident(client: TestClient) -> None:
    client.post(
        "/api/v1/device-health",
        headers=auth("demo-edge"),
        json={
            "site_id": "site-atlas-main",
            "gate_id": "gate-atlas-north",
            "camera_id": "camera-atlas-north-anpr",
            "device_id": "camera-atlas-north-anpr",
            "device_type": "camera",
            "status": "offline",
            "detail": "Stale-plan evaluation signal",
            "reported_at": "2026-08-24T12:20:00+00:00",
        },
    )
    run = _create_run(
        client,
        gate_id="gate-atlas-north",
        key="duplicate-precondition-run-01",
    ).json()
    assert run["pending_approval"]["tool_name"] == "create_incident"

    competing = client.post(
        "/api/v1/incidents",
        headers=auth("demo-operator"),
        json={
            "site_id": "site-atlas-main",
            "gate_id": "gate-atlas-north",
            "title": "Operator opened incident during agent handoff",
            "severity": "warning",
            "description": "A concurrent operator action invalidated the agent proposal.",
        },
    )
    assert competing.status_code == 201
    incident_count = len(
        client.get(
            "/api/v1/incidents?site_id=site-atlas-main",
            headers=auth("demo-viewer"),
        ).json()
    )

    failed = client.post(
        f"/api/v1/agent/runs/{run['id']}/decisions",
        headers=auth("demo-operator"),
        json={
            "decision": "approved",
            "reason": "Approve based on the earlier inspection evidence",
            "idempotency_key": "duplicate-precondition-approval-01",
        },
    ).json()
    assert failed["status"] == "failed"
    assert failed["failure_code"] == "effect_precondition_failed"
    assert failed["failure_detail"] == (
        "The proposed action no longer satisfies commit-time safety checks."
    )
    assert (
        len(
            client.get(
                "/api/v1/incidents?site_id=site-atlas-main",
                headers=auth("demo-viewer"),
            ).json()
        )
        == incident_count
    )


def test_concurrent_create_binds_key_to_winning_request(client: TestClient) -> None:
    application = cast(FastAPI, client.app)
    repository = cast(Repository, application.state.repository)
    rendezvous = Barrier(2)

    class BarrierPlanner(DeterministicGateHealthPlanner):
        name = "barrier_create_race_planner"

        def plan(self, request: AgentRunCreate, gate: GateRead) -> AgentPlan:
            rendezvous.wait(timeout=5)
            return super().plan(request, gate)

    service = AgentWorkflowService(repository, planner=BarrierPlanner())
    requests = (
        AgentRunCreate(
            objective="Inspect the Sports gate during the concurrent create regression",
            gate_id="gate-atlas-sports",
            intent=AgentIntent.GATE_HEALTH_TRIAGE,
            idempotency_key="concurrent-create-binding-01",
        ),
        AgentRunCreate(
            objective="Inspect the South gate during the concurrent create regression",
            gate_id="gate-atlas-south",
            intent=AgentIntent.GATE_HEALTH_TRIAGE,
            idempotency_key="concurrent-create-binding-01",
        ),
    )

    def attempt(payload: AgentRunCreate) -> tuple[str, str]:
        try:
            run = service.create_run("org-atlas", "race-operator", payload)
        except ConflictError:
            return ("conflict", "")
        return ("created", run.id)

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(attempt, requests))
    assert sorted(kind for kind, _ in outcomes) == ["conflict", "created"]
    with repository.database.connect() as connection:
        rows = connection.execute(
            "SELECT gate_id, objective FROM agent_runs WHERE organization_id = ? "
            "AND created_by = ? AND idempotency_key = ?",
            ("org-atlas", "race-operator", "concurrent-create-binding-01"),
        ).fetchall()
    assert len(rows) == 1
    assert (rows[0]["gate_id"], rows[0]["objective"]) in {
        (requests[0].gate_id, requests[0].objective),
        (requests[1].gate_id, requests[1].objective),
    }


def test_concurrent_exact_decision_retry_executes_once(client: TestClient) -> None:
    run = _create_run(
        client,
        gate_id="gate-atlas-service",
        key="concurrent-exact-decision-run-01",
    ).json()
    application = cast(FastAPI, client.app)
    service = cast(AgentWorkflowService, application.state.agent_service)
    request = AgentApprovalDecisionCreate(
        decision=AgentApprovalDecision.APPROVED,
        reason="Concurrent exact retry should share one committed result",
        idempotency_key="concurrent-exact-decision-01",
    )
    rendezvous = Barrier(2)

    def decide_once(_: int) -> tuple[str, int]:
        rendezvous.wait(timeout=5)
        result = service.decide("org-atlas", run["id"], "operator-omar", request)
        return (result.status.value, len(result.audit_events))

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(decide_once, (1, 2)))
    assert outcomes[0] == outcomes[1]
    assert outcomes[0][0] == "completed"
    persisted = service.get_run("org-atlas", run["id"])
    assert sum(event.event_type == "approval.approved" for event in persisted.audit_events) == 1
    assert sum(event.event_type == "run.completed" for event in persisted.audit_events) == 1


def test_concurrent_approvals_from_different_actors_conflict(client: TestClient) -> None:
    run = _create_run(
        client,
        gate_id="gate-atlas-service",
        key="concurrent-actor-decision-run-01",
    ).json()
    application = cast(FastAPI, client.app)
    service = cast(AgentWorkflowService, application.state.agent_service)
    rendezvous = Barrier(2)

    def decide_as(actor_id: str) -> str:
        rendezvous.wait(timeout=5)
        try:
            service.decide(
                "org-atlas",
                run["id"],
                actor_id,
                AgentApprovalDecisionCreate(
                    decision=AgentApprovalDecision.APPROVED,
                    reason="Only one approving actor may own this decision",
                    idempotency_key="concurrent-actor-decision-01",
                ),
            )
        except ConflictError:
            return "conflict"
        return "completed"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(decide_as, ("operator-omar", "admin-amal")))
    assert sorted(outcomes) == ["completed", "conflict"]
    persisted = service.get_run("org-atlas", run["id"])
    assert persisted.status.value == "completed"
    assert persisted.approval is not None
    assert persisted.approval.decided_by in {"operator-omar", "admin-amal"}


def test_concurrent_approve_reject_race_has_one_terminal_decision(client: TestClient) -> None:
    run = _create_run(
        client,
        gate_id="gate-atlas-north",
        key="concurrent-opposite-decision-run-01",
    ).json()
    assert run["pending_approval"]["tool_name"] == "create_incident"
    application = cast(FastAPI, client.app)
    service = cast(AgentWorkflowService, application.state.agent_service)
    repository = cast(Repository, application.state.repository)
    before = [
        incident
        for incident in repository.list_incidents("org-atlas", site_id="site-atlas-main")
        if incident.gate_id == "gate-atlas-north"
    ]
    rendezvous = Barrier(2)

    def decide_opposite(decision: str) -> str:
        rendezvous.wait(timeout=5)
        try:
            service.decide(
                "org-atlas",
                run["id"],
                "operator-omar" if decision == "approved" else "admin-amal",
                AgentApprovalDecisionCreate(
                    decision=AgentApprovalDecision(decision),
                    reason=f"Concurrent operator chose {decision}",
                    idempotency_key=f"concurrent-opposite-{decision}-01",
                ),
            )
        except ConflictError:
            return "conflict"
        return decision

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(decide_opposite, ("approved", "rejected")))
    assert outcomes.count("conflict") == 1
    persisted = service.get_run("org-atlas", run["id"])
    assert persisted.status.value in {"completed", "rejected"}
    assert persisted.approval is not None
    after = [
        incident
        for incident in repository.list_incidents("org-atlas", site_id="site-atlas-main")
        if incident.gate_id == "gate-atlas-north"
    ]
    expected_effects = 1 if persisted.approval.decision.value == "approved" else 0
    assert len(after) == len(before) + expected_effects


def test_health_ingestion_rejects_wrong_gate_and_phantom_camera(client: TestClient) -> None:
    wrong_gate = client.post(
        "/api/v1/device-health",
        headers=auth("demo-edge"),
        json={
            "site_id": "site-atlas-main",
            "gate_id": "gate-atlas-service",
            "camera_id": "camera-atlas-north-anpr",
            "device_id": "camera-atlas-north-anpr",
            "device_type": "camera",
            "status": "online",
            "detail": "Wrong-gate regression fixture",
            "reported_at": datetime.now(UTC).isoformat(),
        },
    )
    assert wrong_gate.status_code == 422
    phantom = client.post(
        "/api/v1/device-health",
        headers=auth("demo-edge"),
        json={
            "site_id": "site-atlas-main",
            "gate_id": "gate-atlas-north",
            "camera_id": None,
            "device_id": "phantom-camera",
            "device_type": "camera",
            "status": "online",
            "detail": "Unregistered camera regression fixture",
            "reported_at": datetime.now(UTC).isoformat(),
        },
    )
    assert phantom.status_code == 422
    future = _report_camera_health(
        client,
        gate_id="gate-atlas-south",
        camera_id="camera-atlas-south-anpr",
        reported_at=datetime.now(UTC) + timedelta(minutes=10),
    )
    assert future.status_code == 422


def test_health_readiness_requires_every_configured_camera_and_ignores_phantoms(
    client: TestClient,
) -> None:
    created = client.post(
        "/api/v1/cameras",
        headers=auth("demo-admin"),
        json={
            "site_id": "site-atlas-main",
            "gate_id": "gate-atlas-sports",
            "code": "SPORTS-OVERVIEW-02",
            "name": "Sports secondary overview",
            "role": "overview",
            "stream_profile": "secondary",
        },
    )
    assert created.status_code == 201
    _report_camera_health(
        client,
        gate_id="gate-atlas-sports",
        camera_id="camera-atlas-sports-anpr",
    )
    application = cast(FastAPI, client.app)
    repository = cast(Repository, application.state.repository)
    with repository.database.transaction() as connection:
        connection.execute(
            "INSERT INTO device_health "
            "(id, organization_id, site_id, gate_id, camera_id, device_id, device_type, "
            "status, latency_ms, detail, reported_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "health-phantom-sports",
                "org-atlas",
                "site-atlas-main",
                "gate-atlas-sports",
                None,
                "phantom-camera",
                "camera",
                "online",
                1.0,
                "Legacy phantom health row",
                datetime.now(UTC).isoformat(),
            ),
        )
    run = _create_run(
        client,
        gate_id="gate-atlas-sports",
        key="partial-camera-coverage-01",
    ).json()
    evidence = run["steps"][1]["output"]
    assert evidence["expected_count"] == 2
    assert evidence["fresh_count"] == 1
    assert evidence["ready"] is False
    assert evidence["missing_device_ids"] == [created.json()["id"]]
    assert all(report["device_id"] != "phantom-camera" for report in evidence["reports"])
    assert run["pending_approval"]["tool_name"] == "create_incident"


def test_out_of_order_health_does_not_regress_camera_state(client: TestClient) -> None:
    current_time = datetime.now(UTC)
    fresh = _report_camera_health(
        client,
        gate_id="gate-atlas-sports",
        camera_id="camera-atlas-sports-anpr",
        status="online",
        reported_at=current_time,
    )
    assert fresh.status_code == 201
    before = client.get(
        "/api/v1/cameras/camera-atlas-sports-anpr",
        headers=auth("demo-viewer"),
    ).json()
    older = _report_camera_health(
        client,
        gate_id="gate-atlas-sports",
        camera_id="camera-atlas-sports-anpr",
        status="offline",
        reported_at=current_time - timedelta(minutes=2),
    )
    assert older.status_code == 201
    after = client.get(
        "/api/v1/cameras/camera-atlas-sports-anpr",
        headers=auth("demo-viewer"),
    ).json()
    assert after["status"] == before["status"] == "online"
    assert after["last_seen_at"] == before["last_seen_at"]
    reports = client.get(
        "/api/v1/device-health?site_id=site-atlas-main&latest_only=true",
        headers=auth("demo-viewer"),
    ).json()
    selected = next(
        report for report in reports if report["device_id"] == "camera-atlas-sports-anpr"
    )
    assert selected["status"] == "online"


def test_mixed_offset_latest_health_uses_absolute_instant_everywhere(
    client: TestClient,
) -> None:
    newer_at = datetime.now(UTC) - timedelta(seconds=15)
    older_at = (newer_at - timedelta(seconds=30)).astimezone(timezone(timedelta(hours=1)))
    # This is the legacy failure shape: the older wall-clock text sorts after the newer UTC text.
    assert older_at < newer_at
    assert older_at.isoformat() > newer_at.isoformat()

    newer = _report_camera_health(
        client,
        gate_id="gate-atlas-sports",
        camera_id="camera-atlas-sports-anpr",
        status="offline",
        reported_at=newer_at,
    )
    assert newer.status_code == 201
    newer_report = newer.json()

    application = cast(FastAPI, client.app)
    repository = cast(Repository, application.state.repository)
    with repository.database.transaction() as connection:
        connection.execute(
            "INSERT INTO device_health "
            "(id, organization_id, site_id, gate_id, camera_id, device_id, device_type, "
            "status, latency_ms, detail, reported_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "health-legacy-offset-older-online",
                "org-atlas",
                "site-atlas-main",
                "gate-atlas-sports",
                "camera-atlas-sports-anpr",
                "camera-atlas-sports-anpr",
                "camera",
                "online",
                1.0,
                "Older legacy +01 report whose text sorts after UTC",
                older_at.isoformat(),
            ),
        )
        connection.executemany(
            "INSERT INTO device_health "
            "(id, organization_id, site_id, gate_id, camera_id, device_id, device_type, "
            "status, latency_ms, detail, reported_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    report_id,
                    "org-atlas",
                    "site-atlas-main",
                    "gate-atlas-sports",
                    None,
                    "edge-tie-break-probe",
                    "edge_probe",
                    status,
                    None,
                    "Equal-instant deterministic ordering fixture",
                    older_at.isoformat(),
                )
                for report_id, status in (
                    ("health-equal-instant-a", "online"),
                    ("health-equal-instant-z", "offline"),
                )
            ],
        )

    for suffix in ("?latest_only=true", "?site_id=site-atlas-main&latest_only=true"):
        response = client.get(f"/api/v1/device-health{suffix}", headers=auth("demo-viewer"))
        assert response.status_code == 200
        selected = [
            report
            for report in response.json()
            if report["device_id"] == "camera-atlas-sports-anpr"
        ]
        assert len(selected) == 1
        assert selected[0]["id"] == newer_report["id"]
        assert selected[0]["status"] == "offline"
        tied = [
            report for report in response.json() if report["device_id"] == "edge-tie-break-probe"
        ]
        assert len(tied) == 1
        assert tied[0]["id"] == "health-equal-instant-z"

    camera = client.get(
        "/api/v1/cameras/camera-atlas-sports-anpr",
        headers=auth("demo-viewer"),
    ).json()
    assert camera["status"] == "offline"
    assert camera["last_seen_at"] == newer_report["reported_at"]

    run = _create_run(
        client,
        gate_id="gate-atlas-sports",
        key="mixed-offset-latest-health-01",
    ).json()
    evidence = run["steps"][1]["output"]
    assert evidence["reports"][0]["id"] == newer_report["id"]
    assert evidence["reports"][0]["status"] == "offline"
    assert "unhealthy_status" in evidence["attention_reasons"]
    assert run["status"] == "awaiting_approval"
    assert run["pending_approval"]["tool_name"] == "create_incident"


@pytest.mark.parametrize(
    ("gate_id", "camera_id", "offset", "reason"),
    [
        ("gate-atlas-sports", "camera-atlas-sports-anpr", timedelta(minutes=-10), "stale_health"),
        ("gate-atlas-south", "camera-atlas-south-anpr", timedelta(minutes=10), "future_health"),
    ],
)
def test_stale_and_future_health_require_attention(
    client: TestClient,
    gate_id: str,
    camera_id: str,
    offset: timedelta,
    reason: str,
) -> None:
    reported_at = datetime.now(UTC) + offset
    if offset > timedelta(0):
        application = cast(FastAPI, client.app)
        repository = cast(Repository, application.state.repository)
        with repository.database.transaction() as connection:
            connection.execute(
                "INSERT INTO device_health "
                "(id, organization_id, site_id, gate_id, camera_id, device_id, device_type, "
                "status, latency_ms, detail, reported_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    f"legacy-future-{camera_id}",
                    "org-atlas",
                    "site-atlas-main",
                    gate_id,
                    camera_id,
                    camera_id,
                    "camera",
                    "online",
                    1.0,
                    "Legacy future-dated health row",
                    reported_at.isoformat(),
                ),
            )
    else:
        response = _report_camera_health(
            client,
            gate_id=gate_id,
            camera_id=camera_id,
            reported_at=reported_at,
        )
        assert response.status_code == 201
    run = _create_run(
        client,
        gate_id=gate_id,
        key=f"timestamp-attention-{reason}-01",
    ).json()
    evidence = run["steps"][1]["output"]
    assert evidence["ready"] is False
    assert reason in evidence["attention_reasons"]
    assert run["pending_approval"]["tool_name"] == "create_incident"


def test_incident_selection_skips_assigned_newer_item_for_actionable_one(
    client: TestClient,
) -> None:
    newer = client.post(
        "/api/v1/incidents",
        headers=auth("demo-operator"),
        json={
            "site_id": "site-atlas-main",
            "gate_id": "gate-atlas-service",
            "title": "Already owned parallel investigation",
            "severity": "warning",
            "description": "Newer assigned incident should not hide the actionable seed incident.",
        },
    ).json()
    client.patch(
        f"/api/v1/incidents/{newer['id']}",
        headers=auth("demo-operator"),
        json={"status": "investigating", "assigned_to": "operator-other"},
    )
    run = _create_run(
        client,
        gate_id="gate-atlas-service",
        key="select-actionable-incident-01",
    ).json()
    assert run["pending_approval"]["tool_name"] == "start_incident_investigation"
    assert run["steps"][3]["input"]["incident_id"] == "incident-atlas-camera"


def test_all_assigned_incidents_complete_without_reassignment_or_duplicate(
    client: TestClient,
) -> None:
    updated = client.patch(
        "/api/v1/incidents/incident-atlas-camera",
        headers=auth("demo-operator"),
        json={"status": "investigating", "assigned_to": "operator-existing"},
    )
    assert updated.status_code == 200
    run = _create_run(
        client,
        gate_id="gate-atlas-service",
        key="assigned-incident-safe-complete-01",
    ).json()
    assert run["status"] == "completed"
    assert run["pending_approval"] is None
    assert [step["status"] for step in run["steps"]][-2:] == ["skipped", "skipped"]
