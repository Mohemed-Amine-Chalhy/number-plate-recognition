import assert from "node:assert/strict";
import test from "node:test";

import {
  AGENT_TOOL_POLICY,
  agentDecisionWasCommitted,
  buildAgentDecisionRequest,
  buildAgentRunRequest,
  canApproveAgentRun,
  createIdempotencyKey,
  createReferenceAgentRun,
  decideAgentRunWithRecovery,
  deriveEvidenceCoverage,
  normalizeAgentRun,
  prepareAgentRunDraft,
  summarizeAgentEvidence,
} from "../agentic.mjs";
import { DEMO_DATA } from "../demo-data.mjs";

function referenceSnapshot({ gateStatus = "open", devices, incidents = [] } = {}) {
  return {
    meta: { generatedAt: "2026-08-25T10:00:00Z" },
    gates: [{ id: "gate-test", code: "G09", name: { en: "Test gate" }, status: gateStatus }],
    devices:
      devices === undefined
        ? [{ id: "CAM-G09", gateId: "gate-test", status: "online" }]
        : devices,
    incidents,
  };
}

function consequentialSteps(run) {
  return run.steps.filter(({ risk }) => risk === "consequential");
}

test("reference workflow matches the backend plan order and approval contract", () => {
  const run = createReferenceAgentRun(DEMO_DATA, { gateId: "gate-residential" });

  assert.equal(run.mode, "reference");
  assert.equal(run.gateId, "gate-residential");
  assert.equal(run.status, "awaiting_approval");
  assert.deepEqual(
    run.plan.steps.map(({ toolName }) => toolName),
    [
      "get_gate",
      "get_latest_device_health",
      "list_open_gate_incidents",
      "start_incident_investigation",
      "create_incident",
    ],
  );
  assert.deepEqual(
    run.steps.slice(0, 3).map(({ risk, status }) => [risk, status]),
    [
      ["read_only", "succeeded"],
      ["read_only", "succeeded"],
      ["read_only", "succeeded"],
    ],
  );
  assert.equal(run.steps[3].status, "awaiting_approval");
  assert.equal(run.steps[3].input.incident_id, "INC-0412");
  assert.equal(run.pendingApproval.stepId, run.steps[3].id);
  assert.equal(run.steps[4].status, "skipped");
  assert.deepEqual(
    run.steps[3].policyChecks.map(({ name, outcome }) => [name, outcome]),
    [
      ["tool_allowlisted", "allow"],
      ["organization_scope", "allow"],
      ["gate_scope", "allow"],
      ["human_approval", "approval_required"],
    ],
  );
  assert.equal(canApproveAgentRun(run), true);
  assert.equal(deriveEvidenceCoverage(run), 100);
});

test("reference trajectory handles each incident and health branch safely", async (t) => {
  await t.test("the first unassigned unresolved incident is actionable", () => {
    const snapshot = referenceSnapshot({
      incidents: [
        {
          id: "INC-OWNED",
          gateId: "gate-test",
          status: "open",
          owner: "Desk one",
        },
        {
          id: "INC-UNASSIGNED",
          gateId: "gate-test",
          status: "investigating",
          owner: "Unassigned",
        },
      ],
    });
    const run = createReferenceAgentRun(snapshot, { gateId: "gate-test" });
    assert.equal(run.pendingApproval.toolName, "start_incident_investigation");
    assert.equal(run.steps[3].input.incident_id, "INC-UNASSIGNED");
  });

  await t.test("assigned investigating incident suppresses both action branches", () => {
    const run = createReferenceAgentRun(
      referenceSnapshot({
        gateStatus: "degraded",
        incidents: [
          {
            id: "INC-OWNED",
            gateId: "gate-test",
            status: "investigating",
            owner: "operator-7",
          },
        ],
      }),
      { gateId: "gate-test" },
    );
    assert.equal(run.status, "completed");
    assert.equal(run.pendingApproval, null);
    assert.deepEqual(consequentialSteps(run).map(({ status }) => status), ["skipped", "skipped"]);
  });

  await t.test("resolved incidents do not suppress a needed creation", () => {
    const run = createReferenceAgentRun(
      referenceSnapshot({
        devices: [{ id: "CAM-G09", gateId: "gate-test", status: "degraded" }],
        incidents: [{ id: "INC-DONE", gateId: "gate-test", status: "resolved" }],
      }),
      { gateId: "gate-test" },
    );
    assert.equal(run.pendingApproval.toolName, "create_incident");
    assert.equal(run.steps[4].policyChecks.at(-1).outcome, "approval_required");
  });

  await t.test("healthy evidence completes without a consequential proposal", () => {
    const run = createReferenceAgentRun(referenceSnapshot(), { gateId: "gate-test" });
    assert.equal(run.status, "completed");
    assert.deepEqual(consequentialSteps(run).map(({ status }) => status), ["skipped", "skipped"]);
  });

  await t.test("gate-only degradation proposes creation", () => {
    const run = createReferenceAgentRun(referenceSnapshot({ gateStatus: "degraded" }), {
      gateId: "gate-test",
    });
    assert.equal(run.pendingApproval.toolName, "create_incident");
  });

  await t.test("missing device evidence proposes creation", () => {
    const run = createReferenceAgentRun(referenceSnapshot({ devices: [] }), {
      gateId: "gate-test",
    });
    assert.equal(run.pendingApproval.toolName, "create_incident");
    assert.equal(run.steps[1].output.device_count, 0);
  });
});

test("live contract normalization preserves trace, evidence and policy checks", () => {
  const run = normalizeAgentRun({
    run: {
      id: "run-live",
      gate_id: "gate-live",
      objective: "Triage the gate",
      status: "completed",
      trace: {
        trace_id: "trace-live",
        correlation_id: "corr-live",
        planner_name: "bounded-planner",
        planner_version: "2",
        policy_name: "bounded-campus-operations",
        policy_version: "3",
      },
      plan: {
        summary: "Read then draft",
        steps: [{ sequence: 1, tool_name: "get_gate", risk: "read_only", rationale: "Scope" }],
      },
      steps: [
        {
          id: "step-live",
          sequence: 1,
          tool_name: "get_gate",
          risk: "read_only",
          status: "succeeded",
          input: { gate_id: "gate-live" },
          output: { queue: 4 },
          policy_checks: [
            {
              code: "gate_scope",
              outcome: "allow",
              detail: "Gate belongs to organization scope",
              policy_name: "bounded-campus-operations",
              policy_version: "3",
            },
          ],
        },
      ],
      approval: {
        decision: "approved",
        reason: "Exact operator rationale",
        decided_by: "operator-7",
      },
      audit_events: [
        {
          event_type: "agent.approval.approved",
          actor_type: "human",
          actor_id: "operator-7",
          summary: "Approved",
          metadata: { gate_id: "gate-live" },
        },
      ],
    },
  });

  assert.equal(run.mode, "live");
  assert.equal(run.trace.traceId, "trace-live");
  assert.equal(run.trace.planner, "bounded-planner");
  assert.equal(run.trace.policy, "bounded-campus-operations");
  assert.deepEqual(run.steps[0].output, { queue: 4 });
  assert.deepEqual(run.steps[0].policyChecks[0], {
    id: "check-1",
    name: "gate_scope",
    outcome: "allow",
    passed: true,
    blocked: false,
    detail: "Gate belongs to organization scope",
    policyName: "bounded-campus-operations",
    policyVersion: "3",
  });
  assert.deepEqual(run.approval, {
    decision: "approved",
    reason: "Exact operator rationale",
    decidedBy: "operator-7",
    decidedAt: null,
  });
  assert.equal(run.auditEvents[0].actorType, "human");
  assert.deepEqual(run.auditEvents[0].metadata, { gate_id: "gate-live" });
});

test("unknown tools, risks, and policy outcomes fail closed", () => {
  const run = normalizeAgentRun({
    id: "run-drift",
    status: "awaiting_approval",
    plan: { steps: [{ sequence: 1, tool_name: "surprise_actuator", risk: "safe-ish" }] },
    steps: [
      {
        id: "step-drift",
        sequence: 1,
        tool_name: "surprise_actuator",
        risk: "safe-ish",
        status: "awaiting_approval",
        policy_checks: [
          { code: "future_policy", outcome: "maybe", passed: true, detail: "new value" },
        ],
      },
    ],
    pending_approval: { step_id: "step-drift", tool_name: "surprise_actuator" },
  });

  assert.equal(run.plan.steps[0].toolKnown, false);
  assert.equal(run.steps[0].risk, "unknown");
  assert.equal(run.steps[0].policyChecks[0].outcome, "unknown");
  assert.equal(run.steps[0].policyChecks[0].blocked, true);
  assert.equal(run.contractBlocked, true);
  assert.equal(canApproveAgentRun(run), false);
});

test("failure contracts are normalized and diagnostics are redacted", () => {
  const run = normalizeAgentRun({
    id: "run-failed",
    status: "failed",
    failure_code: "run_dependency_failed",
    failure_detail: "authorization: bearer-secret token=top-secret",
    steps: [
      {
        id: "step-failed",
        tool_name: "get_gate",
        risk: "read_only",
        status: "failed",
        error_code: "tool_dependency_failed",
        error_detail: "Bearer abc.def credential=private-value",
      },
    ],
  });

  assert.equal(run.failureCode, "run_dependency_failed");
  assert.equal(run.steps[0].errorCode, "tool_dependency_failed");
  assert.doesNotMatch(`${run.failureDetail} ${run.steps[0].errorDetail}`, /top-secret|abc\.def|private-value/);
  assert.match(`${run.failureDetail} ${run.steps[0].errorDetail}`, /\[redacted\]/);
});

test("agent requests normalize decision reasons and keep durable idempotency keys", () => {
  assert.deepEqual(
    buildAgentRunRequest({ objective: "  Inspect the gate  ", gateId: "gate-a", idempotencyKey: "run-fixed" }),
    {
      objective: "Inspect the gate",
      gate_id: "gate-a",
      intent: "gate_health_triage",
      idempotency_key: "run-fixed",
    },
  );
  assert.deepEqual(
    buildAgentDecisionRequest({
      decision: "approved",
      reason: "  Reviewed by operator  ",
      idempotencyKey: "decision-fixed",
    }),
    {
      decision: "approved",
      reason: "Reviewed by operator",
      idempotency_key: "decision-fixed",
    },
  );
  assert.throws(() => buildAgentDecisionRequest({ decision: "execute", idempotencyKey: "bad" }), /approved or rejected/);
  assert.equal(createIdempotencyKey("agent-run", () => "uuid-fixed"), "agent-run-uuid-fixed");
  const decisionKey = createIdempotencyKey(
    "agent-decision",
    () => "12345678-1234-1234-1234-123456789abc",
  );
  assert.equal(decisionKey, "agent-decision-12345678-1234-1234-1234-123456789abc");
  assert.ok(decisionKey.length <= 80);
  assert.deepEqual(
    AGENT_TOOL_POLICY.tools.map(({ name, risk }) => [name, risk]),
    [
      ["get_gate", "read_only"],
      ["get_latest_device_health", "read_only"],
      ["list_open_gate_incidents", "read_only"],
      ["start_incident_investigation", "consequential"],
      ["create_incident", "consequential"],
    ],
  );
});

test("run drafts reuse one key after an ambiguous create failure", () => {
  let generated = 0;
  const makeKey = () => `run-key-${++generated}`;
  const first = prepareAgentRunDraft(null, { objective: "Inspect", gateId: "gate-a" }, makeKey);
  const retry = prepareAgentRunDraft({ ...first, status: "failed" }, { objective: "Inspect", gateId: "gate-a" }, makeKey);
  const changed = prepareAgentRunDraft(retry, { objective: "Inspect again", gateId: "gate-a" }, makeKey);

  assert.equal(first.idempotencyKey, "run-key-1");
  assert.equal(retry.idempotencyKey, first.idempotencyKey);
  assert.equal(retry.status, "pending");
  assert.equal(changed.idempotencyKey, "run-key-2");
});

test("decision timeout after commit reconciles the exact reason without a second effect", async () => {
  const originalError = new Error("request timed out");
  const payload = buildAgentDecisionRequest({
    decision: "approved",
    reason: "  Camera evidence checked by shift lead.  ",
    idempotencyKey: "decision-stable",
  });
  let decisionCalls = 0;
  let readCalls = 0;
  const api = {
    async decideAgentRun(_runId, submitted) {
      decisionCalls += 1;
      assert.equal(submitted.idempotency_key, "decision-stable");
      throw originalError;
    },
    async agentRun() {
      readCalls += 1;
      return {
        id: "run-1",
        status: "completed",
        approval: {
          decision: "approved",
          reason: "Camera evidence checked by shift lead.",
          decided_by: "operator-7",
        },
      };
    },
  };

  const result = await decideAgentRunWithRecovery(api, "run-1", payload);
  assert.equal(result.recovered, true);
  assert.equal(result.run.approval.reason, payload.reason);
  assert.equal(agentDecisionWasCommitted(result.run, "approved", payload.reason), true);
  assert.equal(decisionCalls, 1);
  assert.equal(readCalls, 1);
});

test("decision reconciliation rethrows the original timeout when the reason differs", async () => {
  const originalError = new Error("ambiguous timeout");
  const api = {
    async decideAgentRun() {
      throw originalError;
    },
    async agentRun() {
      return {
        id: "run-2",
        status: "completed",
        approval: { decision: "approved", reason: "Different operation" },
      };
    },
  };
  await assert.rejects(
    () =>
      decideAgentRunWithRecovery(api, "run-2", {
        decision: "approved",
        reason: "Expected operation",
        idempotency_key: "decision-stable",
      }),
    (error) => error === originalError,
  );
});

test("decision reconciliation never masks a deterministic client conflict", async () => {
  const conflict = new Error("API returned 409");
  let readCalls = 0;
  const api = {
    async decideAgentRun() {
      throw conflict;
    },
    async agentRun() {
      readCalls += 1;
      return { approval: { decision: "approved", reason: "Same words" } };
    },
  };
  await assert.rejects(
    () =>
      decideAgentRunWithRecovery(api, "run-3", {
        decision: "approved",
        reason: "Same words",
        idempotency_key: "decision-conflict",
      }),
    (error) => error === conflict,
  );
  assert.equal(readCalls, 0);
});

test("evidence coverage starts at zero and is unavailable without read steps", () => {
  assert.equal(
    deriveEvidenceCoverage({
      steps: [{ risk: "read_only", status: "pending", output: {}, policyChecks: [] }],
    }),
    0,
  );
  assert.equal(deriveEvidenceCoverage({ steps: [] }), null);
});

test("live nested tool outputs become concise safe evidence instead of object coercions", () => {
  const run = normalizeAgentRun({
    id: "run-nested",
    status: "awaiting_approval",
    steps: [
      {
        id: "gate-step",
        sequence: 1,
        tool_name: "get_gate",
        risk: "read_only",
        status: "succeeded",
        output: {
          gate: {
            id: "gate-atlas-residence",
            code: "G03",
            status: "degraded",
            queue_estimate: 7,
            metadata: { private_note: "do not render" },
          },
        },
      },
      {
        id: "health-step",
        sequence: 2,
        tool_name: "get_latest_device_health",
        risk: "read_only",
        status: "succeeded",
        output: {
          count: 2,
          reports: [
            { device_id: "CAM-G03-B", status: "degraded", metadata: { provider: "edge" } },
            { device_id: "BAR-G03-A", status: "online", authorization: "never-render" },
          ],
        },
      },
      {
        id: "incident-step",
        sequence: 3,
        tool_name: "list_open_gate_incidents",
        risk: "read_only",
        status: "succeeded",
        output: {
          count: 2,
          incidents: [
            { id: "INC-0412", status: "investigating", description: "private detail" },
            { id: "INC-0411", status: "assigned", metadata: { reporter: "private" } },
          ],
        },
      },
    ],
  });

  const facts = run.steps.map((step) => summarizeAgentEvidence(step));
  assert.deepEqual(facts[0], [
    { key: "gate_code", value: "G03" },
    { key: "status", value: "degraded" },
    { key: "queue", value: 7 },
  ]);
  assert.deepEqual(facts[1], [
    { key: "device_count", value: 2 },
    { key: "report_statuses", value: "degraded × 1 · online × 1" },
    { key: "device_ids", value: "CAM-G03-B, BAR-G03-A" },
  ]);
  assert.deepEqual(facts[2], [
    { key: "open_incident_count", value: 2 },
    { key: "incident_statuses", value: "investigating × 1 · assigned × 1" },
    { key: "incident_ids", value: "INC-0412, INC-0411" },
  ]);
  const renderedValues = facts.flat().map(({ value }) => String(value)).join(" · ");
  assert.doesNotMatch(renderedValues, /\[object Object\]/);
  assert.doesNotMatch(renderedValues, /private|never-render|authorization|metadata/);
});
