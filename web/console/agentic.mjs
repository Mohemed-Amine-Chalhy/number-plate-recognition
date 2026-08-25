/**
 * Agent-run view model and deterministic reference workflow.
 *
 * The browser never executes tools directly. It submits a fixed intent and
 * retains the objective as operator context, then renders the server's plan,
 * trace, evidence and approval boundary. The reference workflow mirrors that
 * contract for offline demos.
 */

export const AGENT_INTENT = "gate_health_triage";

export const AGENT_TOOL_POLICY = Object.freeze({
  version: "2026-08-25",
  tools: Object.freeze([
    Object.freeze({ name: "get_gate", risk: "read_only" }),
    Object.freeze({ name: "get_latest_device_health", risk: "read_only" }),
    Object.freeze({ name: "list_open_gate_incidents", risk: "read_only" }),
    Object.freeze({ name: "start_incident_investigation", risk: "consequential" }),
    Object.freeze({ name: "create_incident", risk: "consequential" }),
  ]),
});

const AGENT_TOOL_NAMES = new Set(AGENT_TOOL_POLICY.tools.map(({ name }) => name));

const RUN_STATUSES = new Set([
  "idle",
  "running",
  "awaiting_approval",
  "completed",
  "rejected",
  "failed",
]);
const STEP_STATUSES = new Set([
  "pending",
  "running",
  "awaiting_approval",
  "succeeded",
  "skipped",
  "failed",
]);
const RISKS = new Set(["read_only", "consequential"]);
const POLICY_OUTCOMES = new Set(["allow", "approval_required", "deny"]);

function safeArray(value) {
  return Array.isArray(value) ? value : [];
}

function safeObject(value) {
  return value && typeof value === "object" && !Array.isArray(value) ? value : {};
}

export function sanitizeAgentDiagnostic(value, maximumLength = 420) {
  return String(value ?? "")
    .replace(/(bearer\s+)[\w.~+/=-]+/gi, "$1[redacted]")
    .replace(
      /((?:token|secret|password|credential|authorization)\s*[:=]\s*)[^\s,;]+/gi,
      "$1[redacted]",
    )
    .slice(0, maximumLength);
}

const SENSITIVE_OUTPUT_KEY = /token|secret|password|credential|authorization|cookie|description|email|phone|metadata/i;

function safeIdentifierList(records, keys) {
  return records
    .map((record) => keys.map((key) => record?.[key]).find(Boolean))
    .filter(Boolean)
    .map(String);
}

function statusSummary(records, statusFormatter) {
  const counts = new Map();
  records.forEach((record) => {
    const status = String(record?.status ?? "unknown");
    counts.set(status, (counts.get(status) ?? 0) + 1);
  });
  return [...counts.entries()]
    .map(([status, count]) => `${statusFormatter(status)} × ${count}`)
    .join(" · ");
}

/**
 * Reduce structured tool output to a small, explicitly allowlisted fact set.
 * This prevents accidental raw-object rendering and avoids surfacing verbose
 * descriptions, metadata, credentials, or unrelated provider fields.
 */
export function summarizeAgentEvidence(step, options = {}) {
  const output = safeObject(step?.output);
  const statusFormatter = options.statusFormatter ?? ((status) => status);
  const noneLabel = options.noneLabel ?? "none";
  const structuredLabel = options.structuredLabel ?? "structured result available";
  const recordLabel = options.recordLabel ?? ((count) => `${count} records`);
  const values = Object.values(output);

  // Preserve the compact, already-flat reference trajectory exactly.
  if (values.every((value) => value == null || typeof value !== "object")) {
    return Object.entries(output).map(([key, value]) => ({ key, value }));
  }

  if (step?.toolName === "get_gate" && output.gate) {
    const gate = safeObject(output.gate);
    return [
      { key: "gate_code", value: gate.code ?? gate.id ?? noneLabel },
      { key: "status", value: gate.status ?? "unknown" },
      { key: "queue", value: Number(gate.queue_estimate ?? gate.queue ?? 0) },
    ];
  }

  if (step?.toolName === "get_latest_device_health" && Array.isArray(output.reports)) {
    const reports = output.reports.map(safeObject);
    const deviceIds = safeIdentifierList(reports, ["device_id", "id"]);
    return [
      { key: "device_count", value: Number(output.count ?? reports.length) },
      { key: "report_statuses", value: statusSummary(reports, statusFormatter) || noneLabel },
      { key: "device_ids", value: deviceIds.join(", ") || noneLabel },
    ];
  }

  if (step?.toolName === "list_open_gate_incidents" && Array.isArray(output.incidents)) {
    const incidents = output.incidents.map(safeObject);
    const incidentIds = safeIdentifierList(incidents, ["id"]);
    return [
      { key: "open_incident_count", value: Number(output.count ?? incidents.length) },
      { key: "incident_statuses", value: statusSummary(incidents, statusFormatter) || noneLabel },
      { key: "incident_ids", value: incidentIds.join(", ") || noneLabel },
    ];
  }

  if (
    ["create_incident", "start_incident_investigation"].includes(step?.toolName) &&
    output.incident
  ) {
    const incident = safeObject(output.incident);
    return [
      { key: "incident_ids", value: incident.id ?? noneLabel },
      { key: "incident_statuses", value: statusFormatter(incident.status ?? "unknown") },
    ];
  }

  const facts = [];
  Object.entries(output).forEach(([key, value]) => {
    if (SENSITIVE_OUTPUT_KEY.test(key) || value == null) return;
    if (["string", "number", "boolean"].includes(typeof value)) {
      facts.push({ key, value });
    } else if (Array.isArray(value)) {
      const primitives = value.filter((item) => item == null || typeof item !== "object");
      facts.push({
        key,
        value: primitives.length === value.length ? primitives.join(", ") || noneLabel : recordLabel(value.length),
      });
    } else {
      facts.push({ key, value: structuredLabel });
    }
  });
  return facts.slice(0, 5);
}

function normalizeStatus(value, allowed, fallback) {
  const status = String(value ?? "").toLowerCase();
  return allowed.has(status) ? status : fallback;
}

function normalizeRisk(value) {
  const risk = String(value ?? "").toLowerCase();
  return RISKS.has(risk) ? risk : "unknown";
}


function normalizePolicyOutcome(value, fallback = "unknown") {
  const outcome = String(value ?? "").toLowerCase();
  return POLICY_OUTCOMES.has(outcome) ? outcome : fallback;
}

function normalizePolicyChecks(value) {
  if (Array.isArray(value)) {
    return value.map((check, index) => {
      if (typeof check === "string") {
        return {
          id: `check-${index + 1}`,
          name: check,
          outcome: "unknown",
          passed: false,
          blocked: true,
          detail: "Unstructured policy result",
          policyName: null,
          policyVersion: null,
        };
      }
      const inferredOutcome =
        check?.passed === false || check?.status === "failed"
          ? "deny"
          : check?.passed === true || check?.status === "passed"
            ? "allow"
            : "unknown";
      const hasExplicitOutcome = check?.outcome != null && String(check.outcome).trim() !== "";
      const outcome = hasExplicitOutcome
        ? normalizePolicyOutcome(check.outcome)
        : inferredOutcome;
      return {
        id: check?.id ?? `check-${index + 1}`,
        name: check?.code ?? check?.name ?? check?.policy ?? "policy_check",
        outcome,
        passed: outcome === "allow" || outcome === "approval_required",
        blocked: outcome === "deny" || outcome === "unknown",
        detail: sanitizeAgentDiagnostic(check?.detail ?? ""),
        policyName: check?.policy_name ?? check?.policyName ?? null,
        policyVersion: check?.policy_version ?? check?.policyVersion ?? null,
      };
    });
  }
  return Object.entries(safeObject(value)).map(([name, value], index) => {
    const outcome = normalizePolicyOutcome(
      value,
      value === true || value === "passed" ? "allow" : value === false ? "deny" : "unknown",
    );
    return {
      id: `check-${index + 1}`,
      name,
      outcome,
      passed: outcome === "allow" || outcome === "approval_required",
      blocked: outcome === "deny" || outcome === "unknown",
      detail: "",
      policyName: null,
      policyVersion: null,
    };
  });
}

function normalizePlanStep(step, index) {
  const toolName = step?.tool_name ?? step?.toolName ?? "unknown_tool";
  return {
    sequence: Number(step?.sequence) || index + 1,
    toolName,
    toolKnown: AGENT_TOOL_NAMES.has(toolName),
    risk: normalizeRisk(step?.risk),
    rationale: step?.rationale ?? "",
  };
}

function normalizeExecutedStep(step, index) {
  const toolName = step?.tool_name ?? step?.toolName ?? "unknown_tool";
  return {
    id: step?.id ?? `step-${index + 1}`,
    sequence: Number(step?.sequence) || index + 1,
    toolName,
    toolKnown: AGENT_TOOL_NAMES.has(toolName),
    risk: normalizeRisk(step?.risk),
    status: normalizeStatus(step?.status, STEP_STATUSES, "failed"),
    input: safeObject(step?.input),
    output: safeObject(step?.output),
    policyChecks: normalizePolicyChecks(step?.policy_checks ?? step?.policyChecks),
    startedAt: step?.started_at ?? step?.startedAt ?? null,
    completedAt: step?.completed_at ?? step?.completedAt ?? null,
    errorCode: sanitizeAgentDiagnostic(step?.error_code ?? step?.errorCode ?? "", 96) || null,
    errorDetail: sanitizeAgentDiagnostic(
      step?.error_detail ?? step?.errorDetail ?? step?.error ?? "",
    ) || null,
  };
}

function normalizePendingApproval(value) {
  if (!value) return null;
  return {
    stepId: value.step_id ?? value.stepId ?? null,
    toolName: value.tool_name ?? value.toolName ?? "create_incident",
    reason: value.reason ?? "",
    requestedAt: value.requested_at ?? value.requestedAt ?? null,
  };
}

function normalizeAuditEvent(event, index) {
  return {
    id: event?.id ?? `audit-${index + 1}`,
    eventType: event?.event_type ?? event?.eventType ?? event?.type ?? "agent.event",
    actorType: event?.actor_type ?? event?.actorType ?? "agent",
    actorId: event?.actor_id ?? event?.actorId ?? event?.actor ?? event?.created_by ?? "unknown",
    occurredAt: event?.occurred_at ?? event?.created_at ?? event?.occurredAt ?? null,
    summary: event?.summary ?? event?.detail ?? event?.reason ?? "",
    metadata: safeObject(event?.metadata),
  };
}

/** Normalize the live API response without inventing missing execution evidence. */
export function normalizeAgentRun(payload) {
  const raw = payload?.run ?? payload?.data ?? payload ?? {};
  const plan = safeObject(raw.plan);
  const trace = safeObject(raw.trace);
  const approval = raw.approval ? safeObject(raw.approval) : null;
  const normalized = {
    id: raw.id ?? "agent-run-unavailable",
    organizationId: raw.organization_id ?? raw.organizationId ?? null,
    siteId: raw.site_id ?? raw.siteId ?? null,
    gateId: raw.gate_id ?? raw.gateId ?? null,
    objective: raw.objective ?? "",
    intent: raw.intent ?? AGENT_INTENT,
    status: normalizeStatus(raw.status, RUN_STATUSES, "failed"),
    failureCode: sanitizeAgentDiagnostic(raw.failure_code ?? raw.failureCode ?? "", 96) || null,
    failureDetail: sanitizeAgentDiagnostic(raw.failure_detail ?? raw.failureDetail ?? "") || null,
    createdBy: raw.created_by ?? raw.createdBy ?? "agent",
    createdAt: raw.created_at ?? raw.createdAt ?? null,
    updatedAt: raw.updated_at ?? raw.updatedAt ?? null,
    trace: {
      traceId: trace.trace_id ?? trace.traceId ?? null,
      correlationId: trace.correlation_id ?? trace.correlationId ?? null,
      planner: trace.planner_name ?? trace.plannerName ?? trace.planner ?? "unknown",
      plannerVersion: trace.planner_version ?? trace.plannerVersion ?? "unknown",
      policy: trace.policy_name ?? trace.policyName ?? "unknown",
      policyVersion: trace.policy_version ?? trace.policyVersion ?? "unknown",
    },
    plan: {
      summary: plan.summary ?? "",
      steps: safeArray(plan.steps).map(normalizePlanStep),
    },
    steps: safeArray(raw.steps).map(normalizeExecutedStep),
    pendingApproval: normalizePendingApproval(raw.pending_approval ?? raw.pendingApproval),
    approval: approval
      ? {
          decision: approval.decision ?? null,
          reason: approval.reason ?? "",
          decidedBy: approval.decided_by ?? approval.decidedBy ?? null,
          decidedAt: approval.decided_at ?? approval.decidedAt ?? null,
        }
      : null,
    auditEvents: safeArray(raw.audit_events ?? raw.auditEvents).map(normalizeAuditEvent),
    mode: raw.mode === "reference" ? "reference" : "live",
  };
  normalized.contractBlocked = [
    ...normalized.plan.steps,
    ...normalized.steps,
  ].some((step) => !step.toolKnown || step.risk === "unknown") || normalized.steps.some(
    (step) => step.policyChecks.some((check) => check.blocked),
  );
  return normalized;
}

export function canApproveAgentRun(run) {
  if (!run || run.status !== "awaiting_approval" || !run.pendingApproval || run.contractBlocked) {
    return false;
  }
  const step = safeArray(run.steps).find(({ id }) => id === run.pendingApproval.stepId);
  if (
    !step ||
    !step.toolKnown ||
    step.risk !== "consequential" ||
    step.status !== "awaiting_approval"
  ) {
    return false;
  }
  const checks = new Map(step.policyChecks.map((check) => [check.name, check]));
  return (
    checks.get("tool_allowlisted")?.outcome === "allow" &&
    checks.get("organization_scope")?.outcome === "allow" &&
    checks.get("gate_scope")?.outcome === "allow" &&
    checks.get("human_approval")?.outcome === "approval_required"
  );
}

function selectReferenceGate(snapshot, requestedGateId) {
  const gates = safeArray(snapshot?.gates);
  return (
    gates.find((gate) => gate?.id === requestedGateId) ??
    gates.find((gate) => gate?.status === "degraded") ??
    gates[0] ??
    null
  );
}

function localizedName(value) {
  if (value && typeof value === "object") return value.en ?? Object.values(value)[0] ?? "Gate";
  return value ?? "Gate";
}

/**
 * Produce the same auditable shape as the live service from the checked-in
 * snapshot. Outputs are computed from the snapshot rather than presented as
 * live observations.
 */
export function createReferenceAgentRun(snapshot, options = {}) {
  const gate = selectReferenceGate(snapshot, options.gateId);
  const gateId = gate?.id ?? "gate-unavailable";
  const devices = safeArray(snapshot?.devices).filter((device) => device?.gateId === gateId);
  const incidents = safeArray(snapshot?.incidents).filter(
    (incident) => incident?.gateId === gateId && incident?.status !== "resolved",
  );
  const attentionDevices = devices.filter((device) => device?.status !== "online");
  // The service returns unresolved incidents newest-first. Select its first
  // unassigned open/investigating record; never reassign an owned incident.
  const actionableIncident = incidents.find((incident) => {
    const status = String(incident?.status ?? "").toLowerCase();
    const assignee = incident?.assigned_to ?? incident?.assignedTo ?? incident?.owner;
    const normalizedAssignee = String(assignee ?? "").trim().toLowerCase();
    const assigned = Boolean(normalizedAssignee && normalizedAssignee !== "unassigned");
    return ["open", "investigating"].includes(status) && !assigned;
  });
  const objective =
    String(options.objective ?? "").trim() ||
    `Triage gate health and prepare a human-reviewed response for ${localizedName(gate?.name)}`;
  const runId = `reference-${gateId}`;
  const createdAt = snapshot?.meta?.generatedAt ?? null;
  const hasUnresolvedIncident = incidents.length > 0;
  const gateNeedsAttention = ["congested", "degraded", "maintenance", "offline"].includes(
    String(gate?.status ?? "").toLowerCase(),
  );
  const unhealthyEvidence = devices.length === 0 || attentionDevices.length > 0 || gateNeedsAttention;
  const needsNewIncident = !hasUnresolvedIncident && unhealthyEvidence;
  const hasConsequence = Boolean(actionableIncident) || needsNewIncident;
  const policyCheck = (code, outcome = "allow") => ({
    code,
    outcome,
    detail:
      outcome === "approval_required"
        ? "A human decision is required before this step can execute."
        : "Reference policy condition satisfied.",
    policy_name: "bounded-campus-operations",
    policy_version: AGENT_TOOL_POLICY.version,
  });
  const planSteps = [
    {
      sequence: 1,
      tool_name: "get_gate",
      risk: "read_only",
      rationale: "Establish queue, wait and operating state inside the selected gate scope.",
    },
    {
      sequence: 2,
      tool_name: "get_latest_device_health",
      risk: "read_only",
      rationale: "Correlate edge-device health without invoking an actuator.",
    },
    {
      sequence: 3,
      tool_name: "list_open_gate_incidents",
      risk: "read_only",
      rationale: "Avoid duplicating an incident and retain operational context.",
    },
  ];
  planSteps.push(
    {
      sequence: 4,
      tool_name: "start_incident_investigation",
      risk: "consequential",
      rationale: "Prefer an existing incident and pause before changing its assignment or status.",
    },
    {
      sequence: 5,
      tool_name: "create_incident",
      risk: "consequential",
      rationale: "Create a new record only when device evidence needs attention and no open incident exists.",
    },
  );

  const executedSteps = [
    {
      id: `${runId}-gate`,
      sequence: 1,
      tool_name: "get_gate",
      risk: "read_only",
      status: "succeeded",
      input: { gate_id: gateId },
      output: {
        gate_code: gate?.code ?? "—",
        status: gate?.status ?? "unknown",
        queue: Number(gate?.queue ?? 0),
        wait_minutes: Number(gate?.waitMinutes ?? 0),
      },
      policy_checks: [
        policyCheck("tool_allowlisted"),
        policyCheck("organization_scope"),
        policyCheck("gate_scope"),
      ],
      started_at: createdAt,
      completed_at: createdAt,
    },
    {
      id: `${runId}-devices`,
      sequence: 2,
      tool_name: "get_latest_device_health",
      risk: "read_only",
      status: "succeeded",
      input: { gate_id: gateId, latest_only: true },
      output: {
        device_count: devices.length,
        attention_count: attentionDevices.length,
        attention_devices: attentionDevices.map((device) => device.id).join(", ") || "none",
      },
      policy_checks: [
        policyCheck("tool_allowlisted"),
        policyCheck("organization_scope"),
        policyCheck("gate_scope"),
      ],
      started_at: createdAt,
      completed_at: createdAt,
    },
    {
      id: `${runId}-incidents`,
      sequence: 3,
      tool_name: "list_open_gate_incidents",
      risk: "read_only",
      status: "succeeded",
      input: { gate_id: gateId, open_only: true },
      output: {
        open_incident_count: incidents.length,
        incident_ids: incidents.map((incident) => incident.id).join(", ") || "none",
      },
      policy_checks: [
        policyCheck("tool_allowlisted"),
        policyCheck("organization_scope"),
        policyCheck("gate_scope"),
      ],
      started_at: createdAt,
      completed_at: createdAt,
    },
  ];
  const evidenceRefs = executedSteps.map((step) => step.id);
  executedSteps.push(
    {
      id: `${runId}-investigation`,
      sequence: 4,
      tool_name: "start_incident_investigation",
      risk: "consequential",
      status: actionableIncident ? "awaiting_approval" : "skipped",
      input: actionableIncident
        ? {
            gate_id: gateId,
            incident_id: actionableIncident.id,
            evidence_refs: evidenceRefs,
          }
        : { gate_id: gateId },
      output: {},
      policy_checks: actionableIncident
        ? [
            policyCheck("tool_allowlisted"),
            policyCheck("organization_scope"),
            policyCheck("gate_scope"),
            policyCheck("human_approval", "approval_required"),
          ]
        : [],
      started_at: createdAt,
    },
    {
      id: `${runId}-create`,
      sequence: 5,
      tool_name: "create_incident",
      risk: "consequential",
      status: needsNewIncident ? "awaiting_approval" : "skipped",
      input: needsNewIncident
        ? {
            gate_id: gateId,
            title: "Gate health triage handoff",
            evidence_refs: evidenceRefs,
          }
        : { gate_id: gateId },
      output: {},
      policy_checks: needsNewIncident
        ? [
            policyCheck("tool_allowlisted"),
            policyCheck("organization_scope"),
            policyCheck("gate_scope"),
            policyCheck("human_approval", "approval_required"),
          ]
        : [],
      started_at: createdAt,
    },
  );
  const pendingStep = executedSteps.find((step) => step.status === "awaiting_approval") ?? null;

  return normalizeAgentRun({
    id: runId,
    gate_id: gateId,
    objective,
    intent: AGENT_INTENT,
    status: hasConsequence ? "awaiting_approval" : "completed",
    created_by: "reference-workflow",
    created_at: createdAt,
    updated_at: createdAt,
    mode: "reference",
    trace: {
      trace_id: `trace-${gateId}`,
      correlation_id: `reference-${gateId}`,
      planner_name: "deterministic-gate-triage",
      planner_version: AGENT_TOOL_POLICY.version,
      policy_name: "bounded-campus-operations",
      policy_version: AGENT_TOOL_POLICY.version,
    },
    plan: {
      summary: hasConsequence
        ? "Correlate gate, device and incident evidence, then pause at the operational handoff."
        : "Correlate gate, device and incident evidence; no operational handoff is required.",
      steps: planSteps,
    },
    steps: executedSteps,
    pending_approval: hasConsequence
      ? {
          step_id: pendingStep.id,
          tool_name: pendingStep.tool_name,
          reason: actionableIncident
            ? "Starting an investigation changes incident assignment and status, so a person must decide."
            : "Creating an operational incident changes shared records, so a person must decide.",
          requested_at: createdAt,
        }
      : null,
    audit_events: [
      {
        id: `${runId}-audit-1`,
        event_type: "agent.plan.created",
        actor_type: "agent",
        actor_id: "reference-workflow",
        summary: "Bounded gate-health plan created from reference data.",
        metadata: { gate_id: gateId },
        occurred_at: createdAt,
      },
      {
        id: `${runId}-audit-2`,
        event_type: hasConsequence ? "agent.approval.requested" : "agent.run.completed",
        actor_type: "service",
        actor_id: "policy-engine",
        summary: hasConsequence
          ? "Consequential step paused for a human decision."
          : "Read-only run completed without a proposed mutation.",
        metadata: { gate_id: gateId },
        occurred_at: createdAt,
      },
    ],
  });
}

/** Coverage measures available execution evidence, not recommendation correctness or authority. */
export function deriveEvidenceCoverage(run) {
  const readable = safeArray(run?.steps).filter((step) => step.risk === "read_only");
  if (!readable.length) return null;
  const completion = readable.filter((step) => step.status === "succeeded").length / readable.length;
  const withEvidence = readable.filter((step) => Object.keys(safeObject(step.output)).length > 0).length / readable.length;
  const checks = readable.flatMap((step) => safeArray(step.policyChecks));
  const policyCoverage = checks.length
    ? checks.filter((check) => check.outcome === "allow").length / checks.length
    : 0;
  return Math.round(Math.min(100, completion * 55 + withEvidence * 30 + policyCoverage * 15));
}

export function prepareAgentRunDraft(existing, { objective, gateId }, idempotencyKeyFactory) {
  const normalizedObjective = String(objective ?? "").trim();
  if (
    existing &&
    existing.objective === normalizedObjective &&
    existing.gateId === gateId &&
    ["pending", "failed"].includes(existing.status)
  ) {
    return { ...existing, status: "pending" };
  }
  return {
    objective: normalizedObjective,
    gateId,
    idempotencyKey: idempotencyKeyFactory(),
    status: "pending",
  };
}

export function agentDecisionWasCommitted(run, decision, reason = null) {
  if (run?.approval?.decision !== decision) return false;
  return reason == null || run.approval.reason === reason;
}

function isAmbiguousDecisionFailure(error) {
  const message = String(error?.message ?? "");
  // A deterministic client rejection cannot be a timeout-after-commit. Keep
  // 4xx conflicts/validation failures visible instead of accepting a possibly
  // unrelated earlier approval that happens to carry the same text.
  return !/\bAPI returned 4\d\d\b/i.test(message);
}

export async function decideAgentRunWithRecovery(api, runId, payload) {
  try {
    return { run: normalizeAgentRun(await api.decideAgentRun(runId, payload)), recovered: false };
  } catch (decisionError) {
    if (!isAmbiguousDecisionFailure(decisionError)) throw decisionError;
    try {
      const reconciled = normalizeAgentRun(await api.agentRun(runId));
      if (agentDecisionWasCommitted(reconciled, payload.decision, payload.reason)) {
        return { run: reconciled, recovered: true };
      }
    } catch {
      // The original error remains the most useful retry signal.
    }
    throw decisionError;
  }
}

export function buildAgentRunRequest({ objective, gateId, idempotencyKey }) {
  return {
    objective: String(objective ?? "").trim(),
    gate_id: gateId,
    intent: AGENT_INTENT,
    idempotency_key: idempotencyKey,
  };
}

export function buildAgentDecisionRequest({ decision, reason, idempotencyKey }) {
  if (decision !== "approved" && decision !== "rejected") {
    throw new TypeError("Agent decision must be approved or rejected");
  }
  return {
    decision,
    reason: String(reason ?? "").trim(),
    idempotency_key: idempotencyKey,
  };
}

export function createIdempotencyKey(prefix, randomUUID, now = Date.now()) {
  const suffix = typeof randomUUID === "function" ? randomUUID() : String(now);
  return `${prefix}-${suffix}`;
}
