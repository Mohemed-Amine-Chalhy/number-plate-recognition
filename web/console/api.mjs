import { mergeSnapshot } from "./core.mjs?v=0.1.1";

/** The browser adapter mirrors the control API's public v1 resource names. */
const RESOURCE_ROUTES = Object.freeze({
  dashboard: "/dashboard",
  session: "/session",
  organizations: "/organizations",
  sites: "/sites",
  gates: "/gates",
  cameras: "/cameras",
  accessRequests: "/access-requests",
  accessGrants: "/access-grants",
  passages: "/passages?limit=20",
  events: "/events?after_sequence=0&limit=50",
  incidents: "/incidents",
  devices: "/device-health?latest_only=true&limit=100",
});

function unwrap(payload, key) {
  if (payload == null) return null;
  if (Object.hasOwn(payload, key)) return payload[key];
  if (Object.hasOwn(payload, "data")) return payload.data;
  return payload;
}

function asArray(value) {
  if (Array.isArray(value)) return value;
  if (Array.isArray(value?.items)) return value.items;
  return null;
}

function gateStatus(status) {
  if (status === "operational") return "open";
  if (status === "congested" || status === "degraded") return "degraded";
  if (status === "offline" || status === "disabled") return "maintenance";
  return status || "open";
}

function decisionStatus(status) {
  if (status === "rejected") return "denied";
  if (status === "review_required" || status === "no_match") return "review";
  if (status === "allowed") return "approved";
  return status || "pending";
}

function minutesSince(isoDate) {
  const timestamp = Date.parse(isoDate);
  if (!Number.isFinite(timestamp)) return 0;
  return Math.max(0, Math.round((Date.now() - timestamp) / 60_000));
}

export function normalizeGates(items, seedGates) {
  const positions = [
    [20, 71],
    [75, 22],
    [76, 72],
    [15, 22],
  ];
  return (asArray(items) ?? []).map((gate, index) => {
    const seed = seedGates[index % seedGates.length] ?? {};
    const [x, y] = positions[index % positions.length];
    return {
      ...seed,
      id: gate.id ?? seed.id,
      code: gate.code ?? seed.code,
      name: gate.name ?? seed.name,
      zone: gate.direction ?? seed.zone,
      status: gateStatus(gate.status),
      x,
      y,
      queue: gate.queue_estimate ?? seed.queue ?? 0,
      waitMinutes: Math.max(0, Math.ceil((gate.queue_estimate ?? seed.queue ?? 0) / 2)),
      lanes: gate.direction === "bidirectional" ? 2 : seed.lanes ?? 1,
    };
  });
}

export function normalizeAccessRequests(items, seedRequests) {
  return (asArray(items) ?? []).map((request, index) => {
    const seed = seedRequests[index % seedRequests.length] ?? {};
    return {
      ...seed,
      id: request.id ?? seed.id,
      person: request.requested_for_name ?? seed.person,
      plate: request.plate_text ?? seed.plate ?? "—",
      host: request.requested_by ?? seed.host,
      window:
        request.valid_from && request.valid_until
          ? `${new Date(request.valid_from).toLocaleDateString()} · ${new Date(
              request.valid_from,
            ).toLocaleTimeString([], {
              hour: "2-digit",
              minute: "2-digit",
            })}–${new Date(request.valid_until).toLocaleTimeString([], {
              hour: "2-digit",
              minute: "2-digit",
            })}`
          : seed.window,
      reason: request.purpose ?? seed.reason,
      risk: request.status === "pending" ? "medium" : "low",
      status: decisionStatus(request.status),
      submittedMinutes: minutesSince(request.created_at),
    };
  });
}

export function normalizeIncidents(items) {
  return (asArray(items) ?? []).map((incident) => ({
    id: incident.id,
    severity:
      incident.severity === "critical"
        ? "high"
        : incident.severity === "warning"
          ? "medium"
          : "low",
    title: incident.title,
    description: incident.description,
    gateId: incident.gate_id,
    status: incident.status,
    owner: incident.assigned_to ?? incident.created_by ?? "Unassigned",
    minutesAgo: minutesSince(incident.created_at),
  }));
}

export function normalizeDevices(items) {
  return (asArray(items) ?? []).map((device) => ({
    id: device.device_id ?? device.id,
    gateId: device.gate_id,
    type: device.device_type ?? "Device",
    status:
      device.status === "offline" || device.status === "unknown"
        ? "maintenance"
        : device.status ?? "maintenance",
    latency: Math.round(device.latency_ms ?? 0),
    uptime: device.status === "online" ? 99.98 : device.status === "degraded" ? 97.4 : 95,
    detail: device.detail,
  }));
}

function normalizeArrivals(events, seedArrivals) {
  return (asArray(events) ?? [])
    .filter((event) => event.passage_id || String(event.event_type).startsWith("authorization"))
    .map((event, index) => {
      const seed = seedArrivals[index % seedArrivals.length] ?? {};
      const outcome = event.metadata?.outcome ?? event.event_type?.split(".").at(-1);
      const confidence = event.metadata?.confidence;
      return {
        ...seed,
        id: event.id ?? seed.id,
        gateId: event.gate_id ?? seed.gateId,
        plate: event.metadata?.plate_text ?? "—",
        person: event.summary ?? seed.person,
        organization: event.source ?? "Campus event stream",
        purpose: event.event_type ?? seed.purpose,
        decision: decisionStatus(outcome),
        confidence:
          confidence == null ? 0 : Math.round(Number(confidence) * 1000) / 10,
        minutesAgo: minutesSince(event.occurred_at),
        color: "Event stream",
        avatar: String(event.source ?? "EV").slice(0, 2).toUpperCase(),
      };
    });
}

function normalizeDirectory(grants, seedDirectory) {
  const liveRecords = (asArray(grants) ?? []).map((grant) => ({
    id: grant.id,
    kind: grant.plate_text ? "vehicle" : "person",
    name: grant.subject_name,
    plate: grant.plate_text,
    organization: grant.subject_kind,
    status: grant.status === "revoked" ? "blocked" : grant.status,
    access: grant.gate_id ? `Gate ${grant.gate_id}` : "Campus-wide grant",
    initials: grant.plate_text
      ? grant.plate_text.slice(-2)
      : grant.subject_name
          ?.split(/\s+/)
          .map((word) => word[0])
          .join("")
          .slice(0, 2),
  }));
  return liveRecords.length ? liveRecords : seedDirectory;
}

function uiResources(raw, seed) {
  const dashboard = raw.dashboard ?? {};
  const gates = asArray(raw.gates) ?? asArray(dashboard.gates);
  const incidents = asArray(raw.incidents) ?? asArray(dashboard.open_incidents);
  const devices = asArray(raw.devices) ?? asArray(dashboard.device_health);
  const events = asArray(raw.events) ?? asArray(dashboard.recent_events);
  const resources = {};

  if (gates?.length) resources.gates = normalizeGates(gates, seed.gates);
  const arrivals = normalizeArrivals(events, seed.arrivals);
  if (arrivals.length) resources.arrivals = arrivals;
  const requests = normalizeAccessRequests(raw.accessRequests, seed.requests);
  if (requests.length) resources.requests = requests;
  resources.directory = normalizeDirectory(raw.accessGrants, seed.directory);
  const normalizedIncidents = normalizeIncidents(incidents);
  if (normalizedIncidents.length) resources.incidents = normalizedIncidents;
  const normalizedDevices = normalizeDevices(devices);
  if (normalizedDevices.length) resources.devices = normalizedDevices;
  if (dashboard.counts) {
    const analyticsGates = resources.gates ?? seed.gates;
    const trafficTotal = analyticsGates.reduce(
      (total, gate) => total + Math.max(1, Number(gate.throughput) || Number(gate.queue) || 1),
      0,
    );
    resources.analytics = {
      ...seed.analytics,
      totalEntries: dashboard.counts.passages_today ?? seed.analytics.totalEntries,
      gateTraffic: analyticsGates.map((gate) => ({
        gateId: gate.id,
        value: Math.round(
          (Math.max(1, Number(gate.throughput) || Number(gate.queue) || 1) / trafficTotal) * 100,
        ),
      })),
    };
  }
  const site = asArray(raw.sites)?.[0];
  resources.meta = {
    ...seed.meta,
    campusName: site?.name ?? seed.meta.campusName,
    session: raw.session ?? null,
    organization: asArray(raw.organizations)?.[0] ?? null,
  };
  return resources;
}

export class CampusApi {
  constructor(
    {
      baseUrl,
      timeoutMs = 1800,
      organizationId = null,
      tokenProvider = () => "",
      gateCommandPath = null,
    },
    fetchImplementation = globalThis.fetch,
  ) {
    this.baseUrl = String(baseUrl).replace(/\/$/, "");
    this.timeoutMs = timeoutMs;
    this.organizationId = organizationId;
    this.tokenProvider = tokenProvider;
    this.gateCommandPath = gateCommandPath;
    this.fetchImplementation = fetchImplementation;
  }

  get supportsGateCommands() {
    return Boolean(this.gateCommandPath);
  }

  async request(path, options = {}) {
    if (typeof this.fetchImplementation !== "function") throw new Error("Fetch is unavailable");
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      const token = this.tokenProvider?.();
      const authHeaders = token ? { Authorization: `Bearer ${token}` } : {};
      const organizationHeaders = this.organizationId
        ? { "X-Organization-ID": this.organizationId }
        : {};
      const response = await this.fetchImplementation(`${this.baseUrl}${path}`, {
        ...options,
        headers: {
          Accept: "application/json",
          ...authHeaders,
          ...organizationHeaders,
          ...options.headers,
        },
        signal: controller.signal,
      });
      if (!response.ok) throw new Error(`API returned ${response.status}`);
      return await response.json();
    } finally {
      clearTimeout(timeout);
    }
  }

  async loadSnapshot(seed) {
    const entries = Object.entries(RESOURCE_ROUTES);
    const settled = await Promise.allSettled(
      entries.map(async ([key, route]) => [key, unwrap(await this.request(route), key)]),
    );
    const raw = {};
    let successCount = 0;
    settled.forEach((result) => {
      if (result.status !== "fulfilled") return;
      const [key, value] = result.value;
      if (value == null) return;
      raw[key] = value;
      successCount += 1;
    });
    return {
      data: mergeSnapshot(seed, uiResources(raw, seed)),
      source:
        successCount === 0 ? "demo" : successCount === entries.length ? "live" : "hybrid",
      resourcesLoaded: successCount,
      resourcesTotal: entries.length,
    };
  }

  async gateCommand(gateId, command) {
    if (!this.gateCommandPath) throw new Error("Gate actuator endpoint is not configured");
    const path = this.gateCommandPath.replace("{gateId}", encodeURIComponent(gateId));
    return this.request(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ command }),
    });
  }

  async decideRequest(requestId, decision) {
    return this.request(`/access-requests/${encodeURIComponent(requestId)}/decision`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        decision: decision === "denied" ? "rejected" : decision,
        reason: "Decision recorded from the campus access console",
      }),
    });
  }

  async acknowledgeIncident(incidentId, assignedTo = null) {
    return this.request(`/incidents/${encodeURIComponent(incidentId)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status: "investigating", assigned_to: assignedTo }),
    });
  }

  async demoIdentities() {
    return this.request("/demo-identities");
  }
}
