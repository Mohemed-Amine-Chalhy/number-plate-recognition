import assert from "node:assert/strict";
import test from "node:test";

import {
  CampusApi,
  gateCameraHealth,
  normalizeAccessRequests,
  normalizeDevices,
  normalizeGates,
  normalizeIncidents,
} from "../api.mjs";
import { DEMO_DATA } from "../demo-data.mjs";

function jsonResponse(payload, status = 200) {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

test("API requests include demo bearer auth and explicit tenant scope", async () => {
  let captured;
  const api = new CampusApi(
    {
      baseUrl: "/api/v1/",
      organizationId: "org-atlas",
      tokenProvider: () => "demo-operator",
    },
    async (url, options) => {
      captured = { url, options };
      return jsonResponse({ subject: "operator-omar" });
    },
  );
  await api.request("/session");
  assert.equal(captured.url, "/api/v1/session");
  assert.equal(captured.options.headers.Authorization, "Bearer demo-operator");
  assert.equal(captured.options.headers["X-Organization-ID"], "org-atlas");
});

test("API invokes browser fetch without rebinding its receiver", async () => {
  let receiver = "not-called";
  async function browserStyleFetch() {
    receiver = this;
    return jsonResponse({ ok: true });
  }

  const api = new CampusApi({ baseUrl: "/api/v1" }, browserStyleFetch);
  await api.request("/session");
  assert.equal(receiver, undefined);
});

test("access decision maps UI denial to the backend rejected contract", async () => {
  let captured;
  const api = new CampusApi({ baseUrl: "/api/v1" }, async (url, options) => {
    captured = { url, options };
    return jsonResponse({ request: {}, grant: null });
  });
  await api.decideRequest("request/unsafe", "denied");
  assert.equal(captured.url, "/api/v1/access-requests/request%2Funsafe/decision");
  assert.deepEqual(JSON.parse(captured.options.body), {
    decision: "rejected",
    reason: "Decision recorded from the campus access console",
  });
});

test("resource normalizers map the typed API vocabulary into UI states", () => {
  const gates = normalizeGates(
    [
      {
        id: "gate-a",
        code: "A",
        name: "Arrival",
        status: "operational",
        queue_estimate: 3,
        latitude: 32.231,
        longitude: -7.947,
      },
    ],
    DEMO_DATA.gates,
  );
  assert.equal(gates[0].status, "open");
  assert.equal(gates[0].queue, 3);
  assert.equal(gates[0].latitude, 32.231);
  assert.equal(gates[0].longitude, -7.947);
  assert.equal(gates[0].x, undefined);
  assert.equal(gates[0].y, undefined);
  assert.equal(gates[0].throughput, "—");
  assert.equal(gates[0].operator, "—");
  assert.equal(gates[0].cameraHealth, "—");
  assert.equal(gates[0].plate, "—");
  assert.equal(gates[0].confidence, "—");

  const [unconfiguredGate] = normalizeGates(
    [{ id: "gate-unconfigured", status: "operational" }],
    DEMO_DATA.gates,
  );
  assert.deepEqual(
    {
      id: unconfiguredGate.id,
      code: unconfiguredGate.code,
      name: unconfiguredGate.name,
      zone: unconfiguredGate.zone,
      operator: unconfiguredGate.operator,
      throughput: unconfiguredGate.throughput,
      cameraHealth: unconfiguredGate.cameraHealth,
    },
    {
      id: "gate-unconfigured",
      code: "gate-unconfigured",
      name: "gate-unconfigured",
      zone: "—",
      operator: "—",
      throughput: "—",
      cameraHealth: "—",
    },
  );

  const requests = normalizeAccessRequests(
    [
      {
        id: "request-a",
        requested_for_name: "Visitor",
        status: "rejected",
        created_at: new Date().toISOString(),
      },
    ],
    DEMO_DATA.requests,
  );
  assert.equal(requests[0].status, "denied");

  assert.equal(normalizeIncidents([{ severity: "critical" }])[0].severity, "high");
  assert.equal(normalizeDevices([{ status: "offline" }])[0].status, "maintenance");
  assert.equal(
    gateCameraHealth("gate-a", [
      { gateId: "gate-a", type: "Plate camera", status: "online" },
      { gateId: "gate-a", type: "Context camera", status: "degraded" },
      { gateId: "gate-a", type: "Barrier", status: "maintenance" },
    ]),
    84,
  );
  assert.equal(gateCameraHealth("gate-missing", []), null);
});

test("gate normalization matches seeded map positions by stable identity, not response order", () => {
  const seededPosition = (gateId) => {
    const gate = DEMO_DATA.gates.find(({ id }) => id === gateId);
    assert.ok(gate, `missing seeded gate ${gateId}`);
    return [gate.x, gate.y];
  };
  const gates = normalizeGates(
    [
      { id: "gate-logistics", code: "G04", status: "operational" },
      { id: "gate-sports", code: "G06", status: "operational" },
      { id: "gate-main", code: "G01", status: "operational" },
      { id: "gate-south", code: "G05", status: "operational" },
      { id: "gate-new", code: "G99", status: "operational" },
    ],
    DEMO_DATA.gates,
  );

  assert.deepEqual(
    gates.map(({ x, y }) => [x, y]),
    [
      seededPosition("gate-logistics"),
      seededPosition("gate-sports"),
      seededPosition("gate-main"),
      seededPosition("gate-south"),
      [undefined, undefined],
    ],
  );
});

test("snapshot falls back completely when the API is unavailable", async () => {
  const api = new CampusApi({ baseUrl: "/api/v1", timeoutMs: 50 }, async () => {
    throw new TypeError("connection refused");
  });
  const snapshot = await api.loadSnapshot(DEMO_DATA);
  assert.equal(snapshot.source, "demo");
  assert.equal(snapshot.resourcesLoaded, 0);
  assert.equal(snapshot.data.gates.length, 6);
});

test("snapshot reports hybrid state and normalizes live gate resources", async () => {
  const api = new CampusApi({ baseUrl: "/api/v1" }, async (url) => {
    if (url.endsWith("/gates")) {
      return jsonResponse([
        {
          id: "gate-live",
          code: "NORTH-EAST",
          name: "Live gate",
          direction: "inbound",
          status: "congested",
          queue_estimate: 8,
        },
      ]);
    }
    if (url.includes("/device-health")) {
      return jsonResponse([
        {
          id: "health-live",
          gate_id: "gate-live",
          device_id: "camera-live",
          device_type: "camera",
          status: "online",
          latency_ms: 44,
        },
      ]);
    }
    return jsonResponse({ detail: "not ready" }, 503);
  });
  const snapshot = await api.loadSnapshot(DEMO_DATA);
  assert.equal(snapshot.source, "hybrid");
  assert.equal(snapshot.data.gates[0].id, "gate-live");
  assert.equal(snapshot.data.gates[0].status, "degraded");
  assert.equal(snapshot.data.gates[0].throughput, "—");
  assert.equal(snapshot.data.gates[0].operator, "—");
  assert.equal(snapshot.data.gates[0].cameraHealth, 100);
});

test("gate commands stay disabled until an actuator route is configured", async () => {
  const api = new CampusApi({ baseUrl: "/api/v1" }, async () => jsonResponse({ ok: true }));
  assert.equal(api.supportsGateCommands, false);
  await assert.rejects(() => api.gateCommand("gate-a", "open"), /not configured/);
});
