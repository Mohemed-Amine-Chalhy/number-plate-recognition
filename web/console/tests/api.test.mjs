import assert from "node:assert/strict";
import test from "node:test";

import {
  CampusApi,
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
    [{ id: "gate-a", code: "A", name: "Arrival", status: "operational", queue_estimate: 3 }],
    DEMO_DATA.gates,
  );
  assert.equal(gates[0].status, "open");
  assert.equal(gates[0].queue, 3);

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
});

test("snapshot falls back completely when the API is unavailable", async () => {
  const api = new CampusApi({ baseUrl: "/api/v1", timeoutMs: 50 }, async () => {
    throw new TypeError("connection refused");
  });
  const snapshot = await api.loadSnapshot(DEMO_DATA);
  assert.equal(snapshot.source, "demo");
  assert.equal(snapshot.resourcesLoaded, 0);
  assert.equal(snapshot.data.gates.length, 4);
});

test("snapshot reports hybrid state and normalizes live gate resources", async () => {
  const api = new CampusApi({ baseUrl: "/api/v1" }, async (url) => {
    if (url.endsWith("/gates")) {
      return jsonResponse([
        {
          id: "gate-live",
          code: "LIVE",
          name: "Live gate",
          direction: "inbound",
          status: "congested",
          queue_estimate: 8,
        },
      ]);
    }
    return jsonResponse({ detail: "not ready" }, 503);
  });
  const snapshot = await api.loadSnapshot(DEMO_DATA);
  assert.equal(snapshot.source, "hybrid");
  assert.equal(snapshot.data.gates[0].id, "gate-live");
  assert.equal(snapshot.data.gates[0].status, "degraded");
});

test("gate commands stay disabled until an actuator route is configured", async () => {
  const api = new CampusApi({ baseUrl: "/api/v1" }, async () => jsonResponse({ ok: true }));
  assert.equal(api.supportsGateCommands, false);
  await assert.rejects(() => api.gateCommand("gate-a", "open"), /not configured/);
});
