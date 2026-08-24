import assert from "node:assert/strict";
import test from "node:test";

import {
  buildCampusMapModel,
  clampPercent,
  projectCoordinates,
  resolveGateMapLabel,
  resolveGateMapPosition,
  resolveSelectedGate,
} from "../campus-map.mjs";
import { TENANT_CONFIG } from "../config.mjs";
import { DEMO_DATA } from "../demo-data.mjs";

test("tenant map configuration references the versioned illustration and stable approaches", () => {
  assert.equal(TENANT_CONFIG.map.assetUrl, "./assets/campus-map-illustrated-v2.webp");
  assert.equal(TENANT_CONFIG.map.aspectRatio, "3 / 2");
  const approaches = [
    ["gate-main", "gate-atlas-north", { x: 56, y: 6 }],
    ["gate-innovation", "gate-atlas-research", { x: 68, y: 18 }],
    ["gate-logistics", "gate-atlas-service", { x: 85, y: 42 }],
    ["gate-residential", "gate-atlas-residence", { x: 55, y: 83 }],
    ["gate-south", "gate-atlas-south", { x: 39, y: 94 }],
    ["gate-sports", "gate-atlas-sports", { x: 21, y: 65 }],
  ];
  for (const [demoId, liveId, expected] of approaches) {
    assert.deepEqual(TENANT_CONFIG.map.gatePositions[demoId], expected);
    assert.deepEqual(TENANT_CONFIG.map.gatePositions[liveId], expected);
  }
  assert.equal(TENANT_CONFIG.map.gateLabels["gate-atlas-research"], "G02");
  assert.equal(TENANT_CONFIG.map.landmarks.length, 4);
  assert.equal(new Set(TENANT_CONFIG.map.landmarks.map(({ id }) => id)).size, 4);
  assert.ok(Object.isFrozen(TENANT_CONFIG.map));
});

test("map labels are configurable and safely compact long API codes", () => {
  assert.equal(
    resolveGateMapLabel(
      { id: "gate-atlas-research", code: "NORTH-EAST" },
      TENANT_CONFIG.map,
    ),
    "G02",
  );
  assert.equal(resolveGateMapLabel({ id: "unknown", code: "NORTH-EAST" }), "NE");
  assert.equal(resolveGateMapLabel({ id: "unknown", code: "SPORTS" }), "SPO");
  assert.equal(resolveGateMapLabel({ id: "unknown", code: "LIVE" }), "LIVE");
});

test("six-gate fixtures keep map, device, and analytics topology aligned", () => {
  const gateIds = new Set(DEMO_DATA.gates.map(({ id }) => id));
  assert.equal(gateIds.size, 6);
  assert.ok(
    DEMO_DATA.gates.every(
      ({ id, name, zone }) =>
        TENANT_CONFIG.map.gatePositions[id] &&
        ["en", "fr", "ar"].every((locale) => name[locale] && zone[locale]),
    ),
  );

  const deviceGateIds = new Set(DEMO_DATA.devices.map(({ gateId }) => gateId));
  assert.deepEqual(deviceGateIds, gateIds);
  assert.equal(DEMO_DATA.devices.length, 12);
  assert.equal(DEMO_DATA.devices.filter(({ status }) => status === "online").length, 9);

  assert.deepEqual(
    new Set(DEMO_DATA.analytics.gateTraffic.map(({ gateId }) => gateId)),
    gateIds,
  );
  assert.equal(
    DEMO_DATA.analytics.gateTraffic.reduce((total, { value }) => total + value, 0),
    100,
  );
});

test("map percentages and geographic projection are bounded", () => {
  assert.equal(clampPercent(-3), 0);
  assert.equal(clampPercent(108), 100);
  assert.equal(clampPercent("not-a-number", 42), 42);

  const bounds = { north: 34, east: -6, south: 32, west: -8 };
  assert.deepEqual(projectCoordinates(33, -7, bounds), {
    x: 50,
    y: 50,
    source: "geographic",
  });
  assert.deepEqual(projectCoordinates(35, -9, bounds), {
    x: 0,
    y: 0,
    source: "geographic",
  });
  assert.equal(projectCoordinates(null, -7, bounds), null);
  assert.equal(projectCoordinates(33, -7, { ...bounds, north: 31 }), null);
  assert.equal(projectCoordinates(33, -7, { ...bounds, north: 94 }), null);
});

test("tenant positions take priority and respect the configured edge inset", () => {
  const config = {
    edgeInset: 8,
    bounds: { north: 34, east: -6, south: 32, west: -8 },
    gatePositions: {
      "gate-main": { x: 2, y: 98 },
      G02: { x: 25, y: 30 },
    },
  };
  assert.deepEqual(resolveGateMapPosition({ id: "gate-main", x: 60, y: 60 }, config), {
    x: 8,
    y: 92,
    source: "configured",
  });
  assert.deepEqual(resolveGateMapPosition({ id: "another", code: "G02" }, config), {
    x: 25,
    y: 30,
    source: "configured",
  });
  assert.equal(
    resolveGateMapPosition({ id: "geo", latitude: 33, longitude: -7 }, config).source,
    "geographic",
  );
});

test("gate-owned positions remain compatible with deterministic demo records", () => {
  assert.deepEqual(resolveGateMapPosition({ id: "gate-a", x: 23, y: 71 }), {
    x: 23,
    y: 71,
    source: "gate",
  });
  assert.deepEqual(resolveGateMapPosition({ id: "gate-a", mapPosition: { x: 31, y: 47 } }), {
    x: 31,
    y: 47,
    source: "gate",
  });
  assert.equal(
    resolveGateMapPosition({ id: "gate-a", mapPosition: { x: null }, x: 24, y: 65 }).x,
    24,
  );
});

test("fallback slots are stable across API response ordering and never overlap", () => {
  const gates = ["foxtrot", "bravo", "echo", "alpha", "delta", "charlie"].map((id) => ({ id }));
  const forward = buildCampusMapModel(gates, {});
  const reversed = buildCampusMapModel([...gates].reverse(), {});
  const positions = (model) =>
    new Map(model.gates.map(({ gate, position }) => [gate.id, `${position.x}:${position.y}`]));

  assert.deepEqual(positions(forward), positions(reversed));
  assert.equal(new Set(positions(forward).values()).size, gates.length);
  assert.ok(forward.gates.every(({ position }) => position.source === "fallback"));
});

test("selection falls back safely as demo and live gate sets change", () => {
  const gates = [{ id: "gate-a" }, { id: "gate-b" }];
  assert.equal(resolveSelectedGate(gates, "gate-b"), gates[1]);
  assert.equal(resolveSelectedGate(gates, "missing"), gates[0]);
  assert.equal(resolveSelectedGate([], "missing"), null);

  const model = buildCampusMapModel(gates, {}, "gate-b");
  assert.equal(model.selectedGate, gates[1]);
  assert.deepEqual(
    model.gates.map(({ selected }) => selected),
    [false, true],
  );
});
