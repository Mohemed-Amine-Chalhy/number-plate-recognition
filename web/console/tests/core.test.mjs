import assert from "node:assert/strict";
import test from "node:test";

import {
  chartScale,
  escapeHTML,
  filterDirectory,
  gateSummary,
  isRTL,
  mergeSnapshot,
  normalizeLocale,
  resolveRoute,
  resolveAuthToken,
  safeStorage,
  translate,
} from "../core.mjs";
import { MESSAGES } from "../i18n.mjs";

test("locale and route inputs are normalized to supported values", () => {
  assert.equal(normalizeLocale("fr-CA"), "fr");
  assert.equal(normalizeLocale("de"), "en");
  assert.equal(isRTL("ar-MA"), true);
  assert.equal(isRTL("fr"), false);
  assert.equal(resolveRoute("#/operations?site=main"), "operations");
  assert.equal(resolveRoute("#/not-a-route"), "command");
});

test("translation falls back to English and interpolates variables", () => {
  assert.equal(translate(MESSAGES, "nav.command", "fr"), "Centre de contrôle");
  assert.equal(translate(MESSAGES, "time.minutes", "ar", { count: 4 }), "منذ 4 دقائق");
  assert.equal(translate(MESSAGES, "missing.key", "ar"), "missing.key");
});

test("French and Arabic dictionaries cover every English interface key", () => {
  const referenceKeys = Object.keys(MESSAGES.en).sort();
  assert.deepEqual(Object.keys(MESSAGES.fr).sort(), referenceKeys);
  assert.deepEqual(Object.keys(MESSAGES.ar).sort(), referenceKeys);
});

test("dynamic content is escaped before entering templates", () => {
  assert.equal(
    escapeHTML('<img src=x onerror="alert(1)">'),
    "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;",
  );
});

test("gate summary and directory filtering tolerate incomplete collections", () => {
  assert.deepEqual(
    gateSummary([
      { status: "open", queue: 2 },
      { status: "degraded", queue: 5 },
      { status: "maintenance" },
    ]),
    { total: 3, open: 1, degraded: 1, maintenance: 1, queue: 7 },
  );
  const records = [
    { kind: "person", name: "Nadia", organization: "Research" },
    { kind: "vehicle", name: "Fleet", plate: "12345-A-6" },
  ];
  assert.equal(filterDirectory(records, "research", "person").length, 1);
  assert.equal(filterDirectory(records, "12345", "vehicle").length, 1);
  assert.equal(filterDirectory(records, "missing", "all").length, 0);
});

test("chart scaling is bounded and snapshot merge rejects incompatible shapes", () => {
  assert.deepEqual(chartScale([0, 50, 100], 100), [0, 50, 100]);
  const seed = { list: [1], object: { stable: true } };
  assert.deepEqual(mergeSnapshot(seed, { list: { invalid: true }, object: ["invalid"] }), seed);
  assert.deepEqual(mergeSnapshot(seed, { list: [2, 3] }).list, [2, 3]);
});

test("storage facade survives privacy-mode storage failures", () => {
  const storage = {
    getItem() {
      throw new Error("blocked");
    },
    setItem() {
      throw new Error("blocked");
    },
  };
  const safe = safeStorage(storage);
  assert.equal(safe.get("theme", "light"), "light");
  assert.equal(safe.set("theme", "dark"), false);
});

test("blank persisted auth migrates to the configured role token", () => {
  const roleTokens = { operator: "demo-operator", admin: "demo-admin" };
  assert.equal(resolveAuthToken("", "operator", roleTokens), "demo-operator");
  assert.equal(resolveAuthToken("   ", "admin", roleTokens), "demo-admin");
  assert.equal(resolveAuthToken("custom-session-token", "operator", roleTokens), "custom-session-token");
  assert.equal(resolveAuthToken("", "unknown", roleTokens), "");
});
