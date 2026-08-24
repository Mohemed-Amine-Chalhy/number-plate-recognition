import assert from "node:assert/strict";
import { readFile, stat } from "node:fs/promises";
import test from "node:test";

const directory = new URL("../", import.meta.url);

async function source(name) {
  return readFile(new URL(name, directory), "utf8");
}

test("entry page is self-contained and has baseline accessible structure", async () => {
  const html = await source("index.html");
  assert.match(html, /<a class="skip-link" href="#main-content">/);
  assert.match(html, /<meta name="viewport"/);
  assert.match(html, /<noscript>/);
  assert.doesNotMatch(html, /https?:\/\//);
  assert.doesNotMatch(html, /\son\w+=/i);
});

test("console exposes every requested product workspace", async () => {
  const app = await source("app.mjs");
  for (const route of [
    "command",
    "gates",
    "access",
    "directory",
    "operations",
    "analytics",
    "setup",
  ]) {
    assert.match(app, new RegExp(`route: "${route}"|${route}: render`));
  }
  assert.match(app, /aria-modal="true"/);
  assert.match(app, /role="tablist"/);
  assert.match(app, /aria-live="polite"/);
});

test("white-label, multilingual, RTL, dark, offline, and reduced-motion hooks exist", async () => {
  const [config, app, styles, translations] = await Promise.all([
    source("config.mjs"),
    source("app.mjs"),
    source("styles.css"),
    source("i18n.mjs"),
  ]);
  assert.match(config, /branding:/);
  assert.match(config, /logoUrl:/);
  assert.match(app, /document\.documentElement\.dir/);
  assert.match(app, /addEventListener\("offline"/);
  assert.match(styles, /\[data-theme="dark"\]/);
  assert.match(styles, /@media \(prefers-reduced-motion: reduce\)/);
  assert.match(styles, /:focus-visible/);
  assert.match(translations, /fr: Object\.freeze/);
  assert.match(translations, /ar: Object\.freeze/);
});

test("seeded tenant contains six stable configurable gate records", async () => {
  const data = await source("demo-data.mjs");
  const gateIds = [...data.matchAll(/id: "gate-[^"]+"/g)].map((match) => match[0]);
  assert.equal(gateIds.length, 6);
  assert.equal(new Set(gateIds).size, 6);
});

test("illustrated campus map keeps interaction, accessibility, and tenant boundaries in HTML", async () => {
  const [app, config, styles, asset] = await Promise.all([
    source("app.mjs"),
    source("config.mjs"),
    source("styles.css"),
    stat(new URL("assets/campus-map-illustrated-v2.webp", directory)),
  ]);

  assert.ok(asset.size > 100_000, "the checked-in map should be a real optimized illustration");
  assert.ok(asset.size < 1_000_000, "the map should remain practical for an operations console");
  assert.match(config, /map:\s*Object\.freeze/);
  assert.match(config, /campus-map-illustrated-v2\.webp/);
  assert.match(config, /gateLabels:\s*Object\.freeze/);
  assert.match(app, /data-campus-map/);
  assert.match(app, /data-map-gate/);
  assert.match(app, /resolveGateMapLabel\(item, mapConfig\)/);
  assert.match(app, /code:\s*item\.code/);
  assert.match(app, /aria-describedby="map-instructions"/);
  assert.match(app, /aria-controls="selected-gate-card"/);
  assert.match(app, /id="selected-gate-card"/);
  assert.doesNotMatch(app, /class="map-road/);
  assert.doesNotMatch(app, /class="map-building/);
  assert.match(styles, /\.campus-map-art/);
  assert.match(styles, /\.gate-pin:hover,\s*\n\.gate-pin\[aria-pressed="true"\]/);
  assert.match(styles, /top:\s*clamp\(var\(--map-pin-safe-edge\),\s*var\(--gate-y\)/);
  assert.match(styles, /left:\s*clamp\(var\(--map-pin-safe-edge\),\s*var\(--gate-x\)/);
  assert.match(
    styles,
    /@media \(max-width: 620px\)[\s\S]*?\.map-legend\s*\{[\s\S]*?position:\s*static/,
  );
  assert.doesNotMatch(styles, /\[dir="rtl"\]\s+\.gate-pin/);
});

test("gate workspaces use gate-local events and localized empty states", async () => {
  const [app, translations] = await Promise.all([source("app.mjs"), source("i18n.mjs")]);
  assert.match(app, /arrivalForGate\(state\.data\.arrivals, gateId\)/);
  assert.match(app, /workspace\.noRecentEvent/);
  assert.match(app, /workspace\.noRecentEventBody/);
  assert.doesNotMatch(app, /find\(\(arrival\) => arrival\.decision === "review"\)/);
  assert.match(translations, /"workspace\.noRecentEvent":/);
  assert.match(translations, /"workspace\.noRecentEventBody":/);
});

test("device health copy is derived from current records", async () => {
  const app = await source("app.mjs");
  assert.match(app, /deviceHealthSummary\(state\.data\.devices\)/);
  assert.match(app, /t\("metric\.healthDetail",\s*\{\s*online:/);
  assert.doesNotMatch(app, /state\.data\.devices\.filter\(\(device\) => device\.status === "online"\)/);
});

test("browser entrypoints share one cache generation", async () => {
  const [html, app, api, core] = await Promise.all([
    source("index.html"),
    source("app.mjs"),
    source("api.mjs"),
    source("core.mjs"),
  ]);
  for (const entrypoint of [html, app, api, core]) {
    assert.doesNotMatch(entrypoint, /\?v=0\.2\.2/);
  }
  assert.match(html, /styles\.css\?v=0\.2\.3/);
  assert.match(html, /app\.mjs\?v=0\.2\.3/);
  assert.match(app, /api\.mjs\?v=0\.2\.3/);
  assert.match(api, /core\.mjs\?v=0\.2\.3/);
  assert.match(core, /config\.mjs\?v=0\.2\.3/);
});
