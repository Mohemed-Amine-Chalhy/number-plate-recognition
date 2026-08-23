import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
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

test("seeded tenant contains four stable configurable gate records", async () => {
  const data = await source("demo-data.mjs");
  const gateIds = [...data.matchAll(/id: "gate-[^"]+"/g)].map((match) => match[0]);
  assert.equal(gateIds.length, 4);
  assert.equal(new Set(gateIds).size, 4);
});
