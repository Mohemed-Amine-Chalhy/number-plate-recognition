import { CampusApi } from "./api.mjs?v=0.2.3";
import { buildCampusMapModel, resolveGateMapLabel } from "./campus-map.mjs?v=0.2.3";
import { ROLE_OPTIONS, TENANT_CONFIG } from "./config.mjs?v=0.2.3";
import {
  arrivalForGate,
  chartScale,
  deviceHealthSummary,
  escapeHTML,
  filterDirectory,
  formatNumber,
  formatRelativeMinutes,
  gateSummary,
  isRTL,
  normalizeLocale,
  resolveRoute,
  resolveAuthToken,
  safeStorage,
  translate,
} from "./core.mjs?v=0.2.3";
import { DEMO_DATA } from "./demo-data.mjs?v=0.2.3";
import { MESSAGES } from "./i18n.mjs?v=0.2.3";

const root = document.querySelector("#app");
const preferences = safeStorage(globalThis.localStorage);

const NAV_ITEMS = Object.freeze([
  { route: "command", label: "nav.command", icon: "⌘", group: "workspace" },
  { route: "gates", label: "nav.gates", icon: "▦", group: "workspace" },
  { route: "access", label: "nav.access", icon: "✓", group: "workspace" },
  { route: "directory", label: "nav.directory", icon: "◎", group: "manage" },
  { route: "operations", label: "nav.operations", icon: "!", group: "manage" },
  { route: "analytics", label: "nav.analytics", icon: "↗", group: "manage" },
  { route: "setup", label: "nav.setup", icon: "·", group: "manage" },
]);

const storedRole = preferences.get("campus.role", TENANT_CONFIG.defaults.role);
const initialRole = ROLE_OPTIONS.some((role) => role.id === storedRole)
  ? storedRole
  : TENANT_CONFIG.defaults.role;

const state = {
  route: resolveRoute(globalThis.location.hash),
  locale: normalizeLocale(preferences.get("campus.locale", TENANT_CONFIG.defaults.locale)),
  theme: preferences.get("campus.theme", TENANT_CONFIG.defaults.theme) === "dark" ? "dark" : "light",
  role: initialRole,
  tenantName: preferences.get("campus.tenantName", TENANT_CONFIG.branding.name),
  apiBaseUrl: preferences.get("campus.apiBaseUrl", TENANT_CONFIG.api.baseUrl),
  token: resolveAuthToken(
    preferences.get("campus.apiToken", ""),
    initialRole,
    TENANT_CONFIG.api.demoRoleTokens,
  ),
  data: structuredClone(DEMO_DATA),
  source: "loading",
  selectedGateId: "gate-main",
  directoryKind: "all",
  directoryQuery: "",
  setupStep: 0,
  drawerOpen: false,
  modal: null,
  toast: null,
  busy: false,
  networkOnline: globalThis.navigator.onLine,
};

let toastTimer = null;
let lastFocusedElement = null;
let api = createApi();

function createApi() {
  return new CampusApi(
    {
      ...TENANT_CONFIG.api,
      baseUrl: state.apiBaseUrl,
      tokenProvider: () => state.token,
    },
    globalThis.fetch,
  );
}

function t(key, variables) {
  return translate(MESSAGES, key, state.locale, variables);
}

function h(value) {
  return escapeHTML(value);
}

function localized(value) {
  if (value && typeof value === "object") {
    return value[state.locale] ?? value.en ?? Object.values(value)[0] ?? "";
  }
  return value ?? "";
}

function clamp(value, minimum, maximum) {
  return Math.max(minimum, Math.min(maximum, Number(value) || 0));
}

function statusText(status) {
  return t(`status.${status}`);
}

function statusPill(status) {
  const normalized = String(status || "pending").toLowerCase();
  return `<span class="status-pill status-${h(normalized)}">${h(statusText(normalized))}</span>`;
}

function riskPill(risk) {
  return `<span class="badge risk-${h(risk)}">${h(String(risk).toUpperCase())}</span>`;
}

function sourcePill(extraClass = "") {
  return `<span class="data-badge data-${h(state.source)} ${h(extraClass)}" role="status">${h(
    t(`source.${state.source}`),
  )}</span>`;
}

function gateById(gateId) {
  return state.data.gates.find((gate) => gate.id === gateId) ?? state.data.gates[0];
}

function selectedArrival(gateId) {
  return arrivalForGate(state.data.arrivals, gateId);
}

function roleLabel(role = state.role) {
  return t(ROLE_OPTIONS.find((item) => item.id === role)?.labelKey ?? "role.viewer");
}

function applyDocumentPreferences() {
  document.documentElement.lang = state.locale;
  document.documentElement.dir = isRTL(state.locale) ? "rtl" : "ltr";
  document.documentElement.dataset.theme = state.theme;
  document.documentElement.style.setProperty("--brand", TENANT_CONFIG.branding.accent);
  document.documentElement.style.setProperty("--brand-strong", TENANT_CONFIG.branding.accentStrong);
  document.documentElement.style.setProperty("--brand-soft", TENANT_CONFIG.branding.accentSoft);
  document.title = `${t(`page.${state.route}`)} · ${TENANT_CONFIG.branding.shortName}`;
  document.querySelector(".skip-link")?.replaceChildren(document.createTextNode(t("skip.main")));
  document.querySelector('meta[name="theme-color"]')?.setAttribute(
    "content",
    state.theme === "dark" ? "#101211" : "#f2f1ed",
  );
}

function renderNavigation(group) {
  return NAV_ITEMS.filter((item) => item.group === group)
    .map((item) => {
      const count = item.route === "access" ? state.data.requests.filter((request) => request.status === "pending").length : 0;
      return `<a class="nav-link" href="#/${item.route}" ${
        state.route === item.route ? 'aria-current="page"' : ""
      } data-close-drawer>
        <span class="nav-icon" aria-hidden="true">${h(item.icon)}</span>
        <span>${h(t(item.label))}</span>
        ${
          count
            ? `<span class="nav-count" aria-label="${h(count)} ${h(t("nav.pendingCount"))}">${h(count)}</span>`
            : ""
        }
      </a>`;
    })
    .join("");
}

function renderRoleOptions() {
  return ROLE_OPTIONS.map(
    (role) =>
      `<option value="${h(role.id)}" ${role.id === state.role ? "selected" : ""}>${h(
        t(role.labelKey),
      )}</option>`,
  ).join("");
}

function renderShell() {
  if (!root) return;
  applyDocumentPreferences();
  const pendingIncidents = state.data.incidents.filter((incident) => incident.status !== "monitoring").length;
  const campusName = state.data.meta?.campusName ?? TENANT_CONFIG.branding.supportLabel;
  const sessionName = state.data.meta?.session?.display_name ?? "Salma El Idrissi";
  const sessionInitials = sessionName
    .split(/\s+/)
    .map((part) => part[0])
    .join("")
    .slice(0, 2)
    .toUpperCase();
  root.innerHTML = `
    <div class="app-shell ${state.drawerOpen ? "drawer-open" : ""}">
      <button class="drawer-backdrop" type="button" data-action="close-drawer" aria-label="${h(
        t("action.close"),
      )}"></button>
      <aside class="sidebar" aria-label="${h(TENANT_CONFIG.branding.productName)}">
        <div class="brand">
          <div class="brand-visual">
            <span class="brand-fallback" aria-hidden="true">${h(TENANT_CONFIG.branding.fallbackMark)}</span>
            <img class="tenant-logo" src="${h(TENANT_CONFIG.branding.logoUrl)}" alt="${h(
              TENANT_CONFIG.branding.logoAlt,
            )}" data-tenant-logo />
          </div>
          <div class="brand-copy">
            <span class="brand-name">${h(state.tenantName)}</span>
            <span class="brand-product">${h(TENANT_CONFIG.branding.productName)}</span>
          </div>
        </div>
        <div class="side-body">
          <div>
            <span class="nav-group-label">${h(t("nav.workspace"))}</span>
            <nav class="primary-nav" aria-label="${h(t("nav.workspace"))}">${renderNavigation("workspace")}</nav>
          </div>
          <div>
            <span class="nav-group-label">${h(t("nav.manage"))}</span>
            <nav class="primary-nav" aria-label="${h(t("nav.manage"))}">${renderNavigation("manage")}</nav>
          </div>
          <div class="sidebar-preferences">
            <label class="field"><span>${h(t("role.label"))}</span><select class="field-select" data-action="role">${renderRoleOptions()}</select></label>
            <label class="field"><span>${h(t("locale.label"))}</span><select class="field-select" data-action="locale">
              <option value="en" ${state.locale === "en" ? "selected" : ""}>English</option>
              <option value="fr" ${state.locale === "fr" ? "selected" : ""}>Français</option>
              <option value="ar" ${state.locale === "ar" ? "selected" : ""}>العربية</option>
            </select></label>
          </div>
          <div class="site-status-card">
            <div class="site-status-head">
              <strong>${h(campusName)}</strong>
              ${sourcePill()}
            </div>
            <p>${h(t(`source.detail.${state.source}`))}</p>
            <a class="soft-button compact-button" href="#/operations">${h(pendingIncidents)} ${h(
              t("operations.incidents").toLowerCase(),
            )}</a>
          </div>
        </div>
      </aside>
      <header class="topbar">
        <div class="topbar-actions">
          <button class="icon-button mobile-menu-button" type="button" data-action="open-drawer" aria-label="${h(
            t("action.menu"),
          )}" aria-expanded="${state.drawerOpen}">☰</button>
          <div class="topbar-title">
            <span class="eyebrow">${h(campusName)}</span>
            <h1>${h(t(`page.${state.route}`))}</h1>
          </div>
        </div>
        <div class="topbar-actions">
          ${sourcePill("topbar-source")}
          <select class="language-select" data-action="locale" aria-label="Language">
            <option value="en" ${state.locale === "en" ? "selected" : ""}>EN</option>
            <option value="fr" ${state.locale === "fr" ? "selected" : ""}>FR</option>
            <option value="ar" ${state.locale === "ar" ? "selected" : ""}>AR</option>
          </select>
          <button class="icon-button" type="button" data-action="theme" aria-label="${h(t("action.theme"))}" title="${h(
            t("action.theme"),
          )}">${state.theme === "dark" ? "☼" : "◐"}</button>
          <label class="profile-button">
            <span class="avatar brand-avatar" aria-hidden="true">${h(sessionInitials)}</span>
            <span class="profile-copy">
              <strong>${h(sessionName)}</strong>
              <small>${h(t("role.label"))}</small>
            </span>
            <select class="role-select" data-action="role" aria-label="${h(t("role.label"))}">
              ${renderRoleOptions()}
            </select>
            <span aria-hidden="true">⌄</span>
          </label>
        </div>
      </header>
      <main class="main-content" id="main-content" tabindex="-1">${renderCurrentPage()}</main>
      ${renderModal()}
      <div class="toast-region" aria-live="polite" aria-atomic="true">${renderToast()}</div>
    </div>`;
  bindLogoFallback();
  if (state.modal) queueMicrotask(() => root.querySelector("[data-modal-primary]")?.focus());
}

function renderPageIntro(actions = "") {
  return `<div class="page-intro">
    <div>
      <h2>${h(t(`page.${state.route}`))}</h2>
      <p>${h(t(`desc.${state.route}`))}</p>
    </div>
    ${actions ? `<div class="intro-actions">${actions}</div>` : ""}
  </div>`;
}

function renderCurrentPage() {
  const pages = {
    command: renderCommandCenter,
    gates: renderGateWorkspace,
    access: renderAccess,
    directory: renderDirectory,
    operations: renderOperations,
    analytics: renderAnalytics,
    setup: renderSetup,
  };
  return pages[state.route]?.() ?? renderCommandCenter();
}

function renderMetrics() {
  const summary = gateSummary(state.data.gates);
  const pending = state.data.requests.filter((request) => request.status === "pending").length;
  const wait = summary.open ? Math.round(state.data.gates.reduce((sum, gate) => sum + gate.waitMinutes, 0) / summary.total) : 0;
  const health = deviceHealthSummary(state.data.devices);
  const metrics = [
    ["metric.entries", formatNumber(state.data.analytics.totalEntries, state.locale), t("metric.entriesDetail"), "↗"],
    ["metric.pending", formatNumber(pending, state.locale), t("metric.pendingDetail"), "✓"],
    ["metric.wait", `${formatNumber(wait, state.locale)} ${t("unit.minutesShort")}`, t("metric.waitDetail"), "◷"],
    [
      "metric.health",
      `${formatNumber(health.online, state.locale)}/${formatNumber(health.total, state.locale)}`,
      t("metric.healthDetail", {
        online: formatNumber(health.online, state.locale),
        attention: formatNumber(health.attention, state.locale),
      }),
      "●",
    ],
  ];
  return `<div class="metric-grid">${metrics
    .map(
      ([label, value, detail, icon]) => `<article class="metric-card">
        <div><span class="metric-label">${h(t(label))}</span><strong class="metric-value">${h(value)}</strong></div>
        <span class="metric-symbol" aria-hidden="true">${h(icon)}</span>
        <span class="metric-detail">${h(detail)}</span>
      </article>`,
    )
    .join("")}</div>`;
}

function renderMap() {
  const mapConfig = TENANT_CONFIG.map;
  const model = buildCampusMapModel(state.data.gates, mapConfig, state.selectedGateId);
  const gate = model.selectedGate;
  const landmarks = Array.isArray(mapConfig.landmarks) ? mapConfig.landmarks : [];
  return `<section class="panel map-panel" aria-labelledby="map-title">
    <header class="panel-header">
      <div><h3 id="map-title">${h(t("map.title"))}</h3><p>${h(t("map.subtitle"))}</p></div>
      <span class="metric-chip">${h(localized(state.data.meta?.weather))}</span>
    </header>
    <p class="sr-only" id="map-instructions">${h(t("map.instructions"))}</p>
    <div class="campus-map" data-campus-map role="group" aria-labelledby="map-title" aria-describedby="map-instructions" style="--map-aspect:${h(
      mapConfig.aspectRatio,
    )}">
      <div class="campus-map-viewport">
        <img class="campus-map-art" src="${h(mapConfig.assetUrl)}" alt="" aria-hidden="true" />
        <span class="campus-map-shade" aria-hidden="true"></span>
        <div class="campus-landmark-layer" aria-hidden="true">
          ${landmarks
            .map(
              (landmark) => `<span class="campus-landmark" data-map-landmark="${h(landmark.id)}" style="--landmark-x:${clamp(
                landmark.x,
                4,
                96,
              )}%;--landmark-y:${clamp(landmark.y, 4, 96)}%">${h(t(landmark.labelKey))}</span>`,
            )
            .join("")}
        </div>
        <span class="map-compass" aria-hidden="true"><b>↑</b>${h(t("map.north"))}</span>
        <div class="gate-pin-layer">
      ${model.gates
        .map(
          ({ gate: item, position, selected }) => `<button class="gate-pin ${h(
            item.status,
          )}" type="button" data-action="select-map-gate" data-map-gate data-gate-id="${h(
            item.id,
          )}" style="--gate-x:${position.x}%;--gate-y:${position.y}%" aria-label="${h(
            t("map.gateLabel", {
              name: localized(item.name),
              code: item.code,
              status: statusText(item.status),
              queue: formatNumber(item.queue, state.locale),
              wait: `${formatNumber(item.waitMinutes, state.locale)} ${t("unit.minutesShort")}`,
            }),
          )}" aria-controls="selected-gate-card" aria-pressed="${selected}"><span>${h(
            resolveGateMapLabel(item, mapConfig),
          )}</span></button>`,
        )
        .join("")}
        </div>
      </div>
      <div class="map-legend" aria-label="${h(t("map.legend.label"))}">
        <span class="legend-item"><i class="legend-dot open"></i>${h(t("map.legend.open"))}</span>
        <span class="legend-item"><i class="legend-dot degraded"></i>${h(t("map.legend.attention"))}</span>
        <span class="legend-item"><i class="legend-dot maintenance"></i>${h(t("map.legend.closed"))}</span>
      </div>
      <article class="gate-map-card" id="selected-gate-card" aria-live="polite" aria-atomic="true">
        <span class="gate-selection-label">${h(t("map.selectedGate"))}</span>
        <div class="gate-card-head">
          <div><strong>${h(localized(gate.name))}</strong><small>${h(localized(gate.zone))} · ${h(gate.lanes)} ${h(
            t("gate.lanes"),
          )}</small></div>
          ${statusPill(gate.status)}
        </div>
        <div class="gate-card-metrics">
          <div><small>${h(t("gate.queue"))}</small><strong>${h(gate.queue)}</strong></div>
          <div><small>${h(t("gate.wait"))}</small><strong>${h(gate.waitMinutes)} ${h(t("unit.minutesShort"))}</strong></div>
          <div><small>${h(t("gate.throughput"))}</small><strong>${h(gate.throughput ?? "—")}</strong></div>
        </div>
        <button class="soft-button compact-button" type="button" data-action="open-gate" data-gate-id="${h(gate.id)}">${h(
          t("action.view"),
        )} <span aria-hidden="true">→</span></button>
      </article>
    </div>
  </section>`;
}

function renderArrivals() {
  const activityKey = state.source === "live" ? "arrivals.live" : "arrivals.scenario";
  return `<section class="panel" aria-labelledby="arrivals-title">
    <header class="panel-header">
      <div><h3 id="arrivals-title">${h(t("arrivals.title"))}</h3><p>${h(t("arrivals.subtitle"))}</p></div>
      <span class="live-indicator ${state.source === "live" ? "" : "scenario-indicator"}">${h(t(activityKey))}</span>
    </header>
    <div class="live-list">${state.data.arrivals
      .map(
        (arrival) => `<article class="arrival-row">
          <span class="avatar ${arrival.decision === "review" ? "brand-avatar" : ""}" aria-hidden="true">${h(arrival.avatar)}</span>
          <div class="arrival-copy">
            <strong>${h(arrival.plate)}</strong>
            <span>${h(localized(arrival.person))}</span>
            <small>${h(localized(gateById(arrival.gateId)?.name))} · ${h(
              formatRelativeMinutes(arrival.minutesAgo, MESSAGES, state.locale),
            )}</small>
          </div>
          <div class="arrival-state">${statusPill(arrival.decision)}<small>${h(arrival.confidence)}%</small></div>
        </article>`,
      )
      .join("")}</div>
  </section>`;
}

function renderCommandCenter() {
  return `<div class="page-stack">
    ${renderPageIntro(`<button class="soft-button" type="button" data-action="refresh"><span aria-hidden="true">↻</span>${h(
      t("action.refresh"),
    )}</button>`)}
    ${renderMetrics()}
    <div class="command-grid">${renderMap()}${renderArrivals()}</div>
    <aside class="attention-strip">
      <span class="attention-icon" aria-hidden="true">!</span>
      <div><strong>${h(t("attention.title"))}</strong><small>${h(t("attention.body"))}</small></div>
      <button class="soft-button compact-button" type="button" data-action="open-gate" data-gate-id="gate-residential">${h(
        t("action.review"),
      )}</button>
    </aside>
  </div>`;
}

function renderGateTabs() {
  return `<div class="gate-tabs" role="tablist" aria-label="${h(t("page.gates"))}">${state.data.gates
    .map(
      (gate) => `<button class="gate-tab" type="button" role="tab" aria-selected="${
        gate.id === state.selectedGateId
      }" data-action="select-gate" data-gate-id="${h(gate.id)}"><span class="mini-dot ${h(gate.status)}"></span>${h(
        localized(gate.name),
      )}</button>`,
    )
    .join("")}</div>`;
}

function optionalPercentage(value) {
  if (value === null || value === undefined || value === "") return "—";
  const numeric = Number(value);
  return Number.isFinite(numeric) ? `${formatNumber(numeric, state.locale)}%` : "—";
}

function renderGateControls(gate) {
  return `<div class="control-grid">
    <button class="primary-button" type="button" data-action="gate-command" data-command="open" data-gate-id="${h(
      gate.id,
    )}">${h(t("action.openBarrier"))}</button>
    <button class="soft-button" type="button" data-action="gate-command" data-command="hold" data-gate-id="${h(
      gate.id,
    )}">${h(t("action.holdLane"))}</button>
    <button class="soft-button" type="button" data-action="gate-command" data-command="intercom" data-gate-id="${h(
      gate.id,
    )}">${h(t("action.intercom"))}</button>
    <button class="soft-button" type="button" data-action="open-access">${h(t("action.review"))}</button>
  </div>`;
}

function renderGateWorkspace() {
  const gate = gateById(state.selectedGateId);
  const arrival = selectedArrival(gate.id);
  const matched = arrival ? arrival.decision !== "denied" : false;
  const cameraControls = `<div class="camera-controls">
    <button class="icon-button" type="button" aria-label="Snapshot">⌁</button>
    <button class="icon-button" type="button" aria-label="Expand camera">↗</button>
  </div>`;
  const recognition = arrival
    ? `<div class="plate-read"><span>${h(t("workspace.recognition"))} · ${h(arrival.confidence)}%</span><strong>${h(
        arrival.plate,
      )}</strong></div>${cameraControls}`
    : `<div class="plate-read empty-plate-read"><span>${h(t("workspace.noRecentEvent"))}</span><strong>—</strong></div>${cameraControls}`;
  const observationPanel = arrival
    ? `<section class="panel" aria-labelledby="match-heading">
        <header class="panel-header">
          <div><h3 id="match-heading">${h(matched ? t("workspace.match") : t("workspace.noMatch"))}</h3><p>${h(
            arrival.purpose,
          )}</p></div>
          ${statusPill(arrival.decision)}
        </header>
        <div class="gate-control-body">
          <div class="person-match">
            <span class="avatar brand-avatar" aria-hidden="true">${h(arrival.avatar)}</span>
            <div><strong>${h(localized(arrival.person))}</strong><span>${h(arrival.organization)}</span><small>${h(
              arrival.color,
            )}</small></div>
          </div>
          <dl class="detail-list">
            <div class="detail-row"><dt>${h(t("workspace.vehicle"))}</dt><dd>${h(arrival.plate)}</dd></div>
            <div class="detail-row"><dt>${h(t("workspace.accessWindow"))}</dt><dd>06:00–22:00</dd></div>
            <div class="detail-row"><dt>${h(t("workspace.lastSeen"))}</dt><dd>${h(
              formatRelativeMinutes(arrival.minutesAgo, MESSAGES, state.locale),
            )}</dd></div>
            <div class="detail-row"><dt>${h(t("gate.cameraHealth"))}</dt><dd>${h(
              optionalPercentage(gate.cameraHealth),
            )}</dd></div>
          </dl>
          ${renderGateControls(gate)}
        </div>
      </section>`
    : `<section class="panel" aria-labelledby="match-heading">
        <header class="panel-header">
          <div><h3 id="match-heading">${h(t("workspace.noRecentEvent"))}</h3><p>${h(
            t("workspace.noRecentEventBody", { gate: localized(gate.name) }),
          )}</p></div>
        </header>
        <div class="gate-control-body">
          <div class="empty-state gate-empty-state" role="status">
            <span class="metric-symbol" aria-hidden="true">◌</span>
            <strong>${h(t("workspace.noRecentEvent"))}</strong>
            <p>${h(t("workspace.noRecentEventBody", { gate: localized(gate.name) }))}</p>
          </div>
          <dl class="detail-list">
            <div class="detail-row"><dt>${h(t("gate.operator"))}</dt><dd>${h(gate.operator ?? "—")}</dd></div>
            <div class="detail-row"><dt>${h(t("gate.cameraHealth"))}</dt><dd>${h(
              optionalPercentage(gate.cameraHealth),
            )}</dd></div>
          </dl>
          ${renderGateControls(gate)}
        </div>
      </section>`;
  return `<div class="page-stack">
    ${renderPageIntro()}
    ${renderGateTabs()}
    <div class="workspace-grid">
      <section class="panel" aria-labelledby="camera-heading">
        <header class="panel-header">
          <div><h3 id="camera-heading">${h(localized(gate.name))}</h3><p>${h(gate.code)} · ${h(
            gate.operator ?? "—",
          )}</p></div>
          ${statusPill(gate.status)}
        </header>
        <div class="camera-stage" role="img" aria-label="${h(`${t("workspace.live")} · ${localized(gate.name)}`)}">
          <div class="camera-topline"><span class="live-indicator ${
            state.source === "live" ? "" : "scenario-indicator"
          }">${h(t(state.source === "live" ? "arrivals.live" : "arrivals.scenario"))}</span><span>CAM-${h(
            gate.code,
          )}-A · 1080p</span></div>
          <div class="recognition-overlay">${recognition}</div>
        </div>
      </section>
      ${observationPanel}
    </div>
  </div>`;
}

function renderAccess() {
  const pending = state.data.requests.filter((request) => request.status === "pending");
  const approved = state.data.requests.filter((request) => request.status === "approved").length;
  const denied = state.data.requests.filter((request) => request.status === "denied").length;
  const canDecide = state.role === "admin";
  return `<div class="page-stack">
    ${renderPageIntro(`<button class="primary-button" type="button" data-action="open-first-request" ${
      pending.length ? "" : "disabled"
    }>${h(t("action.review"))} ${h(pending.length)}</button>`)}
    <div class="split-grid">
      <section class="panel" aria-labelledby="request-queue-heading">
        <header class="panel-header">
          <div><h3 id="request-queue-heading">${h(t("access.queue"))}</h3><p>${h(t("access.subtitle"))}</p></div>
          <span class="metric-chip">${h(pending.length)} ${h(t("access.pending").toLowerCase())}</span>
        </header>
        <div class="request-list">
          ${state.data.requests
            .map(
              (request) => `<article class="request-card" id="request-${h(request.id)}">
                <div class="request-head">
                  <div><strong>${h(request.person)}</strong><small>${h(request.id)} · ${h(
                    formatRelativeMinutes(request.submittedMinutes, MESSAGES, state.locale),
                  )}</small></div>
                  <div class="inline-actions">${riskPill(request.risk)}${statusPill(request.status)}</div>
                </div>
                <div class="request-meta">
                  <div><span>${h(t("workspace.vehicle"))}</span><strong>${h(request.plate)}</strong></div>
                  <div><span>${h(t("access.host"))}</span><strong>${h(request.host)}</strong></div>
                  <div><span>${h(t("access.window"))}</span><strong>${h(request.window)}</strong></div>
                </div>
                <div><span class="eyebrow">${h(t("access.reason"))}</span><strong>${h(request.reason)}</strong></div>
                <div class="decision-actions">
                  <button class="ghost-button compact-button" type="button" data-action="request-decision" data-request-id="${h(
                    request.id,
                  )}" data-decision="denied" ${request.status !== "pending" || !canDecide ? "disabled" : ""}>${h(
                    t("action.reject"),
                  )}</button>
                  <button class="primary-button compact-button" type="button" data-action="request-decision" data-request-id="${h(
                    request.id,
                  )}" data-decision="approved" ${request.status !== "pending" || !canDecide ? "disabled" : ""}>${h(
                    t("action.approve"),
                  )}</button>
                </div>
              </article>`,
            )
            .join("")}
        </div>
      </section>
      <div class="page-stack">
        <section class="panel">
          <header class="panel-header"><h3>${h(t("access.queue"))}</h3></header>
          <div class="panel-body queue-summary">
            <div><strong>${h(pending.length)}</strong><span>${h(t("access.pending"))}</span></div>
            <div><strong>${h(approved + 46)}</strong><span>${h(t("access.approvedToday"))}</span></div>
            <div><strong>${h(denied + 3)}</strong><span>${h(t("access.deniedToday"))}</span></div>
            <div><strong>48 ${h(t("unit.secondsShort"))}</strong><span>${h(t("access.median"))}</span></div>
          </div>
        </section>
        <section class="panel">
          <header class="panel-header"><h3>${h(t("access.activity"))}</h3></header>
          <div class="panel-body">
            <ol class="timeline">
              <li><strong>REQ-2379 · Atlas Facilities</strong><small>${h(t("status.approved"))} · Salma E. · 09:31</small></li>
              <li><strong>REQ-2378 · Amina Zahra</strong><small>${h(t("status.denied"))} · Youssef K. · 09:24</small></li>
              <li><strong>Visitor pass · GreenTech</strong><small>${h(t("status.active"))} · System · 09:18</small></li>
            </ol>
          </div>
        </section>
      </div>
    </div>
  </div>`;
}

function renderDirectory() {
  const records = filterDirectory(state.data.directory, state.directoryQuery, state.directoryKind);
  const filterButtons = [
    ["all", "directory.all"],
    ["person", "directory.people"],
    ["vehicle", "directory.vehicles"],
  ];
  return `<div class="page-stack">
    ${renderPageIntro()}
    <section class="panel" aria-labelledby="directory-heading">
      <header class="panel-header">
        <div class="filter-row" role="group" aria-label="${h(t("directory.status"))}">
          ${filterButtons
            .map(
              ([kind, label]) => `<button class="filter-button" type="button" data-action="directory-kind" data-kind="${h(
                kind,
              )}" aria-pressed="${state.directoryKind === kind}">${h(t(label))}</button>`,
            )
            .join("")}
        </div>
        <form class="search-form" data-directory-search>
          <label class="sr-only" for="directory-query">${h(t("directory.search"))}</label>
          <input class="search-input" id="directory-query" name="query" type="search" value="${h(
            state.directoryQuery,
          )}" placeholder="${h(t("directory.search"))}" />
          <button class="soft-button" type="submit">${h(t("action.search"))}</button>
        </form>
      </header>
      ${
        records.length
          ? `<div class="table-wrap"><table class="data-table">
              <thead><tr><th>${h(t("directory.record"))}</th><th>${h(t("directory.organization"))}</th><th>${h(
                t("directory.access"),
              )}</th><th>${h(t("directory.status"))}</th><th><span class="sr-only">${h(t("action.review"))}</span></th></tr></thead>
              <tbody>${records
                .map(
                  (record) => `<tr>
                    <td><div class="entity-cell"><span class="avatar ${record.kind === "vehicle" ? "" : "brand-avatar"}">${h(
                      record.initials,
                    )}</span><div><strong>${h(record.name)}</strong><small>${h(record.plate ?? record.email)}</small></div></div></td>
                    <td>${h(record.organization)}</td><td>${h(record.access)}</td><td>${statusPill(record.status)}</td>
                    <td><button class="ghost-button compact-button" type="button">${h(t("action.review"))}</button></td>
                  </tr>`,
                )
                .join("")}</tbody>
            </table></div>`
          : `<div class="empty-state"><span class="metric-symbol" aria-hidden="true">⌕</span><strong>${h(
              t("directory.empty"),
            )}</strong><p>${h(t("directory.emptyBody"))}</p></div>`
      }
    </section>
  </div>`;
}

function renderOperations() {
  const health = deviceHealthSummary(state.data.devices);
  const canAcknowledge = ["operator", "attendant", "admin"].includes(state.role);
  return `<div class="page-stack">
    ${renderPageIntro(`<button class="soft-button" type="button" data-action="refresh"><span aria-hidden="true">↻</span>${h(
      t("action.refresh"),
    )}</button>`)}
    <div class="metric-grid">
      <article class="metric-card"><div><span class="metric-label">${h(t("operations.incidents"))}</span><strong class="metric-value">${h(
        state.data.incidents.length,
      )}</strong></div><span class="metric-symbol">!</span><span class="metric-detail">1 high · 1 medium · 1 planned</span></article>
      <article class="metric-card"><div><span class="metric-label">${h(t("operations.health"))}</span><strong class="metric-value">${h(
        health.online,
      )}/${h(health.total)}</strong></div><span class="metric-symbol">●</span><span class="metric-detail">${h(
        t("metric.healthDetail", {
          online: formatNumber(health.online, state.locale),
          attention: formatNumber(health.attention, state.locale),
        }),
      )}</span></article>
      <article class="metric-card"><div><span class="metric-label">Event delivery</span><strong class="metric-value">99.98%</strong></div><span class="metric-symbol">↯</span><span class="metric-detail">Last 24 hours</span></article>
      <article class="metric-card"><div><span class="metric-label">Recognition latency</span><strong class="metric-value">218 ms</strong></div><span class="metric-symbol">◷</span><span class="metric-detail">p95 at the edge</span></article>
    </div>
    <div class="split-grid">
      <section class="panel" aria-labelledby="incidents-heading">
        <header class="panel-header"><div><h3 id="incidents-heading">${h(t("operations.incidents"))}</h3><p>${h(
          t("operations.incidentsSubtitle"),
        )}</p></div><span class="metric-chip">${h(state.data.incidents.length)}</span></header>
        <div class="incident-list">${state.data.incidents
          .map(
            (incident) => `<article class="incident-card severity-rail ${h(incident.severity)}">
              <div class="incident-head"><div><strong>${h(incident.title)}</strong><small>${h(incident.id)} · ${h(
                localized(gateById(incident.gateId)?.name),
              )}</small></div>${statusPill(incident.status)}</div>
              <p>${h(incident.description)}</p>
              <div class="incident-footer"><small>${h(t("operations.owner"))}: ${h(incident.owner)} · ${h(
                formatRelativeMinutes(incident.minutesAgo, MESSAGES, state.locale),
              )}</small><button class="soft-button compact-button" type="button" data-action="ack-incident" data-incident-id="${h(
                incident.id,
              )}" ${canAcknowledge ? "" : "disabled"}>${h(t("action.acknowledge"))}</button></div>
            </article>`,
          )
          .join("")}</div>
      </section>
      <section class="panel" aria-labelledby="health-heading">
        <header class="panel-header"><div><h3 id="health-heading">${h(t("operations.health"))}</h3><p>${h(
          t("operations.healthSubtitle"),
        )}</p></div></header>
        <div class="health-grid">${state.data.devices
          .map(
            (device) => `<article class="device-card ${h(device.status)}">
              <div class="device-head"><strong>${h(device.id)}</strong>${statusPill(device.status)}</div>
              <small>${h(device.type)} · ${h(localized(gateById(device.gateId)?.name))}</small>
              <div class="health-bar" title="${h(t("operations.uptime"))}: ${h(device.uptime)}%"><span style="--value:${clamp(
                device.uptime,
                0,
                100,
              )}%"></span></div>
              <small>${h(device.latency)} ms ${h(t("operations.latency"))} · ${h(device.uptime)}% ${h(
                t("operations.uptime"),
              )}</small>
            </article>`,
          )
          .join("")}</div>
      </section>
    </div>
  </div>`;
}

function renderBarChart(items, labelKey) {
  const values = items.map((item) => (typeof item === "number" ? item : item.value));
  const heights = chartScale(values, Math.max(...values));
  return `<div class="bar-chart" style="--columns:${items.length}" role="img" aria-label="${h(t(labelKey))}">${items
    .map((item, index) => {
      const value = typeof item === "number" ? item : item.value;
      const label = typeof item === "number" ? `${String(index + 6).padStart(2, "0")}:00` : item.label;
      return `<div class="bar-group" title="${h(label)} · ${h(value)}"><span class="bar" style="--bar-height:${clamp(
        heights[index],
        1,
        100,
      )}%"></span><small>${h(label)}</small></div>`;
    })
    .join("")}</div>`;
}

function renderAnalytics() {
  const analytics = state.data.analytics;
  const approvedEnd = analytics.decisions.approved;
  const reviewEnd = approvedEnd + analytics.decisions.review;
  return `<div class="page-stack">
    ${renderPageIntro(`<button class="soft-button" type="button" data-action="print"><span aria-hidden="true">⇩</span>${h(
      t("action.export"),
    )}</button>`)}
    <div class="metric-grid">
      <article class="metric-card"><div><span class="metric-label">${h(t("metric.entries"))}</span><strong class="metric-value">${h(
        formatNumber(analytics.totalEntries, state.locale),
      )}</strong></div><span class="metric-symbol">↗</span><span class="metric-detail">${h(t("metric.entriesDetail"))}</span></article>
      <article class="metric-card"><div><span class="metric-label">${h(t("analytics.approvalRate"))}</span><strong class="metric-value">${h(
        analytics.approvalRate,
      )}%</strong></div><span class="metric-symbol">✓</span><span class="metric-detail">+1.2 pp this week</span></article>
      <article class="metric-card"><div><span class="metric-label">${h(t("analytics.median"))}</span><strong class="metric-value">${h(
        analytics.medianDecisionSeconds,
      )} ${h(t("unit.secondsShort"))}</strong></div><span class="metric-symbol">◷</span><span class="metric-detail">Recognition to decision</span></article>
      <article class="metric-card"><div><span class="metric-label">${h(t("analytics.peak"))}</span><strong class="metric-value">${h(
        analytics.peakHour,
      )}</strong></div><span class="metric-symbol">⌁</span><span class="metric-detail">151 entries at peak</span></article>
    </div>
    <div class="analytics-grid">
      <section class="panel"><header class="panel-header"><div><h3>${h(t("analytics.hourly"))}</h3><p>${h(
        t("analytics.hourlySubtitle"),
      )}</p></div><span class="metric-chip">${h(formatNumber(analytics.totalEntries, state.locale))} ${h(
        t("unit.entries"),
      )}</span></header><div class="panel-body">${renderBarChart(analytics.hourly, "analytics.hourly")}</div></section>
      <section class="panel"><header class="panel-header"><h3>${h(t("analytics.decisions"))}</h3></header>
        <div class="donut-wrap"><div class="donut" style="--approved:${approvedEnd}%;--review:${reviewEnd}%"><div class="donut-copy"><strong>${h(
          analytics.approvalRate,
        )}%</strong><span>${h(t("analytics.approvalRate"))}</span></div></div>
        <div class="legend-list">
          <div class="legend-row"><i class="legend-dot open"></i><span>${h(t("analytics.approved"))}</span><strong>${h(
            analytics.decisions.approved,
          )}%</strong></div>
          <div class="legend-row"><i class="legend-dot degraded"></i><span>${h(t("analytics.review"))}</span><strong>${h(
            analytics.decisions.review,
          )}%</strong></div>
          <div class="legend-row"><i class="legend-dot maintenance"></i><span>${h(t("analytics.denied"))}</span><strong>${h(
            analytics.decisions.denied,
          )}%</strong></div>
        </div></div>
      </section>
    </div>
    <div class="analytics-grid">
      <section class="panel"><header class="panel-header"><h3>${h(t("analytics.week"))}</h3></header><div class="panel-body">${renderBarChart(
        analytics.days,
        "analytics.week",
      )}</div></section>
      <section class="panel"><header class="panel-header"><h3>${h(t("analytics.gates"))}</h3></header><div class="panel-body progress-list">${analytics.gateTraffic
        .map((item) => {
          const gate = gateById(item.gateId);
          return `<div class="progress-item"><div class="progress-head"><span>${h(localized(gate?.name))}</span><strong>${h(
            item.value,
          )}%</strong></div><div class="progress-bar"><span style="--value:${clamp(item.value, 0, 100)}%"></span></div></div>`;
        })
        .join("")}</div></section>
    </div>
  </div>`;
}

function renderSetupStepBody() {
  const gateSummaryText = `${state.data.gates.length} ${t("setup.configuredGates").toLowerCase()}`;
  const deviceSummaryText = `${state.data.devices.length} ${t("setup.configuredDevices").toLowerCase()}`;
  if (state.setupStep === 0) {
    return `<h3>${h(t("setup.tenantTitle"))}</h3><p>${h(t("setup.tenantBody"))}</p>
      <div class="form-grid">
        <div class="field"><label for="tenant-name">${h(t("setup.tenantName"))}</label><input class="field-input" id="tenant-name" name="tenantName" value="${h(
          state.tenantName,
        )}" /></div>
        <div class="field"><label for="api-url">${h(t("setup.apiUrl"))}</label><input class="field-input" id="api-url" name="apiUrl" value="${h(
          state.apiBaseUrl,
        )}" spellcheck="false" /></div>
        <div class="field full"><label for="api-token">${h(t("setup.token"))}</label><input class="field-input" id="api-token" name="apiToken" type="password" value="${h(
          state.token,
        )}" autocomplete="off" /><small>${h(t("setup.tokenHelp"))}</small></div>
      </div>`;
  }
  if (state.setupStep === 1) {
    return `<h3>${h(t("setup.siteTitle"))}</h3><p>${h(t("setup.siteBody"))}</p>
      <div class="form-grid"><div class="field"><label for="site-name">${h(t("setup.site"))}</label><input class="field-input" id="site-name" value="${h(
        state.data.meta.campusName,
      )}" /></div><div class="field"><label for="timezone">${h(t("setup.timezone"))}</label><select class="field-select" id="timezone"><option>Africa/Casablanca</option><option>UTC</option></select></div></div>
      <div class="gate-tabs">${state.data.gates
        .map((gate) => `<span class="gate-tab"><span class="mini-dot ${h(gate.status)}"></span>${h(localized(gate.name))}</span>`)
        .join("")}</div>`;
  }
  if (state.setupStep === 2) {
    return `<h3>${h(t("setup.devicesTitle"))}</h3><p>${h(t("setup.devicesBody"))}</p><div class="health-grid">${state.data.devices
      .slice(0, 6)
      .map(
        (device) => `<div class="connection-card"><span class="connection-icon">●</span><div><strong>${h(
          device.id,
        )}</strong><small>${h(device.type)} · ${h(localized(gateById(device.gateId)?.name))}</small></div>${statusPill(
          device.status,
        )}</div>`,
      )
      .join("")}</div>`;
  }
  return `<h3>${h(t("setup.readyTitle"))}</h3><p>${h(t("setup.readyBody"))}</p>
    <div class="connection-card"><span class="connection-icon">✓</span><div><strong>${h(t("setup.connection"))}</strong><small>${h(
      state.source === "live" ? t("setup.connectionLive") : t("setup.connectionDemo"),
    )}</small></div>${sourcePill()}</div>
    <div class="queue-summary"><div><strong>${h(state.data.gates.length)}</strong><span>${h(gateSummaryText)}</span></div><div><strong>${h(
      state.data.devices.length,
    )}</strong><span>${h(deviceSummaryText)}</span></div></div>`;
}

function renderSetup() {
  const steps = ["setup.stepTenant", "setup.stepSite", "setup.stepDevices", "setup.stepReady"];
  return `<div class="page-stack">
    ${renderPageIntro()}
    <div class="setup-layout">
      <nav class="panel setup-steps" aria-label="${h(t("page.setup"))}">${steps
        .map(
          (label, index) => `<button class="setup-step" type="button" data-action="setup-step" data-step="${index}" ${
            state.setupStep === index ? 'aria-current="step"' : ""
          }><span class="step-number">${index < state.setupStep ? "✓" : index + 1}</span><span>${h(t(label))}</span></button>`,
        )
        .join("")}</nav>
      <section class="panel"><form class="setup-form" data-setup-form>${renderSetupStepBody()}
        <footer class="setup-footer">
          <button class="ghost-button" type="button" data-action="setup-back" ${state.setupStep === 0 ? "disabled" : ""}>${h(
            t("action.back"),
          )}</button>
          <button class="primary-button" type="submit">${h(
            state.setupStep === 3 ? t("action.finish") : t("action.next"),
          )}</button>
        </footer>
      </form></section>
    </div>
  </div>`;
}

function renderModal() {
  if (!state.modal) return "";
  const gate = gateById(state.modal.gateId);
  const commandLabels = {
    open: t("action.openBarrier"),
    hold: t("action.holdLane"),
    intercom: t("action.intercom"),
  };
  return `<div class="modal-backdrop" data-action="modal-backdrop">
    <section class="modal" role="dialog" aria-modal="true" aria-labelledby="modal-heading" aria-describedby="modal-description">
      <header class="modal-header"><div><span class="eyebrow">${h(gate.code)}</span><h2 id="modal-heading">${h(
        t("workspace.commandTitle"),
      )}</h2></div><button class="icon-button" type="button" data-action="close-modal" aria-label="${h(
        t("action.close"),
      )}">×</button></header>
      <div class="modal-body"><p id="modal-description">${h(t("workspace.commandBody"))}</p><div class="modal-summary"><strong>${h(
        commandLabels[state.modal.command] ?? state.modal.command,
      )}</strong><span>${h(localized(gate.name))} · ${h(gate.operator)}</span>${statusPill(gate.status)}</div></div>
      <footer class="modal-footer"><button class="ghost-button" type="button" data-action="close-modal">${h(
        t("action.cancel"),
      )}</button><button class="primary-button" type="button" data-action="confirm-command" data-modal-primary ${
        state.busy ? "disabled" : ""
      }>${h(t("action.confirm"))}</button></footer>
    </section>
  </div>`;
}

function renderToast() {
  if (!state.toast) return "";
  return `<div class="toast ${state.toast.type === "error" ? "error" : ""}"><span class="toast-mark">${
    state.toast.type === "error" ? "!" : "✓"
  }</span><div><strong>${h(state.toast.title)}</strong>${state.toast.detail ? `<span>${h(state.toast.detail)}</span>` : ""}</div></div>`;
}

function bindLogoFallback() {
  const logo = root?.querySelector("[data-tenant-logo]");
  if (!logo) return;
  logo.addEventListener("error", () => {
    logo.hidden = true;
  });
  if (logo.complete && logo.naturalWidth === 0) logo.hidden = true;
}

function showToast(title, detail = "", type = "success") {
  state.toast = { title, detail, type };
  renderShell();
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    state.toast = null;
    renderShell();
  }, 3200);
}

function closeModal() {
  state.modal = null;
  state.busy = false;
  renderShell();
  queueMicrotask(() => lastFocusedElement?.focus());
}

async function refreshData({ announce = false } = {}) {
  if (!state.networkOnline) {
    state.data = structuredClone(DEMO_DATA);
    state.source = "offline";
    renderShell();
    return;
  }
  state.source = "loading";
  renderShell();
  try {
    const snapshot = await api.loadSnapshot(structuredClone(DEMO_DATA));
    state.data = snapshot.data;
    state.source = snapshot.source;
    if (!state.data.gates.some((gate) => gate.id === state.selectedGateId)) {
      state.selectedGateId = state.data.gates[0]?.id ?? state.selectedGateId;
    }
  } catch {
    state.data = structuredClone(DEMO_DATA);
    state.source = "demo";
  }
  renderShell();
  if (announce) showToast(t("toast.refreshed"), t(`source.detail.${state.source}`));
}

async function confirmGateCommand() {
  if (!state.modal || state.busy) return;
  state.busy = true;
  renderShell();
  const { gateId, command } = state.modal;
  if (state.source === "demo" || state.source === "offline" || !api.supportsGateCommands) {
    closeModal();
    showToast(t("toast.demoCommand"), `${localized(gateById(gateId).name)} · ${command}`);
    return;
  }
  try {
    await api.gateCommand(gateId, command);
    closeModal();
    showToast(t("toast.liveCommand"), localized(gateById(gateId).name));
  } catch (error) {
    state.busy = false;
    state.modal = null;
    renderShell();
    showToast(t("toast.commandError"), error instanceof Error ? error.message : "API error", "error");
  }
}

async function decideRequest(requestId, decision) {
  const request = state.data.requests.find((item) => item.id === requestId);
  if (!request || request.status !== "pending") return;
  if (state.source !== "demo" && state.source !== "offline") {
    try {
      await api.decideRequest(requestId, decision);
    } catch (error) {
      showToast(t("toast.commandError"), error instanceof Error ? error.message : "API error", "error");
      return;
    }
  }
  request.status = decision;
  renderShell();
  showToast(t("toast.decision"), `${request.id} · ${statusText(decision)}`);
}

async function acknowledgeIncident(incidentId) {
  const incident = state.data.incidents.find((item) => item.id === incidentId);
  if (!incident) return;
  if (state.source !== "demo" && state.source !== "offline") {
    try {
      await api.acknowledgeIncident(incidentId, state.data.meta?.session?.subject ?? null);
    } catch (error) {
      showToast(t("toast.commandError"), error instanceof Error ? error.message : "API error", "error");
      return;
    }
  }
  incident.status = "investigating";
  incident.owner = state.data.meta?.session?.display_name ?? incident.owner;
  renderShell();
  showToast(t("toast.incident"), incident.id);
}

function trapModalFocus(event) {
  if (event.key !== "Tab" || !state.modal) return;
  const modal = root?.querySelector(".modal");
  const focusable = [...(modal?.querySelectorAll('button:not(:disabled), input:not(:disabled), select:not(:disabled), [href]') ?? [])];
  if (!focusable.length) return;
  const first = focusable[0];
  const last = focusable.at(-1);
  if (event.shiftKey && document.activeElement === first) {
    event.preventDefault();
    last.focus();
  } else if (!event.shiftKey && document.activeElement === last) {
    event.preventDefault();
    first.focus();
  }
}

root?.addEventListener("click", (event) => {
  const control = event.target.closest("[data-action]");
  if (!control) return;
  const action = control.dataset.action;
  if (action === "open-drawer") {
    state.drawerOpen = true;
    renderShell();
  } else if (action === "close-drawer" || action === "modal-backdrop" && event.target === control) {
    if (action === "modal-backdrop") closeModal();
    else {
      state.drawerOpen = false;
      renderShell();
    }
  } else if (action === "theme") {
    state.theme = state.theme === "dark" ? "light" : "dark";
    preferences.set("campus.theme", state.theme);
    renderShell();
  } else if (action === "refresh") {
    void refreshData({ announce: true });
  } else if (action === "select-map-gate") {
    state.selectedGateId = control.dataset.gateId;
    renderShell();
    queueMicrotask(() => root.querySelector(`[data-map-gate][data-gate-id="${CSS.escape(state.selectedGateId)}"]`)?.focus());
  } else if (action === "select-gate") {
    state.selectedGateId = control.dataset.gateId;
    renderShell();
  } else if (action === "open-gate") {
    state.selectedGateId = control.dataset.gateId;
    globalThis.location.hash = "/gates";
  } else if (action === "open-access") {
    globalThis.location.hash = "/access";
  } else if (action === "gate-command") {
    lastFocusedElement = control;
    state.modal = { gateId: control.dataset.gateId, command: control.dataset.command };
    renderShell();
  } else if (action === "close-modal") {
    closeModal();
  } else if (action === "confirm-command") {
    void confirmGateCommand();
  } else if (action === "request-decision") {
    void decideRequest(control.dataset.requestId, control.dataset.decision);
  } else if (action === "open-first-request") {
    root.querySelector(".request-card button:not(:disabled)")?.focus();
  } else if (action === "directory-kind") {
    state.directoryKind = control.dataset.kind;
    renderShell();
  } else if (action === "ack-incident") {
    void acknowledgeIncident(control.dataset.incidentId);
  } else if (action === "print") {
    globalThis.print();
  } else if (action === "setup-step") {
    state.setupStep = clamp(control.dataset.step, 0, 3);
    renderShell();
  } else if (action === "setup-back") {
    state.setupStep = clamp(state.setupStep - 1, 0, 3);
    renderShell();
  }
});

root?.addEventListener("change", (event) => {
  const control = event.target.closest("[data-action]");
  if (!control) return;
  if (control.dataset.action === "locale") {
    state.locale = normalizeLocale(control.value);
    preferences.set("campus.locale", state.locale);
    renderShell();
  } else if (control.dataset.action === "role") {
    state.role = control.value;
    preferences.set("campus.role", state.role);
    if (Object.values(TENANT_CONFIG.api.demoRoleTokens).includes(state.token)) {
      state.token = TENANT_CONFIG.api.demoRoleTokens[state.role] ?? state.token;
      preferences.set("campus.apiToken", state.token);
      api = createApi();
      void refreshData();
    }
    renderShell();
    showToast(roleLabel());
  }
});

root?.addEventListener("submit", (event) => {
  if (event.target.matches("[data-directory-search]")) {
    event.preventDefault();
    state.directoryQuery = new FormData(event.target).get("query")?.toString() ?? "";
    renderShell();
  }
  if (event.target.matches("[data-setup-form]")) {
    event.preventDefault();
    if (state.setupStep === 0) {
      const submitted = new FormData(event.target);
      state.tenantName = submitted.get("tenantName")?.toString().trim() || state.tenantName;
      state.apiBaseUrl = submitted.get("apiUrl")?.toString().trim() || state.apiBaseUrl;
      state.token = submitted.get("apiToken")?.toString().trim() ?? state.token;
      preferences.set("campus.tenantName", state.tenantName);
      preferences.set("campus.apiBaseUrl", state.apiBaseUrl);
      preferences.set("campus.apiToken", state.token);
      api = createApi();
    }
    if (state.setupStep < 3) {
      state.setupStep += 1;
      renderShell();
    } else {
      showToast(t("toast.setup"), t(`source.detail.${state.source}`));
      void refreshData();
    }
  }
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    if (state.modal) closeModal();
    else if (state.drawerOpen) {
      state.drawerOpen = false;
      renderShell();
    }
  }
  trapModalFocus(event);
});

globalThis.addEventListener("hashchange", () => {
  state.route = resolveRoute(globalThis.location.hash);
  state.drawerOpen = false;
  renderShell();
  queueMicrotask(() => document.querySelector("#main-content")?.focus({ preventScroll: true }));
});

globalThis.addEventListener("offline", () => {
  state.networkOnline = false;
  state.source = "offline";
  renderShell();
});

globalThis.addEventListener("online", () => {
  state.networkOnline = true;
  void refreshData();
});

renderShell();
void refreshData();
