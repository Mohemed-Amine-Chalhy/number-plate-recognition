import { CampusApi } from "./api.mjs?v=0.3.0";
import {
  AGENT_TOOL_POLICY,
  canApproveAgentRun,
  buildAgentDecisionRequest,
  buildAgentRunRequest,
  createIdempotencyKey,
  createReferenceAgentRun,
  decideAgentRunWithRecovery,
  deriveEvidenceCoverage,
  normalizeAgentRun,
  prepareAgentRunDraft,
  summarizeAgentEvidence,
} from "./agentic.mjs?v=0.3.0";
import { buildCampusMapModel, resolveGateMapLabel } from "./campus-map.mjs?v=0.3.0";
import { ROLE_OPTIONS, TENANT_CONFIG } from "./config.mjs?v=0.3.0";
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
} from "./core.mjs?v=0.3.0";
import { DEMO_DATA } from "./demo-data.mjs?v=0.3.0";
import { MESSAGES } from "./i18n.mjs?v=0.3.0";

const root = document.querySelector("#app");
const preferences = safeStorage(globalThis.localStorage);

const NAV_ITEMS = Object.freeze([
  { route: "command", label: "nav.command", icon: "⌘", group: "workspace" },
  { route: "gates", label: "nav.gates", icon: "▦", group: "workspace" },
  { route: "access", label: "nav.access", icon: "✓", group: "workspace" },
  { route: "directory", label: "nav.directory", icon: "◎", group: "manage" },
  { route: "operations", label: "nav.operations", icon: "!", group: "manage" },
  { route: "agent", label: "nav.agent", icon: "✦", group: "manage" },
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
  agentBusy: false,
  agentError: null,
  agentRunDraft: null,
  agentObjective: null,
  agentGateId: "gate-residential",
  agentRun: createReferenceAgentRun(DEMO_DATA, { gateId: "gate-residential" }),
  provenance: {
    liveResources: [],
    liveGateIds: [],
    sessionConfirmed: false,
    agentEndpointConfirmed: false,
  },
  networkOnline: globalThis.navigator.onLine,
};

let toastTimer = null;
let lastFocusDescriptor = null;
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

function cssEscape(value) {
  const text = String(value ?? "");
  return globalThis.CSS?.escape ? globalThis.CSS.escape(text) : text.replace(/[^a-zA-Z0-9_-]/g, "\\$&");
}

function rememberFocus(control) {
  if (!control?.dataset) {
    lastFocusDescriptor = null;
    return;
  }
  lastFocusDescriptor = {
    action: control.dataset.action ?? null,
    decision: control.dataset.decision ?? null,
    gateId: control.dataset.gateId ?? null,
    command: control.dataset.command ?? null,
    requestId: control.dataset.requestId ?? null,
  };
}

function restoreRememberedFocus() {
  const descriptor = lastFocusDescriptor;
  lastFocusDescriptor = null;
  queueMicrotask(() => {
    if (!descriptor?.action) {
      root?.querySelector("#main-content")?.focus({ preventScroll: true });
      return;
    }
    const attributes = Object.entries(descriptor)
      .filter(([, value]) => value)
      .map(([key, value]) => `[data-${key.replace(/[A-Z]/g, (letter) => `-${letter.toLowerCase()}`)}="${cssEscape(value)}"]`)
      .join("");
    const target = root?.querySelector(attributes);
    (target ?? root?.querySelector("#main-content"))?.focus({ preventScroll: true });
  });
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

function gateForAgentRun(run) {
  if (run?.mode === "reference") {
    return DEMO_DATA.gates.find((gate) => gate.id === run.gateId) ?? gateById(run?.gateId);
  }
  return gateById(run?.gateId);
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
  const backgroundInert = state.modal ? 'inert aria-hidden="true"' : "";
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
      <button class="drawer-backdrop" type="button" data-action="close-drawer" ${backgroundInert} aria-label="${h(
        t("action.close"),
      )}"></button>
      <aside class="sidebar" ${backgroundInert} aria-label="${h(TENANT_CONFIG.branding.productName)}">
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
      <header class="topbar" ${backgroundInert}>
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
      <main class="main-content" id="main-content" tabindex="-1" ${backgroundInert}>${renderCurrentPage()}</main>
      ${renderModal()}
      <div class="toast-region" aria-live="polite" aria-atomic="true" ${backgroundInert}>${renderToast()}</div>
    </div>`;
  bindLogoFallback();
  if (state.modal) {
    queueMicrotask(() =>
      root
        .querySelector(state.agentBusy ? ".modal" : "[data-modal-initial], [data-modal-primary]")
        ?.focus(),
    );
  }
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
    agent: renderAgentOperations,
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
  return `<div class="page-stack command-page">
    ${renderPageIntro(`<button class="soft-button" type="button" data-action="refresh"><span aria-hidden="true">↻</span>${h(
      t("action.refresh"),
    )}</button>`)}
    ${renderMetrics()}
    ${renderAgentBriefing()}
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

function renderAgentBriefing() {
  const run = state.agentRun;
  const gate = gateForAgentRun(run);
  const coverage = deriveEvidenceCoverage(run);
  const coverageText = coverage == null
    ? t("agent.coverageUnavailable")
    : `${formatNumber(coverage, state.locale)}%`;
  return `<aside class="agent-briefing" aria-labelledby="agent-briefing-title">
    <span class="agent-orb" aria-hidden="true">✦</span>
    <div class="agent-briefing-copy">
      <span class="eyebrow">${h(t("agent.eyebrow"))}</span>
      <strong id="agent-briefing-title">${h(t("agent.briefingTitle"))}</strong>
      <small>${h(
        t("agent.briefingBody", {
          gate: localized(gate?.name),
          coverage: coverageText,
        }),
      )}</small>
    </div>
    <div class="agent-briefing-state">
      <span class="agent-source-badge compact ${h(run.mode)}"><span aria-hidden="true">${
        run.mode === "reference" ? "◇" : "●"
      }</span>${h(run.mode === "reference" ? t("agent.referenceTrace") : t("agent.liveTrace"))}</span>
      ${agentRunStatusPill(run?.status)}
      <a class="soft-button compact-button" href="#/agent">${h(t("agent.openWorkspace"))}<span aria-hidden="true">→</span></a>
    </div>
  </aside>`;
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

function agentRunStatusPill(status = "idle") {
  const normalized = String(status).toLowerCase();
  return `<span class="agent-state agent-state-${h(normalized)}" role="status"><i aria-hidden="true"></i>${h(
    t(`agent.status.${normalized}`),
  )}</span>`;
}

function agentToolLabel(toolName) {
  const key = `agent.tool.${toolName}`;
  const label = t(key);
  return label === key ? String(toolName).replaceAll("_", " ") : label;
}

function agentRationale(step, run) {
  const key = `agent.rationale.${step.toolName}`;
  const localizedRationale = t(key);
  return run?.mode === "reference" || !step.rationale
    ? localizedRationale === key
      ? step.rationale
      : localizedRationale
    : step.rationale;
}

function agentStepStatus(step) {
  const key = `agent.stepStatus.${step.status}`;
  const label = t(key);
  return `<span class="agent-step-status ${h(step.status)}"><span aria-hidden="true">${
    step.status === "succeeded"
      ? "✓"
      : step.status === "awaiting_approval"
        ? "Ⅱ"
        : step.status === "skipped"
          ? "–"
          : step.status === "failed"
            ? "!"
            : "·"
  }</span>${h(label)}</span>`;
}

function agentOutputLabel(key) {
  const translationKey = `agent.output.${key}`;
  const label = t(translationKey);
  return label === translationKey ? String(key).replaceAll("_", " ") : label;
}

function agentOutputValue(key, value) {
  if (key === "status") {
    const statusKey = `status.${String(value).toLowerCase()}`;
    const translatedStatus = t(statusKey);
    if (translatedStatus !== statusKey) return translatedStatus;
  }
  if (key === "reference_result" && value === "human_handoff_recorded") {
    return t("agent.value.human_handoff_recorded");
  }
  if (typeof value === "boolean") return value ? t("agent.yes") : t("agent.no");
  return value;
}

function renderAgentEvidence(step) {
  const entries = summarizeAgentEvidence(step, {
    statusFormatter: (status) => agentOutputValue("status", status),
    noneLabel: t("agent.none"),
    structuredLabel: t("agent.structuredResult"),
    recordLabel: (count) => t("agent.records", { count: formatNumber(count, state.locale) }),
  });
  if (!entries.length) {
    if (step.status === "awaiting_approval") {
      return `<p class="agent-paused-note"><span aria-hidden="true">Ⅱ</span>${h(t("agent.pausedBeforeTool"))}</p>`;
    }
    if (step.status === "skipped") {
      return `<p class="agent-paused-note skipped"><span aria-hidden="true">↳</span>${h(t("agent.branchSkipped"))}</p>`;
    }
    return "";
  }
  return `<dl class="agent-evidence">${entries
    .slice(0, 5)
    .map(
      ({ key, value }) => `<div><dt>${h(agentOutputLabel(key))}</dt><dd>${h(
        agentOutputValue(key, value),
      )}</dd></div>`,
    )
    .join("")}</dl>`;
}

function renderAgentChecks(step) {
  if (!step.policyChecks?.length) return "";
  return `<div class="agent-checks" aria-label="${h(t("agent.policyChecks"))}">${step.policyChecks
    .map(
      (check) => {
        const appearance = check.outcome === "approval_required"
          ? "approval"
          : check.outcome === "allow"
            ? "passed"
            : "failed";
        const outcomeKey = `agent.policyOutcome.${check.outcome}`;
        return `<details class="agent-check ${h(appearance)}" ${check.blocked ? "open" : ""}>
          <summary><span aria-hidden="true">${
            check.outcome === "approval_required" ? "Ⅱ" : check.outcome === "allow" ? "✓" : "!"
          }</span><span>${h(agentOutputLabel(check.name))}</span><small>${h(t(outcomeKey))}</small></summary>
          <div><p>${h(check.detail || t("agent.policyDetailUnavailable"))}</p><small>${h(
            t("agent.policySource", {
              policy: check.policyName ?? t("agent.coverageUnavailable"),
              version: check.policyVersion ?? t("agent.coverageUnavailable"),
            }),
          )}</small></div>
        </details>`;
      },
    )
    .join("")}</div>`;
}

function renderAgentFailure(code, detail) {
  if (!code && !detail) return "";
  return `<dl class="agent-failure" role="alert">
    ${code ? `<div><dt>${h(t("agent.errorCode"))}</dt><dd><code>${h(code)}</code></dd></div>` : ""}
    ${detail ? `<div><dt>${h(t("agent.errorDetail"))}</dt><dd>${h(detail)}</dd></div>` : ""}
  </dl>`;
}

function renderAgentDecision(run) {
  if (run?.status === "awaiting_approval" && run.pendingApproval) {
    const isInvestigation = run.pendingApproval.toolName === "start_incident_investigation";
    const approvalAllowed = canApproveAgentRun(run);
    return `<section class="panel agent-decision-card" aria-labelledby="agent-decision-heading">
      <div class="agent-decision-icon" aria-hidden="true">Ⅱ</div>
      <span class="eyebrow">${h(t("agent.approvalBoundary"))}</span>
      <h3 id="agent-decision-heading">${h(t("agent.humanDecisionRequired"))}</h3>
      <p>${h(t(isInvestigation ? "agent.approval.reasonInvestigation" : "agent.approval.reasonCreate"))}</p>
      <div class="agent-action-preview">
        <small>${h(t("agent.proposedAction"))}</small>
        <strong>${h(agentToolLabel(run.pendingApproval.toolName))}</strong>
        <span>${h(t("agent.noActuation"))}</span>
      </div>
      ${approvalAllowed
        ? `<div class="agent-decision-actions">
            <button class="ghost-button" type="button" data-action="agent-decision" data-decision="rejected" ${
              state.agentBusy ? "disabled" : ""
            }>${h(t("agent.rejectHandoff"))}</button>
            <button class="primary-button" type="button" data-action="agent-decision" data-decision="approved" ${
              state.agentBusy ? "disabled" : ""
            }>${h(t("agent.approveHandoff"))}</button>
          </div>`
        : `<div class="agent-contract-blocked" role="alert"><strong>${h(t("agent.contractBlocked"))}</strong><span>${h(
            t("agent.contractBlockedBody"),
          )}</span></div>`}
    </section>`;
  }
  const decision = run?.approval?.decision ?? (run?.status === "rejected" ? "rejected" : null);
  const pendingRun = run?.status === "running";
  const failedRun = run?.status === "failed";
  const title = decision === "rejected"
    ? t("agent.handoffRejected")
    : pendingRun
      ? t("agent.analysisInProgress")
      : failedRun
        ? t("agent.analysisFailed")
        : t("agent.noDecisionPending");
  const body = decision === "rejected"
    ? t("agent.handoffRejectedBody")
    : pendingRun
      ? t("agent.analysisInProgressBody")
      : failedRun
        ? t("agent.analysisFailedBody")
        : t("agent.noDecisionPendingBody");
  return `<section class="panel agent-decision-card decision-complete" aria-labelledby="agent-decision-heading">
    <div class="agent-decision-icon" aria-hidden="true">${decision === "rejected" || failedRun ? "×" : pendingRun ? "…" : "✓"}</div>
    <span class="eyebrow">${h(t("agent.approvalBoundary"))}</span>
    <h3 id="agent-decision-heading">${h(title)}</h3>
    <p>${h(body)}</p>
    ${run?.approval
      ? `<dl class="agent-decision-record">
          <div><dt>${h(t("agent.decisionActor"))}</dt><dd>${h(run.approval.decidedBy ?? "—")}</dd></div>
          <div><dt>${h(t("agent.decisionReason"))}</dt><dd>${h(run.approval.reason || "—")}</dd></div>
        </dl>`
      : ""}
    ${renderAgentFailure(run?.failureCode, run?.failureDetail)}
  </section>`;
}

function renderAgentOperations() {
  const run = state.agentRun;
  const gate = gateForAgentRun(run);
  const coverage = deriveEvidenceCoverage(run);
  const coverageText = coverage == null
    ? t("agent.coverageUnavailable")
    : `${formatNumber(coverage, state.locale)}%`;
  const succeededReads = run.steps.filter(
    (step) => step.risk === "read_only" && step.status === "succeeded",
  ).length;
  const operationalWrites =
    run.mode === "live"
      ? run.steps.filter((step) => step.risk === "consequential" && step.status === "succeeded").length
      : 0;
  const writeDetailKey =
    run.mode === "reference" && run.status !== "awaiting_approval"
      ? "agent.referenceNoMutation"
      : operationalWrites
        ? "agent.executedAfterApproval"
        : "agent.pausedBeforeMutation";
  const traceLabel = run.mode === "reference" ? t("agent.referenceTrace") : t("agent.liveTrace");
  return `<div class="page-stack agent-page">
    ${renderPageIntro(`<span class="agent-source-badge ${h(run.mode)}"><span aria-hidden="true">${
      run.mode === "reference" ? "◇" : "●"
    }</span>${h(traceLabel)}</span>`)}
    <section class="panel agent-hero" aria-labelledby="agent-objective-heading">
      <div class="agent-hero-heading">
        <span class="agent-orb large" aria-hidden="true">✦</span>
        <div><span class="eyebrow">${h(t("agent.eyebrow"))}</span><h3 id="agent-objective-heading">${h(
          t("agent.heroTitle"),
        )}</h3><p>${h(t("agent.heroBody"))}</p></div>
        ${agentRunStatusPill(run.status)}
      </div>
      <form class="agent-objective-form" data-agent-run-form>
        <label class="field agent-objective-field"><span>${h(t("agent.objective"))}</span><textarea class="field-input" name="objective" rows="2" required>${h(
          state.agentRunDraft?.objective ??
            state.agentObjective ??
            (run.mode === "reference" ? t("agent.defaultObjective") : run.objective),
        )}</textarea></label>
        <label class="field"><span>${h(t("agent.scope"))}</span><select class="field-select" name="gateId" data-action="agent-gate">${state.data.gates
          .map(
            (item) => `<option value="${h(item.id)}" ${item.id === state.agentGateId ? "selected" : ""}>${h(
              `${item.code} · ${localized(item.name)}`,
            )}</option>`,
          )
          .join("")}</select></label>
        <button class="primary-button agent-run-button" type="submit" ${state.agentBusy ? "disabled" : ""}><span aria-hidden="true">${
          state.agentBusy ? "…" : "✦"
        }</span>${h(
          state.agentBusy
            ? t("agent.running")
            : state.agentRunDraft?.status === "failed"
              ? t("agent.retryAnalysis")
              : t("agent.runAnalysis"),
        )}</button>
      </form>
      ${canUseLiveAgent(state.agentGateId)
        ? `<p class="agent-provenance-note live"><span aria-hidden="true">●</span>${h(t("agent.liveGateConfirmed"))}</p>`
        : `<p class="agent-provenance-note reference"><span aria-hidden="true">◇</span>${h(t("agent.referenceFallback"))}</p>`}
      ${state.agentError ? `<p class="agent-error" role="alert">${h(state.agentError)}</p>` : ""}
    </section>
    <div class="agent-metric-grid">
      <article><span>${h(t("agent.evidenceCoverage"))}</span><strong>${h(coverageText)}</strong><small>${h(
        t("agent.coverageBasis"),
      )}</small></article>
      <article><span>${h(t("agent.readToolsCompleted"))}</span><strong>${h(succeededReads)}/${h(
        run.steps.filter((step) => step.risk === "read_only").length,
      )}</strong><small>${h(t("agent.readOnlyBoundary"))}</small></article>
      <article><span>${h(t("agent.operationalWrites"))}</span><strong>${h(operationalWrites)}</strong><small>${h(
        t(writeDetailKey),
      )}</small></article>
      <article><span>${h(t("agent.scope"))}</span><strong>${h(gate?.code ?? "—")}</strong><small>${h(
        localized(gate?.name),
      )}</small></article>
    </div>
    <div class="agent-workbench">
      <section class="panel agent-trace-panel" aria-labelledby="agent-plan-heading">
        <header class="panel-header"><div><h3 id="agent-plan-heading">${h(t("agent.planAndTrace"))}</h3><p>${h(
          run.mode === "reference" ? t("agent.planSummary") : run.plan.summary || t("agent.planSummary"),
        )}</p></div><span class="metric-chip">${h(t("agent.steps", { count: run.plan.steps.length }))}</span></header>
        <ol class="agent-plan-list" aria-label="${h(t("agent.plan"))}">${run.plan.steps
          .map((step) => {
            const executed = run.steps.find((item) => item.sequence === step.sequence) ?? {
              ...step,
              status: "pending",
            };
            return `<li class="agent-plan-step ${h(executed.status)}">
              <span class="agent-sequence">${h(step.sequence)}</span>
              <div><strong>${h(agentToolLabel(step.toolName))}</strong><small>${h(
                agentRationale(step, run),
              )}</small></div>
              <span class="agent-risk ${h(step.risk)}">${h(t(`agent.risk.${step.risk}`))}</span>
              ${agentStepStatus(executed)}
            </li>`;
          })
          .join("")}</ol>
        <div class="agent-trace-heading"><div><span class="eyebrow">${h(t("agent.executionTrace"))}</span><strong>${h(
          t("agent.evidenceByStep"),
        )}</strong></div><code>${h(run.trace.traceId ?? run.id)}</code></div>
        <div class="agent-trace-list">${run.steps
          .map(
            (step) => `<article class="agent-trace-step ${h(step.status)}">
              <div class="agent-trace-line"><span class="agent-tool-icon" aria-hidden="true">${
                step.risk === "read_only" ? "⌕" : "↗"
              }</span><div><strong>${h(agentToolLabel(step.toolName))}</strong><small>${h(
                t(`agent.risk.${step.risk}`),
              )} · ${h(step.id)}</small></div>${agentStepStatus(step)}</div>
              ${renderAgentEvidence(step)}
              ${renderAgentFailure(step.errorCode, step.errorDetail)}
              ${renderAgentChecks(step)}
            </article>`,
          )
          .join("")}</div>
      </section>
      <aside class="agent-side-stack">
        ${renderAgentDecision(run)}
        <section class="panel agent-policy-card" aria-labelledby="agent-policy-heading">
          <header><span class="agent-policy-icon" aria-hidden="true">⌾</span><div><span class="eyebrow">${h(
            t("agent.guardrails"),
          )}</span><h3 id="agent-policy-heading">${h(t("agent.boundedTools"))}</h3></div></header>
          <p>${h(t("agent.boundedToolsBody"))}</p>
          <ul>${AGENT_TOOL_POLICY.tools
            .map(
              (tool) => `<li><span>${h(agentToolLabel(tool.name))}</span><small class="agent-risk ${h(tool.risk)}">${h(
                t(`agent.risk.${tool.risk}`),
              )}</small></li>`,
            )
            .join("")}</ul>
        </section>
        <section class="panel agent-provenance-card" aria-labelledby="agent-provenance-heading">
          <header><span class="eyebrow">${h(t("agent.provenance"))}</span><h3 id="agent-provenance-heading">${h(
            t("agent.traceability"),
          )}</h3></header>
          <dl>
            <div><dt>${h(t("agent.traceId"))}</dt><dd><code>${h(run.trace.traceId ?? "—")}</code></dd></div>
            <div><dt>${h(t("agent.correlationId"))}</dt><dd><code>${h(
              run.trace.correlationId ?? "—",
            )}</code></dd></div>
            <div><dt>${h(t("agent.planner"))}</dt><dd>${h(run.trace.planner)} · ${h(
              run.trace.plannerVersion,
            )}</dd></div>
            <div><dt>${h(t("agent.policy"))}</dt><dd>${h(run.trace.policy)} · ${h(
              run.trace.policyVersion,
            )}</dd></div>
          </dl>
          <div class="agent-audit-list">${run.auditEvents
            .slice(-3)
            .map(
              (event) => `<div><span aria-hidden="true">●</span><p><strong>${h(
                t(`agent.audit.${event.eventType}`) === `agent.audit.${event.eventType}`
                  ? event.summary || event.eventType
                  : t(`agent.audit.${event.eventType}`),
              )}</strong><small>${h(event.actorType)} · ${h(event.actorId)}</small></p></div>`,
            )
            .join("")}</div>
        </section>
      </aside>
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
  if (state.modal.type === "agent-decision") return renderAgentDecisionModal();
  const gate = gateById(state.modal.gateId);
  const commandLabels = {
    open: t("action.openBarrier"),
    hold: t("action.holdLane"),
    intercom: t("action.intercom"),
  };
  return `<div class="modal-backdrop" data-action="modal-backdrop">
    <section class="modal" role="dialog" aria-modal="true" tabindex="-1" aria-labelledby="modal-heading" aria-describedby="modal-description">
      <header class="modal-header"><div><span class="eyebrow">${h(gate.code)}</span><h2 id="modal-heading">${h(
        t("workspace.commandTitle"),
      )}</h2></div><button class="icon-button" type="button" data-action="close-modal" aria-label="${h(
        t("action.close"),
      )}">×</button></header>
      <div class="modal-body"><p id="modal-description">${h(t("workspace.commandBody"))}</p><div class="modal-summary"><strong>${h(
        commandLabels[state.modal.command] ?? state.modal.command,
      )}</strong><span>${h(localized(gate.name))} · ${h(gate.operator)}</span>${statusPill(gate.status)}</div></div>
      <footer class="modal-footer"><button class="ghost-button" type="button" data-action="close-modal" data-modal-initial>${h(
        t("action.cancel"),
      )}</button><button class="primary-button" type="button" data-action="confirm-command" data-modal-primary ${
        state.busy ? "disabled" : ""
      }>${h(t("action.confirm"))}</button></footer>
  </section>
  </div>`;
}

function renderAgentDecisionModal() {
  const decision = state.modal?.decision;
  const approving = decision === "approved";
  const run = state.agentRun;
  const reason = state.modal?.reason ?? "";
  const normalizedReasonLength = reason.trim().length;
  const reasonInvalid = normalizedReasonLength < 3 || normalizedReasonLength > 500;
  return `<div class="modal-backdrop" data-action="modal-backdrop">
    <section class="modal" role="dialog" aria-modal="true" tabindex="-1" aria-labelledby="modal-heading" aria-describedby="modal-description">
      <header class="modal-header"><div><span class="eyebrow">${h(t("agent.approvalBoundary"))}</span><h2 id="modal-heading">${h(
        approving ? t("agent.modalApproveTitle") : t("agent.modalRejectTitle"),
      )}</h2></div><button class="icon-button" type="button" data-action="close-modal" ${state.agentBusy ? "disabled" : ""} aria-label="${h(
        t("action.close"),
      )}">×</button></header>
      <div class="modal-body"><p id="modal-description">${h(
        approving
          ? t(run.mode === "reference" ? "agent.modalApproveBodyReference" : "agent.modalApproveBody")
          : t("agent.modalRejectBody"),
      )}</p><div class="modal-summary"><strong>${h(
        agentToolLabel(run.pendingApproval?.toolName),
      )}</strong><span>${h(localized(gateForAgentRun(run)?.name))}</span><span class="agent-risk consequential">${h(
        t("agent.risk.consequential"),
      )}</span></div>
      <label class="field agent-decision-reason" for="agent-decision-reason"><span>${h(t("agent.reasonLabel"))}</span>
        <textarea class="field-input" id="agent-decision-reason" rows="4" required minlength="3" maxlength="500" data-action="agent-decision-reason" aria-describedby="agent-reason-help${
          state.modal?.error ? " agent-reason-error" : ""
        }" aria-invalid="${Boolean(state.modal?.error)}" ${state.agentBusy ? "disabled" : ""}>${h(reason)}</textarea>
        <small id="agent-reason-help">${h(t("agent.reasonHelp"))}</small>
      </label>
      ${state.modal?.error ? `<p class="agent-modal-error" id="agent-reason-error" role="alert">${h(state.modal.error)}</p>` : ""}
      <p class="agent-modal-note"><span aria-hidden="true">⌾</span>${h(t("agent.modalAuditNote"))}</p></div>
      <footer class="modal-footer"><button class="ghost-button" type="button" data-action="close-modal" data-modal-initial ${
        state.agentBusy ? "disabled" : ""
      }>${h(
        t("action.cancel"),
      )}</button><button class="${approving ? "primary-button" : "danger-button"}" type="button" data-action="confirm-agent-decision" data-modal-primary ${
        state.agentBusy || reasonInvalid ? "disabled" : ""
      }>${h(approving ? t("agent.confirmApprove") : t("agent.confirmReject"))}</button></footer>
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
  if (state.modal?.type === "agent-decision" && state.agentBusy) return;
  state.modal = null;
  state.busy = false;
  renderShell();
  restoreRememberedFocus();
}

function emptyProvenance() {
  return {
    liveResources: [],
    liveGateIds: [],
    sessionConfirmed: false,
    agentEndpointConfirmed: false,
  };
}

async function refreshData({ announce = false } = {}) {
  const previousData = state.data;
  if (!state.networkOnline) {
    state.data = structuredClone(DEMO_DATA);
    state.source = "offline";
    state.provenance = emptyProvenance();
    state.agentGateId = reconcileAgentGateId(previousData, state.agentGateId);
    renderShell();
    return;
  }
  state.source = "loading";
  renderShell();
  try {
    const snapshot = await api.loadSnapshot(structuredClone(DEMO_DATA));
    state.data = snapshot.data;
    state.source = snapshot.source;
    state.provenance = snapshot.provenance ?? emptyProvenance();
    if (!state.data.gates.some((gate) => gate.id === state.selectedGateId)) {
      state.selectedGateId = state.data.gates[0]?.id ?? state.selectedGateId;
    }
    state.agentGateId = reconcileAgentGateId(previousData, state.agentGateId);
  } catch {
    state.data = structuredClone(DEMO_DATA);
    state.source = "demo";
    state.provenance = emptyProvenance();
    state.agentGateId = reconcileAgentGateId(previousData, state.agentGateId);
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

function reconcileAgentGateId(previousData, preferredGateId) {
  if (state.data.gates.some((gate) => gate.id === preferredGateId)) return preferredGateId;
  const previousCode = previousData?.gates?.find((gate) => gate.id === preferredGateId)?.code;
  return (
    state.data.gates.find((gate) => previousCode && gate.code === previousCode)?.id ??
    state.data.gates.find((gate) => gate.status === "degraded")?.id ??
    state.data.gates[0]?.id ??
    preferredGateId
  );
}

function canUseLiveAgent(gateId) {
  return (
    state.networkOnline &&
    ["live", "hybrid"].includes(state.source) &&
    state.provenance.sessionConfirmed === true &&
    state.provenance.liveGateIds.includes(gateId)
  );
}

function referenceGateIdFor(gateId) {
  const currentCode = state.data.gates.find((gate) => gate.id === gateId)?.code;
  return (
    DEMO_DATA.gates.find((gate) => gate.id === gateId)?.id ??
    DEMO_DATA.gates.find((gate) => currentCode && gate.code === currentCode)?.id ??
    DEMO_DATA.gates.find((gate) => gate.status === "degraded")?.id ??
    DEMO_DATA.gates[0]?.id
  );
}

async function runAgentAnalysis(objective, requestedGateId) {
  if (state.agentBusy) return;
  const gateId = state.data.gates.some((gate) => gate.id === requestedGateId)
    ? requestedGateId
    : state.data.gates[0]?.id;
  if (!gateId) return;
  state.agentBusy = true;
  state.agentError = null;
  state.agentGateId = gateId;
  state.agentObjective = objective;
  const liveEligible = canUseLiveAgent(gateId);
  if (liveEligible) {
    state.agentRunDraft = prepareAgentRunDraft(
      state.agentRunDraft,
      { objective, gateId },
      () => createIdempotencyKey(
        "agent-run",
        globalThis.crypto?.randomUUID?.bind(globalThis.crypto),
      ),
    );
  } else {
    state.agentRunDraft = null;
  }
  renderShell();
  try {
    if (liveEligible) {
      const payload = buildAgentRunRequest({
        objective: state.agentRunDraft.objective,
        gateId: state.agentRunDraft.gateId,
        idempotencyKey: state.agentRunDraft.idempotencyKey,
      });
      state.agentRun = normalizeAgentRun(await api.createAgentRun(payload));
      state.agentRunDraft = null;
      state.provenance.agentEndpointConfirmed = true;
    } else {
      state.agentRun = createReferenceAgentRun(DEMO_DATA, {
        objective,
        gateId: referenceGateIdFor(gateId),
      });
    }
  } catch (error) {
    state.agentError = error instanceof Error ? error.message : t("agent.runError");
    if (state.agentRunDraft) state.agentRunDraft.status = "failed";
  } finally {
    state.agentBusy = false;
    renderShell();
  }
}

function applyReferenceAgentDecision(decision, reason) {
  const run = structuredClone(state.agentRun);
  const step = run.steps.find((item) => item.id === run.pendingApproval?.stepId);
  if (step) {
    step.status = decision === "approved" ? "succeeded" : "skipped";
    step.output =
      decision === "approved"
        ? { reference_result: "human_handoff_recorded", live_mutation: false }
        : {};
  }
  run.status = decision === "approved" ? "completed" : "rejected";
  run.approval = {
    decision,
    reason,
    decidedBy: state.data.meta?.session?.display_name ?? roleLabel(),
    decidedAt: new Date().toISOString(),
  };
  run.pendingApproval = null;
  run.auditEvents.push({
    id: `${run.id}-audit-decision`,
    eventType: decision === "approved" ? "agent.approval.approved" : "agent.approval.rejected",
    actorType: "human",
    actorId: state.data.meta?.session?.subject ?? state.role,
    occurredAt: run.approval.decidedAt,
    summary: run.approval.reason,
    metadata: { reference_only: true },
  });
  state.agentRun = run;
}

async function confirmAgentDecision() {
  if (state.modal?.type !== "agent-decision" || state.agentBusy) return;
  const modal = state.modal;
  const decision = modal.decision;
  const reason = modal.reason ?? "";
  const submittedReason = reason.trim();
  if (submittedReason.length < 3 || submittedReason.length > 500) {
    state.modal.error = t("agent.reasonRequired");
    renderShell();
    return;
  }
  const runId = state.agentRun.id;
  state.agentBusy = true;
  renderShell();
  try {
    if (state.agentRun.mode === "live") {
      const payload = buildAgentDecisionRequest({
        decision,
        reason: submittedReason,
        idempotencyKey: modal.idempotencyKey,
      });
      const result = await decideAgentRunWithRecovery(api, runId, payload);
      state.agentRun = result.run;
      state.provenance.agentEndpointConfirmed = true;
    } else {
      applyReferenceAgentDecision(decision, submittedReason);
    }
    state.modal = null;
    state.agentBusy = false;
    showToast(
      decision === "approved" ? t("agent.toastApproved") : t("agent.toastRejected"),
      state.agentRun.mode === "reference" ? t("agent.referenceDecisionDetail") : runId,
    );
    restoreRememberedFocus();
  } catch (error) {
    state.agentBusy = false;
    state.modal.error = error instanceof Error ? error.message : t("agent.decisionError");
    renderShell();
  }
}

function trapModalFocus(event) {
  if (event.key !== "Tab" || !state.modal) return;
  const modal = root?.querySelector(".modal");
  const focusable = [...(modal?.querySelectorAll('button:not(:disabled), input:not(:disabled), textarea:not(:disabled), select:not(:disabled), [href]') ?? [])];
  if (!focusable.length) {
    event.preventDefault();
    modal?.focus();
    return;
  }
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
    rememberFocus(control);
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
  } else if (action === "agent-decision") {
    if (!canApproveAgentRun(state.agentRun)) return;
    rememberFocus(control);
    state.modal = {
      type: "agent-decision",
      decision: control.dataset.decision,
      reason: "",
      idempotencyKey: createIdempotencyKey(
        // The run is already part of the decision endpoint and database scope.
        // Keeping it out of the key preserves the API's 80-character contract.
        "agent-decision",
        globalThis.crypto?.randomUUID?.bind(globalThis.crypto),
      ),
      error: null,
    };
    renderShell();
  } else if (action === "confirm-agent-decision") {
    void confirmAgentDecision();
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

root?.addEventListener("input", (event) => {
  const control = event.target.closest('[data-action="agent-decision-reason"]');
  if (!control || state.modal?.type !== "agent-decision") return;
  state.modal.reason = control.value;
  state.modal.error = null;
  const confirm = root.querySelector('[data-action="confirm-agent-decision"]');
  const reasonLength = control.value.trim().length;
  if (confirm) confirm.disabled = reasonLength < 3 || reasonLength > 500 || state.agentBusy;
  control.removeAttribute("aria-invalid");
  root.querySelector("#agent-reason-error")?.remove();
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
  } else if (control.dataset.action === "agent-gate") {
    state.agentGateId = control.value;
    state.agentRunDraft = null;
    state.agentError = null;
    if (state.agentRun.mode === "reference") {
      state.agentRun = createReferenceAgentRun(DEMO_DATA, {
        gateId: referenceGateIdFor(state.agentGateId),
        objective: t("agent.defaultObjective"),
      });
    }
    renderShell();
  }
});

root?.addEventListener("submit", (event) => {
  if (event.target.matches("[data-agent-run-form]")) {
    event.preventDefault();
    const submitted = new FormData(event.target);
    void runAgentAnalysis(
      submitted.get("objective")?.toString().trim() || t("agent.defaultObjective"),
      submitted.get("gateId")?.toString() || state.agentGateId,
    );
  }
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
