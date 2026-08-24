import { SUPPORTED_LOCALES } from "./config.mjs?v=0.2.3";

const LOCALE_TAGS = Object.freeze({ en: "en-GB", fr: "fr-FR", ar: "ar-MA" });
const ROUTES = new Set([
  "command",
  "gates",
  "access",
  "directory",
  "operations",
  "analytics",
  "setup",
]);

export function normalizeLocale(value, fallback = "en") {
  const locale = String(value ?? "").toLowerCase().split("-")[0];
  return SUPPORTED_LOCALES.includes(locale) ? locale : fallback;
}

export function isRTL(locale) {
  return normalizeLocale(locale) === "ar";
}

export function resolveRoute(hash) {
  const route = String(hash ?? "")
    .replace(/^#\/?/, "")
    .split(/[/?]/)[0]
    .toLowerCase();
  return ROUTES.has(route) ? route : "command";
}

export function translate(messages, key, locale, variables = {}) {
  const selected = normalizeLocale(locale);
  const template = messages[selected]?.[key] ?? messages.en?.[key] ?? key;
  return String(template).replace(/\{(\w+)\}/g, (_, name) =>
    Object.hasOwn(variables, name) ? String(variables[name]) : `{${name}}`,
  );
}

export function escapeHTML(value) {
  return String(value ?? "").replace(
    /[&<>'"]/g,
    (character) =>
      ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        "'": "&#39;",
        '"': "&quot;",
      })[character],
  );
}

export function formatNumber(value, locale, options = {}) {
  return new Intl.NumberFormat(LOCALE_TAGS[normalizeLocale(locale)], options).format(value);
}

export function formatTime(isoDate, locale, timeZone = "Africa/Casablanca") {
  return new Intl.DateTimeFormat(LOCALE_TAGS[normalizeLocale(locale)], {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
    timeZone,
  }).format(new Date(isoDate));
}

export function formatRelativeMinutes(minutes, messages, locale) {
  if (minutes <= 0) return translate(messages, "time.now", locale);
  if (minutes === 1) return translate(messages, "time.minute", locale);
  return translate(messages, "time.minutes", locale, { count: formatNumber(minutes, locale) });
}

export function gateSummary(gates) {
  const safeGates = Array.isArray(gates) ? gates : [];
  return safeGates.reduce(
    (summary, gate) => {
      summary.total += 1;
      if (gate.status === "open") summary.open += 1;
      if (gate.status === "degraded") summary.degraded += 1;
      if (gate.status === "maintenance") summary.maintenance += 1;
      summary.queue += Number(gate.queue ?? 0);
      return summary;
    },
    { total: 0, open: 0, degraded: 0, maintenance: 0, queue: 0 },
  );
}

/** Return only an observation that belongs to the selected gate. */
export function arrivalForGate(arrivals, gateId) {
  const safeArrivals = Array.isArray(arrivals) ? arrivals : [];
  return safeArrivals.find((arrival) => arrival?.gateId === gateId) ?? null;
}

/** Summarize the current collection instead of relying on fixture totals. */
export function deviceHealthSummary(devices) {
  const safeDevices = Array.isArray(devices) ? devices : [];
  const online = safeDevices.filter((device) => device?.status === "online").length;
  return { online, attention: safeDevices.length - online, total: safeDevices.length };
}

export function filterDirectory(records, query = "", kind = "all") {
  const normalizedQuery = String(query).trim().toLocaleLowerCase();
  return (Array.isArray(records) ? records : []).filter((record) => {
    const kindMatches = kind === "all" || record.kind === kind;
    if (!kindMatches) return false;
    if (!normalizedQuery) return true;
    const haystack = [record.name, record.plate, record.organization, record.email, record.status]
      .filter(Boolean)
      .join(" ")
      .toLocaleLowerCase();
    return haystack.includes(normalizedQuery);
  });
}

export function chartScale(values, ceiling = 100) {
  const safeValues = (Array.isArray(values) ? values : []).map((value) =>
    Math.max(0, Number(value) || 0),
  );
  const maximum = Math.max(ceiling, ...safeValues);
  return safeValues.map((value) => Math.round((value / maximum) * 1000) / 10);
}

export function mergeSnapshot(seed, resources) {
  const merged = structuredClone(seed);
  for (const [key, value] of Object.entries(resources ?? {})) {
    if (value == null) continue;
    if (Array.isArray(merged[key]) && !Array.isArray(value)) continue;
    if (!Array.isArray(merged[key]) && typeof merged[key] === "object" && Array.isArray(value)) {
      continue;
    }
    merged[key] = value;
  }
  return merged;
}

export function safeStorage(storage) {
  return {
    get(key, fallback) {
      try {
        return storage?.getItem(key) ?? fallback;
      } catch {
        return fallback;
      }
    },
    set(key, value) {
      try {
        storage?.setItem(key, value);
        return true;
      } catch {
        return false;
      }
    },
  };
}

export function resolveAuthToken(storedToken, role, roleTokens) {
  const explicitToken = String(storedToken ?? "").trim();
  if (explicitToken) return explicitToken;
  return String(roleTokens?.[role] ?? "").trim();
}
