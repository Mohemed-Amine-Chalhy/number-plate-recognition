/**
 * Pure presentation model for the illustrated campus map.
 *
 * The API owns geographic coordinates while each tenant owns its illustration
 * and optional pin overrides. Keeping projection here prevents transport data
 * from depending on the dimensions or layout of a particular map asset.
 */

const DEFAULT_EDGE_INSET = 6;

function finiteNumber(value) {
  if (value === null || value === undefined || value === "") return null;
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

/** Clamp a map coordinate to the percentage coordinate space. */
export function clampPercent(value, fallback = 50) {
  const fallbackNumber = finiteNumber(fallback) ?? 50;
  const numeric = finiteNumber(value) ?? fallbackNumber;
  return Math.min(100, Math.max(0, numeric));
}

function normalizeInset(value) {
  return Math.min(49, clampPercent(value, DEFAULT_EDGE_INSET));
}

function normalizedPosition(position, edgeInset, source) {
  if (!position || typeof position !== "object") return null;
  const x = finiteNumber(position.x);
  const y = finiteNumber(position.y);
  if (x === null || y === null) return null;

  const inset = normalizeInset(edgeInset);
  return {
    x: Math.min(100 - inset, Math.max(inset, x)),
    y: Math.min(100 - inset, Math.max(inset, y)),
    source,
  };
}

function validBounds(bounds) {
  if (!bounds || typeof bounds !== "object") return null;
  const north = finiteNumber(bounds.north);
  const east = finiteNumber(bounds.east);
  const south = finiteNumber(bounds.south);
  const west = finiteNumber(bounds.west);
  if (
    north === null ||
    east === null ||
    south === null ||
    west === null ||
    north > 90 ||
    south < -90 ||
    east > 180 ||
    west < -180 ||
    north <= south ||
    east <= west
  ) {
    return null;
  }
  return { north, east, south, west };
}

/**
 * Project latitude/longitude into a north-up percentage coordinate space.
 * Bounds that cross the antimeridian are intentionally rejected; callers can
 * supply an explicit pin override for those uncommon illustrated maps.
 */
export function projectCoordinates(latitude, longitude, bounds, edgeInset = 0) {
  const latitudeNumber = finiteNumber(latitude);
  const longitudeNumber = finiteNumber(longitude);
  const normalizedBounds = validBounds(bounds);
  if (
    latitudeNumber === null ||
    longitudeNumber === null ||
    latitudeNumber < -90 ||
    latitudeNumber > 90 ||
    longitudeNumber < -180 ||
    longitudeNumber > 180 ||
    normalizedBounds === null
  ) {
    return null;
  }

  const { north, east, south, west } = normalizedBounds;
  const rawX = ((longitudeNumber - west) / (east - west)) * 100;
  const rawY = ((north - latitudeNumber) / (north - south)) * 100;
  return normalizedPosition({ x: rawX, y: rawY }, edgeInset, "geographic");
}

function configuredPosition(gate, mapConfig, edgeInset) {
  const positions = mapConfig?.gatePositions;
  if (!positions || typeof positions !== "object") return null;
  const byId = gate?.id ? positions[gate.id] : null;
  const byCode = gate?.code ? positions[gate.code] : null;
  return normalizedPosition(byId ?? byCode, edgeInset, "configured");
}

function gatePosition(gate, edgeInset) {
  return (
    normalizedPosition(gate?.mapPosition, edgeInset, "gate") ??
    normalizedPosition({ x: gate?.x, y: gate?.y }, edgeInset, "gate")
  );
}

function fallbackPosition(index, total, edgeInset) {
  const safeTotal = Math.max(1, Math.trunc(finiteNumber(total) ?? 1));
  const safeIndex = Math.max(0, Math.trunc(finiteNumber(index) ?? 0)) % safeTotal;
  if (safeTotal === 1) return normalizedPosition({ x: 50, y: 50 }, edgeInset, "fallback");

  // Arrange unknown gates clockwise on an ellipse. This remains legible on
  // wide map assets and is deterministic once callers sort stable gate keys.
  const angle = -Math.PI / 2 + (safeIndex / safeTotal) * Math.PI * 2;
  return normalizedPosition(
    { x: 50 + Math.cos(angle) * 38, y: 50 + Math.sin(angle) * 34 },
    edgeInset,
    "fallback",
  );
}

/** Resolve one gate position using tenant overrides before transport data. */
export function resolveGateMapPosition(
  gate,
  mapConfig = {},
  fallbackIndex = 0,
  fallbackCount = 1,
) {
  const edgeInset = mapConfig?.edgeInset ?? DEFAULT_EDGE_INSET;
  return (
    configuredPosition(gate, mapConfig, edgeInset) ??
    gatePosition(gate, edgeInset) ??
    projectCoordinates(gate?.latitude, gate?.longitude, mapConfig?.bounds, edgeInset) ??
    fallbackPosition(fallbackIndex, fallbackCount, edgeInset)
  );
}

/** Resolve selection safely while data changes between demo and live modes. */
export function resolveSelectedGate(gates, selectedGateId) {
  const safeGates = Array.isArray(gates) ? gates : [];
  return safeGates.find((gate) => gate?.id === selectedGateId) ?? safeGates[0] ?? null;
}

/** Resolve a short visual label while keeping the full code for accessible copy. */
export function resolveGateMapLabel(gate, mapConfig = {}) {
  const labels = mapConfig?.gateLabels;
  const configured =
    labels && typeof labels === "object" ? labels[gate?.id] ?? labels[gate?.code] : null;
  const rawLabel = String(configured ?? gate?.code ?? "").trim();
  if (rawLabel.length <= 4) return rawLabel;
  const segments = rawLabel.split(/[^a-z0-9]+/i).filter(Boolean);
  if (segments.length > 1) return segments.map((segment) => segment[0]).join("").slice(0, 4).toUpperCase();
  return rawLabel.slice(0, 3).toUpperCase();
}

function stableGateKey(gate, index) {
  return String(gate?.id ?? gate?.code ?? `gate-${index}`).toLocaleLowerCase("en");
}

function hasResolvedPosition(gate, mapConfig) {
  const edgeInset = mapConfig?.edgeInset ?? DEFAULT_EDGE_INSET;
  return Boolean(
    configuredPosition(gate, mapConfig, edgeInset) ??
      gatePosition(gate, edgeInset) ??
      projectCoordinates(gate?.latitude, gate?.longitude, mapConfig?.bounds, edgeInset),
  );
}

/**
 * Build a stable render model without mutating API or seeded gate records.
 * Unknown gates are sorted only for fallback-slot allocation, so response
 * ordering does not move their pins around the illustration.
 */
export function buildCampusMapModel(gates, mapConfig = {}, selectedGateId = null) {
  const safeGates = Array.isArray(gates) ? gates.filter(Boolean) : [];
  const selectedGate = resolveSelectedGate(safeGates, selectedGateId);
  const unresolved = safeGates
    .map((gate, index) => ({ gate, key: stableGateKey(gate, index) }))
    .filter(({ gate }) => !hasResolvedPosition(gate, mapConfig))
    .sort((left, right) => left.key.localeCompare(right.key, "en"));
  const fallbackSlots = new Map(unresolved.map(({ gate }, index) => [gate, index]));

  return {
    selectedGate,
    gates: safeGates.map((gate) => ({
      gate,
      position: resolveGateMapPosition(
        gate,
        mapConfig,
        fallbackSlots.get(gate) ?? 0,
        unresolved.length || 1,
      ),
      selected: gate === selectedGate,
    })),
  };
}
