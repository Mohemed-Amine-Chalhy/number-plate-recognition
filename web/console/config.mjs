/**
 * White-label tenant configuration.
 *
 * Deployments can replace this object at build time or serve an equivalent
 * JSON document from their own configuration service. The UI deliberately
 * does not infer logo dimensions or tenant colors from the demo brand.
 */
export const TENANT_CONFIG = Object.freeze({
  tenantId: "um6p-demo",
  campusId: "ben-guerir",
  branding: Object.freeze({
    name: "UM6P Campus Access",
    shortName: "UM6P",
    productName: "Campus Access",
    logoUrl: "./assets/tenant-logo.png?v=0.3.0",
    logoAlt: "UM6P",
    fallbackMark: "U6",
    accent: "#b51f37",
    accentStrong: "#8e1328",
    accentSoft: "#f9e8ec",
    supportLabel: "Campus operations",
  }),
  map: Object.freeze({
    /** Replace the asset and pin table together for another site. */
    assetUrl: "./assets/campus-map-illustrated-v2.webp",
    aspectRatio: "3 / 2",
    edgeInset: 6,
    /** Set north/east/south/west when the illustration is geographically aligned. */
    bounds: null,
    gatePositions: Object.freeze({
      "gate-main": Object.freeze({ x: 56, y: 6 }),
      "gate-innovation": Object.freeze({ x: 68, y: 18 }),
      "gate-logistics": Object.freeze({ x: 85, y: 42 }),
      "gate-residential": Object.freeze({ x: 55, y: 83 }),
      "gate-south": Object.freeze({ x: 39, y: 94 }),
      "gate-sports": Object.freeze({ x: 21, y: 65 }),
      "gate-atlas-north": Object.freeze({ x: 56, y: 6 }),
      "gate-atlas-research": Object.freeze({ x: 68, y: 18 }),
      "gate-atlas-service": Object.freeze({ x: 85, y: 42 }),
      "gate-atlas-residence": Object.freeze({ x: 55, y: 83 }),
      "gate-atlas-south": Object.freeze({ x: 39, y: 94 }),
      "gate-atlas-sports": Object.freeze({ x: 21, y: 65 }),
    }),
    gateLabels: Object.freeze({
      "gate-main": "G01",
      "gate-innovation": "G02",
      "gate-logistics": "G04",
      "gate-residential": "G03",
      "gate-south": "G05",
      "gate-sports": "G06",
      "gate-atlas-north": "G01",
      "gate-atlas-research": "G02",
      "gate-atlas-service": "G04",
      "gate-atlas-residence": "G03",
      "gate-atlas-south": "G05",
      "gate-atlas-sports": "G06",
    }),
    landmarks: Object.freeze([
      Object.freeze({ id: "residences", labelKey: "map.landmark.residences", x: 48, y: 24 }),
      Object.freeze({ id: "innovation", labelKey: "map.landmark.innovation", x: 72, y: 35 }),
      Object.freeze({ id: "academic", labelKey: "map.landmark.academic", x: 52, y: 52 }),
      Object.freeze({ id: "sports", labelKey: "map.landmark.sports", x: 34, y: 66 }),
    ]),
  }),
  api: Object.freeze({
    baseUrl: "/api/v1",
    timeoutMs: 1800,
    agentTimeoutMs: 8000,
    refreshMs: 30_000,
    organizationId: "org-atlas",
    /** Replace these demo-only credentials when using a real identity provider. */
    demoRoleTokens: Object.freeze({
      operator: "demo-operator",
      attendant: "demo-operator",
      admin: "demo-admin",
      viewer: "demo-viewer",
    }),
    /** Optional future endpoint, for example: "/gates/{gateId}/commands". */
    gateCommandPath: null,
  }),
  defaults: Object.freeze({
    locale: "en",
    theme: "light",
    role: "operator",
    timeZone: "Africa/Casablanca",
  }),
});

export const SUPPORTED_LOCALES = Object.freeze(["en", "fr", "ar"]);

export const ROLE_OPTIONS = Object.freeze([
  { id: "operator", labelKey: "role.operator" },
  { id: "attendant", labelKey: "role.attendant" },
  { id: "admin", labelKey: "role.admin" },
  { id: "viewer", labelKey: "role.viewer" },
]);
