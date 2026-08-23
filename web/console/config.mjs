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
    logoUrl: "./assets/tenant-logo.png?v=0.1.1",
    logoAlt: "UM6P",
    fallbackMark: "U6",
    accent: "#b51f37",
    accentStrong: "#8e1328",
    accentSoft: "#f9e8ec",
    supportLabel: "Campus operations",
  }),
  api: Object.freeze({
    baseUrl: "/api/v1",
    timeoutMs: 1800,
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
