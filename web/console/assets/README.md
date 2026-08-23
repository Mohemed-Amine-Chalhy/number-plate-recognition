# Tenant assets

`tenant-logo.png` is the demo tenant logo downloaded from the official UM6P communication
package at <https://com.um6p.ma/images/logo_um6p.png> on 2026-08-23. Its use in this repository
is based on the project owner's stated authorization.

The console does not depend on this artwork. Replace the file and update `branding.logoUrl`,
`logoAlt`, colors, and names in `../config.mjs` to deploy the same product for another
organization. If an image cannot load, the accessible text mark in `branding.fallbackMark` is
shown instead.
