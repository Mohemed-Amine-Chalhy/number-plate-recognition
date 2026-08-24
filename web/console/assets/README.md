# Console assets

## Reference tenant mark

`tenant-logo.png` is the mark used by the included UM6P-themed reference configuration. The source
asset was retrieved from the official UM6P communications site at
<https://com.um6p.ma/images/logo_um6p.png> on 2026-08-23.

The console does not depend on this artwork. Replace the file and update `branding.logoUrl`,
`logoAlt`, `fallbackMark`, colors, and names in `../config.mjs` to present the platform for another
organization. If the image cannot load, the accessible text mark is shown automatically.

## Illustrated campus map

`campus-map-illustrated-v2.webp` is the command-center map artwork. It was rendered as a local
illustration from the project author's annotated campus boundary and six-gate reference. The image
provides the campus footprint and visual context, while gate markers, selection, status, queue, wait,
and throughput remain data-driven UI elements.

Map artwork and gate data are separate assets: another organization can replace the illustration and
configure its own gate coordinates without changing the command-center workflow.
