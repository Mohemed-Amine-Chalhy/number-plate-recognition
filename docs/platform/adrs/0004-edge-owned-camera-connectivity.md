# ADR-0004: Outbound site edge agent owns camera connectivity

- Status: Proposed target
- Date: 2026-08-23
- Owners: Platform, site IT, and camera integration

← [ADR index](README.md) · [Camera onboarding](../camera-edge-onboarding.md)

## Context

Cameras normally sit on private LAN/VLANs; ONVIF discovery is local-network oriented and RTSP media
is persistent/high-bandwidth. Direct cloud reachability would require inbound routing or camera
exposure, distribute credentials centrally, and fail badly during WAN interruption.

## Decision

Deploy a site edge agent on each relevant camera network zone. It will:

- discover/probe cameras and keep credentials locally/encrypted;
- maintain RTSP decode sessions and select bounded frames/bursts;
- report device/stream/config health;
- upload evidence and control messages through outbound authenticated connections;
- keep last-known-good configuration and a bounded offline spool;
- reject stale/duplicate future commands.

The public API/browser never receives camera credentials or raw RTSP URLs. Browser preview uses a
separate WebRTC/HLS gateway/session.

## Consequences

### Positive

- Preserves private camera network boundary and reduces WAN video volume.
- Local reconnect/capture continues during central outages.
- Vendor/protocol complexity stays out of the control API.
- Creates a path for optional edge inference only when outage SLO requires it.

### Negative

- New device fleet requiring enrollment, upgrades, monitoring, and replacement.
- Per-site installation and disk/network capacity planning.
- Store-forward reconciliation and desired/applied config protocol are required.

## Alternatives considered

- Cloud directly polls RTSP/ONVIF: rejected for connectivity, secret, and outage reasons.
- Camera vendor cloud/P2P as primary integration: rejected as a default due to lock-in/control; may
  be an explicit adapter later.
- Forward continuous video centrally: rejected as default; bounded preview/incident streams remain
  optional.

## Validation

- Camera restart, address change, WAN outage, disk watermark, clock skew, certificate rotation, and
  duplicate message drills.
- Credential-redaction tests and network verification that cameras are not publicly reachable.
