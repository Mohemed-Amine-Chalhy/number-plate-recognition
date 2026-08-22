# Security policy

## Supported versions

Security fixes are applied to the current default branch. No historical release line is currently maintained. When versioned releases begin, this section will list their support windows.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability, privacy exposure, leaked credential, unsafe model artifact, or bypass of upload/session controls.

Use GitHub's private vulnerability reporting feature for this repository if it is enabled. Otherwise contact the repository owner through a private channel listed on their GitHub profile and include only the minimum information needed to establish a secure follow-up channel. If no private channel is available, open a public issue requesting private security contact **without** vulnerability details, secrets, images, or plate values.

Include, when safe:

- affected revision/version and deployment mode;
- impact and realistic attack scenario;
- minimal reproduction using synthetic/non-sensitive data;
- relevant configuration with secrets removed;
- suggested mitigation, if known.

Do not send production images, full plate values, access tokens, private weights, or personal information. Do not access data that is not yours, disrupt a service, perform denial-of-service testing, or retain/exfiltrate data while researching.

## Response expectations

This is currently a community project without a contractual response SLA. Maintainers should acknowledge a valid private report as soon as practical, coordinate triage and remediation privately, and agree on disclosure timing with the reporter. Public disclosure should wait until users have a reasonable opportunity to apply a fix.

## Deployment responsibility

The repository is not secure merely because its checks pass. Deployers must provide TLS, authentication, network isolation, rate/request limits, privacy-safe logging, artifact verification, vulnerability management, monitoring, incident response, and applicable legal/privacy controls. Review [docs/privacy-security.md](docs/privacy-security.md) and [docs/deployment.md](docs/deployment.md).

The current runtime forces Ultralytics offline/no-auto-install behavior; missing local artifacts fail instead of being fetched. No vehicle photographs ship in the current tree/container. Previously removed photographs may remain in Git history and remote caches, so the owner must assess a coordinated purge/cache cleanup before public release rather than assuming current-tree deletion erased them.

The three bundled manifest entries deliberately use null sources, unverified provenance/license status, and `production_approved: false`. Those external provenance/rights decisions—and representative quality acceptance—are unresolved production-release gates. `NPR_ENVIRONMENT=production` fails closed on the machine-readable fields. A checksum proves local integrity, not trust, ownership, safety, quality, or permission.
