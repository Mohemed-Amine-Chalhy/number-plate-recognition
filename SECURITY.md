# Security policy

Security fixes are applied to the current default branch; no historical release line is maintained.

## Reporting a vulnerability

Use GitHub's private vulnerability reporting feature for this repository when available. Otherwise contact the repository owner through a private channel listed on their GitHub profile. Do not publish exploit details or credentials in a public issue.

Include the affected revision, impact, a minimal reproduction, relevant configuration with secrets removed, and a suggested mitigation when known.

Useful areas to test include malformed image handling, upload and cascade limits, model-file verification, dependency boundaries, session isolation, and error sanitization. Avoid destructive or denial-of-service testing against systems you do not own.
