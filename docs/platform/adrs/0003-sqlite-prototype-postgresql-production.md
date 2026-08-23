# ADR-0003: SQLite for prototype, PostgreSQL for replicated production

- Status: Accepted
- Date: 2026-08-23
- Owners: Control-plane engineering

← [ADR index](README.md) · [Backup and restore](../backup-restore.md)

## Context

The case study needs a self-contained, inspectable API without adding a database service. SQLite
provides transactions, foreign keys, indexes, and deterministic local persistence. A production
multi-site control plane needs concurrent writers, replicas, robust migrations/backups, row-level
tenant controls, and managed availability.

## Decision

Use Python's standard `sqlite3` behind a repository boundary for the local prototype. Use short-lived
connections, foreign keys, busy timeout, WAL, explicit transactions, and a schema version. Run one
writer service instance.

Before a replicated/live deployment, implement and validate a PostgreSQL repository/migration path,
row-level security or equivalent defense in depth, managed backup/PITR, and load/isolation tests.

## Consequences

### Positive

- Zero external service for reviewers and deterministic video.
- SQL/schema remains visible rather than hidden behind an in-memory fake.
- Repository boundary gives a concrete migration seam.

### Negative

- Single-writer/host boundary; no horizontal API replicas against the same file.
- Prototype schema initialization is not a substitute for a mature migration toolchain.
- Some PostgreSQL constraints/types/policies cannot be proven in SQLite.

## Alternatives considered

- PostgreSQL required for every local run: rejected for demo friction.
- In-memory dictionaries: rejected because persistence, constraints, backup, and transactions matter.
- SQLite on network storage/multiple replicas: rejected because it misuses the prototype boundary.

## Migration trigger

Any of these requires PostgreSQL work before promotion:

- multiple API writer replicas;
- live multi-site tenant data;
- sustained lock contention or event volume outside measured SQLite capacity;
- row-level policy requirement;
- production RPO/RTO/PITR requirement.
