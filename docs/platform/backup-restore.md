# Backup and restore

← [Platform documentation index](README.md)

## Scope

This runbook covers the prototype SQLite database and the target production data categories. A
backup is not complete until a restore is verified. Demo data is regenerable; real configuration,
requests, grants, passages, decisions, incidents, and events are not.

## Recovery inventory

| Data | Prototype location | Target location | Recovery concern |
| --- | --- | --- | --- |
| Control-plane state | SQLite file from `CONTROL_API_DB_PATH` | PostgreSQL | Transactionally consistent restore |
| SQLite WAL/SHM | Beside live DB when WAL active | N/A | Do not copy DB file alone while live |
| Evidence media | Not required for deterministic demo | Object storage | Object/version/retention consistency |
| Edge buffered metadata/media | Target only | Edge SQLite + spool | Store-forward and disk replacement |
| Config/secrets | Environment/local configuration | Secret/config manager | Restore references, rotate exposed secrets |
| Model artifacts | Manifest-pinned repository files | Immutable artifact registry | Verify hash and semantic contract |
| Console demo fixtures | Version-controlled source | Build artifact | Regenerate, never merge with live data |

## Define RPO and RTO

Before a pilot, the product/operations owner must set:

- **RPO**: maximum acceptable committed-data loss;
- **RTO**: maximum acceptable restoration time;
- evidence/media retention and whether metadata-only recovery is acceptable;
- who may initiate/approve a restore;
- how gate operations continue manually during recovery.

Do not copy generic numbers into the runbook. Use measured backup size, event rate, network, and
restore rehearsal results. The [pilot plan](pilot-rollout.md#pilot-success-measures) records targets.

## SQLite online backup

The default prototype database is `.runtime/campus-control.sqlite3`. Confirm the effective
`CONTROL_API_DB_PATH`; do not assume it when an environment variable is set.

Use SQLite's online backup API instead of copying a live `.sqlite3` file while WAL writes may be
active. The following creates a consistent destination without modifying the source:

```powershell
$SourceDb = (Resolve-Path -LiteralPath ".runtime/campus-control.sqlite3").Path
$BackupDir = Join-Path (Resolve-Path -LiteralPath ".runtime").Path "backups"
New-Item -ItemType Directory -Force -Path $BackupDir | Out-Null
$BackupDb = Join-Path $BackupDir "campus-control-backup.sqlite3"

uv run --project services/control_api --frozen python -c `
  "import sqlite3,sys; src=sqlite3.connect(sys.argv[1]); dst=sqlite3.connect(sys.argv[2]); src.backup(dst); dst.close(); src.close()" `
  $SourceDb $BackupDb
```

Use timestamped/immutable names in an actual schedule rather than overwriting the previous backup.
The fixed name above is only a readable manual example.

## Verify a SQLite backup

Check integrity and schema metadata against the **backup**, not the live path:

```powershell
$BackupDb = (Resolve-Path -LiteralPath ".runtime/backups/campus-control-backup.sqlite3").Path
uv run --project services/control_api --frozen python -c `
  "import sqlite3,sys; db=sqlite3.connect(sys.argv[1]); print(db.execute('PRAGMA integrity_check').fetchone()[0]); print(db.execute('SELECT version, applied_at FROM schema_metadata ORDER BY version').fetchall()); db.close()" `
  $BackupDb
Get-FileHash -Algorithm SHA256 -LiteralPath $BackupDb
```

Record:

- source environment and database identity;
- backup start/end and application version;
- file size and SHA-256;
- `integrity_check` result (`ok` expected);
- schema versions and selected row counts;
- encryption/access location and expiry;
- verifier.

A hash detects later change; it does not prove the backup contains the intended transaction or can
serve the application.

## SQLite restore rehearsal

Restore to a **new path** first. Do not overwrite the only live file during validation.

1. Stop or isolate the test API that will use the restored path.
2. Copy the verified backup into a new empty restore directory.
3. Set `CONTROL_API_DB_PATH` to that restored copy.
4. Set `CONTROL_API_SEED_DEMO=false` so validation does not add fixtures.
5. Start the same application version that produced/supports the schema.
6. Check `/health/ready`, organization-scoped lists, event sequence, request/grant links, passage
   detail, and a read-only dashboard.
7. Run a bounded write/read transaction only in an isolated rehearsal environment.
8. Record restore duration and any missing external media/config.
9. Shut down the rehearsal and retain/delete it according to the backup handling policy.

For an incident restore, switch traffic to the verified restored database through deployment
configuration. Preserve the failed database separately for diagnosis; never merge two divergent
SQLite writers manually.

## Production PostgreSQL target

The production strategy should include:

- managed snapshots or physical backups plus point-in-time recovery/WAL archive;
- logical export only as a secondary portability/debug path;
- encryption and separate backup access role;
- automated retention and failed-backup alerts;
- restore into an isolated database, migration/application compatibility test, then controlled
  cutover;
- organization-level export is not equivalent to full disaster recovery;
- periodic recovery drills with measured RPO/RTO.

PostgreSQL backups must be coordinated with object storage and the event broker. A database row that
references an expired/missing object should be detected in restore verification.

## Object storage recovery

- Enable versioning/retention only to the degree required by the product policy.
- Inventory objects by organization, capture/media identity, checksum, and expiry.
- Back up configuration needed to map database media references to buckets/keys.
- Test signed URL/media gateway behavior after restore.
- Do not preserve deleted high-sensitivity evidence indefinitely through an undocumented backup
  retention exception.

## Edge recovery

An edge agent should be replaceable from central desired configuration and a new workload identity.
Before replacing a failed disk/device:

1. revoke the old edge identity if compromise/loss is possible;
2. preserve the spool only when its chain of custody and integrity are adequate;
3. enroll the replacement and fetch last-known desired configuration;
4. reconcile buffered events by agent/boot/sequence idempotently;
5. never replay expired physical commands;
6. repeat camera connection/calibration acceptance where hardware/network changed.

## Recovery verification checklist

- [ ] Integrity/schema check passes.
- [ ] Application version is compatible.
- [ ] Organization isolation remains enforced.
- [ ] Highest event sequence and key counts match the recovery point.
- [ ] Request → grant and passage → observation → decision links resolve.
- [ ] Referenced evidence objects exist or are explicitly marked unavailable.
- [ ] Demo fixtures were not inserted into a live restore.
- [ ] Credentials were restored through secret references, not a copied `.env` leak.
- [ ] Health and observability point to the restored environment.
- [ ] RPO/RTO and any loss/gap are recorded and communicated.

## Related documents

- [Deployment runbook](deployment-runbook.md)
- [Troubleshooting](troubleshooting.md)
- [Security and privacy](security-and-privacy.md#backup-security)
- [Data model](data-and-workflows.md)
