# Vault Platform Production Runbook (R9)

This runbook is the operator entry point for the `memories` Obsidian Vault.
The Vault contains canonical Markdown and control metadata. Durable queues,
portfolio events, catalog generations, vectors, checkpoints, and logs live in
the single external runtime root resolved by `tools/archivist/runtime_layout.py`.

## Ownership

| Area | Owner | First response |
|---|---|---|
| runtime layout, broker, registry, backup | platform owner | stop writers, inspect health and lease |
| invalid producer commands | producer owner | inspect receipt error and correct payload |
| human-edit conflicts and retention | data owner | resolve/quarantine; never overwrite silently |
| incident coordination | incident owner | record owner and incident ID in evidence |

Production defaults:

```powershell
Set-Location C:\ChinoDoc\Projects\Claude\invest-agents
$vault = "memories"
$runtime = "data\vault_runtime"
```

The canonical runtime for this Vault is `data/vault_runtime/memories`.
Do not set both `INVEST_VAULT_RUNTIME_BASE` and `OBSIDIAN_CATALOG_RUNTIME_PATH`
to different values. Do not put runtime state under the Vault or in the Sync
tree.

## Schedule and process topology

| Job | Frequency | Command | Mutation |
|---|---:|---|---|
| broker drain/recovery | continuous or 1 minute | `python scripts/run_vault_r9_cycle.py --vault $vault --runtime-base $runtime --write-enabled --output scratch/vault-r9/operations/cycle-<UTC>.json` | yes, lease guarded |
| human-edit reconciliation | 5 minutes | same cycle command | yes only for safe body edits; conflicts are retained |
| catalog/index health | 5 minutes or post-commit | same cycle command | health is read-only; rebuild is separate |
| integrity audit | daily | `python scripts/run_vault_r9_preflight.py --vault $vault --runtime-base $runtime --output scratch/vault-r9/preflight/daily-<UTC>.json` | no |
| full recovery bundle | daily | `python scripts/backup_vault_platform_r9.py --vault $vault --runtime-base $runtime --output-dir scratch/vault-r9/recovery --run-id <UTC>` | lease + backup only |
| restore rehearsal | monthly and before migration | restore command below | staging only |

On Windows Task Scheduler, each task must run from the repository directory
with the repository virtual-environment Python and must write its evidence to
`scratch/`, never to `memories/`. A supervisor must treat non-zero exit status
as a failed job; skipped/cancelled checks are not successful checks.

## Health and SLOs

Run a read-only cycle first:

```powershell
python scripts/run_vault_r9_cycle.py --vault $vault --runtime-base $runtime --output scratch/vault-r9/operations/health-<UTC>.json
```

Required normal-state SLOs:

- oldest pending queue age `< 60` seconds;
- dead letters `0`, or an incident owner and incident reference exists;
- unreviewed reconciliation conflicts `< 24` hours;
- catalog/vector eligible-set and policy digests are present and consistent;
- runtime ambiguity, stale lease after process exit, and registry/policy mismatch are `0`;
- last verified backup age `< 24` hours.

If any SLO fails, stop release/cutover. Inspect the JSON evidence, then use
the broker and recovery runbooks before retrying:

- [broker operations](vault-broker-operations.md)
- [broker recovery](vault-broker-recovery.md)
- [human-edit conflicts](vault-human-edit-conflicts.md)
- [portfolio projection rebuild](portfolio-projection-rebuild.md)
- [AI policy incident](vault-ai-policy-incident.md)

## Broker drain, retry, and reconciliation

The write port rejects client-supplied canonical paths. Producers submit an
idempotent, path-independent command; the broker and ArtifactWriter choose the
canonical route. Use the scheduled cycle for normal recovery:

```powershell
python scripts/run_vault_r9_cycle.py --vault $vault --runtime-base $runtime --write-enabled --output scratch/vault-r9/operations/manual-cycle-<UTC>.json
```

For a specific receipt, use the existing broker CLI only after reviewing the
receipt and owning incident:

```powershell
python -m tools.archivist.broker_cli --vault $vault --runtime-base $runtime health
python -m tools.archivist.broker_cli --vault $vault --runtime-base $runtime retry <command-id>
```

Never delete a command, receipt, revision, or conflict to make health green.

## Backup

The backup command acquires the maintenance lease, drains/reclaims the
broker, performs online SQLite backups, snapshots the Vault, records runtime
high-water marks/generation IDs and registry/policy digests, hashes every
artifact, and releases the lease in `finally`:

```powershell
python scripts/backup_vault_platform_r9.py `
  --vault $vault `
  --runtime-base $runtime `
  --output-dir scratch/vault-r9/recovery `
  --run-id backup-<UTC>
```

Confirm `status=PASS`, `broker_quiescence.after.queue_depth=0`, and keep the
manifest SHA-256. Retention is at least 7 daily, 4 weekly, and 90 days for a
pre-migration/release bundle. Backups must remain outside the Vault.

## Staging restore and rebuild

Run the non-destructive operator rehearsal before a migration or release:

```powershell
python scripts/rehearse_vault_r9_runbook.py `
  --vault $vault `
  --runtime-base $runtime `
  --operator vault-platform-owner `
  --output scratch/vault-r9/operations/runbook-rehearsal.json
```

Restore is staging-only and refuses an existing destination:

```powershell
python scripts/restore_vault_platform_r9.py `
  --bundle scratch/vault-r9/recovery/<run-id> `
  --restore-root scratch/vault-r9/recovery/restore-<run-id>
```

Verify `restore-report.json`, the bundle manifest, Vault file count/hash,
SQLite integrity, broker/portfolio high-water marks, and registry/policy
digests. A restored staging tree must not be activated by this command. Any
live activation is a separately reviewed change with an explicit target and
rollback hold.

To rebuild derived stores after a validated restore or policy change:

```powershell
python scripts/rebuild_vault_derived_r9.py `
  --vault $vault `
  --runtime-base $runtime `
  --run-dir scratch/vault-r9/operations/derived-rebuild-<UTC>
```

This command leases the Vault, rebuilds an immutable catalog generation and a
validated vector generation, publishes pointers atomically, and releases the
lease. Never edit the active catalog/vector database in place.

## Rollback

Rollback triggers include runtime ambiguity, command/receipt mismatch,
portfolio stream hash mismatch, duplicate revision, unresolved writer finding,
unrecoverable lease, failed restore proof, or AI policy leakage.

1. Stop producer/scheduled writers.
2. Acquire and retain the maintenance lease; preserve incident evidence.
3. Keep the old runtime and pre-change bundle; do not delete either during the
   rollback window.
4. Restore the broker/portfolio databases and Vault only into a reviewed
   staging target, then verify hashes and replay idempotently.
5. Point configuration back to the reviewed pre-cutover runtime, run the R8
   regression/acceptance checks, and reopen producers only after health is
   green.
6. Record the owner, incident ID, manifest SHA, rollback start/end, and
   remaining risk.

## Release and freeze gate

Before release, run R9 preflight, two operations cycles, a full backup, a clean
staging restore, relevant tests, and the 60-minute post-cutover observation.
The architecture is not frozen while any A123-A144 gate is `FAIL`, `NOT_RUN`,
or `REVIEW`; external branch protection must be evidenced by the repository
host, not inferred from workflow YAML.

Capture the repository-host setting with the read-only audit command:

```powershell
python scripts/audit_github_branch_protection_r9.py `
  --repository gnoMarkII/Trinity-Wealth-Engine `
  --branch main `
  --output scratch/vault-r9/acceptance/branch-protection-evidence.json
```

The required host configuration is a protected `main` branch with the exact
required check `Vault architecture and write-boundary contracts`, and merges
blocked when that check is skipped, cancelled, or failed. The audit command
does not change repository settings; an owner with repository administration
access must apply the setting and rerun the audit if it reports `FAIL`.
