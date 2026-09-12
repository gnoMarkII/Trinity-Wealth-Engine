# Obsidian + AI Readiness R5 Operations

This runbook is for the `memories` vault after R5 hardening. Markdown and
attachments remain the source of truth. Catalog and vector data are immutable,
rebuildable read models outside the vault.

## Runtime locations

- Vault: `memories/`
- Catalog pointer: `memories/.system/catalog_generation_active.json`
- Vector pointer: `memories/.system/vector_generation_active.json`
- Catalog runtime: `data/vault_runtime/memories/`
- Vector runtime: `data/vector_runtime/vault_v2/`
- Evidence: `scratch/vault-v2/remediation-r5/<run-id>/`

Do not place SQLite files, WAL/SHM files, Chroma data, model caches, locks, or
benchmark output under `memories/`.

## Rebuild and verify

Use the maintenance lease owner for every live rebuild:

```powershell
$env:VAULT_MAINTENANCE_OWNER = "codex-vault-r5"
python scripts/rebuild_catalog_generation_r5.py --vault memories --run-dir scratch/vault-v2/remediation-r5/<run-id>
python scripts/rebuild_vector_generation_r5.py --vault memories --run-dir scratch/vault-v2/remediation-r5/<run-id>
python scripts/validate_ai_answer_contract_r5.py --vault memories --output scratch/vault-v2/remediation-r5/<run-id>/answer-contract-r5.json
```

The workers build a new generation, validate it, and publish only a small
pointer. Queries use read-only immutable SQLite access and the active vector
manifest. Chroma's query bookkeeping is isolated in
`data/vector_runtime/vault_v2/query_cache/<generation-id>/`; it must never
write the published generation itself.

## Outbox and trust policy

Writer projection failures go to the external catalog outbox at
`data/vault_runtime/memories/outbox/catalog_outbox.jsonl`. A catalog rebuild
reconciles events as `pending -> processing -> consumed` or `failed`; consumed
records remain as an audit ledger. Never delete an unresolved event manually.

All existing notes are exploration-safe, but not production-eligible until a
human or an evidence-bearing verifier reviews their source and content. R5
therefore permits research-mode answers with path/date/trust labels and blocks
production-mode answers for `not_reviewed`, unavailable, stale, or excluded
evidence.

## Rollback

List generation directories, then switch only to a validated generation:

```powershell
Get-ChildItem data/vault_runtime/memories/catalog -Directory
Get-ChildItem data/vector_runtime/vault_v2/generations -Filter *.json

python scripts/rollback_catalog_generation_r5.py --vault memories --generation-id <catalog-generation-id> --output scratch/vault-v2/remediation-r5/<run-id>/catalog-rollback.json
python scripts/rollback_vector_generation_r5.py --vault memories --generation-id <vector-generation-id> --output scratch/vault-v2/remediation-r5/<run-id>/vector-rollback.json
```

After rollback, run the answer-contract smoke test and final acceptance. Do
not restore a catalog or vector database by copying files into the vault.

## Final acceptance

```powershell
python scripts/run_vault_v2_acceptance_r4.py --vault memories --run-dir scratch/vault-v2/remediation-r5/<run-id>/acceptance-r4-final --observation scratch/vault-v2/remediation-r5/<run-id>/observation-60m.json
python scripts/run_vault_v2_acceptance_r5.py --vault memories --run-dir scratch/vault-v2/remediation-r5/<run-id> --r4-report scratch/vault-v2/remediation-r5/<run-id>/acceptance-r4-final/acceptance/acceptance-report.json --rehearsal-report <rehearsal-acceptance-report.json>
```

The hand-off is complete only when A01–A30 are evidence-derived `PASS`, the
60-minute observation has zero unexplained mutations, and
`remaining-items.json` is empty. Autonomous trade execution remains outside
the R5 scope.
