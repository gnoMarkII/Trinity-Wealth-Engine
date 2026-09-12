# Multi-App Vault R6 Operations

The canonical source is `memories/**/*.md` plus ordinary attachments. The
`.obsidian` directory is an optional Obsidian adapter. Catalog/vector data is
external under `data/**`; do not copy a database into the Vault to repair a
query problem.

## Capture and publish

1. New notes from any app go to `00_Inbox/` with
   `capture_status: pending_normalization` and `search_scope: excluded`.
2. A normalizer validates UTF-8/YAML/path/source fields and allocates
   `note_id` and `document_key` centrally.
3. Publish with an atomic move into the canonical folder. A failed note stays
   in Inbox with a machine-readable reason; do not invent an identity in a
   template.

## Validate and rebuild

```powershell
python scripts/run_vault_v2_acceptance_r6.py `
  --vault memories `
  --output scratch/r6-acceptance.json

python scripts/validate_multi_app_r6.py `
  --vault memories `
  --run-dir scratch/vault-v2/remediation-r6/<run-id> `
  --owner <maintenance-owner> `
  --require-derived
```

Acquire a maintenance lease before any writer, navigation generator, or
derived-generation rebuild. Rebuild catalog first, then vector from that
catalog, and activate each generation through its pointer. Keep the prior
generation until query/citation smoke passes.

## Rollback

Stop writers, restore the approved snapshot to staging, verify the archive
checksum and exact file set, then perform the approved atomic swap. Roll back
catalog/vector pointers independently; pointer rollback must not edit Markdown.
Run the R5 A01–A30 acceptance plus the R6 portability gates after rollback and
again after reapply.

## Portability rules

Use relative Markdown links for internal notes. Do not add `[[wikilink]]`,
`![[embed]]`, absolute machine links, `obsidian://`, Dataview, Meta Bind, or
executable HTML to canonical notes. Adapter views must have a visible link or
source URL fallback to canonical content.
