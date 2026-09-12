# Multi-App Vault Storage Contract

Status: active, schema version 2

This contract defines the portable storage boundary for `memories`. Markdown,
YAML frontmatter, ordinary attachments, and small identity/control records are
the canonical source. Obsidian, indexers, catalogs, vector stores, and query
caches are adapters or derived read models; they must not become the only copy
of user knowledge.

All application mutations enter through
`application.knowledge.write_ports.KnowledgeWritePort`. The default local
implementation is `KnowledgeWriteBroker`, whose command queue and receipts
live under `data/vault_runtime/<vault-id>/broker/`, outside the synced vault.
Only its injected `ArtifactWriterKnowledgeAdapter` may translate a command into
the canonical Markdown/attachment writer. Repeating an idempotency key with a
different command fingerprint is a conflict; repeating the same command may
return the original committed receipt without creating another revision.

## Canonical and non-canonical scopes

| Scope | Role | Portability rule |
|---|---|---|
| `*.md` outside excluded/adapter paths | canonical content | UTF-8 Markdown + YAML frontmatter; readable without Obsidian |
| `90_Attachments/**` | canonical binary attachments | ordinary relative paths; no app URI required |
| `.system/identity_allocations.json` and `.system/retired_notes.jsonl` | canonical control metadata | versioned JSON/JSONL; preserve stable identity and tombstones |
| `.obsidian/**` | Obsidian adapter | optional; deleting it must not delete canonical content |
| `00_Index/App_Views/Obsidian/**` | Obsidian adapter view | may use app syntax; must point to canonical notes |
| `data/**` and `scratch/**` outside the vault | derived/evidence | databases, vectors, caches, snapshots, and reports live here |
| `.system/*_generation_active.json` | derived pointer | small relative-path pointer only; no absolute machine path |
| `.system/maintenance.json` | operational control | lease state only; never a knowledge source |
| `data/vault_runtime/**/broker/**` | write runtime | durable commands, leases, fencing tokens, receipts; never canonical knowledge |

`00_Inbox/**` is capture scope. A captured note must use
`capture_status: pending_normalization` and `search_scope: excluded` until a
normalizer validates its schema, allocates identity, chooses its canonical
path, and publishes it atomically.

## Text, metadata, links, and paths

- New text files use UTF-8 and LF. Readers accept CRLF so a Windows editor does
  not create a false content change. Derived content hashes use decoded UTF-8
  text with universal-newline normalization.
- Frontmatter is YAML safe-loader compatible. Use scalar, list, and mapping
  values that survive a generic YAML round trip. Dates and times use ISO-8601
  forms. Unknown metadata must be preserved by writers.
- Internal links are standard relative Markdown links such as
  `[label](../Folder/Note.md#heading)`. Spaces, Unicode, and parentheses are
  URI-encoded in destinations. External links use normal HTTPS/HTTP URLs.
- Canonical content must not require `[[wikilink]]`, `![[embed]]`,
  `obsidian://`, Dataview, Meta Bind, executable HTML, or JavaScript. Those
  forms are allowed only in explicitly declared adapter paths.
- Canonical paths are vault-relative, use `/` as the logical separator, stay
  at or below 180 characters for Markdown paths, avoid Windows reserved names,
  and use NFC-normalized filenames. A resolver must fail closed on an
  unresolved or ambiguous target.
- Stable `note_id` and `document_key` are independent of path. Rename and link
  migration must preserve both values. Templates must not contain fake or
  placeholder identities.

## Derived runtime and migration compatibility

Catalog/vector generations are immutable external read models. They are built
from one canonical snapshot, carry relative paths plus canonical content
hashes, and are activated by an atomic pointer switch. A rebuild may preserve
durable identity and tombstones but must not silently create a revision or
change canonical content.

Schema changes require a monotonic `schema_version`, an explicit reader
compatibility rule, and a reversible migration. A writer that cannot validate a
note must leave it in capture scope with a machine-readable reason; it must not
publish a partial canonical record.

The profile and policy registry is `schemas/vault/registry.json` with companion
policies in `schemas/vault/policies/`. Its field-ownership policy distinguishes
system, policy, producer, and human fields. Human edits to human-owned fields
are valid input and must be reconciled through an explicit revision; generated
navigation, portfolio projections, and derived artifacts must not overwrite
human-owned content silently. Vector indexing additionally checks lifecycle,
verification, sensitivity, and `search_scope` gates.

The machine-readable form is
`memories/.system/storage_contract.json`. The R6 read-only acceptance runner
is `scripts/run_vault_v2_acceptance_r6.py`.
