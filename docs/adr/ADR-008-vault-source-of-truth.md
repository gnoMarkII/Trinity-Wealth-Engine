# ADR-008: Vault source-of-truth boundaries

Status: accepted (R8)

`memories` is the canonical source for research knowledge, human narrative,
attachments, stable identities, and lifecycle tombstones. Obsidian is an
editor/adapter, not the database boundary. Catalogs, vector stores, caches,
provider cursors, queue state, and portfolio transactions live outside the
sync tree and are rebuildable or backed up by their owning runtime.

The machine-readable contract is `memories/.system/storage_contract.json` and
the profile/policy registry is `schemas/vault/registry.json`. A data class may
not have two physical canonical writers.

Consequences: Markdown remains portable and human-readable; derived systems
must carry relative paths and content hashes; recovery starts from a vault
snapshot plus external runtime backups rather than from an index.
