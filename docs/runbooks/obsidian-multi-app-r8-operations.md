# Obsidian multi-app storage R8 operations

## Boundaries

- Application callers submit `KnowledgeWriteCommand` through
  `KnowledgeWritePort`; they do not pass a Vault path.
- Broker SQLite, portfolio transactions, catalog generations, vectors,
  reconciliation events, and receipts live under external `data/**` runtime.
- Research Markdown/frontmatter and human narrative remain canonical.
- Portfolio numeric state is replayed from
  `tools.portfolio.transaction_store.PortfolioTransactionStore`; Markdown is a
  projection with `search_scope: excluded`.

## Local broker operations

```text
python -m tools.archivist.broker_cli --vault memories health
python -m tools.archivist.broker_cli --vault memories drain --limit 100
python -m tools.archivist.broker_cli --vault memories retry <command-id>
```

Keep a broker database backup with the external runtime backup. Do not delete
the queue or receipts to clear a stuck command; inspect the event trail first.
Commands in `committing` or `committed` are not cancellable. A dead letter is
retried explicitly and records an `operator_retry` event.

## Human-edit reconciliation

`HumanEditReconciler.reconcile_once()` is shadow/read-only by default. It
records malformed YAML, identity drift, system-field edits, rename findings,
and delete grace-period evidence without touching canonical files. Enable
`write_enabled=True` only after the producer migration gate is green. Body
edits import through the broker with both the current body hash and the
baseline revision ID; concurrent application edits therefore fail closed.

## AI/index policy

Every new vector generation is manifest version 2 and records the registry
digest, policy digest, and eligible-set fingerprint. The primary answer
namespace accepts only public/internal reviewed or published notes. Capture,
portfolio projections, navigation, generated/draft/superseded/retired,
disputed, confidential, and restricted content is excluded from that path.
Every citation carries `note_id`, `revision_id`, and the current content hash.

## Rollback

1. Stop broker workers and keep the external SQLite database as evidence.
2. Restore the Vault snapshot and verify the tree fingerprint.
3. Restore the prior catalog/vector immutable generation pointers.
4. Run registry validation and writer inventory in read-only mode.
5. Reconcile queued commands using the same idempotency keys; never recreate
   a command with a new key merely to bypass a conflict.

The canonical recovery source is the Vault snapshot plus external runtime
backup; catalog/vector pointers are rebuildable and never the only copy.
