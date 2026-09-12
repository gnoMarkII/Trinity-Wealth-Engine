# ADR-011: Portfolio operational state and Markdown projections

Status: accepted (R8)

Transactions, provider cursors, units, cost basis, calculated balances, and
numeric progress are owned by the transactional runtime. Holdings, watchlists,
performance, and other Obsidian-facing files are rebuildable projections with
`projection_of`, `projection_version`, `generated_at`, and `source_checkpoint`.

Trading journal prose, investment thesis, goal narrative, and human
annotations remain canonical knowledge. Projection fields are not an implicit
transaction API; edits to generated fields are conflicts or annotations only.
Projection rebuilds are checkpointed, deterministic, and rollback-safe.
