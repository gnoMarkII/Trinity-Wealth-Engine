# ADR-009: One application write broker

Status: accepted (R8)

All application write intent uses `KnowledgeWritePort`. The local
`KnowledgeWriteBroker` persists commands and receipts in external runtime
SQLite, uses idempotency fingerprints, leases, and monotonic fencing tokens,
then delegates the canonical commit to `ArtifactWriterKnowledgeAdapter`.

Callers submit logical intent and metadata, never an absolute path or a second
physical writer. A same-payload retry reuses the committed receipt; a reused
idempotency key with a different fingerprint is a conflict. The broker is
exactly-once-equivalent: it does not claim distributed exactly-once execution,
but the content-addressed artifact engine makes replay safe.
