# Vault broker operations (R8)

The knowledge-write broker is the only application write boundary. Its SQLite
queue and receipts live outside `memories/`; canonical Markdown remains in the
Vault.

## Inspect and drain

```powershell
python -m tools.archivist.broker_cli --vault memories health
python -m tools.archivist.broker_cli --vault memories drain --limit 100
```

Review `queue_depth`, `oldest_pending_age_seconds`, conflicts, and dead letters
before restarting producers.

## Retry

Retry only after inspecting the error code and provider state:

```powershell
python -m tools.archivist.broker_cli --vault memories retry <command-id>
```

Every retry is recorded in `broker_events` with `operator_retry=true`.

## Guardrails

Do not pass a filesystem path from an app command. Do not edit broker rows by
hand. Use the recovery runbook for expired leases and the human-edit runbook
for Markdown conflicts.
