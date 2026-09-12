# Vault broker recovery (R8)

1. Stop producers and acquire the Vault maintenance lease if canonical files
   must be restored.
2. Snapshot the external broker database and the Vault before changing state.
3. Inspect health and recover expired leases:

   ```powershell
   python -m api.workers.vault_write_worker
   python -m tools.archivist.broker_cli --vault memories health
   ```

4. Reconcile committed receipts against `.system/artifacts/heads` and the
   immutable artifact store. Never restore a Vault snapshot without replaying
   commands committed after that snapshot.
5. Release the maintenance lease only after identities, tombstones, receipts,
   and active catalog/vector pointers agree.

The recovery invariant is: one idempotency key maps to one command fingerprint;
replay may reuse a receipt but may not create a second logical revision.
