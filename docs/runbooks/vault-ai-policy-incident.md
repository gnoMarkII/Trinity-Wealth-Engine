# Vault AI policy incident (R8)

If restricted, generated, draft, disputed, or projection content appears in a
public retrieval namespace:

1. Stop publication of the affected catalog/vector generation.
2. Inspect registry and policy digests in the generation manifest.
3. Rebuild the eligible catalog/vector generation from the canonical snapshot.
4. Verify citation tuples contain `note_id`, `revision_id`, and content hash.
5. Record the incident and release the pointer only after namespace leakage is
   zero.

Useful checks:

```powershell
python scripts/run_vault_r8_preflight.py --vault memories
python scripts/run_vault_r8_acceptance.py --vault memories --skip-tests
```

The answer contract is fail-closed for namespace mismatch and incomplete
citations; do not manually edit a generated index to hide a policy violation.
