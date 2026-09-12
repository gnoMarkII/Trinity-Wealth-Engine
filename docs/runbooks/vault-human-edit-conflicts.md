# Human-edit reconciliation (R8)

The reconciler is read-only by default:

```powershell
python scripts/reconcile_human_edits_r8.py --vault memories
```

Only safe body edits with unchanged system-owned frontmatter may be imported:

```powershell
python scripts/reconcile_human_edits_r8.py --vault memories --write-enabled
```

Malformed YAML, identity/frontmatter edits, missing identities, and concurrent
stale updates remain conflicts. Resolve them by preserving the original file,
reviewing the external reconciliation event, and submitting a new command with
the expected revision. Never apply last-writer-wins to a stale human edit.
