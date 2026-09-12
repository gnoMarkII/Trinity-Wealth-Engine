# Portfolio projection rebuild (R8)

Portfolio transactions are the source; Markdown is a deterministic projection.
Rebuild from the external event store and checkpoint:

```powershell
python scripts/rebuild_portfolio_projections_r8.py \
  --vault memories --portfolio-id <portfolio-id> \
  --projections-json <projection-input.json> --write-enabled
```

Without `--write-enabled`, the command is a read-only checkpoint/replay
preflight. Generated content must stay inside the managed block; human notes
outside that block are preserved and their annotation hash is evidence.
