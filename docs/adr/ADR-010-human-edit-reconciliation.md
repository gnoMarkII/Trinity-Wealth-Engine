# ADR-010: Human edits are first-class revisions

Status: accepted (R8)

Manual edits in canonical Markdown are not overwritten as stale noise. A
reconciler compares the current file with the last committed artifact head,
validates YAML and field ownership, and either imports a manual revision,
creates a recoverable conflict/quarantine event, or reports a delete grace
period. System-owned identity/revision fields are never silently accepted from
an editor. Malformed or partially synced files remain untouched.

Generated blocks use stable managed markers. Projection refreshes replace only
the managed block and preserve the human annotation region byte-for-byte.
