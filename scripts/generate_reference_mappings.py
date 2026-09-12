"""Generate deterministic reference mapping for Vault V2 remediation (Task F06).

Reads scratch/vault-v2/remediation-r2/baseline/inventory.json and builds:
- scratch/vault-v2/remediation-r2/reference-candidates.json
- scratch/vault-v2/remediation-r2/unresolved-items.md
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from application.knowledge.identity import build_document_key
from tools.archivist.vault_paths import VaultPaths


def main() -> None:
    base_dir = Path("scratch/vault-v2/remediation-r2")
    inv_file = base_dir / "baseline" / "inventory.json"
    if not inv_file.exists():
        print(f"Error: Inventory file {inv_file} not found.")
        return

    print("Loading baseline inventory...")
    with inv_file.open("r", encoding="utf-8") as f:
        inv_data = json.load(f)

    files = inv_data.get("inventory", [])
    print(f"Loaded {len(files)} file audit records.")

    vp = VaultPaths()

    # Index all active files by lowercase title / stem / rel_path for link resolution
    stem_to_record: dict[str, list[dict[str, Any]]] = {}
    path_to_record: dict[str, dict[str, Any]] = {}

    for rec in files:
        if rec.get("is_excluded"):
            continue
        rel_path = rec.get("relative_path", "").replace("\\", "/")
        path_to_record[rel_path.lower()] = rec
        stem = Path(rel_path).stem.lower()
        stem_to_record.setdefault(stem, []).append(rec)

    mappings: list[dict[str, Any]] = []
    unresolved_items: list[dict[str, Any]] = []

    wikilink_re = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]")

    for rec in files:
        if rec.get("is_excluded"):
            continue

        rel_path = rec.get("relative_path", "").replace("\\", "/")
        sha256 = rec.get("sha256", "")
        props = rec.get("properties", {}) or {}
        stem = Path(rel_path).stem
        title = props.get("title") or stem
        entity_type = props.get("entity_type") or "note"
        ticker = props.get("ticker") or (props.get("tickers")[0] if props.get("tickers") else None)
        date_val = props.get("date") or props.get("published_at") or props.get("as_of")

        # Build canonical document_key and deterministic note_id
        doc_key = props.get("document_key")
        if not doc_key:
            doc_key = build_document_key(
                kind=entity_type,
                source_identity=ticker or stem,
                role="primary",
                as_of=str(date_val) if date_val else None,
            )

        doc_hash = hashlib.sha256(doc_key.encode("utf-8")).hexdigest()[:12]
        note_id = props.get("note_id") or f"note_{doc_hash}"
        revision_id = f"rev_{sha256[:10]}" if sha256 else "rev_initial"

        # Canonical target path
        meta_for_path = dict(props)
        meta_for_path["entity_type"] = entity_type
        if ticker:
            meta_for_path["ticker"] = ticker
        meta_for_path["title"] = title
        canonical_target = str(vp.note_path(meta_for_path, filename=Path(rel_path).name)).replace("\\", "/")

        # Inspect links
        wikilinks = rec.get("links", [])
        resolved_links: list[dict[str, str]] = []
        file_unresolved: list[str] = []

        for link in wikilinks:
            clean_link = link.strip().lower()
            if not clean_link:
                continue
            # Try exact path match
            match = path_to_record.get(clean_link) or path_to_record.get(f"{clean_link}.md")
            if not match:
                # Try stem match
                candidates = stem_to_record.get(clean_link, [])
                if candidates:
                    match = candidates[0]

            if match:
                m_props = match.get("properties", {}) or {}
                m_sha = match.get("sha256", "")
                m_id = m_props.get("note_id") or f"note_{m_sha[:12]}"
                resolved_links.append({
                    "link_text": link,
                    "target_rel_path": match.get("rel_path", "").replace("\\", "/"),
                    "target_note_id": m_id,
                })
            else:
                file_unresolved.append(link)

        mapping_entry = {
            "old_path": rel_path,
            "old_sha256": sha256,
            "title": title,
            "entity_type": entity_type,
            "note_id": note_id,
            "revision_id": revision_id,
            "canonical_target": canonical_target,
            "resolved_links_count": len(resolved_links),
            "unresolved_links_count": len(file_unresolved),
            "resolved_links": resolved_links,
        }
        mappings.append(mapping_entry)

        if file_unresolved:
            unresolved_items.append({
                "source_path": rel_path,
                "unresolved_links": file_unresolved,
            })

    # Save candidates JSON
    out_candidates = base_dir / "reference-candidates.json"
    with out_candidates.open("w", encoding="utf-8") as f:
        json.dump({
            "total_records": len(mappings),
            "unresolved_sources": len(unresolved_items),
            "mappings": mappings,
        }, f, indent=2)

    # Save unresolved markdown report
    out_unresolved = base_dir / "unresolved-items.md"
    with out_unresolved.open("w", encoding="utf-8") as f:
        f.write("# Obsidian Vault V2 — Unresolved Reference Items (Task F06)\n\n")
        f.write(f"- Total scanned active files: {len(mappings)}\n")
        f.write(f"- Files with unresolved links: {len(unresolved_items)}\n\n")
        f.write("## Sample Unresolved Links\n\n")
        for item in unresolved_items[:50]:
            f.write(f"### `{item['source_path']}`\n")
            for ul in item["unresolved_links"]:
                f.write(f"- `[[{ul}]]`\n")
            f.write("\n")

    print(f"Generated {out_candidates} ({len(mappings)} mappings)")
    print(f"Generated {out_unresolved} ({len(unresolved_items)} items with missing links)")


if __name__ == "__main__":
    main()
