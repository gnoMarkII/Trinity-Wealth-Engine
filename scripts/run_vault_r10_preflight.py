"""Read-only R10 preflight for portable links, retrieval, and Concepts cleanup."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_vault_r9_preflight import run as run_r9_preflight  # noqa: E402
from tools.archivist.catalog_runtime import resolve_catalog_path  # noqa: E402
from tools.archivist.concepts_cleanup import build_cleanup_plan, scan_concepts  # noqa: E402
from tools.archivist.recovery_bundle import capture_runtime_state  # noqa: E402
from tools.archivist.vault_audit import scan_vault  # noqa: E402


_REVIEW_FIELDS = (
    "search_scope",
    "lifecycle_status",
    "content_status",
    "retention_class",
    "trust_tier",
    "review_state",
    "review_owner",
    "review_reason",
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _catalog_manifest(vault: Path) -> dict[str, Any]:
    try:
        database = resolve_catalog_path(vault, require_exists=True)
    except Exception as exc:  # noqa: BLE001 - preflight evidence boundary
        return {"error": str(exc)}
    manifest = database.parent / "manifest.json"
    payload = _read_json(manifest)
    payload["database"] = str(database)
    payload["manifest"] = str(manifest)
    return payload


def _active_wikilink_counts(vault: Path) -> dict[str, int]:
    wikilinks = 0
    embeds = 0
    for path in sorted(vault.rglob("*.md")):
        relative = path.relative_to(vault)
        if ".system" in relative.parts or "40_Archive" in relative.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        embeds += len(re.findall(r"!\[\[[^\]]+\]\]", text))
        wikilinks += len(re.findall(r"(?<!\!)\[\[[^\]]+\]\]", text))
    return {"wikilinks": wikilinks, "embeds": embeds}


def _review_metadata_complete(plan: dict[str, Any]) -> bool:
    for row in plan.get("concepts") or []:
        if row.get("disposition") != "REVIEW":
            continue
        if any(not row.get(field) for field in _REVIEW_FIELDS):
            return False
        if not row.get("reason"):
            return False
    return True


def run(vault: Path, runtime_base: Path | None) -> dict[str, Any]:
    root = vault.resolve().parent
    vault = vault.resolve()
    r9 = run_r9_preflight(vault, runtime_base)
    concepts = scan_concepts(vault)
    plan = build_cleanup_plan(concepts)
    audit = scan_vault(vault).to_dict()
    runtime_state = capture_runtime_state(vault, runtime_root=(runtime_base / vault.name).resolve() if runtime_base else None)
    catalog_manifest = _catalog_manifest(vault)
    wiki_counts = _active_wikilink_counts(vault)
    catalog = runtime_state.get("catalog") or {}
    vector = runtime_state.get("vector") or {}
    checks = {
        "r9_preflight": r9.get("status") == "PASS",
        "portable_links_clean": all(
            int((audit.get("stats") or {}).get(key, 0)) == 0
            for key in ("broken_links", "ambiguous_links", "parse_errors")
        ),
        "concept_inventory_complete": len(plan.get("concepts") or []) == int(concepts.get("concept_file_count") or 0)
        and all(bool(row.get("disposition")) for row in plan.get("concepts") or []),
        "review_apply_boundary": all(
            not bool(row.get("apply_eligible")) for row in plan.get("concepts") or [] if row.get("disposition") == "REVIEW"
        ),
        "review_metadata_complete": _review_metadata_complete(plan),
        "no_active_wikilink_or_embed": wiki_counts["wikilinks"] == 0 and wiki_counts["embeds"] == 0,
        "catalog_link_projection": int((catalog.get("counts") or {}).get("note_links", 0)) > 0
        and int((catalog_manifest.get("target_counts", {}).get("counts") or {}).get("note_links", 0))
        == int((catalog.get("counts") or {}).get("note_links", 0)),
        "catalog_vector_policy_parity": (
            catalog.get("registry_digest") == vector.get("registry_digest") == r9.get("runtime_state", {}).get("registry_digest")
            and catalog.get("policy_digest") == vector.get("policy_digest") == r9.get("runtime_state", {}).get("policy_digest")
            and int(catalog.get("eligible_note_count") or 0) == int(vector.get("eligible_note_count") or 0)
        ),
        "graph_snapshot_resolved": int(concepts.get("unresolved_edge_count") or 0) == 0,
        "lossless_retrieval_code_present": all(
            token in (root / relative).read_text(encoding="utf-8")
            for relative, token in (
                ("tools/archivist/hybrid_retriever.py", "matched_chunk"),
                ("tools/archivist/core.py", "read_note_chunk"),
                ("tools/archivist/search.py", "search_memories_with_evidence"),
            )
        ),
        "stub_creation_default_blocked": (
            "allow_stub_creation: bool = False" in (root / "tools/archivist/vault_link_healer.py").read_text(encoding="utf-8")
            and "allow_stub_creation: bool = False" in (root / "tools/macro/news_funnel.py").read_text(encoding="utf-8")
        ),
    }
    return {
        "schema": "vault-r10-preflight-v1",
        "status": "PASS" if all(checks.values()) else "BLOCKED",
        "vault": str(vault),
        "checks": checks,
        "r9_preflight": r9,
        "concepts": {
            "snapshot": {
                key: concepts.get(key)
                for key in ("snapshot_fingerprint", "total_markdown_files", "concept_file_count", "eligible_note_count", "edge_count", "unresolved_edge_count")
            },
            "plan": {
                "policy_digest": plan.get("policy_digest"),
                "disposition_counts": plan.get("disposition_counts"),
                "apply_item_count": plan.get("apply_item_count"),
            },
        },
        "audit": {
            "total_files": audit.get("total_files"),
            "active_files_count": audit.get("active_files_count"),
            "excluded_files_count": audit.get("excluded_files_count"),
            "stats": audit.get("stats"),
        },
        "active_wikilinks": wiki_counts,
        "runtime_state": runtime_state,
        "catalog_manifest": {
            key: catalog_manifest.get(key)
            for key in ("generation_id", "database", "manifest", "eligible_note_count", "target_counts", "registry_digest", "policy_digest")
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", type=Path, default=Path("memories"))
    parser.add_argument("--runtime-base", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("scratch/vault-r10/preflight/r10-preflight.json"))
    args = parser.parse_args()
    report = run(args.vault, args.runtime_base)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve())}, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
