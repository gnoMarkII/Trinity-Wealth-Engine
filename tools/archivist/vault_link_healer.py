"""Report broken links without manufacturing Concepts notes.

The old implementation treated a missing target as permission to create an
empty note. That made the Vault look connected while polluting the knowledge
base. Production callers are now report-only; the legacy stub path remains an
explicit opt-in for migration fixtures only.
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

from core.logger import get_logger
from tools.archivist.core import _atomic_write_text, _sanitize_filename
from tools.archivist.vault_audit import scan_vault
from tools.archivist.artifact_writer import ArtifactWriter
from tools.archivist.vault_paths import VaultPaths
from tools.archivist.maintenance_guard import assert_write_allowed

logger = get_logger(__name__)


def heal_broken_links(
    vault_root: Optional[Union[str, Path]] = None,
    dry_run: bool = False,
    *,
    allow_stub_creation: bool = False,
) -> dict[str, int]:
    """Scan broken links; never create a stub unless explicitly opted in.

    ``allow_stub_creation`` is retained solely for a controlled legacy
    migration/test path. News and other production producers must use the
    default report-only behavior and store unresolved names as structured
    metadata or visible plain text.
    """
    v_root = Path(vault_root).resolve() if vault_root else Path(os.getenv("OBSIDIAN_VAULT_PATH", "./memories")).resolve()
    if not v_root.exists():
        return {"scanned": 0, "broken_found": 0, "stubs_created": 0}

    logger.info("Scanning vault for broken links at: %s", v_root)
    audit_res = scan_vault(v_root)
    broken_issues = [iss for iss in audit_res.issues if iss.issue_type == "broken_link"]

    if not dry_run and not allow_stub_creation:
        logger.warning(
            "Broken-link healer is report-only; %d missing targets were not turned into Concept stubs",
            len(broken_issues),
        )
        return {
            "scanned": audit_res.total_files,
            "broken_found": len(broken_issues),
            "stubs_created": 0,
            "stub_creation_blocked": len(broken_issues),
        }

    concepts_dir = v_root / "30_Knowledge_Base" / "Concepts"
    if not dry_run and allow_stub_creation:
        assert_write_allowed(v_root)
        concepts_dir.mkdir(parents=True, exist_ok=True)

    stubs_created = 0
    seen_targets = set()
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Connect to catalog if exists
    cat = None
    try:
        from tools.archivist.catalog_adapter import SqliteNoteCatalogAdapter
        from tools.archivist.catalog_runtime import resolve_catalog_path
        cat_db = resolve_catalog_path(v_root, require_exists=True)
        if cat_db.exists():
            cat = SqliteNoteCatalogAdapter(db_path=cat_db, vault_root=v_root)
    except Exception:
        pass

    for issue in broken_issues:
        raw_target = None
        if isinstance(issue.details, dict):
            raw_target = issue.details.get("target")
        elif isinstance(issue.details, str):
            # Formats like "Target not found: [[TargetName]]"
            m = re.search(r"\[\[(.*?)\]\]", issue.details)
            if m:
                raw_target = m.group(1)
            else:
                raw_target = issue.details
        if not raw_target:
            continue

        clean_target = raw_target.split("|")[0].split("#")[0].strip()
        if not clean_target or clean_target in seen_targets:
            continue
        seen_targets.add(clean_target)

        # Sanitize name
        safe_name = _sanitize_filename(clean_target)
        if not safe_name or safe_name.lower() in ("untitled", "unknown", "none"):
            continue

        stub_path = concepts_dir / f"{safe_name}.md"
        if stub_path.exists():
            continue

        if not dry_run:
            content = (
                f"---\n"
                f"title: \"{clean_target}\"\n"
                f"entity_type: concept\n"
                f"tags:\n"
                f"  - concept/stub\n"
                f"  - referential-integrity\n"
                f"date: \"{today_str}\"\n"
                f"---\n\n"
                f"# {clean_target}\n\n"
                f"> 💡 โน้ตนี้ถูกสร้างขึ้นโดยระบบอัตโนมัติ (Concept Stub) เพื่อรักษาความสมบูรณ์ของการเชื่อมโยงเครือข่ายความรู้ (GraphRAG)\n\n"
                f"## รายละเอียด\n"
                f"ยังไม่มีเนื้อหาฉบับเต็ม สามารถเพิ่มข้อมูลเพิ่มเติมได้ในภายหลัง\n"
            )
            committed = ArtifactWriter(vault_paths=VaultPaths(v_root)).write_note(
                metadata={
                    "schema_version": 2,
                    "title": clean_target,
                    "entity_type": "concept",
                    "tags": ["concept/stub", "referential-integrity"],
                    "date": today_str,
                },
                body=(
                    f"# {clean_target}\n\n"
                    "> Automatically created concept stub for a repaired reference.\n"
                ),
                filename=safe_name,
            )
            stub_path = committed.primary_file
            if cat:
                try:
                    cat.upsert_note_from_file(stub_path)
                except Exception as e:
                    logger.warning("Failed to catalog stub %s: %s", stub_path, e)

        stubs_created += 1

    return {
        "scanned": audit_res.total_files,
        "broken_found": len(broken_issues),
        "stubs_created": stubs_created,
        "stub_creation_blocked": 0,
    }
