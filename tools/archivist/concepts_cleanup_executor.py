"""Guarded apply/rollback primitives for the R10 Concepts cleanup.

The planner is read-only.  This module is the explicitly invoked maintenance
boundary: it takes a lease, verifies the planner snapshot, preserves preimages,
rewrites only resolvable Markdown links, and moves retired content to an
external quarantine.  No operation here hard-deletes a canonical file.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from tools.archivist.concepts_cleanup import scan_concepts
from tools.archivist.core import _atomic_write_text
from tools.archivist.maintenance_guard import (
    acquire_maintenance_lease,
    release_maintenance_lease,
)
from tools.archivist.metadata import dump_note, parse_note
from tools.archivist.portable_links import (
    clear_vault_link_cache,
    iter_internal_markdown_links,
    render_relative_markdown_link,
    resolve_vault_target_detailed,
    rewrite_internal_markdown_links,
)
from tools.archivist.vault_paths import VaultPaths


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_text(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _validate_external_quarantine(vault: Path, quarantine_root: Path) -> Path:
    root = quarantine_root.resolve()
    if root == vault.resolve() or root.is_relative_to(vault.resolve()):
        raise ValueError("quarantine must be outside the Vault")
    return root


def _metadata_for_relocation(record: Mapping[str, Any], text: str) -> str:
    metadata, body, issues = parse_note(text)
    if issues:
        raise ValueError(f"cannot relocate malformed frontmatter: {record.get('path')}")
    disposition = str(record.get("target_path") or "")
    if "/NotebookLM_Sources/" in disposition:
        entity_type = "briefing_book"
        role = "source"
    elif "/Macroeconomics/Daily_Snapshots/" in disposition:
        entity_type = "macro_snapshot"
        role = "baseline"
    elif "/Stocks/TSLA/Analysis/" in disposition:
        entity_type = "equity_analysis"
        role = "analysis"
    else:
        raise ValueError(f"unsupported relocation target: {disposition}")
    metadata["entity_type"] = entity_type
    metadata["document_role"] = role
    metadata["search_scope"] = "included"
    metadata.setdefault("content_status", "published")
    metadata.setdefault("retention_class", "permanent")
    metadata.setdefault("sensitivity", "internal")
    metadata.setdefault("source_verification_status", "not_reviewed")
    metadata.setdefault("content_verification_status", "not_reviewed")
    metadata.setdefault("trust_tier", "T3")
    metadata.setdefault("production_eligible", False)
    return dump_note(metadata, body)


def _is_archive(relative_path: str) -> bool:
    return "40_Archive" in Path(relative_path).parts


def _build_rewrites(
    root: Path,
    move_map: Mapping[str, Optional[str]],
    selected: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, str], dict[str, str]]:
    """Return updated active texts and their preimage texts."""
    all_texts: dict[str, str] = {}
    for path in sorted(root.rglob("*.md"), key=lambda item: _relative(root, item)):
        rel = _relative(root, path)
        if ".system" in Path(rel).parts or _is_archive(rel):
            continue
        try:
            all_texts[rel] = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

    retired = {path for path, target in move_map.items() if target is None}
    relocated = {path for path, target in move_map.items() if target is not None}
    updates: dict[str, str] = {}
    preimages: dict[str, str] = {}

    for source_rel, original_text in all_texts.items():
        if source_rel in retired:
            continue
        new_source_rel = move_map.get(source_rel) or source_rel
        working = original_text
        if source_rel in relocated:
            working = _metadata_for_relocation(selected[source_rel], working)

        def replace(link):
            resolution = resolve_vault_target_detailed(root, link.destination, source=source_rel)
            if resolution.status != "resolved" or resolution.target is None:
                return None
            target_rel = _relative(root, resolution.target)
            mapped_target = move_map.get(target_rel, target_rel)
            if mapped_target is None:
                # A reference to an empty retired stub becomes visible plain
                # text so the source keeps the human-readable entity mention
                # without preserving a link to a non-knowledge page.
                return str(link.label or target_rel).replace("]", "\\]")
            if mapped_target == target_rel and new_source_rel == source_rel:
                return None
            return render_relative_markdown_link(
                root,
                new_source_rel,
                mapped_target,
                label=link.label or Path(target_rel).stem,
                fragment=link.fragment,
            )

        rewritten = rewrite_internal_markdown_links(working, replace)
        if rewritten != original_text or source_rel in relocated:
            updates[source_rel] = rewritten
            preimages[source_rel] = original_text
    return updates, preimages


def _copy_preimages(root: Path, quarantine: Path, preimages: Mapping[str, str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for rel, text in sorted(preimages.items()):
        target = quarantine / "preimages" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8", newline="\n")
        result[rel] = _relative(quarantine, target)
    return result


def _append_tombstones(vault: Path, rows: list[Mapping[str, Any]], *, owner: str, status: str) -> None:
    if not rows:
        return
    from tools.archivist.write_adapter import ArtifactWriterKnowledgeAdapter

    adapter = ArtifactWriterKnowledgeAdapter(vault_paths=VaultPaths(vault))
    for row in rows:
        adapter._append_tombstone(  # noqa: SLF001 - guarded maintenance boundary
            note_id=str(row.get("note_id") or "") or None,
            document_key=str(row.get("document_key") or "") or None,
            status=status,
            reason="r10_concepts_cleanup",
        )


def _policy_fix_rows(root: Path, snapshot: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in snapshot.get("concepts") or []:
        if not record.get("auto_stub"):
            continue
        policy_fields = (
            "search_scope",
            "lifecycle_status",
            "content_status",
            "retention_class",
            "trust_tier",
            "source_verification_status",
            "content_verification_status",
            "review_state",
            "review_owner",
            "review_reason",
        )
        if all(record.get(field) for field in policy_fields) and record.get("production_eligible") is False:
            continue
        path = root / str(record["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        original = path.read_text(encoding="utf-8")
        metadata, body, issues = parse_note(original)
        if issues:
            raise ValueError(f"cannot repair malformed stub metadata: {record['path']}")
        metadata.update(
            {
                "search_scope": "excluded",
                "lifecycle_status": "stub",
                "content_status": "generated",
                "retention_class": metadata.get("retention_class") or "ephemeral",
                "trust_tier": metadata.get("trust_tier") or "T3",
                "production_eligible": False,
                "source_verification_status": metadata.get("source_verification_status") or "not_reviewed",
                "content_verification_status": metadata.get("content_verification_status") or "not_reviewed",
                "review_state": metadata.get("review_state") or "pending",
                "review_owner": metadata.get("review_owner") or "data-owner",
                "review_reason": metadata.get("review_reason")
                or "referenced auto-stub; promote from evidence or convert links to plain text before retirement",
            }
        )
        updated = dump_note(metadata, body)
        rows.append(
            {
                "path": record["path"],
                "note_id": record.get("note_id"),
                "document_key": record.get("document_key"),
                "pre_content_sha256": _sha256_text(original),
                "post_content_sha256": _sha256_text(updated),
                "original_text": original,
                "updated_text": updated,
            }
        )
    return rows


def apply_cleanup(
    vault_root: str | Path,
    plan_path: str | Path,
    quarantine_root: str | Path,
    *,
    run_id: str,
    owner: str = "codex-r10",
    policy_only: bool = False,
) -> dict[str, Any]:
    """Apply only high-confidence plan items after an explicit operator call."""
    root = Path(vault_root).resolve()
    plan = json.loads(Path(plan_path).resolve().read_text(encoding="utf-8"))
    if not isinstance(plan, dict) or not isinstance(plan.get("concepts"), list):
        raise ValueError("invalid R10 cleanup plan")
    snapshot = scan_concepts(root)
    if snapshot.get("snapshot_fingerprint") != plan.get("snapshot_fingerprint"):
        raise RuntimeError(
            "Vault changed since the R10 plan was built; regenerate the inventory/plan before apply"
        )
    quarantine = _validate_external_quarantine(root, Path(quarantine_root) / run_id)
    quarantine.mkdir(parents=True, exist_ok=True)
    os.environ["VAULT_MAINTENANCE_OWNER"] = owner
    lease = acquire_maintenance_lease(
        root,
        owner=owner,
        purpose=f"r10-concepts-cleanup:{run_id}",
        ttl_seconds=3600,
        baseline_tree_fingerprint=str(snapshot.get("snapshot_fingerprint") or ""),
    )
    journal_path = quarantine / "journal.json"
    journal: dict[str, Any] = {
        "run_id": run_id,
        "owner": owner,
        "status": "prepared",
        "lease_id": lease.lease_id,
        "plan_path": str(Path(plan_path).resolve()),
        "snapshot_fingerprint": snapshot.get("snapshot_fingerprint"),
        "started_at": _utc_now(),
    }
    _write_json(journal_path, journal)
    try:
        if policy_only:
            rows = _policy_fix_rows(root, snapshot)
            preimages = {str(row["path"]): str(row["original_text"]) for row in rows}
            preimage_paths = _copy_preimages(root, quarantine, preimages)
            entries = [
                {
                    "path": row["path"],
                    "disposition": "POLICY_FIX",
                    "target_path": None,
                    "quarantine_path": None,
                    "note_id": row.get("note_id"),
                    "document_key": row.get("document_key"),
                    "pre_content_sha256": row["pre_content_sha256"],
                    "post_content_sha256": row["post_content_sha256"],
                    "preimage_path": preimage_paths[str(row["path"])],
                }
                for row in rows
            ]
            changed_files = [
                {
                    "path": row["path"],
                    "post_path": row["path"],
                    "pre_content_sha256": row["pre_content_sha256"],
                    "post_content_sha256": row["post_content_sha256"],
                    "preimage_path": preimage_paths[str(row["path"])],
                }
                for row in rows
            ]
            manifest = {
                "manifest_version": 1,
                "run_id": run_id,
                "mode": "policy_only",
                "snapshot_fingerprint": snapshot.get("snapshot_fingerprint"),
                "entries": entries,
                "changed_files": changed_files,
                "created_at": _utc_now(),
            }
            _write_json(quarantine / "manifest-pre.json", manifest)
            for row in rows:
                _atomic_write_text(root / str(row["path"]), str(row["updated_text"]))
            clear_vault_link_cache()
            manifest["status"] = "APPLIED"
            _write_json(quarantine / "manifest.json", manifest)
            journal.update({"status": "applied", "changed_files": len(rows), "finished_at": _utc_now()})
            _write_json(journal_path, journal)
            return {"status": "PASS", "mode": "policy_only", "changed_files": len(rows), "quarantine": str(quarantine)}

        selected_rows = {
            str(row["path"]): row
            for row in plan["concepts"]
            if row.get("apply_eligible")
            and str(row.get("disposition") or "") in {"RETIRE", "RELOCATE"}
        }
        if not selected_rows:
            raise RuntimeError("R10 plan has no approved high-confidence apply items")
        move_map: dict[str, Optional[str]] = {
            path: (str(row.get("target_path")) if str(row.get("disposition")) == "RELOCATE" else None)
            for path, row in selected_rows.items()
        }
        for old_rel, new_rel in move_map.items():
            source = root / old_rel
            if not source.is_file() or _sha256_file(source) != str(selected_rows[old_rel].get("content_sha256")):
                raise RuntimeError(f"source changed since plan: {old_rel}")
            if new_rel:
                target = root / new_rel
                if target.exists() or target.is_symlink():
                    raise RuntimeError(f"relocation target collision: {new_rel}")

        updates, preimages = _build_rewrites(root, move_map, selected_rows)
        # Every moved source is a preimage even when its body has no links.
        for old_rel in move_map:
            if old_rel not in preimages:
                preimages[old_rel] = (root / old_rel).read_text(encoding="utf-8")
        preimage_paths = _copy_preimages(root, quarantine, preimages)
        entries: list[dict[str, Any]] = []
        for old_rel, row in sorted(selected_rows.items()):
            new_rel = move_map[old_rel]
            destination = new_rel or f"files/{old_rel}"
            post_text = updates.get(old_rel, preimages[old_rel]) if new_rel else None
            entries.append(
                {
                    "path": old_rel,
                    "disposition": row.get("disposition"),
                    "target_path": new_rel,
                    "quarantine_path": destination,
                    "note_id": row.get("note_id"),
                    "document_key": row.get("document_key"),
                    "pre_content_sha256": _sha256_text(preimages[old_rel]),
                    "post_content_sha256": _sha256_text(post_text) if post_text is not None else None,
                    "preimage_path": preimage_paths[old_rel],
                }
            )
        changed_files = []
        for rel, original in sorted(preimages.items()):
            changed_path = move_map.get(rel) or rel
            if rel in move_map and move_map[rel] is None:
                post_hash = None
            else:
                post_hash = _sha256_text(updates.get(rel, original))
            changed_files.append(
                {
                    "path": rel,
                    "post_path": changed_path,
                    "pre_content_sha256": _sha256_text(original),
                    "post_content_sha256": post_hash,
                    "preimage_path": preimage_paths[rel],
                }
            )
        manifest = {
            "manifest_version": 1,
            "run_id": run_id,
            "mode": "cleanup",
            "status": "PREPARED",
            "snapshot_fingerprint": snapshot.get("snapshot_fingerprint"),
            "plan_policy_digest": plan.get("policy_digest"),
            "entries": entries,
            "changed_files": changed_files,
            "created_at": _utc_now(),
        }
        _write_json(quarantine / "manifest-pre.json", manifest)

        # Move canonical files first.  All preimages and collision checks are
        # complete, and the maintenance lease fences other Vault writers.
        for old_rel, new_rel in sorted(move_map.items()):
            source = root / old_rel
            if new_rel:
                target = root / new_rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(target))
            else:
                target = quarantine / "files" / old_rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(target))

        # Re-render relocated content at its new source path and update active
        # inbound/generated notes.  Link changes are limited to targets that
        # resolved in the pre-apply snapshot.
        for old_rel, new_rel in sorted(move_map.items()):
            if not new_rel:
                continue
            target = root / new_rel
            _atomic_write_text(target, updates.get(old_rel, preimages[old_rel]))
        for rel, content in sorted(updates.items()):
            if rel in move_map:
                continue
            _atomic_write_text(root / rel, content)

        clear_vault_link_cache()

        retire_rows = [row for row in entries if row.get("disposition") == "RETIRE"]
        _append_tombstones(root, retire_rows, owner=owner, status="retired")
        manifest["status"] = "APPLIED"
        manifest["applied_at"] = _utc_now()
        _write_json(quarantine / "manifest.json", manifest)
        journal.update(
            {
                "status": "applied",
                "selected_files": len(entries),
                "rewritten_files": len(updates),
                "retired_files": len(retire_rows),
                "relocated_files": sum(1 for item in entries if item.get("disposition") == "RELOCATE"),
                "finished_at": _utc_now(),
            }
        )
        _write_json(journal_path, journal)
        return {
            "status": "PASS",
            "mode": "cleanup",
            "selected_files": len(entries),
            "rewritten_files": len(updates),
            "retired_files": len(retire_rows),
            "relocated_files": sum(1 for item in entries if item.get("disposition") == "RELOCATE"),
            "quarantine": str(quarantine),
        }
    except Exception as exc:
        journal.update({"status": "FAILED", "error": str(exc), "failed_at": _utc_now()})
        _write_json(journal_path, journal)
        raise
    finally:
        try:
            release_maintenance_lease(root, owner=owner, reason="r10 cleanup finished")
        finally:
            os.environ.pop("VAULT_MAINTENANCE_OWNER", None)


def rollback_cleanup(
    vault_root: str | Path,
    manifest_path: str | Path,
    *,
    owner: str = "codex-r10-rollback",
) -> dict[str, Any]:
    """Restore one applied run, refusing to overwrite unexpected edits."""
    root = Path(vault_root).resolve()
    manifest_file = Path(manifest_path).resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("status") != "APPLIED":
        raise RuntimeError("only an APPLIED R10 manifest can be rolled back")
    quarantine = manifest_file.parent
    os.environ["VAULT_MAINTENANCE_OWNER"] = owner
    lease = acquire_maintenance_lease(root, owner=owner, purpose="r10-concepts-rollback", ttl_seconds=3600)
    moved_back = 0
    restored_files = 0
    try:
        for change in manifest.get("changed_files") or []:
            post_path = root / str(change.get("post_path") or change["path"])
            if not post_path.is_file():
                continue
            expected_post = str(change.get("post_content_sha256") or "")
            if expected_post and _sha256_file(post_path) != expected_post:
                raise RuntimeError(f"rollback refused: current file changed {change['post_path']}")
        # Move relocated files back to a recoverable external rollback area,
        # then restore exact preimages to their original paths.
        discarded = quarantine / "rollback-discard"
        for entry in manifest.get("entries") or []:
            target_rel = entry.get("target_path")
            if not target_rel:
                continue
            current = root / str(target_rel)
            if current.is_file():
                target = discarded / str(target_rel)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(current), str(target))
                moved_back += 1
        retired_paths = {
            str(entry["path"])
            for entry in (manifest.get("entries") or [])
            if entry.get("disposition") == "RETIRE"
        }
        for change in manifest.get("changed_files") or []:
            original_rel = str(change["path"])
            # The canonical retired preimage is restored by moving the
            # quarantined file below. Writing it first would make the move
            # platform-dependent and could overwrite a user's path.
            if original_rel in retired_paths:
                continue
            preimage = quarantine / str(change["preimage_path"])
            if not preimage.is_file():
                raise RuntimeError(f"rollback preimage missing: {preimage}")
            target = root / original_rel
            target.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write_text(target, preimage.read_text(encoding="utf-8"))
            restored_files += 1
        clear_vault_link_cache()
        retired = [entry for entry in manifest.get("entries") or [] if entry.get("disposition") == "RETIRE"]
        for entry in retired:
            current = quarantine / "files" / str(entry["path"])
            target = root / str(entry["path"])
            if current.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(current), str(target))
        _append_tombstones(root, retired, owner=owner, status="restored")
        rollback_manifest = dict(manifest)
        rollback_manifest["status"] = "ROLLED_BACK"
        rollback_manifest["rolled_back_at"] = _utc_now()
        _write_json(quarantine / "manifest-rollback.json", rollback_manifest)
        return {
            "status": "PASS",
            "restored_files": restored_files,
            "moved_back": moved_back,
            "quarantine": str(quarantine),
            "lease_id": lease.lease_id,
        }
    finally:
        release_maintenance_lease(root, owner=owner, reason="r10 rollback finished")
        os.environ.pop("VAULT_MAINTENANCE_OWNER", None)


__all__ = ["apply_cleanup", "rollback_cleanup"]
