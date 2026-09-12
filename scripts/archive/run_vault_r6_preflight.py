"""Create the guarded R6 baseline, portability inventory, and rollback point.

This command is intentionally read-only with respect to note content.  It
creates an R6 maintenance lease and an external snapshot so the subsequent
multi-app migration can be rehearsed and rolled back deterministically.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.archivist.maintenance_guard import (  # noqa: E402
    acquire_maintenance_lease,
    release_maintenance_lease,
)
from tools.archivist.vault_backup import create_vault_snapshot, restore_vault_snapshot  # noqa: E402


OWNER = "codex-vault-r6"
_WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}
_WIKILINK_RE = re.compile(r"!??\[\[[^\]]+\]\]")
_MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\([^\)]+\)")
_ABSOLUTE_LINK_RE = re.compile(r"(?:\[[^\]]+\]\()(?:(?:[A-Za-z]:[\\/])|(?:file://)|(?:/))")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _inventory(root: Path) -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rel = path.relative_to(root).as_posix()
        stat = path.stat()
        row = {
            "relative_path": rel,
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": _sha256(path),
        }
        rows.append(row)
        encoded = json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return rows, digest.hexdigest()


def _body_without_code(text: str) -> str:
    lines: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if re.match(r"^\s*```", line):
            in_fence = not in_fence
            continue
        if not in_fence:
            lines.append(re.sub(r"`[^`]*`", "", line))
    return "\n".join(lines)


def _portable_inventory(vault: Path) -> dict[str, Any]:
    markdown = sorted(path for path in vault.rglob("*.md") if path.is_file())
    wikilink_rows: list[dict[str, Any]] = []
    markdown_link_rows: list[dict[str, Any]] = []
    app_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    case_map: dict[str, list[str]] = {}
    counts = {
        "markdown_files": len(markdown),
        "files_with_wikilinks": 0,
        "files_with_wikilink_embeds": 0,
        "files_with_markdown_links": 0,
        "files_with_absolute_links": 0,
        "files_with_dataview": 0,
        "files_with_meta_bind": 0,
        "files_with_obsidian_uri": 0,
        "files_with_iframe": 0,
        "files_with_active_script": 0,
    }
    for path in markdown:
        rel = path.relative_to(vault).as_posix()
        case_map.setdefault(rel.casefold(), []).append(rel)
        text = path.read_text(encoding="utf-8")
        body = _body_without_code(text)
        wikilinks = _WIKILINK_RE.findall(body)
        embeds = re.findall(r"!\[\[[^\]]+\]\]", body)
        md_links = _MARKDOWN_LINK_RE.findall(body)
        absolute = _ABSOLUTE_LINK_RE.findall(body)
        dataview = bool(re.search(r"```(?:dataview|dataviewjs)\b", text, re.IGNORECASE))
        meta_bind = bool(re.search(r"meta-bind|INPUT\[", text, re.IGNORECASE))
        obsidian_uri = bool(re.search(r"obsidian://", text, re.IGNORECASE))
        iframe = bool(re.search(r"<iframe\b", text, re.IGNORECASE))
        active_script = bool(re.search(r"<script\b|\son[a-z]+\s*=", text, re.IGNORECASE))
        if wikilinks:
            counts["files_with_wikilinks"] += 1
            wikilink_rows.append({"relative_path": rel, "count": len(wikilinks), "tokens": wikilinks[:100]})
        if embeds:
            counts["files_with_wikilink_embeds"] += 1
        if md_links:
            counts["files_with_markdown_links"] += 1
            markdown_link_rows.append({"relative_path": rel, "count": len(md_links)})
        if absolute:
            counts["files_with_absolute_links"] += 1
        if dataview:
            counts["files_with_dataview"] += 1
        if meta_bind:
            counts["files_with_meta_bind"] += 1
        if obsidian_uri:
            counts["files_with_obsidian_uri"] += 1
        if iframe:
            counts["files_with_iframe"] += 1
        if active_script:
            counts["files_with_active_script"] += 1
        if wikilinks or embeds or dataview or meta_bind or obsidian_uri or iframe or active_script:
            app_rows.append({
                "relative_path": rel,
                "wikilinks": len(wikilinks),
                "embeds": len(embeds),
                "dataview": dataview,
                "meta_bind": meta_bind,
                "obsidian_uri": obsidian_uri,
                "iframe": iframe,
                "active_script": active_script,
            })

        if len(rel) > 160 or len(rel) > 180:
            path_rows.append({
                "relative_path": rel,
                "length": len(rel),
                "hard_violation": len(rel) > 180,
                "reason": "path_length",
            })
        stem = path.stem.rstrip(". ")
        if stem.split(".", 1)[0].upper() in _WINDOWS_RESERVED:
            path_rows.append({"relative_path": rel, "reason": "reserved_name", "hard_violation": True})

    collisions = [values for values in case_map.values() if len(values) > 1]
    settings_path = vault / ".obsidian" / "app.json"
    templates_config = vault / ".obsidian" / "templates.json"
    daily_config = vault / ".obsidian" / "daily-notes.json"
    return {
        "status": "PASS",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "counts": counts,
        "wikilink_inventory": wikilink_rows,
        "markdown_link_inventory": markdown_link_rows,
        "app_syntax_inventory": app_rows,
        "path_policy_candidates": sorted(path_rows, key=lambda item: item["relative_path"]),
        "casefold_collisions": collisions,
        "config": {
            "app_json_exists": settings_path.is_file(),
            "templates_json_exists": templates_config.is_file(),
            "daily_notes_json_exists": daily_config.is_file(),
            "template_files": sorted(
                path.relative_to(vault).as_posix()
                for path in (vault / "99_Templates").glob("*.md")
                if path.is_file()
            ),
        },
    }


def _snapshot_hash_match(snapshot: Path, vault: Path, restore_dir: Path) -> dict[str, Any]:
    restored_count = restore_vault_snapshot(snapshot, restore_dir, verify_checksum=True)
    with zipfile.ZipFile(snapshot, "r") as archive:
        members = sorted(item.filename for item in archive.infolist() if not item.is_dir())
    mismatches: list[str] = []
    for rel in members:
        live = vault / Path(rel)
        restored = restore_dir / Path(rel)
        if not live.is_file() or not restored.is_file() or _sha256(live) != _sha256(restored):
            mismatches.append(rel)
            if len(mismatches) >= 50:
                break
    return {
        "status": "PASS" if not mismatches else "FAIL",
        "snapshot": str(snapshot),
        "snapshot_sha256": _sha256(snapshot),
        "archive_file_count": len(members),
        "restored_file_count": restored_count,
        "hashes_match": not mismatches,
        "mismatches": mismatches,
        "restore_dir": str(restore_dir),
    }


def run(vault: Path, run_dir: Path, *, owner: str = OWNER, ttl_seconds: int = 4 * 60 * 60) -> dict[str, Any]:
    vault = vault.resolve()
    run_dir = run_dir.resolve()
    if not vault.is_dir():
        raise FileNotFoundError(vault)
    if run_dir.is_relative_to(vault):
        raise ValueError(f"R6 evidence must be outside the vault: {run_dir}")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty R6 run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    baseline_rows, baseline_fingerprint = _inventory(vault)
    portability = _portable_inventory(vault)
    _write_jsonl(run_dir / "baseline-inventory.jsonl", baseline_rows)
    _write_json(run_dir / "baseline-portability.json", portability)
    _write_jsonl(run_dir / "baseline-link-inventory.jsonl", portability["wikilink_inventory"])
    _write_jsonl(run_dir / "baseline-app-syntax.jsonl", portability["app_syntax_inventory"])
    _write_json(
        run_dir / "baseline.json",
        {
            "phase": "F00",
            "started_at": started_at,
            "vault_root": str(vault),
            "file_count": len(baseline_rows),
            "tree_fingerprint": baseline_fingerprint,
            "owner": owner,
            "portability_counts": portability["counts"],
        },
    )
    lease = None
    try:
        lease = acquire_maintenance_lease(
            vault,
            owner=owner,
            purpose="R6 multi-app vault portability migration",
            ttl_seconds=ttl_seconds,
            baseline_tree_fingerprint=baseline_fingerprint,
        )
        _write_json(run_dir / "lease.json", lease.__dict__)
        snapshot, snapshot_hash = create_vault_snapshot(vault_root=vault, backup_dir=run_dir / "snapshots")
        restore_proof = _snapshot_hash_match(snapshot, vault, run_dir / "snapshot-restore")
        restore_proof.update({"snapshot_checksum": snapshot_hash, "baseline_fingerprint": baseline_fingerprint})
        _write_json(run_dir / "snapshot-restore-proof.json", restore_proof)
        if restore_proof["status"] != "PASS":
            raise RuntimeError(f"R6 snapshot restore proof failed: {restore_proof['mismatches'][:3]}")
        after_rows, after_fingerprint = _inventory(vault)
        _write_jsonl(run_dir / "post-snapshot-inventory.jsonl", after_rows)
        result = {
            "status": "PASS",
            "phase": "F00",
            "run_dir": str(run_dir),
            "vault_root": str(vault),
            "owner": owner,
            "lease_id": lease.lease_id,
            "lease_status": lease.status,
            "baseline_file_count": len(baseline_rows),
            "post_snapshot_file_count": len(after_rows),
            "baseline_tree_fingerprint": baseline_fingerprint,
            "post_snapshot_tree_fingerprint": after_fingerprint,
            "snapshot_restore_status": restore_proof["status"],
            "snapshot": str(snapshot),
            "snapshot_sha256": snapshot_hash,
            "lease_left_active": True,
        }
        _write_json(run_dir / "run.json", result)
        print(json.dumps(result, ensure_ascii=False))
        return result
    except Exception:
        if lease is not None:
            try:
                release_maintenance_lease(vault, owner=owner, reason="F00 preflight failed")
            except Exception:
                pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", type=Path, default=Path("memories"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--owner", default=OWNER)
    parser.add_argument("--ttl-seconds", type=int, default=4 * 60 * 60)
    args = parser.parse_args()
    run(args.vault, args.run_dir, owner=args.owner, ttl_seconds=args.ttl_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
