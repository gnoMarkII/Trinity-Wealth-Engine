"""Apply the proof-backed Vault V2 R3 file repair set.

The command is intentionally conservative: it snapshots and restore-verifies the
vault first, guards every existing file by SHA-256, preserves custom frontmatter
and body bytes, and rewrites only links with a unique canonical target.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import uuid
import zipfile
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from application.knowledge.identity import NoteIdentity, build_document_key
from tools.archivist.artifact_writer import ArtifactWriter
from tools.archivist.catalog_adapter import SqliteNoteCatalogAdapter
from tools.archivist.catalog_runtime import resolve_catalog_path
from tools.archivist.identity_store import DurableIdentityStore
from tools.archivist.metadata import dump_note, parse_note
from tools.archivist.vault_audit import scan_vault, write_audit_report
from tools.archivist.vault_backup import create_vault_snapshot, restore_vault_snapshot
from tools.archivist.vault_paths import VaultPaths


WORKSPACE = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = WORKSPACE / "scratch" / "vault-v2" / "remediation-r3"
WIKILINK_RE = re.compile(r"\[\[([^\]|]+?)(?:\|([^\]]+))?\]\]")
H1_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
TOP_LEVEL_KEY_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_-]*)\s*:")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_fingerprint(root: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".chroma_index" in path.parts:
            continue
        rel = path.relative_to(root).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\n")
        count += 1
    return digest.hexdigest(), count


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def _atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.repair-{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_bytes(value)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _decode_markdown(raw: bytes) -> tuple[str, bool]:
    has_bom = raw.startswith(b"\xef\xbb\xbf")
    payload = raw[3:] if has_bom else raw
    return payload.decode("utf-8"), has_bom


def _encode_markdown(text: str, has_bom: bool) -> bytes:
    encoded = text.encode("utf-8")
    return (b"\xef\xbb\xbf" + encoded) if has_bom else encoded


def _split_frontmatter(text: str) -> tuple[list[str] | None, str, str]:
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].rstrip("\r\n") != "---":
        return None, text, newline
    closing = None
    for index in range(1, len(lines)):
        if lines[index].rstrip("\r\n") == "---":
            closing = index
            break
    if closing is None:
        raise ValueError("unclosed YAML frontmatter")
    return lines[: closing + 1], "".join(lines[closing + 1 :]), newline


def _yaml_scalar(value: Any) -> str:
    if isinstance(value, (date, datetime)):
        value = value.isoformat()[:10]
    rendered = yaml.safe_dump(
        value,
        allow_unicode=True,
        default_flow_style=True,
        sort_keys=False,
    ).strip()
    if rendered.endswith("\n..."):
        rendered = rendered[:-4]
    if rendered.endswith("..."):
        rendered = rendered[:-3].rstrip()
    return rendered


def _patch_markdown(
    raw: bytes,
    updates: dict[str, Any],
    canonical_links: dict[str, str],
    directory_links: dict[str, str],
) -> tuple[bytes, Counter[str]]:
    text, has_bom = _decode_markdown(raw)
    frontmatter, body, newline = _split_frontmatter(text)
    if frontmatter is None:
        frontmatter = [f"---{newline}", f"---{newline}"]

    key_lines: dict[str, int] = {}
    for index, line in enumerate(frontmatter[1:-1], start=1):
        match = TOP_LEVEL_KEY_RE.match(line.rstrip("\r\n"))
        if match:
            key_lines[match.group(1)] = index
    for key, value in updates.items():
        rendered = f"{key}: {_yaml_scalar(value)}{newline}"
        if key in key_lines:
            frontmatter[key_lines[key]] = rendered
        else:
            frontmatter.insert(-1, rendered)

    counts: Counter[str] = Counter()

    def replace_link(match: re.Match[str]) -> str:
        target = match.group(1).strip()
        alias = match.group(2)
        anchor_at = min(
            [pos for pos in (target.find("#"), target.find("^")) if pos >= 0],
            default=-1,
        )
        base = target if anchor_at < 0 else target[:anchor_at]
        anchor = "" if anchor_at < 0 else target[anchor_at:]

        replacement = directory_links.get(base)
        category = "directory"
        if replacement is None and "/" not in base and "\\" not in base:
            ticker = Path(base).stem.upper()
            replacement = canonical_links.get(ticker)
            category = ticker
        if replacement is None:
            return match.group(0)

        counts[category] += 1
        rendered_target = f"{replacement}{anchor}"
        if alias is not None:
            return f"[[{rendered_target}|{alias}]]"
        return f"[[{rendered_target}]]"

    body = WIKILINK_RE.sub(replace_link, body)
    new_text = "".join(frontmatter) + body
    return _encode_markdown(new_text, has_bom), counts


def _infer_entity_type(relative_path: str) -> str | None:
    path = Path(relative_path)
    parts = path.parts
    if relative_path == "index.md":
        return "index"
    if "Earnings" in parts:
        return "earnings_call"
    if "Quant" in parts:
        return "quant_snapshot"
    if "Analysis" in parts:
        return "equity_analysis"
    if "Stocks" in parts:
        parent = path.parent.name.upper()
        return "stock_hub" if path.stem.upper() == parent else "equity_analysis"
    if "YouTube_Summaries" in parts:
        return "youtube_summary"
    if "News" in parts:
        return "company_news"
    if "Books" in parts:
        return "book_note"
    if "Daily_Snapshots" in parts:
        return "macro_snapshot"
    if "Macroeconomics" in parts or "Strategies" in parts:
        return "macro_strategy"
    if "NotebookLM_Sources" in parts:
        return "briefing_book"
    if "Concepts" in parts:
        return "concept"
    return None


def _new_note_id(used: set[str]) -> str:
    while True:
        candidate = f"note_{uuid.uuid4().hex[:12]}"
        if candidate not in used:
            used.add(candidate)
            return candidate


def _snapshot_restore_proof(vault: Path, run_dir: Path) -> dict[str, Any]:
    backup_dir = run_dir / "snapshot"
    snapshot, checksum = create_vault_snapshot(vault_root=vault, backup_dir=backup_dir)
    if _sha256_file(snapshot) != checksum:
        raise RuntimeError("snapshot checksum changed immediately after creation")
    restore_dir = run_dir / "restore_probe"
    restored_count = restore_vault_snapshot(snapshot, restore_dir, verify_checksum=True)

    mismatches: list[str] = []
    with zipfile.ZipFile(snapshot, "r") as archive:
        members = sorted(item.filename for item in archive.infolist() if not item.is_dir())
    for rel in members:
        live_path = vault / Path(rel)
        restored_path = restore_dir / Path(rel)
        if not live_path.is_file() or _sha256_file(live_path) != _sha256_file(restored_path):
            mismatches.append(rel)
            if len(mismatches) >= 20:
                break
    if mismatches:
        raise RuntimeError(f"snapshot restore proof differs from live vault: {mismatches}")
    return {
        "path": str(snapshot),
        "sha256": checksum,
        "members": len(members),
        "restored_files": restored_count,
        "restore_proof": "PASS",
    }


def _catalog_by_path(vault: Path) -> dict[str, Any]:
    try:
        catalog_file = resolve_catalog_path(vault, require_exists=True)
    except FileNotFoundError:
        return {}
    catalog = SqliteNoteCatalogAdapter(
        db_path=catalog_file,
        vault_root=vault,
        read_only=True,
    )
    return {
        entry.relative_path.replace("\\", "/"): entry
        # Repair planning must see unresolved/legacy rows as evidence too.
        # The normal iterator intentionally hides those rows from search, but
        # hiding them here would allocate a fresh authoritative note_id for a
        # note whose identity is still unresolved.
        for entry in catalog.iter_notes(page_size=500, include_non_searchable=True)
    }


def _nav_note(
    note_id: str,
    document_key: str,
    title: str,
    body: str,
    today: str,
) -> bytes:
    return dump_note(
        {
            "schema_version": 2,
            "note_id": note_id,
            "document_key": document_key,
            "entity_type": "index",
            "title": title,
            "date": today,
            "scope": "navigation",
            "generated_by": "vault_v2_live_repair_r3",
        },
        body,
    ).encode("utf-8")


def _link_line(relative_path: str, title: str) -> str:
    target = str(Path(relative_path).with_suffix("")).replace("\\", "/")
    safe_title = title.replace("|", "-").replace("]", "")
    return f"- [[{target}|{safe_title}]]"


def _make_navigation_files(
    vault: Path,
    active_records: list[Any],
    used_ids: set[str],
    identity_records: list[NoteIdentity],
    today: str,
    planned_stock_hubs: set[str],
) -> dict[str, bytes]:
    specs = {
        "Stocks_Hub.md": ("stocks", "Stocks & Equities Hub", "30_Knowledge_Base/Stocks/"),
        "Macro_Hub.md": ("macro", "Macroeconomics & Strategy Hub", "30_Knowledge_Base/Macroeconomics/"),
        "News_Hub.md": ("news", "News Hub", "30_Knowledge_Base/News/"),
        "YouTube_Hub.md": ("youtube", "YouTube Summaries Hub", "30_Knowledge_Base/YouTube_Summaries/"),
        "NotebookLM_Sources_Hub.md": (
            "notebooklm-sources",
            "NotebookLM Sources Hub",
            "30_Knowledge_Base/NotebookLM_Sources/",
        ),
        "Concepts_Hub.md": ("concepts", "Concepts Hub", "30_Knowledge_Base/Concepts/"),
    }
    by_prefix: dict[str, list[tuple[str, str]]] = {}
    for _, (_, _, prefix) in specs.items():
        candidates: list[tuple[str, str]] = []
        for record in active_records:
            if not record.relative_path.startswith(prefix) or record.relative_path.endswith("/index.md"):
                continue
            if prefix.endswith("Stocks/"):
                parts = Path(record.relative_path).parts
                if len(parts) < 4 or Path(record.relative_path).stem.upper() != parts[2].upper():
                    continue
            title = str(record.properties.get("title") or Path(record.relative_path).stem)
            candidates.append((record.relative_path, title))
        if prefix.endswith("Stocks/"):
            for ticker in sorted(planned_stock_hubs):
                rel = f"30_Knowledge_Base/Stocks/{ticker}/{ticker}.md"
                candidates.append((rel, ticker))
        by_prefix[prefix] = sorted(set(candidates))[:100]

    created: dict[str, bytes] = {}
    nav_targets: dict[str, tuple[str, str]] = {}
    for filename, (source, title, prefix) in specs.items():
        rel = f"00_Index/{filename}"
        nav_targets[source] = (rel, title)
        if (vault / rel).exists():
            continue
        document_key = build_document_key("navigation", source, role="hub")
        note_id = _new_note_id(used_ids)
        identity_records.append(NoteIdentity(note_id=note_id, document_key=document_key))
        lines = [f"# {title}", ""]
        targets = by_prefix[prefix]
        if targets:
            lines.extend(_link_line(path, label) for path, label in targets)
        else:
            lines.append("No notes yet.")
        created[rel] = _nav_note(note_id, document_key, title, "\n".join(lines), today)

    home_rel = "00_Index/Home.md"
    if not (vault / home_rel).exists():
        document_key = build_document_key("navigation", "home", role="hub")
        note_id = _new_note_id(used_ids)
        identity_records.append(NoteIdentity(note_id=note_id, document_key=document_key))
        lines = ["# Investment Knowledge Base", ""]
        for source in ("stocks", "macro", "news", "youtube", "notebooklm-sources", "concepts"):
            rel, title = nav_targets[source]
            lines.append(_link_line(rel, title))
        created[home_rel] = _nav_note(
            note_id,
            document_key,
            "Investment Knowledge Base",
            "\n".join(lines),
            today,
        )
    return created


def _issue_targets(audit: Any, issue_type: str) -> Counter[str]:
    targets: Counter[str] = Counter()
    for issue in audit.issues:
        if issue.issue_type != issue_type:
            continue
        match = re.search(r"\[\[([^\]]+)\]\]", issue.details)
        targets[match.group(1) if match else issue.details] += 1
    return targets


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", type=Path, default=WORKSPACE / "memories")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--apply", action="store_true", help="Required to mutate the vault")
    args = parser.parse_args(argv)
    if not args.apply:
        parser.error("--apply is required; this command has no implicit mutation mode")

    vault = args.vault.resolve()
    if not vault.is_dir():
        raise FileNotFoundError(f"vault does not exist: {vault}")
    run_id = datetime.now(timezone.utc).strftime("live_repair_%Y%m%dT%H%M%SZ")
    run_dir = args.output_root.resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # SQLite may materialize persistent ``-wal``/``-shm`` companions even for
    # a URI read-only connection.  Open the catalog before freezing the
    # baseline so those transient companions cannot trip the mutation guard.
    catalog = _catalog_by_path(vault)
    before_fingerprint, before_file_count = _tree_fingerprint(vault)
    before_audit = scan_vault(vault)
    write_audit_report(before_audit, run_dir / "before")
    snapshot = _snapshot_restore_proof(vault, run_dir)
    if _tree_fingerprint(vault) != (before_fingerprint, before_file_count):
        raise RuntimeError("vault changed while snapshot/restore proof was being created")

    active_records = [
        record
        for record in before_audit.inventory
        if record.extension == ".md" and not record.is_excluded
    ]
    record_by_path = {record.relative_path: record for record in active_records}
    used_ids: set[str] = set()
    identity_records: list[NoteIdentity] = []
    identity_sources: Counter[str] = Counter()
    metadata_counts: Counter[str] = Counter()
    blocked: list[dict[str, str]] = []

    parsed: dict[str, tuple[bytes, dict[str, Any], str]] = {}
    for record in active_records:
        path = vault / record.relative_path
        raw = path.read_bytes()
        if _sha256_bytes(raw) != record.sha256:
            raise RuntimeError(f"baseline hash mismatch before planning: {record.relative_path}")
        try:
            text, _ = _decode_markdown(raw)
            meta, body, issues = parse_note(text)
        except (UnicodeDecodeError, ValueError) as exc:
            blocked.append({"relative_path": record.relative_path, "reason": str(exc)})
            continue
        if issues:
            blocked.append({"relative_path": record.relative_path, "reason": str(issues)})
            continue
        parsed[record.relative_path] = (raw, meta, body)
        existing_id = str(meta.get("note_id") or "").strip()
        catalog_entry = catalog.get(record.relative_path)
        if existing_id:
            if existing_id in used_ids:
                raise RuntimeError(f"duplicate active note_id: {existing_id}")
            used_ids.add(existing_id)
            if catalog_entry and str(catalog_entry.note_id) != existing_id:
                raise RuntimeError(
                    f"catalog/file identity conflict at {record.relative_path}: "
                    f"{catalog_entry.note_id} != {existing_id}"
                )
            stored_key = str(meta.get("document_key") or "").strip()
            if stored_key and catalog_entry and stored_key != str(catalog_entry.document_key):
                raise RuntimeError(f"catalog/file document_key conflict at {record.relative_path}")
            document_key = (
                stored_key
                or (str(catalog_entry.document_key) if catalog_entry else "")
                or f"legacy-import:v1:{existing_id}"
            )
            identity_records.append(NoteIdentity(existing_id, document_key))
            identity_sources["existing_file"] += 1

    broken_targets_before = _issue_targets(before_audit, "broken_link")
    broken_stems = {
        Path(target.split("#", 1)[0].split("^", 1)[0]).stem.upper()
        for target in broken_targets_before
    }
    canonical_links: dict[str, str] = {}
    planned_stock_hubs: set[str] = set()
    for ticker in ("AAPL", "SPY", "CPALL", "KBANK", "FTNT", "AOT", "ADVANC", "NVDA", "AMZN"):
        rel = f"30_Knowledge_Base/Stocks/{ticker}/{ticker}.md"
        if (vault / rel).is_file():
            canonical_links[ticker] = str(Path(rel).with_suffix("")).replace("\\", "/")
        elif ticker in {"NVDA", "AMZN"} and ticker in broken_stems:
            canonical_links[ticker] = str(Path(rel).with_suffix("")).replace("\\", "/")
            planned_stock_hubs.add(ticker)

    directory_links = {
        "30_Knowledge_Base/Stocks": "00_Index/Stocks_Hub",
        "30_Knowledge_Base/Macroeconomics": "00_Index/Macro_Hub",
        "30_Knowledge_Base/News": "00_Index/News_Hub",
        "30_Knowledge_Base/YouTube_Summaries": "00_Index/YouTube_Hub",
        "30_Knowledge_Base/NotebookLM_Sources": "00_Index/NotebookLM_Sources_Hub",
        "30_Knowledge_Base/Concepts": "00_Index/Concepts_Hub",
    }

    staged: dict[str, bytes] = {}
    staged_reasons: dict[str, list[str]] = {}
    link_counts: Counter[str] = Counter()
    for relative_path, (raw, meta, body) in parsed.items():
        updates: dict[str, Any] = {}
        reasons: list[str] = []
        schema = meta.get("schema_version")
        if isinstance(schema, int) and schema > 2:
            blocked.append({"relative_path": relative_path, "reason": f"future schema {schema}"})
            continue
        if schema != 2:
            updates["schema_version"] = 2
            metadata_counts["schema_version"] += 1
            reasons.append("schema_version")
        if not meta.get("title"):
            match = H1_RE.search(body)
            updates["title"] = match.group(1).strip() if match else Path(relative_path).stem.replace("_", " ")
            metadata_counts["title"] += 1
            reasons.append("title")
        if not meta.get("entity_type") and not meta.get("type"):
            inferred = _infer_entity_type(relative_path)
            if inferred:
                updates["entity_type"] = inferred
                metadata_counts["entity_type"] += 1
                reasons.append("entity_type")
        if not any(meta.get(key) for key in ("date", "published_date", "analysis_date", "as_of")):
            if meta.get("created"):
                updates["date"] = meta["created"]
                metadata_counts["date_from_created"] += 1
                reasons.append("date_from_created")

        if not meta.get("note_id"):
            catalog_entry = catalog.get(relative_path)
            if catalog_entry:
                note_id = str(catalog_entry.note_id)
                document_key = str(catalog_entry.document_key)
                if note_id in used_ids:
                    raise RuntimeError(f"catalog note_id collision for {relative_path}: {note_id}")
                used_ids.add(note_id)
                identity_sources["catalog_proven"] += 1
            else:
                note_id = _new_note_id(used_ids)
                document_key = f"legacy-import:v1:{_sha256_bytes(raw)}"
                identity_sources["explicit_content_import"] += 1
            updates["note_id"] = note_id
            identity_records.append(NoteIdentity(note_id, document_key))
            metadata_counts["note_id"] += 1
            reasons.append("note_id")

        patched, per_file_links = _patch_markdown(raw, updates, canonical_links, directory_links)
        link_counts.update(per_file_links)
        if per_file_links:
            reasons.append("canonical_links")
        if patched != raw:
            staged[relative_path] = patched
            staged_reasons[relative_path] = reasons

    navigation_files = _make_navigation_files(
        vault,
        active_records,
        used_ids,
        identity_records,
        today,
        planned_stock_hubs,
    )
    for rel, value in navigation_files.items():
        staged[rel] = value
        staged_reasons[rel] = ["navigation_hub"]

    stock_hub_identities: dict[str, NoteIdentity] = {}
    for ticker in sorted(planned_stock_hubs):
        document_key = build_document_key("stock_hub", ticker, role="hub")
        identity = NoteIdentity(_new_note_id(used_ids), document_key)
        identity_records.append(identity)
        stock_hub_identities[ticker] = identity

    key_to_id: dict[str, str] = {}
    id_to_key: dict[str, str] = {}
    for identity in identity_records:
        if identity.document_key in key_to_id and key_to_id[identity.document_key] != identity.note_id:
            raise RuntimeError(f"duplicate document_key in repair plan: {identity.document_key}")
        if identity.note_id in id_to_key and id_to_key[identity.note_id] != identity.document_key:
            raise RuntimeError(f"duplicate note_id in repair plan: {identity.note_id}")
        key_to_id[identity.document_key] = identity.note_id
        id_to_key[identity.note_id] = identity.document_key

    stage_root = run_dir / "staged"
    changes: list[dict[str, Any]] = []
    for rel, value in sorted(staged.items()):
        target = vault / rel
        before_hash = _sha256_file(target) if target.is_file() else None
        expected = record_by_path.get(rel)
        if expected and before_hash != expected.sha256:
            raise RuntimeError(f"target changed while repair was planned: {rel}")
        if expected is None and target.exists():
            raise RuntimeError(f"planned create target already exists: {rel}")
        stage_path = stage_root / rel
        stage_path.parent.mkdir(parents=True, exist_ok=True)
        stage_path.write_bytes(value)
        changes.append(
            {
                "relative_path": rel,
                "action": "update" if before_hash else "create",
                "before_sha256": before_hash,
                "after_sha256": _sha256_bytes(value),
                "reasons": staged_reasons[rel],
                "status": "planned",
            }
        )

    plan = {
        "run_id": run_id,
        "vault_root": str(vault),
        "baseline_fingerprint": before_fingerprint,
        "baseline_file_count": before_file_count,
        "snapshot": snapshot,
        "metadata_updates": dict(metadata_counts),
        "identity_sources": dict(identity_sources),
        "identity_records": len(identity_records),
        "link_rewrites": dict(link_counts),
        "planned_stock_hubs": sorted(planned_stock_hubs),
        "blocked": blocked,
        "changes": changes,
    }
    _write_json(run_dir / "repair-plan.json", plan)
    _write_json(run_dir / "repair-journal.json", {"run_id": run_id, "changes": changes})

    if _tree_fingerprint(vault) != (before_fingerprint, before_file_count):
        raise RuntimeError("vault changed after planning and before apply")
    for change in changes:
        target = vault / change["relative_path"]
        actual = _sha256_file(target) if target.is_file() else None
        if actual != change["before_sha256"]:
            raise RuntimeError(f"pre-hash guard failed: {change['relative_path']}")
    for ticker in planned_stock_hubs:
        if (vault / f"30_Knowledge_Base/Stocks/{ticker}/{ticker}.md").exists():
            raise RuntimeError(f"stock hub appeared concurrently: {ticker}")

    identity_store = DurableIdentityStore(root=vault)
    identity_result = identity_store.bulk_import_note_identities(identity_records)
    for change in changes:
        target = vault / change["relative_path"]
        actual = _sha256_file(target) if target.is_file() else None
        if actual != change["before_sha256"]:
            raise RuntimeError(f"pre-hash guard failed during apply: {change['relative_path']}")
        _atomic_write_bytes(target, (stage_root / change["relative_path"]).read_bytes())
        if _sha256_file(target) != change["after_sha256"]:
            raise RuntimeError(f"post-hash verification failed: {change['relative_path']}")
        change["status"] = "applied"
        _write_json(run_dir / "repair-journal.json", {"run_id": run_id, "changes": changes})

    artifact_writer = ArtifactWriter(
        vault_paths=VaultPaths(vault, layout_version=2),
        identity_store=identity_store,
    )
    artifact_results: dict[str, Any] = {}
    for ticker, identity in sorted(stock_hub_identities.items()):
        result = artifact_writer.write_note(
            {
                "schema_version": 2,
                "note_id": identity.note_id,
                "document_key": identity.document_key,
                "entity_type": "stock_hub",
                "title": ticker,
                "ticker": ticker,
                "date": today,
                "status": "stub",
                "source_verification_status": "not_reviewed",
                "content_verification_status": "not_reviewed",
            },
            f"# {ticker}\n\nThis ticker hub is ready for verified research notes.\n",
        )
        artifact_results[ticker] = {
            "note_id": result.note_id,
            "revision_id": result.revision_id,
            "revision": result.revision,
            "content_hash": result.content_hash,
            "primary_file": str(result.primary_file),
            "manifest": str(result.manifest_path),
        }

    writable_catalog = SqliteNoteCatalogAdapter(vault_root=vault)
    catalog_sync = writable_catalog.sync_from_vault(force=True)
    outbox_processed = writable_catalog.process_outbox()

    after_audit = scan_vault(vault)
    write_audit_report(after_audit, run_dir / "after")
    after_fingerprint, after_file_count = _tree_fingerprint(vault)

    active_ids: dict[str, str] = {}
    schema_versions: Counter[str] = Counter()
    for record in after_audit.inventory:
        if record.is_excluded or record.extension != ".md":
            continue
        note_id = str(record.properties.get("note_id") or "").strip()
        if note_id:
            if note_id in active_ids:
                raise RuntimeError(
                    f"post-repair duplicate note_id {note_id}: "
                    f"{active_ids[note_id]} and {record.relative_path}"
                )
            active_ids[note_id] = record.relative_path
        schema_versions[str(record.properties.get("schema_version"))] += 1
    unintended: list[str] = []
    changed_paths = {change["relative_path"] for change in changes}
    changed_paths.update(
        f"30_Knowledge_Base/Stocks/{ticker}/{ticker}.md" for ticker in planned_stock_hubs
    )
    for record in active_records:
        if record.relative_path in changed_paths:
            continue
        path = vault / record.relative_path
        if not path.is_file() or _sha256_file(path) != record.sha256:
            unintended.append(record.relative_path)
    if unintended:
        raise RuntimeError(f"unplanned active Markdown changes detected: {unintended[:20]}")
    if after_audit.stats["broken_links"] > before_audit.stats["broken_links"]:
        raise RuntimeError("broken-link count increased after repair")
    if after_audit.stats["ambiguous_links"] > before_audit.stats["ambiguous_links"]:
        raise RuntimeError("ambiguous-link count increased after repair")
    if any(version != "2" for version in schema_versions):
        raise RuntimeError(f"active notes remain outside schema v2: {dict(schema_versions)}")

    remaining = {
        "broken_links": _issue_targets(after_audit, "broken_link").most_common(),
        "ambiguous_links": _issue_targets(after_audit, "ambiguous_link").most_common(),
        "missing_metadata": [
            {
                "relative_path": issue.relative_path,
                "details": issue.details,
            }
            for issue in after_audit.issues
            if issue.issue_type == "missing_metadata"
        ],
        "blocked": blocked,
    }
    _write_json(run_dir / "remaining-issues.json", remaining)
    result = {
        "run_id": run_id,
        "status": "PASS",
        "vault_root": str(vault),
        "before": {
            "fingerprint": before_fingerprint,
            "file_count": before_file_count,
            "audit": before_audit.stats,
        },
        "after": {
            "fingerprint": after_fingerprint,
            "file_count": after_file_count,
            "audit": after_audit.stats,
            "schema_versions": dict(schema_versions),
            "unique_active_note_ids": len(active_ids),
        },
        "snapshot": snapshot,
        "direct_file_changes": len(changes),
        "artifact_hubs": artifact_results,
        "identity_import": {
            "imported": identity_result["imported"],
            "reused": identity_result["reused"],
            "total": identity_result["total"],
        },
        "catalog_sync": catalog_sync,
        "catalog_outbox_processed": outbox_processed,
        "unintended_active_changes": unintended,
        "evidence_dir": str(run_dir),
    }
    _write_json(run_dir / "result.json", result)
    report = [
        "# Vault V2 R3 Live File Repair",
        "",
        f"- Status: **{result['status']}**",
        f"- Run: `{run_id}`",
        f"- Vault: `{vault}`",
        f"- Snapshot restore proof: **{snapshot['restore_proof']}**",
        f"- Snapshot SHA-256: `{snapshot['sha256']}`",
        f"- Direct Markdown creates/updates: **{len(changes):,}**",
        f"- Managed stock hubs created: **{len(artifact_results)}**",
        f"- Identity mappings imported/reused: **{identity_result['imported']:,}/{identity_result['reused']:,}**",
        "",
        "## Audit delta",
        "",
        "| Finding | Before | After |",
        "|---|---:|---:|",
    ]
    for key in ("missing_metadata", "broken_links", "ambiguous_links", "duplicate_filenames", "parse_errors"):
        report.append(f"| {key} | {before_audit.stats[key]:,} | {after_audit.stats[key]:,} |")
    report.extend(
        [
            "",
            "## Verification",
            "",
            f"- Active schema versions: `{dict(schema_versions)}`",
            f"- Unique active note IDs: `{len(active_ids):,}`",
            f"- Unplanned active Markdown changes: `{len(unintended)}`",
            f"- Catalog sync: `{catalog_sync}`",
            "- Remaining low-confidence findings are preserved in `remaining-issues.json`.",
            "",
        ]
    )
    (run_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    _write_json(args.output_root.resolve() / "latest-live-repair.json", result)
    print(json.dumps(result, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
