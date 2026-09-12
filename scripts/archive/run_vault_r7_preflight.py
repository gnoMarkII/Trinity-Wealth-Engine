"""Read-only R7 vault preflight and deterministic tree fingerprint.

The command intentionally does not acquire a maintenance lease or mutate the
vault.  It is used before and after a repair so the evidence records both the
profile debt and the exact input tree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from tools.archivist.metadata import parse_note, validate_capture_note, validate_note


CORE_FIELDS = ("schema_version", "note_id", "document_key", "entity_type", "title")
CAPTURE_FIELDS = ("capture_status", "search_scope", "captured_at", "capture_source")
EXCLUDED_DIRS = {".git", ".obsidian"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_fingerprint(root: Path, files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _iter_markdown(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*.md"):
        relative_parts = set(path.relative_to(root).parts)
        if relative_parts & EXCLUDED_DIRS:
            continue
        files.append(path)
    return files


def _has_wikilink(text: str) -> bool:
    return "[[" in text or "![[" in text or "obsidian://" in text


def collect(root: Path) -> dict[str, Any]:
    files = _iter_markdown(root)
    profile_counts: Counter[str] = Counter()
    entity_counts: Counter[str] = Counter()
    missing_counts: Counter[str] = Counter()
    issue_counts: Counter[str] = Counter()
    note_ids: dict[str, list[str]] = {}
    document_keys: dict[str, list[str]] = {}
    wikilinks: list[str] = []
    malformed: list[str] = []
    missing_samples: dict[str, list[str]] = {field: [] for field in CORE_FIELDS}
    recent: list[dict[str, Any]] = []

    for path in files:
        relative = path.relative_to(root).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            issue_counts["unreadable"] += 1
            malformed.append(f"{relative}: {exc}")
            continue

        metadata, body, parse_issues = parse_note(text)
        for issue in parse_issues:
            issue_counts[str(issue.get("code") or "parse_error")] += 1
        if parse_issues:
            malformed.append(relative)

        raw_entity = str(metadata.get("entity_type") or "").strip().lower()
        entity_counts[raw_entity or "<missing>"] += 1
        if _has_wikilink(text):
            wikilinks.append(relative)

        path_parts = set(Path(relative).parts)
        archive_like = ".backups" in path_parts or "Revisions" in path_parts or "40_Archive" in path_parts
        template_like = "99_Templates" in path_parts
        capture_like = relative == "00_Inbox/index.md" or relative.startswith("00_Inbox/")
        if archive_like:
            profile_counts["archive_or_backup"] += 1
        elif template_like:
            profile_counts["template"] += 1
            ok, issues = validate_capture_note(metadata) if raw_entity == "capture" else (True, [])
            if not ok:
                for issue in issues:
                    issue_counts[f"template:{issue.get('code', 'invalid')}"] += 1
        elif capture_like or raw_entity == "capture":
            profile_counts["capture"] += 1
            ok, issues = validate_capture_note(metadata)
            if not ok:
                for issue in issues:
                    issue_counts[f"capture:{issue.get('code', 'invalid')}"] += 1
        elif raw_entity == "navigation" or metadata.get("document_role") == "navigation":
            profile_counts["navigation"] += 1
        elif metadata.get("document_role") in {"portfolio_state", "holdings", "watchlist", "goals", "journal", "portfolio_item"} or raw_entity in {"portfolio_state", "holding", "watchlist_item", "goal"}:
            profile_counts["portfolio_state"] += 1
        else:
            profile_counts["published"] += 1
            for field in CORE_FIELDS:
                value = metadata.get(field)
                if value is None or (isinstance(value, str) and not value.strip()):
                    missing_counts[field] += 1
                    if len(missing_samples[field]) < 25:
                        missing_samples[field].append(relative)
            if metadata.get("schema_version") != 2:
                missing_counts["schema_version_v2"] += 1
            _, issues = validate_note(metadata, mode="strict")
            for issue in issues:
                issue_counts[str(issue.get("code") or "validation_error")] += 1

        if metadata.get("note_id"):
            note_ids.setdefault(str(metadata["note_id"]), []).append(relative)
        if metadata.get("document_key"):
            document_keys.setdefault(str(metadata["document_key"]), []).append(relative)

        stat = path.stat()
        recent.append({"path": relative, "mtime_utc": stat.st_mtime, "size": stat.st_size})

    duplicate_note_ids = {key: value for key, value in note_ids.items() if len(value) > 1}
    duplicate_document_keys = {key: value for key, value in document_keys.items() if len(value) > 1}
    recent.sort(key=lambda item: (item["mtime_utc"], item["path"]), reverse=True)

    return {
        "schema": "vault-r7-preflight-v1",
        "vault_root": str(root),
        "environment_vault": os.getenv("OBSIDIAN_VAULT_PATH"),
        "markdown_count": len(files),
        "tree_fingerprint": tree_fingerprint(root, files),
        "profiles": dict(profile_counts),
        "entity_types": dict(entity_counts),
        "missing": dict(missing_counts),
        "missing_samples": missing_samples,
        "issues": dict(issue_counts),
        "duplicate_note_ids": duplicate_note_ids,
        "duplicate_document_keys": duplicate_document_keys,
        "wikilink_count": len(wikilinks),
        "wikilink_samples": wikilinks[:25],
        "malformed_samples": malformed[:25],
        "recent_files": recent[:25],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", type=Path, default=Path(os.getenv("OBSIDIAN_VAULT_PATH", "memories")))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.vault.resolve()
    report = collect(root)
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8", newline="\n")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
