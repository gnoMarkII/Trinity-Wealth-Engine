"""Apply R6 multi-app configuration, templates, and portable media fallbacks."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.archivist.maintenance_guard import MaintenanceLeaseError, assert_write_allowed, load_maintenance_lease  # noqa: E402


TEMPLATES: dict[str, str] = {
    "Inbox_Capture_Template.md": """---
schema_version: 2
capture_status: pending_normalization
search_scope: excluded
title: Untitled capture
entity_type: concept
date_status: unknown
source_verification_status: not_reviewed
content_verification_status: not_reviewed
---

# Untitled capture

## Source / context

- Source URL:
- Captured at:
- Next normalization action:

## Notes
""",
    "Daily_Note_Template.md": """---
schema_version: 2
capture_status: pending_normalization
search_scope: excluded
document_role: daily_log
title: Daily note
date_status: unknown
source_verification_status: not_reviewed
content_verification_status: not_reviewed
---

# Daily note

## Capture
""",
    "Research_Note_Template.md": """---
schema_version: 2
capture_status: pending_normalization
search_scope: excluded
entity_type: concept
title: Research note
date_status: unknown
source_verification_status: not_reviewed
content_verification_status: not_reviewed
---

# Research note

## Question

## Evidence

## Working notes
""",
    "Source_Note_Template.md": """---
schema_version: 2
capture_status: pending_normalization
search_scope: excluded
entity_type: concept
title: Source note
source_url:
date_status: unknown
source_verification_status: not_reviewed
content_verification_status: not_reviewed
---

# Source note

## Extracted facts

## Source context
""",
    "Stock_Analysis_Template.md": """---
schema_version: 2
capture_status: pending_normalization
search_scope: excluded
entity_type: equity_analysis
title: Stock analysis
ticker:
date_status: unknown
source_verification_status: not_reviewed
content_verification_status: not_reviewed
---

# Stock analysis

## Thesis

## Evidence

## Risks
""",
    "Macro_Note_Template.md": """---
schema_version: 2
capture_status: pending_normalization
search_scope: excluded
entity_type: macro_snapshot
title: Macro note
date_status: unknown
source_verification_status: not_reviewed
content_verification_status: not_reviewed
---

# Macro note

## Indicators

## Interpretation
""",
    "Book_Note_Template.md": """---
schema_version: 2
capture_status: pending_normalization
search_scope: excluded
entity_type: book_note
title: Book note
date_status: unknown
source_verification_status: not_reviewed
content_verification_status: not_reviewed
author:
genre:
date_read:
rating:
tags:
  - book
  - investment_philosophy
---

# Book note

> Author:  | Read: 

## Core ideas

## Investment principles

## Mental models

## Application
""",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".r6tmp", dir=str(path.parent))
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _assert_lease(vault: Path, owner: str) -> None:
    lease = load_maintenance_lease(vault)
    if lease is None or not lease.is_active() or lease.owner != owner:
        raise MaintenanceLeaseError(f"R6 config apply requires active lease owned by {owner!r}")
    assert_write_allowed(vault / ".obsidian" / "app.json", owner=owner)


def _portable_iframe_fallbacks(vault: Path, journal: list[dict[str, Any]]) -> dict[str, Any]:
    touched: list[dict[str, Any]] = []
    iframe_count = 0
    for path in sorted(vault.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        if not re.search(r"<iframe\b", text, re.IGNORECASE):
            continue
        iframe_count += 1
        source_match = re.search(r"(?m)^source_url:\s*['\"]?(https?://[^'\"\s]+)", text)
        url = source_match.group(1) if source_match else ""
        if not url:
            iframe_match = re.search(r"<iframe\b[^>]*\bsrc=[\"']([^\"']+)[\"']", text, re.IGNORECASE)
            embedded = iframe_match.group(1).strip() if iframe_match else ""
            youtube_id = re.search(r"youtube\.com/embed/([A-Za-z0-9_-]+)", embedded)
            if youtube_id:
                url = f"https://www.youtube.com/watch?v={youtube_id.group(1)}"
            else:
                touched.append({"path": path.relative_to(vault).as_posix(), "status": "BLOCKED", "reason": "missing_source_url"})
                continue

        # A prior remediation pass could have appended a second source_url
        # after an invalid placeholder (for example a bare YouTube ID).  Keep
        # exactly one canonical URL so every app sees the same metadata.
        if text.startswith("---") and "\n---" in text[3:]:
            marker = text.find("\n---", 3)
            frontmatter = text[3:marker]
            source_lines = re.findall(r"(?m)^source_url:\s*.*$", frontmatter)
            if source_lines and (len(source_lines) > 1 or not re.search(r"(?m)^source_url:\s*['\"]?https?://", frontmatter)):
                kept_lines = [
                    line for line in frontmatter.splitlines(keepends=True)
                    if not re.match(r"^source_url:\s*", line)
                ]
                newline = "" if not kept_lines or kept_lines[-1].endswith("\n") else "\n"
                kept_lines.append(newline + f"source_url: {url}\n")
                normalized = text[:3] + "".join(kept_lines) + text[marker:]
                if normalized != text:
                    before = _sha256(path)
                    _atomic_write(path, normalized)
                    after = _sha256(path)
                    text = normalized
                    rel = path.relative_to(vault).as_posix()
                    journal.append({"operation": "iframe_source_url_normalize", "path": rel, "before_sha256": before, "after_sha256": after})
                    touched.append({"path": rel, "status": "PASS", "changed": True, "reason": "deduplicated_source_url", "before_sha256": before, "after_sha256": after})
        if re.search(rf"\[[^\]]+\]\([^)]*{re.escape(url)}[^)]*\)", text):
            if not any(item.get("path") == path.relative_to(vault).as_posix() for item in touched):
                touched.append({"path": path.relative_to(vault).as_posix(), "status": "PASS", "changed": False})
            continue
        lines = text.splitlines(keepends=True)
        inserted = False
        for index, line in enumerate(lines):
            if re.search(r"<iframe\b", line, re.IGNORECASE):
                newline = "\n" if line.endswith("\n") else ""
                lines.insert(index, f"- [Open source video]({url}){newline}")
                inserted = True
                break
        if not inserted:
            touched.append({"path": path.relative_to(vault).as_posix(), "status": "BLOCKED", "reason": "iframe_not_located"})
            continue
        before = _sha256(path)
        _atomic_write(path, "".join(lines))
        after = _sha256(path)
        rel = path.relative_to(vault).as_posix()
        journal.append({"operation": "iframe_fallback", "path": rel, "before_sha256": before, "after_sha256": after})
        touched.append({"path": rel, "status": "PASS", "changed": True, "before_sha256": before, "after_sha256": after})
    blocked = [item for item in touched if item["status"] != "PASS"]
    return {"status": "PASS" if not blocked else "BLOCKED", "total_iframe_notes": iframe_count, "changed": sum(bool(item.get("changed")) for item in touched), "blocked": blocked, "records": touched}


def apply(vault: Path, run_dir: Path, *, owner: str) -> dict[str, Any]:
    vault = vault.resolve()
    run_dir = run_dir.resolve()
    _assert_lease(vault, owner)
    journal: list[dict[str, Any]] = []

    app_path = vault / ".obsidian" / "app.json"
    app = json.loads(app_path.read_text(encoding="utf-8"))
    before = _sha256(app_path)
    app.update({"useMarkdownLinks": True, "newLinkFormat": "relative", "alwaysUpdateLinks": True, "attachmentFolderPath": "90_Attachments"})
    _atomic_write(app_path, json.dumps(app, ensure_ascii=False, indent=2) + "\n")
    journal.append({"operation": "obsidian_app_config", "path": ".obsidian/app.json", "before_sha256": before, "after_sha256": _sha256(app_path)})

    configs = {
        ".obsidian/templates.json": {"folder": "99_Templates"},
        ".obsidian/daily-notes.json": {"folder": "01_Daily_Logs", "format": "YYYY-MM-DD", "template": "99_Templates/Daily_Note_Template.md"},
    }
    for rel, payload in configs.items():
        path = vault / rel
        old_hash = _sha256(path) if path.is_file() else None
        _atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        journal.append({"operation": "obsidian_plugin_config", "path": rel, "before_sha256": old_hash, "after_sha256": _sha256(path)})

    template_records: list[dict[str, Any]] = []
    for name, content in TEMPLATES.items():
        path = vault / "99_Templates" / name
        old_hash = _sha256(path) if path.is_file() else None
        _atomic_write(path, content.rstrip() + "\n")
        after = _sha256(path)
        journal.append({"operation": "template", "path": path.relative_to(vault).as_posix(), "before_sha256": old_hash, "after_sha256": after})
        template_records.append({"path": path.relative_to(vault).as_posix(), "status": "PASS", "before_sha256": old_hash, "after_sha256": after})

    iframe_result = _portable_iframe_fallbacks(vault, journal)
    app_view_disposition = {
        "status": "PASS",
        "allowlisted_adapter_paths": ["20_Portfolio_Management/Portfolio_Dashboard.md", "00_Index/App_Views/Obsidian/**"],
        "reason": "Existing Portfolio Dashboard is explicitly search_scope=excluded and remains a legacy adapter with canonical data outside the dynamic view.",
    }
    _write_json(run_dir / "template-contract-result.json", {"status": "PASS", "template_count": len(template_records), "templates": template_records, "configs": list(configs)})
    _write_json(run_dir / "iframe-fallback-report.json", iframe_result)
    _write_json(run_dir / "app-view-dispositions.json", app_view_disposition)
    journal_path = run_dir / "mutation-journal-r6-config.jsonl"
    with journal_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in journal:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    result = {
        "status": "PASS" if iframe_result["status"] == "PASS" else "BLOCKED",
        "config_count": len(configs),
        "template_count": len(template_records),
        "iframe_notes": iframe_result["total_iframe_notes"],
        "iframe_fallbacks_added": iframe_result["changed"],
        "journal_count": len(journal),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "owner": owner,
    }
    _write_json(run_dir / "config-apply-result.json", result)
    print(json.dumps(result, ensure_ascii=False))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", type=Path, default=Path("memories"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--owner", default="codex-vault-r6")
    args = parser.parse_args()
    result = apply(args.vault, args.run_dir, owner=args.owner)
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
