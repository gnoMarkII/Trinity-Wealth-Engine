"""Vault Audit Engine for Obsidian Vault V2.

Performs deterministic, read-only scanning of an Obsidian Vault.
Computes SHA-256 hashes, parses YAML frontmatter/properties, inspects wikilinks,
detects duplicate titles, broken links, missing required properties, and orphaned sidecars.
Zero mutation: never modifies, moves, or deletes any file in the scanned vault.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

import frontmatter as fm

from tools.archivist.portable_links import (
    iter_internal_markdown_links,
    resolve_vault_target_detailed,
)

log = logging.getLogger(__name__)

# Standard wikilink and markdown link patterns
_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

# Folders to exclude from searchable knowledge base (marked as excluded in inventory)
_EXCLUDED_DIR_PREFIXES = (
    ".",  # .obsidian, .trash, .sync_history, .chroma_index, .pre_migration_backup_*
)
_EXCLUDED_DIR_NAMES = {
    ".obsidian",
    ".trash",
    ".sync_history",
    ".chroma_index",
    ".system",
    "40_Archive",
    "99_Templates",
}

# T14-C: Broken-link suppression — patterns that are NOT real wikilinks.
# Matches Obsidian Dataview expressions, template variables, and JS-like expressions.
_BROKEN_LINK_SUPPRESS_RE = re.compile(
    r"[\[\](){}]"
    r"|^panels\["
    r"|^dv\."
    r"|^this\."
    r"|dataview"
    r"|\$\{"
    r"|\\n",
    re.IGNORECASE,
)

# T14-D: Portfolio folder stems that are intentionally duplicated across sub-portfolios.
# Stem (lowercased) in this set won't trigger duplicate_filename warnings.
_PORTFOLIO_DUPLICATE_ALLOWLIST: set[str] = {
    "portfolio_holdings",
    "trading_journal",
    "watchlist",
    "portfolio_summary",
    "performance_report",
    # Directory index notes intentionally repeat at different hierarchy levels.
    # Links are path-qualified, so these do not create ambiguous note targets.
    "index",
}

# T14-C v2: System-generated or index files that legitimately reference many notes
# including those in excluded folders — skip broken/ambiguous link checks for these only if explicitly requested.
_LINK_CHECK_SKIP_FILES: set[str] = set()


@dataclass
class FileAuditRecord:
    relative_path: str
    size_bytes: int
    sha256: str
    extension: str
    is_excluded: bool
    exclusion_reason: Optional[str] = None
    has_frontmatter: bool = False
    properties: dict[str, Any] = field(default_factory=dict)
    links: list[str] = field(default_factory=list)
    artifact_candidates: list[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class AuditIssue:
    issue_type: str
    severity: str  # "error" | "warning" | "info"
    relative_path: str
    details: str


@dataclass
class VaultAuditResult:
    vault_root: str
    scanned_at: str
    total_files: int
    active_files_count: int
    excluded_files_count: int
    scanned_by_extension: dict[str, int]
    inventory: list[FileAuditRecord]
    issues: list[AuditIssue]
    stats: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "vault_root": self.vault_root,
            "scanned_at": self.scanned_at,
            "total_files": self.total_files,
            "active_files_count": self.active_files_count,
            "excluded_files_count": self.excluded_files_count,
            "scanned_by_extension": self.scanned_by_extension,
            "inventory": [asdict(r) for r in self.inventory],
            "issues": [asdict(i) for i in self.issues],
            "stats": self.stats,
        }


def _compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def _is_path_excluded(rel_path: Path) -> tuple[bool, Optional[str]]:
    parts = rel_path.parts
    for part in parts:
        if part in _EXCLUDED_DIR_NAMES:
            return True, f"directory_{part}"
        if any(part.startswith(prefix) for prefix in _EXCLUDED_DIR_PREFIXES):
            return True, f"hidden_or_backup_{part}"
    if rel_path.name.endswith(".quality.json"):
        return False, None
    if rel_path.name.startswith("."):
        return True, "hidden_file"
    return False, None


def _extract_links_from_text(text: str) -> list[str]:
    links: list[str] = []
    # 1. Wikilinks [[target|alias]] -> target
    for match in _WIKILINK_RE.finditer(text):
        target = match.group(1).strip()
        if target:
            links.append(target)
    # 2. Portable Markdown links. Use the same parser as GraphRAG so URI
    # decoding, source-relative paths, and balanced parentheses stay aligned.
    links.extend(link.destination for link in iter_internal_markdown_links(text))
    return links


def scan_vault(root: Path | str) -> VaultAuditResult:
    """Read-only scan of the vault directory.
    
    Traverses all files, computes SHA256, parses metadata/links for markdown,
    records any errors without modifying or deleting anything.
    """
    root_path = Path(root).resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"Vault root does not exist: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"Vault root is not a directory: {root_path}")

    records: list[FileAuditRecord] = []
    issues: list[AuditIssue] = []
    ext_counts: dict[str, int] = {}
    active_count = 0
    excluded_count = 0

    # Collect and sort all files deterministically
    all_files: list[Path] = []
    for dirpath, _, filenames in os.walk(root_path):
        dp = Path(dirpath)
        for fn in filenames:
            all_files.append(dp / fn)
    all_files.sort(key=lambda p: p.relative_to(root_path).as_posix())

    # Map of lowercase stem to list of relative paths for duplicate detection
    stem_to_paths: dict[str, list[str]] = {}
    # Set of all relative paths (as posix) and stems for link resolution checks
    all_rel_paths_posix: set[str] = set()
    all_stems: set[str] = set()

    for file_path in all_files:
        rel = file_path.relative_to(root_path)
        rel_posix = rel.as_posix()
        ext = file_path.suffix.lower()
        ext_counts[ext] = ext_counts.get(ext, 0) + 1

        all_rel_paths_posix.add(rel_posix)
        all_stems.add(file_path.stem)

        is_excluded, exclusion_reason = _is_path_excluded(rel)
        if is_excluded:
            excluded_count += 1
        else:
            active_count += 1
            # Duplicate note detection is about Markdown note identity. JSON
            # companions deliberately share a stem with their Markdown source
            # and must not create duplicate-note warnings.
            if ext == ".md":
                stem_key = file_path.stem.lower()
                stem_to_paths.setdefault(stem_key, []).append(rel_posix)

        size_bytes = 0
        sha256_hex = ""
        error_msg: Optional[str] = None
        has_fm = False
        props: dict[str, Any] = {}
        links: list[str] = []
        artifact_candidates: list[str] = []

        try:
            size_bytes = file_path.stat().st_size
            sha256_hex = _compute_sha256(file_path)
        except Exception as e:
            error_msg = f"Failed to read file: {e}"
            issues.append(
                AuditIssue(
                    issue_type="read_error",
                    severity="error",
                    relative_path=rel_posix,
                    details=error_msg,
                )
            )

        if not error_msg and ext == ".md":
            try:
                raw_bytes = file_path.read_bytes()
                try:
                    text_content = raw_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    text_content = raw_bytes.decode("latin-1")
                    issues.append(
                        AuditIssue(
                            issue_type="encoding_warning",
                            severity="warning",
                            relative_path=rel_posix,
                            details="File not valid UTF-8, decoded with latin-1",
                        )
                    )

                # Parse frontmatter
                try:
                    parsed = fm.loads(text_content)
                    if parsed.metadata:
                        has_fm = True
                        props = dict(parsed.metadata)
                    body = parsed.content
                except Exception as fe:
                    issues.append(
                        AuditIssue(
                            issue_type="parse_error",
                            severity="error",
                            relative_path=rel_posix,
                            details=f"Malformed YAML frontmatter: {fe}",
                        )
                    )
                    body = text_content

                # Extract links
                links = _extract_links_from_text(body)

                # Check companion sidecar candidate (.json with same stem)
                companion_json = file_path.with_suffix(".json")
                if companion_json.exists():
                    artifact_candidates.append(companion_json.relative_to(root_path).as_posix())

                # Check companion canvas
                companion_canvas = file_path.with_suffix(".canvas")
                if companion_canvas.exists():
                    artifact_candidates.append(companion_canvas.relative_to(root_path).as_posix())

                # Check quality companion (.quality.json)
                companion_quality = file_path.parent / f"{file_path.name}.quality.json"
                if companion_quality.exists():
                    artifact_candidates.append(companion_quality.relative_to(root_path).as_posix())

            except Exception as e:
                error_msg = f"Error processing markdown: {e}"
                issues.append(
                    AuditIssue(
                        issue_type="read_error",
                        severity="error",
                        relative_path=rel_posix,
                        details=error_msg,
                    )
                )

        elif not error_msg and ext == ".json":
            try:
                content = file_path.read_text(encoding="utf-8")
                json.loads(content)
            except Exception as je:
                issues.append(
                    AuditIssue(
                        issue_type="parse_error",
                        severity="error",
                        relative_path=rel_posix,
                        details=f"Malformed JSON: {je}",
                    )
                )

        record = FileAuditRecord(
            relative_path=rel_posix,
            size_bytes=size_bytes,
            sha256=sha256_hex,
            extension=ext,
            is_excluded=is_excluded,
            exclusion_reason=exclusion_reason,
            has_frontmatter=has_fm,
            properties=props,
            links=links,
            artifact_candidates=artifact_candidates,
            error=error_msg,
        )
        records.append(record)

    # Secondary analysis for active (non-excluded) markdown files
    for r in records:
        if r.is_excluded:
            continue

        # 1. Check duplicate filenames
        if r.extension == ".md":
            stem_key = Path(r.relative_path).stem.lower()
            paths_with_stem = stem_to_paths.get(stem_key, [])
            if len(paths_with_stem) > 1 and stem_key not in _PORTFOLIO_DUPLICATE_ALLOWLIST:
                # Add duplicate warning once per path
                issues.append(
                    AuditIssue(
                        issue_type="duplicate_filename",
                        severity="warning",
                        relative_path=r.relative_path,
                        details=f"Stem '{stem_key}' appears in multiple active files: {paths_with_stem}",
                    )
                )

            # 2. Check missing required properties
            p = r.properties
            missing_props = []
            if "schema_version" not in p:
                missing_props.append("schema_version")
            if "note_id" not in p:
                missing_props.append("note_id")
            if "title" not in p:
                missing_props.append("title")
            if "entity_type" not in p and "type" not in p:
                missing_props.append("entity_type")
            if (
                "date" not in p
                and "published_date" not in p
                and "analysis_date" not in p
                and "as_of" not in p
                and p.get("date_status") not in {"unknown", "not_applicable"}
            ):
                missing_props.append("date")

            if missing_props:
                issues.append(
                    AuditIssue(
                        issue_type="missing_metadata",
                        severity="info",
                        relative_path=r.relative_path,
                        details=f"Missing recommended properties: {', '.join(missing_props)}",
                    )
                )

            # 3. Check broken and ambiguous links
            # Skip link checks for system-generated index files (T14-C)
            if r.relative_path in _LINK_CHECK_SKIP_FILES:
                continue

            for target in r.links:
                # The shared resolver applies source-relative Markdown
                # semantics and only accepts a unique target.
                clean_target = target.split("#", 1)[0].split("^", 1)[0].strip()
                if not clean_target or _BROKEN_LINK_SUPPRESS_RE.search(clean_target):
                    continue

                resolution = resolve_vault_target_detailed(
                    root_path,
                    clean_target,
                    source=r.relative_path,
                )
                if resolution.status == "missing":
                    issues.append(
                        AuditIssue(
                            issue_type="broken_link",
                            severity="warning",
                            relative_path=r.relative_path,
                            details=f"Target not found: [[{target}]]",
                        )
                    )
                elif resolution.status == "ambiguous":
                    candidates = [
                        candidate.relative_to(root_path).as_posix()
                        for candidate in resolution.candidates
                    ]
                    issues.append(
                        AuditIssue(
                            issue_type="ambiguous_link",
                            severity="warning",
                            relative_path=r.relative_path,
                            details=f"Ambiguous target [[{target}]] matches {len(candidates)} files: {candidates}",
                        )
                    )

    # Aggregate stats
    stats: dict[str, Any] = {
        "read_errors": sum(1 for i in issues if i.issue_type == "read_error"),
        "parse_errors": sum(1 for i in issues if i.issue_type == "parse_error"),
        "duplicate_filenames": sum(1 for i in issues if i.issue_type == "duplicate_filename"),
        "broken_links": sum(1 for i in issues if i.issue_type == "broken_link"),
        "ambiguous_links": sum(1 for i in issues if i.issue_type == "ambiguous_link"),
        "missing_metadata": sum(1 for i in issues if i.issue_type == "missing_metadata"),
        "total_issues": len(issues),
    }

    return VaultAuditResult(
        vault_root=str(root_path),
        scanned_at=datetime.now(timezone.utc).isoformat(),
        total_files=len(records),
        active_files_count=active_count,
        excluded_files_count=excluded_count,
        scanned_by_extension=ext_counts,
        inventory=records,
        issues=issues,
        stats=stats,
    )


def _json_serial(obj: Any) -> Any:
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    return str(obj)


def write_audit_report(result: VaultAuditResult, output_dir: Path | str) -> tuple[Path, Path]:
    """Writes inventory.json and audit.md to output_dir.
    
    Guaranteed to write outside the vault root.
    Returns (inventory_json_path, audit_md_path).
    """
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. inventory.json
    inv_file = out_path / "inventory.json"
    with inv_file.open("w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False, default=_json_serial)

    # 2. audit.md
    audit_file = out_path / "audit.md"
    md_lines: list[str] = [
        "# Vault V2 Audit Report",
        "",
        f"- **Vault Root**: `{result.vault_root}`",
        f"- **Scanned At**: `{result.scanned_at}`",
        f"- **Total Files Scanned**: `{result.total_files}`",
        f"- **Active Knowledge Files**: `{result.active_files_count}`",
        f"- **Excluded Files (System/Trash/Backup)**: `{result.excluded_files_count}`",
        "",
        "## File Breakdown by Extension",
        "",
        "| Extension | Count |",
        "|---|---|",
    ]
    for ext, count in sorted(result.scanned_by_extension.items(), key=lambda x: x[1], reverse=True):
        md_lines.append(f"| `{ext or '(none)'}` | {count} |")

    md_lines.extend([
        "",
        "## Summary of Detected Issues",
        "",
        "| Issue Type | Count |",
        "|---|---|",
    ])
    for itype, icount in result.stats.items():
        if itype != "total_issues":
            md_lines.append(f"| `{itype}` | {icount} |")
    md_lines.append(f"| **Total Issues** | **{result.stats.get('total_issues', 0)}** |")

    # Group issues by severity
    errors = [i for i in result.issues if i.severity == "error"]
    warnings = [i for i in result.issues if i.severity == "warning"]
    infos = [i for i in result.issues if i.severity == "info"]

    if errors:
        md_lines.extend([
            "",
            "## Critical Errors",
            "",
            "| Path | Issue | Details |",
            "|---|---|---|",
        ])
        for e in errors[:50]:
            md_lines.append(f"| `{e.relative_path}` | `{e.issue_type}` | {e.details} |")
        if len(errors) > 50:
            md_lines.append(f"| ... | ... | *(truncated {len(errors) - 50} more errors)* |")

    if warnings:
        md_lines.extend([
            "",
            "## Warnings (Broken/Ambiguous Links & Duplicates)",
            "",
            "| Path | Issue | Details |",
            "|---|---|---|",
        ])
        for w in warnings[:50]:
            md_lines.append(f"| `{w.relative_path}` | `{w.issue_type}` | {w.details} |")
        if len(warnings) > 50:
            md_lines.append(f"| ... | ... | *(truncated {len(warnings) - 50} more warnings)* |")

    if infos:
        md_lines.extend([
            "",
            "## Info (Missing Recommended Metadata)",
            "",
            f"Found {len(infos)} active files missing standard metadata properties (e.g. note_id, entity_type, or date).",
            "See `inventory.json` for the full machine-readable inventory.",
        ])

    with audit_file.open("w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    return inv_file, audit_file
