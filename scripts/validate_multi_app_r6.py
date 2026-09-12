"""Validate the R6 multi-app canonical Markdown contract.

The validator is read-only with respect to the content contract, except for
the explicit generator idempotence smoke which is intentionally run against a
leased rehearsal/live tree.  It writes only evidence files below ``run_dir``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import unicodedata
from datetime import date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.archivist.metadata import parse_note  # noqa: E402


WIKILINK_RE = re.compile(r"(?<!\!)\[\[[^\[\]]+\]\]")
EMBED_RE = re.compile(r"!\[\[[^\[\]]+\]\]")
MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[([^\]]*)\]\(([^)]+)\)")
FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
EXTERNAL_RE = re.compile(r"^(?:[a-z][a-z0-9+.-]*:|//)", re.IGNORECASE)
ABSOLUTE_RE = re.compile(r"^(?:[A-Za-z]:[\\/]|/|\\\\|file://)")
APP_SYNTAX_RE = re.compile(
    r"(?:obsidian://|\bdataview(?:js)?\b|meta-bind|meta-bind-button|<%|<script\b|javascript:)",
    re.IGNORECASE,
)
ACTIVE_SCRIPT_RE = re.compile(r"(?:<script\b|javascript:)", re.IGNORECASE)
ALLOWLISTED_ADAPTERS = (
    "20_Portfolio_Management/Portfolio_Dashboard.md",
    "00_Index/App_Views/Obsidian/",
)
RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_content_sha256(path: Path) -> str:
    """Hash decoded note content with universal-newline normalization.

    Catalog and vector generations hash note text rather than raw filesystem
    bytes. This keeps derived identity stable when a vault moves between
    Windows (CRLF) and Unix-like applications (LF).
    """
    return hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _is_allowlisted(rel: str) -> bool:
    return rel == ALLOWLISTED_ADAPTERS[0] or rel.startswith(ALLOWLISTED_ADAPTERS[1])


def _visible_body(text: str) -> str:
    """Return rendered Markdown text with fenced/inline code removed."""
    lines: list[str] = []
    in_fence = False
    fence_char = ""
    for line in text.splitlines(keepends=True):
        fence = FENCE_RE.match(line)
        if fence:
            marker = fence.group(1)
            if not in_fence:
                in_fence = True
                fence_char = marker[0]
            elif marker[0] == fence_char:
                in_fence = False
            continue
        if not in_fence:
            lines.append(re.sub(r"(`+).*?\1", "", line))
    return "".join(lines)


def _frontmatter_and_body(text: str) -> tuple[dict[str, Any], str, list[dict[str, str]]]:
    return parse_note(text)


def _portable_yaml(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool, int, float, date, datetime)):
        return True
    if isinstance(value, list):
        return all(_portable_yaml(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, (str, int, float, bool)) and _portable_yaml(item) for key, item in value.items())
    return False


def _resolve_internal(root: Path, source: Path, destination: str) -> tuple[str | None, str | None]:
    value = destination.strip()
    if value.startswith("<") and value.endswith(">"):
        value = value[1:-1]
    value = unquote(value)
    if not value or value.startswith("#"):
        return _rel(source, root), None
    if EXTERNAL_RE.match(value):
        return None, "external"
    if ABSOLUTE_RE.match(value):
        return None, "absolute_internal"
    path_part, _fragment = (value.split("#", 1) + [""])[:2] if "#" in value else (value, "")
    path_part = path_part.split("?", 1)[0]
    if not path_part:
        return _rel(source, root), None
    candidate = (source.parent / path_part).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None, "escapes_vault"
    if candidate.is_dir() and (candidate / "index.md").is_file():
        candidate = candidate / "index.md"
    if not candidate.is_file() and candidate.suffix.lower() != ".md":
        with_md = candidate.with_suffix(".md")
        if with_md.is_file():
            candidate = with_md
    if not candidate.is_file():
        return None, "missing_target"
    return _rel(candidate, root), None


def _link_rows(root: Path, path: Path, text: str) -> list[dict[str, Any]]:
    _meta, body, _issues = _frontmatter_and_body(text)
    visible = _visible_body(body)
    rows: list[dict[str, Any]] = []
    for match in MARKDOWN_LINK_RE.finditer(visible):
        destination = match.group(2).strip()
        target, error = _resolve_internal(root, path, destination)
        fragment = destination.split("#", 1)[1] if "#" in destination else ""
        rows.append({
            "source_path": _rel(path, root),
            "destination": destination,
            "target": target,
            "error": error,
            "fragment": unquote(fragment),
        })
    return rows


def _heading_slugs(text: str) -> set[str]:
    _meta, body, _issues = _frontmatter_and_body(text)
    slugs: set[str] = set()
    for line in body.splitlines():
        match = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", line)
        if not match:
            continue
        heading = re.sub(r"[`*_~]", "", match.group(1)).strip().casefold()
        heading = re.sub(r"[^\w\-\s]", "", heading, flags=re.UNICODE)
        slugs.add(re.sub(r"\s+", "-", heading))
    return slugs


def _validate_metadata(vault: Path, markdown: list[Path]) -> dict[str, Any]:
    parse_errors: list[dict[str, Any]] = []
    nonportable: list[dict[str, Any]] = []
    duplicate_ids: dict[str, list[str]] = {}
    missing_core: dict[str, list[str]] = {"schema_version": [], "entity_type": [], "title": []}
    ids: dict[str, list[str]] = {}
    for path in markdown:
        text = path.read_text(encoding="utf-8")
        meta, _body, issues = parse_note(text)
        rel = _rel(path, vault)
        if issues:
            parse_errors.append({"path": rel, "issues": issues})
        if not _portable_yaml(meta):
            nonportable.append({"path": rel})
        for field in missing_core:
            if not meta.get(field):
                missing_core[field].append(rel)
        note_id = meta.get("note_id")
        if note_id:
            ids.setdefault(str(note_id), []).append(rel)
    duplicate_ids = {key: paths for key, paths in ids.items() if len(paths) > 1}
    return {
        "status": "PASS" if not parse_errors and not nonportable and not duplicate_ids else "BLOCKED",
        "parse_error_count": len(parse_errors),
        "nonportable_count": len(nonportable),
        "duplicate_note_id_count": len(duplicate_ids),
        "missing_core_field_counts": {key: len(value) for key, value in missing_core.items()},
        "parse_errors": parse_errors[:100],
        "nonportable": nonportable[:100],
        "duplicate_note_ids": duplicate_ids,
    }


def _validate_links(vault: Path, markdown: list[Path], run_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in markdown:
        rows.extend(_link_rows(vault, path, path.read_text(encoding="utf-8")))
    broken = [row for row in rows if row["error"] not in (None, "external")]
    absolute = [row for row in rows if row["error"] == "absolute_internal"]
    fragment_errors: list[dict[str, Any]] = []
    for row in rows:
        if row["error"] is not None or not row["fragment"] or row["fragment"].startswith("^"):
            continue
        target = vault / str(row["target"])
        if row["fragment"].casefold() not in _heading_slugs(target.read_text(encoding="utf-8")):
            fragment_errors.append(row)

    expected_rows = _load_jsonl(run_dir / "link-rewrite-plan.jsonl")
    expected_edges = {
        (str(row.get("source_final_path")), str(row.get("final_target")))
        for row in expected_rows
        if row.get("source_final_path")
        and row.get("final_target")
        # Root/index and generated navigation hubs are projections.  They may
        # be bounded and regenerated; searchable note-to-note relationships
        # are the identity graph that must remain lossless.
        and str(row.get("source_final_path")) != "index.md"
        and not str(row.get("source_final_path")).startswith("00_Index/")
        and str(row.get("source_final_path")) != "20_Portfolio_Management/Portfolio_Dashboard.md"
    }
    current_edges = {
        (str(row["source_path"]), str(row["target"]))
        for row in rows
        if row.get("error") is None and row.get("target")
    }
    lost_edges = sorted(expected_edges - current_edges)
    _write_json(run_dir / "current-link-inventory.json", {
        "markdown_link_count": len(rows),
        "internal_edge_count": len(current_edges),
        "external_link_count": sum(row["error"] == "external" for row in rows),
        "broken": broken,
    })
    return {
        "A34_status": "PASS" if not broken else "BLOCKED",
        "A35_status": "PASS" if not absolute else "BLOCKED",
        "A36_status": "PASS" if not lost_edges else "BLOCKED",
        "A37_status": "PASS" if not fragment_errors else "BLOCKED",
        "markdown_link_count": len(rows),
        "internal_edge_count": len(current_edges),
        "broken_count": len(broken),
        "absolute_internal_count": len(absolute),
        "fragment_error_count": len(fragment_errors),
        "expected_edge_count": len(expected_edges),
        "lost_edge_count": len(lost_edges),
        "broken": broken[:100],
        "lost_edges": lost_edges[:100],
    }


def _validate_portability(vault: Path, markdown: list[Path]) -> dict[str, Any]:
    wikilinks: list[dict[str, Any]] = []
    embeds: list[dict[str, Any]] = []
    app_syntax: list[dict[str, Any]] = []
    scripts: list[dict[str, Any]] = []
    iframe_rows: list[dict[str, Any]] = []
    for path in markdown:
        rel = _rel(path, vault)
        text = path.read_text(encoding="utf-8")
        _meta, body, _issues = _frontmatter_and_body(text)
        visible = _visible_body(body)
        if not _is_allowlisted(rel):
            wikilinks.extend({"path": rel, "token": token} for token in WIKILINK_RE.findall(visible))
            embeds.extend({"path": rel, "token": token} for token in EMBED_RE.findall(visible))
            app_syntax.extend({"path": rel, "token": match.group(0)} for match in APP_SYNTAX_RE.finditer(visible))
            scripts.extend({"path": rel, "token": match.group(0)} for match in ACTIVE_SCRIPT_RE.finditer(visible))
        if re.search(r"<iframe\b", text, re.IGNORECASE):
            source_match = re.search(r"(?m)^source_url:\s*['\"]?(https?://[^'\"\s]+)", text)
            source_url = source_match.group(1) if source_match else ""
            links = _link_rows(vault, path, text)
            visible_links = [str(row["destination"]) for row in links if row["error"] == "external"]
            iframe_rows.append({
                "path": rel,
                "has_source_url": bool(source_url),
                "has_visible_source_link": bool(source_url and any(source_url in link for link in visible_links)),
            })
    iframe_missing = [row for row in iframe_rows if not row["has_source_url"] and not row["has_visible_source_link"]]
    return {
        "A32_status": "PASS" if not wikilinks else "BLOCKED",
        "A33_status": "PASS" if not embeds else "BLOCKED",
        "A38_status": "PASS" if not app_syntax else "BLOCKED",
        "A39_status": "PASS" if not iframe_missing else "BLOCKED",
        "A40_status": "PASS" if not scripts else "BLOCKED",
        "wikilink_count": len(wikilinks),
        "embed_count": len(embeds),
        "app_syntax_count": len(app_syntax),
        "active_script_count": len(scripts),
        "iframe_count": len(iframe_rows),
        "iframe_missing_count": len(iframe_missing),
        "wikilinks": wikilinks[:100],
        "embeds": embeds[:100],
        "app_syntax": app_syntax[:100],
        "iframes_missing_source": iframe_missing[:100],
    }


def _validate_paths(vault: Path) -> dict[str, Any]:
    files = [path for path in vault.rglob("*") if path.is_file()]
    markdown = [path for path in files if path.suffix.lower() == ".md"]
    too_long = [{"path": _rel(path, vault), "length": len(_rel(path, vault))} for path in markdown if len(_rel(path, vault)) > 180]
    long_assets = [{"path": _rel(path, vault), "length": len(_rel(path, vault))} for path in files if path.suffix.lower() != ".md" and len(_rel(path, vault)) > 180]
    collisions: dict[str, list[str]] = {}
    for path in files:
        rel = _rel(path, vault)
        key = unicodedata.normalize("NFC", rel).casefold()
        collisions.setdefault(key, []).append(rel)
    collisions = {key: values for key, values in collisions.items() if len(values) > 1}
    reserved = [
        _rel(path, vault)
        for path in files
        if path.stem.upper().split(".", 1)[0] in RESERVED_NAMES
    ]
    return {
        "status": "PASS" if not too_long and not collisions and not reserved else "BLOCKED",
        "too_long_count": len(too_long),
        "long_non_markdown_asset_count": len(long_assets),
        "casefold_nfc_collision_count": len(collisions),
        "reserved_name_count": len(reserved),
        "too_long": too_long[:100],
        "long_non_markdown_assets": long_assets[:100],
        "collisions": collisions,
        "reserved_names": reserved[:100],
    }


def _validate_configs(vault: Path) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    app_path = vault / ".obsidian" / "app.json"
    try:
        app = json.loads(app_path.read_text(encoding="utf-8"))
    except Exception as exc:
        app = {}
        checks["app_error"] = str(exc)
    checks["app_settings"] = {
        key: app.get(key)
        for key in ("useMarkdownLinks", "newLinkFormat", "alwaysUpdateLinks", "attachmentFolderPath")
    }
    checks["templates_config"] = (vault / ".obsidian" / "templates.json").is_file()
    checks["daily_config"] = (vault / ".obsidian" / "daily-notes.json").is_file()
    expected = {
        "useMarkdownLinks": True,
        "newLinkFormat": "relative",
        "alwaysUpdateLinks": True,
        "attachmentFolderPath": "90_Attachments",
    }
    status = all(app.get(key) == value for key, value in expected.items()) and checks["templates_config"] and checks["daily_config"]
    return {"status": "PASS" if status else "BLOCKED", **checks}


def _validate_templates(vault: Path) -> dict[str, Any]:
    folder = vault / "99_Templates"
    templates = sorted(folder.glob("*.md")) if folder.is_dir() else []
    failures: list[dict[str, Any]] = []
    for path in templates:
        text = path.read_text(encoding="utf-8")
        if "capture_status: pending_normalization" not in text or "search_scope: excluded" not in text:
            failures.append({"path": _rel(path, vault), "reason": "missing capture contract"})
        if re.search(r"(?m)^note_id:", text):
            failures.append({"path": _rel(path, vault), "reason": "template contains a fake identity"})
    return {
        "status": "PASS" if templates and not failures else "BLOCKED",
        "template_count": len(templates),
        "failure_count": len(failures),
        "failures": failures,
    }


def _validate_writer_hardening(root: Path, vault: Path, run_dir: Path) -> dict[str, Any]:
    source_files = [
        root / "tools/archivist/indexer.py",
        root / "tools/archivist/writer.py",
        root / "tools/portfolio/adapters/markdown/journal_format.py",
        root / "tools/portfolio/adapters/markdown/journal_vault_adapter.py",
        root / "tools/portfolio/adapters/markdown/repository_adapter.py",
        root / "tools/portfolio/journal.py",
        root / "tools/market/news.py",
        root / "tools/market/consensus.py",
        root / "tools/market/technical.py",
        root / "tools/market/equity_report_formatter.py",
        root / "tools/macro/news_funnel.py",
    ]
    direct_emit: list[dict[str, Any]] = []
    for path in source_files:
        if not path.is_file():
            continue
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if "[[" in line and re.search(r"(?:f[\"']|append\(|return\s+f|join\(|format\()", line):
                if "re." not in line and "wikilink" not in line.lower():
                    direct_emit.append({"path": str(path.relative_to(root)), "line": line_no, "text": line.strip()})

    smoke: dict[str, Any] = {}
    try:
        from tools.macro.news_funnel import _format_6_sections
        from tools.portfolio.adapters.markdown.journal_format import inject_journal_links
        from tools.archivist import indexer

        smoke["news_funnel_has_wikilink"] = "[[" in _format_6_sections("summary", ["AAPL"], ["rates"])
        smoke["journal_has_wikilink"] = "[[" in inject_journal_links(
            "**[BUY]** FTNT **[10 units]**",
            vault_root=vault,
            source_path=vault / "20_Portfolio_Management/Current_Holdings/Portfolios/default/Trading_Journal.md",
        )
        indexer._build_cache_from_disk(vault)
        indexer._write_index_from_cache(vault)
        smoke["index_has_wikilink"] = "[[" in (vault / "index.md").read_text(encoding="utf-8")
    except Exception as exc:
        smoke["error"] = str(exc)
    status = not direct_emit and not any(value for key, value in smoke.items() if key.endswith("has_wikilink")) and "error" not in smoke
    result = {"status": "PASS" if status else "BLOCKED", "direct_emit_count": len(direct_emit), "direct_emit": direct_emit, "smoke": smoke}
    _write_json(run_dir / "writer-hardening.json", result)
    return result


def _validate_idempotence(vault: Path, run_dir: Path, owner: str) -> dict[str, Any]:
    os.environ["VAULT_MAINTENANCE_OWNER"] = owner
    before = {_rel(path, vault): _sha256(path) for path in vault.rglob("*.md")}
    try:
        from tools.archivist.navigation_builder import build_navigation_indices
        build_navigation_indices(vault)
        after_first = {_rel(path, vault): _sha256(path) for path in vault.rglob("*.md")}
        build_navigation_indices(vault)
        after_second = {_rel(path, vault): _sha256(path) for path in vault.rglob("*.md")}
        from tools.archivist import indexer
        indexer._build_cache_from_disk(vault)
        indexer._write_index_from_cache(vault)
        index_first = {_rel(path, vault): _sha256(path) for path in vault.rglob("*.md")}
        indexer._build_cache_from_disk(vault)
        indexer._write_index_from_cache(vault)
        index_second = {_rel(path, vault): _sha256(path) for path in vault.rglob("*.md")}
    except Exception as exc:
        result = {"status": "BLOCKED", "error": str(exc)}
        _write_json(run_dir / "generator-idempotence.json", result)
        return result
    changed_second = sorted(path for path in set(after_first) | set(after_second) if after_first.get(path) != after_second.get(path))
    changed_index_second = sorted(path for path in set(index_first) | set(index_second) if index_first.get(path) != index_second.get(path))
    result = {
        "status": "PASS" if not changed_second and not changed_index_second else "BLOCKED",
        "initial_file_count": len(before),
        "after_first_file_count": len(after_first),
        "navigation_second_run_changed": changed_second,
        "index_second_run_changed": changed_index_second,
    }
    _write_json(run_dir / "generator-idempotence.json", result)
    return result


def _validate_derived(vault: Path, run_dir: Path, require: bool) -> dict[str, Any]:
    pointers = {
        "catalog": vault / ".system" / "catalog_generation_active.json",
        "vector": vault / ".system" / "vector_generation_active.json",
    }
    records: dict[str, Any] = {}
    failures: list[str] = []
    for name, pointer_path in pointers.items():
        if not pointer_path.is_file():
            failures.append(f"missing_{name}_pointer")
            records[name] = {"status": "MISSING", "path": str(pointer_path)}
            continue
        try:
            payload = json.loads(pointer_path.read_text(encoding="utf-8"))
            if name == "catalog":
                from tools.archivist.catalog_runtime import resolve_catalog_path
                target = resolve_catalog_path(vault, require_exists=True)
                catalog_missing: list[str] = []
                catalog_hash_mismatch: list[str] = []
                with sqlite3.connect(f"file:{target.as_posix()}?mode=ro&immutable=1", uri=True) as conn:
                    rows = conn.execute(
                        "SELECT relative_path, content_sha256 FROM note_catalog WHERE record_state = 'active'"
                    ).fetchall()
                for rel, expected_sha in rows:
                    path = vault / str(rel)
                    if not path.is_file():
                        catalog_missing.append(str(rel))
                    elif _canonical_content_sha256(path) != str(expected_sha):
                        catalog_hash_mismatch.append(str(rel))
                catalog_ok = bool(target.is_file()) and not catalog_missing and not catalog_hash_mismatch
                if not catalog_ok:
                    failures.append("catalog_content_hash_mismatch")
            else:
                from tools.archivist.vector_generation import (
                    load_active_manifest,
                    manifest_path,
                    vector_runtime_path,
                )
                manifest = load_active_manifest(vault)
                target = manifest_path(vault, str(payload.get("generation_id"))) if manifest else None
                state_path = vector_runtime_path(vault) / "vector_index_state.json"
                state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
                indexed_files = state.get("indexed_files") if isinstance(state, dict) else {}
                vector_missing: list[str] = []
                vector_hash_mismatch: list[str] = []
                for rel, expected_sha in (indexed_files or {}).items():
                    path = vault / str(rel)
                    if not path.is_file():
                        vector_missing.append(str(rel))
                    elif _canonical_content_sha256(path) != str(expected_sha):
                        vector_hash_mismatch.append(str(rel))
                vector_corpus_ok = bool(manifest and state.get("corpus_fingerprint") == manifest.get("corpus_fingerprint"))
                if not vector_corpus_ok or vector_missing or vector_hash_mismatch:
                    failures.append("vector_content_hash_or_corpus_mismatch")
                records_extra = {
                    "manifest_note_count": manifest.get("eligible_note_count") if manifest else None,
                    "indexed_file_count": len(indexed_files or {}),
                    "corpus_fingerprint_match": vector_corpus_ok,
                    "missing_count": len(vector_missing),
                    "hash_mismatch_count": len(vector_hash_mismatch),
                }
            exists = bool(target and target.is_file())
            if not exists:
                failures.append(f"missing_{name}_target")
            records[name] = {
                "status": "PASS" if exists and (name != "catalog" or (not catalog_missing and not catalog_hash_mismatch)) and (name != "vector" or (vector_corpus_ok and not vector_missing and not vector_hash_mismatch)) else "BLOCKED",
                "pointer": payload,
                "target": str(target) if target else None,
                **({
                    "catalog_row_count": len(rows),
                    "missing_count": len(catalog_missing),
                    "hash_mismatch_count": len(catalog_hash_mismatch),
                } if name == "catalog" else records_extra),
            }
        except Exception as exc:
            failures.append(f"invalid_{name}_pointer")
            records[name] = {"status": "BLOCKED", "error": str(exc)}
    status = "PASS" if not failures else ("BLOCKED" if require else "DEFERRED")
    result = {"status": status, "required": require, "failures": failures, "records": records}
    _write_json(run_dir / "derived-read-model-validation.json", result)
    return result


def _validate_render(vault: Path, markdown: list[Path]) -> dict[str, Any]:
    try:
        from markdown_it import MarkdownIt
        parser = MarkdownIt("commonmark")
        rendered = 0
        for path in markdown:
            _meta, body, _issues = parse_note(path.read_text(encoding="utf-8"))
            parser.render(body)
            rendered += 1
        return {"status": "PASS", "rendered_count": rendered, "engine": "markdown_it.commonmark"}
    except Exception as exc:
        return {"status": "BLOCKED", "error": str(exc)}


def validate(root: Path, vault: Path, run_dir: Path, *, owner: str, require_derived: bool) -> dict[str, Any]:
    root = root.resolve()
    vault = vault.resolve()
    run_dir = run_dir.resolve()
    os.environ["VAULT_MAINTENANCE_OWNER"] = owner
    markdown = sorted(path for path in vault.rglob("*.md") if path.is_file())
    metadata = _validate_metadata(vault, markdown)
    portability = _validate_portability(vault, markdown)
    links = _validate_links(vault, markdown, run_dir)
    paths = _validate_paths(vault)
    configs = _validate_configs(vault)
    templates = _validate_templates(vault)
    writer = _validate_writer_hardening(root, vault, run_dir)
    idempotence = _validate_idempotence(vault, run_dir, owner)
    render = _validate_render(vault, markdown)
    derived = _validate_derived(vault, run_dir, require_derived)

    checks = {
        "A31_metadata_parse": metadata["status"],
        "A32_no_wikilinks": portability["A32_status"],
        "A33_no_embeds": portability["A33_status"],
        "A34_internal_links_resolve": links["A34_status"],
        "A35_no_absolute_internal_links": links["A35_status"],
        "A36_link_graph_preserved": links["A36_status"],
        "A37_fragments_resolve": links["A37_status"],
        "A38_app_syntax_isolated": portability["A38_status"],
        "A39_iframe_fallbacks": portability["A39_status"],
        "A40_no_active_scripts": portability["A40_status"],
        "A41_path_safety": paths["status"],
        "A42_frontmatter_portable": metadata["status"],
        "A43_app_config": configs["status"],
        "A44_template_contract": templates["status"],
        "A45_writer_hardening": writer["status"],
        "A46_generator_idempotence": idempotence["status"],
        "A47_derived_read_models": derived["status"],
        "A48_commonmark_render": render["status"],
    }
    blocking_values = {"BLOCKED"}
    if require_derived:
        blocking_values.add("DEFERRED")
    result = {
        "status": "PASS" if not any(value in blocking_values for value in checks.values()) else "BLOCKED",
        "vault": str(vault),
        "markdown_count": len(markdown),
        "checks": checks,
        "metadata": metadata,
        "portability": portability,
        "links": links,
        "paths": paths,
        "configs": configs,
        "templates": templates,
        "writer_hardening": writer,
        "idempotence": idempotence,
        "derived": derived,
        "render": render,
    }
    _write_json(run_dir / "multi-app-validation.json", result)
    print(json.dumps(result, ensure_ascii=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", type=Path, default=Path("memories"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--owner", default="codex-vault-r6")
    parser.add_argument("--require-derived", action="store_true")
    args = parser.parse_args()
    result = validate(args.root, args.vault, args.run_dir, owner=args.owner, require_derived=args.require_derived)
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
