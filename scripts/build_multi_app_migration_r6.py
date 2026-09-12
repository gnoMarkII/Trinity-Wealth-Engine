"""Build a deterministic, external R6 path/link migration plan.

The builder never edits the Vault.  It parses Markdown outside code fences,
resolves legacy Obsidian wikilinks against the frozen tree, proposes short
cross-platform paths, and emits row-level evidence for the apply step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.archivist.metadata import parse_note  # noqa: E402


WIKILINK_RE = re.compile(r"(?<!\!)\[\[([^\[\]]+)\]\]")
EMBED_RE = re.compile(r"!\[\[([^\[\]]+)\]\]")
MD_LINK_RE = re.compile(r"(?<!!)\[([^\]]*)\]\(([^\)]+)\)")
FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
PATH_SAFE = "/:@-._~!$&'+,;=@"
MAX_SOFT_PATH = 160


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


def _parse_segments(text: str) -> list[tuple[str, str]]:
    """Return (kind, text) segments, preserving fences and ordinary lines."""
    segments: list[tuple[str, str]] = []
    in_fence = False
    fence_char = ""
    for line in text.splitlines(keepends=True):
        match = FENCE_RE.match(line)
        if match:
            marker = match.group(1)
            if not in_fence:
                in_fence = True
                fence_char = marker[0]
            elif marker[0] == fence_char:
                in_fence = False
            segments.append(("fence", line))
        else:
            segments.append(("fence" if in_fence else "body", line))
    return segments


def _split_link(raw: str) -> tuple[str, str | None, str | None]:
    value = raw.strip()
    alias = None
    if "|" in value:
        value, alias = value.split("|", 1)
        alias = alias.strip()
    fragment = None
    if "#" in value:
        value, fragment = value.split("#", 1)
        fragment = fragment.strip()
    return value.strip(), alias, fragment


def _norm_rel(value: str) -> str:
    value = value.replace("\\", "/")
    value = re.sub(r"/+/", "/", value)
    while value.startswith("./"):
        value = value[2:]
    return value.strip("/")


def _uri_path(rel: str) -> str:
    return "/".join(quote(part, safe=PATH_SAFE) for part in rel.split("/"))


def _uri_fragment(fragment: str) -> str:
    return quote(fragment.strip(), safe="-._~")


def _relative_link(source_rel: str, target_rel: str, fragment: str | None) -> str:
    source_parent = Path(source_rel).parent
    relative = Path(target_rel).relative_to(Path(target_rel).anchor) if False else Path(
        __import__("os").path.relpath(target_rel, start=source_parent.as_posix())
    )
    target = relative.as_posix()
    if target == ".":
        target = Path(target_rel).name
    if not target.startswith(".") and "/" not in target:
        target = f"./{target}"
    encoded = _uri_path(target)
    if fragment:
        encoded += f"#{_uri_fragment(fragment)}"
    return encoded


def _candidate_target(raw_target: str, source_rel: str, files: set[str], by_stem: dict[str, list[str]], by_title: dict[str, list[str]], by_alias: dict[str, list[str]]) -> tuple[str | None, list[str], str]:
    target = _norm_rel(raw_target)
    if not target:
        return None, [], "empty_target"
    if target.startswith("^"):
        return None, [], "block_reference"
    source_parent = Path(source_rel).parent.as_posix()
    candidates: list[tuple[str, str]] = []

    raw_candidates = []
    if target.startswith("/"):
        raw_candidates.append(target.lstrip("/"))
    else:
        raw_candidates.append(_norm_rel(Path(source_parent, target).as_posix()))
        raw_candidates.append(target)
    expanded: list[str] = []
    for item in raw_candidates:
        expanded.append(item)
        if not item.lower().endswith(".md"):
            expanded.append(f"{item}.md")
        if item in {".", ""}:
            expanded.append(f"{item}/index.md")
        elif not item.lower().endswith("/index.md"):
            expanded.append(f"{item}/index.md")
    for item in expanded:
        if item in files and item not in [row[0] for row in candidates]:
            candidates.append((item, "path"))
    if len(candidates) == 1:
        return candidates[0][0], [candidates[0][0]], candidates[0][1]
    if len(candidates) > 1:
        return None, [item[0] for item in candidates], "ambiguous_path"

    stem = Path(target).stem.casefold()
    stem_candidates = by_stem.get(stem, [])
    if len(stem_candidates) == 1:
        return stem_candidates[0], stem_candidates, "stem"
    if len(stem_candidates) > 1:
        return None, stem_candidates, "ambiguous_stem"

    title_candidates = by_title.get(target.casefold(), [])
    if len(title_candidates) == 1:
        return title_candidates[0], title_candidates, "title"
    if len(title_candidates) > 1:
        return None, title_candidates, "ambiguous_title"

    alias_candidates = by_alias.get(target.casefold(), [])
    if len(alias_candidates) == 1:
        return alias_candidates[0], alias_candidates, "alias"
    if len(alias_candidates) > 1:
        return None, alias_candidates, "ambiguous_alias"
    return None, [], "unresolved"


def _short_path(path: Path, root: Path) -> tuple[str, str | None]:
    rel = path.relative_to(root).as_posix()
    if len(rel) <= MAX_SOFT_PATH:
        return rel, None
    digest = _sha256(path)[:10]
    stem = path.stem
    # Keep the beginning of the human title, then add a stable content suffix.
    # The suffix prevents collisions without using the mutable path as identity.
    budget = max(24, MAX_SOFT_PATH - len(path.parent.relative_to(root).as_posix()) - len(path.suffix) - len(digest) - 3)
    prefix = re.sub(r"\s+", " ", unicodedata.normalize("NFC", stem)).strip()
    prefix = prefix[:budget].rstrip(" .-_")
    new_name = f"{prefix}__{digest}{path.suffix}"
    return path.parent.relative_to(root).joinpath(new_name).as_posix(), "path_length"


def build_plan(vault: Path, run_dir: Path) -> dict[str, Any]:
    vault = vault.resolve()
    run_dir = run_dir.resolve()
    markdown = sorted(path for path in vault.rglob("*.md") if path.is_file())
    files = {path.relative_to(vault).as_posix() for path in markdown}
    by_stem: dict[str, list[str]] = {}
    by_title: dict[str, list[str]] = {}
    by_alias: dict[str, list[str]] = {}
    metadata_rows: dict[str, dict[str, Any]] = {}
    for path in markdown:
        rel = path.relative_to(vault).as_posix()
        text = path.read_text(encoding="utf-8")
        meta, _body, issues = parse_note(text)
        metadata_rows[rel] = {"metadata": meta, "issues": issues}
        by_stem.setdefault(path.stem.casefold(), []).append(rel)
        title = meta.get("title")
        if title:
            by_title.setdefault(str(title).casefold(), []).append(rel)
        aliases = meta.get("aliases") or []
        if isinstance(aliases, str):
            aliases = [aliases]
        if isinstance(aliases, list):
            for alias in aliases:
                by_alias.setdefault(str(alias).casefold(), []).append(rel)

    path_map: list[dict[str, Any]] = []
    final_paths: dict[str, str] = {}
    for path in markdown:
        old = path.relative_to(vault).as_posix()
        new, reason = _short_path(path, vault)
        final_paths[old] = new
        if old != new:
            path_map.append({
                "old_path": old,
                "new_path": new,
                "reason": reason,
                "note_id": metadata_rows[old]["metadata"].get("note_id"),
                "document_key": metadata_rows[old]["metadata"].get("document_key"),
                "sha256": _sha256(path),
            })
    collisions: dict[str, list[str]] = {}
    for old, new in final_paths.items():
        collisions.setdefault(new.casefold(), []).append(old)
    final_collisions = [values for values in collisions.values() if len(values) > 1]

    link_rows: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []
    block_refs: list[dict[str, Any]] = []
    rewrite_counts = {"wikilinks": 0, "same_target": 0, "with_alias": 0, "with_fragment": 0}
    for path in markdown:
        source_old = path.relative_to(vault).as_posix()
        source_final = final_paths[source_old]
        text = path.read_text(encoding="utf-8")
        for line_no, (kind, line) in enumerate(_parse_segments(text), start=1):
            if kind != "body":
                continue
            for match in WIKILINK_RE.finditer(line):
                raw = match.group(1)
                target, alias, fragment = _split_link(raw)
                resolved, candidates, resolution = _candidate_target(
                    target, source_old, files, by_stem, by_title, by_alias
                )
                row = {
                    "source_path": source_old,
                    "source_final_path": source_final,
                    "line": line_no,
                    "column": match.start() + 1,
                    "raw": match.group(0),
                    "target": target,
                    "alias": alias,
                    "fragment": fragment,
                    "resolved_target": resolved,
                    "final_target": final_paths.get(resolved) if resolved else None,
                    "candidates": candidates,
                    "resolution": resolution,
                }
                link_rows.append(row)
                rewrite_counts["wikilinks"] += 1
                if alias:
                    rewrite_counts["with_alias"] += 1
                if fragment:
                    rewrite_counts["with_fragment"] += 1
                if fragment and fragment.startswith("^"):
                    block_refs.append(row)
                elif not resolved:
                    (ambiguous if resolution.startswith("ambiguous") else unresolved).append(row)
                elif final_paths.get(resolved) == source_final and not fragment:
                    rewrite_counts["same_target"] += 1

    plan = {
        "status": "PASS" if not unresolved and not ambiguous and not block_refs and not final_collisions else "BLOCKED",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "vault_root": str(vault),
        "markdown_count": len(markdown),
        "path_map_count": len(path_map),
        "link_count": len(link_rows),
        "rewrite_counts": rewrite_counts,
        "unresolved_count": len(unresolved),
        "ambiguous_count": len(ambiguous),
        "block_reference_count": len(block_refs),
        "final_path_collision_count": len(final_collisions),
        "contract": {
            "link_style": "relative_markdown",
            "max_relative_path_length": MAX_SOFT_PATH,
            "rewrite_code_fences": False,
            "rewrite_external_urls": False,
        },
    }
    _write_json(run_dir / "migration-plan.json", plan)
    _write_jsonl(run_dir / "path-map.jsonl", path_map)
    _write_jsonl(run_dir / "link-rewrite-plan.jsonl", link_rows)
    _write_jsonl(run_dir / "unresolved-links.jsonl", unresolved)
    _write_jsonl(run_dir / "ambiguous-links.jsonl", ambiguous)
    _write_jsonl(run_dir / "fragment-inventory.jsonl", [row for row in link_rows if row.get("fragment")])
    _write_json(run_dir / "path-collision-report.json", {"status": "PASS" if not final_collisions else "FAIL", "collisions": final_collisions})
    _write_json(run_dir / "link-resolution.json", {
        "status": "PASS" if not unresolved and not ambiguous and not block_refs else "BLOCKED",
        "unresolved": unresolved,
        "ambiguous": ambiguous,
        "block_references": block_refs,
    })
    print(json.dumps(plan, ensure_ascii=False))
    return plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", type=Path, default=Path("memories"))
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    plan = build_plan(args.vault, args.run_dir)
    return 0 if plan["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
