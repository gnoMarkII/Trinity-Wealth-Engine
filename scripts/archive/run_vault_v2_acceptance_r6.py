"""Read-only R6 portability acceptance for a Vault tree.

Unlike the live migration validator, this runner never rewrites notes or
regenerates navigation. It is intended to expose the baseline defect set and
to provide a small, machine-readable portability gate for any future app.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.archivist.metadata import parse_note  # noqa: E402


FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
WIKILINK_RE = re.compile(r"(?<!\!)\[\[[^\[\]]+\]\]")
EMBED_RE = re.compile(r"!\[\[[^\[\]]+\]\]")
MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")
APP_SYNTAX_RE = re.compile(
    r"(?:obsidian://|\bdataview(?:js)?\b|meta-bind|meta-bind-button|<%|<script\b|javascript:)",
    re.IGNORECASE,
)
ALLOWLIST = (
    "20_Portfolio_Management/Portfolio_Dashboard.md",
    "00_Index/App_Views/Obsidian/",
)


def _visible(text: str) -> str:
    lines: list[str] = []
    in_fence = False
    fence_char = ""
    for line in text.splitlines():
        fence = FENCE_RE.match(line)
        if fence:
            marker = fence.group(1)[0]
            if not in_fence:
                in_fence, fence_char = True, marker
            elif marker == fence_char:
                in_fence = False
            continue
        if not in_fence:
            lines.append(re.sub(r"`[^`]*`", "", line))
    return "\n".join(lines)


def _allowlisted(rel: str) -> bool:
    return rel == ALLOWLIST[0] or rel.startswith(ALLOWLIST[1])


def _resolve_internal(vault: Path, source: Path, destination: str) -> bool:
    value = unquote(destination.strip().split("#", 1)[0])
    if not value or urlparse(value).scheme or value.startswith("//"):
        return True
    if re.match(r"^(?:[A-Za-z]:[\\/]|/|\\\\|file://)", value):
        return False
    target = (source.parent / value).resolve()
    try:
        target.relative_to(vault)
    except ValueError:
        return False
    if target.is_file():
        return True
    return target.with_suffix(".md").is_file()


def accept(vault: Path, output: Path) -> dict[str, object]:
    vault = vault.resolve()
    markdown = sorted(path for path in vault.rglob("*.md") if path.is_file())
    parse_errors: list[dict[str, object]] = []
    wikilinks: list[dict[str, object]] = []
    embeds: list[dict[str, object]] = []
    app_syntax: list[dict[str, object]] = []
    broken: list[dict[str, object]] = []
    too_long: list[str] = []
    for path in markdown:
        rel = path.relative_to(vault).as_posix()
        text = path.read_text(encoding="utf-8")
        _meta, _body, issues = parse_note(text)
        if issues:
            parse_errors.append({"path": rel, "issues": issues})
        visible = _visible(text)
        path_parts = set(path.relative_to(vault).parts)
        excluded_lifecycle_copy = bool(
            {"40_Archive", "Revisions", ".backups", "99_Templates"} & path_parts
        )
        if not _allowlisted(rel) and not excluded_lifecycle_copy:
            wikilinks.extend({"path": rel, "token": token} for token in WIKILINK_RE.findall(visible))
            embeds.extend({"path": rel, "token": token} for token in EMBED_RE.findall(visible))
            app_syntax.extend({"path": rel, "token": match.group(0)} for match in APP_SYNTAX_RE.finditer(visible))
        if not excluded_lifecycle_copy:
            for match in MARKDOWN_LINK_RE.finditer(visible):
                destination = match.group(1).strip().strip("<>")
                if not _resolve_internal(vault, path, destination):
                    broken.append({"path": rel, "destination": destination})
        if len(rel) > 180:
            too_long.append(rel)
    checks = {
        "A31_metadata_parse": not parse_errors,
        "A32_no_wikilinks": not wikilinks,
        "A33_no_embeds": not embeds,
        "A34_internal_links_resolve": not broken,
        "A38_app_syntax_isolated": not app_syntax,
        "A41_markdown_path_safety": not too_long,
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "vault": str(vault),
        "read_only": True,
        "markdown_count": len(markdown),
        "checks": {key: "PASS" if value else "FAIL" for key, value in checks.items()},
        "counts": {
            "parse_error_count": len(parse_errors),
            "wikilink_count": len(wikilinks),
            "embed_count": len(embeds),
            "app_syntax_count": len(app_syntax),
            "broken_link_count": len(broken),
            "too_long_markdown_count": len(too_long),
        },
        "samples": {
            "parse_errors": parse_errors[:20],
            "wikilinks": wikilinks[:20],
            "embeds": embeds[:20],
            "app_syntax": app_syntax[:20],
            "broken_links": broken[:20],
            "too_long": too_long[:20],
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return 0 if accept(args.vault, args.output)["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
