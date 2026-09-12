"""Portable Markdown link rendering for canonical Vault content."""
from __future__ import annotations

import posixpath
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote, unquote
from typing import Callable, Iterator, Optional, Union


_PATH_SAFE = "/:@-._~!$&'+,;=@"
_URI_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")


@dataclass(frozen=True)
class MarkdownLink:
    """One ordinary Markdown link found in a note body."""

    label: str
    destination: str
    fragment: Optional[str] = None
    line: int = 1
    start: int = 0
    end: int = 0


@dataclass(frozen=True)
class LinkResolution:
    """Detailed, fail-closed result for one internal link target."""

    status: str
    clean_target: str
    target: Optional[Path] = None
    candidates: tuple[Path, ...] = ()


def _split_markdown_destination(raw: str) -> tuple[str, Optional[str]]:
    """Extract the destination from ``[label](destination "title")``."""
    value = raw.strip()
    if value.startswith("<"):
        close = value.find(">")
        if close >= 0:
            destination = value[1:close]
        else:
            destination = value[1:]
    else:
        destination = value.split(None, 1)[0] if value else ""
    fragment: Optional[str] = None
    if "#" in destination:
        destination, fragment = destination.split("#", 1)
        fragment = unquote(fragment)
    return destination.strip(), fragment


def iter_markdown_links(text: str) -> Iterator[MarkdownLink]:
    """Yield ordinary Markdown links without treating images as graph edges.

    The small scanner handles balanced parentheses in destinations, encoded
    spaces, and optional Markdown link titles.  It intentionally does not
    parse Obsidian wikilinks: canonical notes use this portable format and
    adapters can inspect wikilinks separately when needed.
    """
    source = text or ""
    opener = re.compile(r"(?<!!)\[([^\]\n]*)\]\(")
    for match in opener.finditer(source):
        position = match.end()
        depth = 1
        escaped = False
        while position < len(source) and depth:
            char = source[position]
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            position += 1
        if depth:
            continue
        raw_destination = source[match.end(): position - 1]
        destination, fragment = _split_markdown_destination(raw_destination)
        if not destination:
            continue
        yield MarkdownLink(
            label=match.group(1),
            destination=destination,
            fragment=fragment,
            line=source.count("\n", 0, match.start()) + 1,
            start=match.start(),
            end=position,
        )


def is_external_destination(destination: str) -> bool:
    """Return whether a Markdown destination is not a Vault graph edge."""
    value = str(destination or "").strip()
    return (
        not value
        or value.startswith("#")
        or value.startswith("//")
        or bool(_URI_SCHEME_RE.match(value))
    )


def iter_internal_markdown_links(text: str) -> Iterator[MarkdownLink]:
    """Yield only Markdown links that may resolve to a Vault note."""
    for link in iter_markdown_links(text):
        if not is_external_destination(link.destination):
            yield link


def rewrite_internal_markdown_links(
    text: str,
    callback: Callable[[MarkdownLink], Optional[str]],
) -> str:
    """Rewrite selected internal links while leaving surrounding Markdown intact."""
    source = text or ""
    replacements: list[tuple[int, int, str]] = []
    for link in iter_internal_markdown_links(source):
        replacement = callback(link)
        if replacement is not None:
            replacements.append((link.start, link.end, replacement))
    for start, end, replacement in reversed(replacements):
        source = source[:start] + replacement + source[end:]
    return source


def encode_markdown_destination(value: str) -> str:
    """Encode a relative Markdown destination without encoding separators."""
    return "/".join(quote(part, safe=_PATH_SAFE) for part in value.replace("\\", "/").split("/"))


def render_relative_markdown_link(
    vault_root: Union[str, Path],
    source: Union[str, Path],
    target: Union[str, Path],
    label: Optional[str] = None,
    fragment: Optional[str] = None,
) -> str:
    """Render a portable relative Markdown link between Vault files."""
    root = Path(vault_root).resolve()
    source_path = Path(source)
    target_path = Path(target)
    if source_path.is_absolute():
        source_rel = source_path.resolve().relative_to(root).as_posix()
    else:
        source_rel = source_path.as_posix()
    if target_path.is_absolute():
        target_rel = target_path.resolve().relative_to(root).as_posix()
    else:
        target_rel = target_path.as_posix()
    destination = posixpath.relpath(target_rel, start=posixpath.dirname(source_rel))
    destination = encode_markdown_destination(destination)
    if fragment:
        destination += "#" + quote(str(fragment).strip(), safe="-._~")
    display = label or Path(target_rel).stem
    display = str(display).replace("]", "\\]")
    return f"[{display}]({destination})"


def _clean_target(target: str) -> str:
    """Remove legacy link decoration before resolving a Vault target."""
    value = unquote(str(target or "").strip()).replace("\\", "/")
    if value.startswith("<") and value.endswith(">"):
        value = value[1:-1]
    if value.startswith("[[") and value.endswith(']]'):
        value = value[2:-2]
    if "|" in value:
        value = value.split("|", 1)[0]
    if "#" in value:
        value = value.split("#", 1)[0]
    return value.strip()


@lru_cache(maxsize=16)
def _vault_markdown_files(vault_root: str) -> tuple[str, ...]:
    root = Path(vault_root)
    if not root.exists():
        return ()
    return tuple(
        p.relative_to(root).as_posix()
        for p in sorted(root.rglob("*.md"))
        if ".system" not in p.parts and not any(part.startswith(".") for part in p.relative_to(root).parts)
    )


def resolve_vault_target(
    vault_root: Union[str, Path],
    target: str,
    source: Optional[Union[str, Path]] = None,
) -> Optional[Path]:
    """Resolve a legacy/path-like target to one canonical Markdown file.

    Resolution is deliberately conservative: an ambiguous basename is not
    guessed.  This keeps generated links portable and prevents a writer from
    silently linking to the wrong note.
    """
    result = resolve_vault_target_detailed(vault_root, target, source=source)
    return result.target if result.status == "resolved" else None


def clear_vault_link_cache() -> None:
    """Forget the file-name snapshot after a guarded Vault mutation."""
    _vault_markdown_files.cache_clear()


def _safe_relative_source(root: Path, source: Optional[Union[str, Path]]) -> Optional[Path]:
    if source is None:
        return None
    raw = Path(source)
    try:
        resolved = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
        return resolved.relative_to(root)
    except (OSError, ValueError):
        return None


def resolve_vault_target_detailed(
    vault_root: Union[str, Path],
    target: str,
    *,
    source: Optional[Union[str, Path]] = None,
) -> LinkResolution:
    """Resolve a target with source-relative Markdown semantics.

    Exact source-relative paths win.  Bare legacy basenames are accepted only
    when they identify one file; two matching stems are reported as
    ``ambiguous`` instead of guessed.  A caller can therefore distinguish a
    genuine broken edge from a legacy audit false positive.
    """
    root = Path(vault_root).resolve()
    raw_value = str(target or "").strip()
    if is_external_destination(raw_value):
        return LinkResolution("external", _clean_target(raw_value))
    clean = _clean_target(raw_value)
    if not clean:
        return LinkResolution("anchor", clean)

    clean = clean.replace("\\", "/")
    source_rel = _safe_relative_source(root, source)
    raw = Path(clean)
    candidate_paths: list[Path] = []

    def add_candidate(path: Path) -> None:
        try:
            resolved = path.resolve()
            resolved.relative_to(root)
        except (OSError, ValueError):
            return
        if resolved not in candidate_paths:
            candidate_paths.append(resolved)

    if raw.is_absolute():
        add_candidate(raw)
    else:
        # Markdown paths are relative to the source note.  For a bare stem we
        # also retain root-relative legacy compatibility below.
        if source_rel is not None:
            add_candidate(root / source_rel.parent / raw)
        add_candidate(root / raw)

    expanded: list[Path] = []
    for candidate in candidate_paths:
        expanded.append(candidate)
        if candidate.suffix.lower() != ".md":
            expanded.append(candidate.with_suffix(".md"))
        if candidate.is_dir():
            expanded.extend((candidate / "index.md", candidate / f"{candidate.name}.md"))

    matches: list[Path] = []
    for candidate in expanded:
        try:
            candidate = candidate.resolve()
            candidate.relative_to(root)
        except (OSError, ValueError):
            continue
        if candidate.is_file() and candidate.suffix.lower() == ".md" and candidate not in matches:
            matches.append(candidate)
    if len(matches) == 1:
        return LinkResolution("resolved", clean, target=matches[0], candidates=(matches[0],))
    if len(matches) > 1:
        return LinkResolution("ambiguous", clean, candidates=tuple(matches))

    # A bare name is a supported legacy adapter form.  Never use this fallback
    # for ``../`` or other explicit path expressions.
    if "/" not in clean and not clean.startswith("."):
        wanted = Path(clean).stem.casefold()
        rels = _vault_markdown_files(str(root))
        basename_matches = tuple(
            (root / rel)
            for rel in rels
            if Path(rel).stem.casefold() == wanted
        )
        if len(basename_matches) == 1:
            return LinkResolution("resolved", clean, target=basename_matches[0], candidates=basename_matches)
        if len(basename_matches) > 1:
            return LinkResolution("ambiguous", clean, candidates=basename_matches)
    return LinkResolution("missing", clean)


def render_resolved_markdown_link(
    vault_root: Union[str, Path],
    source: Union[str, Path],
    target: str,
    label: Optional[str] = None,
) -> str:
    """Render a link only when ``target`` resolves unambiguously.

    Unresolved targets become escaped plain text.  A broken Markdown link is
    worse for multi-app consumers than a visible, searchable label.
    """
    resolved = resolve_vault_target(vault_root, target, source=source)
    display = label or _clean_target(target) or str(target)
    if resolved is None:
        return str(display).replace("]", "\\]")
    return render_relative_markdown_link(vault_root, source, resolved, label=display)


def render_symbol_markdown_link(
    vault_root: Union[str, Path],
    source: Union[str, Path],
    symbol: str,
    label: Optional[str] = None,
) -> str:
    """Render a link to the canonical stock/entity note when it exists."""
    clean = _clean_target(symbol).strip()
    if not clean:
        return ""
    root = Path(vault_root).resolve()
    preferred = root / "30_Knowledge_Base" / "Stocks" / clean / f"{clean}.md"
    target = preferred if preferred.is_file() else clean
    return render_resolved_markdown_link(root, source, str(target), label=label or clean)


def default_vault_root() -> Path:
    return Path(os.getenv("OBSIDIAN_VAULT_PATH", "./memories")).resolve()


__all__ = [
    "default_vault_root",
    "encode_markdown_destination",
    "clear_vault_link_cache",
    "is_external_destination",
    "iter_internal_markdown_links",
    "iter_markdown_links",
    "LinkResolution",
    "MarkdownLink",
    "render_relative_markdown_link",
    "rewrite_internal_markdown_links",
    "render_resolved_markdown_link",
    "render_symbol_markdown_link",
    "resolve_vault_target",
    "resolve_vault_target_detailed",
]
