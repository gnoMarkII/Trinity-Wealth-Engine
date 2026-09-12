"""Navigation index builder for the Obsidian Vault V2 layout.

Generated links are emitted only for notes that exist. Directory links are
represented by a small ``index.md`` note when the directory is present, so an
Obsidian click always opens a document rather than a folder. Existing manual
text outside the generated marker is retained on regeneration.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

from tools.archivist.core import _atomic_write_text
from tools.archivist.maintenance_guard import assert_write_allowed
from tools.archivist.portable_links import render_relative_markdown_link
from tools.archivist.vault_paths import VaultPaths

logger = logging.getLogger(__name__)

_GEN_START = "<!-- vault-v2:generated:start -->"
_GEN_END = "<!-- vault-v2:generated:end -->"


def _note_link(
    vault_root: Path,
    target: Path,
    label: Optional[str] = None,
    *,
    source: Optional[Path] = None,
) -> str:
    source_path = source or (vault_root / "index.md")
    return render_relative_markdown_link(vault_root, source_path, target, label=label)


def _write_generated(path: Path, generated_body: str) -> None:
    """Merge a generated block while preserving manual content around it."""
    normalized = generated_body.rstrip()

    def split_frontmatter(value: str) -> tuple[str, str]:
        # Obsidian only recognizes YAML frontmatter when it is the first
        # document block. Keep the generated marker after frontmatter instead
        # of turning a navigation note into an unparseable Markdown file.
        if value.startswith("---"):
            marker = value.find("\n---", 3)
            if marker >= 0:
                end = marker + len("\n---")
                return value[:end], value[end:].lstrip("\n")
        return "", value

    frontmatter, body = split_frontmatter(normalized)
    full_block = (
        f"{frontmatter}\n{_GEN_START}\n{body}\n{_GEN_END}"
        if frontmatter
        else f"{_GEN_START}\n{body}\n{_GEN_END}"
    )
    body_block = f"{_GEN_START}\n{body}\n{_GEN_END}"

    if path.is_file():
        existing = path.read_text(encoding="utf-8")
        start = existing.find(_GEN_START)
        end = existing.find(_GEN_END, start + len(_GEN_START)) if start >= 0 else -1
        if start >= 0 and end >= 0:
            # R4's first live repair left the original frontmatter/body and
            # appended a second frontmatter block before the marker.  Keep
            # the durable first metadata block, discard that stale duplicate,
            # and make the generated section the sole body projection.
            first_fm_end = existing.find("\n---", 3) + len("\n---") if existing.startswith("---") else -1
            duplicate_frontmatter = existing[:start].count("schema_version:") > 1
            if duplicate_frontmatter and first_fm_end > len("---"):
                content = existing[:first_fm_end].rstrip() + "\n\n" + body_block + "\n"
            elif start == 0 or not existing[:start].strip():
                content = full_block + "\n"
            else:
                content = existing[:start].rstrip() + "\n\n" + body_block + existing[end + len(_GEN_END):]
        else:
            separator = "\n\n" if existing.strip() else ""
            content = existing.rstrip() + separator + full_block + "\n"
    else:
        content = full_block + "\n"
    _atomic_write_text(path, content)


def _ensure_directory_index(
    vault_root: Path,
    directory: Path,
    title: str,
    today: Optional[str] = None,
) -> Optional[Path]:
    if not directory.is_dir():
        return None
    target = directory / "index.md"
    children = sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() == ".md" and p.name != "index.md" and not p.name.startswith(".")
    )
    lines = [
        "---",
        "schema_version: 2",
        f"note_id: nav_{directory.relative_to(vault_root).as_posix().replace('/', '_').lower()}",
        f"document_key: navigation:v2:{directory.relative_to(vault_root).as_posix().replace('/', '_').lower()}",
        "entity_type: concept", "document_role: navigation", "search_scope: excluded",
        f"title: {title}",
        f"date: {today or datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
        "date_status: known",
        "source_verification_status: not_reviewed",
        "content_verification_status: not_reviewed",
        "scope: navigation",
        "generated_by: vault_v2_navigation_builder",
        "---",
        "",
        f"# {title}",
        "",
    ]
    if children:
        lines.extend(
            f"- {_note_link(vault_root, child, source=target)}"
            for child in children[:200]
        )
    else:
        lines.append("No notes yet.")
    _write_generated(target, "\n".join(lines))
    return target


def _dated_notes(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []

    def _sort_key(path: Path) -> tuple[str, str]:
        # Filesystem mtimes differ after snapshot restore and cross-device
        # copy.  Navigation projections must therefore sort by durable note
        # date/path, never by mtime.
        note_date = ""
        try:
            head = path.read_text(encoding="utf-8")[:2500]
            match = re.search(r"(?m)^date:\s*['\"]?(\d{4}-\d{2}-\d{2})", head)
            note_date = match.group(1) if match else ""
        except OSError:
            pass
        return note_date, path.relative_to(directory).as_posix()

    return sorted(
        (p for p in directory.rglob("*.md") if p.name != "index.md" and not p.name.startswith(".")),
        key=_sort_key,
        reverse=True,
    )


def _preserve_existing_section_links(
    source: Path,
    directory: Path,
    selected: list[Path],
) -> list[Path]:
    """Keep valid existing section links when a bounded hub is regenerated."""
    if not source.is_file():
        return selected
    preserved: list[Path] = []
    try:
        content = source.read_text(encoding="utf-8")
    except OSError:
        return selected
    for destination in re.findall(r"(?<!!)\[[^\]]+\]\(([^)]+)\)", content):
        destination = destination.split("#", 1)[0].strip()
        if not destination or re.match(r"^[a-z][a-z0-9+.-]*:", destination, re.IGNORECASE):
            continue
        target = (source.parent / destination).resolve()
        try:
            target.relative_to(directory.resolve())
        except ValueError:
            continue
        if target.is_file() and target.suffix.lower() == ".md":
            preserved.append(target)
    merged: list[Path] = []
    for path in [*selected, *preserved]:
        if path not in merged:
            merged.append(path)
    def _stable_key(path: Path) -> tuple[str, str]:
        note_date = ""
        try:
            head = path.read_text(encoding="utf-8")[:2500]
            match = re.search(r"(?m)^date:\s*['\"]?(\d{4}-\d{2}-\d{2})", head)
            note_date = match.group(1) if match else ""
        except OSError:
            pass
        return note_date, path.relative_to(directory).as_posix()
    return sorted(merged, key=_stable_key, reverse=True)


def build_navigation_indices(vault_root: Optional[Union[str, Path]] = None) -> dict[str, Path]:
    """Generate concise V2 navigation notes and return their paths."""
    vp = VaultPaths(vault_root)
    root = vp.root
    assert_write_allowed(root)
    index_dir = root / "00_Index"
    index_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    stocks_dir = root / "30_Knowledge_Base" / "Stocks"
    macro_dir = root / "30_Knowledge_Base" / "Macroeconomics"
    news_dir = root / "30_Knowledge_Base" / "News"
    sources_dir = root / "30_Knowledge_Base" / "NotebookLM_Sources"
    audio_dir = root / "30_Knowledge_Base" / "NotebookLM_Audio"

    tickers = sorted(
        d.name for d in stocks_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
    ) if stocks_dir.is_dir() else []

    stock_targets: dict[str, tuple[Optional[Path], Optional[Path], Optional[Path]]] = {}
    for ticker in tickers:
        ticker_dir = stocks_dir / ticker
        hub = ticker_dir / f"{ticker}.md"
        earnings_index = _ensure_directory_index(root, ticker_dir / "Earnings", f"{ticker} Earnings Calls", today)
        analysis_index = _ensure_directory_index(root, ticker_dir / "Analysis", f"{ticker} Analysis", today)
        stock_targets[ticker] = (hub if hub.is_file() else None, earnings_index, analysis_index)

    macro_notes = _preserve_existing_section_links(index_dir / "Macro_Hub.md", macro_dir, _dated_notes(macro_dir)[:15])
    news_notes = _preserve_existing_section_links(index_dir / "News_Hub.md", news_dir, _dated_notes(news_dir)[:15])

    generated: dict[str, Path] = {}

    home_lines = [
        "---", "schema_version: 2", "note_id: nav_home_v2", "document_key: navigation:v2:home", "entity_type: concept", "document_role: navigation", "search_scope: excluded",
        "title: Obsidian Vault Home", f"date: {today}", "date_status: known", "source_verification_status: not_reviewed", "content_verification_status: not_reviewed", "scope: navigation",
        "generated_by: vault_v2_navigation_builder", "---", "",
        "# Investment Knowledge Base — Home", "",
        f"> Vault V2 active · last indexed `{today}`", "",
        "## Knowledge Domains", "",
        f"- {_note_link(root, index_dir / 'Stocks_Hub.md', 'Equity & Stock Intelligence', source=index_dir / 'Home.md')}",
        f"- {_note_link(root, index_dir / 'Macro_Hub.md', 'Macroeconomics & Asset Allocation', source=index_dir / 'Home.md')}",
        f"- {_note_link(root, index_dir / 'News_Hub.md', 'News & Expert Insights', source=index_dir / 'Home.md')}",
        f"- {_note_link(root, index_dir / 'Audio_Hub.md', 'Audio Briefing Room', source=index_dir / 'Home.md')}",
        "", "## Quick Navigation", "",
    ]
    if tickers:
        labels = []
        for ticker in tickers[:10]:
            hub = stock_targets[ticker][0]
            labels.append(_note_link(root, hub, ticker, source=index_dir / 'Home.md') if hub else ticker)
        home_lines.append(f"**Tracked equities ({len(tickers)})**: {', '.join(labels)}")
        if len(tickers) > 10:
            home_lines.append(f"…and {len(tickers) - 10} more in {_note_link(root, index_dir / 'Stocks_Hub.md', 'Stocks Hub', source=index_dir / 'Home.md')}.")
    home_lines.extend([
        "", "## Architecture & Maintenance", "",
        "- Canonical format: Obsidian Vault V2", "- Revisions: `40_Archive/Revisions/`",
        "- Read model: external catalog generation (vault runtime)", "",
    ])
    while len("\n".join(home_lines)) > 4000 and len(home_lines) > 18:
        home_lines.pop(-3 if home_lines[-1] == "" else -1)
    home_file = index_dir / "Home.md"
    _write_generated(home_file, "\n".join(home_lines))
    generated["Home"] = home_file

    stock_lines = [
        "---", "schema_version: 2", "note_id: nav_stocks_hub_v2", "document_key: navigation:v2:stocks", "entity_type: concept", "document_role: navigation", "search_scope: excluded",
        "title: Stocks & Equities Hub", f"date: {today}", "date_status: known", "source_verification_status: not_reviewed", "content_verification_status: not_reviewed", "scope: navigation",
        "generated_by: vault_v2_navigation_builder", "---", "",
        "# Equity & Stock Intelligence Hub", "", f"Total companies monitored: **{len(tickers)}**", "",
    ]
    for ticker in tickers[:100]:
        hub, earnings_index, analysis_index = stock_targets[ticker]
        destination = _note_link(root, hub, ticker, source=index_dir / 'Stocks_Hub.md') if hub else f"**{ticker}**"
        extras = []
        if earnings_index:
            extras.append(_note_link(root, earnings_index, "Earnings Calls", source=index_dir / 'Stocks_Hub.md'))
        if analysis_index:
            extras.append(_note_link(root, analysis_index, "Analysis", source=index_dir / 'Stocks_Hub.md'))
        stock_lines.append(f"- {destination}" + (" · " + " · ".join(extras) if extras else ""))
    while len("\n".join(stock_lines)) > 6000 and len(stock_lines) > 12:
        stock_lines.pop()
    stocks_file = index_dir / "Stocks_Hub.md"
    _write_generated(stocks_file, "\n".join(stock_lines))
    generated["Stocks_Hub"] = stocks_file

    macro_lines = [
        "---", "schema_version: 2", "note_id: nav_macro_hub_v2", "document_key: navigation:v2:macro", "entity_type: concept", "document_role: navigation", "search_scope: excluded",
        "title: Macroeconomics & Strategy Hub", f"date: {today}", "date_status: known", "source_verification_status: not_reviewed", "content_verification_status: not_reviewed", "scope: navigation",
        "generated_by: vault_v2_navigation_builder", "---", "", "# Macroeconomics & Strategy Hub", "",
        "## Recent strategic directions and snapshots", "",
    ]
    macro_lines.extend(f"- {_note_link(root, p, source=index_dir / 'Macro_Hub.md')}" for p in macro_notes)
    macro_file = index_dir / "Macro_Hub.md"
    _write_generated(macro_file, "\n".join(macro_lines))
    generated["Macro_Hub"] = macro_file

    news_lines = [
        "---", "schema_version: 2", "note_id: nav_news_hub_v2", "document_key: navigation:v2:news", "entity_type: concept", "document_role: navigation", "search_scope: excluded",
        "title: News & YouTube Insights Hub", f"date: {today}", "date_status: known", "source_verification_status: not_reviewed", "content_verification_status: not_reviewed", "scope: navigation",
        "generated_by: vault_v2_navigation_builder", "---", "", "# News & Video Insights Hub", "",
        "## Recent articles and insights", "",
    ]
    news_lines.extend(f"- {_note_link(root, p, source=index_dir / 'News_Hub.md')}" for p in news_notes)
    news_file = index_dir / "News_Hub.md"
    _write_generated(news_file, "\n".join(news_lines))
    generated["News_Hub"] = news_file

    audio_index = _ensure_directory_index(root, sources_dir, "NotebookLM Source Books")
    audio_lines = [
        "---", "schema_version: 2", "note_id: nav_audio_hub_v2", "document_key: navigation:v2:audio", "entity_type: concept", "document_role: navigation", "search_scope: excluded",
        "title: NotebookLM & Audio Briefing Hub", f"date: {today}", "date_status: known", "source_verification_status: not_reviewed", "content_verification_status: not_reviewed", "scope: navigation",
        "generated_by: vault_v2_navigation_builder", "---", "", "# NotebookLM & Audio Hub", "",
        "- Source books: `30_Knowledge_Base/NotebookLM_Sources/`",
        "- Generated audio: `30_Knowledge_Base/NotebookLM_Audio/`",
        "- Manifests retain provider IDs and recovery status.", "",
    ]
    if audio_index:
        audio_lines.append(f"- {_note_link(root, audio_index, 'Open source catalog', source=index_dir / 'Audio_Hub.md')}")
    if audio_dir.is_dir():
        audio_files = _dated_notes(audio_dir)[:20]
        audio_lines.extend(f"- {_note_link(root, p, source=index_dir / 'Audio_Hub.md')}" for p in audio_files)
    audio_file = index_dir / "Audio_Hub.md"
    _write_generated(audio_file, "\n".join(audio_lines))
    generated["Audio_Hub"] = audio_file

    # Keep the historical root Home useful without replacing its manual
    # sections. The generated link is a small additive bridge to 00_Index.
    root_index = root / "index.md"
    if root_index.is_file():
        existing = root_index.read_text(encoding="utf-8")
        bridge = _note_link(root, home_file, "Open V2 Home", source=root_index)
        if bridge not in existing:
            _atomic_write_text(root_index, existing.rstrip() + f"\n\n- {bridge}\n")

    logger.info("Navigation indices built successfully in %s", index_dir)
    return generated
