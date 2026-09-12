"""Link Resolver for Obsidian Vault V2.

Resolves wikilinks [[target]] and standard markdown links [text](target)
to canonical relative paths using the note catalog and vault paths.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Union

from application.knowledge.ports import LinkResolverPort, NoteCatalogPort
from tools.archivist.vault_paths import VaultPaths


class VaultLinkResolver(LinkResolverPort):
    """Resolves wikilinks and paths within the vault."""

    def __init__(
        self,
        catalog: Optional[NoteCatalogPort] = None,
        vault_root: Optional[Union[str, Path]] = None,
    ) -> None:
        self._catalog = catalog
        self._vp = VaultPaths(vault_root)

    def resolve_target(self, link_target: str, context_path: Optional[str] = None) -> Optional[str]:
        """Resolves link target into a canonical relative path (posix format) or None if unresolved."""
        if not link_target:
            return None

        clean = link_target.strip()
        # Strip wikilink brackets if present [[target]]
        if clean.startswith("[[") and clean.endswith("]]"):
            clean = clean[2:-2]

        # Strip display alias [[target|alias]]
        if "|" in clean:
            clean = clean.split("|")[0].strip()

        # Strip section anchor [[target#heading]]
        if "#" in clean:
            clean = clean.split("#")[0].strip()

        if not clean:
            return None

        # 1. If context_path given, check relative path resolution first
        if context_path and ("/" in clean or "\\" in clean or clean.startswith(".")):
            ctx = Path(context_path).parent
            candidate = (ctx / clean).as_posix()
            if not candidate.endswith(".md"):
                candidate = f"{candidate}.md"
            if self._catalog:
                entry = self._catalog.get_by_path(candidate)
                if entry:
                    return entry.relative_path
            target_file = self._vp.root / candidate
            if target_file.exists():
                return candidate

        # 2. Check catalog exact relative_path match
        candidate_rel = clean if clean.endswith(".md") else f"{clean}.md"
        if self._catalog:
            entry = self._catalog.get_by_path(candidate_rel)
            if entry:
                return entry.relative_path

        # 3. Check if clean is a ticker symbol (e.g. "FTNT")
        if re.match(r"^[A-Z0-9.\-]{1,10}$", clean, re.IGNORECASE):
            ticker_upper = clean.upper()
            if self._catalog:
                entries = self._catalog.find_notes(ticker=ticker_upper, entity_type="stock_hub", limit=1)
                if entries:
                    return entries[0].relative_path

            # Fallback path for ticker Hub
            hub_path = Path("30_Knowledge_Base") / "Stocks" / ticker_upper / f"{ticker_upper}.md"
            if (self._vp.root / hub_path).exists():
                return hub_path.as_posix()

        # 4. Search by filename or title in catalog
        stem = Path(clean).stem
        if self._catalog:
            # Look up across catalog by filename ending
            with self._catalog._get_conn() as conn:  # type: ignore[attr-defined]
                cur = conn.execute(
                    "SELECT relative_path FROM note_catalog WHERE relative_path LIKE ? OR title = ? LIMIT 1;",
                    (f"%/{stem}.md", stem),
                )
                row = cur.fetchone()
                if row:
                    return row["relative_path"]

        # 5. Fallback filesystem check using legacy candidates
        for cand in self._vp.legacy_candidates({"entity_type": "concept", "title": stem}):
            if cand.exists():
                try:
                    return cand.relative_to(self._vp.root).as_posix()
                except ValueError:
                    pass

        return None
