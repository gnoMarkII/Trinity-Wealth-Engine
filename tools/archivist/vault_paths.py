"""Centralized Vault Path Management for Obsidian Vault V2.

Resolves deterministic canonical note paths, handles layout version detection,
prevents path traversal outside vault root, and provides legacy V1 candidate paths.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Optional, Union

from schemas.knowledge_metadata import CommonNoteMetadata
from tools.archivist.vault_policy import sanitize_filename


def _extract_year_month(date_val: Optional[str]) -> tuple[str, str]:
    """Extracts YYYY and MM from date string. Returns ('_undated', '') if unknown."""
    if not date_val:
        return "_undated", ""
    date_str = str(date_val).strip()
    m = re.match(r"^(\d{4})-(\d{2})(?:-\d{2})?", date_str)
    if m:
        return m.group(1), m.group(2)
    m_year = re.match(r"^(\d{4})", date_str)
    if m_year:
        return m_year.group(1), "01"
    return "_undated", ""


class VaultPaths:
    """Manages paths within an Obsidian Vault instance with injectable root."""

    def __init__(
        self,
        root: Union[str, Path, None] = None,
        layout_version: Optional[int] = None,
    ) -> None:
        if root is not None:
            self._root = Path(root).resolve()
        else:
            # Fallback to environment variable or default ./memories
            env_path = os.getenv("OBSIDIAN_VAULT_PATH", "./memories")
            self._root = Path(env_path).resolve()

        if layout_version is not None:
            self._layout_version = layout_version
        else:
            self._layout_version = self._detect_layout_version()

    @property
    def root(self) -> Path:
        return self._root

    @property
    def layout_version(self) -> int:
        return self._layout_version

    def _detect_layout_version(self) -> int:
        """Reads layout from .system/vault_config.json. If missing, default is V1."""
        config_path = self._root / ".system" / "vault_config.json"
        if not config_path.exists():
            return 1
        try:
            with config_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return int(data.get("layout_version", 1))
        except Exception:
            return 1

    def safe_resolve(self, relative_path: Union[str, Path]) -> Path:
        """Safely resolves a relative path against the vault root.

        Raises ValueError if path traversal outside the vault root is attempted.
        """
        p = Path(relative_path)
        if p.is_absolute():
            # If absolute, verify it is inside self._root
            resolved = p.resolve()
        else:
            resolved = (self._root / p).resolve()

        try:
            resolved.relative_to(self._root)
        except ValueError:
            raise ValueError(f"Path traversal outside vault root is forbidden: {relative_path}")

        return resolved

    def revision_path(self, note_id: str, revision_id: str, filename: str) -> Path:
        """Returns frozen revision snapshot path in 40_Archive/Revisions/NOTE_ID/REVISION_ID/."""
        safe_note = sanitize_filename(note_id)
        safe_rev = sanitize_filename(revision_id)
        safe_fn = sanitize_filename(filename)
        rel = Path("40_Archive") / "Revisions" / safe_note / safe_rev / safe_fn
        return self.safe_resolve(rel)

    def note_path(
        self,
        metadata: Union[dict[str, Any], CommonNoteMetadata],
        filename: Optional[str] = None,
    ) -> Path:
        """Returns canonical file path for note according to Vault V2 rules."""
        if isinstance(metadata, CommonNoteMetadata):
            meta_dict = metadata.model_dump(mode="python")
        else:
            meta_dict = dict(metadata)

        entity_type = str(meta_dict.get("entity_type", "concept")).strip().lower()
        title = meta_dict.get("title", filename or "Untitled")
        date_val = (
            meta_dict.get("date")
            or meta_dict.get("published_date")
            or meta_dict.get("analysis_date")
            or meta_dict.get("as_of")
            or meta_dict.get("authored_date")
        )
        ticker = str(meta_dict.get("ticker", "")).strip().upper()
        if not ticker and "tickers" in meta_dict and meta_dict["tickers"]:
            ticker = str(meta_dict["tickers"][0]).strip().upper()

        safe_title = sanitize_filename(filename or title)
        if not safe_title.endswith(".md"):
            safe_title = f"{safe_title}.md"

        kb = Path("30_Knowledge_Base")

        # 1. Stock Hub
        if entity_type == "stock_hub":
            target_ticker = ticker or sanitize_filename(Path(safe_title).stem)
            stem_upper = Path(safe_title).stem.upper()
            if not filename and (stem_upper == "UNTITLED" or stem_upper == target_ticker):
                return self.safe_resolve(kb / "Stocks" / target_ticker / f"{target_ticker}.md")
            if stem_upper != target_ticker:
                return self.safe_resolve(kb / "Stocks" / target_ticker / "Analysis" / safe_title)
            return self.safe_resolve(kb / "Stocks" / target_ticker / f"{target_ticker}.md")

        # 2. Equity Analysis
        elif entity_type == "equity_analysis":
            target_ticker = ticker or "UNKNOWN"
            return self.safe_resolve(kb / "Stocks" / target_ticker / "Analysis" / safe_title)

        # 3. Quant Snapshot
        elif entity_type in ("quant_snapshot", "equity_quant_snapshot"):
            target_ticker = ticker or "UNKNOWN"
            return self.safe_resolve(kb / "Stocks" / target_ticker / "Quant" / safe_title)

        # 4. Earnings Call
        elif entity_type == "earnings_call":
            target_ticker = ticker or "UNKNOWN"
            return self.safe_resolve(kb / "Stocks" / target_ticker / "Earnings" / safe_title)

        # 5. News & Articles
        elif entity_type in ("company_news", "article", "article_note"):
            year, month = _extract_year_month(date_val)
            if year == "_undated":
                return self.safe_resolve(kb / "News" / "_undated" / safe_title)
            return self.safe_resolve(kb / "News" / year / month / safe_title)

        # 6. YouTube Summaries
        elif entity_type in ("youtube_summary", "youtube_insight"):
            year, month = _extract_year_month(date_val)
            if year == "_undated":
                return self.safe_resolve(kb / "YouTube_Summaries" / "_undated" / safe_title)
            return self.safe_resolve(kb / "YouTube_Summaries" / year / month / safe_title)

        # 7. Macro Daily Snapshot
        elif entity_type in ("macro_snapshot", "macro_country", "macro_global", "macro_regional"):
            year, month = _extract_year_month(date_val)
            if year == "_undated":
                return self.safe_resolve(kb / "Macroeconomics" / "Daily_Snapshots" / "_undated" / safe_title)
            return self.safe_resolve(kb / "Macroeconomics" / "Daily_Snapshots" / year / month / safe_title)

        # 8. Macro Strategy
        elif entity_type == "macro_strategy":
            year, month = _extract_year_month(date_val)
            if year == "_undated":
                return self.safe_resolve(kb / "Macroeconomics" / "Strategies" / "_undated" / safe_title)
            return self.safe_resolve(kb / "Macroeconomics" / "Strategies" / year / month / safe_title)

        # 9. Indicator Series
        elif entity_type == "indicator_series":
            series_name = filename or safe_title
            if not series_name.endswith(".json") and not series_name.endswith(".md"):
                series_name = f"{series_name}.json"
            return self.safe_resolve(kb / "Macroeconomics" / "Indicator_Series" / series_name)

        # 10. Briefing Book (NotebookLM Sources)
        elif entity_type == "briefing_book" or "Daily Briefing" in safe_title:
            year, month = _extract_year_month(date_val)
            if year == "_undated":
                return self.safe_resolve(kb / "NotebookLM_Sources" / "_undated" / safe_title)
            return self.safe_resolve(kb / "NotebookLM_Sources" / year / month / safe_title)

        # 11. Book Note
        elif entity_type in ("book_note", "book"):
            return self.safe_resolve(kb / "Books" / safe_title)

        # 12. Portfolio projections.  These are managed, excluded-from-AI
        # Markdown views of transactional runtime state; the portfolio id is
        # part of routing, never inferred from a client filesystem path.
        elif entity_type in ("portfolio_state", "holding", "watchlist_item", "goal"):
            portfolio_id = sanitize_filename(str(meta_dict.get("portfolio_id") or "default"))
            portfolio_root = Path("20_Portfolio_Management") / "Current_Holdings" / "Portfolios" / portfolio_id
            if entity_type == "portfolio_state":
                return self.safe_resolve(portfolio_root / "Portfolio_Holdings.md")
            if entity_type == "holding":
                return self.safe_resolve(portfolio_root / "Holdings" / safe_title)
            if entity_type == "watchlist_item":
                return self.safe_resolve(portfolio_root / "Watchlist_Items" / safe_title)
            return self.safe_resolve(Path("20_Portfolio_Management") / "Goals" / "Items" / safe_title)

        # Default: Concepts / General
        else:
            return self.safe_resolve(kb / "Concepts" / safe_title)

    def legacy_candidates(
        self,
        metadata: Union[dict[str, Any], CommonNoteMetadata],
        filename: Optional[str] = None,
    ) -> list[Path]:
        """Returns list of candidate paths where this note might exist under V1 layout."""
        if isinstance(metadata, CommonNoteMetadata):
            meta_dict = metadata.model_dump(mode="python")
        else:
            meta_dict = dict(metadata)

        entity_type = str(meta_dict.get("entity_type", "")).strip().lower()
        title = meta_dict.get("title", filename or "Untitled")
        ticker = str(meta_dict.get("ticker", "")).strip().upper()
        safe_title = sanitize_filename(filename or title)
        if not safe_title.endswith(".md"):
            safe_title = f"{safe_title}.md"

        candidates: list[Path] = []
        kb = self._root / "30_Knowledge_Base"

        if entity_type in ("stock_hub", "equity_analysis", "quant_snapshot", "equity_quant_snapshot", "earnings_call"):
            if ticker:
                candidates.append(kb / "Stocks" / ticker / safe_title)
                candidates.append(kb / "Stocks" / f"{ticker}.md")
                candidates.append(kb / "Stocks" / ticker / "Quant" / safe_title)
                candidates.append(kb / "Stocks" / ticker / "Analysis" / safe_title)
            candidates.append(kb / "Stocks" / safe_title)
            candidates.append(kb / "Equities" / safe_title)
            candidates.append(kb / "Earnings_Calls" / safe_title)

        elif entity_type in ("company_news", "article"):
            candidates.append(kb / "News" / safe_title)
            candidates.append(kb / "News" / "Inbox" / safe_title)

        elif entity_type == "youtube_summary":
            candidates.append(kb / "YouTube_Summaries" / safe_title)
            candidates.append(kb / "YouTube_Summaries" / "Inbox" / safe_title)

        elif entity_type in ("macro_strategy", "macro_snapshot", "macro_country", "macro_global", "macro_regional"):
            candidates.append(kb / "Macroeconomics" / "Daily_Snapshots" / safe_title)
            candidates.append(kb / "Macroeconomics" / "Strategies" / safe_title)
            candidates.append(kb / "Strategies" / safe_title)

        elif entity_type in ("book_note", "book"):
            candidates.append(kb / "Books" / safe_title)

        elif entity_type == "briefing_book" or "Daily Briefing" in safe_title:
            candidates.append(self._root / "NotebookLM_Sources" / safe_title)
            candidates.append(kb / "NotebookLM_Sources" / safe_title)

        # General concept fallback
        candidates.append(kb / "Concepts" / safe_title)

        return candidates
