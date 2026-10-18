"""Equity NotebookLM Bundle Builder — Assembles frozen research bundle on disk.

Hexagonal Architecture Invariant:
Implements EquityExportBundlePort. Operates as a driven adapter in tools/ compiling
an EquityCorpusSnapshot into atomic markdown files, inventory, and corpus snapshot.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from application.equity.notebooklm_export_ports import (
    EquityCorpusSnapshot,
    EquityExportBundlePort,
)
from tools.market.notebooklm_bundle_formatter import (
    format_earnings_calls,
    format_financials,
    format_historical_analyses,
    format_insider_ownership,
    format_latest_analysis,
    format_market_derivatives,
    format_news_and_references,
    format_price_and_notes,
    format_research_guide,
    format_structured_appendix,
)


def _compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class EquityExportBundleBuilder(EquityExportBundlePort):
    """Builds and seals a deterministic frozen equity export bundle on disk."""

    def __init__(self, export_root: Optional[Path] = None) -> None:
        configured = (
            export_root
            or os.getenv("EQUITY_NOTEBOOKLM_EXPORT_DIR")
            or "data/notebooklm_equity_exports"
        )
        self._export_root = Path(configured).resolve()

    @property
    def export_root(self) -> Path:
        return self._export_root

    def build_bundle(
        self, export_id: str, snapshot: EquityCorpusSnapshot
    ) -> Tuple[Path, str, Dict[str, Any]]:
        bundle_dir = self._export_root / export_id
        sources_dir = bundle_dir / "sources"
        sources_dir.mkdir(parents=True, exist_ok=True)

        # 1. Generate text for each part
        parts: List[Tuple[str, str, str]] = [
            ("00-research-guide.md", "Research Guide & Inventory", format_research_guide(snapshot)),
            ("01-latest-analysis.md", f"{snapshot.ticker} Investment Thesis & Scorecard", format_latest_analysis(snapshot)),
            ("02-analysis-history.md", f"{snapshot.ticker} Historical Analyses", format_historical_analyses(snapshot)),
            ("03-financials.md", f"{snapshot.ticker} Financials & Filings", format_financials(snapshot)),
            ("04-insider-ownership.md", f"{snapshot.ticker} Insider & Ownership", format_insider_ownership(snapshot)),
            ("05-earnings-calls.md", f"{snapshot.ticker} Earnings Calls", format_earnings_calls(snapshot)),
            ("06-news.md", f"{snapshot.ticker} Curated News", format_news_and_references(snapshot)),
            ("07-market-derivatives.md", f"{snapshot.ticker} Options & Derivatives", format_market_derivatives(snapshot)),
            ("08-price-and-notes.md", f"{snapshot.ticker} Market Price & User Notes", format_price_and_notes(snapshot)),
            ("09-structured-appendix.md", f"{snapshot.ticker} Structured Appendix", format_structured_appendix(snapshot)),
        ]

        source_inventory: List[Dict[str, Any]] = []

        # 2. Write markdown files and compute cumulative rendered files hash
        files_hasher = hashlib.sha256()
        for filename, title, content in parts:
            file_path = sources_dir / filename
            content_bytes = content.encode("utf-8")
            file_path.write_bytes(content_bytes)

            file_hash = _compute_sha256(content_bytes)
            files_hasher.update(filename.encode("utf-8"))
            files_hasher.update(file_hash.encode("utf-8"))

            source_inventory.append({
                "file_name": filename,
                "relative_path": f"sources/{filename}",
                "title": title,
                "size_bytes": len(content_bytes),
                "sha256": file_hash,
            })

        bundle_files_hash = files_hasher.hexdigest()

        # Semantic data hash independent of fleeting snapshot timestamp
        semantic_payload = {
            "ticker": snapshot.ticker,
            "market": snapshot.market,
            "latest_analysis": snapshot.latest_analysis,
            "historical_analyses": snapshot.historical_analyses,
            "financials": snapshot.financials,
            "sec_financials": snapshot.sec_financials,
            "sec_insider_trades": snapshot.sec_insider_trades,
            "earnings_calls": snapshot.earnings_calls,
            "news": snapshot.news,
            "options_chain": snapshot.options_chain,
            "max_pain": snapshot.max_pain,
            "consensus": snapshot.consensus,
            "notes": snapshot.notes,
            "portfolio_holding": snapshot.portfolio_holding,
        }
        content_hash = hashlib.sha256(
            json.dumps(semantic_payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

        # 3. Write corpus.json
        corpus_data = {
            "ticker": snapshot.ticker,
            "market": snapshot.market,
            "snapshot_at": snapshot.snapshot_at,
            "content_hash": content_hash,
            "latest_analysis": snapshot.latest_analysis,
            "historical_analyses": snapshot.historical_analyses,
            "financials": snapshot.financials,
            "sec_financials": snapshot.sec_financials,
            "sec_insider_trades": snapshot.sec_insider_trades,
            "earnings_calls": snapshot.earnings_calls,
            "news": snapshot.news,
            "options_chain": snapshot.options_chain,
            "max_pain": snapshot.max_pain,
            "consensus": snapshot.consensus,
            "notes": snapshot.notes,
            "portfolio_holding": snapshot.portfolio_holding,
            "metadata": snapshot.metadata,
        }
        corpus_path = bundle_dir / "corpus.json"
        corpus_path.write_text(json.dumps(corpus_data, indent=2, default=str), encoding="utf-8")

        # 4. Write inventory.json
        inventory: Dict[str, Any] = {
            "export_id": export_id,
            "ticker": snapshot.ticker,
            "market": snapshot.market,
            "snapshot_at": snapshot.snapshot_at,
            "content_hash": content_hash,
            "bundle_files_hash": bundle_files_hash,
            "total_sources": len(source_inventory),
            "sources": source_inventory,
            "counts": snapshot.metadata.get("counts", {}),
            "warnings": snapshot.metadata.get("warnings", []),
            "has_latest_analysis": snapshot.metadata.get(
                "has_latest_analysis", bool(snapshot.latest_analysis)
            ),
            "has_financials": snapshot.metadata.get(
                "has_financials",
                bool(snapshot.financials or snapshot.sec_financials or snapshot.latest_analysis),
            ),
            "has_sec_filings": snapshot.metadata.get(
                "has_sec_filings",
                bool(snapshot.sec_financials or snapshot.metadata.get("has_sec_data")),
            ),
            "has_insider_trades": snapshot.metadata.get(
                "has_insider_trades",
                bool(snapshot.sec_insider_trades or snapshot.metadata.get("has_sec_data")),
            ),
            "has_sec_data": snapshot.metadata.get("has_sec_data", False),
            "has_options_data": snapshot.metadata.get("has_options_data", False),
            "has_portfolio_holding": snapshot.metadata.get("has_portfolio_holding", False),
        }
        inventory_path = bundle_dir / "inventory.json"
        inventory_path.write_text(json.dumps(inventory, indent=2), encoding="utf-8")

        return bundle_dir, content_hash, inventory
