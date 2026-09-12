"""Tests for VaultPaths, layout resolution, path security, and legacy candidate routing."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tools.archivist.vault_paths import VaultPaths
from tools.archivist.vault_policy import is_searchable, sanitize_filename


def test_vault_paths_security_and_traversal(tmp_path: Path) -> None:
    """safe_resolve must reject path traversals outside the vault root."""
    vault = VaultPaths(root=tmp_path / "vault1")
    
    # Valid relative path
    valid_p = vault.safe_resolve("30_Knowledge_Base/Stocks/AAPL/AAPL.md")
    assert str(valid_p).startswith(str(vault.root))

    # Path traversal attack
    with pytest.raises(ValueError, match="Path traversal outside vault root"):
        vault.safe_resolve("../outside.md")

    with pytest.raises(ValueError, match="Path traversal outside vault root"):
        vault.safe_resolve("sub/../../escape.txt")


def test_vault_isolation(tmp_path: Path) -> None:
    """Two Vault instances must not share configuration, roots, or layouts."""
    root1 = tmp_path / "vault1"
    root2 = tmp_path / "vault2"
    root1.mkdir()
    root2.mkdir()

    # Configure root1 as V2
    sys1 = root1 / ".system"
    sys1.mkdir()
    (sys1 / "vault_config.json").write_text(json.dumps({"layout_version": 2}), encoding="utf-8")

    v1 = VaultPaths(root=root1)
    v2 = VaultPaths(root=root2)

    assert v1.root != v2.root
    assert v1.layout_version == 2
    assert v2.layout_version == 1  # Missing config defaults to 1


def test_note_path_generation_all_types(tmp_path: Path) -> None:
    """note_path must compute canonical V2 paths matching the specification table."""
    vp = VaultPaths(root=tmp_path / "test_vault")

    # 1. Stock Hub
    hub_p = vp.note_path({"entity_type": "stock_hub", "ticker": "FTNT"})
    assert hub_p.as_posix().endswith("30_Knowledge_Base/Stocks/FTNT/FTNT.md")

    # 2. Equity Analysis
    ana_p = vp.note_path(
        {"entity_type": "equity_analysis", "ticker": "FTNT", "date": "2026-09-05"},
        filename="2026-09-05 FTNT Equity Analysis",
    )
    assert ana_p.as_posix().endswith("30_Knowledge_Base/Stocks/FTNT/Analysis/2026-09-05 FTNT Equity Analysis.md")

    # 3. Quant Snapshot
    quant_p = vp.note_path(
        {"entity_type": "quant_snapshot", "ticker": "FTNT", "date": "2026-09-05"},
        filename="2026-09-05 FTNT Quant Snapshot",
    )
    assert quant_p.as_posix().endswith("30_Knowledge_Base/Stocks/FTNT/Quant/2026-09-05 FTNT Quant Snapshot.md")

    # 4. Earnings Call
    ec_p = vp.note_path(
        {"entity_type": "earnings_call", "ticker": "FTNT", "period": "2026-Q2"},
        filename="2026-Q2 FTNT Earnings Call",
    )
    assert ec_p.as_posix().endswith("30_Knowledge_Base/Stocks/FTNT/Earnings/2026-Q2 FTNT Earnings Call.md")

    # 5. News & Articles (with and without date)
    news_p = vp.note_path(
        {"entity_type": "company_news", "date": "2026-09-06"},
        filename="2026-09-06 Tech Update",
    )
    assert news_p.as_posix().endswith("30_Knowledge_Base/News/2026/09/2026-09-06 Tech Update.md")

    news_undated = vp.note_path(
        {"entity_type": "company_news"},
        filename="Undated News",
    )
    assert news_undated.as_posix().endswith("30_Knowledge_Base/News/_undated/Undated News.md")

    # 6. YouTube Summaries
    yt_p = vp.note_path(
        {"entity_type": "youtube_summary", "date": "2026-09-05"},
        filename="Market Outlook",
    )
    assert yt_p.as_posix().endswith("30_Knowledge_Base/YouTube_Summaries/2026/09/Market Outlook.md")

    # 7. Macro Snapshot
    macro_snap_p = vp.note_path(
        {"entity_type": "macro_snapshot", "as_of": "2026-09-06"},
        filename="Daily Snapshot",
    )
    assert macro_snap_p.as_posix().endswith("30_Knowledge_Base/Macroeconomics/Daily_Snapshots/2026/09/Daily Snapshot.md")

    # 8. Macro Strategy
    macro_strat_p = vp.note_path(
        {"entity_type": "macro_strategy", "as_of": "2026-09-06"},
        filename="Daily Macro Strategy",
    )
    assert macro_strat_p.as_posix().endswith("30_Knowledge_Base/Macroeconomics/Strategies/2026/09/Daily Macro Strategy.md")

    # 9. Indicator Series
    indicator_p = vp.note_path(
        {"entity_type": "indicator_series"},
        filename="Yield_Spread.json",
    )
    assert indicator_p.as_posix().endswith("30_Knowledge_Base/Macroeconomics/Indicator_Series/Yield_Spread.json")

    # 10. Briefing Book
    briefing_p = vp.note_path(
        {"entity_type": "briefing_book", "authored_date": "2026-09-06"},
        filename="Daily Briefing",
    )
    assert briefing_p.as_posix().endswith("30_Knowledge_Base/NotebookLM_Sources/2026/09/Daily Briefing.md")

    # 11. Revision snapshot path
    rev_p = vp.revision_path(note_id="note_123", revision_id="rev_abc", filename="doc.md")
    assert rev_p.as_posix().endswith("40_Archive/Revisions/note_123/rev_abc/doc.md")


def test_vault_policy_searchability() -> None:
    """is_searchable must exclude revisions, system files, trash, and non-markdown files."""
    # Active notes are searchable
    assert is_searchable("30_Knowledge_Base/Stocks/FTNT/FTNT.md") is True
    assert is_searchable("30_Knowledge_Base/News/2026/09/Update.md") is True

    # Revisions in 40_Archive/Revisions/ are EXCLUDED from current search
    assert is_searchable("40_Archive/Revisions/note_1/rev_1/FTNT.md") is False

    # System files and excluded directories
    assert is_searchable("index.md") is False
    assert is_searchable("Portfolio_Holdings.md") is False
    assert is_searchable(".trash/Deleted.md") is False
    assert is_searchable(".obsidian/workspace.json") is False
    assert is_searchable(".system/vault_config.json") is False
    assert is_searchable("30_Knowledge_Base/Stocks/FTNT/Analysis/FTNT.json") is False  # Non-markdown
