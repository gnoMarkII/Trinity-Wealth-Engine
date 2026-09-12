import json
import sqlite3
from pathlib import Path
import pytest

from schemas.micro_quant_schemas import MicroQuantOutput, QuantSignals, EquitySentimentContext
from tools.archivist.catalog_adapter import SqliteNoteCatalogAdapter
from tools.archivist.vault_maintenance import run_catalog_maintenance, detect_orphaned_sidecars
from tools.market.equity_sidecar import write_equity_sidecar
from tools.market.adapters.equity_vault_query_adapter import EquityVaultQueryAdapter


def _dummy_output(ticker: str = "TEST", date: str = "2026-09-07") -> MicroQuantOutput:
    return MicroQuantOutput(
        ticker=ticker,
        market="US",
        analysis_date=date,
        base_case_summary="Test summary",
        narrative_analysis="Test narrative",
        quant_signals=QuantSignals(
            evaluated_at=f"{date}T10:00:00Z",
            ticker=ticker,
            market="US",
            composite_score=85.0,
            overall_stance="BULLISH",
            fundamental_score=80.0,
            technical_score=90.0,
            valuation_score=75.0,
            sentiment_score=85.0,
            risk_score=20.0,
        ),
        sentiment_context=EquitySentimentContext(
            evaluated_at=f"{date}T10:00:00Z",
            market_sentiment="bullish",
            sources_summary="Summary",
        ),
    )


def test_sidecar_catalog_indexing_and_fast_query(tmp_path: Path):
    """Verifies write_equity_sidecar registers records in sidecar_catalog and EquityVaultQueryAdapter queries it directly."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / ".system").mkdir(parents=True, exist_ok=True)

    # 1. Write sidecar
    out = _dummy_output("NVDA", "2026-09-07")
    
    import os
    os.environ["OBSIDIAN_VAULT_PATH"] = str(vault)
    try:
        write_equity_sidecar(out)

        cat = SqliteNoteCatalogAdapter(vault_root=vault)
        sidecars = cat.get_sidecars("NVDA")
        assert len(sidecars) >= 2  # user and system tier
        assert any("30_Knowledge_Base" in s for s in sidecars)
        assert any(".system/sidecars" in s for s in sidecars)

        # 2. Query via EquityVaultQueryAdapter
        adapter = EquityVaultQueryAdapter(vault_path=vault)
        files = adapter._sidecar_files("NVDA")
        assert len(files) >= 1
        assert files[0].exists()

        # Query detail
        detail = adapter.get_detail("NVDA")
        assert detail is not None
        assert detail["ticker"] == "NVDA"
        assert detail["composite_score"] == 85.0

    finally:
        os.environ.pop("OBSIDIAN_VAULT_PATH", None)


def test_vault_maintenance_and_compaction(tmp_path: Path):
    """Verifies that run_catalog_maintenance optimizes and vacuums the catalog."""
    vault = tmp_path / "vault"
    vault.mkdir()
    cat_db = vault / ".system" / "vault_catalog.db"
    cat_db.parent.mkdir(parents=True, exist_ok=True)

    cat = SqliteNoteCatalogAdapter(db_path=cat_db, vault_root=vault)
    # Add dummy entries
    cat.upsert_sidecar(
        ticker="AAPL",
        evaluation_date="2026-09-07",
        relative_path="30_Knowledge_Base/Stocks/AAPL/Analysis/AAPL Equity Analysis 2026-09-07.json",
        storage_tier="user",
        mtime=12345.0,
        file_size=100,
        sha256="abc",
    )

    # Run maintenance
    res = run_catalog_maintenance(vault_root=vault, vacuum=True)
    assert res["status"] == "success"
    assert "initial_size_bytes" in res
    assert "final_size_bytes" in res


def test_detect_orphaned_sidecars(tmp_path: Path):
    """Verifies that orphaned JSON sidecars without markdown hub are flagged."""
    vault = tmp_path / "vault"
    orphan_json = vault / "30_Knowledge_Base" / "Stocks" / "ORPHAN" / "Analysis" / "ORPHAN Equity Analysis 2026-09-07.json"
    orphan_json.parent.mkdir(parents=True, exist_ok=True)
    orphan_json.write_text("{}", encoding="utf-8")

    orphans = detect_orphaned_sidecars(vault_root=vault)
    assert len(orphans) == 1
    assert orphans[0]["ticker"] == "ORPHAN"
    assert orphans[0]["reason"] == "missing_hub_markdown"
