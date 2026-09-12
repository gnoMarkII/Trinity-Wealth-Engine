"""Tests for FTNT Equity Pilot under Obsidian Vault V2 Structure.

Verifies:
- list_latest returns ticker='FTNT' (not 'Analysis').
- EquitySidecarValuationAdapter resolves sidecars from Stocks/FTNT/Analysis/.
- quant_history reads and saves into Stocks/FTNT/Quant/.
- Earnings Call notes write and list from Stocks/FTNT/Earnings/.
- No duplicate reports from latest pointers or revisions.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from schemas.micro_quant_schemas import EquitySentimentContext, MicroQuantOutput, QuantSignals
from tools.content.earnings_call.adapters.obsidian_adapter import ObsidianEarningsCallAdapter
from tools.market.adapters.equity_research import EquitySidecarValuationAdapter
from tools.market.adapters.equity_vault_query_adapter import EquityVaultQueryAdapter
from tools.market.equity_sidecar import write_equity_sidecar
from tools.market.quant_history import get_equity_score_trend, save_equity_quant_snapshot


def _create_mock_quant_output(ticker: str = "FTNT", date_str: str = "2026-09-05") -> MicroQuantOutput:
    return MicroQuantOutput(
        ticker=ticker,
        market="US",
        analysis_date=date_str,
        quant_signals=QuantSignals(
            ticker=ticker,
            market="US",
            company_name="Fortinet Inc.",
            evaluated_at=f"{date_str}T12:00:00Z",
            composite_score=78.5,
            value_score=70.0,
            quality_score=85.0,
            growth_score=80.0,
            momentum_score=75.0,
        ),
        sentiment_context=EquitySentimentContext(
            evaluated_at=f"{date_str}T12:00:00Z",
            market_sentiment="bullish",
            sources_summary="Positive earnings call and guidance.",
        ),
        narrative_analysis="Strong cybersecurity position.",
        base_case_summary="Target $84.50.",
        generated_by="micro_quant_agent",
    )


def test_ftnt_pilot_v2_reading_and_latest(tmp_path: Path, monkeypatch) -> None:
    """list_latest must return ticker='FTNT' instead of 'Analysis' when reading V2 nested folders."""
    vault_dir = tmp_path / "vault_v2"
    # Set layout_version = 2 in config
    sys_dir = vault_dir / ".system"
    sys_dir.mkdir(parents=True)
    (sys_dir / "vault_config.json").write_text(json.dumps({"layout_version": 2}), encoding="utf-8")

    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(vault_dir))

    # 1. Write FTNT Hub
    hub_dir = vault_dir / "30_Knowledge_Base" / "Stocks" / "FTNT"
    hub_dir.mkdir(parents=True)
    (hub_dir / "FTNT.md").write_text("---\ntitle: FTNT\n---\n# FTNT Hub", encoding="utf-8")

    # 2. Write FTNT Equity Analysis sidecar using write_equity_sidecar
    mock_output = _create_mock_quant_output()
    write_equity_sidecar(mock_output)

    # Verify written location is V2 Analysis/ folder
    analysis_dir = hub_dir / "Analysis"
    assert analysis_dir.exists()
    sidecar_path = analysis_dir / "FTNT Equity Analysis 2026-09-05.json"
    assert sidecar_path.exists()

    # Write companion Markdown report
    md_report = analysis_dir / "FTNT Equity Analysis 2026-09-05.md"
    md_report.write_text(
        "---\ntitle: FTNT Analysis\nticker: FTNT\n---\n# FTNT Report\nDetailed content for [[FTNT]].",
        encoding="utf-8",
    )

    # 3. Test EquityVaultQueryAdapter
    adapter = EquityVaultQueryAdapter(vault_path=vault_dir)

    # list_latest must return ticker=FTNT, NOT Analysis
    latest_list = adapter.list_latest()
    assert len(latest_list) == 1
    item = latest_list[0]
    assert item["ticker"] == "FTNT"  # Crucial assertion: NOT 'Analysis'!
    assert item["composite_score"] == 78.5
    assert item["source_file"] is not None
    assert "FTNT Equity Analysis 2026-09-05.md" in item["source_file"]

    # get_detail must return matching detail
    detail = adapter.get_detail("FTNT")
    assert detail is not None
    assert detail["ticker"] == "FTNT"

    # 4. Test Valuation Adapter
    val_adapter = EquitySidecarValuationAdapter(vault_path=vault_dir)
    latest_val = val_adapter.latest("FTNT")
    assert latest_val is not None
    assert latest_val.ticker == "FTNT"
    assert latest_val.quant_signals.composite_score == 78.5


def test_ftnt_quant_and_earnings_v2(tmp_path: Path, monkeypatch) -> None:
    """Quant snapshots and Earnings calls are written and read from V2 subfolders."""
    vault_dir = tmp_path / "vault_v2"
    sys_dir = vault_dir / ".system"
    sys_dir.mkdir(parents=True)
    (sys_dir / "vault_config.json").write_text(json.dumps({"layout_version": 2}), encoding="utf-8")

    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(vault_dir))

    # 1. Quant history save & read
    signals = QuantSignals(
        ticker="FTNT",
        market="US",
        company_name="Fortinet",
        evaluated_at="2026-09-05T12:00:00Z",
        composite_score=78.5,
        value_score=70.0,
        quality_score=85.0,
        growth_score=80.0,
        momentum_score=75.0,
    )
    save_equity_quant_snapshot(signals)

    quant_file = vault_dir / "30_Knowledge_Base" / "Stocks" / "FTNT" / "Quant" / "FTNT_2026-09-05.md"
    assert quant_file.exists()

    trend = get_equity_score_trend("FTNT")
    assert len(trend) == 1
    assert trend[0]["composite_score"] == 78.5

    # 2. Earnings call write & list
    ec_adapter = ObsidianEarningsCallAdapter(vault_path=vault_dir)
    written_path = ec_adapter.write_note(
        ticker="FTNT",
        period="2026-Q2",
        transcript="Call transcript content.",
        highlights="## 🤖 AI Highlights\nRevenue grew 15%.",
    )
    assert "Stocks/FTNT/Earnings" in written_path

    notes = ec_adapter.list_notes_for_ticker("FTNT")
    assert len(notes) == 1
    assert notes[0].ticker == "FTNT"
    assert notes[0].period == "2026-Q2"
    assert "Revenue grew 15%" in notes[0].highlights
