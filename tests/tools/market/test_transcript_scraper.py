"""Unit tests for transcript scraper and transcript vault adapter."""
from pathlib import Path
import pytest

from tools.market.transcript_scraper import (
    parse_date_str,
    parse_quarter_label,
    parse_stockanalysis_html,
)
from tools.market.adapters.transcript_vault_adapter import (
    save_transcript_text,
    load_transcript_text,
    save_promise_ledger,
    load_promise_ledger,
    list_cached_transcripts,
)


def test_parse_date_str():
    assert parse_date_str("Aug 26, 2026") == "2026-08-26"
    assert parse_date_str("February 26, 2025") == "2025-02-26"
    assert parse_date_str("2024-11-20") == "2024-11-20"


def test_parse_quarter_label():
    assert parse_quarter_label("Earnings Call: Q4 2025") == "FY2025Q4"
    assert parse_quarter_label("Q1 2026 Earnings Release") == "FY2026Q1"
    assert parse_quarter_label("2024 Q3") == "FY2024Q3"
    assert parse_quarter_label("Conference Call Without Quarter") is None


def test_parse_stockanalysis_html():
    sample_html = """
    <html>
        <body>
            <article>
                <div class="font-bold text-lg text-default md:text-xl">Stewart Stecker</div>
                <div class="text-sm italic text-muted">Head of Investor Relations, NVIDIA</div>
                <p class="text-default mt-2">
                    <span class="transcript-sentence">Good afternoon, everyone, and welcome to NVIDIA's conference call.</span>
                    <span class="transcript-sentence">With me today are Jensen Huang and Colette Kress.</span>
                </p>
                <div class="font-bold text-lg text-default md:text-xl">Colette Kress</div>
                <div class="text-sm italic text-muted">CFO, NVIDIA</div>
                <p class="text-default mt-2">
                    <span class="transcript-sentence">Thanks, Stewart. Q4 revenue was $39.3 billion.</span>
                </p>
            </article>
        </body>
    </html>
    """
    text, words = parse_stockanalysis_html(sample_html)
    assert words > 20
    assert "### Stewart Stecker (Head of Investor Relations, NVIDIA)" in text
    assert "### Colette Kress (CFO, NVIDIA)" in text
    assert "Q4 revenue was $39.3 billion." in text


def test_parse_stockanalysis_html_fallback():
    fallback_html = """
    <html>
        <body>
            <p><span class="transcript-sentence">This is the first sentence of the call.</span></p>
            <p><span class="transcript-sentence">This is the second sentence of the call.</span></p>
        </body>
    </html>
    """
    text, words = parse_stockanalysis_html(fallback_html)
    assert words == 16
    assert "This is the first sentence of the call." in text
    assert "This is the second sentence of the call." in text


def test_transcript_vault_adapter(tmp_path: Path):
    test_base_dir = tmp_path / "cache" / "transcripts"
    ticker = "NVDA"
    quarter = "FY2025Q4"
    content = "Sample full transcript content for testing."

    # Save and Load text
    saved_path = save_transcript_text(ticker, quarter, content, base_dir=test_base_dir)
    assert saved_path.is_file()
    loaded_text = load_transcript_text(ticker, quarter, base_dir=test_base_dir)
    assert loaded_text == content

    # Check listing
    save_transcript_text(ticker, "FY2025Q3", "Earlier call", base_dir=test_base_dir)
    quarters = list_cached_transcripts(ticker, base_dir=test_base_dir)
    assert quarters == ["FY2025Q3", "FY2025Q4"]

    # Save and Load promise ledger
    ledger_data = {
        "framework_version": "0.1",
        "ticker": "NVDA",
        "promises": [{"id": "P-NVDA-FY2025Q4-01", "status": "Pending"}],
    }
    ledger_path = save_promise_ledger(ticker, ledger_data, base_dir=test_base_dir)
    assert ledger_path.is_file()
    loaded_ledger = load_promise_ledger(ticker, base_dir=test_base_dir)
    assert loaded_ledger == ledger_data
