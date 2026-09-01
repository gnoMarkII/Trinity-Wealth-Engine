"""Unit tests for Point-in-Time Fundamentals Ledger (Phase 4 & v3.1)."""
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.market.pit_fundamentals_ledger import (
    PITLedgerEntry,
    append_pit_entry,
    compute_pit_multiple_bands,
)


def test_pit_ledger_append_and_idempotency():
    with TemporaryDirectory() as tmp_dir:
        base_path = Path(tmp_dir)
        entry1 = PITLedgerEntry(
            as_of_date="2026-01-15",
            ticker="AAPL",
            valuation_price=185.0,
            ttm_eps=6.50,
            pe_multiple=28.5,
        )
        append_pit_entry(entry1, base_dir=base_path)

        # Append identical date
        append_pit_entry(entry1, base_dir=base_path)

        # Should only have 1 entry
        bands = compute_pit_multiple_bands("AAPL", current_pe=28.5, base_dir=base_path)
        assert bands.observations_count == 1
        assert bands.status == "unavailable"  # 1 is < 250


def test_pit_insufficient_history_returns_unavailable():
    with TemporaryDirectory() as tmp_dir:
        base_path = Path(tmp_dir)
        bands = compute_pit_multiple_bands("MSFT", current_pe=32.0, base_dir=base_path)
        assert bands.status == "unavailable"
        assert "pit_ledger_not_initialized" in bands.flags[0]


def test_pit_multiple_bands_calculation_when_sufficient():
    with TemporaryDirectory() as tmp_dir:
        base_path = Path(tmp_dir)
        from datetime import date, timedelta
        start_date = date(2024, 1, 1)
        # Generate 260 distinct daily PIT entries
        for i in range(260):
            cur_date = start_date + timedelta(days=i)
            entry = PITLedgerEntry(
                as_of_date=cur_date.strftime("%Y-%m-%d"),
                ticker="GOOGL",
                valuation_price=150.0 + i * 0.1,
                pe_multiple=20.0 + (i % 20),  # PE range 20 to 39
            )
            append_pit_entry(entry, base_dir=base_path)

        bands = compute_pit_multiple_bands("GOOGL", current_pe=25.0, base_dir=base_path)
        assert bands.status == "available"
        assert bands.observations_count >= 250
        assert bands.pe_median_3y is not None
        assert bands.current_pe_percentile is not None
        assert 0.0 <= bands.current_pe_percentile <= 100.0
