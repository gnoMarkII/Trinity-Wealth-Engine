"""Unit tests สำหรับ tools/market/quant_history.py (Vault-native Forward-tracking)"""
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from schemas.micro_quant_schemas import QuantSignals
from tools.market.quant_history import save_equity_quant_snapshot, get_equity_score_trend


def _make_signals(ticker="AAPL", market="US", evaluated_at=None, value_score=80.0, composite_score=None) -> QuantSignals:
    return QuantSignals(
        ticker=ticker,
        market=market,
        value_score=value_score,
        quality_score=70.0,
        momentum_score=60.0,
        beta=1.2,
        volatility_pct=25.0,
        mdd_pct=-15.0,
        composite_score=composite_score,
        evaluated_at=evaluated_at or datetime.now(timezone.utc).isoformat(),
    )


class TestSaveEquityQuantSnapshot:
    def test_writes_file_with_expected_path_and_frontmatter(self, tmp_vault):
        signals = _make_signals(evaluated_at="2026-07-20T10:00:00+00:00")
        save_equity_quant_snapshot(signals)

        expected = tmp_vault / "30_Knowledge_Base" / "Stocks" / "AAPL" / "Quant" / "AAPL_2026-07-20.md"
        assert expected.exists()
        content = expected.read_text(encoding="utf-8")
        assert "entity_type: quant_snapshot" in content
        assert "ticker: AAPL" in content
        assert "Value Score | 80.0" in content

    def test_no_leftover_temp_files_after_write(self, tmp_vault):
        signals = _make_signals(evaluated_at="2026-07-20T10:00:00+00:00")
        save_equity_quant_snapshot(signals)
        ticker_dir = tmp_vault / "30_Knowledge_Base" / "Stocks" / "AAPL" / "Quant"
        tmp_files = list(ticker_dir.glob("*.tmp"))
        assert tmp_files == []

    def test_rerun_same_day_overwrites_not_duplicates(self, tmp_vault):
        save_equity_quant_snapshot(_make_signals(evaluated_at="2026-07-20T09:00:00+00:00", value_score=50.0))
        save_equity_quant_snapshot(_make_signals(evaluated_at="2026-07-20T18:00:00+00:00", value_score=90.0))
        ticker_dir = tmp_vault / "30_Knowledge_Base" / "Stocks" / "AAPL" / "Quant"
        files = list(ticker_dir.glob("AAPL_*.md"))
        assert len(files) == 1
        assert "Value Score | 90.0" in files[0].read_text(encoding="utf-8")

    def test_write_failure_does_not_raise(self, tmp_vault):
        signals = _make_signals()
        with patch("tools.archivist.artifact_writer._atomic_write_text", side_effect=OSError("disk full")):
            save_equity_quant_snapshot(signals)  # ต้องไม่ raise ออกมา — เป็น side-effect เสริมเท่านั้น


class TestGetEquityScoreTrend:
    def test_no_history_returns_empty_list(self, tmp_vault):
        assert get_equity_score_trend("NOHISTORY") == []

    def test_returns_saved_snapshot(self, tmp_vault):
        save_equity_quant_snapshot(_make_signals(evaluated_at="2026-07-20T10:00:00+00:00", value_score=80.0))
        trend = get_equity_score_trend("AAPL", days=90)
        assert len(trend) == 1
        assert trend[0]["value_score"] == 80.0

    def test_filters_out_snapshots_older_than_days_cutoff(self, tmp_vault):
        now = datetime.now(timezone.utc)
        recent = (now - timedelta(days=5)).isoformat()
        old = (now - timedelta(days=200)).isoformat()
        save_equity_quant_snapshot(_make_signals(evaluated_at=recent, value_score=85.0))
        save_equity_quant_snapshot(_make_signals(evaluated_at=old, value_score=40.0))

        trend = get_equity_score_trend("AAPL", days=90)
        assert len(trend) == 1
        assert trend[0]["value_score"] == 85.0

    def test_composite_score_persisted_in_snapshot(self, tmp_vault):
        save_equity_quant_snapshot(_make_signals(evaluated_at="2026-07-20T10:00:00+00:00", composite_score=82.5))
        expected = tmp_vault / "30_Knowledge_Base" / "Stocks" / "AAPL" / "Quant" / "AAPL_2026-07-20.md"
        content = expected.read_text(encoding="utf-8")
        assert "Composite Score" in content
        assert "82.5" in content

        trend = get_equity_score_trend("AAPL", days=90)
        assert trend[0]["composite_score"] == 82.5
