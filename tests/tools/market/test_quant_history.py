"""Unit tests สำหรับ tools/market/quant_history.py (Vault-native Forward-tracking)"""
from datetime import date, datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from schemas.micro_quant_schemas import QuantSignals
from tools.market.pit_fundamentals_ledger import get_default_pit_dir
from tools.market.quant_history import (
    evaluate_signal_factor_efficacy,
    get_equity_score_trend,
    save_equity_quant_snapshot,
)


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

    def test_write_failure_does_not_raise(self, tmp_vault, caplog):
        signals = _make_signals(evaluated_at="2026-07-20T10:00:00+00:00")
        with patch("tools.archivist.artifact_writer._atomic_write_text", side_effect=OSError("disk full")) as mocked_write:
            save_equity_quant_snapshot(signals)  # ต้องไม่ raise ออกมา — เป็น side-effect เสริมเท่านั้น
        mocked_write.assert_called()
        assert "save_equity_quant_snapshot failed" in caplog.text
        assert "disk full" in caplog.text
        ticker_dir = tmp_vault / "30_Knowledge_Base" / "Stocks" / "AAPL" / "Quant"
        assert list(ticker_dir.glob("*.tmp")) == []


class TestGetEquityScoreTrend:
    def test_no_history_returns_empty_list(self, tmp_vault):
        assert get_equity_score_trend("NOHISTORY") == []

    def test_returns_saved_snapshot(self, tmp_vault):
        save_equity_quant_snapshot(_make_signals(evaluated_at="2026-07-20T10:00:00+00:00", value_score=80.0))
        trend = get_equity_score_trend("AAPL", days=90, as_of=date(2026, 8, 1))
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

        trend = get_equity_score_trend("AAPL", days=90, as_of=date(2026, 8, 1))
        assert trend[0]["composite_score"] == 82.5

    def test_save_snapshot_records_pit_entry_and_evaluates_efficacy(self, tmp_vault):
        signals = _make_signals(evaluated_at="2026-07-20T10:00:00+00:00", value_score=80.0)
        signals.raw_analysis_price = 215.5
        save_equity_quant_snapshot(signals)

        pit_file = get_default_pit_dir() / "AAPL.json"
        assert pit_file.exists()
        assert "215.5" in pit_file.read_text(encoding="utf-8")

        efficacy = evaluate_signal_factor_efficacy(
            factor_values=[10.0, 20.0, 30.0, 40.0, 50.0],
            forward_returns=[0.01, 0.02, 0.03, 0.04, 0.05],
            ic_series=[0.08, 0.06, 0.07, 0.09, 0.05, 0.07, 0.08],
        )
        assert efficacy["rank_ic"] == 1.0
        assert efficacy["newey_west_tstat"] is not None
        assert efficacy["newey_west_tstat"] > 2.0
