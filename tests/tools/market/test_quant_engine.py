"""Unit tests สำหรับ tools/market/quant_engine.py"""
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from tools.market.quant_engine import (
    compute_beta,
    compute_volatility,
    compute_mdd,
    compute_technical_indicators,
    compute_growth_rates,
    compute_price_percentile,
    _get_price_history,
    _HISTORY_CACHE,
    _HISTORY_ERROR_CACHE,
)
from tools.market.financial_autopsy import FinancialAutopsyPeriod


def _make_close_series(n: int, start: float = 100.0, daily_return: float = 0.0) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=n)
    prices = [start]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + daily_return))
    return pd.DataFrame({"Close": prices}, index=dates)


class TestGetPriceHistory:
    def setup_method(self):
        _HISTORY_CACHE.clear()
        _HISTORY_ERROR_CACHE.clear()

    @patch("tools.market.quant_engine.yf.Ticker")
    def test_fetch_and_cache(self, mock_ticker_cls):
        df = _make_close_series(100)
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = df
        mock_ticker_cls.return_value = mock_ticker

        result1 = _get_price_history("AAPL", "1y")
        result2 = _get_price_history("AAPL", "1y")
        assert len(result1) == 100
        assert len(result2) == 100
        assert mock_ticker_cls.call_count == 1  # เรียกครั้งที่สอง ต้องมาจาก cache ไม่ยิง yfinance ซ้ำ

    @patch("tools.market.quant_engine.yf.Ticker")
    def test_fetch_error_cached_short_ttl(self, mock_ticker_cls):
        mock_ticker = MagicMock()
        # ใช้ error message ที่ไม่ตรงกับ transient pattern -> ไม่ retry ไม่ sleep
        mock_ticker.history.side_effect = Exception("Symbol not found in Yahoo Finance")
        mock_ticker_cls.return_value = mock_ticker

        with pytest.raises(Exception):
            _get_price_history("BADSYM", "1y")
        with pytest.raises(Exception):
            _get_price_history("BADSYM", "1y")
        assert mock_ticker_cls.call_count == 1  # ครั้งที่สองต้องมาจาก error cache


class TestDataQualityGuard:
    def setup_method(self):
        _HISTORY_CACHE.clear()
        _HISTORY_ERROR_CACHE.clear()

    @patch("tools.market.quant_engine._get_price_history")
    def test_insufficient_trading_days_marks_invalid(self, mock_hist):
        mock_hist.return_value = _make_close_series(10)  # < 45 วัน (เช่น หุ้น IPO ใหม่)

        vol, q = compute_volatility("NEWCO")
        assert vol is None
        assert q.is_valid is False
        assert q.stale_reason == "insufficient_trading_history"

        mdd, q2 = compute_mdd("NEWCO")
        assert mdd is None
        assert q2.is_valid is False

        tech, q3 = compute_technical_indicators("NEWCO")
        assert tech is None
        assert q3.is_valid is False

    @patch("tools.market.quant_engine._get_price_history")
    def test_beta_insufficient_overlap(self, mock_hist):
        def side_effect(symbol, period):
            if symbol == "THINSTOCK":
                return _make_close_series(100, start=50.0, daily_return=0.001)
            return _make_close_series(10, start=4000.0, daily_return=0.0005)  # overlap แค่ 10 วัน

        mock_hist.side_effect = side_effect
        beta, q = compute_beta("THINSTOCK", benchmark="^GSPC")
        assert beta is None
        assert q.is_valid is False
        assert q.trading_days < 45

    @patch("tools.market.quant_engine._get_price_history")
    def test_fetch_exception_marks_fetch_error(self, mock_hist):
        mock_hist.side_effect = RuntimeError("cached failure")
        vol, q = compute_volatility("ERRCO")
        assert vol is None
        assert q.is_valid is False
        assert q.stale_reason == "fetch_error"


class TestComputeFormulas:
    def setup_method(self):
        _HISTORY_CACHE.clear()
        _HISTORY_ERROR_CACHE.clear()

    @patch("tools.market.quant_engine._get_price_history")
    def test_compute_volatility_near_zero_for_constant_return(self, mock_hist):
        mock_hist.return_value = _make_close_series(100, start=100.0, daily_return=0.001)
        vol, q = compute_volatility("STEADYCO")
        assert q.is_valid is True
        assert vol is not None
        assert vol < 1.0  # return คงที่ทุกวัน -> std ≈ 0

    @patch("tools.market.quant_engine._get_price_history")
    def test_compute_mdd_negative_for_declining_price(self, mock_hist):
        mock_hist.return_value = _make_close_series(100, start=100.0, daily_return=-0.01)
        mdd, q = compute_mdd("DECLINECO")
        assert q.is_valid is True
        assert mdd is not None
        assert mdd < 0

    @patch("tools.market.quant_engine._get_price_history")
    def test_compute_technical_indicators_uptrend_bullish(self, mock_hist):
        mock_hist.return_value = _make_close_series(100, start=100.0, daily_return=0.01)
        tech, q = compute_technical_indicators("UPCO")
        assert q.is_valid is True
        assert tech is not None
        # ขาขึ้นต่อเนื่อง ไม่มีวันขาดทุนเลยใน 14 วันล่าสุด -> RSI ต้องแตะเพดาน 100 (ไม่ใช่ NaN)
        assert tech["rsi_14"] == 100.0
        assert tech["macd_signal"] == "bullish"

    @patch("tools.market.quant_engine._get_price_history")
    def test_compute_technical_indicators_flat_price_neutral_rsi(self, mock_hist):
        mock_hist.return_value = _make_close_series(100, start=100.0, daily_return=0.0)
        tech, q = compute_technical_indicators("FLATCO")
        assert q.is_valid is True
        assert tech is not None
        assert tech["rsi_14"] == 50.0  # ราคาไม่ขยับเลย -> เป็นกลาง ไม่ใช่ NaN

    @patch("tools.market.quant_engine._get_price_history")
    def test_compute_beta_known_ratio(self, mock_hist):
        n = 100
        dates = pd.bdate_range("2024-01-01", periods=n)
        rng = np.random.default_rng(42)
        bench_returns = rng.normal(0.0005, 0.01, n - 1)
        bench_prices = [4000.0]
        for r in bench_returns:
            bench_prices.append(bench_prices[-1] * (1 + r))
        stock_prices = [50.0]
        for r in bench_returns:
            stock_prices.append(stock_prices[-1] * (1 + 1.5 * r))  # beta คาดว่าใกล้ 1.5

        def side_effect(symbol, period):
            if symbol == "BETACO":
                return pd.DataFrame({"Close": stock_prices}, index=dates)
            return pd.DataFrame({"Close": bench_prices}, index=dates)

        mock_hist.side_effect = side_effect
        beta, q = compute_beta("BETACO", benchmark="^GSPC")
        assert q.is_valid is True
        assert beta is not None
        assert 1.3 <= beta <= 1.7


def _period(fiscal_period_end, total_revenue=None, net_income=None):
    return FinancialAutopsyPeriod(
        fiscal_period_end=fiscal_period_end,
        total_revenue=total_revenue,
        net_income=net_income,
    )


class TestComputeGrowthRates:
    def test_insufficient_periods_returns_none_with_flag(self):
        result, flags = compute_growth_rates([_period("2026-06-30", total_revenue=1000.0)])
        assert result == {"revenue_growth_yoy_pct": None, "net_income_growth_yoy_pct": None}
        assert flags == ["insufficient_periods:growth"]

    def test_normal_annual_gap_computes_growth(self):
        periods = [
            _period("2026-06-30", total_revenue=1200.0, net_income=200.0),
            _period("2025-06-30", total_revenue=1000.0, net_income=150.0),  # 365 days ก่อนหน้า
        ]
        result, flags = compute_growth_rates(periods)
        assert flags == []
        assert result["revenue_growth_yoy_pct"] == 20.0
        assert result["net_income_growth_yoy_pct"] == pytest.approx(33.33, abs=0.01)

    def test_non_annual_gap_rejected(self):
        # ห่างกันแค่ ~90 วัน (quarterly ปนมา) — ไม่ใช่ YoY จริง
        periods = [
            _period("2026-06-30", total_revenue=1200.0),
            _period("2026-03-31", total_revenue=1000.0),
        ]
        result, flags = compute_growth_rates(periods)
        assert result == {"revenue_growth_yoy_pct": None, "net_income_growth_yoy_pct": None}
        assert flags == ["non_annual_period_gap:growth"]

    def test_base_year_negative_or_zero_revenue_skipped(self):
        periods = [
            _period("2026-06-30", total_revenue=1200.0, net_income=200.0),
            _period("2025-06-28", total_revenue=-500.0, net_income=100.0),  # ~367 days, revenue ฐานติดลบ
        ]
        result, flags = compute_growth_rates(periods)
        assert result["revenue_growth_yoy_pct"] is None
        assert "base_year_negative:revenue_growth" in flags
        assert result["net_income_growth_yoy_pct"] is not None  # net_income ฐานยังบวก คำนวณได้ตามปกติ

    def test_missing_values_leave_none_without_crashing(self):
        periods = [
            _period("2026-06-30", total_revenue=None, net_income=None),
            _period("2025-06-30", total_revenue=1000.0, net_income=150.0),
        ]
        result, flags = compute_growth_rates(periods)
        assert result["revenue_growth_yoy_pct"] is None
        assert result["net_income_growth_yoy_pct"] is None
        assert flags == []


class TestComputePricePercentile:
    def setup_method(self):
        _HISTORY_CACHE.clear()
        _HISTORY_ERROR_CACHE.clear()

    @patch("tools.market.quant_engine._get_price_history")
    def test_insufficient_history_returns_none(self, mock_hist):
        mock_hist.return_value = _make_close_series(100)  # < 250 วันขั้นต่ำ
        percentile, zscore, q = compute_price_percentile("SHORTCO")
        assert percentile is None
        assert zscore is None
        assert q.is_valid is False
        assert q.stale_reason == "insufficient_trading_history"

    @patch("tools.market.quant_engine._get_price_history")
    def test_price_at_historical_high_gives_100_percentile(self, mock_hist):
        mock_hist.return_value = _make_close_series(260, start=100.0, daily_return=0.001)  # ขาขึ้นต่อเนื่อง
        percentile, zscore, q = compute_price_percentile("UPCO")
        assert q.is_valid is True
        assert percentile == 100.0  # ราคาล่าสุด = สูงสุดในช่วง
        assert zscore is not None
        assert zscore > 0  # อยู่เหนือค่าเฉลี่ย

    @patch("tools.market.quant_engine._get_price_history")
    def test_price_at_historical_low_gives_low_percentile(self, mock_hist):
        mock_hist.return_value = _make_close_series(260, start=200.0, daily_return=-0.001)  # ขาลงต่อเนื่อง
        percentile, zscore, q = compute_price_percentile("DOWNCO")
        assert q.is_valid is True
        assert percentile < 5.0  # ราคาล่าสุด = ต่ำสุดในช่วง
        assert zscore is not None
        assert zscore < 0

    @patch("tools.market.quant_engine._get_price_history")
    def test_flat_price_gives_neutral_zscore(self, mock_hist):
        mock_hist.return_value = _make_close_series(260, start=100.0, daily_return=0.0)
        percentile, zscore, q = compute_price_percentile("FLATCO")
        assert q.is_valid is True
        assert zscore == 0.0

    @patch("tools.market.quant_engine._get_price_history")
    def test_fetch_exception_returns_none_with_fetch_error(self, mock_hist):
        mock_hist.side_effect = RuntimeError("cached failure")
        percentile, zscore, q = compute_price_percentile("ERRCO")
        assert percentile is None
        assert zscore is None
        assert q.stale_reason == "fetch_error"

    @patch("tools.market.quant_engine._get_price_history")
    def test_trailing_nan_latest_close_unavailable(self, mock_hist):
        df = _make_close_series(260, start=100.0, daily_return=0.001)
        df.iloc[-1, df.columns.get_loc("Close")] = float("nan")
        mock_hist.return_value = df
        percentile, zscore, q = compute_price_percentile("NANCO")
        assert percentile is None
        assert zscore is None
        assert q.is_valid is False
        assert q.stale_reason == "latest_close_unavailable"

    @patch("tools.market.quant_engine._get_price_history")
    def test_internal_nan_handled_correctly(self, mock_hist):
        df = _make_close_series(265, start=100.0, daily_return=0.001)
        # Introduce 5 internal NaNs (not latest)
        df.iloc[10:15, df.columns.get_loc("Close")] = float("nan")
        mock_hist.return_value = df
        percentile, zscore, q = compute_price_percentile("INTERNALNANCO")
        assert q.is_valid is True
        assert percentile is not None
        assert zscore is not None
        assert q.trading_days == 260

    @patch("tools.market.quant_engine._get_price_history")
    def test_cleanup_drops_below_min_days_returns_insufficient_history(self, mock_hist):
        df = _make_close_series(260, start=100.0, daily_return=0.001)
        # Introduce 20 internal NaNs -> 240 valid (< 250)
        df.iloc[10:30, df.columns.get_loc("Close")] = float("nan")
        mock_hist.return_value = df
        percentile, zscore, q = compute_price_percentile("TOOMANYNANCO")
        assert percentile is None
        assert zscore is None
        assert q.is_valid is False
        assert q.stale_reason == "insufficient_trading_history"


