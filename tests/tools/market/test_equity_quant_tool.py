"""Unit tests สำหรับ tools/market/equity_quant_tool.py"""
import json
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from tools.market.asset_resolver import ResolvedAsset, AssetClass
from tools.market.financial_autopsy import (
    FinancialAutopsyFetchResult,
    FinancialAutopsySnapshot,
    FinancialAutopsyPeriod,
)
from tools.market.quant_engine import PriceSeriesQuality
from tools.market.equity_quant_tool import (
    compute_equity_quant_signals,
    _get_info_cached,
    _INFO_CACHE,
    _INFO_ERROR_CACHE,
)
from schemas.micro_quant_schemas import QuantSignals, SmartMoneyFlags


def _resolved_asset(ticker="AAPL", market="US"):
    return ResolvedAsset(
        raw_symbol=ticker,
        provider_symbol=ticker,
        asset_class=AssetClass.STOCK_US,
        market=market,
        confidence="high",
        eligible_for_financial_autopsy=True,
    )


def _autopsy_result(ticker="AAPL", market="US", two_periods=False):
    periods = [
        FinancialAutopsyPeriod(
            fiscal_period_end="2026-06-30",
            free_cash_flow=500.0,
            total_debt=1000.0,
            total_revenue=5000.0,
            net_income=800.0,
            payout_ratio_pct=25.0,
            ebit=900.0,
            operating_income=850.0,
            interest_expense=50.0,
            tax_expense=150.0,
            income_before_tax=750.0,
            source_tier="primary_best_effort",
            period_type="annual",
        )
    ]
    if two_periods:
        periods.append(
            FinancialAutopsyPeriod(
                fiscal_period_end="2025-06-30",
                free_cash_flow=400.0,
                total_debt=900.0,
                total_revenue=4000.0,
                net_income=600.0,
                source_tier="primary_best_effort",
                period_type="annual",
            )
        )
    snapshot = FinancialAutopsySnapshot(
        ticker=ticker,
        provider_symbol=ticker,
        market=market,
        currency="USD",
        unit="raw",
        retrieval_timestamp="2026-07-20 12:00:00",
        periods=periods,
        current_pe=18.0,
        source_tier="primary_best_effort",
        period_type="annual",
    )
    return FinancialAutopsyFetchResult(asset=_resolved_asset(ticker, market), status="success", snapshot=snapshot)


_VALID_QUALITY = PriceSeriesQuality(trading_days=100, is_valid=True)
_INVALID_QUALITY = PriceSeriesQuality(trading_days=5, is_valid=False, stale_reason="insufficient_trading_history")


class TestComputeEquityQuantSignalsHappyPath:
    @patch("tools.market.equity_quant_tool._get_price_history")
    @patch("tools.market.equity_quant_tool.compute_smart_money_flags")
    @patch("tools.market.equity_quant_tool.compute_dcf_valuation")
    @patch("tools.market.equity_quant_tool.load_latest_macro_observables")
    @patch("tools.market.equity_quant_tool.fetch_earnings_revision_data")
    @patch("tools.market.equity_quant_tool.compute_price_percentile")
    @patch("tools.market.equity_quant_tool.fetch_peer_metrics")
    @patch("tools.market.equity_quant_tool.compute_technical_indicators")
    @patch("tools.market.equity_quant_tool.compute_mdd")
    @patch("tools.market.equity_quant_tool.compute_volatility")
    @patch("tools.market.equity_quant_tool.compute_beta")
    @patch("tools.market.equity_quant_tool._yf_info")
    @patch("tools.market.equity_quant_tool.get_financial_autopsy")
    @patch("tools.market.equity_quant_tool.resolve_asset")
    def test_returns_valid_quant_signals_json(
        self, mock_resolve, mock_autopsy, mock_info, mock_beta, mock_vol, mock_mdd, mock_tech, mock_peers, mock_pctile,
        mock_revisions, mock_macro_reg, mock_dcf, mock_smart_money, mock_history,
    ):
        _INFO_CACHE.clear()
        _INFO_ERROR_CACHE.clear()
        mock_resolve.return_value = _resolved_asset()
        mock_autopsy.return_value = _autopsy_result(two_periods=True)
        dates = pd.date_range(end=pd.Timestamp("2026-08-28"), periods=250, freq="B")
        n = len(dates)
        mock_history.return_value = pd.DataFrame({
            "Close": [200.0] * n,
            "High": [205.0] * n,
            "Low": [195.0] * n,
            "Open": [200.0] * n,
            "Volume": [5_000_000] * n,
        }, index=dates)
        mock_info.return_value = {
            "shortName": "Apple Inc.",
            "sector": "Technology",
            "trailingPE": 18.0,
            "priceToBook": 5.0,
            "enterpriseToEbitda": 12.0,
            "returnOnEquity": 0.25,
            "profitMargins": 0.18,
            "fiftyDayAverage": 190.0,
            "twoHundredDayAverage": 180.0,
            "currentPrice": 200.0,
            "targetMeanPrice": 220.0,
            "targetHighPrice": 240.0,
            "targetLowPrice": 190.0,
            "trailingAnnualDividendYield": 0.02,  # -> 2.0%
            "debtToEquity": 50.0,  # 0.5x
            "currentRatio": 2.0,
            "averageVolume": 5_000_000,
            "averageVolume10Day": 6_000_000,
            "freeCashflow": 500.0,
            "marketCap": 10000.0,
            "operatingCashflow": 700.0,
            "ebitda": 1000.0,
            "totalDebt": 1000.0,
            "totalCash": 200.0,
            "totalStockholderEquity": 3000.0,
            "sharesOutstanding": 100.0,
        }
        mock_beta.return_value = (1.2, _VALID_QUALITY)
        mock_vol.return_value = (28.5, _VALID_QUALITY)
        mock_mdd.return_value = (-18.0, _VALID_QUALITY)
        mock_tech.return_value = ({"rsi_14": 65.0, "macd": 1.2, "macd_signal_line": 0.9, "macd_signal": "bullish"}, _VALID_QUALITY)
        mock_peers.return_value = [{"symbol": "MSFT", "pe": 20.0}, {"symbol": "GOOGL", "pe": 22.0}]
        mock_pctile.return_value = (72.5, 1.1, _VALID_QUALITY)
        mock_revisions.return_value = ({"up_last_30d": 3, "down_last_30d": 1, "estimate_current": 9.0, "estimate_30d_ago": 8.5}, None)
        mock_macro_reg.return_value = {}
        mock_dcf.return_value = (None, [])
        mock_smart_money.return_value = (SmartMoneyFlags(), [])

        result = compute_equity_quant_signals.invoke({"ticker": "AAPL", "market": "US"})
        assert not result.startswith("Error:")

        signals = QuantSignals.model_validate(json.loads(result))
        assert signals.ticker == "AAPL"
        assert signals.market == "US"
        assert signals.company_name == "Apple Inc."
        assert signals.beta == 1.2
        assert signals.volatility_pct == 28.5
        assert signals.mdd_pct == -18.0
        assert signals.value_score is not None
        assert signals.quality_score is not None
        assert signals.momentum_score is not None
        assert signals.upside_pct == 20.0  # (240-200)/200*100
        assert signals.downside_pct == -5.0  # (190-200)/200*100

        # 5 Pillars ใหม่
        assert signals.revenue_growth_yoy_pct == 25.0  # (5000-4000)/4000*100
        assert signals.net_income_growth_yoy_pct == pytest.approx(33.33, abs=0.01)
        assert signals.growth_score is not None
        assert signals.dividend_yield_pct == 2.0
        assert signals.payout_ratio_pct == 25.0
        assert signals.dividend_score is not None
        assert signals.de_ratio_pct == 50.0
        assert signals.current_ratio == 2.0
        assert signals.solvency_score == 100.0
        assert signals.adtv_local_currency == 6_000_000 * 200.0  # ใช้ averageVolume10Day * currentPrice
        assert signals.composite_score is not None
        assert signals.peer_sector == "Technology"
        assert signals.peer_count == 2
        assert signals.peer_relative_score is not None
        assert signals.price_percentile_5y == 72.5
        assert signals.price_zscore_5y == 1.1
        assert signals.eps_revision_net_30d == 2
        assert signals.eps_estimate_change_30d_pct == pytest.approx(5.88, abs=0.01)
        assert signals.earnings_momentum_score is not None

        assert signals.data_quality_flags == []

    @patch("tools.market.equity_quant_tool.fetch_earnings_revision_data")
    @patch("tools.market.equity_quant_tool.compute_price_percentile")
    @patch("tools.market.equity_quant_tool.compute_technical_indicators")
    @patch("tools.market.equity_quant_tool.compute_mdd")
    @patch("tools.market.equity_quant_tool.compute_volatility")
    @patch("tools.market.equity_quant_tool.compute_beta")
    @patch("tools.market.equity_quant_tool._yf_info")
    @patch("tools.market.equity_quant_tool.get_financial_autopsy")
    @patch("tools.market.equity_quant_tool.resolve_asset")
    def test_th_market_uses_set_benchmark_not_gspc(
        self, mock_resolve, mock_autopsy, mock_info, mock_beta, mock_vol, mock_mdd, mock_tech, mock_pctile,
        mock_revisions,
    ):
        _INFO_CACHE.clear()
        _INFO_ERROR_CACHE.clear()
        mock_resolve.return_value = _resolved_asset(ticker="PTT", market="TH")
        mock_autopsy.return_value = _autopsy_result(ticker="PTT", market="TH")
        mock_info.return_value = {"currentPrice": 35.0}
        mock_beta.return_value = (0.9, _VALID_QUALITY)
        mock_vol.return_value = (None, _INVALID_QUALITY)
        mock_mdd.return_value = (None, _INVALID_QUALITY)
        mock_tech.return_value = (None, _INVALID_QUALITY)
        mock_pctile.return_value = (None, None, _INVALID_QUALITY)
        mock_revisions.return_value = (None, "missing_earnings_revision_data:earnings_momentum")

        compute_equity_quant_signals.invoke({"ticker": "PTT", "market": "TH"})

        mock_beta.assert_called_once_with("PTT", benchmark="^SET.BK")

    @patch("tools.market.equity_quant_tool.fetch_earnings_revision_data")
    @patch("tools.market.equity_quant_tool.compute_price_percentile")
    @patch("tools.market.equity_quant_tool.compute_technical_indicators")
    @patch("tools.market.equity_quant_tool.compute_mdd")
    @patch("tools.market.equity_quant_tool.compute_volatility")
    @patch("tools.market.equity_quant_tool.compute_beta")
    @patch("tools.market.equity_quant_tool._yf_info")
    @patch("tools.market.equity_quant_tool.get_financial_autopsy")
    @patch("tools.market.equity_quant_tool.resolve_asset")
    def test_us_market_uses_gspc_benchmark(
        self, mock_resolve, mock_autopsy, mock_info, mock_beta, mock_vol, mock_mdd, mock_tech, mock_pctile,
        mock_revisions,
    ):
        _INFO_CACHE.clear()
        _INFO_ERROR_CACHE.clear()
        mock_resolve.return_value = _resolved_asset()
        mock_autopsy.return_value = _autopsy_result()
        mock_info.return_value = {"currentPrice": 200.0}
        mock_beta.return_value = (1.0, _VALID_QUALITY)
        mock_vol.return_value = (None, _INVALID_QUALITY)
        mock_mdd.return_value = (None, _INVALID_QUALITY)
        mock_tech.return_value = (None, _INVALID_QUALITY)
        mock_pctile.return_value = (None, None, _INVALID_QUALITY)
        mock_revisions.return_value = (None, "missing_earnings_revision_data:earnings_momentum")

        compute_equity_quant_signals.invoke({"ticker": "AAPL", "market": "US"})

        mock_beta.assert_called_once_with("AAPL", benchmark="^GSPC")

    @patch("tools.market.equity_quant_tool.fetch_earnings_revision_data")
    @patch("tools.market.equity_quant_tool.compute_price_percentile")
    @patch("tools.market.equity_quant_tool.compute_technical_indicators")
    @patch("tools.market.equity_quant_tool.compute_mdd")
    @patch("tools.market.equity_quant_tool.compute_volatility")
    @patch("tools.market.equity_quant_tool.compute_beta")
    @patch("tools.market.equity_quant_tool._yf_info")
    @patch("tools.market.equity_quant_tool.get_financial_autopsy")
    @patch("tools.market.equity_quant_tool.resolve_asset")
    def test_low_solvency_score_flagged_and_excluded_from_composite(
        self, mock_resolve, mock_autopsy, mock_info, mock_beta, mock_vol, mock_mdd, mock_tech, mock_pctile,
        mock_revisions,
    ):
        """solvency_score ต่ำต้องแนบ flag high_leverage_risk แต่ต้องไม่ทำให้ composite_score เปลี่ยนไป
        (solvency ไม่ได้ถูกใช้ในสูตร composite เลย)"""
        _INFO_CACHE.clear()
        _INFO_ERROR_CACHE.clear()
        mock_resolve.return_value = _resolved_asset(ticker="LEVERAGED")
        mock_autopsy.return_value = _autopsy_result(ticker="LEVERAGED")
        info_high_leverage = {
            "trailingPE": 18.0, "priceToBook": 5.0, "returnOnEquity": 0.25, "profitMargins": 0.18,
            "currentPrice": 200.0, "debtToEquity": 500.0, "currentRatio": 0.2,
        }
        info_low_leverage = dict(info_high_leverage, debtToEquity=50.0, currentRatio=2.0)
        mock_beta.return_value = (None, _INVALID_QUALITY)
        mock_vol.return_value = (None, _INVALID_QUALITY)
        mock_mdd.return_value = (None, _INVALID_QUALITY)
        mock_tech.return_value = (None, _INVALID_QUALITY)
        mock_pctile.return_value = (None, None, _INVALID_QUALITY)
        mock_revisions.return_value = (None, "missing_earnings_revision_data:earnings_momentum")

        mock_info.return_value = info_high_leverage
        result_high = compute_equity_quant_signals.invoke({"ticker": "LEVERAGED", "market": "US"})
        signals_high = QuantSignals.model_validate(json.loads(result_high))

        _INFO_CACHE.clear()
        _INFO_ERROR_CACHE.clear()
        mock_info.return_value = info_low_leverage
        result_low = compute_equity_quant_signals.invoke({"ticker": "LEVERAGED", "market": "US"})
        signals_low = QuantSignals.model_validate(json.loads(result_low))

        assert signals_high.solvency_score < signals_low.solvency_score
        assert "high_leverage_risk:solvency" in signals_high.data_quality_flags
        # composite ต้องเท่ากันทั้งคู่ (value/quality/growth/momentum/dividend inputs เหมือนกันหมด)
        assert signals_high.composite_score == signals_low.composite_score

    @patch("tools.market.equity_quant_tool.fetch_earnings_revision_data")
    @patch("tools.market.equity_quant_tool.compute_price_percentile")
    @patch("tools.market.equity_quant_tool.compute_technical_indicators")
    @patch("tools.market.equity_quant_tool.compute_mdd")
    @patch("tools.market.equity_quant_tool.compute_volatility")
    @patch("tools.market.equity_quant_tool.compute_beta")
    @patch("tools.market.equity_quant_tool._yf_info")
    @patch("tools.market.equity_quant_tool.get_financial_autopsy")
    @patch("tools.market.equity_quant_tool.resolve_asset")
    def test_negative_pe_flags_instead_of_scoring(
        self, mock_resolve, mock_autopsy, mock_info, mock_beta, mock_vol, mock_mdd, mock_tech, mock_pctile,
        mock_revisions,
    ):
        _INFO_CACHE.clear()
        _INFO_ERROR_CACHE.clear()
        mock_resolve.return_value = _resolved_asset(ticker="LOSSCO")
        autopsy = _autopsy_result(ticker="LOSSCO")
        autopsy.snapshot.current_pe = -5.0
        mock_autopsy.return_value = autopsy
        mock_info.return_value = {"trailingPE": -5.0}
        mock_beta.return_value = (None, _INVALID_QUALITY)
        mock_vol.return_value = (None, _INVALID_QUALITY)
        mock_mdd.return_value = (None, _INVALID_QUALITY)
        mock_tech.return_value = (None, _INVALID_QUALITY)
        mock_pctile.return_value = (None, None, _INVALID_QUALITY)
        mock_revisions.return_value = (None, "missing_earnings_revision_data:earnings_momentum")

        result = compute_equity_quant_signals.invoke({"ticker": "LOSSCO", "market": "US"})
        signals = QuantSignals.model_validate(json.loads(result))
        assert signals.value_score is None
        assert signals.momentum_score is None
        assert "negative_earnings:pe_undefined" in signals.data_quality_flags
        assert "insufficient_trading_history:beta" in signals.data_quality_flags
        assert "insufficient_trading_history:technical_indicators" in signals.data_quality_flags


    @patch("tools.market.equity_quant_tool.fetch_earnings_revision_data")
    @patch("tools.market.equity_quant_tool.compute_price_percentile")
    @patch("tools.market.equity_quant_tool.compute_technical_indicators")
    @patch("tools.market.equity_quant_tool.compute_mdd")
    @patch("tools.market.equity_quant_tool.compute_volatility")
    @patch("tools.market.equity_quant_tool.compute_beta")
    @patch("tools.market.equity_quant_tool._yf_info")
    @patch("tools.market.equity_quant_tool.get_financial_autopsy")
    @patch("tools.market.equity_quant_tool.resolve_asset")
    def test_zero_pe_from_autopsy_does_not_silently_fallback_to_info(
        self, mock_resolve, mock_autopsy, mock_info, mock_beta, mock_vol, mock_mdd, mock_tech, mock_pctile,
        mock_revisions,
    ):
        """autopsy.current_pe == 0.0 เป็นค่า falsy ใน Python — ต้องไม่ถูกมองข้ามไปใช้ info['trailingPE'] แทน"""
        _INFO_CACHE.clear()
        _INFO_ERROR_CACHE.clear()
        mock_resolve.return_value = _resolved_asset(ticker="ZEROPE")
        autopsy = _autopsy_result(ticker="ZEROPE")
        autopsy.snapshot.current_pe = 0.0
        mock_autopsy.return_value = autopsy
        # ถ้าโค้ดมี falsy-zero bug จะเผลอไปใช้ trailingPE=15.0 นี้แทน ทำให้ value_score ไม่ None
        mock_info.return_value = {"trailingPE": 15.0}
        mock_beta.return_value = (None, _INVALID_QUALITY)
        mock_vol.return_value = (None, _INVALID_QUALITY)
        mock_mdd.return_value = (None, _INVALID_QUALITY)
        mock_tech.return_value = (None, _INVALID_QUALITY)
        mock_pctile.return_value = (None, None, _INVALID_QUALITY)
        mock_revisions.return_value = (None, "missing_earnings_revision_data:earnings_momentum")

        result = compute_equity_quant_signals.invoke({"ticker": "ZEROPE", "market": "US"})
        signals = QuantSignals.model_validate(json.loads(result))
        # P/E=0.0 ไม่เข้าเงื่อนไข pe<=0 ของ compute_value_score (0.0 ไม่ <= 0 เป็น False เพราะ 0<=0 คือ True จริง)
        # แต่ประเด็นคือค่าที่ใช้ต้องเป็น 0.0 จาก autopsy ไม่ใช่ 15.0 จาก info ที่หลุดมาแทน
        assert signals.value_score is None  # pe=0.0 -> เข้า guard "negative_earnings:pe_undefined" (pe<=0)
        assert "negative_earnings:pe_undefined" in signals.data_quality_flags


class TestGetInfoCached:
    def test_caches_successful_result(self):
        _INFO_CACHE.clear()
        _INFO_ERROR_CACHE.clear()
        with patch("tools.market.equity_quant_tool._yf_info") as mock_yf_info:
            mock_yf_info.return_value = {"shortName": "Test Co"}
            result1 = _get_info_cached("TESTCO")
            result2 = _get_info_cached("TESTCO")
            assert result1 == {"shortName": "Test Co"}
            assert result2 == {"shortName": "Test Co"}
            mock_yf_info.assert_called_once()

    def test_caches_failure_short_ttl(self):
        _INFO_CACHE.clear()
        _INFO_ERROR_CACHE.clear()
        with patch("tools.market.equity_quant_tool._yf_info") as mock_yf_info:
            mock_yf_info.side_effect = RuntimeError("network down")
            with pytest.raises(Exception):
                _get_info_cached("BADCO")
            with pytest.raises(Exception):
                _get_info_cached("BADCO")
            mock_yf_info.assert_called_once()


class TestComputeEquityQuantSignalsErrorPath:
    @patch("tools.market.equity_quant_tool.resolve_asset")
    def test_exception_returns_error_string_not_raise(self, mock_resolve):
        mock_resolve.side_effect = RuntimeError("boom")
        result = compute_equity_quant_signals.invoke({"ticker": "BROKEN", "market": "US"})
        assert "QUANT_EXECUTION_ERROR" in result or result.startswith("Error:")
