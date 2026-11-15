"""Unit tests สำหรับ tools/market/financial_autopsy.py"""
from unittest.mock import patch, MagicMock
from datetime import datetime
import pandas as pd
import pytest
from pydantic import ValidationError
from tools.market.asset_resolver import ResolvedAsset, AssetClass
from tools.market.financial_autopsy import (
    FinancialAutopsyPeriod,
    FinancialAutopsySnapshot,
    FinancialAutopsyFetchResult,
    get_financial_autopsy,
    ingest_financial_autopsy,
    _AUTOPSY_SUCCESS_CACHE,
    _AUTOPSY_ERROR_CACHE,
    _build_rolling_ttm_period,
)


def test_fetch_result_state_invariants():
    asset = ResolvedAsset(
        raw_symbol="PTT",
        provider_symbol="PTT.BK",
        asset_class=AssetClass.STOCK_TH,
        market="TH",
        confidence="high",
        eligible_for_financial_autopsy=True,
    )
    # Success without snapshot -> raises ValidationError
    with pytest.raises(ValidationError):
        FinancialAutopsyFetchResult(asset=asset, status="success", snapshot=None)

    # Error with snapshot -> raises ValidationError
    snap = FinancialAutopsySnapshot(
        ticker="PTT",
        provider_symbol="PTT.BK",
        market="TH",
        currency="THB",
        unit="raw",
        retrieval_timestamp="2026-07-20 12:00:00",
        periods=[],
    )
    with pytest.raises(ValidationError):
        FinancialAutopsyFetchResult(asset=asset, status="error", snapshot=snap, error_message="Some error")

    # Error without error_message -> raises ValidationError
    with pytest.raises(ValidationError):
        FinancialAutopsyFetchResult(asset=asset, status="error", snapshot=None, error_message=None)


def test_get_financial_autopsy_ineligible_asset():
    _AUTOPSY_SUCCESS_CACHE.clear()
    _AUTOPSY_ERROR_CACHE.clear()
    asset = ResolvedAsset(
        raw_symbol="QQQ",
        provider_symbol="QQQ",
        asset_class=AssetClass.ETF,
        market="US",
        confidence="high",
        eligible_for_financial_autopsy=False,
    )
    res = get_financial_autopsy(asset)
    assert res.status == "unavailable"
    assert res.error_code == "INELIGIBLE_ASSET"
    assert res.snapshot is None


@patch("tools.market.financial_autopsy._fetch_autopsy_raw")
def test_get_financial_autopsy_success_and_caching(mock_fetch):
    _AUTOPSY_SUCCESS_CACHE.clear()
    _AUTOPSY_ERROR_CACHE.clear()
    asset = ResolvedAsset(
        raw_symbol="NVDA",
        provider_symbol="NVDA",
        asset_class=AssetClass.STOCK_US,
        market="US",
        confidence="high",
        eligible_for_financial_autopsy=True,
    )

    # Mock dataframes
    dates = ["2026-01-31", "2025-01-31"]
    df_fin = pd.DataFrame(
        {"2026-01-31": [1000.0, 200.0], "2025-01-31": [800.0, 150.0]},
        index=["Total Revenue", "Net Income"]
    )
    df_cf = pd.DataFrame(
        {"2026-01-31": [250.0, 300.0, -50.0, -20.0], "2025-01-31": [180.0, 220.0, -40.0, -15.0]},
        index=["Free Cash Flow", "Operating Cash Flow", "Capital Expenditure", "Common Stock Dividend Paid"]
    )
    df_bs = pd.DataFrame(
        {"2026-01-31": [500.0], "2025-01-31": [450.0]},
        index=["Total Debt"]
    )
    mock_info = {"currency": "USD", "trailingPE": 40.5, "forwardPE": 30.2}

    mock_fetch.return_value = (mock_info, df_fin, df_cf, df_bs)

    res = get_financial_autopsy(asset)
    assert res.status == "success"
    assert res.snapshot is not None
    assert len(res.snapshot.periods) == 2
    assert res.snapshot.periods[0].free_cash_flow == 250.0
    assert res.snapshot.periods[0].total_debt == 500.0
    assert res.snapshot.periods[0].payout_ratio_pct == 10.0  # (20 / 200) * 100

    # Verify success caching
    res_cached = get_financial_autopsy(asset)
    assert res_cached == res
    assert mock_fetch.call_count == 1


@patch("tools.market.financial_autopsy._fetch_autopsy_raw")
def test_get_financial_autopsy_timeout_or_error(mock_fetch):
    _AUTOPSY_SUCCESS_CACHE.clear()
    _AUTOPSY_ERROR_CACHE.clear()
    asset = ResolvedAsset(
        raw_symbol="FAILCO",
        provider_symbol="FAILCO",
        asset_class=AssetClass.STOCK_US,
        market="US",
        confidence="high",
        eligible_for_financial_autopsy=True,
    )
    mock_fetch.side_effect = Exception("Connection reset by peer")
    res = get_financial_autopsy(asset)
    assert res.status == "error"
    assert res.error_code == "PROVIDER_ERROR"
    assert "Connection reset by peer" in res.error_message


def test_ingest_financial_autopsy_tool():
    _AUTOPSY_SUCCESS_CACHE.clear()
    _AUTOPSY_ERROR_CACHE.clear()
    # Test tool with ineligible ETF symbol directly
    out_json = ingest_financial_autopsy.invoke({"symbol": "QQQ", "market": "US"})
    assert "INELIGIBLE_ASSET" in out_json


@patch("tools.market.financial_autopsy._fetch_autopsy_raw")
def test_stress_resource_accumulation_under_timeout(mock_fetch):
    """Stress test assuring repeated timeouts do not crash or accumulate blocking worker pool indefinitely."""
    import time
    from concurrent.futures import TimeoutError as FuturesTimeoutError
    _AUTOPSY_SUCCESS_CACHE.clear()
    _AUTOPSY_ERROR_CACHE.clear()
    asset = ResolvedAsset(
        raw_symbol="SLOW",
        provider_symbol="SLOW",
        asset_class=AssetClass.STOCK_US,
        market="US",
        confidence="high",
        eligible_for_financial_autopsy=True,
    )
    # Simulate a slow provider or immediate FuturesTimeoutError representing network timeout
    def slow_or_timeout(sym, timeout):
        raise FuturesTimeoutError("Simulated network timeout")

    mock_fetch.side_effect = slow_or_timeout
    for _ in range(5):
        res = get_financial_autopsy(asset)
        assert res.status == "error"
        assert res.error_code == "PROVIDER_TIMEOUT"
        _AUTOPSY_ERROR_CACHE.clear()  # Clear short TTL cache to force re-invocation


@patch("tools.market.financial_autopsy._fetch_autopsy_raw")
def test_get_financial_autopsy_fiscal_alignment_and_debt_summing(mock_fetch):
    from tools.market.asset_resolver import ResolvedAsset, AssetClass
    from tools.market.financial_autopsy import get_financial_autopsy, _AUTOPSY_SUCCESS_CACHE, _AUTOPSY_ERROR_CACHE

    _AUTOPSY_SUCCESS_CACHE.clear()
    _AUTOPSY_ERROR_CACHE.clear()

    info = {"currency": "THB", "trailingPE": 15.0}
    # financials has column 2025-12-31, cashflow has 2025-12-30 (diff = 1 day <= 45 days)
    # balance_sheet has no 'Total Debt' row, but has 'Long Term Debt' = 500 and 'Short Term Debt' = 100
    dates_fin = [pd.Timestamp("2025-12-31")]
    dates_cf = [pd.Timestamp("2025-12-30")]
    dates_bs = [pd.Timestamp("2025-12-31")]

    financials = pd.DataFrame(index=["Total Revenue", "Net Income"], columns=dates_fin)
    financials.loc["Total Revenue", pd.Timestamp("2025-12-31")] = 1000.0
    financials.loc["Net Income", pd.Timestamp("2025-12-31")] = 200.0

    cashflow = pd.DataFrame(index=["Free Cash Flow"], columns=dates_cf)
    cashflow.loc["Free Cash Flow", pd.Timestamp("2025-12-30")] = 150.0

    balance_sheet = pd.DataFrame(index=["Long Term Debt", "Short Term Debt"], columns=dates_bs)
    balance_sheet.loc["Long Term Debt", pd.Timestamp("2025-12-31")] = 500.0
    balance_sheet.loc["Short Term Debt", pd.Timestamp("2025-12-31")] = 100.0

    mock_fetch.return_value = (info, financials, cashflow, balance_sheet)

    asset = ResolvedAsset(
        raw_symbol="ADVANC",
        provider_symbol="ADVANC.BK",
        asset_class=AssetClass.STOCK_TH,
        market="TH",
        confidence="high",
        eligible_for_financial_autopsy=True,
    )
    res = get_financial_autopsy(asset)
    assert res.status == "success"
    assert res.snapshot is not None
    # Verify that exactly 1 canonical period record is created (not two separate ones for Dec 31 and Dec 30)
    assert len(res.snapshot.periods) == 1
    p = res.snapshot.periods[0]
    assert p.fiscal_period_end == "2025-12-31"
    assert p.total_revenue == 1000.0
    assert p.free_cash_flow == 150.0
    # Verify that total_debt summed long term (500) + short term (100) = 600
    assert p.total_debt == 600.0


@pytest.mark.parametrize(
    "dates",
    [
        ["2026-09-30", "2026-06-30", "2025-12-31", "2025-09-30"],
        ["2026-09-30", "2025-12-31", "2025-09-30", "2025-06-30"],
    ],
)
def test_rolling_ttm_rejects_missing_fiscal_quarters(dates):
    financials = pd.DataFrame(
        [[200.0, 100.0, 80.0, 60.0], [80.0, 20.0, 20.0, 20.0]],
        index=["Total Revenue", "Gross Profit"],
        columns=pd.to_datetime(dates),
    )
    assert _build_rolling_ttm_period(financials, None) is None


def test_rolling_ttm_accepts_off_calendar_52_53_week_quarter_ends():
    dates = pd.to_datetime(["2026-10-03", "2026-06-27", "2026-03-28", "2025-12-27"])
    financials = pd.DataFrame(
        [[200.0, 100.0, 80.0, 60.0], [80.0, 20.0, 20.0, 20.0]],
        index=["Total Revenue", "Gross Profit"],
        columns=dates,
    )
    ttm = _build_rolling_ttm_period(financials, None)
    assert ttm is not None
    assert ttm.period_type == "ttm"
    assert ttm.total_revenue == 440.0
    assert ttm.gross_profit == 140.0


def test_rolling_ttm_does_not_fill_missing_latest_income_period_from_older_quarter():
    financial_dates = pd.to_datetime(["2026-06-30", "2026-03-31", "2025-12-31"])
    cashflow_dates = pd.to_datetime(["2026-09-30", "2026-06-30", "2026-03-31", "2025-12-31"])
    financials = pd.DataFrame(
        [[100.0, 80.0, 60.0], [20.0, 20.0, 20.0]],
        index=["Total Revenue", "Gross Profit"],
        columns=financial_dates,
    )
    cashflow = pd.DataFrame(
        [[50.0, 25.0, 20.0, 15.0]],
        index=["Operating Cash Flow"],
        columns=cashflow_dates,
    )

    ttm = _build_rolling_ttm_period(financials, cashflow)
    assert ttm is not None
    assert ttm.fiscal_period_end == "2026-09-30 (TTM)"
    assert ttm.total_revenue is None
    assert ttm.gross_profit is None

    from tools.market.sector_kpi_engine import compute_sector_kpis

    for industry, name in (
        ("Semiconductors", "Gross Margin (%)"),
        ("Software - Infrastructure", "Subscription / Gross Margin"),
    ):
        summary = compute_sector_kpis(
            sector="Technology",
            periods=[],
            industry=industry,
            ttm_period=ttm,
        )
        gross_margin = next(kpi for kpi in summary.kpis if kpi.name == name)
        assert gross_margin.status == "unavailable"
        assert gross_margin.value is None
        assert gross_margin.period_basis == "ttm"
