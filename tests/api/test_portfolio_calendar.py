"""Unit tests for GET /api/portfolio/calendar endpoint with isolated portfolio state."""
import sys
from unittest.mock import patch
import pytest


@pytest.fixture
def isolated_calendar_portfolio(tmp_path, monkeypatch):
    """Set up isolated vault path with known Holdings and Watchlist for calendar testing."""
    from tests.conftest import _reset_portfolio_modules
    mods = _reset_portfolio_modules(tmp_path, monkeypatch)
    core = mods.core
    watchlist = mods.watchlist

    # Initialize portfolio state with 1 US holding, 1 TH holding, and 1 watchlist item
    post, state = core._load_or_init()
    h_us = core.Holding(
        symbol="AAPL",
        company_name="Apple Inc.",
        asset_type="Stock",
        units=10.0,
        current_price_thb=6000.0,
        avg_cost_thb=5500.0,
    )
    h_th = core.Holding(
        symbol="PTT",
        company_name="PTT Public Company Limited",
        asset_type="Stock",
        units=1000.0,
        current_price_thb=35.0,
        avg_cost_thb=33.0,
    )
    h_fail = core.Holding(
        symbol="FAILTICKER",
        company_name="Failed Company",
        asset_type="Stock",
        units=1.0,
        current_price_thb=10.0,
        avg_cost_thb=10.0,
    )
    state.holdings = [h_us, h_th, h_fail]
    core._save(post, state)

    # Add Watchlist item
    watchlist.add_to_watchlist.invoke({"symbol": "NVDA", "asset_type": "Stock", "target_price": 120.0})

    return tmp_path


@pytest.fixture
def empty_calendar_portfolio(tmp_path, monkeypatch):
    """Set up isolated empty portfolio vault path."""
    from tests.conftest import _reset_portfolio_modules
    mods = _reset_portfolio_modules(tmp_path, monkeypatch)
    core = mods.core

    post, state = core._load_or_init()
    state.holdings = []
    core._save(post, state)
    return tmp_path


def test_get_portfolio_calendar_success_and_th_mapping(authed_client, isolated_calendar_portfolio, monkeypatch):
    """Test calendar route correctly maps provider_symbol (e.g. PTT -> PTT.BK), populates company_name for holdings, and returns DTO events.

    FAILTICKER resolves with confidence='low' (ไม่ใช่ ticker จริง) จึงถูกกรองออกก่อนถึงขั้น fetch
    calendar เลย ตาม Phase 1 ของ fix — ไม่ปรากฏใน tickers_fetched/tickers_failed อีกต่อไป
    (เดิมทดสอบ path "fetch แล้ว fail" ผ่าน FAILTICKER, ย้ายไปทดสอบแยกใน
    test_get_portfolio_calendar_tickers_failed_on_fetch_error แทน)
    """
    mock_cal = {
        "AAPL": {
            "Earnings Date": ["2026-10-30"],
            "Earnings Average": 1.98,
            "Earnings Low": 1.90,
            "Earnings High": 2.05,
            "Ex-Dividend Date": "2026-08-10",
        },
        "PTT.BK": {
            "Ex-Dividend Date": "2026-09-01",
        },
        "NVDA": {
            "Earnings Date": ["2026-11-15"],
        },
    }

    def mock_get_asset_calendar(provider_symbol):
        return mock_cal.get(provider_symbol, {})

    monkeypatch.setattr("tools.market.calendar.get_asset_calendar", mock_get_asset_calendar)

    res = authed_client.get("/api/portfolio/calendar")
    assert res.status_code == 200
    data = res.json()

    assert "generated_at" in data
    assert data["tickers_fetched"] == 3
    assert data["tickers_failed"] == []

    events = data["events"]
    tickers = [e["ticker"] for e in events]
    assert "AAPL" in tickers
    assert "PTT" in tickers
    assert "NVDA" in tickers
    assert "FAILTICKER" not in tickers

    # Verify company_name logic
    aapl_event = next(e for e in events if e["ticker"] == "AAPL")
    assert aapl_event["company_name"] == "Apple Inc."
    assert aapl_event["bucket"] == "holding"

    ptt_event = next(e for e in events if e["ticker"] == "PTT")
    assert ptt_event["company_name"] == "PTT Public Company Limited"
    assert ptt_event["bucket"] == "holding"

    nvda_event = next(e for e in events if e["ticker"] == "NVDA")
    assert nvda_event["company_name"] is None
    assert nvda_event["bucket"] == "watchlist"


def test_get_portfolio_calendar_filters_low_confidence_tickers(authed_client, isolated_calendar_portfolio, monkeypatch):
    """ticker ที่ resolve ไม่ผ่าน (confidence='low', เช่น FAILTICKER) ต้องถูกกรองออกก่อนถึง fetch
    ไม่ให้ไปรบกวน shared yfinance session ของ ticker อื่นที่ resolve ผ่าน (root cause ของบั๊กจริง)"""
    calls: list[str] = []

    def mock_get_asset_calendar(provider_symbol):
        calls.append(provider_symbol)
        return {}

    monkeypatch.setattr("tools.market.calendar.get_asset_calendar", mock_get_asset_calendar)

    res = authed_client.get("/api/portfolio/calendar")
    assert res.status_code == 200
    assert "FAILTICKER" not in calls


def test_get_portfolio_calendar_tickers_failed_on_fetch_error(authed_client, isolated_calendar_portfolio, monkeypatch):
    """ticker ที่ resolve ผ่าน (confidence สูงพอ) แต่ fetch calendar จริงพัง ต้องยังโผล่ใน tickers_failed"""
    from tools.market.asset_resolver import ResolvedAsset, AssetClass

    def mock_resolve_asset(symbol):
        clean = symbol.strip().upper()
        return ResolvedAsset(
            raw_symbol=clean,
            provider_symbol=clean,
            asset_class=AssetClass.STOCK_US,
            market="US",
            confidence="medium",
            eligible_for_financial_autopsy=True,
        )

    def mock_get_asset_calendar(provider_symbol):
        if provider_symbol == "FAILTICKER":
            raise RuntimeError("Network error fetching FAILTICKER")
        return {}

    monkeypatch.setattr("tools.market.asset_resolver.resolve_asset", mock_resolve_asset)
    monkeypatch.setattr("tools.market.calendar.get_asset_calendar", mock_get_asset_calendar)

    res = authed_client.get("/api/portfolio/calendar")
    assert res.status_code == 200
    data = res.json()
    assert "FAILTICKER" in data["tickers_failed"]


def test_get_portfolio_calendar_empty_state(authed_client, empty_calendar_portfolio, monkeypatch):
    """Test calendar route returns clean empty state when no holdings or watchlist exist."""
    monkeypatch.setattr("tools.market.calendar.get_asset_calendar", lambda sym: {})

    res = authed_client.get("/api/portfolio/calendar")
    assert res.status_code == 200
    data = res.json()

    assert data["events"] == []
    assert data["tickers_fetched"] == 0
    assert data["tickers_failed"] == []
