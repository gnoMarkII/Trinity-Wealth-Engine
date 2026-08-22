from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.routes_ohlcv import _calculate_pivot_levels, _CACHE

client = TestClient(app)


@pytest.fixture(autouse=True)
def override_require_session():
    from api.auth import require_session
    app.dependency_overrides[require_session] = lambda: {"user_id": "mock_user"}
    yield
    app.dependency_overrides = {}


@pytest.fixture(autouse=True)
def clear_cache():
    _CACHE.clear()
    yield
    _CACHE.clear()


def _make_sample_df(start_date: str, days: int, base_price: float = 150.0) -> pd.DataFrame:
    dates = pd.date_range(start=start_date, periods=days, freq="D", tz="America/New_York")
    data = {
        "Open": [base_price + i * 0.5 for i in range(days)],
        "High": [base_price + i * 0.5 + 2.0 for i in range(days)],
        "Low": [base_price + i * 0.5 - 1.0 for i in range(days)],
        "Close": [base_price + i * 0.5 + 1.0 for i in range(days)],
        "Volume": [1000000 + i * 1000 for i in range(days)],
    }
    return pd.DataFrame(data, index=dates)


def _make_monthly_df(dates_list: list[str]) -> pd.DataFrame:
    dates = [pd.Timestamp(d, tz="America/New_York") for d in dates_list]
    data = {
        "Open": [100.0, 110.0, 120.0][: len(dates)],
        "High": [120.0, 130.0, 140.0][: len(dates)],
        "Low": [90.0, 100.0, 110.0][: len(dates)],
        "Close": [115.0, 125.0, 135.0][: len(dates)],
        "Volume": [50000000, 60000000, 70000000][: len(dates)],
    }
    return pd.DataFrame(data, index=dates)


def test_pivot_levels_calculation_current_month_ongoing():
    # If the last candle is the current ongoing month, should use iloc[-2]
    now_ny = datetime.now(ZoneInfo("America/New_York"))
    current_month_str = now_ny.strftime("%Y-%m-01")
    
    # Calculate previous month date
    first_of_curr = now_ny.replace(day=1)
    prev_month_last_day = first_of_curr - timedelta(days=1)
    prev_month_str = prev_month_last_day.strftime("%Y-%m-01")

    monthly_df = pd.DataFrame(
        {
            "Open": [100.0, 110.0],
            "High": [120.0, 140.0],  # iloc[-2] High = 120.0
            "Low": [90.0, 100.0],    # iloc[-2] Low = 90.0
            "Close": [110.0, 130.0], # iloc[-2] Close = 110.0
            "Volume": [10000, 20000],
        },
        index=[
            pd.Timestamp(prev_month_str, tz="America/New_York"),
            pd.Timestamp(current_month_str, tz="America/New_York"),
        ],
    )

    levels, period, as_of = _calculate_pivot_levels(monthly_df, "America/New_York")
    assert period == "monthly"
    assert as_of == prev_month_last_day.strftime("%Y-%m")
    assert levels is not None
    # Pivot = (120 + 90 + 110) / 3 = 320 / 3 = 106.6667
    raw_pivot = 320.0 / 3.0
    assert levels.pivot == round(raw_pivot, 4)
    assert levels.r1 == round(2.0 * raw_pivot - 90.0, 4)
    assert levels.s1 == round(2.0 * raw_pivot - 120.0, 4)


def test_pivot_levels_calculation_past_month_completed():
    # If the last candle is a past completed month, should use iloc[-1]
    monthly_df = pd.DataFrame(
        {
            "Open": [100.0],
            "High": [150.0],
            "Low": [100.0],
            "Close": [140.0],
            "Volume": [10000],
        },
        index=[pd.Timestamp("2025-01-01", tz="America/New_York")],
    )

    levels, period, as_of = _calculate_pivot_levels(monthly_df, "America/New_York")
    assert period == "monthly"
    assert as_of == "2025-01"
    assert levels is not None
    # Pivot = (150 + 100 + 140) / 3 = 390 / 3 = 130.0
    assert levels.pivot == 130.0
    assert levels.r1 == 2.0 * 130.0 - 100.0  # 160.0
    assert levels.r2 == 130.0 + (150.0 - 100.0)  # 180.0
    assert levels.r3 == 150.0 + 2.0 * (130.0 - 100.0)  # 210.0
    assert levels.s1 == 2.0 * 130.0 - 150.0  # 110.0
    assert levels.s2 == 130.0 - (150.0 - 100.0)  # 80.0
    assert levels.s3 == 100.0 - 2.0 * (150.0 - 130.0)  # 60.0
    assert levels.s4 == 60.0 - (150.0 - 100.0)  # 10.0


@patch("api.routes_ohlcv.yf.Ticker")
def test_get_equity_ohlcv_success_us(mock_ticker_cls):
    mock_instance = MagicMock()
    mock_ticker_cls.return_value = mock_instance

    chart_df = _make_sample_df("2026-01-01", 10, base_price=100.0)
    daily_df = _make_sample_df("2026-01-06", 5, base_price=105.0)
    monthly_df = pd.DataFrame(
        {
            "Open": [100.0, 110.0],
            "High": [120.0, 140.0],
            "Low": [90.0, 100.0],
            "Close": [110.0, 130.0],
            "Volume": [10000, 20000],
        },
        index=[
            pd.Timestamp("2026-06-01", tz="America/New_York"),
            pd.Timestamp("2026-07-01", tz="America/New_York"),
        ],
    )

    def mock_history(period, interval, *args, **kwargs):
        if interval == "1mo":
            return monthly_df
        if period == "5d" and interval == "1d":
            return daily_df
        return chart_df

    mock_instance.history.side_effect = mock_history

    response = client.get("/api/equity/AAPL/ohlcv?range=6mo&interval=1d")
    assert response.status_code == 200
    data = response.json()

    assert data["ticker"] == "AAPL"
    assert data["market"] == "US"
    assert data["currency"] == "USD"
    assert len(data["candles"]) == 10
    
    # Check millisecond timestamps
    first_candle = data["candles"][0]
    assert first_candle["timestamp"] > 10**12
    assert "open" in first_candle
    assert "high" in first_candle
    assert "low" in first_candle
    assert "close" in first_candle
    assert "volume" in first_candle

    assert data["current_price"] is not None
    assert data["price_change"] is not None
    assert data["price_change_pct"] is not None
    assert data["price_as_of"] is not None
    assert data["pivot_levels"] is not None
    assert data["pivot_period"] == "monthly"


@patch("api.routes_ohlcv.yf.Ticker")
def test_get_equity_ohlcv_th_market(mock_ticker_cls):
    mock_instance = MagicMock()
    mock_ticker_cls.return_value = mock_instance

    chart_df = _make_sample_df("2026-01-01", 5, base_price=35.0)
    mock_instance.history.return_value = chart_df

    response = client.get("/api/equity/PTT/ohlcv")
    assert response.status_code == 200
    data = response.json()

    assert data["ticker"] == "PTT"
    assert data["market"] == "TH"
    assert data["currency"] == "THB"


@patch("api.routes_ohlcv.yf.Ticker")
def test_get_equity_ohlcv_caching(mock_ticker_cls):
    mock_instance = MagicMock()
    mock_ticker_cls.return_value = mock_instance
    mock_instance.history.return_value = _make_sample_df("2026-01-01", 5, base_price=50.0)

    # First request
    res1 = client.get("/api/equity/NVDA/ohlcv?range=3mo&interval=1d")
    assert res1.status_code == 200
    call_count_1 = mock_instance.history.call_count

    # Second request with same params should hit cache
    res2 = client.get("/api/equity/NVDA/ohlcv?range=3mo&interval=1d")
    assert res2.status_code == 200
    assert mock_instance.history.call_count == call_count_1


def test_get_equity_ohlcv_invalid_params():
    res1 = client.get("/api/equity/AAPL/ohlcv?range=invalid")
    assert res1.status_code == 400
    assert "Invalid" in res1.json()["detail"]

    res2 = client.get("/api/equity/AAPL/ohlcv?interval=invalid")
    assert res2.status_code == 400
    assert "Invalid interval" in res2.json()["detail"]

    res3 = client.get("/api/equity/A..B/ohlcv")
    assert res3.status_code == 400
    assert "Path traversal" in res3.json()["detail"]

    res4 = client.get("/api/equity/AAPL$$/ohlcv")
    assert res4.status_code == 400
    assert "Invalid ticker format" in res4.json()["detail"]


@patch("api.routes_ohlcv.yf.Ticker")
def test_get_equity_ohlcv_not_found(mock_ticker_cls):
    mock_instance = MagicMock()
    mock_ticker_cls.return_value = mock_instance
    mock_instance.history.return_value = pd.DataFrame()

    response = client.get("/api/equity/UNKNOWN_TICKER/ohlcv")
    assert response.status_code == 404


def test_get_fetch_period_strategy():
    from api.routes_ohlcv import _get_fetch_period
    assert _get_fetch_period("6mo", "1d") == "2y"
    assert _get_fetch_period("1mo", "1d") == "2y"
    assert _get_fetch_period("3mo", "1d") == "2y"
    assert _get_fetch_period("1y", "1d") == "3y"
    assert _get_fetch_period("5y", "1wk") == "10y"
    assert _get_fetch_period("max", "1d") == "max"
    assert _get_fetch_period("5y", "1mo") == "10y"
    assert _get_fetch_period("max", "1mo") == "max"


def test_calculate_warmup_metadata_max_range():
    from api.schemas import OHLCVCandleDTO
    from api.routes_ohlcv import _calculate_warmup_metadata

    candles = [
        OHLCVCandleDTO(timestamp=1000, open=10, high=11, low=9, close=10, volume=100),
        OHLCVCandleDTO(timestamp=2000, open=10, high=11, low=9, close=10, volume=100),
        OHLCVCandleDTO(timestamp=3000, open=10, high=11, low=9, close=10, volume=100),
    ]

    start_ts, avail, req, status, ind_warmup = _calculate_warmup_metadata(candles, "max", "1d", "America/New_York")
    assert start_ts == 1000
    assert avail == 0
    assert req == 0
    assert status == "not_applicable"


def test_calculate_warmup_metadata_sufficient_and_insufficient():
    from api.schemas import OHLCVCandleDTO
    from api.routes_ohlcv import _calculate_warmup_metadata

    # 1d series: 350 daily bars spaced by 86400 * 1000 ms
    base_ts = 1700000000000
    day_ms = 86400 * 1000

    candles_350 = [
        OHLCVCandleDTO(
            timestamp=base_ts + i * day_ms,
            open=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=1000,
        )
        for i in range(350)
    ]

    # For 1mo range (approx 30 days = 30 bars), warmup bars approx 320 >= 200 -> sufficient
    start_ts, avail, req, status, ind_warmup = _calculate_warmup_metadata(candles_350, "1mo", "1d", "America/New_York")
    assert req == 200
    assert avail >= 200
    assert status == "sufficient"
    assert start_ts in [c.timestamp for c in candles_350]
    assert ind_warmup["EMA200"].status == "full"

    # Short IPO series: 60 daily bars -> insufficient for 1d
    candles_60 = candles_350[:60]
    start_ts_60, avail_60, req_60, status_60, ind_warmup_60 = _calculate_warmup_metadata(candles_60, "1mo", "1d", "America/New_York")
    assert req_60 == 200
    assert avail_60 < 200
    assert status_60 == "insufficient"
    assert ind_warmup_60["EMA200"].status == "unavailable"


def test_calculate_warmup_metadata_leap_year_and_month_end():
    from api.schemas import OHLCVCandleDTO
    from api.routes_ohlcv import _calculate_warmup_metadata
    from datetime import datetime
    from zoneinfo import ZoneInfo

    tz = ZoneInfo("America/New_York")
    
    # Series ending on March 31 of a leap year (2024-03-31)
    dt_end = datetime(2024, 3, 31, 16, 0, tzinfo=tz)
    dt_feb29 = datetime(2024, 2, 29, 16, 0, tzinfo=tz)
    dt_feb28 = datetime(2024, 2, 28, 16, 0, tzinfo=tz)

    candles = [
        OHLCVCandleDTO(timestamp=int(dt_feb28.timestamp() * 1000), open=100, high=101, low=99, close=100, volume=100),
        OHLCVCandleDTO(timestamp=int(dt_feb29.timestamp() * 1000), open=100, high=101, low=99, close=100, volume=100),
        OHLCVCandleDTO(timestamp=int(dt_end.timestamp() * 1000), open=100, high=101, low=99, close=100, volume=100),
    ]

    # 1mo cutoff from 2024-03-31 with relativedelta gives 2024-02-29
    start_ts, _, _, _, _ = _calculate_warmup_metadata(candles, "1mo", "1d", "America/New_York")
    assert start_ts == int(dt_feb29.timestamp() * 1000)

    # 1y cutoff from 2024-02-29 (leap day) to non-leap year (2023-02-28)
    dt_2023_feb28 = datetime(2023, 2, 28, 16, 0, tzinfo=tz)
    dt_2024_feb29 = datetime(2024, 2, 29, 16, 0, tzinfo=tz)
    candles_leap = [
        OHLCVCandleDTO(timestamp=int(dt_2023_feb28.timestamp() * 1000), open=100, high=101, low=99, close=100, volume=100),
        OHLCVCandleDTO(timestamp=int(dt_2024_feb29.timestamp() * 1000), open=100, high=101, low=99, close=100, volume=100),
    ]
    start_ts_leap, _, _, _, _ = _calculate_warmup_metadata(candles_leap, "1y", "1d", "America/New_York")
    assert start_ts_leap == int(dt_2023_feb28.timestamp() * 1000)


@patch("api.routes_ohlcv.yf.Ticker")
def test_get_equity_ohlcv_metadata_and_invariants(mock_ticker_cls):
    mock_instance = MagicMock()
    mock_ticker_cls.return_value = mock_instance

    # Generate 500 days of data
    chart_df = _make_sample_df("2024-01-01", 500, base_price=100.0)
    mock_instance.history.return_value = chart_df

    response = client.get("/api/equity/AAPL/ohlcv?range=6mo&interval=1d")
    assert response.status_code == 200
    data = response.json()

    assert data["requested_range"] == "6mo"
    assert data["interval"] == "1d"
    assert data["display_start_timestamp"] is not None
    assert data["available_warmup_bars"] >= 200
    assert data["required_warmup_bars"] == 200
    assert data["warmup_status"] == "sufficient"

    # Verify Invariants:
    timestamps = [c["timestamp"] for c in data["candles"]]
    assert len(timestamps) == len(set(timestamps)), "Timestamps must be unique"
    for i in range(len(timestamps) - 1):
        assert timestamps[i] < timestamps[i + 1], "Timestamps must be strictly increasing"

    assert data["display_start_timestamp"] in timestamps
    visible_candles = [c for c in data["candles"] if c["timestamp"] >= data["display_start_timestamp"]]
    assert len(visible_candles) >= 1



@patch("tools.market.earnings.yf.Ticker")
@patch("api.routes_ohlcv.yf.Ticker")
def test_corporate_actions_raw_cache_multi_interval_reuse(mock_ticker_cls, mock_earnings_ticker_cls):
    from api.routes_ohlcv import _ACTION_CACHE
    import tools.market.earnings as earn_mod
    _ACTION_CACHE.clear()
    earn_mod._CACHE.clear()
    earn_mod._BURST_FAIL_CACHE.clear()

    mock_instance = MagicMock()
    mock_ticker_cls.return_value = mock_instance
    mock_earnings_ticker_cls.return_value = mock_instance

    # Mock OHLCV history
    daily_df = _make_sample_df("2024-01-01", 300, base_price=100.0)
    mock_instance.history.return_value = daily_df

    # Mock Earnings dates (reported inside the history)
    ed_df = pd.DataFrame(
        {
            "Reported EPS": [1.50],
            "EPS Estimate": [1.20],
        },
        index=[pd.Timestamp("2024-03-15", tz="America/New_York")],
    )
    mock_instance.get_earnings_dates.return_value = ed_df

    # Mock Dividends
    div_s = pd.Series([0.25], index=[pd.Timestamp("2024-05-10", tz="America/New_York")])
    mock_instance.dividends = div_s

    # Mock Splits
    splits_s = pd.Series([4.0], index=[pd.Timestamp("2024-06-07", tz="America/New_York")])
    mock_instance.splits = splits_s

    # 1. First call: 1d
    res1 = client.get("/api/equity/AAPL/ohlcv?range=6mo&interval=1d")
    assert res1.status_code == 200
    data1 = res1.json()
    assert len(data1["events"]) == 3
    assert mock_instance.get_earnings_dates.call_count == 1

    # Check Earnings Beat
    e_event = next(e for e in data1["events"] if e["event_type"] == "earnings")
    assert e_event["color"] == "green"
    assert e_event["label"] == "E"
    assert e_event["eps_actual"] == 1.50
    assert e_event["eps_estimate"] == 1.20
    assert e_event["mapping_method"] == "reported_date"

    # Check Split
    s_event = next(e for e in data1["events"] if e["event_type"] == "split")
    assert s_event["color"] == "purple"
    assert s_event["label"] == "S"
    assert s_event["split_formatted"] == "4-for-1 forward split"

    # 2. Second call: 1wk (should reuse raw action cache without calling get_earnings_dates again)
    weekly_dates = pd.date_range(start="2024-01-05", periods=52, freq="W-FRI", tz="America/New_York")
    weekly_df = pd.DataFrame(
        {
            "Open": [100.0 + i for i in range(52)],
            "High": [105.0 + i for i in range(52)],
            "Low": [95.0 + i for i in range(52)],
            "Close": [102.0 + i for i in range(52)],
            "Volume": [5000000 + i * 10000 for i in range(52)],
        },
        index=weekly_dates,
    )
    mock_instance.history.return_value = weekly_df

    res2 = client.get("/api/equity/AAPL/ohlcv?range=1y&interval=1wk")
    assert res2.status_code == 200
    data2 = res2.json()
    assert len(data2["events"]) >= 1
    # Ensure get_earnings_dates was NOT called a second time due to raw cache reuse
    assert mock_instance.get_earnings_dates.call_count == 1


@patch("tools.market.earnings.yf.Ticker")
@patch("api.routes_ohlcv.yf.Ticker")
def test_corporate_actions_future_scheduled_earnings_filtered(mock_ticker_cls, mock_earnings_ticker_cls):
    from api.routes_ohlcv import _ACTION_CACHE
    import tools.market.earnings as earn_mod
    _ACTION_CACHE.clear()
    earn_mod._CACHE.clear()
    earn_mod._BURST_FAIL_CACHE.clear()

    mock_instance = MagicMock()
    mock_ticker_cls.return_value = mock_instance
    mock_earnings_ticker_cls.return_value = mock_instance

    daily_df = _make_sample_df("2024-01-01", 100, base_price=100.0) # Ends ~2024-04-10
    mock_instance.history.return_value = daily_df

    # Scheduled earnings on 2024-12-01 (far future compared to latest daily candle)
    ed_df = pd.DataFrame(
        {
            "Reported EPS": [None],
            "EPS Estimate": [2.00],
        },
        index=[pd.Timestamp("2024-12-01", tz="America/New_York")],
    )
    mock_instance.get_earnings_dates.return_value = ed_df
    mock_instance.dividends = pd.Series([], dtype=float)
    mock_instance.splits = pd.Series([], dtype=float)

    res = client.get("/api/equity/AAPL/ohlcv?range=3mo&interval=1d")
    assert res.status_code == 200
    data = res.json()
    # Future scheduled earnings must be omitted
    earnings_events = [e for e in data["events"] if e["event_type"] == "earnings"]
    assert len(earnings_events) == 0


@patch("tools.market.earnings.yf.Ticker")
@patch("api.routes_ohlcv.yf.Ticker")
def test_corporate_actions_partial_status_and_missing_sources(mock_ticker_cls, mock_earnings_ticker_cls):
    from api.routes_ohlcv import _ACTION_CACHE
    import tools.market.earnings as earn_mod
    _ACTION_CACHE.clear()
    earn_mod._CACHE.clear()
    earn_mod._BURST_FAIL_CACHE.clear()

    mock_instance = MagicMock()
    mock_ticker_cls.return_value = mock_instance
    mock_earnings_ticker_cls.return_value = mock_instance

    daily_df = _make_sample_df("2024-01-01", 100, base_price=100.0)
    mock_instance.history.return_value = daily_df

    # Earnings succeeds
    ed_df = pd.DataFrame(
        {
            "Reported EPS": [1.00],
            "EPS Estimate": [1.00], # In-line -> blue
        },
        index=[pd.Timestamp("2024-02-15", tz="America/New_York")],
    )
    mock_instance.get_earnings_dates.return_value = ed_df

    # Dividends raises an exception (simulating failure without retry delay)
    def _raise_div():
        raise ValueError("Dividends provider network error")
    type(mock_instance).dividends = property(lambda self: _raise_div())
    mock_instance.splits = pd.Series([], dtype=float)

    res = client.get("/api/equity/AAPL/ohlcv?range=3mo&interval=1d")
    assert res.status_code == 200
    data = res.json()

    meta = data["events_metadata"]
    assert meta["status"] == "partial"
    assert "dividends" in meta["missing_sources"]
    assert meta["earnings_status"] == "ok"
    assert meta["dividends_status"] == "failed"
    assert len(data["events"]) == 1
    assert data["events"][0]["color"] == "blue" # In-line


@patch("tools.market.earnings.yf.Ticker")
@patch("api.routes_ohlcv.yf.Ticker")
def test_corporate_actions_failure_isolation(mock_ticker_cls, mock_earnings_ticker_cls):
    from api.routes_ohlcv import _ACTION_CACHE
    import tools.market.earnings as earn_mod
    _ACTION_CACHE.clear()
    earn_mod._CACHE.clear()
    earn_mod._BURST_FAIL_CACHE.clear()

    mock_instance = MagicMock()
    mock_ticker_cls.return_value = mock_instance
    mock_earnings_ticker_cls.return_value = mock_instance

    daily_df = _make_sample_df("2024-01-01", 100, base_price=100.0)
    mock_instance.history.return_value = daily_df

    # All action fetches fail (using ValueError to skip retry loop)
    mock_instance.get_earnings_dates.side_effect = ValueError("Rate limited")
    def _raise_err():
        raise ValueError("Timeout")
    type(mock_instance).dividends = property(lambda self: _raise_err())
    type(mock_instance).splits = property(lambda self: _raise_err())

    # OHLCV endpoint must still succeed with HTTP 200
    res = client.get("/api/equity/AAPL/ohlcv?range=3mo&interval=1d")
    assert res.status_code == 200
    data = res.json()
    assert len(data["candles"]) == 100
    assert data["events"] == []
    assert data["events_metadata"]["status"] == "unavailable"


@patch("api.routes_ohlcv.yf.Ticker")
def test_52w_high_low_consistency_and_coverage(mock_ticker_cls):
    mock_instance = MagicMock()
    mock_ticker_cls.return_value = mock_instance

    # 400 days of data starting from 2023-01-01
    daily_df = _make_sample_df("2023-01-01", 400, base_price=100.0)
    mock_instance.history.return_value = daily_df
    mock_instance.get_earnings_dates.return_value = None
    mock_instance.dividends = pd.Series([], dtype=float)
    mock_instance.splits = pd.Series([], dtype=float)

    res = client.get("/api/equity/AAPL/ohlcv?range=6mo&interval=1d")
    assert res.status_code == 200
    data = res.json()

    assert data["price_basis"] == "provider_proportional_adj_close_ratio"
    assert data["week52_high"] is not None
    assert data["week52_low"] is not None
    # 400 calendar days between start and end -> coverage calendar days ~ 399
    assert 360 <= data["week52_coverage_calendar_days"] <= 410


def test_capability_matrix_invalid_combinations_400():
    # Test invalid interval
    res = client.get("/api/equity/AAPL/ohlcv?range=1mo&interval=5m")
    assert res.status_code == 400
    assert "Invalid interval" in res.json()["detail"]

    # Test invalid combinations
    invalid_pairs = [
        ("15m", "1y"),
        ("15m", "max"),
        ("1h", "5y"),
        ("1wk", "1mo"),
        ("1mo", "1y"),
    ]
    for intv, rng in invalid_pairs:
        resp = client.get(f"/api/equity/AAPL/ohlcv?range={rng}&interval={intv}")
        assert resp.status_code == 400
        assert f"Invalid interval and range combination '{intv}' with '{rng}'" in resp.json()["detail"]


@patch("api.routes_ohlcv.yf.Ticker")
def test_capability_matrix_valid_combinations_success(mock_ticker_cls):
    mock_instance = MagicMock()
    mock_ticker_cls.return_value = mock_instance

    sample_df = _make_sample_df("2024-01-01", 100, base_price=100.0)
    mock_instance.history.return_value = sample_df
    mock_instance.get_earnings_dates.return_value = None
    mock_instance.dividends = pd.Series([], dtype=float)
    mock_instance.splits = pd.Series([], dtype=float)

    valid_pairs = [
        ("15m", "5d"),
        ("1h", "3mo"),
        ("1d", "6mo"),
        ("1wk", "1y"),
        ("1mo", "5y"),
    ]
    for intv, rng in valid_pairs:
        resp = client.get(f"/api/equity/AAPL/ohlcv?range={rng}&interval={intv}")
        assert resp.status_code == 200, f"Failed for {intv} + {rng}: {resp.text}"
        data = resp.json()
        assert data["interval"] == intv
        assert data["requested_range"] == rng
        assert "allowed_ranges" in data
        assert rng in data["allowed_ranges"]
        assert "indicator_warmup" in data
        assert "effective_capabilities" in data


@patch("api.routes_ohlcv.yf.Ticker")
def test_3tier_warmup_contract_and_burn_in(mock_ticker_cls):
    mock_instance = MagicMock()
    mock_ticker_cls.return_value = mock_instance

    # 1. Dataset with 400 bars for 6mo range (has plenty of pre-roll -> EMA200 status="full")
    df_400 = _make_sample_df("2023-01-01", 400, base_price=150.0)
    mock_instance.history.return_value = df_400
    mock_instance.get_earnings_dates.return_value = None
    mock_instance.dividends = pd.Series([], dtype=float)
    mock_instance.splits = pd.Series([], dtype=float)

    res_full = client.get("/api/equity/AAPL/ohlcv?range=6mo&interval=1d")
    assert res_full.status_code == 200
    data_full = res_full.json()
    assert data_full["indicator_warmup"]["EMA200"]["status"] == "full"
    assert data_full["indicator_warmup"]["EMA200"]["burn_in_bars_remaining"] == 0

    # 2. Dataset with 220 bars (total bars >= 200, but pre-roll before 6mo cutoff is only ~95 bars -> EMA200 status="partial")
    from api.routes_ohlcv import _CACHE
    _CACHE.clear()
    df_220 = _make_sample_df("2024-01-01", 220, base_price=150.0)
    mock_instance.history.return_value = df_220

    res_partial = client.get("/api/equity/AAPL/ohlcv?range=6mo&interval=1d")
    assert res_partial.status_code == 200
    data_partial = res_partial.json()
    assert data_partial["indicator_warmup"]["EMA200"]["status"] == "partial"
    assert data_partial["indicator_warmup"]["EMA200"]["burn_in_bars_remaining"] > 0
    assert data_partial["indicator_warmup"]["EMA200"]["first_reliable_timestamp"] is not None
    assert data_partial["indicator_warmup"]["EMA200"]["burn_in_policy"]["algorithm_version"] == "ema200_v1.0"

    # 3. Dataset with 10 bars (total bars < 15 -> RSI14 status="unavailable")
    _CACHE.clear()
    df_10 = _make_sample_df("2024-01-01", 10, base_price=150.0)
    mock_instance.history.return_value = df_10

    res_unavail = client.get("/api/equity/AAPL/ohlcv?range=6mo&interval=1d")
    assert res_unavail.status_code == 200
    data_unavail = res_unavail.json()
    assert data_unavail["indicator_warmup"]["RSI14"]["status"] == "unavailable"

