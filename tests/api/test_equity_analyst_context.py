import json
import math
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.state_db import get_connection, get_analyst_context_cache, upsert_analyst_context_cache
from tools.market.earnings import finite_or_none, fetch_earnings_dates, EarningsFetchResult


@pytest.fixture
def client():
    return TestClient(app)


def test_finite_or_none():
    assert finite_or_none(123.45) == 123.45
    assert finite_or_none("45.67") == 45.67
    assert finite_or_none(math.nan) is None
    assert finite_or_none(float("nan")) is None
    assert finite_or_none(float("inf")) is None
    assert finite_or_none(float("-inf")) is None
    assert finite_or_none(None) is None
    assert finite_or_none("invalid_str") is None


def test_analyst_context_cache_crud(tmp_path):
    db_path = str(tmp_path / "test_state.sqlite")
    conn = get_connection(db_path)

    # Empty cache
    assert get_analyst_context_cache(conn, "AAPL") is None

    # Upsert valid data
    now_wall = time.time()
    upsert_analyst_context_cache(
        conn,
        "AAPL",
        {
            "provider_symbol": "AAPL",
            "market": "US",
            "currency": "USD",
            "exchange_tz": "America/New_York",
            "target_mean": 250.0,
            "target_high": 300.0,
            "target_low": 200.0,
            "num_analysts": 35,
            "next_earnings_date": "2026-10-25",
            "earnings_history": [
                {"date_str": "2026-07-25", "eps_actual": 1.40, "eps_estimate": 1.35}
            ],
            "source_as_of": "2026-08-22T00:00:00",
            "data_status": "ok",
            "synced_at": now_wall,
        },
    )

    cached = get_analyst_context_cache(conn, "AAPL")
    assert cached is not None
    assert cached["ticker"] == "AAPL"
    assert cached["target_mean"] == 250.0
    assert cached["num_analysts"] == 35
    assert isinstance(cached["earnings_history"], list)
    assert len(cached["earnings_history"]) == 1
    assert cached["earnings_history"][0]["eps_actual"] == 1.40

    # Corrupted JSON recovery (object instead of list or syntax error)
    conn.execute(
        "UPDATE analyst_context_cache SET eps_history_json = '{\"not\": \"a list\"}' WHERE ticker = 'AAPL'"
    )
    conn.commit()
    assert get_analyst_context_cache(conn, "AAPL") is None

    conn.execute(
        "UPDATE analyst_context_cache SET eps_history_json = 'corrupted_json{{{' WHERE ticker = 'AAPL'"
    )
    conn.commit()
    assert get_analyst_context_cache(conn, "AAPL") is None


def test_shared_earnings_fetch_sanitization():
    with patch("yfinance.Ticker") as mock_ticker:
        import pandas as pd
        mock_df = pd.DataFrame(
            [
                {"Reported EPS": float("nan"), "EPS Estimate": 1.25},
                {"Reported EPS": 1.50, "EPS Estimate": float("inf")},
            ],
            index=[pd.Timestamp("2026-05-01"), pd.Timestamp("2026-02-01")],
        )
        instance = MagicMock()
        instance.get_earnings_dates.return_value = mock_df
        mock_ticker.return_value = instance

        res = fetch_earnings_dates("TEST_SYM_SAN", "America/New_York")
        assert res.status == "ok"
        assert len(res.rows) == 2
        assert res.rows[0]["eps_actual"] is None
        assert res.rows[0]["eps_estimate"] == 1.25
        assert res.rows[1]["eps_actual"] == 1.50
        assert res.rows[1]["eps_estimate"] is None


def test_get_equity_analyst_context_endpoint(client):
    # Mock auth session
    app.dependency_overrides[require_session] = lambda: {"user_id": "test_user"}
    conn = get_connection()
    conn.execute("DELETE FROM analyst_context_cache WHERE ticker = 'TESTMOCK'")
    conn.commit()

    try:
        with patch("api.routes_equity.yf.Ticker") as mock_ticker, \
             patch("api.routes_equity.get_asset_calendar") as mock_cal, \
             patch("api.routes_equity.fetch_earnings_dates") as mock_earnings:

            # Mock yfinance price targets
            instance = MagicMock()
            instance.get_analyst_price_targets.return_value = {
                "mean": 180.5,
                "high": 210.0,
                "low": 150.0,
            }
            instance.info = {"numberOfAnalystOpinions": 42}
            mock_ticker.return_value = instance

            # Mock calendar with future date
            from datetime import date, timedelta
            future_day = date.today() + timedelta(days=45)
            mock_cal.return_value = {
                "Earnings Date": [future_day]
            }

            # Mock earnings
            mock_earnings.return_value = EarningsFetchResult(
                rows=[{"date_str": "2026-08-10", "timestamp_ms": 1786320000000, "eps_actual": 2.10, "eps_estimate": 2.05}],
                status="ok",
                source_as_of="2026-08-22T00:00:00",
            )

            res = client.get("/api/equity/TESTMOCK/analyst-context")
            assert res.status_code == 200
            data = res.json()
            assert data["ticker"] == "TESTMOCK"
            assert data["market"] == "US"
            assert data["currency"] == "USD"
            assert data["target_mean"] == 180.5
            assert data["num_analysts"] == 42
            assert data["next_earnings_date"] == future_day.strftime("%Y-%m-%d")
            assert data["days_to_earnings"] is not None
            assert data["data_status"] == "ok"
            assert len(data["earnings_history"]) == 1
    finally:
        app.dependency_overrides.pop(require_session, None)


from api.auth import require_session
