"""Tests for Actual Portfolio Hub Read Endpoints (/api/portfolio/actual/*)."""
import importlib
import sys
from unittest.mock import patch
import pytest
from filelock import Timeout


@pytest.fixture
def isolated_api_portfolio(tmp_path, monkeypatch):
    """Set up isolated vault path and reattach modules for API tests."""
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(tmp_path))

    for mod_name in list(sys.modules):
        if mod_name.startswith("tools.portfolio.") or mod_name.startswith("tools.portfolio_tools"):
            del sys.modules[mod_name]

    import tools.portfolio.core as core
    import tools.portfolio.trading as trading
    import tools.portfolio.watchlist as watchlist
    import tools.portfolio.goals as goals
    import tools.portfolio.performance as perf
    import tools.portfolio.journal as journal

    # Reattach fresh modules to api.routes_portfolio
    if "api.routes_portfolio" in sys.modules:
        import api.routes_portfolio as rp
        rp.portfolio_core = core
        rp.portfolio_trading = trading
        rp.portfolio_watchlist = watchlist
        rp.portfolio_goals = goals
        rp.portfolio_perf = perf
        rp.portfolio_journal = journal

    # Initialize basic files
    post, state = core._load_or_init()
    h = core.Holding(
        symbol="CASH_THB",
        asset_type="Cash",
        units=100_000.0,
        current_price_thb=1.0,
        avg_cost_thb=1.0,
    )
    state.holdings = [h]
    core._save(post, state)

    watchlist.add_to_watchlist.invoke({"symbol": "AAPL", "asset_type": "Stock", "target_price": 180.0})
    goals.set_goal.invoke({"name": "Emergency Fund", "target_amount_thb": 500_000.0, "goal_type": "cash_target"})
    perf.record_performance_snapshot.invoke({"refresh_prices": False})
    journal.append_trading_journal.invoke({"entry": "Test API journal entry."})

    return tmp_path


def test_actual_portfolio_state_route_no_network_calls(authed_client, isolated_api_portfolio):
    with patch("yfinance.Ticker") as mock_ticker:
        r = authed_client.get("/api/portfolio/actual/state?refresh_prices=false&fetch_fundamentals=false")
        assert r.status_code == 200
        data = r.json()
        assert data["summary"]["total_value_thb"] == 100_000.0
        assert len(data["holdings"]) == 1
        assert data["holdings"][0]["symbol"] == "CASH_THB"
        # Zero network calls when fetch_fundamentals=false and refresh_prices=false
        assert mock_ticker.call_count == 0


def test_actual_portfolio_state_route_lock_timeout(authed_client, isolated_api_portfolio):
    with patch("api.routes_portfolio.portfolio_core.get_structured_portfolio_state", side_effect=Timeout("test lock")):
        r = authed_client.get("/api/portfolio/actual/state")
        assert r.status_code == 503
        assert "timeout" in r.json()["detail"].lower()


def test_actual_allocations_route(authed_client, isolated_api_portfolio):
    r = authed_client.get("/api/portfolio/actual/allocations")
    assert r.status_code == 200
    data = r.json()
    assert "summaries" in data
    cash_bucket = next((s for s in data["summaries"] if s["bucket_id"] == "cash"), None)
    assert cash_bucket is not None


def test_actual_watchlist_route(authed_client, isolated_api_portfolio):
    r = authed_client.get("/api/portfolio/actual/watchlist")
    assert r.status_code == 200
    data = r.json()
    assert len(data["items"]) == 1
    assert data["items"][0]["symbol"] == "AAPL"


def test_actual_goals_route(authed_client, isolated_api_portfolio):
    r = authed_client.get("/api/portfolio/actual/goals")
    assert r.status_code == 200
    data = r.json()
    assert data["n_goals"] == 1
    assert data["goals"][0]["name"] == "Emergency Fund"


def test_actual_performance_route(authed_client, isolated_api_portfolio):
    r = authed_client.get("/api/portfolio/actual/performance")
    assert r.status_code == 200
    data = r.json()
    assert len(data) >= 1
    assert data[0]["Total_NAV"] == 100_000.0


def test_actual_journal_route(authed_client, isolated_api_portfolio):
    r = authed_client.get("/api/portfolio/actual/journal")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 1
    assert "Test API journal entry." in data[0]["content"]


def test_value_error_mapping(authed_client, isolated_api_portfolio):
    with patch("api.routes_portfolio.portfolio_core.get_structured_bucket_allocation", side_effect=ValueError("Invalid state")):
        r = authed_client.get("/api/portfolio/actual/allocations")
        assert r.status_code == 400
        assert "Invalid state" in r.json()["detail"]
