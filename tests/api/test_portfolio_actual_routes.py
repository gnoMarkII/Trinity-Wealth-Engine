"""Tests for Actual Portfolio Hub Read Endpoints (/api/portfolio/actual/*)."""
import importlib
import sys
from unittest.mock import MagicMock, patch
import pytest
from filelock import Timeout
from api.main import app
from api.dependencies import get_portfolio_service


@pytest.fixture
def isolated_api_portfolio(tmp_path, monkeypatch):
    """Set up isolated vault path and reattach modules for API tests."""
    from tests.conftest import _reset_portfolio_modules
    mods = _reset_portfolio_modules(tmp_path, monkeypatch)
    core = mods.core
    trading = mods.trading
    watchlist = mods.watchlist
    goals = mods.goals
    perf = mods.perf
    journal = mods.journal

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
    fake_service = MagicMock()
    fake_service.get_structured_portfolio_state.side_effect = Timeout("test lock")
    app.dependency_overrides[get_portfolio_service] = lambda: fake_service
    try:
        r = authed_client.get("/api/portfolio/actual/state")
        assert r.status_code == 503
        assert "timeout" in r.json()["detail"].lower()
    finally:
        app.dependency_overrides.pop(get_portfolio_service, None)


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
    fake_service = MagicMock()
    fake_service.get_structured_bucket_allocation.side_effect = ValueError("Invalid state")
    app.dependency_overrides[get_portfolio_service] = lambda: fake_service
    try:
        r = authed_client.get("/api/portfolio/actual/allocations")
        assert r.status_code == 400
        assert "Invalid state" in r.json()["detail"]
    finally:
        app.dependency_overrides.pop(get_portfolio_service, None)


def test_actual_transactions_route_and_note_update(authed_client, isolated_api_portfolio):
    import tools.portfolio.trading as trading
    trading.execute_trade.invoke({
        "symbol": "PTT",
        "asset_type": "Stock",
        "action": "buy",
        "units": 100.0,
        "price": 35.0,
        "currency": "THB",
        "notes": "Initial trade note",
    })

    # GET transactions
    r = authed_client.get("/api/portfolio/actual/transactions")
    assert r.status_code == 200
    data = r.json()
    assert data["portfolio_id"] == "default"
    assert len(data["transactions"]) == 1
    tx = data["transactions"][0]
    assert tx["symbol"] == "PTT"
    assert tx["action"] == "BUY"
    assert tx["units"] == 100.0
    assert tx["price"] == 35.0
    assert tx["notes"] == "Initial trade note"
    assert data["summary"]["total_buy_count"] == 1
    assert data["summary"]["total_buy_thb"] == 3500.0

    tx_id = tx["transaction_id"]

    # PATCH note
    patch_r = authed_client.patch(
        f"/api/portfolio/actual/transactions/{tx_id}/note",
        json={"notes": "Updated note via API"}
    )
    assert patch_r.status_code == 200
    updated_tx = patch_r.json()
    assert updated_tx["transaction_id"] == tx_id
    assert updated_tx["notes"] == "Updated note via API"

    # GET again to verify persistence
    r2 = authed_client.get("/api/portfolio/actual/transactions")
    assert r2.status_code == 200
    assert r2.json()["transactions"][0]["notes"] == "Updated note via API"


def test_actual_transactions_keeps_pvd_rows_without_changing_buy_sell_summary(authed_client, monkeypatch):
    """Existing transaction consumers still receive their ordinary trade totals."""
    import json

    base = {"Timestamp": "2026-09-30 00:00:00", "Currency": "THB", "Notes": ""}
    metadata = json.dumps({"as_of_date": "2026-09-30", "total_nav": "1250.00"})
    rows = [
        dict(base, Transaction_ID="stock-buy", Symbol="PTT", Action="BUY", Units="100", Price="35", Cost_THB="3500.00", Realized_PnL_THB=""),
        dict(base, Transaction_ID="stock-sell", Symbol="PTT", Action="SELL", Units="20", Price="40", Cost_THB="700.00", Realized_PnL_THB="100.00"),
        dict(base, Transaction_ID="pvd-snapshot", Symbol="PVD_SINSA_FI", Action="PVD_SNAPSHOT", Units="123.4567", Price="10.1234", Cost_THB="900.00", Realized_PnL_THB="", Gross_Amount="1250.00", Reconciliation_Adjustment="0.01", Source="BBLAM_PVD", Asset_Type="PVD", Cash_Adjusted="NO", PVD_Metadata_JSON=metadata),
    ]
    service = MagicMock()
    service.get_structured_trades_log.return_value = rows
    monkeypatch.setitem(app.dependency_overrides, get_portfolio_service, lambda: service)

    response = authed_client.get("/api/portfolio/actual/transactions?portfolio_id=mixed")
    assert response.status_code == 200
    data = response.json()
    service.get_structured_trades_log.assert_called_once_with(portfolio_id="mixed", symbol=None)
    assert data["portfolio_id"] == "mixed"
    assert data["summary"] == {
        "total_buy_count": 1,
        "total_sell_count": 1,
        "total_buy_thb": 3500.0,
        "total_sell_thb": 800.0,
        "total_realized_pnl_thb": 100.0,
    }
    assert len(data["transactions"]) == 3
    pvd = next(row for row in data["transactions"] if row["action"] == "PVD_SNAPSHOT")
    assert pvd["gross_amount"] == "1250.00"
    assert pvd["pvd_metadata_json"] == metadata
    assert pvd["asset_type"] == "PVD"
    assert pvd["cash_adjusted"] == "NO"
    assert pvd["reconciliation_adjustment"] == "0.01"
    assert next(row for row in data["transactions"] if row["transaction_id"] == "stock-buy")["reconciliation_adjustment"] is None
