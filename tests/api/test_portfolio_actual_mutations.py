"""Tests for Actual Portfolio Hub Mutation Endpoints (/api/portfolio/actual/*)."""
import sys
from unittest.mock import patch
import pytest
from filelock import Timeout


@pytest.fixture
def isolated_mutation_portfolio(tmp_path, monkeypatch):
    """Set up isolated vault path and reattach modules for mutation API tests."""
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

    if "api.routes_portfolio" in sys.modules:
        import api.routes_portfolio as rp
        rp.portfolio_core = core
        rp.portfolio_trading = trading
        rp.portfolio_watchlist = watchlist
        rp.portfolio_goals = goals
        rp.portfolio_perf = perf
        rp.portfolio_journal = journal

    # Initialize basic holdings
    post, state = core._load_or_init()
    h_cash = core.Holding(
        symbol="CASH_THB",
        asset_type="Cash",
        units=100_000.0,
        current_price_thb=1.0,
        avg_cost_thb=1.0,
    )
    h_stock = core.Holding(
        symbol="PTT",
        asset_type="Stock",
        units=100.0,
        current_price_thb=35.0,
        avg_cost_thb=30.0,
        bucket_id="thai_equities",
    )
    state.holdings = [h_cash, h_stock]
    core._save(post, state)

    watchlist.add_to_watchlist.invoke({"symbol": "TSLA", "asset_type": "Stock", "target_price": 250.0})
    goals.set_goal.invoke({"name": "Retirement Fund", "target_amount_thb": 10_000_000.0, "goal_type": "nav_target"})

    return tmp_path


def test_upsert_allocation_targets(authed_client, isolated_mutation_portfolio):
    payload = {
        "targets": [
            {"bucket_id": "us_growth", "name": "US Growth", "target_percent": 60.0, "color": "#10B981"},
            {"bucket_id": "cash", "name": "Cash Reserve", "target_percent": 40.0, "color": "#6B7280"}
        ]
    }
    r = authed_client.put("/api/portfolio/actual/allocations/targets", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert len(data["allocation_targets"]) == 2
    assert data["allocation_targets"][0]["bucket_id"] == "us_growth"


def test_assign_holding_bucket(authed_client, isolated_mutation_portfolio):
    authed_client.put("/api/portfolio/actual/allocations/targets", json={"targets": [{"bucket_id": "energy_sector", "name": "Energy", "target_percent": 100.0, "color": None}]})
    r = authed_client.put("/api/portfolio/actual/holdings/PTT/bucket", json={"bucket_id": "energy_sector"})
    assert r.status_code == 200
    data = r.json()
    ptt = next((h for h in data["holdings"] if h["symbol"] == "PTT"), None)
    assert ptt is not None
    assert ptt["bucket_id"] == "energy_sector"


def test_assign_holding_bucket_invalid_raises_400(authed_client, isolated_mutation_portfolio):
    r = authed_client.put("/api/portfolio/actual/holdings/PTT/bucket", json={"bucket_id": "non_existent_bucket"})
    assert r.status_code == 400


def test_batch_assign_holding_buckets(authed_client, isolated_mutation_portfolio):
    authed_client.put("/api/portfolio/actual/allocations/targets", json={"targets": [{"bucket_id": "blue_chips", "name": "Blue Chips", "target_percent": 100.0, "color": None}]})
    r = authed_client.put("/api/portfolio/actual/holdings/batch-bucket", json={"symbols": ["PTT"], "bucket_id": "blue_chips"})
    assert r.status_code == 200
    data = r.json()
    ptt = next((h for h in data["holdings"] if h["symbol"] == "PTT"), None)
    assert ptt["bucket_id"] == "blue_chips"


def test_batch_remove_holdings(authed_client, isolated_mutation_portfolio):
    r = authed_client.post("/api/portfolio/actual/holdings/batch-delete", json={"symbols": ["PTT"]})
    assert r.status_code == 200
    data = r.json()
    assert not any(h["symbol"] == "PTT" for h in data["holdings"])


def test_reset_portfolio_clean_slate(authed_client, isolated_mutation_portfolio):
    r = authed_client.post("/api/portfolio/actual/reset")
    assert r.status_code == 200
    data = r.json()
    assert len(data["holdings"]) == 0
    assert data["summary"]["total_value_thb"] == 0.0


def test_execute_trade_endpoint(authed_client, isolated_mutation_portfolio):
    payload = {
        "symbol": "AAPL",
        "asset_type": "Stock",
        "action": "buy",
        "units": 10.0,
        "price": 180.0,
        "currency": "THB",
        "bucket_id": "tech_stocks"
    }
    r = authed_client.post("/api/portfolio/actual/trade", json=payload)
    assert r.status_code == 200
    data = r.json()
    aapl = next((h for h in data["holdings"] if h["symbol"] == "AAPL"), None)
    assert aapl is not None
    assert aapl["units"] == 10.0
    assert aapl["bucket_id"] == "tech_stocks"


def test_manage_cash_flow_endpoint(authed_client, isolated_mutation_portfolio):
    r = authed_client.post("/api/portfolio/actual/cashflow", json={"amount": 50_000.0, "action": "deposit", "currency": "THB"})
    assert r.status_code == 200
    data = r.json()
    cash = next((h for h in data["holdings"] if h["symbol"] == "CASH_THB"), None)
    assert cash["units"] == 150_000.0


def test_record_income_endpoint(authed_client, isolated_mutation_portfolio):
    r = authed_client.post("/api/portfolio/actual/income", json={"income_type": "Dividend", "amount_thb": 2_500.0, "source_symbol": "PTT"})
    assert r.status_code == 200
    data = r.json()
    assert data["summary"]["passive_income_ytd"] == 2_500.0


def test_edit_holding_endpoint(authed_client, isolated_mutation_portfolio):
    r = authed_client.put("/api/portfolio/actual/holdings/PTT/edit", json={"units": 200.0, "reason": "Stock split correction"})
    assert r.status_code == 200
    data = r.json()
    ptt = next((h for h in data["holdings"] if h["symbol"] == "PTT"), None)
    assert ptt["units"] == 200.0


def test_remove_holding_endpoint(authed_client, isolated_mutation_portfolio):
    r = authed_client.delete("/api/portfolio/actual/holdings/PTT")
    assert r.status_code == 200
    data = r.json()
    assert not any(h["symbol"] == "PTT" for h in data["holdings"])


def test_upsert_watchlist_item_endpoint(authed_client, isolated_mutation_portfolio):
    r = authed_client.put("/api/portfolio/actual/watchlist/MSFT", json={"asset_type": "Stock", "target_price": 420.0, "notes": "AI leader"})
    assert r.status_code == 200
    data = r.json()
    assert any(item["symbol"] == "MSFT" and item["target_price"] == 420.0 for item in data["items"])


def test_remove_watchlist_item_endpoint(authed_client, isolated_mutation_portfolio):
    r = authed_client.delete("/api/portfolio/actual/watchlist/TSLA")
    assert r.status_code == 200
    data = r.json()
    assert not any(item["symbol"] == "TSLA" for item in data["items"])


def test_upsert_goal_endpoint(authed_client, isolated_mutation_portfolio):
    r = authed_client.put("/api/portfolio/actual/goals/Dividend%20Target", json={"goal_type": "passive_income_ytd", "target_amount_thb": 100_000.0})
    assert r.status_code == 200
    data = r.json()
    assert any(g["name"] == "Dividend Target" and g["target_amount_thb"] == 100_000.0 for g in data["goals"])


def test_remove_goal_endpoint(authed_client, isolated_mutation_portfolio):
    r = authed_client.delete("/api/portfolio/actual/goals/Retirement%20Fund")
    assert r.status_code == 200
    data = r.json()
    assert not any(g["name"] == "Retirement Fund" for g in data["goals"])


def test_mutation_value_error_mapping(authed_client, isolated_mutation_portfolio):
    # Invalid trade units should raise ValueError and map to 400
    r = authed_client.post("/api/portfolio/actual/trade", json={
        "symbol": "AAPL", "asset_type": "Stock", "action": "buy", "units": -5.0, "price": 100.0
    })
    assert r.status_code == 400
    assert "units ต้องมากกว่า 0" in r.json()["detail"]


def test_mutation_timeout_error_mapping(authed_client, isolated_mutation_portfolio):
    with patch("api.routes_portfolio.portfolio_trading.structured_execute_trade", side_effect=Timeout("test lock")):
        r = authed_client.post("/api/portfolio/actual/trade", json={
            "symbol": "AAPL", "asset_type": "Stock", "action": "buy", "units": 5.0, "price": 100.0
        })
        assert r.status_code == 503
        assert "timeout" in r.json()["detail"].lower()


def test_append_journal_endpoint(authed_client, isolated_mutation_portfolio):
    r = authed_client.post("/api/portfolio/actual/journal", json={"entry": "Bought more PTT on dip."})
    assert r.status_code == 200
    data = r.json()
    assert any("Bought more PTT on dip." in e["content"] for e in data)
