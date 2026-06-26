"""Test portfolio_tools: recalc, execute_trade, record_income — ใช้ Vault แยกผ่าน tmp_path"""
import json

import pytest


class TestRecalcHolding:
    def test_cash_holding(self, isolated_portfolio):
        pt = isolated_portfolio
        h = pt.Holding(symbol="CASH_THB", asset_type="Cash", units=100_000.0)
        pt._recalc_holding(h, current_fx=36.5)
        assert h.market_value_thb == 100_000.0
        assert h.unrealized_pnl_percent is None

    def test_usd_holding_uses_current_fx(self, isolated_portfolio):
        pt = isolated_portfolio
        h = pt.Holding(
            symbol="AAPL",
            asset_type="Stock",
            units=10,
            avg_cost_usd=150.0,
            current_price_usd=200.0,
            fx_rate=35.0,  # historical fx
        )
        pt._recalc_holding(h, current_fx=36.5)  # current fx
        # ต้องใช้ current_fx=36.5 ไม่ใช่ h.fx_rate=35.0
        assert h.market_value_thb == 10 * 200.0 * 36.5
        assert h.unrealized_pnl_percent == pytest.approx(33.33, rel=1e-2)

    def test_thb_holding(self, isolated_portfolio):
        pt = isolated_portfolio
        h = pt.Holding(
            symbol="PTT",
            asset_type="Stock",
            units=1000,
            avg_cost_thb=30.0,
            current_price_thb=35.0,
        )
        pt._recalc_holding(h, current_fx=36.5)
        assert h.market_value_thb == 35_000.0
        assert h.unrealized_pnl_percent == pytest.approx(16.67, rel=1e-2)

    def test_incomplete_holding_resets_to_zero(self, isolated_portfolio):
        pt = isolated_portfolio
        h = pt.Holding(symbol="X", asset_type="Stock", units=10, market_value_thb=999.0)
        pt._recalc_holding(h, current_fx=36.5)
        # ไม่มี cost/price ครบ → reset เป็น 0 (Anti-Drift: กัน stale value)
        assert h.market_value_thb == 0.0
        assert h.unrealized_pnl_percent is None


class TestRecalcSummary:
    def test_summary_aggregates(self, isolated_portfolio):
        pt = isolated_portfolio
        state = pt.PortfolioState(
            last_updated="2026-05-20T00:00:00",
            fx_rates={"USDTHB": 36.5},
            holdings=[
                pt.Holding(symbol="CASH_THB", asset_type="Cash", units=100_000.0),
                pt.Holding(
                    symbol="AAPL", asset_type="Stock", units=10,
                    avg_cost_usd=150.0, current_price_usd=200.0,
                ),
                pt.Holding(
                    symbol="PTT", asset_type="Stock", units=1000,
                    avg_cost_thb=30.0, current_price_thb=35.0,
                ),
            ],
        )
        pt._recalc_all(state)
        # market values: 100k cash + 10*200*36.5 = 73,000 + 35,000 = 208,000
        assert state.summary.total_value_thb == pytest.approx(208_000.0)
        # unrealized: (200-150)*10*36.5 + (35-30)*1000 = 18,250 + 5,000 = 23,250
        assert state.summary.total_unrealized_profit == pytest.approx(23_250.0)


class TestGetPortfolioState:
    def test_returns_json(self, isolated_portfolio):
        pt = isolated_portfolio
        pt._record_income_locked("Other", 100_000.0, None)
        result = pt.get_portfolio_state.invoke({"refresh_prices": False})
        data = json.loads(result)
        assert data["doc_type"] == "portfolio_master"
        assert data["fx_rates"]["USDTHB"] == 36.5
        cash = next(h for h in data["holdings"] if h["symbol"] == "CASH_THB")
        assert cash["units"] == 100_000.0


class TestCashUsdRecalc:
    def test_cash_usd_market_value_uses_fx(self, isolated_portfolio):
        pt = isolated_portfolio
        h = pt.Holding(symbol="CASH_USD", asset_type="Cash", units=1_000.0)
        pt._recalc_holding(h, current_fx=36.5)
        # USD cash → market_value_thb = units × current_fx
        assert h.market_value_thb == 36_500.0
        assert h.unrealized_pnl_percent is None

    def test_summary_aggregates_both_cash_pots(self, isolated_portfolio):
        pt = isolated_portfolio
        state = pt.PortfolioState(
            last_updated="2026-05-22T00:00:00",
            fx_rates={"USDTHB": 36.5},
            holdings=[
                pt.Holding(symbol="CASH_THB", asset_type="Cash", units=100_000.0),
                pt.Holding(symbol="CASH_USD", asset_type="Cash", units=1_000.0),
            ],
        )
        pt._recalc_all(state)
        # 100k THB + 1000 × 36.5 = 136,500
        assert state.summary.total_value_thb == pytest.approx(136_500.0)
        # ไม่มี non-cash holding → unrealized = 0
        assert state.summary.total_unrealized_profit == pytest.approx(0.0)

    def test_compute_total_cost_includes_cash_usd_at_fx(self, isolated_portfolio):
        pt = isolated_portfolio
        state = pt.PortfolioState(
            last_updated="2026-05-22T00:00:00",
            fx_rates={"USDTHB": 36.5},
            holdings=[
                pt.Holding(symbol="CASH_THB", asset_type="Cash", units=50_000.0),
                pt.Holding(symbol="CASH_USD", asset_type="Cash", units=2_000.0),
            ],
        )
        # cost = 50,000 + 2,000 × 36.5 = 123,000
        cost = pt._compute_total_cost(state, current_fx=36.5)
        assert cost == pytest.approx(123_000.0)


class TestComputeAllocationBreakdown:
    def test_group_by_asset_type(self, isolated_portfolio):
        pt = isolated_portfolio
        pt._manage_cash_flow_locked(100_000.0, "deposit", "THB")
        pt._execute_trade_locked(
            symbol="PTT", asset_type="Stock", action="buy",
            units=1000, price=30.0, currency="THB",
        )
        # หลัง buy: cash 70k + PTT (mv 30k) = NAV 100k → Stock 30%, Cash 70%
        result = pt.compute_allocation_breakdown.invoke({"group_by": "asset_type"})
        data = json.loads(result)
        assert data["group_by"] == "asset_type"
        assert data["total_nav_thb"] == pytest.approx(100_000.0)
        groups = {b["group"]: b for b in data["breakdown"]}
        assert "Stock" in groups and "Cash" in groups
        assert groups["Stock"]["pct"] == pytest.approx(30.0)
        assert groups["Cash"]["pct"] == pytest.approx(70.0)
        # sum ของ pct ทั้งหมดต้อง ≈ 100
        total_pct = sum(b["pct"] for b in data["breakdown"])
        assert total_pct == pytest.approx(100.0, abs=0.1)

    def test_group_by_currency(self, isolated_portfolio):
        pt = isolated_portfolio
        pt._manage_cash_flow_locked(73_000.0, "deposit", "THB")
        pt._manage_cash_flow_locked(2_000.0, "deposit", "USD")
        # NAV = 73,000 + 2000*36.5 = 73,000 + 73,000 = 146,000 → THB 50%, USD 50%
        result = pt.compute_allocation_breakdown.invoke({"group_by": "currency"})
        data = json.loads(result)
        groups = {b["group"]: b for b in data["breakdown"]}
        assert "THB" in groups and "USD" in groups
        assert groups["THB"]["pct"] == pytest.approx(50.0)
        assert groups["USD"]["pct"] == pytest.approx(50.0)

    def test_invalid_group_by_raises(self, isolated_portfolio):
        pt = isolated_portfolio
        with pytest.raises(ValueError, match="group_by"):
            pt.compute_allocation_breakdown.invoke({"group_by": "sector"})



import pytest
import tools.portfolio.core as core

class TestLoadOrInitCore:
    def test_load_no_metadata(self, isolated_portfolio, tmp_vault, monkeypatch):
        pt = isolated_portfolio
        test_path = tmp_vault / "20_Portfolio_Management/Current_Holdings/Portfolio_Holdings.md"
        monkeypatch.setattr(core, "PORTFOLIO_PATH", test_path)
        
        # Create a blank markdown file
        test_path.parent.mkdir(parents=True, exist_ok=True)
        with open(test_path, "w", encoding="utf-8") as f:
            f.write("Hello World")
        
        # It should log warning and init new state
        post, state = core._load_or_init()
        assert len(state.holdings) == 2  # Default cash symbols

class TestHoldingSidecars:
    def test_holding_to_md_usd(self, isolated_portfolio):
        pt = isolated_portfolio
        h = pt.Holding(
            symbol="AAPL",
            asset_type="Stock",
            units=10,
            avg_cost_usd=150.0,
            current_price_usd=200.0,
            market_value_thb=70000.0
        )
        md = core._holding_to_md(h)
        assert "currency: USD" in md
        assert "avg_cost: 150.0" in md
        assert "current_price: 200.0" in md

    def test_sidecar_deletion(self, isolated_portfolio):
        pt = isolated_portfolio
        pt._manage_cash_flow_locked(500_000, "deposit", "THB")
        pt._manage_cash_flow_locked(500_000, "deposit", "USD")
        pt._execute_trade_locked("PTT", "Stock", "buy", 100, 30.0, "THB")
        pt._execute_trade_locked("AAPL", "Stock", "buy", 10, 150.0, "USD")
        
        import tools.portfolio.core as cl
        sidecars = list(cl.HOLDINGS_DIR.glob("*.md"))
        assert len(sidecars) == 2
        
        pt._execute_trade_locked("PTT", "Stock", "sell", 100, 35.0, "THB")
        sidecars_after = list(cl.HOLDINGS_DIR.glob("*.md"))
        assert len(sidecars_after) == 1
        assert sidecars_after[0].stem == "AAPL"

class TestRecalcEdgeCases:
    def test_recalc_fx_missing(self, isolated_portfolio):
        pt = isolated_portfolio
        post, state = pt._load_or_init()
        state.fx_rates = {} # Empty
        pt._recalc_all(state)
        # Should not crash but log warning
        
    def test_require_cash_lazy_creation(self, isolated_portfolio):
        pt = isolated_portfolio
        post, state = pt._load_or_init()
        # Remove CASH_USD to test lazy creation
        state.holdings = [h for h in state.holdings if h.symbol != core.CASH_USD_SYMBOL]
        cash = core._require_cash(state, "USD")
        assert cash.symbol == core.CASH_USD_SYMBOL
        assert len(state.holdings) == 2

    def test_require_fx_missing_raises(self, isolated_portfolio):
        pt = isolated_portfolio
        post, state = pt._load_or_init()
        state.fx_rates = {}
        with pytest.raises(ValueError, match="ไม่พบ fx_rates.USDTHB"):
            core._require_fx(state)

class TestGetPortfolioStateRefresh:
    def test_get_portfolio_state_refresh_prices(self, isolated_portfolio, monkeypatch):
        pt = isolated_portfolio
        import tools.portfolio.prices as prices
        monkeypatch.setattr(prices, "_refresh_prices", lambda state: {"PTT": "ok"})
        
        result = pt.get_portfolio_state.func(refresh_prices=True)
        import json
        data = json.loads(result)
        assert "_price_refresh" in data
        assert data["_price_refresh"]["PTT"] == "ok"

    def test_get_portfolio_state_lock_timeout(self, isolated_portfolio, monkeypatch):
        pt = isolated_portfolio
        def mock_lock(*args, **kwargs):
            from filelock import Timeout
            raise Timeout("mock")
        monkeypatch.setattr("tools.portfolio.core._portfolio_lock.acquire", mock_lock)
        
        result = pt.get_portfolio_state.func()
        import json
        data = json.loads(result)
        assert "error" in data
        assert "portfolio lock timeout" in data["error"]

class TestHoldingCurrency:
    def test_holding_currency_unknown(self):
        h = core.Holding(symbol="WEIRD", asset_type="Asset", units=10, market_value_thb=100.0)
        assert core._holding_currency(h) == "UNKNOWN"
        
    def test_holding_currency_usd(self):
        h = core.Holding(symbol="AAPL", asset_type="Stock", units=10, market_value_thb=100.0, avg_cost_usd=10.0)
        assert core._holding_currency(h) == "USD"
        
    def test_holding_currency_thb(self):
        h = core.Holding(symbol="PTT", asset_type="Stock", units=10, market_value_thb=100.0, avg_cost_thb=10.0)
        assert core._holding_currency(h) == "THB"

class TestComputeAllocationExceptions:
    def test_invalid_group_by_raises(self, isolated_portfolio):
        pt = isolated_portfolio
        with pytest.raises(ValueError, match="group_by ต้องเป็น"):
            pt.compute_allocation_breakdown.func(group_by="invalid")

    def test_compute_allocation_lock_timeout(self, isolated_portfolio, monkeypatch):
        pt = isolated_portfolio
        def mock_lock(*args, **kwargs):
            from filelock import Timeout
            raise Timeout("mock")
        monkeypatch.setattr("tools.portfolio.core._portfolio_lock.acquire", mock_lock)
        
        with pytest.raises(ValueError, match="portfolio lock timeout"):
            pt.compute_allocation_breakdown.func()
