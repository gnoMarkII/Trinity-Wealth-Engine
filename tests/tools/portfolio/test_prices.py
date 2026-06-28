"""Test portfolio_tools: recalc, execute_trade, record_income — ใช้ Vault แยกผ่าน tmp_path"""
import json

import pytest


class TestSyncMarketPrices:
    def test_sync_updates_prices_and_recalcs(self, isolated_portfolio, monkeypatch):
        pt = isolated_portfolio
        pt._manage_cash_flow_locked(500_000.0, "deposit", "THB")
        pt._execute_trade_locked(
            symbol="PTT", asset_type="Stock", action="buy",
            units=1000, price=30.0, currency="THB",
        )

        # Mock _refresh_prices ให้บังคับราคาเป็น 40 และ return status ok
        def fake_refresh(state):
            for h in state.holdings:
                if h.symbol == "PTT":
                    h.current_price_thb = 40.0
            return {"PTT": "ok"}

        import tools.portfolio.prices as prices
        monkeypatch.setattr(prices, "_refresh_prices", fake_refresh)
        result = pt.sync_market_prices.invoke({})
        assert "[SYNC]" in result
        assert "refreshed 1/1" in result

        _, state = pt._load_or_init()
        ptt = pt._find_holding(state, "PTT")
        assert ptt.current_price_thb == 40.0
        # Anti-Drift: market_value + unrealized ต้อง recalc ตามอัตโนมัติ
        assert ptt.market_value_thb == 40_000.0
        assert state.summary.total_unrealized_profit == pytest.approx(10_000.0)

    def test_sync_reports_failed_symbols(self, isolated_portfolio, monkeypatch):
        pt = isolated_portfolio
        pt._manage_cash_flow_locked(500_000.0, "deposit", "THB")
        pt._execute_trade_locked(
            symbol="PTT", asset_type="Stock", action="buy",
            units=1000, price=30.0, currency="THB",
        )

        def fake_refresh(state):
            return {"PTT": "timeout"}

        import tools.portfolio.prices as prices
        monkeypatch.setattr(prices, "_refresh_prices", fake_refresh)
        result = pt.sync_market_prices.invoke({})
        assert "refreshed 0/1" in result
        assert "PTT=timeout" in result

    def test_sync_empty_portfolio(self, isolated_portfolio, monkeypatch):
        pt = isolated_portfolio
        # ไม่มี non-cash holding → refresh_info ว่าง
        monkeypatch.setattr(pt, "_refresh_prices", lambda state: {})
        result = pt.sync_market_prices.invoke({})
        assert "no non-cash holdings" in result


class TestUpdateFxRate:
    def test_manual_rate_updates_fx_and_recalcs(self, isolated_portfolio):
        pt = isolated_portfolio
        pt._manage_cash_flow_locked(1_000.0, "deposit", "USD")
        # default fx = 36.5 → CASH_USD market_value = 36,500

        result = pt._update_fx_rate_locked(35.0)
        assert "[FX manual]" in result
        assert "36.5000" in result and "35.0000" in result

        _, state = pt._load_or_init()
        assert state.fx_rates["USDTHB"] == 35.0
        # CASH_USD market_value = 1000 × 35 = 35,000
        cash_usd = pt._find_holding(state, "CASH_USD")
        assert cash_usd.market_value_thb == pytest.approx(35_000.0)
        assert state.summary.total_value_thb == pytest.approx(35_000.0)

    def test_auto_fetch_when_rate_none(self, isolated_portfolio, monkeypatch):
        pt = isolated_portfolio
        import tools.portfolio.trading as trading
        monkeypatch.setattr(trading, "_fetch_fx_rate", lambda: 34.80)
        result = pt._update_fx_rate_locked(None)
        assert "[FX yfinance]" in result
        assert "34.8000" in result

        _, state = pt._load_or_init()
        assert state.fx_rates["USDTHB"] == 34.80

    def test_auto_fetch_failure_raises(self, isolated_portfolio, monkeypatch):
        pt = isolated_portfolio
        import tools.portfolio.trading as trading
        monkeypatch.setattr(trading, "_fetch_fx_rate", lambda: None)
        with pytest.raises(ValueError, match="auto-fetch FX ล้มเหลว"):
            pt._update_fx_rate_locked(None)

    def test_negative_rate_raises(self, isolated_portfolio):
        pt = isolated_portfolio
        result = pt.update_fx_rate.invoke({"rate": -1.0})

        assert isinstance(result, str) and result.startswith("Error:")
        assert "rate ต้องมากกว่า 0" in result
    def test_zero_rate_raises(self, isolated_portfolio):
        pt = isolated_portfolio
        result = pt.update_fx_rate.invoke({"rate": 0})

        assert isinstance(result, str) and result.startswith("Error:")
        assert "rate ต้องมากกว่า 0" in result
    def test_fx_change_affects_usd_holding_market_value(self, isolated_portfolio):
        """FX ขึ้น → market_value USD holding ใน THB เพิ่ม + unrealized P/L (THB) เพิ่ม"""
        pt = isolated_portfolio
        pt._manage_cash_flow_locked(50_000.0, "deposit", "USD")
        pt._execute_trade_locked(
            symbol="AAPL", asset_type="Stock", action="buy",
            units=10, price=200.0, currency="USD",
        )
        # simulate ราคาขึ้น 200 → 250
        post, state = pt._load_or_init()
        aapl = pt._find_holding(state, "AAPL")
        aapl.current_price_usd = 250.0
        pt._save(post, state)

        _, state = pt._load_or_init()
        unrealized_at_365 = state.summary.total_unrealized_profit
        # unrealized = (250 - 200) × 10 × 36.5 = 18,250

        # FX ขึ้นเป็น 37.0 → unrealized = (250-200) × 10 × 37 = 18,500
        pt._update_fx_rate_locked(37.0)
        _, state2 = pt._load_or_init()
        assert state2.summary.total_unrealized_profit == pytest.approx(18_500.0)
        assert state2.summary.total_unrealized_profit > unrealized_at_365



import pytest
import tools.portfolio.prices as prices
from tools.portfolio.prices import _yf_symbol, _fetch_last_price, _fetch_fx_rate, fetch_latest_price, sync_market_prices, _refresh_prices

class TestYfSymbol:
    def test_yf_symbol_thb(self):
        assert _yf_symbol("PTT", "THB") == "PTT.BK"
        assert _yf_symbol("PTT.BK", "THB") == "PTT.BK"

    def test_yf_symbol_usd(self):
        assert _yf_symbol("AAPL", "USD") == "AAPL"

class TestFetchLastPrice:
    def test_fetch_last_price_fast_info_success(self, monkeypatch):
        class MockFastInfo:
            last_price = 150.0
        class MockTicker:
            fast_info = MockFastInfo()
        monkeypatch.setattr(prices.yf, "Ticker", lambda sym: MockTicker())
        assert _fetch_last_price("AAPL") == 150.0

    def test_fetch_last_price_history_fallback_success(self, monkeypatch):
        import pandas as pd
        class MockFastInfo:
            last_price = None
        class MockTicker:
            fast_info = MockFastInfo()
            def history(self, period):
                return pd.DataFrame({"Close": [140.0, 145.0]})
        monkeypatch.setattr(prices.yf, "Ticker", lambda sym: MockTicker())
        assert _fetch_last_price("AAPL") == 145.0

    def test_fetch_last_price_history_empty(self, monkeypatch):
        import pandas as pd
        class MockFastInfo:
            last_price = None
        class MockTicker:
            fast_info = MockFastInfo()
            def history(self, period):
                return pd.DataFrame({"Close": []})
        monkeypatch.setattr(prices.yf, "Ticker", lambda sym: MockTicker())
        assert _fetch_last_price("AAPL") is None

    def test_fetch_last_price_exception(self, monkeypatch):
        def mock_raise(sym):
            raise ValueError("API Down")
        monkeypatch.setattr(prices.yf, "Ticker", mock_raise)
        assert _fetch_last_price("AAPL") is None

class TestFetchFxRate:
    def test_fetch_fx_rate_success(self, monkeypatch):
        monkeypatch.setattr(prices, "_fetch_last_price", lambda sym: 35.5)
        assert _fetch_fx_rate() == 35.5

class TestFetchLatestPrice:
    def test_fetch_latest_price_success(self, monkeypatch):
        monkeypatch.setattr(prices, "_fetch_last_price", lambda sym: 30.0)
        assert fetch_latest_price("PTT", "THB") == 30.0

    def test_fetch_latest_price_invalid_currency(self):
        with pytest.raises(ValueError, match="currency ต้องเป็น"):
            fetch_latest_price("PTT", "EUR")

class TestRefreshPrices:
    def test_refresh_prices_success(self, isolated_portfolio, monkeypatch):
        pt = isolated_portfolio
        pt._manage_cash_flow_locked(500_000, "deposit", "THB")
        pt._manage_cash_flow_locked(500_000, "deposit", "USD")
        pt._execute_trade_locked("PTT", "Stock", "buy", 100, 30.0, "THB")
        pt._execute_trade_locked("AAPL", "Stock", "buy", 10, 150.0, "USD")
        _, state = pt._load_or_init()
        
        monkeypatch.setattr(prices, "_fetch_last_price", lambda sym: 40.0 if sym == "PTT.BK" else 160.0)
        results = _refresh_prices(state)
        
        assert results["PTT"] == "ok"
        assert results["AAPL"] == "ok"
        
        ptt = pt._find_holding(state, "PTT")
        assert ptt.current_price_thb == 40.0
        aapl = pt._find_holding(state, "AAPL")
        assert aapl.current_price_usd == 160.0

    def test_refresh_prices_exception(self, isolated_portfolio, monkeypatch):
        pt = isolated_portfolio
        pt._manage_cash_flow_locked(500_000, "deposit", "THB")
        pt._execute_trade_locked("PTT", "Stock", "buy", 100, 30.0, "THB")
        _, state = pt._load_or_init()
        
        def mock_raise(sym):
            raise ValueError("fetch failed")
        monkeypatch.setattr(prices, "_fetch_last_price", mock_raise)
        results = _refresh_prices(state)
        
        assert "error: fetch failed" in results["PTT"]

    def test_refresh_prices_none(self, isolated_portfolio, monkeypatch):
        pt = isolated_portfolio
        pt._manage_cash_flow_locked(500_000, "deposit", "THB")
        pt._execute_trade_locked("PTT", "Stock", "buy", 100, 30.0, "THB")
        _, state = pt._load_or_init()
        
        monkeypatch.setattr(prices, "_fetch_last_price", lambda sym: None)
        results = _refresh_prices(state)
        
        assert results["PTT"] == "no_data"

    def test_refresh_prices_timeout(self, isolated_portfolio, monkeypatch):
        pt = isolated_portfolio
        pt._manage_cash_flow_locked(500_000, "deposit", "THB")
        pt._execute_trade_locked("PTT", "Stock", "buy", 100, 30.0, "THB")
        _, state = pt._load_or_init()
        
        import time
        def mock_timeout(sym):
            time.sleep(0.5)
            return 40.0
        monkeypatch.setattr(prices, "_fetch_last_price", mock_timeout)
        monkeypatch.setattr(prices, "_PRICE_FETCH_TIMEOUT", 0.1)
        
        results = _refresh_prices(state)
        assert results["PTT"] == "timeout"

class TestSyncMarketPricesExceptions:
    def test_sync_market_prices_lock_timeout(self, monkeypatch):
        def mock_lock(*args, **kwargs):
            from filelock import Timeout
            raise Timeout("mock")
        monkeypatch.setattr(prices._portfolio_lock, "acquire", mock_lock)
        
        result = sync_market_prices.func()

        assert isinstance(result, str) and result.startswith("Error:")
        assert "portfolio lock" in result
    def test_sync_market_prices_empty_portfolio(self, monkeypatch, isolated_portfolio):
        pt = isolated_portfolio
        monkeypatch.setattr(prices, "_refresh_prices", lambda state: {})
        result = pt.sync_market_prices.func()
        assert "no non-cash holdings" in result
