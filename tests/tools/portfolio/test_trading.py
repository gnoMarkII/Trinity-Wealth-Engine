"""Test portfolio_tools: recalc, execute_trade, record_income — ใช้ Vault แยกผ่าน tmp_path"""
import json

import pytest


class TestExecuteTrade:
    def test_buy_new_thb_holding(self, isolated_portfolio):
        pt = isolated_portfolio
        # bootstrap with cash
        pt._record_income_locked("Other", 500_000.0, None)

        # Use the public wrapper to cover line 116
        result = pt.execute_trade.func(
            symbol="PTT", asset_type="Stock", action="buy",
            units=1000, price=30.0, currency="THB",
        )
        assert "[BUY] PTT" in result
        assert "CASH_THB" in result and "คงเหลือ" in result

        # Verify state
        _, state = pt._load_or_init()
        holding = pt._find_holding(state, "PTT")
        assert holding is not None
        assert holding.units == 1000
        assert holding.avg_cost_thb == 30.0
        cash = pt._find_holding(state, "CASH_THB")
        assert cash.units == 500_000.0 - 30_000.0

    def test_insufficient_cash_raises(self, isolated_portfolio):
        pt = isolated_portfolio
        with pytest.raises(ValueError, match="Insufficient cash"):
            pt._execute_trade_locked(
                symbol="PTT", asset_type="Stock", action="buy",
                units=1_000_000, price=30.0, currency="THB",
            )

    def test_sell_realized_profit(self, isolated_portfolio):
        pt = isolated_portfolio
        pt._record_income_locked("Other", 500_000.0, None)
        pt._execute_trade_locked(
            symbol="PTT", asset_type="Stock", action="buy",
            units=1000, price=30.0, currency="THB",
        )
        result = pt._execute_trade_locked(
            symbol="PTT", asset_type="Stock", action="sell",
            units=500, price=35.0, currency="THB",
        )
        assert "[SELL] PTT" in result
        assert "+2,500" in result  # (35-30) * 500

        _, state = pt._load_or_init()
        holding = pt._find_holding(state, "PTT")
        assert holding.units == 500  # หลังขาย 500 จาก 1000
        assert state.summary.total_realized_profit_ytd == 2_500.0

    def test_sell_too_many_raises(self, isolated_portfolio):
        pt = isolated_portfolio
        pt._record_income_locked("Other", 500_000.0, None)
        pt._execute_trade_locked(
            symbol="PTT", asset_type="Stock", action="buy",
            units=1000, price=30.0, currency="THB",
        )
        with pytest.raises(ValueError, match="Insufficient units to sell"):
            pt._execute_trade_locked(
                symbol="PTT", asset_type="Stock", action="sell",
                units=2000, price=35.0, currency="THB",
            )

    def test_cannot_mix_currency(self, isolated_portfolio):
        pt = isolated_portfolio
        # USD trades หักจาก CASH_USD (แยกจาก CASH_THB ตามสถาปัตยกรรมใหม่)
        pt._manage_cash_flow_locked(amount=10_000.0, action="deposit", currency="USD")
        pt._record_income_locked("Other", 500_000.0, None)  # THB ไว้ทดสอบ THB buy ด้านล่าง
        pt._execute_trade_locked(
            symbol="AAPL", asset_type="Stock", action="buy",
            units=10, price=150.0, currency="USD",
        )
        with pytest.raises(ValueError, match="USD อยู่แล้ว"):
            pt._execute_trade_locked(
                symbol="AAPL", asset_type="Stock", action="buy",
                units=5, price=150.0, currency="THB",
            )

    def test_weighted_avg_cost(self, isolated_portfolio):
        pt = isolated_portfolio
        pt._record_income_locked("Other", 500_000.0, None)
        pt._execute_trade_locked(
            symbol="PTT", asset_type="Stock", action="buy",
            units=1000, price=30.0, currency="THB",
        )
        result = pt._execute_trade_locked(
            symbol="PTT", asset_type="Stock", action="buy",
            units=1000, price=40.0, currency="THB",
        )
        _, state = pt._load_or_init()
        holding = pt._find_holding(state, "PTT")
        assert holding.units == 2000
        # weighted avg = (1000*30 + 1000*40) / 2000 = 35
        assert holding.avg_cost_thb == 35.0
        # Spec: tool output must announce avg cost change on subsequent buys
        assert "Avg cost updated" in result


class TestRecordIncome:
    def test_dividend_increments_cash_and_passive_income(self, isolated_portfolio):
        pt = isolated_portfolio
        # Call public wrapper to cover line 272
        result = pt.record_income.func(income_type="Dividend", amount_thb=5_000.0)
        assert "[DIV]" in result
        assert "5,000.00" in result

        _, state = pt._load_or_init()
        cash = pt._find_holding(state, "CASH_THB")
        assert cash.units == 5_000.0
        assert state.summary.passive_income_ytd == 5_000.0

    def test_dividend_with_source(self, isolated_portfolio):
        pt = isolated_portfolio
        pt._record_income_locked("Other", 500_000.0, None)
        pt._execute_trade_locked(
            symbol="PTT", asset_type="Stock", action="buy",
            units=1000, price=30.0, currency="THB",
        )
        pt._record_income_locked("Dividend", 3_000.0, "PTT")

        _, state = pt._load_or_init()
        ptt = pt._find_holding(state, "PTT")
        assert ptt.accumulated_dividend_thb == 3_000.0

    def test_zero_amount_raises(self, isolated_portfolio):
        pt = isolated_portfolio
        # validation is in the public wrapper, not _locked — call the wrapper
        result = pt.record_income.invoke({
            "income_type": "Dividend",
            "amount_thb": 0,
        })

        assert isinstance(result, str) and result.startswith("Error:")
        assert "ต้องมากกว่า 0" in result
    def test_unknown_source_raises(self, isolated_portfolio):
        pt = isolated_portfolio
        with pytest.raises(ValueError, match="ไม่พบ"):
            pt._record_income_locked("Dividend", 1_000.0, "UNKNOWN_SYMBOL")


class TestManageCashFlow:
    def test_deposit_thb_increases_cash_thb(self, isolated_portfolio):
        pt = isolated_portfolio
        # Call public wrapper to cover line 529
        result = pt.manage_cash_flow.func(amount=50_000.0, action="deposit", currency="THB")
        assert "[DEPOSIT]" in result and "CASH_THB" in result
        _, state = pt._load_or_init()
        cash = pt._find_holding(state, "CASH_THB")
        assert cash.units == 50_000.0

    def test_deposit_usd_creates_cash_usd_holding(self, isolated_portfolio):
        pt = isolated_portfolio
        result = pt._manage_cash_flow_locked(1_500.0, "deposit", "USD")
        assert "CASH_USD" in result and "USD" in result
        _, state = pt._load_or_init()
        cash_usd = pt._find_holding(state, "CASH_USD")
        assert cash_usd is not None
        assert cash_usd.units == 1_500.0
        # market_value_thb = 1500 × 36.5 (default fx) = 54,750
        assert cash_usd.market_value_thb == pytest.approx(54_750.0)

    def test_withdraw_decreases_cash(self, isolated_portfolio):
        pt = isolated_portfolio
        pt._manage_cash_flow_locked(100_000.0, "deposit", "THB")
        pt._manage_cash_flow_locked(30_000.0, "withdraw", "THB")
        _, state = pt._load_or_init()
        cash = pt._find_holding(state, "CASH_THB")
        assert cash.units == 70_000.0

    def test_withdraw_insufficient_raises(self, isolated_portfolio):
        pt = isolated_portfolio
        pt._manage_cash_flow_locked(10_000.0, "deposit", "THB")
        with pytest.raises(ValueError, match="Insufficient cash"):
            pt._manage_cash_flow_locked(20_000.0, "withdraw", "THB")

    def test_invalid_action_raises(self, isolated_portfolio):
        pt = isolated_portfolio
        with pytest.raises(Exception):
            pt.manage_cash_flow.invoke({
                "amount": 1_000.0, "action": "transfer", "currency": "THB"
            })
    def test_invalid_currency_raises(self, isolated_portfolio):
        pt = isolated_portfolio
        with pytest.raises(Exception):
            pt.manage_cash_flow.invoke({
                "amount": 1_000.0, "action": "deposit", "currency": "EUR"
            })
    def test_zero_amount_raises(self, isolated_portfolio):
        pt = isolated_portfolio
        result = pt.manage_cash_flow.invoke({
            "amount": 0, "action": "deposit", "currency": "THB"
        })

        assert isinstance(result, str) and result.startswith("Error:")
        assert "ต้องมากกว่า 0" in result
    def test_deposit_preserves_unrealized_pnl(self, isolated_portfolio):
        """Anti-Drift §3.2: deposit เพิ่ม NAV เท่ากับเพิ่ม cost basis → unrealized P/L คงเดิม"""
        pt = isolated_portfolio
        # Setup: ซื้อ PTT แล้ว simulate ราคาขึ้น และ persist ลงดิสก์
        pt._manage_cash_flow_locked(500_000.0, "deposit", "THB")
        pt._execute_trade_locked(
            symbol="PTT", asset_type="Stock", action="buy",
            units=1000, price=30.0, currency="THB",
        )
        post, state = pt._load_or_init()
        ptt = pt._find_holding(state, "PTT")
        ptt.current_price_thb = 35.0
        pt._save(post, state)  # persist price + recalc

        _, state = pt._load_or_init()
        unrealized_before = state.summary.total_unrealized_profit
        nav_before = state.summary.total_value_thb
        cost_before = pt._compute_total_cost(state, current_fx=36.5)

        # Deposit เพิ่ม → ทั้ง NAV และ cost ต้องขึ้น 100k เท่ากัน, P/L คงเดิม
        pt._manage_cash_flow_locked(100_000.0, "deposit", "THB")
        _, state2 = pt._load_or_init()
        cost_after = pt._compute_total_cost(state2, current_fx=36.5)
        assert state2.summary.total_value_thb == pytest.approx(nav_before + 100_000.0)
        assert cost_after == pytest.approx(cost_before + 100_000.0)
        assert state2.summary.total_unrealized_profit == pytest.approx(unrealized_before)


class TestEditHolding:
    def test_edit_units(self, isolated_portfolio):
        pt = isolated_portfolio
        pt._manage_cash_flow_locked(500_000.0, "deposit", "THB")
        pt._execute_trade_locked(
            symbol="PTT", asset_type="Stock", action="buy",
            units=1000, price=30.0, currency="THB",
        )
        result = pt._edit_holding_locked(
            "PTT", units=950.0, avg_cost=None,
            accumulated_dividend_thb=None, asset_type=None,
            reason="fixed bonus share calc",
        )
        assert "[EDIT PTT]" in result
        assert "units: 1000 → 950" in result
        _, state = pt._load_or_init()
        ptt = pt._find_holding(state, "PTT")
        assert ptt.units == 950.0

    def test_edit_avg_cost_usd_routes_correctly(self, isolated_portfolio):
        pt = isolated_portfolio
        pt._manage_cash_flow_locked(5_000.0, "deposit", "USD")
        pt._execute_trade_locked(
            symbol="AAPL", asset_type="Stock", action="buy",
            units=10, price=200.0, currency="USD",
        )
        # Call the public wrapper to cover line 680 (return _edit_holding_locked)
        pt.edit_holding.func(
            symbol="AAPL", units=None, avg_cost=180.0,
            accumulated_dividend_thb=None, asset_type=None,
            reason="corrected entry price"
        )
        _, state = pt._load_or_init()
        aapl = pt._find_holding(state, "AAPL")
        assert aapl.avg_cost_usd == 180.0
        assert aapl.avg_cost_thb is None  # ไม่ไปสร้าง field ผิดสกุล

    def test_edit_cash_sentinel_rejected(self, isolated_portfolio):
        pt = isolated_portfolio
        result = pt.edit_holding.invoke({
            "symbol": "CASH_THB", "units": 100.0, "reason": "test"
        })

        assert isinstance(result, str) and result.startswith("Error:")
        assert "manage_cash_flow" in result
    def test_edit_nonexistent_symbol_raises(self, isolated_portfolio):
        pt = isolated_portfolio
        with pytest.raises(ValueError, match="ไม่พบ"):
            pt._edit_holding_locked(
                "GHOST", units=100.0, avg_cost=None,
                accumulated_dividend_thb=None, asset_type=None, reason="x",
            )

    def test_edit_no_field_raises(self, isolated_portfolio):
        pt = isolated_portfolio
        result = pt.edit_holding.invoke({"symbol": "PTT", "reason": "test"})

        assert isinstance(result, str) and result.startswith("Error:")
        assert "อย่างน้อย 1 field" in result
    def test_edit_negative_units_raises(self, isolated_portfolio):
        pt = isolated_portfolio
        result = pt.edit_holding.invoke({"symbol": "PTT", "units": -5, "reason": "x"})

        assert isinstance(result, str) and result.startswith("Error:")
        assert "units ต้องมากกว่า 0" in result
    def test_edit_negative_accumulated_dividend_raises(self, isolated_portfolio):
        pt = isolated_portfolio
        result = pt.edit_holding.invoke({
            "symbol": "PTT", "accumulated_dividend_thb": -100, "reason": "x"
        })

        assert isinstance(result, str) and result.startswith("Error:")
        assert "accumulated_dividend_thb" in result
    def test_edit_appends_to_trading_journal(self, isolated_portfolio, tmp_vault):
        pt = isolated_portfolio
        pt._manage_cash_flow_locked(500_000.0, "deposit", "THB")
        pt._execute_trade_locked(
            symbol="PTT", asset_type="Stock", action="buy",
            units=1000, price=30.0, currency="THB",
        )
        pt._edit_holding_locked(
            "PTT", units=950.0, avg_cost=None,
            accumulated_dividend_thb=None, asset_type=None,
            reason="dividend stock split",
        )
        journal_path = tmp_vault / "20_Portfolio_Management" / "Journals_and_Reports" / "Trading_Journal.md"
        text = journal_path.read_text(encoding="utf-8")
        assert "[EDIT PTT]" in text
        assert "dividend stock split" in text

    def test_edit_no_actual_change_raises(self, isolated_portfolio):
        pt = isolated_portfolio
        pt._manage_cash_flow_locked(500_000.0, "deposit", "THB")
        pt._execute_trade_locked(
            symbol="PTT", asset_type="Stock", action="buy",
            units=1000, price=30.0, currency="THB",
        )
        with pytest.raises(ValueError, match="เหมือนเดิม"):
            pt._edit_holding_locked(
                "PTT", units=1000.0, avg_cost=30.0,
                accumulated_dividend_thb=None, asset_type=None,
                reason="noop",
            )




import tools.portfolio.trading

class TestBatchImportHoldings:
    def test_invalid_mode_raises(self, isolated_portfolio):
        pt = isolated_portfolio
        result = pt.batch_import_holdings.func(assets_list=[], mode="invalid")
            
        assert isinstance(result, str) and result.startswith("Error:")
        assert "mode" in result
    def test_not_list_raises(self, isolated_portfolio):
        pt = isolated_portfolio
        result = pt.batch_import_holdings.func(assets_list="not_a_list")

        assert isinstance(result, str) and result.startswith("Error:")
        assert "assets_list ต้องเป็น list" in result
    def test_empty_list_merge_raises(self, isolated_portfolio):
        pt = isolated_portfolio
        result = pt.batch_import_holdings.func(assets_list=[])
            
        assert isinstance(result, str) and result.startswith("Error:")
        assert "assets_list ต้องเป็น list ที่ไม่ว่าง" in result
    def test_invalid_item_format_raises(self, isolated_portfolio):
        pt = isolated_portfolio
        result = pt.batch_import_holdings.func(assets_list=["invalid"])
            
        assert isinstance(result, str) and result.startswith("Error:")
        assert "ไม่ใช่ dict" in result
    def test_missing_fields_raises(self, isolated_portfolio):
        pt = isolated_portfolio
        result = pt.batch_import_holdings.func(assets_list=[{"symbol": "PTT"}])
            
        assert isinstance(result, str) and result.startswith("Error:")
        assert "field ขาดหรือ format ผิด" in result
    def test_empty_symbol_raises(self, isolated_portfolio):
        pt = isolated_portfolio
        result = pt.batch_import_holdings.func(assets_list=[{
            "symbol": "", "asset_type": "Stock", "units": 10, "avg_cost": 10, "currency": "THB"
        }])

        assert isinstance(result, str) and result.startswith("Error:")
        assert "symbol ว่าง" in result
    def test_cash_sentinel_raises(self, isolated_portfolio):
        pt = isolated_portfolio
        result = pt.batch_import_holdings.func(assets_list=[{
            "symbol": "CASH_THB", "asset_type": "Cash", "units": 10, "avg_cost": 1, "currency": "THB"
        }])

        assert isinstance(result, str) and result.startswith("Error:")
        assert "cash sentinel (CASH_THB/CASH_USD)" in result
    def test_negative_values_raises(self, isolated_portfolio):
        pt = isolated_portfolio
        result = pt.batch_import_holdings.func(assets_list=[{
            "symbol": "PTT", "asset_type": "Stock", "units": -10, "avg_cost": 10, "currency": "THB"
        }])
        assert isinstance(result, str) and result.startswith("Error:")
        assert "units ต้องมากกว่า 0" in result
        result = pt.batch_import_holdings.func(assets_list=[{
            "symbol": "PTT", "asset_type": "Stock", "units": 10, "avg_cost": -10, "currency": "THB"
        }])

        assert isinstance(result, str) and result.startswith("Error:")
        assert "avg_cost ต้องมากกว่า 0" in result
    def test_invalid_currency_raises(self, isolated_portfolio):
        pt = isolated_portfolio
        result = pt.batch_import_holdings.func(assets_list=[{
            "symbol": "PTT", "asset_type": "Stock", "units": 10, "avg_cost": 10, "currency": "EUR"
        }])

        assert isinstance(result, str) and result.startswith("Error:")
        assert "currency ต้องเป็น " in result
    def test_duplicate_symbol_raises(self, isolated_portfolio):
        pt = isolated_portfolio
        result = pt.batch_import_holdings.func(assets_list=[
            {"symbol": "PTT", "asset_type": "Stock", "units": 10, "avg_cost": 10, "currency": "THB"},
            {"symbol": "ptt", "asset_type": "Stock", "units": 20, "avg_cost": 15, "currency": "THB"}
        ])

        assert isinstance(result, str) and result.startswith("Error:")
        assert "ซ้ำ" in result
    def test_invalid_current_price_raises(self, isolated_portfolio):
        pt = isolated_portfolio
        result = pt.batch_import_holdings.func(assets_list=[{
            "symbol": "PTT", "asset_type": "Stock", "units": 10, "avg_cost": 10, "currency": "THB",
            "current_price": "invalid"
        }])
        assert isinstance(result, str) and result.startswith("Error:")
        assert "current_price format ผิด" in result
        result = pt.batch_import_holdings.func(assets_list=[{
            "symbol": "PTT", "asset_type": "Stock", "units": 10, "avg_cost": 10, "currency": "THB",
            "current_price": -5
        }])

        assert isinstance(result, str) and result.startswith("Error:")
        assert "current_price ต้องมากกว่า 0" in result
    def test_import_with_provided_price(self, isolated_portfolio):
        pt = isolated_portfolio
        pt._manage_cash_flow_locked(amount=1000, action="deposit", currency="THB")
        pt._manage_cash_flow_locked(amount=1000, action="deposit", currency="USD")
        
        result = pt.batch_import_holdings.func(assets_list=[
            {"symbol": "PTT", "asset_type": "Stock", "units": 100, "avg_cost": 30.0, "currency": "THB", "current_price": 35.0},
            {"symbol": "AAPL", "asset_type": "Stock", "units": 10, "avg_cost": 150.0, "currency": "USD", "current_price": 180.0}
        ], mode="merge")
        
        assert "[IMPORT MERGE]" in result
        assert "provided=2" in result
        
        _, state = pt._load_or_init()
        ptt = pt._find_holding(state, "PTT")
        aapl = pt._find_holding(state, "AAPL")
        
        assert ptt.units == 100
        assert ptt.avg_cost_thb == 30.0
        assert ptt.current_price_thb == 35.0
        
        assert aapl.units == 10
        assert aapl.avg_cost_usd == 150.0
        assert aapl.current_price_usd == 180.0

    def test_import_with_fetch_success(self, isolated_portfolio, monkeypatch):
        pt = isolated_portfolio
        monkeypatch.setattr(tools.portfolio.trading, "fetch_latest_price", lambda sym, cur: 40.0)
        
        result = pt.batch_import_holdings.func(assets_list=[
            {"symbol": "PTT", "asset_type": "Stock", "units": 100, "avg_cost": 30.0, "currency": "THB"}
        ], mode="merge")
        
        assert "fetched=1" in result
        _, state = pt._load_or_init()
        ptt = pt._find_holding(state, "PTT")
        assert ptt.current_price_thb == 40.0

    def test_import_with_fetch_failure(self, isolated_portfolio, monkeypatch):
        pt = isolated_portfolio
        def mock_fail(*args):
            raise ValueError("fetch failed")
        monkeypatch.setattr(tools.portfolio.trading, "fetch_latest_price", mock_fail)
        
        result = pt.batch_import_holdings.func(assets_list=[
            {"symbol": "PTT", "asset_type": "Stock", "units": 100, "avg_cost": 30.0, "currency": "THB"}
        ], mode="merge")
        
        assert "fallback=1" in result
        _, state = pt._load_or_init()
        ptt = pt._find_holding(state, "PTT")
        assert ptt.current_price_thb == 30.0

    def test_import_with_fetch_timeout(self, isolated_portfolio, monkeypatch):
        pt = isolated_portfolio
        import time
        def mock_timeout(*args):
            time.sleep(0.5)
            return 40.0
        monkeypatch.setattr(tools.portfolio.trading, "fetch_latest_price", mock_timeout)
        monkeypatch.setattr(tools.portfolio.trading, "_PRICE_FETCH_TIMEOUT", 0.1)
        
        result = pt.batch_import_holdings.func(assets_list=[
            {"symbol": "PTT", "asset_type": "Stock", "units": 100, "avg_cost": 30.0, "currency": "THB"}
        ], mode="merge")
        
        assert "fallback=1" in result

    def test_import_overwrite_mode(self, isolated_portfolio):
        pt = isolated_portfolio
        pt._manage_cash_flow_locked(amount=1000, action="deposit", currency="THB")
        pt._manage_cash_flow_locked(amount=1000, action="deposit", currency="USD")
        pt._execute_trade_locked("OLD", "Stock", "buy", 10, 10, "THB")
        
        result = pt.batch_import_holdings.func(assets_list=[
            {"symbol": "NEW", "asset_type": "Stock", "units": 100, "avg_cost": 30.0, "currency": "THB", "current_price": 35.0}
        ], mode="overwrite", reset_cash_usd=True)
        
        assert "[IMPORT OVERWRITE]" in result
        _, state = pt._load_or_init()
        old = pt._find_holding(state, "OLD")
        assert old is None
        new = pt._find_holding(state, "NEW")
        assert new is not None
        
        cash_usd = pt._find_holding(state, "CASH_USD")
        assert cash_usd.units == 0.0

    def test_import_existing_holding_merge(self, isolated_portfolio):
        pt = isolated_portfolio
        pt._manage_cash_flow_locked(amount=500_000, action="deposit", currency="THB")
        pt._execute_trade_locked("PTT", "Stock", "buy", 100, 30.0, "THB")
        pt._record_income_locked("Dividend", 50.0, "PTT")
        
        pt.batch_import_holdings.func(assets_list=[
            {"symbol": "PTT", "asset_type": "Stock", "units": 200, "avg_cost": 35.0, "currency": "THB", "current_price": 40.0}
        ], mode="merge")
        
        _, state = pt._load_or_init()
        ptt = pt._find_holding(state, "PTT")
        assert ptt.units == 200
        assert ptt.avg_cost_thb == 35.0
        assert ptt.accumulated_dividend_thb == 50.0 # Preserved

    def test_import_lock_timeout(self, isolated_portfolio, monkeypatch):
        pt = isolated_portfolio
        def mock_lock(*args, **kwargs):
            from filelock import Timeout
            raise Timeout("mock")
        monkeypatch.setattr(tools.portfolio.trading._portfolio_lock, "acquire", mock_lock)
        
        result = pt.batch_import_holdings.func(assets_list=[
            {"symbol": "PTT", "asset_type": "Stock", "units": 100, "avg_cost": 30.0, "currency": "THB", "current_price": 35.0}
        ], mode="merge")


        assert isinstance(result, str) and result.startswith("Error:")
        assert "portfolio lock" in result
class TestUpdateFxRate:
    def test_update_fx_manual(self, isolated_portfolio):
        pt = isolated_portfolio
        result = pt.update_fx_rate.func(rate=35.5)
        assert "[FX manual]" in result
        assert "35.5" in result
        
        _, state = pt._load_or_init()
        assert state.fx_rates.get("USDTHB") == 35.5

    def test_update_fx_negative_raises(self, isolated_portfolio):
        pt = isolated_portfolio
        result = pt.update_fx_rate.func(rate=-5.0)
            
        assert isinstance(result, str) and result.startswith("Error:")
        assert "rate ต้องมากกว่า 0" in result
    def test_update_fx_auto_fetch(self, isolated_portfolio, monkeypatch):
        pt = isolated_portfolio
        monkeypatch.setattr(tools.portfolio.trading, "_fetch_fx_rate", lambda: 36.0)
        
        result = pt.update_fx_rate.func()
        assert "[FX yfinance]" in result
        assert "36.0" in result
        
        _, state = pt._load_or_init()
        assert state.fx_rates.get("USDTHB") == 36.0

    def test_update_fx_auto_fetch_failure(self, isolated_portfolio, monkeypatch):
        pt = isolated_portfolio
        monkeypatch.setattr(tools.portfolio.trading, "_fetch_fx_rate", lambda: None)
        
        result = pt.update_fx_rate.func()

        assert isinstance(result, str) and result.startswith("Error:")
        assert "auto-fetch FX ล้มเหลว" in result
    def test_update_fx_lock_timeout(self, isolated_portfolio, monkeypatch):
        pt = isolated_portfolio
        def mock_lock(*args, **kwargs):
            from filelock import Timeout
            raise Timeout("mock")
        monkeypatch.setattr(tools.portfolio.trading._portfolio_lock, "acquire", mock_lock)
        
        result = pt.update_fx_rate.func(rate=35.0)


        assert isinstance(result, str) and result.startswith("Error:")
        assert "portfolio lock" in result
class TestTradingExceptions:
    def test_execute_trade_lock_timeout(self, isolated_portfolio, monkeypatch):
        pt = isolated_portfolio
        def mock_lock(*args, **kwargs):
            from filelock import Timeout
            raise Timeout("mock")
        monkeypatch.setattr(tools.portfolio.trading._portfolio_lock, "acquire", mock_lock)
        
        result = pt.execute_trade.func(symbol="PTT", asset_type="Stock", action="buy", units=100, price=30.0, currency="THB")
            
        assert isinstance(result, str) and result.startswith("Error:")
        assert "portfolio lock" in result
    def test_execute_trade_invalid_action(self, isolated_portfolio):
        pt = isolated_portfolio
        result = pt.execute_trade.func(symbol="PTT", asset_type="Stock", action="hold", units=100, price=30.0, currency="THB")

        assert isinstance(result, str) and result.startswith("Error:")
        assert "action ต้องเป็น " in result
    def test_execute_trade_negative_units_price(self, isolated_portfolio):
        pt = isolated_portfolio
        result = pt.execute_trade.func(symbol="PTT", asset_type="Stock", action="buy", units=-100, price=30.0, currency="THB")
        assert isinstance(result, str) and result.startswith("Error:")
        assert "units ต้องมากกว่า 0" in result
        result = pt.execute_trade.func(symbol="PTT", asset_type="Stock", action="buy", units=100, price=-30.0, currency="THB")
            
        assert isinstance(result, str) and result.startswith("Error:")
        assert "price ต้องมากกว่า 0" in result
    def test_execute_trade_cash_sentinel(self, isolated_portfolio):
        pt = isolated_portfolio
        result = pt.execute_trade.func(symbol="CASH_THB", asset_type="Cash", action="buy", units=100, price=1.0, currency="THB")
            
        assert isinstance(result, str) and result.startswith("Error:")
        assert "cash sentinel (CASH_THB/CASH_USD)" in result
    def test_execute_trade_sell_not_found(self, isolated_portfolio):
        pt = isolated_portfolio
        with pytest.raises(ValueError, match="ไม่มีในพอร์ต"):
            pt._execute_trade_locked(symbol="UNKNOWN", asset_type="Stock", action="sell", units=100, price=30.0, currency="THB")

    def test_execute_trade_usd_success(self, isolated_portfolio):
        pt = isolated_portfolio
        pt._manage_cash_flow_locked(500_000, "deposit", "USD")
        pt._execute_trade_locked("AAPL", "Stock", "buy", 10, 150.0, "USD")
        
        # Test buying USD when already USD
        result = pt._execute_trade_locked("AAPL", "Stock", "buy", 10, 160.0, "USD")
        assert "Avg cost updated to $" in result
        
        # Test selling USD when already USD
        result = pt._execute_trade_locked("AAPL", "Stock", "sell", 10, 170.0, "USD")
        assert "[SELL]" in result

    def test_execute_trade_currency_mismatch_sell(self, isolated_portfolio):
        pt = isolated_portfolio
        pt._manage_cash_flow_locked(500_000, "deposit", "THB")
        pt._manage_cash_flow_locked(500_000, "deposit", "USD")
        pt._execute_trade_locked("PTT", "Stock", "buy", 100, 30.0, "THB")
        pt._execute_trade_locked("AAPL", "Stock", "buy", 10, 150.0, "USD")
        
        with pytest.raises(ValueError, match="cost เป็น THB ไม่สามารถขายด้วย USD ได้"):
            pt._execute_trade_locked("PTT", "Stock", "sell", 10, 35.0, "USD")
            
        with pytest.raises(ValueError, match="cost เป็น USD ไม่สามารถขายด้วย THB ได้"):
            pt._execute_trade_locked("AAPL", "Stock", "sell", 5, 180.0, "THB")
            
    def test_execute_trade_currency_mismatch_buy(self, isolated_portfolio):
        pt = isolated_portfolio
        pt._manage_cash_flow_locked(500_000, "deposit", "THB")
        pt._manage_cash_flow_locked(500_000, "deposit", "USD")
        pt._execute_trade_locked("PTT", "Stock", "buy", 100, 30.0, "THB")
        
        with pytest.raises(ValueError, match="เป็นสินทรัพย์ THB อยู่แล้ว"):
            pt._execute_trade_locked("PTT", "Stock", "buy", 100, 1.0, "USD")

    def test_execute_trade_deletes_when_empty(self, isolated_portfolio):
        pt = isolated_portfolio
        pt._manage_cash_flow_locked(500_000, "deposit", "THB")
        pt._execute_trade_locked("PTT", "Stock", "buy", 100, 30.0, "THB")
        pt._execute_trade_locked("PTT", "Stock", "sell", 100, 35.0, "THB")
        _, state = pt._load_or_init()
        ptt = pt._find_holding(state, "PTT")
        assert ptt is None

    def test_record_income_lock_timeout(self, isolated_portfolio, monkeypatch):
        pt = isolated_portfolio
        def mock_lock(*args, **kwargs):
            from filelock import Timeout
            raise Timeout("mock")
        monkeypatch.setattr(tools.portfolio.trading._portfolio_lock, "acquire", mock_lock)
        
        result = pt.record_income.func(income_type="Dividend", amount_thb=1000.0)
            
        assert isinstance(result, str) and result.startswith("Error:")
        assert "portfolio lock" in result
    def test_record_income_cash_source(self, isolated_portfolio):
        pt = isolated_portfolio
        with pytest.raises(ValueError, match="source_symbol ห้ามเป็น cash sentinel"):
            pt._record_income_locked("Dividend", 1000.0, "CASH_THB")

    def test_manage_cash_flow_lock_timeout(self, isolated_portfolio, monkeypatch):
        pt = isolated_portfolio
        def mock_lock(*args, **kwargs):
            from filelock import Timeout
            raise Timeout("mock")
        monkeypatch.setattr(tools.portfolio.trading._portfolio_lock, "acquire", mock_lock)
        
        result = pt.manage_cash_flow.func(amount=1000.0, action="deposit", currency="THB")

        assert isinstance(result, str) and result.startswith("Error:")
        assert "portfolio lock" in result
    def test_manage_cash_flow_negative_amount(self, isolated_portfolio):
        pt = isolated_portfolio
        result = pt.manage_cash_flow.func(amount=-1000.0, action="deposit", currency="THB")

        assert isinstance(result, str) and result.startswith("Error:")
        assert "amount ต้องมากกว่า 0" in result
    def test_manage_cash_flow_invalid_action_currency(self, isolated_portfolio):
        pt = isolated_portfolio
        result = pt.manage_cash_flow.func(amount=1000.0, action="steal", currency="THB")
        assert isinstance(result, str) and result.startswith("Error:")
        assert "action ต้องเป็น " in result
        result = pt.manage_cash_flow.func(amount=1000.0, action="deposit", currency="EUR")

        assert isinstance(result, str) and result.startswith("Error:")
        assert "currency ต้องเป็น " in result
    def test_edit_holding_lock_timeout(self, isolated_portfolio, monkeypatch):
        pt = isolated_portfolio
        def mock_lock(*args, **kwargs):
            from filelock import Timeout
            raise Timeout("mock")
        monkeypatch.setattr(tools.portfolio.trading._portfolio_lock, "acquire", mock_lock)
        
        result = pt.edit_holding.func(symbol="PTT", units=100, reason="test")

        assert isinstance(result, str) and result.startswith("Error:")
        assert "portfolio lock" in result
    def test_edit_holding_negative_units(self, isolated_portfolio):
        pt = isolated_portfolio
        result = pt.edit_holding.func(symbol="PTT", units=-100, reason="test")
        assert isinstance(result, str) and result.startswith("Error:")
        assert "units ต้องมากกว่า 0" in result
        result = pt.edit_holding.func(symbol="PTT", avg_cost=-10.0, reason="test")

        assert isinstance(result, str) and result.startswith("Error:")
        assert "avg_cost ต้องมากกว่า 0" in result
    def test_edit_holding_cash_raises(self, isolated_portfolio):
        pt = isolated_portfolio
        with pytest.raises(ValueError, match="เป็น Cash holding"):
            pt._edit_holding_locked("CASH_THB", units=100.0, avg_cost=None, accumulated_dividend_thb=None, asset_type=None, reason="x")

    def test_edit_holding_missing_avg_cost(self, isolated_portfolio):
        pt = isolated_portfolio
        pt._manage_cash_flow_locked(500_000, "deposit", "THB")
        pt._execute_trade_locked("PTT", "Stock", "buy", 100, 30.0, "THB")
        _, state = pt._load_or_init()
        ptt = pt._find_holding(state, "PTT")
        ptt.avg_cost_thb = None # force missing
        pt._save(_, state)
        
        with pytest.raises(ValueError, match="ไม่มี avg_cost_thb/usd เดิม"):
            pt._edit_holding_locked("PTT", units=None, avg_cost=35.0, accumulated_dividend_thb=None, asset_type=None, reason="x")

    def test_edit_holding_asset_type_and_dividend(self, isolated_portfolio):
        pt = isolated_portfolio
        pt._manage_cash_flow_locked(500_000, "deposit", "THB")
        pt._execute_trade_locked("PTT", "Stock", "buy", 100, 30.0, "THB")
        
        result = pt._edit_holding_locked("PTT", units=None, avg_cost=35.0, accumulated_dividend_thb=100.0, asset_type="ETF", reason="fix")
        assert "asset_type: Stock → ETF" in result
        assert "accumulated_dividend_thb" in result
        assert "avg_cost_thb" in result
        _, state = pt._load_or_init()
        ptt = pt._find_holding(state, "PTT")
        assert ptt.asset_type == "ETF"
        assert ptt.accumulated_dividend_thb == 100.0
        assert ptt.avg_cost_thb == 35.0

class TestComputeTotalCostEdgeCases:
    def test_compute_total_cost_incomplete_pairs(self, isolated_portfolio):
        pt = isolated_portfolio
        pt._manage_cash_flow_locked(500_000, "deposit", "THB")
        pt._execute_trade_locked("PTT", "Stock", "buy", 100, 30.0, "THB")
        _, state = pt._load_or_init()
        
        ptt = pt._find_holding(state, "PTT")
        # To cover 772 completely, we need current_price_thb = None
        ptt.avg_cost_usd = None
        ptt.current_price_usd = None
        ptt.avg_cost_thb = 30.0
        ptt.current_price_thb = None
        
        total = pt._compute_total_cost(state, 36.5)
        # Only Cash_THB should be counted since PTT is incomplete
        assert total == 500_000 - 3000

        ptt.avg_cost_thb = None
        ptt.current_price_thb = 30.0
        total2 = pt._compute_total_cost(state, 36.5)
        assert total2 == 500_000 - 3000

    def test_compute_total_cost_usd_success(self, isolated_portfolio):
        pt = isolated_portfolio
        pt._manage_cash_flow_locked(500_000, 'deposit', 'USD')
        pt._execute_trade_locked('AAPL', 'Stock', 'buy', 10, 150.0, 'USD')
        _, state = pt._load_or_init()
        total = pt._compute_total_cost(state, 36.5)
        assert total > 0
