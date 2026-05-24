"""Test portfolio_tools: recalc, execute_trade, record_income — ใช้ Vault แยกผ่าน tmp_path"""
import json

import pytest


# --- Pure-function tests (ไม่แตะ disk) ---


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


# --- Integration tests: real disk write/read via tmp_vault ---


class TestExecuteTrade:
    def test_buy_new_thb_holding(self, isolated_portfolio):
        pt = isolated_portfolio
        # bootstrap with cash
        pt._record_income_locked("Other", 500_000.0, None)

        result = pt._execute_trade_locked(
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
        result = pt._record_income_locked("Dividend", 5_000.0, None)
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
        with pytest.raises(ValueError, match="ต้องมากกว่า 0"):
            pt.record_income.invoke({
                "income_type": "Dividend",
                "amount_thb": 0,
            })

    def test_unknown_source_raises(self, isolated_portfolio):
        pt = isolated_portfolio
        with pytest.raises(ValueError, match="ไม่พบ"):
            pt._record_income_locked("Dividend", 1_000.0, "UNKNOWN_SYMBOL")


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


# --- CASH_USD support: recalc, cost basis bottom-up ---


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


# --- manage_cash_flow ---


class TestManageCashFlow:
    def test_deposit_thb_increases_cash_thb(self, isolated_portfolio):
        pt = isolated_portfolio
        result = pt._manage_cash_flow_locked(50_000.0, "deposit", "THB")
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
        with pytest.raises(ValueError, match="action"):
            pt.manage_cash_flow.invoke({
                "amount": 1_000.0, "action": "transfer", "currency": "THB"
            })

    def test_invalid_currency_raises(self, isolated_portfolio):
        pt = isolated_portfolio
        with pytest.raises(ValueError, match="currency"):
            pt.manage_cash_flow.invoke({
                "amount": 1_000.0, "action": "deposit", "currency": "EUR"
            })

    def test_zero_amount_raises(self, isolated_portfolio):
        pt = isolated_portfolio
        with pytest.raises(ValueError, match="ต้องมากกว่า 0"):
            pt.manage_cash_flow.invoke({
                "amount": 0, "action": "deposit", "currency": "THB"
            })

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


# --- sync_market_prices ---


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

        monkeypatch.setattr(pt, "_refresh_prices", fake_refresh)
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

        monkeypatch.setattr(pt, "_refresh_prices", fake_refresh)
        result = pt.sync_market_prices.invoke({})
        assert "refreshed 0/1" in result
        assert "PTT=timeout" in result

    def test_sync_empty_portfolio(self, isolated_portfolio, monkeypatch):
        pt = isolated_portfolio
        # ไม่มี non-cash holding → refresh_info ว่าง
        monkeypatch.setattr(pt, "_refresh_prices", lambda state: {})
        result = pt.sync_market_prices.invoke({})
        assert "no non-cash holdings" in result


# --- update_fx_rate ---


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
        monkeypatch.setattr(pt, "_fetch_fx_rate", lambda: 34.80)
        result = pt._update_fx_rate_locked(None)
        assert "[FX yfinance]" in result
        assert "34.8000" in result

        _, state = pt._load_or_init()
        assert state.fx_rates["USDTHB"] == 34.80

    def test_auto_fetch_failure_raises(self, isolated_portfolio, monkeypatch):
        pt = isolated_portfolio
        monkeypatch.setattr(pt, "_fetch_fx_rate", lambda: None)
        with pytest.raises(ValueError, match="auto-fetch FX ล้มเหลว"):
            pt._update_fx_rate_locked(None)

    def test_negative_rate_raises(self, isolated_portfolio):
        pt = isolated_portfolio
        with pytest.raises(ValueError, match="rate ต้องมากกว่า 0"):
            pt.update_fx_rate.invoke({"rate": -1.0})

    def test_zero_rate_raises(self, isolated_portfolio):
        pt = isolated_portfolio
        with pytest.raises(ValueError, match="rate ต้องมากกว่า 0"):
            pt.update_fx_rate.invoke({"rate": 0})

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


# --- record_performance_snapshot: cash_balance ต้องรวมทั้ง 2 cash pots ---


class TestPerformanceSnapshotCashAggregation:
    def test_cash_balance_includes_cash_usd_in_thb_equivalent(
        self, isolated_portfolio, monkeypatch, tmp_vault
    ):
        """Anti-Drift §3.2: Cash_Balance ใน CSV ต้องรวม CASH_USD × fx
        ไม่งั้น downstream consumer (chart, report) จะเห็น
        Cash_Balance + non_cash ≠ Total_NAV → drift
        """
        pt = isolated_portfolio
        # ฝาก THB และ USD ทั้ง 2 pot
        pt._manage_cash_flow_locked(100_000.0, "deposit", "THB")
        pt._manage_cash_flow_locked(1_000.0, "deposit", "USD")
        # default fx = 36.5 → 1000 USD = 36,500 THB

        result = pt.record_performance_snapshot.invoke({"refresh_prices": False})
        assert "[PERF]" in result
        # Cash ต้องเป็น 100,000 + 1000*36.5 = 136,500
        assert "136,500.00" in result

        # Verify CSV row directly
        import csv
        csv_path = tmp_vault / "20_Portfolio_Management" / "Journals_and_Reports" / "Performance_Log.csv"
        with csv_path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        last = rows[-1]
        assert float(last["Cash_Balance"]) == pytest.approx(136_500.0)
        # Invariant: Cash_Balance ต้อง == Total_NAV เมื่อไม่มี non-cash holdings
        assert float(last["Cash_Balance"]) == pytest.approx(float(last["Total_NAV"]))


# --- edit_holding (correction tool) ---


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
        pt._edit_holding_locked(
            "AAPL", units=None, avg_cost=180.0,
            accumulated_dividend_thb=None, asset_type=None,
            reason="corrected entry price",
        )
        _, state = pt._load_or_init()
        aapl = pt._find_holding(state, "AAPL")
        assert aapl.avg_cost_usd == 180.0
        assert aapl.avg_cost_thb is None  # ไม่ไปสร้าง field ผิดสกุล

    def test_edit_cash_sentinel_rejected(self, isolated_portfolio):
        pt = isolated_portfolio
        with pytest.raises(ValueError, match="manage_cash_flow"):
            pt.edit_holding.invoke({
                "symbol": "CASH_THB", "units": 100.0, "reason": "test"
            })

    def test_edit_nonexistent_symbol_raises(self, isolated_portfolio):
        pt = isolated_portfolio
        with pytest.raises(ValueError, match="ไม่พบ"):
            pt._edit_holding_locked(
                "GHOST", units=100.0, avg_cost=None,
                accumulated_dividend_thb=None, asset_type=None, reason="x",
            )

    def test_edit_no_field_raises(self, isolated_portfolio):
        pt = isolated_portfolio
        with pytest.raises(ValueError, match="อย่างน้อย 1 field"):
            pt.edit_holding.invoke({"symbol": "PTT", "reason": "test"})

    def test_edit_negative_units_raises(self, isolated_portfolio):
        pt = isolated_portfolio
        with pytest.raises(ValueError, match="units ต้องมากกว่า 0"):
            pt.edit_holding.invoke({"symbol": "PTT", "units": -5, "reason": "x"})

    def test_edit_negative_accumulated_dividend_raises(self, isolated_portfolio):
        pt = isolated_portfolio
        with pytest.raises(ValueError, match="accumulated_dividend_thb"):
            pt.edit_holding.invoke({
                "symbol": "PTT", "accumulated_dividend_thb": -100, "reason": "x"
            })

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


# --- compute_allocation_breakdown ---


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


# --- read_performance_history ---


class TestReadPerformanceHistory:
    def test_no_file_returns_error_msg(self, isolated_portfolio):
        pt = isolated_portfolio
        result = pt.read_performance_history.invoke({"days": 30})
        data = json.loads(result)
        assert "error" in data and "ยังไม่มี" in data["error"]

    def test_invalid_days_raises(self, isolated_portfolio):
        pt = isolated_portfolio
        with pytest.raises(ValueError, match="days"):
            pt.read_performance_history.invoke({"days": 0})

    def test_metrics_calculation(self, isolated_portfolio, tmp_vault):
        pt = isolated_portfolio
        # เขียน CSV ตรงๆ ด้วย rows ที่ควบคุมได้ — เลี่ยง yfinance + simulate trend
        import csv
        csv_path = tmp_vault / "20_Portfolio_Management" / "Journals_and_Reports" / "Performance_Log.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Date", "Total_NAV", "Total_Cost", "Unrealized_PnL", "Cash_Balance"])
            # Trend: up → peak → drawdown
            for date, nav in [
                ("2026-04-01", 1_000_000),
                ("2026-04-02", 1_050_000),
                ("2026-04-03", 1_080_000),  # peak
                ("2026-04-04", 1_020_000),  # -5.56% from peak
                ("2026-04-05", 1_050_000),
            ]:
                w.writerow([date, f"{nav:.2f}", f"{nav:.2f}", "0.00", f"{nav:.2f}"])

        result = pt.read_performance_history.invoke({"days": 30})
        data = json.loads(result)
        assert data["n_observations"] == 5
        assert data["first_nav"] == 1_000_000.0
        assert data["latest_nav"] == 1_050_000.0
        assert data["change_abs"] == 50_000.0
        assert data["change_pct"] == pytest.approx(5.0)
        assert data["max_nav"] == 1_080_000.0
        assert data["min_nav"] == 1_000_000.0
        # max_drawdown: peak 1.08M → trough 1.02M = (1.02-1.08)/1.08 × 100 = -5.5556%
        assert data["max_drawdown_pct"] == pytest.approx(-5.56, abs=0.01)

    def test_tail_days(self, isolated_portfolio, tmp_vault):
        pt = isolated_portfolio
        import csv
        csv_path = tmp_vault / "20_Portfolio_Management" / "Journals_and_Reports" / "Performance_Log.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Date", "Total_NAV", "Total_Cost", "Unrealized_PnL", "Cash_Balance"])
            for i in range(10):
                w.writerow([f"2026-04-{i+1:02d}", f"{1_000_000 + i*1000:.2f}", "0.00", "0.00", "0.00"])

        result = pt.read_performance_history.invoke({"days": 3})
        data = json.loads(result)
        assert data["n_observations"] == 3
        # tail 3: เริ่ม 2026-04-08 (NAV 1_007_000), จบ 2026-04-10 (NAV 1_009_000)
        assert data["first_date"] == "2026-04-08"
        assert data["latest_date"] == "2026-04-10"


# --- read_watchlist ---


class TestReadWatchlist:
    def test_empty_watchlist(self, isolated_portfolio):
        pt = isolated_portfolio
        result = pt.read_watchlist.invoke({})
        data = json.loads(result)
        assert data["n_items"] == 0
        assert data["items"] == []

    def test_returns_added_items(self, isolated_portfolio):
        pt = isolated_portfolio
        pt.add_to_watchlist.invoke({
            "symbol": "NVDA", "asset_type": "Stock",
            "target_price": 100.0, "notes": "wait for dip"
        })
        pt.add_to_watchlist.invoke({
            "symbol": "PTT", "asset_type": "Stock", "target_price": 28.0
        })
        result = pt.read_watchlist.invoke({})
        data = json.loads(result)
        assert data["n_items"] == 2
        symbols = {it["symbol"] for it in data["items"]}
        assert symbols == {"NVDA", "PTT"}
        nvda = next(it for it in data["items"] if it["symbol"] == "NVDA")
        assert nvda["target_price"] == 100.0
        assert nvda["notes"] == "wait for dip"


# --- read_trading_journal ---


class TestReadTradingJournal:
    def test_no_file_returns_error_msg(self, isolated_portfolio):
        pt = isolated_portfolio
        result = pt.read_trading_journal.invoke({"days": 30})
        data = json.loads(result)
        assert "error" in data and "ยังไม่มี" in data["error"]

    def test_invalid_days_raises(self, isolated_portfolio):
        pt = isolated_portfolio
        with pytest.raises(ValueError, match="days"):
            pt.read_trading_journal.invoke({"days": 0})

    def test_invalid_limit_raises(self, isolated_portfolio):
        pt = isolated_portfolio
        with pytest.raises(ValueError, match="limit"):
            pt.read_trading_journal.invoke({"limit": 0})

    def test_reads_entries_newest_first(self, isolated_portfolio):
        pt = isolated_portfolio
        pt.append_trading_journal.invoke({"entry": "first entry — buy AAPL"})
        pt.append_trading_journal.invoke({"entry": "second entry — sell PTT"})
        pt.append_trading_journal.invoke({"entry": "third entry — market crash"})

        result = pt.read_trading_journal.invoke({})
        data = json.loads(result)
        assert data["n_total_in_window"] == 3
        assert data["n_returned"] == 3
        # newest first → third comes before first
        contents = [e["content"] for e in data["entries"]]
        assert "third entry" in contents[0]
        assert "first entry" in contents[2]

    def test_keyword_filter_case_insensitive(self, isolated_portfolio):
        pt = isolated_portfolio
        pt.append_trading_journal.invoke({"entry": "Bought AAPL because earnings beat"})
        pt.append_trading_journal.invoke({"entry": "Sold PTT due to dividend cut"})
        pt.append_trading_journal.invoke({"entry": "AAPL split announced"})

        result = pt.read_trading_journal.invoke({"keyword": "aapl"})
        data = json.loads(result)
        assert data["n_total_in_window"] == 2
        for e in data["entries"]:
            assert "aapl" in e["content"].lower()

    def test_limit_caps_entries(self, isolated_portfolio):
        pt = isolated_portfolio
        for i in range(5):
            pt.append_trading_journal.invoke({"entry": f"entry {i}"})
        result = pt.read_trading_journal.invoke({"limit": 2})
        data = json.loads(result)
        assert data["n_total_in_window"] == 5
        assert data["n_returned"] == 2

    def test_days_filter_excludes_old(self, isolated_portfolio, tmp_vault):
        pt = isolated_portfolio
        # เขียน journal entries ผสม — ใหม่และเก่า (เก่าเขียน timestamp ด้วยมือเลย)
        journal_path = tmp_vault / "20_Portfolio_Management" / "Journals_and_Reports" / "Trading_Journal.md"
        journal_path.parent.mkdir(parents=True, exist_ok=True)
        journal_path.write_text(
            "\n## [2020-01-01 10:00:00]\n\nold entry far in past\n"
            "\n## [2026-05-22 14:00:00]\n\nentry today\n",
            encoding="utf-8",
        )
        # days=30 → 2020 entry ตกขอบ, 2026 อยู่ใน window (assuming current date >= 2026-04-22)
        # Note: ทดสอบนี้พึ่งวันที่ system — ใช้ days=10000 เพื่อให้ปลอดภัย แล้วเช็คว่า old ก็ติด
        result_all = pt.read_trading_journal.invoke({"days": 10_000})
        data_all = json.loads(result_all)
        assert data_all["n_total_in_window"] == 2

        result_recent = pt.read_trading_journal.invoke({"days": 30})
        data_recent = json.loads(result_recent)
        # 2020 ต้องถูก filter ออก
        assert data_recent["n_total_in_window"] == 1
        assert "today" in data_recent["entries"][0]["content"]

    def test_edit_holding_entries_appear_in_journal(self, isolated_portfolio):
        """integration: edit_holding auto-logs → read_trading_journal เห็น"""
        pt = isolated_portfolio
        pt._manage_cash_flow_locked(500_000.0, "deposit", "THB")
        pt._execute_trade_locked(
            symbol="PTT", asset_type="Stock", action="buy",
            units=1000, price=30.0, currency="THB",
        )
        pt._edit_holding_locked(
            "PTT", units=950.0, avg_cost=None,
            accumulated_dividend_thb=None, asset_type=None,
            reason="bonus share correction",
        )
        result = pt.read_trading_journal.invoke({"keyword": "EDIT PTT"})
        data = json.loads(result)
        assert data["n_total_in_window"] >= 1


class TestGoals:
    def test_set_goal_new(self, isolated_portfolio):
        pt = isolated_portfolio
        result = pt.set_goal.invoke({
            "name": "พอร์ต 10 ล้าน",
            "goal_type": "nav_target",
            "target_amount_thb": 10_000_000.0,
            "deadline": "2031-12-31",
        })
        assert "[GOAL SET]" in result
        assert "พอร์ต 10 ล้าน" in result
        assert "10,000,000.00 THB" in result
        assert "2031-12-31" in result
        assert "total: 1" in result

    def test_set_goal_update_existing(self, isolated_portfolio):
        pt = isolated_portfolio
        pt.set_goal.invoke({
            "name": "เป้า", "goal_type": "nav_target", "target_amount_thb": 1_000_000.0,
        })
        result = pt.set_goal.invoke({
            "name": "เป้า", "goal_type": "nav_target", "target_amount_thb": 2_000_000.0,
        })
        assert "[GOAL UPD]" in result
        assert "2,000,000.00 THB" in result
        assert "total: 1" in result  # ไม่ซ้ำ ยังเป็น 1

    def test_set_goal_multiple(self, isolated_portfolio):
        pt = isolated_portfolio
        pt.set_goal.invoke({"name": "A", "goal_type": "nav_target", "target_amount_thb": 1e6})
        pt.set_goal.invoke({"name": "B", "goal_type": "cash_target", "target_amount_thb": 5e5})
        result = pt.set_goal.invoke({
            "name": "C", "goal_type": "passive_income_ytd", "target_amount_thb": 3e5,
        })
        assert "total: 3" in result

    def test_set_goal_invalid_name_raises(self, isolated_portfolio):
        pt = isolated_portfolio
        with pytest.raises(Exception, match="name"):
            pt.set_goal.invoke({
                "name": "   ", "goal_type": "nav_target", "target_amount_thb": 1e6,
            })

    def test_set_goal_invalid_target_raises(self, isolated_portfolio):
        pt = isolated_portfolio
        with pytest.raises(Exception):
            pt.set_goal.invoke({
                "name": "X", "goal_type": "nav_target", "target_amount_thb": 0.0,
            })

    def test_set_goal_invalid_deadline_raises(self, isolated_portfolio):
        pt = isolated_portfolio
        with pytest.raises(Exception, match="deadline"):
            pt.set_goal.invoke({
                "name": "X", "goal_type": "nav_target",
                "target_amount_thb": 1e6, "deadline": "31-12-2031",
            })

    def test_set_goal_preserves_created_date_on_update(self, isolated_portfolio):
        pt = isolated_portfolio
        pt.set_goal.invoke({"name": "G", "goal_type": "nav_target", "target_amount_thb": 1e6})

        # อ่าน created_date จาก Goals.md
        import frontmatter
        goals_path = pt.GOALS_PATH
        post = frontmatter.load(goals_path)
        original_date = post.metadata["goals"][0]["created_date"]

        pt.set_goal.invoke({"name": "G", "goal_type": "nav_target", "target_amount_thb": 2e6})
        post2 = frontmatter.load(goals_path)
        assert post2.metadata["goals"][0]["created_date"] == original_date

    def test_remove_goal(self, isolated_portfolio):
        pt = isolated_portfolio
        pt.set_goal.invoke({"name": "Del", "goal_type": "nav_target", "target_amount_thb": 1e6})
        result = pt.remove_goal.invoke({"name": "Del"})
        assert "[GOAL DEL]" in result
        assert "remaining: 0" in result

    def test_remove_goal_not_found_raises(self, isolated_portfolio):
        pt = isolated_portfolio
        with pytest.raises(Exception, match="ไม่พบ"):
            pt.remove_goal.invoke({"name": "ไม่มีอยู่จริง"})

    def test_get_goals_progress_nav_target(self, isolated_portfolio):
        pt = isolated_portfolio
        # เติมเงินสด → NAV = 200,000
        pt._manage_cash_flow_locked(200_000.0, "deposit", "THB")
        pt.set_goal.invoke({
            "name": "NAV เป้า", "goal_type": "nav_target", "target_amount_thb": 1_000_000.0,
        })
        result = pt.get_goals_progress.invoke({})
        data = json.loads(result)
        assert data["n_goals"] == 1
        g = data["goals"][0]
        assert g["goal_type"] == "nav_target"
        assert g["current_amount_thb"] == pytest.approx(200_000.0, rel=1e-3)
        assert g["progress_pct"] == pytest.approx(20.0, rel=1e-2)

    def test_get_goals_progress_cash_target(self, isolated_portfolio):
        pt = isolated_portfolio
        pt._manage_cash_flow_locked(300_000.0, "deposit", "THB")
        pt.set_goal.invoke({
            "name": "เงินฉุกเฉิน", "goal_type": "cash_target", "target_amount_thb": 400_000.0,
        })
        result = pt.get_goals_progress.invoke({})
        data = json.loads(result)
        g = data["goals"][0]
        assert g["goal_type"] == "cash_target"
        assert g["current_amount_thb"] == pytest.approx(300_000.0, rel=1e-3)
        assert g["progress_pct"] == pytest.approx(75.0, rel=1e-2)

    def test_get_goals_progress_passive_income(self, isolated_portfolio):
        pt = isolated_portfolio
        pt._manage_cash_flow_locked(500_000.0, "deposit", "THB")
        pt._execute_trade_locked(
            symbol="PTT", asset_type="Stock", action="buy",
            units=1000, price=30.0, currency="THB",
        )
        pt._record_income_locked("Dividend", 50_000.0, "PTT")
        pt.set_goal.invoke({
            "name": "passive 500k", "goal_type": "passive_income_ytd",
            "target_amount_thb": 500_000.0,
        })
        result = pt.get_goals_progress.invoke({})
        data = json.loads(result)
        g = data["goals"][0]
        assert g["goal_type"] == "passive_income_ytd"
        assert g["current_amount_thb"] == pytest.approx(50_000.0, rel=1e-3)
        assert g["progress_pct"] == pytest.approx(10.0, rel=1e-2)

    def test_get_goals_progress_deadline_days_left(self, isolated_portfolio):
        pt = isolated_portfolio
        pt.set_goal.invoke({
            "name": "D", "goal_type": "nav_target",
            "target_amount_thb": 1e6, "deadline": "2031-12-31",
        })
        result = pt.get_goals_progress.invoke({})
        data = json.loads(result)
        g = data["goals"][0]
        assert "deadline" in g
        assert "deadline_days_left" in g
        assert g["deadline_days_left"] > 0

    def test_get_goals_progress_empty(self, isolated_portfolio):
        pt = isolated_portfolio
        result = pt.get_goals_progress.invoke({})
        data = json.loads(result)
        assert data["n_goals"] == 0
        assert data["goals"] == []

    def test_sidecar_files_created(self, isolated_portfolio):
        """set_goal ต้องสร้าง sidecar .md ไฟล์ใน Goals/Items/"""
        pt = isolated_portfolio
        pt.set_goal.invoke({"name": "My Goal", "goal_type": "nav_target", "target_amount_thb": 1e6})
        items_dir = pt.GOALS_ITEMS_DIR
        sidecars = list(items_dir.glob("*.md"))
        assert len(sidecars) == 1
        assert sidecars[0].stem == "My_Goal"

    def test_sidecar_deleted_on_remove(self, isolated_portfolio):
        """remove_goal ต้องลบ sidecar .md ด้วย"""
        pt = isolated_portfolio
        pt.set_goal.invoke({"name": "Del Goal", "goal_type": "nav_target", "target_amount_thb": 1e6})
        pt.remove_goal.invoke({"name": "Del Goal"})
        items_dir = pt.GOALS_ITEMS_DIR
        sidecars = list(items_dir.glob("*.md"))
        assert len(sidecars) == 0
