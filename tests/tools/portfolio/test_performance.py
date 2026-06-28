"""Test portfolio_tools: recalc, execute_trade, record_income — ใช้ Vault แยกผ่าน tmp_path"""
import json

import pytest


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


class TestReadPerformanceHistory:
    def test_no_file_returns_error_msg(self, isolated_portfolio):
        pt = isolated_portfolio
        result = pt.read_performance_history.invoke({"days": 30})
        data = json.loads(result)
        assert "error" in data and "ยังไม่มี" in data["error"]

    def test_invalid_days_raises(self, isolated_portfolio):
        pt = isolated_portfolio
        result = pt.read_performance_history.invoke({"days": 0})

        assert isinstance(result, str) and result.startswith("Error:")
        assert "days" in result
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



import pytest
import tools.portfolio.performance as performance
import csv

class TestComputeTotalCost:
    def test_compute_total_cost_usd_and_thb(self, isolated_portfolio):
        pt = isolated_portfolio
        post, state = pt._load_or_init()
        # Add a USD holding
        h_usd = pt.Holding(symbol="AAPL", asset_type="Stock", units=10, avg_cost_usd=150.0, current_price_usd=200.0)
        # Add a THB holding
        h_thb = pt.Holding(symbol="PTT", asset_type="Stock", units=100, avg_cost_thb=30.0, current_price_thb=35.0)
        state.holdings.extend([h_usd, h_thb])
        
        # 10 * 150 * 36.5 = 54750
        # 100 * 30 = 3000
        # total = 57750
        cost = performance._compute_total_cost(state, current_fx=36.5)
        assert cost == pytest.approx(57750.0)

class TestPerformanceSnapshotAddons:
    def test_snapshot_refresh_prices(self, isolated_portfolio, monkeypatch):
        pt = isolated_portfolio
        import tools.portfolio.prices as prices
        monkeypatch.setattr(prices, "_refresh_prices", lambda state: {"PTT": "ok"})
        result = pt.record_performance_snapshot.func(refresh_prices=True)
        assert "[PERF]" in result

    def test_snapshot_lock_timeout(self, isolated_portfolio, monkeypatch):
        pt = isolated_portfolio
        def mock_lock(*args, **kwargs):
            from filelock import Timeout
            raise Timeout("mock")
        monkeypatch.setattr("tools.portfolio.performance._portfolio_lock.acquire", mock_lock)
        
        result = pt.record_performance_snapshot.func()

        assert isinstance(result, str) and result.startswith("Error:")
        assert "portfolio lock" in result
class TestReadPerformanceHistoryAddons:
    def test_empty_csv(self, isolated_portfolio, tmp_vault):
        pt = isolated_portfolio
        csv_path = tmp_vault / "20_Portfolio_Management" / "Journals_and_Reports" / "Performance_Log.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Date", "Total_NAV", "Total_Cost", "Unrealized_PnL", "Cash_Balance"])
            # No data rows
            
        result = pt.read_performance_history.func()
        import json
        data = json.loads(result)
        assert "error" in data
        assert "ว่างเปล่า" in data["error"]
