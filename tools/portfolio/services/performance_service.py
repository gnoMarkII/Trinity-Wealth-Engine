"""PortfolioPerformanceService — Snapshot, NAV History, Drawdown."""
import json
from datetime import datetime
from typing import Optional, List, Dict

from tools.portfolio.domain.constants import _MONEY_DP
from tools.portfolio.domain.calculations import compute_total_cost, recalc_all
from tools.portfolio.domain.validator import validate_portfolio_id
from tools.portfolio.ports.repository_port import PortfolioRepositoryPort
from tools.portfolio.ports.performance_port import PerformanceRepositoryPort
from tools.portfolio.ports.price_port import MarketPricePort


def _require_fx(state) -> float:
    fx = state.fx_rates.get("USDTHB")
    if fx is None or fx <= 0:
        raise ValueError("ไม่พบ fx_rates.USDTHB ที่ valid ใน portfolio")
    return fx


class PortfolioPerformanceService:
    """Handles Performance Snapshots and NAV History."""

    def __init__(
        self,
        repo: PortfolioRepositoryPort,
        perf_repo: PerformanceRepositoryPort,
        price_provider: MarketPricePort,
    ) -> None:
        self.repo = repo
        self.perf_repo = perf_repo
        self.price_provider = price_provider

    def record_performance_snapshot(self, refresh_prices: bool = True, portfolio_id: str = "default") -> str:
        try:
            pid = validate_portfolio_id(portfolio_id)
            # Load state (with optional price refresh)
            from tools.portfolio.domain.ledger_change import LedgerChange
            with self.repo.unit_of_work(pid) as uow:
                state = uow.load_state()
                if refresh_prices:
                    from tools.portfolio.prices import _refresh_prices
                    _refresh_prices(state)
                recalc_all(state)
                uow.commit(state, LedgerChange(kind="unchanged"))

            current_fx = _require_fx(state)
            total_nav = state.summary.total_value_thb
            unrealized = state.summary.total_unrealized_profit
            total_cost = compute_total_cost(state, current_fx)
            cash_bal = round(sum(h.market_value_thb for h in state.holdings if h.asset_type == "Cash"), _MONEY_DP)

            row = {
                "Date": datetime.now().strftime("%Y-%m-%d"),
                "Total_NAV": total_nav,
                "Total_Cost": total_cost,
                "Unrealized_PnL": unrealized,
                "Cash_Balance": cash_bal,
                "Realized_PnL_YTD": state.summary.total_realized_profit_ytd,
                "Passive_Income_YTD": state.summary.passive_income_ytd,
            }
            self.perf_repo.upsert_snapshot(pid, row)
            return f"[PERF SNAPSHOT] บันทึก snapshot วันที่ {row['Date']} (NAV: {total_nav:,.2f} THB) สำเร็จ"
        except Exception as e:
            return f"Error: {e}"

    def read_performance_history(self, days: int = 30, portfolio_id: str = "default") -> str:
        rows = self.get_structured_performance_history(days=days, portfolio_id=portfolio_id)
        return json.dumps(rows, ensure_ascii=False, indent=2)

    def get_structured_performance_history(
        self, days: Optional[int] = None, portfolio_id: str = "default"
    ) -> List[Dict]:
        return self.perf_repo.read_history(portfolio_id=portfolio_id, days=days)
