import pytest
from typing import Optional, List, Dict, Tuple, Literal
from tools.portfolio.domain.constants import CASH_THB_SYMBOL, CASH_USD_SYMBOL
from tools.portfolio.domain.models import (
    PortfolioState,
    Holding,
    Summary,
    PortfolioMeta,
    AllocationTarget,
    WatchlistState,
    GoalsState,
    GoalItem,
    _now_iso,
    default_allocation_targets,
)
from tools.portfolio.domain.ledger_change import LedgerChange
from tools.portfolio.ports.repository_port import PortfolioRepositoryPort, PortfolioUnitOfWork
from tools.portfolio.ports.watchlist_port import WatchlistRepositoryPort
from tools.portfolio.ports.goals_port import GoalsRepositoryPort
from tools.portfolio.ports.performance_port import PerformanceRepositoryPort
from tools.portfolio.ports.journal_port import TradeJournalPort
from tools.portfolio.ports.price_port import MarketPricePort
from tools.portfolio.service import PortfolioService


class InMemoryPortfolioUnitOfWork(PortfolioUnitOfWork):
    def __init__(self, repo: "InMemoryPortfolioRepository", portfolio_id: str):
        self.repo = repo
        self.portfolio_id = portfolio_id

    def __enter__(self) -> "InMemoryPortfolioUnitOfWork":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> Optional[bool]:
        return None

    def load_state(self) -> PortfolioState:
        return self.repo.load_state(self.portfolio_id)

    def commit(self, state: PortfolioState, ledger_change: Optional[LedgerChange] = None) -> None:
        self.repo._states[self.portfolio_id] = state.model_copy(deep=True)
        if ledger_change and ledger_change.kind == "append" and ledger_change.row:
            self.repo._trades.setdefault(self.portfolio_id, []).append(ledger_change.row)
        elif ledger_change and ledger_change.kind == "replace_all" and ledger_change.rows is not None:
            self.repo._trades[self.portfolio_id] = [dict(r) for r in ledger_change.rows]

    def rollback(self) -> None:
        pass


class InMemoryPortfolioRepository(PortfolioRepositoryPort):
    def __init__(self):
        self._states: Dict[str, PortfolioState] = {}
        self._trades: Dict[str, List[Dict]] = {}
        self._portfolios: List[PortfolioMeta] = [
            PortfolioMeta(id="default", name="พอร์ตลงทุนหลัก", is_default=True)
        ]

    def unit_of_work(self, portfolio_id: str = "default") -> PortfolioUnitOfWork:
        return InMemoryPortfolioUnitOfWork(self, portfolio_id)

    def load_state(self, portfolio_id: str = "default") -> PortfolioState:
        if portfolio_id not in self._states:
            self._states[portfolio_id] = PortfolioState(
                last_updated=_now_iso(),
                summary=Summary(),
                fx_rates={"USDTHB": 35.0},
                holdings=[
                    Holding(symbol=CASH_THB_SYMBOL, asset_type="Cash", units=0.0),
                    Holding(symbol=CASH_USD_SYMBOL, asset_type="Cash", units=0.0),
                ],
            )
        return self._states[portfolio_id].model_copy(deep=True)

    def read_trade_log(self, portfolio_id: str = "default", symbol: Optional[str] = None) -> List[Dict]:
        rows = self._trades.get(portfolio_id, [])
        if symbol:
            return [r for r in rows if r.get("Symbol") == symbol]
        return list(rows)

    def list_portfolios(self) -> List[PortfolioMeta]:
        return list(self._portfolios)

    def create_portfolio(self, name: str, portfolio_id: Optional[str] = None) -> PortfolioMeta:
        pid = portfolio_id or name.lower().replace(" ", "_")
        meta = PortfolioMeta(id=pid, name=name, is_default=False)
        self._portfolios.append(meta)
        return meta

    def delete_portfolio(self, portfolio_id: str) -> None:
        self._portfolios = [p for p in self._portfolios if p.id != portfolio_id]

    def rename_portfolio(self, portfolio_id: str, new_name: str) -> PortfolioMeta:
        meta = next(p for p in self._portfolios if p.id == portfolio_id)
        meta.name = new_name
        return meta

    def portfolio_exists(self, portfolio_id: str) -> bool:
        if portfolio_id == "default":
            return True
        return any(p.id == portfolio_id for p in self._portfolios)


class InMemoryWatchlistRepository(WatchlistRepositoryPort):
    def __init__(self):
        self._watchlists: Dict[str, WatchlistState] = {}

    def load_watchlist(self, portfolio_id: str = "default") -> WatchlistState:
        if portfolio_id not in self._watchlists:
            self._watchlists[portfolio_id] = WatchlistState(last_updated=_now_iso(), items=[])
        return self._watchlists[portfolio_id].model_copy(deep=True)

    def save_watchlist(self, state: WatchlistState, portfolio_id: str = "default") -> None:
        self._watchlists[portfolio_id] = state.model_copy(deep=True)


class InMemoryGoalsRepository(GoalsRepositoryPort):
    def __init__(self):
        self._state = GoalsState(last_updated=_now_iso(), goals=[])

    def load_goals(self, portfolio_id: Optional[str] = None) -> GoalsState:
        return self._state.model_copy(deep=True)

    def save_goals(self, state: GoalsState) -> None:
        self._state = state.model_copy(deep=True)


class InMemoryPerformanceRepository(PerformanceRepositoryPort):
    def __init__(self):
        self._history: Dict[str, List[Dict]] = {}

    def upsert_snapshot(self, portfolio_id: str, row: Dict) -> None:
        date = row.get("Date")
        rows = self._history.setdefault(portfolio_id, [])
        for i, r in enumerate(rows):
            if r.get("Date") == date:
                rows[i] = dict(row)
                return
        rows.append(dict(row))

    def read_history(self, portfolio_id: str = "default", days: Optional[int] = None) -> List[Dict]:
        return list(self._history.get(portfolio_id, []))


class InMemoryJournalAdapter(TradeJournalPort):
    def __init__(self):
        self._entries: Dict[str, List[Dict]] = {}

    def append_journal(self, entry: str, portfolio_id: str = "default") -> List[Dict]:
        item = {"timestamp": _now_iso(), "content": entry}
        self._entries.setdefault(portfolio_id, []).append(item)
        return list(self._entries[portfolio_id])

    def read_journal(self, days: Optional[int] = 365, keyword: Optional[str] = None, limit: int = 100, portfolio_id: str = "default") -> List[Dict]:
        return list(self._entries.get(portfolio_id, []))


class MockPriceAdapter(MarketPricePort):
    def __init__(self):
        self.prices = {"PTT": 35.0, "AAPL": 220.0}

    def fetch_price(self, symbol: str, currency: Literal["THB", "USD"]) -> Optional[float]:
        return self.prices.get(symbol, 100.0)

    def fetch_fx_rate(self, date_str: Optional[str] = None, fallback_rate: Optional[float] = None) -> Tuple[float, Literal["historical", "live", "fallback"]]:
        return 35.0, "live"

    def refresh_portfolio_prices(self, state: PortfolioState) -> Dict[str, str]:
        for h in state.holdings:
            if h.symbol in self.prices:
                p = self.prices[h.symbol]
                if h.avg_cost_usd is not None:
                    h.current_price_usd = p
                    h.current_price_thb = p * 35.0
                else:
                    h.current_price_thb = p
                    h.current_price_usd = p / 35.0
        return {k: f"{v}" for k, v in self.prices.items()}

    def fetch_fundamentals(self, state: PortfolioState, force: bool = False) -> Dict[str, str]:
        return {}


def test_full_portfolio_flow_with_di():
    repo = InMemoryPortfolioRepository()
    watchlist_repo = InMemoryWatchlistRepository()
    goals_repo = InMemoryGoalsRepository()
    perf_repo = InMemoryPerformanceRepository()
    journal_provider = InMemoryJournalAdapter()
    price_provider = MockPriceAdapter()

    svc = PortfolioService(
        repo=repo,
        watchlist_repo=watchlist_repo,
        goals_repo=goals_repo,
        perf_repo=perf_repo,
        journal_provider=journal_provider,
        price_provider=price_provider,
    )

    # 1. Deposit Cash
    svc.manage_cash_flow(amount=100000.0, action="deposit", currency="THB")
    svc.manage_cash_flow(amount=2000.0, action="deposit", currency="USD")

    state = svc.get_structured_portfolio_state()
    cash_thb = next(h for h in state.holdings if h.symbol == CASH_THB_SYMBOL)
    cash_usd = next(h for h in state.holdings if h.symbol == CASH_USD_SYMBOL)
    assert cash_thb.units == 100000.0
    assert cash_usd.units == 2000.0

    # 2. Buy PTT: 1000 @ 30 THB -> Cost 30,000 THB
    svc.structured_execute_trade(symbol="PTT", asset_type="Stock", action="buy", units=1000.0, price=30.0, currency="THB")
    state = svc.get_structured_portfolio_state()
    ptt = next(h for h in state.holdings if h.symbol == "PTT")
    assert ptt.units == 1000.0
    assert ptt.avg_cost_thb == 30.0
    assert next(h for h in state.holdings if h.symbol == CASH_THB_SYMBOL).units == 70000.0

    # 3. Buy AAPL: 10 @ 150 USD -> Cost 1500 USD
    svc.structured_execute_trade(symbol="AAPL", asset_type="Stock", action="buy", units=10.0, price=150.0, currency="USD")
    state = svc.get_structured_portfolio_state()
    aapl = next(h for h in state.holdings if h.symbol == "AAPL")
    assert aapl.units == 10.0
    assert aapl.avg_cost_usd == 150.0
    assert next(h for h in state.holdings if h.symbol == CASH_USD_SYMBOL).units == 500.0

    # 4. Sell PTT: 500 @ 40 THB -> Realized profit (40-30)*500 = 5000 THB
    svc.structured_execute_trade(symbol="PTT", asset_type="Stock", action="sell", units=500.0, price=40.0, currency="THB")
    state = svc.get_structured_portfolio_state()
    ptt = next(h for h in state.holdings if h.symbol == "PTT")
    assert ptt.units == 500.0
    assert state.summary.total_realized_profit_ytd == 5000.0
    assert next(h for h in state.holdings if h.symbol == CASH_THB_SYMBOL).units == 90000.0

    # 5. Check Trade Log
    trades = svc.get_structured_trades_log()
    assert len(trades) == 3

    # 6. Watchlist Operations
    svc.add_to_watchlist("NVDA", "Stock", target_price=100.0, notes="Wait for dip")
    w = svc.get_structured_watchlist()
    assert len(w.items) == 1
    assert w.items[0].symbol == "NVDA"

    svc.remove_from_watchlist("NVDA")
    assert len(svc.get_structured_watchlist().items) == 0

    # 7. Goals Operations
    svc.set_goal(name="Emergency Fund", goal_type="cash_target", target_amount_thb=100000.0)
    goals = svc.get_structured_goals()
    assert len(goals) == 1
    assert goals[0]["name"] == "Emergency Fund"

    # 8. Journal Operations
    svc.append_trading_journal("Good trade on PTT")
    entries = svc.get_structured_journal()
    assert len(entries) == 1
    assert "Good trade on PTT" in entries[0]["content"]

    # 9. Performance Snapshot
    msg = svc.record_performance_snapshot()
    assert "[PERF SNAPSHOT]" in msg
    hist = svc.get_structured_performance_history()
    assert len(hist) == 1
