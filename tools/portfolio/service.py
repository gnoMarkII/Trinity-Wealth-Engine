"""Unified Orchestrator Facade for Portfolio Management (Hexagonal Architecture).

This module exposes the unified `PortfolioService` facade, delegating domain operations
to specialized sub-services under `tools.portfolio.services`.
"""
from typing import Optional, List, Dict, Tuple, Literal, Union

from tools.portfolio.domain.models import (
    PortfolioState,
    PortfolioMeta,
    AllocationTarget,
    WatchlistState,
)
from tools.portfolio.ports.repository_port import PortfolioRepositoryPort
from tools.portfolio.ports.watchlist_port import WatchlistRepositoryPort
from tools.portfolio.ports.goals_port import GoalsRepositoryPort
from tools.portfolio.ports.performance_port import PerformanceRepositoryPort
from tools.portfolio.ports.journal_port import TradeJournalPort
from tools.portfolio.ports.price_port import MarketPricePort

def _require_fx(state: PortfolioState) -> float:
    fx = state.fx_rates.get("USDTHB")
    if fx is None or fx <= 0:
        raise ValueError("ไม่พบ fx_rates.USDTHB ที่ valid ใน portfolio")
    return fx


class PortfolioService:
    """Unified Orchestrator Facade implementing Hexagonal Architecture for Portfolio Management."""

    def __init__(
        self,
        repo: Optional[PortfolioRepositoryPort] = None,
        watchlist_repo: Optional[WatchlistRepositoryPort] = None,
        goals_repo: Optional[GoalsRepositoryPort] = None,
        perf_repo: Optional[PerformanceRepositoryPort] = None,
        journal_provider: Optional[TradeJournalPort] = None,
        price_provider: Optional[MarketPricePort] = None,
    ):
        """Initialize self.  See help(type(self)) for accurate signature."""
        from tools.portfolio.bootstrap import (
            resolve_legacy_portfolio_dependencies,
        )

        self._bind_dependencies(resolve_legacy_portfolio_dependencies(
            repo=repo,
            watchlist_repo=watchlist_repo,
            goals_repo=goals_repo,
            perf_repo=perf_repo,
            journal_provider=journal_provider,
            price_provider=price_provider,
        ))

    def _bind_dependencies(self, deps) -> None:
        """Bind an already-composed dependency bundle to the facade.

        Kept private so the historical constructor/signature remains frozen;
        the API composition root can inject the complete bundle (including the
        DividendHistoryPort) without making the facade construct adapters.
        """
        from tools.portfolio.bootstrap import build_portfolio_application

        self._deps = deps
        self.repo = self._deps.repo
        self.watchlist_repo = self._deps.watchlist_repo
        self.goals_repo = self._deps.goals_repo
        self.perf_repo = self._deps.perf_repo
        self.journal_provider = self._deps.journal_provider
        self.price_provider = self._deps.price_provider
        self._app = build_portfolio_application(self._deps)

        # Initialize Sub-services
        self._state_service = self._app.state_service
        self._trading_service = self._app.trading_service
        self._cash_flow_service = self._app.cash_flow_service
        self._ledger_service = self._app.ledger_service
        self._goal_service = self._app.goal_service
        self._perf_service = self._app.performance_service
        self._watchlist_service = self._app.watchlist_service
        self._journal_service = self._app.journal_service
        self._batch_import_service = self._app.batch_import_service
        self._dime_sync_service = self._app.dime_sync_service
        self._wealthx_sync_service = self._app.wealthx_sync_service
        self._scbam_sync_service = self._app.scbam_sync_service

    # =========================================================================
    # 0. Portfolio Lifecycle & Management
    # =========================================================================

    def list_portfolios(self) -> List[PortfolioMeta]:
        return self._state_service.list_portfolios()

    def create_portfolio(self, name: str, portfolio_id: Optional[str] = None) -> PortfolioMeta:
        return self._state_service.create_portfolio(name=name, portfolio_id=portfolio_id)

    def delete_portfolio(self, portfolio_id: str) -> None:
        self._state_service.delete_portfolio(portfolio_id=portfolio_id)

    def update_portfolio_name(self, portfolio_id: str, name: str) -> PortfolioMeta:
        return self._state_service.update_portfolio_name(portfolio_id=portfolio_id, name=name)

    # =========================================================================
    # 1. Core State Queries & Buckets
    # =========================================================================

    def get_portfolio_state(self, refresh_prices: bool = True, portfolio_id: str = "default") -> str:
        return self._state_service.get_portfolio_state(refresh_prices=refresh_prices, portfolio_id=portfolio_id)

    def get_structured_portfolio_state(
        self, refresh_prices: bool = False, fetch_fundamentals: bool = False, portfolio_id: str = "default"
    ) -> PortfolioState:
        return self._state_service.get_structured_portfolio_state(
            refresh_prices=refresh_prices, fetch_fundamentals=fetch_fundamentals, portfolio_id=portfolio_id
        )

    def get_structured_bucket_allocation(
        self, state: Optional[PortfolioState] = None, portfolio_id: str = "default"
    ) -> Tuple[List[Dict], Optional[str]]:
        return self._state_service.get_structured_bucket_allocation(state=state, portfolio_id=portfolio_id)

    def structured_assign_holding_bucket(
        self, symbol: str, bucket_id: Optional[str], portfolio_id: str = "default"
    ) -> PortfolioState:
        return self._state_service.structured_assign_holding_bucket(symbol=symbol, bucket_id=bucket_id, portfolio_id=portfolio_id)

    def structured_batch_assign_holding_buckets(
        self, symbols: List[str], bucket_id: Optional[str], portfolio_id: str = "default"
    ) -> PortfolioState:
        return self._state_service.structured_batch_assign_holding_buckets(
            symbols=symbols, bucket_id=bucket_id, portfolio_id=portfolio_id
        )

    def structured_batch_remove_holdings(
        self, symbols: List[str], portfolio_id: str = "default"
    ) -> PortfolioState:
        return self._trading_service.structured_batch_remove_holdings(symbols=symbols, portfolio_id=portfolio_id)

    def structured_reset_clean_slate(self, portfolio_id: str = "default") -> PortfolioState:
        return self._state_service.structured_reset_clean_slate(portfolio_id=portfolio_id)

    def structured_upsert_allocation_targets(
        self, targets: List[AllocationTarget], portfolio_id: str = "default"
    ) -> PortfolioState:
        return self._state_service.structured_upsert_allocation_targets(targets=targets, portfolio_id=portfolio_id)

    def compute_allocation_breakdown(
        self, group_by: Literal["asset_type", "currency"] = "asset_type", portfolio_id: str = "default"
    ) -> str:
        return self._state_service.compute_allocation_breakdown(group_by=group_by, portfolio_id=portfolio_id)

    # =========================================================================
    # 2. Trade & Holdings Operations
    # =========================================================================

    def execute_trade(
        self,
        symbol: str,
        asset_type: str,
        action: Literal["buy", "sell"],
        units: float,
        price: float,
        currency: Literal["THB", "USD"] = "THB",
        notes: str = "",
        portfolio_id: str = "default",
    ) -> str:
        return self._trading_service.execute_trade(
            symbol=symbol,
            asset_type=asset_type,
            action=action,
            units=units,
            price=price,
            currency=currency,
            notes=notes,
            portfolio_id=portfolio_id,
        )

    def structured_execute_trade(
        self,
        symbol: str,
        asset_type: str,
        action: Literal["buy", "sell"],
        units: float,
        price: float,
        currency: Literal["THB", "USD"] = "THB",
        exchange_rate: Optional[float] = None,
        date: Optional[str] = None,
        notes: str = "",
        bucket_id: Optional[str] = None,
        portfolio_id: str = "default",
    ) -> PortfolioState:
        return self._trading_service.structured_execute_trade(
            symbol=symbol,
            asset_type=asset_type,
            action=action,
            units=units,
            price=price,
            currency=currency,
            exchange_rate=exchange_rate,
            date=date,
            notes=notes,
            bucket_id=bucket_id,
            portfolio_id=portfolio_id,
        )

    def _execute_trade_internal(
        self,
        symbol: str,
        asset_type: str,
        action: Literal["buy", "sell"],
        units: float,
        price: float,
        currency: Literal["THB", "USD"] = "THB",
        exchange_rate: Optional[float] = None,
        date: Optional[str] = None,
        notes: str = "",
        bucket_id: Optional[str] = None,
        portfolio_id: str = "default",
    ) -> Tuple[str, PortfolioState]:
        return self._trading_service._execute_trade_internal(
            symbol=symbol,
            asset_type=asset_type,
            action=action,
            units=units,
            price=price,
            currency=currency,
            exchange_rate=exchange_rate,
            date=date,
            notes=notes,
            bucket_id=bucket_id,
            portfolio_id=portfolio_id,
        )

    def batch_import_holdings(
        self,
        assets_list: Union[List[Dict], str],
        mode: Literal["merge", "overwrite"] = "merge",
        reset_cash_usd: bool = False,
        portfolio_id: str = "default",
    ) -> str:
        return self._trading_service.batch_import_holdings(
            assets_list=assets_list,
            mode=mode,
            reset_cash_usd=reset_cash_usd,
            portfolio_id=portfolio_id,
        )

    def edit_holding(
        self,
        symbol: str,
        units: Optional[float] = None,
        avg_cost: Optional[float] = None,
        accumulated_dividend_thb: Optional[float] = None,
        asset_type: Optional[str] = None,
        reason: str = "",
        portfolio_id: str = "default",
    ) -> str:
        return self._trading_service.edit_holding(
            symbol=symbol,
            units=units,
            avg_cost=avg_cost,
            accumulated_dividend_thb=accumulated_dividend_thb,
            asset_type=asset_type,
            reason=reason,
            portfolio_id=portfolio_id,
        )

    def structured_edit_holding(
        self,
        symbol: str,
        units: Optional[float] = None,
        avg_cost: Optional[float] = None,
        accumulated_dividend_thb: Optional[float] = None,
        asset_type: Optional[str] = None,
        reason: str = "",
        bucket_id: Optional[str] = None,
        portfolio_id: str = "default",
    ) -> PortfolioState:
        return self._trading_service.structured_edit_holding(
            symbol=symbol,
            units=units,
            avg_cost=avg_cost,
            accumulated_dividend_thb=accumulated_dividend_thb,
            asset_type=asset_type,
            reason=reason,
            bucket_id=bucket_id,
            portfolio_id=portfolio_id,
        )

    def _edit_holding_internal(
        self,
        symbol: str,
        units: Optional[float] = None,
        avg_cost: Optional[float] = None,
        accumulated_dividend_thb: Optional[float] = None,
        asset_type: Optional[str] = None,
        reason: str = "",
        bucket_id: Optional[str] = None,
        portfolio_id: str = "default",
    ) -> Tuple[str, PortfolioState]:
        return self._trading_service._edit_holding_internal(
            symbol=symbol,
            units=units,
            avg_cost=avg_cost,
            accumulated_dividend_thb=accumulated_dividend_thb,
            asset_type=asset_type,
            reason=reason,
            bucket_id=bucket_id,
            portfolio_id=portfolio_id,
        )

    def structured_remove_holding(self, symbol: str, portfolio_id: str = "default") -> PortfolioState:
        return self._trading_service.structured_remove_holding(symbol=symbol, portfolio_id=portfolio_id)

    # =========================================================================
    # 3. Cash Flow & Income Operations
    # =========================================================================

    def manage_cash_flow(
        self,
        amount: float,
        action: Literal["deposit", "withdraw"],
        currency: Literal["THB", "USD"] = "THB",
        portfolio_id: str = "default",
    ) -> str:
        return self._cash_flow_service.manage_cash_flow(
            amount=amount, action=action, currency=currency, portfolio_id=portfolio_id
        )

    def structured_manage_cash_flow(
        self,
        amount: float,
        action: Literal["deposit", "withdraw"],
        currency: Literal["THB", "USD"] = "THB",
        exchange_rate: Optional[float] = None,
        date: Optional[str] = None,
        notes: str = "",
        portfolio_id: str = "default",
    ) -> PortfolioState:
        return self._cash_flow_service.structured_manage_cash_flow(
            amount=amount,
            action=action,
            currency=currency,
            exchange_rate=exchange_rate,
            date=date,
            notes=notes,
            portfolio_id=portfolio_id,
        )

    def _manage_cash_flow_internal(
        self,
        amount: float,
        action: Literal["deposit", "withdraw"],
        currency: Literal["THB", "USD"] = "THB",
        exchange_rate: Optional[float] = None,
        date: Optional[str] = None,
        notes: str = "",
        portfolio_id: str = "default",
    ) -> Tuple[str, PortfolioState]:
        return self._cash_flow_service._manage_cash_flow_internal(
            amount=amount,
            action=action,
            currency=currency,
            exchange_rate=exchange_rate,
            date=date,
            notes=notes,
            portfolio_id=portfolio_id,
        )

    def record_income(
        self,
        income_type: Literal["Dividend", "Interest", "Rental", "Other"],
        amount_thb: float,
        source_symbol: Optional[str] = None,
        portfolio_id: str = "default",
    ) -> str:
        return self._cash_flow_service.record_income(
            income_type=income_type, amount_thb=amount_thb, source_symbol=source_symbol, portfolio_id=portfolio_id
        )

    def structured_record_income(
        self,
        income_type: Literal["Dividend", "Interest", "Rental", "Other"],
        amount_thb: float,
        source_symbol: Optional[str] = None,
        date: Optional[str] = None,
        notes: str = "",
        portfolio_id: str = "default",
    ) -> PortfolioState:
        return self._cash_flow_service.structured_record_income(
            income_type=income_type,
            amount_thb=amount_thb,
            source_symbol=source_symbol,
            date=date,
            notes=notes,
            portfolio_id=portfolio_id,
        )

    def _record_income_internal(
        self,
        income_type: Literal["Dividend", "Interest", "Rental", "Other"],
        amount_thb: float,
        source_symbol: Optional[str] = None,
        date: Optional[str] = None,
        notes: str = "",
        portfolio_id: str = "default",
    ) -> Tuple[str, PortfolioState]:
        return self._cash_flow_service._record_income_internal(
            income_type=income_type,
            amount_thb=amount_thb,
            source_symbol=source_symbol,
            date=date,
            notes=notes,
            portfolio_id=portfolio_id,
        )

    def update_fx_rate(self, rate: Optional[float] = None, portfolio_id: str = "default") -> str:
        return self._trading_service.update_fx_rate(rate=rate, portfolio_id=portfolio_id)

    def sync_market_prices(self, portfolio_id: str = "default") -> str:
        return self._trading_service.sync_market_prices(portfolio_id=portfolio_id)

    # =========================================================================
    # 4. Ledger Operations
    # =========================================================================

    def get_structured_trades_log(
        self, portfolio_id: str = "default", symbol: Optional[str] = None
    ) -> List[Dict]:
        return self._ledger_service.get_structured_trades_log(portfolio_id=portfolio_id, symbol=symbol)

    def update_trade_note(self, tx_id: str, notes: str, portfolio_id: str = "default") -> Dict:
        return self._ledger_service.update_trade_note(tx_id=tx_id, notes=notes, portfolio_id=portfolio_id)

    def edit_transaction(
        self,
        tx_id: str,
        timestamp: Optional[str] = None,
        units: Optional[float] = None,
        price: Optional[float] = None,
        fx_rate: Optional[float] = None,
        notes: Optional[str] = None,
        adjust_cash: bool = True,
        portfolio_id: str = "default",
    ) -> PortfolioState:
        return self._ledger_service.edit_transaction(
            tx_id=tx_id,
            timestamp=timestamp,
            units=units,
            price=price,
            fx_rate=fx_rate,
            notes=notes,
            adjust_cash=adjust_cash,
            portfolio_id=portfolio_id,
        )

    def void_transaction(
        self, tx_id: str, portfolio_id: str = "default", adjust_cash: Optional[bool] = None
    ) -> PortfolioState:
        return self._ledger_service.void_transaction(
            tx_id=tx_id, portfolio_id=portfolio_id, adjust_cash=adjust_cash
        )

    def delete_transaction(
        self, tx_id: str, adjust_cash: bool = True, portfolio_id: str = "default"
    ) -> PortfolioState:
        return self.void_transaction(tx_id=tx_id, portfolio_id=portfolio_id)

    # =========================================================================
    # 5. Watchlist Operations
    # =========================================================================

    def add_to_watchlist(
        self, symbol: str, asset_type: str, target_price: Optional[float] = None, notes: str = "", portfolio_id: str = "default"
    ) -> str:
        return self._watchlist_service.add_to_watchlist(
            symbol=symbol, asset_type=asset_type, target_price=target_price, notes=notes, portfolio_id=portfolio_id
        )

    def remove_from_watchlist(self, symbol: str, portfolio_id: str = "default") -> str:
        return self._watchlist_service.remove_from_watchlist(symbol=symbol, portfolio_id=portfolio_id)

    def read_watchlist(self, portfolio_id: str = "default") -> str:
        return self._watchlist_service.read_watchlist(portfolio_id=portfolio_id)

    def get_structured_watchlist(self, portfolio_id: str = "default") -> WatchlistState:
        return self._watchlist_service.get_structured_watchlist(portfolio_id=portfolio_id)

    def structured_upsert_watchlist_item(
        self, symbol: str, asset_type: str, target_price: Optional[float] = None, notes: str = "", portfolio_id: str = "default"
    ) -> WatchlistState:
        return self._watchlist_service.structured_upsert_watchlist_item(
            symbol=symbol, asset_type=asset_type, target_price=target_price, notes=notes, portfolio_id=portfolio_id
        )

    def structured_remove_watchlist_item(self, symbol: str, portfolio_id: str = "default") -> WatchlistState:
        return self._watchlist_service.structured_remove_watchlist_item(symbol=symbol, portfolio_id=portfolio_id)

    # =========================================================================
    # 6. Goals Operations
    # =========================================================================

    def set_goal(
        self,
        name: str,
        goal_type: Literal["nav_target", "cash_target", "passive_income_ytd", "bucket_target"],
        target_amount_thb: float,
        deadline: Optional[str] = None,
        years_from_now: Optional[int] = None,
        notes: Optional[str] = None,
        portfolio_id: str = "default",
        bucket_id: Optional[str] = None,
    ) -> str:
        return self._goal_service.set_goal(
            name=name,
            goal_type=goal_type,
            target_amount_thb=target_amount_thb,
            deadline=deadline,
            years_from_now=years_from_now,
            notes=notes,
            portfolio_id=portfolio_id,
            bucket_id=bucket_id,
        )

    def remove_goal(self, name: str) -> str:
        return self._goal_service.remove_goal(name=name)

    def get_goals_progress(self, portfolio_id: str = "default") -> str:
        return self._goal_service.get_goals_progress(portfolio_id=portfolio_id)

    def get_structured_goals(self, portfolio_id: Optional[str] = None) -> List[Dict]:
        return self._goal_service.get_structured_goals(portfolio_id=portfolio_id)

    def structured_upsert_goal(
        self,
        name: str,
        goal_type: Literal["nav_target", "cash_target", "passive_income_ytd", "bucket_target"],
        target_amount_thb: float,
        deadline: Optional[str] = None,
        years_from_now: Optional[int] = None,
        notes: Optional[str] = None,
        portfolio_id: str = "default",
        bucket_id: Optional[str] = None,
    ) -> List[Dict]:
        return self._goal_service.structured_upsert_goal(
            name=name,
            goal_type=goal_type,
            target_amount_thb=target_amount_thb,
            deadline=deadline,
            years_from_now=years_from_now,
            notes=notes,
            portfolio_id=portfolio_id,
            bucket_id=bucket_id,
        )

    def structured_remove_goal(self, name: str, portfolio_id: Optional[str] = None) -> List[Dict]:
        return self._goal_service.structured_remove_goal(name=name, portfolio_id=portfolio_id)

    # =========================================================================
    # 7. Journal Operations
    # =========================================================================

    def append_trading_journal(self, entry: str, portfolio_id: str = "default") -> str:
        return self._journal_service.append_trading_journal(entry=entry, portfolio_id=portfolio_id)

    def read_trading_journal(
        self, days: int = 30, keyword: Optional[str] = None, limit: int = 20, portfolio_id: str = "default"
    ) -> str:
        return self._journal_service.read_trading_journal(
            days=days, keyword=keyword, limit=limit, portfolio_id=portfolio_id
        )

    def get_structured_journal(
        self, days: Optional[int] = 365, keyword: Optional[str] = None, limit: int = 100, portfolio_id: str = "default"
    ) -> List[Dict]:
        return self._journal_service.get_structured_journal(
            days=days, keyword=keyword, limit=limit, portfolio_id=portfolio_id
        )

    def structured_append_journal(self, entry: str, portfolio_id: str = "default") -> List[Dict]:
        return self._journal_service.structured_append_journal(entry=entry, portfolio_id=portfolio_id)

    # =========================================================================
    # 8. Performance Operations
    # =========================================================================

    def record_performance_snapshot(self, refresh_prices: bool = True, portfolio_id: str = "default") -> str:
        return self._perf_service.record_performance_snapshot(
            refresh_prices=refresh_prices, portfolio_id=portfolio_id
        )

    def read_performance_history(self, days: int = 30, portfolio_id: str = "default") -> str:
        return self._perf_service.read_performance_history(days=days, portfolio_id=portfolio_id)

    def get_structured_performance_history(
        self, days: Optional[int] = None, portfolio_id: str = "default"
    ) -> List[Dict]:
        return self._perf_service.get_structured_performance_history(days=days, portfolio_id=portfolio_id)

    # =========================================================================
    # 9. Prices & Dividends Operations
    # =========================================================================

    def fetch_fx_rate(
        self, date_str: Optional[str] = None, fallback_rate: Optional[float] = None
    ) -> Tuple[float, Literal["historical", "live", "fallback"]]:
        return self._cash_flow_service.fetch_fx_rate(date_str=date_str, fallback_rate=fallback_rate)

    def fetch_latest_price(self, symbol: str, currency: Literal["THB", "USD"] = "THB") -> Optional[float]:
        return self._cash_flow_service.fetch_latest_price(symbol=symbol, currency=currency)

    def sync_dividends_from_history(self, portfolio_id: str = "default") -> Dict:
        """Sync dividend payouts and history for holdings."""
        return self._cash_flow_service.sync_dividends_from_history(portfolio_id=portfolio_id)
