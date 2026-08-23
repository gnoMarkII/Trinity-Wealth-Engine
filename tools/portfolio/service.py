import json
import uuid
import time
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Literal, Union

from core.logger import get_logger
from tools.portfolio.domain.constants import (
    CASH_THB_SYMBOL,
    CASH_USD_SYMBOL,
    _CASH_SYMBOLS,
    _FLOAT_EPS,
    _MONEY_DP,
    _COST_DP,
    _PCT_DP,
    _EDITABLE_HOLDING_FIELDS,
)
from tools.portfolio.domain.models import (
    PortfolioState,
    Holding,
    Summary,
    PortfolioMeta,
    AllocationTarget,
    WatchlistState,
    WatchlistItem,
    GoalsState,
    GoalItem,
    _now_iso,
    default_allocation_targets,
)
from tools.portfolio.domain.ledger_change import LedgerChange
from tools.portfolio.domain.errors import (
    InvalidTradeError,
    InsufficientCashError,
    HoldingNotFoundError,
    PortfolioNotFoundError,
)
from tools.portfolio.domain.calculations import (
    calc_weighted_avg_cost,
    calc_realized_pnl,
    calc_holding_currency,
    recalc_holding,
    compute_total_cost,
    recalc_summary,
    recalc_fundamentals_derived,
    recalc_all,
    compute_allocation_breakdown,
    compute_target_allocation_variance,
    _replay_symbol_trades,
)
from tools.portfolio.adapters.markdown.paths import get_trades_log_filepath
from tools.portfolio.domain.validator import (
    validate_portfolio_id,
    validate_trade_request,
    validate_cash_availability,
    validate_holding_for_sell,
    validate_cash_flow_request,
)
from tools.portfolio.ports.repository_port import PortfolioRepositoryPort
from tools.portfolio.ports.watchlist_port import WatchlistRepositoryPort
from tools.portfolio.ports.goals_port import GoalsRepositoryPort
from tools.portfolio.ports.performance_port import PerformanceRepositoryPort
from tools.portfolio.ports.journal_port import TradeJournalPort
from tools.portfolio.ports.price_port import MarketPricePort

log = get_logger(__name__)


def _find_holding(state: PortfolioState, symbol: str) -> Optional[Holding]:
    return next((h for h in state.holdings if h.symbol == symbol), None)


def _require_cash(state: PortfolioState, currency: Literal["THB", "USD"] = "THB") -> Holding:
    sym = CASH_THB_SYMBOL if currency == "THB" else CASH_USD_SYMBOL
    cash = _find_holding(state, sym)
    if cash is None:
        cash = Holding(symbol=sym, asset_type="Cash", units=0.0, market_value_thb=0.0)
        state.holdings.append(cash)
    return cash


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
        if repo is None:
            from tools.portfolio.adapters.markdown.repository_adapter import MarkdownVaultRepositoryAdapter
            from tools.portfolio.adapters.sqlite_mirror_decorator import SqliteMirroredPortfolioRepository
            md_repo = MarkdownVaultRepositoryAdapter()
            self.repo = SqliteMirroredPortfolioRepository(underlying_repo=md_repo)
        else:
            self.repo = repo

        if watchlist_repo is None:
            from tools.portfolio.adapters.markdown.watchlist_adapter import MarkdownWatchlistAdapter
            self.watchlist_repo = MarkdownWatchlistAdapter()
        else:
            self.watchlist_repo = watchlist_repo

        if goals_repo is None:
            from tools.portfolio.adapters.markdown.goals_adapter import MarkdownGoalsAdapter
            self.goals_repo = MarkdownGoalsAdapter()
        else:
            self.goals_repo = goals_repo

        if perf_repo is None:
            from tools.portfolio.adapters.markdown.performance_adapter import MarkdownPerformanceAdapter
            self.perf_repo = MarkdownPerformanceAdapter()
        else:
            self.perf_repo = perf_repo

        if journal_provider is None:
            from tools.portfolio.adapters.markdown.journal_vault_adapter import JournalVaultAdapter
            self.journal_provider = JournalVaultAdapter()
        else:
            self.journal_provider = journal_provider

        if price_provider is None:
            from tools.portfolio.adapters.price_yfinance_adapter import PriceYFinanceAdapter
            self.price_provider = PriceYFinanceAdapter()
        else:
            self.price_provider = price_provider

    # =========================================================================
    # 0. Portfolio Lifecycle & Management
    # =========================================================================

    def list_portfolios(self) -> List[PortfolioMeta]:
        return self.repo.list_portfolios()

    def create_portfolio(self, name: str, portfolio_id: Optional[str] = None) -> PortfolioMeta:
        return self.repo.create_portfolio(name=name, portfolio_id=portfolio_id)

    def delete_portfolio(self, portfolio_id: str) -> None:
        self.repo.delete_portfolio(portfolio_id=portfolio_id)

    def update_portfolio_name(self, portfolio_id: str, name: str) -> PortfolioMeta:
        return self.repo.rename_portfolio(portfolio_id=portfolio_id, new_name=name)

    # =========================================================================
    # 1. Core State Queries & Buckets
    # =========================================================================

    def get_portfolio_state(self, refresh_prices: bool = True, portfolio_id: str = "default") -> str:
        from filelock import Timeout
        pid = validate_portfolio_id(portfolio_id)
        refresh_info: Dict[str, str] = {}
        try:
            with self.repo.unit_of_work(pid) as uow:
                state = uow.load_state()
                if refresh_prices:
                    from tools.portfolio.prices import _refresh_prices
                    refresh_info = _refresh_prices(state)
                    uow.commit(state, LedgerChange(kind="unchanged"))
                else:
                    recalc_all(state)
        except Timeout:
            return json.dumps({"error": f"portfolio lock timeout for '{portfolio_id}'"}, ensure_ascii=False)

        dump = state.model_dump(exclude_none=True)
        if refresh_info:
            dump["_price_refresh"] = refresh_info
        return json.dumps(dump, ensure_ascii=False, indent=2)

    def get_structured_portfolio_state(
        self, refresh_prices: bool = False, fetch_fundamentals: bool = False, portfolio_id: str = "default"
    ) -> PortfolioState:
        pid = validate_portfolio_id(portfolio_id)
        if refresh_prices or fetch_fundamentals:
            with self.repo.unit_of_work(pid) as uow:
                state = uow.load_state()
                if refresh_prices:
                    from tools.portfolio.prices import _refresh_prices
                    _refresh_prices(state)
                if fetch_fundamentals:
                    from tools.portfolio.prices import _fetch_fundamentals
                    _fetch_fundamentals(state, force=False)
                recalc_all(state)
                uow.commit(state, LedgerChange(kind="unchanged"))
                return state

        return self.repo.load_state(pid)

    def get_structured_bucket_allocation(
        self, state: Optional[PortfolioState] = None, portfolio_id: str = "default"
    ) -> Tuple[List[Dict], Optional[str]]:
        pid = validate_portfolio_id(portfolio_id)
        st = state or self.repo.load_state(pid)
        return compute_target_allocation_variance(st)

    def structured_assign_holding_bucket(
        self, symbol: str, bucket_id: Optional[str], portfolio_id: str = "default"
    ) -> PortfolioState:
        pid = validate_portfolio_id(portfolio_id)
        with self.repo.unit_of_work(pid) as uow:
            state = uow.load_state()
            if bucket_id is not None:
                valid_bucket_ids = {t.bucket_id for t in state.allocation_targets}
                if bucket_id not in valid_bucket_ids:
                    raise ValueError(f"ไม่พบ bucket_id '{bucket_id}' ใน allocation targets")
            clean_sym = symbol.strip().upper()
            h = _find_holding(state, clean_sym)
            if not h:
                raise ValueError(f"ไม่พบสินทรัพย์ '{clean_sym}' ในพอร์ต")
            h.bucket_id = bucket_id
            uow.commit(state, LedgerChange(kind="unchanged"))
            return state

    def structured_batch_assign_holding_buckets(
        self, symbols: List[str], bucket_id: Optional[str], portfolio_id: str = "default"
    ) -> PortfolioState:
        pid = validate_portfolio_id(portfolio_id)
        with self.repo.unit_of_work(pid) as uow:
            state = uow.load_state()
            if bucket_id is not None:
                valid_bucket_ids = {t.bucket_id for t in state.allocation_targets}
                if bucket_id not in valid_bucket_ids:
                    raise ValueError(f"ไม่พบ bucket_id '{bucket_id}' ใน allocation targets")
            sym_set = {s.strip().upper() for s in symbols}
            found = 0
            for h in state.holdings:
                if h.symbol in sym_set:
                    h.bucket_id = bucket_id
                    found += 1
            if found == 0 and symbols:
                raise ValueError("ไม่พบสินทรัพย์ตามที่ระบุในพอร์ต")
            uow.commit(state, LedgerChange(kind="unchanged"))
            return state

    def structured_batch_remove_holdings(
        self, symbols: List[str], portfolio_id: str = "default"
    ) -> PortfolioState:
        pid = validate_portfolio_id(portfolio_id)
        with self.repo.unit_of_work(pid) as uow:
            state = uow.load_state()
            sym_set = {s.strip().upper() for s in symbols}
            before_len = len(state.holdings)
            state.holdings = [h for h in state.holdings if h.symbol not in sym_set]
            if len(state.holdings) == before_len and symbols:
                raise ValueError("ไม่พบสินทรัพย์ตามที่ระบุเพื่อลบ")
            uow.commit(state, LedgerChange(kind="unchanged"))
            return state

    def structured_reset_clean_slate(self, portfolio_id: str = "default") -> PortfolioState:
        import shutil
        from datetime import datetime
        from tools.portfolio.adapters.markdown.paths import get_portfolio_filepath, get_holdings_dir

        pid = validate_portfolio_id(portfolio_id)
        with self.repo.unit_of_work(pid) as uow:
            state = uow.load_state()

            # Create backup before wiping
            portfolio_file = get_portfolio_filepath(pid)
            holdings_dir = get_holdings_dir(pid)
            backups_dir = portfolio_file.parent / ".backups"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dest = backups_dir / timestamp

            try:
                backup_dest.mkdir(parents=True, exist_ok=True)
                if portfolio_file.exists():
                    shutil.copy2(portfolio_file, backup_dest / portfolio_file.name)
                if holdings_dir.exists():
                    dest_holdings = backup_dest / "Holdings"
                    dest_holdings.mkdir(parents=True, exist_ok=True)
                    for f in holdings_dir.glob("*.md"):
                        shutil.copy2(f, dest_holdings / f.name)
            except Exception as e:
                raise ValueError(f"สำรองข้อมูลก่อนล้างพอร์ตไม่สำเร็จ: {e}")

            # Clean sidecars
            if holdings_dir.exists():
                for f in holdings_dir.glob("*.md"):
                    try:
                        f.unlink(missing_ok=True)
                    except Exception:
                        pass

            new_state = PortfolioState(
                last_updated=_now_iso(),
                allocation_targets=default_allocation_targets(),
                fx_rates={"USDTHB": 36.5},
                holdings=[],
            )
            uow.commit(new_state, LedgerChange(kind="replace_all", rows=[]))
            return new_state

    def structured_upsert_allocation_targets(
        self, targets: List[AllocationTarget], portfolio_id: str = "default"
    ) -> PortfolioState:
        pid = validate_portfolio_id(portfolio_id)
        with self.repo.unit_of_work(pid) as uow:
            state = uow.load_state()
            total_pct = sum(t.target_percent for t in targets)
            if abs(total_pct - 100.0) > 0.01:
                raise ValueError(f"ผลรวม target_percent ({total_pct:.1f}%) ต้องเท่ากับ 100%")
            state.allocation_targets = targets
            uow.commit(state, LedgerChange(kind="unchanged"))
            return state

    def compute_allocation_breakdown(
        self, group_by: Literal["asset_type", "currency"] = "asset_type", portfolio_id: str = "default"
    ) -> str:
        from filelock import Timeout
        from tools.tool_errors import LOCK_TIMEOUT
        pid = validate_portfolio_id(portfolio_id)
        try:
            state = self.repo.load_state(pid)
            breakdown = compute_allocation_breakdown(state, group_by=group_by)
            total_nav = state.summary.total_value_thb
            return json.dumps(
                {
                    "group_by": group_by,
                    "total_nav_thb": total_nav,
                    "breakdown": breakdown,
                    "generated_at": _now_iso(),
                },
                ensure_ascii=False,
                indent=2,
            )
        except Timeout:
            return LOCK_TIMEOUT.format(detail=f"portfolio lock '{pid}'")

    # =========================================================================
    # 2. Portfolio Meta Management
    # =========================================================================

    def list_portfolios(self) -> List[PortfolioMeta]:
        return self.repo.list_portfolios()

    def create_portfolio(self, name: str, portfolio_id: Optional[str] = None) -> PortfolioMeta:
        return self.repo.create_portfolio(name, portfolio_id=portfolio_id)

    def delete_portfolio(self, portfolio_id: str) -> None:
        self.repo.delete_portfolio(portfolio_id)

    def update_portfolio_name(self, portfolio_id: str, name: str) -> PortfolioMeta:
        return self.repo.rename_portfolio(portfolio_id, name)

    # =========================================================================
    # 3. Trade & Cash Operations
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
        from filelock import Timeout
        from tools.tool_errors import LOCK_TIMEOUT
        try:
            res_str, state = self._execute_trade_internal(
                symbol=symbol,
                asset_type=asset_type,
                action=action,
                units=units,
                price=price,
                currency=currency,
                notes=notes,
                portfolio_id=portfolio_id,
            )
            return res_str
        except Timeout:
            return LOCK_TIMEOUT.format(detail=f"portfolio lock '{portfolio_id}'")
        except Exception as e:
            return f"Error: {e}"

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
        _, state = self._execute_trade_internal(
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
        return state

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
        clean_sym = symbol.strip().upper()
        if clean_sym in _CASH_SYMBOLS:
            raise ValueError(f"ห้ามเทรด cash sentinel ({clean_sym}) ผ่าน execute_trade — ให้ใช้ manage_cash_flow สำหรับฝาก/ถอนเงินสด")
        if units <= 0:
            raise ValueError("units ต้องมากกว่า 0")
        if price <= 0:
            raise ValueError("price ต้องมากกว่า 0")
        if action not in ("buy", "sell"):
            raise ValueError("action ต้องเป็น 'buy' หรือ 'sell'")
        if currency not in ("THB", "USD"):
            raise ValueError("currency ต้องเป็น 'THB' หรือ 'USD'")

        pid = validate_portfolio_id(portfolio_id)
        with self.repo.unit_of_work(pid) as uow:
            state = uow.load_state()
            today_str = _now_iso()[:10]
            is_backdate = bool(date and date.strip() and date.strip() < today_str)

            if exchange_rate is not None and exchange_rate > 0:
                trade_fx = float(exchange_rate)
                if not is_backdate:
                    state.fx_rates["USDTHB"] = trade_fx
            elif currency == "USD":
                import tools.portfolio.prices as prices_mod
                rate_val, _ = prices_mod.fetch_fx_rate(date_str=date, fallback_rate=state.fx_rates.get("USDTHB", 36.5))
                trade_fx = rate_val
                if not is_backdate:
                    state.fx_rates["USDTHB"] = trade_fx
            else:
                trade_fx = 1.0

            cash = _require_cash(state, currency)

            total_cost_native = units * price
            fx_rate_for_log = trade_fx if currency == "USD" else 1.0
            tx_id = f"tx_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
            ts_str = f"{date.strip()} 12:00:00" if date and date.strip() else _now_iso().replace("T", " ")

            ledger_row: Dict = {
                "Transaction_ID": tx_id,
                "Timestamp": ts_str,
                "Symbol": clean_sym,
                "Action": action.upper(),
                "Units": f"{units:g}",
                "Price": f"{price:.2f}",
                "Currency": currency,
                "FX_Rate": f"{trade_fx:.4f}" if currency == "USD" else "",
                "Cost_THB": "",
                "Realized_PnL_THB": "",
                "Notes": notes or "",
            }

            holding = _find_holding(state, clean_sym)

            if action == "buy":
                if holding is not None and holding.status == "active":
                    holding_ccy = "USD" if (holding.avg_cost_usd is not None and holding.avg_cost_usd > 0) else "THB"
                    if currency == "USD" and holding_ccy == "THB":
                        raise ValueError(f"สินทรัพย์ {clean_sym} เป็นสินทรัพย์ THB อยู่แล้ว ไม่สามารถซื้อด้วย USD ได้")
                    if currency == "THB" and holding_ccy == "USD":
                        raise ValueError(f"สินทรัพย์ {clean_sym} มีสกุลเงินเป็น USD อยู่แล้ว ไม่สามารถทำรายการด้วย THB ได้")

                if cash.units < total_cost_native - _FLOAT_EPS:
                    raise ValueError(f"Insufficient cash in CASH_{currency} (required: {total_cost_native:,.2f}, available: {cash.units:,.2f})")

                cash.units -= total_cost_native
                cost_thb = round(total_cost_native * fx_rate_for_log, _MONEY_DP)
                ledger_row["Cost_THB"] = f"{cost_thb:.2f}"

                if holding is None or holding.status == "archived":
                    if holding is None:
                        holding = Holding(
                            symbol=clean_sym,
                            asset_type=asset_type,
                            units=0.0,
                            bucket_id=bucket_id,
                        )
                        state.holdings.append(holding)
                    else:
                        holding.status = "active"
                        holding.archived_at = None
                        holding.units = 0.0

                    global_fx = state.fx_rates.get("USDTHB", 36.5)
                    if currency == "USD":
                        holding.avg_cost_usd = price
                        holding.current_price_usd = price
                        holding.current_price_thb = round(price * global_fx, _MONEY_DP)
                    else:
                        holding.avg_cost_thb = price
                        holding.current_price_thb = price
                        holding.current_price_usd = round(price / global_fx, _MONEY_DP)

                old_units = holding.units
                old_cost = (holding.avg_cost_usd if currency == "USD" else holding.avg_cost_thb) or 0.0
                new_avg = calc_weighted_avg_cost(old_units, old_cost, units, price)
                holding.units += units

                if currency == "USD":
                    holding.avg_cost_usd = new_avg
                    holding.current_price_usd = price
                else:
                    holding.avg_cost_thb = new_avg
                    holding.current_price_thb = price

                curr_prefix = "$" if currency == "USD" else "฿"
                avg_text = f" (Avg cost updated to {curr_prefix}{new_avg:,.2f})" if old_units > _FLOAT_EPS else ""
                res_str = f"[BUY] {clean_sym} {units:g} units @ {curr_prefix}{price:,.2f}{avg_text} | CASH_{currency} คงเหลือ: {cash.units:,.2f} {currency}"

            elif action == "sell":
                if holding is None or holding.status == "archived" or holding.units <= _FLOAT_EPS:
                    raise ValueError(f"สินทรัพย์ {clean_sym} ไม่มีในพอร์ต หรือจำนวนหน่วยเป็น 0")

                holding_ccy = "USD" if (holding.avg_cost_usd is not None and holding.avg_cost_usd > 0) else "THB"
                if currency == "USD" and holding_ccy == "THB":
                    raise ValueError(f"สินทรัพย์ {clean_sym} มี cost เป็น THB ไม่สามารถขายด้วย USD ได้")
                if currency == "THB" and holding_ccy == "USD":
                    raise ValueError(f"สินทรัพย์ {clean_sym} มี cost เป็น USD ไม่สามารถขายด้วย THB ได้")

                if units > holding.units + _FLOAT_EPS:
                    raise ValueError(f"Insufficient units to sell for {clean_sym} (requested: {units:g}, available: {holding.units:g})")

                cash.units += total_cost_native
                avg_cost = (holding.avg_cost_usd if currency == "USD" else holding.avg_cost_thb) or 0.0
                realized_pnl = calc_realized_pnl(avg_cost, units, price, fx_rate=fx_rate_for_log)
                cost_basis_thb = round(avg_cost * units * fx_rate_for_log, _MONEY_DP)

                ledger_row["Cost_THB"] = f"{cost_basis_thb:.2f}"
                ledger_row["Realized_PnL_THB"] = f"{realized_pnl:.2f}"
                state.summary.total_realized_profit_ytd = round(
                    state.summary.total_realized_profit_ytd + realized_pnl, _MONEY_DP
                )

                holding.units = max(0.0, holding.units - units)
                if holding.units <= _FLOAT_EPS:
                    state.holdings = [h for h in state.holdings if h.symbol != clean_sym]

                sign = "+" if realized_pnl >= 0 else ""
                res_str = f"[SELL] {clean_sym} {units:g} units @ {price:,.2f} {currency} | Realized P/L: {sign}{realized_pnl:,.2f} {currency} | CASH_{currency} คงเหลือ: {cash.units:,.2f} {currency}"

            recalc_all(state)
            uow.commit(state, LedgerChange(kind="append", row=ledger_row, tx_id=tx_id))

            content = f"**[TRADE NOTE - {clean_sym}]** {action.upper()} {units:g} @ {price:,.4f} {currency}"
            if notes and notes.strip():
                content += f" — {notes.strip()}"
            from tools.portfolio.journal import _write_journal_entry
            _write_journal_entry(content, date_str=date, portfolio_id=pid)

            return res_str, state

    def batch_import_holdings(
        self,
        assets_list: Union[List[Dict], str],
        mode: Literal["merge", "overwrite"] = "merge",
        reset_cash_usd: bool = False,
        portfolio_id: str = "default",
    ) -> str:
        from filelock import Timeout
        from tools.tool_errors import LOCK_TIMEOUT
        if mode not in ("merge", "overwrite"):
            return "Error: mode ต้องเป็น 'merge' หรือ 'overwrite'"

        if not isinstance(assets_list, (list, str)):
            return "Error: assets_list ต้องเป็น list"

        if isinstance(assets_list, str):
            try:
                raw_list = json.loads(assets_list)
            except Exception:
                return "Error: assets_list ต้องเป็น list"
        else:
            raw_list = assets_list

        if not isinstance(raw_list, list):
            return "Error: assets_list ต้องเป็น list"

        if mode == "merge" and len(raw_list) == 0:
            return "Error: assets_list ต้องเป็น list ที่ไม่ว่าง"

        for item in raw_list:
            if not isinstance(item, dict):
                return "Error: item ใน assets_list ไม่ใช่ dict"
            for required_k in ("symbol", "units", "avg_cost"):
                if required_k not in item:
                    return f"Error: item field ขาดหรือ format ผิด ({required_k})"
            sym = str(item.get("symbol", "")).strip().upper()
            if not sym:
                return "Error: symbol ว่าง"
            if sym in _CASH_SYMBOLS:
                return "Error: ห้ามนำเข้า cash sentinel (CASH_THB/CASH_USD) ผ่าน batch_import_holdings"
            try:
                u = float(item["units"])
                c = float(item["avg_cost"])
            except (ValueError, TypeError):
                return "Error: units หรือ avg_cost ต้องเป็นตัวเลข"
            if u <= 0:
                return "Error: units ต้องมากกว่า 0"
            if c <= 0:
                return "Error: avg_cost ต้องมากกว่า 0"
            curr = str(item.get("currency", "THB")).strip().upper()
            if curr not in ("THB", "USD"):
                return "Error: currency ต้องเป็น 'THB' หรือ 'USD'"
            if item.get("current_price") is not None:
                try:
                    cp = float(item["current_price"])
                except (ValueError, TypeError):
                    return "Error: current_price format ผิด"
                if cp <= 0:
                    return "Error: current_price ต้องมากกว่า 0"

        syms = [str(i["symbol"]).strip().upper() for i in raw_list]
        if len(syms) != len(set(syms)):
            return "Error: มี duplicate symbol ซ้ำกันใน assets_list"

        pid = validate_portfolio_id(portfolio_id)
        try:
            import tools.portfolio.trading as trading_mod
            lock = trading_mod._get_portfolio_lock(pid)
            with lock:
                with self.repo.unit_of_work(pid) as uow:
                    state = uow.load_state()
                    current_fx = _require_fx(state)

                    if mode == "overwrite":
                        cash_holdings = [h for h in state.holdings if h.asset_type == "Cash"]
                        if reset_cash_usd:
                            for ch in cash_holdings:
                                if ch.symbol == CASH_USD_SYMBOL:
                                    ch.units = 0.0
                        state.holdings = cash_holdings

                    provided_count = 0
                    fetched_count = 0
                    fallback_count = 0

                    for item in raw_list:
                        sym = str(item["symbol"]).strip().upper()
                        asset_type = item.get("asset_type", "Stock")
                        units = float(item["units"])
                        avg_cost = float(item["avg_cost"])
                        ccy = str(item.get("currency", "THB")).strip().upper()

                        h = _find_holding(state, sym)
                        if h is None:
                            h = Holding(symbol=sym, asset_type=asset_type, units=0.0)
                            state.holdings.append(h)

                        h.units = units
                        h.status = "active"
                        h.archived_at = None

                        if item.get("current_price") is not None:
                            cprice = float(item["current_price"])
                            provided_count += 1
                        else:
                            fetched = None
                            try:
                                timeout = getattr(trading_mod, "_PRICE_FETCH_TIMEOUT", 6.0)
                                import concurrent.futures
                                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                                    future = executor.submit(trading_mod.fetch_latest_price, sym, ccy)
                                    try:
                                        fetched = future.result(timeout=timeout)
                                    except Exception:
                                        fetched = None
                            except Exception:
                                fetched = None

                            if fetched is not None and fetched > 0:
                                cprice = fetched
                                fetched_count += 1
                            else:
                                cprice = avg_cost
                                fallback_count += 1

                        if ccy == "USD":
                            h.avg_cost_usd = avg_cost
                            h.avg_cost_thb = None
                            h.current_price_usd = cprice
                            h.current_price_thb = round(cprice * current_fx, _MONEY_DP)
                        else:
                            h.avg_cost_thb = avg_cost
                            h.avg_cost_usd = None
                            h.current_price_thb = cprice
                            h.current_price_usd = round(cprice / current_fx, _MONEY_DP)

                    recalc_all(state)
                    uow.commit(state, LedgerChange(kind="unchanged"))
                    mode_tag = "OVERWRITE" if mode == "overwrite" else "MERGE"
                    return f"[IMPORT {mode_tag}] นำเข้าสำเร็จ {len(raw_list)} รายการ (provided={provided_count}, fetched={fetched_count}, fallback={fallback_count})"
        except Timeout:
            return LOCK_TIMEOUT.format(detail=f"portfolio lock '{pid}'")
        except Exception as e:
            return f"Error: {e}"

    def manage_cash_flow(
        self,
        amount: float,
        action: Literal["deposit", "withdraw"],
        currency: Literal["THB", "USD"] = "THB",
        portfolio_id: str = "default",
    ) -> str:
        from filelock import Timeout
        from tools.tool_errors import LOCK_TIMEOUT
        try:
            res_str, _ = self._manage_cash_flow_internal(
                amount=amount, action=action, currency=currency, portfolio_id=portfolio_id
            )
            return res_str
        except Timeout:
            return LOCK_TIMEOUT.format(detail=f"portfolio lock '{portfolio_id}'")
        except Exception as e:
            return f"Error: {e}"

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
        _, state = self._manage_cash_flow_internal(
            amount=amount,
            action=action,
            currency=currency,
            exchange_rate=exchange_rate,
            date=date,
            notes=notes,
            portfolio_id=portfolio_id,
        )
        return state

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
        if amount <= 0:
            raise ValueError("amount ต้องมากกว่า 0")
        if action not in ("deposit", "withdraw"):
            raise ValueError("action ต้องเป็น 'deposit' หรือ 'withdraw'")
        if currency not in ("THB", "USD"):
            raise ValueError("currency ต้องเป็น 'THB' หรือ 'USD'")

        pid = validate_portfolio_id(portfolio_id)
        with self.repo.unit_of_work(pid) as uow:
            state = uow.load_state()
            cash = _require_cash(state, currency)

            if action == "withdraw":
                if cash.units < amount - _FLOAT_EPS:
                    raise ValueError(f"Insufficient cash in CASH_{currency} (requested: {amount:,.2f}, available: {cash.units:,.2f})")
                cash.units -= amount
            else:
                cash.units += amount

            current_fx = exchange_rate or _require_fx(state)
            recalc_all(state)
            uow.commit(state, LedgerChange(kind="unchanged"))
            res_str = f"[{action.upper()}] {currency} | {'+' if action == 'deposit' else '-'}{amount:,.2f} {currency} | CASH_{currency} คงเหลือ: {cash.units:,.2f} {currency}"

            content = f"**[CASH FLOW NOTE]** {action.upper()} {amount:,.2f} {currency}"
            if notes and notes.strip():
                content += f" — {notes.strip()}"
            from tools.portfolio.journal import _write_journal_entry
            _write_journal_entry(content, date_str=date, portfolio_id=pid)

            return res_str, state

    def record_income(
        self,
        income_type: Literal["Dividend", "Interest", "Rental", "Other"],
        amount_thb: float,
        source_symbol: Optional[str] = None,
        portfolio_id: str = "default",
    ) -> str:
        from filelock import Timeout
        from tools.tool_errors import LOCK_TIMEOUT
        try:
            res_str, _ = self._record_income_internal(
                income_type=income_type,
                amount_thb=amount_thb,
                source_symbol=source_symbol,
                portfolio_id=portfolio_id,
            )
            return res_str
        except Timeout:
            return LOCK_TIMEOUT.format(detail=f"portfolio lock '{portfolio_id}'")
        except Exception as e:
            return f"Error: {e}"

    def structured_record_income(
        self,
        income_type: Literal["Dividend", "Interest", "Rental", "Other"],
        amount_thb: float,
        source_symbol: Optional[str] = None,
        date: Optional[str] = None,
        notes: str = "",
        portfolio_id: str = "default",
    ) -> PortfolioState:
        _, state = self._record_income_internal(
            income_type=income_type,
            amount_thb=amount_thb,
            source_symbol=source_symbol,
            date=date,
            notes=notes,
            portfolio_id=portfolio_id,
        )
        return state

    def _record_income_internal(
        self,
        income_type: Literal["Dividend", "Interest", "Rental", "Other"],
        amount_thb: float,
        source_symbol: Optional[str] = None,
        date: Optional[str] = None,
        notes: str = "",
        portfolio_id: str = "default",
    ) -> Tuple[str, PortfolioState]:
        if amount_thb <= 0:
            raise ValueError("amount_thb ต้องมากกว่า 0")
        clean_src = source_symbol.strip().upper() if source_symbol else None
        if clean_src and clean_src in _CASH_SYMBOLS:
            raise ValueError(f"source_symbol ห้ามเป็น cash sentinel ({source_symbol})")

        pid = validate_portfolio_id(portfolio_id)
        with self.repo.unit_of_work(pid) as uow:
            state = uow.load_state()
            if clean_src:
                h = _find_holding(state, clean_src)
                if not h:
                    raise ValueError(f"ไม่พบสินทรัพย์ '{source_symbol}' ในพอร์ต")

            cash = _require_cash(state, "THB")
            cash.units += amount_thb

            state.summary.passive_income_ytd = round(
                state.summary.passive_income_ytd + amount_thb, _MONEY_DP
            )
            if income_type == "Dividend":
                state.summary.total_accumulated_dividend = round(
                    state.summary.total_accumulated_dividend + amount_thb, _MONEY_DP
                )
                if clean_src:
                    h = _find_holding(state, clean_src)
                    if h:
                        h.accumulated_dividend_thb = round(
                            (h.accumulated_dividend_thb or 0.0) + amount_thb, _MONEY_DP
                        )

            recalc_all(state)
            uow.commit(state, LedgerChange(kind="unchanged"))
            tag_map = {"Dividend": "DIV", "Interest": "INT", "Rental": "RENT", "Other": "INCOME"}
            tag = tag_map.get(income_type, "INCOME")
            src_str = f" จาก {clean_src}" if clean_src else ""
            res_str = f"[{tag}] +{amount_thb:,.2f} THB{src_str} | passive_income_ytd: {state.summary.passive_income_ytd:,.2f} | เงินสดคงเหลือ: {cash.units:,.2f} บาท"

            src_note = f" ({clean_src})" if clean_src else ""
            content = f"**[INCOME NOTE - {income_type}{src_note}]** +{amount_thb:,.2f} THB"
            if notes and notes.strip():
                content += f" — {notes.strip()}"
            from tools.portfolio.journal import _write_journal_entry
            _write_journal_entry(content, date_str=date, portfolio_id=pid)

            return res_str, state

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
        from filelock import Timeout
        from tools.tool_errors import LOCK_TIMEOUT
        try:
            res_str, _ = self._edit_holding_internal(
                symbol=symbol,
                units=units,
                avg_cost=avg_cost,
                accumulated_dividend_thb=accumulated_dividend_thb,
                asset_type=asset_type,
                reason=reason,
                portfolio_id=portfolio_id,
            )
            return res_str
        except Timeout:
            return LOCK_TIMEOUT.format(detail=f"portfolio lock '{portfolio_id}'")
        except Exception as e:
            return f"Error: {e}"

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
        _, state = self._edit_holding_internal(
            symbol=symbol,
            units=units,
            avg_cost=avg_cost,
            accumulated_dividend_thb=accumulated_dividend_thb,
            asset_type=asset_type,
            reason=reason,
            bucket_id=bucket_id,
            portfolio_id=portfolio_id,
        )
        return state

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
        clean_sym = symbol.strip().upper()
        if clean_sym in _CASH_SYMBOLS:
            raise ValueError(f"สินทรัพย์ {clean_sym} เป็น Cash holding ห้ามแก้ไขผ่าน edit_holding — ให้ใช้ manage_cash_flow แทน")
        if units is None and avg_cost is None and accumulated_dividend_thb is None and asset_type is None and bucket_id is None:
            raise ValueError("ต้องระบุอย่างน้อย 1 field ที่ต้องการแก้ไข")
        if units is not None and units <= 0:
            raise ValueError("units ต้องมากกว่า 0")
        if avg_cost is not None and avg_cost <= 0:
            raise ValueError("avg_cost ต้องมากกว่า 0")
        if accumulated_dividend_thb is not None and accumulated_dividend_thb < 0:
            raise ValueError("accumulated_dividend_thb ต้องไม่ติดลบ (>= 0)")

        pid = validate_portfolio_id(portfolio_id)
        with self.repo.unit_of_work(pid) as uow:
            state = uow.load_state()
            h = _find_holding(state, clean_sym)
            if not h or h.status == "archived":
                raise ValueError(f"ไม่พบสินทรัพย์ '{clean_sym}' ในพอร์ต")

            changes = []
            if units is not None and abs(units - h.units) > _FLOAT_EPS:
                changes.append(f"units: {h.units:g} → {units:g}")
                h.units = units

            if avg_cost is not None:
                if h.avg_cost_usd is not None:
                    if abs(avg_cost - h.avg_cost_usd) > _FLOAT_EPS:
                        changes.append(f"avg_cost_usd: ${h.avg_cost_usd:.2f} → ${avg_cost:.2f}")
                        h.avg_cost_usd = avg_cost
                elif h.avg_cost_thb is not None:
                    if abs(avg_cost - h.avg_cost_thb) > _FLOAT_EPS:
                        changes.append(f"avg_cost_thb: {h.avg_cost_thb:.2f} → {avg_cost:.2f}")
                        h.avg_cost_thb = avg_cost
                else:
                    raise ValueError(f"สินทรัพย์ {clean_sym} ไม่มี avg_cost_thb/usd เดิม ไม่สามารถแก้ไข avg_cost ได้")

            if accumulated_dividend_thb is not None:
                old_div = h.accumulated_dividend_thb or 0.0
                if abs(accumulated_dividend_thb - old_div) > _FLOAT_EPS:
                    changes.append(f"accumulated_dividend_thb: {old_div:.2f} → {accumulated_dividend_thb:.2f}")
                    h.accumulated_dividend_thb = accumulated_dividend_thb

            if asset_type is not None and asset_type != h.asset_type:
                changes.append(f"asset_type: {h.asset_type} → {asset_type}")
                h.asset_type = asset_type

            if bucket_id is not None and bucket_id != h.bucket_id:
                changes.append(f"bucket_id: {h.bucket_id} → {bucket_id}")
                h.bucket_id = bucket_id

            if not changes:
                raise ValueError("ค่าที่ระบุไม่มีการเปลี่ยนแปลงจากข้อมูลเดิม (เหมือนเดิม)")

            reason_text = reason.strip() or "(no reason given)"
            from tools.portfolio.journal import _write_journal_entry
            _write_journal_entry(
                f"**[EDIT {clean_sym}]** {' | '.join(changes)}\n\nReason: {reason_text}",
                portfolio_id=pid,
            )

            recalc_all(state)
            uow.commit(state, LedgerChange(kind="unchanged"))
            res_str = f"[EDIT {clean_sym}] {', '.join(changes)} (เหตุผล: {reason})"
            return res_str, state

    def structured_remove_holding(self, symbol: str, portfolio_id: str = "default") -> PortfolioState:
        clean_sym = symbol.strip().upper()
        if clean_sym in _CASH_SYMBOLS:
            raise ValueError("ห้ามลบ cash sentinel")
        pid = validate_portfolio_id(portfolio_id)
        with self.repo.unit_of_work(pid) as uow:
            state = uow.load_state()
            target = _find_holding(state, clean_sym)
            if not target:
                raise ValueError(f"ไม่พบสินทรัพย์ {clean_sym} ในพอร์ต")
            state.holdings.remove(target)
            recalc_all(state)
            uow.commit(state, LedgerChange(kind="unchanged"))
            from tools.portfolio.journal import _write_journal_entry
            _write_journal_entry(f"**[REMOVE {clean_sym}]** ลบสินทรัพย์ออกจากพอร์ตโดยตรง", portfolio_id=pid)
            return state

    def structured_batch_remove_holdings(self, symbols: List[str], portfolio_id: str = "default") -> PortfolioState:
        pid = validate_portfolio_id(portfolio_id)
        with self.repo.unit_of_work(pid) as uow:
            state = uow.load_state()
            from tools.portfolio.journal import _write_journal_entry
            for sym in symbols:
                clean_sym = sym.strip().upper()
                if clean_sym in _CASH_SYMBOLS:
                    continue
                target = _find_holding(state, clean_sym)
                if target:
                    state.holdings.remove(target)
                    _write_journal_entry(f"**[REMOVE {clean_sym}]** ลบสินทรัพย์ออกจากพอร์ตโดยตรง", portfolio_id=pid)
            recalc_all(state)
            uow.commit(state, LedgerChange(kind="unchanged"))
            return state

    def update_fx_rate(self, rate: Optional[float] = None, portfolio_id: str = "default") -> str:
        from filelock import Timeout
        from tools.tool_errors import LOCK_TIMEOUT
        if rate is not None:
            if rate <= 0:
                return "Error: rate ต้องมากกว่า 0"
            new_rate = float(rate)
            source = "manual"
        else:
            try:
                import tools.portfolio.trading as trading_fresh
                new_rate = trading_fresh._fetch_fx_rate()
            except Exception:
                new_rate = None
            if new_rate is None or new_rate <= 0:
                return "Error: auto-fetch FX ล้มเหลว กรุณาระบุ rate ด้วยตนเอง"
            source = "yfinance"

        pid = validate_portfolio_id(portfolio_id)
        try:
            with self.repo.unit_of_work(pid) as uow:
                state = uow.load_state()
                old_rate = state.fx_rates.get("USDTHB", 36.5)
                state.fx_rates["USDTHB"] = new_rate
                recalc_all(state)
                uow.commit(state, LedgerChange(kind="unchanged"))
                return f"[FX {source}] USDTHB: {old_rate:.4f} → {new_rate:.4f}"
        except Timeout:
            return LOCK_TIMEOUT.format(detail=f"portfolio lock '{pid}'")
        except Exception as e:
            return f"Error: {e}"

    def sync_market_prices(self, portfolio_id: str = "default") -> str:
        from tools.portfolio.prices import _sync_market_prices_impl
        return _sync_market_prices_impl(portfolio_id=portfolio_id)

    # =========================================================================
    # 4. Ledger Operations
    # =========================================================================

    def get_structured_trades_log(
        self, portfolio_id: str = "default", symbol: Optional[str] = None
    ) -> List[Dict]:
        return self.repo.read_trade_log(portfolio_id, symbol=symbol)

    def update_trade_note(self, tx_id: str, notes: str, portfolio_id: str = "default") -> Dict:
        pid = validate_portfolio_id(portfolio_id)
        with self.repo.unit_of_work(pid) as uow:
            state = uow.load_state()
            fpath = get_trades_log_filepath(pid)
            from tools.portfolio.adapters.markdown.repository_adapter import (
                _read_and_migrate_trade_log_locked,
                _sanitize_csv_field,
            )
            rows = _read_and_migrate_trade_log_locked(fpath)
            found = False
            updated_row = {}
            sanitized_notes = _sanitize_csv_field(notes)
            for r in rows:
                if r.get("Transaction_ID") == tx_id:
                    r["Notes"] = sanitized_notes
                    found = True
                    item_dict = {k.lower(): v for k, v in r.items()}
                    item_dict.update({k: v for k, v in r.items()})
                    updated_row = item_dict
                    break
            if not found:
                raise ValueError(f"ไม่พบ transaction id '{tx_id}'")

            uow.commit(state, LedgerChange(kind="replace_all", rows=rows, tx_id=tx_id))
            return updated_row

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
        pid = validate_portfolio_id(portfolio_id)
        with self.repo.unit_of_work(pid) as uow:
            state = uow.load_state()
            fpath = get_trades_log_filepath(pid)
            from tools.portfolio.adapters.markdown.repository_adapter import _read_and_migrate_trade_log_locked
            rows = _read_and_migrate_trade_log_locked(fpath)
            target_row = next((r for r in rows if r.get("Transaction_ID") == tx_id), None)
            if not target_row:
                raise ValueError(f"ไม่พบ transaction id '{tx_id}'")

            old_units = float(target_row.get("Units", 0))
            old_price = float(target_row.get("Price", 0))
            action = target_row.get("Action", "BUY").upper()

            if timestamp is not None:
                target_row["Timestamp"] = timestamp
            if units is not None:
                target_row["Units"] = f"{units:g}"
            if price is not None:
                target_row["Price"] = f"{price:.2f}"
            if fx_rate is not None:
                target_row["FX_Rate"] = f"{fx_rate:.4f}"
            if notes is not None:
                target_row["Notes"] = notes

            new_units_val = float(target_row.get("Units", 0))
            new_price_val = float(target_row.get("Price", 0))

            sym = target_row.get("Symbol")
            ccy = target_row.get("Currency", "THB")
            sym_rows = [r for r in rows if r.get("Symbol") == sym]
            updated_sym_rows, final_units, final_avg, total_realized = _replay_symbol_trades(sym_rows, sym, ccy)

            row_map = {r["Transaction_ID"]: r for r in updated_sym_rows}
            new_rows = [row_map.get(r["Transaction_ID"], r) for r in rows]

            if adjust_cash:
                cash = _require_cash(state, ccy)
                if action == "BUY":
                    delta_cash = - (new_units_val * new_price_val) - (- (old_units * old_price))
                    cash.units += delta_cash
                elif action == "SELL":
                    delta_cash = (new_units_val * new_price_val) - (old_units * old_price)
                    cash.units += delta_cash

            h = _find_holding(state, sym)
            if final_units > _FLOAT_EPS:
                if h is None:
                    h = Holding(symbol=sym, asset_type="Stock", units=final_units)
                    state.holdings.append(h)
                h.units = final_units
                h.status = "active"
                h.archived_at = None
                if ccy == "USD":
                    h.avg_cost_usd = final_avg
                else:
                    h.avg_cost_thb = final_avg
                if h.dividend_source == "synced":
                    h.dividend_source = None
            else:
                if h:
                    state.holdings = [item for item in state.holdings if item.symbol != sym]

            state.summary.total_realized_profit_ytd = round(
                sum(float(r.get("Realized_PnL_THB") or 0.0) for r in new_rows), _MONEY_DP
            )
            recalc_all(state)
            uow.commit(state, LedgerChange(kind="replace_all", rows=new_rows, tx_id=tx_id))
            return state

    def delete_transaction(
        self, tx_id: str, adjust_cash: bool = True, portfolio_id: str = "default"
    ) -> PortfolioState:
        pid = validate_portfolio_id(portfolio_id)
        with self.repo.unit_of_work(pid) as uow:
            state = uow.load_state()
            fpath = get_trades_log_filepath(pid)
            from tools.portfolio.adapters.markdown.repository_adapter import _read_and_migrate_trade_log_locked
            rows = _read_and_migrate_trade_log_locked(fpath)
            target_row = next((r for r in rows if r.get("Transaction_ID") == tx_id), None)
            if not target_row:
                raise ValueError(f"ไม่พบ transaction id '{tx_id}'")

            old_units = float(target_row.get("Units", 0))
            old_price = float(target_row.get("Price", 0))
            action = target_row.get("Action", "BUY").upper()
            sym = target_row.get("Symbol")
            ccy = target_row.get("Currency", "THB")

            if adjust_cash:
                cash = _require_cash(state, ccy)
                if action == "BUY":
                    cash.units += old_units * old_price
                elif action == "SELL":
                    cash.units -= old_units * old_price

            filtered_rows = [r for r in rows if r.get("Transaction_ID") != tx_id]
            sym_rows = [r for r in filtered_rows if r.get("Symbol") == sym]
            updated_sym_rows, final_units, final_avg, total_realized = _replay_symbol_trades(sym_rows, sym, ccy)

            row_map = {r["Transaction_ID"]: r for r in updated_sym_rows}
            new_rows = [row_map.get(r["Transaction_ID"], r) for r in filtered_rows]

            h = _find_holding(state, sym)
            if final_units > _FLOAT_EPS:
                if h is None:
                    h = Holding(symbol=sym, asset_type="Stock", units=final_units)
                    state.holdings.append(h)
                h.units = final_units
                h.status = "active"
                h.archived_at = None
                if ccy == "USD":
                    h.avg_cost_usd = final_avg
                else:
                    h.avg_cost_thb = final_avg
                if h.dividend_source == "synced":
                    h.dividend_source = None
            else:
                if h:
                    state.holdings = [item for item in state.holdings if item.symbol != sym]

            state.summary.total_realized_profit_ytd = round(
                sum(float(r.get("Realized_PnL_THB") or 0.0) for r in new_rows), _MONEY_DP
            )
            recalc_all(state)
            uow.commit(state, LedgerChange(kind="replace_all", rows=new_rows, tx_id=tx_id))
            return state

    # =========================================================================
    # 5. Watchlist Operations
    # =========================================================================

    def add_to_watchlist(
        self, symbol: str, asset_type: str, target_price: Optional[float] = None, notes: str = "", portfolio_id: str = "default"
    ) -> str:
        from filelock import Timeout
        from tools.tool_errors import LOCK_TIMEOUT
        clean_sym = symbol.strip().upper()
        if not clean_sym:
            return "Error: symbol ต้องไม่ว่าง"
        if target_price is not None and target_price <= 0:
            return "Error: target_price ต้องมากกว่า 0"

        pid = validate_portfolio_id(portfolio_id)
        try:
            state = self.watchlist_repo.load_watchlist(pid)
            item = next((it for it in state.items if it.symbol == clean_sym), None)
            is_update = item is not None
            if item:
                item.asset_type = asset_type
                if target_price is not None:
                    item.target_price = target_price
                if notes:
                    item.notes = notes
            else:
                state.items.append(
                    WatchlistItem(
                        symbol=clean_sym,
                        asset_type=asset_type,
                        target_price=target_price,
                        notes=notes or None,
                        added_date=_now_iso()[:10],
                    )
                )
            self.watchlist_repo.save_watchlist(state, pid)
            if is_update:
                return f"[WATCH UPD] อัปเดต {clean_sym} ใน Watchlist (target_price={target_price})"
            return f"[WATCH ADD] เพิ่ม {clean_sym} ({asset_type}) เข้า Watchlist (target_price={target_price})"
        except Timeout:
            return LOCK_TIMEOUT.format(detail=f"watchlist lock '{pid}'")
        except OSError:
            raise
        except Exception as e:
            return f"Error: {e}"

    def remove_from_watchlist(self, symbol: str, portfolio_id: str = "default") -> str:
        from filelock import Timeout
        from tools.tool_errors import LOCK_TIMEOUT
        clean_sym = symbol.strip().upper()
        if not clean_sym:
            return "Error: symbol ต้องไม่ว่าง"

        pid = validate_portfolio_id(portfolio_id)
        try:
            state = self.watchlist_repo.load_watchlist(pid)
            orig_len = len(state.items)
            state.items = [it for it in state.items if it.symbol != clean_sym]
            if len(state.items) == orig_len:
                return f"Error: ไม่พบ {clean_sym} ใน Watchlist"
            self.watchlist_repo.save_watchlist(state, pid)
            return f"[WATCH DEL] ลบ {clean_sym} ออกจาก Watchlist สำเร็จ (remaining: {len(state.items)})"
        except Timeout:
            return LOCK_TIMEOUT.format(detail=f"watchlist lock '{pid}'")
        except Exception as e:
            return f"Error: {e}"

    def read_watchlist(self, portfolio_id: str = "default") -> str:
        from filelock import Timeout
        pid = validate_portfolio_id(portfolio_id)
        try:
            state = self.get_structured_watchlist(portfolio_id=pid)
            items_list = [it.model_dump(exclude_none=True) for it in state.items]
            return json.dumps({"n_items": len(items_list), "items": items_list}, ensure_ascii=False, indent=2)
        except Timeout:
            return json.dumps({"error": f"watchlist lock timeout for '{pid}'"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def get_structured_watchlist(self, portfolio_id: str = "default") -> WatchlistState:
        return self.watchlist_repo.load_watchlist(portfolio_id)

    def structured_upsert_watchlist_item(
        self, symbol: str, asset_type: str, target_price: Optional[float] = None, notes: str = "", portfolio_id: str = "default"
    ) -> WatchlistState:
        clean_sym = symbol.strip().upper()
        if not clean_sym:
            raise ValueError("symbol ต้องไม่ว่าง")
        if target_price is not None and target_price <= 0:
            raise ValueError("target_price ต้องมากกว่า 0")
        state = self.watchlist_repo.load_watchlist(portfolio_id)
        item = next((it for it in state.items if it.symbol == clean_sym), None)
        if item:
            item.asset_type = asset_type
            if target_price is not None:
                item.target_price = target_price
            if notes:
                item.notes = notes
        else:
            state.items.append(
                WatchlistItem(
                    symbol=clean_sym,
                    asset_type=asset_type,
                    target_price=target_price,
                    notes=notes or None,
                    added_date=_now_iso()[:10],
                )
            )
        self.watchlist_repo.save_watchlist(state, portfolio_id)
        return state

    def structured_remove_watchlist_item(self, symbol: str, portfolio_id: str = "default") -> WatchlistState:
        clean_sym = symbol.strip().upper()
        if not clean_sym:
            raise ValueError("symbol ต้องไม่ว่าง")
        state = self.watchlist_repo.load_watchlist(portfolio_id)
        orig_len = len(state.items)
        state.items = [it for it in state.items if it.symbol != clean_sym]
        if len(state.items) == orig_len:
            raise ValueError(f"ไม่พบ {clean_sym} ใน Watchlist")
        self.watchlist_repo.save_watchlist(state, portfolio_id)
        return state

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
        try:
            self.structured_upsert_goal(
                name=name,
                goal_type=goal_type,
                target_amount_thb=target_amount_thb,
                deadline=deadline,
                years_from_now=years_from_now,
                notes=notes,
                portfolio_id=portfolio_id,
                bucket_id=bucket_id,
            )
            return f"[GOAL] บันทึกเป้าหมาย '{name}' {target_amount_thb:,.2f} THB สำเร็จ"
        except Exception as e:
            return f"Error: {e}"

    def remove_goal(self, name: str) -> str:
        try:
            self.structured_remove_goal(name=name)
            return f"[GOAL] ลบเป้าหมาย '{name}' สำเร็จ"
        except Exception as e:
            return f"Error: {e}"

    def get_goals_progress(self, portfolio_id: str = "default") -> str:
        goals = self.get_structured_goals(portfolio_id=portfolio_id)
        return json.dumps(goals, ensure_ascii=False, indent=2)

    def get_structured_goals(self, portfolio_id: Optional[str] = None) -> List[Dict]:
        state = self.goals_repo.load_goals(portfolio_id=portfolio_id)
        return [g.model_dump(exclude_none=True) for g in state.goals]

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
        state = self.goals_repo.load_goals(portfolio_id=None)
        clean_name = name.strip()
        existing = next((g for g in state.goals if g.name == clean_name and g.portfolio_id == portfolio_id), None)
        if existing:
            existing.goal_type = goal_type
            existing.target_amount_thb = target_amount_thb
            existing.deadline = deadline
            existing.years_from_now = years_from_now
            existing.notes = notes
            existing.bucket_id = bucket_id
        else:
            state.goals.append(
                GoalItem(
                    name=clean_name,
                    goal_type=goal_type,
                    target_amount_thb=target_amount_thb,
                    deadline=deadline,
                    years_from_now=years_from_now,
                    notes=notes,
                    created_date=_now_iso()[:10],
                    portfolio_id=portfolio_id,
                    bucket_id=bucket_id,
                )
            )
        self.goals_repo.save_goals(state)
        return self.get_structured_goals(portfolio_id=portfolio_id)

    def structured_remove_goal(self, name: str, portfolio_id: Optional[str] = None) -> List[Dict]:
        clean_name = name.strip()
        state = self.goals_repo.load_goals(portfolio_id=None)
        state.goals = [
            g for g in state.goals if not (g.name == clean_name and (portfolio_id is None or g.portfolio_id == portfolio_id))
        ]
        self.goals_repo.save_goals(state)
        return self.get_structured_goals(portfolio_id=portfolio_id)

    # =========================================================================
    # 7. Journal Operations
    # =========================================================================

    def append_trading_journal(self, entry: str, portfolio_id: str = "default") -> str:
        try:
            self.journal_provider.append_journal(entry, portfolio_id=portfolio_id)
            return f"[JOURNAL] บันทึกสำเร็จ | [{_now_iso()}] | {len(entry)} chars"
        except Exception as e:
            return f"Error: {e}"

    def read_trading_journal(
        self, days: int = 30, keyword: Optional[str] = None, limit: int = 20, portfolio_id: str = "default"
    ) -> str:
        entries = self.get_structured_journal(days=days, keyword=keyword, limit=limit, portfolio_id=portfolio_id)
        if not entries:
            return "ไม่พบบันทึกการเทรดตามเงื่อนไข"
        lines = []
        for e in entries:
            lines.append(f"## [{e.get('timestamp')}]\n\n{e.get('content')}\n")
        return "\n".join(lines)

    def get_structured_journal(
        self, days: Optional[int] = 365, keyword: Optional[str] = None, limit: int = 100, portfolio_id: str = "default"
    ) -> List[Dict]:
        return self.journal_provider.read_journal(days=days, keyword=keyword, limit=limit, portfolio_id=portfolio_id)

    def structured_append_journal(self, entry: str, portfolio_id: str = "default") -> List[Dict]:
        return self.journal_provider.append_journal(entry, portfolio_id=portfolio_id)

    # =========================================================================
    # 8. Performance Operations
    # =========================================================================

    def record_performance_snapshot(self, refresh_prices: bool = True, portfolio_id: str = "default") -> str:
        try:
            pid = validate_portfolio_id(portfolio_id)
            state = self.get_structured_portfolio_state(refresh_prices=refresh_prices, portfolio_id=pid)
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

    # =========================================================================
    # 9. Prices & Dividends Operations
    # =========================================================================

    def fetch_fx_rate(
        self, date_str: Optional[str] = None, fallback_rate: Optional[float] = None
    ) -> Tuple[float, Literal["historical", "live", "fallback"]]:
        return self.price_provider.fetch_fx_rate(date_str=date_str, fallback_rate=fallback_rate)

    def fetch_latest_price(self, symbol: str, currency: Literal["THB", "USD"] = "THB") -> Optional[float]:
        return self.price_provider.fetch_price(symbol, currency)

    def sync_dividends_from_history(self, portfolio_id: str = "default") -> Dict:
        """Sync dividend payouts and history for holdings."""
        pid = validate_portfolio_id(portfolio_id)
        from tools.portfolio.dividends import sync_dividends_from_history as _div_sync
        return _div_sync(portfolio_id=pid)
