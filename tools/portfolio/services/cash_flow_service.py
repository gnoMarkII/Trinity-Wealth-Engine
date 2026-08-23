"""PortfolioCashFlowService — Deposit/Withdraw, Record Income, FX, Dividends."""
from typing import Optional, Literal, Tuple

from tools.portfolio.domain.constants import _FLOAT_EPS, _MONEY_DP, CASH_THB_SYMBOL, CASH_USD_SYMBOL, _CASH_SYMBOLS
from tools.portfolio.domain.models import PortfolioState, Holding, _now_iso
from tools.portfolio.domain.ledger_change import LedgerChange
from tools.portfolio.domain.calculations import recalc_all
from tools.portfolio.domain.validator import validate_portfolio_id
from tools.portfolio.ports.repository_port import PortfolioRepositoryPort
from tools.portfolio.ports.price_port import MarketPricePort
from tools.portfolio.ports.journal_port import TradeJournalPort
from tools.portfolio.ports.dividend_port import DividendHistoryPort


def _find_holding(state: PortfolioState, symbol: str):
    return next((h for h in state.holdings if h.symbol == symbol), None)


def _require_cash(state: PortfolioState, currency: Literal["THB", "USD"] = "THB"):
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


class PortfolioCashFlowService:
    """Handles Deposit, Withdraw, Income Recording, FX updates, and Dividends sync."""

    def __init__(
        self,
        repo: PortfolioRepositoryPort,
        price_provider: MarketPricePort,
        dividend_provider: Optional[DividendHistoryPort] = None,
        journal_provider: Optional[TradeJournalPort] = None,
    ) -> None:
        self.repo = repo
        self.price_provider = price_provider
        self.dividend_provider = dividend_provider
        self.journal_provider = journal_provider

    def _append_journal(self, content: str, date_str: Optional[str] = None, portfolio_id: str = "default") -> None:
        if self.journal_provider:
            self.journal_provider.append_system_entry(content, date_str=date_str, portfolio_id=portfolio_id)
        else:
            from tools.portfolio.journal import _write_journal_entry
            _write_journal_entry(content, date_str=date_str, portfolio_id=portfolio_id)

    # ------------------------------------------------------------------
    # Cash Flow (Deposit / Withdraw)
    # ------------------------------------------------------------------

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
                    raise ValueError(
                        f"Insufficient cash in CASH_{currency} (requested: {amount:,.2f}, available: {cash.units:,.2f})"
                    )
                cash.units -= amount
            else:
                cash.units += amount

            recalc_all(state)
            uow.commit(state, LedgerChange(kind="unchanged"))
            res_str = (
                f"[{action.upper()}] {currency} | "
                f"{'+' if action == 'deposit' else '-'}{amount:,.2f} {currency} | "
                f"CASH_{currency} คงเหลือ: {cash.units:,.2f} {currency}"
            )

            content = f"**[CASH FLOW NOTE]** {action.upper()} {amount:,.2f} {currency}"
            if notes and notes.strip():
                content += f" — {notes.strip()}"
            self._append_journal(content, date_str=date, portfolio_id=pid)

            return res_str, state

    # ------------------------------------------------------------------
    # Income Recording
    # ------------------------------------------------------------------

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

            state.summary.passive_income_ytd = round(state.summary.passive_income_ytd + amount_thb, _MONEY_DP)
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
            res_str = (
                f"[{tag}] +{amount_thb:,.2f} THB{src_str} | "
                f"passive_income_ytd: {state.summary.passive_income_ytd:,.2f} | "
                f"เงินสดคงเหลือ: {cash.units:,.2f} บาท"
            )

            src_note = f" ({clean_src})" if clean_src else ""
            content = f"**[INCOME NOTE - {income_type}{src_note}]** +{amount_thb:,.2f} THB"
            if notes and notes.strip():
                content += f" — {notes.strip()}"
            self._append_journal(content, date_str=date, portfolio_id=pid)

            return res_str, state

    # ------------------------------------------------------------------
    # FX & Dividends
    # ------------------------------------------------------------------

    def fetch_fx_rate(
        self, date_str: Optional[str] = None, fallback_rate: Optional[float] = None
    ) -> Tuple[float, Literal["historical", "live", "fallback"]]:
        return self.price_provider.fetch_fx_rate(date_str=date_str, fallback_rate=fallback_rate)

    def fetch_latest_price(self, symbol: str, currency: Literal["THB", "USD"] = "THB") -> Optional[float]:
        return self.price_provider.fetch_price(symbol, currency)

    def sync_dividends_from_history(self, portfolio_id: str = "default"):
        pid = validate_portfolio_id(portfolio_id)
        from tools.portfolio.dividends import sync_dividends_from_history as _div_sync
        return _div_sync(portfolio_id=pid)
