"""PortfolioLedgerService — Edit/Delete Transactions, Replay PnL."""
from typing import Optional, Dict

from tools.portfolio.domain.constants import _FLOAT_EPS, _MONEY_DP
from tools.portfolio.domain.models import PortfolioState, Holding
from tools.portfolio.domain.ledger_change import LedgerChange
from tools.portfolio.domain.calculations import recalc_all, _replay_symbol_trades
from tools.portfolio.domain.validator import validate_portfolio_id
from tools.portfolio.ports.repository_port import PortfolioRepositoryPort


def _find_holding(state: PortfolioState, symbol: str):
    return next((h for h in state.holdings if h.symbol == symbol), None)


def _require_cash(state: PortfolioState, currency: str):
    from tools.portfolio.domain.constants import CASH_THB_SYMBOL, CASH_USD_SYMBOL
    sym = CASH_THB_SYMBOL if currency == "THB" else CASH_USD_SYMBOL
    cash = _find_holding(state, sym)
    if cash is None:
        cash = Holding(symbol=sym, asset_type="Cash", units=0.0, market_value_thb=0.0)
        state.holdings.append(cash)
    return cash


class PortfolioLedgerService:
    """Handles Trade Log reads, edit/delete transaction with PnL replay."""

    def __init__(self, repo: PortfolioRepositoryPort) -> None:
        self.repo = repo

    # ------------------------------------------------------------------
    # Trade Log Read
    # ------------------------------------------------------------------

    def get_structured_trades_log(self, portfolio_id: str = "default", symbol: Optional[str] = None):
        return self.repo.read_trade_log(portfolio_id, symbol=symbol)

    def update_trade_note(self, tx_id: str, notes: str, portfolio_id: str = "default") -> Dict:
        pid = validate_portfolio_id(portfolio_id)
        with self.repo.unit_of_work(pid) as uow:
            state = uow.load_state()
            rows = uow.read_trade_log_locked()
            found = False
            updated_row: Dict = {}
            raw_notes = (notes or "").strip()
            clean_notes = "'" + raw_notes if raw_notes and raw_notes[0] in ("=", "+", "-", "@", "\t", "\r") else raw_notes
            for r in rows:
                if r.get("Transaction_ID") == tx_id or r.get("transaction_id") == tx_id:
                    r["Notes"] = clean_notes
                    r["notes"] = clean_notes
                    found = True
                    item_dict = {k.lower(): v for k, v in r.items()}
                    item_dict.update({k: v for k, v in r.items()})
                    updated_row = item_dict
                    break
            if not found:
                raise ValueError(f"ไม่พบ transaction id '{tx_id}'")
            uow.commit(state, LedgerChange(kind="replace_all", rows=rows, tx_id=tx_id))
            return updated_row

    # ------------------------------------------------------------------
    # Edit Transaction
    # ------------------------------------------------------------------

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
            rows = uow.read_trade_log_locked()
            target_row = next(
                (r for r in rows if r.get("Transaction_ID") == tx_id or r.get("transaction_id") == tx_id),
                None,
            )
            if not target_row:
                raise ValueError(f"ไม่พบ transaction id '{tx_id}'")

            old_units = float(target_row.get("Units") or target_row.get("units") or 0)
            old_price = float(target_row.get("Price") or target_row.get("price") or 0)
            action = str(target_row.get("Action") or target_row.get("action") or "BUY").upper()

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

            new_units_val = float(target_row.get("Units") or target_row.get("units") or 0)
            new_price_val = float(target_row.get("Price") or target_row.get("price") or 0)

            sym = target_row.get("Symbol") or target_row.get("symbol")
            ccy = target_row.get("Currency") or target_row.get("currency") or "THB"
            sym_rows = [r for r in rows if (r.get("Symbol") or r.get("symbol")) == sym]
            updated_sym_rows, final_units, final_avg, total_realized = _replay_symbol_trades(sym_rows, sym, ccy)

            row_map = {r.get("Transaction_ID") or r.get("transaction_id"): r for r in updated_sym_rows}
            new_rows = [row_map.get(r.get("Transaction_ID") or r.get("transaction_id"), r) for r in rows]

            if adjust_cash:
                cash = _require_cash(state, ccy)
                if action == "BUY":
                    delta_cash = -(new_units_val * new_price_val) - (-(old_units * old_price))
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
                sum(float(r.get("Realized_PnL_THB") or r.get("realized_pnl_thb") or 0.0) for r in new_rows),
                _MONEY_DP,
            )
            recalc_all(state)
            uow.commit(state, LedgerChange(kind="replace_all", rows=new_rows, tx_id=tx_id))
            return state

    # ------------------------------------------------------------------
    # Delete Transaction
    # ------------------------------------------------------------------

    def delete_transaction(
        self, tx_id: str, adjust_cash: bool = True, portfolio_id: str = "default"
    ) -> PortfolioState:
        pid = validate_portfolio_id(portfolio_id)
        with self.repo.unit_of_work(pid) as uow:
            state = uow.load_state()
            rows = uow.read_trade_log_locked()
            target_row = next(
                (r for r in rows if r.get("Transaction_ID") == tx_id or r.get("transaction_id") == tx_id),
                None,
            )
            if not target_row:
                raise ValueError(f"ไม่พบ transaction id '{tx_id}'")

            old_units = float(target_row.get("Units") or target_row.get("units") or 0)
            old_price = float(target_row.get("Price") or target_row.get("price") or 0)
            action = str(target_row.get("Action") or target_row.get("action") or "BUY").upper()
            sym = target_row.get("Symbol") or target_row.get("symbol")
            ccy = target_row.get("Currency") or target_row.get("currency") or "THB"

            if adjust_cash:
                cash = _require_cash(state, ccy)
                if action == "BUY":
                    cash.units += old_units * old_price
                elif action == "SELL":
                    cash.units -= old_units * old_price

            filtered_rows = [
                r for r in rows if (r.get("Transaction_ID") or r.get("transaction_id")) != tx_id
            ]
            sym_rows = [r for r in filtered_rows if (r.get("Symbol") or r.get("symbol")) == sym]
            updated_sym_rows, final_units, final_avg, total_realized = _replay_symbol_trades(sym_rows, sym, ccy)

            row_map = {r.get("Transaction_ID") or r.get("transaction_id"): r for r in updated_sym_rows}
            new_rows = [row_map.get(r.get("Transaction_ID") or r.get("transaction_id"), r) for r in filtered_rows]

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
                sum(float(r.get("Realized_PnL_THB") or r.get("realized_pnl_thb") or 0.0) for r in new_rows),
                _MONEY_DP,
            )
            recalc_all(state)
            uow.commit(state, LedgerChange(kind="replace_all", rows=new_rows, tx_id=tx_id))
            return state
