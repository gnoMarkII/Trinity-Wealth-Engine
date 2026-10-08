import time
import uuid
from decimal import Decimal
from typing import Optional, Dict, List

from core.logger import get_logger
from tools.portfolio.domain.constants import _FLOAT_EPS, _MONEY_DP
from tools.portfolio.domain.models import PortfolioState, Holding, _now_iso
from tools.portfolio.domain.ledger_change import LedgerChange
from tools.portfolio.domain.events import SystemJournalEvent
from tools.portfolio.domain.mutation import PortfolioMutation
from tools.portfolio.domain.calculations import recalc_all, _replay_symbol_trades
from tools.portfolio.domain.validator import validate_portfolio_id
from tools.portfolio.ports.repository_port import PortfolioRepositoryPort
from tools.portfolio.ports.journal_port import TradeJournalPort
from ._mutation_commit import commit_mutation

log = get_logger(__name__)


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

    def __init__(
        self,
        repo: PortfolioRepositoryPort,
        journal_provider: Optional[TradeJournalPort] = None,
    ) -> None:
        self.repo = repo
        self.journal_provider = journal_provider

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
            commit_mutation(
                uow,
                state,
                PortfolioMutation(
                    ledger_change=LedgerChange(kind="replace_all", rows=rows, tx_id=tx_id),
                    system_journal_events=[
                        SystemJournalEvent.from_entry(
                            event_type="trade_note_updated",
                            message=f"**[TRADE NOTE UPDATED]** {tx_id}",
                            metadata={"transaction_id": tx_id},
                        )
                    ],
                ),
                journal_provider=self.journal_provider,
                portfolio_id=pid,
            )
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

            # Check if this is a Dime import and caller is attempting to edit economic fields
            is_dime = str(target_row.get("Source") or "").strip().upper() == "DIME"
            if is_dime and any(x is not None for x in (timestamp, units, price, fx_rate)):
                raise ValueError(
                    "รายการที่นำเข้าจาก Dime ไม่สามารถแก้ไขตัวเลขได้โดยตรง "
                    "หากต้องการแก้ไขให้ทำการ Void รายการนี้แล้วนำเข้าฉบับใหม่ (อนุญาตเฉพาะแก้ไข Notes เท่านั้น)"
                )

            old_units = float(target_row.get("Units") or target_row.get("units") or 0)
            old_price = float(target_row.get("Price") or target_row.get("price") or 0)
            action = str(target_row.get("Action") or target_row.get("action") or "BUY").upper()

            changed_fields = []
            if timestamp is not None:
                target_row["Timestamp"] = timestamp
                changed_fields.append("timestamp")
            if units is not None:
                target_row["Units"] = f"{units:g}"
                changed_fields.append("units")
            if price is not None:
                target_row["Price"] = f"{price:.2f}"
                changed_fields.append("price")
            if fx_rate is not None:
                target_row["FX_Rate"] = f"{fx_rate:.4f}"
                changed_fields.append("fx_rate")
            if notes is not None:
                target_row["Notes"] = notes
                changed_fields.append("notes")

            new_units_val = float(target_row.get("Units") or target_row.get("units") or 0)
            new_price_val = float(target_row.get("Price") or target_row.get("price") or 0)

            if units is not None or price is not None:
                if target_row.get("Net_Amount") is not None or target_row.get("net_amount") is not None:
                    target_row["Net_Amount"] = f"{new_units_val * new_price_val:.2f}"
                if target_row.get("Gross_Amount") is not None or target_row.get("gross_amount") is not None:
                    target_row["Gross_Amount"] = f"{new_units_val * new_price_val:.2f}"


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
            commit_mutation(
                uow,
                state,
                PortfolioMutation(
                    ledger_change=LedgerChange(kind="replace_all", rows=new_rows, tx_id=tx_id),
                    system_journal_events=[
                        SystemJournalEvent.from_entry(
                            event_type="transaction_edited",
                            message=f"**[TRANSACTION EDITED]** {sym} ({tx_id})",
                            metadata={"transaction_id": tx_id, "symbol": sym, "fields": changed_fields},
                        )
                    ],
                ),
                journal_provider=self.journal_provider,
                portfolio_id=pid,
            )
            return state

    # ------------------------------------------------------------------
    # Void Transaction (Idempotent Non-Destructive Reversal)
    # ------------------------------------------------------------------

    def void_transaction(
        self, tx_id: str, portfolio_id: str = "default", adjust_cash: Optional[bool] = None
    ) -> PortfolioState:
        """Non-destructive idempotent void/reversal strictly mirroring original transaction."""
        pid = validate_portfolio_id(portfolio_id)
        with self.repo.unit_of_work(pid) as uow:
            state = uow.load_state()
            rows = uow.read_trade_log_locked()
            target_row = next(
                (r for r in rows if (r.get("Transaction_ID") or r.get("transaction_id")) == tx_id),
                None,
            )
            if not target_row:
                raise ValueError(f"ไม่พบ transaction id '{tx_id}'")

            action = str(target_row.get("Action") or target_row.get("action") or "BUY").upper()
            rel_tx_id = str(target_row.get("Related_Transaction_ID") or target_row.get("related_transaction_id") or "")

            # 1. Guard against voiding an existing reversal row
            if action.startswith("VOID_") or action == "REVERSAL" or rel_tx_id:
                raise ValueError("ไม่สามารถยกเลิกรายการที่เป็น Reversal หรือ Void ได้")

            # 2. Idempotency check: Has this transaction already been voided?
            existing_reversal = next(
                (r for r in rows if str(r.get("Related_Transaction_ID") or r.get("related_transaction_id") or "") == tx_id),
                None,
            )
            if existing_reversal:
                log.info(
                    "Transaction %s already voided by reversal %s. Idempotent return.",
                    tx_id,
                    existing_reversal.get("Transaction_ID"),
                )
                return state

            # 3. Strict Mirror Invariant: Reversal strictly mirrors target_row["Cash_Adjusted"]
            # Ignore any caller adjust_cash override to maintain ledger consistency
            effective_cash_adjusted = str(target_row.get("Cash_Adjusted") or "YES").strip().upper() == "YES"

            sym = target_row.get("Symbol") or target_row.get("symbol")
            ccy = target_row.get("Currency") or target_row.get("currency") or "THB"

            # Determine net cash amount to reverse
            net_amt_raw = target_row.get("Net_Amount")
            if net_amt_raw is not None and str(net_amt_raw).strip() != "":
                try:
                    net_cash_amount = float(Decimal(str(net_amt_raw)))
                except Exception:
                    units_val = float(target_row.get("Units") or 0)
                    price_val = float(target_row.get("Price") or 0)
                    net_cash_amount = units_val * price_val
            else:
                units_val = float(target_row.get("Units") or 0)
                price_val = float(target_row.get("Price") or 0)
                net_cash_amount = units_val * price_val

            if effective_cash_adjusted:
                cash = _require_cash(state, ccy)
                if action == "BUY":
                    cash.units += net_cash_amount
                elif action == "SELL":
                    cash.units -= net_cash_amount

            # 4. Create Reversal Row mirroring economic fields
            rev_tx_id = f"tx_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
            reversal_action = f"VOID_{action}"
            now_str = _now_iso().replace("T", " ")

            reversal_row = dict(target_row)
            reversal_row.update({
                "Transaction_ID": rev_tx_id,
                "Timestamp": now_str,
                "Action": reversal_action,
                "Related_Transaction_ID": tx_id,
                "Cost_THB": "0.00",
                "Realized_PnL_THB": "0.00",
                "Notes": f"[VOID] Reversal of {tx_id}" + (f" | {target_row.get('Notes')}" if target_row.get('Notes') else ""),
                "Cash_Adjusted": "YES" if effective_cash_adjusted else "NO",
            })

            new_rows = list(rows) + [reversal_row]

            sym_rows = [r for r in new_rows if (r.get("Symbol") or r.get("symbol")) == sym]
            updated_sym_rows, final_units, final_avg, total_realized = _replay_symbol_trades(sym_rows, sym, ccy)

            row_map = {r.get("Transaction_ID") or r.get("transaction_id"): r for r in updated_sym_rows}
            final_rows = [row_map.get(r.get("Transaction_ID") or r.get("transaction_id"), r) for r in new_rows]

            h = _find_holding(state, sym)
            if final_units > _FLOAT_EPS:
                if h is None:
                    h = Holding(symbol=sym, asset_type=target_row.get("Asset_Type") or "Stock", units=final_units)
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
                sum(float(r.get("Realized_PnL_THB") or r.get("realized_pnl_thb") or 0.0) for r in final_rows),
                _MONEY_DP,
            )
            recalc_all(state)

            commit_mutation(
                uow,
                state,
                PortfolioMutation(
                    ledger_change=LedgerChange(kind="replace_all", rows=final_rows, tx_id=rev_tx_id),
                    system_journal_events=[
                        SystemJournalEvent.from_entry(
                            event_type="transaction_voided",
                            message=f"**[TRANSACTION VOIDED]** {sym} ({tx_id}) reversed by {rev_tx_id}",
                            metadata={"transaction_id": tx_id, "reversal_id": rev_tx_id, "symbol": sym},
                        )
                    ],
                ),
                journal_provider=self.journal_provider,
                portfolio_id=pid,
            )
            return state

    # ------------------------------------------------------------------
    # Delete Transaction (Retired Hard Delete -> Delegates to void_transaction)
    # ------------------------------------------------------------------

    def delete_transaction(
        self, tx_id: str, adjust_cash: bool = True, portfolio_id: str = "default"
    ) -> PortfolioState:
        """Retires hard delete and delegates to void_transaction (preserving audit trail)."""
        return self.void_transaction(tx_id=tx_id, portfolio_id=portfolio_id)
