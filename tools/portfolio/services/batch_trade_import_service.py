"""Batch Trade Import Service (Hexagonal Architecture).

Coordinates ingestion of TradeImportItem batches into the authoritative ledger and portfolio state
under an atomic Unit of Work. Enforces SHA-256 fingerprint deduplication, intra-batch checks,
and cash adjustment by Net_Amount.
"""
import time
import uuid
from decimal import Decimal
from typing import List, Optional, Set, Tuple

from core.logger import get_logger
from tools.portfolio.domain.calculations import _replay_symbol_trades, recalc_all
from tools.portfolio.domain.constants import _FLOAT_EPS, _MONEY_DP
from tools.portfolio.domain.errors import TradeDuplicateError, TradeReconciliationError
from tools.portfolio.domain.events import SystemJournalEvent
from tools.portfolio.domain.ledger_change import LedgerChange
from tools.portfolio.domain.models import PortfolioState, Holding, TradeImportItem
from tools.portfolio.domain.mutation import PortfolioMutation
from tools.portfolio.domain.validator import validate_portfolio_id
from tools.portfolio.ports.journal_port import TradeJournalPort
from tools.portfolio.ports.repository_port import PortfolioRepositoryPort
from ._mutation_commit import commit_mutation

log = get_logger(__name__)


def _find_holding(state: PortfolioState, symbol: str) -> Optional[Holding]:
    return next((h for h in state.holdings if h.symbol == symbol), None)


def _require_cash(state: PortfolioState, currency: str) -> Holding:
    cash_sym = f"CASH_{currency.upper()}"
    h = _find_holding(state, cash_sym)
    if not h:
        h = Holding(symbol=cash_sym, asset_type="Cash", units=0.0, market_value_thb=0.0)
        state.holdings.append(h)
    return h


class BatchTradeImportService:
    """Service for batch-importing external trades into the authoritative ledger."""

    def __init__(
        self,
        repo: PortfolioRepositoryPort,
        journal: Optional[TradeJournalPort] = None,
        journal_provider: Optional[TradeJournalPort] = None,
    ):
        self.repo = repo
        self.journal = journal or journal_provider
        self.journal_provider = self.journal



    def execute_batch_import(
        self,
        items: List[TradeImportItem],
        portfolio_id: str = "default",
    ) -> PortfolioState:
        """Atomic batch import with Order-ID identity verification, conflict detection, and deduplication."""
        pid = validate_portfolio_id(portfolio_id)
        if not items:
            return self.repo.load_state(pid)

        with self.repo.unit_of_work(pid) as uow:
            state = uow.load_state()
            existing_rows = uow.read_trade_log_locked()

            # 1. Map existing transactions by Fingerprint and (Confirmation_No, Order_ID)
            existing_fps = {
                str(r.get("Fingerprint") or "").strip()
                for r in existing_rows
                if r.get("Fingerprint")
            }
            existing_identity_map: dict = {}
            for r in existing_rows:
                c_no = str(r.get("Confirmation_No") or "").strip()
                o_id = str(r.get("Order_ID") or "").strip()
                if c_no and o_id:
                    existing_identity_map[(c_no, o_id)] = r

            # Legacy natural key map for rows migrated from old ledgers without Order_ID
            legacy_existing_keys = {
                (
                    str(r.get("Confirmation_No") or "").strip(),
                    str(r.get("Symbol") or "").strip().upper(),
                    str(r.get("Action") or "").strip().upper(),
                    str(r.get("Units") or "").strip(),
                    str(r.get("Price") or "").strip(),
                    str(r.get("Timestamp") or "")[:10].strip(),
                )
                for r in existing_rows
                if r.get("Confirmation_No") and not str(r.get("Order_ID") or "").strip()
            }

            # 2. Check for intra-batch duplicates and conflicts
            batch_identity_map: dict = {}
            batch_fps: Set[str] = set()
            valid_items: List[TradeImportItem] = []

            for item in items:
                c_no = item.confirmation_no.strip()
                o_id = (item.order_id or "").strip()

                if not o_id and item.source.upper() in ("DIME", "WEALTHX"):
                    src_label = "WealthX" if item.source.upper() == "WEALTHX" else "Dime"
                    raise ValueError(f"รายการ {item.symbol} ขาด Order ID ไม่สามารถระบุตัวตนของธุรกรรมได้ (ห้ามใช้ fallback line_index สำหรับ {src_label})")

                identity_key = (c_no, o_id) if o_id else None

                # Intra-batch conflict and deduplication
                if identity_key and identity_key in batch_identity_map:
                    prev = batch_identity_map[identity_key]
                    # Check for financial conflict on identical order ID
                    if (
                        prev.symbol.strip().upper() != item.symbol.strip().upper()
                        or prev.action.strip().upper() != item.action.strip().upper()
                        or prev.units != item.units
                        or prev.price != item.price
                        or prev.net_amount != item.net_amount
                        or prev.fees.commission != item.fees.commission
                    ):
                        raise TradeReconciliationError(
                            f"ตรวจพบ Conflict: มีคำสั่งซื้อขายซ้ำกันในเอกสาร (Confirmation: {c_no}, Order: {o_id}) แต่มียอดเงินหรือค่าธรรมเนียมขัดแย้งกัน"
                        )
                    # Exact duplicate across emails -> consolidate safely
                    log.info("Consolidating intra-batch duplicate trade %s (Conf: %s, Order: %s)", item.symbol, c_no, o_id)
                    continue

                # Cross-check against existing ledger rows
                if identity_key and identity_key in existing_identity_map:
                    ex = existing_identity_map[identity_key]
                    ex_sym = str(ex.get("Symbol") or "").strip().upper()
                    ex_act = str(ex.get("Action") or "").strip().upper()
                    ex_units = str(ex.get("Units") or "").strip()
                    ex_price = str(ex.get("Price") or "").strip()
                    ex_net = str(ex.get("Net_Amount") or "").strip()

                    if (
                        ex_sym != item.symbol.strip().upper()
                        or ex_act != item.action.strip().upper()
                        or (ex_units and ex_units != f"{item.units:g}")
                        or (ex_price and float(ex_price) != float(item.price))
                        or (ex_net and float(ex_net) != float(item.net_amount))
                    ):
                        raise TradeReconciliationError(
                            f"ตรวจพบ Conflict กับ Ledger เดิม: คำสั่งซื้อขาย (Confirmation: {c_no}, Order: {o_id}) มีใน Ledger แล้วแต่ตัวเลขขัดแย้งกัน"
                        )
                    log.info("Skipping already imported identical trade %s (Conf: %s, Order: %s)", item.symbol, c_no, o_id)
                    continue

                # Check if fingerprint or legacy natural key already present
                if item.fingerprint in existing_fps:
                    log.info("Skipping already imported item %s (Conf: %s, FP: %s)", item.symbol, item.confirmation_no, item.fingerprint)
                    continue

                legacy_k = (
                    c_no,
                    item.symbol.strip().upper(),
                    item.action.strip().upper(),
                    str(item.units).strip(),
                    str(item.price).strip(),
                    item.trade_date[:10].strip(),
                )
                if not o_id and legacy_k in legacy_existing_keys:
                    log.info("Skipping already imported legacy item %s (Conf: %s)", item.symbol, item.confirmation_no)
                    continue

                if identity_key:
                    batch_identity_map[identity_key] = item
                batch_fps.add(item.fingerprint)
                valid_items.append(item)

            if not valid_items:
                log.info("All %d items in batch are duplicates or already imported.", len(items))
                return state

            new_ledger_rows = []
            touched_symbols: Set[str] = set()

            for item in valid_items:
                tx_id = f"tx_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
                ts_str = f"{item.trade_date} 12:00:00" if len(item.trade_date) <= 10 else item.trade_date
                clean_sym = item.symbol.strip().upper()
                touched_symbols.add(clean_sym)

                price_str = f"{item.price:f}".rstrip("0").rstrip(".") if "." in f"{item.price:f}" else f"{item.price:f}"
                row_dict = {
                    "Transaction_ID": tx_id,
                    "Timestamp": ts_str,
                    "Symbol": clean_sym,
                    "Action": item.action.upper(),
                    "Units": f"{item.units:g}",
                    "Price": price_str,
                    "Currency": item.currency.upper(),
                    "FX_Rate": f"{item.exchange_rate:.4f}" if item.exchange_rate else "",
                    "Cost_THB": "",
                    "Realized_PnL_THB": "",
                    "Notes": f"{item.source} confirmation {item.confirmation_no}" if item.source else f"Confirmation {item.confirmation_no}",
                    "Gross_Amount": f"{item.gross_amount:.2f}",
                    "Commission": f"{item.fees.commission:.2f}",
                    "VAT": f"{item.fees.vat:.2f}",
                    "Other_Fees": f"{item.fees.other_fees:.2f}",
                    "Net_Amount": f"{item.net_amount:.2f}",
                    "Fee_Currency": item.fees.fee_currency.upper(),
                    "Confirmation_No": item.confirmation_no,
                    "Order_ID": item.order_id or "",
                    "Settlement_Date": item.settlement_date or "",
                    "Source": item.source,
                    "Fingerprint": item.fingerprint,
                    "Cash_Adjusted": "YES" if item.cash_adjusted else "NO",
                    "Related_Transaction_ID": "",
                }
                new_ledger_rows.append(row_dict)

                # Cash adjustment by Net_Amount (incorporating fees)
                if item.cash_adjusted:
                    cash = _require_cash(state, item.currency)
                    net_cash = float(item.net_amount)
                    if item.action.upper() == "BUY":
                        cash.units -= net_cash
                    elif item.action.upper() == "SELL":
                        cash.units += net_cash

            all_ledger_rows = list(existing_rows) + new_ledger_rows

            # Replay all trades for touched symbols
            for sym in touched_symbols:
                sym_rows = [r for r in all_ledger_rows if (r.get("Symbol") or r.get("symbol")) == sym]
                ccy = sym_rows[0].get("Currency") or sym_rows[0].get("currency") or "THB"
                updated_sym_rows, final_units, final_avg, total_realized = _replay_symbol_trades(sym_rows, sym, ccy)

                # Update row map
                row_map = {r.get("Transaction_ID") or r.get("transaction_id"): r for r in updated_sym_rows}
                all_ledger_rows = [row_map.get(r.get("Transaction_ID") or r.get("transaction_id"), r) for r in all_ledger_rows]

                h = _find_holding(state, sym)
                item_asset_type = next((getattr(it, "asset_type", None) for it in valid_items if it.symbol.strip().upper() == sym and getattr(it, "asset_type", None)), None)
                if not item_asset_type:
                    c_no = sym_rows[0].get("Confirmation_No") or sym_rows[0].get("confirmation_no") or ""
                    if str(c_no).startswith("DIMEMF"):
                        item_asset_type = "Fund"
                    else:
                        item_asset_type = "Stock"

                if final_units > _FLOAT_EPS:
                    if h is None:
                        h = Holding(symbol=sym, asset_type=item_asset_type, units=final_units)
                        state.holdings.append(h)
                    h.units = final_units
                    h.status = "active"
                    h.archived_at = None
                    if ccy == "USD":
                        h.avg_cost_usd = final_avg
                    else:
                        h.avg_cost_thb = final_avg
                else:
                    if h:
                        state.holdings = [item for item in state.holdings if item.symbol != sym]

            state.summary.total_realized_profit_ytd = round(
                sum(float(r.get("Realized_PnL_THB") or r.get("realized_pnl_thb") or 0.0) for r in all_ledger_rows),
                _MONEY_DP,
            )
            recalc_all(state)

            commit_mutation(
                uow,
                state,
                PortfolioMutation(
                    ledger_change=LedgerChange(kind="replace_all", rows=all_ledger_rows),
                    system_journal_events=[
                        SystemJournalEvent.from_entry(
                            event_type="batch_trades_imported",
                            message=f"**[DIME BATCH IMPORT]** Successfully imported {len(valid_items)} trades",
                            metadata={"count": len(valid_items), "symbols": list(touched_symbols)},
                        )
                    ],
                ),
                journal_provider=self.journal_provider,
                portfolio_id=pid,
            )
            return state
