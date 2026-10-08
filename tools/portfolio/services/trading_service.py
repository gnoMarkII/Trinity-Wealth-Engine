"""PortfolioTradingService — Buy/Sell, Batch Import, Holding Edits, FX/Price Sync."""
import json
import time
import uuid
from typing import Optional, List, Dict, Tuple, Literal, Union

from decimal import Decimal
from tools.portfolio.domain.constants import (
    CASH_THB_SYMBOL,
    CASH_USD_SYMBOL,
    _CASH_SYMBOLS,
    _FLOAT_EPS,
    _MONEY_DP,
)
from tools.portfolio.domain.models import (
    PortfolioState,
    Holding,
    _now_iso,
    MONEY_QUANTUM,
    quantize_decimal,
)
from tools.portfolio.domain.ledger_change import LedgerChange
from tools.portfolio.domain.events import SystemJournalEvent
from tools.portfolio.domain.mutation import PortfolioMutation
from tools.portfolio.domain.calculations import (
    calc_weighted_avg_cost,
    calc_realized_pnl,
    recalc_all,
    _replay_symbol_trades,
)
from tools.portfolio.domain.validator import validate_portfolio_id
from tools.portfolio.ports.repository_port import PortfolioRepositoryPort
from tools.portfolio.ports.price_port import MarketPricePort
from tools.portfolio.ports.journal_port import TradeJournalPort
from ._mutation_commit import commit_mutation


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


class PortfolioTradingService:
    """Handles Trade execution, Batch holdings import, Holding edits/removals, and Price/FX sync."""

    def __init__(
        self,
        repo: PortfolioRepositoryPort,
        price_provider: MarketPricePort,
        journal_provider: TradeJournalPort,
    ) -> None:
        self.repo = repo
        self.price_provider = price_provider
        self.journal_provider = journal_provider

    # =========================================================================
    # Trade Execution
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
                rate_val, _ = self.price_provider.fetch_fx_rate(date_str=date, fallback_rate=state.fx_rates.get("USDTHB", 36.5))
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

            cost_native = Decimal(str(units)) * Decimal(str(price))
            gross_str = f"{quantize_decimal(cost_native, MONEY_QUANTUM):f}"
            net_str = gross_str
            price_str = f"{Decimal(str(price)):f}" if "." in str(price) else f"{price:.2f}"

            ledger_row: Dict = {
                "Transaction_ID": tx_id,
                "Timestamp": ts_str,
                "Symbol": clean_sym,
                "Action": action.upper(),
                "Units": f"{units:g}",
                "Price": price_str,
                "Currency": currency,
                "FX_Rate": f"{trade_fx:.4f}" if currency == "USD" else "",
                "Cost_THB": "",
                "Realized_PnL_THB": "",
                "Notes": notes or "",
                "Gross_Amount": gross_str,
                "Commission": "0.00",
                "VAT": "0.00",
                "Other_Fees": "0.00",
                "Net_Amount": net_str,
                "Fee_Currency": currency,
                "Confirmation_No": "",
                "Settlement_Date": "",
                "Source": "MANUAL",
                "Fingerprint": "",
                "Cash_Adjusted": "YES",
                "Related_Transaction_ID": "",
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

            content = f"**[TRADE NOTE - {clean_sym}]** {action.upper()} {units:g} @ {price:,.4f} {currency}"
            if notes and notes.strip():
                content += f" — {notes.strip()}"
            recalc_all(state)
            commit_mutation(
                uow,
                state,
                PortfolioMutation(
                    ledger_change=LedgerChange(kind="append", row=ledger_row, tx_id=tx_id),
                    system_journal_events=[
                        SystemJournalEvent.from_entry(
                            event_type="trade_executed",
                            message=content,
                            date_str=date,
                            metadata={"transaction_id": tx_id, "symbol": clean_sym},
                        )
                    ],
                ),
                journal_provider=self.journal_provider,
                portfolio_id=pid,
            )

            return res_str, state

    # =========================================================================
    # Batch Holdings Import
    # =========================================================================

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
                            if self.price_provider:
                                fetched = self.price_provider.fetch_price(sym, ccy)
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

    # =========================================================================
    # Holding Edits and Removals
    # =========================================================================

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
            journal_message = f"**[EDIT {clean_sym}]** {' | '.join(changes)}\n\nReason: {reason_text}"
            recalc_all(state)
            commit_mutation(
                uow,
                state,
                PortfolioMutation(
                    ledger_change=LedgerChange(kind="unchanged"),
                    system_journal_events=[
                        SystemJournalEvent.from_entry(
                            event_type="holding_edited",
                            message=journal_message,
                            metadata={"symbol": clean_sym},
                        )
                    ],
                ),
                journal_provider=self.journal_provider,
                portfolio_id=pid,
            )
            res_str = f"[EDIT {clean_sym}] {', '.join(changes)} (เหตุผล: {reason})"
            return res_str, state

    def _purge_holding_transactions_locked(
        self,
        uow,
        state: PortfolioState,
        symbols: List[str],
    ) -> Tuple[List[Dict], List[SystemJournalEvent], bool]:
        """Hard purge all transactions for the given symbols from the ledger, adjusting cash if necessary."""
        rows = uow.read_trade_log_locked()
        if not rows:
            return rows, [], False

        sym_set = {s.strip().upper() for s in symbols if s}
        if not sym_set:
            return rows, [], False

        # 1. Identify already voided target IDs
        voided_target_ids = {
            str(r.get("Related_Transaction_ID") or r.get("related_transaction_id") or "").strip()
            for r in rows
            if str(r.get("Action") or r.get("action") or "").strip().upper().startswith("VOID_")
            or str(r.get("Action") or r.get("action") or "").strip().upper() == "REVERSAL"
        }

        # 2. Adjust cash for active transactions belonging to symbols being purged
        journal_events: List[SystemJournalEvent] = []
        purged_tx_ids: Set[str] = set()

        for r in rows:
            tx_id = str(r.get("Transaction_ID") or r.get("transaction_id") or "").strip()
            sym = str(r.get("Symbol") or r.get("symbol") or "").strip().upper()
            action = str(r.get("Action") or r.get("action") or "").strip().upper()
            rel_tx_id = str(r.get("Related_Transaction_ID") or r.get("related_transaction_id") or "").strip()

            if sym not in sym_set:
                continue

            if tx_id:
                purged_tx_ids.add(tx_id)

            # Only active (non-voided, non-reversal) transactions need cash refund
            if action.startswith("VOID_") or action == "REVERSAL" or rel_tx_id or tx_id in voided_target_ids:
                continue

            ccy = str(r.get("Currency") or r.get("currency") or "THB").strip().upper()
            effective_cash_adjusted = str(r.get("Cash_Adjusted") or r.get("cash_adjusted") or "YES").strip().upper() == "YES"

            net_amt_raw = r.get("Net_Amount") or r.get("net_amount")
            if net_amt_raw is not None and str(net_amt_raw).strip() != "":
                try:
                    net_cash_amount = float(Decimal(str(net_amt_raw)))
                except Exception:
                    units_val = float(r.get("Units") or r.get("units") or 0)
                    price_val = float(r.get("Price") or r.get("price") or 0)
                    net_cash_amount = units_val * price_val
            else:
                units_val = float(r.get("Units") or r.get("units") or 0)
                price_val = float(r.get("Price") or r.get("price") or 0)
                net_cash_amount = units_val * price_val

            if effective_cash_adjusted:
                cash = _require_cash(state, ccy)
                if action == "BUY":
                    cash.units += net_cash_amount
                elif action == "SELL":
                    cash.units -= net_cash_amount

        # 3. Purge all rows for sym_set and any reversal rows referring to purged tx_ids
        final_rows = [
            r for r in rows
            if str(r.get("Symbol") or r.get("symbol") or "").strip().upper() not in sym_set
            and str(r.get("Related_Transaction_ID") or r.get("related_transaction_id") or "").strip() not in purged_tx_ids
            and str(r.get("Transaction_ID") or r.get("transaction_id") or "").strip() not in purged_tx_ids
        ]

        for sym in sym_set:
            purged_count = len([r for r in rows if str(r.get("Symbol") or r.get("symbol") or "").strip().upper() == sym])
            if purged_count > 0:
                journal_events.append(
                    SystemJournalEvent.from_entry(
                        event_type="transactions_purged",
                        message=f"**[TRANSACTIONS PURGED]** ลบประวัติธุรกรรมทั้งหมดของ {sym} ({purged_count} รายการ) ออกจากระบบ",
                        metadata={"symbol": sym, "purged_count": purged_count},
                    )
                )

        state.summary.total_realized_profit_ytd = round(
            sum(float(row.get("Realized_PnL_THB") or row.get("realized_pnl_thb") or 0.0) for row in final_rows),
            _MONEY_DP,
        )

        had_changes = len(final_rows) != len(rows)
        return final_rows, journal_events, had_changes

    _void_holding_transactions_locked = _purge_holding_transactions_locked

    def structured_remove_holding(self, symbol: str, portfolio_id: str = "default") -> PortfolioState:
        clean_sym = symbol.strip().upper()
        if clean_sym in _CASH_SYMBOLS:
            raise ValueError("ห้ามลบ cash sentinel")
        pid = validate_portfolio_id(portfolio_id)
        with self.repo.unit_of_work(pid) as uow:
            state = uow.load_state()
            target = _find_holding(state, clean_sym)
            rows = uow.read_trade_log_locked()
            has_trades = any(str(r.get("Symbol") or r.get("symbol") or "").strip().upper() == clean_sym for r in rows)
            if not target and not has_trades:
                raise ValueError(f"ไม่พบสินทรัพย์ {clean_sym} ในพอร์ต")
            if target:
                state.holdings.remove(target)

            final_rows, purge_events, had_changes = self._purge_holding_transactions_locked(
                uow, state, [clean_sym]
            )
            ledger_change = (
                LedgerChange(kind="replace_all", rows=final_rows, tx_id="holding_remove")
                if had_changes
                else LedgerChange(kind="unchanged")
            )

            remove_event = SystemJournalEvent.from_entry(
                event_type="holding_removed",
                message=f"**[REMOVE {clean_sym}]** ลบสินทรัพย์ออกจากพอร์ตโดยตรง",
                metadata={"symbol": clean_sym},
            )

            recalc_all(state)
            commit_mutation(
                uow,
                state,
                PortfolioMutation(
                    ledger_change=ledger_change,
                    system_journal_events=purge_events + [remove_event],
                    deleted_symbols=[clean_sym],
                ),
                journal_provider=self.journal_provider,
                portfolio_id=pid,
            )
            return state

    def structured_batch_remove_holdings(self, symbols: List[str], portfolio_id: str = "default") -> PortfolioState:
        pid = validate_portfolio_id(portfolio_id)
        with self.repo.unit_of_work(pid) as uow:
            state = uow.load_state()
            remove_events: List[SystemJournalEvent] = []
            clean_syms = []
            for sym in symbols:
                clean_sym = sym.strip().upper()
                if clean_sym in _CASH_SYMBOLS:
                    continue
                clean_syms.append(clean_sym)
                target = _find_holding(state, clean_sym)
                if target:
                    state.holdings.remove(target)
                    remove_events.append(
                        SystemJournalEvent.from_entry(
                            event_type="holding_removed",
                            message=f"**[REMOVE {clean_sym}]** ลบสินทรัพย์ออกจากพอร์ตโดยตรง",
                            metadata={"symbol": clean_sym},
                        )
                    )

            final_rows, purge_events, had_changes = self._purge_holding_transactions_locked(
                uow, state, clean_syms
            )
            ledger_change = (
                LedgerChange(kind="replace_all", rows=final_rows, tx_id="batch_holding_remove")
                if had_changes
                else LedgerChange(kind="unchanged")
            )

            recalc_all(state)
            commit_mutation(
                uow,
                state,
                PortfolioMutation(
                    ledger_change=ledger_change,
                    system_journal_events=purge_events + remove_events,
                    deleted_symbols=clean_syms,
                ),
                journal_provider=self.journal_provider,
                portfolio_id=pid,
            )
            return state

    # =========================================================================
    # Market Price & FX Updates
    # =========================================================================

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
                rate_val, _ = self.price_provider.fetch_fx_rate()
                new_rate = rate_val
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
        from filelock import Timeout
        from tools.tool_errors import LOCK_TIMEOUT

        pid = validate_portfolio_id(portfolio_id)
        try:
            with self.repo.unit_of_work(pid) as uow:
                state = uow.load_state()
                has_non_cash = any(
                    h.asset_type != "Cash" and h.status == "active" and h.units > _FLOAT_EPS
                    for h in state.holdings
                )
                results = self.price_provider.refresh_portfolio_prices(state)
                if not results and not has_non_cash:
                    return f"[SYNC] {pid}: no non-cash holdings to update"

                # Filter out auxiliary rates like USDTHB when assessing holding refresh counts
                holding_results = {k: v for k, v in results.items() if k != "USDTHB"}
                target_items = holding_results if holding_results else results

                def _is_failed(v: str) -> bool:
                    v_lower = str(v).lower()
                    return (
                        v_lower.startswith("fetch failed")
                        or v_lower.startswith("error")
                        or v_lower.startswith("timeout")
                        or v_lower == "no_data"
                    )

                total_count = len(target_items)
                failed_items = [f"{key}={value}" for key, value in target_items.items() if _is_failed(value)]
                success_count = total_count - len(failed_items)
                recalc_all(state)
                uow.commit(state, LedgerChange(kind="unchanged"))

                if failed_items:
                    return (
                        f"[SYNC] updated prices: refreshed {success_count}/{total_count} "
                        f"({', '.join(failed_items)})"
                    )
                return f"[SYNC] updated prices: refreshed {success_count}/{total_count}"
        except Timeout:
            return LOCK_TIMEOUT.format(detail=f"portfolio lock '{pid}'")
        except Exception as exc:
            return f"Error: {exc}"
