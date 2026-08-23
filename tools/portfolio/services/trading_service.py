"""PortfolioTradingService — Buy/Sell, Batch Import, Holding Edits, FX/Price Sync."""
import json
import time
import uuid
from typing import Optional, List, Dict, Tuple, Literal, Union

from core.logger import get_logger
from tools.portfolio.domain.constants import (
    CASH_THB_SYMBOL,
    CASH_USD_SYMBOL,
    _CASH_SYMBOLS,
    _FLOAT_EPS,
    _MONEY_DP,
)
from tools.portfolio.domain.models import PortfolioState, Holding, _now_iso
from tools.portfolio.domain.ledger_change import LedgerChange
from tools.portfolio.domain.calculations import (
    calc_weighted_avg_cost,
    calc_realized_pnl,
    recalc_all,
)
from tools.portfolio.domain.validator import validate_portfolio_id
from tools.portfolio.ports.repository_port import PortfolioRepositoryPort
from tools.portfolio.ports.price_port import MarketPricePort
from tools.portfolio.ports.journal_port import TradeJournalPort

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
            self.journal_provider.append_system_entry(content, date_str=date, portfolio_id=pid)

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
                            import tools.portfolio.trading as trading_mod
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
            self.journal_provider.append_system_entry(
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
            self.journal_provider.append_system_entry(
                f"**[REMOVE {clean_sym}]** ลบสินทรัพย์ออกจากพอร์ตโดยตรง", portfolio_id=pid
            )
            return state

    def structured_batch_remove_holdings(self, symbols: List[str], portfolio_id: str = "default") -> PortfolioState:
        pid = validate_portfolio_id(portfolio_id)
        with self.repo.unit_of_work(pid) as uow:
            state = uow.load_state()
            for sym in symbols:
                clean_sym = sym.strip().upper()
                if clean_sym in _CASH_SYMBOLS:
                    continue
                target = _find_holding(state, clean_sym)
                if target:
                    state.holdings.remove(target)
                    self.journal_provider.append_system_entry(
                        f"**[REMOVE {clean_sym}]** ลบสินทรัพย์ออกจากพอร์ตโดยตรง", portfolio_id=pid
                    )
            recalc_all(state)
            uow.commit(state, LedgerChange(kind="unchanged"))
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
                import tools.portfolio.trading as trading_fresh
                if hasattr(trading_fresh, "_fetch_fx_rate") and callable(trading_fresh._fetch_fx_rate):
                    new_rate = trading_fresh._fetch_fx_rate()
                    if isinstance(new_rate, tuple):
                        new_rate = new_rate[0]
                else:
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
        from tools.portfolio.prices import _sync_market_prices_impl
        return _sync_market_prices_impl(portfolio_id=portfolio_id)
