"""PortfolioCashFlowService — Deposit/Withdraw, Record Income, FX, Dividends."""
from datetime import date as date_type, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import math
import json
import uuid
from typing import Dict, List, Optional, Literal, Tuple

from tools.portfolio.domain.constants import _FLOAT_EPS, _MONEY_DP, CASH_THB_SYMBOL, CASH_USD_SYMBOL, _CASH_SYMBOLS
from tools.portfolio.domain.models import PortfolioState, Holding, DividendRound, _now_iso
from tools.portfolio.domain.ledger_change import LedgerChange
from tools.portfolio.domain.events import SystemJournalEvent
from tools.portfolio.domain.mutation import PortfolioMutation
from tools.portfolio.domain.calculations import recalc_all
from tools.portfolio.domain.validator import validate_portfolio_id
from tools.portfolio.ports.repository_port import PortfolioRepositoryPort
from tools.portfolio.ports.price_port import MarketPricePort
from tools.portfolio.ports.journal_port import TradeJournalPort
from tools.portfolio.ports.dividend_port import DividendHistoryPort
from ._mutation_commit import commit_mutation


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
        idempotency_key: Optional[str] = None,
    ) -> PortfolioState:
        _, state = self._manage_cash_flow_internal(
            amount=amount,
            action=action,
            currency=currency,
            exchange_rate=exchange_rate,
            date=date,
            notes=notes,
            portfolio_id=portfolio_id,
            idempotency_key=idempotency_key,
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
        idempotency_key: Optional[str] = None,
    ) -> Tuple[str, PortfolioState]:
        try:
            amount_decimal = Decimal(str(amount))
        except (InvalidOperation, TypeError, ValueError) as exc:
            raise ValueError("amount must be a positive finite decimal") from exc
        if not amount_decimal.is_finite() or amount_decimal <= Decimal("0"):
            raise ValueError("amount must be a positive finite decimal")
        amount_float = float(amount_decimal)
        if not math.isfinite(amount_float) or amount_float <= 0:
            raise ValueError("amount is outside the supported numeric range")
        if action not in ("deposit", "withdraw"):
            raise ValueError("action ต้องเป็น 'deposit' หรือ 'withdraw'")
        if currency not in ("THB", "USD"):
            raise ValueError("currency ต้องเป็น 'THB' หรือ 'USD'")

        pid = validate_portfolio_id(portfolio_id)
        idempotency_key = str(idempotency_key or f"cashflow-{uuid.uuid4().hex}").strip()
        if not idempotency_key:
            raise ValueError("idempotency_key is required")
        request_payload = {
            "portfolio_id": pid,
            "action": action,
            "amount_native_decimal": str(amount_decimal),
            "currency": currency,
            "date": str(date or ""),
            "exchange_rate": str(exchange_rate) if exchange_rate is not None else None,
            "notes": str(notes or "").strip(),
        }
        request_digest = hashlib.sha256(
            json.dumps(request_payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if date:
            try:
                parsed_occurred = datetime.fromisoformat(str(date).strip().replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError("date must be an ISO-8601 date or timestamp") from exc
            if parsed_occurred.tzinfo is None:
                parsed_occurred = parsed_occurred.replace(tzinfo=timezone.utc)
        else:
            parsed_occurred = datetime.now(timezone.utc)
        occurred_at_utc = parsed_occurred.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

        fx_rate_decimal: Optional[Decimal] = Decimal("1") if currency == "THB" else None
        fx_source_id: Optional[str] = "CURRENCY_IDENTITY" if currency == "THB" else None
        if currency == "USD":
            if exchange_rate is not None:
                try:
                    candidate = Decimal(str(exchange_rate))
                except (InvalidOperation, TypeError, ValueError) as exc:
                    raise ValueError("exchange_rate must be a positive finite decimal") from exc
                if not candidate.is_finite() or candidate <= Decimal("0"):
                    raise ValueError("exchange_rate must be a positive finite decimal")
                fx_rate_decimal = candidate
                fx_source_id = "REQUEST_SUPPLIED"
            else:
                try:
                    fetched_rate, fetched_source = self.price_provider.fetch_fx_rate(
                        date_str=occurred_at_utc[:10], fallback_rate=None
                    )
                    candidate = Decimal(str(fetched_rate))
                    if candidate.is_finite() and candidate > Decimal("0") and fetched_source in ("historical", "live"):
                        fx_rate_decimal = candidate
                        fx_source_id = f"market_price_provider:{fetched_source}"
                except Exception:
                    fx_rate_decimal = None
                    fx_source_id = None

        lookup_event = getattr(self.repo, "get_cash_flow_event", None)
        if callable(lookup_event):
            prior = lookup_event(pid, idempotency_key)
            if prior:
                if prior.get("request_digest") != request_digest:
                    raise ValueError("idempotency_key was reused for a different cash-flow request")
                return "[CASH FLOW] Duplicate request already committed.", self.repo.load_state(pid)

        with self.repo.unit_of_work(pid) as uow:
            state = uow.load_state()
            # Recheck while holding the portfolio UnitOfWork lock to fence concurrent retries.
            if callable(lookup_event):
                prior = lookup_event(pid, idempotency_key)
                if prior:
                    if prior.get("request_digest") != request_digest:
                        raise ValueError("idempotency_key was reused for a different cash-flow request")
                    return "[CASH FLOW] Duplicate request already committed.", state
            cash = _require_cash(state, currency)

            if action == "withdraw":
                if cash.units < amount_float - _FLOAT_EPS:
                    raise ValueError(
                        f"Insufficient cash in CASH_{currency} (requested: {amount_float:,.2f}, available: {cash.units:,.2f})"
                    )
                cash.units -= amount_float
            else:
                cash.units += amount_float

            res_str = (
                f"[{action.upper()}] {currency} | "
                f"{'+' if action == 'deposit' else '-'}{amount_decimal:,.2f} {currency} | "
                f"CASH_{currency} คงเหลือ: {cash.units:,.2f} {currency}"
            )

            content = f"**[CASH FLOW NOTE]** {action.upper()} {amount_float:,.2f} {currency}"
            if notes and notes.strip():
                content += f" — {notes.strip()}"
            recalc_all(state)
            signed_native_amount = amount_decimal if action == "deposit" else -amount_decimal
            signed_thb_amount = (
                (signed_native_amount * fx_rate_decimal).quantize(Decimal("0.01"))
                if fx_rate_decimal is not None else None
            )
            cash_flow_event = {
                "event_id": f"cashflow_{uuid.uuid4().hex}",
                "idempotency_key": idempotency_key,
                "request_digest": request_digest,
                "action": action,
                "amount_native_decimal": str(signed_native_amount),
                "currency": currency,
                "fx_rate_to_thb_decimal": str(fx_rate_decimal) if fx_rate_decimal is not None else None,
                "fx_source_id": fx_source_id,
                "amount_thb_decimal": str(signed_thb_amount) if signed_thb_amount is not None else None,
                "occurred_at_utc": occurred_at_utc,
                "flow_kind": "EXTERNAL_CONTRIBUTION",
                "flow_coverage_status": "COMPLETE" if signed_thb_amount is not None else "MISSING_FX",
            }
            commit_mutation(
                uow,
                state,
                PortfolioMutation(
                    ledger_change=LedgerChange(kind="unchanged"),
                    durable_events=[{"event_type": "cash_flow_recorded", "payload": cash_flow_event}],
                    system_journal_events=[
                        SystemJournalEvent.from_entry(
                            event_type="cash_flow_recorded",
                            message=content,
                            date_str=date,
                            metadata={"action": action, "currency": currency},
                        )
                    ],
                ),
                journal_provider=self.journal_provider,
                portfolio_id=pid,
            )

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
            recalc_all(state)
            commit_mutation(
                uow,
                state,
                PortfolioMutation(
                    ledger_change=LedgerChange(kind="unchanged"),
                    system_journal_events=[
                        SystemJournalEvent.from_entry(
                            event_type="income_recorded",
                            message=content,
                            date_str=date,
                            metadata={"income_type": income_type, "source_symbol": clean_src},
                        )
                    ],
                ),
                journal_provider=self.journal_provider,
                portfolio_id=pid,
            )

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
        """Fetch dividend history through the port and persist derived totals atomically."""
        if self.dividend_provider is None:
            raise RuntimeError("Dividend history provider is not configured")

        pid = validate_portfolio_id(portfolio_id)
        with self.repo.unit_of_work(pid) as uow:
            state = uow.load_state()
            targets = [
                holding
                for holding in state.holdings
                if holding.asset_type != "Cash"
                and holding.asset_type != "PVD"
                and not holding.symbol.upper().startswith("PVD_")
                and holding.status == "active"
                and holding.dividend_source != "manual"
            ]
            skipped_manual = [
                holding.symbol
                for holding in state.holdings
                if holding.asset_type not in ("Cash", "PVD") and not holding.symbol.upper().startswith("PVD_") and holding.dividend_source == "manual"
            ]
            if not targets:
                return {
                    "synced_symbols": 0,
                    "total_rounds": 0,
                    "total_received_rounds": 0,
                    "total_upcoming_rounds": 0,
                    "total_dividend_thb": 0.0,
                    "total_upcoming_thb": 0.0,
                    "skipped_manual": skipped_manual,
                    "details": {},
                }

            histories = self.dividend_provider.fetch_dividend_history([h.symbol for h in targets])
            trade_rows = uow.read_trade_log_locked()
            fallback_fx = state.fx_rates.get("USDTHB", 36.5)
            details: Dict[str, List[Dict]] = {}
            total_rounds = total_received_rounds = total_upcoming_rounds = 0
            total_received_thb = total_upcoming_thb = 0.0

            for holding in targets:
                currency = "USD" if holding.avg_cost_usd is not None else "THB"
                rounds: List[Dict] = []
                for record in histories.get(holding.symbol, []) or []:
                    ex_date = self._dividend_date(record.get("ex_date") or record.get("date"))
                    if ex_date is None:
                        continue
                    dps = float(record.get("dps", record.get("amount", 0.0)) or 0.0)
                    if dps <= 0:
                        continue
                    units_held = self._units_held_before(trade_rows, holding.symbol, ex_date)
                    if units_held <= _FLOAT_EPS:
                        continue

                    pay_date = self._dividend_date(record.get("pay_date")) or (ex_date + timedelta(days=21))
                    status = "received" if pay_date <= date_type.today() and ex_date <= date_type.today() else "upcoming"
                    record_currency = str(record.get("currency") or currency).upper()
                    tax_rate = 0.10 if record_currency == "THB" else 0.15 if record_currency == "USD" else 0.0
                    fx_rate = 1.0
                    if record_currency == "USD":
                        try:
                            fx_rate, _ = self.price_provider.fetch_fx_rate(
                                date_str=ex_date.isoformat(), fallback_rate=fallback_fx
                            )
                        except Exception:
                            fx_rate = fallback_fx
                    gross_native = round(units_held * dps, 4)
                    net_native = round(gross_native * (1.0 - tax_rate), 4)
                    net_thb = round(net_native * fx_rate, _MONEY_DP)
                    round_data = {
                        "symbol": holding.symbol,
                        "ex_date": ex_date.isoformat(),
                        "pay_date": pay_date.isoformat(),
                        "dps": round(dps, 4),
                        "currency": record_currency,
                        "units_held": units_held,
                        "status": status,
                        "gross_native": gross_native,
                        "net_native": net_native,
                        "gross_thb": round(gross_native * fx_rate, _MONEY_DP),
                        "tax_rate": tax_rate,
                        "net_thb": net_thb,
                        "fx_rate": round(fx_rate, 4),
                    }
                    rounds.append(round_data)

                rounds.sort(key=lambda item: item["ex_date"], reverse=True)
                received = [item for item in rounds if item["status"] == "received"]
                upcoming = [item for item in rounds if item["status"] == "upcoming"]
                holding.accumulated_dividend_thb = round(sum(item["net_thb"] for item in received), _MONEY_DP)
                holding.accumulated_dividend_native = round(sum(item["net_native"] for item in received), 4)
                holding.upcoming_dividend_thb = round(sum(item["net_thb"] for item in upcoming), _MONEY_DP)
                holding.upcoming_dividend_native = round(sum(item["net_native"] for item in upcoming), 4)
                holding.dividend_rounds = [DividendRound(**item) for item in rounds]
                holding.dividend_source = "synced"
                if rounds:
                    details[holding.symbol] = rounds
                total_rounds += len(rounds)
                total_received_rounds += len(received)
                total_upcoming_rounds += len(upcoming)
                total_received_thb += sum(item["net_thb"] for item in received)
                total_upcoming_thb += sum(item["net_thb"] for item in upcoming)

            state.summary.total_accumulated_dividend = round(
                sum(holding.accumulated_dividend_thb or 0.0 for holding in state.holdings), _MONEY_DP
            )
            recalc_all(state)
            uow.commit(state, LedgerChange(kind="unchanged"))

        return {
            "synced_symbols": len(details),
            "total_rounds": total_rounds,
            "total_received_rounds": total_received_rounds,
            "total_upcoming_rounds": total_upcoming_rounds,
            "total_dividend_thb": round(total_received_thb, _MONEY_DP),
            "total_upcoming_thb": round(total_upcoming_thb, _MONEY_DP),
            "skipped_manual": skipped_manual,
            "details": details,
        }

    @staticmethod
    def _dividend_date(value) -> Optional[date_type]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date_type):
            return value
        try:
            return datetime.fromisoformat(str(value)[:10]).date()
        except (TypeError, ValueError):
            return None

    @classmethod
    def _units_held_before(cls, rows: List[Dict], symbol: str, before: date_type) -> float:
        units = 0.0
        clean_symbol = symbol.strip().upper()
        timeline = []
        for row in rows:
            row_symbol = str(row.get("Symbol", row.get("symbol", ""))).strip().upper()
            if row_symbol != clean_symbol:
                continue
            trade_date = cls._dividend_date(row.get("Timestamp", row.get("timestamp")))
            if trade_date is None or trade_date >= before:
                continue
            try:
                quantity = float(row.get("Units", row.get("units", 0.0)) or 0.0)
            except (TypeError, ValueError):
                continue
            timeline.append((trade_date, str(row.get("Action", row.get("action", ""))).upper(), quantity))
        for _, action, quantity in sorted(timeline, key=lambda item: item[0]):
            if quantity <= 0:
                continue
            if action == "BUY":
                units += quantity
            elif action == "SELL":
                units = max(0.0, units - quantity)
        return round(units, 8)
