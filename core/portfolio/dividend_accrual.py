"""Ex-Dividend Receivable Accrual & True Economic NAV.

Architecture Layer: core/portfolio/ (Pure Domain)
Strict Pure Python — no numpy, pandas, pydantic, sqlite3, or API imports.
Enforces:
- Ex-dividend receivable accrual derived from ex-date entitled units (immutable
  entitlement snapshots or dated trade history prior to ex_date) rather than
  current post-XD share balances.
- Retaining earned receivables after post-XD full or partial sales until settlement.
- Settlement-aware clearing so early/recorded cash settlements do not double-count
  with accrued receivables.
- Finite Decimal normalization across all monetary, unit, tax, and FX inputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

_TWO_DP = Decimal("0.01")
_HUNDRED = Decimal("100")
_CASH_SENTINELS = frozenset({"CASH_THB", "CASH_USD", "CASH", "THB", "USD"})


def _parse_finite_decimal(
    value: Any,
    field_name: str,
    *,
    allow_zero: bool = True,
    allow_negative: bool = False,
) -> Decimal:
    """Strictly converts numbers/strings/Decimals to finite Decimal; rejects bools and NaN/Inf."""
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{field_name} must be a finite numeric value, got {value!r}")
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"{field_name} cannot be empty")
    try:
        dec = value if isinstance(value, Decimal) else Decimal(str(value).strip())
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field_name}: {value!r}") from exc
    if not dec.is_finite():
        raise ValueError(f"{field_name} must be finite, got {value!r}")
    if not allow_negative and dec < Decimal("0"):
        raise ValueError(f"{field_name} must be non-negative, got {dec}")
    if not allow_zero and dec == Decimal("0"):
        raise ValueError(f"{field_name} must be positive, got {dec}")
    return dec


def _normalize_iso_date(value: Any, field_name: str = "date") -> str:
    """Normalizes date/datetime/ISO strings into strict YYYY-MM-DD."""
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if not isinstance(value, str) or len(value.strip()) < 10:
        raise ValueError(f"{field_name} must be an ISO YYYY-MM-DD date, got {value!r}")
    iso_part = value.strip()[:10]
    try:
        date.fromisoformat(iso_part)
    except ValueError as exc:
        raise ValueError(f"{field_name} is not a valid ISO YYYY-MM-DD date: {value!r}") from exc
    return iso_part


def _infer_market(
    ticker: str,
    *,
    market: Any = None,
    currency: Any = None,
    obj: Any = None,
) -> str:
    clean_tick = ticker.strip().upper()
    if clean_tick.endswith(".BK"):
        return "TH"
    if market is not None and str(market).strip():
        m = str(market).strip().upper()
        if m in ("TH", "THB", "SET"):
            return "TH"
        if m in ("US", "USD", "NYSE", "NASDAQ"):
            return "US"
    cur = currency
    if cur is None and obj is not None:
        cur = obj.get("currency") if isinstance(obj, dict) else getattr(obj, "currency", None)
    if cur is not None and str(cur).strip():
        c = str(cur).strip().upper()
        if c == "THB":
            return "TH"
        if c == "USD":
            return "US"
    if obj is not None:
        avg_usd = obj.get("avg_cost_usd") if isinstance(obj, dict) else getattr(obj, "avg_cost_usd", None)
        avg_thb = obj.get("avg_cost_thb") if isinstance(obj, dict) else getattr(obj, "avg_cost_thb", None)
        if avg_usd is not None:
            return "US"
        if avg_thb is not None:
            return "TH"
    return ""


def _security_identity(ticker: str, market: Optional[str] = None) -> Tuple[str, str]:
    """Canonicalizes security base symbol and market tag (e.g. PTT.BK == PTT in TH)."""
    clean_tick = str(ticker or "").strip().upper()
    if not clean_tick:
        return ("", "")
    if clean_tick.endswith(".BK"):
        return (clean_tick[:-3], "TH")
    clean_market = str(market or "").strip().upper()
    if clean_market in ("TH", "US"):
        return (clean_tick, clean_market)
    return (clean_tick, "")


@dataclass(frozen=True)
class DividendScheduleEntry:
    """Dividend announcement schedule entry."""

    ticker: str
    ex_date: str  # "YYYY-MM-DD"
    pay_date: str  # "YYYY-MM-DD"
    dps_native: Decimal  # Dividend Per Share in native currency
    market: Literal["TH", "US"] = "TH"
    currency: Literal["THB", "USD"] = "THB"
    withholding_tax_pct: Optional[Decimal] = None  # Defaults: TH=10.0%, US=15.0%
    settlement_date: Optional[str] = None
    settlement_confirmed: Optional[bool] = None

    def __post_init__(self) -> None:
        clean_ticker = str(self.ticker or "").strip().upper()
        if not clean_ticker:
            raise ValueError("ticker is required on DividendScheduleEntry")
        object.__setattr__(self, "ticker", clean_ticker)

        ex_iso = _normalize_iso_date(self.ex_date, "ex_date")
        pay_iso = _normalize_iso_date(self.pay_date, "pay_date")
        object.__setattr__(self, "ex_date", ex_iso)
        object.__setattr__(self, "pay_date", pay_iso)

        if self.settlement_date is not None and str(self.settlement_date).strip():
            object.__setattr__(self, "settlement_date", _normalize_iso_date(self.settlement_date, "settlement_date"))
        else:
            object.__setattr__(self, "settlement_date", None)

        dps_dec = _parse_finite_decimal(self.dps_native, "dps_native", allow_zero=True)
        object.__setattr__(self, "dps_native", dps_dec)

        inferred_market = _infer_market(clean_ticker, market=self.market, currency=self.currency)
        resolved_market: Literal["TH", "US"] = "US" if inferred_market == "US" else "TH"
        object.__setattr__(self, "market", resolved_market)

        cur_upper = str(self.currency or ("USD" if resolved_market == "US" else "THB")).strip().upper()
        resolved_currency: Literal["THB", "USD"] = "USD" if cur_upper == "USD" else "THB"
        object.__setattr__(self, "currency", resolved_currency)

        if self.withholding_tax_pct is None:
            wht = Decimal("15.0") if (resolved_market == "US" or resolved_currency == "USD") else Decimal("10.0")
        else:
            wht = _parse_finite_decimal(self.withholding_tax_pct, "withholding_tax_pct", allow_zero=True)
            if wht > _HUNDRED:
                raise ValueError(f"withholding_tax_pct must be <= 100, got {wht}")
        object.__setattr__(self, "withholding_tax_pct", wht)


@dataclass(frozen=True)
class DividendEntitlementSnapshot:
    """Immutable record of entitled shares pinned on or before ex_date."""

    ticker: str
    ex_date: str
    entitled_units: Decimal
    market: Optional[str] = None
    source: str = "entitlement_snapshot"

    def __post_init__(self) -> None:
        clean_tick = str(self.ticker or "").strip().upper()
        if not clean_tick:
            raise ValueError("ticker is required on DividendEntitlementSnapshot")
        object.__setattr__(self, "ticker", clean_tick)
        object.__setattr__(self, "ex_date", _normalize_iso_date(self.ex_date, "ex_date"))
        object.__setattr__(
            self,
            "entitled_units",
            _parse_finite_decimal(self.entitled_units, "entitled_units", allow_zero=True),
        )


@dataclass(frozen=True)
class DividendSettlementRecord:
    """Confirmed dividend cash settlement clearing an accrued receivable."""

    ticker: str
    settled_date: str
    settled_amount_native: Optional[Decimal] = None
    settled_amount_thb: Optional[Decimal] = None
    currency: Optional[str] = None
    ex_date: Optional[str] = None
    status: str = "settled"

    def __post_init__(self) -> None:
        clean_tick = str(self.ticker or "").strip().upper()
        if not clean_tick:
            raise ValueError("ticker is required on DividendSettlementRecord")
        object.__setattr__(self, "ticker", clean_tick)
        object.__setattr__(self, "settled_date", _normalize_iso_date(self.settled_date, "settled_date"))
        if self.ex_date is not None and str(self.ex_date).strip():
            object.__setattr__(self, "ex_date", _normalize_iso_date(self.ex_date, "ex_date"))
        else:
            object.__setattr__(self, "ex_date", None)
        if self.settled_amount_native is not None:
            object.__setattr__(
                self,
                "settled_amount_native",
                _parse_finite_decimal(self.settled_amount_native, "settled_amount_native", allow_zero=True),
            )
        if self.settled_amount_thb is not None:
            object.__setattr__(
                self,
                "settled_amount_thb",
                _parse_finite_decimal(self.settled_amount_thb, "settled_amount_thb", allow_zero=True),
            )


def normalize_dividend_schedule_entry(entry: Any) -> DividendScheduleEntry:
    """Converts dicts, DividendRound-like models, or DividendScheduleEntry objects to canonical form."""
    if isinstance(entry, DividendScheduleEntry):
        return entry
    if isinstance(entry, dict):
        get_val = entry.get
    else:
        get_val = lambda k, default=None: getattr(entry, k, default)

    ticker = get_val("ticker") or get_val("symbol")
    ex_date = get_val("ex_date") or get_val("date")
    pay_date = get_val("pay_date")
    if pay_date is None or str(pay_date).strip() == "":
        raise ValueError(f"Dividend schedule for {ticker!r} is missing pay_date")

    dps_raw = get_val("dps_native")
    if dps_raw is None:
        dps_raw = get_val("dps")
    if dps_raw is None:
        dps_raw = get_val("amount")

    raw_currency = get_val("currency")
    raw_market = get_val("market")
    inferred_market = _infer_market(str(ticker or ""), market=raw_market, currency=raw_currency)
    market: Literal["TH", "US"] = "US" if inferred_market == "US" else "TH"
    if raw_currency is not None and str(raw_currency).strip():
        currency: Literal["THB", "USD"] = "USD" if str(raw_currency).strip().upper() == "USD" else "THB"
    else:
        currency = "USD" if market == "US" else "THB"

    wht_raw = get_val("withholding_tax_pct")
    if wht_raw is None and get_val("tax_rate") is not None:
        tr = _parse_finite_decimal(get_val("tax_rate"), "tax_rate", allow_zero=True)
        wht_raw = tr * _HUNDRED if tr <= Decimal("1") else tr

    return DividendScheduleEntry(
        ticker=str(ticker or ""),
        ex_date=str(ex_date or ""),
        pay_date=str(pay_date),
        dps_native=_parse_finite_decimal(dps_raw, "dps_native", allow_zero=True),
        market=market,
        currency=currency,
        withholding_tax_pct=wht_raw,
        settlement_date=get_val("settlement_date"),
        settlement_confirmed=get_val("settlement_confirmed"),
    )


@dataclass(frozen=True)
class AccruedDividendItem:
    """Individual accrued receivable item for a holding."""

    ticker: str
    units: Decimal
    dps_native: Decimal
    gross_amount_native: Decimal
    net_amount_native: Decimal
    withholding_tax_native: Decimal
    net_amount_thb: Decimal
    currency: Literal["THB", "USD"]
    ex_date: str
    pay_date: str
    days_to_payment: int
    market: str = "TH"
    entitlement_source: str = "holdings_fallback_unverified"
    gross_amount_thb: Decimal = Decimal("0.00")
    withholding_tax_thb: Decimal = Decimal("0.00")
    settlement_status: str = "ACCRUED_UNSETTLED"


@dataclass(frozen=True)
class AccruedDividendSummary:
    """Aggregated portfolio dividend receivables with entitlement provenance."""

    accrued_receivable_thb: Decimal
    accrued_receivable_usd: Decimal
    items: List[AccruedDividendItem] = field(default_factory=list)
    entitlement_basis: str = "none"
    entitlement_verified: bool = True
    warnings: List[str] = field(default_factory=list)


def _aggregate_holdings(holdings: Sequence[Any]) -> Tuple[Dict[Tuple[str, str], Decimal], Dict[str, List[Tuple[str, Decimal]]]]:
    """Aggregates duplicate holding rows by canonical security identity."""
    exact_map: Dict[Tuple[str, str], Decimal] = {}
    by_base: Dict[str, Dict[str, Decimal]] = {}

    for h in holdings or ():
        if isinstance(h, dict):
            sym = h.get("symbol") or h.get("ticker")
            raw_u = h.get("units")
            asset_type = h.get("asset_type")
            currency = h.get("currency")
            market = h.get("market")
        else:
            sym = getattr(h, "symbol", None) or getattr(h, "ticker", None)
            raw_u = getattr(h, "units", None)
            asset_type = getattr(h, "asset_type", None)
            currency = getattr(h, "currency", None)
            market = getattr(h, "market", None)

        if not sym or raw_u is None:
            continue
        clean_sym = str(sym).strip().upper()
        if clean_sym in _CASH_SENTINELS or str(asset_type or "").strip().lower() == "cash":
            continue

        dec_u = _parse_finite_decimal(raw_u, f"holdings[{clean_sym}].units", allow_zero=True)
        if dec_u <= Decimal("0"):
            continue

        inferred_mkt = _infer_market(clean_sym, market=market, currency=currency, obj=h)
        base, norm_mkt = _security_identity(clean_sym, inferred_mkt)
        exact_map[(base, norm_mkt)] = exact_map.get((base, norm_mkt), Decimal("0")) + dec_u
        by_base.setdefault(base, {})[norm_mkt] = by_base.get(base, {}).get(norm_mkt, Decimal("0")) + dec_u

    base_index = {base: list(mkt_dict.items()) for base, mkt_dict in by_base.items()}
    return exact_map, base_index


def _extract_entitlement_snapshots(
    snapshots: Optional[Sequence[Any]],
) -> Dict[Tuple[str, str, str], Tuple[Decimal, str]]:
    """Indexes explicit ex-date entitlement snapshots keyed by (base, market, ex_date)."""
    indexed: Dict[Tuple[str, str, str], Tuple[Decimal, str]] = {}
    if not snapshots:
        return indexed

    for idx, snap in enumerate(snapshots):
        if isinstance(snap, DividendEntitlementSnapshot):
            tick, ex_d, units, mkt, src = snap.ticker, snap.ex_date, snap.entitled_units, snap.market, snap.source
        elif isinstance(snap, dict):
            tick = snap.get("ticker") or snap.get("symbol")
            ex_d = _normalize_iso_date(snap.get("ex_date"), f"entitlement_snapshots[{idx}].ex_date")
            raw_u = snap.get("entitled_units", snap.get("units", snap.get("units_held")))
            units = _parse_finite_decimal(raw_u, f"entitlement_snapshots[{idx}].entitled_units", allow_zero=True)
            mkt = snap.get("market") or snap.get("currency")
            src = str(snap.get("source") or "entitlement_snapshot")
        else:
            tick = getattr(snap, "ticker", None) or getattr(snap, "symbol", None)
            ex_d = _normalize_iso_date(getattr(snap, "ex_date", None), f"entitlement_snapshots[{idx}].ex_date")
            raw_u = getattr(snap, "entitled_units", None)
            if raw_u is None:
                raw_u = getattr(snap, "units_held", None)
            if raw_u is None:
                raw_u = getattr(snap, "units", None)
            units = _parse_finite_decimal(raw_u, f"entitlement_snapshots[{idx}].entitled_units", allow_zero=True)
            mkt = getattr(snap, "market", None) or getattr(snap, "currency", None)
            src = str(getattr(snap, "source", "entitlement_snapshot"))

        if not tick:
            continue
        inferred_mkt = _infer_market(str(tick), market=mkt, obj=snap)
        base, norm_mkt = _security_identity(str(tick), inferred_mkt)
        key = (base, norm_mkt, ex_d)
        if key in indexed:
            prev_units, prev_src = indexed[key]
            indexed[key] = (prev_units + units, prev_src)
        else:
            indexed[key] = (units, src)
    return indexed


def _build_trade_timelines(
    trade_history: Sequence[Any],
) -> Dict[Tuple[str, str], List[Tuple[str, str, int, str, Decimal]]]:
    """Builds chronologically sorted BUY/SELL ledger rows keyed by (base, market)."""
    timelines: Dict[Tuple[str, str], List[Tuple[str, str, int, str, Decimal]]] = {}
    for seq_idx, row in enumerate(trade_history):
        if isinstance(row, dict):
            raw_sym = row.get("Symbol", row.get("symbol", row.get("ticker")))
            raw_action = row.get("Action", row.get("action"))
            raw_units = row.get("Units", row.get("units", row.get("quantity")))
            raw_ts = row.get("Timestamp", row.get("timestamp", row.get("trade_date", row.get("date"))))
            raw_cur = row.get("Currency", row.get("currency"))
            raw_mkt = row.get("Market", row.get("market"))
            raw_status = row.get("Status", row.get("status", ""))
        else:
            raw_sym = getattr(row, "symbol", None) or getattr(row, "ticker", None)
            raw_action = getattr(row, "action", None)
            raw_units = getattr(row, "units", None)
            if raw_units is None:
                raw_units = getattr(row, "quantity", None)
            raw_ts = (
                getattr(row, "timestamp", None)
                or getattr(row, "trade_date", None)
                or getattr(row, "date", None)
            )
            raw_cur = getattr(row, "currency", None)
            raw_mkt = getattr(row, "market", None)
            raw_status = getattr(row, "status", "")

        status_u = str(raw_status or "").strip().upper()
        if status_u in ("VOID", "VOIDED", "CANCELLED", "CANCELED"):
            continue
        action_u = str(raw_action or "").strip().upper()
        if action_u not in ("BUY", "SELL"):
            continue
        if not raw_sym or raw_ts is None or raw_units is None:
            continue

        clean_sym = str(raw_sym).strip().upper()
        if clean_sym in _CASH_SENTINELS:
            continue

        trade_day = _normalize_iso_date(raw_ts, f"trade_history[{seq_idx}].date")
        ts_sort = raw_ts.isoformat() if isinstance(raw_ts, (date, datetime)) else str(raw_ts).strip()
        qty = _parse_finite_decimal(raw_units, f"trade_history[{seq_idx}].units", allow_zero=True)
        if qty <= Decimal("0"):
            continue

        inferred_mkt = _infer_market(clean_sym, market=raw_mkt, currency=raw_cur, obj=row)
        base, norm_mkt = _security_identity(clean_sym, inferred_mkt)
        timelines.setdefault((base, norm_mkt), []).append((trade_day, ts_sort, seq_idx, action_u, qty))

    for entries in timelines.values():
        entries.sort(key=lambda item: (item[0], item[1], item[2]))
    return timelines


def _entitled_units_from_timeline(
    timelines: Dict[Tuple[str, str], List[Tuple[str, str, int, str, Decimal]]],
    base: str,
    market: str,
    ex_date: str,
) -> Decimal:
    """Computes shares held strictly prior to ex_date (trade_date < ex_date)."""
    rows: List[Tuple[str, str, int, str, Decimal]] = []
    if (base, market) in timelines:
        rows.extend(timelines[(base, market)])
    if (base, "") in timelines:
        rows.extend(timelines[(base, "")])
    if not rows:
        return Decimal("0")

    rows.sort(key=lambda item: (item[0], item[1], item[2]))
    running = Decimal("0")
    for trade_day, _, _, action_u, qty in rows:
        if trade_day >= ex_date:
            continue
        if action_u == "BUY":
            running += qty
        elif action_u == "SELL":
            running = max(Decimal("0"), running - qty)
    return running


def _parse_settlement_records(
    records: Optional[Sequence[Any]],
) -> List[Tuple[str, str, DividendSettlementRecord]]:
    parsed: List[Tuple[str, str, DividendSettlementRecord]] = []
    if not records:
        return parsed
    for idx, rec in enumerate(records):
        if isinstance(rec, DividendSettlementRecord):
            sr = rec
        elif isinstance(rec, dict):
            tick = rec.get("ticker") or rec.get("symbol") or rec.get("source_symbol")
            sdate = rec.get("settled_date") or rec.get("settlement_date") or rec.get("date")
            ex_d = rec.get("ex_date")
            raw_nat = rec.get("settled_amount_native", rec.get("net_amount_native", rec.get("amount")))
            raw_thb = rec.get("settled_amount_thb", rec.get("net_amount_thb"))
            cur = rec.get("currency")
            status = str(rec.get("status") or "settled")
            sr = DividendSettlementRecord(
                ticker=str(tick or ""),
                settled_date=str(sdate or ""),
                settled_amount_native=raw_nat,
                settled_amount_thb=raw_thb,
                currency=cur,
                ex_date=ex_d,
                status=status,
            )
        else:
            tick = getattr(rec, "ticker", None) or getattr(rec, "symbol", None)
            sdate = (
                getattr(rec, "settled_date", None)
                or getattr(rec, "settlement_date", None)
                or getattr(rec, "date", None)
            )
            ex_d = getattr(rec, "ex_date", None)
            raw_nat = getattr(rec, "settled_amount_native", None)
            if raw_nat is None:
                raw_nat = getattr(rec, "net_amount_native", None)
            if raw_nat is None:
                raw_nat = getattr(rec, "amount", None)
            raw_thb = getattr(rec, "settled_amount_thb", None)
            if raw_thb is None:
                raw_thb = getattr(rec, "net_amount_thb", None)
            cur = getattr(rec, "currency", None)
            status = str(getattr(rec, "status", "settled"))
            sr = DividendSettlementRecord(
                ticker=str(tick or ""),
                settled_date=str(sdate or ""),
                settled_amount_native=raw_nat,
                settled_amount_thb=raw_thb,
                currency=cur,
                ex_date=ex_d,
                status=status,
            )
        inferred_mkt = _infer_market(sr.ticker, currency=sr.currency, obj=rec)
        base, norm_mkt = _security_identity(sr.ticker, inferred_mkt)
        parsed.append((base, norm_mkt, sr))
    return parsed


def compute_accrued_dividend_receivable(
    holdings: List[Any],
    dividend_schedules: List[Any],
    as_of_date: str,
    usd_thb_fx_rate: Any = Decimal("35.50"),
    *,
    trade_history: Optional[Sequence[Any]] = None,
    entitlement_snapshots: Optional[Sequence[Any]] = None,
    settlement_records: Optional[Sequence[Any]] = None,
    require_verified_entitlement: bool = False,
) -> AccruedDividendSummary:
    """Computes accrued net dividend receivables as of a given date.

    Entitlement Policy:
    1. If an immutable ``entitlement_snapshots`` entry matches ``(ticker, ex_date)``,
       its ``entitled_units`` is authoritative.
    2. Otherwise, if ``trade_history`` is supplied (even an empty sequence ``[]``),
       entitled units are derived strictly as shares held prior to ``ex_date``
       (``trade_date < ex_date``). Post-XD buys receive 0 entitlement and post-XD
       sells retain the receivable until settled.
    3. If both ``entitlement_snapshots`` and ``trade_history`` are unavailable
       (``None``):
       - When ``require_verified_entitlement=True``, fails closed (0 accrued) and
         records ``entitlement_basis="unavailable"``.
       - Otherwise (default backward-compatible mode), falls back to aggregated
         current holdings while flagging ``entitlement_verified=False`` and
         ``entitlement_basis="holdings_fallback_unverified"`` plus explicit warnings.
    """
    clean_as_of = _normalize_iso_date(as_of_date, "as_of_date")
    fx_rate = _parse_finite_decimal(usd_thb_fx_rate, "usd_thb_fx_rate", allow_zero=False)

    exact_holdings, base_holdings = _aggregate_holdings(holdings)
    snapshot_map = _extract_entitlement_snapshots(entitlement_snapshots)
    timelines = _build_trade_timelines(trade_history) if trade_history is not None else None
    parsed_settlements = _parse_settlement_records(settlement_records)
    consumed_settlement_indices: set[int] = set()

    # Deduplicate announcement schedules by canonical event key
    dedup_schedules: List[DividendScheduleEntry] = []
    seen_events: set[Tuple[str, str, str, str, Decimal, str]] = set()
    for raw_sched in dividend_schedules or ():
        sched = normalize_dividend_schedule_entry(raw_sched)
        base, norm_mkt = _security_identity(sched.ticker, sched.market)
        dedup_key = (base, norm_mkt, sched.ex_date, sched.pay_date, sched.dps_native, sched.currency)
        if dedup_key in seen_events:
            continue
        seen_events.add(dedup_key)
        dedup_schedules.append(sched)

    total_thb = Decimal("0.00")
    total_usd = Decimal("0.00")
    items: List[AccruedDividendItem] = []
    sources_used: set[str] = set()
    warnings: List[str] = []
    fallback_warned = False
    unavailable_warned = False

    d_asof = date.fromisoformat(clean_as_of)

    for sched in dedup_schedules:
        base, norm_mkt = _security_identity(sched.ticker, sched.market)
        ex_d = sched.ex_date
        pay_d = sched.pay_date

        if clean_as_of < ex_d:
            continue

        # Enforce accrual end window unless settlement explicitly marked pending past pay_date
        end_d = sched.settlement_date or pay_d
        if sched.settlement_confirmed is not False and clean_as_of >= end_d:
            continue

        # Determine ex-date entitled units
        entitled_units = Decimal("0")
        entitlement_source = "unavailable"

        snap_hit = snapshot_map.get((base, norm_mkt, ex_d)) or snapshot_map.get((base, "", ex_d))
        if snap_hit is not None:
            entitled_units, entitlement_source = snap_hit
        elif timelines is not None:
            entitled_units = _entitled_units_from_timeline(timelines, base, norm_mkt, ex_d)
            entitlement_source = "trade_history"
        elif require_verified_entitlement:
            entitled_units = Decimal("0")
            entitlement_source = "unavailable"
            if not unavailable_warned:
                warnings.append(
                    "Ex-date entitlement history is unavailable and fail-closed verification is enabled; "
                    "unverified holdings fallback skipped."
                )
                unavailable_warned = True
        else:
            entitlement_source = "holdings_fallback_unverified"
            candidates = [
                u for mkt, u in base_holdings.get(base, []) if not norm_mkt or mkt in ("", norm_mkt)
            ]
            entitled_units = sum(candidates, Decimal("0"))
            if entitled_units > Decimal("0") and not fallback_warned:
                warnings.append(
                    "Ex-date entitlement history unavailable; fell back to current aggregated "
                    "holdings units without ex-date verification."
                )
                fallback_warned = True

        if entitled_units <= Decimal("0") or sched.dps_native <= Decimal("0"):
            continue

        gross_native = (entitled_units * sched.dps_native).quantize(_TWO_DP, rounding=ROUND_HALF_UP)
        tax_rate = sched.withholding_tax_pct / _HUNDRED
        tax_native = (gross_native * tax_rate).quantize(_TWO_DP, rounding=ROUND_HALF_UP)
        net_native = gross_native - tax_native
        if net_native <= Decimal("0"):
            continue

        # Clear against settlement evidence up to clean_as_of (prevents early-settlement double counting)
        settled_native = Decimal("0.00")
        fully_settled = False
        settlement_status = "ACCRUED_UNSETTLED"
        for s_idx, (s_base, s_mkt, sr) in enumerate(parsed_settlements):
            if s_idx in consumed_settlement_indices:
                continue
            if s_base != base:
                continue
            if s_mkt and norm_mkt and s_mkt != norm_mkt:
                continue
            if sr.settled_date > clean_as_of:
                continue
            if sr.ex_date is not None:
                if sr.ex_date != ex_d:
                    continue
            elif sr.settled_date < ex_d:
                continue

            sr_status = sr.status.strip().lower()
            consumed_settlement_indices.add(s_idx)
            if sr.settled_amount_native is not None:
                settled_native += sr.settled_amount_native
            elif sr.settled_amount_thb is not None:
                if sched.currency == "USD":
                    settled_native += (sr.settled_amount_thb / fx_rate).quantize(_TWO_DP, rounding=ROUND_HALF_UP)
                else:
                    settled_native += sr.settled_amount_thb
            elif sr_status in ("settled", "received", "paid", "completed"):
                fully_settled = True
                break

        if fully_settled or settled_native >= net_native:
            continue
        if settled_native > Decimal("0"):
            net_native = (net_native - settled_native).quantize(_TWO_DP, rounding=ROUND_HALF_UP)
            settlement_status = "PARTIALLY_SETTLED"

        if sched.currency == "USD":
            gross_thb = (gross_native * fx_rate).quantize(_TWO_DP, rounding=ROUND_HALF_UP)
            tax_thb = (tax_native * fx_rate).quantize(_TWO_DP, rounding=ROUND_HALF_UP)
            net_thb = (net_native * fx_rate).quantize(_TWO_DP, rounding=ROUND_HALF_UP)
            total_usd += net_native
            total_thb += net_thb
        else:
            gross_thb = gross_native
            tax_thb = tax_native
            net_thb = net_native
            total_thb += net_thb

        d_pay = date.fromisoformat(pay_d)
        days_left = max(0, (d_pay - d_asof).days)
        sources_used.add(entitlement_source)

        items.append(
            AccruedDividendItem(
                ticker=sched.ticker,
                units=entitled_units,
                dps_native=sched.dps_native,
                gross_amount_native=gross_native,
                net_amount_native=net_native,
                withholding_tax_native=tax_native,
                net_amount_thb=net_thb,
                currency=sched.currency,
                ex_date=ex_d,
                pay_date=pay_d,
                days_to_payment=days_left,
                market=norm_mkt or sched.market,
                entitlement_source=entitlement_source,
                gross_amount_thb=gross_thb,
                withholding_tax_thb=tax_thb,
                settlement_status=settlement_status,
            )
        )

    if not items:
        if unavailable_warned:
            entitlement_basis = "unavailable"
            entitlement_verified = False
        elif timelines is not None or snapshot_map:
            entitlement_basis = "entitlement_snapshot" if snapshot_map and timelines is None else "trade_history"
            entitlement_verified = True
        else:
            entitlement_basis = "none"
            entitlement_verified = True
    elif sources_used == {"holdings_fallback_unverified"}:
        entitlement_basis = "holdings_fallback_unverified"
        entitlement_verified = False
    elif "holdings_fallback_unverified" in sources_used:
        entitlement_basis = "mixed"
        entitlement_verified = False
    elif sources_used == {"entitlement_snapshot"}:
        entitlement_basis = "entitlement_snapshot"
        entitlement_verified = True
    elif sources_used == {"trade_history"}:
        entitlement_basis = "trade_history"
        entitlement_verified = True
    else:
        entitlement_basis = "mixed"
        entitlement_verified = True

    return AccruedDividendSummary(
        accrued_receivable_thb=total_thb.quantize(_TWO_DP, rounding=ROUND_HALF_UP),
        accrued_receivable_usd=total_usd.quantize(_TWO_DP, rounding=ROUND_HALF_UP),
        items=items,
        entitlement_basis=entitlement_basis,
        entitlement_verified=entitlement_verified,
        warnings=warnings,
    )
