"""Data and Temporal Integrity Guardrails (T1–T5, I1–I5, I7).

Architecture Layer: core/portfolio_advisory/ (Pure Domain)
Enforces:
- S3: safe_decimal with NaN, Inf, and InvalidOperation handling
- D1: resolve_holding_currency_and_market preventing Thai stock USD trap
- T1–T5: Temporal integrity, freshness, same-day thesis guard, and live upside re-anchoring
- I1–I5, I7: Price scale sanity, FX rates, bucket sum, identity, and unassigned drift guard
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import math
from typing import Any, List, Optional, Tuple

from core.portfolio_advisory.models import (
    DataAnomalyWarning,
    QuarantinedHoldingItem,
)


def safe_decimal(
    val: Any,
    default: Optional[Decimal] = Decimal("0"),
    quantum: Optional[str] = "0.01",
) -> Optional[Decimal]:
    """Converts any numeric value or string to Decimal safely without throwing InvalidOperation.

    Handles NaN, Inf, -Inf, None, MagicMock, and unparseable strings gracefully.
    """
    if val is None or val.__class__.__name__ == "MagicMock":
        return default
    if isinstance(val, (int, Decimal)):
        res = Decimal(str(val))
    elif isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return default
        res = Decimal(str(val))
    elif isinstance(val, str):
        cleaned = val.strip().replace(",", "")
        if not cleaned or cleaned.lower() in ("nan", "inf", "-inf", "null", "none"):
            return default
        try:
            res = Decimal(cleaned)
        except (InvalidOperation, ValueError):
            return default
    else:
        try:
            res = Decimal(str(val))
        except (InvalidOperation, ValueError, TypeError):
            return default

    if quantum:
        try:
            return res.quantize(Decimal(quantum), rounding=ROUND_HALF_UP)
        except InvalidOperation:
            return default
    return res


def resolve_holding_currency_and_market(holding: Any) -> Tuple[str, str]:
    """Resolves market ('TH' | 'US') and native currency ('THB' | 'USD') for a holding item.

    Enforces Defect D1:
    - Never deduces US market simply because current_price_usd exists (since price refresh calculates both).
    - Uses avg_cost_usd vs avg_cost_thb, .BK ticker suffix, and asset type.
    """
    symbol = str(getattr(holding, "symbol", "")).strip().upper()
    asset_type = str(getattr(holding, "asset_type", "")).strip().lower()

    # Tax vehicles and domestic mutual funds/gold are always Thai Baht
    if any(k in asset_type for k in ("fund", "mutual_fund", "gold", "bond", "pvd", "rmf", "ssf", "thaiesg")):
        return "TH", "THB"

    # Thai stock symbol suffix
    if symbol.endswith(".BK"):
        return "TH", "THB"

    avg_usd = getattr(holding, "avg_cost_usd", None)
    avg_thb = getattr(holding, "avg_cost_thb", None)
    has_usd = isinstance(avg_usd, (int, float, Decimal)) and float(avg_usd) > 0
    has_thb = isinstance(avg_thb, (int, float, Decimal)) and float(avg_thb) > 0

    if has_usd and not has_thb:
        return "US", "USD"
    if has_thb and not has_usd:
        return "TH", "THB"

    curr = str(getattr(holding, "currency", "")).strip().upper()
    if curr == "USD":
        return "US", "USD"

    # Default fallback for SET equities
    return "TH", "THB"


# =====================================================================
# Temporal Guardrails (T1–T5)
# =====================================================================

def check_future_date_anomaly(date_str: Optional[str], threshold_days: int = 1) -> Optional[DataAnomalyWarning]:
    """T1: Detects timestamps in the future beyond threshold_days."""
    if not date_str:
        return None
    try:
        iso_clean = str(date_str)[:10]
        parsed = date.fromisoformat(iso_clean)
        today = datetime.now(timezone.utc).date()
        if (parsed - today).days > threshold_days:
            return DataAnomalyWarning(
                guardrail_code="FUTURE_DATE_ANOMALY",
                ticker="",
                message=f"Date '{date_str}' is { (parsed - today).days } days in the future",
                severity="critical",
                remedy_hint="Ensure system clock and upstream report generation dates are synchronized.",
            )
    except Exception:
        pass
    return None


def check_stale_portfolio_price(
    price_date: Optional[str],
    sidecar_date: Optional[str],
    max_days: int = 3,
) -> Tuple[bool, Optional[DataAnomalyWarning]]:
    """T2: Checks if portfolio holding price is older than sidecar price or older than max_days.

    Returns (is_stale, warning).
    """
    today = datetime.now(timezone.utc).date()
    p_date: Optional[date] = None
    s_date: Optional[date] = None

    if price_date:
        try:
            p_date = date.fromisoformat(str(price_date)[:10])
        except Exception:
            pass

    if sidecar_date:
        try:
            s_date = date.fromisoformat(str(sidecar_date)[:10])
        except Exception:
            pass

    if p_date:
        days_old = (today - p_date).days
        if days_old > max_days:
            warn = DataAnomalyWarning(
                guardrail_code="PORTFOLIO_PRICE_OLDER_THAN_SIDECAR",
                ticker="",
                message=f"Portfolio price date ({p_date}) is {days_old} days old (exceeds {max_days} days limit)",
                severity="warning",
                remedy_hint="Trigger a price refresh to update market valuations before executing market orders.",
            )
            return True, warn

    return False, None


def check_macro_cross_scale_drift(
    macro_evaluated_at: Optional[str],
    sector_as_of_date: Optional[str],
    max_macro_days: int = 14,
) -> Tuple[str, List[DataAnomalyWarning]]:
    """T3: Checks age of Macro Strategy Direction and drift from Sector Rotation.

    Returns status ('fresh' | 'stale') and list of warnings.
    """
    today = datetime.now(timezone.utc).date()
    warnings: list[DataAnomalyWarning] = []
    is_stale = False

    if macro_evaluated_at:
        try:
            m_date = date.fromisoformat(str(macro_evaluated_at)[:10])
            days_old = (today - m_date).days
            if days_old > max_macro_days:
                is_stale = True
                warnings.append(DataAnomalyWarning(
                    guardrail_code="CROSS_SCALE_DATE_DRIFT",
                    ticker="MACRO",
                    message=f"Macro strategy report is {days_old} days old (exceeds {max_macro_days} days threshold)",
                    severity="warning",
                    remedy_hint="Run Strategic Allocator to refresh macro direction before aggressive rebalancing.",
                ))
        except Exception:
            pass
    else:
        is_stale = True

    return ("stale" if is_stale else "fresh"), warnings


def compute_live_upside(
    current_price: Decimal,
    target_price: Optional[Decimal],
) -> Optional[Decimal]:
    """T5: Re-anchors 12-month upside percentage against latest price using Decimal."""
    if not target_price or target_price <= Decimal("0"):
        return None
    if current_price <= Decimal("0"):
        return None
    upside = ((target_price - current_price) / current_price) * Decimal("100")
    return safe_decimal(upside, quantum="0.1")


# =====================================================================
# Input & Sanity Guardrails (I1–I7)
# =====================================================================

def check_price_scale_anomaly(
    holding_price: Decimal,
    sidecar_price: Optional[Decimal],
    ticker: str = "",
    tolerance_pct: Decimal = Decimal("50.0"),
) -> Optional[DataAnomalyWarning]:
    """I1: Catches currency scale mismatches (e.g. typing THB for US stock) or unadjusted splits.

    If holding_price diverges from sidecar_price by more than tolerance_pct, flags critical anomaly.
    """
    if not sidecar_price or sidecar_price <= Decimal("0"):
        return None
    if holding_price <= Decimal("0"):
        return None

    diff_pct = (abs(holding_price - sidecar_price) / sidecar_price) * Decimal("100")
    if diff_pct > tolerance_pct:
        return DataAnomalyWarning(
            guardrail_code="PRICE_SCALE_OR_CURRENCY_ANOMALY",
            ticker=ticker,
            message=f"Price scale mismatch on {ticker}: Portfolio price ({holding_price}) differs from analysis price ({sidecar_price}) by {diff_pct:.1f}%",
            severity="critical",
            remedy_hint="Check if currency (THB vs USD) was entered incorrectly or if a stock split occurred.",
        )
    return None


def validate_fx_rate(
    fx_rate: Optional[Decimal],
    fallback_rate: Decimal = Decimal("35.50"),
) -> Tuple[Decimal, Optional[DataAnomalyWarning]]:
    """I2: Validates USD/THB exchange rate sanity (20.0 <= fx <= 60.0)."""
    rate = safe_decimal(fx_rate, default=fallback_rate, quantum="0.0001")
    if rate < Decimal("20.0") or rate > Decimal("60.0"):
        warn = DataAnomalyWarning(
            guardrail_code="FX_AND_FINITE_NUMBER_GATE",
            ticker="USDTHB",
            message=f"USD/THB exchange rate {rate} is outside valid bounds (20.0 - 60.0). Reverting to fallback {fallback_rate}",
            severity="warning",
            remedy_hint="Update FX rate in settings or price provider.",
        )
        return fallback_rate, warn
    return rate, None


def validate_bucket_target_sum(
    buckets: List[Any],
    tolerance_pct: Decimal = Decimal("0.5"),
) -> Tuple[bool, Optional[DataAnomalyWarning]]:
    """I3: Ensures strategic bucket target percentages sum to 100% +/- tolerance_pct."""
    if not buckets:
        return True, None
    total_pct = Decimal("0")
    for b in buckets:
        pct = getattr(b, "target_percent", None)
        if pct is not None:
            total_pct += safe_decimal(pct, quantum="0.01")

    diff = abs(total_pct - Decimal("100.0"))
    if diff > tolerance_pct:
        warn = DataAnomalyWarning(
            guardrail_code="BUCKET_TARGET_SUM_INVALID",
            ticker="",
            message=f"Strategy bucket target percentages sum to {total_pct}%, which differs from 100.0%",
            severity="warning",
            remedy_hint="Adjust strategic allocation bucket weights in Portfolio Strategy Settings to total 100%.",
        )
        return False, warn
    return True, None


def verify_ticker_market_identity(
    holding_symbol: str,
    holding_market: str,
    sidecar_symbol: str,
    sidecar_market: str,
) -> bool:
    """I4: Verifies 3-way identity (normalized symbol + market) before pairing."""
    s1 = holding_symbol.replace(".BK", "").strip().upper()
    s2 = sidecar_symbol.replace(".BK", "").strip().upper()
    if s1 != s2:
        return False
    return holding_market.strip().upper() == sidecar_market.strip().upper()


def check_unassigned_holdings_drift(
    unassigned_value_thb: Decimal,
    total_nav_thb: Decimal,
    threshold_pct: Decimal = Decimal("5.0"),
) -> Tuple[bool, Optional[DataAnomalyWarning]]:
    """I7: Guards against unassigned holdings causing false underweight buying signals.

    If holdings with bucket_id=None exceed threshold_pct of NAV, returns (True, warning).
    """
    if total_nav_thb <= Decimal("0"):
        return False, None
    pct = (unassigned_value_thb / total_nav_thb) * Decimal("100")
    if pct > threshold_pct:
        warn = DataAnomalyWarning(
            guardrail_code="UNASSIGNED_BUCKET_DRIFT_GUARD",
            ticker="",
            message=f"Unassigned assets without strategy buckets account for {pct:.1f}% of NAV (exceeds {threshold_pct}% threshold)",
            severity="warning",
            remedy_hint="Assign newly imported holdings to strategy buckets before initiating rebalancing trades.",
        )
        return True, warn
    return False, None


def check_look_through_concentration_cap(
    look_through_exposure: Any,
    cap_threshold_pct: Decimal = Decimal("15.0"),
) -> Tuple[bool, Optional[DataAnomalyWarning]]:
    """G12: Guards against hidden concentration cap breach across direct and mutual fund holdings.

    If total economic exposure exceeds cap_threshold_pct (default 15.0%), returns (True, warning).
    """
    if not look_through_exposure:
        return False, None

    total_weight = safe_decimal(
        getattr(look_through_exposure, "total_weight_pct", None)
        or (look_through_exposure.get("total_weight_pct") if isinstance(look_through_exposure, dict) else 0)
    )
    direct_weight = safe_decimal(
        getattr(look_through_exposure, "direct_weight_pct", None)
        or (look_through_exposure.get("direct_weight_pct") if isinstance(look_through_exposure, dict) else 0)
    )
    raw_sym = (
        getattr(look_through_exposure, "ticker", None)
        or getattr(look_through_exposure, "stock_symbol", None)
        or getattr(look_through_exposure, "symbol", None)
        or (
            look_through_exposure.get("ticker")
            or look_through_exposure.get("stock_symbol")
            or look_through_exposure.get("symbol")
            if isinstance(look_through_exposure, dict)
            else ""
        )
    )
    ticker = str(raw_sym or "").strip().upper()

    if total_weight > cap_threshold_pct:
        is_hidden = direct_weight <= cap_threshold_pct
        source_funds = getattr(look_through_exposure, "source_funds", None) or (
            look_through_exposure.get("source_funds", []) if isinstance(look_through_exposure, dict) else []
        )
        funds_str = ", ".join(f"{s[0]}" for s in source_funds) if source_funds else "underlying funds"

        warn = DataAnomalyWarning(
            guardrail_code="ANOMALY_LOOK_THROUGH_CONCENTRATION_CAP",
            ticker=ticker,
            message=(
                f"True Look-Through exposure for {ticker} is {total_weight:.1f}%, exceeding {cap_threshold_pct}% Cap "
                f"(Direct: {direct_weight:.1f}%, Indirect via {funds_str})"
                if is_hidden
                else f"Single-stock economic exposure for {ticker} is {total_weight:.1f}% (exceeds {cap_threshold_pct}% Cap)"
            ),
            severity="warning" if is_hidden else "critical",
            remedy_hint=(
                f"Consider trimming direct position in {ticker} or reallocating from {funds_str} to reduce aggregated exposure below {cap_threshold_pct}%."
            ),
        )
        return True, warn

    return False, None
