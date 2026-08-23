from typing import Literal, Optional, List, Dict, Tuple
from core.logger import get_logger
from .constants import (
    CASH_THB_SYMBOL,
    CASH_USD_SYMBOL,
    _FLOAT_EPS,
    _MONEY_DP,
    _COST_DP,
    _PCT_DP,
    MARKET_CAP_MEGA_USD,
    MARKET_CAP_LARGE_USD,
    MARKET_CAP_MID_USD,
)
from .models import Holding, PortfolioState, Summary

log = get_logger(__name__)


def calc_weighted_avg_cost(prev_units: float, prev_cost: float, new_units: float, new_price: float) -> float:
    """คำนวณ Weighted-Average Cost Basis."""
    total_units = prev_units + new_units
    if total_units <= _FLOAT_EPS:
        return 0.0
    total_cost = (prev_units * prev_cost) + (new_units * new_price)
    return round(total_cost / total_units, _COST_DP)


def calc_realized_pnl(avg_cost: float, sell_units: float, sell_price: float, fx_rate: float = 1.0) -> float:
    """คำนวณ Realized P&L ในสกุลเงิน THB."""
    pnl_native = (sell_price - avg_cost) * sell_units
    return round(pnl_native * fx_rate, _MONEY_DP)


def calc_holding_currency(h: Holding) -> str:
    """Determine currency of a holding."""
    if h.symbol == CASH_USD_SYMBOL:
        return "USD"
    if h.symbol == CASH_THB_SYMBOL:
        return "THB"
    if h.avg_cost_usd is not None:
        return "USD"
    if h.avg_cost_thb is not None:
        return "THB"
    return "UNKNOWN"


def recalc_holding(h: Holding, current_fx: float) -> None:
    """คำนวณ market_value_thb และ unrealized_pnl_percent ให้ holding 1 ตัว."""
    if h.asset_type == "Cash":
        if h.symbol == CASH_USD_SYMBOL:
            h.market_value_thb = round(h.units * current_fx, _MONEY_DP)
        else:
            h.market_value_thb = round(h.units, _MONEY_DP)
        h.unrealized_pnl_percent = None
        h.accumulated_dividend_thb = None
        return

    if h.avg_cost_usd is not None and h.current_price_usd is not None:
        h.market_value_thb = round(h.units * h.current_price_usd * current_fx, _MONEY_DP)
        h.unrealized_pnl_percent = (
            round((h.current_price_usd - h.avg_cost_usd) / h.avg_cost_usd * 100, _PCT_DP)
            if h.avg_cost_usd
            else 0.0
        )
    elif h.avg_cost_thb is not None and h.current_price_thb is not None:
        h.market_value_thb = round(h.units * h.current_price_thb, _MONEY_DP)
        h.unrealized_pnl_percent = (
            round((h.current_price_thb - h.avg_cost_thb) / h.avg_cost_thb * 100, _PCT_DP)
            if h.avg_cost_thb
            else 0.0
        )
    else:
        log.warning("Holding %s has incomplete cost/price pair — market value reset to 0", h.symbol)
        h.market_value_thb = 0.0
        h.unrealized_pnl_percent = None


def compute_total_cost(state: PortfolioState, current_fx: float) -> float:
    """รวมต้นทุนทุก holding ใน THB."""
    total = 0.0
    for h in state.holdings:
        if h.asset_type == "Cash":
            if h.symbol == CASH_USD_SYMBOL:
                total += h.units * current_fx
            else:
                total += h.units
            continue
        if h.avg_cost_usd is not None and h.current_price_usd is not None:
            total += h.units * h.avg_cost_usd * current_fx
        elif h.avg_cost_thb is not None and h.current_price_thb is not None:
            total += h.units * h.avg_cost_thb
    return round(total, _MONEY_DP)


def recalc_summary(state: PortfolioState, current_fx: float) -> None:
    """รวม total_value_thb และ total_unrealized_profit จาก holdings."""
    total_value = 0.0
    total_unrealized = 0.0

    for h in state.holdings:
        total_value += h.market_value_thb
        if h.asset_type == "Cash":
            continue
        if h.avg_cost_usd is not None and h.current_price_usd is not None:
            total_unrealized += (h.current_price_usd - h.avg_cost_usd) * h.units * current_fx
        elif h.avg_cost_thb is not None and h.current_price_thb is not None:
            total_unrealized += (h.current_price_thb - h.avg_cost_thb) * h.units

    state.summary.total_value_thb = round(total_value, _MONEY_DP)
    state.summary.total_cost_basis_thb = compute_total_cost(state, current_fx)
    state.summary.total_unrealized_profit = round(total_unrealized, _MONEY_DP)


def recalc_fundamentals_derived(state: PortfolioState) -> None:
    """คำนวณ derived fundamentals/metrics ให้ทุก holding (market_cap_tier, yield_on_cost, unrealized_pnl_value)."""
    current_fx = state.fx_rates.get("USDTHB", 0.0) or 0.0
    for h in state.holdings:
        if h.asset_type == "Cash":
            setattr(h, "market_cap_tier", "N/A")
            setattr(h, "yield_on_cost", None)
            setattr(h, "unrealized_pnl_value", 0.0)
            if h.bucket_id is None:
                h.bucket_id = "cash"
            continue

        # Market cap tier
        mcap = getattr(h, "market_cap_value", None)
        if mcap is not None and isinstance(mcap, (int, float)) and mcap > 0:
            if mcap >= MARKET_CAP_MEGA_USD:
                setattr(h, "market_cap_tier", "Mega")
            elif mcap >= MARKET_CAP_LARGE_USD:
                setattr(h, "market_cap_tier", "Large")
            elif mcap >= MARKET_CAP_MID_USD:
                setattr(h, "market_cap_tier", "Mid")
            else:
                setattr(h, "market_cap_tier", "Small")
        else:
            setattr(h, "market_cap_tier", "N/A")

        # Yield on cost
        div_rate = getattr(h, "dividend_per_share", None)
        if div_rate is not None and isinstance(div_rate, (int, float)) and div_rate >= 0:
            if h.avg_cost_usd is not None and h.avg_cost_usd > 0:
                setattr(h, "yield_on_cost", round((div_rate / h.avg_cost_usd) * 100, _PCT_DP))
            elif h.avg_cost_thb is not None and h.avg_cost_thb > 0:
                setattr(h, "yield_on_cost", round((div_rate / h.avg_cost_thb) * 100, _PCT_DP))
            else:
                setattr(h, "yield_on_cost", None)
        else:
            setattr(h, "yield_on_cost", None)

        # Unrealized PnL Value (THB)
        if h.avg_cost_usd is not None and h.current_price_usd is not None:
            cost_thb = h.units * h.avg_cost_usd * current_fx
            setattr(h, "unrealized_pnl_value", round(h.market_value_thb - cost_thb, _MONEY_DP))
        elif h.avg_cost_thb is not None and h.current_price_thb is not None:
            cost_thb = h.units * h.avg_cost_thb
            setattr(h, "unrealized_pnl_value", round(h.market_value_thb - cost_thb, _MONEY_DP))
        else:
            setattr(h, "unrealized_pnl_value", None)


def recalc_all(state: PortfolioState) -> None:
    """Anti-Drift: คำนวณใหม่ทั้งหมดโดยใช้ fx_rates.USDTHB ปัจจุบันของพอร์ต."""
    current_fx = state.fx_rates.get("USDTHB", 0.0) or 0.0
    if current_fx <= 0:
        log.warning("fx_rates.USDTHB missing or invalid — USD holdings will compute as 0")
    for h in state.holdings:
        recalc_holding(h, current_fx)
    recalc_summary(state, current_fx)
    recalc_fundamentals_derived(state)


def compute_allocation_breakdown(
    state: PortfolioState, group_by: Literal["asset_type", "currency"] = "asset_type"
) -> List[Dict]:
    """คำนวณ Breakdown การจัดสรรสินทรัพย์ตาม asset_type หรือ currency."""
    recalc_all(state)
    total_nav = state.summary.total_value_thb

    buckets: Dict[str, Dict] = {}
    for h in state.holdings:
        key = h.asset_type if group_by == "asset_type" else calc_holding_currency(h)
        b = buckets.setdefault(key, {"value_thb": 0.0, "count": 0})
        b["value_thb"] += h.market_value_thb
        b["count"] += 1

    breakdown = [
        {
            "group": k,
            "value_thb": round(v["value_thb"], _MONEY_DP),
            "pct": round((v["value_thb"] / total_nav * 100) if total_nav > 0 else 0.0, _PCT_DP),
            "count": v["count"],
        }
        for k, v in buckets.items()
    ]
    breakdown.sort(key=lambda x: x["value_thb"], reverse=True)
    return breakdown


def compute_target_allocation_variance(
    state: PortfolioState
) -> Tuple[List[Dict], Optional[str]]:
    """คำนวณเปรียบเทียบ Target Allocation กับ Actual Allocation ปัจจุบัน."""
    recalc_all(state)
    total_nav = state.summary.total_value_thb

    bucket_values: Dict[str, float] = {}
    unassigned_thb = 0.0

    for h in state.holdings:
        if h.bucket_id:
            bucket_values[h.bucket_id] = bucket_values.get(h.bucket_id, 0.0) + h.market_value_thb
        else:
            unassigned_thb += h.market_value_thb

    total_target_pct = sum(t.target_percent for t in state.allocation_targets)
    warning_flag: Optional[str] = None
    if abs(total_target_pct - 100.0) > 0.01:
        warning_flag = f"ผลรวมเป้าหมาย ({total_target_pct:.1f}%) ไม่เท่ากับ 100%"

    summaries = []
    for t in state.allocation_targets:
        actual_thb = bucket_values.get(t.bucket_id, 0.0)
        actual_pct = round((actual_thb / total_nav) * 100, _PCT_DP) if total_nav > 0 else 0.0
        variance = round(actual_pct - t.target_percent, _PCT_DP)

        summaries.append({
            "bucket_id": t.bucket_id,
            "name": t.name,
            "target_percent": t.target_percent,
            "actual_value_thb": round(actual_thb, _MONEY_DP),
            "actual_percent": actual_pct,
            "variance": variance,
            "color": t.color,
        })

    if unassigned_thb > _FLOAT_EPS:
        actual_pct = round((unassigned_thb / total_nav) * 100, _PCT_DP) if total_nav > 0 else 0.0
        summaries.append({
            "bucket_id": "unassigned",
            "name": "Unassigned",
            "target_percent": 0.0,
            "actual_value_thb": round(unassigned_thb, _MONEY_DP),
            "actual_percent": actual_pct,
            "variance": actual_pct,
            "color": "#64748B",
        })

    return summaries, warning_flag


def _replay_symbol_trades(
    trades_rows: list[dict],
    symbol: str,
    currency: str,
) -> tuple[list[dict], float, float, float]:
    """Pure helper to replay all trades for a specific symbol chronologically.

    Args:
        trades_rows: List of dicts representing transactions for this symbol.
        symbol: Asset ticker symbol.
        currency: 'THB' or 'USD'.

    Returns:
        tuple of (updated_rows, final_units, final_avg_cost, total_realized_pnl_thb)

    Raises:
        ValueError: If cumulative units held drop below zero at any point.
    """
    sorted_rows = sorted(trades_rows, key=lambda r: str(r.get("Timestamp") or ""))
    units_held = 0.0
    cumulative_cost_native = 0.0
    avg_cost_native = 0.0
    total_realized_pnl_thb = 0.0
    updated_rows: list[dict] = []

    for row in sorted_rows:
        r = dict(row)
        action = str(r.get("Action") or "BUY").strip().upper()
        try:
            units = float(r.get("Units") or 0.0)
        except (ValueError, TypeError):
            units = 0.0
        try:
            price = float(r.get("Price") or 0.0)
        except (ValueError, TypeError):
            price = 0.0

        fx_raw = r.get("FX_Rate")
        fx_rate = float(fx_raw) if fx_raw is not None and str(fx_raw).strip() != "" else None

        if action == "BUY":
            amount_native = units * price
            cumulative_cost_native += amount_native
            units_held += units
            avg_cost_native = (cumulative_cost_native / units_held) if units_held > _FLOAT_EPS else 0.0
            cost_thb = amount_native * fx_rate if currency == "USD" and fx_rate is not None else amount_native

            r["Cost_THB"] = f"{cost_thb:.2f}"
            r["Realized_PnL_THB"] = ""
            r["Units"] = f"{units:g}"
            r["Price"] = f"{price:.2f}"
            updated_rows.append(r)

        elif action == "SELL":
            if units > units_held + _FLOAT_EPS:
                raise ValueError(
                    f"Replay failed for {symbol}: Insufficient units to sell at {r.get('Timestamp')} "
                    f"(held: {units_held:g}, tried to sell: {units:g})"
                )
            realized_native = (price - avg_cost_native) * units
            realized_thb = realized_native * fx_rate if currency == "USD" and fx_rate is not None else realized_native
            cost_thb = avg_cost_native * units * (fx_rate if currency == "USD" and fx_rate is not None else 1.0)

            units_held = max(0.0, units_held - units)
            cumulative_cost_native = units_held * avg_cost_native
            total_realized_pnl_thb += realized_thb

            r["Cost_THB"] = f"{cost_thb:.2f}"
            r["Realized_PnL_THB"] = f"{realized_thb:.2f}"
            r["Units"] = f"{units:g}"
            r["Price"] = f"{price:.2f}"
            updated_rows.append(r)

        else:
            updated_rows.append(r)

    return updated_rows, units_held, avg_cost_native, total_realized_pnl_thb
