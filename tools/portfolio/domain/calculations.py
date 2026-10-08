from decimal import Decimal, ROUND_HALF_UP
from typing import Literal, Optional, List, Dict, Tuple, Union
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
from .models import (
    Holding,
    PortfolioState,
    Summary,
    TradeFeeBreakdown,
    TradeImportItem,
    UNITS_QUANTUM,
    PRICE_QUANTUM,
    MONEY_QUANTUM,
    quantize_decimal,
)

log = get_logger(__name__)


def validate_reconciliation_invariant(
    units: Union[Decimal, float, str],
    price: Union[Decimal, float, str],
    gross_amount: Union[Decimal, float, str],
    fees: Union[TradeFeeBreakdown, Dict, Decimal, float, str],
    net_amount: Union[Decimal, float, str],
    action: str,
) -> Tuple[bool, str]:
    """Pure two-stage reconciliation for financial trade records.

    Stage 1 (Line-item level): Units * Price ~= Gross Amount within +/- 0.01
    Stage 2 (Statement level): Gross +/- Fees == Net Amount within +/- 0.01
    """
    try:
        u = quantize_decimal(units, UNITS_QUANTUM)
        p = quantize_decimal(price, PRICE_QUANTUM)
        gross = quantize_decimal(gross_amount, MONEY_QUANTUM)
        net = quantize_decimal(net_amount, MONEY_QUANTUM)

        if isinstance(fees, TradeFeeBreakdown):
            total_fees = fees.total_fees
        elif isinstance(fees, dict):
            total_fees = quantize_decimal(
                Decimal(str(fees.get("commission", "0.00") or "0.00"))
                + Decimal(str(fees.get("vat", "0.00") or "0.00"))
                + Decimal(str(fees.get("other_fees", "0.00") or "0.00")),
                MONEY_QUANTUM,
            )
        else:
            total_fees = quantize_decimal(fees, MONEY_QUANTUM)

        # Stage 2: Statement level Gross +/- Fees == Net
        act = action.strip().upper()
        if act == "BUY":
            expected_net = quantize_decimal(gross + total_fees, MONEY_QUANTUM)
        elif act == "SELL":
            expected_net = quantize_decimal(gross - total_fees, MONEY_QUANTUM)
        else:
            return False, f"Unsupported action '{action}' for reconciliation"

        stmt_diff = abs(expected_net - net)
        if stmt_diff > Decimal("0.01"):
            return False, f"Statement mismatch: Gross ({gross}) {'+' if act == 'BUY' else '-'} Fees ({total_fees}) = {expected_net} != Net ({net}), diff={stmt_diff}"

        # Stage 1: Line-item Units * Price vs Gross
        computed_gross = quantize_decimal(u * p, MONEY_QUANTUM)
        line_diff = abs(computed_gross - gross)

        # In fractional share trading (e.g. US fractional shares on Dime/DriveWealth), brokers
        # round the displayed execution unit price on confirmation notes to 2 decimal places
        # (e.g. $7.23 instead of $7.2325), causing a small rounding discrepancy between
        # Units * DisplayPrice and the exact Gross.
        # If the statement balance (Gross +/- Fees == Net) holds exactly (stmt_diff <= 0.01),
        # we allow the line-item rounding discrepancy up to the theoretical maximum rounding
        # bound from a 2-decimal rounded price: Units * 0.005 + 0.01
        is_fractional = (u % Decimal("1")) != Decimal("0")
        max_allowed_line_diff = Decimal("0.01")
        if is_fractional and stmt_diff <= Decimal("0.01"):
            fractional_bound = quantize_decimal(u * Decimal("0.005") + Decimal("0.01"), MONEY_QUANTUM)
            max_allowed_line_diff = max(Decimal("0.01"), fractional_bound)

        if line_diff > max_allowed_line_diff:
            return False, f"Line-item mismatch: Units ({u}) * Price ({p}) = {computed_gross} != Gross ({gross}), diff={line_diff}"

        return True, "OK"
    except Exception as e:
        return False, f"Reconciliation calculation error: {e}"


def allocate_document_fees_pro_rata(
    line_gross_amounts: List[Union[Decimal, float, str]],
    total_fee: Union[Decimal, float, str],
) -> List[Decimal]:
    """Allocate lump-sum document fee proportionally across lines by gross amount.

    Tie-breaker: Residual pennies are allocated to the line with the largest gross amount.
    """
    fee = quantize_decimal(total_fee, MONEY_QUANTUM)
    grosses = [quantize_decimal(g, MONEY_QUANTUM) for g in line_gross_amounts]
    total_gross = sum(grosses)

    if not grosses:
        return []
    if total_gross <= Decimal("0.00") or fee == Decimal("0.00"):
        n = Decimal(len(grosses))
        base = quantize_decimal(fee / n, MONEY_QUANTUM)
        allocated = [base for _ in grosses]
        residual = fee - sum(allocated)
        if residual > Decimal("0.00"):
            allocated[0] += residual
        return allocated

    allocated = []
    for g in grosses:
        line_fee = quantize_decimal((g / total_gross) * fee, MONEY_QUANTUM)
        allocated.append(line_fee)

    residual = fee - sum(allocated)
    if residual != Decimal("0.00"):
        max_idx = max(range(len(grosses)), key=lambda i: grosses[i])
        allocated[max_idx] += residual

    return allocated


def calc_weighted_avg_cost(
    prev_units: Union[Decimal, float],
    prev_cost: Union[Decimal, float],
    new_units: Union[Decimal, float],
    new_price: Union[Decimal, float],
) -> float:
    """คำนวณ Weighted-Average Cost Basis ด้วย Decimal Precision."""
    u_prev = Decimal(str(prev_units))
    c_prev = Decimal(str(prev_cost))
    u_new = Decimal(str(new_units))
    p_new = Decimal(str(new_price))

    total_units = u_prev + u_new
    if total_units <= Decimal("0.0000001"):
        return 0.0
    total_cost = (u_prev * c_prev) + (u_new * p_new)
    res = total_cost / total_units
    return float(quantize_decimal(res, Decimal("0.000001")))


def calc_realized_pnl(
    avg_cost: Union[Decimal, float],
    sell_units: Union[Decimal, float],
    sell_price: Union[Decimal, float],
    fx_rate: Union[Decimal, float] = 1.0,
) -> float:
    """คำนวณ Realized P&L ในสกุลเงิน THB ด้วย Decimal Precision."""
    c_avg = Decimal(str(avg_cost))
    u_sell = Decimal(str(sell_units))
    p_sell = Decimal(str(sell_price))
    fx = Decimal(str(fx_rate))

    pnl_native = (p_sell - c_avg) * u_sell
    pnl_thb = quantize_decimal(pnl_native * fx, MONEY_QUANTUM)
    return float(pnl_thb)


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
    voided_tx_ids = {
        str(r.get("Related_Transaction_ID") or "").strip()
        for r in sorted_rows
        if r.get("Related_Transaction_ID") and str(r.get("Action") or "").strip().upper().startswith("VOID_")
    }

    units_held_dec = Decimal("0.0")
    cumulative_cost_native_dec = Decimal("0.0")
    avg_cost_native_dec = Decimal("0.0")
    total_realized_pnl_thb_dec = Decimal("0.0")
    updated_rows: list[dict] = []

    for row in sorted_rows:
        r = dict(row)
        tx_id = str(r.get("Transaction_ID") or "").strip()
        action = str(r.get("Action") or "BUY").strip().upper()

        # Handle non-destructive void/reversal pairing:
        # Rows that were voided or are reversal records have 0 economic impact on holdings
        is_reversal = action.startswith("VOID_") or action == "REVERSAL"
        is_voided = bool(tx_id and tx_id in voided_tx_ids)
        if is_reversal or is_voided:
            r["Cost_THB"] = "0.00"
            r["Realized_PnL_THB"] = "0.00"
            updated_rows.append(r)
            continue

        try:
            units = Decimal(str(r.get("Units") or 0.0))
        except (ValueError, TypeError):
            units = Decimal("0.0")
        try:
            price = Decimal(str(r.get("Price") or 0.0))
        except (ValueError, TypeError):
            price = Decimal("0.0")

        fx_raw = r.get("FX_Rate")
        fx_rate = Decimal(str(fx_raw)) if fx_raw is not None and str(fx_raw).strip() != "" else None

        if action == "BUY":
            # If Net_Amount is recorded (e.g. Dime imports with fees), use it for cost basis
            net_amt_raw = r.get("Net_Amount")
            if net_amt_raw is not None and str(net_amt_raw).strip() != "":
                try:
                    amount_native = Decimal(str(net_amt_raw))
                except (ValueError, TypeError):
                    amount_native = units * price
            else:
                amount_native = units * price

            cumulative_cost_native_dec += amount_native
            units_held_dec += units
            avg_cost_native_dec = (
                (cumulative_cost_native_dec / units_held_dec)
                if units_held_dec > Decimal("1e-7")
                else Decimal("0.0")
            )
            cost_thb = amount_native * fx_rate if currency == "USD" and fx_rate is not None else amount_native

            r["Cost_THB"] = f"{cost_thb:.2f}"
            r["Realized_PnL_THB"] = ""
            r["Units"] = f"{units:g}"
            # Preserve full precision price string if available
            r["Price"] = str(r.get("Price") or f"{price:.2f}")
            updated_rows.append(r)

        elif action == "SELL":
            if units > units_held_dec + Decimal("1e-6"):
                raise ValueError(
                    f"Replay failed for {symbol}: Insufficient units to sell at {r.get('Timestamp')} "
                    f"(held: {units_held_dec:g}, tried to sell: {units:g})"
                )

            net_amt_raw = r.get("Net_Amount")
            if net_amt_raw is not None and str(net_amt_raw).strip() != "":
                try:
                    net_proceeds = Decimal(str(net_amt_raw))
                    realized_native = net_proceeds - (avg_cost_native_dec * units)
                except (ValueError, TypeError):
                    realized_native = (price - avg_cost_native_dec) * units
            else:
                realized_native = (price - avg_cost_native_dec) * units

            realized_thb = realized_native * fx_rate if currency == "USD" and fx_rate is not None else realized_native
            cost_thb = avg_cost_native_dec * units * (fx_rate if currency == "USD" and fx_rate is not None else Decimal("1.0"))

            units_held_dec = max(Decimal("0.0"), units_held_dec - units)
            cumulative_cost_native_dec = units_held_dec * avg_cost_native_dec
            total_realized_pnl_thb_dec += realized_thb

            r["Cost_THB"] = f"{cost_thb:.2f}"
            r["Realized_PnL_THB"] = f"{realized_thb:.2f}"
            r["Units"] = f"{units:g}"
            r["Price"] = str(r.get("Price") or f"{price:.2f}")
            updated_rows.append(r)

        else:
            updated_rows.append(r)

    return (
        updated_rows,
        float(units_held_dec),
        float(quantize_decimal(avg_cost_native_dec, PRICE_QUANTUM)),
        float(quantize_decimal(total_realized_pnl_thb_dec, MONEY_QUANTUM)),
    )


def extract_active_ledger_identities(
    rows: list[dict],
) -> dict[tuple[str, str], dict]:
    """Extract active (non-voided, non-reversal) transactions keyed by (Confirmation_No, Order_ID).

    Any transaction row that has been voided by a subsequent reversal row (identified by
    Action starting with 'VOID_' referencing the original Transaction_ID via Related_Transaction_ID)
    is excluded. The reversal row itself is also excluded.

    This ensures that when a transaction is deleted/voided from the portfolio, its natural
    identity is considered vacated, allowing it to be safely re-imported from trade confirmations.
    """
    voided_target_ids: set[str] = set()
    for r in rows:
        act = str(r.get("Action") or "").strip().upper()
        rel_id = str(r.get("Related_Transaction_ID") or "").strip()
        if (act.startswith("VOID_") or act == "REVERSAL") and rel_id:
            voided_target_ids.add(rel_id)

    active_map: dict[tuple[str, str], dict] = {}
    for r in rows:
        if not r:
            continue
        tx_id = str(r.get("Transaction_ID") or "").strip()
        act = str(r.get("Action") or "").strip().upper()

        # Skip reversal records and original records that were voided
        if act.startswith("VOID_") or act == "REVERSAL" or (tx_id and tx_id in voided_target_ids):
            continue

        c_no = str(r.get("Confirmation_No") or "").strip()
        o_id = str(r.get("Order_ID") or "").strip()
        if c_no and o_id:
            active_map[(c_no, o_id)] = dict(r)

    return active_map
