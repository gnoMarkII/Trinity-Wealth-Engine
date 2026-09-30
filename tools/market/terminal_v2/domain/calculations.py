"""Pure Mathematical and Analytical Calculations for Terminal V2.

Strict Hexagonal Architecture & Domain Rules:
1. Pure standard library only (0 external dependencies).
2. Pure, deterministic functions with no side effects or I/O.
3. Explicit edge-case handling: missing data, division by zero, min sample thresholds.
4. Mathematical Invariants:
   - Max Pain: Standard 100-multiplier series only; argmin cash payout across static OI.
   - Commodity Volatility: 52-week percentile requires min 100 historical samples.
   - Auction Demand: Uses prior completed auctions (up to 8) of identical type & term; NO WI tail.
   - Financial Ratios: Free Cash Flow, FCF Margin, Debt-to-OCF with guard for non-positive denominators.
   - Insider Buying: Filters strictly non-derivative P/S open market transactions within 90 days.
"""
from datetime import datetime, timedelta
from typing import Dict, Optional, Sequence, Tuple

from tools.market.terminal_v2.domain.models import (
    InsiderTransaction,
    OptionContract,
    OptionsMaxPainResult,
    OptionsPutCallRatios,
    PolicyRateItem,
)


# ============================================================================
# 1. Equity Derivatives & Options Calculations
# ============================================================================

def calculate_max_pain(
    contracts: Sequence[OptionContract],
    underlying: str,
    expiry: str,
    spot_price: Optional[float] = None,
) -> OptionsMaxPainResult:
    """Calculate deterministic Max Pain for a specific expiration date.

    Max Pain is the strike price where the aggregate dollar payout across all
    expiring option contracts is minimized for holders (most option value expires worthless).

    Mathematical Formulation:
    For candidate strike S in set of all listed strikes for expiry:
        Payout(S) = Sum_{calls} [ OI * multiplier * max(S - strike, 0) ]
                  + Sum_{puts}  [ OI * multiplier * max(strike - S, 0) ]
        MaxPainStrike = argmin_S Payout(S)

    Strict Assumptions & Constraints:
    - Only standard 100-multiplier contracts are included. Non-standard series are excluded.
    - Assumes open interest remains static until expiration.
    - Ignores dynamic dealer delta/gamma hedging and real-time order flow.
    - Not a forecast, price target, or proof of market maker profit.
    """
    valid_contracts = []
    excluded_count = 0

    for c in contracts:
        if c.expiry != expiry:
            continue
        if c.multiplier != 100 or not c.is_standard:
            excluded_count += 1
            continue
        if c.open_interest < 0 or c.strike <= 0:
            excluded_count += 1
            continue
        valid_contracts.append(c)

    if not valid_contracts:
        raise ValueError(f"No valid standard option contracts found for {underlying} on {expiry}")

    candidate_strikes = sorted({c.strike for c in valid_contracts})
    min_payout: Optional[float] = None
    best_strike: Optional[float] = None

    for s in candidate_strikes:
        total_payout = 0.0
        for c in valid_contracts:
            if c.open_interest <= 0:
                continue
            if c.side == "call":
                intrinsic = max(s - c.strike, 0.0)
            else:
                intrinsic = max(c.strike - s, 0.0)
            total_payout += intrinsic * c.open_interest * c.multiplier

        if min_payout is None or total_payout < min_payout:
            min_payout = total_payout
            best_strike = s
        elif total_payout == min_payout:
            # Tie breaker: prefer strike closest to spot_price if available
            if spot_price is not None and best_strike is not None:
                if abs(s - spot_price) < abs(best_strike - spot_price):
                    best_strike = s

    assert best_strike is not None
    assert min_payout is not None

    distance_from_spot = None
    distance_pct = None
    if spot_price is not None and spot_price > 0:
        distance_from_spot = spot_price - best_strike
        distance_pct = (distance_from_spot / spot_price) * 100.0

    return OptionsMaxPainResult(
        underlying=underlying.upper(),
        expiry=expiry,
        strike=best_strike,
        minimum_theoretical_payout=min_payout,
        candidate_count=len(candidate_strikes),
        excluded_contract_count=excluded_count,
        spot_price=spot_price,
        distance_from_spot=distance_from_spot,
        distance_pct=distance_pct,
        assumptions="Standard 100-multiplier series only; static open interest; zero hedging assumptions.",
        limitations="Analytical folk/composite indicator; NOT a price target, prediction, or evidence of market maker profits.",
    )


def calculate_put_call_ratios(
    contracts: Sequence[OptionContract],
    underlying: str,
    expiry: str,
) -> OptionsPutCallRatios:
    """Calculate Put/Call volume and open interest ratios for a specific expiration."""
    put_vol = 0
    call_vol = 0
    put_oi = 0
    call_oi = 0

    for c in contracts:
        if c.expiry != expiry or not c.is_standard:
            continue
        if c.side == "put":
            put_vol += max(c.volume, 0)
            put_oi += max(c.open_interest, 0)
        elif c.side == "call":
            call_vol += max(c.volume, 0)
            call_oi += max(c.open_interest, 0)

    vol_ratio = (put_vol / call_vol) if call_vol > 0 else None
    oi_ratio = (put_oi / call_oi) if call_oi > 0 else None

    return OptionsPutCallRatios(
        underlying=underlying.upper(),
        expiry=expiry,
        put_volume=put_vol,
        call_volume=call_vol,
        volume_ratio=vol_ratio,
        put_open_interest=put_oi,
        call_open_interest=call_oi,
        oi_ratio=oi_ratio,
    )


# ============================================================================
# 2. Macro Spreads, Percentiles & Regimes
# ============================================================================

def calculate_rate_spread_bps(rate_a_pct: float, rate_b_pct: float) -> float:
    """Calculate rate spread in basis points: (Rate A% - Rate B%) * 100."""
    return round((rate_a_pct - rate_b_pct) * 100.0, 4)


def calculate_cot_percentile(current_net: int, history_net: Sequence[int]) -> float:
    """Calculate the 52-week percentile rank (0.0 to 100.0) for current net positioning.

    Returns 50.0 if history is empty or contains only one point.
    """
    if not history_net:
        return 50.0

    min_val = min(history_net)
    max_val = max(history_net)

    if max_val == min_val:
        return 50.0

    pct = ((current_net - min_val) / (max_val - min_val)) * 100.0
    return round(max(0.0, min(100.0, pct)), 2)


def calculate_policy_rate_spreads(
    rates: Sequence[PolicyRateItem],
    benchmark_country: str = "TH",
) -> Dict[str, float]:
    """Calculate the policy interest rate spreads in basis points against a benchmark country.

    Formula: (country_rate - benchmark_rate) * 100 (in bps)
    """
    benchmark = next((r for r in rates if r.country == benchmark_country), None)
    if not benchmark:
        return {}

    bench_val = benchmark.rate_value
    spreads: Dict[str, float] = {}

    for item in rates:
        diff_bps = round((item.rate_value - bench_val) * 100.0, 1)
        spreads[item.country] = diff_bps

    return spreads


def classify_fsi_regime(fsi_value: float) -> str:
    """Classify the OFR Financial Stress Index into standard systemic risk regimes."""
    if fsi_value < -0.5:
        return "calm"
    elif fsi_value <= 0.5:
        return "normal"
    elif fsi_value <= 1.5:
        return "elevated"
    return "severe"


# ============================================================================
# 3. Commodity Volatility & Treasury Demand
# ============================================================================

def calculate_commodity_vol_percentile(
    current_val: float,
    history_closes: Sequence[float],
    min_samples: int = 100,
    max_lookback: int = 252,
) -> Tuple[Optional[float], int]:
    """Calculate 52-week percentile of commodity volatility close.

    Formula: 100 * (count of closes <= current_val) / total sample count.
    Uses up to `max_lookback` recent closes (representing ~52 trading weeks).
    Returns (None, sample_count) if sample_count < min_samples.
    """
    valid = [float(v) for v in history_closes if v is not None and v > 0]
    recent = valid[-max_lookback:] if len(valid) > max_lookback else valid
    sample_count = len(recent)

    if sample_count < min_samples:
        return None, sample_count

    lte_count = sum(1 for v in recent if v <= current_val)
    percentile = round((lte_count / sample_count) * 100.0, 2)
    return percentile, sample_count


def classify_commodity_vol_regime(percentile: Optional[float]) -> Optional[str]:
    """Classify volatility percentile into a heuristic regime label.

    Static thresholds (Heuristic):
    - >= 85.0: extreme_panic
    - >= 65.0: elevated
    - >= 35.0: normal
    - < 35.0:  complacent
    """
    if percentile is None:
        return None
    if percentile >= 85.0:
        return "extreme_panic"
    if percentile >= 65.0:
        return "elevated"
    if percentile >= 35.0:
        return "normal"
    return "complacent"


def calculate_auction_demand_summary(
    latest_btc: Optional[float],
    prior_btcs: Sequence[float],
    min_samples: int = 3,
    max_prior: int = 8,
) -> Tuple[Optional[float], Optional[float], int]:
    """Calculate auction demand moving average and delta.

    Strict Invariant:
    - Uses prior completed auctions (up to `max_prior`, default 8) of the SAME type and term.
    - Does NOT compute auction tail (which requires When-Issued market yields).
    - Returns (prior_mean, demand_delta, sample_count).
    - If prior samples < min_samples or latest_btc is None: returns (None, None, sample_count).
    """
    valid_prior = [float(b) for b in prior_btcs if b is not None and b > 0][:max_prior]
    sample_count = len(valid_prior)

    if latest_btc is None or latest_btc <= 0 or sample_count < min_samples:
        return None, None, sample_count

    prior_mean = sum(valid_prior) / sample_count
    demand_delta = latest_btc - prior_mean
    return round(prior_mean, 4), round(demand_delta, 4), sample_count


# ============================================================================
# 4. Corporate Financial Facts & Insider Transactions
# ============================================================================

def calculate_financial_ratios(
    revenue: Optional[float],
    operating_cash_flow: Optional[float],
    capex: Optional[float],
    long_term_debt: Optional[float],
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Compute Free Cash Flow, FCF Margin, and Debt-to-OCF ratio.

    Rules:
    - FCF = operating_cash_flow - capex (if both available).
    - FCF Margin = FCF / revenue (if FCF and revenue > 0).
    - Debt to OCF = long_term_debt / operating_cash_flow (if debt >= 0 and OCF > 0).
    - Returns (fcf, fcf_margin, debt_to_ocf). Missing/invalid values are None.
    """
    fcf: Optional[float] = None
    if operating_cash_flow is not None and capex is not None:
        fcf = round(operating_cash_flow - capex, 2)

    fcf_margin: Optional[float] = None
    if fcf is not None and revenue is not None and revenue > 0:
        fcf_margin = round(fcf / revenue, 4)

    debt_to_ocf: Optional[float] = None
    if long_term_debt is not None and operating_cash_flow is not None and operating_cash_flow > 0:
        debt_to_ocf = round(long_term_debt / operating_cash_flow, 4)

    return fcf, fcf_margin, debt_to_ocf


def calculate_insider_net_buying_90d(
    transactions: Sequence[InsiderTransaction],
    as_of_date_str: str,
) -> Tuple[Optional[float], float, float, int]:
    """Calculate 90-day net insider buying ratio from Form 4 transactions.

    Rules:
    - Strictly filters non-derivative open-market transactions with code 'P' (Purchase) or 'S' (Sale).
    - Excludes option exercises (M), equity grants/awards (A), gifts (G), and other non-open-market codes.
    - Transaction date must fall within [as_of_date - 90 days, as_of_date].
    - Notional = shares * price_per_share.
    - Net Buy Ratio = (sum_P - sum_S) / (sum_P + sum_S).
    - If sum_P + sum_S == 0: returns (None, 0.0, 0.0, 0).
    """
    try:
        as_of_dt = datetime.strptime(as_of_date_str[:10], "%Y-%m-%d")
    except (ValueError, TypeError):
        as_of_dt = datetime.utcnow()

    cutoff_dt = as_of_dt - timedelta(days=90)

    p_notional_sum = 0.0
    s_notional_sum = 0.0
    eligible_count = 0

    for tx in transactions:
        if not tx.transaction_date or tx.transaction_code not in ("P", "S"):
            continue

        try:
            tx_dt = datetime.strptime(tx.transaction_date[:10], "%Y-%m-%d")
        except (ValueError, TypeError):
            continue

        # Check 90-day window
        if not (cutoff_dt <= tx_dt <= as_of_dt):
            continue

        if tx.shares is None or tx.shares <= 0 or tx.price_per_share is None or tx.price_per_share <= 0:
            continue

        notional = tx.shares * tx.price_per_share
        if tx.transaction_code == "P":
            p_notional_sum += notional
            eligible_count += 1
        elif tx.transaction_code == "S":
            s_notional_sum += notional
            eligible_count += 1

    total_notional = p_notional_sum + s_notional_sum
    if total_notional <= 0:
        return None, 0.0, 0.0, 0

    net_buy_ratio = round((p_notional_sum - s_notional_sum) / total_notional, 4)
    return net_buy_ratio, round(p_notional_sum, 2), round(s_notional_sum, 2), eligible_count
