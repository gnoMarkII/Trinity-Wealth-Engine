from decimal import Decimal
import pytest

from core.portfolio_advisory.circuit_breaker import (
    DrawdownPostureTier,
    PortfolioDrawdownPosture,
    evaluate_drawdown_circuit_breaker,
    compute_realized_tracking_error,
)


def test_drawdown_posture_graduated_tiers():
    peak = Decimal("1000000.00")

    # 1. At peak / new high -> NORMAL
    p_peak = evaluate_drawdown_circuit_breaker(current_nav_thb=peak, peak_nav_thb=peak)
    assert p_peak.drawdown_pct == Decimal("0.00")
    assert p_peak.posture_tier == DrawdownPostureTier.NORMAL
    assert p_peak.hurdle_adjustment == Decimal("0.0")

    p_above = evaluate_drawdown_circuit_breaker(current_nav_thb=Decimal("1100000.00"), peak_nav_thb=peak)
    assert p_above.drawdown_pct == Decimal("0.00")
    assert p_above.posture_tier == DrawdownPostureTier.NORMAL

    # 2. 5% Drawdown (<= 8.0%) -> NORMAL
    p_5 = evaluate_drawdown_circuit_breaker(current_nav_thb=Decimal("950000.00"), peak_nav_thb=peak)
    assert p_5.drawdown_pct == Decimal("5.00")
    assert p_5.posture_tier == DrawdownPostureTier.NORMAL

    # 3. 10% Drawdown (8.0% < DD <= 12.0%) -> TIER_1_CAUTION
    p_10 = evaluate_drawdown_circuit_breaker(current_nav_thb=Decimal("900000.00"), peak_nav_thb=peak)
    assert p_10.drawdown_pct == Decimal("10.00")
    assert p_10.posture_tier == DrawdownPostureTier.TIER_1_CAUTION
    assert p_10.hurdle_adjustment == Decimal("0.5")
    assert any("CAUTION ACTIVE" in r for r in p_10.action_restrictions)

    # 4. 14% Drawdown (12.0% < DD <= 15.0%) -> TIER_2_THROTTLE
    p_14 = evaluate_drawdown_circuit_breaker(current_nav_thb=Decimal("860000.00"), peak_nav_thb=peak)
    assert p_14.drawdown_pct == Decimal("14.00")
    assert p_14.posture_tier == DrawdownPostureTier.TIER_2_THROTTLE
    assert p_14.hurdle_adjustment == Decimal("1.0")
    assert any("CAPITAL THROTTLE ACTIVE" in r for r in p_14.action_restrictions)

    # 5. 18% Drawdown (> 15.0%) -> TIER_3_DEFENSIVE
    p_18 = evaluate_drawdown_circuit_breaker(current_nav_thb=Decimal("820000.00"), peak_nav_thb=peak)
    assert p_18.drawdown_pct == Decimal("18.00")
    assert p_18.posture_tier == DrawdownPostureTier.TIER_3_DEFENSIVE
    assert p_18.hurdle_adjustment == Decimal("2.0")
    assert any("EMERGENCY DEFENSIVE MODE" in r for r in p_18.action_restrictions)


def test_realized_tracking_error():
    # 1. Identical returns -> TE 0.00%
    rets_p = [Decimal("0.01")] * 20
    rets_b = [Decimal("0.01")] * 20
    te_zero = compute_realized_tracking_error(rets_p, rets_b)
    assert te_zero == Decimal("0.00")

    # 2. Constant alpha (e.g. +10 bps daily) -> std of excess return is 0 -> TE 0.00%
    rets_p_const = [Decimal("0.011")] * 20
    te_const = compute_realized_tracking_error(rets_p_const, rets_b)
    assert te_const == Decimal("0.00")

    # 3. Variable excess returns -> non-zero tracking error
    # Alternate +1% and -1% relative to benchmark
    rets_p_var = [Decimal("0.02") if i % 2 == 0 else Decimal("0.00") for i in range(20)]
    te_var = compute_realized_tracking_error(rets_p_var, rets_b)
    assert te_var is not None
    assert te_var > Decimal("10.0")  # ~1% daily std * sqrt(252) = ~15.87%

    # 4. Insufficient data (< 10 observations)
    assert compute_realized_tracking_error(rets_p[:5], rets_b[:5]) is None


def test_r16_unrounded_threshold_boundaries_validation_and_engine_enforcement():
    """R16 / W03: Unrounded DD comparison, invalid threshold rejection, unknown peak, TE alignment, and engine effect."""
    from core.portfolio_advisory.engine import evaluate_portfolio_advisory
    from core.portfolio_advisory.models import AdvisoryActionCode

    peak = Decimal("1000000")

    # Exact R16 reproduction: DD = 8.004% displays 8.00% but MUST trigger TIER_1_CAUTION (> 8.0%)
    posture_edge = evaluate_drawdown_circuit_breaker(current_nav_thb=Decimal("919960"), peak_nav_thb=peak)
    assert posture_edge.drawdown_pct == Decimal("8.00")
    assert posture_edge.posture_tier == DrawdownPostureTier.TIER_1_CAUTION
    assert posture_edge.hurdle_adjustment == Decimal("0.5")

    # Exact boundary at 8.00% stays NORMAL; 12.001% triggers TIER_2; 15.001% triggers TIER_3
    assert evaluate_drawdown_circuit_breaker(Decimal("920000"), peak).posture_tier == DrawdownPostureTier.NORMAL
    assert evaluate_drawdown_circuit_breaker(Decimal("880000"), peak).posture_tier == DrawdownPostureTier.TIER_1_CAUTION
    assert evaluate_drawdown_circuit_breaker(Decimal("879990"), peak).posture_tier == DrawdownPostureTier.TIER_2_THROTTLE
    assert evaluate_drawdown_circuit_breaker(Decimal("850000"), peak).posture_tier == DrawdownPostureTier.TIER_2_THROTTLE
    assert evaluate_drawdown_circuit_breaker(Decimal("849990"), peak).posture_tier == DrawdownPostureTier.TIER_3_DEFENSIVE

    # Reversed thresholds & invalid inputs raise ValueError
    with pytest.raises(ValueError, match="Thresholds"):
        evaluate_drawdown_circuit_breaker(peak, peak, Decimal("12.0"), Decimal("8.0"), Decimal("15.0"))
    with pytest.raises(ValueError, match="finite"):
        evaluate_drawdown_circuit_breaker(Decimal("NaN"), peak)

    # Unknown or <=0 peak reports explicit unavailable status
    unk = evaluate_drawdown_circuit_breaker(peak, None)
    assert unk.is_peak_available is False
    assert unk.status == "UNAVAILABLE_UNKNOWN_PEAK"
    assert unk.posture_tier == DrawdownPostureTier.NORMAL

    # Tracking error rejects mismatched series length
    with pytest.raises(ValueError, match="mismatch"):
        compute_realized_tracking_error([Decimal("0.01")] * 11, [Decimal("0.01")] * 10)

    # Engine enforcement: Tier-2 allows Core with conv 9.0 (>= 8.5), throttles Core conv 8.0 and Satellite conv 9.5;
    # Tier-3 halts all new buys.
    class H:
        def __init__(self, symbol, units, price, bucket_id):
            self.symbol = symbol
            self.asset_type = "Stock"
            self.units = Decimal(str(units))
            self.current_price_thb = Decimal(str(price))
            self.market_value_thb = Decimal(str(units * price))
            self.avg_cost_thb = Decimal(str(price))
            self.bucket_id = bucket_id
            self.market = "TH"
            self.currency = "THB"
            self.status = "active"
            self.name = symbol

    class B:
        def __init__(self, bid, name, pct):
            self.bucket_id = bid
            self.name = name
            self.target_percent = pct

    class S:
        holdings = [
            H("CORE9.BK", 1000, 100.0, "core"),       # 100k (below 25% target)
            H("CORE8.BK", 1000, 100.0, "core"),       # 100k (below 25% target)
            H("SAT9.BK", 1000, 100.0, "satellite"),   # 100k (below 50% target)
        ]
        summary = type("Sum", (), {"usd_thb_rate": Decimal("35.5"), "cash_balance_thb": Decimal("560000")})()
        buckets = [B("core", "Core Compounders", Decimal("50")), B("satellite", "Tactical Growth", Decimal("50"))]

    # NAV = 860,000 THB vs peak 1,000,000 THB -> 14% DD (TIER_2_THROTTLE, hurdle +1.0 -> 8.5 required)
    sidecars = {
        sym: {
            "ticker": sym,
            "market": "TH",
            "quant_signals": {
                "raw_analysis_price": 100.0,
                "volatility_pct": 20.0,
                "adtv_local_currency": 20_000_000.0,
                "deterministic_scorecard": {"core_conviction_score": conv, "action_stance": "ACCUMULATE_NOW"},
                "tactical_setup": {"buy_zone_max": 110.0, "buy_zone_min": 90.0},
            },
        }
        for sym, conv in [("CORE9.BK", 9.0), ("CORE8.BK", 8.0), ("SAT9.BK", 9.5)]
    }

    rep_t2 = evaluate_portfolio_advisory(
        S(),
        equity_sidecars=sidecars,
        peak_nav_thb=Decimal("1000000"),
    )
    assert rep_t2.risk_radar.drawdown_posture_tier == "TIER_2_THROTTLE"
    assert rep_t2.risk_radar.drawdown_status == "AVAILABLE"
    assert rep_t2.risk_radar.drawdown_hurdle_adjustment == Decimal("1.0")
    by_sym_t2 = {h.symbol: h for h in rep_t2.holdings_advisory}
    assert by_sym_t2["CORE9.BK"].action_code == AdvisoryActionCode.ACCUMULATE_NOW
    assert by_sym_t2["CORE8.BK"].action_code == AdvisoryActionCode.HOLD_WITHIN_BAND  # conv 8.0 < 8.5 hurdle
    assert by_sym_t2["SAT9.BK"].action_code == AdvisoryActionCode.HOLD_WITHIN_BAND   # non-Core blocked in Tier 2

    # Peak 1,050,000 vs flow-adjusted current 860,000 -> ~18.10% DD (TIER_3_DEFENSIVE -> halts all buys including CORE9)
    rep_t3 = evaluate_portfolio_advisory(
        S(),
        equity_sidecars=sidecars,
        peak_nav_thb=Decimal("1050000"),
        flow_adjusted_current_nav_thb=Decimal("860000"),
    )
    assert rep_t3.risk_radar.drawdown_posture_tier == "TIER_3_DEFENSIVE"
    by_sym_t3 = {h.symbol: h for h in rep_t3.holdings_advisory}
    assert by_sym_t3["CORE9.BK"].action_code == AdvisoryActionCode.HOLD_WITHIN_BAND
    assert rep_t3.cash_plan.immediate_deploy_thb == Decimal("0")
